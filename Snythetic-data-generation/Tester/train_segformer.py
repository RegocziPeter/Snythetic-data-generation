import torch
from datasets import load_dataset, concatenate_datasets
import numpy as np
from model import get_model
import os
from PIL import Image

from huggingface_hub import login, hf_hub_download
from albumentations.pytorch.transforms import ToTensorV2
import json

from transformers import (
    TrainingArguments,
    Trainer
)

import albumentations as A
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as T
import psutil

ONLY_EVAL = False
ONLY_TEST = True
NUM_EPOCHS = 70
LR = 1e-4

# The generated synthetic dataset, introduced in the paper
SYNTHETIC_DATASETS = ["regpeter/omnigen_cityscapes_1024_512"]

IDD_test_ds_id = "<IDD_DATASET_ID>"  # Replace with actual HF dataset ID
BDD_test_ds_id = "<BDD_DATASET_ID>"  # Replace with actual HF dataset ID
TEST_DATASETS = [IDD_test_ds_id, BDD_test_ds_id]


TRAIN_BATCH_SIZE = 16 # 16
EVAL_BATCH_SIZE =  8 if not ONLY_TEST else 1 # for DINO 16

IMAGE_SIZE = (512,1024)
CROP_SIZE = (512, 512)
LOGGING_STEP_NUM = 375  #375
SHUFFLE_DATASET = True
USE_AUGMENTATION= True
RESUME_CHECKPOINT = False
MODEL_NAME = "segformer"
# if MODEL_ID is not None, it will load the model from the checkpoint
MODEL_ID = None #"checkpoints/checkpoint-15375/model.safetensors"

hf_token = "<HF_TOKEN>" # Replace with your actual token
login(hf_token)

# The cityscapes dataset on HF hub should contain the images and the segmentation masks with 1024*512 resolution.
# Each pixel in the segmentation mask should have a value between 0-18 for the classes and 255 for the ignore class.
# The dataset should contain 'train' and 'val' splits. Under them there should be 'image' and 'label' fields.
cityscapes_hf_dataset_id = "<CITYSCAPES_DATASET_ID>"
ds_cityscapes = load_dataset(cityscapes_hf_dataset_id)

if SHUFFLE_DATASET:
    ds_cityscapes = ds_cityscapes.shuffle(seed=1)
DO_EVAL = True if "val" in ds_cityscapes and ds_cityscapes["val"] else False
ds_cityscapes = ds_cityscapes
train_ds_cityscapes = ds_cityscapes["train"]
val_ds = ds_cityscapes["val"] if DO_EVAL else None
train_ds = train_ds_cityscapes

# Concatenate synthetic datasets to train_ds
if SYNTHETIC_DATASETS:
    synthetic_dss = [load_dataset(syn_ds, split="train") for syn_ds in SYNTHETIC_DATASETS]
    train_ds = concatenate_datasets([train_ds] + synthetic_dss)
print("Train dataset size: ", train_ds)

if ONLY_TEST:
    test_dss = [load_dataset(test_ds, split="val") for test_ds in TEST_DATASETS]
    print("Test dataset sizes: ", [len(test_ds) for test_ds in test_dss])

train_ds = train_ds.rename_column("image", "pixel_values")
val_ds = val_ds.rename_column("image", "pixel_values") if val_ds else None
test_dss = [test_ds.rename_column("image", "pixel_values") for test_ds in test_dss] if ONLY_TEST else None
print(train_ds)

label_repo_id = "huggingface/label-files"
label_json = "cityscapes-id2label.json"
id2label = json.load(open(hf_hub_download(label_repo_id, label_json, repo_type="dataset"), "r"))
id2label = {int(k):v for k,v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)
print(num_labels, list(label2id.keys()))

model = get_model(
    model_name=MODEL_NAME,
    model_id=MODEL_ID,
    num_classes=num_labels,)
model.to("cuda")

print("Number of parameters in the model: ", sum(p.numel() for p in model.parameters()))
print("Number of trainable parameters in the model: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("Number of frozen parameters in the model: ", sum(p.numel() for p in model.parameters() if not p.requires_grad))

train_augmentations = A.Compose([
    A.RandomCrop(height=CROP_SIZE[0], width=CROP_SIZE[1], p=1.0),  # Random resized cropping
    A.HorizontalFlip(p=0.5),  # Random horizontal flipping
    A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, p=1.0),
    # # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    # A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    # A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    # # A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
    # A.ElasticTransform(p=0.3),
    # A.GridDistortion(p=0.3),
    # A.CLAHE(p=0.3),
])

def base_transforms(image, mask):
    # Convert to PyTorch tensor using torchvision
    image_tensor = T.ToTensor()(image)
    image_tensor = T.Normalize(mean=(0.485, 0.456, 0.406), std=( 0.229, 0.224, 0.225))(image_tensor)
    mask_tensor = torch.as_tensor(mask, dtype=torch.long)
    return image_tensor, mask_tensor
    

def train_transforms(example_batch):
    images = example_batch['pixel_values']
    labels = example_batch['label']
    
    input_images = []
    input_labels = []
    
    for image, label in zip(images, labels):
        image = np.array(image)
        label = np.array(label)
        if USE_AUGMENTATION:
            augmented = train_augmentations(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']

        image_tensor, mask_tensor = base_transforms(image=image, mask=label)
        input_images.append(image_tensor)
        input_labels.append(mask_tensor.long())

    return {
        "pixel_values": torch.stack(input_images),
        "labels": torch.stack(input_labels)
    }

def val_transforms(example_batch):
    images = example_batch['pixel_values']
    labels = example_batch['label']
    
    input_images = []
    input_labels = []
    
    for image, label in zip(images, labels):
        image = np.array(image)
        label = np.array(label)

        image_tensor, mask_tensor = base_transforms(image=image, mask=label)
        input_images.append(image_tensor)
        input_labels.append(mask_tensor.long())

    return {
        "pixel_values": torch.stack(input_images),
        "labels": torch.stack(input_labels)
    }

train_ds.set_transform(train_transforms)
if DO_EVAL:
    val_ds.set_transform(val_transforms)
if ONLY_TEST:
    for test_ds in test_dss:
        test_ds.set_transform(val_transforms)
print("Transforms applied")

num_classes = len(id2label)
confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

def preprocess_logits_for_metrics(logits, labels):
    """
    Instead of accumulating all predictions, update a global confusion matrix.
    Uses torch operations where possible.
    """
    global confusion_matrix
    # Check CPU memory usage
    mem = psutil.virtual_memory()
    if mem.percent > 80:
        raise RuntimeError(f"CPU memory usage is above 80%: {mem.percent}%")
    logits = F.interpolate(logits, size=labels.shape[1:], mode="bilinear", align_corners=False)
    pred_labels = torch.argmax(logits, axis=1)  # (batch, H, W)
    true_labels = labels
    # Flatten
    pred_labels = pred_labels.view(-1)
    true_labels = true_labels.view(-1)
    # Ignore index 255
    mask = true_labels != 255
    pred_labels = pred_labels[mask]
    true_labels = true_labels[mask]
    # Update confusion matrix using torch bincount
    if pred_labels.numel() > 0:
        inds = true_labels * num_classes + pred_labels
        cm = torch.bincount(inds, minlength=num_classes*num_classes).reshape(num_classes, num_classes)
        confusion_matrix += cm.cpu().numpy()
    # Return dummy tensor
    return torch.zeros(1, dtype=torch.long)

def compute_metrics(eval_pred):
    """
    Compute mean IoU and accuracy from the confusion matrix.
    """
    global confusion_matrix
    # IoU per class
    intersection = np.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(axis=1)
    predicted_set = confusion_matrix.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    iou = intersection / (union + 1e-10)
    mean_iou = np.nanmean(iou)
    accuracy = intersection.sum() / (confusion_matrix.sum() + 1e-10)
    # Reset confusion matrix for next eval
    confusion_matrix[:] = 0
    return {
        "mean_iou": mean_iou,
        "accuracy": accuracy,
    }

def collate_fn(examples):
    batch = {}
    batch["pixel_values"] = torch.stack([example["pixel_values"] for example in examples])
    batch["labels"] = torch.stack([example["labels"] for example in examples])
    return batch


# Define training arguments
training_args = TrainingArguments(
    output_dir="./checkpoints",
    #resume_from_checkpoint=RESUME_CHECKPOINT,
    eval_strategy="steps" if DO_EVAL else None,
    eval_steps = LOGGING_STEP_NUM,
    save_strategy="steps",
    save_steps=LOGGING_STEP_NUM,
    learning_rate=LR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.0001,
    #max_grad_norm=0.1,
    logging_dir="./logs",
    logging_steps=LOGGING_STEP_NUM//5,
    save_total_limit=3,
    load_best_model_at_end=True if DO_EVAL else False,
    metric_for_best_model="mean_iou",
    report_to="none",
    remove_unused_columns=True,
    eval_accumulation_steps=8,
    save_safetensors=False if MODEL_NAME == "deeplab" else True,
    eval_do_concat_batches=False,  # NOTE: For IDD, where not all images has the same size
    dataloader_num_workers=max(1, (os.cpu_count() or 2) // 2),
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

if ONLY_EVAL:
    eval_results = trainer.evaluate()
    print("Eval results: ", eval_results)

if not ONLY_EVAL and not ONLY_TEST:
    trainer.train(resume_from_checkpoint=True if RESUME_CHECKPOINT else None)
    # trainer.train(resume_from_checkpoint=True)

# Results:
  # Segformer normal: Cityscapes val: 0.6837, BDD: 0.3484 , IDD: 0.3537
  # Segformer (+drawing): Cityscapes val: 0.6922, BDD: 0.3894, IDD: 0.4243
  # Segformer (+realistic): Cityscapes val: 0.6887, BDD: 0.3845, IDD: 0.4186


if ONLY_TEST:
    train_ds = None
    val_ds = None
    for test_ds in test_dss:
        print(f"Starting manual evaluation for: {test_ds}")
        
        # Reset confusion matrix before starting
        confusion_matrix[:] = 0

        # Create a PyTorch DataLoader (using your collate_fn)
        test_dataloader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=EVAL_BATCH_SIZE, # Use your defined batch size
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=os.cpu_count() // 2 or 1
        )
        
        # 3. Custom Evaluation Loop
        with torch.no_grad():
            for step, batch in enumerate(tqdm(test_dataloader, desc="Evaluating")):
                model.eval()
                
                # Move tensors to GPU
                pixel_values = batch["pixel_values"].to("cuda")
                labels = batch["labels"].to("cuda")
                
                # Forward pass
                logits = model(pixel_values) # assuming output is a logits attribute
                
                # Accumulate metrics (this is your original preprocess logic)
                logits = F.interpolate(logits, size=labels.shape[1:], mode="bilinear", align_corners=False)
                pred_labels = torch.argmax(logits, axis=1)
                true_labels = labels
                
                pred_labels = pred_labels.view(-1)
                true_labels = true_labels.view(-1)
                
                mask = true_labels != 255
                pred_labels = pred_labels[mask]
                true_labels = true_labels[mask]
                
                if pred_labels.numel() > 0:
                    inds = true_labels * num_classes + pred_labels
                    cm = torch.bincount(inds, minlength=num_classes*num_classes).reshape(num_classes, num_classes)
                    confusion_matrix += cm.cpu().numpy()
                    
                    # Explicitly delete all tensors moved/created on GPU
                    del cm, inds
                
                del pixel_values, labels, logits, pred_labels, true_labels, mask
                
                torch.cuda.empty_cache()

        # 5. Compute and print final metrics (using your original compute_metrics logic)
        results = compute_metrics(None) # Pass None as compute_metrics ignores it
        print(f"Test results for {test_ds.column_names}: {results}")
        continue  # no visualization
        assert 1==2, "Return"
        # --- Visualization of test set predictions ---
        os.makedirs("test_images", exist_ok=True)
        def get_palette(num_classes):
            palette = [
                (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153),
                (153,153,153), (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152),
                ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,  0)
            ]
            while len(palette) < num_classes:
                palette.append(tuple(np.random.randint(0, 255, 3)))
            return palette

        palette = get_palette(num_labels)

        def colorize_mask(mask):
            color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            for idx, color in enumerate(palette):
                color_mask[mask == idx] = color
            return color_mask

        test_ds_vis = load_dataset("regpeter/BDD_dataset_val", split="val")

        for i in tqdm(range(len(test_ds))):
            orig_img = np.array(test_ds_vis[i]['image'])
            batch = test_ds[i]
            input_img = batch['pixel_values'].unsqueeze(0).to("cuda")
            with torch.no_grad():
                logits = model(input_img)
                pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
            pred_color = colorize_mask(pred)
            concat_img = np.concatenate([orig_img, pred_color], axis=1)
            out_path = os.path.join("test_images", f"test_{i:04d}.png")
            Image.fromarray(concat_img).save(out_path)

            # Overlay
            overlay = (0.4 * orig_img + 0.6 * pred_color).astype(np.uint8)
            overlay_path = os.path.join("test_images", f"test_{i:04d}_overlay.png")
            Image.fromarray(overlay).save(overlay_path)
            # print(f"Saved {out_path} and {overlay_path}")
        print("Test images saved in test_images folder.")
