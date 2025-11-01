from tqdm import tqdm
import os
from utils.model import get_model
import torch
from PIL import Image
from transformers import AutoModel
from huggingface_hub import login
from datasets import Dataset
import numpy as np
from pathlib import Path
import psutil
import torchvision.transforms as transforms

# Parameters
LOGITS_ALREADY_CREATED = False  # Set to True if datasets already exist
DATASETS_ALREADY_CREATED = False  # Set to True if datasets already exist
PUSH_TO_HUB = True
MAX_CPU_RAM = 28 * 1024 * 1024 * 1024  # 28 GB, adjust as needed
model_name = "ensemble"  #"regpeter/DINOv2_complex_autolabel_Cityscapes"
model_type = "ensemble"  # or "dino_v2_MS"
NUM_CLASSES = 20  # Number of classes for segmentation
MSP_THRESHOLD = 0.6  # Threshold for max softmax probability
ENTROPY_THRESHOLD = 0.3  # Threshold for entropy uncertainty method: % value of max_entropy
IGNORE_THRESHOLD = 0.1  # Threshold for ignoring low-confidence pixels in evaluation
M2SR_THRESHOLD = 4  # Threshold for max 2 softmax ratio: It is Max_softmax / Second_Max_softmax

hf_token = "<HF_TOKEN>" # replace with your HuggingFace token
login(hf_token)

# Load the DINOv2 complex model
print(f"Loading model: {model_name}")
model = get_model(model_type, model_name, num_classes=NUM_CLASSES)

# Define image preprocessing transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Set device
device = "cuda"
model = model.to(device)
model.eval()

# Define image folders to process
image_folders = [
    "LIST_OF_IMAGE_FOLDERS"  # Replace with your image folders. Images are generated with use_flux.py
    ]

def check_ram_usage():
    """Check current RAM usage and raise an error if it exceeds the limit."""
    current_ram = psutil.virtual_memory().used
    if current_ram > MAX_CPU_RAM:
        raise RuntimeError(f"CPU RAM usage exceeded: {current_ram / (1024**3):.2f} GB > {MAX_CPU_RAM / (1024**3):.2f} GB")

def process_image(image, model, transform, device):
    """Process a single image and return logits"""
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)

    return logits.cpu().numpy()


def collect_and_process_images():
    """Collect all images and process them, saving logits to disk to save memory."""
    os.makedirs("logits", exist_ok=True)
    data_samples = {"image": [], "logits": []}
    for folder in image_folders:
        print(f"Processing images in {folder}")
        image_files = list(Path(folder).glob("*.png"))
        assert len(image_files) > 0, "Dataset does not contain any png file."

        for image_path in tqdm(image_files, desc=f"{os.path.basename(folder)}"):
            image = Image.open(image_path).convert("RGB")
            logits = process_image(image, model, transform, device)
            # Save logits to disk
            logits_filename = f"logits/{image_path.stem}.npy"
            np.save(logits_filename, logits)
            # Optionally, save image path (not image object) to save memory
            data_samples["image"].append(str(image_path))
            data_samples["logits"].append(logits_filename)
            
    return data_samples

def create_segmentation_labels(data_samples, uncertainty_method: str | None = None, ignore_threshold: dict[str, float] | None = None):
    """
    Generates segmentation labels from model logits for a batch of data samples.
    Args:
        data_samples: {"image": [], "logits": []}
        ignore_threshold (float, optional): If specified, pixels with softmax probability below this threshold
            are set to 255 in the output (ignore class).
    Returns:
        dict: {"image": [], "label": []}
    """

    seg_labels = {"image": [], "label": []}
    uncertainty_mean_list = []
    for image_path, logits_path in tqdm(zip(data_samples["image"], data_samples["logits"]), 
                                        desc=f"Creating segmentation labels, with uncertainty method: {uncertainty_method}",
                                        total=len(data_samples["image"])):
        logit = np.load(logits_path)  # shape [batch_size, num_classes, height, width]
        logit = torch.from_numpy(logit)
        logit = logit.squeeze(0)  # Remove batch dimension
        logit = logit.permute(1, 2, 0)  # Change to [height, width, num_classes] for softmax
        segmentation_labels = torch.argmax(logit, dim=-1)
        if uncertainty_method is not None:
            softmax_logits = torch.softmax(logit, dim=-1)
            if uncertainty_method == "entropy":
                uncertainty = -torch.sum(softmax_logits * torch.log(softmax_logits + 1e-8), dim=-1)
                segmentation_labels[uncertainty > ignore_threshold["entropy"]] = 255
                uncertainty_mean_list.append(uncertainty.mean().item())

                # Set pixels to 255 where the probability of the ignore class (last class) exceeds IGNORE_THRESHOLD
                ignore_class_probs = softmax_logits[..., NUM_CLASSES - 1]
                segmentation_labels[ignore_class_probs > IGNORE_THRESHOLD] = 255

            elif uncertainty_method == "max_softmax":
                certainty = torch.max(softmax_logits, dim=-1).values
                segmentation_labels[certainty < ignore_threshold["max_softmax"]] = 255
                uncertainty_mean_list.append(certainty.mean().item())

                # Set pixels to 255 where the probability of the ignore class (last class) exceeds IGNORE_THRESHOLD
                ignore_class_probs = softmax_logits[..., NUM_CLASSES - 1]
                segmentation_labels[ignore_class_probs > IGNORE_THRESHOLD] = 255

            elif uncertainty_method == "M2SR":
                values, _ = torch.topk(softmax_logits, 2, dim=-1)
                max_softmax = values[..., 0]
                second_max_softmax = values[..., 1]
                certainty = max_softmax / (second_max_softmax + 1e-8)
                segmentation_labels[certainty < ignore_threshold["M2SR"]] = 255
                uncertainty_mean_list.append(certainty.mean().item())

                # Set pixels to 255 where the probability of the ignore class (last class) exceeds IGNORE_THRESHOLD
                ignore_class_probs = softmax_logits[..., NUM_CLASSES - 1]
                segmentation_labels[ignore_class_probs > IGNORE_THRESHOLD] = 255

            else:
                raise ValueError(f"Unsupported uncertainty method: {uncertainty_method}")

        image = Image.open(image_path).convert("RGB")
        seg_map = segmentation_labels
        seg_map[seg_map == (NUM_CLASSES-1)] = 255  # Set ignore class to 255
        segmentation_labels_pil = Image.fromarray(seg_map.numpy().astype(np.uint8), mode='L')
        seg_labels["image"].append(image)
        seg_labels["label"].append(segmentation_labels_pil)
        check_ram_usage()  # Check RAM usage after processing each image

    if uncertainty_method is not None:
        print(f"Mean uncertainty for {uncertainty_method}: {np.mean(uncertainty_mean_list)}")
    return seg_labels

def load_images_and_labels(image_folders, logits_folder="logits"):
    """
    Load images and their corresponding logits from the specified folders.
    Args:
        image_folders (list): List of folders containing images.
        logits_folder (str): Folder containing precomputed logits.
    Returns:
        dict: {"image": [], "logits": []}
    """
    data_samples = {"image": [], "logits": []}
    for folder in image_folders:
        print(f"Loading images from {folder}")
        image_files = list(Path(folder).glob("*.png"))
        assert len(image_files) > 0, "Dataset does not contain any png file."

        for image_path in tqdm(image_files, desc=f"{os.path.basename(folder)}"):
            data_samples["image"].append(str(image_path))
            logits_filename = f"{logits_folder}/{image_path.stem}.npy"
            data_samples["logits"].append(logits_filename)
    
    return data_samples


def main():
    if LOGITS_ALREADY_CREATED:
        print("Loading precomputed logits...")
        data_samples = load_images_and_labels(image_folders)
    else:
        print("Starting image processing and dataset creation...")
        data_samples = collect_and_process_images()

    if not DATASETS_ALREADY_CREATED:
        seg_dataset = create_segmentation_labels(data_samples, uncertainty_method=None, ignore_threshold=None)
        seg_dataset = Dataset.from_dict(seg_dataset)
        seg_dataset_name = "Flux_image_seg_drawing"
        seg_dataset.save_to_disk(f"./{seg_dataset_name}")

        # # Create segmentation labels with max softmax probability uncertainty method
        # seg_ignored_uncertainty_MSP = create_segmentation_labels(data_samples, uncertainty_method="max_softmax", ignore_threshold={"max_softmax": MSP_THRESHOLD})
        # seg_ignored_uncertainty_MSP = Dataset.from_dict(seg_ignored_uncertainty_MSP)
        # seg_ignored_uncertainty_MSP_name = "Flux_image_seg_uncertainty_MSP"
        # seg_ignored_uncertainty_MSP.save_to_disk(f"./{seg_ignored_uncertainty_MSP_name}")

        # # Create segmentation labels with ENTROPY uncertainty method
        # max_entropy = np.log(20)  # Assuming 20 classes, max entropy is log(20) = 2.996
        # entropy_threshold = ENTROPY_THRESHOLD * max_entropy  # Adjust threshold as needed
        # seg_ignored_uncertainty_entropy = create_segmentation_labels(data_samples, uncertainty_method="entropy", ignore_threshold={"entropy": entropy_threshold})
        # seg_ignored_uncertainty_entropy = Dataset.from_dict(seg_ignored_uncertainty_entropy)
        # seg_ignored_uncertainty_entropy_name = "Flux_image_seg_uncertainty_entropy"
        # seg_ignored_uncertainty_entropy.save_to_disk(f"./{seg_ignored_uncertainty_entropy_name}")

        # # Create segmentation labels with max 2 softmax ratio (M2SR) uncertainty method
        # seg_ignored_uncertainty_M2SR = create_segmentation_labels(data_samples, uncertainty_method="M2SR", ignore_threshold={"M2SR": M2SR_THRESHOLD})
        # seg_ignored_uncertainty_M2SR = Dataset.from_dict(seg_ignored_uncertainty_M2SR)
        # seg_ignored_uncertainty_M2SR_name = "Flux_image_seg_uncertainty_M2SR"
        # seg_ignored_uncertainty_M2SR.save_to_disk(f"./{seg_ignored_uncertainty_M2SR_name}")

    if PUSH_TO_HUB:
        seg_dataset_name = "Flux_image_seg_drawing"
        if 'seg_dataset' not in locals():
            seg_dataset = Dataset.load_from_disk(f"./{seg_dataset_name}")
        seg_dataset.push_to_hub(seg_dataset_name, private=True)
    # if PUSH_TO_HUB:
    #     seg_ignored_uncertainty_MSP_name = "Flux_image_seg_uncertainty_MSP"
    #     if 'seg_ignored_uncertainty_MSP' not in locals():
    #         seg_ignored_uncertainty_MSP = Dataset.load_from_disk(f"./{seg_ignored_uncertainty_MSP_name}")
    #     seg_ignored_uncertainty_MSP.push_to_hub(seg_ignored_uncertainty_MSP_name, private=True)
    # if PUSH_TO_HUB:
    #     seg_ignored_uncertainty_entropy_name = "Flux_image_seg_uncertainty_entropy"
    #     if 'seg_ignored_uncertainty_entropy' not in locals():
    #         seg_ignored_uncertainty_entropy = Dataset.load_from_disk(f"./{seg_ignored_uncertainty_entropy_name}")
    #     seg_ignored_uncertainty_entropy.push_to_hub(seg_ignored_uncertainty_entropy_name, private=True)
    # if PUSH_TO_HUB:
    #     seg_ignored_uncertainty_M2SR_name = "Flux_image_seg_uncertainty_M2SR"
    #     if 'seg_ignored_uncertainty_M2SR' not in locals():
    #         seg_ignored_uncertainty_M2SR = Dataset.load_from_disk(f"./{seg_ignored_uncertainty_M2SR_name}")
    #     seg_ignored_uncertainty_M2SR.push_to_hub(seg_ignored_uncertainty_M2SR_name, private=True)

if __name__ == "__main__":
    main()