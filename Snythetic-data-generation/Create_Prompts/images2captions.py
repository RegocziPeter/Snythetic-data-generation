from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
from huggingface_hub import login


hf_token = "<HF_TOKEN>" # replace with your HuggingFace token
login(hf_token)

# The cityscapes dataset on HF hub should contain the images and the segmentation masks with 1024*512 resolution.
# Each pixel in the segmentation mask should have a value between 0-18 for the classes and 255 for the ignore class.
# The dataset should contain 'train' and 'val' splits. Under them there should be 'image' and 'label' fields.
cityscapes_hf_dataset_id = "<CITYSCAPES_DATASET_ID>"  # replace with your Cityscapes dataset ID on HuggingFace
dataset = load_dataset(cityscapes_hf_dataset_id, split='train')

device = "cuda"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load Florence-2-large model and processor
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    torch_dtype=torch_dtype,
    trust_remote_code=True
).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

def generate_caption(example):
    image = example['image']
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    prompt = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            min_new_tokens=200,
            num_beams=3,
            early_stopping=False,
            do_sample=False,
            #temperature=0.1,  # only if do_sample True
            length_penalty=1.5,      # Optional: encourages longer captions
            repetition_penalty=1.5
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task="<MORE_DETAILED_CAPTION>",
        image_size=(image.width, image.height)
    )["<MORE_DETAILED_CAPTION>"]
    # Florence-2 may return special tokens; clean up if needed
    caption = parsed_answer.strip()
    example['caption'] = caption
    return example

# Apply the caption generation to each example in the dataset
# Set batched=False to process one image at a time (to avoid OOM errors)
new_dataset = dataset.map(generate_caption, batched=False)

# Push to Hugging Face Hub (optional)
new_dataset.push_to_hub("cityscapes_train_1024_512_captioned", private=True)