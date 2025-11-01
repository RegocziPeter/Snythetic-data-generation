import os
from datasets import load_dataset, load_from_disk
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

# Parameters
DATASET_NAME = "Flux_image_seg"  # Replace with your dataset name or path
OUTPUT_DIR = "visu_preds_images"

def colorize_mask(mask, num_classes=21):
    # Simple color palette for up to 21 classes (Pascal VOC style)
    palette = np.array([
        [128, 0, 0],      # 1
        [0, 128, 0],      # 2
        [128, 128, 0],    # 3
        [0, 0, 128],      # 4
        [128, 0, 128],    # 5
        [0, 128, 128],    # 6
        [128, 128, 128],  # 7
        [64, 0, 0],       # 8
        [192, 0, 0],      # 9
        [64, 128, 0],     # 10
        [192, 128, 0],    # 11
        [64, 0, 128],     # 12
        [192, 0, 128],    # 13
        [64, 128, 128],   # 14
        [192, 128, 128],  # 15
        [0, 64, 0],       # 16
        [128, 64, 0],     # 17
        [0, 192, 0],      # 18
        [128, 192, 0],    # 19
        [0, 0, 0],        # 20
    ], dtype=np.uint8)
    mask = np.array(mask)
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx in range(num_classes):
        color_mask[mask == class_idx] = palette[class_idx % len(palette)]
    return Image.fromarray(color_mask)

def overlay_images(image, mask, alpha=0.7):
    return Image.blend(image.convert("RGBA"), mask.convert("RGBA"), alpha)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        dataset = load_from_disk(DATASET_NAME)
        print(f"Loaded dataset from disk: {DATASET_NAME}")
    except Exception as e:
        print(f"Could not load from disk, falling back to load_dataset: {e}")
        dataset = load_dataset(DATASET_NAME, split=None)
    for idx, item in enumerate(dataset):
        img = item["image"]
        label = item["label"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        if not isinstance(label, Image.Image):
            label = Image.fromarray(np.array(label))
        color_mask = colorize_mask(label)
        color_mask = color_mask.resize(img.size, resample=Image.NEAREST)
        overlay = overlay_images(img, color_mask, alpha=0.5)
        overlay.save(os.path.join(OUTPUT_DIR, f"{idx:06d}.png"))
        print(f"Saved {idx:06d}.png")

if __name__ == "__main__":
    main()
