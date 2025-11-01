## Tester

This folder is dedicated to evaluating whether synthetic data improves the robustness of semantic segmentation models.

### Purpose
The scripts and notebooks here are used to train and test segmentation models (such as DeepLab and SegFormer) on datasets that include synthetic samples. The goal is to analyze if adding synthetic data helps the models generalize better and improves robustness.

### Usage
**1. To train DeepLabv3+ (`Tester/train_deeplab.ipynb`):**
- All datasets need to be in Cityscapes format
- Put the Cityscapes dataset under *./Cityscapes*, the BDD dataset under *./BDD*, and the IDD under *./IDD*
- (optional) If you have Huggingface datasets, use `Other_Scripts/hf-ds-to-cityscapes.ipynb` to create a local (or kaggle) dataset

**2. To train Segformer:**
- Edit parameters as needed in `Tester/train_segformer.py`
- You can change the Segformer version in `Tester/model.py`
