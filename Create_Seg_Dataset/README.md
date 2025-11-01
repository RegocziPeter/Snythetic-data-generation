# Create_Seg_Dataset

This folder contains scripts for generating images and labeling them using an ensemble of DINO-based autolabeling models.

## Features
- Generate synthetic image datasets with FLUX.1-dev
- Label images automatically with ensemble of DINO models
- Utilities for visualization

## Setup
```powershell
pip install -r requirements.txt
```

## How to Use
1. **Generate images: `use_flux.py`:** Edit the scripts and set your Hugging Face token, other parameters and the prompt (realistic, drawing style, intersection, etc.).

2. **Create semantic segmentation labels: `run_labeling.py`:**
- Edit the scripts and set your Hugging Face token and other parameters
- Use the same image folders, created by *use_flux.py*.
- In `utils/ensemble.py`, set the models that you want to use in the ensemble
- In `utils/model.py`, the ensemble-based uncertainty method can be set, the non-ensemble-based uncertainty methods can be set in *run_labeling.py*

3. (optional) **Visualize the overlayed images and masks: `visu_preds.py`.**