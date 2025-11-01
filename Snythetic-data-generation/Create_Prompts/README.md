# Create_Prompts

Purpose
-------
Quickly generate text prompts (for a text-to-image model) from your images. The two scripts in this folder:

- `images2captions.py`: creates detailed captions from images.
- `captions2prompts.py`: refines captions into varied prompts for text-to-image training or generation.

Quick start
-----------
1) Install dependencies (PowerShell):

```powershell
python -m pip install -r Create_Prompts\requirements.txt
```

2) Edit the scripts and set your Hugging Face token and dataset IDs (search for `hf_token` and dataset placeholders).

(You can create a Huggingface dataset from a local dataset with `Other_Scripts\hf-dataset-creator.ipynb`.)

3) Run (from repo root):

```powershell
python .\Create_Prompts\images2captions.py

python .\Create_Prompts\captions2prompts.py
```

