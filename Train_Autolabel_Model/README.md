
## Train the Autolabeling model

### Quick start

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Fill the placeholders in ```train.py``` (HF token, model name, hyperparameters, etc.).

(You can create a Huggingface dataset from a local dataset with ```Other_Scripts\hf-dataset-creator.ipynb```.)

 In ```model.py```, set FREEZE_START_LAYER variable based on how many layers to unfreeze in DINOv2/DINOv3.

The *complex* head refers to the convolutional head.

3. Train (example):

```powershell
python Train_Autolabel_Model/train.py
```

Files
- `train.py` — training entrypoint and CLI
- `model.py` — model utilities and checkpointing
- `requirements.txt` — dependencies
- `models/` — model implementations (dino_*, segformer, ensemble)


