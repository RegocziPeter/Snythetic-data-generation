import os
from utils.dino_mlp import Dinov2WithMLPHead
from utils.dino_complex import Dinov2WithComplexHead
from huggingface_hub import hf_hub_download
from utils.dino_linear import Dinov2WithLinearHead
from utils.dino_ms import Dinov2WithMSHead
from transformers import Dinov2Model
from safetensors.torch import load_file


def get_model(model_name, model_id, num_classes):
    backbone = Dinov2Model.from_pretrained("facebook/dinov2-base")
    hidden_size = backbone.config.hidden_size
    if model_name == "dino_v2_linear":
        model = Dinov2WithLinearHead(backbone, hidden_size=hidden_size, num_classes=num_classes)
    elif model_name == "dino_v2_complex":
        model = Dinov2WithComplexHead(backbone, hidden_size=hidden_size, num_classes=num_classes)
    elif model_name == "dino_v2_MS":
        model = Dinov2WithMSHead(
            backbone, hidden_size=hidden_size, num_classes=num_classes, resize_factors=None #[1, 0.5, 0.75, 1.5]
        )
    elif model_name == "ensemble":
        from utils.ensamble import EnsembleModel
        uncertainty_method = "variance"  # "entropy" or "variance" or None
        if uncertainty_method == "MI":
            uncertainty_threshold = 0.4
        elif uncertainty_method == "variance":
            uncertainty_threshold = 0.015
        else:
            uncertainty_threshold = None
        model = EnsembleModel(num_classes=num_classes, ignore_class=None, uncertainty_method=uncertainty_method, uncertainty_threshold=uncertainty_threshold)
        return model
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    # If model_id is a local file, use it directly; otherwise, download 'model.safetensors' from HuggingFace Hub
    if os.path.isfile(model_id):
        safetensor_path = model_id
    else:
        safetensor_path = hf_hub_download(repo_id=model_id, filename="model.safetensors", resume_download=True)
    state_dict = load_file(safetensor_path, device="cuda")
    model.load_state_dict(state_dict)
    model.to("cuda")
    return model
