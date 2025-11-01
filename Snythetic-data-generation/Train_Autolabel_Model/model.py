from transformers import Dinov2Model, AutoModel
from models.dino_linear import Dinov2WithLinearHead
from models.dino_v3_linear import Dinov3WithLinearHead
from models.dino_complex import Dinov2WithComplexHead
from models.dino_v3_complex import Dinov3WithComplexHead
from models.dino_segformer import DinoWithSegformerHead
from models.ensamble import EnsembleModel
from models.segformer import Segformer
from safetensors.torch import load_file


def get_model(model_name, model_id, num_classes):
    if "dino_v2" in model_name:
        FREEZE_START_LAYER = 10

        backbone = Dinov2Model.from_pretrained("facebook/dinov2-base")
        hidden_size = backbone.config.hidden_size
        if model_name == "dino_v2_linear":
            model = Dinov2WithLinearHead(backbone, hidden_size=hidden_size, num_classes=num_classes)
        if model_name == "dino_v2_complex":
            model = Dinov2WithComplexHead(backbone, hidden_size=hidden_size, num_classes=num_classes)
        elif model_name == "dino_v2_segformer":
            model = DinoWithSegformerHead(backbone, "dinov2", num_classes=num_classes)

        for name, param in model.named_parameters():
            if name.startswith("dinov2"):
                param.requires_grad = False
            # Unfreeze the patch_embedding layers
            # if "dinov2.embeddings.patch_embeddings" in name:
            #     param.requires_grad = True
            # Unfreeze the last N encoder layers
            if "dinov2.encoder.layer" in name and int(name.split(".")[3]) >= FREEZE_START_LAYER:
                param.requires_grad = True

            if model_name == "dino_v2_linear" or model_name == "dino_v2_complex":
                # Unfreeze LayerNorm layers, sometimes helps
                if "norm" in name:
                    param.requires_grad = True

    elif "dino_v3" in model_name:
        FREEZE_START_LAYER = 10

        backbone = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
        hidden_size = backbone.config.hidden_size
        if model_name == "dino_v3_linear":
            model = Dinov3WithLinearHead(backbone, hidden_size=hidden_size, num_classes=num_classes)
        elif model_name == "dino_v3_complex":
            model = Dinov3WithComplexHead(backbone, hidden_size=hidden_size, num_classes=num_classes)
        elif model_name == "dino_v3_segformer":
            model = DinoWithSegformerHead(backbone, "dinov3", num_classes=num_classes)
        for name, param in model.named_parameters():
            if name.startswith("dinov3"):
                param.requires_grad = False
            # Unfreeze the patch_embedding layers
            # if "dinov3.embeddings.patch_embeddings" in name:
            #     param.requires_grad = True
            # Unfreeze the last 4 encoder layers
            if "dinov3.layer" in name and int(name.split(".")[2]) >= FREEZE_START_LAYER:
                param.requires_grad = True

            if model_name == "dino_v3_linear" or model_name == "dino_v3_complex" or model_name == "dino_v3_segformer":
                # Unfreeze LayerNorm layers, usually helps
                if "norm" in name:
                    param.requires_grad = True

    if model_name == "ensemble":
        model = EnsembleModel(num_classes=num_classes)
    
    elif model_name == "segformer":
        model = Segformer(num_classes=num_classes, version="b1")

    if model_id is not None:
        state_dict = load_file(model_id, device="cuda")
        model.load_state_dict(state_dict)
        model.to("cuda")
    return model
