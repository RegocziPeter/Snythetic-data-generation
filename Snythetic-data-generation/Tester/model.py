from models.segformer import Segformer
from safetensors.torch import load_file


def get_model(model_name, model_id, num_classes):
    if model_name == "segformer":
        model = Segformer(num_classes=num_classes, version="b0")
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if model_id is not None:
        state_dict = load_file(model_id, device="cuda")
        model.load_state_dict(state_dict)
        model.to("cuda")
    return model
