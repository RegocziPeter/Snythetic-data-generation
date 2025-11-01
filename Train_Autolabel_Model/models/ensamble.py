from models.dino_complex import Dinov2WithComplexHead
from models.dino_v3_complex import Dinov3WithComplexHead
from models.dino_segformer import DinoWithSegformerHead
from safetensors.torch import load_file
from transformers import AutoModel
import torch

class EnsembleModel(torch.nn.Module):
	def __init__(self, num_classes):
		super().__init__()
		# Create a separate backbone for each model to avoid weight sharing issues
		backbone1 = AutoModel.from_pretrained("facebook/dinov2-base")
		backbone2 = AutoModel.from_pretrained("facebook/dinov2-base")
		backbone3 = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
		backbone4 = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
		backbone5 = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")

		model1 = Dinov2WithComplexHead(backbone1, hidden_size=backbone1.config.hidden_size, num_classes=num_classes)
		model2 = Dinov2WithComplexHead(backbone2, hidden_size=backbone2.config.hidden_size, num_classes=num_classes)
		model3 = Dinov3WithComplexHead(backbone3, hidden_size=backbone3.config.hidden_size, num_classes=num_classes)
		model4 = Dinov3WithComplexHead(backbone4, hidden_size=backbone4.config.hidden_size, num_classes=num_classes)
		model5 = DinoWithSegformerHead(backbone5, "dinov3", num_classes=num_classes)

		model1_id = "dinov2_complex/model.safetensors"
		model2_id = "dinov2_complex_onlyHead/model.safetensors"
		model3_id = "dinov3_complex/model.safetensors"
		model4_id = "dinov3_complex_onlyHead/model.safetensors"
		model5_id = "dinov3_segformer/model.safetensors"

		model_ids = [model1_id, model2_id, model3_id, model4_id, model5_id]
		models = [model1, model2, model3, model4, model5]
		assert len(model_ids) == len(models), "Number of model IDs must match number of models"

		self.models = torch.nn.ModuleList(models)
		for i, model in enumerate(self.models):
			state_dict = load_file(model_ids[i], device="cuda")
			model.load_state_dict(state_dict)
			model.to("cuda")

	def forward(self, pixel_values=None, images=None, output_hidden_states=False, output_attentions=False, labels=None):
		if images is not None:
			pixel_values = images
		if pixel_values is None:
			raise ValueError("Either pixel_values or images must be provided to the model.")
		logits_list = []
		for model in self.models:
			logits = model(pixel_values)
			logits_list.append(logits)
		logits_stack = torch.stack(logits_list)  # (num_models, batch, classes, H, W)
		avg_logits = torch.mean(logits_stack, dim=0)  # (batch, classes, H, W)

		if labels is not None:
			loss = torch.tensor(-1.0)
			return loss, avg_logits
		return avg_logits
