from utils.dino_complex import Dinov2WithComplexHead
from utils.dino_v3_complex import Dinov3WithComplexHead
from utils.dino_segformer import DinoWithSegformerHead
from safetensors.torch import load_file
from transformers import AutoModel
import torch

class EnsembleModel(torch.nn.Module):
	def __init__(self, num_classes, ignore_class=None, uncertainty_method=None, uncertainty_threshold=2.0):
		super().__init__()
		# Always set ignore_class to last class index for consistency with downstream code
		if ignore_class is None:
			ignore_class = num_classes - 1
		self.ignore_class = ignore_class
		self.uncertainty_method = uncertainty_method
		self.uncertainty_threshold = uncertainty_threshold
		if self.ignore_class != num_classes - 1:
			print(f"[WARNING] EnsembleModel: ignore_class ({self.ignore_class}) != num_classes-1 ({num_classes-1}). This may cause ignore mapping issues.")
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

		# if labels is not None:
		# 	loss = torch.tensor(-1.0)
		# 	return loss, avg_logits
		
		if self.uncertainty_method is None:
			return avg_logits

		pixel_uncertainty = None
		if self.uncertainty_method == "variance":
			probs_stack = torch.softmax(logits_stack, dim=2)         # (num_models, batch, classes, H, W)
			variance = torch.var(probs_stack, dim=0)                 # (batch, classes, H, W)
			pixel_uncertainty = variance.mean(dim=1)                 # (batch, H, W)
			# print(f'pixel_uncertainty shape: {pixel_uncertainty.shape}, min: {pixel_uncertainty.min().item()}, max: {pixel_uncertainty.max().item()}, mean: {pixel_uncertainty.mean().item()}')
		elif self.uncertainty_method == "MI":  # Mutual Information
			probs = torch.softmax(logits_stack, dim=2)  # (num_models, batch, classes, H, W)
			avg_probs = torch.mean(probs, dim=0)  # (batch, classes, H, W)
			entropy_avg = -torch.sum(avg_probs * torch.log(avg_probs + 1e-6), dim=1)  # (batch, H, W)
			entropy_per_model = -torch.sum(probs * torch.log(probs + 1e-6), dim=2)  # (num_models, batch, H, W)
			avg_entropy = torch.mean(entropy_per_model, dim=0)  # (batch, H, W)
			pixel_uncertainty = entropy_avg - avg_entropy

		mask = pixel_uncertainty > self.uncertainty_threshold  # (batch, H, W)
		# Uncomment to print percentage of pixels masked:
		# true_pct = (mask.float().mean() * 100).item()
		# print(f"Mask True percentage: {true_pct:.2f}%")
		if mask.any():
			high, low = 10.0, -10.0
			avg_logits = avg_logits.clone()
			# Create a mask for all classes except ignore class
			class_mask = torch.ones(avg_logits.size(1), dtype=torch.bool, device=avg_logits.device)
			class_mask[self.ignore_class] = False
			
			# Set all non-ignore class logits to low where mask is True
			avg_logits[:, class_mask, :, :] = torch.where(
				mask.unsqueeze(1), low, avg_logits[:, class_mask, :, :]
			)
			# Set ignore class logit to high where mask is True
			avg_logits[:, self.ignore_class, :, :] = torch.where(
				mask, high, avg_logits[:, self.ignore_class, :, :]
			)
			# print(1111, avg_logits.permute(0,2,3,1)[mask])
		return avg_logits
