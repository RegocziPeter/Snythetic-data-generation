import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig


class Segformer(nn.Module):
    def __init__(self, num_classes: int, version: str):
        super(Segformer, self).__init__()

        assert version in ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"], "Version must be one of ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']"
        
        # Create SegFormer B0 specific configuration
        config = SegformerConfig.from_pretrained(
            f"nvidia/segformer-{version}-finetuned-ade-512-512",
            num_labels=num_classes
        )
        
        # Initialize model and load pre-trained SegFormer B0 encoder weights
        # This follows the original approach: pre-trained encoder + random decoder
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            f"nvidia/mit-{version}",  # SegFormer B0 encoder weights
            config=config,
            ignore_mismatched_sizes=True  # Ignore decoder head size mismatch
        )
    
    def forward(self, pixel_values=None, images=None, labels=None, output_hidden_states=False, output_attentions=False):
        # Handle both pixel_values and images input formats for compatibility
        if images is not None:
            pixel_values = images
        if pixel_values is None:
            raise ValueError("Either pixel_values or images must be provided to the model.")
        
        # Forward pass through the model with SegFormer's built-in loss
        outputs = self.model(
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        
        logits = outputs.logits
        
        # Upsample logits to match input size
        if logits.shape[2:] != pixel_values.shape[2:]:
            logits = torch.nn.functional.interpolate(
                logits, 
                size=pixel_values.shape[2:], 
                mode="bilinear", 
                align_corners=False
            )
        
        # Return loss and logits if labels provided, otherwise just logits
        if labels is not None:
            return outputs.loss, logits
        
        return logits