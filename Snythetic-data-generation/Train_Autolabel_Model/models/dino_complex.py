from models.utils import dice_loss
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexClassifier(nn.Module):
    def __init__(self, in_channels, num_labels=1, hidden_channels=256, dropout=0.1):
        super(ComplexClassifier, self).__init__()

        self.in_channels = in_channels

        self.classifier = nn.Sequential(
            # Expand to hidden dimension
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),

            nn.Dropout(dropout),

            # Another conv for context
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),

            nn.Dropout(dropout),

            # Project to output classes
            nn.Conv2d(hidden_channels, num_labels, kernel_size=1)
        )

    def forward(self, embeddings, original_height, original_width, patch_size):
        # Dynamically calculate height and width based on input size
        batch_size, num_tokens, _ = embeddings.shape
        height = original_height // patch_size
        width = original_width // patch_size

        embeddings = embeddings.reshape(batch_size, height, width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2).contiguous()

        return self.classifier(embeddings)


class Dinov2WithComplexHead(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_classes: int):
        super(Dinov2WithComplexHead, self).__init__()

        self.dinov2 = backbone
        self.classifier = ComplexClassifier(hidden_size, num_classes)

    def forward(self, pixel_values=None, images=None, output_hidden_states=False, output_attentions=False, labels=None):
        if images is not None:
            pixel_values = images
        if pixel_values is None:
            raise ValueError("Either pixel_values or images must be provided to the model.")

        outputs = self.dinov2(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )
        # exclude CLS token
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]
        original_height, original_width = pixel_values.shape[2:]

        patch_size = self.dinov2.config.patch_size
        logits = self.classifier(patch_embeddings, original_height, original_width, patch_size)
        logits = F.interpolate(
            logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False
        )

        if labels is not None:
            if labels.dtype != torch.long:
                labels = labels.long()
            loss = dice_loss(logits, labels, ignore_index=255)
            return loss, logits

        return logits
