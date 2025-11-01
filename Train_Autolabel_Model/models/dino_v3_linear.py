import torch


class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1, 1))

    def forward(self, embeddings, original_height, original_width, patch_size):
        # Dynamically calculate height and width based on input size
        batch_size, num_tokens, _ = embeddings.shape
        height = original_height // patch_size
        width = original_width // patch_size
        # print(f"Shape of patch embeddings: {embeddings.shape}")
        
        # expected_tokens = height * width
        # embeddings = embeddings[:, :expected_tokens, :]

        embeddings = embeddings.reshape(batch_size, height, width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2).contiguous()

        return self.classifier(embeddings)


class Dinov3WithLinearHead(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, hidden_size: int, num_classes: int):
        super(Dinov3WithLinearHead, self).__init__()

        self.dinov3 = backbone
        self.classifier = LinearClassifier(hidden_size, num_classes)

    def forward(self, pixel_values=None, images=None, output_hidden_states=False, output_attentions=False, labels=None):
        if images is not None:
            pixel_values = images
        if pixel_values is None:
            raise ValueError("Either pixel_values or images must be provided to the model.")

        outputs = self.dinov3(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )
        # get the patch embeddings - so we exclude the CLS token and registers
        patch_embeddings = outputs.last_hidden_state[:, 5:, :]
        original_height, original_width = pixel_values.shape[2:]

        patch_size = self.dinov3.config.patch_size
        logits = self.classifier(patch_embeddings, original_height, original_width, patch_size)
        logits = torch.nn.functional.interpolate(
            logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False
        )

        if labels is not None:
            if labels.dtype != torch.long:
                labels = labels.long()
            loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=255)
            return loss, logits

        return logits

