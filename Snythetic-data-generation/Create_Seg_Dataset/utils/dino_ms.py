import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class Dinov2WithMSHead(torch.nn.Module):
    def __init__(
        self, backbone: torch.nn.Module, hidden_size: int, num_classes: int, resize_factors: list[float] | None = None
    ):
        super(Dinov2WithMSHead, self).__init__()

        self.dinov2 = backbone
        self.hidden_size = hidden_size
        self.classifier = BNHead(num_classes, hidden_size, resize_factors=resize_factors)

    def forward(self, pixel_values=None, images=None, output_hidden_states=True, output_attentions=False, labels=None):
        # Accept both pixel_values and images for compatibility with HuggingFace Trainer
        if images is not None:
            pixel_values = images
        if pixel_values is None:
            raise ValueError("Either pixel_values or images must be provided to the model.")

        outputs = self.dinov2(pixel_values, output_attentions=output_attentions, output_hidden_states=True)
        # get the last 4 hidden states; exclude the CLS token
        patch_embeddings = [output[:, 1:, :] for output in outputs.hidden_states[-4:]]
        original_height, original_width = pixel_values.shape[2:]
        patch_size = self.dinov2.config.patch_size

        # Reshape to 4D tensor
        batch_size, num_tokens, _ = patch_embeddings[0].shape
        height = original_height // patch_size
        width = original_width // patch_size

        embeddings = [
            patch_embedding.reshape(batch_size, height, width, self.hidden_size) for patch_embedding in patch_embeddings
        ]
        embeddings = [embedding.permute(0, 3, 1, 2).contiguous() for embedding in embeddings]

        logits = self.classifier(embeddings)  # use last 4 layers
        logits = torch.nn.functional.interpolate(
            logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False
        )

        if labels is not None:
            # labels: (batch, H, W), logits: (batch, num_classes, H, W)
            if labels.dtype != torch.long:
                labels = labels.long()
            loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=255)
            return loss, logits

        return logits


class BNHead(torch.nn.Module):
    """Just a batchnorm."""

    def __init__(self, num_classes, channels, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = channels * 4
        self.bn = nn.SyncBatchNorm(self.in_channels)
        self.resize_factors = resize_factors

        self.in_index = [0, 1, 2, 3]
        self.input_transform = "resize_concat"
        self.align_corners = False

        self.conv_seg = nn.Conv2d(self.in_channels, num_classes, kernel_size=1)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # print("inputs", [i.shape for i in inputs])
        x = self._transform_inputs(inputs)
        # print("x", x.shape)
        feats = self.bn(x)
        # print("feats", feats.shape)
        return feats

    def cls_seg(self, feat):
        """Classify each pixel."""
        output = self.conv_seg(feat)
        return output

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == "resize_concat":
            # accept lists (for cls token)
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list
            # an image descriptor can be a local descriptor with resolution 1x1
            for i, x in enumerate(inputs):
                if len(x.shape) == 2:  # Handle 3D tensors
                    inputs[i] = x[:, :, None, None]

            # select indices
            inputs = [inputs[i] for i in self.in_index]
            # Resizing shenanigans
            # print("before", *(x.shape for x in inputs))
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs), (len(self.resize_factors), len(inputs))
                inputs = [
                    resize(input_data=x, scale_factor=f, mode="bilinear" if f >= 1 else "area")
                    for x, f in zip(inputs, self.resize_factors)
                ]
                # print("after", *(x.shape for x in inputs))
            upsampled_inputs = [
                resize(input_data=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output


def resize(input_data, size=None, scale_factor=None, mode="nearest", align_corners=None, warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input_data, size, scale_factor, mode, align_corners)
