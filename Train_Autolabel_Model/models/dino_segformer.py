import json
from models.utils import dice_loss
import torch
import os
from transformers import SegformerDecodeHead, SegformerConfig
from huggingface_hub import hf_hub_download


class DinoWithSegformerHead(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, backbone_type: str, num_classes: int, feature_layers=[-7, -5, -3, -1]):
        super(DinoWithSegformerHead, self).__init__()
        device = torch.device("cuda")
        self.backbone = backbone.to(device)
        self.backbone_type = backbone_type
        # Freeze backbone parameters
        for name, param in backbone.named_parameters():
            print(name)
            # dino v2
            if "encoder.layer" in name:
                if int(name.split(".")[2]) >= 10:
                    param.requires_grad = True
                else: param.requires_grad = False
            elif "norm" in name:
                param.requires_grad = True
            # dino v3
            elif "layer" in name and int(name.split(".")[1]) >= 10:
                param.requires_grad = True
            elif "norm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
       
        self.feature_layers = feature_layers

        label_repo_id = "huggingface/label-files"
        label_json = "cityscapes-id2label.json"
        id2label = json.load(open(hf_hub_download(label_repo_id, label_json, repo_type="dataset"), "r"))
        id2label = {int(k):v for k,v in id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
        id2label[19] = "ignore"
        label2id["ignore"] = 19

        hidden_sizes = self.backbone.config.hidden_size
        self.hidden_sizes = hidden_sizes
        segformer_config = SegformerConfig(
            hidden_sizes=[hidden_sizes]*4,
            feature_strides=[4, 8, 16, 32],
            id2label=id2label,
            label2id=label2id
        )
        self.decode_head = SegformerDecodeHead(
            config=segformer_config
        )

    def forward(self, pixel_values=None, images=None, output_hidden_states=False, output_attentions=False, labels=None):
        if images is not None:
            pixel_values = images
        if pixel_values is None:
            raise ValueError("Either pixel_values or images must be provided to the model.")

        # 1. Extract features from backbone
        features = self.backbone(pixel_values, output_hidden_states=True)
        encoder_hidden_states = features.hidden_states

        # 2. Select and reshape features from specified layers
        selected_features = []
        original_height, original_width = pixel_values.shape[2:]
        patch_size = self.backbone.config.patch_size
        hidden_size = self.backbone.config.hidden_size
        for i, idx in enumerate(self.feature_layers):
            feat = encoder_hidden_states[idx]
            if self.backbone_type is "dinov2":
                feat = feat[:, 1:, :]  # Exclude CLS token
            elif self.backbone_type is "dinov3":
                feat = feat[:, 5:, :]  # Exclude CLS and register tokens
            else:
                raise ValueError("Unsupported backbone type")
            batch_size, num_tokens, _ = feat.shape
            height = original_height // patch_size
            width = original_width // patch_size
            hs = hidden_size if isinstance(hidden_size, int) else hidden_size[idx]
            embeddings = feat.reshape(batch_size, height, width, hs)
            embeddings = embeddings.permute(0, 3, 1, 2).contiguous()
            selected_features.append(embeddings)

        # 3. Pass features to Segformer head
        logits = self.decode_head(selected_features)
        logits = torch.nn.functional.interpolate(
            logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False
        )

        if labels is not None:
            if labels.dtype != torch.long:
                labels = labels.long()
            loss = dice_loss(logits, labels, ignore_index=255)
            return loss, logits

        return logits

