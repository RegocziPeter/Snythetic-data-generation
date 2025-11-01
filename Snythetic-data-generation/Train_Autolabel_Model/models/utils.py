import torch


def dice_loss(inputs, targets, ignore_index=255, smooth=1e-6):
    """
    inputs: (N, C, H, W) logits
    targets: (N, H, W) ground truth labels
    """
    # Convert targets to one-hot
    targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
    if ignore_index is not None:
        mask = (targets != ignore_index).unsqueeze(1)
        inputs = inputs * mask
        targets_one_hot = targets_one_hot * mask
    inputs = torch.softmax(inputs, dim=1)
    intersection = (inputs * targets_one_hot).sum(dim=(0,2,3))
    union = inputs.sum(dim=(0,2,3)) + targets_one_hot.sum(dim=(0,2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()