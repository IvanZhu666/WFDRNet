import torch
import torch.nn.functional as F
from typing import List


def cosine_similarity_loss(output_list: List[torch.Tensor]) -> torch.Tensor:
    if not output_list:
        raise ValueError("The output_list should not be empty.")

    loss = 0.0
    for feature_map in output_list:
        if feature_map.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got {feature_map.ndim}D.")
        spatial_sum = torch.sum(feature_map)
        spatial_area = feature_map.shape[2] * feature_map.shape[3]
        loss += spatial_sum / spatial_area

    return loss


def focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = -1,
        gamma: float = 4.0,
        reduction: str = "mean"
) -> torch.Tensor:

    if inputs.shape != targets.shape:
        raise ValueError("inputs and targets must have the same shape.")

    inputs = inputs.float()
    targets = targets.float()

    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    pt = torch.where(targets == 1, inputs, 1 - inputs)
    focal_weight = (1 - pt).pow(gamma)
    loss = ce_loss * focal_weight

    if alpha >= 0:
        alpha_weight = torch.where(targets == 1, alpha, 1 - alpha)
        loss *= alpha_weight

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError("reduction must be one of ['mean', 'sum', 'none']")
