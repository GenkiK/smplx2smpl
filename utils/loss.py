import math
from os import path as osp

import numpy as np
import torch
from torch import nn

from utils.rotation import axis_angle_to_matrix


def compute_geodesic_distance(
    pose1: torch.Tensor,
    pose2: torch.Tensor,
    reduction: str = "none",
    eps: float = torch.finfo(torch.float32).eps,
) -> torch.Tensor:
    """
    Compute the Geodesic Distance between two SMPL pose parameters.

    Args:
        pose1 (torch.Tensor): The first pose parameter.
                              Shape is (B, J, 3) in axis-angle representation.
        pose2 (torch.Tensor): The second pose parameter. Same shape as pose1.
        reduction (str): Reduction to apply to the output: 'none' | 'mean' | 'sum'.
                         'sum': Sum of distances across all joints (default)
                         'mean': Mean distance across all joints
                         'none': Return distances for each joint as is (B, J)

    Returns:
        torch.Tensor: Computed Geodesic Distance.
                      If reduction is 'none', the shape is (B, J),
                      otherwise it is a scalar tensor.
    """
    assert pose1.shape == pose2.shape, "pose1 and pose2 must have the same shape."
    assert pose1.ndim == 3, "Input tensors must be 3-dimensional with shape (B, J, 3)."

    # Get batch size and number of joints
    B = pose1.shape[0]

    # Convert axis-angle representation to rotation matrices
    # Reshape (B, J * 3) -> (B * J, 3) for conversion, then reshape back
    R1 = axis_angle_to_matrix(pose1.view(-1, 3)).view(B, -1, 3, 3)
    R2 = axis_angle_to_matrix(pose2.view(-1, 3)).view(B, -1, 3, 3)

    # Compute relative rotation
    # torch.matmul supports broadcasting, so (B, J, 3, 3) works directly
    dR = torch.matmul(R2, R1.transpose(-1, -2))

    # Compute the trace of the relative rotation matrix
    # Using einsum for simplicity: '...ii->...' sums diagonal elements
    traces = torch.einsum("...ii->...", dR)  # Shape: (B, J)

    # Compute angle (Geodesic Distance) from the trace
    # Clamp values to [-1, 1] to handle numerical errors
    cos_theta = (traces - 1) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)  # Shape: (B, J)

    # Improve gradient quality for small angles (theta ≈ 0)
    # For θ→0, the gradient of acos diverges as 1/√(1−x²) ≈ 1/θ ⇒ Replace with Frobenius approximation for smooth gradients.
    small = theta < 1e-4
    if small.any():
        A = 0.5 * (dR - dR.transpose(-1, -2))
        # Frobenius norm is √2 θ + O(θ³)
        alt_theta = (A.square().sum((-2, -1)).sqrt()) / math.sqrt(2.0)
        theta = torch.where(small, alt_theta, theta)

    # Apply the specified reduction
    if reduction == "sum":
        return torch.sum(theta)
    elif reduction == "mean":
        return torch.mean(theta)
    elif reduction == "none":
        return theta
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")


def get_reduction_method(reduction="mean"):
    if reduction == "mean":
        return torch.mean
    elif reduction == "sum":
        return torch.sum
    elif reduction == "none":
        return lambda x: x
    else:
        raise ValueError("Unknown reduction method: {}".format(reduction))


class WeightedMSELoss(nn.Module):
    def __init__(self, reduction="mean", **kwargs):
        super(WeightedMSELoss, self).__init__()
        self.reduce_str = reduction
        self.reduce = get_reduction_method(reduction)

    def forward(self, input, target, weights=None):
        diff = input - target
        if weights is None:
            return diff.pow(2).sum() / diff.shape[0]
        else:
            return (weights.unsqueeze(dim=-1) * diff.pow(2)).sum() / diff.shape[0]


class VertexEdgeLoss(nn.Module):
    def __init__(
        self,
        norm_type="l2",
        gt_edges=None,
        gt_edge_path="",
        est_edges=None,
        est_edge_path="",
        robustifier=None,
        edge_thresh=0.0,
        epsilon=1e-8,
        reduction="sum",
        **kwargs,
    ):
        super(VertexEdgeLoss, self).__init__()

        assert norm_type in ["l1", "l2"], "Norm type must be [l1, l2]"
        self.norm_type = norm_type
        self.epsilon = epsilon
        self.reduction = reduction
        assert self.reduction in ["sum", "mean"]

        gt_edge_path = osp.expandvars(gt_edge_path)
        est_edge_path = osp.expandvars(est_edge_path)
        assert osp.exists(gt_edge_path) or gt_edges is not None, "gt_edges must not be None or gt_edge_path must exist"
        assert osp.exists(est_edge_path) or est_edges is not None, "est_edges must not be None or est_edge_path must exist"
        if osp.exists(gt_edge_path) and gt_edges is None:
            gt_edges = np.load(gt_edge_path)
        if osp.exists(est_edge_path) and est_edges is None:
            est_edges = np.load(est_edge_path)

        self.register_buffer("gt_connections", torch.tensor(gt_edges, dtype=torch.long))
        self.register_buffer("est_connections", torch.tensor(est_edges, dtype=torch.long))

    def extra_repr(self):
        msg = [
            f"Norm type: {self.norm_type}",
        ]
        if self.has_connections:
            msg.append(f"GT Connections shape: {self.gt_connections.shape}")
            msg.append(f"Est Connections shape: {self.est_connections.shape}")
        return "\n".join(msg)

    def compute_edges(self, points, connections):
        edge_points = torch.index_select(points, 1, connections.view(-1)).reshape(points.shape[0], -1, 2, 3)
        return edge_points[:, :, 1] - edge_points[:, :, 0]

    def forward(self, gt_vertices, est_vertices, weights=None):
        gt_edges = self.compute_edges(gt_vertices, connections=self.gt_connections)
        est_edges = self.compute_edges(est_vertices, connections=self.est_connections)

        raw_edge_diff = gt_edges - est_edges

        batch_size = gt_vertices.shape[0]
        if self.norm_type == "l2":
            edge_diff = raw_edge_diff.pow(2)
        elif self.norm_type == "l1":
            edge_diff = raw_edge_diff.abs()
        else:
            raise NotImplementedError(f"Loss type not implemented: {self.loss_type}")
        if self.reduction == "sum":
            return edge_diff.sum()
        elif self.reduction == "mean":
            return edge_diff.sum() / batch_size


def build_loss(type="l2", reduction="mean", **kwargs) -> nn.Module:
    if type == "l2":
        return WeightedMSELoss(reduction=reduction, **kwargs)
    elif type == "vertex-edge":
        return VertexEdgeLoss(reduction=reduction, **kwargs)
    elif type == "l1":
        return nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss type: {type}")
