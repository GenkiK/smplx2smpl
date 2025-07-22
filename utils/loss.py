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
    2つのSMPLポーズパラメータ間のGeodesic Distanceを計算します。

    Args:
        pose1 (torch.Tensor): 1つ目のポーズパラメータ。
                              形状は (B, J, 3) の軸角度表現。
        pose2 (torch.Tensor): 2つ目のポーズパラメータ。pose1と同じ形状。
        reduction (str): 出力に適用するリダクション: 'none' | 'mean' | 'sum'。
                         'sum': 全関節の距離の合計（デフォルト）
                         'mean': 全関節の距離の平均
                         'none': 各関節の距離をそのまま返す (B, J)

    Returns:
        torch.Tensor: 計算されたGeodesic Distance。
                      reductionが 'none' の場合は形状 (B, J) のテンソル、
                      それ以外の場合はスカラーテンソル。
    """
    assert pose1.shape == pose2.shape, "pose1とpose2の形状は一致する必要があります。"
    assert pose1.ndim == 3, "入力テンソルは3次元でなければなりません。形状は (B, J, 3) です。"

    # バッチサイズ、関節数を取得
    B = pose1.shape[0]

    # 軸角度表現を回転行列に変換
    # (B, J * 3) -> (B * J, 3) に変形してから変換し、元の形状に戻す
    R1 = axis_angle_to_matrix(pose1.view(-1, 3)).view(B, -1, 3, 3)
    R2 = axis_angle_to_matrix(pose2.view(-1, 3)).view(B, -1, 3, 3)

    # 相対的な回転を計算
    # torch.matmulはブロードキャストをサポートしているため、(B, J, 3, 3)のままでOK
    dR = torch.matmul(R2, R1.transpose(-1, -2))

    # 相対回転行列のトレースを計算
    # einsumを使うと簡潔に書ける: '...ii->...' は対角成分の和を取る
    traces = torch.einsum("...ii->...", dR)  # 形状: (B, J)

    # トレースから角度（Geodesic Distance）を計算
    # acosの入力は[-1, 1]の範囲にある必要があるため、clampで数치誤差による範囲外の値を丸める
    cos_theta = (traces - 1) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)  # 形状: (B, J)

    # 近接角(theta ≒ 0)での勾配品質向上
    # θ→0 では acos の勾配が 1/√(1−x²) ≈ 1/θ と発散 ⇒ Frobenius 近似に置き換え，滑らかな勾配を確保。
    small = theta < 1e-4
    if small.any():
        A = 0.5 * (dR - dR.transpose(-1, -2))
        # Frobenius norm は √2 θ + O(θ³)
        alt_theta = (A.square().sum((-2, -1)).sqrt()) / math.sqrt(2.0)
        theta = torch.where(small, alt_theta, theta)

    # 指定されたリダクションを適用
    if reduction == "sum":
        return torch.sum(theta)
    elif reduction == "mean":
        return torch.mean(theta)
    elif reduction == "none":
        return theta
    else:
        raise ValueError(f"無効なreductionタイプです: {reduction}")


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
