from typing import Callable

import torch
from smplx import build_layer
from smplx.body_models import SMPL, SMPLXLayer


def create_gender2smplx_model(device: torch.device) -> dict[str, SMPLXLayer]:
    return {
        gender: build_layer(
            f"models/smplx/SMPLX_{gender.upper()}.npz",
            model_type="smplx",
            flat_hand_mean=True,
            use_pca=False,
            num_betas=20,
            num_expression_coeffs=10,
        ).to(device)
        for gender in ["male", "neutral", "female"]
    }


def build_smpl_forward_closure(smpl_model: SMPL, var_dict: dict[str, torch.Tensor]) -> Callable:
    def smpl_forward():
        return smpl_model(return_full_pose=True, **var_dict)

    return smpl_forward
