from pathlib import Path
from typing import Callable

import numpy as np
import smplx
import torch
from smplx.body_models import SMPL
from smplx.utils import SMPLOutput, SMPLXOutput
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from parahome import ParahomeTemporalDataset
from utils.log import summary_closure
from utils.loss import build_loss, compute_geodesic_distance
from utils.optimize import build_optimizer, get_variables, minimize
from utils.plot import plot_3d
from utils.smpl import build_smpl_forward_closure, create_gender2smplx_model
from utils.vertices import apply_deformation_transfer, read_deformation_transfer

Z_OFFSET = 0.8
XY_OFFSET = 0.5


def collate_fn(dict_batch):
    # HACK: assume batch_size is 1
    ret_dict = {}
    for k in dict_batch[0]:
        ret_dict[k] = dict_batch[0][k]
    return ret_dict


def main(
    smplx_dir: str,
    per_frame_result_dir: Path,
    temporal_result_dir: Path,
    use_cuda: bool = False,
    do_visualize: bool = False,
    save_img_dir: Path | None = None,
    window_size: int = 100,
    n_duplicated_frames: int = 10,
    smooth_pose_vel_weight: float = 0.0,
    smooth_pose_acc_weight: float = 0.0,
    smooth_joints_vel_weight: float = 0.0,
    smooth_joints_acc_weight: float = 0.0,
    smooth_verts_vel_weight: float = 0.0,
    smooth_verts_acc_weight: float = 0.0,
    smooth_transl_vel_weight: float = 0.0,
    smooth_transl_acc_weight: float = 0.0,
    first_idx: int | None = None,
    last_idx: int | None = None,
    verbose: bool = False,
    consider_gender: bool = True,
    num_workers: int = 0,
) -> None:
    if not smplx_dir.exists():
        print(f"Session path does not exist: {smplx_dir}. Skipping.", flush=True)
        return

    if save_img_dir is not None:
        do_visualize = True
        save_img_dir.mkdir(parents=True, exist_ok=True)

    device_name = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    print("Using device:", device_name, flush=True)
    device = torch.device(device_name)

    def_matrix = read_deformation_transfer(Path("transfer_data/smplx2smpl_deftrafo_setup.pkl"), device=device)
    gender2smplx_model = create_gender2smplx_model(device)

    dataset = ParahomeTemporalDataset(
        window_size=window_size,
        n_duplicated_frames=n_duplicated_frames,
        per_frame_result_dir=per_frame_result_dir,
        temporal_result_dir=temporal_result_dir,
        session_dir=smplx_dir,
        device=device,
        first_idx=first_idx,
        last_idx=last_idx,
    )
    gender = dataset.gender
    smpl_gender = gender if consider_gender else "neutral"
    n_iters = len(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    smpl_model = smplx.create(
        model_path=f"models/smpl/SMPL_{smpl_gender.upper()}.pkl",
        gender=smpl_gender,
        model_type="smpl",
        batch_size=window_size,
    ).to(device)

    for i_loop, data in tqdm(enumerate(dataloader), total=n_iters, desc=f"Session: {smplx_dir.name}"):
        frame_idxs = data["frame_idxs"]
        smplx_output: SMPLXOutput = gender2smplx_model[gender](
            return_verts=True,
            **{k.replace("smplx_", ""): v for k, v in data.items() if k.startswith("smplx_")},
        )
        smplx_verts = smplx_output.vertices.detach()
        exp_cfg = {
            "verbose": verbose,
            "summary_steps": 50,
            "optim": {
                "type": "lbfgs",
                "maxiters": 100,
                "gtol": 1e-08,
                "ftol": 1e-08,
            },
            "verts_sample_freq": 1,
        }
        var_dict = run_fitting(
            exp_cfg,
            smplx_verts,
            smpl_model,
            def_matrix,
            device,
            prev_result_dict={k.replace("smpl_", ""): v for k, v in data.items() if k.startswith("smpl_")},
            n_duplicated_frames=n_duplicated_frames,
            is_first_batch=(i_loop == 0),
            smooth_pose_vel_weight=smooth_pose_vel_weight,
            smooth_pose_acc_weight=smooth_pose_acc_weight,
            smooth_joints_vel_weight=smooth_joints_vel_weight,
            smooth_joints_acc_weight=smooth_joints_acc_weight,
            smooth_verts_vel_weight=smooth_verts_vel_weight,
            smooth_verts_acc_weight=smooth_verts_acc_weight,
            smooth_transl_vel_weight=smooth_transl_vel_weight,
            smooth_transl_acc_weight=smooth_transl_acc_weight,
        )

        batch_result_transl = var_dict["transl"].detach().cpu().numpy().reshape(window_size, 3)
        batch_result_global_orient = var_dict["global_orient"].detach().cpu().numpy().reshape(window_size, 3)
        batch_result_body_pose = var_dict["body_pose"].detach().cpu().numpy().reshape(window_size, 69)
        batch_result_betas = var_dict["betas"].detach().cpu().numpy().reshape(window_size, 10)

        batch_result_arr = np.empty((window_size, 3 + 72 + 10), dtype=np.float32)
        batch_result_arr[:, :3] = batch_result_transl
        batch_result_arr[:, 3:6] = batch_result_global_orient
        batch_result_arr[:, 6:75] = batch_result_body_pose
        batch_result_arr[:, 75:] = batch_result_betas

        for i, frame_idx in enumerate(frame_idxs):
            save_path = temporal_result_dir / f"{frame_idx:010d}.npz"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(save_path, arr=batch_result_arr[i], gender=smpl_gender)

        if do_visualize:
            for i, frame_idx in enumerate(frame_idxs):
                orig_vertices = smplx_output.vertices[i].detach().cpu().numpy()[::2]  # [n_verts/2, 3]
                fitted_vertices = var_dict["vertices"][i].detach().cpu().numpy()[::2]  # [n_verts/2, 3]
                save_path = save_img_dir / f"{frame_idx:010d}.jpg" if save_img_dir is not None else None
                plot_3d(
                    # gt_verts=orig_vertices,
                    fit_verts=fitted_vertices,
                    save_path=save_path,
                    xmin=dataset.xmin - XY_OFFSET,
                    xmax=dataset.xmax + XY_OFFSET,
                    ymin=dataset.ymin - XY_OFFSET,
                    ymax=dataset.ymax + XY_OFFSET,
                    zmin=dataset.zmin - Z_OFFSET,
                    zmax=dataset.zmax + Z_OFFSET,
                )


def build_temporal_closure(
    smpl_model: SMPL,
    var_dict: dict[str, torch.Tensor],
    optimizer_dict,
    gt_vertices: torch.Tensor,
    vertex_loss: nn.Module,
    params_to_opt: torch.Tensor | None = None,
    verts_sample_freq: int = 1,
    n_duplicated_frames: int = 10,
    is_first_batch: bool = False,
    smooth_pose_vel_weight: float = 0.0,
    smooth_pose_acc_weight: float = 0.0,
    smooth_joints_vel_weight: float = 0.0,
    smooth_joints_acc_weight: float = 0.0,
    smooth_verts_vel_weight: float = 0.0,
    smooth_verts_acc_weight: float = 0.0,
    smooth_transl_vel_weight: float = 0.0,
    smooth_transl_acc_weight: float = 0.0,
    verbose: bool = False,
) -> Callable:
    """Builds the closure for the vertex objective"""
    optimizer = optimizer_dict["optimizer"]
    create_graph = optimizer_dict["create_graph"]

    model_forward = build_smpl_forward_closure(smpl_model, var_dict)

    if params_to_opt is None:
        params_to_opt = [var_dict[key] for key in var_dict]

    def closure(backward=True):
        if backward:
            optimizer.zero_grad()

        model_output: SMPLOutput = model_forward()
        est_vertices = model_output.vertices

        if verts_sample_freq > 1:
            est_vertices = est_vertices[:, ::verts_sample_freq, :]
            new_gt_vertices = gt_vertices[:, ::verts_sample_freq, :]
        else:
            new_gt_vertices = gt_vertices

        loss = vertex_loss(
            est_vertices,
            new_gt_vertices,
        )

        smooth_joints_vel_loss = 0.0
        smooth_joints_acc_loss = 0.0
        smooth_pose_vel_loss = 0.0
        smooth_pose_acc_loss = 0.0
        smooth_verts_vel_loss = 0.0
        smooth_verts_acc_loss = 0.0
        smooth_transl_vel_loss = 0.0
        smooth_transl_acc_loss = 0.0

        if smooth_joints_acc_weight > 0 or smooth_joints_vel_weight > 0:
            joints_vel = model_output.joints[1:] - model_output.joints[:-1]
            if smooth_joints_vel_weight > 0:
                smooth_joints_vel_loss = torch.mean(joints_vel.pow(2)) * smooth_joints_vel_weight
            if smooth_joints_acc_weight > 0:
                joints_acc = joints_vel[1:] - joints_vel[:-1]
                smooth_joints_acc_loss = torch.mean(joints_acc.pow(2)) * smooth_joints_acc_weight
        if smooth_pose_acc_weight > 0 or smooth_pose_vel_weight > 0:
            full_pose = model_output.full_pose  # [bs, 72]
            window_size = full_pose.shape[0]
            full_pose_aa = full_pose.reshape(window_size, -1, 3)
            theta = compute_geodesic_distance(full_pose_aa[:-1], full_pose_aa[1:], reduction="none")
            if smooth_pose_acc_weight > 0:
                smooth_pose_vel_loss = theta.square().mean() * smooth_pose_vel_weight
            if smooth_pose_acc_weight > 0:
                theta_next = compute_geodesic_distance(full_pose_aa[1:-1], full_pose_aa[2:], reduction="none")
                smooth_pose_acc_loss = (theta_next - theta[:-1]).square().mean() * smooth_pose_acc_weight
        if smooth_verts_acc_weight > 0 or smooth_verts_vel_weight > 0:
            verts_vel = model_output.vertices[1:] - model_output.vertices[:-1]
            if smooth_verts_vel_weight > 0:
                smooth_verts_vel_loss = torch.mean(verts_vel.pow(2)) * smooth_verts_vel_weight
            if smooth_verts_acc_weight > 0:
                verts_acc = verts_vel[1:] - verts_vel[:-1]
                smooth_verts_acc_loss = torch.mean(verts_acc.pow(2)) * smooth_verts_acc_weight

        if smooth_transl_vel_weight > 0 or smooth_transl_acc_weight > 0:
            transl_vel = model_output.transl[1:] - model_output.transl[:-1]
            if smooth_transl_vel_weight > 0:
                smooth_transl_vel_loss = torch.mean(transl_vel.pow(2)) * smooth_transl_vel_weight
            if smooth_transl_acc_weight > 0:
                transl_acc = transl_vel[1:] - transl_vel[:-1]
                smooth_transl_acc_loss = torch.mean(transl_acc.pow(2)) * smooth_transl_acc_weight

        if verbose:
            print(f"Loss: {loss.item():.4f}", flush=True)
            print(f"Smooth joints vel loss: {smooth_joints_vel_loss.item():.4f}", flush=True)
            print(f"Smooth joints acc loss: {smooth_joints_acc_loss.item():.4f}", flush=True)
            print(f"Smooth pose vel loss: {smooth_pose_vel_loss.item():.4f}", flush=True)
            print(f"Smooth pose acc loss: {smooth_pose_acc_loss.item():.4f}", flush=True)
            print(f"Smooth verts vel loss: {smooth_verts_vel_loss.item():.4f}", flush=True)
            print(f"Smooth verts acc loss: {smooth_verts_acc_loss.item():.4f}", flush=True)
            print(f"Smooth transl vel loss: {smooth_transl_vel_loss.item():.4f}", flush=True)
            print(f"Smooth transl acc loss: {smooth_transl_acc_loss.item():.4f}", flush=True)

        total_loss = (
            loss
            + smooth_joints_vel_loss
            + smooth_joints_acc_loss
            + smooth_pose_vel_loss
            + smooth_pose_acc_loss
            + smooth_verts_vel_loss
            + smooth_verts_acc_loss
            + smooth_transl_vel_loss
            + smooth_transl_acc_loss
        )

        if backward:
            if create_graph:
                # Use this instead of .backward to avoid GPU memory leaks
                grads = torch.autograd.grad(total_loss, params_to_opt, create_graph=True)
                torch.autograd.backward(params_to_opt, grads, create_graph=True)
            else:
                total_loss.backward()
        if not is_first_batch:
            for k in var_dict:
                if var_dict[k].grad is not None:
                    var_dict[k].grad[:n_duplicated_frames] = 0

        return total_loss

    return closure


def run_fitting(
    exp_cfg,
    smplx_verts: torch.Tensor,
    smpl_model: SMPL,
    def_matrix: torch.Tensor,
    device: torch.device,
    prev_result_dict: dict[str, torch.Tensor],
    n_duplicated_frames: int,
    is_first_batch: bool,
    smooth_pose_vel_weight: float = 0.0,
    smooth_pose_acc_weight: float = 0.0,
    smooth_joints_vel_weight: float = 0.0,
    smooth_joints_acc_weight: float = 0.0,
    smooth_verts_vel_weight: float = 0.0,
    smooth_verts_acc_weight: float = 0.0,
    smooth_transl_vel_weight: float = 0.0,
    smooth_transl_acc_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    batch_size = len(smplx_verts)
    summary_steps = exp_cfg.get("summary_steps")
    verbose = exp_cfg.get("verbose")

    # Get the parameters from the model
    var_dict = get_variables(batch_size, smpl_model, prev_result_dict=prev_result_dict, fix_betas=True)

    # Build the optimizer object for the current batch
    optim_cfg = exp_cfg.get("optim", {})

    def_vertices = apply_deformation_transfer(def_matrix, smplx_verts)

    def log_closure():
        return summary_closure(def_vertices, var_dict, smpl_model)

    temporal_fitting_cfg = exp_cfg.get("temporal_fitting", {})
    temporal_loss = build_loss(**temporal_fitting_cfg)
    temporal_loss = temporal_loss.to(device=device)

    #  Optimize all model parameters with vertex-based loss
    optimizer_dict = build_optimizer(list(var_dict.values()), optim_cfg)
    closure = build_temporal_closure(
        smpl_model,
        var_dict,
        optimizer_dict,
        def_vertices,
        vertex_loss=temporal_loss,
        verts_sample_freq=exp_cfg.get("verts_sample_freq", 1),
        n_duplicated_frames=n_duplicated_frames,
        is_first_batch=is_first_batch,
        smooth_pose_vel_weight=smooth_pose_vel_weight,
        smooth_pose_acc_weight=smooth_pose_acc_weight,
        smooth_joints_vel_weight=smooth_joints_vel_weight,
        smooth_joints_acc_weight=smooth_joints_acc_weight,
        smooth_verts_vel_weight=smooth_verts_vel_weight,
        smooth_verts_acc_weight=smooth_verts_acc_weight,
        smooth_transl_vel_weight=smooth_transl_vel_weight,
        smooth_transl_acc_weight=smooth_transl_acc_weight,
        verbose=verbose,
    )
    minimize(
        optimizer_dict["optimizer"],
        closure,
        params=list(var_dict.values()),
        summary_closure=log_closure,
        summary_steps=summary_steps,
        verbose=verbose,
        **optim_cfg,
    )
    var_dict["vertices"] = smpl_model(return_full_pose=True, **var_dict).vertices
    return var_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SMPL-X fitting on Parahome dataset")
    parser.add_argument("--session_name", type=str, required=True, help="Name of the session to process")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA for fitting")
    parser.add_argument("--do_visualize", action="store_true", help="Visualize the fitting results")
    parser.add_argument("--root_save_img_dir", type=Path, default=None, help="Directory to save visualization images")
    parser.add_argument("--window_size", type=int, default=60, help="Window size for fitting")
    parser.add_argument("--n_duplicated_frames", type=int, default=10, help="Number of duplicated frames for fitting")
    parser.add_argument("--smooth_pose_vel_weight", type=float, default=1000, help="Weight for pose velocity smoothing")
    parser.add_argument("--smooth_pose_acc_weight", type=float, default=1000, help="Weight for pose acceleration smoothing")
    parser.add_argument("--smooth_joints_vel_weight", type=float, default=3000, help="Weight for joints velocity smoothing")
    parser.add_argument("--smooth_joints_acc_weight", type=float, default=3000, help="Weight for joints acceleration smoothing")
    parser.add_argument("--smooth_verts_vel_weight", type=float, default=3000, help="Weight for vertices velocity smoothing")
    parser.add_argument("--smooth_verts_acc_weight", type=float, default=3000, help="Weight for vertices acceleration smoothing")
    parser.add_argument("--smooth_transl_vel_weight", type=float, default=5000, help="Weight for translation velocity smoothing")
    parser.add_argument("--smooth_transl_acc_weight", type=float, default=5000, help="Weight for translation acceleration smoothing")
    parser.add_argument("--consider_gender", action="store_true", help="Consider gender")
    parser.add_argument("--root_smplx_dir", type=Path, required=True, help="Root directory for source data")
    parser.add_argument("--root_per_frame_result_dir", type=Path, required=True, help="Directory for per-frame results")
    parser.add_argument("--root_temporal_result_dir", type=Path, required=True, help="Directory for temporal results")
    parser.add_argument("--first_idx", type=int, default=None, help="Start index for processing frames")
    parser.add_argument("--last_idx", type=int, default=None, help="End index for processing frames")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    args = parser.parse_args()
    main(
        smplx_dir=args.root_smplx_dir / args.session_name,
        per_frame_result_dir=args.root_per_frame_result_dir / args.session_name,
        temporal_result_dir=args.root_temporal_result_dir / args.session_name,
        use_cuda=args.use_cuda,
        do_visualize=args.do_visualize,
        save_img_dir=args.root_save_img_dir / args.session_name if args.root_save_img_dir else None,
        window_size=args.window_size,
        n_duplicated_frames=args.n_duplicated_frames,
        smooth_pose_vel_weight=args.smooth_pose_vel_weight,
        smooth_pose_acc_weight=args.smooth_pose_acc_weight,
        smooth_joints_vel_weight=args.smooth_joints_vel_weight,
        smooth_joints_acc_weight=args.smooth_joints_acc_weight,
        smooth_verts_vel_weight=args.smooth_verts_vel_weight,
        smooth_verts_acc_weight=args.smooth_verts_acc_weight,
        smooth_transl_vel_weight=args.smooth_transl_vel_weight,
        smooth_transl_acc_weight=args.smooth_transl_acc_weight,
        consider_gender=args.consider_gender,
        first_idx=args.first_idx,
        last_idx=args.last_idx,
        verbose=args.verbose,
    )
