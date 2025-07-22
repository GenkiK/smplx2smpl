from pathlib import Path
from typing import Callable

import numpy as np
import smplx
import torch
from smplx.body_models import SMPL
from smplx.utils import SMPLXOutput
from torch import nn
from tqdm import tqdm

from parahome import ParahomeDataset
from utils.log import summary_closure
from utils.loss import build_loss
from utils.optimize import build_optimizer, get_variables, minimize
from utils.plot import plot_3d
from utils.smpl import build_smpl_forward_closure, create_gender2smplx_model
from utils.vertices import apply_deformation_transfer, get_vertices_per_edge, read_deformation_transfer

Z_OFFSET = 0.8
XY_OFFSET = 0.5


def main(
    src_dir: Path,
    tgt_dir: Path,
    use_cuda: bool = False,
    do_visualize: bool = False,
    save_img_dir: Path | None = None,
    fit_only_nan: bool = False,
    consider_gender: bool = True,
    first_idx: int | None = None,
    last_idx: int | None = None,
    verbose: bool = False,
) -> None:
    if not src_dir.exists():
        print(f"{src_dir} does not exist. Skipping.", flush=True)
        return

    if save_img_dir is not None:
        do_visualize = True
        save_img_dir.mkdir(parents=True, exist_ok=True)

    device_name = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    print("Using device:", device_name, flush=True)
    device = torch.device(device_name)

    deformation_transfer_path = Path("transfer_data/smplx2smpl_deftrafo_setup.pkl")
    def_matrix = read_deformation_transfer(deformation_transfer_path, device=device)
    gender2smplx_model = create_gender2smplx_model(device)

    dataset = ParahomeDataset(src_dir, device=device, first_idx=first_idx, last_idx=last_idx)
    gender = dataset.gender
    smpl_gender = gender if consider_gender else "neutral"
    seqlen = len(dataset)

    smpl_model = smplx.create(
        model_path=f"models/smpl/SMPL_{smpl_gender.upper()}.pkl",
        gender=smpl_gender,
        model_type="smpl",
        batch_size=1,
    ).to(device)

    prev_result = None
    prev_result_dict = None
    for frame_idx in tqdm(range(seqlen), desc=f"Processing session {src_dir.name}", total=seqlen):
        save_path = tgt_dir / f"{frame_idx:010d}.npz"
        if fit_only_nan and save_path.exists():
            past_result_arr = np.load(save_path)["arr"]
            if not np.isnan(past_result_arr).any():
                prev_result: np.ndarray | None = past_result_arr
                continue
        if prev_result is not None:
            if isinstance(prev_result, np.ndarray):
                prev_result = torch.from_numpy(prev_result).to(device)
            prev_result_dict = {
                "transl": prev_result[:3].unsqueeze(0),
                "global_orient": prev_result[3:6].reshape(1, 1, 3),
                "body_pose": prev_result[6:75].reshape(1, -1, 3),
                "betas": prev_result[75:].unsqueeze(0),
            }
        while_loop_cnt = 0
        smplx_data = dataset[frame_idx]
        while True:
            print(f"Processing index {frame_idx} (attempt {while_loop_cnt})", flush=True)
            smplx_output: SMPLXOutput = gender2smplx_model[gender](
                return_verts=True,
                **smplx_data,
            )
            smplx_verts = smplx_output.vertices.detach()
            exp_cfg = {
                "verbose": verbose,
                "summary_steps": 50,
                "edge_fitting": {},
                "optim": {
                    "type": "lbfgs",
                    "maxiters": 200,
                    "edge_gtol": 1e-05,
                    "vertex_gtol": 1e-02,
                },
                "verts_sample_freq": 1,
            }
            var_dict = run_fitting(exp_cfg, smplx_verts, smpl_model, def_matrix, device, prev_result_dict)

            result_transl = var_dict["transl"].detach().cpu().numpy().ravel()  # [3]
            result_global_orient = var_dict["global_orient"].detach().cpu().numpy().ravel()  # [3]
            result_body_pose = var_dict["body_pose"].detach().cpu().numpy().ravel()  # [69]
            result_betas = var_dict["betas"].detach().cpu().numpy().ravel()  # [10]

            result_arr = np.empty(3 + 72 + 10, dtype=np.float32)
            result_arr[:3] = result_transl
            result_arr[3:6] = result_global_orient
            result_arr[6:75] = result_body_pose
            result_arr[75:] = result_betas

            if not np.isnan(result_arr).any():
                break
            while_loop_cnt += 1

            if while_loop_cnt >= 2:
                print(f"Failed to fit SMPL-X model for index {frame_idx} after 2 attempts. Save with nan.", flush=True)
                break

        save_path = tgt_dir / f"{frame_idx:010d}.npz"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path, arr=result_arr, gender=smpl_gender)

        if not np.isnan(result_arr).any():
            prev_result = result_arr

        if do_visualize:
            orig_vertices = smplx_output.vertices.detach().cpu().numpy().squeeze(0)[::2]  # [n_verts/2, 3]
            fitted_vertices = var_dict["vertices"].detach().cpu().numpy().squeeze(0)[::2]  # [n_verts/2, 3]
            save_path = save_img_dir / f"{frame_idx:010d}.jpg" if save_img_dir is not None else None
            plot_3d(
                gt_verts=orig_vertices,
                fit_verts=fitted_vertices,
                save_path=save_path,
                xmin=dataset.xmin - XY_OFFSET,
                xmax=dataset.xmax + XY_OFFSET,
                ymin=dataset.ymin - XY_OFFSET,
                ymax=dataset.ymax + XY_OFFSET,
                zmin=dataset.zmin - Z_OFFSET,
                zmax=dataset.zmax + Z_OFFSET,
            )


def build_edge_closure(
    body_model: nn.Module,
    var_dict: dict[str, torch.Tensor],
    edge_loss: nn.Module,
    optimizer_dict,
    gt_vertices: torch.Tensor,
) -> Callable:
    """Builds the closure for the edge objective"""
    optimizer = optimizer_dict["optimizer"]
    create_graph = optimizer_dict["create_graph"]

    params_to_opt = [p for key, p in var_dict.items() if "pose" in key]
    model_forward = build_smpl_forward_closure(body_model, var_dict)

    def closure(backward=True):
        if backward:
            optimizer.zero_grad()

        body_model_output = model_forward()
        est_vertices = body_model_output["vertices"]

        loss = edge_loss(est_vertices, gt_vertices)
        if backward:
            if create_graph:
                # Use this instead of .backward to avoid GPU memory leaks
                grads = torch.autograd.grad(loss, params_to_opt, create_graph=True)
                torch.autograd.backward(params_to_opt, grads, create_graph=True)
            else:
                loss.backward()
        return loss

    return closure


def build_vertex_closure(
    body_model: nn.Module,
    var_dict: dict[str, torch.Tensor],
    optimizer_dict,
    gt_vertices: torch.Tensor,
    vertex_loss: nn.Module,
    params_to_opt: torch.Tensor | None = None,
    verts_sample_freq: int = 1,
) -> Callable:
    """Builds the closure for the vertex objective"""
    optimizer = optimizer_dict["optimizer"]
    create_graph = optimizer_dict["create_graph"]

    model_forward = build_smpl_forward_closure(body_model, var_dict)

    if params_to_opt is None:
        params_to_opt = [p for key, p in var_dict.items()]

    def closure(backward=True):
        if backward:
            optimizer.zero_grad()

        body_model_output = model_forward()
        est_vertices = body_model_output["vertices"]

        if verts_sample_freq > 1:
            est_vertices = est_vertices[:, ::verts_sample_freq, :]
            new_gt_vertices = gt_vertices[:, ::verts_sample_freq, :]
        else:
            new_gt_vertices = gt_vertices

        loss = vertex_loss(
            est_vertices,
            new_gt_vertices,
        )
        if backward:
            if create_graph:
                # Use this instead of .backward to avoid GPU memory leaks
                grads = torch.autograd.grad(loss, params_to_opt, create_graph=True)
                torch.autograd.backward(params_to_opt, grads, create_graph=True)
            else:
                loss.backward()

        return loss

    return closure


def run_fitting(
    exp_cfg,
    smplx_verts: torch.Tensor,
    smpl_model: SMPL,
    def_matrix: torch.Tensor,
    device: torch.device,
    prev_result_dict: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    batch_size = len(smplx_verts)
    summary_steps = exp_cfg.get("summary_steps")
    verbose = exp_cfg.get("verbose")

    # Get the parameters from the model
    var_dict = get_variables(batch_size, smpl_model, prev_result_dict=prev_result_dict, fix_betas=False)

    # Build the optimizer object for the current batch
    optim_cfg = exp_cfg.get("optim", {})

    def_vertices = apply_deformation_transfer(def_matrix, smplx_verts)

    f_sel = np.ones_like(smpl_model.faces[:, 0], dtype=np.bool_)
    vpe = get_vertices_per_edge(smpl_model.v_template.detach().cpu().numpy(), smpl_model.faces[f_sel])

    def log_closure():
        return summary_closure(def_vertices, var_dict, smpl_model)

    edge_fitting_cfg = exp_cfg.get("edge_fitting", {})
    edge_loss = build_loss(type="vertex-edge", gt_edges=vpe, est_edges=vpe, **edge_fitting_cfg)
    edge_loss = edge_loss.to(device=device)

    vertex_fitting_cfg = exp_cfg.get("vertex_fitting", {})
    vertex_loss = build_loss(**vertex_fitting_cfg)
    vertex_loss = vertex_loss.to(device=device)

    # Optimize edge-based loss to initialize pose
    if "edge_gtol" in optim_cfg:
        if "gtol" in optim_cfg:
            optim_cfg["gtol_orig"] = optim_cfg["gtol"]
        optim_cfg["gtol"] = optim_cfg["edge_gtol"]

    optimizer_dict = build_optimizer(list(var_dict.values()), optim_cfg)
    closure = build_edge_closure(
        smpl_model,
        var_dict,
        edge_loss,
        optimizer_dict,
        def_vertices,
    )

    minimize(
        optimizer_dict["optimizer"],
        closure,
        params=var_dict.values(),
        summary_closure=log_closure,
        summary_steps=summary_steps,
        verbose=verbose,
        **optim_cfg,
    )

    if "transl" in var_dict:
        optimizer_dict = build_optimizer([var_dict["transl"]], optim_cfg)
        closure = build_vertex_closure(
            smpl_model,
            var_dict,
            optimizer_dict,
            def_vertices,
            vertex_loss=vertex_loss,
            params_to_opt=[var_dict["transl"]],
            verts_sample_freq=exp_cfg.get("verts_sample_freq", 1),
        )
        # Optimize translation
        minimize(
            optimizer_dict["optimizer"],
            closure,
            params=[var_dict["transl"]],
            summary_closure=log_closure,
            summary_steps=summary_steps,
            verbose=verbose,
            **optim_cfg,
        )

    #  Optimize all model parameters with vertex-based loss
    if "gtol_orig" in optim_cfg:
        optim_cfg["gtol"] = optim_cfg.pop("gtol_orig")
    if "vertex_gtol" in optim_cfg:
        optim_cfg["gtol"] = optim_cfg["vertex_gtol"]
    optimizer_dict = build_optimizer(list(var_dict.values()), optim_cfg)
    closure = build_vertex_closure(
        smpl_model,
        var_dict,
        optimizer_dict,
        def_vertices,
        vertex_loss=vertex_loss,
        verts_sample_freq=exp_cfg.get("verts_sample_freq", 1),
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
    var_dict["vertices"] = smpl_model(return_verts=True, **var_dict).vertices
    return var_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SMPL-X fitting on Parahome dataset")
    parser.add_argument("--root_src_dir", type=Path, required=True, help="Root directory of the source data")
    parser.add_argument("--root_tgt_dir", type=Path, required=True, help="Root directory to save the results")
    parser.add_argument("--session_name", type=str, required=True, help="Name of the session to process")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA for fitting")
    parser.add_argument("--do_visualize", action="store_true", help="Visualize the fitting results")
    parser.add_argument("--root_save_img_dir", type=Path, default=None, help="Directory to save visualization images")
    parser.add_argument(
        "--fit_only_nan",
        action="store_true",
        help="Fit frames with NaN values and unfitted frames. This is useful when you want to restart fitting",
    )
    parser.add_argument("--consider_gender", action="store_true", help="Consider gender")
    parser.add_argument("--first_idx", type=int, default=None, help="First index to process (inclusive)")
    parser.add_argument("--last_idx", type=int, default=None, help="Last index to process (inclusive)")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information during the fitting process",
    )
    args = parser.parse_args()
    main(
        src_dir=args.root_src_dir / args.session_name,
        tgt_dir=args.root_tgt_dir / args.session_name,
        use_cuda=args.use_cuda,
        do_visualize=args.do_visualize,
        save_img_dir=args.root_save_img_dir / args.session_name if args.root_save_img_dir else None,
        fit_only_nan=args.fit_only_nan,
        consider_gender=args.consider_gender,
        first_idx=args.first_idx,
        last_idx=args.last_idx,
        verbose=args.verbose,
    )
