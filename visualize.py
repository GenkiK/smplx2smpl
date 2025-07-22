import pickle
from pathlib import Path

import numpy as np
import smplx
import torch
from tqdm import tqdm

from utils.plot import plot_3d

Z_OFFSET = 0.8
XY_OFFSET = 0.5


def main(
    smplx_dir: Path,  # use to extract min and max x, y, z values
    src_dir: Path | None,
    tgt_dir: Path,
    first_idx: int | None,
    last_idx: int | None,
    verts_downsample_rate: int = 1,
):
    tgt_dir.mkdir(parents=True, exist_ok=True)
    model_type = "smplx" if src_dir is None else "smpl"
    with open(smplx_dir / "smplx_pose.pkl", "rb") as f:
        smplx_pose = pickle.load(f)
    with open(smplx_dir / "smplx_params.pkl", "rb") as f:
        smplx_params = pickle.load(f)

    xmin, ymin, zmin = smplx_pose["transl"].amin(0)
    xmax, ymax, zmax = smplx_pose["transl"].amax(0)
    xmin -= XY_OFFSET
    xmax += XY_OFFSET
    ymin -= XY_OFFSET
    ymax += XY_OFFSET
    zmin -= Z_OFFSET
    zmax += Z_OFFSET
    if model_type == "smplx":
        print("Visualizing SMPL-X model", flush=True)
        visualize_smplx(
            smplx_pose,
            smplx_params,
            tgt_dir,
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
            first_idx=first_idx,
            last_idx=last_idx,
            downsample_rate=verts_downsample_rate,
        )
    else:
        print("Visualizing SMPL model", flush=True)
        visualize_smpl(
            src_dir,
            tgt_dir,
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
            first_idx,
            last_idx,
            downsample_rate=verts_downsample_rate,
        )


def visualize_smplx(
    smplx_pose: dict[str, torch.Tensor],
    smplx_params: dict[str, torch.Tensor],
    tgt_dir,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
    zmin: int,
    zmax: int,
    first_idx: int | None = None,
    last_idx: int | None = None,
    downsample_rate: int = 1,
):
    orig_seqlen = len(smplx_pose["transl"])
    last_idx = orig_seqlen - 1 if last_idx is None else last_idx
    first_idx = 0 if first_idx is None else first_idx
    for key in smplx_pose:
        smplx_pose[key] = smplx_pose[key][first_idx : last_idx + 1]

    seqlen = len(smplx_pose["transl"])
    gender: str = smplx_params["gender"]
    betas: torch.Tensor = smplx_params["beta"]

    seq_transl = smplx_pose["transl"]
    seq_global_orient = smplx_pose["global_orient"]
    seq_body_pose = smplx_pose["body_pose"]
    seq_hand_pose = smplx_pose["hand_pose"].reshape(seqlen, 30, 3)  # [seqlen, 30, 3]
    seq_left_hand_pose = seq_hand_pose[:, :15, :].reshape(seqlen, -1)
    seq_right_hand_pose = seq_hand_pose[:, 15:, :].reshape(seqlen, -1)

    smplx_model = smplx.create(
        model_path=f"models/smplx/SMPLX_{gender.upper()}.npz",
        model_type="smplx",
        gender=gender,
        flat_hand_mean=True,
        use_pca=False,
        num_betas=20,
        num_expression_coeffs=10,
        batch_size=1,
    )
    for frame_idx in tqdm(range(seqlen), total=seqlen):
        transl = seq_transl[frame_idx : frame_idx + 1]
        global_orient = seq_global_orient[frame_idx : frame_idx + 1]
        body_pose = seq_body_pose[frame_idx : frame_idx + 1]
        betas = betas
        left_hand_pose = seq_left_hand_pose[frame_idx : frame_idx + 1]
        right_hand_pose = seq_right_hand_pose[frame_idx : frame_idx + 1]
        verts = smplx_model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
        ).vertices
        verts = verts.detach().numpy().squeeze(0)[::downsample_rate]
        plot_3d(
            gt_verts=verts,
            save_path=tgt_dir / f"{frame_idx:010d}.jpg",
            title="GT SMPL-X",
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            zmin=zmin,
            zmax=zmax,
            plot_legend=False,
        )


def visualize_smpl(
    src_dir: Path,
    tgt_dir,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
    zmin: int,
    zmax: int,
    first_idx: int | None = None,
    last_idx: int | None = None,
    downsample_rate: int = 1,
):
    npz_paths = sorted(src_dir.glob("*.npz"))
    orig_seqlen = len(npz_paths)
    last_idx = orig_seqlen - 1 if last_idx is None else last_idx
    first_idx = 0 if first_idx is None else first_idx
    npz_paths = npz_paths[first_idx : last_idx + 1]
    gender: str = np.load(npz_paths[0])["gender"].item()
    smpl_model = smplx.create(
        model_path=f"models/smpl/SMPL_{gender.upper()}.pkl",
        gender=gender,
        model_type="smpl",
        batch_size=1,
    )
    for npz_path in tqdm(npz_paths):
        result_arr = np.load(npz_path)["arr"]
        assert not np.isnan(result_arr).any(), f"NaN values found in result file: {npz_path}"
        transl = result_arr[:3]
        global_orient = result_arr[3:6]
        body_pose = result_arr[6:75]
        betas = result_arr[75:85]

        verts = smpl_model(
            betas=torch.from_numpy(betas).unsqueeze(0),
            global_orient=torch.from_numpy(global_orient).unsqueeze(0),
            body_pose=torch.from_numpy(body_pose).unsqueeze(0),
            transl=torch.from_numpy(transl).unsqueeze(0),
        ).vertices
        verts = verts.detach().numpy().squeeze(0)[::downsample_rate]
        plot_3d(
            fit_verts=verts,
            save_path=tgt_dir / f"{npz_path.stem}.jpg",
            title="Fitted SMPL",
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            zmin=zmin,
            zmax=zmax,
            plot_legend=False,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SMPL-X fitting on Parahome dataset")
    parser.add_argument("--session_name", type=str, required=True, help="Name of the session to process")
    parser.add_argument("--root_smplx_dir", type=Path, required=True, help="Directory containing SMPL-X models")
    parser.add_argument("--root_src_dir", type=Path, default=None, help="Directory of fitted SMPL data. If not provided, will plot SMPL-X data")
    parser.add_argument("--root_save_img_dir", type=Path, required=True, help="Directory to save visualization images")
    parser.add_argument("--first_idx", type=int, default=None, help="Start index for processing frames")
    parser.add_argument("--last_idx", type=int, default=None, help="End index for processing frames")
    parser.add_argument("--verts_downsample_rate", type=int, default=2, help="Downsample rate for vertices in visualization")
    args = parser.parse_args()
    main(
        smplx_dir=args.root_smplx_dir / args.session_name,
        src_dir=args.root_src_dir / args.session_name if args.root_src_dir else None,
        tgt_dir=args.root_save_img_dir / args.session_name,
        first_idx=args.first_idx,
        last_idx=args.last_idx,
        verts_downsample_rate=args.verts_downsample_rate,
    )
