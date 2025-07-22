from pathlib import Path

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm

from utils.data import split_indices_with_overlap

PELVIS_IDX = 0


def count_consecutive_true(mask: torch.Tensor):
    # Trueに対応する要素にはそのTrueが属する塊の大きさ，Falseに対応する要素には0が入ったテンソルを返す

    diff = torch.diff(mask.int(), prepend=torch.tensor([0], dtype=torch.int32), append=torch.tensor([0], dtype=torch.int32))  # 端を追加して差分を取る
    starts = torch.where(diff == 1)[0]  # True の開始位置
    ends = torch.where(diff == -1)[0]  # True の終了位置

    # ブロードキャストを用いて True の塊のサイズを設定
    indices = torch.arange(len(mask))[:, None]  # 配列のインデックス
    mask = (indices >= starts) & (indices < ends)  # True の範囲をマスク
    out = torch.sum(mask * (ends - starts), dim=1)  # 塊のサイズを適用
    return out


def smoothen(
    b_transl: torch.Tensor,
    b_global_orient: torch.Tensor,
    b_pose: torch.Tensor,
    b_betas: torch.Tensor,
    b_joints: torch.Tensor,
    thresh: float = 0.2,
    thresh_betas: float = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # smplifyx.smoothen.smoothen_dataと基本的に同じだが，pose_embeddingを受け取らない（代わりにbetas, pose）・jointがopenposeを仮定しないという違いがある．

    # b_transl: [bs, 3]
    # b_global_orient: [bs, 3]
    # b_joints: [bs, 24, 3]
    # b_pose: [bs, 23, 3]
    # b_betas: [bs, 10]

    # 1. jointsを見て動作の滑らかさを見て異常値を決定する
    bs = b_joints.shape[0]
    b_outlier_mask = torch.zeros(bs, dtype=torch.bool)
    b_pelvis_outlier_mask = torch.zeros(bs, dtype=torch.bool)
    b_vel = b_joints[1:] - b_joints[:-1]  # [bs-1, n_joints, 3]
    b_acc = b_vel[1:] - b_vel[:-1]  # [bs-2, n_joints, 3]
    b_acc_norm = torch.norm(b_acc, dim=-1)  # [bs-2, n_joints]
    b_outlier_mask[1:-1] = (b_acc_norm > thresh).sum(-1) > 0
    b_pelvis_outlier_mask[1:-1] = b_acc_norm[:, PELVIS_IDX] > thresh

    # 1.5. b_betasの変化量が大きい場合も異常値とする
    b_betas_vel_norm = torch.norm(b_betas[1:] - b_betas[:-1], dim=-1)  # [bs-1, n_betas] -> [bs-1]
    ## 末尾が異常だとnext_idxsがout of rangeになるので，末尾は考慮しない．次にwindowに任せる
    b_outlier_mask[1:-1] = torch.logical_or(b_betas_vel_norm > thresh_betas, b_outlier_mask[1:])[:-1]  # [bs-2]

    if (~b_outlier_mask).all():
        return b_transl, b_global_orient, b_pose, b_betas

    # b_outlier_maskはTrueが連続している可能性もある
    head_outlier_idxs = torch.where(torch.diff(b_outlier_mask.int()) == 1)[0] + 1
    n_consecutive_outliers = count_consecutive_true(b_outlier_mask)[head_outlier_idxs]
    prev_idxs = head_outlier_idxs - 1
    next_idxs = head_outlier_idxs + n_consecutive_outliers
    b_inlier_mask = ~b_outlier_mask

    # 2. b_translを線形補間
    b_transl_cp = b_transl.clone()
    b_transl_cp[b_outlier_mask] = torch.repeat_interleave((b_transl[prev_idxs] + b_transl[next_idxs]) / 2, repeats=n_consecutive_outliers, dim=0)
    # root jointが異常でなければ補間しない．実際は元に戻している
    b_transl_cp[~b_pelvis_outlier_mask] = b_transl[~b_pelvis_outlier_mask]
    b_transl = b_transl_cp

    # 3. b_global_orientをslerpで補間
    b_inlier_idxs_np = torch.where(b_inlier_mask)[0].float().numpy()
    b_global_orient_inlier_scipy = R.from_rotvec(b_global_orient[b_inlier_mask].float())
    slerp = Slerp(b_inlier_idxs_np, b_global_orient_inlier_scipy)
    b_global_orient[b_outlier_mask] = torch.from_numpy(slerp(torch.where(b_outlier_mask)[0].float().numpy()).as_rotvec().astype(np.float32))

    # 4. b_poseをslerpで補間
    n_pose_params = b_pose.shape[1]
    for pose_idx in range(n_pose_params):
        b_pose_inlier_scipy = R.from_rotvec(b_pose[b_inlier_mask, pose_idx, :].float())
        slerp = Slerp(b_inlier_idxs_np, b_pose_inlier_scipy)
        b_pose[b_outlier_mask, pose_idx, :] = torch.from_numpy(slerp(torch.where(b_outlier_mask)[0].float().numpy()).as_rotvec().astype(np.float32))

    # 5. b_betasを平均値で補間
    b_betas[b_outlier_mask] = b_betas[b_inlier_mask].mean(dim=0)
    return b_transl, b_global_orient, b_pose, b_betas


def main(
    src_dir: Path,
    tgt_dir: Path,
    window_size: int,
    n_duplicated_frames: int,
) -> None:
    smpl_model = smplx.create(
        gender="neutral",
        model_path="models/smpl/SMPL_NEUTRAL.pkl",
        model_type="smpl",
        batch_size=window_size,
    )
    tgt_dir.mkdir(exist_ok=True, parents=True)
    if not src_dir.exists():
        print(f"Session path does not exist: {src_dir}. Skipping.", flush=True)
        return

    seqlen = len(list(src_dir.glob("*.npz")))
    windows = split_indices_with_overlap(seqlen, window_size, n_duplicated_frames)

    for window_idxs in tqdm(windows, desc=f"Smoothing {src_dir.name}"):
        window_interp_smpl_arr = np.empty((window_size, 85), dtype=np.float32)
        for i, frame_idx in enumerate(window_idxs):
            if (tgt_dir / f"{frame_idx:010d}.npz").exists():
                # すでにスムージング済みのフレームはスキップ
                interp_smpl_path = tgt_dir / f"{frame_idx:010d}.npz"
            else:
                interp_smpl_path = src_dir / f"{frame_idx:010d}.npz"
            data = np.load(interp_smpl_path)
            interp_smpl_arr = data["arr"]
            gender = data["gender"]
            assert not np.isnan(interp_smpl_arr).any(), f"NaN found in {interp_smpl_path}"
            window_interp_smpl_arr[i] = interp_smpl_arr

        window_transl = torch.from_numpy(window_interp_smpl_arr[:, :3])
        window_global_orient = torch.from_numpy(window_interp_smpl_arr[:, 3:6])
        window_pose = torch.from_numpy(window_interp_smpl_arr[:, 6:75])
        window_betas = torch.from_numpy(window_interp_smpl_arr[:, 75:85])
        window_joints = smpl_model(
            transl=window_transl,
            global_orient=window_global_orient,
            body_pose=window_pose,
            betas=window_betas,
        ).joints
        window_transl, window_global_orient, window_pose, window_betas = smoothen(
            b_transl=window_transl,
            b_global_orient=window_global_orient,
            b_pose=window_pose.reshape(window_size, -1, 3),
            b_betas=window_betas,
            b_joints=window_joints,
        )
        window_interp_smpl_arr[:, :3] = window_transl.numpy()
        window_interp_smpl_arr[:, 3:6] = window_global_orient.numpy()
        window_interp_smpl_arr[:, 6:75] = window_pose.reshape(window_size, -1).numpy()
        window_interp_smpl_arr[:, 75:85] = window_betas.numpy()

        for i, frame_idx in enumerate(window_idxs):
            smoothed_smpl_path = tgt_dir / f"{frame_idx:010d}.npz"
            np.savez(smoothed_smpl_path, arr=window_interp_smpl_arr[i].astype(np.float32), gender=gender)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SMPL-X fitting on Parahome dataset")
    parser.add_argument("--root_src_dir", type=Path, required=True, help="Name of the session to process")
    parser.add_argument("--root_tgt_dir", type=Path, required=True, help="Name of the session to process")
    parser.add_argument("--session_name", type=str, required=True, help="Name of the session to process")
    parser.add_argument("--window_size", type=int, default=100, help="Size of the window for smoothing")
    parser.add_argument("--n_duplicated_frames", type=int, default=20, help="Number of duplicated frames for smoothing")
    args = parser.parse_args()
    main(
        src_dir=args.root_src_dir / args.session_name,
        tgt_dir=args.root_tgt_dir / args.session_name,
        window_size=args.window_size,
        n_duplicated_frames=args.n_duplicated_frames,
    )
