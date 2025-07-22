from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm


def interpolate_points(p0: np.ndarray, p1: np.ndarray, n_nan_frames: int):
    """
    Linearly interpolate between two 3D points p0 and p1 at equal intervals.

    Parameters:
        p0 (array-like): Starting point (length 3)
        p1 (array-like): Ending point (length 3)
        n_nan_frames (int): Number of interpolation points (excluding the endpoints)

    Returns:
        np.ndarray: Interpolated array of shape (n_frames x 3)
    """
    n_frames = n_nan_frames + 2  # Includes both endpoints
    p0 = np.asarray(p0)
    p1 = np.asarray(p1)
    alphas = np.linspace(0, 1, n_frames)[:, None]  # (n_frames, 1)
    return (1 - alphas) * p0 + alphas * p1


def interp_nan(src_dir: Path, tgt_dir: Path) -> None:
    tgt_dir.mkdir(exist_ok=True, parents=True)
    if not src_dir.exists():
        print(f"Session path does not exist: {src_dir}. Skipping.", flush=True)
        return

    is_nan_seq = False
    nan_frame_idxs = []
    prev_result = None
    for frame_idx, smpl_path in tqdm(enumerate(sorted(src_dir.glob("*.npz")))):
        data = np.load(smpl_path)
        smpl_arr = data["arr"]
        gender = data["gender"]
        if np.isnan(smpl_arr).any():  # nan
            nan_frame_idxs.append(frame_idx)
            is_nan_seq = True
        elif not is_nan_seq:  # not nan, not is_nan_seq
            prev_result = smpl_arr
            interp_path = tgt_dir / f"{frame_idx:010d}.npz"
            np.savez(interp_path, arr=smpl_arr.astype(np.float32), gender=gender)
        else:  # not nan, is_nan_seq; nan sequence has ended, so interpolate
            is_nan_seq = False
            next_result = smpl_arr
            print(nan_frame_idxs, flush=True)

            if prev_result is None:
                # If nan starts from frame_idx=0
                for i, nan_frame_idx in enumerate(nan_frame_idxs):
                    if i == 0:
                        assert nan_frame_idx == 0, "The frame before nan should not be nan"
                    interp_result = smpl_arr.astype(np.float32)
                    interp_path = tgt_dir / f"{i:010d}.npz"
                    np.savez(interp_path, arr=interp_result, gender=gender)
            else:
                # Linearly interpolate transl
                prev_transl = prev_result[:3]
                next_transl = next_result[:3]
                num_nans = len(nan_frame_idxs)
                interp_transl = interpolate_points(prev_transl, next_transl, num_nans)

                # Interpolate global_orient using slerp
                global_orient_prev_next = R.from_rotvec(np.stack([prev_result[3:6], next_result[3:6]], axis=0))
                slerp = Slerp(np.array([nan_frame_idxs[0] - 1, nan_frame_idxs[-1] + 1]).astype(np.float32), global_orient_prev_next)
                interp_global_orient = slerp(np.asarray(nan_frame_idxs).astype(np.float32)).as_rotvec().astype(np.float32)

                # Interpolate pose_params using slerp
                prev_pose = prev_result[6:75].reshape(-1, 3)
                next_pose = next_result[6:75].reshape(-1, 3)
                interp_pose = np.empty((num_nans, 23, 3), dtype=np.float32)  # 69 is the number of SMPL pose_params
                n_pose_params = prev_pose.shape[0]
                for pose_idx in range(n_pose_params):
                    pose_prev_next = R.from_rotvec(np.stack([prev_pose[pose_idx], next_pose[pose_idx]], axis=0))
                    slerp = Slerp(np.array([nan_frame_idxs[0] - 1, nan_frame_idxs[-1] + 1]).astype(np.float32), pose_prev_next)
                    interp_pose[:, pose_idx, :] = slerp(np.asarray(nan_frame_idxs).astype(np.float32)).as_rotvec().astype(np.float32)

                # Interpolate betas using the average
                prev_betas = prev_result[75:]
                next_betas = next_result[75:]
                interp_betas = (prev_betas + next_betas) / 2

                for i, nan_frame_idx in enumerate(nan_frame_idxs):
                    interp_result = np.concatenate(
                        [
                            interp_transl[i],
                            interp_global_orient[i],
                            interp_pose[i].ravel(),
                            interp_betas,
                        ]
                    )
                    interp_result = interp_result.astype(np.float32)
                    interp_path = tgt_dir / f"{nan_frame_idx:010d}.npz"
                    np.savez(interp_path, arr=interp_result, gender=gender)

            np.savez(tgt_dir / f"{frame_idx:010d}.npz", arr=smpl_arr.astype(np.float32), gender=gender)  # Save
            nan_frame_idxs = []  # Reset nan_frame_idxs
            prev_result = smpl_arr

    if is_nan_seq:  # If the last nan sequence has not ended
        # Fill all with prev_result
        for i, nan_frame_idx in enumerate(nan_frame_idxs):
            interp_result = prev_result.astype(np.float32)
            interp_path = tgt_dir / f"{nan_frame_idx:010d}.npz"
            np.savez(interp_path, arr=interp_result, gender=gender)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SMPL-X fitting on Parahome dataset")
    parser.add_argument("--root_src_dir", type=Path, required=True, help="Name of the session to process")
    parser.add_argument("--root_tgt_dir", type=Path, required=True, help="Name of the session to process")
    parser.add_argument("--session_name", type=str, required=True, help="Name of the session to process")
    args = parser.parse_args()
    interp_nan(
        src_dir=args.root_src_dir / args.session_name,
        tgt_dir=args.root_tgt_dir / args.session_name,
    )
