import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

from utils.data import split_indices_with_overlap
from utils.rotation import axis_angle_to_matrix


class ParahomeDataset(data.Dataset):
    def __init__(
        self,
        session_dir: Path,
        device: torch.device = torch.device("cpu"),
        first_idx: int | None = None,
        last_idx: int | None = None,
    ):
        self.device = device
        with open(session_dir / "smplx_params.pkl", "rb") as f:
            smplx_params = pickle.load(f)
        with open(session_dir / "smplx_pose.pkl", "rb") as f:
            smplx_pose = pickle.load(f)

        self.gender: str = smplx_params["gender"]
        self.betas: torch.Tensor = smplx_params["beta"].squeeze(0).to(device)  # [20]

        self.seq_body_pose = axis_angle_to_matrix(smplx_pose["body_pose"].reshape(-1, 21, 3)).to(device)  # [seqlen, 63] -> [seqlen, 21, 3, 3]
        self.seq_global_orient = axis_angle_to_matrix(smplx_pose["global_orient"]).to(device)  # [seqlen, 3, 3]
        self.seq_transl: torch.Tensor = smplx_pose["transl"].to(device)  # [seqlen, 3]
        seq_hand_pose: torch.Tensor = smplx_pose["hand_pose"].reshape(-1, 30, 3)  # [seqlen, 30, 3]
        self.seq_left_hand_pose = axis_angle_to_matrix(seq_hand_pose[:, :15, :]).to(device)  # [seqlen, 15, 3] -> [seqlen, 15, 3, 3]
        self.seq_right_hand_pose = axis_angle_to_matrix(seq_hand_pose[:, 15:, :]).to(device)  # [seqlen, 15, 3] -> [seqlen, 15, 3, 3]

        orig_seqlen = len(self.seq_body_pose)
        last_idx = orig_seqlen - 1 if last_idx is None else last_idx
        first_idx = 0 if first_idx is None else first_idx

        self.seq_body_pose = self.seq_body_pose[first_idx : last_idx + 1]
        self.seq_global_orient = self.seq_global_orient[first_idx : last_idx + 1]
        self.seq_transl = self.seq_transl[first_idx : last_idx + 1]
        self.seq_left_hand_pose = self.seq_left_hand_pose[first_idx : last_idx + 1]
        self.seq_right_hand_pose = self.seq_right_hand_pose[first_idx : last_idx + 1]

        self.seqlen = len(self.seq_body_pose)

        # These values are used for visualization
        self.xmin, self.ymin, self.zmin = self.seq_transl.amin(0)
        self.xmax, self.ymax, self.zmax = self.seq_transl.amax(0)

    def __len__(self):
        return self.seqlen

    def __getitem__(self, idx):
        return {
            "betas": self.betas.unsqueeze(0),
            "body_pose": self.seq_body_pose[idx : idx + 1],
            "global_orient": self.seq_global_orient[idx : idx + 1],
            "transl": self.seq_transl[idx : idx + 1],
            "left_hand_pose": self.seq_left_hand_pose[idx : idx + 1],
            "right_hand_pose": self.seq_right_hand_pose[idx : idx + 1],
        }


class ParahomeTemporalDataset(ParahomeDataset):
    def __init__(
        self,
        window_size: int,
        n_duplicated_frames: int,
        per_frame_result_dir: Path,
        temporal_result_dir: Path,
        session_dir: Path,
        device=torch.device("cpu"),
        first_idx: int | None = None,
        last_idx: int | None = None,
    ):
        super().__init__(session_dir, device, first_idx, last_idx)

        self.window_size = window_size
        self.windows = torch.from_numpy(split_indices_with_overlap(self.seqlen, window_size, n_duplicated_frames)).to(device)
        self.per_frame_result_dir = per_frame_result_dir
        assert per_frame_result_dir.exists(), f"Result directory does not exist: {per_frame_result_dir}"
        self.temporal_result_dir = temporal_result_dir

        self.determine_betas()

    def determine_betas(self):
        all_betas = np.empty((self.seqlen, 10), dtype=np.float32)
        all_transl = np.empty((self.seqlen, 3), dtype=np.float32)
        for i in tqdm(range(self.seqlen), total=self.seqlen, desc="Determining betas"):
            result_path = self.per_frame_result_dir / f"{i:010d}.npz"
            if not result_path.exists():
                continue
            result_arr = np.load(result_path)["arr"]
            assert not np.isnan(result_arr).any(), f"NaN values found in result file: {result_path}"
            all_betas[i, :] = result_arr[75:]
            all_transl[i, :] = result_arr[:3]
        self.smpl_betas = torch.from_numpy(all_betas.mean(0)).float().to(self.device)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int):
        frame_idxs = self.windows[idx]
        smpl_seq_transl = np.empty((self.window_size, 3), dtype=np.float32)
        smpl_seq_global_orient = np.empty((self.window_size, 3), dtype=np.float32)
        smpl_seq_body_pose = np.empty((self.window_size, 69), dtype=np.float32)

        for i, frame_idx in enumerate(frame_idxs):
            frame_idx_str = f"{frame_idx.item():010d}.npz"
            if (self.temporal_result_dir / frame_idx_str).exists():
                result_path = self.temporal_result_dir / frame_idx_str
            else:
                result_path = self.per_frame_result_dir / frame_idx_str
            assert result_path.exists(), f"Result file does not exist: {result_path}"
            result_arr = np.load(result_path)["arr"]
            smpl_seq_transl[i] = result_arr[:3]
            smpl_seq_global_orient[i] = result_arr[3:6]
            smpl_seq_body_pose[i] = result_arr[6:75]

        smpl_seq_transl = torch.from_numpy(smpl_seq_transl).to(self.device)
        smpl_seq_global_orient = torch.from_numpy(smpl_seq_global_orient).to(self.device)
        smpl_seq_body_pose = torch.from_numpy(smpl_seq_body_pose).to(self.device)

        return {
            "smplx_betas": self.betas.unsqueeze(0).expand(self.window_size, -1),
            "smplx_body_pose": self.seq_body_pose[frame_idxs],
            "smplx_global_orient": self.seq_global_orient[frame_idxs],
            "smplx_transl": self.seq_transl[frame_idxs],
            "smplx_left_hand_pose": self.seq_left_hand_pose[frame_idxs],
            "smplx_right_hand_pose": self.seq_right_hand_pose[frame_idxs],
            #
            "smpl_betas": self.smpl_betas.unsqueeze(0).expand(self.window_size, -1),
            "smpl_transl": smpl_seq_transl,
            "smpl_global_orient": smpl_seq_global_orient,
            "smpl_body_pose": smpl_seq_body_pose,
            #
            "frame_idxs": frame_idxs,
        }
