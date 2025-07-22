import numpy as np


def split_indices_with_overlap(seqlen: int, window_size: int, n_duplicated_frames: int):
    assert window_size > n_duplicated_frames, "window_size must be larger than n_duplicated_frames"

    step = window_size - n_duplicated_frames
    last_start = max(seqlen - window_size, 0)

    start_indices = np.arange(0, last_start + 1, step)
    if start_indices[-1] != last_start:
        start_indices = np.append(start_indices, last_start)

    window_offsets = np.arange(window_size)
    windows = start_indices[:, None] + window_offsets[None, :]
    return windows
