from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_3d(
    gt_verts: np.ndarray | None = None,
    fit_verts: np.ndarray | None = None,
    gt_kpts: np.ndarray | None = None,
    fit_kpts: np.ndarray | None = None,
    save_path: Path | None = None,
    elev: int = 20,
    azim: int = -225,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
    title: str | None = None,
    verts_s: int = 1,
    kpts_s: int = 20,
    plot_legend: bool = True,
) -> None:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    if gt_verts is not None:
        ax.scatter(gt_verts[:, 0], gt_verts[:, 1], gt_verts[:, 2], c="blue", marker="o", label="GT", s=verts_s)
    if fit_verts is not None:
        ax.scatter(fit_verts[:, 0], fit_verts[:, 1], fit_verts[:, 2], c="red", marker="^", label="Fit", s=verts_s)

    if gt_kpts is not None:
        ax.scatter(gt_kpts[:, 0], gt_kpts[:, 1], gt_kpts[:, 2], c="green", marker="x", label="GT Joints", s=kpts_s)
        for i, joint in enumerate(gt_kpts):
            ax.text(joint[0], joint[1], joint[2], str(i), color="green", fontsize=12)

    if fit_kpts is not None:
        ax.scatter(fit_kpts[:, 0], fit_kpts[:, 1], fit_kpts[:, 2], c="orange", marker="*", label="Fit Joints", s=kpts_s)
        for i, joint in enumerate(fit_kpts):
            ax.text(joint[0], joint[1], joint[2], str(i), color="cyan", fontsize=12)

    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    if zmin is not None and zmax is not None:
        ax.set_zlim(zmin, zmax)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    if plot_legend:
        ax.legend()
    if title is not None:
        ax.set_title(title)

    ax.set_aspect("equal")

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.clf()
    plt.close()
