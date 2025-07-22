import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


def apply_deformation_transfer(def_matrix: torch.Tensor, smplx_verts: torch.Tensor, use_normals: bool = False) -> torch.Tensor:
    """Applies the deformation transfer on the given meshes"""
    if use_normals:
        raise NotImplementedError
    else:
        def_vertices = torch.einsum("mn,bni->bmi", [def_matrix, smplx_verts])
        return def_vertices


def read_deformation_transfer(
    deformation_transfer_path: Path,
    device: torch.device | None = None,
    use_normal: bool = False,
) -> torch.Tensor:
    """Reads a deformation transfer"""
    if device is None:
        device = torch.device("cpu")
    assert deformation_transfer_path.exists(), f"Deformation transfer path does not exist: {deformation_transfer_path}"
    # Read the deformation transfer matrix
    with open(deformation_transfer_path, "rb") as f:
        def_transfer_setup = pickle.load(f, encoding="latin1")
    if "mtx" in def_transfer_setup:
        def_matrix = def_transfer_setup["mtx"]
        if hasattr(def_matrix, "todense"):
            def_matrix = def_matrix.todense()
        def_matrix = np.array(def_matrix, dtype=np.float32)
        if not use_normal:
            num_verts = def_matrix.shape[1] // 2
            def_matrix = def_matrix[:, :num_verts]
    elif "matrix" in def_transfer_setup:
        def_matrix = def_transfer_setup["matrix"]
    else:
        valid_keys = ["mtx", "matrix"]
        raise KeyError(f"Deformation transfer setup must contain {valid_keys}")

    return torch.tensor(def_matrix, device=device, dtype=torch.float32)


def row(A):
    return A.reshape((1, -1))


def col(A):
    return A.reshape((-1, 1))


def get_vert_connectivity(mesh_v, mesh_f):
    """Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12."""

    vpv = sp.csc_matrix((len(mesh_v), len(mesh_v)))

    # for each column in the faces...
    for i in range(3):
        IS = mesh_f[:, i]
        JS = mesh_f[:, (i + 1) % 3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    return vpv


def get_vertices_per_edge(mesh_v, mesh_f):
    """Returns an Ex2 array of adjacencies between vertices, where
    each element in the array is a vertex index. Each edge is included
    only once. If output of get_faces_per_edge is provided, this is used to
    avoid call to get_vert_connectivity()"""

    vc = sp.coo_matrix(get_vert_connectivity(mesh_v, mesh_f))
    result = np.hstack((col(vc.row), col(vc.col)))
    result = result[result[:, 0] < result[:, 1]]  # for uniqueness

    return result
