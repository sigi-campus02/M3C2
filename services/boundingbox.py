"""Clip multiple point clouds to their common oriented bounding box.

The helper functions in this module transform all point clouds into the
local frame of the first cloud, compute the intersection of their
oriented bounding boxes and write the clipped results back to disk.
"""

# clip_obb_overlap_multi.py
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List, Tuple

def read_ply(path: str) -> o3d.geometry.PointCloud:
    """Read a ``.ply`` file and return an Open3D point cloud.

    Parameters
    ----------
    path:
        File system path to the PLY file.

    Returns
    -------
    o3d.geometry.PointCloud
        Loaded point cloud instance.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    RuntimeError
        If the file is empty or cannot be parsed as a point cloud.
    """

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found:\n  {p}")
    pc = o3d.io.read_point_cloud(str(p))
    if pc.is_empty():
        raise RuntimeError(
            f"Empty or invalid PLY read (possibly wrong format):\n  {p}"
        )
    return pc

def to_local_frame(xyz: np.ndarray, R: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Transform points into the local oriented bounding box frame.

    Parameters
    ----------
    xyz:
        Array of shape ``(N,3)`` with world coordinates.
    R:
        Rotation matrix defining the OBB axes.
    C:
        Center of the reference OBB.

    Returns
    -------
    np.ndarray
        Points expressed in the local OBB coordinate system.
    """

    # First translate to the OBB centre and then apply the inverse rotation
    return (xyz - C) @ R

def to_world_frame(xyz_local: np.ndarray, R: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Transform local OBB coordinates back into world space."""

    # Inverse of :func:`to_local_frame`: rotate then translate back
    return xyz_local @ R.T + C

def clip_obbf_aligned_many(
    in_paths: List[str],
    out_paths: List[str],
    pad: float = 0.0,
) -> None:
    """Clip several point clouds to the intersection of their oriented bounding boxes.

    Parameters
    ----------
    in_paths:
        Sequence of input PLY file paths.
    out_paths:
        Output file paths; must match ``in_paths`` in length.
    pad:
        Optional padding applied to the intersection bounds in local
        coordinates.

    Raises
    ------
    ValueError
        If ``in_paths`` and ``out_paths`` differ in length.
    RuntimeError
        If no overlapping volume can be determined.
    """

    if len(in_paths) != len(out_paths):
        raise ValueError("in_paths and out_paths must be equally long.")

    # 1) Load all clouds
    pcs = [read_ply(p) for p in in_paths]

    # 2) Use the oriented bounding box of the first cloud as reference
    ref_obb = pcs[0].get_oriented_bounding_box()
    C = ref_obb.center  # (3,)
    R = ref_obb.R       # (3,3) rotation matrix (columns = OBB axes)

    # 3) Transform every cloud into the local OBB frame
    xyz_locals: List[np.ndarray] = []
    color_locals: List[np.ndarray | None] = []
    normal_locals: List[np.ndarray | None] = []
    for pc in pcs:
        xyz = np.asarray(pc.points)
        xyz_l = to_local_frame(xyz, R, C)
        xyz_locals.append(xyz_l)

        if pc.has_colors():
            color_locals.append(np.asarray(pc.colors))
        else:
            color_locals.append(None)

        if pc.has_normals():
            # Only rotate normals, they are already relative to the origin
            nrm = np.asarray(pc.normals)
            normal_locals.append(nrm @ R)
        else:
            normal_locals.append(None)

    # 4) Determine the intersection of the axis-aligned bounding boxes
    # in the local frame
    mins = np.vstack([xyz_l.min(axis=0) for xyz_l in xyz_locals])
    maxs = np.vstack([xyz_l.max(axis=0) for xyz_l in xyz_locals])
    inter_min = mins.max(axis=0)
    inter_max = maxs.min(axis=0)

    # Optional padding around the intersection
    if pad != 0.0:
        inter_min -= pad
        inter_max += pad

    if np.any(inter_min >= inter_max):
        raise RuntimeError(
            "No common OBB overlap found in the reference frame."
        )

    # 5) Clip each cloud and transform back to world coordinates
    for xyz_l, cols, nrms, outp in zip(
        xyz_locals, color_locals, normal_locals, out_paths
    ):
        # Mask points that lie within the intersection bounds
        mask = np.all((xyz_l >= inter_min) & (xyz_l <= inter_max), axis=1)
        clipped_local = xyz_l[mask]

        # Convert clipped points back to world coordinates
        clipped_world = to_world_frame(clipped_local, R, C)

        pc_out = o3d.geometry.PointCloud()
        pc_out.points = o3d.utility.Vector3dVector(clipped_world)

        if cols is not None:
            pc_out.colors = o3d.utility.Vector3dVector(cols[mask])

        if nrms is not None:
            # Rotate normals back to world frame: n_world = n_local @ R.T
            nrms_world = (nrms[mask]) @ R.T
            pc_out.normals = o3d.utility.Vector3dVector(nrms_world)

        outp = str(Path(outp).expanduser().resolve())
        o3d.io.write_point_cloud(outp, pc_out, write_ascii=False, compressed=False)
        print(f"{outp}  |  {xyz_l.shape[0]} -> {clipped_local.shape[0]} points")

    print("Common OBB bounds (in the local frame of the first cloud):")
    print("  min:", inter_min, "\n  max:", inter_max)

if __name__ == "__main__":
    # === HIER deine drei (oder mehr) Dateien eintragen ===
    in_paths  = [
        r"data\Multi-illumination\Job_0378_8400-110\1-1\Job_0378_8400-110-rad-1-1_cloud.ply",
        r"data\Multi-illumination\Job_0378_8400-110\1-1\Job_0378_8400-110-rad-1-1-AI_cloud.ply",
    ]
    out_paths = [
        r"data\Multi-illumination\Job_0378_8400-110\1-1\Job_0378_8400-110-rad-1-1_cloud_overlap.ply",
        r"data\Multi-illumination\Job_0378_8400-110\1-1\Job_0378_8400-110-rad-1-1-AI_cloud_overlap.ply",
    ]

    # Kleines numerisches Polster (optional), z.B. 1e-6 oder 1e-4 je nach Maßstab
    clip_obbf_aligned_many(in_paths, out_paths, pad=0.0)
