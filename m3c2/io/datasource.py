"""Utilities for loading heterogeneous point cloud data.

This module centralises the logic for reading point cloud pairs from a number
of supported file formats and converting them to a unified XYZ representation.
Only the necessary optional dependencies are imported for the requested file
types, enabling this module to be imported even when those libraries are not
installed.  The :class:`DataSource` class is the public entry point and exposes
the :meth:`DataSource.load_points` method which returns the moving epoch, the
reference epoch and the core points as NumPy arrays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np

try:  # Optional dependency
    import py4dgeo  # type: ignore
except Exception:  # pragma: no cover - handled in tests
    py4dgeo = None  # type: ignore

try:  # Optional dependency
    from plyfile import PlyData  # type: ignore
except Exception:  # pragma: no cover - handled in tests
    PlyData = None  # type: ignore

try:  # Optional dependency
    import laspy  # type: ignore
except Exception:  # pragma: no cover - handled in tests
    laspy = None  # type: ignore


@dataclass
class DataSource:
    """Load point cloud data for a pair of epochs.

    Parameters
    ----------
    folder:
        Base directory containing the point cloud files.
    mov_basename, ref_basename:
        Basenames of the moving and reference point cloud files without the
        file extension. Supported extensions are ``.xyz``, ``.las``, ``.laz``,
        ``.ply``, ``.obj`` and ``.gpc``.
    mov_as_corepoints:
        If ``True`` the moving epoch is used as core points, otherwise the
        reference epoch is used.
    use_subsampled_corepoints:
        Subsampling factor applied to the core points.

    Raises
    ------
    TypeError
        Raised by :meth:`load_points` if the resulting core points are not an
        ``np.ndarray``.
    """

    folder: str
    mov_basename: str = "mov"
    ref_basename: str = "ref"
    mov_as_corepoints: bool = True
    use_subsampled_corepoints: int = 1

    mov_base: Path = field(init=False)
    ref_base: Path = field(init=False)

    def __post_init__(self) -> None:
        self.folder = Path(self.folder)
        self.mov_base = self.folder / self.mov_basename
        self.ref_base = self.folder / self.ref_basename
        self.folder.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # helpers
    def _detect(self, base: Path) -> tuple[str | None, Path | None]:
        """Detect the available file type for ``base``.

        Parameters
        ----------
        base:
            File path without extension for which to determine the existing
            file.

        Returns
        -------
        tuple
            A ``(kind, path)`` tuple where ``kind`` is one of
            ``{"xyz", "laslike", "ply", "obj", "gpc"}`` or ``None`` if no
            file exists and ``path`` is the resolved file path.
        """

        # Map each supported extension to its potential file path
        mapping = {
            "xyz": base.with_suffix(".xyz"),
            "las": base.with_suffix(".las"),
            "laz": base.with_suffix(".laz"),
            "ply": base.with_suffix(".ply"),
            "obj": base.with_suffix(".obj"),
            "gpc": base.with_suffix(".gpc"),
        }

        # Check for existing files in order of preference
        if mapping["xyz"].exists():
            return "xyz", mapping["xyz"]
        if mapping["las"].exists() or mapping["laz"].exists():
            return "laslike", mapping["las"] if mapping["las"].exists() else mapping["laz"]
        if mapping["ply"].exists():
            return "ply", mapping["ply"]
        if mapping["obj"].exists():
            return "obj", mapping["obj"]
        if mapping["gpc"].exists():
            return "gpc", mapping["gpc"]
        return None, None

    def _read_las_or_laz_to_xyz_array(self, path: Path) -> np.ndarray:
        """Read a LAS or LAZ file and return its XYZ coordinates.

        Parameters
        ----------
        path:
            Path to the LAS/LAZ file.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(N, 3)`` containing ``x``, ``y`` and ``z``
            coordinates as ``float64`` values.

        Raises
        ------
        RuntimeError
            If ``laspy`` or its optional ``lazrs`` dependency is not
            installed.
        """

        if laspy is None:
            raise RuntimeError("LAS/LAZ gefunden, aber 'laspy' ist nicht installiert.")

        try:
            # ``laspy`` transparently handles LAS or LAZ files
            las = laspy.read(str(path))
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency issue
            raise RuntimeError(
                "LAZ erkannt, bitte 'pip install \"laspy[lazrs]\"' installieren."
            ) from exc

        # Stack the coordinate arrays column-wise into an ``(N, 3)`` array
        return np.vstack([las.x, las.y, las.z]).T.astype(np.float64)

    def _read_obj_to_xyz_array(self, path: Path) -> np.ndarray:
        """Parse an OBJ file and extract its vertex coordinates.

        Parameters
        ----------
        path:
            Path to the OBJ file containing ``v`` entries.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(N, 3)`` with vertex coordinates.
        """

        vertices: list[list[float]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                # OBJ vertex definitions start with ``v`` followed by x, y, z
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.asarray(vertices, dtype=np.float64)

    def _read_gpc_to_xyz_array(self, path: Path) -> np.ndarray:
        """Read a GPC file and return its XYZ coordinates.

        Parameters
        ----------
        path:
            Path to the GPC file.

        Returns
        -------
        numpy.ndarray
            Array of ``float64`` coordinates with shape ``(N, 3)``.
        """

        return np.loadtxt(path, dtype=np.float64, usecols=(0, 1, 2))

    def _ensure_xyz(self, base: Path, detected: tuple[str | None, Path | None]) -> Path:
        """Ensure that an ``.xyz`` file exists for ``base``.

        Parameters
        ----------
        base:
            Path without extension specifying the desired output location.
        detected:
            ``(kind, path)`` tuple as returned by :meth:`_detect`.

        Returns
        -------
        pathlib.Path
            Path to the resulting ``.xyz`` file.

        Raises
        ------
        FileNotFoundError
            If no supported file for ``base`` exists.
        RuntimeError
            If a required optional dependency is missing during conversion.
        """

        kind, path = detected
        xyz = base.with_suffix(".xyz")

        # If an ``.xyz`` file already exists simply return it
        if kind == "xyz" and path:
            return path

        # Convert LAS/LAZ files to ``.xyz`` using ``laspy``
        if kind == "laslike" and path:
            logging.info("[%s] Konvertiere LAS/LAZ → XYZ …", base)
            arr = self._read_las_or_laz_to_xyz_array(path)
            np.savetxt(xyz, arr, fmt="%.6f")
            return xyz

        # Convert PLY files to ``.xyz`` using ``plyfile``
        if kind == "ply" and path:
            if PlyData is None:
                raise RuntimeError("PLY gefunden, aber 'plyfile' ist nicht installiert.")
            logging.info("[%s] Konvertiere PLY → XYZ …", base)
            ply = PlyData.read(str(path))
            v = ply["vertex"]
            arr = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)
            np.savetxt(xyz, arr, fmt="%.6f")
            return xyz

        # Convert OBJ files by extracting their vertices
        if kind == "obj" and path:
            logging.info("[%s] Konvertiere OBJ → XYZ …", base)
            arr = self._read_obj_to_xyz_array(path)
            np.savetxt(xyz, arr, fmt="%.6f")
            return xyz

        # Convert GPC files to ``.xyz`` using plain text loading
        if kind == "gpc" and path:
            logging.info("[%s] Konvertiere GPC → XYZ …", base)
            arr = self._read_gpc_to_xyz_array(path)
            np.savetxt(xyz, arr, fmt="%.6f")
            return xyz

        # No suitable file found for conversion
        raise FileNotFoundError(f"Fehlt: {base}.xyz/.las/.laz/.ply/.obj/.gpc")

    # ------------------------------------------------------------------
    # public API
    def load_points(self) -> Tuple[object, object, np.ndarray]:
        """Load the moving and reference epochs and derive core points.

        Returns
        -------
        tuple
            ``(mov, ref, corepoints)`` where ``mov`` and ``ref`` are the
            objects returned by :mod:`py4dgeo` and ``corepoints`` is an
            ``np.ndarray`` of shape ``(N, 3)``.

        Raises
        ------
        RuntimeError
            If :mod:`py4dgeo` is not available.
        TypeError
            If the derived core points are not an ``np.ndarray``.
        """

        if py4dgeo is None:  # pragma: no cover - handled via tests
            raise RuntimeError("'py4dgeo' ist nicht installiert.")

        # Determine the available file types for moving and reference epochs
        m_kind, m_path = self._detect(self.mov_base)
        r_kind, r_path = self._detect(self.ref_base)

        # Choose the appropriate py4dgeo reader based on detected types
        if m_kind == r_kind == "xyz":
            logging.info("Nutze py4dgeo.read_from_xyz")
            mov, ref = py4dgeo.read_from_xyz(str(m_path), str(r_path))
        elif m_kind == "laslike" and r_kind == "laslike":
            logging.info("Nutze py4dgeo.read_from_las (unterstützt .las und .laz)")
            mov, ref = py4dgeo.read_from_las(str(m_path), str(r_path))
        elif m_kind == r_kind == "ply" and hasattr(py4dgeo, "read_from_ply"):
            logging.info("Nutze py4dgeo.read_from_ply")
            mov, ref = py4dgeo.read_from_ply(str(m_path), str(r_path))
        else:
            # Convert heterogeneous types to XYZ and use the generic reader
            m_xyz = self._ensure_xyz(self.mov_base, (m_kind, m_path))
            r_xyz = self._ensure_xyz(self.ref_base, (r_kind, r_path))
            logging.info("Mischtypen → konvertiert zu XYZ → py4dgeo.read_from_xyz")
            mov, ref = py4dgeo.read_from_xyz(str(m_xyz), str(r_xyz))

        # Extract core points from the configured epoch, optionally subsampled
        if self.mov_as_corepoints:
            logging.info(
                "Nutze mov als Corepoints und nutze Subsamling: %s",
                self.use_subsampled_corepoints,
            )
            corepoints = (
                mov.cloud[:: self.use_subsampled_corepoints]
                if hasattr(mov, "cloud")
                else mov
            )
        else:
            logging.info(
                "Nutze ref als Corepoints und nutze Subsamling: %s",
                self.use_subsampled_corepoints,
            )
            corepoints = (
                ref.cloud[:: self.use_subsampled_corepoints]
                if hasattr(ref, "cloud")
                else ref
            )

        if not isinstance(corepoints, np.ndarray):
            raise TypeError("Unerwarteter Typ für corepoints; erwarte np.ndarray (Nx3).")

        return mov, ref, corepoints


__all__ = ["DataSource"]

