"""Image <-> pitch coordinate transforms built on a calibrated homography.

Given a :class:`~offside.calibration.Calibration` (a 3x3 homography H mapping
image pixels to pitch metres and its inverse), this module projects points in
either direction. It handles single points and batches, validates inputs, and
fails safely on invalid transforms (e.g. points on the horizon line where the
homogeneous w-coordinate is zero).

Coordinate convention (see PROJECT_CONTEXT.md):
    * Image: pixels (x, y), top-left origin.
    * Pitch: metres (X = length 0..105 default, Y = width 0..68 default).

IMPORTANT: when projecting players onto the pitch, pass their **ground-contact
(foot) point** — not the bounding-box centre or head — so the projected position
corresponds to where the player stands on the plane.

Depends only on NumPy and OpenCV (no Ultralytics / PyTorch).
"""

from __future__ import annotations

import cv2
import numpy as np

from offside.calibration import Calibration


class GeometryError(ValueError):
    """Raised for invalid transform inputs or non-mappable points."""


def _as_point_array(points: np.ndarray | list | tuple) -> np.ndarray:
    """Coerce input into an (N, 2) float64 array, accepting a single (x, y)."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != 2:
            raise GeometryError(f"A single point must have 2 coordinates, got {arr.shape[0]}.")
        arr = arr.reshape(1, 2)
    elif arr.ndim == 2:
        if arr.shape[1] != 2:
            raise GeometryError(f"Points must have shape (N, 2), got {arr.shape}.")
        if arr.shape[0] == 0:
            raise GeometryError("Points array is empty.")
    else:
        raise GeometryError(f"Points must be 1-D (x, y) or 2-D (N, 2), got {arr.ndim}-D.")
    if not np.all(np.isfinite(arr)):
        raise GeometryError("Points contain non-finite values (NaN/Inf).")
    return arr


def _apply_homography(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to (N, 2) points, guarding against w ~ 0.

    We compute the transform manually (rather than cv2.perspectiveTransform) so
    we can detect points that map to/near the horizon (homogeneous w close to 0),
    which would otherwise produce infinities silently.
    """
    n = points.shape[0]
    homogeneous = np.hstack([points, np.ones((n, 1), dtype=np.float64)])  # (N, 3)
    transformed = homogeneous @ matrix.T  # (N, 3)
    w = transformed[:, 2]
    if np.any(np.abs(w) < 1e-9):
        raise GeometryError(
            "One or more points map to the horizon line (degenerate projection); "
            "they cannot be represented on the target plane."
        )
    return transformed[:, :2] / w[:, np.newaxis]


def _transform(
    points: np.ndarray | list | tuple, matrix: np.ndarray, single_input: bool
) -> np.ndarray:
    arr = _as_point_array(points)
    out = _apply_homography(arr, matrix)
    return out[0] if single_input else out


def _is_single(points: np.ndarray | list | tuple) -> bool:
    arr = np.asarray(points, dtype=np.float64)
    return arr.ndim == 1


def image_to_pitch(
    calibration: Calibration, image_points: np.ndarray | list | tuple
) -> np.ndarray:
    """Map image pixel coordinates to pitch metres.

    Args:
        calibration: a successful calibration holding the homography.
        image_points: a single (x, y) or an (N, 2) array of pixel points.

    Returns:
        A (2,) array for a single input point, or an (N, 2) array for a batch,
        in pitch metres.

    Raises:
        GeometryError: for malformed input or non-mappable points.
    """
    return _transform(image_points, calibration.homography, _is_single(image_points))


def pitch_to_image(
    calibration: Calibration, pitch_points: np.ndarray | list | tuple
) -> np.ndarray:
    """Map pitch metres to image pixel coordinates.

    Args:
        calibration: a successful calibration holding the inverse homography.
        pitch_points: a single (X, Y) or an (N, 2) array of pitch points (metres).

    Returns:
        A (2,) array for a single input point, or an (N, 2) array for a batch,
        in image pixels.

    Raises:
        GeometryError: for malformed input or non-mappable points.
    """
    return _transform(pitch_points, calibration.homography_inv, _is_single(pitch_points))
