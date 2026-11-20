"""Tests for calibration / homography estimation.

Deterministic: correspondences are generated from a known homography. No model
weights, internet, or GPU required.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from offside.calibration import (
    Calibration,
    CalibrationError,
    Correspondences,
    PitchDimensions,
    calibrate,
    make_correspondences,
)

# A fixed, realistic-ish image->pitch homography used to synthesize points.
# (Perspective-like: maps a quadrilateral of image pixels onto the pitch plane.)
_IMG_QUAD = np.array([[100, 700], [1180, 700], [980, 300], [300, 300]], dtype=np.float64)
_PITCH_QUAD = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype=np.float64)
_KNOWN_H = cv2.getPerspectiveTransform(_IMG_QUAD.astype(np.float32), _PITCH_QUAD.astype(np.float32))


def _project(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), h).reshape(-1, 2)


def test_calibrate_four_points_recovers_homography() -> None:
    corr = make_correspondences(_IMG_QUAD, _PITCH_QUAD)
    calib = calibrate(corr)
    assert isinstance(calib, Calibration)
    assert calib.num_points == 4
    assert calib.inliers is None  # exact solve, no RANSAC
    # Reprojection error should be ~0 for an exact 4-point solve.
    assert calib.reprojection_error_px < 1e-6
    assert calib.reprojection_error_m < 1e-6


def test_reprojection_error_exposed_and_small_with_extra_points() -> None:
    # Add interior points consistent with the known homography.
    extra_img = np.array([[640, 500], [500, 450], [800, 450]], dtype=np.float64)
    img = np.vstack([_IMG_QUAD, extra_img])
    pit = _project(_KNOWN_H, img)
    calib = calibrate(make_correspondences(img, pit))
    assert calib.num_points == 7
    assert calib.reprojection_error_px < 1.0
    assert calib.reprojection_error_m < 0.5


def test_ransac_rejects_outlier() -> None:
    img = np.vstack([_IMG_QUAD, np.array([[640, 500], [500, 450]], dtype=np.float64)])
    pit = _project(_KNOWN_H, img)
    # Corrupt one correspondence into a gross outlier.
    pit_bad = pit.copy()
    pit_bad[-1] = [999.0, 999.0]
    calib = calibrate(make_correspondences(img, pit_bad))
    assert calib.inliers is not None
    assert calib.num_inliers is not None and calib.num_inliers >= 4
    # The corrupted last point should be flagged as an outlier.
    assert calib.inliers[-1] == False  # noqa: E712


def test_default_and_custom_pitch_dimensions() -> None:
    calib = calibrate(make_correspondences(_IMG_QUAD, _PITCH_QUAD))
    assert calib.pitch.length == 105.0
    assert calib.pitch.width == 68.0

    custom = PitchDimensions(length=100.0, width=64.0)
    calib2 = calibrate(make_correspondences(_IMG_QUAD, _PITCH_QUAD), pitch=custom)
    assert calib2.pitch.length == 100.0
    assert calib2.pitch.width == 64.0


def test_pitch_dimensions_must_be_positive() -> None:
    with pytest.raises(CalibrationError):
        PitchDimensions(length=0.0, width=68.0)


def test_too_few_points_rejected() -> None:
    with pytest.raises(CalibrationError):
        Correspondences(
            image_points=np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64),
            pitch_points=np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64),
        )


def test_mismatched_counts_rejected() -> None:
    with pytest.raises(CalibrationError):
        Correspondences(
            image_points=np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64),
            pitch_points=np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64),
        )


def test_non_finite_coordinates_rejected() -> None:
    with pytest.raises(CalibrationError):
        make_correspondences(
            [[0, 0], [1, 0], [0, 1], [np.nan, 1]],
            [[0, 0], [1, 0], [0, 1], [1, 1]],
        )


def test_collinear_image_points_rejected() -> None:
    # All image points on a line -> degenerate.
    img = [[0, 0], [1, 1], [2, 2], [3, 3]]
    pit = [[0, 0], [105, 0], [105, 68], [0, 68]]
    with pytest.raises(CalibrationError):
        calibrate(make_correspondences(img, pit))


def test_duplicate_points_rejected() -> None:
    img = [[0, 0], [0, 0], [10, 0], [0, 10]]  # effectively 3 distinct points
    pit = [[0, 0], [0, 0], [105, 0], [0, 68]]
    with pytest.raises(CalibrationError):
        calibrate(make_correspondences(img, pit))


def test_wrong_shape_rejected() -> None:
    with pytest.raises(CalibrationError):
        make_correspondences([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
                             [[0, 0], [1, 0], [0, 1], [1, 1]])
