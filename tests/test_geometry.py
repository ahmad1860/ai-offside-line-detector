"""Tests for image<->pitch coordinate transforms.

Deterministic: built on a known homography via a real Calibration. No model
weights, internet, or GPU required.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from offside.calibration import calibrate, make_correspondences
from offside.geometry import GeometryError, image_to_pitch, pitch_to_image

_IMG_QUAD = np.array([[100, 700], [1180, 700], [980, 300], [300, 300]], dtype=np.float64)
_PITCH_QUAD = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype=np.float64)


def _calib():
    return calibrate(make_correspondences(_IMG_QUAD, _PITCH_QUAD))


def test_image_to_pitch_known_corners() -> None:
    calib = _calib()
    for img_pt, expected in zip(_IMG_QUAD, _PITCH_QUAD):
        out = image_to_pitch(calib, img_pt)
        assert out.shape == (2,)
        np.testing.assert_allclose(out, expected, atol=1e-6)


def test_pitch_to_image_known_corners() -> None:
    calib = _calib()
    for pitch_pt, expected in zip(_PITCH_QUAD, _IMG_QUAD):
        out = pitch_to_image(calib, pitch_pt)
        np.testing.assert_allclose(out, expected, atol=1e-6)


def test_round_trip_single_point() -> None:
    calib = _calib()
    img_pt = np.array([640.0, 500.0])
    pitch_pt = image_to_pitch(calib, img_pt)
    back = pitch_to_image(calib, pitch_pt)
    np.testing.assert_allclose(back, img_pt, atol=1e-6)


def test_round_trip_batch() -> None:
    calib = _calib()
    img_pts = np.array([[640, 500], [300, 650], [900, 450], [500, 600]], dtype=np.float64)
    pitch_pts = image_to_pitch(calib, img_pts)
    assert pitch_pts.shape == (4, 2)
    back = pitch_to_image(calib, pitch_pts)
    assert back.shape == (4, 2)
    np.testing.assert_allclose(back, img_pts, atol=1e-6)


def test_pitch_points_within_bounds_for_interior_image_point() -> None:
    calib = _calib()
    # A point inside the image quad should land inside the pitch rectangle.
    pitch_pt = image_to_pitch(calib, np.array([640.0, 500.0]))
    assert 0.0 <= pitch_pt[0] <= 105.0
    assert 0.0 <= pitch_pt[1] <= 68.0


def test_single_vs_batch_shapes() -> None:
    calib = _calib()
    single = image_to_pitch(calib, [640, 500])
    assert single.shape == (2,)
    batch = image_to_pitch(calib, [[640, 500]])
    assert batch.shape == (1, 2)


def test_rejects_wrong_single_point_length() -> None:
    calib = _calib()
    with pytest.raises(GeometryError):
        image_to_pitch(calib, [640, 500, 0])


def test_rejects_wrong_batch_shape() -> None:
    calib = _calib()
    with pytest.raises(GeometryError):
        image_to_pitch(calib, np.zeros((3, 3)))


def test_rejects_empty_batch() -> None:
    calib = _calib()
    with pytest.raises(GeometryError):
        image_to_pitch(calib, np.zeros((0, 2)))


def test_rejects_non_finite() -> None:
    calib = _calib()
    with pytest.raises(GeometryError):
        image_to_pitch(calib, [np.inf, 500])


def test_horizon_point_raises() -> None:
    # Construct a homography whose inverse sends a chosen pitch point to w=0.
    # Easiest: use a pure projective matrix and a point on its horizon line.
    h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]])  # w = y + 1
    h_inv = np.linalg.inv(h)
    from offside.calibration import Calibration, PitchDimensions

    calib = Calibration(
        homography=h,
        homography_inv=h_inv,
        pitch=PitchDimensions(),
        reprojection_error_px=0.0,
        reprojection_error_m=0.0,
        num_points=4,
    )
    # Pitch point with y = -1 makes w = 0 under h -> horizon.
    with pytest.raises(GeometryError):
        image_to_pitch(calib, [5.0, -1.0])
