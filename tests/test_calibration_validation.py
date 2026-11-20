"""V1.1 calibration validation: flexible landmark sets, degeneracy, spread, RANSAC units.

A known "true camera" maps the pitch plane to the image; catalogue landmarks are
projected through it to make noise-free correspondences. Any non-degenerate set of
landmarks must then recover the same mapping, and degenerate sets must be refused.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from offside.calibration import (
    MAX_CORRESPONDENCES,
    CalibrationError,
    PitchDimensions,
    assess_point_set,
    calibrate,
    make_correspondences,
)
from offside.pitch_model import get_landmark

P = PitchDimensions(105.0, 68.0)
IMG_SIZE = (1280, 720)

_CORNERS = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype=np.float32)
_CAMERA_IMG = np.array([[150, 640], [1130, 640], [900, 250], [380, 250]], dtype=np.float32)
H_PITCH_TO_IMAGE = cv2.getPerspectiveTransform(_CORNERS, _CAMERA_IMG)


def to_image(pitch_pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(pitch_pts, dtype=np.float64).reshape(-1, 1, 2), H_PITCH_TO_IMAGE.astype(np.float64)
    ).reshape(-1, 2)


def from_landmarks(keys: list[str], pitch: PitchDimensions = P):
    pit = np.array([get_landmark(k).pitch_xy(pitch) for k in keys])
    return make_correspondences(to_image(pit), pit, labels=keys)


# Probe points over the whole pitch used to compare recovered mappings.
_PROBES = np.array([[5, 5], [30, 20], [52.5, 34], [80, 50], [100, 64], [15, 60]], dtype=np.float64)


def assert_recovers_true_camera(calib, tol_px: float = 1e-3) -> None:
    recovered = cv2.perspectiveTransform(
        _PROBES.reshape(-1, 1, 2), calib.homography_inv
    ).reshape(-1, 2)
    np.testing.assert_allclose(recovered, to_image(_PROBES), atol=tol_px)


# --- flexible combinations ---------------------------------------------------

VALID_COMBINATIONS = {
    "pitch corners": ["corner_x0_y0", "corner_xL_y0", "corner_xL_yW", "corner_x0_yW"],
    "penalty-area front corners (both ends)": [
        "x0_pa_front_y0", "x0_pa_front_yW", "xL_pa_front_y0", "xL_pa_front_yW",
    ],
    "goal-line points (both ends)": [
        "x0_pa_goalline_y0", "x0_pa_goalline_yW", "xL_pa_goalline_y0", "xL_pa_goalline_yW",
    ],
    "one end: goal-area + penalty-area fronts": [
        "x0_ga_front_y0", "x0_ga_front_yW", "x0_pa_front_y0", "x0_pa_front_yW",
    ],
    "mixed, five points": [
        "corner_x0_y0", "xL_pa_front_yW", "centre_circle_y0", "x0_ga_front_yW", "halfway_yW",
    ],
    "centre circle + penalty marks": [
        "centre_circle_y0", "centre_circle_yW", "x0_penalty_mark", "xL_penalty_mark",
    ],
}


@pytest.mark.parametrize("name", list(VALID_COMBINATIONS))
def test_any_non_degenerate_landmark_combination_recovers_the_same_camera(name: str) -> None:
    calib = calibrate(from_landmarks(VALID_COMBINATIONS[name]), P, image_size=IMG_SIZE)
    assert_recovers_true_camera(calib)
    assert calib.reprojection_error_px < 1e-3


def test_extra_points_on_a_shared_line_are_fine_when_a_good_quad_exists() -> None:
    keys = VALID_COMBINATIONS["pitch corners"] + ["halfway_y0", "centre_mark", "halfway_yW"]
    calib = calibrate(from_landmarks(keys), P, image_size=IMG_SIZE)
    assert_recovers_true_camera(calib)
    assert calib.num_points == 7


def test_correspondence_order_does_not_change_the_result() -> None:
    keys = VALID_COMBINATIONS["mixed, five points"]
    a = calibrate(from_landmarks(keys), P)
    b = calibrate(from_landmarks(list(reversed(keys))), P)
    np.testing.assert_allclose(a.homography, b.homography, rtol=1e-6, atol=1e-9)


def test_image_and_pitch_points_stay_paired_when_labels_are_shuffled_together() -> None:
    keys = VALID_COMBINATIONS["mixed, five points"]
    corr = from_landmarks(keys)
    perm = [3, 0, 4, 2, 1]
    shuffled = make_correspondences(
        corr.image_points[perm], corr.pitch_points[perm], [keys[i] for i in perm]
    )
    assert_recovers_true_camera(calibrate(shuffled, P))


# --- degenerate sets are refused ---------------------------------------------

def test_all_points_on_the_halfway_line_rejected() -> None:
    keys = ["halfway_y0", "halfway_yW", "centre_mark", "centre_circle_y0", "centre_circle_yW"]
    with pytest.raises(CalibrationError, match="cannot determine a homography"):
        calibrate(from_landmarks(keys), P)


def test_three_of_four_on_one_goal_line_rejected() -> None:
    keys = ["x0_pa_goalline_y0", "x0_pa_goalline_yW", "x0_ga_goalline_y0", "centre_mark"]
    with pytest.raises(CalibrationError, match="cannot determine a homography"):
        calibrate(from_landmarks(keys), P)


def test_four_collinear_plus_one_off_line_rejected() -> None:
    # Every four-point subset contains at least three points of the line.
    keys = ["halfway_y0", "centre_circle_y0", "centre_circle_yW", "halfway_yW", "corner_x0_y0"]
    with pytest.raises(CalibrationError):
        calibrate(from_landmarks(keys), P)


def test_collinear_in_the_image_only_is_rejected() -> None:
    # Pitch points are fine but the clicks all fall on one image line.
    pit = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype=float)
    img = np.array([[100, 100], [200, 200], [300, 300], [400, 400]], dtype=float)
    with pytest.raises(CalibrationError):
        calibrate(make_correspondences(img, pit), P)


def test_duplicate_image_location_rejected() -> None:
    pit = np.array([[0, 0], [105, 0], [105, 68], [0, 68], [52.5, 34]], dtype=float)
    img = to_image(pit)
    img[4] = img[0]  # the centre mark clicked on top of the corner
    with pytest.raises(CalibrationError, match="same image location"):
        calibrate(make_correspondences(img, pit, ["a", "b", "c", "d", "e"]), P)


def test_duplicate_pitch_location_rejected() -> None:
    pit = np.array([[0, 0], [105, 0], [105, 68], [0, 68], [0, 0]], dtype=float)
    img = to_image(pit[:4]).tolist() + [[640.0, 400.0]]
    with pytest.raises(CalibrationError, match="same pitch location"):
        calibrate(make_correspondences(img, pit), P)


# --- assess_point_set: count, spread ------------------------------------------

def test_assess_reports_too_few_points() -> None:
    a = assess_point_set([[0, 0], [1, 1], [2, 5]], [[0, 0], [1, 1], [2, 5]])
    assert not a.ok and a.count == 3
    assert "at least 4" in a.errors[0]


def test_assess_reports_mismatched_counts_and_non_finite() -> None:
    assert not assess_point_set([[0, 0]] * 4, [[0, 0]] * 3).ok
    assert not assess_point_set([[0, 0], [1, 0], [0, 1], [np.nan, 1]], [[0, 0]] * 4).ok


def test_assess_rejects_too_many_points() -> None:
    n = MAX_CORRESPONDENCES + 1
    pts = np.random.default_rng(0).uniform(0, 100, size=(n, 2))
    a = assess_point_set(pts, pts)
    assert not a.ok and "Too many" in a.errors[0]


def test_assess_clean_wide_set_has_no_warnings() -> None:
    corr = from_landmarks(VALID_COMBINATIONS["pitch corners"])
    a = assess_point_set(corr.image_points, corr.pitch_points, pitch=P, image_size=IMG_SIZE)
    assert a.ok and a.warnings == ()
    assert a.pitch_coverage == pytest.approx(1.0, abs=1e-6)
    assert a.conditioning is not None and a.conditioning > 0.1


def test_assess_warns_about_weak_conditioning() -> None:
    # Fourth point only ~4% of the baseline off the line through the others.
    pts = [[0, 0], [100, 0], [50, 5], [30, -4]]
    a = assess_point_set(pts, pts, pitch=PitchDimensions(105.0, 68.0))
    assert a.ok
    assert "weak_conditioning" in [w.code for w in a.warnings]


def test_assess_warns_when_landmarks_cover_little_of_the_pitch() -> None:
    corr = from_landmarks(VALID_COMBINATIONS["one end: goal-area + penalty-area fronts"])
    a = assess_point_set(corr.image_points, corr.pitch_points, pitch=P, image_size=IMG_SIZE)
    assert a.ok
    assert "poor_pitch_coverage" in [w.code for w in a.warnings]
    assert a.pitch_coverage is not None and a.pitch_coverage < 0.15


def test_assess_warns_when_clicks_cover_little_of_the_frame() -> None:
    pit = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype=float)
    img = np.array([[600, 300], [680, 300], [680, 360], [600, 360]], dtype=float)  # tiny region
    a = assess_point_set(img, pit, pitch=P, image_size=IMG_SIZE)
    assert a.ok
    assert "poor_image_coverage" in [w.code for w in a.warnings]


def test_image_coverage_only_computed_when_size_supplied() -> None:
    keys = VALID_COMBINATIONS["pitch corners"]
    assert calibrate(from_landmarks(keys), P).image_coverage is None
    with_size = calibrate(from_landmarks(keys), P, image_size=IMG_SIZE)
    assert with_size.image_coverage is not None and 0.0 < with_size.image_coverage <= 1.0


# --- reprojection honesty -----------------------------------------------------

def test_exact_four_point_fit_warns_that_error_is_meaningless() -> None:
    calib = calibrate(from_landmarks(VALID_COMBINATIONS["pitch corners"]), P)
    assert "exact_fit" in calib.warning_codes
    assert calib.reprojection_error_px < 1e-3


def test_redundant_points_give_a_real_consistency_check_without_exact_fit_warning() -> None:
    calib = calibrate(from_landmarks(VALID_COMBINATIONS["pitch corners"] + ["centre_mark", "halfway_y0"]), P)
    assert "exact_fit" not in calib.warning_codes


def test_noisy_clicks_are_reflected_in_the_error_and_warned_when_large() -> None:
    keys = VALID_COMBINATIONS["pitch corners"] + ["centre_mark", "halfway_y0", "halfway_yW"]
    corr = from_landmarks(keys)
    rng = np.random.default_rng(1)
    noisy = corr.image_points + rng.normal(0, 2.0, size=corr.image_points.shape)
    calib = calibrate(make_correspondences(noisy, corr.pitch_points, keys), P)
    assert 0.3 < calib.reprojection_error_px < 6.0
    assert len(calib.point_errors_px) == len(keys)


# --- RANSAC works in IMAGE PIXELS (regression: it used to be metres) ----------

def _six_consistent_points():
    keys = VALID_COMBINATIONS["pitch corners"] + ["centre_mark", "halfway_y0"]
    return keys, from_landmarks(keys)


def test_twelve_pixel_misclick_is_rejected_as_an_outlier() -> None:
    keys, corr = _six_consistent_points()
    img = corr.image_points.copy()
    img[5] += [12.0, 0.0]  # halfway_y0 clicked 12 px off
    calib = calibrate(make_correspondences(img, corr.pitch_points, keys), P)
    assert calib.inliers is not None
    assert calib.outlier_indices == (5,)
    assert "outliers_rejected" in calib.warning_codes
    assert "halfway_y0" in next(w.message for w in calib.warnings if w.code == "outliers_rejected")
    # The residual of the rejected click is the largest and roughly the click error.
    assert int(np.argmax(calib.point_errors_px)) == 5
    assert calib.point_errors_px[5] > 8.0
    assert_recovers_true_camera(calib, tol_px=0.05)


def test_small_click_noise_is_not_treated_as_outliers() -> None:
    keys, corr = _six_consistent_points()
    img = corr.image_points + np.array([[1.5, -1.0], [-1.0, 1.5], [1.0, 1.0], [-1.5, -1.0], [0.5, 1.0], [1.0, -0.5]])
    calib = calibrate(make_correspondences(img, corr.pitch_points, keys), P)
    assert calib.outlier_indices == ()
    assert "outliers_rejected" not in calib.warning_codes


def test_ransac_threshold_must_be_positive() -> None:
    with pytest.raises(CalibrationError):
        calibrate(from_landmarks(VALID_COMBINATIONS["pitch corners"]), P, ransac_reproj_threshold=0.0)


def test_labels_must_match_point_count() -> None:
    with pytest.raises(CalibrationError):
        make_correspondences([[0, 0]] * 4, [[0, 0]] * 4, labels=["a", "b"])
