"""Tests for the pitch model: landmark catalogue and markings.

Expected coordinates are worked out by hand from the Laws-of-the-Game sizes
(penalty area 16.5 m deep / 40.32 m wide, goal area 5.5 m / 18.32 m, penalty mark
11 m, centre circle radius 9.15 m), not copied from the implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from offside.calibration import PitchDimensions
from offside.pitch_model import (
    LEGACY_KEY_ALIASES,
    PITCH_LANDMARKS,
    PitchModelError,
    available_landmarks,
    canonical_key,
    get_landmark,
    pitch_markings,
    validate_custom_pitch_xy,
)

P = PitchDimensions(105.0, 68.0)


def xy(key: str, pitch: PitchDimensions = P) -> tuple[float, float]:
    return get_landmark(key).pitch_xy(pitch)


# --- catalogue structure ----------------------------------------------------

def test_catalogue_size_and_unique_keys() -> None:
    keys = [lm.key for lm in PITCH_LANDMARKS]
    assert len(keys) == len(set(keys)) == 31
    assert len(available_landmarks(P)) == 31  # all fit a standard pitch


def test_every_landmark_has_label_group_description() -> None:
    for lm in PITCH_LANDMARKS:
        assert lm.label and lm.group and lm.description


def test_labels_use_pitch_coordinates_not_image_directions() -> None:
    # The old wording (near/far/left/right) was ambiguous; labels must be in pitch terms.
    for lm in PITCH_LANDMARKS:
        low = lm.label.lower()
        for word in ("near", "far", "left", "right"):
            assert word not in low, f"{lm.label!r} uses ambiguous {word!r}"


# --- hand-computed coordinates (105 x 68) -----------------------------------

@pytest.mark.parametrize(
    "key, expected",
    [
        ("corner_x0_y0", (0.0, 0.0)),
        ("corner_x0_yW", (0.0, 68.0)),
        ("corner_xL_y0", (105.0, 0.0)),
        ("corner_xL_yW", (105.0, 68.0)),
        ("halfway_y0", (52.5, 0.0)),
        ("halfway_yW", (52.5, 68.0)),
        ("centre_mark", (52.5, 34.0)),
        ("centre_circle_y0", (52.5, 34.0 - 9.15)),
        ("centre_circle_yW", (52.5, 34.0 + 9.15)),
        ("x0_pa_goalline_y0", (0.0, 34.0 - 20.16)),
        ("x0_pa_goalline_yW", (0.0, 34.0 + 20.16)),
        ("x0_pa_front_y0", (16.5, 34.0 - 20.16)),
        ("x0_pa_front_yW", (16.5, 34.0 + 20.16)),
        ("xL_pa_front_y0", (105.0 - 16.5, 34.0 - 20.16)),
        ("xL_pa_front_yW", (105.0 - 16.5, 34.0 + 20.16)),
        ("x0_ga_goalline_y0", (0.0, 34.0 - 9.16)),
        ("x0_ga_front_yW", (5.5, 34.0 + 9.16)),
        ("xL_ga_goalline_yW", (105.0, 34.0 + 9.16)),
        ("xL_ga_front_y0", (105.0 - 5.5, 34.0 - 9.16)),
        ("x0_penalty_mark", (11.0, 34.0)),
        ("xL_penalty_mark", (94.0, 34.0)),
    ],
)
def test_catalogue_coordinates(key: str, expected: tuple[float, float]) -> None:
    assert xy(key) == pytest.approx(expected, abs=1e-9)


def test_penalty_arc_points_are_on_the_arc_and_on_the_box_front_line() -> None:
    # Arc: radius 9.15 m about the penalty mark; meets the area front line (x = 16.5).
    for end, mark in (("x0", "x0_penalty_mark"), ("xL", "xL_penalty_mark")):
        for side in ("y0", "yW"):
            ax, ay = xy(f"{end}_pa_arc_{side}")
            mx, my = xy(mark)
            assert math.hypot(ax - mx, ay - my) == pytest.approx(9.15, abs=1e-9)
    assert xy("x0_pa_arc_y0")[0] == pytest.approx(16.5)
    assert xy("xL_pa_arc_yW")[0] == pytest.approx(105.0 - 16.5)


def test_box_widths_match_laws() -> None:
    pa = abs(xy("x0_pa_goalline_yW")[1] - xy("x0_pa_goalline_y0")[1])
    ga = abs(xy("xL_ga_goalline_yW")[1] - xy("xL_ga_goalline_y0")[1])
    assert pa == pytest.approx(40.32)
    assert ga == pytest.approx(18.32)


def test_ends_are_mirror_images() -> None:
    for side in ("y0", "yW"):
        for part in ("pa_front", "pa_goalline", "pa_arc", "ga_front", "ga_goalline"):
            x0, y0 = xy(f"x0_{part}_{side}")
            xl, yl = xy(f"xL_{part}_{side}")
            assert xl == pytest.approx(105.0 - x0)
            assert yl == pytest.approx(y0)


# --- derived from the CURRENT pitch dimensions -------------------------------

def test_coordinates_follow_pitch_dimensions() -> None:
    p = PitchDimensions(100.0, 64.0)
    assert xy("corner_xL_yW", p) == (100.0, 64.0)
    assert xy("centre_mark", p) == (50.0, 32.0)
    assert xy("xL_pa_front_y0", p) == pytest.approx((100.0 - 16.5, 32.0 - 20.16))
    assert xy("halfway_yW", p) == (50.0, 64.0)


def test_marking_sizes_do_not_scale_with_pitch_length() -> None:
    # A penalty area is 16.5 m deep on ANY full-size pitch (regression: it used to be
    # defined as a fraction of a 105 m pitch, so it grew/shrank with the length).
    for length in (100.0, 105.0, 120.0):
        p = PitchDimensions(length, 68.0)
        assert xy("x0_pa_front_yW", p)[0] == pytest.approx(16.5)
        assert length - xy("xL_pa_front_yW", p)[0] == pytest.approx(16.5)
        assert xy("x0_penalty_mark", p)[0] == pytest.approx(11.0)


def test_landmarks_not_fitting_a_small_pitch_are_not_offered() -> None:
    narrow = PitchDimensions(105.0, 40.0)  # penalty area (40.32 m) is wider than the pitch
    keys = {lm.key for lm in available_landmarks(narrow)}
    assert "x0_pa_front_y0" not in keys
    assert "corner_x0_y0" in keys
    assert "centre_mark" in keys
    assert len(keys) < 31


# --- legacy keys -------------------------------------------------------------

def test_legacy_keys_resolve_to_same_points() -> None:
    assert xy("corner_00") == (0.0, 0.0)
    assert xy("corner_11") == (105.0, 68.0)
    assert xy("half_center") == (52.5, 34.0)
    assert xy("pbox_left_near") == pytest.approx((16.5, 34.0 - 20.16))
    assert xy("pbox_right_far") == pytest.approx((105.0 - 16.5, 34.0 + 20.16))
    for old, new in LEGACY_KEY_ALIASES.items():
        assert canonical_key(old) == new
        assert get_landmark(old) is get_landmark(new)


def test_unknown_key_rejected() -> None:
    with pytest.raises(PitchModelError):
        get_landmark("not_a_landmark")


# --- custom points -----------------------------------------------------------

def test_custom_point_accepts_on_pitch_and_small_overhang() -> None:
    assert validate_custom_pitch_xy(10.0, 20.0, P) == (10.0, 20.0)
    assert validate_custom_pitch_xy(-2.0, 70.0, P) == (-2.0, 70.0)  # just outside the lines


@pytest.mark.parametrize("x, y", [(float("nan"), 1.0), (1.0, float("inf")), (500.0, 10.0), (10.0, -50.0)])
def test_custom_point_rejects_invalid(x: float, y: float) -> None:
    with pytest.raises(PitchModelError):
        validate_custom_pitch_xy(x, y, P)


def test_custom_point_rejects_non_numbers() -> None:
    with pytest.raises(PitchModelError):
        validate_custom_pitch_xy("a", 1.0, P)  # type: ignore[arg-type]


# --- markings ----------------------------------------------------------------

def test_markings_for_standard_pitch() -> None:
    lines = pitch_markings(P)
    assert {"outline", "halfway_line", "centre_circle"} <= set(lines)
    assert {"x0_penalty_area", "xL_penalty_area", "x0_goal_area", "xL_goal_area"} <= set(lines)
    outline = lines["outline"]
    assert np.allclose(outline[0], outline[-1])  # closed
    circle = lines["centre_circle"]
    radii = np.hypot(circle[:, 0] - 52.5, circle[:, 1] - 34.0)
    assert np.allclose(radii, 9.15)


def test_markings_omit_boxes_that_do_not_fit() -> None:
    lines = pitch_markings(PitchDimensions(105.0, 40.0))
    assert "x0_penalty_area" not in lines
    assert "x0_goal_area" in lines  # 18.32 m wide still fits
