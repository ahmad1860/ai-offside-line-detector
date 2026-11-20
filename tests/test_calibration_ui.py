"""V1.1 UI-helper tests: flexible/custom landmarks, staleness, direction, clicks.

No Streamlit import; deterministic; a known "true camera" generates the clicks.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.offside_ui import (
    DIAGRAM_TARGET_RGB,
    CalibrationStatus,
    Correspondence,
    DuplicateLandmarkError,
    UIValidationError,
    add_correspondence,
    analysis_signature,
    assess_correspondences,
    build_calibration,
    calibration_input_signature,
    calibration_point_table,
    check_calibration,
    click_stamp,
    click_to_frame_xy,
    correspondence_frames,
    correspondences_to_arrays,
    describe_attacking_direction,
    direction_label,
    direction_target_x,
    landmark_by_key,
    make_custom_correspondence,
    remove_correspondence,
    render_direction_diagram,
    validate_pitch_dimensions,
    validate_ready_to_run,
)
from offside.calibration import PitchDimensions
from offside.offside import (
    AttackingDirection,
    BallPosition,
    OffsideStatus,
    PlayerOnPitch,
    TeamSide,
    evaluate_offside,
)

P = PitchDimensions(105.0, 68.0)
IMG = (1280, 720)

_CORNERS = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype=np.float32)
_QUAD = np.array([[150, 640], [1130, 640], [900, 250], [380, 250]], dtype=np.float32)
_H = cv2.getPerspectiveTransform(_CORNERS, _QUAD).astype(np.float64)


def px_to_image(x: float, y: float) -> tuple[float, float]:
    pt = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float64), _H)[0, 0]
    return float(pt[0]), float(pt[1])


def corr(key: str, frame_index: int | None = 0, pitch: PitchDimensions = P) -> Correspondence:
    ix, iy = px_to_image(*landmark_by_key(key).pitch_xy(pitch))
    return Correspondence(key, ix, iy, frame_index=frame_index)


CORNER_KEYS = ["corner_x0_y0", "corner_xL_y0", "corner_xL_yW", "corner_x0_yW"]
FOUR = [corr(k) for k in CORNER_KEYS]


def assert_matches_camera(calib, tol: float = 1e-3) -> None:
    probes = np.array([[10, 10], [52.5, 34], [90, 55]], dtype=np.float64)
    got = cv2.perspectiveTransform(probes.reshape(-1, 1, 2), calib.homography_inv).reshape(-1, 2)
    want = cv2.perspectiveTransform(probes.reshape(-1, 1, 2), _H).reshape(-1, 2)
    np.testing.assert_allclose(got, want, atol=tol)


# --- landmarks derive from the selected pitch --------------------------------

def test_calibrating_from_penalty_area_landmarks_through_the_ui_helpers() -> None:
    keys = ["x0_pa_front_y0", "x0_pa_front_yW", "xL_pa_front_y0", "xL_pa_front_yW"]
    assert_matches_camera(build_calibration([corr(k) for k in keys], P, IMG))


def test_landmark_coordinates_follow_the_selected_pitch_size() -> None:
    small = PitchDimensions(100.0, 64.0)
    # Clicks generated for the 100x64 interpretation of the SAME landmarks.
    keys = CORNER_KEYS + ["centre_mark"]
    cs = []
    for k in keys:
        x, y = landmark_by_key(k).pitch_xy(small)
        # Scale the 100x64 pitch onto the same camera quad.
        ix, iy = px_to_image(x * 105.0 / 100.0, y * 68.0 / 64.0)
        cs.append(Correspondence(k, ix, iy, frame_index=0))
    ok = build_calibration(cs, small, IMG)
    # Mapping the 100x64 centre must hit the centre click...
    centre_img = np.array([[[50.0, 32.0]]])
    got = cv2.perspectiveTransform(centre_img, ok.homography_inv)[0, 0]
    np.testing.assert_allclose(got, px_to_image(52.5, 34.0), atol=1e-2)
    # ...whereas interpreting the same clicks on a 105x68 pitch gives a different mapping.
    other = build_calibration(cs, P, IMG)
    assert not np.allclose(ok.homography, other.homography)


def test_landmark_that_does_not_exist_on_a_small_pitch_is_rejected() -> None:
    narrow = PitchDimensions(105.0, 40.0)
    cs = [corr(k, pitch=narrow) for k in CORNER_KEYS] + [Correspondence("x0_pa_front_y0", 100, 100, 0)]
    with pytest.raises(UIValidationError, match="does not exist"):
        correspondences_to_arrays(cs, narrow)
    s = assess_correspondences(cs, narrow, IMG)
    assert s.placed == 5 and s.valid == 4
    assert s.invalid_labels and not s.can_build


# --- custom points -----------------------------------------------------------

def test_custom_point_keys_are_unique_and_coordinates_are_kept() -> None:
    first = make_custom_correspondence([], pitch_x=30.0, pitch_y=20.0, image_x=500, image_y=400,
                                       pitch=P, frame_index=0, label="  corner flag base ")
    assert first.landmark_key == "custom_1" and first.is_custom
    assert first.custom_pitch_xy == (30.0, 20.0)
    assert first.custom_label == "corner flag base"
    second = make_custom_correspondence([first], pitch_x=40.0, pitch_y=10.0, image_x=600, image_y=500, pitch=P)
    assert second.landmark_key == "custom_2"
    assert "custom X=30" in first.display_label()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(pitch_x=float("nan"), pitch_y=1.0, image_x=10, image_y=10),
        dict(pitch_x=500.0, pitch_y=1.0, image_x=10, image_y=10),
        dict(pitch_x=10.0, pitch_y=-90.0, image_x=10, image_y=10),
        dict(pitch_x=10.0, pitch_y=10.0, image_x=float("inf"), image_y=10),
        dict(pitch_x=10.0, pitch_y=10.0, image_x=5000, image_y=10),  # outside the frame
    ],
)
def test_custom_point_validation(kwargs) -> None:
    with pytest.raises(UIValidationError):
        make_custom_correspondence([], pitch=P, image_size=IMG, **kwargs)


def test_custom_point_is_paired_with_its_own_metres_in_the_arrays() -> None:
    ix, iy = px_to_image(30.0, 20.0)
    custom = make_custom_correspondence([], pitch_x=30.0, pitch_y=20.0, image_x=ix, image_y=iy,
                                        pitch=P, frame_index=0)
    cs = [corr("corner_x0_y0"), custom, corr("corner_xL_yW"), corr("corner_xL_y0")]
    img, pit = correspondences_to_arrays(cs, P)
    assert pit[1] == [30.0, 20.0]
    assert img[1] == pytest.approx([ix, iy])
    assert pit[0] == [0.0, 0.0] and pit[2] == [105.0, 68.0]


def test_custom_points_calibrate_together_with_catalogue_points() -> None:
    customs = []
    # (0,0), (25,15), (80,50) would be almost collinear and rightly refused; use spread points.
    for x, y in ((25.0, 55.0), (80.0, 50.0)):
        ix, iy = px_to_image(x, y)
        customs.append(make_custom_correspondence(customs, pitch_x=x, pitch_y=y, image_x=ix,
                                                  image_y=iy, pitch=P, frame_index=0))
    cs = [corr("corner_x0_y0"), corr("corner_xL_y0")] + customs
    assert_matches_camera(build_calibration(cs, P, IMG))


def test_nearly_collinear_custom_choice_is_refused_with_an_explanation() -> None:
    customs = []
    for x, y in ((25.0, 15.0), (80.0, 50.0)):  # with the origin: y ~= 0.6 x
        ix, iy = px_to_image(x, y)
        customs.append(make_custom_correspondence(customs, pitch_x=x, pitch_y=y, image_x=ix,
                                                  image_y=iy, pitch=P, frame_index=0))
    cs = [corr("corner_x0_y0"), corr("corner_xL_y0")] + customs
    with pytest.raises(UIValidationError, match="cannot determine a homography"):
        build_calibration(cs, P, IMG)


def test_calibrating_from_custom_points_only() -> None:
    customs: list[Correspondence] = []
    for x, y in ((10.0, 10.0), (90.0, 12.0), (85.0, 58.0), (12.0, 55.0)):
        ix, iy = px_to_image(x, y)
        customs = add_correspondence(
            customs,
            make_custom_correspondence(customs, pitch_x=x, pitch_y=y, image_x=ix, image_y=iy,
                                       pitch=P, frame_index=0),
            P,
        )
    assert_matches_camera(build_calibration(customs, P, IMG))


# --- duplicates, replace, remove ---------------------------------------------

def test_adding_an_already_placed_landmark_is_refused_unless_replacing() -> None:
    cs = list(FOUR)
    again = Correspondence("corner_x0_y0", 999.0, 999.0, frame_index=0)
    with pytest.raises(DuplicateLandmarkError, match="already placed"):
        add_correspondence(cs, again, P)
    replaced = add_correspondence(cs, again, P, replace=True)
    assert len(replaced) == 4
    assert replaced[0].image_x == 999.0  # replaced in place, order preserved


def test_legacy_key_counts_as_the_same_landmark() -> None:
    with pytest.raises(DuplicateLandmarkError):
        add_correspondence(FOUR, Correspondence("corner_00", 5.0, 5.0, 0), P)


def test_custom_point_on_top_of_a_catalogue_landmark_is_refused() -> None:
    clash = make_custom_correspondence(FOUR, pitch_x=0.0, pitch_y=0.0, image_x=400, image_y=400, pitch=P)
    with pytest.raises(DuplicateLandmarkError, match="same pitch location"):
        add_correspondence(FOUR, clash, P)


def test_add_rejects_landmark_not_on_this_pitch() -> None:
    narrow = PitchDimensions(105.0, 40.0)
    with pytest.raises(UIValidationError, match="does not exist"):
        add_correspondence([], Correspondence("x0_pa_front_y0", 1.0, 1.0, 0), narrow)


def test_remove_correspondence_by_key_including_legacy_keys() -> None:
    assert [c.landmark_key for c in remove_correspondence(FOUR, "corner_xL_y0")] == [
        "corner_x0_y0", "corner_xL_yW", "corner_x0_yW",
    ]
    assert len(remove_correspondence(FOUR, "corner_00")) == 3  # legacy alias
    assert len(remove_correspondence(FOUR, "nonexistent")) == 4


# --- live assessment ---------------------------------------------------------

def test_assess_counts_and_blocks_until_four_valid_landmarks() -> None:
    s = assess_correspondences(FOUR[:3], P, IMG)
    assert (s.placed, s.valid) == (3, 3) and not s.can_build
    assert assess_correspondences(FOUR, P, IMG).can_build


def test_assess_flags_collinear_landmark_choices() -> None:
    cs = [corr(k) for k in ["halfway_y0", "centre_mark", "halfway_yW", "centre_circle_y0"]]
    s = assess_correspondences(cs, P, IMG)
    assert not s.can_build
    assert any("cannot determine a homography" in e for e in s.assessment.errors)


def test_assess_warns_when_landmarks_are_clustered() -> None:
    keys = ["x0_ga_front_y0", "x0_ga_front_yW", "x0_pa_front_y0", "x0_pa_front_yW"]
    s = assess_correspondences([corr(k) for k in keys], P, IMG)
    assert s.can_build
    assert "poor_pitch_coverage" in [w.code for w in s.assessment.warnings]


def test_build_calibration_reports_rejected_outlier_by_name() -> None:
    cs = [corr(k) for k in CORNER_KEYS + ["centre_mark", "halfway_y0"]]
    bad = cs[5]
    cs[5] = Correspondence(bad.landmark_key, bad.image_x + 12.0, bad.image_y, frame_index=0)
    calib = build_calibration(cs, P, IMG)
    rows = calibration_point_table(calib)
    assert len(rows) == 6
    assert [r["used"] for r in rows] == [True, True, True, True, True, False]
    assert "Halfway line" in rows[5]["landmark"]
    assert rows[5]["error_px"] > 8.0


def test_build_calibration_rejects_duplicates_and_too_few() -> None:
    with pytest.raises(UIValidationError):
        build_calibration(FOUR[:3], P, IMG)
    with pytest.raises(UIValidationError):
        build_calibration(FOUR + [FOUR[0]], P, IMG)


# --- pitch dimensions --------------------------------------------------------

def test_validate_pitch_dimensions_accepts_defaults_and_custom() -> None:
    assert validate_pitch_dimensions(105.0, 68.0) == PitchDimensions(105.0, 68.0)
    assert validate_pitch_dimensions(100, 64) == PitchDimensions(100.0, 64.0)


@pytest.mark.parametrize(
    "length, width",
    [(105.0, 105.0), (60.0, 68.0), (float("nan"), 68.0), (105.0, float("inf")),
     (10.0, 5.0), (500.0, 68.0), (105.0, 200.0), ("a", 68.0)],
)
def test_validate_pitch_dimensions_rejects_bad_values(length, width) -> None:
    with pytest.raises(UIValidationError):
        validate_pitch_dimensions(length, width)


def test_pitch_error_messages_state_units() -> None:
    with pytest.raises(UIValidationError, match=r"\bm\b"):
        validate_pitch_dimensions(500.0, 68.0)


# --- staleness: pitch change / landmark change / other frame ------------------

def _built(pitch: PitchDimensions = P, cs=None):
    cs = list(FOUR) if cs is None else cs
    return cs, build_calibration(cs, pitch, IMG), calibration_input_signature(cs, pitch)


def test_no_calibration_is_reported_missing() -> None:
    assert check_calibration(None, None, FOUR, P, 0).status is CalibrationStatus.MISSING


def test_fresh_calibration_on_its_own_frame_is_ready() -> None:
    cs, calib, sig = _built()
    check = check_calibration(calib, sig, cs, P, 0)
    assert check.ready and check.status is CalibrationStatus.READY


def test_changing_the_pitch_dimensions_invalidates_the_calibration() -> None:
    cs, calib, sig = _built()
    for other in (PitchDimensions(100.0, 68.0), PitchDimensions(105.0, 64.0)):
        assert check_calibration(calib, sig, cs, other, 0).status is CalibrationStatus.OUTDATED
    assert calibration_input_signature(cs, P) != calibration_input_signature(cs, PitchDimensions(100.0, 68.0))


def test_changing_the_landmarks_invalidates_the_calibration() -> None:
    cs, calib, sig = _built()
    moved = list(cs)
    moved[0] = Correspondence(cs[0].landmark_key, cs[0].image_x + 3.0, cs[0].image_y, 0)
    assert check_calibration(calib, sig, moved, P, 0).status is CalibrationStatus.OUTDATED
    extra = cs + [corr("centre_mark")]
    assert check_calibration(calib, sig, extra, P, 0).status is CalibrationStatus.OUTDATED
    assert check_calibration(calib, sig, cs[:3], P, 0).status is CalibrationStatus.OUTDATED


def test_changing_a_custom_pitch_coordinate_invalidates_the_calibration() -> None:
    ix, iy = px_to_image(30.0, 20.0)
    a = make_custom_correspondence([], pitch_x=30.0, pitch_y=20.0, image_x=ix, image_y=iy, pitch=P, frame_index=0)
    b = make_custom_correspondence([], pitch_x=31.0, pitch_y=20.0, image_x=ix, image_y=iy, pitch=P, frame_index=0)
    base = FOUR[:3]
    assert calibration_input_signature(base + [a], P) != calibration_input_signature(base + [b], P)


def test_calibration_made_on_another_frame_is_not_ready_here() -> None:
    cs, calib, sig = _built()
    check = check_calibration(calib, sig, cs, P, frame_index=7)
    assert check.status is CalibrationStatus.WRONG_FRAME
    assert "frame 0" in check.message and "frame 7" in check.message
    assert "Recalibrate" in check.message


def test_points_without_a_recorded_frame_are_not_flagged() -> None:
    cs = [corr(k, frame_index=None) for k in CORNER_KEYS]
    calib = build_calibration(cs, P, IMG)
    sig = calibration_input_signature(cs, P)
    assert check_calibration(calib, sig, cs, P, 12).ready


def test_correspondence_frames_lists_distinct_source_frames() -> None:
    cs = [corr("corner_x0_y0", 3), corr("corner_xL_y0", 3), corr("centre_mark", 9), corr("halfway_y0", None)]
    assert correspondence_frames(cs) == (3, 9)


def test_analysis_signature_depends_on_pitch() -> None:
    kw = dict(video_name="v", frame_index=0, attacking_direction=AttackingDirection.POSITIVE_X,
              correspondences=FOUR, team_sides={"p0": TeamSide.ATTACKING}, ball_xy=(1.0, 2.0))
    a = analysis_signature(pitch=P, **kw)
    assert a == analysis_signature(pitch=P, **kw)
    assert a != analysis_signature(pitch=PitchDimensions(100.0, 68.0), **kw)
    assert analysis_signature(**kw)  # still callable without a pitch (compat)


def test_readiness_uses_the_specific_calibration_problem() -> None:
    m = validate_ready_to_run(
        has_video=True, frame_selected=True, has_calibration=False,
        calibration_problem="Recalibrate on this frame.",
        attacking_direction=AttackingDirection.POSITIVE_X,
        player_ids=["p0"], team_sides={"p0": TeamSide.ATTACKING},
    )
    assert not m.ok and "Recalibrate on this frame." in m.reasons


# --- attacking direction -------------------------------------------------------

def test_direction_target_x_for_both_directions_and_pitch_sizes() -> None:
    assert direction_target_x(AttackingDirection.POSITIVE_X, P) == 105.0
    assert direction_target_x(AttackingDirection.NEGATIVE_X, P) == 0.0
    assert direction_target_x(AttackingDirection.POSITIVE_X, PitchDimensions(100.0, 64.0)) == 100.0


def test_direction_text_mentions_the_attacked_goal_line() -> None:
    assert "105" in direction_label(AttackingDirection.POSITIVE_X, P)
    assert "X = 0" in direction_label(AttackingDirection.NEGATIVE_X, P)
    pos = describe_attacking_direction(AttackingDirection.POSITIVE_X, P)
    neg = describe_attacking_direction(AttackingDirection.NEGATIVE_X, P)
    assert "X = 105" in pos and "larger" in pos
    assert "X = 0" in neg and "smaller" in neg


@pytest.mark.parametrize("direction, expect_right", [(AttackingDirection.POSITIVE_X, True),
                                                      (AttackingDirection.NEGATIVE_X, False)])
def test_diagram_highlights_the_attacked_goal_line_on_the_correct_side(direction, expect_right) -> None:
    img = render_direction_diagram(P, direction)
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
    mask = np.all(img == np.array(DIAGRAM_TARGET_RGB, dtype=np.uint8), axis=2)
    assert mask.any()
    mean_x = float(np.mean(np.nonzero(mask)[1]))
    assert (mean_x > img.shape[1] / 2) == expect_right


def test_diagram_aspect_follows_the_pitch_dimensions() -> None:
    wide = render_direction_diagram(PitchDimensions(110.0, 60.0), AttackingDirection.POSITIVE_X)
    squat = render_direction_diagram(PitchDimensions(100.0, 80.0), AttackingDirection.POSITIVE_X)
    assert squat.shape[0] > wide.shape[0]


@pytest.mark.parametrize("direction", list(AttackingDirection))
def test_displayed_direction_agrees_with_the_offside_calculation(direction) -> None:
    # "toward" is derived ONLY from what the UI shows as the attacked goal line.
    toward = 1.0 if direction_target_x(direction, P) > P.length / 2 else -1.0
    mid = P.length / 2

    def verdict(attacker_offset: float) -> OffsideStatus:
        defenders = [PlayerOnPitch("d1", mid + toward * 30, 30.0, TeamSide.DEFENDING),
                     PlayerOnPitch("d2", mid + toward * 20, 40.0, TeamSide.DEFENDING)]
        attacker = PlayerOnPitch("a", mid + toward * attacker_offset, 34.0, TeamSide.ATTACKING)
        ball = BallPosition(mid + toward * 8, 34.0, "manual")
        return evaluate_offside([attacker], defenders, ball, direction, P.length).status

    assert verdict(40.0) is OffsideStatus.OFFSIDE_POSITION   # beyond the line, toward the shown goal
    assert verdict(10.0) is OffsideStatus.ONSIDE             # behind the line


# --- click handling ------------------------------------------------------------

def test_click_is_scaled_from_displayed_to_true_frame_pixels() -> None:
    # 1800 px wide frame shown at 900 px: a click at (450, 200) is (900, 400) in the frame.
    assert click_to_frame_xy({"x": 450, "y": 200, "width": 900}, 1800, 900) == (900.0, 400.0)


def test_click_is_unscaled_when_shown_at_full_size() -> None:
    assert click_to_frame_xy({"x": 321, "y": 123, "width": 640}, 640, 640) == (321.0, 123.0)


def test_click_uses_the_components_reported_width_over_the_requested_one() -> None:
    # The component actually shows 600 px even though 900 was requested.
    assert click_to_frame_xy({"x": 300, "y": 150, "width": 600}, 1200, 900) == (600.0, 300.0)


def test_click_falls_back_to_requested_width_when_none_is_reported() -> None:
    assert click_to_frame_xy({"x": 450, "y": 100}, 1800, 900) == (900.0, 200.0)
    assert click_to_frame_xy({"x": 5, "y": 6, "width": 0}, 1800, 900) == (10.0, 12.0)


def test_click_stamp_identifies_a_click_event() -> None:
    assert click_stamp({"x": 1, "y": 2, "unix_time": 123}) == 123
    assert click_stamp({"x": 1, "y": 2}) == (1, 2)
    assert click_stamp({"x": 1, "y": 2, "unix_time": 1}) != click_stamp({"x": 1, "y": 2, "unix_time": 2})
