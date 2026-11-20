"""Tests for the annotation module.

Builds FrameAnalysis objects directly (no pipeline/detector needed) on synthetic
frames. No weights, internet, or GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

import cv2

from offside.annotate import (
    DEFAULT_STYLE,
    AnnotationStyle,
    annotate_frame,
    direction_arrow_points,
    draw_calibration_overlay,
    draw_correspondence_markers,
    fit_text,
    label_rect,
    style_metrics,
)
from offside.calibration import calibrate, make_correspondences
from offside.geometry import image_to_pitch
from offside.detection import BoundingBox, Detection, ObjectClass
from offside.offside import (
    AttackerClassification,
    AttackingDirection,
    BallPosition,
    OffsideResult,
    OffsideStatus,
    ReferenceKind,
    TeamSide,
)
from offside.pipeline import (
    BallInfo,
    BallSource,
    DetectedPlayer,
    FrameAnalysis,
)
from offside.team_cluster import assign_manual

_IMG_QUAD = [[0, 700], [1000, 700], [1000, 0], [0, 0]]
_PITCH_QUAD = [[0, 0], [105, 0], [105, 68], [0, 68]]


def _calib():
    return calibrate(make_correspondences(_IMG_QUAD, _PITCH_QUAD))


def _frame():
    return np.full((700, 1000, 3), 30, dtype=np.uint8)


def _player(pid: str, cx: float, foot_y: float):
    box = BoundingBox(cx - 20, foot_y - 80, cx + 20, foot_y)
    det = Detection(box=box, confidence=0.9, coco_class_id=0, object_class=ObjectClass.PLAYER)
    return DetectedPlayer(
        player_id=pid,
        detection=det,
        foot_image_xy=(cx, foot_y),
        pitch_xy=(cx / 1000.0 * 105.0, 0.0),
    )


def _analysis(status: OffsideStatus, *, ball: BallInfo, with_line: bool, attackers=()):
    offside = OffsideResult(
        status=status,
        explanation="test explanation that is reasonably short.",
        attacking_direction=AttackingDirection.POSITIVE_X,
        offside_line_x=60.0 if with_line else None,
        reference_kind=ReferenceKind.SECOND_LAST_OPPONENT if with_line else None,
        ball=BallPosition(50.0, 34.0) if ball.pitch_xy else None,
        attackers=attackers,
    )
    players = (_player("p0", 900, 650), _player("p1", 550, 650), _player("p2", 600, 650))
    teams = assign_manual(
        ["p0", "p1", "p2"],
        {"p0": TeamSide.ATTACKING, "p1": TeamSide.DEFENDING, "p2": TeamSide.DEFENDING},
    )
    return FrameAnalysis(
        frame_index=3,
        attacking_direction=AttackingDirection.POSITIVE_X,
        players=players,
        ball=ball,
        offside=offside,
        calibration=_calib(),
        team_assignment=teams,
    )


def test_does_not_mutate_input_frame() -> None:
    frame = _frame()
    original = frame.copy()
    ball = BallInfo(source=BallSource.MANUAL, image_xy=(500, 650), pitch_xy=(52.5, 0.0))
    out = annotate_frame(frame, _analysis(OffsideStatus.ONSIDE, ball=ball, with_line=True))
    assert np.array_equal(frame, original)
    assert not np.array_equal(out, original)
    assert out.shape == frame.shape


def test_confident_result_draws_offside_line() -> None:
    ball = BallInfo(source=BallSource.MANUAL, image_xy=(500, 650), pitch_xy=(52.5, 0.0))
    analysis = _analysis(OffsideStatus.OFFSIDE_POSITION, ball=ball, with_line=True)
    frame = _frame()
    out = annotate_frame(frame, analysis)
    # The offside line is red (0,0,255); confirm red pixels exist where none did.
    red = np.logical_and.reduce([out[:, :, 2] > 200, out[:, :, 1] < 80, out[:, :, 0] < 80])
    assert red.any()


def test_insufficient_data_draws_no_offside_line() -> None:
    ball = BallInfo(source=BallSource.UNAVAILABLE, note="no ball")
    analysis = _analysis(OffsideStatus.INSUFFICIENT_DATA, ball=ball, with_line=False)
    out = annotate_frame(_frame(), analysis)
    # No long red horizontal-ish line should be drawn. We check that the count of
    # pure-red line pixels is very small (text/banners use other colours).
    red = np.logical_and.reduce([out[:, :, 2] > 200, out[:, :, 1] < 80, out[:, :, 0] < 80])
    # Allow a tiny number of incidental red pixels, but not a full pitch-width line.
    assert red.sum() < 50


def _has_color(img: np.ndarray, bgr: tuple[int, int, int]) -> np.ndarray:
    return np.all(img == np.array(bgr, dtype=np.uint8), axis=2)


def test_player_boxes_use_cyan_and_blue_and_labels_use_orange() -> None:
    ball = BallInfo(source=BallSource.MANUAL, image_xy=(500, 650), pitch_xy=(52.5, 0.0))
    out = annotate_frame(_frame(), _analysis(OffsideStatus.ONSIDE, ball=ball, with_line=True))
    s = DEFAULT_STYLE
    assert _has_color(out, s.attacker_box).any()          # cyan box (p0 is attacking)
    assert _has_color(out, s.defender_box).any()          # blue box (p1, p2 defending)
    assert _has_color(out, s.label_background).any()      # orange id-label background
    assert _has_color(out, s.label_text).any()            # white label text
    # Old green box/label colour must be gone (it vanished against the pitch).
    assert not _has_color(out, (0, 200, 0)).any()


def _hue(bgr: tuple[int, int, int]) -> int:
    px = np.array([[bgr]], dtype=np.uint8)
    return int(cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0, 0])  # OpenCV hue: 0..179


def test_default_colours_follow_the_requested_design() -> None:
    s = DEFAULT_STYLE
    assert 85 <= _hue(s.attacker_box) <= 130   # cyan/blue family
    assert 85 <= _hue(s.defender_box) <= 130
    assert 5 <= _hue(s.label_background) <= 25  # amber/orange
    assert s.label_text == (255, 255, 255)      # white
    assert _hue(s.offside_line) in (0, 179) or _hue(s.offside_line) < 5  # red


def test_each_meaning_has_its_own_colour() -> None:
    s = DEFAULT_STYLE
    meanings = {
        "attacker box": s.attacker_box, "defender box": s.defender_box,
        "id label": s.label_background, "offside line": s.offside_line,
        "ball": s.ball, "ON pill": s.onside_mark_background,
        "OFF pill": s.offside_mark_background, "calibration": s.calibration,
        "unassigned box": s.unassigned_box,
    }
    assert len(set(meanings.values())) == len(meanings)
    # The offside line must not share a colour with any player box.
    assert s.offside_line not in (s.attacker_box, s.defender_box, s.unassigned_box)


def test_custom_style_is_honoured() -> None:
    style = AnnotationStyle(attacker_box=(1, 2, 3), label_background=(4, 5, 6))
    ball = BallInfo(source=BallSource.MANUAL, image_xy=(500, 650), pitch_xy=(52.5, 0.0))
    out = annotate_frame(_frame(), _analysis(OffsideStatus.ONSIDE, ball=ball, with_line=True), style)
    assert _has_color(out, (1, 2, 3)).any()
    assert _has_color(out, (4, 5, 6)).any()
    assert not _has_color(out, DEFAULT_STYLE.attacker_box).any()


def test_offside_line_is_red_with_a_white_casing() -> None:
    ball = BallInfo(source=BallSource.MANUAL, image_xy=(500, 650), pitch_xy=(52.5, 0.0))
    out = annotate_frame(_frame(), _analysis(OffsideStatus.OFFSIDE_POSITION, ball=ball, with_line=True))
    assert _has_color(out, DEFAULT_STYLE.offside_line).any()
    assert _has_color(out, DEFAULT_STYLE.offside_line_casing).any()


def test_label_for_a_player_at_the_top_edge_is_not_hidden_under_the_banner() -> None:
    # Regression: a box touching the top edge got its label painted over by the banner.
    box = BoundingBox(300, 0, 340, 120)
    det = Detection(box=box, confidence=0.9, coco_class_id=0, object_class=ObjectClass.PLAYER)
    player = DetectedPlayer("p0", det, (320.0, 120.0), (30.0, 5.0))
    base = _analysis(OffsideStatus.INSUFFICIENT_DATA,
                     ball=BallInfo(source=BallSource.UNAVAILABLE), with_line=False)
    teams = assign_manual(["p0"], {"p0": TeamSide.DEFENDING})
    analysis = FrameAnalysis(3, base.attacking_direction, (player,), base.ball, base.offside,
                             base.calibration, teams)
    out = annotate_frame(_frame(), analysis)
    banner_h = style_metrics(DEFAULT_STYLE, out.shape).banner_height
    orange = _has_color(out, DEFAULT_STYLE.label_background)
    assert orange[banner_h:, :].any()      # label sits below the banner
    assert not orange[:banner_h, :].any()  # nothing orange is hidden under it


def test_unassigned_player_gets_neutral_box_and_question_mark_label() -> None:
    base = _analysis(OffsideStatus.INSUFFICIENT_DATA,
                     ball=BallInfo(source=BallSource.UNAVAILABLE), with_line=False)
    analysis = FrameAnalysis(3, base.attacking_direction, base.players, base.ball, base.offside,
                             base.calibration, None)
    out = annotate_frame(_frame(), analysis)
    assert _has_color(out, DEFAULT_STYLE.unassigned_box).any()


# --- layout helpers ----------------------------------------------------------

@pytest.mark.parametrize(
    "anchor_x, anchor_top, anchor_bottom, prefer",
    [
        (0, 0, 40, "above"), (-30, 5, 60, "above"), (990, 300, 400, "above"),
        (500, 695, 700, "below"), (1100, 10, 20, "below"), (0, 350, 360, "above"),
    ],
)
def test_label_rect_always_inside_the_image(anchor_x, anchor_top, anchor_bottom, prefer) -> None:
    x0, y0, x1, y1 = label_rect((120, 24), anchor_x, anchor_top, anchor_bottom, 1000, 700, 4, prefer)
    assert 0 <= x0 < x1 <= 1000
    assert 0 <= y0 < y1 <= 700


def test_label_rect_respects_min_y() -> None:
    _x0, y0, _x1, _y1 = label_rect((120, 24), 10, 0, 40, 1000, 700, 4, "above", min_y=34)
    assert y0 >= 34


def test_fit_text_shortens_to_fit_and_leaves_short_text_alone() -> None:
    assert fit_text("short", 400, 0.6, 2) == "short"
    long = "x" * 300
    fitted = fit_text(long, 200, 0.6, 2)
    assert fitted.endswith("...") and len(fitted) < len(long)
    (w, _h), _b = cv2.getTextSize(fitted, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    assert w <= 200


def test_style_metrics_scale_with_frame_width_and_clamp() -> None:
    small = style_metrics(DEFAULT_STYLE, (360, 640, 3))
    ref = style_metrics(DEFAULT_STYLE, (720, 1280, 3))
    big = style_metrics(DEFAULT_STYLE, (1080, 1920, 3))
    huge = style_metrics(DEFAULT_STYLE, (4000, 8000, 3))
    assert small.scale == DEFAULT_STYLE.min_scale
    assert ref.scale == pytest.approx(1.0)
    assert big.scale == pytest.approx(1.5)
    assert huge.scale == DEFAULT_STYLE.max_scale
    assert small.font < ref.font < big.font
    assert all(v >= 1 for v in (small.box, small.line, small.text_thickness, small.foot, small.pad))


# --- calibration overlay and markers -----------------------------------------

def test_calibration_overlay_draws_pitch_lines_without_mutating() -> None:
    frame = _frame()
    before = frame.copy()
    out = draw_calibration_overlay(frame, _calib(), AttackingDirection.POSITIVE_X)
    assert np.array_equal(frame, before)
    assert _has_color(out, DEFAULT_STYLE.calibration).any()


def test_calibration_overlay_draws_the_arrow_only_when_a_direction_is_given() -> None:
    # (The arrow and the label text are both white, so compare renders instead of colours.)
    calib = _calib()
    without = draw_calibration_overlay(_frame(), calib, None)
    with_dir = draw_calibration_overlay(_frame(), calib, AttackingDirection.POSITIVE_X)
    assert _has_color(without, DEFAULT_STYLE.calibration).any()
    differs = np.any(with_dir != without, axis=2)
    assert differs.any()
    _tail, (hx, hy) = direction_arrow_points(calib, AttackingDirection.POSITIVE_X)
    assert differs[max(0, hy - 4): hy + 5, max(0, hx - 4): hx + 5].any()  # drawn at the head


def test_direction_arrow_points_toward_the_attacked_end_for_both_directions() -> None:
    calib = _calib()
    for direction, sign in ((AttackingDirection.POSITIVE_X, +1), (AttackingDirection.NEGATIVE_X, -1)):
        tail, head = direction_arrow_points(calib, direction)
        tail_x = float(image_to_pitch(calib, list(tail))[0])
        head_x = float(image_to_pitch(calib, list(head))[0])
        assert (head_x - tail_x) * sign > 0, direction
        assert abs(head_x - tail_x) > 5.0  # a visible arrow, not a dot


def test_markers_are_numbered_drawn_and_do_not_mutate() -> None:
    frame = _frame()
    before = frame.copy()
    out = draw_correspondence_markers(frame, [("a", 100, 100), ("b", 300, 200)], numbers=[2, 5])
    assert np.array_equal(frame, before)
    assert _has_color(out, DEFAULT_STYLE.calibration).any()
    assert _has_color(out, DEFAULT_STYLE.label_text).any()  # the number text


def test_markers_reject_mismatched_numbers_and_bad_frame() -> None:
    with pytest.raises(ValueError):
        draw_correspondence_markers(_frame(), [("a", 1, 1)], numbers=[1, 2])
    with pytest.raises(ValueError):
        draw_correspondence_markers(np.zeros((5, 5), dtype=np.uint8), [])
    with pytest.raises(ValueError):
        draw_calibration_overlay(np.zeros((5, 5), dtype=np.uint8), _calib())


def test_ball_annotated_with_source() -> None:
    ball = BallInfo(source=BallSource.DETECTOR, image_xy=(500, 650), pitch_xy=(52.5, 0.0),
                    detector_confidence=0.8)
    out = annotate_frame(_frame(), _analysis(OffsideStatus.ONSIDE, ball=ball, with_line=True))
    # Ball drawn in yellow (0,255,255).
    yellow = np.logical_and.reduce([out[:, :, 0] < 80, out[:, :, 1] > 200, out[:, :, 2] > 200])
    assert yellow.any()


def test_rejects_invalid_frame() -> None:
    ball = BallInfo(source=BallSource.UNAVAILABLE)
    analysis = _analysis(OffsideStatus.INSUFFICIENT_DATA, ball=ball, with_line=False)
    with pytest.raises(ValueError):
        annotate_frame(np.zeros((10, 10), dtype=np.uint8), analysis)  # not 3-channel


def test_per_attacker_marks_for_confident_result() -> None:
    attackers = (
        AttackerClassification("p0", OffsideStatus.OFFSIDE_POSITION, 90.0, 34.0, 30.0, False),
    )
    ball = BallInfo(source=BallSource.MANUAL, image_xy=(500, 650), pitch_xy=(52.5, 0.0))
    analysis = _analysis(OffsideStatus.OFFSIDE_POSITION, ball=ball, with_line=True,
                         attackers=attackers)
    out = annotate_frame(_frame(), analysis)
    assert out.shape == (700, 1000, 3)  # renders without error
