"""Tests for the single-frame integration pipeline.

Synthetic clip + mocked detector + known calibration. No real weights, internet,
or GPU. The homography here maps image pixels to pitch metres such that a larger
image-x maps to a larger pitch-X (so geometry is easy to reason about).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from offside.calibration import calibrate, make_correspondences
from offside.detection import BoundingBox, Detection, ObjectClass
from offside.offside import AttackingDirection, OffsideStatus, TeamSide
from offside.pipeline import (
    BallSource,
    PipelineError,
    analyze_frame,
    assign_player_ids,
)
from offside.team_cluster import assign_manual

# --- shared synthetic calibration -------------------------------------------
# Image quad -> pitch quad. Image is ~1000x700; pitch 105x68.
_IMG_QUAD = [[0, 700], [1000, 700], [1000, 0], [0, 0]]
_PITCH_QUAD = [[0, 0], [105, 0], [105, 68], [0, 68]]


def _calib():
    return calibrate(make_correspondences(_IMG_QUAD, _PITCH_QUAD))


def _player(cx: float, foot_y: float, cls=ObjectClass.PLAYER, conf=0.9) -> Detection:
    # Build a box whose foot_point (bottom-centre) is (cx, foot_y).
    half_w, h = 20.0, 80.0
    box = BoundingBox(cx - half_w, foot_y - h, cx + half_w, foot_y)
    coco = 0 if cls is ObjectClass.PLAYER else 32
    return Detection(box=box, confidence=conf, coco_class_id=coco, object_class=cls)


def _ball(cx: float, cy: float, conf=0.9) -> Detection:
    box = BoundingBox(cx - 5, cy - 5, cx + 5, cy + 5)
    return Detection(box=box, confidence=conf, coco_class_id=32, object_class=ObjectClass.BALL)


class FakeDetector:
    """Returns a fixed detection list regardless of the frame."""

    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections

    def detect(self, frame):  # noqa: ANN001
        return list(self._detections)


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (120, 90))
    assert writer.isOpened()
    try:
        for i in range(15):
            writer.write(np.full((90, 120, 3), i * 10 % 255, dtype=np.uint8))
    finally:
        writer.release()
    return path


# --- id assignment ----------------------------------------------------------

def test_assign_player_ids_is_positional_and_deterministic() -> None:
    # Two players; ids are ordered by foot x.
    players = [_player(cx=800, foot_y=650), _player(cx=200, foot_y=650)]
    ids = assign_player_ids(players)
    # Player at x=200 should be p0, x=800 should be p1.
    assert ids == ["p1", "p0"]


# --- end-to-end success ------------------------------------------------------

def _run(clip, detections, teams, direction, ball_xy=None):
    assignment = assign_manual(list(teams.keys()), teams)
    return analyze_frame(
        video_path=clip,
        frame_index=3,
        detector=FakeDetector(detections),
        calibration=_calib(),
        attacking_direction=direction,
        team_assignment=assignment,
        manual_ball_image_xy=ball_xy,
    )


def test_end_to_end_offside_positive_x(clip: Path) -> None:
    # Attacking toward +X (larger image-x). Defenders at image x=600 (last) and
    # x=550 (second-last) => pitch line ~57.75. Attacker at x=900 is well beyond.
    dets = [
        _player(cx=900, foot_y=650),  # attacker (most goalward)
        _player(cx=600, foot_y=650),  # defender last
        _player(cx=550, foot_y=650),  # defender second-last
    ]
    ids = assign_player_ids([d for d in dets])
    # Map by position: x=550->p0, x=600->p1, x=900->p2
    teams = {"p0": TeamSide.DEFENDING, "p1": TeamSide.DEFENDING, "p2": TeamSide.ATTACKING}
    analysis = _run(clip, dets, teams, AttackingDirection.POSITIVE_X, ball_xy=(500, 650))
    assert analysis.status is OffsideStatus.OFFSIDE_POSITION
    assert analysis.ball.source is BallSource.MANUAL
    assert ids == ["p2", "p1", "p0"]


def test_end_to_end_onside(clip: Path) -> None:
    # ids are assigned by foot-x ascending: 540->p0, 570->p1, 600->p2.
    dets = [
        _player(cx=540, foot_y=650),  # p0: attacker, behind both defenders
        _player(cx=600, foot_y=650),  # p2: defender (last)
        _player(cx=570, foot_y=650),  # p1: defender (second-last)
    ]
    teams = {"p0": TeamSide.ATTACKING, "p1": TeamSide.DEFENDING, "p2": TeamSide.DEFENDING}
    analysis = _run(clip, dets, teams, AttackingDirection.POSITIVE_X, ball_xy=(500, 650))
    # Attacker p0 (x=540) is behind the second-last defender (x=570) -> onside.
    assert analysis.status is OffsideStatus.ONSIDE


def test_both_directions(clip: Path) -> None:
    dets = [
        _player(cx=100, foot_y=650),  # attacker near x=0 (goalward for -X)
        _player(cx=400, foot_y=650),  # defender
        _player(cx=450, foot_y=650),  # defender
    ]
    teams = {"p0": TeamSide.ATTACKING, "p1": TeamSide.DEFENDING, "p2": TeamSide.DEFENDING}
    analysis = _run(clip, dets, teams, AttackingDirection.NEGATIVE_X, ball_xy=(500, 650))
    # Attacking toward X=0; attacker at pitch-x ~10.5 is beyond defenders -> offside.
    assert analysis.status is OffsideStatus.OFFSIDE_POSITION


# --- ball handling -----------------------------------------------------------

def test_manual_ball_is_authoritative_over_detector(clip: Path) -> None:
    dets = [
        _player(cx=900, foot_y=650),
        _player(cx=600, foot_y=650),
        _player(cx=550, foot_y=650),
        _ball(cx=950, cy=600, conf=0.99),  # detector ball, but manual overrides
    ]
    teams = {"p0": TeamSide.DEFENDING, "p1": TeamSide.DEFENDING, "p2": TeamSide.ATTACKING}
    analysis = _run(clip, dets, teams, AttackingDirection.POSITIVE_X, ball_xy=(500, 650))
    assert analysis.ball.source is BallSource.MANUAL
    assert analysis.ball.image_xy == (500.0, 650.0)


def test_detector_ball_accepted_when_single_confident(clip: Path) -> None:
    dets = [
        _player(cx=900, foot_y=650),
        _player(cx=600, foot_y=650),
        _player(cx=550, foot_y=650),
        _ball(cx=500, cy=650, conf=0.8),
    ]
    teams = {"p0": TeamSide.DEFENDING, "p1": TeamSide.DEFENDING, "p2": TeamSide.ATTACKING}
    analysis = _run(clip, dets, teams, AttackingDirection.POSITIVE_X, ball_xy=None)
    assert analysis.ball.source is BallSource.DETECTOR
    assert analysis.ball.detector_confidence == pytest.approx(0.8)


def test_low_confidence_ball_is_unavailable_insufficient(clip: Path) -> None:
    dets = [
        _player(cx=900, foot_y=650),
        _player(cx=600, foot_y=650),
        _player(cx=550, foot_y=650),
        _ball(cx=500, cy=650, conf=0.2),  # below threshold
    ]
    teams = {"p0": TeamSide.DEFENDING, "p1": TeamSide.DEFENDING, "p2": TeamSide.ATTACKING}
    analysis = _run(clip, dets, teams, AttackingDirection.POSITIVE_X, ball_xy=None)
    assert analysis.ball.source is BallSource.UNAVAILABLE
    assert analysis.status is OffsideStatus.INSUFFICIENT_DATA


def test_multiple_ball_candidates_unavailable(clip: Path) -> None:
    dets = [
        _player(cx=900, foot_y=650),
        _player(cx=600, foot_y=650),
        _player(cx=550, foot_y=650),
        _ball(cx=500, cy=650, conf=0.8),
        _ball(cx=300, cy=650, conf=0.8),
    ]
    teams = {"p0": TeamSide.DEFENDING, "p1": TeamSide.DEFENDING, "p2": TeamSide.ATTACKING}
    analysis = _run(clip, dets, teams, AttackingDirection.POSITIVE_X, ball_xy=None)
    assert analysis.ball.source is BallSource.UNAVAILABLE
    assert analysis.status is OffsideStatus.INSUFFICIENT_DATA


# --- team-assignment validation ---------------------------------------------

def test_missing_team_assignment_raises(clip: Path) -> None:
    dets = [_player(cx=900, foot_y=650), _player(cx=600, foot_y=650), _player(cx=550, foot_y=650)]
    teams = {"p0": TeamSide.DEFENDING, "p1": TeamSide.DEFENDING}  # p2 missing
    with pytest.raises(PipelineError):
        _run(clip, dets, teams, AttackingDirection.POSITIVE_X, ball_xy=(500, 650))


def test_unknown_team_assignment_raises(clip: Path) -> None:
    dets = [_player(cx=900, foot_y=650), _player(cx=600, foot_y=650), _player(cx=550, foot_y=650)]
    teams = {
        "p0": TeamSide.DEFENDING, "p1": TeamSide.DEFENDING, "p2": TeamSide.ATTACKING,
        "p9": TeamSide.ATTACKING,  # not detected
    }
    with pytest.raises(PipelineError):
        _run(clip, dets, teams, AttackingDirection.POSITIVE_X, ball_xy=(500, 650))


# --- insufficient defenders --------------------------------------------------

def test_insufficient_defenders(clip: Path) -> None:
    dets = [_player(cx=900, foot_y=650), _player(cx=550, foot_y=650)]
    teams = {"p0": TeamSide.DEFENDING, "p1": TeamSide.ATTACKING}
    analysis = _run(clip, dets, teams, AttackingDirection.POSITIVE_X, ball_xy=(500, 650))
    assert analysis.status is OffsideStatus.INSUFFICIENT_DATA


# --- foot-point projection ---------------------------------------------------

def test_foot_point_used_for_projection(clip: Path) -> None:
    # A single player; verify the projected pitch point corresponds to the FOOT
    # (bottom-centre), not the box centre. foot_y=700 -> pitch Y=0 in this quad.
    dets = [
        _player(cx=500, foot_y=700),
        _player(cx=600, foot_y=700),
        _player(cx=550, foot_y=700),
    ]
    teams = {"p0": TeamSide.ATTACKING, "p1": TeamSide.DEFENDING, "p2": TeamSide.DEFENDING}
    analysis = _run(clip, dets, teams, AttackingDirection.POSITIVE_X, ball_xy=(500, 700))
    p0 = next(p for p in analysis.players if p.player_id == "p0")
    # foot at image y=700 maps to pitch Y=0 (bottom edge of the quad).
    assert p0.pitch_xy[1] == pytest.approx(0.0, abs=1e-6)


# --- invalid frame index -----------------------------------------------------

def test_invalid_frame_index(clip: Path) -> None:
    dets = [_player(cx=900, foot_y=650), _player(cx=600, foot_y=650), _player(cx=550, foot_y=650)]
    teams = {"p0": TeamSide.DEFENDING, "p1": TeamSide.DEFENDING, "p2": TeamSide.ATTACKING}
    assignment = assign_manual(list(teams.keys()), teams)
    from offside.video_io import VideoReadError

    with pytest.raises(VideoReadError):
        analyze_frame(
            video_path=clip,
            frame_index=9999,  # beyond clip
            detector=FakeDetector(dets),
            calibration=_calib(),
            attacking_direction=AttackingDirection.POSITIVE_X,
            team_assignment=assignment,
            manual_ball_image_xy=(500, 650),
        )
