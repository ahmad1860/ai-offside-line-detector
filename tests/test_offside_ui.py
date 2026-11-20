"""Tests for the non-UI Streamlit helper logic (app/offside_ui.py).

These do not import or run Streamlit. Deterministic; no weights/internet/GPU.
"""

from __future__ import annotations

import pytest

from app.offside_ui import (
    PITCH_LANDMARKS,
    Correspondence,
    UIValidationError,
    analysis_signature,
    build_calibration,
    correspondences_to_arrays,
    landmark_by_key,
    parse_attacking_direction,
    prepare_ball_xy,
    prepare_team_assignment,
    validate_ready_to_run,
)
from offside.calibration import Calibration, PitchDimensions
from offside.offside import AttackingDirection, TeamSide

_PITCH = PitchDimensions()  # 105 x 68

# Four corner landmarks give a clean, non-degenerate calibration.
_FOUR = [
    Correspondence("corner_00", 100, 700),
    Correspondence("corner_10", 1180, 700),
    Correspondence("corner_11", 980, 300),
    Correspondence("corner_01", 300, 300),
]


# --- landmark catalogue ------------------------------------------------------

def test_landmark_lookup_and_pitch_coordinates() -> None:
    lm = landmark_by_key("corner_11")
    assert lm.pitch_xy(_PITCH) == (105.0, 68.0)
    assert landmark_by_key("corner_00").pitch_xy(_PITCH) == (0.0, 0.0)


def test_unknown_landmark_rejected() -> None:
    with pytest.raises(UIValidationError):
        landmark_by_key("not_a_landmark")


def test_all_landmarks_within_pitch() -> None:
    for lm in PITCH_LANDMARKS:
        x, y = lm.pitch_xy(_PITCH)
        assert 0.0 <= x <= 105.0
        assert 0.0 <= y <= 68.0


# --- landmark -> correspondence conversion ----------------------------------

def test_correspondences_to_arrays_shapes() -> None:
    img, pitch = correspondences_to_arrays(_FOUR, _PITCH)
    assert len(img) == 4 and len(pitch) == 4
    assert pitch[0] == [0.0, 0.0]  # corner_00


def test_too_few_correspondences_rejected() -> None:
    with pytest.raises(UIValidationError):
        correspondences_to_arrays(_FOUR[:3], _PITCH)


def test_duplicate_landmark_rejected() -> None:
    dup = _FOUR + [Correspondence("corner_00", 50, 50)]
    with pytest.raises(UIValidationError):
        correspondences_to_arrays(dup, _PITCH)


# --- calibration building ----------------------------------------------------

def test_build_calibration_success() -> None:
    calib = build_calibration(_FOUR, _PITCH)
    assert isinstance(calib, Calibration)
    assert calib.reprojection_error_px < 1e-6


def test_build_calibration_collinear_rejected() -> None:
    collinear = [
        Correspondence("corner_00", 0, 0),
        Correspondence("corner_10", 1, 1),
        Correspondence("corner_11", 2, 2),
        Correspondence("corner_01", 3, 3),
    ]
    with pytest.raises(UIValidationError):
        build_calibration(collinear, _PITCH)


# --- team assignment preparation --------------------------------------------

def test_prepare_team_assignment_success() -> None:
    sides = {"p0": TeamSide.ATTACKING, "p1": TeamSide.DEFENDING}
    ta = prepare_team_assignment(["p0", "p1"], sides)
    assert ta.team_of("p0") is TeamSide.ATTACKING


def test_prepare_team_assignment_missing_rejected() -> None:
    with pytest.raises(UIValidationError):
        prepare_team_assignment(["p0", "p1"], {"p0": TeamSide.ATTACKING})


def test_prepare_team_assignment_unknown_rejected() -> None:
    with pytest.raises(UIValidationError):
        prepare_team_assignment(["p0"], {"p0": TeamSide.ATTACKING, "pX": TeamSide.DEFENDING})


# --- ball coordinate preparation --------------------------------------------

def test_prepare_ball_xy_none() -> None:
    assert prepare_ball_xy(None) is None


def test_prepare_ball_xy_pair() -> None:
    assert prepare_ball_xy((12.0, 34.0)) == (12.0, 34.0)


def test_prepare_ball_xy_bad_length() -> None:
    with pytest.raises(UIValidationError):
        prepare_ball_xy((1.0, 2.0, 3.0))


def test_prepare_ball_xy_non_finite() -> None:
    with pytest.raises(UIValidationError):
        prepare_ball_xy((float("nan"), 2.0))


# --- attacking direction parsing --------------------------------------------

def test_parse_direction() -> None:
    assert parse_attacking_direction("positive_x") is AttackingDirection.POSITIVE_X
    assert parse_attacking_direction("negative_x") is AttackingDirection.NEGATIVE_X


def test_parse_direction_invalid() -> None:
    with pytest.raises(UIValidationError):
        parse_attacking_direction("sideways")


# --- stale-result signature --------------------------------------------------

def test_signature_stable_for_same_inputs() -> None:
    kw = dict(
        video_name="clip.mp4",
        frame_index=10,
        attacking_direction=AttackingDirection.POSITIVE_X,
        correspondences=_FOUR,
        team_sides={"p0": TeamSide.ATTACKING},
        ball_xy=(100.0, 200.0),
    )
    assert analysis_signature(**kw) == analysis_signature(**kw)


def test_signature_changes_with_inputs() -> None:
    base = dict(
        video_name="clip.mp4",
        frame_index=10,
        attacking_direction=AttackingDirection.POSITIVE_X,
        correspondences=_FOUR,
        team_sides={"p0": TeamSide.ATTACKING},
        ball_xy=(100.0, 200.0),
    )
    sig = analysis_signature(**base)
    # Change frame.
    assert analysis_signature(**{**base, "frame_index": 11}) != sig
    # Change ball.
    assert analysis_signature(**{**base, "ball_xy": (101.0, 200.0)}) != sig
    # Change team.
    assert analysis_signature(**{**base, "team_sides": {"p0": TeamSide.DEFENDING}}) != sig
    # Change direction.
    assert analysis_signature(**{**base, "attacking_direction": AttackingDirection.NEGATIVE_X}) != sig


# --- readiness validation ----------------------------------------------------

def test_validate_ready_all_present() -> None:
    m = validate_ready_to_run(
        has_video=True,
        frame_selected=True,
        has_calibration=True,
        attacking_direction=AttackingDirection.POSITIVE_X,
        player_ids=["p0", "p1"],
        team_sides={"p0": TeamSide.ATTACKING, "p1": TeamSide.DEFENDING},
    )
    assert m.ok
    assert m.reasons == ()


def test_validate_ready_collects_all_missing() -> None:
    m = validate_ready_to_run(
        has_video=False,
        frame_selected=False,
        has_calibration=False,
        attacking_direction=None,
        player_ids=[],
        team_sides={},
    )
    assert not m.ok
    assert len(m.reasons) >= 4


def test_validate_ready_flags_unassigned_players() -> None:
    m = validate_ready_to_run(
        has_video=True,
        frame_selected=True,
        has_calibration=True,
        attacking_direction=AttackingDirection.POSITIVE_X,
        player_ids=["p0", "p1"],
        team_sides={"p0": TeamSide.ATTACKING},
    )
    assert not m.ok
    assert any("p1" in r for r in m.reasons)


def test_validate_ready_ball_not_required() -> None:
    # Ball is intentionally not a prerequisite (missing ball -> INSUFFICIENT_DATA).
    m = validate_ready_to_run(
        has_video=True,
        frame_selected=True,
        has_calibration=True,
        attacking_direction=AttackingDirection.POSITIVE_X,
        player_ids=["p0", "p1"],
        team_sides={"p0": TeamSide.ATTACKING, "p1": TeamSide.DEFENDING},
    )
    assert m.ok
