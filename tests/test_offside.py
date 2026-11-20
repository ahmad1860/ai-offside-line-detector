"""Tests for offside-position geometry and decision logic.

Deterministic synthetic pitch coordinates (metres) with known football answers.
Pitch length 105 m => halfway line at X = 52.5 m. No weights/internet/GPU.

Reasoning aids:
* POSITIVE_X: attacking toward X=105; goalward = larger X; opponents' half X>52.5.
* NEGATIVE_X: attacking toward X=0;  goalward = smaller X; opponents' half X<52.5.
"""

from __future__ import annotations

import math

import pytest

from offside.offside import (
    AttackerClassification,
    AttackingDirection,
    BallPosition,
    OffsideError,
    OffsideResult,
    OffsideStatus,
    PlayerOnPitch,
    ReferenceKind,
    TeamSide,
    evaluate_offside,
    second_last_opponent,
)

PITCH_LEN = 105.0


def atk(pid: str, x: float, y: float = 34.0) -> PlayerOnPitch:
    return PlayerOnPitch(pid, x, y, TeamSide.ATTACKING)


def dfn(pid: str, x: float, y: float = 34.0) -> PlayerOnPitch:
    return PlayerOnPitch(pid, x, y, TeamSide.DEFENDING)


def _classification(result: OffsideResult, pid: str) -> AttackerClassification:
    return next(c for c in result.attackers if c.player_id == pid)


# --- 1. Clear offside attacker ----------------------------------------------

def test_clear_offside_attacker_positive_x() -> None:
    # Defenders at X=80 (last) and X=75 (second-last). Ball behind at X=60.
    # Attacker at X=90 is beyond the second-last opponent -> offside position.
    result = evaluate_offside(
        attackers=[atk("A", 90.0)],
        defenders=[dfn("D1", 80.0), dfn("D2", 75.0)],
        ball=BallPosition(60.0, 34.0),
        direction=AttackingDirection.POSITIVE_X,
        pitch_length=PITCH_LEN,
    )
    assert result.status is OffsideStatus.OFFSIDE_POSITION
    assert result.offside_line_x == pytest.approx(75.0)
    assert result.reference_kind is ReferenceKind.SECOND_LAST_OPPONENT
    assert _classification(result, "A").status is OffsideStatus.OFFSIDE_POSITION


# --- 2. Clear onside attacker -----------------------------------------------

def test_clear_onside_attacker() -> None:
    # Attacker at X=70 is behind the second-last opponent (X=75) -> onside.
    result = evaluate_offside(
        attackers=[atk("A", 70.0)],
        defenders=[dfn("D1", 80.0), dfn("D2", 75.0)],
        ball=BallPosition(60.0, 34.0),
        direction=AttackingDirection.POSITIVE_X,
        pitch_length=PITCH_LEN,
    )
    assert result.status is OffsideStatus.ONSIDE
    assert _classification(result, "A").status is OffsideStatus.ONSIDE


# --- 3. Attacker exactly level with the line --------------------------------

def test_attacker_level_is_onside() -> None:
    # Attacker exactly on the line (X=75) -> level -> onside.
    result = evaluate_offside(
        attackers=[atk("A", 75.0)],
        defenders=[dfn("D1", 80.0), dfn("D2", 75.0)],
        ball=BallPosition(60.0, 34.0),
        direction=AttackingDirection.POSITIVE_X,
        pitch_length=PITCH_LEN,
    )
    assert _classification(result, "A").status is OffsideStatus.ONSIDE


# --- 4. Ball more goalward than the second-last opponent --------------------

def test_ball_more_goalward_defines_line() -> None:
    # Ball at X=85 is more goalward than second-last opponent (X=75).
    # Attacker at X=80 is beyond the defender but behind the ball -> onside.
    result = evaluate_offside(
        attackers=[atk("A", 80.0)],
        defenders=[dfn("D1", 90.0), dfn("D2", 75.0)],
        ball=BallPosition(85.0, 34.0),
        direction=AttackingDirection.POSITIVE_X,
        pitch_length=PITCH_LEN,
    )
    assert result.reference_kind is ReferenceKind.BALL
    assert result.offside_line_x == pytest.approx(85.0)
    assert _classification(result, "A").status is OffsideStatus.ONSIDE


# --- 5. Second-last opponent more goalward than the ball --------------------

def test_defender_more_goalward_defines_line() -> None:
    result = evaluate_offside(
        attackers=[atk("A", 80.0)],
        defenders=[dfn("D1", 90.0), dfn("D2", 78.0)],
        ball=BallPosition(60.0, 34.0),
        direction=AttackingDirection.POSITIVE_X,
        pitch_length=PITCH_LEN,
    )
    assert result.reference_kind is ReferenceKind.SECOND_LAST_OPPONENT
    assert result.offside_line_x == pytest.approx(78.0)
    assert _classification(result, "A").status is OffsideStatus.OFFSIDE_POSITION


# --- 6. Both attacking directions -------------------------------------------

def test_negative_x_direction() -> None:
    # Attacking toward X=0: goalward = smaller X. Opponents' half is X<52.5.
    # Defenders at X=25 (last) and X=30 (second-last). Ball at X=40.
    # Attacker at X=20 is more goalward (smaller) than line(30) -> offside.
    result = evaluate_offside(
        attackers=[atk("A", 20.0), atk("B", 35.0)],
        defenders=[dfn("D1", 25.0), dfn("D2", 30.0)],
        ball=BallPosition(40.0, 34.0),
        direction=AttackingDirection.NEGATIVE_X,
        pitch_length=PITCH_LEN,
    )
    assert result.offside_line_x == pytest.approx(30.0)
    assert result.reference_kind is ReferenceKind.SECOND_LAST_OPPONENT
    assert _classification(result, "A").status is OffsideStatus.OFFSIDE_POSITION
    assert _classification(result, "B").status is OffsideStatus.ONSIDE  # behind line


# --- 7. Own-half exemption --------------------------------------------------

def test_own_half_exemption_positive_x() -> None:
    # Attacker at X=40 is in own half (<52.5). Even though defenders/ball are
    # far back, the attacker cannot be offside in their own half.
    result = evaluate_offside(
        attackers=[atk("A", 40.0)],
        defenders=[dfn("D1", 30.0), dfn("D2", 20.0)],
        ball=BallPosition(10.0, 34.0),
        direction=AttackingDirection.POSITIVE_X,
        pitch_length=PITCH_LEN,
    )
    c = _classification(result, "A")
    assert c.in_own_half is True
    assert c.status is OffsideStatus.ONSIDE


def test_own_half_exemption_negative_x() -> None:
    # Attacking toward X=0; own half is X>52.5. Attacker at X=60 is in own half.
    result = evaluate_offside(
        attackers=[atk("A", 60.0)],
        defenders=[dfn("D1", 70.0), dfn("D2", 80.0)],
        ball=BallPosition(90.0, 34.0),
        direction=AttackingDirection.NEGATIVE_X,
        pitch_length=PITCH_LEN,
    )
    c = _classification(result, "A")
    assert c.in_own_half is True
    assert c.status is OffsideStatus.ONSIDE


# --- 8. Player on the halfway line ------------------------------------------

def test_halfway_line_is_own_half_both_directions() -> None:
    for direction in (AttackingDirection.POSITIVE_X, AttackingDirection.NEGATIVE_X):
        result = evaluate_offside(
            attackers=[atk("A", 52.5)],
            defenders=[dfn("D1", 80.0), dfn("D2", 75.0)]
            if direction is AttackingDirection.POSITIVE_X
            else [dfn("D1", 25.0), dfn("D2", 30.0)],
            ball=BallPosition(
                60.0 if direction is AttackingDirection.POSITIVE_X else 40.0, 34.0
            ),
            direction=direction,
            pitch_length=PITCH_LEN,
        )
        c = _classification(result, "A")
        assert c.in_own_half is True, direction
        assert c.status is OffsideStatus.ONSIDE, direction


# --- 9. Fewer than two defenders --------------------------------------------

def test_fewer_than_two_defenders_insufficient_data() -> None:
    result = evaluate_offside(
        attackers=[atk("A", 90.0)],
        defenders=[dfn("D1", 80.0)],
        ball=BallPosition(60.0, 34.0),
        direction=AttackingDirection.POSITIVE_X,
        pitch_length=PITCH_LEN,
    )
    assert result.status is OffsideStatus.INSUFFICIENT_DATA
    assert result.attackers == ()


def test_second_last_opponent_requires_two() -> None:
    with pytest.raises(OffsideError):
        second_last_opponent([dfn("D1", 80.0)], AttackingDirection.POSITIVE_X)


# --- 10. Missing ball -------------------------------------------------------

def test_missing_ball_insufficient_data() -> None:
    result = evaluate_offside(
        attackers=[atk("A", 90.0)],
        defenders=[dfn("D1", 80.0), dfn("D2", 75.0)],
        ball=None,
        direction=AttackingDirection.POSITIVE_X,
        pitch_length=PITCH_LEN,
    )
    assert result.status is OffsideStatus.INSUFFICIENT_DATA
    assert "ball" in result.explanation.lower()
    assert result.offside_line_x is None


# --- 11. Multiple attackers with different classifications ------------------

def test_multiple_attackers_mixed() -> None:
    result = evaluate_offside(
        attackers=[atk("A", 90.0), atk("B", 70.0), atk("C", 76.0)],
        defenders=[dfn("D1", 80.0), dfn("D2", 75.0)],
        ball=BallPosition(60.0, 34.0),
        direction=AttackingDirection.POSITIVE_X,
        pitch_length=PITCH_LEN,
    )
    # Line = 75 (second-last opponent). A(90) & C(76) offside; B(70) onside.
    assert _classification(result, "A").status is OffsideStatus.OFFSIDE_POSITION
    assert _classification(result, "B").status is OffsideStatus.ONSIDE
    assert _classification(result, "C").status is OffsideStatus.OFFSIDE_POSITION
    assert len(result.offside_players) == 2


# --- 12. Goalkeeper not treated specially -----------------------------------

def test_goalkeeper_not_special_cased() -> None:
    # Keeper is NOT the most goalward here: an outfield defender (X=95) is ahead
    # of the keeper (X=90). Second-last opponent must be the keeper at X=90 by
    # pure position, not by any keeper rule.
    defenders = [dfn("OUTFIELD", 95.0), dfn("KEEPER", 90.0), dfn("D3", 70.0)]
    slo = second_last_opponent(defenders, AttackingDirection.POSITIVE_X)
    assert slo.player_id == "KEEPER"  # second-most-goalward, purely positional
    result = evaluate_offside(
        attackers=[atk("A", 92.0)],
        defenders=defenders,
        ball=BallPosition(60.0, 34.0),
        direction=AttackingDirection.POSITIVE_X,
        pitch_length=PITCH_LEN,
    )
    assert result.offside_line_x == pytest.approx(90.0)
    assert _classification(result, "A").status is OffsideStatus.OFFSIDE_POSITION


# --- 13. Numerical tolerance around the line --------------------------------

def test_tolerance_band_keeps_near_level_onside() -> None:
    # Attacker 2 cm beyond the line with a 5 cm tolerance -> treated as level/onside.
    result = evaluate_offside(
        attackers=[atk("A", 75.02)],
        defenders=[dfn("D1", 80.0), dfn("D2", 75.0)],
        ball=BallPosition(60.0, 34.0),
        direction=AttackingDirection.POSITIVE_X,
        pitch_length=PITCH_LEN,
        tolerance_m=0.05,
    )
    assert _classification(result, "A").status is OffsideStatus.ONSIDE


def test_tolerance_zero_makes_tiny_margin_offside() -> None:
    # Same geometry, zero tolerance -> the 2 cm beyond now counts as offside.
    result = evaluate_offside(
        attackers=[atk("A", 75.02)],
        defenders=[dfn("D1", 80.0), dfn("D2", 75.0)],
        ball=BallPosition(60.0, 34.0),
        direction=AttackingDirection.POSITIVE_X,
        pitch_length=PITCH_LEN,
        tolerance_m=0.0,
    )
    assert _classification(result, "A").status is OffsideStatus.OFFSIDE_POSITION


# --- 14. Invalid / non-finite coordinates -----------------------------------

def test_non_finite_player_rejected() -> None:
    with pytest.raises(OffsideError):
        PlayerOnPitch("A", math.inf, 34.0, TeamSide.ATTACKING)


def test_non_finite_ball_rejected() -> None:
    with pytest.raises(OffsideError):
        BallPosition(math.nan, 34.0)


# --- 15. Missing / invalid direction and inputs -----------------------------

def test_invalid_direction_rejected() -> None:
    with pytest.raises(OffsideError):
        evaluate_offside(
            attackers=[atk("A", 90.0)],
            defenders=[dfn("D1", 80.0), dfn("D2", 75.0)],
            ball=BallPosition(60.0, 34.0),
            direction="forward",  # type: ignore[arg-type]
            pitch_length=PITCH_LEN,
        )


def test_invalid_pitch_length_rejected() -> None:
    with pytest.raises(OffsideError):
        evaluate_offside(
            attackers=[atk("A", 90.0)],
            defenders=[dfn("D1", 80.0), dfn("D2", 75.0)],
            ball=BallPosition(60.0, 34.0),
            direction=AttackingDirection.POSITIVE_X,
            pitch_length=0.0,
        )


def test_negative_tolerance_rejected() -> None:
    with pytest.raises(OffsideError):
        evaluate_offside(
            attackers=[atk("A", 90.0)],
            defenders=[dfn("D1", 80.0), dfn("D2", 75.0)],
            ball=BallPosition(60.0, 34.0),
            direction=AttackingDirection.POSITIVE_X,
            pitch_length=PITCH_LEN,
            tolerance_m=-1.0,
        )


# --- bonus: tie between ball and second-last opponent -----------------------

def test_tie_reference_when_ball_and_defender_coincide() -> None:
    result = evaluate_offside(
        attackers=[atk("A", 80.0)],
        defenders=[dfn("D1", 90.0), dfn("D2", 75.0)],
        ball=BallPosition(75.0, 34.0),
        direction=AttackingDirection.POSITIVE_X,
        pitch_length=PITCH_LEN,
    )
    assert result.reference_kind is ReferenceKind.TIE
    assert result.offside_line_x == pytest.approx(75.0)
