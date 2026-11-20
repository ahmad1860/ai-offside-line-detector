"""Offside-position geometry and decision logic.

This is the football reasoning core. It works entirely in **pitch coordinates**
(metres) — players and the ball must already be projected onto the pitch plane
(upstream, via :mod:`offside.geometry`). The module is therefore independent of
Ultralytics, PyTorch, OpenCV, Streamlit, and any video handling.

What it computes (see PROJECT_CONTEXT.md "Critical football / offside rules"):

* The **offside reference line** is the more goalward of the **second-last
  opponent** and the **ball**, along the attacking axis (pitch X).
* An attacker is in an **offside position** when they are strictly more goalward
  than that line AND in the opponents' half (the **own-half exemption**). Being
  level with the line is **onside**.
* The **ball is mandatory**: without a valid ball position the result is
  ``INSUFFICIENT_DATA`` — we never produce a confident verdict from defenders
  alone (DECISIONS.md D8).

What it deliberately does NOT do:

* It judges offside **position only**, never an offside **offence** (active
  involvement in play is a separate judgement this system does not make, D12).
* The numerical ``tolerance_m`` absorbs tiny floating-point noise at the line.
  It does **not** compensate for camera-calibration error and does not make the
  system VAR-accurate.

Coordinate convention: pitch X = length (0..length), pitch Y = width (0..width);
the halfway line is at ``length / 2``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

# Default numerical tolerance (metres) for deciding "level with the line".
DEFAULT_TOLERANCE_M = 0.05

# Minimum defenders required to identify a second-last opponent.
MIN_DEFENDERS = 2


class OffsideError(ValueError):
    """Raised for invalid offside-evaluation inputs."""


class AttackingDirection(Enum):
    """Which way the attacking team is going, expressed on the pitch-X axis.

    POSITIVE_X: larger X is more goalward (attacking toward X = pitch length).
    NEGATIVE_X: smaller X is more goalward (attacking toward X = 0).
    """

    POSITIVE_X = "positive_x"
    NEGATIVE_X = "negative_x"

    @property
    def sign(self) -> float:
        """+1 for POSITIVE_X, -1 for NEGATIVE_X.

        Multiplying a pitch-X value by this sign yields a "goalward score": a
        larger score is always more goalward, regardless of direction. This lets
        one code path serve both attacking directions.
        """
        return 1.0 if self is AttackingDirection.POSITIVE_X else -1.0


class TeamSide(Enum):
    """Which team a player belongs to, relative to the phase of play."""

    ATTACKING = "attacking"
    DEFENDING = "defending"


class OffsideStatus(Enum):
    """Overall outcome of an evaluation."""

    OFFSIDE_POSITION = "offside_position"
    ONSIDE = "onside"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_DATA = "insufficient_data"


class ReferenceKind(Enum):
    """Which reference defined the offside line."""

    SECOND_LAST_OPPONENT = "second_last_opponent"
    BALL = "ball"
    TIE = "tie"  # ball and second-last opponent coincide (within tolerance)


def _check_finite_xy(x: float, y: float, what: str) -> None:
    if not (math.isfinite(x) and math.isfinite(y)):
        raise OffsideError(f"{what} has non-finite coordinates: ({x}, {y}).")


@dataclass(frozen=True)
class PlayerOnPitch:
    """A player's position on the pitch plane (metres) plus their team side."""

    player_id: str
    x: float
    y: float
    team: TeamSide

    def __post_init__(self) -> None:
        _check_finite_xy(self.x, self.y, f"Player {self.player_id!r}")


@dataclass(frozen=True)
class BallPosition:
    """The ball's position on the pitch plane (metres).

    ``source`` is free-form provenance metadata (e.g. "detection", "manual")
    recorded for transparency; it does not affect the geometry.
    """

    x: float
    y: float
    source: str = "unspecified"

    def __post_init__(self) -> None:
        _check_finite_xy(self.x, self.y, "Ball")


@dataclass(frozen=True)
class AttackerClassification:
    """Per-attacker outcome."""

    player_id: str
    status: OffsideStatus  # OFFSIDE_POSITION or ONSIDE
    x: float
    y: float
    # Signed goalward margin (metres) relative to the offside line:
    #   > 0  => beyond the line (toward goal); < 0 => behind it; ~0 => level.
    margin_m: float
    in_own_half: bool


@dataclass(frozen=True)
class OffsideResult:
    """Full result of an offside-position evaluation."""

    status: OffsideStatus
    explanation: str
    attacking_direction: AttackingDirection | None = None
    offside_line_x: float | None = None
    reference_kind: ReferenceKind | None = None
    second_last_opponent: PlayerOnPitch | None = None
    ball: BallPosition | None = None
    attackers: tuple[AttackerClassification, ...] = field(default_factory=tuple)

    @property
    def offside_players(self) -> tuple[AttackerClassification, ...]:
        return tuple(
            a for a in self.attackers if a.status is OffsideStatus.OFFSIDE_POSITION
        )


def _goalward_score(x: float, direction: AttackingDirection) -> float:
    """Project a pitch-X onto a monotone 'goalward' axis (larger = nearer goal)."""
    return direction.sign * x


def second_last_opponent(
    defenders: list[PlayerOnPitch], direction: AttackingDirection
) -> PlayerOnPitch:
    """Return the second-most-goalward defender (the "second-last opponent").

    Ordering is purely positional along the attacking axis. The goalkeeper is
    NOT treated specially: whoever is the second-most-goalward defender is the
    reference, whether or not that happens to be the keeper.

    Raises:
        OffsideError: if fewer than two defenders are supplied.
    """
    if len(defenders) < MIN_DEFENDERS:
        raise OffsideError(
            f"Need at least {MIN_DEFENDERS} defenders to find the second-last "
            f"opponent, got {len(defenders)}."
        )
    # Sort by goalward score descending: index 0 = last man, index 1 = second-last.
    ordered = sorted(
        defenders, key=lambda p: _goalward_score(p.x, direction), reverse=True
    )
    return ordered[1]


def _halfway_line_x(pitch_length: float) -> float:
    return pitch_length / 2.0


def _in_own_half(
    player_x: float, direction: AttackingDirection, pitch_length: float
) -> bool:
    """True if the attacker is in (or on the line of) their own half.

    The attacking team's own half is the half AWAY from the goal they attack.
    - POSITIVE_X (attacking toward X = length): own half is small X, i.e.
      ``x <= length/2``.
    - NEGATIVE_X (attacking toward X = 0): own half is large X, i.e.
      ``x >= length/2``.

    A player exactly on the halfway line is treated as being in their own half
    for the exemption, so they cannot be offside (consistent with the law that
    a player level with the halfway line is not in an offside position).
    """
    half = _halfway_line_x(pitch_length)
    if direction is AttackingDirection.POSITIVE_X:
        return player_x <= half
    return player_x >= half


def evaluate_offside(
    attackers: list[PlayerOnPitch],
    defenders: list[PlayerOnPitch],
    ball: BallPosition | None,
    direction: AttackingDirection,
    pitch_length: float,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
) -> OffsideResult:
    """Classify each attacker as onside / offside-position at a chosen moment.

    Args:
        attackers: attacking-team players, already projected to pitch metres.
        defenders: defending-team players, already projected to pitch metres.
        ball: the ball position (metres). **Mandatory** — if ``None`` the result
            is ``INSUFFICIENT_DATA``.
        direction: the attacking direction along pitch X.
        pitch_length: pitch length in metres (defines the halfway line).
        tolerance_m: distance (metres) within which an attacker is treated as
            level with the line (hence onside). Absorbs numerical noise only.

    Returns:
        An :class:`OffsideResult`. Status is ``INSUFFICIENT_DATA`` when the ball
        is missing or fewer than two defenders are available; otherwise
        ``OFFSIDE_POSITION`` if any attacker is in an offside position, else
        ``ONSIDE``.

    Raises:
        OffsideError: for invalid inputs (bad direction, non-positive pitch
            length, negative tolerance, or non-finite coordinates in inputs).
    """
    if not isinstance(direction, AttackingDirection):
        raise OffsideError(f"direction must be an AttackingDirection, got {direction!r}.")
    if not (math.isfinite(pitch_length) and pitch_length > 0):
        raise OffsideError(f"pitch_length must be a positive number, got {pitch_length!r}.")
    if not (math.isfinite(tolerance_m) and tolerance_m >= 0):
        raise OffsideError(f"tolerance_m must be a non-negative number, got {tolerance_m!r}.")

    # Mandatory ball (D8): never guess a verdict without it.
    if ball is None:
        return OffsideResult(
            status=OffsideStatus.INSUFFICIENT_DATA,
            explanation=(
                "No ball position supplied. Offside depends on the ball at the moment "
                "of the pass, so no verdict is produced (INSUFFICIENT_DATA)."
            ),
            attacking_direction=direction,
        )

    # Need at least two defenders to form the reference.
    if len(defenders) < MIN_DEFENDERS:
        return OffsideResult(
            status=OffsideStatus.INSUFFICIENT_DATA,
            explanation=(
                f"Only {len(defenders)} defender(s) available; at least {MIN_DEFENDERS} "
                "are required to determine the second-last opponent (INSUFFICIENT_DATA)."
            ),
            attacking_direction=direction,
            ball=ball,
        )

    slo = second_last_opponent(defenders, direction)

    # Offside line = more goalward of {second-last opponent, ball}.
    slo_score = _goalward_score(slo.x, direction)
    ball_score = _goalward_score(ball.x, direction)
    if abs(slo_score - ball_score) <= tolerance_m:
        reference_kind = ReferenceKind.TIE
        line_x = slo.x  # coincident within tolerance; either works
    elif ball_score > slo_score:
        reference_kind = ReferenceKind.BALL
        line_x = ball.x
    else:
        reference_kind = ReferenceKind.SECOND_LAST_OPPONENT
        line_x = slo.x

    line_score = _goalward_score(line_x, direction)

    classifications: list[AttackerClassification] = []
    any_offside = False
    for atk in attackers:
        atk_score = _goalward_score(atk.x, direction)
        margin = atk_score - line_score  # >0 means beyond the line toward goal
        in_own_half = _in_own_half(atk.x, direction, pitch_length)

        if in_own_half:
            status = OffsideStatus.ONSIDE  # own-half exemption
        elif margin > tolerance_m:
            status = OffsideStatus.OFFSIDE_POSITION
        else:
            status = OffsideStatus.ONSIDE  # behind the line, or level within tolerance

        if status is OffsideStatus.OFFSIDE_POSITION:
            any_offside = True

        classifications.append(
            AttackerClassification(
                player_id=atk.player_id,
                status=status,
                x=atk.x,
                y=atk.y,
                margin_m=margin,
                in_own_half=in_own_half,
            )
        )

    overall = OffsideStatus.OFFSIDE_POSITION if any_offside else OffsideStatus.ONSIDE
    ref_label = {
        ReferenceKind.SECOND_LAST_OPPONENT: "the second-last opponent",
        ReferenceKind.BALL: "the ball",
        ReferenceKind.TIE: "the second-last opponent and the ball (coincident)",
    }[reference_kind]
    n_off = sum(1 for c in classifications if c.status is OffsideStatus.OFFSIDE_POSITION)
    explanation = (
        f"Offside line set by {ref_label} at pitch X = {line_x:.2f} m. "
        f"{n_off} of {len(classifications)} attacker(s) in an offside position. "
        "This indicates offside POSITION only, not an offside offence."
    )

    return OffsideResult(
        status=overall,
        explanation=explanation,
        attacking_direction=direction,
        offside_line_x=line_x,
        reference_kind=reference_kind,
        second_last_opponent=slo,
        ball=ball,
        attackers=tuple(classifications),
    )
