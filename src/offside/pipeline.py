"""End-to-end single-frame offside analysis (integration layer).

Flow: video -> selected frame -> detection -> pitch projection -> team lookup ->
offside evaluation -> structured result (ready for annotation / UI).

SCOPE (V1): this analyses ONE selected "pass-moment" frame. Player ids and team
assignments apply to that frame only and are NOT persistent identities. There is
no player tracking and no multi-frame / whole-video verdict here.

This module is the single integration boundary where the detector (M2, torch)
meets the pure geometry / offside / team modules. It does not put any computer-
vision logic into the M4 rule module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from offside.calibration import Calibration
from offside.detection import Detection, Detector, split_by_class
from offside.geometry import image_to_pitch
from offside.offside import (
    AttackingDirection,
    BallPosition,
    OffsideResult,
    OffsideStatus,
    PlayerOnPitch,
    TeamSide,
    evaluate_offside,
)
from offside.team_cluster import TeamAssignment
from offside.video_io import read_frame

Frame = "object"  # numpy array at runtime; avoid importing numpy types here


class PipelineError(ValueError):
    """Raised for invalid pipeline inputs (frame, assignment, calibration)."""


class BallSource(Enum):
    """Where the ball position came from."""

    MANUAL = "manual"        # user-supplied image coordinate (authoritative)
    DETECTOR = "detector"    # accepted detector candidate (per reliability policy)
    UNAVAILABLE = "unavailable"  # no reliable ball -> INSUFFICIENT_DATA


@dataclass(frozen=True)
class DetectedPlayer:
    """A player detected in the selected frame, with a frame-local id.

    The ``player_id`` is deterministic within this frame only (see module scope);
    never reuse it across frames or runs.
    """

    player_id: str
    detection: Detection
    foot_image_xy: tuple[float, float]
    pitch_xy: tuple[float, float]


@dataclass(frozen=True)
class BallInfo:
    """Resolved ball position (or absence) with provenance."""

    source: BallSource
    image_xy: tuple[float, float] | None = None
    pitch_xy: tuple[float, float] | None = None
    detector_confidence: float | None = None
    note: str = ""


@dataclass(frozen=True)
class FrameAnalysis:
    """Full structured result of a single-frame analysis (annotation-ready)."""

    frame_index: int
    attacking_direction: AttackingDirection
    players: tuple[DetectedPlayer, ...]
    ball: BallInfo
    offside: OffsideResult
    calibration: Calibration
    team_assignment: TeamAssignment | None = field(default=None)

    @property
    def status(self) -> OffsideStatus:
        return self.offside.status


# Default policy: accept a detector ball only if its confidence is at least this
# AND it is the single ball candidate in the frame. Deliberately conservative
# because the COCO "sports ball" class is unreliable in football footage.
DEFAULT_BALL_MIN_CONF = 0.5


def assign_player_ids(players: list[Detection]) -> list[str]:
    """Assign deterministic frame-local ids by position.

    Sorted by foot-point (x, then y) so the same frame always yields the same
    ids regardless of detector output ordering. Ids look like ``p0, p1, ...``.
    These are NOT persistent identities.
    """
    order = sorted(
        range(len(players)),
        key=lambda i: (players[i].box.foot_point[0], players[i].box.foot_point[1]),
    )
    ids = [""] * len(players)
    for rank, original_index in enumerate(order):
        ids[original_index] = f"p{rank}"
    return ids


def _resolve_ball(
    ball_detections: list[Detection],
    calibration: Calibration,
    manual_ball_image_xy: tuple[float, float] | None,
    ball_min_conf: float,
) -> BallInfo:
    """Apply the ball-input policy (manual > reliable detector > unavailable)."""
    # 1) Manual input is authoritative.
    if manual_ball_image_xy is not None:
        pitch = tuple(float(v) for v in image_to_pitch(calibration, list(manual_ball_image_xy)))
        return BallInfo(
            source=BallSource.MANUAL,
            image_xy=(float(manual_ball_image_xy[0]), float(manual_ball_image_xy[1])),
            pitch_xy=(pitch[0], pitch[1]),
            note="Ball position supplied manually (authoritative).",
        )

    # 2) Detector candidate, only under an explicit reliability policy.
    #    High confidence alone is NOT treated as guaranteed correct; we require a
    #    single candidate above threshold and still flag the inherent uncertainty.
    confident = [d for d in ball_detections if d.confidence >= ball_min_conf]
    if len(confident) == 1:
        d = confident[0]
        fx, fy = d.box.center  # ball: use box centre as its position
        pitch = tuple(float(v) for v in image_to_pitch(calibration, [fx, fy]))
        return BallInfo(
            source=BallSource.DETECTOR,
            image_xy=(float(fx), float(fy)),
            pitch_xy=(pitch[0], pitch[1]),
            detector_confidence=float(d.confidence),
            note=(
                "Ball taken from a single detector candidate; detection of the ball "
                "in football footage is unreliable and not guaranteed correct."
            ),
        )

    # 3) Nothing reliable.
    reason = (
        "no ball candidates above threshold"
        if not confident
        else f"{len(confident)} competing ball candidates"
    )
    return BallInfo(
        source=BallSource.UNAVAILABLE,
        note=f"No reliable ball position ({reason}); supply a manual ball coordinate.",
    )


def analyze_frame(
    *,
    video_path: Path | str,
    frame_index: int,
    detector: Detector,
    calibration: Calibration,
    attacking_direction: AttackingDirection,
    team_assignment: TeamAssignment,
    manual_ball_image_xy: tuple[float, float] | None = None,
    max_width: int | None = None,
    tolerance_m: float = 0.05,
    ball_min_conf: float = DEFAULT_BALL_MIN_CONF,
) -> FrameAnalysis:
    """Run the full single-frame offside analysis.

    Args:
        video_path: path to the clip.
        frame_index: the selected pass-moment frame index.
        detector: anything implementing the M2 ``Detector`` protocol.
        calibration: an M3 calibration valid for THIS camera view.
        attacking_direction: which way the attacking team is going.
        team_assignment: maps the frame-local player ids (``p0..``) to sides. Must
            cover every detected player.
        manual_ball_image_xy: optional authoritative ball image coordinate.
        max_width: optional resize applied to the frame before detection. Must
            match the calibration's image scale — see note below.
        tolerance_m: offside "level" tolerance in metres (passed to M4).
        ball_min_conf: detector ball acceptance threshold (reliability policy).

    Returns:
        A :class:`FrameAnalysis`. The embedded ``offside`` carries the M4 status
        (OFFSIDE_POSITION / ONSIDE / UNCERTAIN / INSUFFICIENT_DATA).

    Raises:
        PipelineError: for an invalid frame index, or a team assignment that does
            not match the detected players.
        VideoReadError / GeometryError: propagated from the underlying layers.

    Note on scale: the calibration must correspond to the same image scale as the
    frame passed to the detector. If you resize via ``max_width``, calibrate on
    the resized frame (or omit ``max_width``). This is documented, not auto-fixed.
    """
    frame = read_frame(video_path, frame_index, max_width=max_width)

    detections = detector.detect(frame)
    player_dets, ball_dets = split_by_class(detections)

    ids = assign_player_ids(player_dets)

    # Validate the supplied team assignment against THIS frame's players.
    detected_ids = set(ids)
    assigned_ids = set(team_assignment.assignments.keys())
    missing = detected_ids - assigned_ids
    unknown = assigned_ids - detected_ids
    if missing or unknown:
        raise PipelineError(
            "Team assignment does not match detected players for this frame. "
            f"Unassigned detected players: {sorted(missing)}; "
            f"assignments for non-detected ids: {sorted(unknown)}. "
            "Assignments are frame-local; re-assign for this frame."
        )

    # Project each player's FOOT point (not box centre) to pitch coordinates.
    players: list[DetectedPlayer] = []
    attackers: list[PlayerOnPitch] = []
    defenders: list[PlayerOnPitch] = []
    for pid, det in zip(ids, player_dets):
        foot = det.box.foot_point
        pitch = tuple(float(v) for v in image_to_pitch(calibration, list(foot)))
        side = team_assignment.team_of(pid)
        pop = PlayerOnPitch(player_id=pid, x=pitch[0], y=pitch[1], team=side)
        players.append(
            DetectedPlayer(
                player_id=pid,
                detection=det,
                foot_image_xy=(float(foot[0]), float(foot[1])),
                pitch_xy=(pitch[0], pitch[1]),
            )
        )
        (attackers if side is TeamSide.ATTACKING else defenders).append(pop)

    ball_info = _resolve_ball(ball_dets, calibration, manual_ball_image_xy, ball_min_conf)
    ball_position = (
        None
        if ball_info.pitch_xy is None
        else BallPosition(ball_info.pitch_xy[0], ball_info.pitch_xy[1], source=ball_info.source.value)
    )

    result = evaluate_offside(
        attackers=attackers,
        defenders=defenders,
        ball=ball_position,
        direction=attacking_direction,
        pitch_length=calibration.pitch.length,
        tolerance_m=tolerance_m,
    )

    return FrameAnalysis(
        frame_index=frame_index,
        attacking_direction=attacking_direction,
        players=tuple(players),
        ball=ball_info,
        offside=result,
        calibration=calibration,
        team_assignment=team_assignment,
    )
