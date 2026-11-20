"""Team classification: assign detected players to the two teams.

Produces assignments consumable directly by :mod:`offside.offside` (each player
mapped to a :class:`~offside.offside.TeamSide`). Two methods, kept logically
separate:

* **Manual assignment (primary, reliable).** The caller supplies player-id ->
  team mappings (batch and/or individual overrides). Deterministic and always
  available. This is the path M6 wires first.
* **Jersey-colour clustering (optional, best-effort).** Samples a torso-region
  colour per player and k-means-clusters players into two provisional groups
  with a separation/confidence measure. It does NOT decide which cluster is
  attacking vs defending — the caller maps clusters -> sides explicitly — and it
  does not reliably handle referees, goalkeepers, similar kits, lighting, or
  occlusion.

Dependencies: NumPy + OpenCV only (no Ultralytics / PyTorch / Streamlit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Protocol, Sequence

import cv2
import numpy as np

from offside.offside import TeamSide

Frame = np.ndarray

# Below this separation ratio, a clustering result is flagged unreliable.
DEFAULT_MIN_SEPARATION = 1.5

# Torso crop fractions of the bounding box (avoid head / shorts / background).
_TORSO_TOP = 0.15
_TORSO_BOTTOM = 0.55
_TORSO_INSET_X = 0.20


class TeamClusterError(ValueError):
    """Raised for invalid team-assignment or clustering inputs."""


class AssignmentMethod(Enum):
    """How a team assignment was produced."""

    MANUAL = "manual"
    CLUSTER = "cluster"


class BoxLike(Protocol):
    """Structural type for a pixel bounding box (duck-typed, no hard import)."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class TeamAssignment:
    """Result mapping player ids to teams.

    Attributes:
        assignments: player_id -> TeamSide.
        method: how the assignment was produced.
        reliable: whether the assignment should be trusted downstream. Manual
            assignments are always reliable; cluster-derived assignments are only
            reliable when separation exceeded the configured threshold.
    """

    assignments: Mapping[str, TeamSide]
    method: AssignmentMethod
    reliable: bool = True

    def team_of(self, player_id: str) -> TeamSide:
        if player_id not in self.assignments:
            raise TeamClusterError(f"No team assignment for player {player_id!r}.")
        return self.assignments[player_id]

    def players_in(self, team: TeamSide) -> tuple[str, ...]:
        return tuple(pid for pid, side in self.assignments.items() if side is team)


@dataclass(frozen=True)
class ClusterResult:
    """Provisional two-way colour clustering of players.

    Attributes:
        labels: player_id -> cluster index (0 or 1). NOT a team side.
        separation: ratio of between-cluster distance to mean within-cluster
            spread in Lab space. Larger is better-separated; small means the two
            kits look alike (unreliable).
        reliable: True iff ``separation >= min_separation``.
        cluster_colors: per-cluster median Lab colour (shape (2, 3)).
    """

    labels: Mapping[str, int]
    separation: float
    reliable: bool
    cluster_colors: np.ndarray = field(repr=False)

    def to_team_assignment(
        self, cluster_to_side: Mapping[int, TeamSide]
    ) -> TeamAssignment:
        """Map the two clusters onto team sides (caller decides which is which).

        Args:
            cluster_to_side: a mapping covering both cluster indices {0, 1} to
                ``TeamSide`` values. The two clusters must map to two different
                sides.

        Raises:
            TeamClusterError: if the mapping is incomplete or maps both clusters
                to the same side.
        """
        if set(cluster_to_side.keys()) != {0, 1}:
            raise TeamClusterError(
                "cluster_to_side must map exactly clusters {0, 1} to team sides."
            )
        if len(set(cluster_to_side.values())) != 2:
            raise TeamClusterError(
                "The two clusters must map to two different team sides."
            )
        assignments = {pid: cluster_to_side[c] for pid, c in self.labels.items()}
        # Propagate reliability: a low-confidence cluster must not become a
        # silently-trusted team assignment.
        return TeamAssignment(
            assignments=assignments,
            method=AssignmentMethod.CLUSTER,
            reliable=self.reliable,
        )


# --------------------------------------------------------------------------- #
# Manual assignment (primary)
# --------------------------------------------------------------------------- #

def assign_manual(
    known_player_ids: Sequence[str],
    mapping: Mapping[str, TeamSide],
    *,
    overrides: Mapping[str, TeamSide] | None = None,
    require_all: bool = True,
) -> TeamAssignment:
    """Build a manual team assignment from a batch mapping (+ optional overrides).

    Args:
        known_player_ids: the players the caller detected / cares about.
        mapping: player_id -> TeamSide. May assign many players at once.
        overrides: optional per-player corrections applied on top of ``mapping``.
        require_all: if True, every id in ``known_player_ids`` must end up
            assigned; otherwise a partial assignment is allowed.

    Returns:
        A reliable :class:`TeamAssignment` (method = MANUAL).

    Raises:
        TeamClusterError: for duplicate known ids, unknown ids in mapping/overrides,
            or (when ``require_all``) players left unassigned.
    """
    known = list(known_player_ids)
    known_set = set(known)
    if len(known_set) != len(known):
        dupes = sorted({pid for pid in known if known.count(pid) > 1})
        raise TeamClusterError(f"Duplicate player ids in known_player_ids: {dupes}.")

    combined: dict[str, TeamSide] = {}
    for src_name, src in (("mapping", mapping), ("overrides", overrides or {})):
        for pid, side in src.items():
            if pid not in known_set:
                raise TeamClusterError(
                    f"{src_name} references unknown player id {pid!r}."
                )
            if not isinstance(side, TeamSide):
                raise TeamClusterError(
                    f"{src_name}[{pid!r}] must be a TeamSide, got {side!r}."
                )
            combined[pid] = side  # overrides naturally win (applied second)

    if require_all:
        missing = sorted(known_set - combined.keys())
        if missing:
            raise TeamClusterError(f"Players left unassigned: {missing}.")

    return TeamAssignment(assignments=combined, method=AssignmentMethod.MANUAL, reliable=True)


# --------------------------------------------------------------------------- #
# Jersey-colour clustering (optional, best-effort)
# --------------------------------------------------------------------------- #

def _torso_crop(frame: Frame, box: BoxLike) -> np.ndarray:
    """Return the torso sub-image of a player's bounding box.

    Takes an upper-middle band (skipping head and shorts/legs) inset from the
    sides to reduce grass/background contamination.

    Raises:
        TeamClusterError: if the box is invalid or outside the frame.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = float(box.x1), float(box.y1), float(box.x2), float(box.y2)
    if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
        raise TeamClusterError("Bounding box has non-finite coordinates.")
    if x2 <= x1 or y2 <= y1:
        raise TeamClusterError(f"Degenerate bounding box: ({x1}, {y1}, {x2}, {y2}).")

    bw, bh = x2 - x1, y2 - y1
    cx1 = x1 + _TORSO_INSET_X * bw
    cx2 = x2 - _TORSO_INSET_X * bw
    cy1 = y1 + _TORSO_TOP * bh
    cy2 = y1 + _TORSO_BOTTOM * bh

    # Clamp to frame bounds and integerize.
    ix1, iy1 = int(np.floor(max(0.0, cx1))), int(np.floor(max(0.0, cy1)))
    ix2, iy2 = int(np.ceil(min(float(w), cx2))), int(np.ceil(min(float(h), cy2)))
    if ix2 <= ix1 or iy2 <= iy1:
        raise TeamClusterError("Torso crop is empty (box outside frame or too small).")

    crop = frame[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        raise TeamClusterError("Torso crop is empty after clamping to frame.")
    return crop


def _median_lab_color(crop_bgr: np.ndarray) -> np.ndarray:
    """Median CIELab colour of a BGR crop (robust to grass/background outliers)."""
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float64)
    return np.median(pixels, axis=0)


def cluster_by_jersey(
    frame: Frame,
    boxes: Mapping[str, BoxLike],
    *,
    min_separation: float = DEFAULT_MIN_SEPARATION,
    seed: int = 0,
) -> ClusterResult:
    """Cluster players into two provisional groups by torso colour.

    Deterministic: fixed k-means seed, fixed attempts, and a stable tie-break.

    Args:
        frame: the BGR frame the boxes refer to.
        boxes: player_id -> bounding box. At least two players required.
        min_separation: threshold below which the result is flagged unreliable.
        seed: RNG seed for reproducibility.

    Returns:
        A :class:`ClusterResult` with per-player cluster labels (0/1), a
        separation measure, a reliability flag, and the cluster median colours.

    Raises:
        TeamClusterError: for fewer than two players, or invalid crops.
    """
    if frame is None or getattr(frame, "ndim", 0) != 3 or frame.shape[2] != 3:
        raise TeamClusterError("frame must be a BGR image of shape (H, W, 3).")
    if len(boxes) < 2:
        raise TeamClusterError("Need at least two players to cluster into two teams.")

    player_ids = list(boxes.keys())
    colors = np.vstack([_median_lab_color(_torso_crop(frame, boxes[pid])) for pid in player_ids])
    colors32 = colors.astype(np.float32)

    # Deterministic k-means (k=2). Fixed criteria and attempts.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    cv2.setRNGSeed(seed)
    _compactness, labels, centers = cv2.kmeans(
        colors32, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.ravel().astype(int)

    # Stable label orientation: cluster 0 is the one with the smaller mean L*
    # (darker), so results don't flip between runs/platforms.
    if centers[0, 0] > centers[1, 0]:
        labels = 1 - labels
        centers = centers[::-1]

    # Separation = between-centre distance / mean within-cluster spread.
    between = float(np.linalg.norm(centers[0] - centers[1]))
    within = []
    for c in (0, 1):
        members = colors[labels == c]
        if len(members) == 0:
            # All points collapsed into one cluster -> no separation.
            return ClusterResult(
                labels={pid: 0 for pid in player_ids},
                separation=0.0,
                reliable=False,
                cluster_colors=centers.astype(np.float64),
            )
        within.append(float(np.mean(np.linalg.norm(members - centers[c], axis=1))))
    mean_within = (within[0] + within[1]) / 2.0
    separation = between / mean_within if mean_within > 1e-6 else (between / 1e-6 if between > 0 else 0.0)
    # Cap the degenerate "infinite" separation (identical within-cluster points)
    # to a large finite number so downstream code stays numeric.
    separation = float(min(separation, 1e6))

    reliable = separation >= min_separation
    return ClusterResult(
        labels={pid: int(labels[i]) for i, pid in enumerate(player_ids)},
        separation=separation,
        reliable=reliable,
        cluster_colors=centers.astype(np.float64),
    )
