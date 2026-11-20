"""Tests for team classification (manual assignment + jersey clustering).

Synthetic image data only — no real footage, model weights, internet, or GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from offside.offside import TeamSide
from offside.team_cluster import (
    AssignmentMethod,
    ClusterResult,
    TeamAssignment,
    TeamClusterError,
    assign_manual,
    cluster_by_jersey,
)


@dataclass
class _Box:
    """Minimal BoxLike for tests."""

    x1: float
    y1: float
    x2: float
    y2: float


# --------------------------------------------------------------------------- #
# Manual assignment
# --------------------------------------------------------------------------- #

def test_manual_batch_assignment() -> None:
    known = ["p1", "p2", "p3", "p4"]
    mapping = {
        "p1": TeamSide.ATTACKING,
        "p2": TeamSide.ATTACKING,
        "p3": TeamSide.DEFENDING,
        "p4": TeamSide.DEFENDING,
    }
    result = assign_manual(known, mapping)
    assert isinstance(result, TeamAssignment)
    assert result.method is AssignmentMethod.MANUAL
    assert result.reliable is True
    assert set(result.players_in(TeamSide.ATTACKING)) == {"p1", "p2"}
    assert set(result.players_in(TeamSide.DEFENDING)) == {"p3", "p4"}


def test_manual_individual_override() -> None:
    known = ["p1", "p2"]
    mapping = {"p1": TeamSide.ATTACKING, "p2": TeamSide.ATTACKING}
    result = assign_manual(known, mapping, overrides={"p2": TeamSide.DEFENDING})
    assert result.team_of("p1") is TeamSide.ATTACKING
    assert result.team_of("p2") is TeamSide.DEFENDING  # override wins


def test_manual_unknown_player_rejected() -> None:
    with pytest.raises(TeamClusterError):
        assign_manual(["p1"], {"ghost": TeamSide.ATTACKING})


def test_manual_unknown_override_rejected() -> None:
    with pytest.raises(TeamClusterError):
        assign_manual(["p1"], {"p1": TeamSide.ATTACKING}, overrides={"x": TeamSide.DEFENDING})


def test_manual_missing_assignment_rejected_when_require_all() -> None:
    with pytest.raises(TeamClusterError):
        assign_manual(["p1", "p2"], {"p1": TeamSide.ATTACKING})


def test_manual_partial_allowed_when_not_require_all() -> None:
    result = assign_manual(["p1", "p2"], {"p1": TeamSide.ATTACKING}, require_all=False)
    assert result.team_of("p1") is TeamSide.ATTACKING
    with pytest.raises(TeamClusterError):
        result.team_of("p2")  # genuinely unassigned


def test_manual_duplicate_known_ids_rejected() -> None:
    with pytest.raises(TeamClusterError):
        assign_manual(["p1", "p1"], {"p1": TeamSide.ATTACKING})


def test_manual_empty_input() -> None:
    result = assign_manual([], {})
    assert result.assignments == {}
    assert result.reliable is True


# --------------------------------------------------------------------------- #
# Jersey clustering
# --------------------------------------------------------------------------- #

def _frame_with_players(colors_bgr, box_h=80, box_w=40, gap=10):
    """Build a green-pitch frame with a row of solid-colour player rectangles.

    Returns (frame, boxes) where boxes maps player id -> _Box.
    """
    n = len(colors_bgr)
    width = n * (box_w + gap) + gap
    height = box_h + 40
    frame = np.full((height, width, 3), (40, 120, 40), dtype=np.uint8)  # grass-ish
    boxes = {}
    for i, color in enumerate(colors_bgr):
        x1 = gap + i * (box_w + gap)
        y1 = 20
        x2 = x1 + box_w
        y2 = y1 + box_h
        frame[y1:y2, x1:x2] = color
        boxes[f"p{i}"] = _Box(x1, y1, x2, y2)
    return frame, boxes


def test_cluster_two_clearly_separated_colours() -> None:
    # Two red players, two blue players (BGR).
    red, blue = (0, 0, 200), (200, 0, 0)
    frame, boxes = _frame_with_players([red, red, blue, blue])
    result = cluster_by_jersey(frame, boxes)
    assert isinstance(result, ClusterResult)
    assert result.reliable is True
    # The two reds share a cluster, the two blues share the other.
    labels = result.labels
    assert labels["p0"] == labels["p1"]
    assert labels["p2"] == labels["p3"]
    assert labels["p0"] != labels["p2"]


def test_cluster_is_deterministic() -> None:
    red, blue = (0, 0, 200), (200, 0, 0)
    frame, boxes = _frame_with_players([red, blue, red, blue])
    r1 = cluster_by_jersey(frame, boxes, seed=0)
    r2 = cluster_by_jersey(frame, boxes, seed=0)
    assert r1.labels == r2.labels
    assert r1.separation == pytest.approx(r2.separation)


def test_cluster_single_colour_is_unreliable() -> None:
    same = (0, 0, 200)
    frame, boxes = _frame_with_players([same, same, same, same])
    result = cluster_by_jersey(frame, boxes)
    assert result.reliable is False
    assert result.separation < 1.5


def test_cluster_requires_two_players() -> None:
    frame, boxes = _frame_with_players([(0, 0, 200)])
    with pytest.raises(TeamClusterError):
        cluster_by_jersey(frame, boxes)


def test_cluster_invalid_box_rejected() -> None:
    frame, boxes = _frame_with_players([(0, 0, 200), (200, 0, 0)])
    boxes["p0"] = _Box(50, 50, 10, 10)  # x2<x1, y2<y1 -> degenerate
    with pytest.raises(TeamClusterError):
        cluster_by_jersey(frame, boxes)


def test_cluster_out_of_frame_box_rejected() -> None:
    frame, boxes = _frame_with_players([(0, 0, 200), (200, 0, 0)])
    boxes["p0"] = _Box(10_000, 10_000, 10_100, 10_100)  # fully outside frame
    with pytest.raises(TeamClusterError):
        cluster_by_jersey(frame, boxes)


def test_cluster_non_finite_box_rejected() -> None:
    frame, boxes = _frame_with_players([(0, 0, 200), (200, 0, 0)])
    boxes["p0"] = _Box(float("nan"), 0, 40, 80)
    with pytest.raises(TeamClusterError):
        cluster_by_jersey(frame, boxes)


# --------------------------------------------------------------------------- #
# Cluster -> team-side mapping
# --------------------------------------------------------------------------- #

def test_explicit_cluster_to_side_mapping() -> None:
    red, blue = (0, 0, 200), (200, 0, 0)
    frame, boxes = _frame_with_players([red, red, blue, blue])
    result = cluster_by_jersey(frame, boxes)
    assignment = result.to_team_assignment(
        {0: TeamSide.ATTACKING, 1: TeamSide.DEFENDING}
    )
    assert assignment.method is AssignmentMethod.CLUSTER
    # Each cluster maps to a distinct side; all four players assigned.
    sides = {assignment.team_of(pid) for pid in ["p0", "p1", "p2", "p3"]}
    assert sides == {TeamSide.ATTACKING, TeamSide.DEFENDING}


def test_cluster_to_side_requires_both_clusters() -> None:
    red, blue = (0, 0, 200), (200, 0, 0)
    frame, boxes = _frame_with_players([red, blue])
    result = cluster_by_jersey(frame, boxes)
    with pytest.raises(TeamClusterError):
        result.to_team_assignment({0: TeamSide.ATTACKING})  # missing cluster 1


def test_cluster_to_side_rejects_same_side_for_both() -> None:
    red, blue = (0, 0, 200), (200, 0, 0)
    frame, boxes = _frame_with_players([red, blue])
    result = cluster_by_jersey(frame, boxes)
    with pytest.raises(TeamClusterError):
        result.to_team_assignment({0: TeamSide.ATTACKING, 1: TeamSide.ATTACKING})


def test_low_confidence_cluster_propagates_unreliable_to_assignment() -> None:
    same = (0, 0, 200)
    frame, boxes = _frame_with_players([same, same])
    result = cluster_by_jersey(frame, boxes)
    assert result.reliable is False
    assignment = result.to_team_assignment(
        {0: TeamSide.ATTACKING, 1: TeamSide.DEFENDING}
    )
    # A low-confidence cluster must NOT silently become a trusted assignment.
    assert assignment.reliable is False
