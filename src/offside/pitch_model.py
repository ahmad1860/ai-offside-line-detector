"""Pitch model: landmark catalogue and pitch markings in pitch coordinates.

Pure NumPy; no UI, CV or model dependencies.

Coordinate convention (the single source of truth, shared with calibration and
the offside logic):

    * Pitch X runs along the LENGTH, from the goal line at X = 0 to the goal line
      at X = length.
    * Pitch Y runs across the WIDTH, from the touchline at Y = 0 to the touchline
      at Y = width.
    * Units are metres. "X=0 end" / "X=L end" are the two goal lines and
      "Y=0 side" / "Y=W side" are the two touchlines. These are PITCH
      coordinates, not image left/right: which physical end you call X=0 is your
      choice, but it must be used consistently for every landmark and must match
      the attacking direction.

Marking sizes come from the Laws of the Game and are FIXED lengths that do not
scale with the pitch (a penalty area is 16.5 m deep on any full-size pitch).
Landmark coordinates are therefore derived from the *current* pitch dimensions
plus those fixed offsets, never from fractions of a 105 x 68 reference pitch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from offside.calibration import PitchDimensions

# --- Laws-of-the-Game marking dimensions (metres) --------------------------
PENALTY_AREA_DEPTH_M = 16.5
PENALTY_AREA_WIDTH_M = 40.32      # 16.5 + 7.32 (goal) + 16.5
GOAL_AREA_DEPTH_M = 5.5
GOAL_AREA_WIDTH_M = 18.32         # 5.5 + 7.32 (goal) + 5.5
PENALTY_MARK_DISTANCE_M = 11.0
CENTRE_CIRCLE_RADIUS_M = 9.15
# Where the penalty arc (radius 9.15 m about the penalty mark) meets the
# penalty-area front line (16.5 m from the goal line): half-chord along Y.
PENALTY_ARC_HALF_CHORD_M = math.sqrt(
    CENTRE_CIRCLE_RADIUS_M**2 - (PENALTY_AREA_DEPTH_M - PENALTY_MARK_DISTANCE_M) ** 2
)

# Custom points may sit a little outside the touchlines/goal lines (e.g. a
# marking just outside the pitch) but not arbitrarily far.
CUSTOM_POINT_MARGIN_M = 10.0


class PitchModelError(ValueError):
    """Raised for invalid landmark lookups or custom pitch coordinates."""


def _ref(ref: str, total: float) -> float:
    if ref == "start":
        return 0.0
    if ref == "mid":
        return total / 2.0
    if ref == "end":
        return total
    raise PitchModelError(f"Unknown reference {ref!r}.")


@dataclass(frozen=True)
class PitchLandmark:
    """A named, geometrically defined pitch point.

    The position is ``reference + offset`` on each axis, where the reference is
    the start (0), middle, or end (length / width) of that axis and the offset is
    a signed length in metres along the pitch axis. This keeps fixed-size
    markings correct when the pitch dimensions change.
    """

    key: str
    label: str
    group: str
    description: str
    x_ref: str
    x_off: float
    y_ref: str
    y_off: float

    def pitch_xy(self, pitch: PitchDimensions) -> tuple[float, float]:
        return (
            _ref(self.x_ref, pitch.length) + self.x_off,
            _ref(self.y_ref, pitch.width) + self.y_off,
        )

    def is_valid_for(self, pitch: PitchDimensions) -> bool:
        """True if the point lies on the pitch and is geometrically sensible.

        Small pitches may not contain a full-size penalty area; such landmarks
        are simply not offered.
        """
        x, y = self.pitch_xy(pitch)
        tol = 1e-9
        on_pitch = -tol <= x <= pitch.length + tol and -tol <= y <= pitch.width + tol
        # A marking measured from a goal line must stay inside its own half.
        in_own_half = self.x_ref == "mid" or abs(self.x_off) < pitch.length / 2.0
        return on_pitch and in_own_half


def _build_catalogue() -> tuple[PitchLandmark, ...]:
    items: list[PitchLandmark] = []

    def add(key, label, group, description, x_ref, x_off, y_ref, y_off):
        items.append(PitchLandmark(key, label, group, description, x_ref, x_off, y_ref, y_off))

    g = "Pitch corners"
    add("corner_x0_y0", "Corner (X=0 end, Y=0 side)", g,
        "Where the X=0 goal line meets the Y=0 touchline.", "start", 0.0, "start", 0.0)
    add("corner_x0_yW", "Corner (X=0 end, Y=W side)", g,
        "Where the X=0 goal line meets the Y=W touchline.", "start", 0.0, "end", 0.0)
    add("corner_xL_y0", "Corner (X=L end, Y=0 side)", g,
        "Where the X=L goal line meets the Y=0 touchline.", "end", 0.0, "start", 0.0)
    add("corner_xL_yW", "Corner (X=L end, Y=W side)", g,
        "Where the X=L goal line meets the Y=W touchline.", "end", 0.0, "end", 0.0)

    g = "Halfway line and centre circle"
    add("halfway_y0", "Halfway line ∩ touchline (Y=0 side)", g,
        "Where the halfway line meets the Y=0 touchline.", "mid", 0.0, "start", 0.0)
    add("halfway_yW", "Halfway line ∩ touchline (Y=W side)", g,
        "Where the halfway line meets the Y=W touchline.", "mid", 0.0, "end", 0.0)
    add("centre_mark", "Centre mark", g,
        "The centre spot, midway along both axes.", "mid", 0.0, "mid", 0.0)
    r = CENTRE_CIRCLE_RADIUS_M
    add("centre_circle_y0", "Centre circle ∩ halfway line (Y=0 side)", g,
        f"Where the centre circle (radius {r} m) crosses the halfway line, on the Y=0 side "
        "of the centre mark.", "mid", 0.0, "mid", -r)
    add("centre_circle_yW", "Centre circle ∩ halfway line (Y=W side)", g,
        f"Where the centre circle (radius {r} m) crosses the halfway line, on the Y=W side "
        "of the centre mark.", "mid", 0.0, "mid", +r)

    for end_label, tag, xref, s in (("X=0 end", "x0", "start", +1.0), ("X=L end", "xL", "end", -1.0)):
        for side_label, ytag, ys in (("Y=0 side", "y0", -1.0), ("Y=W side", "yW", +1.0)):
            pa = f"{end_label}: penalty area"
            add(f"{tag}_pa_goalline_{ytag}",
                f"Penalty area ∩ goal line ({end_label}, {side_label})", pa,
                f"Where the side edge of the penalty area meets the {end_label} goal line "
                f"({PENALTY_AREA_WIDTH_M / 2:.2f} m from the goal centre).",
                xref, 0.0, "mid", ys * PENALTY_AREA_WIDTH_M / 2.0)
            add(f"{tag}_pa_front_{ytag}",
                f"Penalty area front corner ({end_label}, {side_label})", pa,
                f"The outer corner of the penalty area, {PENALTY_AREA_DEPTH_M} m from the goal "
                "line.", xref, s * PENALTY_AREA_DEPTH_M, "mid", ys * PENALTY_AREA_WIDTH_M / 2.0)
            add(f"{tag}_pa_arc_{ytag}",
                f"Penalty arc ∩ penalty-area front line ({end_label}, {side_label})", pa,
                "Where the arc of the 'D' meets the front line of the penalty area.",
                xref, s * PENALTY_AREA_DEPTH_M, "mid", ys * PENALTY_ARC_HALF_CHORD_M)
            ga = f"{end_label}: goal area"
            add(f"{tag}_ga_goalline_{ytag}",
                f"Goal area ∩ goal line ({end_label}, {side_label})", ga,
                f"Where the side edge of the goal area meets the {end_label} goal line "
                f"({GOAL_AREA_WIDTH_M / 2:.2f} m from the goal centre).",
                xref, 0.0, "mid", ys * GOAL_AREA_WIDTH_M / 2.0)
            add(f"{tag}_ga_front_{ytag}",
                f"Goal area front corner ({end_label}, {side_label})", ga,
                f"The outer corner of the goal area, {GOAL_AREA_DEPTH_M} m from the goal line.",
                xref, s * GOAL_AREA_DEPTH_M, "mid", ys * GOAL_AREA_WIDTH_M / 2.0)
        add(f"{tag}_penalty_mark", f"Penalty mark ({end_label})", f"{end_label}: penalty area",
            f"The penalty spot, {PENALTY_MARK_DISTANCE_M} m from the {end_label} goal line, "
            "centred across the pitch.", xref, s * PENALTY_MARK_DISTANCE_M, "mid", 0.0)
    return tuple(items)


PITCH_LANDMARKS: tuple[PitchLandmark, ...] = _build_catalogue()

# Keys used by earlier versions (ambiguous near/far/left/right wording) map to
# the coordinate-based keys above so older sessions/tests keep working.
LEGACY_KEY_ALIASES: dict[str, str] = {
    "corner_00": "corner_x0_y0",
    "corner_01": "corner_x0_yW",
    "corner_10": "corner_xL_y0",
    "corner_11": "corner_xL_yW",
    "half_near": "halfway_y0",
    "half_far": "halfway_yW",
    "half_center": "centre_mark",
    "pbox_left_near": "x0_pa_front_y0",
    "pbox_left_far": "x0_pa_front_yW",
    "pbox_right_near": "xL_pa_front_y0",
    "pbox_right_far": "xL_pa_front_yW",
}

_BY_KEY: dict[str, PitchLandmark] = {lm.key: lm for lm in PITCH_LANDMARKS}


def canonical_key(key: str) -> str:
    """Resolve a legacy key to its coordinate-based key (identity otherwise)."""
    return LEGACY_KEY_ALIASES.get(key, key)


def get_landmark(key: str) -> PitchLandmark:
    """Look up a catalogue landmark by key (legacy keys accepted)."""
    resolved = canonical_key(key)
    if resolved not in _BY_KEY:
        raise PitchModelError(f"Unknown landmark key: {key!r}.")
    return _BY_KEY[resolved]


def available_landmarks(pitch: PitchDimensions) -> tuple[PitchLandmark, ...]:
    """Catalogue landmarks that exist on a pitch of the given dimensions."""
    return tuple(lm for lm in PITCH_LANDMARKS if lm.is_valid_for(pitch))


def validate_custom_pitch_xy(x: float, y: float, pitch: PitchDimensions) -> tuple[float, float]:
    """Validate a user-entered custom pitch point (metres, same convention).

    Raises:
        PitchModelError: if non-finite or implausibly far from the pitch.
    """
    try:
        fx, fy = float(x), float(y)
    except (TypeError, ValueError) as exc:
        raise PitchModelError("Custom pitch coordinates must be numbers.") from exc
    if not (math.isfinite(fx) and math.isfinite(fy)):
        raise PitchModelError("Custom pitch coordinates must be finite numbers.")
    m = CUSTOM_POINT_MARGIN_M
    if not (-m <= fx <= pitch.length + m and -m <= fy <= pitch.width + m):
        raise PitchModelError(
            f"Custom point ({fx:g}, {fy:g}) m is more than {m:g} m outside the "
            f"{pitch.length:g} x {pitch.width:g} m pitch. Use X in 0..{pitch.length:g} "
            f"(length) and Y in 0..{pitch.width:g} (width)."
        )
    return (fx, fy)


def _rect_open(x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    """Three sides of a box whose fourth side is the goal line (at x0)."""
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def pitch_markings(pitch: PitchDimensions, circle_points: int = 72) -> dict[str, np.ndarray]:
    """Polylines (N, 2) in pitch metres describing the main pitch markings.

    Used to draw the calibration-check overlay and the direction diagram. Areas
    that do not fit on a small pitch are omitted.
    """
    length, width = pitch.length, pitch.width
    cx, cy = length / 2.0, width / 2.0
    lines: dict[str, np.ndarray] = {
        "outline": np.array(
            [[0, 0], [length, 0], [length, width], [0, width], [0, 0]], dtype=np.float64
        ),
        "halfway_line": np.array([[cx, 0.0], [cx, width]], dtype=np.float64),
    }
    ang = np.linspace(0.0, 2.0 * math.pi, circle_points + 1)
    lines["centre_circle"] = np.stack(
        [cx + CENTRE_CIRCLE_RADIUS_M * np.cos(ang), cy + CENTRE_CIRCLE_RADIUS_M * np.sin(ang)],
        axis=1,
    )
    for tag, goal_x, s in (("x0", 0.0, +1.0), ("xL", length, -1.0)):
        for name, depth, half_w in (
            ("penalty_area", PENALTY_AREA_DEPTH_M, PENALTY_AREA_WIDTH_M / 2.0),
            ("goal_area", GOAL_AREA_DEPTH_M, GOAL_AREA_WIDTH_M / 2.0),
        ):
            if depth < length / 2.0 and cy - half_w >= 0.0 and cy + half_w <= width:
                lines[f"{tag}_{name}"] = _rect_open(goal_x, goal_x + s * depth, cy - half_w, cy + half_w)
    return lines
