"""Non-UI helper logic for the Streamlit app.

All the glue the Streamlit page needs is placed here so it can be unit-tested
without running Streamlit: landmark selection (catalogue + custom points),
correspondence bookkeeping, calibration building/validation/staleness checks,
team-assignment and ball preparation, pitch-dimension validation, the
attacking-direction explanation/diagram, and the stale-result "signature".

This module imports only the project's own modules plus NumPy/OpenCV. It does
NOT import streamlit, so importing it in tests is cheap and safe.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import cv2
import numpy as np

from offside.calibration import (
    MIN_PITCH_SEPARATION_M,
    Calibration,
    PitchDimensions,
    PointSetAssessment,
    assess_point_set,
    calibrate,
    make_correspondences,
)
from offside.offside import AttackingDirection, TeamSide
from offside.pitch_model import (
    PITCH_LANDMARKS,  # noqa: F401  (re-exported for the UI and existing tests)
    PitchLandmark,
    PitchModelError,
    canonical_key,
    get_landmark,
    pitch_markings,
    validate_custom_pitch_xy,
)
from offside.team_cluster import TeamAssignment, assign_manual

# Supported upload types (kept in one place for the UI uploader).
SUPPORTED_VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv")

# Pitch dimension limits accepted by the UI (metres).
MIN_PITCH_LENGTH_M, MAX_PITCH_LENGTH_M = 50.0, 130.0
MIN_PITCH_WIDTH_M, MAX_PITCH_WIDTH_M = 40.0, 90.0

FIT_VS_ACCURACY_NOTE = (
    "Reprojection error shows how well your clicked landmarks agree with each other "
    "(mathematical fit). It does NOT prove the calibration is accurate in the real "
    "world: a consistent but mislabelled or poorly spread set of landmarks can fit "
    "perfectly and still be wrong. Use the overlay to check the projected pitch lines "
    "against the real markings."
)

COORDINATE_HELP = (
    "Pitch coordinates (metres): X runs along the LENGTH from the goal line at X = 0 "
    "to the goal line at X = L; Y runs across the WIDTH from the touchline at Y = 0 to "
    "the touchline at Y = W. 'X=0 end' / 'X=L end' and 'Y=0 side' / 'Y=W side' are "
    "pitch coordinates, NOT image left/right. You choose which physical end is X = 0, "
    "but keep it consistent for every landmark and make the attacking direction match."
)


class UIValidationError(ValueError):
    """Raised for user-input problems the UI should surface as friendly errors."""


class DuplicateLandmarkError(UIValidationError):
    """Raised when a landmark (or its pitch location) is already placed."""


def landmark_by_key(key: str) -> PitchLandmark:
    """Look up a catalogue landmark (legacy keys accepted)."""
    try:
        return get_landmark(key)
    except PitchModelError as exc:
        raise UIValidationError(str(exc)) from exc


# --------------------------------------------------------------------------- #
# Pitch dimensions
# --------------------------------------------------------------------------- #

def validate_pitch_dimensions(length: float, width: float) -> PitchDimensions:
    """Validate user-entered pitch dimensions (metres) and return them.

    Raises:
        UIValidationError: if non-finite, outside the supported range, or if the
            length (X axis) does not exceed the width (Y axis).
    """
    try:
        ln, wd = float(length), float(width)
    except (TypeError, ValueError) as exc:
        raise UIValidationError("Pitch length and width must be numbers (metres).") from exc
    if not (math.isfinite(ln) and math.isfinite(wd)):
        raise UIValidationError("Pitch length and width must be finite numbers (metres).")
    if not (MIN_PITCH_LENGTH_M <= ln <= MAX_PITCH_LENGTH_M):
        raise UIValidationError(
            f"Pitch length must be between {MIN_PITCH_LENGTH_M:g} and "
            f"{MAX_PITCH_LENGTH_M:g} m, got {ln:g} m."
        )
    if not (MIN_PITCH_WIDTH_M <= wd <= MAX_PITCH_WIDTH_M):
        raise UIValidationError(
            f"Pitch width must be between {MIN_PITCH_WIDTH_M:g} and "
            f"{MAX_PITCH_WIDTH_M:g} m, got {wd:g} m."
        )
    if ln <= wd:
        raise UIValidationError(
            f"Pitch length (X axis, {ln:g} m) must be greater than its width "
            f"(Y axis, {wd:g} m)."
        )
    return PitchDimensions(length=ln, width=wd)


# --------------------------------------------------------------------------- #
# Correspondences (landmark <-> clicked pixel)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Correspondence:
    """A user-made pairing of a pitch landmark with a clicked image pixel.

    For a catalogue landmark the pitch coordinates are derived from the CURRENT
    pitch dimensions at use time. For a custom point (``custom_pitch_xy`` set)
    the user-entered metres are used as given.

    ``frame_index`` records which video frame the click was made on, so a
    calibration is never silently reused for a different camera view.
    """

    landmark_key: str
    image_x: float
    image_y: float
    frame_index: int | None = None
    custom_pitch_xy: tuple[float, float] | None = None
    custom_label: str | None = None

    @property
    def is_custom(self) -> bool:
        return self.custom_pitch_xy is not None

    @property
    def canonical(self) -> str:
        return self.landmark_key if self.is_custom else canonical_key(self.landmark_key)

    def display_label(self) -> str:
        if self.is_custom:
            name = self.custom_label or self.landmark_key
            x, y = self.custom_pitch_xy  # type: ignore[misc]
            return f"{name} (custom X={x:g}, Y={y:g} m)"
        return landmark_by_key(self.landmark_key).label

    def pitch_xy(self, pitch: PitchDimensions) -> tuple[float, float]:
        if self.custom_pitch_xy is not None:
            return self.custom_pitch_xy
        return landmark_by_key(self.landmark_key).pitch_xy(pitch)

    def is_valid_for(self, pitch: PitchDimensions) -> bool:
        """Whether this landmark exists on a pitch of the given size."""
        if self.custom_pitch_xy is not None:
            try:
                validate_custom_pitch_xy(*self.custom_pitch_xy, pitch)
            except PitchModelError:
                return False
            return True
        return landmark_by_key(self.landmark_key).is_valid_for(pitch)


def validate_image_point(
    x: float, y: float, image_size: tuple[int, int] | None = None
) -> tuple[float, float]:
    """Check a clicked/entered image coordinate (finite and, if known, inside the frame)."""
    try:
        fx, fy = float(x), float(y)
    except (TypeError, ValueError) as exc:
        raise UIValidationError("Image coordinates must be numbers.") from exc
    if not (math.isfinite(fx) and math.isfinite(fy)):
        raise UIValidationError("Image coordinates must be finite numbers.")
    if image_size is not None:
        w, h = image_size
        if not (0.0 <= fx <= w and 0.0 <= fy <= h):
            raise UIValidationError(
                f"Image point ({fx:.0f}, {fy:.0f}) is outside the {w}x{h} frame."
            )
    return fx, fy


def next_custom_key(existing: Sequence[Correspondence]) -> str:
    """Return the next unused ``custom_N`` key."""
    used = {c.landmark_key for c in existing}
    n = 1
    while f"custom_{n}" in used:
        n += 1
    return f"custom_{n}"


def make_custom_correspondence(
    existing: Sequence[Correspondence],
    *,
    pitch_x: float,
    pitch_y: float,
    image_x: float,
    image_y: float,
    pitch: PitchDimensions,
    frame_index: int | None = None,
    label: str | None = None,
    image_size: tuple[int, int] | None = None,
) -> Correspondence:
    """Build a custom correspondence from real-world metres and a clicked pixel.

    The pitch point uses the same convention as every other landmark (X along the
    length, Y across the width, metres).

    Raises:
        UIValidationError: for non-finite/out-of-range coordinates.
    """
    try:
        px, py = validate_custom_pitch_xy(pitch_x, pitch_y, pitch)
    except PitchModelError as exc:
        raise UIValidationError(str(exc)) from exc
    ix, iy = validate_image_point(image_x, image_y, image_size)
    name = (label or "").strip() or None
    return Correspondence(
        landmark_key=next_custom_key(existing),
        image_x=ix,
        image_y=iy,
        frame_index=frame_index,
        custom_pitch_xy=(px, py),
        custom_label=name,
    )


def add_correspondence(
    existing: Sequence[Correspondence],
    new: Correspondence,
    pitch: PitchDimensions,
    *,
    replace: bool = False,
) -> list[Correspondence]:
    """Return a new list with ``new`` added (or replacing the same landmark).

    A landmark that is already placed is never silently overwritten: pass
    ``replace=True`` to replace it (keeping its position in the list).

    Raises:
        DuplicateLandmarkError: if the landmark is already placed and ``replace``
            is False, or if its pitch location coincides with another placed point.
        UIValidationError: if the landmark does not exist on this pitch.
    """
    if not new.is_valid_for(pitch):
        raise UIValidationError(
            f"{new.display_label()} does not exist on a {pitch.length:g} x {pitch.width:g} m pitch."
        )
    key = new.canonical
    index = next((i for i, c in enumerate(existing) if c.canonical == key), None)
    if index is not None and not replace:
        raise DuplicateLandmarkError(
            f"{new.display_label()} is already placed. Remove it first, or replace it."
        )
    nx, ny = new.pitch_xy(pitch)
    for i, c in enumerate(existing):
        if i == index or not c.is_valid_for(pitch):
            continue
        ox, oy = c.pitch_xy(pitch)
        if math.hypot(nx - ox, ny - oy) < MIN_PITCH_SEPARATION_M:
            raise DuplicateLandmarkError(
                f"{new.display_label()} is at the same pitch location as {c.display_label()}."
            )
    result = list(existing)
    if index is None:
        result.append(new)
    else:
        result[index] = new
    return result


def remove_correspondence(existing: Sequence[Correspondence], key: str) -> list[Correspondence]:
    """Return a new list without the landmark ``key`` (legacy keys accepted)."""
    target = canonical_key(key)
    return [c for c in existing if c.canonical != target]


def correspondences_to_arrays(
    correspondences: Sequence[Correspondence], pitch: PitchDimensions
) -> tuple[list[list[float]], list[list[float]]]:
    """Convert UI correspondences into paired (image_points, pitch_points) lists.

    Index i of both lists refers to the same physical landmark.

    Raises:
        UIValidationError: for fewer than 4 points, an unknown landmark, a
            landmark that does not exist on this pitch, or a duplicate landmark.
    """
    if len(correspondences) < 4:
        raise UIValidationError(
            f"Need at least 4 correspondences, have {len(correspondences)}."
        )
    seen: set[str] = set()
    image_points: list[list[float]] = []
    pitch_points: list[list[float]] = []
    for c in correspondences:
        if c.canonical in seen:
            raise DuplicateLandmarkError(
                f"Landmark {c.landmark_key!r} used more than once; each landmark "
                "must be clicked at a single image location."
            )
        seen.add(c.canonical)
        if not c.is_valid_for(pitch):
            raise UIValidationError(
                f"{c.display_label()} does not exist on a "
                f"{pitch.length:g} x {pitch.width:g} m pitch."
            )
        image_points.append([float(c.image_x), float(c.image_y)])
        px, py = c.pitch_xy(pitch)
        pitch_points.append([px, py])
    return image_points, pitch_points


@dataclass(frozen=True)
class CorrespondenceSummary:
    """Live feedback on the landmarks chosen so far."""

    placed: int
    valid: int
    invalid_labels: tuple[str, ...]
    assessment: PointSetAssessment

    @property
    def can_build(self) -> bool:
        return self.placed == self.valid and self.assessment.ok


def assess_correspondences(
    correspondences: Sequence[Correspondence],
    pitch: PitchDimensions,
    image_size: tuple[int, int] | None = None,
) -> CorrespondenceSummary:
    """Count valid landmarks and check their geometry (nothing is estimated).

    Reports how many placed landmarks are valid for the current pitch, and whether
    they are enough and well spread. Used for live guidance before building.
    """
    invalid = tuple(c.display_label() for c in correspondences if not c.is_valid_for(pitch))
    usable = [c for c in correspondences if c.is_valid_for(pitch)]
    seen: set[str] = set()
    unique: list[Correspondence] = []
    dupes: list[str] = []
    for c in usable:
        if c.canonical in seen:
            dupes.append(c.display_label())
        else:
            seen.add(c.canonical)
            unique.append(c)
    assessment = assess_point_set(
        [[c.image_x, c.image_y] for c in unique],
        [list(c.pitch_xy(pitch)) for c in unique],
        pitch=pitch,
        image_size=image_size,
        labels=[c.display_label() for c in unique],
    )
    if dupes:
        assessment = PointSetAssessment(
            count=assessment.count,
            errors=assessment.errors + tuple(f"{d} is placed more than once." for d in dupes),
            warnings=assessment.warnings,
            conditioning=assessment.conditioning,
            pitch_coverage=assessment.pitch_coverage,
            image_coverage=assessment.image_coverage,
        )
    return CorrespondenceSummary(
        placed=len(correspondences),
        valid=len(unique),
        invalid_labels=invalid,
        assessment=assessment,
    )


def correspondence_frames(correspondences: Sequence[Correspondence]) -> tuple[int, ...]:
    """Distinct source frames of the placed correspondences (provenance)."""
    return tuple(sorted({c.frame_index for c in correspondences if c.frame_index is not None}))


def build_calibration(
    correspondences: Sequence[Correspondence],
    pitch: PitchDimensions,
    image_size: tuple[int, int] | None = None,
) -> Calibration:
    """Validate UI correspondences and build an M3 calibration.

    Raises:
        UIValidationError: wrapping any calibration failure (too few points,
            duplicates, collinear/degenerate, singular homography) with a readable
            message.
    """
    image_points, pitch_points = correspondences_to_arrays(correspondences, pitch)
    labels = [c.display_label() for c in correspondences]
    try:
        corr = make_correspondences(image_points, pitch_points, labels)
        return calibrate(corr, pitch=pitch, image_size=image_size)
    except Exception as exc:  # calibration raises CalibrationError subclasses
        raise UIValidationError(f"Calibration failed: {exc}") from exc


def calibration_point_table(calibration: Calibration) -> list[dict[str, object]]:
    """Per-landmark residuals for display (label, pixel error, whether it was used)."""
    errors = calibration.point_errors_px or ()
    rows: list[dict[str, object]] = []
    for i, err in enumerate(errors):
        label = calibration.labels[i] if calibration.labels else f"#{i + 1}"
        used = True if calibration.inliers is None else bool(calibration.inliers[i])
        rows.append({"landmark": label, "error_px": round(float(err), 2), "used": used})
    return rows


# --------------------------------------------------------------------------- #
# Calibration validity (staleness)
# --------------------------------------------------------------------------- #

def calibration_input_signature(
    correspondences: Sequence[Correspondence], pitch: PitchDimensions
) -> str:
    """Hash of everything a calibration is computed from.

    Includes the pitch dimensions, because catalogue landmark coordinates are
    derived from them: changing the pitch size changes the signature.
    """
    payload = {
        "pitch": [round(pitch.length, 4), round(pitch.width, 4)],
        "corr": sorted(
            [
                c.canonical,
                round(c.image_x, 3),
                round(c.image_y, 3),
                None if c.custom_pitch_xy is None else [round(v, 4) for v in c.custom_pitch_xy],
                c.frame_index,
            ]
            for c in correspondences
        ),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class CalibrationStatus(Enum):
    MISSING = "missing"
    OUTDATED = "outdated"
    WRONG_FRAME = "wrong_frame"
    READY = "ready"


@dataclass(frozen=True)
class CalibrationCheck:
    status: CalibrationStatus
    message: str

    @property
    def ready(self) -> bool:
        return self.status is CalibrationStatus.READY


def check_calibration(
    calibration: Calibration | None,
    built_signature: str | None,
    correspondences: Sequence[Correspondence],
    pitch: PitchDimensions,
    frame_index: int,
) -> CalibrationCheck:
    """Decide whether a stored calibration may be used for the selected frame.

    A calibration is usable only if it still matches the landmarks and pitch size
    it was built from, and the landmarks were clicked on the frame being analysed
    (the camera view must be the same; the app cannot verify a camera is static).
    """
    if calibration is None:
        return CalibrationCheck(
            CalibrationStatus.MISSING,
            "No calibration yet. Place at least 4 landmarks and press 'Build calibration'.",
        )
    if built_signature != calibration_input_signature(correspondences, pitch):
        return CalibrationCheck(
            CalibrationStatus.OUTDATED,
            "The landmarks or pitch dimensions changed since this calibration was built. "
            "Press 'Build calibration' again.",
        )
    other = [f for f in correspondence_frames(correspondences) if f != frame_index]
    if other:
        listed = ", ".join(str(f) for f in other)
        return CalibrationCheck(
            CalibrationStatus.WRONG_FRAME,
            f"This calibration was made on frame {listed}, but frame {frame_index} is "
            "selected. A calibration only applies to the camera view it was made on, and "
            "the app cannot verify the camera has not moved. Recalibrate on this frame.",
        )
    return CalibrationCheck(CalibrationStatus.READY, "Calibration matches the selected frame.")


# --------------------------------------------------------------------------- #
# Click handling
# --------------------------------------------------------------------------- #

def click_to_frame_xy(
    click: Mapping[str, object], frame_width: int, display_width: int
) -> tuple[float, float]:
    """Convert a click on the DISPLAYED image into true frame pixels.

    The click component reports coordinates in the displayed image's pixel space
    (and its displayed ``width``). The frame may have been shown scaled down to
    fit the page, so the click is scaled back up to the real frame resolution.
    """
    shown = float(click.get("width") or display_width)  # type: ignore[arg-type]
    scale = frame_width / shown if shown > 0 else 1.0
    return (float(click["x"]) * scale, float(click["y"]) * scale)  # type: ignore[arg-type]


def click_stamp(click: Mapping[str, object]) -> object:
    """Identity of a click event.

    The component keeps re-reporting its last click on every rerun; comparing
    stamps lets the app record each click once instead of resurrecting a click
    the user has already used.
    """
    return click.get("unix_time", (click.get("x"), click.get("y")))


# --------------------------------------------------------------------------- #
# Teams, ball, direction
# --------------------------------------------------------------------------- #

def prepare_team_assignment(
    player_ids: Sequence[str], sides: Mapping[str, TeamSide]
) -> TeamAssignment:
    """Build an M5 manual assignment, surfacing friendly errors.

    Raises:
        UIValidationError: if any detected player is unassigned, or an assignment
            references a non-detected id.
    """
    try:
        return assign_manual(list(player_ids), dict(sides), require_all=True)
    except Exception as exc:  # TeamClusterError
        raise UIValidationError(str(exc)) from exc


def prepare_ball_xy(
    value: tuple[float, float] | Sequence[float] | None,
) -> tuple[float, float] | None:
    """Normalise a manual ball coordinate, or None if not placed.

    Raises:
        UIValidationError: if a value is given but is not a finite (x, y) pair.
    """
    if value is None:
        return None
    seq = list(value)
    if len(seq) != 2:
        raise UIValidationError("Ball coordinate must be an (x, y) pair.")
    x, y = float(seq[0]), float(seq[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        raise UIValidationError("Ball coordinate must be finite.")
    return (x, y)


def parse_attacking_direction(label: str) -> AttackingDirection:
    """Map a UI label to an AttackingDirection."""
    key = label.strip().lower().replace(" ", "_")
    for d in AttackingDirection:
        if key == d.value:
            return d
    raise UIValidationError(f"Invalid attacking direction: {label!r}.")


def direction_target_x(direction: AttackingDirection, pitch: PitchDimensions) -> float:
    """Pitch X of the goal line the attackers are attacking.

    Derived from the same ``AttackingDirection`` the offside calculation uses, so
    the explanation and the verdict cannot disagree.
    """
    return pitch.length if direction.sign > 0 else 0.0


def direction_label(direction: AttackingDirection, pitch: PitchDimensions) -> str:
    """Short selectbox text for a direction."""
    if direction is AttackingDirection.POSITIVE_X:
        return f"Toward X = {pitch.length:g} m goal line (+X)"
    return "Toward X = 0 goal line (−X)"


def describe_attacking_direction(direction: AttackingDirection, pitch: PitchDimensions) -> str:
    """One-sentence explanation of what the selected direction means."""
    target = direction_target_x(direction, pitch)
    bigger = "larger" if direction.sign > 0 else "smaller"
    return (
        f"Attackers are attacking the goal line at X = {target:g} m. A player is more "
        f"goalward the {bigger} their pitch X is, so the defenders' last line and the "
        "offside line are measured along that axis."
    )


# RGB colours used by the direction diagram (the target goal line colour is
# unique in the image so tests can locate it).
DIAGRAM_TARGET_RGB = (255, 140, 0)
_DIAGRAM_PITCH_RGB = (46, 125, 50)
_DIAGRAM_LINE_RGB = (255, 255, 255)
_DIAGRAM_BG_RGB = (245, 245, 245)
_DIAGRAM_TEXT_RGB = (30, 30, 30)
_DIAGRAM_ARROW_RGB = (20, 20, 20)


def render_direction_diagram(
    pitch: PitchDimensions, direction: AttackingDirection, width_px: int = 480
) -> np.ndarray:
    """Draw a top-down pitch with the X axis, both ends and the attack direction.

    X runs left to right (0 at the left, L at the right) and Y runs bottom to top.
    The attacked goal line is highlighted and an arrow points toward it. Returns
    an RGB uint8 image ready for ``st.image``.
    """
    margin_x, margin_y = 44, 34
    pw = max(120, int(width_px) - 2 * margin_x)
    ph = max(60, int(round(pw * pitch.width / pitch.length)))
    canvas = np.full((ph + 2 * margin_y, pw + 2 * margin_x, 3), _DIAGRAM_BG_RGB, dtype=np.uint8)

    def to_px(x: float, y: float) -> tuple[int, int]:
        return (
            int(round(margin_x + x / pitch.length * pw)),
            int(round(margin_y + ph - y / pitch.width * ph)),  # Y up
        )

    cv2.rectangle(canvas, to_px(0, pitch.width), to_px(pitch.length, 0), _DIAGRAM_PITCH_RGB, -1)
    for name, line in pitch_markings(pitch).items():
        pts = np.array([to_px(x, y) for x, y in line], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, _DIAGRAM_LINE_RGB, 1, cv2.LINE_AA)

    target_x = direction_target_x(direction, pitch)
    cv2.line(canvas, to_px(target_x, 0), to_px(target_x, pitch.width), DIAGRAM_TARGET_RGB, 5, cv2.LINE_8)

    s = direction.sign
    mid_y = pitch.width / 2.0
    tail = to_px(pitch.length / 2.0 - s * pitch.length * 0.18, mid_y)
    head = to_px(pitch.length / 2.0 + s * pitch.length * 0.30, mid_y)
    cv2.arrowedLine(canvas, tail, head, _DIAGRAM_ARROW_RGB, 4, cv2.LINE_AA, 0, 0.3)
    label = "ATTACK"
    (tw, _th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.putText(canvas, label, (to_px(pitch.length / 2.0, mid_y)[0] - tw // 2, tail[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _DIAGRAM_LINE_RGB, 2, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom = canvas.shape[0] - 10
    cv2.putText(canvas, "X = 0", (margin_x - 4, bottom), font, 0.45, _DIAGRAM_TEXT_RGB, 1, cv2.LINE_AA)
    end_text = f"X = {pitch.length:g} m"
    (ew, _), _ = cv2.getTextSize(end_text, font, 0.45, 1)
    cv2.putText(canvas, end_text, (margin_x + pw - ew + 4, bottom), font, 0.45, _DIAGRAM_TEXT_RGB, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Y={pitch.width:g}", (2, margin_y + 10), font, 0.4, _DIAGRAM_TEXT_RGB, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Y=0", (2, margin_y + ph), font, 0.4, _DIAGRAM_TEXT_RGB, 1, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------- #
# Result signature and readiness
# --------------------------------------------------------------------------- #

def analysis_signature(
    *,
    video_name: str,
    frame_index: int,
    attacking_direction: AttackingDirection | None,
    correspondences: Sequence[Correspondence],
    team_sides: Mapping[str, TeamSide],
    ball_xy: tuple[float, float] | None,
    pitch: PitchDimensions | None = None,
) -> str:
    """Stable hash of all inputs a result depends on.

    When any dependency changes (including the pitch dimensions), the signature
    changes; the UI compares the stored result's signature to the current one to
    detect and discard a stale result (so a displayed verdict always matches
    current inputs).
    """
    payload = {
        "video": video_name,
        "frame": int(frame_index),
        "dir": None if attacking_direction is None else attacking_direction.value,
        "pitch": None if pitch is None else [round(pitch.length, 4), round(pitch.width, 4)],
        "corr": sorted(
            [
                c.canonical,
                round(c.image_x, 3),
                round(c.image_y, 3),
                None if c.custom_pitch_xy is None else [round(v, 4) for v in c.custom_pitch_xy],
                c.frame_index,
            ]
            for c in correspondences
        ),
        "teams": sorted([pid, side.value] for pid, side in team_sides.items()),
        "ball": None if ball_xy is None else [round(ball_xy[0], 3), round(ball_xy[1], 3)],
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class MissingInputs:
    """Which prerequisites are still missing before analysis can run."""

    reasons: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.reasons


def validate_ready_to_run(
    *,
    has_video: bool,
    frame_selected: bool,
    has_calibration: bool,
    attacking_direction: AttackingDirection | None,
    player_ids: Sequence[str],
    team_sides: Mapping[str, TeamSide],
    calibration_problem: str | None = None,
) -> MissingInputs:
    """Collect human-readable reasons analysis cannot run yet (empty => ready).

    Note: the ball is intentionally NOT required here — a missing ball is a valid
    path that yields INSUFFICIENT_DATA from the pipeline, not a blocked run.
    ``calibration_problem`` (from :func:`check_calibration`) replaces the generic
    calibration message with the specific reason.
    """
    reasons: list[str] = []
    if not has_video:
        reasons.append("Upload a video.")
    if not frame_selected:
        reasons.append("Select a frame.")
    if not has_calibration:
        reasons.append(
            calibration_problem or "Complete pitch calibration (at least 4 correspondences)."
        )
    if attacking_direction is None:
        reasons.append("Choose an attacking direction.")
    if not player_ids:
        reasons.append("Run player detection.")
    else:
        unassigned = [pid for pid in player_ids if pid not in team_sides]
        if unassigned:
            reasons.append(f"Assign a team for: {', '.join(sorted(unassigned))}.")
    return MissingInputs(reasons=tuple(reasons))
