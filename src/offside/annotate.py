"""Draw a single-frame offside analysis (and calibration aids) onto an image.

Pure rendering: takes a BGR frame and returns an annotated **copy** (the input is
never mutated). No model loading or inference happens here.

Honesty rules enforced visually:
    * The offside line is drawn ONLY when the result is a confident verdict
      (OFFSIDE_POSITION or ONSIDE). For UNCERTAIN / INSUFFICIENT_DATA no line is
      drawn and the banner explains why.
    * The ball is labelled with its source (manual / detector / unavailable).
    * The calibration overlay is a visual sanity check, not a verification: lines
      that sit on the real pitch markings suggest a plausible calibration, but a
      good-looking overlay does not prove real-world accuracy.

Design notes:
    * All colours, thicknesses and font sizes live in :class:`AnnotationStyle`.
      Sizes are scaled with the frame width so labels stay readable on large
      frames that are later shown downscaled.
    * Every distinct meaning has its own colour: cyan/blue boxes (players),
      orange label backgrounds (ids), red (offside line / OFF), dark green (ON),
      yellow (ball), white-on-dark foot points, magenta (calibration aids).
    * Labels are laid out so they stay inside the image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from offside.calibration import Calibration
from offside.geometry import pitch_to_image
from offside.offside import AttackingDirection, OffsideStatus, TeamSide
from offside.pipeline import BallSource, FrameAnalysis
from offside.pitch_model import pitch_markings

Frame = np.ndarray
Color = tuple[int, int, int]  # BGR

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_COORD_LIMIT = 1_000_000  # keep projected coordinates int32-safe for OpenCV


@dataclass(frozen=True)
class AnnotationStyle:
    """Central, overridable look of all annotations. Colours are BGR."""

    # --- colours ---
    attacker_box: Color = (255, 255, 0)        # cyan
    defender_box: Color = (255, 110, 0)        # blue (RGB 0,110,255)
    unassigned_box: Color = (200, 200, 200)    # light grey
    label_background: Color = (0, 120, 235)    # orange (RGB 235,120,0)
    label_text: Color = (255, 255, 255)        # white
    text_outline: Color = (0, 0, 0)            # dark halo behind text
    foot_point: Color = (255, 255, 255)        # white dot with dark ring
    ball: Color = (0, 255, 255)                # yellow
    ball_label_text: Color = (0, 0, 0)
    offside_line: Color = (0, 0, 255)          # red
    offside_line_casing: Color = (255, 255, 255)
    offside_mark_background: Color = (0, 0, 200)   # red pill: player OFF
    onside_mark_background: Color = (0, 130, 0)    # dark green pill: player ON
    calibration: Color = (255, 0, 255)         # magenta: calibration aids
    direction_arrow: Color = (255, 255, 255)
    banner_offside: Color = (0, 0, 200)
    banner_onside: Color = (0, 130, 0)
    banner_uncertain: Color = (0, 140, 200)
    banner_insufficient: Color = (70, 70, 70)
    # --- sizes (at ``reference_width``; scaled with the frame width) ---
    reference_width: int = 1280
    min_scale: float = 0.75
    max_scale: float = 3.0
    box_thickness: int = 3
    offside_line_thickness: int = 4
    casing_extra: int = 2
    label_font_scale: float = 0.6
    label_text_thickness: int = 2
    banner_font_scale: float = 0.8
    banner_height: int = 34
    foot_radius: int = 4
    ball_radius: int = 8
    label_padding: int = 4
    overlay_thickness: int = 2

    def box_color(self, side: TeamSide | None) -> Color:
        if side is TeamSide.ATTACKING:
            return self.attacker_box
        if side is TeamSide.DEFENDING:
            return self.defender_box
        return self.unassigned_box

    def banner_color(self, status: OffsideStatus) -> Color:
        return {
            OffsideStatus.OFFSIDE_POSITION: self.banner_offside,
            OffsideStatus.ONSIDE: self.banner_onside,
            OffsideStatus.UNCERTAIN: self.banner_uncertain,
            OffsideStatus.INSUFFICIENT_DATA: self.banner_insufficient,
        }[status]


DEFAULT_STYLE = AnnotationStyle()

_BANNER_TEXT = {
    OffsideStatus.OFFSIDE_POSITION: "OFFSIDE POSITION",
    OffsideStatus.ONSIDE: "ONSIDE",
    OffsideStatus.UNCERTAIN: "UNCERTAIN",
    OffsideStatus.INSUFFICIENT_DATA: "INSUFFICIENT DATA",
}


@dataclass(frozen=True)
class Metrics:
    """Pixel sizes resolved for one frame size (see :func:`style_metrics`)."""

    scale: float
    box: int
    line: int
    casing: int
    font: float
    text_thickness: int
    banner_font: float
    banner_height: int
    foot: int
    ball: int
    pad: int
    overlay: int


def style_metrics(style: AnnotationStyle, frame_shape: tuple[int, ...]) -> Metrics:
    """Scale the style's sizes to the frame width (never below 1 px)."""
    width = int(frame_shape[1])
    scale = min(style.max_scale, max(style.min_scale, width / float(style.reference_width)))

    def px(v: int) -> int:
        return max(1, int(round(v * scale)))

    return Metrics(
        scale=scale,
        box=px(style.box_thickness),
        line=px(style.offside_line_thickness),
        casing=px(style.casing_extra),
        font=style.label_font_scale * scale,
        text_thickness=px(style.label_text_thickness),
        banner_font=style.banner_font_scale * scale,
        banner_height=px(style.banner_height),
        foot=px(style.foot_radius),
        ball=px(style.ball_radius),
        pad=px(style.label_padding),
        overlay=px(style.overlay_thickness),
    )


def _check_frame(frame: Frame) -> None:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image of shape (H, W, 3).")


def _safe_int_point(p: Sequence[float]) -> tuple[int, int]:
    x = float(np.clip(p[0], -_COORD_LIMIT, _COORD_LIMIT))
    y = float(np.clip(p[1], -_COORD_LIMIT, _COORD_LIMIT))
    return int(round(x)), int(round(y))


def label_rect(
    text_size: tuple[int, int],
    anchor_x: float,
    anchor_top: float,
    anchor_bottom: float,
    img_w: int,
    img_h: int,
    pad: int,
    prefer: str = "above",
    min_y: int = 0,
) -> tuple[int, int, int, int]:
    """Position a label rectangle so that it stays inside the image.

    Args:
        text_size: (width, height) of the label *including* baseline.
        anchor_x: preferred left edge.
        anchor_top / anchor_bottom: vertical extent of the thing being labelled.
        prefer: "above" places the label above ``anchor_top`` (falling back to
            just inside it when there is no room); "below" places it under
            ``anchor_bottom`` (falling back to just inside it).
        min_y: highest row a label may occupy. Used to keep labels out from
            under the status banner, which is painted last.

    Returns:
        (x0, y0, x1, y1) with 0 <= x0, x1 <= img_w and min_y <= y0, y1 <= img_h
        whenever the label fits at all.
    """
    w = text_size[0] + 2 * pad
    h = text_size[1] + 2 * pad
    x0 = int(round(min(max(anchor_x, 0), max(0, img_w - w))))
    if prefer == "above":
        y0 = int(round(anchor_top)) - h
        if y0 < min_y:
            y0 = int(round(anchor_top))  # no room above: sit just inside the top edge
    else:
        y0 = int(round(anchor_bottom))
        if y0 + h > img_h:
            y0 = int(round(anchor_bottom)) - h  # no room below: sit just inside the bottom
    y0 = min(max(y0, min_y), max(min_y, img_h - h))
    return x0, y0, x0 + w, y0 + h


def _text_size(text: str, scale: float, thickness: int) -> tuple[int, int]:
    (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, thickness)
    return tw, th + baseline


def _put_bold_text(
    img: Frame, text: str, org: tuple[int, int], color: Color, outline: Color,
    scale: float, thickness: int,
) -> None:
    """Text with a dark halo so it stays legible on grass, kits and labels."""
    cv2.putText(img, text, org, _FONT, scale, outline, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, scale, color, thickness, cv2.LINE_AA)


def _draw_label(
    img: Frame, text: str, anchor_x: float, anchor_top: float, anchor_bottom: float,
    m: Metrics, background: Color, text_color: Color, outline: Color,
    prefer: str = "above", min_y: int = 0,
) -> tuple[int, int, int, int]:
    """Draw a filled label pill with bold text; returns its rectangle."""
    h, w = img.shape[:2]
    tw, th = _text_size(text, m.font, m.text_thickness)
    x0, y0, x1, y1 = label_rect(
        (tw, th), anchor_x, anchor_top, anchor_bottom, w, h, m.pad, prefer, min_y
    )
    cv2.rectangle(img, (x0, y0), (x1, y1), background, -1)
    baseline = cv2.getTextSize(text, _FONT, m.font, m.text_thickness)[1]
    _put_bold_text(img, text, (x0 + m.pad, y1 - m.pad - baseline), text_color, outline,
                   m.font, m.text_thickness)
    return x0, y0, x1, y1


def fit_text(text: str, max_width: int, scale: float, thickness: int) -> str:
    """Shorten ``text`` with an ellipsis until it fits ``max_width`` pixels."""
    if _text_size(text, scale, thickness)[0] <= max_width:
        return text
    while len(text) > 1:
        text = text[:-1]
        candidate = text.rstrip() + "..."
        if _text_size(candidate, scale, thickness)[0] <= max_width:
            return candidate
    return "..."


# --------------------------------------------------------------------------- #
# Offside analysis annotation
# --------------------------------------------------------------------------- #

def _draw_offside_line(
    img: Frame, analysis: FrameAnalysis, style: AnnotationStyle, m: Metrics
) -> tuple[int, int] | None:
    """Project the offside line (a constant pitch-X) back to image space.

    Draws only the line itself (under the players) and returns the point where
    its label should go, or None when the line is not visible. The label is
    drawn later, on top of the player boxes, so nothing cuts through its text.
    """
    line_x = analysis.offside.offside_line_x
    if line_x is None:
        return None
    width = analysis.calibration.pitch.width
    p1 = pitch_to_image(analysis.calibration, [line_x, 0.0])
    p2 = pitch_to_image(analysis.calibration, [line_x, width])
    h, w = img.shape[:2]
    inside, a, b = cv2.clipLine((0, 0, w, h), _safe_int_point(p1), _safe_int_point(p2))
    if not inside:
        return None  # the line is outside this frame
    # White casing under a thick red line keeps it distinct from boxes and pitch lines.
    cv2.line(img, a, b, style.offside_line_casing, m.line + 2 * m.casing, cv2.LINE_AA)
    cv2.line(img, a, b, style.offside_line, m.line, cv2.LINE_AA)
    return min(a, b, key=lambda p: p[1])


def annotate_frame(
    frame: Frame, analysis: FrameAnalysis, style: AnnotationStyle = DEFAULT_STYLE
) -> Frame:
    """Return an annotated copy of ``frame`` for the given analysis.

    Args:
        frame: the BGR image that was analysed.
        analysis: the structured pipeline result.
        style: colours/sizes; defaults to :data:`DEFAULT_STYLE`.

    Returns:
        A new annotated image. The input frame is not modified.

    Raises:
        ValueError: if ``frame`` is not a 3-channel image.
    """
    _check_frame(frame)
    out = frame.copy()
    m = style_metrics(style, out.shape)
    h, w = out.shape[:2]
    status = analysis.status
    confident = status in (OffsideStatus.OFFSIDE_POSITION, OffsideStatus.ONSIDE)

    # Offside line only for a confident verdict. The line goes under the players;
    # its label is drawn after them so no box edge cuts through the text.
    line_label_at = _draw_offside_line(out, analysis, style, m) if confident else None
    top = m.banner_height  # labels must not sit under the status banner

    atk_status = {c.player_id: c.status for c in analysis.offside.attackers}
    assignment = analysis.team_assignment

    for p in analysis.players:
        box = p.detection.box
        side = assignment.team_of(p.player_id) if assignment is not None else None
        color = style.box_color(side)
        x1, y1, x2, y2 = (int(box.x1), int(box.y1), int(box.x2), int(box.y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, m.box)

        side_text = {TeamSide.ATTACKING: "ATK", TeamSide.DEFENDING: "DEF"}.get(side, "?")
        _draw_label(out, f"{p.player_id} {side_text}", x1, y1, y2, m,
                    style.label_background, style.label_text, style.text_outline, min_y=top)

        fx, fy = int(p.foot_image_xy[0]), int(p.foot_image_xy[1])
        cv2.circle(out, (fx, fy), m.foot + 1, style.text_outline, -1, cv2.LINE_AA)
        cv2.circle(out, (fx, fy), m.foot, style.foot_point, -1, cv2.LINE_AA)

        if confident and p.player_id in atk_status:
            is_off = atk_status[p.player_id] is OffsideStatus.OFFSIDE_POSITION
            bg = style.offside_mark_background if is_off else style.onside_mark_background
            _draw_label(out, "OFFSIDE" if is_off else "ONSIDE", x1, y2, y2, m,
                        bg, style.label_text, style.text_outline, prefer="below", min_y=top)

    ball = analysis.ball
    if ball.image_xy is not None:
        bx, by = int(ball.image_xy[0]), int(ball.image_xy[1])
        cv2.circle(out, (bx, by), m.ball + 2, style.text_outline, m.box, cv2.LINE_AA)
        cv2.circle(out, (bx, by), m.ball, style.ball, m.box, cv2.LINE_AA)
        _draw_label(out, f"ball ({ball.source.value})", bx + m.ball, by - m.ball, by + m.ball, m,
                    style.ball, style.ball_label_text, style.ball, prefer="above", min_y=top)
    elif ball.source is BallSource.UNAVAILABLE:
        _draw_label(out, "ball: unavailable", m.pad, h - m.pad, h - m.pad, m,
                    style.ball, style.ball_label_text, style.ball, prefer="above", min_y=top)

    if line_label_at is not None:
        _draw_label(out, "OFFSIDE LINE", line_label_at[0], line_label_at[1], line_label_at[1], m,
                    style.offside_line, style.label_text, style.text_outline,
                    prefer="below", min_y=top)

    _draw_banner(out, analysis, style, m)
    return out


def _draw_banner(img: Frame, analysis: FrameAnalysis, style: AnnotationStyle, m: Metrics) -> None:
    """Status banner, plus a short explanation when the result is not confident."""
    w = img.shape[1]
    color = style.banner_color(analysis.status)
    cv2.rectangle(img, (0, 0), (w, m.banner_height), color, -1)
    _put_bold_text(img, _BANNER_TEXT[analysis.status], (m.pad * 2, int(m.banner_height * 0.72)),
                   style.label_text, style.text_outline, m.banner_font, m.text_thickness)
    if analysis.status in (OffsideStatus.UNCERTAIN, OffsideStatus.INSUFFICIENT_DATA):
        msg = fit_text(analysis.offside.explanation, w - 4 * m.pad - 2, m.font, m.text_thickness)
        _draw_label(img, msg, 0, m.banner_height + m.pad, m.banner_height + m.pad, m,
                    color, style.label_text, style.text_outline, prefer="below")


# --------------------------------------------------------------------------- #
# Calibration aids
# --------------------------------------------------------------------------- #

def _project_runs(calibration: Calibration, points: np.ndarray) -> list[np.ndarray]:
    """Project pitch points to the image, dropping points behind/near the horizon.

    A homography maps points whose homogeneous w has the opposite sign to the
    pitch centre's to mirrored positions (they would be behind the camera), and
    points with w near 0 to enormous coordinates. Those are skipped and the
    polyline is split into runs of valid consecutive points.
    """
    h_inv = calibration.homography_inv
    cx, cy = calibration.pitch.length / 2.0, calibration.pitch.width / 2.0
    ref = float(h_inv[2, 0] * cx + h_inv[2, 1] * cy + h_inv[2, 2])
    if abs(ref) < 1e-12:
        return []
    homog = np.hstack([points, np.ones((points.shape[0], 1))]) @ h_inv.T
    ratio = homog[:, 2] / ref
    valid = ratio > 0.02
    safe_w = np.where(valid, homog[:, 2], 1.0)
    xy = homog[:, :2] / safe_w[:, None]
    valid &= np.all(np.abs(xy) < _COORD_LIMIT, axis=1)
    runs: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for ok, pt in zip(valid, xy):
        if ok:
            current.append(pt)
        elif current:
            runs.append(np.array(current))
            current = []
    if current:
        runs.append(np.array(current))
    return [r for r in runs if len(r) >= 2]


def draw_calibration_overlay(
    frame: Frame,
    calibration: Calibration,
    direction: AttackingDirection | None = None,
    style: AnnotationStyle = DEFAULT_STYLE,
) -> Frame:
    """Project the calibrated pitch markings (and attack direction) onto the frame.

    This is a *visual sanity check*: if the magenta pitch lines sit on the real
    markings the calibration is plausible; if they drift, recalibrate. It does
    not verify accuracy.

    Args:
        frame: BGR frame the calibration was made on.
        calibration: the calibration to visualise.
        direction: if given, draws an arrow along pitch X toward the attacked
            goal line, using the same :class:`AttackingDirection` the offside
            calculation uses.

    Returns:
        An annotated copy of ``frame``.
    """
    _check_frame(frame)
    out = frame.copy()
    m = style_metrics(style, out.shape)
    h, w = out.shape[:2]
    pitch = calibration.pitch

    for polyline in pitch_markings(pitch).values():
        for run in _project_runs(calibration, polyline):
            pts = np.array([_safe_int_point(p) for p in run], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], False, style.text_outline, m.overlay + 2, cv2.LINE_AA)
            cv2.polylines(out, [pts], False, style.calibration, m.overlay, cv2.LINE_AA)

    cy = pitch.width / 2.0
    for x_val, text in ((0.0, "X=0 end"), (pitch.length, "X=L end")):
        for run in _project_runs(calibration, np.array([[x_val, cy], [x_val, cy]])):
            ax, ay = _safe_int_point(run[0])
            _draw_label(out, text, ax, ay, ay, m, style.calibration, style.label_text,
                        style.text_outline, prefer="below")
            break

    if direction is not None:
        ends = direction_arrow_points(calibration, direction)
        if ends is not None:
            a, b = ends
            cv2.arrowedLine(out, a, b, style.text_outline, m.line + 2 * m.casing, cv2.LINE_AA, 0, 0.25)
            cv2.arrowedLine(out, a, b, style.direction_arrow, m.line, cv2.LINE_AA, 0, 0.25)
            target = "X=L end" if direction is AttackingDirection.POSITIVE_X else "X=0 end"
            _draw_label(out, f"attacking toward {target}", b[0], b[1], b[1], m,
                        style.text_outline, style.label_text, style.text_outline, prefer="below")
    return out


def direction_arrow_points(
    calibration: Calibration, direction: AttackingDirection
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Image-space (tail, head) of the attack-direction arrow, or None if not visible.

    The arrow runs along pitch X through the pitch centre; the head is on the side
    of the goal line being attacked, taken from the same ``direction.sign`` the
    offside calculation uses.
    """
    pitch = calibration.pitch
    reach = min(12.0, pitch.length / 8.0)
    cx, cy = pitch.length / 2.0, pitch.width / 2.0
    s = direction.sign
    runs = _project_runs(calibration, np.array([[cx - s * reach, cy], [cx + s * reach, cy]]))
    if not runs:
        return None
    return _safe_int_point(runs[0][0]), _safe_int_point(runs[0][-1])


def draw_correspondence_markers(
    frame: Frame,
    markers: Sequence[tuple[str, float, float]],
    style: AnnotationStyle = DEFAULT_STYLE,
    numbers: Sequence[int] | None = None,
) -> Frame:
    """Draw numbered crosshairs for already-placed landmarks on a copy of ``frame``.

    Args:
        markers: ``(label, x, y)`` triples in the frame's pixel coordinates.
        numbers: the number drawn beside each marker. Defaults to 1, 2, 3, ... in
            sequence order; pass explicit numbers so they match a list that also
            contains markers not drawn on this frame.
    """
    _check_frame(frame)
    if numbers is not None and len(numbers) != len(markers):
        raise ValueError("numbers must have one entry per marker.")
    out = frame.copy()
    m = style_metrics(style, out.shape)
    labels_n = list(numbers) if numbers is not None else list(range(1, len(markers) + 1))
    for i, (_label, x, y) in zip(labels_n, markers):
        c = _safe_int_point((x, y))
        arm = m.ball + m.pad
        cv2.drawMarker(out, c, style.text_outline, cv2.MARKER_CROSS, arm * 2 + 2, m.overlay + 2, cv2.LINE_AA)
        cv2.drawMarker(out, c, style.calibration, cv2.MARKER_CROSS, arm * 2, m.overlay, cv2.LINE_AA)
        _draw_label(out, str(i), c[0] + arm, c[1] - arm, c[1] + arm, m,
                    style.calibration, style.label_text, style.text_outline, prefer="above")
    return out
