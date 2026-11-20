"""Streamlit UI for the AI Offside Line Detector (single-frame analysis).

Thin presentation layer over the M6 pipeline. All reusable logic lives in
``app.offside_ui`` (testable) and the ``offside`` package; this file only wires
widgets, session state, and rendering.

Run from the project root:

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Make the src/ layout importable when launched via `streamlit run`.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from app.offside_ui import (  # noqa: E402
    COORDINATE_HELP,
    FIT_VS_ACCURACY_NOTE,
    SUPPORTED_VIDEO_SUFFIXES,
    MAX_PITCH_LENGTH_M,
    MAX_PITCH_WIDTH_M,
    MIN_PITCH_LENGTH_M,
    MIN_PITCH_WIDTH_M,
    CalibrationStatus,
    Correspondence,
    UIValidationError,
    add_correspondence,
    analysis_signature,
    assess_correspondences,
    build_calibration,
    calibration_input_signature,
    calibration_point_table,
    check_calibration,
    click_stamp,
    click_to_frame_xy,
    describe_attacking_direction,
    direction_label,
    make_custom_correspondence,
    parse_attacking_direction,
    prepare_ball_xy,
    prepare_team_assignment,
    remove_correspondence,
    render_direction_diagram,
    validate_pitch_dimensions,
    validate_ready_to_run,
)
from offside.annotate import (  # noqa: E402
    annotate_frame,
    draw_calibration_overlay,
    draw_correspondence_markers,
)
from offside.calibration import PitchDimensions  # noqa: E402
from offside.config import load_config  # noqa: E402
from offside.detection import DetectorConfig, YoloDetector, split_by_class  # noqa: E402
from offside.offside import AttackingDirection, OffsideStatus, TeamSide  # noqa: E402
from offside.pipeline import analyze_frame, assign_player_ids  # noqa: E402
from offside.pitch_model import CUSTOM_POINT_MARGIN_M, available_landmarks  # noqa: E402
from offside.video_io import (  # noqa: E402
    VideoReadError,
    frame_index_for_timestamp,
    probe_video,
    read_frame,
)

try:
    from streamlit_image_coordinates import streamlit_image_coordinates as st_image_coords
    _HAS_CLICK = True
except Exception:  # pragma: no cover - component import guard
    _HAS_CLICK = False


# --------------------------------------------------------------------------- #
# Cached resources / helpers
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def get_detector() -> YoloDetector:
    """Load the detector once per session (weights load lazily on first detect)."""
    app = load_config()
    return YoloDetector(
        DetectorConfig(
            model_name=app.detection.model_name,
            models_dir=app.paths.models,
            device="cpu",
            conf_threshold=app.detection.conf_threshold,
        )
    )


def _weights_present() -> bool:
    app = load_config()
    return (app.paths.models / app.detection.model_name).is_file()


def _ss_init() -> None:
    ss = st.session_state
    ss.setdefault("video_path", None)
    ss.setdefault("video_name", None)
    ss.setdefault("frame_index", 0)
    ss.setdefault("correspondences", [])   # list[Correspondence]
    ss.setdefault("calibration", None)     # Calibration | None
    ss.setdefault("calibration_sig", None)  # signature of the inputs it was built from
    ss.setdefault("player_ids", [])        # list[str]
    ss.setdefault("detected_boxes", {})    # pid -> (x1,y1,x2,y2)
    ss.setdefault("detect_key", None)      # (video_name, frame_index) of last detect
    ss.setdefault("team_sides", {})        # pid -> TeamSide
    ss.setdefault("ball_xy", None)         # (x, y) | None
    ss.setdefault("calib_last_click", None)  # (x, y) in true frame px | None
    ss.setdefault("ball_last_click", None)   # (x, y) in true frame px | None
    ss.setdefault("result", None)          # FrameAnalysis | None
    ss.setdefault("result_sig", None)      # signature of inputs when result computed


def _to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


# Max on-screen width (px) for click images, so a large frame fits the page
# instead of rendering at full resolution (which looked "zoomed in").
_CLICK_DISPLAY_WIDTH = 900


def _poll_click(frame_bgr: np.ndarray, key: str, state_key: str) -> None:
    """Show a click image fitted to the page and record a NEW click in session state.

    The click is stored in TRUE frame pixels under ``state_key``. The component
    re-reports its last click on every rerun, so each click is recorded once
    (tracked by its stamp) and a consumed click is not resurrected.
    """
    ss = st.session_state
    real_w = int(frame_bgr.shape[1])
    display_w = min(_CLICK_DISPLAY_WIDTH, real_w)  # never upscale
    coords = st_image_coords(_to_rgb(frame_bgr), width=display_w, key=key)
    if coords is None:
        return
    stamp = click_stamp(coords)
    if ss.get(f"{state_key}_stamp") == stamp:
        return
    ss[f"{state_key}_stamp"] = stamp
    ss[state_key] = click_to_frame_xy(coords, real_w, display_w)


def _current_pitch() -> PitchDimensions | None:
    """The validated pitch dimensions, or None if the inputs are invalid."""
    try:
        return validate_pitch_dimensions(
            st.session_state.get("pitch_len", 105.0), st.session_state.get("pitch_wid", 68.0)
        )
    except UIValidationError:
        return None


def _current_direction() -> AttackingDirection | None:
    label = st.session_state.get("direction_label")
    if not label:
        return None
    try:
        return parse_attacking_direction(label)
    except UIValidationError:
        return None


def _invalidate_result() -> None:
    st.session_state["result"] = None
    st.session_state["result_sig"] = None


def _on_pitch_change() -> None:
    """Changing the pitch size invalidates the calibration and any result.

    Catalogue landmark coordinates are derived from the pitch dimensions, so a
    calibration built for the old size is wrong for the new one.
    """
    st.session_state["calibration"] = None
    st.session_state["calibration_sig"] = None
    _invalidate_result()


def _calibration_check(pitch: PitchDimensions):
    ss = st.session_state
    return check_calibration(
        ss.get("calibration"), ss.get("calibration_sig"), ss["correspondences"],
        pitch, ss["frame_index"],
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

def render_sidebar():
    """Render the sidebar; returns (pitch or None, container for calibration status)."""
    st.sidebar.title("AI Offside Line Detector")
    st.sidebar.caption(
        "Experimental single-camera offside-**position** estimator. "
        "Academic/portfolio project — not VAR and not an official decision."
    )

    st.sidebar.header("1. Video")
    upload = st.sidebar.file_uploader(
        "Upload a short clip",
        type=[s.lstrip(".") for s in SUPPORTED_VIDEO_SUFFIXES],
    )
    if upload is not None:
        _handle_upload(upload)

    st.sidebar.header("Analysis settings")
    st.sidebar.number_input(
        "Pitch length, X axis (m)", MIN_PITCH_LENGTH_M, MAX_PITCH_LENGTH_M, 105.0,
        step=0.5, format="%.2f", key="pitch_len", on_change=_on_pitch_change,
        help="Real length of the pitch in metres (goal line to goal line).",
    )
    st.sidebar.number_input(
        "Pitch width, Y axis (m)", MIN_PITCH_WIDTH_M, MAX_PITCH_WIDTH_M, 68.0,
        step=0.5, format="%.2f", key="pitch_wid", on_change=_on_pitch_change,
        help="Real width of the pitch in metres (touchline to touchline).",
    )
    pitch = _current_pitch()
    if pitch is None:
        try:
            validate_pitch_dimensions(
                st.session_state.get("pitch_len"), st.session_state.get("pitch_wid")
            )
        except UIValidationError as exc:
            st.sidebar.error(str(exc))
    else:
        st.sidebar.caption(
            f"Pitch model: {pitch.length:g} m (X) × {pitch.width:g} m (Y). Changing these "
            "clears the calibration and any result."
        )

    label_pitch = pitch or PitchDimensions()
    st.sidebar.selectbox(
        "Attacking direction",
        [d.value for d in AttackingDirection],
        key="direction_label",
        format_func=lambda v: direction_label(parse_attacking_direction(v), label_pitch),
        help="The goal line the attacking team is attacking, along the pitch X axis.",
        on_change=_invalidate_result,
    )
    direction = _current_direction()
    if direction is not None:
        st.sidebar.image(
            render_direction_diagram(label_pitch, direction),
            caption="Orange line = the goal line being attacked.",
            width="stretch",
        )
        st.sidebar.caption(describe_attacking_direction(direction, label_pitch))

    st.sidebar.header("Calibration status")
    status_box = st.sidebar.container()

    st.sidebar.info(
        "Single-frame analysis only. Player ids (p0, p1, ...) are valid for the "
        "selected frame only — no tracking. Reports offside POSITION, not an "
        "offence.",
        icon="ℹ️",
    )
    if not _weights_present():
        st.sidebar.warning(
            "Model weights not found in models/. Run `python scripts/fetch_models.py` "
            "once (needs internet) before detection.",
            icon="⚠️",
        )
    return pitch, status_box


def _render_calibration_status(container, pitch: PitchDimensions | None) -> None:
    """Fill the sidebar status box (called after the frame is known)."""
    ss = st.session_state
    n = len(ss["correspondences"])
    container.write(f"Landmarks placed: **{n}** (need at least 4)")
    if pitch is None:
        container.error("Fix the pitch dimensions first.")
        return
    check = _calibration_check(pitch)
    if check.status is CalibrationStatus.READY:
        calib = ss["calibration"]
        container.success("Calibration ready for this frame.")
        if calib.warnings:
            container.warning(f"{len(calib.warnings)} calibration warning(s) — see section 4.")
    elif check.status is CalibrationStatus.MISSING:
        container.info("Not calibrated yet.")
    else:
        container.warning(check.message)


def _handle_upload(upload) -> None:
    ss = st.session_state
    # Only re-save when a different file arrives (avoid duplicate temp copies).
    if ss.get("video_name") == upload.name and ss.get("video_path"):
        return
    # Clean up a previous temp file.
    _cleanup_temp_video()
    suffix = Path(upload.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.getbuffer())
    tmp.flush()
    tmp.close()
    ss["video_path"] = tmp.name
    ss["video_name"] = upload.name
    # Reset downstream state for the new video.
    ss["correspondences"] = []
    ss["calibration"] = None
    ss["calibration_sig"] = None
    ss["calib_last_click"] = None
    ss["player_ids"] = []
    ss["detected_boxes"] = {}
    ss["detect_key"] = None
    ss["team_sides"] = {}
    ss["ball_xy"] = None
    _invalidate_result()


def _cleanup_temp_video() -> None:
    path = st.session_state.get("video_path")
    if path and Path(path).is_file():
        try:
            Path(path).unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Main sections
# --------------------------------------------------------------------------- #

def render_frame_selection() -> np.ndarray | None:
    ss = st.session_state
    if not ss.get("video_path"):
        st.info("Upload a video in the sidebar to begin.")
        return None

    try:
        meta = probe_video(ss["video_path"])
    except VideoReadError as exc:
        st.error(f"Could not read the uploaded video: {exc}")
        return None

    st.subheader("2. Select the pass-moment frame")
    cols = st.columns(4)
    cols[0].metric("Resolution", f"{meta.width}x{meta.height}")
    cols[1].metric("FPS", f"{meta.fps:.1f}")
    cols[2].metric("Frames", f"{meta.frame_count}")
    cols[3].metric("Duration", f"{meta.duration_seconds:.1f}s")

    mode = st.radio("Select by", ["Frame index", "Timestamp (s)"], horizontal=True)
    max_index = max(0, meta.frame_count - 1)
    if mode == "Frame index":
        idx = st.slider("Frame index", 0, max_index, min(ss["frame_index"], max_index))
    else:
        ts = st.slider("Timestamp (s)", 0.0, max(0.1, meta.duration_seconds), 0.0, 0.1)
        try:
            idx = min(frame_index_for_timestamp(ss["video_path"], ts), max_index)
        except VideoReadError:
            idx = 0
        st.caption(f"-> frame {idx}")

    if idx != ss["frame_index"]:
        ss["frame_index"] = idx
        ss["calib_last_click"] = None
        _invalidate_result()

    try:
        frame = read_frame(ss["video_path"], ss["frame_index"])
    except VideoReadError as exc:
        st.error(f"Could not read frame {ss['frame_index']}: {exc}")
        return None

    st.image(_to_rgb(frame), caption=f"Frame {ss['frame_index']}", width="stretch")
    return frame


def render_detection(frame: np.ndarray) -> None:
    ss = st.session_state
    st.subheader("3. Detect players")
    st.caption("IDs p0, p1, ... are valid for THIS frame only (no tracking).")

    detect_key = (ss["video_name"], ss["frame_index"])
    if st.button("Detect players", type="primary"):
        if not _weights_present():
            st.error("Model weights missing. Run `python scripts/fetch_models.py` first.")
        else:
            try:
                with st.spinner("Running CPU detection..."):
                    detector = get_detector()
                    players, _balls = split_by_class(detector.detect(frame))
                    ids = assign_player_ids(players)
                ss["player_ids"] = ids
                ss["detected_boxes"] = {
                    pid: (p.box.x1, p.box.y1, p.box.x2, p.box.y2)
                    for pid, p in zip(ids, players)
                }
                ss["detect_key"] = detect_key
                # Drop team assignments for ids that no longer exist.
                ss["team_sides"] = {
                    pid: s for pid, s in ss["team_sides"].items() if pid in ids
                }
                _invalidate_result()
                st.success(f"Detected {len(ids)} player(s).")
            except Exception as exc:
                st.error(f"Detection failed: {exc}")

    if ss["player_ids"] and ss["detect_key"] != detect_key:
        st.warning("Frame changed since last detection — click Detect players again.")

    if ss["player_ids"]:
        preview = frame.copy()
        for pid, (x1, y1, x2, y2) in ss["detected_boxes"].items():
            cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 3)
            cv2.putText(preview, pid, (int(x1), max(14, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(preview, pid, (int(x1), max(14, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        st.image(_to_rgb(preview), caption="Detected players", width="stretch")


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #

def render_calibration(frame: np.ndarray, pitch: PitchDimensions) -> None:
    ss = st.session_state
    frame_idx = ss["frame_index"]
    h, w = frame.shape[:2]
    image_size = (w, h)
    corrs: list[Correspondence] = ss["correspondences"]

    st.subheader("4. Calibrate the pitch")
    st.caption(
        "Pair landmarks you can see with their position on the pitch. Use ANY landmarks "
        "that are visible in this frame: at least 4, not all on one line, and spread across "
        "the pitch. Every click must be the exact physical point the landmark names."
    )
    with st.expander("What do X, Y, 'end' and 'side' mean?"):
        st.write(COORDINATE_HELP)
        st.caption("The diagram in the sidebar shows the axes and the attacking direction.")

    # ---- Step 1: click the image --------------------------------------------
    st.markdown("**Step 1 — click the landmark in the image**")
    here = [(i, c) for i, c in enumerate(corrs, start=1) if c.frame_index in (None, frame_idx)]
    marked = draw_correspondence_markers(
        frame,
        [(c.display_label(), c.image_x, c.image_y) for _i, c in here],
        numbers=[i for i, _c in here],
    )
    if _HAS_CLICK:
        _poll_click(marked, key=f"calib_click_{frame_idx}", state_key="calib_last_click")
    else:
        st.info("Click component unavailable; enter the image pixel coordinates below.")
        st.image(_to_rgb(marked), width="stretch")
        c1, c2, c3 = st.columns(3)
        nx = c1.number_input("Image x", 0.0, float(w), 0.0)
        ny = c2.number_input("Image y", 0.0, float(h), 0.0)
        if c3.button("Use this point"):
            ss["calib_last_click"] = (float(nx), float(ny))
    point = ss.get("calib_last_click")
    if point is None:
        st.info("Click the landmark's exact location in the image above.")
    else:
        st.success(f"Image point selected: ({point[0]:.0f}, {point[1]:.0f}) px.")

    # ---- Step 2: say which landmark it is ------------------------------------
    st.markdown("**Step 2 — say which pitch landmark that point is**")
    tab_catalogue, tab_custom = st.tabs(["Pitch landmark", "Custom point (advanced)"])
    with tab_catalogue:
        _render_catalogue_picker(pitch, point, frame_idx, image_size)
    with tab_custom:
        _render_custom_picker(pitch, point, frame_idx, image_size)

    # ---- Placed landmarks ----------------------------------------------------
    st.markdown(f"**Placed landmarks ({len(corrs)})**")
    if not corrs:
        st.caption("None yet.")
    for i, c in enumerate(corrs, start=1):
        cols = st.columns([0.5, 4.5, 2.2, 2.2, 1.4, 1.2])
        cols[0].write(f"**{i}**")
        cols[1].write(c.display_label())
        cols[2].write(f"image ({c.image_x:.0f}, {c.image_y:.0f})")
        if c.is_valid_for(pitch):
            px, py = c.pitch_xy(pitch)
            cols[3].write(f"pitch ({px:.1f}, {py:.1f}) m")
        else:
            cols[3].write("⚠ not on this pitch")
        other = c.frame_index is not None and c.frame_index != frame_idx
        cols[4].write(f"⚠ frame {c.frame_index}" if other else f"frame {c.frame_index}")
        if cols[5].button("Remove", key=f"rm_{c.canonical}"):
            ss["correspondences"] = remove_correspondence(corrs, c.landmark_key)
            ss["calib_last_click"] = None
            _invalidate_result()
            st.rerun()
    if corrs and st.button("Remove all landmarks"):
        ss["correspondences"] = []
        ss["calib_last_click"] = None
        _invalidate_result()
        st.rerun()

    # ---- Live validation -----------------------------------------------------
    summary = assess_correspondences(corrs, pitch, image_size)
    st.write(f"Valid landmarks: **{summary.valid}** of {summary.placed} placed (need at least 4).")
    if summary.invalid_labels:
        st.error("Not available on this pitch size: " + "; ".join(summary.invalid_labels))
    if summary.valid < 4:
        if summary.placed:
            st.info(f"Place {4 - summary.valid} more valid landmark(s).")
    elif not summary.assessment.ok:
        for msg in summary.assessment.errors:
            st.error(msg)
    else:
        st.success("These landmarks can be used to build a calibration.")
        for wn in summary.assessment.warnings:
            st.warning(wn.message)

    if st.button("Build calibration", type="primary", disabled=not summary.can_build):
        try:
            ss["calibration"] = build_calibration(corrs, pitch, image_size)
            ss["calibration_sig"] = calibration_input_signature(corrs, pitch)
            _invalidate_result()
        except UIValidationError as exc:
            ss["calibration"] = None
            ss["calibration_sig"] = None
            st.error(str(exc))

    _render_calibration_report(frame, pitch)


def _render_catalogue_picker(
    pitch: PitchDimensions, point: tuple[float, float] | None, frame_idx: int,
    image_size: tuple[int, int],
) -> None:
    ss = st.session_state
    corrs: list[Correspondence] = ss["correspondences"]
    available = available_landmarks(pitch)
    groups = list(dict.fromkeys(lm.group for lm in available))
    group = st.selectbox("Landmark group", groups, key="lm_group")
    in_group = [lm for lm in available if lm.group == group]
    placed = {c.canonical for c in corrs}
    options = {f"{'✓ ' if lm.key in placed else ''}{lm.label}": lm for lm in in_group}
    choice = st.selectbox("Landmark (✓ = already placed)", list(options), key=f"lm_choice_{group}")
    lm = options[choice]
    px, py = lm.pitch_xy(pitch)
    st.caption(f"{lm.description}  →  pitch position X = {px:.2f} m, Y = {py:.2f} m")
    already = lm.key in placed
    if already:
        st.warning("This landmark is already placed. Replace it, or remove it from the list below.")
    if st.button(
        "Replace its placement" if already else "Add landmark",
        disabled=point is None, key="add_catalogue",
    ):
        new = Correspondence(lm.key, point[0], point[1], frame_index=frame_idx)
        try:
            ss["correspondences"] = add_correspondence(corrs, new, pitch, replace=already)
            ss["calib_last_click"] = None
            _invalidate_result()
            st.rerun()
        except UIValidationError as exc:
            st.error(str(exc))


def _render_custom_picker(
    pitch: PitchDimensions, point: tuple[float, float] | None, frame_idx: int,
    image_size: tuple[int, int],
) -> None:
    ss = st.session_state
    corrs: list[Correspondence] = ss["correspondences"]
    st.caption(
        "Use this for a visible ground marking that is not in the list. Enter its real-world "
        "position in metres with the SAME convention: X along the length from the X=0 goal "
        "line, Y across the width from the Y=0 touchline."
    )
    c1, c2 = st.columns(2)
    m = CUSTOM_POINT_MARGIN_M
    cx = c1.number_input("Pitch X (m)", -m, pitch.length + m, 0.0, step=0.1, key="custom_x")
    cy = c2.number_input("Pitch Y (m)", -m, pitch.width + m, 0.0, step=0.1, key="custom_y")
    name = st.text_input("Name (optional)", key="custom_name")
    if st.button("Add custom point", disabled=point is None, key="add_custom"):
        try:
            new = make_custom_correspondence(
                corrs, pitch_x=cx, pitch_y=cy, image_x=point[0], image_y=point[1],
                pitch=pitch, frame_index=frame_idx, label=name, image_size=image_size,
            )
            ss["correspondences"] = add_correspondence(corrs, new, pitch)
            ss["calib_last_click"] = None
            _invalidate_result()
            st.rerun()
        except UIValidationError as exc:
            st.error(str(exc))


def _render_calibration_report(frame: np.ndarray, pitch: PitchDimensions) -> None:
    ss = st.session_state
    check = _calibration_check(pitch)
    if check.status is CalibrationStatus.MISSING:
        return
    if not check.ready:
        st.warning(check.message)
        return

    calib = ss["calibration"]
    st.markdown("**Calibration result**")
    cols = st.columns(4)
    cols[0].metric("Mean reprojection error", f"{calib.reprojection_error_px:.2f} px")
    cols[1].metric("Worst landmark", f"{calib.max_reprojection_error_px:.2f} px")
    cols[2].metric("Error on pitch", f"{calib.reprojection_error_m:.2f} m")
    used = calib.num_inliers if calib.num_inliers is not None else calib.num_points
    cols[3].metric("Landmarks used", f"{used} of {calib.num_points}")
    cov = []
    if calib.pitch_coverage is not None:
        cov.append(f"pitch area covered by the landmarks: {calib.pitch_coverage:.0%}")
    if calib.image_coverage is not None:
        cov.append(f"frame area covered: {calib.image_coverage:.0%}")
    if cov:
        st.caption("; ".join(cov))

    for wn in calib.warnings:
        st.warning(wn.message)
    st.caption(FIT_VS_ACCURACY_NOTE)

    with st.expander("Per-landmark residuals"):
        st.dataframe(calibration_point_table(calib), width="stretch")

    overlay = draw_calibration_overlay(frame, calib, _current_direction())
    st.image(
        _to_rgb(overlay),
        caption="Visual check: magenta lines are the pitch markings projected from your "
                "calibration. They should sit on the real lines; if they drift, recalibrate.",
        width="stretch",
    )


# --------------------------------------------------------------------------- #
# Teams and ball
# --------------------------------------------------------------------------- #

def render_team_assignment() -> None:
    ss = st.session_state
    st.subheader("5. Assign teams")
    if not ss["player_ids"]:
        st.info("Detect players first.")
        return
    st.caption("Assign every detected player to a side. Manual assignment is the reliable path.")

    for pid in ss["player_ids"]:
        current = ss["team_sides"].get(pid)
        choice = st.radio(
            f"Player {pid}",
            ["Unassigned", "Attacking", "Defending"],
            index={None: 0, TeamSide.ATTACKING: 1, TeamSide.DEFENDING: 2}[current],
            horizontal=True,
            key=f"team_{pid}",
        )
        new = {"Unassigned": None, "Attacking": TeamSide.ATTACKING,
               "Defending": TeamSide.DEFENDING}[choice]
        if new != current:
            if new is None:
                ss["team_sides"].pop(pid, None)
            else:
                ss["team_sides"][pid] = new
            _invalidate_result()

    unassigned = [p for p in ss["player_ids"] if p not in ss["team_sides"]]
    if unassigned:
        st.warning(f"Unassigned: {', '.join(unassigned)}")
    else:
        st.success("All players assigned.")


def render_ball(frame: np.ndarray) -> None:
    ss = st.session_state
    st.subheader("6. Place the ball")
    st.caption("A manual ball position is authoritative and labelled MANUAL. "
               "Without a reliable ball, the result is INSUFFICIENT_DATA.")

    if _HAS_CLICK:
        st.caption("Click the ball on the image below (image is scaled to fit).")
        _poll_click(frame, key=f"ball_click_{ss['frame_index']}", state_key="ball_last_click")
        last = ss.get("ball_last_click")
        if last is not None and st.button(f"Set ball at ({last[0]:.0f}, {last[1]:.0f})"):
            ss["ball_xy"] = (last[0], last[1])
            ss["ball_last_click"] = None
            _invalidate_result()
    else:
        c1, c2, c3 = st.columns(3)
        bx = c1.number_input("Ball x", 0.0, float(frame.shape[1]), 0.0)
        by = c2.number_input("Ball y", 0.0, float(frame.shape[0]), 0.0)
        if c3.button("Set ball"):
            ss["ball_xy"] = (float(bx), float(by))
            _invalidate_result()

    if ss["ball_xy"] is not None:
        st.write(f"Ball at image ({ss['ball_xy'][0]:.0f}, {ss['ball_xy'][1]:.0f}) — source MANUAL")
        if st.button("Clear ball"):
            ss["ball_xy"] = None
            _invalidate_result()


# --------------------------------------------------------------------------- #
# Run and results
# --------------------------------------------------------------------------- #

def render_run_and_results(frame: np.ndarray, pitch: PitchDimensions) -> None:
    ss = st.session_state
    st.subheader("7. Run analysis")

    check = _calibration_check(pitch)
    missing = validate_ready_to_run(
        has_video=bool(ss.get("video_path")),
        frame_selected=True,
        has_calibration=check.ready,
        calibration_problem=None if check.ready else check.message,
        attacking_direction=_current_direction(),
        player_ids=ss["player_ids"],
        team_sides=ss["team_sides"],
    )
    if not missing.ok:
        st.warning("Before running:\n" + "\n".join(f"- {r}" for r in missing.reasons))
        return

    current_sig = analysis_signature(
        video_name=ss["video_name"] or "",
        frame_index=ss["frame_index"],
        attacking_direction=_current_direction(),
        correspondences=ss["correspondences"],
        team_sides=ss["team_sides"],
        ball_xy=ss["ball_xy"],
        pitch=pitch,
    )

    if st.button("Run analysis", type="primary"):
        try:
            ball_xy = prepare_ball_xy(ss["ball_xy"])
            assignment = prepare_team_assignment(ss["player_ids"], ss["team_sides"])
            with st.spinner("Analysing..."):
                detector = get_detector()
                analysis = analyze_frame(
                    video_path=ss["video_path"],
                    frame_index=ss["frame_index"],
                    detector=detector,
                    calibration=ss["calibration"],
                    attacking_direction=_current_direction(),
                    team_assignment=assignment,
                    manual_ball_image_xy=ball_xy,
                )
            ss["result"] = analysis
            ss["result_sig"] = current_sig
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

    # Only show a result that matches the current inputs.
    result = ss.get("result")
    if result is not None and ss.get("result_sig") == current_sig:
        _render_result(frame, result)
    elif result is not None:
        st.info("Inputs changed since the last run — click Run analysis to refresh.")


def _render_result(frame: np.ndarray, analysis) -> None:
    st.subheader("8. Result")
    status = analysis.status
    banner = {
        OffsideStatus.OFFSIDE_POSITION: ("error", "OFFSIDE POSITION"),
        OffsideStatus.ONSIDE: ("success", "ONSIDE"),
        OffsideStatus.UNCERTAIN: ("warning", "UNCERTAIN"),
        OffsideStatus.INSUFFICIENT_DATA: ("warning", "INSUFFICIENT DATA"),
    }[status]
    getattr(st, banner[0])(banner[1])
    st.write(analysis.offside.explanation)
    for wn in analysis.calibration.warnings:
        st.warning("Calibration: " + wn.message)

    annotated = annotate_frame(frame, analysis)
    st.image(_to_rgb(annotated), caption="Annotated result", width="stretch")

    # Download.
    ok, buf = cv2.imencode(".png", annotated)
    if ok:
        st.download_button(
            "Download annotated image",
            data=buf.tobytes(),
            file_name=f"offside_frame_{analysis.frame_index}.png",
            mime="image/png",
        )

    # Details.
    with st.expander("Details"):
        st.write(f"**Ball source:** {analysis.ball.source.value} — {analysis.ball.note}")
        if analysis.offside.offside_line_x is not None:
            st.write(f"**Offside line:** pitch X = {analysis.offside.offside_line_x:.2f} m")
        if analysis.offside.reference_kind is not None:
            st.write(f"**Reference:** {analysis.offside.reference_kind.value}")
        st.write(
            f"**Calibration fit:** reprojection ~{analysis.calibration.reprojection_error_px:.2f} px / "
            f"{analysis.calibration.reprojection_error_m:.3f} m (mathematical fit, not "
            "real-world accuracy)"
        )
        if analysis.offside.attackers:
            st.write("**Attackers:**")
            for c in analysis.offside.attackers:
                st.write(
                    f"- {c.player_id}: {c.status.value} "
                    f"(margin {c.margin_m:+.2f} m, own_half={c.in_own_half})"
                )
        st.caption(
            "Offside POSITION only — not an offence (active involvement not judged). "
            "Not VAR-grade; accuracy depends on calibration quality."
        )


def _render_workflow(pitch: PitchDimensions | None) -> None:
    frame = render_frame_selection()
    if frame is None:
        return
    if pitch is None:
        st.error("The pitch dimensions are invalid; fix them in the sidebar to continue.")
        return

    render_detection(frame)
    render_calibration(frame, pitch)
    render_team_assignment()
    render_ball(frame)
    render_run_and_results(frame, pitch)


def main() -> None:
    st.set_page_config(page_title="AI Offside Line Detector", layout="wide")
    _ss_init()
    pitch, status_box = render_sidebar()
    _render_workflow(pitch)
    # Filled last, into the container created earlier in the sidebar, so it reflects
    # whatever the sections above changed during this run.
    _render_calibration_status(status_box, pitch)


if __name__ == "__main__":
    main()
