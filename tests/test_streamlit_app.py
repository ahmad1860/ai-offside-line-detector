"""Executes the real Streamlit script in-process (streamlit.testing AppTest).

This runs app/streamlit_app.py end to end without a browser, so widget wiring and
session-state transitions are exercised for real. It does NOT test the third-party
click component's browser behaviour (clicking pixels), which needs manual checking.
No model weights, internet, or GPU are used (detection is never triggered).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from app.offside_ui import Correspondence, calibration_input_signature
from offside.calibration import PitchDimensions

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")

_CORNER_CLICKS = [
    ("corner_x0_y0", 60, 330), ("corner_xL_y0", 580, 330),
    ("corner_xL_yW", 480, 150), ("corner_x0_yW", 160, 150),
]


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
    assert writer.isOpened()
    try:
        for i in range(20):
            writer.write(np.full((360, 640, 3), (40 + i, 120, 40), dtype=np.uint8))
    finally:
        writer.release()
    return path


def corners(n: int = 4, frame_index: int = 0) -> list[Correspondence]:
    return [Correspondence(k, x, y, frame_index=frame_index) for k, x, y in _CORNER_CLICKS[:n]]


def start(clip: Path | None, landmarks: list[Correspondence] | None = None, **state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    if clip is not None:
        at.session_state["video_path"] = str(clip)
        at.session_state["video_name"] = "clip.mp4"
    if landmarks is not None:
        at.session_state["correspondences"] = landmarks
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def button(at: AppTest, label: str):
    found = [b for b in at.button if b.label == label]
    assert found, f"no button {label!r}; have {[b.label for b in at.button]}"
    return found[0]


def build(at: AppTest) -> AppTest:
    return button(at, "Build calibration").click().run()


def texts(elements) -> str:
    return " | ".join(str(e.value) for e in elements)


def test_renders_without_a_video() -> None:
    at = start(None)
    assert not at.exception
    assert "Upload a video" in texts(at.info)


def test_renders_the_full_calibration_workflow_with_a_video(clip: Path) -> None:
    at = start(clip, corners())
    assert not at.exception
    labels = [b.label for b in at.button]
    assert "Detect players" in labels and "Build calibration" in labels
    assert "Add custom point" in labels
    assert "Remove all landmarks" in labels


def test_build_is_disabled_until_four_valid_landmarks(clip: Path) -> None:
    assert button(start(clip, corners(3)), "Build calibration").disabled
    assert not button(start(clip, corners(4)), "Build calibration").disabled


def test_collinear_landmarks_block_building_and_explain_why(clip: Path) -> None:
    line = [Correspondence(k, 100 + 50 * i, 200, frame_index=0)
            for i, k in enumerate(["halfway_y0", "centre_mark", "halfway_yW", "centre_circle_y0"])]
    at = start(clip, line)
    assert not at.exception
    assert button(at, "Build calibration").disabled
    assert "cannot determine a homography" in texts(at.error)


def test_building_stores_a_calibration_tied_to_its_inputs(clip: Path) -> None:
    at = build(start(clip, corners()))
    assert not at.exception
    assert at.session_state["calibration"] is not None
    assert at.session_state["calibration_sig"] == calibration_input_signature(
        corners(), PitchDimensions(105.0, 68.0)
    )
    assert "Calibration ready for this frame" in texts(at.sidebar.success)


def test_exact_four_point_fit_is_warned_about_after_building(clip: Path) -> None:
    at = build(start(clip, corners()))
    assert "exact fit" in texts(at.warning).lower()


@pytest.mark.parametrize("key, new_value", [("pitch_len", 100.0), ("pitch_wid", 64.0)])
def test_changing_pitch_dimensions_clears_the_calibration(clip: Path, key: str, new_value: float) -> None:
    at = build(start(clip, corners()))
    assert at.session_state["calibration"] is not None
    at.sidebar.number_input(key=key).set_value(new_value).run()
    assert not at.exception
    assert at.session_state["calibration"] is None
    assert at.session_state["calibration_sig"] is None
    assert at.session_state["result"] is None
    assert "Not calibrated yet" in texts(at.sidebar.info)


def test_selecting_another_frame_flags_the_calibration_and_blocks_running(clip: Path) -> None:
    at = build(start(clip, corners()))
    at.session_state["frame_index"] = 5
    at.run()
    assert not at.exception
    assert "frame 0" in texts(at.sidebar.warning) and "frame 5" in texts(at.sidebar.warning)
    assert "Recalibrate" in texts(at.warning)  # the run section names the real reason


def test_invalid_pitch_dimensions_are_reported_and_stop_the_workflow(clip: Path) -> None:
    at = start(clip, corners())
    at.sidebar.number_input(key="pitch_len").set_value(60.0)
    at.sidebar.number_input(key="pitch_wid").set_value(70.0).run()  # length <= width
    assert not at.exception
    assert "must be greater than its width" in texts(at.sidebar.error)
    assert "pitch dimensions are invalid" in texts(at.error)
    assert "Build calibration" not in [b.label for b in at.button]


def test_placed_landmark_offers_replace_instead_of_a_silent_duplicate(clip: Path) -> None:
    at = start(clip, corners())
    labels = [b.label for b in at.button]
    assert "Replace its placement" in labels
    assert "Add landmark" not in labels
    assert "already placed" in texts(at.warning)


def test_direction_explanation_follows_the_selection(clip: Path) -> None:
    at = start(clip, corners())
    assert "X = 105" in texts(at.sidebar.caption)
    at.sidebar.selectbox(key="direction_label").select("negative_x").run()
    assert not at.exception
    assert "goal line at X = 0" in texts(at.sidebar.caption)


def test_changing_direction_invalidates_a_stale_result(clip: Path) -> None:
    at = start(clip, corners(), result="old", result_sig="x")
    at.sidebar.selectbox(key="direction_label").select("negative_x").run()
    assert at.session_state["result"] is None
