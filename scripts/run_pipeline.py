"""CLI: run the single-frame offside analysis end-to-end on a local clip.

This performs REAL detection and may download YOLO11n weights on first run
(internet required once). It is a manual demonstration, not part of the tests.

Inputs are supplied via a JSON config so you never edit Python to change them.

Example:
    python scripts/run_pipeline.py --config analysis.json --output outputs/frame.png

JSON config schema (all image coordinates are pixels in the analysed frame;
pitch coordinates are metres):

{
  "video": "data/raw/clip.mp4",
  "frame_index": 120,                  // OR "timestamp_seconds": 4.0
  "attacking_direction": "positive_x", // or "negative_x"
  "max_width": 960,                    // optional; must match calibration scale
  "pitch": {"length": 105.0, "width": 68.0},   // optional, defaults 105x68
  "calibration": {
    "image_points": [[x,y], [x,y], [x,y], [x,y]],
    "pitch_points": [[X,Y], [X,Y], [X,Y], [X,Y]]
  },
  "teams": {                           // frame-local ids p0, p1, ... -> side
    "p0": "attacking", "p1": "defending", ...
  },
  "ball_image_xy": [x, y]              // optional manual ball; omit to use detector policy
}

Run the pipeline once WITHOUT "teams" (or with a placeholder) to see the printed
list of detected player ids for that frame, then fill in the mapping.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make the src/ layout importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from offside.annotate import annotate_frame  # noqa: E402
from offside.calibration import PitchDimensions, calibrate, make_correspondences  # noqa: E402
from offside.config import load_config  # noqa: E402
from offside.detection import DetectorConfig, YoloDetector, split_by_class  # noqa: E402
from offside.offside import AttackingDirection, TeamSide  # noqa: E402
from offside.pipeline import PipelineError, analyze_frame, assign_player_ids  # noqa: E402
from offside.team_cluster import assign_manual  # noqa: E402
from offside.video_io import frame_index_for_timestamp, read_frame  # noqa: E402

_DIRECTIONS = {d.value: d for d in AttackingDirection}
_SIDES = {s.value: s for s in TeamSide}


def _fail(message: str) -> None:
    """Print a clear error and exit non-zero (no confusing traceback)."""
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        _fail(f"Config file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"Config is not valid JSON: {exc}")
    return {}  # unreachable


def _build_calibration(cfg: dict):
    cal = cfg.get("calibration")
    if not cal or "image_points" not in cal or "pitch_points" not in cal:
        _fail("Config must contain 'calibration' with 'image_points' and 'pitch_points'.")
    pitch_cfg = cfg.get("pitch", {})
    pitch = PitchDimensions(
        length=float(pitch_cfg.get("length", 105.0)),
        width=float(pitch_cfg.get("width", 68.0)),
    )
    try:
        corr = make_correspondences(cal["image_points"], cal["pitch_points"])
        calib = calibrate(corr, pitch=pitch)
    except Exception as exc:  # calibration raises CalibrationError subclasses
        _fail(f"Calibration failed: {exc}")
    print(
        f"Calibration OK: reprojection error ~{calib.reprojection_error_px:.2f}px / "
        f"{calib.reprojection_error_m:.3f}m. NOTE: low error means the clicked points are "
        "internally consistent, NOT that the real-world calibration is accurate."
    )
    for warning in calib.warnings:
        print(f"CALIBRATION WARNING: {warning.message}")
    return calib


def _detector() -> YoloDetector:
    app = load_config()
    return YoloDetector(
        DetectorConfig(
            model_name=app.detection.model_name,
            models_dir=app.paths.models,
            device="cpu",
            conf_threshold=app.detection.conf_threshold,
        )
    )


def _print_detected_ids(video: Path, frame_index: int, max_width, detector) -> None:
    frame = read_frame(video, frame_index, max_width=max_width)
    players, _ = split_by_class(detector.detect(frame))
    ids = assign_player_ids(players)
    print(f"Detected {len(ids)} player(s) in frame {frame_index}: {ids}")
    print("Add a 'teams' mapping for these ids to your config, then re-run.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-frame offside analysis.")
    parser.add_argument("--config", required=True, help="Path to the JSON analysis config.")
    parser.add_argument("--output", default="outputs/offside_frame.png", help="Annotated image path.")
    args = parser.parse_args()

    cfg = _load_json(Path(args.config))

    video = Path(cfg.get("video", ""))
    if not video.is_file():
        _fail(f"Video not found: {video!s} (set 'video' in the config).")

    max_width = cfg.get("max_width")

    # Resolve frame index (explicit index or timestamp).
    if "frame_index" in cfg:
        frame_index = int(cfg["frame_index"])
    elif "timestamp_seconds" in cfg:
        frame_index = frame_index_for_timestamp(video, float(cfg["timestamp_seconds"]))
        print(f"timestamp {cfg['timestamp_seconds']}s -> frame {frame_index}")
    else:
        _fail("Config must set either 'frame_index' or 'timestamp_seconds'.")

    direction_key = str(cfg.get("attacking_direction", "")).lower()
    if direction_key not in _DIRECTIONS:
        _fail(f"'attacking_direction' must be one of {list(_DIRECTIONS)}.")
    direction = _DIRECTIONS[direction_key]

    calib = _build_calibration(cfg)
    detector = _detector()

    teams_cfg = cfg.get("teams")
    if not teams_cfg:
        print("No 'teams' mapping supplied; listing detected player ids instead:\n")
        _print_detected_ids(video, frame_index, max_width, detector)
        return

    # Build the manual team assignment.
    known_ids = list(teams_cfg.keys())
    mapping = {}
    for pid, side_key in teams_cfg.items():
        if str(side_key).lower() not in _SIDES:
            _fail(f"teams[{pid!r}] must be one of {list(_SIDES)}, got {side_key!r}.")
        mapping[pid] = _SIDES[str(side_key).lower()]
    try:
        assignment = assign_manual(known_ids, mapping)
    except Exception as exc:
        _fail(f"Invalid team assignment: {exc}")

    ball_xy = cfg.get("ball_image_xy")
    manual_ball = tuple(float(v) for v in ball_xy) if ball_xy else None

    try:
        analysis = analyze_frame(
            video_path=video,
            frame_index=frame_index,
            detector=detector,
            calibration=calib,
            attacking_direction=direction,
            team_assignment=assignment,
            manual_ball_image_xy=manual_ball,
            max_width=max_width,
        )
    except PipelineError as exc:
        _fail(str(exc))
    except Exception as exc:
        _fail(f"Analysis failed: {exc}")

    # Report.
    print("\n=== Analysis ===")
    print(f"frame: {analysis.frame_index}")
    print(f"status: {analysis.status.value}")
    print(f"ball: {analysis.ball.source.value} - {analysis.ball.note}")
    print(f"explanation: {analysis.offside.explanation}")
    for c in analysis.offside.attackers:
        print(f"  {c.player_id}: {c.status.value} (margin {c.margin_m:+.2f} m, own_half={c.in_own_half})")

    # Annotate and save.
    frame = read_frame(video, frame_index, max_width=max_width)
    annotated = annotate_frame(frame, analysis)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    print(f"\nAnnotated frame saved to: {out_path}")
    print("Reminder: single-frame analysis; player ids are frame-local; offside POSITION only.")


if __name__ == "__main__":
    main()
