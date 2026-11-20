"""Demonstration: run YOLO detection on sampled frames of a local video.

This performs REAL inference and may download YOLO11n weights on first run
(internet required once). It is a manual demo, not part of the automated tests.

Usage (from the project root, with the venv active):

    python scripts/test_detection.py --video data/raw/clip.mp4
    python scripts/test_detection.py --video data/raw/clip.mp4 --max-frames 5 --save

Options:
    --video       Path to a local video file (required).
    --stride      Sample every Nth frame (default: from config).
    --max-frames  Stop after this many sampled frames (default: 5).
    --conf        Confidence threshold (default: from config).
    --save        Save annotated frames to outputs/ instead of only summarizing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# Make the src/ layout importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offside.config import load_config
from offside.detection import (
    DetectorConfig,
    YoloDetector,
    draw_detections,
    split_by_class,
)
from offside.video_io import iter_sampled_frames, probe_video


def build_detector(conf_threshold: float, models_dir: Path, model_name: str) -> YoloDetector:
    return YoloDetector(
        DetectorConfig(
            model_name=model_name,
            models_dir=models_dir,
            device="cpu",
            conf_threshold=conf_threshold,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO detection on a local clip.")
    parser.add_argument("--video", required=True, help="Path to a local video file.")
    parser.add_argument("--stride", type=int, default=None, help="Sample every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=5, help="Max sampled frames to process.")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold.")
    parser.add_argument("--save", action="store_true", help="Save annotated frames to outputs/.")
    args = parser.parse_args()

    cfg = load_config()
    video_path = Path(args.video)
    stride = args.stride if args.stride is not None else cfg.video.frame_sample_stride
    conf = args.conf if args.conf is not None else cfg.detection.conf_threshold

    meta = probe_video(video_path)
    print(f"Video: {meta.path}")
    print(f"  {meta.width}x{meta.height} @ {meta.fps:.2f} fps, "
          f"{meta.frame_count} frames (~{meta.duration_seconds:.1f}s)")

    detector = build_detector(conf, cfg.paths.models, cfg.detection.model_name)

    out_dir = cfg.paths.outputs
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    import time

    total_players = 0
    total_balls = 0
    processed = 0
    for frame_index, frame in iter_sampled_frames(
        video_path, stride=stride, max_width=cfg.video.max_width, max_frames=args.max_frames
    ):
        t0 = time.perf_counter()
        detections = detector.detect(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        players, balls = split_by_class(detections)
        total_players += len(players)
        total_balls += len(balls)
        processed += 1

        print(
            f"  frame {frame_index:5d}: {len(players)} players, "
            f"{len(balls)} ball(s)  [{elapsed_ms:.0f} ms]"
        )

        if args.save:
            annotated = draw_detections(frame, detections)
            out_path = out_dir / f"detect_frame_{frame_index:05d}.png"
            cv2.imwrite(str(out_path), annotated)

    print("\nSummary")
    print(f"  frames processed : {processed}")
    print(f"  total players    : {total_players}")
    print(f"  total ball(s)    : {total_balls} (best-effort; often unreliable)")
    if args.save:
        print(f"  annotated frames saved under: {out_dir}")


if __name__ == "__main__":
    main()
