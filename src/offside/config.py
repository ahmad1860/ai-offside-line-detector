"""Typed configuration loading for the AI Offside Line Detector.

Loads ``config/default.yaml`` into frozen dataclasses. All paths are resolved
relative to the project root using :mod:`pathlib`, so the project behaves the
same regardless of the current working directory (important on Windows).

This module is intentionally pure (no video, model, or heavy imports) so it can
be imported and unit-tested without the runtime ML stack installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the project root directory.

    The root is two levels above this file: ``<root>/src/offside/config.py``.
    """
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Filesystem locations, resolved to absolute paths under the project root."""

    data_raw: Path
    samples: Path
    models: Path
    outputs: Path


@dataclass(frozen=True)
class VideoConfig:
    """Frame-sampling and resizing settings tuned for CPU / limited RAM."""

    frame_sample_stride: int
    max_width: int
    max_clip_seconds: int


@dataclass(frozen=True)
class DetectionConfig:
    """Detector settings. ``device`` stays on CPU by default (see SPEC §3)."""

    model_name: str
    device: str
    conf_threshold: float
    iou_threshold: float


@dataclass(frozen=True)
class OffsideConfig:
    """Offside-logic settings.

    ``require_ball`` enforces that a missing ball reference yields an
    UNCERTAIN / INSUFFICIENT_DATA result rather than a silent verdict
    (see SPEC §0.6).
    """

    require_ball: bool


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration aggregate."""

    paths: Paths
    video: VideoConfig
    detection: DetectionConfig
    offside: OffsideConfig


def _resolve(root: Path, relative: str) -> Path:
    """Resolve a config-relative path against the project root."""
    return (root / relative).resolve()


def load_config(config_path: Path | str | None = None) -> AppConfig:
    """Load and validate configuration from a YAML file.

    Args:
        config_path: Optional path to a YAML config. Defaults to
            ``<project_root>/config/default.yaml``.

    Returns:
        A fully populated, frozen :class:`AppConfig`.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If a required section or key is missing.
    """
    root = project_root()
    path = Path(config_path) if config_path is not None else root / "config" / "default.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    paths_raw = raw["paths"]
    paths = Paths(
        data_raw=_resolve(root, paths_raw["data_raw"]),
        samples=_resolve(root, paths_raw["samples"]),
        models=_resolve(root, paths_raw["models"]),
        outputs=_resolve(root, paths_raw["outputs"]),
    )

    video_raw = raw["video"]
    video = VideoConfig(
        frame_sample_stride=int(video_raw["frame_sample_stride"]),
        max_width=int(video_raw["max_width"]),
        max_clip_seconds=int(video_raw["max_clip_seconds"]),
    )

    detection_raw = raw["detection"]
    detection = DetectionConfig(
        model_name=str(detection_raw["model_name"]),
        device=str(detection_raw["device"]),
        conf_threshold=float(detection_raw["conf_threshold"]),
        iou_threshold=float(detection_raw["iou_threshold"]),
    )

    offside_raw = raw["offside"]
    offside = OffsideConfig(require_ball=bool(offside_raw["require_ball"]))

    return AppConfig(paths=paths, video=video, detection=detection, offside=offside)
