"""Tests for the configuration loader.

These are deterministic and require no ML stack or GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from offside.config import AppConfig, load_config, project_root


def test_project_root_contains_config() -> None:
    root = project_root()
    assert (root / "config" / "default.yaml").is_file()


def test_load_default_config_types() -> None:
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    # Paths are resolved to absolute paths under the project root.
    root = project_root()
    for p in (cfg.paths.data_raw, cfg.paths.samples, cfg.paths.models, cfg.paths.outputs):
        assert isinstance(p, Path)
        assert p.is_absolute()
        assert str(p).startswith(str(root))


def test_default_values_match_file() -> None:
    cfg = load_config()
    assert cfg.video.frame_sample_stride == 5
    assert cfg.video.max_width == 960
    assert cfg.video.max_clip_seconds == 15
    assert cfg.detection.model_name == "yolo11n.pt"
    assert cfg.detection.device == "cpu"
    assert cfg.offside.require_ball is True


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist.yaml")
