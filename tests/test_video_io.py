"""Tests for the video I/O layer.

These generate a tiny synthetic clip at runtime (no copyrighted footage) and
verify metadata extraction, frame sampling, resizing, and resource release.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from offside.video_io import (
    VideoReader,
    VideoReadError,
    frame_index_for_timestamp,
    iter_sampled_frames,
    probe_video,
    read_frame,
    resize_to_max_width,
)

# Properties of the synthetic clip used across tests.
_W, _H, _FPS, _N = 160, 120, 10, 20


def _make_clip(path: Path, width: int = _W, height: int = _H, fps: int = _FPS, n: int = _N) -> None:
    """Write a tiny clip of ``n`` frames, each a distinct solid grey level."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), "Could not open VideoWriter (codec unavailable?)"
    try:
        for i in range(n):
            level = int(255 * i / max(1, n - 1))
            frame = np.full((height, width, 3), level, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    path = tmp_path / "sample.mp4"
    _make_clip(path)
    assert path.is_file()
    return path


def test_probe_metadata(clip: Path) -> None:
    meta = probe_video(clip)
    assert meta.width == _W
    assert meta.height == _H
    assert meta.fps > 0
    # Frame count can be off by a frame depending on the codec; allow slack.
    assert abs(meta.frame_count - _N) <= 1
    assert meta.duration_seconds > 0


def test_probe_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(VideoReadError):
        probe_video(tmp_path / "does_not_exist.mp4")


def test_reader_yields_frames_and_releases(clip: Path) -> None:
    with VideoReader(clip) as reader:
        count = sum(1 for _ in reader.frames())
        assert count >= _N - 1
    # After the context exits the capture must be released.
    with pytest.raises(VideoReadError):
        _ = reader.capture


def test_sampling_stride(clip: Path) -> None:
    all_indices = [idx for idx, _ in iter_sampled_frames(clip, stride=1)]
    strided = [idx for idx, _ in iter_sampled_frames(clip, stride=5)]
    # Strided indices are a subset, start at 0, and are multiples of the stride.
    assert strided[0] == 0
    assert all(i % 5 == 0 for i in strided)
    assert len(strided) < len(all_indices)


def test_max_frames_limit(clip: Path) -> None:
    frames = list(iter_sampled_frames(clip, stride=1, max_frames=3))
    assert len(frames) == 3


def test_resize_downscales_only() -> None:
    big = np.zeros((200, 400, 3), dtype=np.uint8)
    out = resize_to_max_width(big, max_width=100)
    assert out.shape[1] == 100
    assert out.shape[0] == 50  # aspect ratio preserved (200 * 100/400)

    small = np.zeros((20, 40, 3), dtype=np.uint8)
    unchanged = resize_to_max_width(small, max_width=100)
    assert unchanged.shape == small.shape  # never upscales


def test_resize_rejects_bad_width() -> None:
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        resize_to_max_width(frame, max_width=0)


def test_iter_rejects_bad_stride(clip: Path) -> None:
    with pytest.raises(ValueError):
        list(iter_sampled_frames(clip, stride=0))


def test_sampled_frames_respect_max_width(clip: Path) -> None:
    # Source is 160 wide; cap at 80 and confirm every yielded frame obeys it.
    for _, frame in iter_sampled_frames(clip, stride=5, max_width=80):
        assert frame.shape[1] <= 80


def test_read_frame_returns_requested_frame(clip: Path) -> None:
    frame = read_frame(clip, index=2)
    assert frame.shape[0] == _H and frame.shape[1] == _W
    assert frame.ndim == 3 and frame.shape[2] == 3


def test_read_frame_resizes(clip: Path) -> None:
    frame = read_frame(clip, index=0, max_width=80)
    assert frame.shape[1] <= 80


def test_read_frame_negative_index_rejected(clip: Path) -> None:
    with pytest.raises(ValueError):
        read_frame(clip, index=-1)


def test_read_frame_beyond_end_raises(clip: Path) -> None:
    with pytest.raises(VideoReadError):
        read_frame(clip, index=10_000)


def test_frame_index_for_timestamp(clip: Path) -> None:
    # fps is 10; 1.0s -> frame 10.
    assert frame_index_for_timestamp(clip, 1.0) == 10


def test_frame_index_for_timestamp_negative_rejected(clip: Path) -> None:
    with pytest.raises(ValueError):
        frame_index_for_timestamp(clip, -0.5)
