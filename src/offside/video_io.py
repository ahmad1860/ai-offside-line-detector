"""CPU-friendly video input and processing.

This module is the foundation for every later milestone. It provides:

* :func:`probe_video` — open a clip, validate it, and read its metadata.
* :class:`VideoReader` — a context manager that guarantees the underlying
  OpenCV ``VideoCapture`` is released, even on error.
* :func:`iter_sampled_frames` — a lazy generator that yields every Nth frame,
  optionally resized so its width does not exceed a configured maximum.
* :func:`resize_to_max_width` — the resize helper used throughout.

Design notes (why it is shaped this way):

* **Memory discipline.** Frames are yielded one at a time by a generator rather
  than collected into a list, so RAM usage stays bounded on an 8 GB machine.
* **Guaranteed cleanup.** The reader is a context manager; the capture is always
  released. Leaked captures can lock files and exhaust handles.
* **No model/geometry imports.** This layer is pure I/O and is unit-testable
  with a tiny synthetic clip (no copyrighted footage required).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

# A frame is a standard OpenCV BGR image: shape (H, W, 3), dtype uint8.
Frame = np.ndarray


class VideoReadError(RuntimeError):
    """Raised when a video cannot be opened or read."""


@dataclass(frozen=True)
class VideoMetadata:
    """Basic metadata describing a video clip.

    Attributes:
        path: Absolute path to the video file.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second reported by the container (may be approximate).
        frame_count: Number of frames reported by the container. Some codecs
            report this inaccurately, so treat it as a hint, not a guarantee.
        duration_seconds: ``frame_count / fps`` when both are known, else 0.0.
    """

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


def _open_capture(path: Path) -> cv2.VideoCapture:
    """Open a :class:`cv2.VideoCapture` or raise :class:`VideoReadError`."""
    if not path.is_file():
        raise VideoReadError(f"Video file not found: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoReadError(
            f"Could not open video (unsupported codec or corrupt file): {path}"
        )
    return capture


def probe_video(path: Path | str) -> VideoMetadata:
    """Open a clip briefly, read its metadata, and release it.

    Args:
        path: Path to a video file.

    Returns:
        A populated :class:`VideoMetadata`.

    Raises:
        VideoReadError: If the file is missing or cannot be opened.
    """
    resolved = Path(path).resolve()
    capture = _open_capture(resolved)
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()

    # Guard against nonsensical values some codecs report.
    fps = fps if fps and fps > 0 else 0.0
    frame_count = frame_count if frame_count > 0 else 0
    duration = (frame_count / fps) if (fps > 0 and frame_count > 0) else 0.0

    return VideoMetadata(
        path=resolved,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration,
    )


class VideoReader:
    """Context manager around a :class:`cv2.VideoCapture`.

    Guarantees the capture is released on exit, including when an exception
    propagates out of the ``with`` block.

    Example:
        >>> with VideoReader(path) as reader:
        ...     for frame in reader.frames():
        ...         ...  # process one frame at a time
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path).resolve()
        self._capture: cv2.VideoCapture | None = None

    def __enter__(self) -> "VideoReader":
        self._capture = _open_capture(self._path)
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    def release(self) -> None:
        """Release the underlying capture if it is open. Safe to call twice."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @property
    def capture(self) -> cv2.VideoCapture:
        if self._capture is None:
            raise VideoReadError("VideoReader used outside of its context manager.")
        return self._capture

    def frames(self) -> Iterator[Frame]:
        """Yield frames one at a time until the stream ends."""
        capture = self.capture
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            yield frame


def resize_to_max_width(frame: Frame, max_width: int) -> Frame:
    """Resize a frame so its width is at most ``max_width``, preserving aspect.

    Frames narrower than ``max_width`` are returned unchanged (we never upscale,
    which would waste CPU and add no information).

    Args:
        frame: BGR image of shape (H, W, 3).
        max_width: Maximum allowed width in pixels. Must be positive.

    Returns:
        The resized frame, or the original if it already fits.
    """
    if max_width <= 0:
        raise ValueError(f"max_width must be positive, got {max_width}")
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    new_size = (max_width, max(1, int(round(height * scale))))
    # INTER_AREA is the recommended interpolation for downscaling.
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def iter_sampled_frames(
    path: Path | str,
    stride: int = 1,
    max_width: int | None = None,
    max_frames: int | None = None,
) -> Iterator[tuple[int, Frame]]:
    """Lazily yield sampled, optionally resized frames from a clip.

    Only every ``stride``-th frame is yielded (frame 0, ``stride``,
    ``2*stride``, ...), which keeps CPU cost down on long clips. Frames are
    read one at a time, so memory stays bounded regardless of clip length.

    Args:
        path: Path to a video file.
        stride: Sample every Nth frame. Must be >= 1.
        max_width: If given, resize each yielded frame so its width is at most
            this value (see :func:`resize_to_max_width`).
        max_frames: If given, stop after yielding this many sampled frames.

    Yields:
        ``(original_frame_index, frame)`` pairs, where ``original_frame_index``
        is the frame's index in the source video (not the sample count).

    Raises:
        ValueError: If ``stride`` < 1.
        VideoReadError: If the file is missing or cannot be opened.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    yielded = 0
    with VideoReader(path) as reader:
        for index, frame in enumerate(reader.frames()):
            if index % stride != 0:
                continue
            if max_width is not None:
                frame = resize_to_max_width(frame, max_width)
            yield index, frame
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break


def read_frame(
    path: Path | str, index: int, max_width: int | None = None
) -> Frame:
    """Read a single frame by its index, optionally resized.

    Used to grab one selected "pass-moment" frame. Seeks with
    ``CAP_PROP_POS_FRAMES`` and verifies the read, since some codecs seek
    imprecisely. The capture is always released.

    Args:
        path: Path to a video file.
        index: Zero-based frame index to read.
        max_width: If given, resize the frame so its width is at most this value.

    Returns:
        The requested BGR frame (optionally resized).

    Raises:
        ValueError: If ``index`` is negative.
        VideoReadError: If the file cannot be opened or the frame cannot be read.
    """
    if index < 0:
        raise ValueError(f"Frame index must be >= 0, got {index}.")
    with VideoReader(path) as reader:
        capture = reader.capture
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(index))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise VideoReadError(
                f"Could not read frame {index} from {Path(path)} "
                "(index beyond end of clip, or codec seek failure)."
            )
        if max_width is not None:
            frame = resize_to_max_width(frame, max_width)
        return frame


def frame_index_for_timestamp(path: Path | str, seconds: float) -> int:
    """Convert a timestamp in seconds to a frame index using the clip's fps.

    Raises:
        VideoReadError: If the clip cannot be opened or reports no usable fps.
        ValueError: If ``seconds`` is negative.
    """
    if seconds < 0:
        raise ValueError(f"Timestamp must be >= 0, got {seconds}.")
    meta = probe_video(path)
    if meta.fps <= 0:
        raise VideoReadError(
            f"Clip reports no usable fps; cannot convert timestamp for {Path(path)}."
        )
    return int(round(seconds * meta.fps))
