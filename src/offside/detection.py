"""Lightweight, CPU-first object detection for the offside detector.

This module wraps a pretrained Ultralytics YOLO model (YOLO11n by default) behind
a small, typed interface. Downstream modules consume the structured results
(:class:`Detection`, :class:`BoundingBox`) and never import Ultralytics directly,
so the detector can be swapped later without touching the rest of the pipeline.

Design notes:

* **CPU-first.** Inference runs on ``device="cpu"`` by default. No CUDA assumed.
* **Lazy weights.** The model is loaded on first use, not at import time, and
  weights are only downloaded when real inference actually runs. Unit tests mock
  the model so they need no weights and no internet.
* **Player-primary, ball best-effort.** Person detection (COCO class 0) is the
  reliable target. Ball detection (COCO "sports ball", class 32) is attempted but
  explicitly marked best-effort — broadcast footage makes the ball hard to detect.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:  # avoid importing ultralytics at module import time
    from ultralytics import YOLO

# COCO class indices used by the stock YOLO models.
COCO_PERSON = 0
COCO_SPORTS_BALL = 32

Frame = np.ndarray


class DetectionError(RuntimeError):
    """Raised when detection input or model loading is invalid."""


class ObjectClass(str, Enum):
    """Semantic classes this project cares about."""

    PLAYER = "player"  # mapped from COCO "person"
    BALL = "ball"      # mapped from COCO "sports ball" (best-effort)
    OTHER = "other"    # any other detected class, retained but unlabeled


_COCO_TO_CLASS = {COCO_PERSON: ObjectClass.PLAYER, COCO_SPORTS_BALL: ObjectClass.BALL}


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates (top-left origin).

    Invariant: ``x1 <= x2`` and ``y1 <= y2``.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise DetectionError(
                f"Invalid bounding box: ({self.x1}, {self.y1}, {self.x2}, {self.y2})"
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-centre point — the baseline ground-contact estimate.

        This is the point projected onto the pitch plane in later milestones.
        A future pose-based estimator can override this via the shared
        ground-contact interface (see SPEC §0.8).
        """
        return ((self.x1 + self.x2) / 2.0, self.y2)


@dataclass(frozen=True)
class Detection:
    """A single detected object."""

    box: BoundingBox
    confidence: float
    coco_class_id: int
    object_class: ObjectClass


class Detector(Protocol):
    """Interface downstream code depends on (keeps Ultralytics out of the API)."""

    def detect(self, frame: Frame) -> list[Detection]:
        ...


def _validate_frame(frame: Frame) -> None:
    if not isinstance(frame, np.ndarray):
        raise DetectionError(f"Frame must be a numpy array, got {type(frame)!r}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise DetectionError(
            f"Frame must have shape (H, W, 3); got shape {getattr(frame, 'shape', None)}"
        )
    if frame.size == 0:
        raise DetectionError("Frame is empty.")


def _classify(coco_id: int) -> ObjectClass:
    return _COCO_TO_CLASS.get(coco_id, ObjectClass.OTHER)


def parse_ultralytics_result(
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
    conf_threshold: float,
    keep_classes: set[int] | None = None,
) -> list[Detection]:
    """Convert raw Ultralytics arrays into typed :class:`Detection` objects.

    Separated from the model so it can be unit-tested without any model or
    weights. This is where confidence and class filtering happen.

    Args:
        boxes_xyxy: Array of shape (N, 4) with [x1, y1, x2, y2] per detection.
        confidences: Array of shape (N,) with confidence scores.
        class_ids: Array of shape (N,) with integer COCO class ids.
        conf_threshold: Minimum confidence to keep.
        keep_classes: If given, only keep detections whose COCO class id is in
            this set (e.g. ``{0, 32}`` for players + ball).

    Returns:
        A list of detections passing the filters, in input order.
    """
    detections: list[Detection] = []
    for box, conf, cls in zip(boxes_xyxy, confidences, class_ids, strict=True):
        conf_f = float(conf)
        cls_i = int(cls)
        if conf_f < conf_threshold:
            continue
        if keep_classes is not None and cls_i not in keep_classes:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        detections.append(
            Detection(
                box=BoundingBox(x1, y1, x2, y2),
                confidence=conf_f,
                coco_class_id=cls_i,
                object_class=_classify(cls_i),
            )
        )
    return detections


@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for :class:`YoloDetector`."""

    model_name: str = "yolo11n.pt"
    models_dir: Path | None = None
    device: str = "cpu"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    detect_players: bool = True
    detect_ball: bool = True  # best-effort
    imgsz: int = 640  # inference size; smaller is faster/lighter on CPU


class YoloDetector:
    """CPU-first YOLO detector with lazy weight loading.

    The underlying Ultralytics model is created on first :meth:`detect` call.
    Weights resolve to ``models_dir / model_name`` if that file exists; otherwise
    Ultralytics will download the named model on first use.
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self._config = config or DetectorConfig()
        self._model: YOLO | None = None

    @property
    def config(self) -> DetectorConfig:
        return self._config

    def _keep_classes(self) -> set[int] | None:
        keep: set[int] = set()
        if self._config.detect_players:
            keep.add(COCO_PERSON)
        if self._config.detect_ball:
            keep.add(COCO_SPORTS_BALL)
        return keep or None

    def _resolve_weights(self) -> str:
        """Return a local weights path if present, else the bare model name.

        Returning the bare name lets Ultralytics fetch the pretrained weights on
        first use (never during import or unit tests).
        """
        models_dir = self._config.models_dir
        if models_dir is not None:
            local = Path(models_dir) / self._config.model_name
            if local.is_file():
                return str(local)
        return self._config.model_name

    def load(self) -> None:
        """Eagerly load the model. Useful to surface load errors up front."""
        if self._model is not None:
            return
        from ultralytics import YOLO  # local import keeps module import light

        self._model = YOLO(self._resolve_weights())

    def detect(self, frame: Frame) -> list[Detection]:
        """Run inference on a single frame and return typed detections.

        Args:
            frame: BGR image of shape (H, W, 3).

        Returns:
            Filtered detections (by confidence and selected classes).

        Raises:
            DetectionError: If the frame is invalid.
        """
        _validate_frame(frame)
        if self._model is None:
            self.load()
        assert self._model is not None

        results = self._model.predict(
            source=frame,
            device=self._config.device,
            conf=self._config.conf_threshold,
            iou=self._config.iou_threshold,
            imgsz=self._config.imgsz,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        return parse_ultralytics_result(
            boxes_xyxy=xyxy,
            confidences=conf,
            class_ids=cls,
            conf_threshold=self._config.conf_threshold,
            keep_classes=self._keep_classes(),
        )


def split_by_class(
    detections: list[Detection],
) -> tuple[list[Detection], list[Detection]]:
    """Split detections into (players, balls). Balls are best-effort."""
    players = [d for d in detections if d.object_class is ObjectClass.PLAYER]
    balls = [d for d in detections if d.object_class is ObjectClass.BALL]
    return players, balls


def draw_detections(frame: Frame, detections: list[Detection]) -> Frame:
    """Draw bounding boxes and confidence labels onto a copy of ``frame``.

    Players are drawn in green, balls in yellow, anything else in grey. Returns a
    new image; the input frame is not modified. (A fuller annotation layer with
    the offside line arrives in milestone M6.)
    """
    import cv2

    colours = {
        ObjectClass.PLAYER: (0, 200, 0),
        ObjectClass.BALL: (0, 215, 255),
        ObjectClass.OTHER: (160, 160, 160),
    }
    out = frame.copy()
    for det in detections:
        colour = colours[det.object_class]
        p1 = (int(det.box.x1), int(det.box.y1))
        p2 = (int(det.box.x2), int(det.box.y2))
        cv2.rectangle(out, p1, p2, colour, 2)
        label = f"{det.object_class.value} {det.confidence:.2f}"
        cv2.putText(
            out, label, (p1[0], max(0, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
        )
    return out
