"""Unit tests for the detection module.

These do NOT download weights and do NOT require internet. The YOLO model is
mocked; the model-independent logic (parsing, filtering, validation, drawing)
is tested directly. A real-inference check is kept separate in
``scripts/test_detection.py`` and is not part of the automated suite.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from offside.detection import (
    COCO_PERSON,
    COCO_SPORTS_BALL,
    BoundingBox,
    Detection,
    DetectionError,
    DetectorConfig,
    ObjectClass,
    YoloDetector,
    draw_detections,
    parse_ultralytics_result,
    split_by_class,
)


def _frame(h: int = 48, w: int = 64) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# --- BoundingBox -------------------------------------------------------------

def test_bbox_properties() -> None:
    box = BoundingBox(10, 20, 30, 60)
    assert box.width == 20
    assert box.height == 40
    assert box.center == (20.0, 40.0)
    assert box.foot_point == (20.0, 60.0)  # bottom-centre


def test_bbox_rejects_inverted_coordinates() -> None:
    with pytest.raises(DetectionError):
        BoundingBox(30, 20, 10, 60)  # x2 < x1
    with pytest.raises(DetectionError):
        BoundingBox(10, 60, 30, 20)  # y2 < y1


# --- parse_ultralytics_result ------------------------------------------------

def _raw(boxes, confs, classes):
    return (
        np.array(boxes, dtype=float).reshape(-1, 4),
        np.array(confs, dtype=float),
        np.array(classes, dtype=float),
    )


def test_parse_maps_classes_and_structure() -> None:
    xyxy, conf, cls = _raw(
        [[0, 0, 10, 20], [5, 5, 8, 9]],
        [0.9, 0.8],
        [COCO_PERSON, COCO_SPORTS_BALL],
    )
    dets = parse_ultralytics_result(xyxy, conf, cls, conf_threshold=0.25)
    assert len(dets) == 2
    assert all(isinstance(d, Detection) for d in dets)
    assert dets[0].object_class is ObjectClass.PLAYER
    assert dets[1].object_class is ObjectClass.BALL
    assert dets[0].coco_class_id == COCO_PERSON


def test_parse_confidence_filtering() -> None:
    xyxy, conf, cls = _raw([[0, 0, 1, 1], [0, 0, 1, 1]], [0.9, 0.10], [0, 0])
    dets = parse_ultralytics_result(xyxy, conf, cls, conf_threshold=0.5)
    assert len(dets) == 1
    assert dets[0].confidence == pytest.approx(0.9)


def test_parse_class_filtering() -> None:
    xyxy, conf, cls = _raw(
        [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
        [0.9, 0.9, 0.9],
        [COCO_PERSON, COCO_SPORTS_BALL, 15],  # 15 = cat (should be dropped)
    )
    dets = parse_ultralytics_result(
        xyxy, conf, cls, conf_threshold=0.25, keep_classes={COCO_PERSON, COCO_SPORTS_BALL}
    )
    classes = {d.object_class for d in dets}
    assert classes == {ObjectClass.PLAYER, ObjectClass.BALL}
    assert len(dets) == 2


def test_parse_empty_input() -> None:
    xyxy = np.empty((0, 4))
    conf = np.empty((0,))
    cls = np.empty((0,))
    assert parse_ultralytics_result(xyxy, conf, cls, conf_threshold=0.25) == []


def test_split_by_class() -> None:
    xyxy, conf, cls = _raw(
        [[0, 0, 1, 1], [0, 0, 1, 1]], [0.9, 0.9], [COCO_PERSON, COCO_SPORTS_BALL]
    )
    dets = parse_ultralytics_result(xyxy, conf, cls, conf_threshold=0.25)
    players, balls = split_by_class(dets)
    assert len(players) == 1
    assert len(balls) == 1


# --- frame validation --------------------------------------------------------

def test_detect_rejects_non_array() -> None:
    det = YoloDetector()
    with pytest.raises(DetectionError):
        det.detect("not a frame")  # type: ignore[arg-type]


def test_detect_rejects_wrong_shape() -> None:
    det = YoloDetector()
    with pytest.raises(DetectionError):
        det.detect(np.zeros((10, 10), dtype=np.uint8))  # missing channel dim


def test_detect_rejects_empty_frame() -> None:
    det = YoloDetector()
    with pytest.raises(DetectionError):
        det.detect(np.zeros((0, 0, 3), dtype=np.uint8))


# --- mocked inference path ---------------------------------------------------

class _FakeTensor:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _FakeBoxes:
    def __init__(self, xyxy, conf, cls) -> None:
        self.xyxy = _FakeTensor(np.array(xyxy, dtype=float).reshape(-1, 4))
        self.conf = _FakeTensor(np.array(conf, dtype=float))
        self.cls = _FakeTensor(np.array(cls, dtype=float))

    def __len__(self) -> int:
        return len(self.conf.numpy())


class _FakeModel:
    def __init__(self, boxes: _FakeBoxes) -> None:
        self._boxes = boxes
        self.calls: list[dict] = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(boxes=self._boxes)]


def test_detector_uses_cpu_and_returns_structured_results() -> None:
    fake = _FakeModel(
        _FakeBoxes(
            xyxy=[[0, 0, 10, 20], [1, 1, 2, 2]],
            conf=[0.95, 0.10],
            cls=[COCO_PERSON, COCO_PERSON],
        )
    )
    det = YoloDetector(DetectorConfig(conf_threshold=0.5, device="cpu"))
    det._model = fake  # inject mock; no weights, no download

    results = det.detect(_frame())
    assert len(results) == 1  # low-confidence one filtered out
    assert results[0].object_class is ObjectClass.PLAYER
    # Confirm we asked Ultralytics for CPU inference.
    assert fake.calls[0]["device"] == "cpu"


def test_detector_handles_no_boxes() -> None:
    det = YoloDetector()
    det._model = _FakeModel(_FakeBoxes(xyxy=[], conf=[], cls=[]))  # type: ignore[arg-type]
    assert det.detect(_frame()) == []


# --- drawing -----------------------------------------------------------------

def test_draw_detections_does_not_mutate_input() -> None:
    frame = _frame()
    original = frame.copy()
    dets = [
        Detection(BoundingBox(1, 1, 10, 10), 0.9, COCO_PERSON, ObjectClass.PLAYER),
    ]
    out = draw_detections(frame, dets)
    assert out.shape == frame.shape
    assert np.array_equal(frame, original)  # input untouched
    assert not np.array_equal(out, original)  # something was drawn
