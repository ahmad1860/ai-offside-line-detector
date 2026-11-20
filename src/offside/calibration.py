"""Manual pitch calibration → homography estimation.

The user identifies known pitch landmarks in an image (e.g. penalty-box corners,
halfway-line intersections) and provides their real-world pitch coordinates in
metres. From >= 4 such correspondences we estimate a 3x3 homography H that maps
image pixels to the top-down pitch plane.

Coordinate convention (see PROJECT_CONTEXT.md):
    * Image points: pixels (x, y), top-left origin.
    * Pitch points: metres (X, Y); X = length (0..105 default), Y = width (0..68 default).

Mathematical requirements (a general planar homography has 8 degrees of freedom):
    * At least 4 correspondences are required. Fewer is never accepted.
    * Those correspondences must contain 4 points in *general position*: no three
      of the four on (nearly) one straight line, in the image AND on the pitch.
      Four points with three on one goal line, for example, do not determine H.
    * With more than 4 points, RANSAC rejects gross outliers (mis-clicks). The
      inlier threshold is in IMAGE PIXELS: H is estimated pitch -> image, where
      the click noise actually lives, and then inverted.

Important caveats (never overstate calibration quality):
    * A calibration is valid ONLY for the camera view it was created in. Camera
      movement, zoom, or a significant change in viewing angle invalidates it and
      requires recalibration.
    * Reprojection error measures how well the points agree with each other, not
      how accurate the calibration is in the real world. With exactly 4 points it
      is ~0 by construction and carries no information.
    * Homography accuracy degrades away from the region covered by the chosen
      points; coverage warnings flag when that region is small.

This module depends only on NumPy and OpenCV (no Ultralytics / PyTorch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

import cv2
import numpy as np

# Standard full-size pitch defaults (metres). Configurable via PitchDimensions.
DEFAULT_PITCH_LENGTH_M = 105.0
DEFAULT_PITCH_WIDTH_M = 68.0

# Minimum correspondences for a planar homography.
MIN_CORRESPONDENCES = 4
# Upper bound for the exhaustive general-position search (C(n, 4) quadruples).
MAX_CORRESPONDENCES = 40

# --- Validation thresholds -------------------------------------------------
# These are engineering heuristics for flagging risky inputs. They are NOT
# accuracy guarantees, and passing them does not make a calibration accurate.
MIN_IMAGE_SEPARATION_PX = 1.0      # two clicks closer than this are duplicates
MIN_PITCH_SEPARATION_M = 0.05      # two landmarks closer than this are duplicates
COLLINEAR_ERROR_RATIO = 0.02       # best quadruple below this => cannot determine H
WEAK_CONDITIONING_RATIO = 0.10     # best quadruple below this => warn
MIN_PITCH_COVERAGE = 0.15          # convex-hull share of the pitch area
MIN_IMAGE_COVERAGE = 0.10          # convex-hull share of the frame area
WARN_MEAN_REPROJ_PX = 4.0
WARN_MAX_REPROJ_PX = 12.0
WARN_MEAN_REPROJ_M = 1.0
DEFAULT_RANSAC_THRESHOLD_PX = 5.0  # inlier threshold in IMAGE pixels
_MAX_HOMOGRAPHY_COND = 1e10


class CalibrationError(ValueError):
    """Raised when calibration input or the resulting homography is invalid."""


@dataclass(frozen=True)
class CalibrationWarning:
    """A non-blocking calibration concern. ``code`` is stable for programmatic use."""

    code: str
    message: str


@dataclass(frozen=True)
class PitchDimensions:
    """Top-down pitch reference dimensions in metres.

    X spans ``length`` (0..length), Y spans ``width`` (0..width).
    """

    length: float = DEFAULT_PITCH_LENGTH_M
    width: float = DEFAULT_PITCH_WIDTH_M

    def __post_init__(self) -> None:
        if not (self.length > 0 and self.width > 0):
            raise CalibrationError(
                f"Pitch dimensions must be positive, got {self.length} x {self.width}"
            )


@dataclass(frozen=True)
class Correspondences:
    """Paired image (pixels) and pitch (metres) points for the same landmarks.

    ``image_points`` and ``pitch_points`` are each (N, 2) float arrays with N >= 4
    and matching N. Index i in both refers to the same physical landmark.
    ``labels`` optionally names each landmark (used only in messages).
    """

    image_points: np.ndarray
    pitch_points: np.ndarray
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        img = np.asarray(self.image_points, dtype=np.float64)
        pit = np.asarray(self.pitch_points, dtype=np.float64)
        _validate_point_array(img, "image_points")
        _validate_point_array(pit, "pitch_points")
        if img.shape[0] != pit.shape[0]:
            raise CalibrationError(
                f"Mismatched correspondence counts: {img.shape[0]} image points vs "
                f"{pit.shape[0]} pitch points."
            )
        if img.shape[0] < MIN_CORRESPONDENCES:
            raise CalibrationError(
                f"Need at least {MIN_CORRESPONDENCES} correspondences, got {img.shape[0]}."
            )
        if self.labels is not None and len(self.labels) != img.shape[0]:
            raise CalibrationError(
                f"Got {len(self.labels)} labels for {img.shape[0]} correspondences."
            )
        # Normalize stored arrays to float64 (frozen dataclass: use object.__setattr__).
        object.__setattr__(self, "image_points", img)
        object.__setattr__(self, "pitch_points", pit)
        if self.labels is not None:
            object.__setattr__(self, "labels", tuple(self.labels))

    @property
    def count(self) -> int:
        return int(self.image_points.shape[0])


@dataclass(frozen=True)
class Calibration:
    """Result of a successful calibration.

    Attributes:
        homography: 3x3 matrix mapping image pixels -> pitch metres.
        homography_inv: 3x3 inverse mapping pitch metres -> image pixels.
        pitch: the pitch dimensions used.
        reprojection_error_px: mean Euclidean error (pixels) when pitch points are
            mapped into the image and compared to the clicked image points,
            over the inliers (all points when no RANSAC was used).
        reprojection_error_m: mean Euclidean error (metres) when image points are
            mapped to the pitch and compared to the supplied pitch points
            (inliers only).
        num_points: number of correspondences supplied.
        inliers: optional boolean mask (length num_points) from RANSAC; None when
            an exact 4-point solve was used.
        max_reprojection_error_px: worst inlier error in pixels.
        point_errors_px: per-correspondence pixel residual for ALL supplied
            points (including any rejected outliers), in input order.
        pitch_coverage / image_coverage: convex-hull share of the pitch area /
            frame area covered by the inlier points (image_coverage is None when
            the image size was not supplied).
        conditioning: quality (0..~0.5) of the best four-point subset; higher is
            better. Low values mean the points are nearly collinear.
        warnings: non-blocking concerns (see :class:`CalibrationWarning`).
        labels: landmark names, if supplied.
    """

    homography: np.ndarray
    homography_inv: np.ndarray
    pitch: PitchDimensions
    reprojection_error_px: float
    reprojection_error_m: float
    num_points: int
    inliers: np.ndarray | None = field(default=None)
    max_reprojection_error_px: float = 0.0
    point_errors_px: tuple[float, ...] | None = None
    pitch_coverage: float | None = None
    image_coverage: float | None = None
    conditioning: float | None = None
    warnings: tuple[CalibrationWarning, ...] = ()
    labels: tuple[str, ...] | None = None

    @property
    def num_inliers(self) -> int | None:
        return None if self.inliers is None else int(np.count_nonzero(self.inliers))

    @property
    def warning_codes(self) -> tuple[str, ...]:
        return tuple(w.code for w in self.warnings)

    @property
    def outlier_indices(self) -> tuple[int, ...]:
        if self.inliers is None:
            return ()
        return tuple(int(i) for i in np.flatnonzero(~self.inliers))


@dataclass(frozen=True)
class PointSetAssessment:
    """Geometry-only assessment of a set of correspondences (no estimation).

    ``errors`` block calibration; ``warnings`` do not but must be shown.
    """

    count: int
    errors: tuple[str, ...] = ()
    warnings: tuple[CalibrationWarning, ...] = ()
    conditioning: float | None = None
    pitch_coverage: float | None = None
    image_coverage: float | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def _validate_point_array(points: np.ndarray, name: str) -> None:
    if points.ndim != 2 or points.shape[1] != 2:
        raise CalibrationError(f"{name} must have shape (N, 2), got {points.shape}.")
    if not np.all(np.isfinite(points)):
        raise CalibrationError(f"{name} contains non-finite values (NaN/Inf).")


def _triple_ratio(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Height of triangle abc over its longest side. 0 means collinear.

    ``2 * area / longest_side**2``: scale-invariant, so one threshold works for
    pixels and metres alike.
    """
    ab, ac, bc = b - a, c - a, c - b
    twice_area = abs(float(ab[0] * ac[1] - ab[1] * ac[0]))
    longest = max(float(np.hypot(*ab)), float(np.hypot(*ac)), float(np.hypot(*bc)))
    if longest <= 0.0:
        return 0.0
    return twice_area / (longest * longest)


def _best_general_position(img: np.ndarray, pit: np.ndarray) -> float:
    """Best achievable quality over all 4-point subsets.

    The quality of a subset is the smallest triple ratio among its four triples,
    taking the worse of the image and pitch values (a subset must be in general
    position in BOTH spaces). Returns 0.0 when no subset is non-collinear.
    """
    n = img.shape[0]
    triple: dict[tuple[int, int, int], float] = {}
    for t in combinations(range(n), 3):
        a, b, c = t
        triple[t] = min(
            _triple_ratio(img[a], img[b], img[c]),
            _triple_ratio(pit[a], pit[b], pit[c]),
        )
    best = 0.0
    for quad in combinations(range(n), 4):
        q = min(triple[t] for t in combinations(quad, 3))
        if q > best:
            best = q
    return best


def _hull_area(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull))


def assess_point_set(
    image_points: np.ndarray | Sequence[Sequence[float]],
    pitch_points: np.ndarray | Sequence[Sequence[float]],
    *,
    pitch: PitchDimensions | None = None,
    image_size: tuple[int, int] | None = None,
    labels: Sequence[str] | None = None,
) -> PointSetAssessment:
    """Check point count, duplicates, general position and spatial spread.

    Pure geometry: nothing is estimated. :func:`calibrate` uses this as its
    single source of validation, and the UI calls it for live feedback while the
    user is still choosing landmarks.

    Args:
        image_points: (N, 2) clicked pixels.
        pitch_points: (N, 2) matching pitch coordinates in metres.
        pitch: pitch dimensions, used for the pitch-coverage check.
        image_size: (width, height) of the frame, used for the image-coverage check.
        labels: optional landmark names for readable messages.
    """
    pitch = pitch or PitchDimensions()
    try:
        img = np.asarray(image_points, dtype=np.float64)
        pit = np.asarray(pitch_points, dtype=np.float64)
        if img.size == 0:
            img = img.reshape(0, 2)
        if pit.size == 0:
            pit = pit.reshape(0, 2)
        _validate_point_array(img, "image_points")
        _validate_point_array(pit, "pitch_points")
    except (CalibrationError, ValueError) as exc:
        return PointSetAssessment(count=0, errors=(str(exc),))

    n = int(img.shape[0])
    if n != pit.shape[0]:
        return PointSetAssessment(
            count=n,
            errors=(f"Mismatched counts: {n} image points vs {pit.shape[0]} pitch points.",),
        )
    if n < MIN_CORRESPONDENCES:
        return PointSetAssessment(
            count=n,
            errors=(
                f"Need at least {MIN_CORRESPONDENCES} correspondences for a homography, "
                f"have {n}.",
            ),
        )
    if n > MAX_CORRESPONDENCES:
        return PointSetAssessment(
            count=n,
            errors=(f"Too many correspondences ({n}); the limit is {MAX_CORRESPONDENCES}.",),
        )

    def name(i: int) -> str:
        return str(labels[i]) if labels is not None else f"#{i + 1}"

    errors: list[str] = []
    for i, j in combinations(range(n), 2):
        if float(np.hypot(*(img[i] - img[j]))) < MIN_IMAGE_SEPARATION_PX:
            errors.append(f"{name(i)} and {name(j)} are at the same image location.")
        if float(np.hypot(*(pit[i] - pit[j]))) < MIN_PITCH_SEPARATION_M:
            errors.append(f"{name(i)} and {name(j)} are the same pitch location.")
    if errors:
        return PointSetAssessment(count=n, errors=tuple(errors[:4]))

    conditioning = _best_general_position(img, pit)
    warnings: list[CalibrationWarning] = []
    if conditioning < COLLINEAR_ERROR_RATIO:
        return PointSetAssessment(
            count=n,
            conditioning=conditioning,
            errors=(
                "These landmarks cannot determine a homography: every group of four "
                "has three or more points on (nearly) one straight line, in the image "
                "or on the pitch (for example several points on the same goal line or "
                "touchline). Add landmarks that are NOT on that line.",
            ),
        )
    if conditioning < WEAK_CONDITIONING_RATIO:
        warnings.append(
            CalibrationWarning(
                "weak_conditioning",
                "The landmarks are close to lying on one line, so the calibration is "
                "sensitive to small click errors. Add landmarks further from that line.",
            )
        )

    pitch_cov = min(1.0, _hull_area(pit) / (pitch.length * pitch.width))
    if pitch_cov < MIN_PITCH_COVERAGE:
        warnings.append(
            CalibrationWarning(
                "poor_pitch_coverage",
                f"The landmarks cover only {pitch_cov:.0%} of the pitch area. Positions "
                "outside that region are extrapolated and may be inaccurate. Prefer "
                "landmarks spread across the pitch (including near where the players are).",
            )
        )
    image_cov: float | None = None
    if image_size is not None:
        w, h = image_size
        if w > 0 and h > 0:
            image_cov = min(1.0, _hull_area(img) / float(w * h))
            if image_cov < MIN_IMAGE_COVERAGE:
                warnings.append(
                    CalibrationWarning(
                        "poor_image_coverage",
                        f"The clicked points cover only {image_cov:.0%} of the frame, so "
                        "click errors are magnified. Spread the landmarks across the image.",
                    )
                )
    return PointSetAssessment(
        count=n,
        warnings=tuple(warnings),
        conditioning=conditioning,
        pitch_coverage=pitch_cov,
        image_coverage=image_cov,
    )


def _project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)


def _normalise(matrix: np.ndarray) -> np.ndarray:
    scale = matrix[2, 2]
    return matrix / scale if abs(scale) > 1e-12 else matrix


def calibrate(
    correspondences: Correspondences,
    pitch: PitchDimensions | None = None,
    ransac_reproj_threshold: float = DEFAULT_RANSAC_THRESHOLD_PX,
    image_size: tuple[int, int] | None = None,
) -> Calibration:
    """Estimate the image→pitch homography from landmark correspondences.

    Uses an exact solve for exactly 4 points and RANSAC for more than 4 (so a
    stray mis-clicked landmark can be rejected as an outlier).

    Args:
        correspondences: image↔pitch point pairs (>= 4).
        pitch: pitch dimensions; defaults to 105 x 68 m.
        ransac_reproj_threshold: RANSAC inlier threshold in IMAGE PIXELS (used
            only when more than 4 correspondences are supplied).
        image_size: optional (width, height) of the frame, enabling the
            image-coverage check.

    Returns:
        A :class:`Calibration` with the homography, its inverse, reprojection
        errors, the RANSAC inlier mask, per-point residuals and any warnings.

    Raises:
        CalibrationError: for duplicate or collinear/degenerate points, too few
            inliers, or an ill-conditioned / non-invertible homography.
    """
    pitch = pitch or PitchDimensions()
    if not ransac_reproj_threshold > 0:
        raise CalibrationError("ransac_reproj_threshold must be positive.")
    img = correspondences.image_points.astype(np.float64)
    pit = correspondences.pitch_points.astype(np.float64)
    labels = correspondences.labels

    assessment = assess_point_set(
        img, pit, pitch=pitch, image_size=image_size, labels=labels
    )
    if not assessment.ok:
        raise CalibrationError(" ".join(assessment.errors))

    # Estimate pitch -> image so the RANSAC threshold is in pixels (where click
    # noise lives), then invert to get image -> pitch.
    if correspondences.count == MIN_CORRESPONDENCES:
        h_pi, _ = cv2.findHomography(pit, img, method=0)
        inliers = None
    else:
        h_pi, mask = cv2.findHomography(
            pit, img, method=cv2.RANSAC, ransacReprojThreshold=ransac_reproj_threshold
        )
        inliers = None if mask is None else mask.ravel().astype(bool)

    if h_pi is None:
        raise CalibrationError(
            "cv2.findHomography failed to estimate a homography from the given points."
        )
    if not np.all(np.isfinite(h_pi)):
        raise CalibrationError("Estimated homography contains non-finite values.")

    use = np.ones(correspondences.count, dtype=bool) if inliers is None else inliers
    final = assessment
    if inliers is not None:
        if int(use.sum()) < MIN_CORRESPONDENCES:
            raise CalibrationError(
                f"Only {int(use.sum())} of {correspondences.count} correspondences agree "
                f"within {ransac_reproj_threshold:g} px; at least {MIN_CORRESPONDENCES} are "
                "needed. Check for mis-clicked or mislabelled landmarks."
            )
        sub_labels = None if labels is None else [l for l, u in zip(labels, use) if u]
        final = assess_point_set(
            img[use], pit[use], pitch=pitch, image_size=image_size, labels=sub_labels
        )
        if not final.ok:
            raise CalibrationError(
                "After rejecting outliers, the remaining landmarks are not usable. "
                + " ".join(final.errors)
            )

    if float(np.linalg.cond(h_pi)) > _MAX_HOMOGRAPHY_COND:
        raise CalibrationError(
            "Estimated homography is singular or ill-conditioned; calibration is unreliable."
        )
    h_pi = _normalise(h_pi)
    homography = _normalise(np.linalg.inv(h_pi))
    if not np.all(np.isfinite(homography)):
        raise CalibrationError("Estimated homography is not invertible.")

    px_err = np.linalg.norm(_project(pit, h_pi) - img, axis=1)
    m_err = np.linalg.norm(_project(img, homography) - pit, axis=1)
    mean_px = float(np.mean(px_err[use]))
    max_px = float(np.max(px_err[use]))
    mean_m = float(np.mean(m_err[use]))

    warnings = list(final.warnings)
    if inliers is None:
        warnings.insert(
            0,
            CalibrationWarning(
                "exact_fit",
                "Exactly 4 landmarks give an exact fit, so the reprojection error is ~0 "
                "by construction and says nothing about accuracy. Add more landmarks to "
                "get a meaningful consistency check.",
            ),
        )
    else:
        rejected = [i for i in range(correspondences.count) if not use[i]]
        if rejected:
            names = ", ".join(labels[i] if labels else f"#{i + 1}" for i in rejected)
            warnings.insert(
                0,
                CalibrationWarning(
                    "outliers_rejected",
                    f"{len(rejected)} landmark(s) disagreed with the others by more than "
                    f"{ransac_reproj_threshold:g} px and were ignored: {names}. They are "
                    "probably mis-clicked or mislabelled; remove and re-place them.",
                ),
            )
    if mean_px > WARN_MEAN_REPROJ_PX or max_px > WARN_MAX_REPROJ_PX:
        warnings.append(
            CalibrationWarning(
                "high_reprojection_px",
                f"Landmarks disagree with the fit (mean {mean_px:.1f} px, worst "
                f"{max_px:.1f} px). Re-check the clicked locations and landmark choices.",
            )
        )
    if mean_m > WARN_MEAN_REPROJ_M:
        warnings.append(
            CalibrationWarning(
                "high_reprojection_m",
                f"On the pitch the landmarks disagree by {mean_m:.2f} m on average, which "
                "is large compared with typical offside margins.",
            )
        )

    return Calibration(
        homography=homography,
        homography_inv=h_pi,
        pitch=pitch,
        reprojection_error_px=mean_px,
        reprojection_error_m=mean_m,
        num_points=correspondences.count,
        inliers=inliers,
        max_reprojection_error_px=max_px,
        point_errors_px=tuple(float(v) for v in px_err),
        pitch_coverage=final.pitch_coverage,
        image_coverage=final.image_coverage,
        conditioning=final.conditioning,
        warnings=tuple(warnings),
        labels=None if labels is None else tuple(labels),
    )


def make_correspondences(
    image_points: Sequence[Sequence[float]],
    pitch_points: Sequence[Sequence[float]],
    labels: Sequence[str] | None = None,
) -> Correspondences:
    """Convenience builder from plain nested sequences."""
    return Correspondences(
        image_points=np.asarray(image_points, dtype=np.float64),
        pitch_points=np.asarray(pitch_points, dtype=np.float64),
        labels=None if labels is None else tuple(labels),
    )
