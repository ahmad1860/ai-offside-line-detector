# PROJECT_CONTEXT.md — Permanent Source of Truth

> This file (with `PROGRESS.md` and `DECISIONS.md`) is the project's durable memory.
> Read it before starting any milestone. Keep it synchronized with the actual code.
> Do not document unimplemented functionality.

## Project identity

- **Name:** AI Offside Line Detector.
- **Goal:** an experimental computer-vision system that estimates offside **positions** from
  single-camera football footage.
- **Nature:** academic / portfolio project. **Not** VAR-certified and **not** an official
  refereeing system.
- **Intent:** demonstrate genuine computer-vision engineering (detection, calibration, homography,
  geometric reasoning), not merely an AI-generated interface.

## Architecture & module responsibilities

Pipeline (change only for a justified architectural reason, and record it in `DECISIONS.md`):

1. `src/offside/video_io.py` — video loading, metadata, frame sampling, resizing. **[implemented M1]**
2. `src/offside/detection.py` — player + best-effort ball detection using YOLO11n. **[implemented M2]**
3. `src/offside/calibration.py` — manual pitch-landmark correspondences → homography. **[implemented M3]**
4. `src/offside/geometry.py` — image↔pitch coordinate transforms. **[implemented M3]**
5. `src/offside/offside.py` — offside-position geometry and decision logic. **[implemented M4]**
6. `src/offside/team_cluster.py` — distinguish the two teams (manual-first; optional clustering). **[implemented M5]**
7. `src/offside/annotate.py` — draw detections, pitch geometry, offside line, results. **[implemented M6]**
8. `src/offside/pipeline.py` — coordinate the complete single-frame workflow. **[implemented M6]**
9. `app/streamlit_app.py` — user-facing interface (thin; wraps M6). **[implemented M7]**
   Non-UI glue lives in `app/offside_ui.py` (testable, no streamlit import).

Supporting module (V1.1):
- `src/offside/pitch_model.py` — pure NumPy. The pitch landmark catalogue (31 geometrically defined
  points), pitch markings polylines, custom-point validation, and the Laws-of-the-Game marking
  sizes. Used by the UI, the calibration overlay and the direction diagram. **[implemented V1.1]**

V1.1 also extended `calibration.py` (general-position/spread validation, pixel-unit RANSAC,
warnings, per-point residuals) and `annotate.py` (central `AnnotationStyle`, calibration overlay,
landmark markers). Public APIs stayed backward compatible.

Optional later:
- `src/offside/pose.py` — pose/keypoint estimation **only if** it materially improves ground-contact
  estimation and earns its CPU cost. **[optional M8]**
- OpenVINO / other CPU optimization **only if** real profiling justifies it. **[optional M9]**

Each module must stay independently testable; avoid unnecessary coupling. Calibration and geometry
must not import Ultralytics or PyTorch.

## Hardware & development environment

- OS / shell: Windows + PowerShell.
- Python: **3.13.7** (project-local venv at `.venv/`).
- Machine: Dell Inspiron 15 5000; 11th-gen Intel CPU; Intel Iris Xe integrated graphics; 8 GB RAM.
- **No NVIDIA GPU, no CUDA.** CPU-first implementation is mandatory. The Iris Xe does not
  accelerate the installed PyTorch build (`torch.cuda.is_available()` is False).

## Current dependency environment (actual installed versions)

| Package | Version |
|---|---|
| torch | 2.15.0+cpu |
| torchvision | 0.29.2+cpu |
| ultralytics | 8.4.201 |
| opencv-python | 5.0.0 |
| numpy | 2.5.4 |
| PyYAML | 6.0.3 |
| pytest | 9.2.0 |
| streamlit | 1.66.1 |
| streamlit-image-coordinates | 0.4.1 |

Do not silently change the Python environment or dependency strategy. If a genuine compatibility
issue arises, explain it and propose the smallest practical change (record it in `DECISIONS.md`).

## Permanent technical constraints

- Process short clips (~3–15 s initially).
- Sample frames and resize them to control CPU and memory.
- Never load an entire video into memory (frame iteration is generator-based).
- No cloud inference, paid APIs, or LLMs as substitutes for computer vision.
- Do not train models from scratch.
- Keep pretrained weights, raw videos, and generated outputs out of Git.
- Keep model loading lazy where appropriate.
- Do not claim real-time performance without measurements.
- Never fabricate accuracy, FPS, or evaluation results.

## Critical football / offside correctness rules

- The offside reference line is the **more goalward of {second-last opponent, ball}** along the
  attacking axis.
- The **attacking direction must be known** (user-selected in V1).
- The **own-half exemption** must be considered (a player in their own half cannot be offside).
- Offside **position** is not automatically an offside **offence**; active involvement in play is a
  separate judgement this initial system does **not** make.
- A **ball position is mandatory** for a reliable verdict.
- If the ball cannot be detected or manually positioned, return `UNCERTAIN / INSUFFICIENT_DATA`.
  Never silently ignore the ball and still produce a confident verdict.
- Use player **ground-contact / foot points** for pitch projection — not bounding-box centres or
  heads.
- Camera movement, zoom, or a significant change in viewing angle **invalidates calibration** and
  requires recalibration. A mathematically valid homography does not guarantee an accurate
  real-world calibration.

## Analysis scope (V1)

- The system analyses **one user-selected "pass-moment" frame** (M6). There is **no player
  tracking** and **no whole-video verdict**. Full-video annotation/tracking is a later milestone.
- **Player ids are frame-local**: assigned deterministically by foot-point position within the
  selected frame (`p0, p1, ...`). They are **not** persistent identities and must not be reused
  across frames or runs.
- **Team assignment is frame-local** too and must cover exactly the players detected in that frame.
- **Ball sources** (pipeline): `MANUAL` (user image coordinate, authoritative) >
  `DETECTOR` (accepted only if a single candidate clears the confidence threshold; never treated as
  guaranteed correct) > `UNAVAILABLE` → M4 returns `INSUFFICIENT_DATA`.
- **Calibration must match the analysed frame's camera view** (and its image scale if `max_width`
  resizing is used). Camera movement/zoom/angle change requires recalibration.

## Calibration rules (V1.1)

- **Minimum is 4 correspondences** for the general homography. Never reduce it. Those four must be
  in **general position**: no three on (nearly) one line, in the image **and** on the pitch (e.g.
  several points on one goal line or on the halfway line cannot calibrate). `calibrate()` refuses
  such sets (`assess_point_set` is the single source of validation, also used for live UI feedback).
- With >4 points RANSAC rejects mis-clicks. The inlier threshold is in **image pixels** (H is
  estimated pitch→image, then inverted). Rejected landmarks are reported by name.
- **Warnings are never silent:** exactly 4 points (error is ~0 by construction), weak conditioning,
  small pitch/frame coverage, high residuals, rejected outliers. Reprojection error is a
  **mathematical-fit** measure, not real-world accuracy.
- **Landmarks:** any combination of the catalogue (corners, halfway/centre circle, penalty area,
  goal area, penalty mark/arc) or **custom points** entered in metres with the same convention.
  Marking sizes are **fixed Laws-of-the-Game lengths** (penalty area 16.5 m deep, 40.32 m wide;
  goal area 5.5 m, 18.32 m; penalty mark 11 m; centre circle 9.15 m) measured from the **current**
  pitch dimensions; they do not scale with the pitch.
- **A calibration is bound to the inputs and frame it was built from.** Changing the pitch
  dimensions, any landmark, or selecting a different frame makes it "not ready" and blocks running
  until it is rebuilt (the app cannot verify that a camera is static, so it never reuses a
  calibration on another frame).
- Multi-frame calibration is **not implemented** (see `SPEC.md` §13 for the design recommendation).

## Pitch coordinate convention

- Top-down pitch plane in **metres**. Default dimensions **105 m (length) × 68 m (width)**,
  configurable (UI limits: length 50–130 m, width 40–90 m, length > width).
- **Pitch X = length** (0 → 105 m by default), **Pitch Y = width** (0 → 68 m by default).
- "**X=0 end**" / "**X=L end**" are the two goal lines; "**Y=0 side**" / "**Y=W side**" are the two
  touchlines. These are **pitch coordinates, not image left/right**. The user chooses which
  physical end is X=0, keeps it consistent for every landmark, and the attacking direction uses the
  same axis (`AttackingDirection.POSITIVE_X` = attacking the X=L goal line).
- Image→pitch uses the homography H; pitch→image uses H⁻¹.
