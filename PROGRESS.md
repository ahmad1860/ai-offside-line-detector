# PROGRESS.md — Milestone Tracker

> Never mark a milestone complete on files alone — require implementation + test evidence.
> Keep in sync with the code. Latest verified full-suite run is recorded per milestone.

| Milestone | Scope | Status |
|---|---|---|
| M0 | Project scaffolding and configuration | Complete |
| M1 | Video input and frame processing | Complete |
| M2 | YOLO11n object detection | Complete |
| M3 | Pitch calibration and homography | Complete |
| M4 | Offside geometry | Complete |
| M5 | Team classification | Complete |
| M6 | Annotation and integrated single-frame pipeline | Complete |
| M7 | Streamlit interface | Complete |
| V1.1 | Annotation readability and flexible pitch calibration | Complete (automated); manual browser and real-footage checks pending |
| M8 | Optional pose estimation | Future/optional |
| M9 | Optional CPU optimization | Future/optional |

---

## M0 — Scaffolding & configuration — Complete
- **Implemented:** repo layout, `.gitignore`, `requirements*.txt`, `pyproject.toml`,
  `config/default.yaml`, typed config loader (`src/offside/config.py`), data/model READMEs.
- **Files:** as above.
- **Limitations:** none material.

## M1 — Video input & frame processing — Complete
- **Implemented:** `src/offside/video_io.py` — `probe_video`, `VideoReader` (context-managed
  release), `iter_sampled_frames` (lazy generator), `resize_to_max_width` (downscale-only).
- **Files:** `src/offside/video_io.py`, `tests/test_video_io.py`.
- **Limitations:** codec support depends on the system OpenCV build; frame counts from some codecs
  are approximate.

## M2 — YOLO11n object detection — Complete
- **Implemented:** `src/offside/detection.py` — typed, Ultralytics-free output
  (`BoundingBox`/`Detection`/`ObjectClass`), lazy CPU model load, confidence/class filtering,
  player-primary + best-effort ball, `foot_point` ground-contact estimate, `draw_detections`.
  Plus `scripts/fetch_models.py` and `scripts/test_detection.py`.
- **Real inference (measured, not fabricated):** YOLO11n detected 4 players on the Ultralytics
  bundled sample image (conf 0.62–0.89); ~2.4 s CPU for a full-size 1080×810 image; model load
  ~2.4 s. Ball not relevant in that image.
- **Limitations:** ball detection unreliable in football footage; CPU inference is not real-time;
  Iris Xe does not accelerate this torch build.

## M3 — Pitch calibration & homography — Complete
- **Implemented:**
  - `src/offside/calibration.py` — `PitchDimensions` (default 105×68 m, configurable),
    `Correspondences` (validates shape, finiteness, ≥4 points, matching counts), `calibrate()`
    (exact 4-point solve; RANSAC for >4 with inlier mask), degeneracy/collinearity + duplicate
    detection, singular-matrix rejection, reprojection error in both pixels and metres,
    `Calibration` result (H, H⁻¹, errors, inliers).
  - `src/offside/geometry.py` — `image_to_pitch()` / `pitch_to_image()` for single points and
    batches, input validation, and a horizon (w≈0) guard that fails safely.
- **Files:** `src/offside/calibration.py`, `src/offside/geometry.py`,
  `tests/test_calibration.py`, `tests/test_geometry.py`.
- **Tests:** 22 new deterministic tests from a known homography (4-pt recovery ~0 error, RANSAC
  outlier rejection, round-trips, error exposure, all invalid-input cases). No weights/internet/GPU.
- **Limitations:** calibration is valid only for the camera view it was made in; camera
  movement/zoom/angle change requires recalibration. Low reprojection error means the supplied
  correspondences are internally consistent, NOT that the real-world calibration is accurate — that
  depends on the user clicking correct, well-spread landmarks. Depends only on NumPy + OpenCV.
- **Independence:** no Ultralytics / PyTorch imports.

## M4 — Offside geometry & decision logic — Complete
- **Implemented:** `src/offside/offside.py` — pure pitch-space reasoning (no CV deps).
  - Enums: `AttackingDirection` (POSITIVE_X/NEGATIVE_X with a `.sign` goalward helper),
    `TeamSide`, `OffsideStatus` (OFFSIDE_POSITION/ONSIDE/UNCERTAIN/INSUFFICIENT_DATA),
    `ReferenceKind` (second-last opponent / ball / tie).
  - Types: `PlayerOnPitch`, `BallPosition` (with `source` provenance),
    `AttackerClassification` (per-player status, signed margin, own-half flag), `OffsideResult`
    (overall status, line X, reference kind, second-last opponent, ball, per-attacker tuple,
    human-readable explanation).
  - `second_last_opponent()` — second-most-goalward defender, purely positional (no keeper rule).
  - `evaluate_offside()` — offside line = more goalward of {second-last opponent, ball};
    strictly-beyond-and-in-opponents'-half = offside position; level = onside; own-half exemption;
    configurable `tolerance_m`. Mandatory ball and ≥2 defenders, else INSUFFICIENT_DATA.
- **Files:** `src/offside/offside.py`, `tests/test_offside.py`.
- **Tests:** 22 new deterministic tests covering all 15 required scenarios (clear offside/onside,
  level, ball-vs-defender reference both ways, both directions, own-half exemption, halfway line,
  <2 defenders, missing ball, mixed attackers, keeper-not-special, tolerance band, non-finite
  coords, invalid direction/pitch/tolerance) plus a ball/defender tie case.
- **Limitations:** reports offside **position** only, never an offside **offence** (no active-
  involvement judgement). `tolerance_m` absorbs numeric noise only — it does not compensate for
  calibration error and implies no VAR-grade accuracy. Players/ball must be pre-projected to pitch
  metres (projection lives in `geometry`, wiring happens in M6).
- **Independence:** no Ultralytics / PyTorch / OpenCV imports.

## M5 — Team classification & assignment — Complete
- **Implemented:** `src/offside/team_cluster.py` — NumPy + OpenCV only (no Ultralytics/PyTorch/UI).
  - **Manual (primary):** `assign_manual(known_ids, mapping, overrides=, require_all=)` — batch
    assignment + per-player overrides; validates duplicate known ids, unknown ids, and (optionally)
    missing assignments. Always `reliable=True`.
  - **Clustering (optional, best-effort):** `cluster_by_jersey(frame, boxes, min_separation=, seed=)`
    — median CIELab torso-region colour (upper-middle crop, inset to avoid grass/shorts/background),
    deterministic `cv2.kmeans` (k=2, fixed seed, stable label orientation by L*), separation measure
    = between-centre / mean within-cluster distance, `reliable` flag vs threshold.
  - Types: `TeamAssignment` (player->TeamSide, method, reliable), `AssignmentMethod`,
    `ClusterResult` (+ `to_team_assignment(cluster_to_side)` which requires both clusters map to two
    distinct sides and **propagates** the unreliable flag). `BoxLike` Protocol keeps it decoupled
    from `detection`.
- **Files:** `src/offside/team_cluster.py`, `tests/test_team_cluster.py`.
- **Tests:** 19 new deterministic tests (manual batch/override/validation/empty; clustering of
  two separated colours; determinism; single-colour degeneracy; <2 players; invalid/out-of-frame/
  non-finite boxes; explicit cluster->side mapping and its validation; low-confidence propagation).
- **Limitations:** clustering is best-effort and does NOT decide which cluster is attacking vs
  defending (caller maps explicitly); it does not reliably handle referees, goalkeepers, similar
  kits, lighting changes, or occlusion. Manual assignment is the dependable path.

## M6 — Annotation & integrated single-frame pipeline — Complete
- **Scope:** single selected "pass-moment" frame. No tracking, no whole-video verdict.
- **Implemented:**
  - `src/offside/video_io.py` (extended): `read_frame(path, index, max_width=)` (seek + verified
    read, always releases) and `frame_index_for_timestamp(path, seconds)`.
  - `src/offside/pipeline.py` — `analyze_frame(...)` integration boundary: read frame → detect →
    split players/ball → deterministic frame-local ids (`assign_player_ids`, positional) → validate
    team assignment covers exactly detected players → project **foot points** to pitch via M3 →
    resolve ball (MANUAL > DETECTOR-if-single-confident > UNAVAILABLE) → `evaluate_offside`. Returns
    `FrameAnalysis` (players, `BallInfo` with `BallSource`, `OffsideResult`, calibration, teams).
  - `src/offside/annotate.py` — `annotate_frame(frame, analysis)` returns an annotated **copy**;
    draws team-coloured boxes + ids + foot points, ball with source, per-attacker ON/OFF marks, and
    the offside line projected back to image space via `pitch_to_image` — **only** for a confident
    verdict. Status banner always; short explanation when UNCERTAIN/INSUFFICIENT_DATA.
  - `scripts/run_pipeline.py` — JSON-config CLI (video, frame_index/timestamp, calibration points,
    attacking direction, team mapping, optional manual ball) with validation and clear errors.
- **Files:** `src/offside/pipeline.py`, `src/offside/annotate.py`, `scripts/run_pipeline.py`,
  `tests/test_pipeline.py`, `tests/test_annotate.py`, extended `video_io.py` + `test_video_io.py`.
- **Tests:** 26 new (13 pipeline, 7 annotation, 6 video_io) — both directions, foot-point
  projection, all three ball sources + unreliable/multi-candidate → INSUFFICIENT_DATA, missing/
  unknown team assignment, insufficient defenders, invalid frame index, resource cleanup; annotation
  draws line only when confident, never mutates input, rejects invalid frames.
- **Real inference (measured, separate from mocked tests):** full pipeline + annotation ran on a
  1-frame clip built from the Ultralytics bundled people image with the real YOLO11n detector —
  4 players detected, foot-points projected, manual ball applied, `offside_position` returned,
  annotated PNG written; ~357 ms with a warm detector. (Scenario is arbitrary; verifies the
  integration path, not accuracy.)
- **Limitations:** single frame only; frame-local non-persistent ids; calibration must match the
  frame's view/scale; ball unreliable without manual input; reports offside POSITION, not offence.

## M7 — Streamlit interface — Complete
- **Scope:** UI milestone only; wraps the M6 pipeline, no new CV logic.
- **Implemented:**
  - `app/offside_ui.py` (testable, no streamlit import): pitch-landmark catalogue
    (`PitchLandmark`, `PITCH_LANDMARKS`), `correspondences_to_arrays`, `build_calibration`,
    `prepare_team_assignment`, `prepare_ball_xy`, `parse_attacking_direction`,
    `analysis_signature` (stale-result hash), `validate_ready_to_run` (ball intentionally not a
    prerequisite — missing ball -> INSUFFICIENT_DATA). Errors surface as `UIValidationError`.
  - `app/streamlit_app.py` (thin): upload+save-to-temp, metadata via `probe_video`, frame select by
    index/timestamp (`read_frame`/`frame_index_for_timestamp`), Detect Players (cached detector via
    `@st.cache_resource`, re-detect prompt when frame changes), click-to-calibrate and click-to-ball
    via `streamlit-image-coordinates` with numeric fallback, manual team assignment (reusing M5),
    Run Analysis -> `analyze_frame`, annotated image + status + per-attacker details + PNG download.
    Session-state workflow; stale results invalidated by signature; temp upload cleaned on replace.
  - `app/__init__.py`; `pyproject.toml` pytest `pythonpath` now includes `.` so `app` imports.
- **Files:** `app/offside_ui.py`, `app/streamlit_app.py`, `app/__init__.py`,
  `tests/test_offside_ui.py`; `requirements.txt` + `pyproject.toml` updated.
- **Tests:** 23 new helper tests (landmark lookup/bounds, correspondence conversion, calibration
  success + collinear rejection, team-assignment prep + validation, ball prep, direction parse,
  signature stability/variation, readiness validation). Streamlit's interactive UI is not browser-
  automated (deliberate).
- **UI launch check (not an accuracy claim):** `streamlit run app/streamlit_app.py` booted headless
  on port 8599, served HTTP 200; both UI modules import without error.
  *Correction (V1.1):* that check only fetched the page's HTML shell; it did not execute the script
  body. Script execution is now genuinely covered by `tests/test_streamlit_app.py` (AppTest).
- **Limitations:** single-frame; frame-local ids; click component is third-party (numeric fallback
  provided); calibration must match the frame's view/scale; not VAR-grade; position not offence.

## V1.1 — Annotation readability and flexible pitch calibration — Complete (automated)
- **Defects found in earlier milestones and fixed (verified by running the old code first):**
  1. M3 `calibrate()` accepted 3-of-4 collinear points (only global collinearity was checked).
  2. M3 RANSAC threshold was applied in **metres**, not pixels as documented: a 12 px mis-click was
     accepted as an inlier. H is now estimated pitch→image so the threshold is in pixels (default 5).
  3. M7 landmark coordinates hardcoded fractions of 105×68 (penalty box scaled with pitch length).
  4. M7 kept `ss["calibration"]` after landmarks/pitch size changed and across frames (stale use).
  5. M7 `analysis_signature` ignored the pitch dimensions.
  6. M7 `use_container_width` is deprecated in Streamlit; migrated to `width="stretch"`.
- **Annotation:** `AnnotationStyle` (central colours/sizes, scaled with frame width). Cyan
  (attackers) / blue (defenders) boxes; orange label pills with bold white text and a dark halo;
  red offside line with white casing; ON/OFF as filled pills (no green text on grass); labels kept
  inside the image and out from under the status banner; offside-line label drawn after the boxes.
  New `draw_calibration_overlay` (projected pitch markings + attack arrow) and
  `draw_correspondence_markers`. Verified by rendering and viewing images; two layout defects found
  that way (label hidden under the banner; line label cut by a box) were fixed.
- **Calibration core:** general-position check over all 4-subsets, duplicate-location checks,
  pitch/image coverage and weak-conditioning warnings, per-point residuals, outlier reporting,
  `exact_fit` warning, `assess_point_set` for live feedback. Minimum remains 4 points.
- **Landmarks:** new `src/offside/pitch_model.py` — 31 landmarks (corners, halfway line, centre
  circle, penalty area/goal area corners and goal-line points, penalty marks, penalty-arc points)
  with unambiguous pitch-coordinate names; legacy keys still resolve. Custom points (metres + click).
  Duplicates refused; replace is an explicit action; frame provenance recorded per landmark.
- **UI:** pitch size validated with units and clears calibration/results on change; attacking-
  direction diagram + explanation, derived from the same `AttackingDirection` the calculation uses;
  numbered markers on the click image; live valid-landmark count and guidance; calibration report
  (mean/worst px, error on pitch, landmarks used, coverage, warnings, residual table, overlay);
  sidebar calibration status; click handling scaled/de-duplicated via tested helpers.
- **Files created:** `src/offside/pitch_model.py`, `tests/test_pitch_model.py`,
  `tests/test_calibration_validation.py`, `tests/test_calibration_ui.py`,
  `tests/test_streamlit_app.py`.
- **Files changed:** `src/offside/calibration.py`, `src/offside/annotate.py`, `app/offside_ui.py`,
  `app/streamlit_app.py`, `tests/test_annotate.py`, `scripts/run_pipeline.py` (prints calibration
  warnings), docs.
- **Tests:** +158 (pitch model 40, calibration validation 30, UI helpers 55, AppTest 13, annotation
  +20). Mutation checks confirmed the tests fail when the collinearity check, pixel RANSAC
  threshold, duplicate check, or the pitch-change handler is removed.
- **Not verified:** real-footage calibration accuracy; the third-party click component's browser
  behaviour (pixel clicking, replace/remove flows); how the annotations look on real footage.
- **Multi-frame calibration:** investigated, **not implemented** (cannot verify a static camera);
  design recommendation in `SPEC.md` §13.

---

### Verified test results
- **M0–M2 combined:** 27 tests passed in 0.80 s.
- **M0–M3 combined:** 49 tests passed in 0.94 s.
- **M0–M4 combined:** 71 tests passed in 1.09 s.
- **M0–M5 combined:** 90 tests passed in 1.41 s.
- **M0–M6 combined:** 116 tests passed in 0.87 s.
- **M0–M7 combined:** 139 tests passed in 1.84 s (Python 3.13.7, CPU). Automated unit tests
  require no model weights, internet, or GPU (detector mocked; CV-free modules use synthetic data).
  Real-inference (M6) is a separate manual verification, noted above.
- **M0–V1.1 combined:** **297 tests passed in 12.84 s** (Python 3.13.7, CPU). 13 of these execute
  the real Streamlit script in-process (AppTest) and account for most of the run time. No model
  weights, internet, or GPU are needed.
