# AI Offside Line Detector — Technical Specification

> Status: **APPROVED.** Building in milestones. Target platform: Windows 10/11, 11th-gen Intel CPU,
> 8 GB RAM, Intel Iris Xe (no NVIDIA/CUDA).
>
> **Note for readers of the public repository:** this is the original design specification, kept as
> a record. Where it differs from the code, `README.md`, `PROJECT_CONTEXT.md` and `PROGRESS.md` are
> authoritative. In particular: V1 analyses one selected frame and outputs an annotated **image**
> (annotated video is not implemented); development used Python 3.13.7 rather than 3.11; and the
> offside rules live in `offside.py`.

## 0. Approved decisions & clarifications

1. **User-selected inputs (V1):** the user manually selects the attacking direction and the
   "moment of the pass" frame. Auto-detection is deferred.
2. **Calibration (V1):** user manually clicks known pitch points to build the homography.
3. **Output:** annotated MP4 video + selected annotated key frames.
4. **Pose:** optional. Evaluate the baseline bounding-box ground-contact estimator first, then
   compare a pose-based estimator on the real hardware before deciding whether to include it.
5. **Offside reference line:** computed from the **frontmost of {second-last opponent, ball}**
   along the attacking axis (whichever is closer to the defending goal defines the line).
6. **Ball is required for a verdict:** if the ball cannot be reliably detected *or* manually
   positioned, the system returns `UNCERTAIN / INSUFFICIENT_DATA` — it never silently ignores the
   ball.
7. **Position vs. offence:** the tool reports offside *position* only. An offside *offence* further
   requires active involvement in play; the tool explicitly does **not** judge that.
8. **Swappable ground-contact estimator:** the bounding-box foot-point estimator and any future
   pose-based estimator implement one shared interface, so the estimator can be replaced without
   touching the offside geometry.
9. CPU-first strategy is unchanged; no unnecessary dependencies; milestone structure unchanged.
10. **Python runtime:** developing on **Python 3.13.7** (3.11 was not available on the machine).
    The architecture is kept Python-version-independent; 3.11 remains supported and we can switch
    later if a dependency requires it. Library compatibility is checked at each install.

---

## 1. Objective

Build a lightweight, single-camera football **offside-position estimation** tool. Given a short local
video clip of a passage of play, the system:

1. Detects players (and the ball) with a lightweight pretrained model.
2. Lets the user calibrate the pitch by marking known pitch lines/points in the image.
3. Computes a **homography** mapping the camera image to a top-down pitch plane.
4. Applies **geometric reasoning** to estimate, for a chosen attacking direction and a chosen
   "moment of the pass", where the offside line sits and which attackers are beyond the
   second-to-last defender.
5. Renders an **annotated video / frames** showing players, the estimated offside line, and
   flagged positions.

This is a **decision-support / estimation** tool, not a refereeing authority. The accuracy
section (§9) makes the limits explicit — we will not claim VAR-grade precision.

---

## 2. Scope and non-goals

**In scope (initial version)**
- Local video upload (short clips, e.g. 3–15 s).
- Frame sampling + resizing for CPU-friendly throughput.
- CPU-first player detection (YOLO11n).
- Manual / assisted pitch calibration → homography.
- Offside-line geometry on the top-down plane.
- Annotated output frames and a stitched output video.
- Streamlit UI as the eventual front end.
- A small, honest test suite (geometry/homography math is deterministic and unit-testable).

**Explicit non-goals (initial version)**
- No training a network from scratch (constraint #3).
- No automatic team classification guarantee (jersey-colour clustering is *best-effort*, see §6.5).
- No automatic "moment of the pass" detection — the user selects the frame (can be automated later).
- No Docker, database, cloud API, paid inference, or LLM dependency (constraint #13).
- No scraping or auto-downloading of copyrighted footage (constraint #10).
- No assumption that the Intel GPU accelerates inference (constraint #8). OpenVINO is an
  optional later experiment, not a baseline.

---

## 3. Model strategy (compatibility checked)

| Concern | Finding | Decision |
|---|---|---|
| Python 3.11 support | Ultralytics supports 3.11 | Use Python 3.11 + venv |
| Detection model | YOLO11n, ~2.6M params, CPU-capable | **Baseline detector** |
| Pose model | `yolo11n-pose` exists but is slow on weak CPUs (~4 FPS even INT8 on a Pi 5) | **Optional**, evaluated in a dedicated milestone, off by default |
| Intel Iris Xe accel | No CUDA/ROCm; PyTorch runs on CPU | Plan for CPU; OpenVINO is an optional later optimization |
| Ball detection | Small/fast-moving; YOLO COCO "sports ball" is unreliable in broadcast frames | Treat ball as best-effort; offside logic must not hard-depend on it |

**Why pose is "evaluate, don't assume":** offside is judged by the *foremost point of a player's
body that can legally play the ball* (feet/torso/head, not arms). Pose keypoints could refine the
"leading edge" of a player versus using the bounding-box bottom-centre. But pose inference cost on
this hardware is high. So: start with bounding-box foot-point projection; add an **optional** pose
refinement milestone and measure the real cost/benefit before committing.

---

## 4. Proposed directory structure

```
AI Offside Line Detector/
├── .gitignore
├── README.md
├── SPEC.md                      # this document
├── requirements.txt             # pinned, CPU-only
├── requirements-dev.txt         # test/lint tooling
├── pyproject.toml               # tooling config (ruff, pytest, mypy)
├── config/
│   └── default.yaml             # frame sampling, resize, thresholds, paths
├── data/
│   ├── raw/                     # user-provided clips (gitignored)
│   ├── samples/                 # tiny synthetic/sample assets committed for tests
│   └── README.md                # dataset & footage-rights guidance (constraint #10)
├── models/                      # downloaded weights (gitignored)
│   └── README.md                # how weights are fetched on first run
├── outputs/                     # annotated frames/videos (gitignored)
├── src/
│   └── offside/
│       ├── __init__.py
│       ├── config.py            # typed config loading (pathlib, dataclasses)
│       ├── video_io.py          # open/sample/resize/release clips safely
│       ├── detection.py         # YOLO11n wrapper (lazy load, CPU)
│       ├── pose.py              # OPTIONAL pose wrapper (added in its own milestone)
│       ├── calibration.py       # point selection → homography (cv2.findHomography)
│       ├── pitch_model.py       # landmark catalogue + pitch markings (V1.1)
│       ├── geometry.py          # projective transforms, offside-line math (pure, testable)
│       ├── team_cluster.py      # best-effort jersey-colour clustering
│       ├── offside.py           # orchestration: detections + homography → offside result
│       ├── annotate.py          # draw players, line, flags onto frames
│       └── pipeline.py          # end-to-end clip → annotated output
├── app/
│   └── streamlit_app.py         # UI (added late, after core pipeline works)
├── scripts/
│   └── fetch_models.py          # explicit, user-initiated weight download
└── tests/
    ├── test_geometry.py         # deterministic math (no model/GPU needed)
    ├── test_calibration.py
    ├── test_video_io.py         # uses a tiny generated clip
    └── test_config.py
```

Everything uses `pathlib` and paths relative to the project root (constraint #12).

---

## 5. Data flow

```
local clip ──► video_io (sample + resize + release)
                   │
                   ▼
            detection (YOLO11n, CPU) ──► player boxes (+ best-effort ball)
                   │
                   ▼
          calibration (user points) ──► homography H (image → pitch plane)
                   │
                   ▼
   geometry: project foot-points via H ──► positions on top-down pitch
                   │
                   ▼
   offside logic: attacking dir + 2nd-last defender ──► offside line + flags
                   │
                   ▼
            annotate ──► output frames/video ──► Streamlit display
```

---

## 6. Component responsibilities

- **config.py** — load `config/default.yaml` into typed dataclasses; resolve paths from project root.
- **video_io.py** — context-managed capture; frame sampling (every Nth frame); resize to a configured
  max width; guaranteed `release()` even on error (constraint #7).
- **detection.py** — lazy-loaded YOLO11n, `device="cpu"`, confidence/IoU thresholds from config,
  returns typed detections. No weights downloaded implicitly at import time.
- **calibration.py** — take ≥4 image↔pitch point correspondences, build H with `cv2.findHomography`
  (pitch→image so RANSAC's threshold is in pixels, then inverted); validate point count, duplicates,
  general position (no 3-of-4 collinear) and spatial spread; report reprojection error honestly
  together with warnings and per-point residuals (V1.1, see §12).
- **pitch_model.py** — pure NumPy catalogue of 31 pitch landmarks (fixed Laws-of-the-Game marking
  sizes measured from the current pitch dimensions), custom-point validation, and pitch-marking
  polylines (V1.1).
- **geometry.py** — pure NumPy: apply H to points, sort players along the attacking axis, compute the
  offside line from the **frontmost of {second-last opponent, ball}**, classify attackers. Returns a
  result with an explicit `UNCERTAIN / INSUFFICIENT_DATA` state when the ball reference is missing.
  Reports offside **position**, not offence. Fully unit-testable, no I/O.
- **ground-contact estimator (interface):** a `GroundContactEstimator` protocol maps a player
  detection → the image point that touches the pitch (the point projected via H). V1 implementation
  is bounding-box foot-point (`BBoxFootEstimator`); a future `PoseFootEstimator` implements the same
  protocol so it can be swapped without changing offside geometry.
- **team_cluster.py** — best-effort: sample jersey pixels, k-means into 2 teams + keeper/ref outliers.
  Clearly labelled as heuristic; user can override.
- **offside.py / pipeline.py** — orchestration and the end-to-end path.
- **annotate.py** — OpenCV drawing of boxes, the offside line, and flags, driven by a central
  `AnnotationStyle`; also the calibration-check overlay and landmark markers (V1.1).

---

## 7. Dependency strategy

**Runtime (`requirements.txt`, CPU-only, pinned at install time):**
- `ultralytics` — YOLO11n detection (pulls a CPU `torch`; we install the CPU wheel explicitly).
- `opencv-python` — video I/O, drawing, homography (`cv2.findHomography`, `perspectiveTransform`).
- `numpy` — geometry math.
- `pyyaml` — config.
- `streamlit` — UI (added when the UI milestone begins).

**Dev (`requirements-dev.txt`):**
- `pytest` — tests.
- `ruff` — lint/format.
- `mypy` — type checking on `src/offside`.

**Install notes (Windows, CPU):**
- Create venv with Python 3.11.
- Install CPU PyTorch wheel explicitly (from the PyTorch CPU index) **before** `ultralytics`, so we
  never pull a CUDA build that won't work on this machine.
- Exact versions will be pinned from what actually resolves on your machine — I will **not** invent
  version numbers; we capture them after a real install.

---

## 8. Git hygiene (constraint #14)

`.gitignore` will exclude: `.venv/`, `models/*` (except `README.md`), `data/raw/*`, `outputs/*`,
`__pycache__/`, `*.mp4`/`*.mov`/`*.avi`, `*.pt`/`*.onnx`, and OS/editor cruft.

---

## 9. Accuracy & honesty policy (constraints #9, #10)

- Offside estimation from a **single broadcast/phone camera** is inherently approximate: lens
  distortion, calibration error, player occlusion, and the difficulty of pinpointing the exact
  "moment of the pass" all introduce error.
- The tool reports **estimates**, shows the homography **reprojection error**, and never presents a
  verdict as authoritative.
- We will **not** fabricate FPS numbers, mAP, or accuracy figures. Any benchmark shown comes from a
  real run on your machine, with the command and output reproduced.
- No footage is scraped or auto-downloaded. Users supply their own clips. `data/README.md` documents
  rights/licensing and points to legitimately usable sources.

---

## 10. Milestone checklist

Each milestone: I state the objective + planned files, implement **only** that milestone, run the
available tests, and report **real** results before moving on.

- [x] **M0 — Scaffolding & tooling.** Repo files, `.gitignore`, `requirements*.txt`, `pyproject.toml`,
      `config/default.yaml`, package skeleton, `data/`+`models/` READMEs. venv on Python 3.13.7;
      5/5 tests pass. **DONE.**
- [x] **M1 — Video I/O.** `video_io.py`: `probe_video`, `VideoReader` (context-managed release),
      `iter_sampled_frames` (lazy generator), `resize_to_max_width` (downscale-only). Tests use a
      synthetic clip (no copyrighted footage). 14/14 tests pass. **DONE.**
- [x] **M2 — Detection.** `detection.py` (typed, Ultralytics-free output; lazy CPU model load;
      conf/class filtering; player-primary, ball best-effort) + `scripts/fetch_models.py` +
      `scripts/test_detection.py` demo. torch 2.15.0+cpu / ultralytics 8.4.201 on Python 3.13.
      Real inference verified: YOLO11n detected 4 players on the bundled sample (~2.4s CPU,
      full-size image). 27/27 unit tests pass (no weights/internet needed). **DONE.**
- [x] **M3 — Calibration & homography.** `calibration.py` (typed correspondences, `cv2.findHomography`
      with exact 4-pt solve / RANSAC for >4, degeneracy + finiteness + invertibility checks,
      reprojection error in px and m, inlier mask) + `geometry.py` (`image_to_pitch`/`pitch_to_image`,
      single + batch, horizon-point guard). Pitch default 105×68 m, configurable. 22 deterministic
      unit tests from a known homography; 49/49 total pass. **DONE.**
- [x] **M4 — Offside geometry.** `offside.py` — pure pitch-space reasoning: `AttackingDirection`,
      `TeamSide`, `PlayerOnPitch`, `BallPosition`, `OffsideResult`; `second_last_opponent()` (purely
      positional, no keeper special case); `evaluate_offside()` with offside line = more goalward of
      {second-last opponent, ball}, level-is-onside, own-half exemption, configurable tolerance,
      mandatory ball → INSUFFICIENT_DATA. 22 unit tests over all required scenarios; 71/71 pass.
      Reports offside **position** only, not offence. **DONE.**
- [x] **M5 — Team classification.** `team_cluster.py` — manual-first `assign_manual` (batch +
      overrides + validation, always reliable) and optional best-effort `cluster_by_jersey`
      (deterministic CIELab torso-colour k-means with separation/reliability measure). Clusters are
      caller-mapped to sides; low confidence propagates. Decoupled via a `BoxLike` Protocol.
      19 unit tests on synthetic frames; 90/90 pass. **DONE.**
- [x] **M6 — Annotation & integrated single-frame pipeline.** `pipeline.py` (`analyze_frame`:
      video→frame→detect→foot-point projection→team lookup→`evaluate_offside`, with the ball policy
      manual > single-confident detector > unavailable) + `annotate.py` (`annotate_frame`: team
      boxes, ball-with-source, per-attacker marks, offside line projected back to image **only when
      confident**, status banner; never mutates input) + `scripts/run_pipeline.py` (JSON-config CLI)
      + `video_io.read_frame`/`frame_index_for_timestamp`. Single-frame V1; frame-local non-
      persistent ids. 26 new tests; 116/116 pass; real-inference integration verified separately.
      **DONE.**
- [x] **M7 — Streamlit UI.** `app/streamlit_app.py` (thin) over the M6 pipeline + `app/offside_ui.py`
      (testable, no-streamlit helpers): upload → frame select (index/timestamp) → detect → click-to-
      calibrate (numeric fallback) → manual team assignment → click-to-place ball → run → annotated
      result + download. Session-state workflow, cached detector, stale-result invalidation.
      `streamlit==1.66.1`, `streamlit-image-coordinates==0.4.1` (verified on 3.13). 23 helper tests;
      139/139 pass; headless launch served HTTP 200 (page shell only; see V1.1). **DONE.**
- [x] **V1.1 — Annotation readability and flexible pitch calibration.** Cyan/blue boxes with orange
      bold-white id labels (central `AnnotationStyle`); `pitch_model.py` with 31 landmarks plus custom
      points; general-position/spread validation and pixel-unit RANSAC in `calibration.py`; UI with
      pitch-size validation, direction diagram, live landmark guidance, calibration report/overlay,
      and calibration bound to its frame and inputs (§12). Multi-frame calibration investigated and
      deliberately not implemented (§13). 297/297 pass. Real-footage accuracy and browser click
      behaviour **not yet verified**. **DONE (automated).**
- [ ] **M8 — Optional pose evaluation.** Add `pose.py`, measure real cost/benefit of `yolo11n-pose`
      on your hardware; keep only if it earns its cost.
- [ ] **M9 — Optional OpenVINO experiment.** Explore Intel-accelerated inference; keep only if it helps.

---

## 11. Open questions for you

1. **Attacking direction & pass moment:** OK to have the user pick these in the UI for v1
   (auto-detection deferred)?
2. **Calibration input:** acceptable to start with the user clicking known pitch points
   (penalty box corners, halfway line) to build the homography?
3. **Output:** annotated MP4 + key frames — does that cover what you want to see?
4. **Pose:** agree to keep pose as an *optional, measured* milestone rather than a baseline?

*(All four were answered and approved; see §0.)*

---

## 12. V1.1 — flexible pitch calibration

**Goal:** real clips rarely show four specific landmarks (e.g. the four pitch corners) at once, so
any suitable combination must work, without weakening the mathematics.

- **Mathematical requirement (unchanged):** a general homography needs **at least 4**
  correspondences, containing 4 points in **general position** — no three on (nearly) one line, in
  the image *and* on the pitch. Several points on one goal line or on the halfway line cannot
  calibrate, however many are chosen.
- **Landmarks:** 31 catalogue points (corners; halfway line ∩ touchlines; centre mark; centre circle ∩
  halfway line; per end: penalty-area and goal-area corners, their goal-line points, penalty mark,
  penalty-arc ∩ penalty-area front line). Sizes are the fixed Laws-of-the-Game lengths measured
  from the **current** pitch dimensions; points that do not fit a small pitch are not offered.
- **Custom point:** enter real-world X and Y in metres (same convention) and click its image
  location. Used for visible markings outside the catalogue.
- **Naming:** "X=0 end"/"X=L end" are the goal lines; "Y=0 side"/"Y=W side" are the touchlines.
  These are pitch coordinates, **not** image left/right; the user picks which end is X=0 and keeps
  it consistent. The attacking direction (`POSITIVE_X` = attacking the X=L goal line) uses the same
  axis; the sidebar diagram and the overlay arrow are drawn from the same enum the offside
  calculation uses.
- **Validation:** blocked — fewer than 4 points, non-finite input, duplicate image or pitch
  locations, no 4-subset in general position. Warned — weak conditioning, small pitch/frame
  coverage, high residuals, rejected outliers, and "exact fit" (exactly 4 points: error is ~0 by
  construction). RANSAC (more than 4 points) works in image pixels and names rejected landmarks.
- **What the numbers mean:** reprojection error is a mathematical-fit measure. It does **not** show
  real-world accuracy. The overlay (projected pitch markings on the frame) is a visual sanity check,
  not verification.
- **Validity:** a calibration is tied to the pitch dimensions and landmarks it was built from and to
  the frame they were clicked on; changing any of them makes it unusable until rebuilt.
- **Pitch dimensions:** editable, validated with units (length 50–130 m, width 40–90 m, length >
  width), default 105 × 68 m; a change clears the calibration and any result.

## 13. Multi-frame calibration — design recommendation (NOT implemented)

**Is it feasible with the current architecture?** Partly. `calibrate()` accepts any list of
(image, pitch) pairs and has no notion of frames, and V1.1 now records each landmark's source
frame, so *accumulating* correspondences is a small change. The part that is **not** available is
verification: pixels from two frames share a coordinate system only if camera position,
orientation, zoom (including digital zoom/crop) and stabilisation are identical, and nothing in the
system can measure that. A mixed-frame set from a moving camera is silently wrong, and its
reprojection error can still look small (RANSAC may even discard the correct points as outliers).

**Current behaviour:** all landmarks must be clicked on the frame being analysed; any other frame is
reported as "calibrated on frame N — recalibrate". A user "camera is static" checkbox was rejected
because it is an unverifiable assertion.

**Recommended design if it is ever built**
1. **Accumulation:** pick a *reference frame*; every correspondence keeps its `frame_index`; the
   calibration describes the reference view only.
2. **Verification (automatic, per extra frame):** match background features (ORB/AKAZE, with player
   boxes masked out) between the extra frame and the **reference** frame (never chain neighbouring
   frames; drift accumulates) and fit a frame-to-frame homography with RANSAC. Accept only if
   there are enough inliers and the transform is the identity within ~1–2 px at the frame corners
   and grid, with no scale change. Add scene-cut detection (histogram distance). If verification
   cannot run (too little texture, e.g. mostly grass), treat the frame as **not verified**.
3. **Invalidation:** a failed or impossible check, a cut, or any frame outside the verified set
   removes that frame's correspondences from the calibration; analysing an unverified frame is
   blocked with "recalibrate".
4. **Visibility:** the landmark table already shows the source frame per correspondence; add a
   verified/unverified badge per frame and the overlay on each frame.
5. **Tests:** synthetic frames with known translation, zoom and rotation; texture-poor frames;
   cuts; and **validation on real clips** before enabling.

**Risks:** slow pans/zooms below the threshold, rolling shutter, digital stabilisation, replays, and
low-texture grass that makes verification fail often. The feature mainly benefits fixed (tactical or
stand) cameras; most broadcast footage pans, so per-frame calibration will stay the norm.

