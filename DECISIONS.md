# DECISIONS.md — Architectural & Technical Decisions

> Append new significant decisions with: decision, rationale, alternatives rejected.
> Do not contradict an existing decision without recording why the change is necessary.

## D1 — Python 3.13.7 retained
**Decision:** develop on Python 3.13.7. **Rationale:** 3.11 was not installed on the machine, and
the full dependency stack (torch/torchvision CPU, ultralytics, opencv, numpy) installs and runs
cleanly on 3.13.7. Architecture is kept version-independent. **Rejected:** blocking on a 3.11
install (no benefit, slows progress).

## D2 — CPU-only PyTorch, no CUDA
**Decision:** install `torch`/`torchvision` from the CPU wheel index; never CUDA builds.
**Rationale:** the laptop has no NVIDIA GPU; CUDA wheels are useless and large. CPU torch is
installed **before** ultralytics so ultralytics does not pull a CUDA build. **Rejected:** default
PyPI torch (may resolve CUDA variants).

## D3 — YOLO11n as the initial detector
**Decision:** use pretrained YOLO11n (nano). **Rationale:** smallest practical pretrained detector
(~2.6M params), CPU-capable, no training. **Rejected:** larger YOLO sizes (too slow on CPU),
training/fine-tuning (out of scope).

## D4 — COCO `person` → player
**Decision:** map COCO class 0 (person) to `PLAYER`. **Rationale:** stock pretrained model has no
football-specific "player" class; person is the correct proxy. **Rejected:** custom-trained player
detector (out of scope for V1).

## D5 — Ball detection is best-effort only
**Decision:** COCO class 32 (sports ball) → `BALL`, flagged best-effort. **Rationale:** the ball is
small/fast and frequently missed in football footage; it is not a football-specific detector.
**Consequence:** offside logic must not depend on reliable automatic ball detection.

## D6 — Manual ball placement fallback
**Decision:** allow the user to manually position the ball when detection fails. **Rationale:** a
ball reference is mandatory for a verdict (see D8); manual placement avoids silently guessing.

## D7 — Manual calibration (V1)
**Decision:** V1 calibration is manual selection of known pitch landmarks mapped to pitch
coordinates. **Rationale:** robust, transparent, and feasible without auto pitch-line detection.
**Rejected:** automatic pitch-line detection (deferred; harder and error-prone on a single camera).

## D8 — Ball mandatory; otherwise UNCERTAIN / INSUFFICIENT_DATA
**Decision:** without a reliable detected or manually placed ball, the offside result is
`UNCERTAIN / INSUFFICIENT_DATA`. **Rationale:** offside depends on the moment of the pass and the
ball position; a confident verdict without the ball would be misleading.

## D9 — Pitch reference 105 × 68 m, configurable
**Decision:** default top-down pitch is 105 m (length, X) × 68 m (width, Y), configurable.
**Rationale:** standard full-size pitch; metres make geometry and distances physically meaningful.
**Rejected:** normalized 0–1 rectangle (less meaningful; harder to reason about distances).

## D10 — Manual pass-moment selection (V1)
**Decision:** the user selects the pass-moment frame. **Rationale:** reliable auto pass detection is
hard; manual selection keeps V1 correct and simple. **Deferred:** automatic detection.

## D11 — Team assignment manual first, clustering later
**Decision:** start with manual team assignment; add jersey-colour clustering only if it proves
useful and reliable. **Rationale:** clustering is heuristic and can fail (kits, lighting, keepers,
refs); manual is a dependable baseline.

## D12 — No VAR-grade or official-referee claims
**Decision:** the tool reports estimates of offside **position** only, shows calibration error, and
never claims refereeing authority. **Rationale:** single-camera estimation is inherently
approximate; honesty is a hard requirement.

## D13 — Swappable ground-contact estimator
**Decision:** player pitch projection uses a ground-contact (foot) point behind a shared interface;
V1 uses bounding-box bottom-centre, a future pose estimator can replace it without changing offside
geometry. **Rationale:** keeps the accuracy-critical estimator replaceable.

## D15 — Offside logic operates in pure pitch space
**Decision:** `offside.py` consumes already-projected `PlayerOnPitch` / `BallPosition` (metres) and
imports no CV libraries. Projection from image to pitch (via `geometry`) and team assignment happen
upstream (M5/M6). **Rationale:** keeps the football rules independently testable with synthetic
coordinates and decoupled from detection/calibration. **Rejected:** passing detections + a
calibration into the offside module (couples it to CV concerns and complicates testing).

## D17 — Team classification: manual-first, clustering optional & caller-mapped
**Decision:** `team_cluster.py` provides deterministic **manual** assignment as the primary,
always-reliable path, and **jersey-colour k-means** (CIELab median of a torso crop, fixed seed) as
an optional best-effort convenience. Clustering returns cluster indices + a separation/reliability
measure; it does **not** decide which cluster is attacking vs defending — the caller maps clusters
to sides explicitly, and an unreliable cluster propagates its `reliable=False` flag into the
resulting assignment. **Rationale:** honours D11; keeps the fragile heuristic from silently becoming
a trusted verdict. **Rejected:** auto-assigning sides from colour (impossible without external
context) and auto-trusting low-separation clusters.

## D19 — M6 is single-frame; frame-local, non-persistent player ids
**Decision:** V1 analyses one selected pass-moment frame. Player ids are assigned deterministically
by foot-point position within that frame (`p0, p1, ...`) and are explicitly NOT persistent across
frames/runs. Team assignment is validated to cover exactly the detected players. **Rationale:**
honest scope — no tracking exists yet, so claiming cross-frame identity or a whole-video verdict
would be false. **Rejected:** copying one frame's verdict across a whole MP4 (misleading).

## D20 — Pipeline ball policy: manual > single-confident detector > unavailable
**Decision:** manual ball image coordinate is authoritative; otherwise a detector ball is accepted
only if exactly one candidate clears `ball_min_conf` (default 0.5), and even then it is flagged as
not guaranteed correct; otherwise the ball is UNAVAILABLE and M4 returns INSUFFICIENT_DATA.
**Rationale:** implements D5/D6/D8 concretely without inventing a ball or over-trusting an
unreliable class. **Rejected:** trusting the highest-confidence ball unconditionally.

## D22 — UI logic split into a testable non-Streamlit helper module
**Decision:** all reusable UI glue lives in `app/offside_ui.py` (no streamlit import);
`app/streamlit_app.py` only wires widgets/state/rendering. **Rationale:** keeps M7 logic unit-
testable without brittle browser automation and keeps the Streamlit layer thin. **Rejected:**
putting logic directly in the Streamlit script (untestable without a UI driver).

## D23 — Click-to-interact via streamlit-image-coordinates, with numeric fallback
**Decision:** use `streamlit-image-coordinates` (0.4.1, verified on Python 3.13) for click-to-
calibrate and click-to-place-ball; if the component is unavailable, fall back to numeric pixel
entry. **Rationale:** a lightweight maintained component gives the required UX; the fallback keeps
the app usable without it. **Rejected:** heavier canvas components; click-only with no fallback.

## D24 — Stale results invalidated by an input signature
**Decision:** a displayed analysis is kept only while a SHA-256 signature of its inputs (video,
frame, direction, correspondences, teams, ball) matches the current inputs; any change invalidates
it. **Rationale:** guarantees the shown verdict always matches current inputs across Streamlit
reruns. **Rejected:** trusting Streamlit rerun ordering to keep results fresh.

## D21 — read_frame / timestamp helpers added to video_io (M6 gap)
**Decision:** add `read_frame(path, index, max_width=)` and `frame_index_for_timestamp()` to
`video_io.py` rather than inventing frame-seeking inside the pipeline. **Rationale:** single-frame
selection is a genuine video-I/O capability M1 lacked; keeping it in `video_io` preserves the
resource-safety guarantees and keeps the pipeline thin. **Rejected:** seeking via the capture
directly in `pipeline.py` (duplicates release logic, weakens the M1 boundary).

## D18 — team_cluster stays decoupled via a BoxLike Protocol
**Decision:** `team_cluster.py` accepts any object exposing `x1/y1/x2/y2` (a `BoxLike` Protocol)
rather than importing `detection.BoundingBox`. **Rationale:** avoids pulling the Ultralytics-adjacent
detection module into team logic and keeps clustering testable with trivial box stubs. **Rejected:**
a hard import of `BoundingBox` (unnecessary coupling).

## D16 — Tolerance is numeric-noise only; goalward sign abstraction
**Decision:** a single configurable `tolerance_m` (default 0.05 m) decides "level" (level = onside),
and both attacking directions are handled by one code path via `AttackingDirection.sign` (goalward
score = sign × X). **Rationale:** level-is-onside is the law; a small epsilon prevents sub-cm
floating-point noise from flipping a verdict; the sign trick avoids duplicated direction logic.
**Explicitly NOT:** a correction for calibration error or a claim of VAR-grade precision.

## D14 — Scripts add `src/` to sys.path
**Decision:** runnable scripts under `scripts/` insert `src/` onto `sys.path` at startup.
**Rationale:** the `src/` layout is only auto-added for pytest (via `pyproject.toml`); plain scripts
need it explicitly. **Rejected:** installing the package (`pip install -e .`) — kept optional to
avoid extra build steps for now.

## D25 — Landmark catalogue and pitch markings live in `pitch_model.py` (V1.1)
**Decision:** a pure-NumPy `src/offside/pitch_model.py` holds the landmark catalogue (31 points),
the pitch-marking polylines, and custom-point validation. Each landmark is `reference + offset`
(start/mid/end of an axis plus a signed metre offset) using the **fixed Laws-of-the-Game sizes**
(penalty area 16.5 × 40.32 m, goal area 5.5 × 18.32 m, penalty mark 11 m, centre circle 9.15 m).
Names are in pitch coordinates ("X=0 end", "Y=W side"), never image left/right. Old keys resolve via
`LEGACY_KEY_ALIASES`. **Rationale:** the old catalogue stored fractions of a 105×68 pitch, so a
penalty area changed size with the pitch length and labels like "near/far/left/right" were
ambiguous; the overlay and diagram also need the same marking geometry. **Rejected:** keeping the
catalogue in `app/offside_ui.py` (CLI/overlay could not reuse it); scaling markings by fraction.

## D26 — Calibration validation: general position + spread, errors vs warnings (V1.1)
**Decision:** the minimum stays **4** correspondences (no 3/2-point shortcuts for a general
homography). `assess_point_set` (single source of validation, also used for live UI feedback)
**blocks** on: <4 points, mismatched/non-finite input, duplicate image or pitch locations, and no
4-subset in general position (best-subset triangle-height ratio below 0.02 in the image or on the
pitch). It **warns** (never silently) on weak conditioning (<0.10), pitch coverage <15 %, frame
coverage <10 %, mean/worst residuals (4 / 12 px) or >1 m on the pitch, rejected outliers, and
"exact fit" with exactly 4 points. **Rationale:** the old global-collinearity check accepted 3-of-4
collinear sets, which do not determine H; flexible landmark choice makes that likely. The numeric
thresholds are engineering heuristics, **not** accuracy guarantees. **Rejected:** blocking on every
poor-coverage case (sometimes the only visible landmarks are clustered; the user must be told, not
prevented); only checking global collinearity.

## D27 — RANSAC threshold is in image pixels (fix to M3) (V1.1)
**Decision:** estimate pitch→image (`findHomography(pitch, image)`) and invert, so
`ransacReprojThreshold` (default now 5 px) is measured in pixels where click noise occurs.
**Rationale:** OpenCV measures the threshold in the *destination* space; M3 estimated image→pitch,
so the documented "3 px" was actually 3 **metres** and a 12 px mis-click was accepted (reproduced
before the fix). `reprojection_error_*` are now computed over inliers; per-point residuals cover
all points. **Rejected:** keeping image→pitch and converting the threshold to metres (varies with
perspective across the pitch).

## D28 — Custom landmarks carry explicit metres in the same convention (V1.1)
**Decision:** a custom correspondence stores the user's pitch (X, Y) in metres plus the clicked pixel;
catalogue correspondences store only the landmark key and derive metres from the *current* pitch.
Custom coordinates are validated (finite, at most 10 m outside the pitch). A landmark already placed
is never silently overwritten (explicit replace); a point whose pitch location coincides with an
existing one is refused. **Rationale:** lets users calibrate from visible markings outside the
catalogue while keeping every image point paired with exactly one pitch point.

## D29 — A calibration is bound to the inputs and frame it was built from (V1.1)
**Decision:** each correspondence records its source `frame_index`; the built calibration stores a
signature of (pitch dimensions, landmarks, custom coordinates, frames). `check_calibration` returns
MISSING / OUTDATED (pitch or landmarks changed) / WRONG_FRAME / READY, and Run is blocked unless
READY. Changing the pitch size also clears the calibration immediately. **Behaviour change:** in M7 a
calibration silently carried over to other frames; that is no longer allowed. **Rationale:** the
app cannot verify that the camera has not moved, so reusing a calibration on another frame could
produce a confident but wrong verdict. **Rejected:** a "camera is static" checkbox (an unverifiable
user assertion that would be the unsafe part of multi-frame calibration).

## D30 — Multi-frame calibration investigated, not implemented (V1.1)
**Decision:** do not accumulate correspondences across frames. **Rationale:** the architecture can
technically hold them (provenance is now recorded) but nothing can verify a static camera, and a
mixed-frame set can still show a small reprojection error. The design recommendation (reference-
frame feature check, per-frame verification, invalidation, UX) is in `SPEC.md` §13. Revisit only
after a static-camera check is validated on real clips. Flexible/custom landmarks address the
original problem (four fixed corners rarely visible at once) without this risk.

## D31 — Annotation style is centralised; each meaning has its own colour (V1.1)
**Decision:** `AnnotationStyle` (frozen dataclass, BGR) holds all annotation colours/sizes; sizes
scale with frame width. Cyan = attacker box, blue = defender box, orange pill + bold white text =
player id, red + white casing = offside line, red/dark-green pills = OFFSIDE/ONSIDE marks, yellow =
ball, white-on-dark dot = foot point, magenta = calibration aids. Labels are clamped inside the
image and kept below the status banner; the offside-line label is drawn after the boxes. Team is
also in the label text (ATK/DEF), so colour is never the only cue. **Rationale:** green text/boxes
vanished against the pitch; defects found by viewing renders (label under banner, line label cut by
a box) drove the layout rules. **Rejected:** loading the style from YAML (no need yet).

## D32 — Calibration overlay is a visual sanity check, never verification (V1.1)
**Decision:** `draw_calibration_overlay` projects the pitch markings and the attack arrow (from the
same `AttackingDirection`) onto the frame so the user can compare them with the real lines. The UI
states that reprojection error is mathematical fit, not real-world accuracy. **Rationale:** gives
users a real-world check that residuals cannot provide, without claiming accuracy.

## D33 — Streamlit is tested by executing the real script (AppTest) (V1.1)
**Decision:** `tests/test_streamlit_app.py` runs `app/streamlit_app.py` in-process with
`streamlit.testing.v1.AppTest`; pure logic stays in `app/offside_ui.py`. **Rationale:** the M7
"HTTP 200" launch check only fetched the page shell and could not catch runtime errors in the
script. AppTest exercises widget wiring and session state without brittle browser automation.
**Limit:** it cannot click pixels in the third-party image component, so click behaviour still needs
a manual check.
