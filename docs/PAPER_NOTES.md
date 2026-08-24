# PAPER_NOTES.md — running results ledger for the research paper

The project ends in a **research paper**. This file is the single running record of paper-worthy
results: metrics **with the test set they were measured on**, ablations, failure cases, figures, and
the methodology decisions behind them. Update after every meaningful experiment. Keep failures — they
are the ablations / limitations sections.

**Provenance rule:** every metric records *date · model/config · dataset · test set*. Any "A beats B"
claim must be a **head-to-head on ONE human-verified held-out set** (two numbers from different test
sets are not comparable — see Open Rigor Items).


*(References below to `AGENTS.md` are to the project's internal working runbook, which is not
part of this release.)*

---

## Working title / framing
Automated **behaviour & affective-state analysis** of a single octopus ("Nity") from continuous
multi-camera aquarium footage: detection → clip extraction → structured behavioural extraction →
distilled local models (caption, segmentation) → ethological time-series (activity budget, circadian,
stimulus-response). Framed as **arousal / behavioural-state**, not emotion.

## Contributions (draft)
1. A full pipeline turning raw 24/7 footage into a quantified behavioural time-series.
2. Teacher→student distillation for **local, laptop-runnable** caption + (WIP) segmentation models.
3. Ethological findings on Nity (activity budget, circadian rhythm, human-presence stimulus response).
4. Methodology lessons (below) that generalize to aquarium/animal video analysis.

---

## Results so far

### R1 — Behaviour analysis (the headline scientific results)
- Corpus: **3,205 clips** structured-extracted via Qwen3-VL-235B (OpenRouter), $2.22, ~$0.0006/clip.
  3,083 present. Full run 2026-07-20.
- **Activity budget:** 41% exploration/manipulation · 33% resting · 14% human-interaction · 9% reaching-out · 2% crawling · 1% swimming.
- **Circadian:** visible-activity ~1–5% overnight → **45% peak at 17:00** (13:00–19:00 plateau) + dawn bump 05–06h. Exposure-normalized (present ÷ all extracted windows/hour).
- **Stimulus response:** human presence **nearly doubles motion (0.045→0.095)** and lifts arousal **0.46→0.68**.
- Colour (colour cameras only): dark_red_brown most common at baseline (~16%) vs during human interaction (~6%).
- Artifacts: `data/behaviour_stats.json`, `data/behaviour_dashboard.html`, published artifact "Nity — behavioural profile".
- CAVEATS (limitations section): `context="enrichment_object"` fires ~66% (means "object in tank", not active enrichment); presence gate dirty upstream. Rate/response *contrasts* robust; absolute levels shift after detector retrain.

### R2 — Caption student (distillation)
- Qwen3-VL-2B + LoRA r16/α32, distilling 235B teacher captions. 3066 train / 392 val, 576 steps (DONE 2026-07-15).
- **Eval (50 held-out val): base emb-sim 0.702 / rougeL 0.269 → LoRA 0.834 / 0.455.**
- Local 4-bit MLX deploy: `models/qwen3vl2b_caption_v1_mlx_4bit` (1.7 GB, ~3 s/caption on 16GB Mac, no GPU).
- Cross-platform HF backend added (base Qwen3-VL-2B + LoRA) for non-Apple hardware.

### R3 — Segmentation (auto-labeler + tiny student) — A100 run 2026-07-23
- Auto-labeler: **GroundingDINO-tiny (box) → SAM2 (mask)**, seed = most-confident frame, **video propagation**
  (temporal consistency) + largest-blob + area-continuity. Phase-0 validated on 4 cameras (before/after):
  IR tool-bleed 11.8%→6.5%, colour bg-bleed 15.5%→5.8%, reflection camera rejected by low seed conf (~0.50).
- Labeled 1,824 colour clips → **4,412 (image,mask) pairs / 77 videos**.
- **Mask pixel-IoU: plateaus ~0.47** (bar 0.85) across TinyUNet(ch8/16/32), LR-ASPP, aug, +IR.
  Diagnosed **video-diversity generalization gap** (only 62 train videos; train IoU 0.68 / val 0.47;
  fails by *mislocating* a right-sized blob). → limitations + motivates DATA_PLAN.
- **Presence gate WIN:** v1 (positives-only) = random (AUC 0.50). **v3 = 4,412 pos + 1,388 empty-mask negatives
  → AUC 0.86 overall, 0.99 vs reflections, 0% reflection-FP at area≥0.01 (88% present-recall).** Model
  `weights/seg/octo_seg_v3_lraspp.pt` (LR-ASPP, 3.2M params). Full trail: `src/SEGMENTATION_LOG.md`, `results/segmentation/`.
- IR (Right_Top) unusable as-is (GroundingDINO low-conf on greyscale, SAM2 grabs bright tools) — needs Phase-0 IR fix.

### R3b — Segmentation diversity retrain (Modal, 2026-08-07) — tests the R3 diversity diagnosis
- The diverse-footage harvest (R5) closed the loop: **530 clips / 276 distinct videos / 149 dates** auto-labeled
  on Modal (A10G, `src/modal_seg_train.py`, GD+SAM2 teacher, min_seed_conf 0.60) → **178 accepted / 732 pairs**
  (345 low-conf recoverable via a lower-conf pass). Merged with old v1 and retrained (LR-ASPP, 60 ep, split BY VIDEO).
- **Head-to-head (best val IoU):** old **62 vid / 4,412 pairs = 0.468** (soft same-week val) → harvest **new-only
  100 vid / 732 pairs = 0.245** (overfits: 588 frames too few) → **merged 176 vid / 5,144 pairs = 0.494** (best,
  on a HARDER diverse-date 35-video val). Model `weights/seg/octo_seg_merged_lraspp.pt`.
- **Interpretation (paper):** diversity helps *robustly but modestly* (+0.026, and on a genuinely harder val);
  the merged val plateaus flat at ~0.49 with NO overfitting. Diversity helped up to ~176 videos, then flat.

### R3c — HQ-teacher upgrade: an HONEST NEGATIVE RESULT (2026-08-08)
Hypothesis: the ~0.49 plateau is teacher-label quality (a distilled student can't beat noisy GD+SAM2-tiny masks).
Motivating evidence: the merged model scored 0.49 vs tiny masks but 0.70 vs HQ masks. Test: upgrade teacher to
**GD-base + SAM2-large**, re-label ALL clips (harvest 740 + old 3,991 = 4,731 HQ pairs), retrain.
- **Result: held-out val IoU FLAT** — tiny 0.494 → 14%-HQ 0.508 → 100%-HQ **0.506**. Clean labels did not help.
- **Why the hypothesis failed:** the 0.70-vs-HQ figure was **train leakage** (evaluated on training clips). The
  clean held-out HQ-vs-HQ number is ~0.51, same as tiny. **Teacher-label quality was NOT the ceiling.**
- **What the ceiling actually is:** the tiny student generalizes to **right-SIZED but mis-LOCALIZED masks**
  (areaErr **1.4%**, Dice 0.62, IoU 0.50). Not fixable by labels or data (both exhausted) — it's a per-frame
  localization limit → points to a **temporal** student as the real lever.
- **Reframe for the paper:** pixel-IoU-vs-teacher-masks is a weak metric here (teacher isn't perfect GT). The
  project needs **area/posture + presence**, and area is accurate to 1.4% — so the model may already be adequate
  for its downstream ethological use. **Open rigor item: human-verified mask val + downstream-task eval** decide
  whether ~0.5 IoU actually matters. Good methodology lesson: *always check train/val leakage before trusting a
  "measure against cleaner labels" signal.*

### R4 — Negative results / ablations (keep for the paper)
- **Behavior classifier** (frozen CLIP feats → MLP): 45% val acc, *below* 50% majority baseline. Lesson: static pooled features can't classify behaviour — needs temporal/motion features.
- **Segmentation arch/aug ablations:** no gain from ch8→ch32, LR-ASPP, or strong augmentation — confirms the limit is data, not model.
- **235B as presence filter:** running 235B over 847 "verified" clips, **534 (63%) came back not-present** — the CLIP+motion extractor massively over-extracts (esp. Right_Left reflections). VLM is a far better presence filter than the detector.

---

## Methodology lessons (paper "what worked / what we learned")
- **Letterbox, not center-crop** for CLIP — aspect-ratio mismatch (not architecture) caused poor field performance.
- **Absolute motion (changed-pixel fraction), not per-video normalized** — normalization passes static videos (lamp flicker). Mask the burned-in timestamp region.
- **Teacher→student distillation** gives local, cheap, offline models (big VLM/SAM2 → tiny student).
- **Temporal consistency (SAM2 video propagation)** kills *transient* auto-label errors — but NOT *consistent* ones (a reflection present every frame). Gate on seed confidence + drop reflection camera.
- **Structured extraction > captioning** for behaviour: a caption is one field; posture/colour/context need a JSON re-prompt.

## R5 — Diverse-footage harvest (data-gen, in progress 2026-07-23)
- **Diagnosis:** whole corpus was 7 dates/one week → both students diversity-limited. Server actually holds
  ~6 collections / ~5 animals; **Nity alone ~209 days**. (Method note: server exposes crawlable HTML listings.)
- **Network finding (measured, Colab→server):** ~5 MB/s download; **parallelism barely helps** (1→5 streams =
  1.6×, near-total server-side cap) → single CPU box, 2–3 streams, NO GPU (network-bound; GPU idles).
  Stream-scan ~5 video-sec/s → stream + early-exit, never bulk-download.
- **Method:** `src/harvest_stream.py` (Modal CPU) — probe-first empty-skip (10 seek-frames; skip if
  `p_visible<0.5` everywhere) + **visibility-only gate** + 2 clips/video (60s spread) + early-exit. A/B (same
  8 vids): motion-gate 1 clip-video → visibility-gate **4 clip-videos** (motion gate was discarding
  still-but-visible octopus, which IS good seg/caption data). Detailed coverage ledger records probe points +
  `unscanned_sec` per video so skipped footage can be mined later.
- Full run: 1,769 Nity colour videos; projected ~6–12 h.

## R6 — Skeleton, tracking and kinematics (2026-08-13/15)
Downstream of segmentation: silhouette → anatomical graph (mantle/head/8 arms) → per-arm kinematics.
Code: `src/skeleton/`, `src/segment_to_skeleton.py`, `src/batch_skeleton_motion.py`. Full trail:
`src/SEGMENTATION_LOG.md`.

**Skeleton extraction phases** (frozen `data/skel_bench50`, 50 frames / 20 videos, model-mask input,
tip-correctness guarded). ⚠️ **All rows below are PRE-GATE and are being re-measured** — commit
`8343d2a` added anti-mess gates as unconditional defaults inside `select_arm_paths`, so even the
"baseline" row is no longer reproducible by today's code.
| phase | arms/frame | tip-match |
|---|---|---|
| baseline extractor | 2.82 | 0.876 |
| + selection floors (2.5×/0.30, prefix 0.70) | 3.65 | — |
| + prep (bin_thresh 96, spur width_factor 0.35) | 4.16 | 0.851 |
| + thin768 segmentation (768², Tversky β=0.8) | 4.64 | — |
| + SAM2 mask refine (offline) | 5.04 | 0.792 |
| clean human-mask ceiling (pre-gate) | 6.15 | — |
- Anatomical head (neck constriction on the mantle→crown line) replaced the 2nd-distance-peak head:
  "plausible" 9%→96% at the time; the same check later read 86% (refine, pre-gate) and **80%**
  post-gate. The "plausible" criterion is loose (head merely between mantle and crown) and is being
  replaced by pixel error against human head clicks (`data/skel_bench50/head_gt.json`, `src/skel_head_eval.py`).
- Measured NULLS: hysteresis on the seg probability field (+0.08 arms, noise — the student genuinely
  does not see the thin arms); zoom-2-pass segmentation (4.38 vs 4.60 — crops are OOD for the student).
- **Anti-mess gates (2026-08-15, `8343d2a`)**: unique-suffix (unshared portion ≥ max(2× root radius,
  30% of length)) + tip-thinness (tip clearance ≤ 0.55× root radius). Motivated by visible tangle
  (duplicate late-forking arms, stubs ending in the fat body). Effect on the `skel_bench_latest`
  harness (no tip guard, SAM2 refine): **4.80 → 3.48 arms/frame**, head-plausible 80%.
- ⚠️ **Provenance defect (recorded honestly):** the figure "≥6 arms rose 3/50 → 17/50" quoted in the
  tex came from a live UI header, not a logged artifact, and is **not reproducible**; the current
  on-disk post-gate value is 11/50. Do not cite 17/50.

**Tracking v2** (frozen 10-clip set, `src/skel_eval_tracking.py`, metrics `src/skeleton/track_metrics.py`):
| run | teleport | arms | verdict |
|---|---|---|---|
| baseline (centroid prior) | 14.85% | 5.1 | — |
| flow prior, unchanged gates | 15.71% | 4.9 | ❌ worse — a better prior admits more noisy detections |
| **flow + tightened gates (adopted)** | **14.27%** | 4.9 | ✅ |
| global tracklet association (2 tunings) | 15.07 / 14.35% | 4.2–4.3 | ❌ negative, kept opt-in |
- **occluded_frac = 41.7%**: nearly half of arm samples in naive tracking were evidence-free holds;
  `compute_motion` now emits rows only from `detected`/`fitted` samples (teleport-confident 16.5% >
  overall 14.3% — held nodes' fake stillness was flattering the average).

**Kinematics × behaviour cross-validation** (state-gated arm-tip speed vs VLM behaviour label):
resting 63 · crawling 101 · human-interaction 136 · exploration 139 · reaching 159 px s⁻¹.
⚠️ n = 41 clips only (crawling n=2, resting n=4), no significance test, speeds in crop-pixel units
(camera-distance confound unaddressed), and computed with the PRE-GATE detector.

## R7 — Frozen benchmark suite (2026-08-15)
`BENCHMARKS.md` + `src/benchmarks.py` — one runner, tagged results in `data/benchmarks.json`,
auto-generated LaTeX table. Three suites: **SEG-TEST** (122 human-mask frames from 5 held-out
videos + 19 negatives), **SKEL-50**, **TRACK-10**.
- **Metric fix:** arms/frame is *not* a score — the anti-mess gates improved the output while the
  count fell 27%. SKEL-50's headline is now **arm-tip F1** vs the human mask's protrusions (greedy
  1-1 within 5% of the diagonal, GT capped at 8), penalising both spurious and missed arms.
- **SEG-TEST head-to-head (same leak-free test, 2026-08-15):**
  | model | IoU mean | IoU median | area err | presence AUC |
  |---|---|---|---|---|
  | clean512tv (paper's current headline) | 0.6075 | 0.6661 | 1.07% | 0.718 |
  | **thin768** | **0.6415** | **0.7193** | 1.05% | **0.794** |
  → thin768 wins on every metric; **promoted to the paper's headline seg model 2026-08-15**
  (abstract/contributions/Sec. V now quote 0.642/0.719 + AUC 0.794 from this one model). This also fixes
  a rigor defect the paper review found: the abstract currently pairs clean512tv's IoU with a
  presence AUC measured on a *different* model (v3 aug-LR-ASPP), reading as one system.
- **SKEL-50 first run (thin768, no refine, post-gate):** tip-F1 **0.419** (precision 0.712,
  **recall 0.353**), arms/frame 3.24 → the gates are over-strict; gate grid in progress.
- **Gate frontier (2026-08-15, `src/skel_gate_grid.py`, corrected GT, 50 frames, no refine;
  artifact `data/skel_diag/gate_grid_result.json`)** — this is what the paper's Table II reports:
  | gates (uniq-scale, uniq-frac, tip-ratio) | P | R | F1 | dup | arms |
  |---|---|---|---|---|---|
  | off (0, 0, ∞) | 0.659 | 0.562 | 0.565 | 0.076 | 4.64 |
  | (1.0, 0.15, 0.85) | 0.673 | 0.535 | 0.550 | 0.038 | 4.28 |
  | **(1.5, 0.20, 1.00) SHIPPED** | 0.722 | 0.502 | 0.539 | 0.000 | 3.68 |
  | (2.0, 0.30, 0.55) 1st attempt | 0.760 | 0.468 | 0.520 | 0.000 | 3.24 |
  ⚠️ The **dup rate is defined by the same unique-suffix criterion the gates enforce**, so strictly
  gated rows are 0 by construction — informative mainly for the gates-off row. Gates-off maxes F1
  but restores the visible tangle; we ship the F1-suboptimal point because downstream tracking needs
  arm identities. **NOT claimed:** that the GT fix *changed the ranking* of configurations — for the
  only pair measured under both GTs the order held (old GT: shipped 0.441 > 1st attempt 0.419; new
  GT: 0.539 > 0.520). What is claimed is that pre-fix recall/F1 are incomparable with post-fix ones,
  so any old ranking had to be re-measured.
- **Kinematics n (recomputed 2026-08-15 from `behaviour_records.json`, same filter as
  `make_figures.py` Fig. 4):** 41 clips carry state-gated kinematics; **40** after dropping
  `behavior="uncertain"` — resting 4 · crawling 2 · human 13 · exploration 17 · reaching 4.
  The paper quoted n=40 (the plotted set) until 2026-08-15; **superseded by R12** (146 clips /
  66 videos, video-level statistics). The n=40 medians are retained here only as the historical
  row and are referenced in the paper as the earlier, mildly optimistic estimate.

## Open rigor items (must close before paper claims)
- **No shared human-verified held-out test sets yet.** Segmentation has none; captioning has partial
  (`data/caption_training_set.json`). Every "A beats B" (e.g. seg gate vs CLIP gate) needs a head-to-head
  on ONE verified set. This is the DATA_PLAN Phase-D deliverable.
- **Footage diversity:** all 13,342 clips from **7 dates (one week, Feb 2026)** — the binding limit for
  both students. DATA_PLAN addresses it (harvest more distinct days).

## Figure/asset inventory (for the paper)
- `data/segmentation_demo/*.mp4` + `phase0_out/` — seg before/after (IR tool-bleed, colour bleed, reflection).
- `data/behaviour_dashboard.html` + artifact — activity budget, circadian, stimulus-response charts.
- `results/segmentation/*.log` — training curves / sweep logs.

## Pointers
Plans: `src/DATA_PLAN.md`, `src/SEGMENTATION_PLAN.md`, `src/TRAINING_PLAN.md`. Trails: `src/SEGMENTATION_LOG.md`.

## BLOCKED — VLM-250 reliability study (2026-08-15)
Code is complete and verified end-to-end locally (`src/vlm_reliability.py`: frames extracted,
detector-scored, disjoint frame set selected correctly), but every API call returns
**HTTP 401 `{"error":{"message":"User not found"}}`**. The key in `.env` is present and well-formed
(`sk-or-v1…`, 73 chars) but is rejected by OpenRouter — revoked/expired account, not a code fault.
**Unblocks with a fresh `OPENROUTER_API_KEY`; then just run**
`venv/bin/python3 src/vlm_reliability.py --run` (~$0.17, 250 clips, resumable).
This is the highest-value open rigor item: every headline behavioural result is grouped by labels
this extractor produced, and their reliability is still unmeasured.

## R7b — Segmentation training configuration (for the paper's reproducibility section)
Deployed model `octo_seg_thin768_lraspp.pt`, trained on Modal (A10G) via `src/modal_seg_train.py`:
- **Architecture** LR-ASPP / MobileNetV3-Large head, `base_ch=16`, **3.218 M parameters**
- **Input** 768×768 (`--in-size 768`); **batch** 8; **optimiser** Adam, **lr** 3e-4 with cosine schedule
- **Epochs** 60; **augmentation** "strong" (h-flip, affine rotate/translate/scale applied to image and
  mask in lock-step, brightness/contrast jitter ±25%, mild sensor noise)
- **Loss** focal Tversky (α=0.2, β=0.8 — β>α penalises false negatives, i.e. missed thin arms)
  + 0.5·BCE for stable pixel gradients
- **Data** 5,143 pairs / 183 source videos = human-verified masks + GD+SAM2 teacher labels
  (old-HQ 3,991 + harvest-HQ 740); **split BY SOURCE VIDEO**, with 5 test videos forced out of *all*
  training sources via `--holdout-videos` (leakage guard added after the incident in R3c)
- **Selection** best epoch by validation IoU

## R8 — Test-time temporal fusion: a NEGATIVE for masks, an unexpected WIN for presence (2026-08-15)
Motivation: R3c concluded the mask model fails by *mislocalizing* a correctly-sized blob and asserted
"a temporal student is the real lever" — an untested claim in the paper's limitations. Also a
reporting defect: every published IoU is single-frame, while the deployed skeleton path
(`segment_to_skeleton.py`, `EMA_ALPHA=0.45`) thresholds an EMA-smoothed probability map.
Method: `src/temporal_fusion.py` + `benchmarks.py --fusion {none,ema,flow}`; neighbours t±1,±2 warped
onto t with DIS optical flow, fused by per-pixel median. **Frame-alignment trap avoided:**
`seed_frame` indexes the labeller's `ffmpeg fps=2, scale='min(1024,iw)'` list, NOT raw video frames,
so neighbours are produced by re-running that identical extraction and asserting the regenerated
frame matches the stored labelled image (align_err 0.4–0.55 vs tolerance 12; **0/141 failures**).

SEG-TEST (122 human-mask frames, 5 held-out videos, 19 negatives), thin768:
| fusion | IoU mean | IoU median | area err | presence AUC |
|---|---|---|---|---|
| none (single frame — what the paper reports) | 0.6415 | 0.7193 | 1.05% | 0.794 |
| flow ±2 (DIS-warped median) | **0.5109** | **0.5505** | 1.06% | **0.9495** |

- **NEGATIVE (pre-registered kill criterion met, ≥+0.01 mean IoU required):** temporal fusion does
  **not** fix mislocalization — it makes masks materially worse (−0.131 mean, −0.169 median IoU),
  consistent with boundary blurring on a fast-deforming animal. The paper's limitation must be
  rewritten from "a temporal student is the real lever" to "**test-time** temporal fusion does not
  fix mislocalization" (a temporal *trained* student remains untested, but this removes the cheap
  evidence for it).
- **UNEXPECTED POSITIVE:** presence AUC **0.794 → 0.9495**. Fusion washes out single-frame
  hallucinations that are inconsistent across neighbouring frames, which is exactly the pipeline's
  dominant false-positive mode (reflections / empty tank). Body-area error is unchanged (1.05→1.06%),
  so mask *size* survives while boundary fidelity degrades — a good trade for a presence GATE and a
  bad one for morphology.
  ⚠️ CAVEAT: only **19 negatives** in the leak-free holdout, so this AUC has wide uncertainty; it
  should be re-measured on more negatives before being claimed as headline.

## R9 — Reflection robustness of the DEPLOYED model, measured for the first time (2026-08-15)
The paper reports presence AUC 0.794 for `thin768` and describes the system as reflection-robust.
Those are two different claims: the 0.794 came from **19 empty-TANK negatives on the same cameras as
the positives**; the *reflection* failure mode (Right_Left — the camera sees the room and a mirrored
human through the glass, and the CLIP detector fires at p_visible=1.0) was only ever measured for the
**v3 negatives model**, never for the deployed thin768. R9 closes that gap.

**Leakage assertion (verified, not assumed):** thin768's training set `/dataset_seg_thin768` =
**4,965 images, 0 of them Right_Left** (checked file-by-file on the Modal volume; the camera is
excluded by construction in `auto_segment.py` and absent from the human label set). Reflections are
therefore a leak-free negative source for this model. New: `src/reflection_negatives.py`,
`src/eval_reflection_presence.py`.

### Right_Left is NOT a pure-reflection camera — 10–20% of frames contain the real animal
42 frames reviewed at full resolution before scoring (indices 0-33 are one frame from each of the 34
 distinct Right_Left source videos; 34-41 add second frames).
**3/42 (7%) unmistakably contain the octopus** (animal spread on the glass with its own mirror image
beside it; arms with clearly resolved suckers), and **5 more are ambiguous → up to 19%**. Frames were
labelled conservatively: ambiguous frames are excluded from the negative set, never counted as empty.
- This revises the 2026-07-05 reading of the 235B result ("534/847 clips came back not-present, almost
  all Right_Left reflections → drop Right_Left"). Right_Left is **mostly** empty, not **purely** empty,
  so the standing "drop Right_Left" rule is discarding a real, if sparse, source of animal data.
- It also re-confirms the 2026-06 hard-negative lesson: an assumed-negative pool must be verified
  before it is trained or scored on. Assuming these 30 frames were empty would have injected a 10–20%
  label error straight into the headline metric.
- ⚠️ PROVENANCE: this review was done by an **AI vision model, not a human**. It is staged for human
  confirmation (`data/reflection_negatives/`, contact sheets + `ui/review_hardneg.py`). Treat the exact
  percentage as provisional; the qualitative finding (some frames plainly contain the animal) is not
  in doubt — see `data/reflection_negatives/pilot_1.jpg` #11 and `pilot_4.jpg` #24, #29.

### Presence separation, by negative type (never pooled), thin768 @ threshold 0.5
n for the CI is the number of **videos**, not frames; CIs are cluster-bootstrapped by source video.
| negative type | n | AUC | AUC CI95 (by video) | FP@recall .90 | FP@recall .80 | FP@area>=.01 | median neg area |
|---|---|---|---|---|---|---|---|
| empty tank (same cameras) | 19 frames | 0.7942 | [0.626, 0.932] | 0.316 | 0.210 | 0.316 | 0.0036 |
| reflection (Right_Left)   | 34 frames / 27 videos | **0.9214** | [0.871, 0.964] | 0.235 | 0.118 | 0.176 | 0.0035 |
(positive median mask area 0.0325; the empty-tank AUC reproduces the benchmark's 0.794 exactly, which
validates the harness.)

**CORRECTION (2026-08-15, same day).** I first wrote this up as "the assumed failure mode is
backwards — the model rejects reflections (0.921) better than the empty tank (0.794)". That
comparison does not hold up and is withdrawn. The empty-tank negatives are **19 frames from only 2
source videos, 18 of them from `2026-02-21/183003` alone** — effectively a single-video estimate.
Comparing it against a 27-video reflection estimate is not a like-for-like contrast, so the *ordering*
of the two AUCs cannot be asserted. Reported descriptively only, with no CI and no A-beats-B claim.

**What DOES survive, and it is the more useful finding:**
1. **The reflection failure mode is comfortably handled.** AUC **0.9214** across 27 source videos,
   CI95 [0.871, 0.964], FP at the deployed gate (area>=0.01) **0.176**. This is a properly-powered,
   leak-free measurement and it validates the paper's "reflection-robust" claim for the first time.
2. **The paper's published presence AUC of 0.794 is effectively a ONE-VIDEO number.** That is a defect
   in the benchmark, not a property of the model: 18/19 of its negatives come from one recording, so it
   is a near-meaningless population estimate and its CI [0.626, 0.932] is correspondingly useless. It
   must be either re-based on negatives drawn from many videos, or reported descriptively with the
   n=2-videos caveat stated. **This is the highest-value fix available to the presence section** and
   it was invisible until negatives were counted BY VIDEO rather than by frame.
3. Lesson, consistent with the leakage rule: **count n in videos at every stage, including the
   negatives.** We applied by-video discipline to training splits and to the kinematics statistics but
   never to the negative sets, and a headline benchmark number silently rested on one recording.

**Consequence for the R8 follow-up:** the referee's pre-registered early-stop was AUC(none) >= 0.93 on
reflections. Measured **0.9214** (stable: 0.9173 on the first 24 negatives, 0.9214 on 34 / 27 videos, so the
estimate is not an artifact of the smaller pilot) — just under the line, so the cycle is not killed, but the headroom
for fusion on this negative type is only +0.079, and the referee's second criterion (>= +0.05 AUC gain)
must clear that ceiling to count.

## R10 — CLIP detector vs mask area as a presence gate, head-to-head on ONE verified set (2026-08-15)
Closes the standing rigor item ("every 'A beats B' needs a head-to-head on one verified set"). The
paper's Sec. III-C claim that the detector is the weak presence filter rested on a single anecdote
(534/847 clips came back `octopus not present` in the 235B captioning run). The two gates had never
been scored against each other. New: `src/eval_presence_headtohead.py`; per-frame scores in
`data/presence_headtohead_frames.csv`.

**REFL-28 — the benchmark this had to be run on.** The detector was TRAINED on Right_Left frames, so
its training sessions must be dropped — and dropped from **both** arms, or the head-to-head commits the
sin it exists to fix. Leakage unit is the recording **session** (`date/segment`), not the camera: this
repo splits by session, and two cameras in one session are the same scene, lighting and animal state at
the same instant. Excluding only the *Right_Left* training sessions leaves 4 further sessions the
detector had already seen through another camera (my first run made exactly this error and got
33 frames; corrected → **28 frames / 22 videos**, dropping 6 frames from 5 sessions).
R9's REFL-34 number stands as a separately-scoped measurement (thin768 is Right_Left-free, so it needs
no exclusion); the REFL-28 row is reported beside it, not over it — segmenter AUC 0.9214 (34 fr/27 vid)
vs 0.9315 (28 fr/22 vid).

### Results — REFL-28, identical frame set for every arm
| arm | AUC | CI95 (cluster-bootstrap by video) | FP @ present-recall 0.90 |
|---|---|---|---|
| **mask area** (thin768) | **0.9315** | [0.884, 0.964] | **0.214** |
| CLIP detector `p_visible` | 0.8053 | [0.747, 0.864] | 0.679 |
| combined (rank-product) | 0.9186 | [0.884, 0.946] | 0.286 |

- **Paired ΔAUC (area − detector) = +0.1263, CI95 [+0.056, +0.198] — excludes 0.** Mask area is the
  better reflection gate, and now on a leak-free verified set rather than an anecdote.
- **The asymmetry makes this stronger, not weaker.** The detector is **in-domain** here — it trained on
  **1,519 Right_Left frames across 11 sessions**. The segmenter is **zero-shot**: not one Right_Left
  frame in its 4,965 training images. The zero-shot model beats the in-domain one by 0.126 AUC. Must be
  written as in-domain vs zero-shot wherever it is claimed.
- **Detector false-positive rate at its deployed threshold (p_visible ≥ 0.6): 32% of reflection frames
  and 68% of empty-tank frames.** This is the concrete number behind "the extraction over-extracts",
  replacing the 63%-not-present anecdote.
- **PRE-REGISTERED NEGATIVE — the combined gate is dead.** ΔAUC (combined − area) = −0.0129,
  CI95 [−0.032, +0.009], includes 0. Kill criterion met: the two signals are redundant, **mask area
  alone is the gate**. (One variant only, rank-product, fixed before running; a `min` variant was
  proposed and dropped as fishing on n=22 videos.)
- **Empty tank: descriptive only, no AUC and no CI** — 19 frames but **2 source videos, 18 from one**.
  Median mask area 0.0036 (neg) vs 0.0325 (pos); median p_visible 0.8179 (neg) vs 0.9989 (pos).

CAVEATS that must travel with any use of this: (a) the detector is scored **per frame at p≥0.6**, a
**proxy** — deployment applies that threshold to >50% of frames in a 20 s window, so this is not "the
deployed gate"; (b) the reflection labels are **AI-verified, not human-verified**, so this stays in
PAPER_NOTES and out of the .tex until human review of the 28 frames; (c) read-only study — no gate,
threshold or default was changed.

## R10 CAVEAT — selection bias in the reflection negative set (found by review, 2026-08-15)
**This qualifies R10's headline and must travel with it.** The reflection negatives were sampled from
`src/octopus_clips_verified/*/Right_Left_*.mp4` — clips the *extraction pipeline* selected, and that
pipeline fires only when the CLIP detector marks >50% of a 20 s window as visible at p>=0.6. So the
reflection negatives are, by construction, **enriched for frames the detector got wrong**. Both arms
are scored on the identical frames, but the *set itself* was chosen by a process that used one of the
two arms. The bias runs **against the detector and in favour of mask area** — i.e. in the direction of
R10's result.

I had this backwards in my own framing (I worried the pool would flatter the detector; it flatters the
segmenter). Consequences:
- R10's paired dAUC **+0.1263 [+0.056, +0.198] is an upper bound**, not an unbiased estimate, on the
  mask-area advantage over the detector on reflections. The *sign* is well supported (the detector
  fires on only 32% of these frames at p>=0.6, so the set is not purely its own false positives), but
  the magnitude is inflated by an unquantified amount.
- An unbiased version needs negatives drawn **detector-independently** — uniform random timestamps
  from whole source videos via input-seek, the way `src/harvest_stream.py` probes — not frames from
  clips the extractor already chose.
- The same latent bias sits in R9's REFL-34 segmenter-only numbers, though there it has no head-to-head
  to distort: it makes the reflection set *harder-than-random* for the detector and roughly
  representative for the segmenter.
- Nothing here is withdrawn; the claim is narrowed to "mask area is the better reflection gate, with the
  effect size an upper bound pending detector-independent sampling".

## R8/R9 CAVEAT — every presence number resting on the 19 empty-tank frames inherits the one-video problem
This includes **R8's fusion presence result** (AUC 0.794 -> 0.9685 ema / 0.9495 flow). Those negatives
are the same 19 frames from 2 recordings, so the fusion presence gain is a one-video observation and
cannot carry a CI either. The fusion *mask* results (IoU 0.642 -> 0.547 / 0.511) are unaffected: they
are computed on 122 positives across 5 held-out videos. Re-testing the fusion presence claim on a
properly-powered empty-frame set is a deliverable of the next cycle.

**Paper action taken:** the .tex now states the $19$ empty frames come from two recordings (18 from
one), reports $0.794$ descriptively, and attaches no confidence interval to it.

## R11 — LEAKAGE AUDIT of the headline segmentation model (2026-08-15): PASSES
Prompted by the discovery that all 5 SEG-TEST holdout videos appear in thin768's dataset manifest
(493 frames), which would invalidate every published IoU if those frames had been trained on.
**They were not.** Verified three independent ways, not asserted:
1. The training command (`/tmp/modal_train_thin768.log`) carries `--holdout-videos /data/holdout.txt`
   and the run prints `forced holdout: 5/5 holdout videos present -> excluded from train`.
2. The split was re-derived from scratch (manifest + `--sources human` filter + `RandomState(42)`
   shuffle + `val_frac 0.2` + forced holdout) and reproduces the logged frame counts **exactly**:
   train 3450 / val 1693. No holdout video appears in the training partition.
3. Presence in the *dataset* manifest is not presence in *training* — the manifest lists all pairs and
   the trainer partitions them by source video.
**Conclusion: the headline IoU 0.6415 / 0.7193 and everything derived from it are leak-free.**

**Paper action taken (2026-08-15):** Sec. III-F of the .tex states that all five SEG-TEST videos
appear in the dataset manifest but are excluded from training by the forced-holdout flag, that the
split was re-derived and reproduces the logged partition exactly (3450 train / 1693 val), and that
the audit also found and fixed a logging bug printing wrong video counts for a correct split.

Two by-products:
- **thin768's true training set is 142 source videos** (not 183 = the dataset, and not 147 = the logged
  figure). Enumerated to `data/thin768_train_videos.json` — this is the exclusion list any future
  negative set must be filtered against, since empty-tank negatives fall squarely in its domain.
- **Logging bug found and fixed** in `src/train_segmenter.py`: the split line printed
  `len(vids)-n_val / n_val`, which ignores videos added to val by `--holdout-videos`, so thin768 logged
  "train 147 / val 36" for a split that was actually 142/41. Frame counts were always correct, so the
  printed numbers were mutually inconsistent — which is precisely what made this audit look like a leak
  at first glance. Now prints the actual partition sizes.

## R8-FINAL — threshold sweep resolves the calibration confound; the negative HOLDS (2026-08-15)
R8 compared fusion arms at the shipped threshold 0.5. That is not a fair comparison: a per-pixel
median over warped neighbours suppresses any pixel not confidently octopus in most frames, so the fused
probability map is systematically shrunk relative to a single-frame map, and a fixed 0.5 handicaps it.
Publishing a negative on that basis would have been wrong. `src/fusion_threshold_sweep.py` caches one
probability map per frame per mode and sweeps the binarisation threshold, scoring **each arm at its own
best operating point**. SEG-TEST, 122 positives / 5 held-out videos / 19 empty-tank negatives:

| mode | best IoU | @ t | best presence AUC | @ t | best FP@recall .90 | @ t |
|---|---|---|---|---|---|---|
| **none** (single frame) | **0.6552** | 0.80 | 0.8192 | 0.80 | 0.316 | 0.10 |
| flow ±2 (DIS-warped median) | 0.5109 | 0.50 | 0.9521 | 0.70 | 0.105 | 0.25 |
| **median ±2 (unwarped CONTROL)** | 0.5527 | 0.55 | 0.9629 | 0.70 | **0.000** | 0.25 |
| ema α=0.45 (deployed config) | 0.5866 | 0.40 | **0.9763** | 0.60 | **0.000** | 0.50 |

**1. The mask negative HOLDS.** Best-vs-best, the strongest fusion arm still loses to single frame:
0.5866 vs 0.6552 = **−0.0686**. The confound was real (it narrowed the gap from −0.094 at t=0.5 to
−0.069 best-vs-best) but does not change the conclusion. Test-time temporal fusion does not fix
mislocalization; it degrades mask fidelity. The paper's limitation must read "**test-time** temporal
fusion does not fix mislocalization", and the claim "a temporal student is the real lever" loses its
cheap supporting evidence (a temporal *trained* student remains untested).

**2. MECHANISM ESTABLISHED — optical flow is not the mechanism, and is actively harmful.** The
unwarped control beats flow on **both** metrics (IoU 0.5527 vs 0.5109; AUC 0.9629 vs 0.9521). Plain
temporal averaging supplies the entire benefit; motion compensation subtracts from it, presumably
because DIS flow is unreliable on a deforming, low-texture animal in dim IR and warping errors smear
the map. Without this control the natural write-up would have been "optical-flow fusion improves the
presence gate" — true in isolation and wrong about why. **EMA — the cheapest arm, already deployed and
requiring no flow computation — is the best of the three.**

**3. The shipped binarisation threshold is mildly suboptimal.** Single-frame peaks at t=0.80
(IoU 0.6552) versus 0.6415 at the shipped 0.5: **+0.0137 IoU for free**, no retraining. Not yet applied
— changing it is a pipeline default and needs its own before/after on SKEL-50, since the skeleton
stage consumes these masks and a thinner mask at t=0.8 may cost arms.

**4. The presence gain survives best-vs-best** (0.8192 → 0.9763, +0.157) **but is measured on the
19 empty-tank frames = 2 source videos**, so it inherits the one-video problem and carries no CI.
Re-testing it on EMPTY-V2 is the pending deliverable.

**Paper action taken (2026-08-15).** Sec. V of the .tex now carries "A negative that needed a
control: test-time temporal fusion" — best-vs-best 0.5866 (ema) vs 0.6552 (single frame), the
unwarped-median control beating flow on both IoU (0.5527 vs 0.5109) and presence AUC (0.9629 vs
0.9521), and the explicit statement that the fusion presence gain is NOT headlined because it rests
on the same 19 two-recording empty frames. The limitation now reads "**test-time** temporal fusion
does not fix mislocalisation; a temporally *trained* student remains untested". Cut for space (still
true, still logged here): the +0.0137 IoU available at threshold 0.80 — noted in the .tex header.

## R12 — Kinematics × behaviour cross-validation, recomputed with video-level statistics (2026-08-15)
The paper's headline cross-validation (skeleton kinematics agree with the VLM's behaviour labels) rested
on **n=40 clips with crawling at n=2**, pooled at clip level — pseudo-replication, since several clips
come from one recording. Recomputed properly.

**Sample.** `src/kinematics_sample.py` drew a video-spread stratified sample: **146 clips / 66 distinct
source videos**, ~25 per behaviour class, ≤2 clips per video. Run as 2 shards
(`batch_skeleton_motion.py --shard i/n`, isolated outputs), merged by `src/merge_shards.py`, which
refuses to pool mixed configs — all 147 records carry one stamp:
`{ckpt: octo_seg_thin768_lraspp.pt, fps: 3.0, refine: false, sha: f456768}`.
**Statistics** (`src/kinematics_stats.py`): aggregated to one value per (video, class) before testing,
Kruskal–Wallis + ε², Holm-corrected Mann–Whitney for resting-vs-each, cluster-bootstrap CIs by video.
The two signals are independent: the skeleton pipeline never sees the behaviour label.

### RAW arm-tip speed (px/s), median [CI95 by video]
| behaviour | median | CI95 | videos |
|---|---|---|---|
| Resting / stationary | 53.05 | [31.6, 62.9] | 24 |
| Human / enrichment interaction | 90.22 | [69.0, 134.8] | 24 |
| Crawling | 91.11 | [79.4, 114.8] | 19 |
| Swimming / jetting | 107.29 | [92.2, 124.4] | 15 |
| Exploration / manipulation | 112.40 | [90.6, 132.0] | 25 |
| Reaching out of water | 141.05 | [127.8, 166.9] | 25 |

**Kruskal–Wallis H=33.18, p=3.5e-06, ε²=0.224** (N=132 video-class units, k=6). All five
resting-vs-X contrasts significant after Holm correction (p_holm 0.0086 → 5e-05).

### SCALE-INVARIANT speed (body-lengths/s = speed ÷ arm-spread)
Resting 0.17 [0.1,0.2] · Human 0.26 [0.2,0.5] · Reaching 0.31 [0.2,0.3] · Exploration 0.36 [0.3,0.5] ·
Crawling 0.36 [0.3,0.4] · Swimming 0.37 [0.3,0.4].
**H=21.40, p=6.8e-04, ε²=0.130**; all five contrasts still significant after Holm.

**PRE-REGISTERED KILL CRITERION (p>0.05 or ε²<0.06) NOT MET — the result stands** and is now
properly powered. It also survives in scale-invariant units, which matters because raw px/s is
confounded by apparent size (distance from camera); normalising by arm-spread removes that.

**Nuance worth reporting: `reaching out of water` is the FASTEST in raw px/s but only 3rd in
body-lengths/s.** Reaching is performed with an extended body, so a large part of its raw tip speed is
extended posture rather than faster motion. Reporting only raw px/s would have overstated it. Swimming
and crawling, by contrast, rise in the normalised ranking.

Versus the old n=40 figure (resting 63 → reaching 159 px/s): same direction, more conservative
magnitudes (53 → 141) — the small-sample version was mildly optimistic, not wrong.

CAVEAT: behaviour labels come from the VLM structured extractor, whose reliability study (VLM-250) is
still BLOCKED on a revoked OpenRouter key. This validates that kinematics track the labels, not that
the labels are correct. Speeds are px/s in crop space (no px→cm calibration).

**Paper action taken (2026-08-15):** R12 replaced the n=40 cross-validation everywhere in the .tex
(abstract, contribution 4, Sec. VI, Fig. 5, limitations). Fig. 5 is now a two-panel figure generated
by `OCEANS_2026/make_figures.py` **through `src/kinematics_stats.collect` on
`data/skeleton_motion_study.json`** — the same loader that produced `data/kinematics_stats.json`, so
the plotted medians are the published medians by construction; it plots per-(video,class) medians,
not clips. The old 63→159 px/s figures are cited in the paper only as the earlier, less conservative
clip-pooled estimate.

## R13 — EMPTY-V2: the presence benchmark repaired, and 0.794 was WRONG as well as under-powered (2026-08-15)
The paper's presence AUC of 0.794 came from 19 empty frames drawn from **2 recordings (18 from one)**.
EMPTY-V2 replaces it with a properly-powered, **detector-independent**, verified set.

**Construction** (`src/empty_negatives.py`). Frames are grabbed at **uniform random timestamps from
whole server videos** by input-seek — never from clips the extractor selected, because extracted clips
exist only where the CLIP detector fired, which enriches the set with that detector's own false
positives (the bias now recorded against R10). Excluded: thin768's 132 training sessions and the
detector's 32 sessions, matched on `(date, HHMM)`. All 120 frames reviewed at full resolution before
scoring: **8 (6.7%) unmistakably contain the animal, 7 more ambiguous (12.5% total)**; ambiguous frames
are excluded rather than assumed empty. Result: **105 verified negatives / 53 source videos.**

**Two sampling defects were caught and fixed before any number was computed** — both would have
produced a confident, wrong result:
1. *Single-date concentration.* The first run drew all 40 frames from 20 recordings on ONE date and ONE
   camera, because one directory listing supplied every recording before the loop advanced —
   reproducing exactly the concentration defect this benchmark exists to fix. Fixed with a per-listing
   cap and round-robin.
2. *Domain mismatch.* The second run crawled both Nity collections, which are **two different physical
   setups** — the 2026-02 lab tank (the positives' domain) and a 2025-09 collection in a different room
   with a different tank. Separating "2026 tank containing an octopus" from "2025 room containing none"
   would have measured SCENE DIFFERENCE and returned a flatteringly high AUC for the wrong reason.
   Fixed by matching the collection; the cross-setup frames are kept separately
   (`data/empty_negatives_crossdomain/`) as a distinct question (FP in an unseen environment).

### Results — thin768, threshold 0.5, CIs cluster-bootstrapped by source video
| negative set | n | videos | AUC | CI95 | FP@R.90 | FP@R.80 | FP@area>=.01 |
|---|---|---|---|---|---|---|---|
| **EMPTY-V2 (empty frames, multi-video)** | 105 | **53** | **0.9170** | [0.839, 0.962] | 0.171 | 0.086 | 0.143 |
| reflection REFL-34 | 34 | 27 | 0.9214 | [0.826, 0.966] | 0.235 | 0.118 | 0.176 |
| old SEG-TEST empty-tank | 19 | **2** | *descriptive only* | — | — | — | — |

**1. The published 0.794 was not merely under-powered — it was PESSIMISTIC.** Properly measured across
53 recordings the model separates empty frames from present frames at **0.917**, not 0.794. The old
figure was dominated by a single unusually hard recording. The paper's presence claim is stronger than
what it currently reports, and can now carry a confidence interval.

**2. NULL RESULT, and it settles the question I withdrew in R9.** Empty-frame AUC 0.9170 and reflection
AUC 0.9214 are statistically indistinguishable (CIs almost entirely overlapping). There is **no
measurable difference between the two failure modes** — neither "reflections are the dominant problem"
(the paper's original framing) nor "the failure mode is backwards" (my withdrawn claim) is supported.
Recording the null explicitly so neither framing returns.

**3. BUG FIXED in my own earlier statistics.** `areas_from_cache` set each positive's `video` to its
image FILENAME, so the cluster bootstrap treated 122 positives as 122 independent recordings when they
come from 5. That understates clustering and yields CIs that are too narrow *in the flattering
direction*. R9's reflection CI [0.871, 0.964] is corrected to **[0.826, 0.966]**; the point estimate
0.9214 is unchanged. Fixed in `src/eval_reflection_presence.py`. **Every cluster bootstrap must group
BOTH arms by true source video — grouping only the negatives is not enough.**

PENDING: re-test R8's fusion presence gain (0.794 -> 0.9685 ema) on EMPTY-V2 — it rests on the same
19 two-recording frames. Needs neighbour frames per negative, i.e. another server pass.
CAVEAT: labels are AI-verified, not human-verified — PAPER_NOTES only until human confirmation.

## R13-FINAL — EMPTY-V2 HUMAN-VERIFIED (2026-08-15). Now citable in the paper.
All **120/120** frames confirmed by a human via `ui/verify_negatives.py` (port 8020). Human labels are
stored in a separate `human` field; the model's labels stay in `review`, so agreement is measurable.

### Model-vs-human agreement: 102/120 = 85.0%
| my label | human said | n | direction |
|---|---|---|---|
| empty | **octopus present** | **9** | **contamination — I would have scored 9 animal-containing frames as negatives** |
| octopus present | empty | 2 | I over-called |
| ambiguous | octopus present | 6 | resolved |
| ambiguous | empty | 1 | resolved |

Final human set: **99 empty / 21 present / 0 ambiguous**. My proposed negative set of 105 was
**8.6% contaminated**. This is the third time in this project that an assumed-empty pool turned out to
contain the animal (166/232 in the 2026-06 hard-negative mining; 7-19% in the reflection pilot).
**An AI-verified negative set is not a substitute for a human one** — 85% agreement sounds high, but it
is the 15% that decides the number.

### Headline (HUMAN-verified, thin768, threshold 0.5, CI cluster-bootstrapped by source video)
| negative set | n | videos | AUC | CI95 | FP@R.90 | FP@area>=.01 |
|---|---|---|---|---|---|---|
| **EMPTY-V2 human-verified** | **99** | **53** | **0.9093** | **[0.833, 0.957]** | 0.182 | 0.152 |
| EMPTY-V2 as I had labelled it | 105 | 53 | 0.9170 | [0.839, 0.962] | 0.171 | 0.143 |
| old SEG-TEST empty-tank (paper's 0.794) | 19 | **2** | descriptive only | — | — | — |

**The paper's 0.794 should become 0.909 [0.833, 0.957] on 53 recordings.** Even after removing my
contamination the properly-powered figure is far above the published one, which was dominated by a
single hard recording.

### LESSON — contamination does not always deflate a metric; here it INFLATED it
Intuitively, octopus-containing frames scored as negatives should *hurt* the AUC. They did the
opposite (0.9170 -> 0.9093 when removed) because **7 of the 9 frames I missed also fell below the
deployed gate — the segmenter missed the animal in exactly the frames I did**. They therefore looked
like unusually clean negatives and flattered the score. When the reviewer and the model share a blind
spot, contamination masquerades as good performance. Do not assume label noise is conservative.

### Deployment statistic worth reporting (uniformly-sampled footage, not curated clips)
At the deployed gate (mask area >= 0.01): **11/21 (52%) of human-confirmed present frames fire**, and
**15/99 (15%) of confirmed empty frames fire**. Note these present frames are uniformly sampled, so
many show the animal small, dim or half-denned — a much harder recall test than SEG-TEST's curated
positives, and a more honest picture of what the gate does on raw footage.

### STILL AI-ONLY: the reflection set (R9 / R10)
`data/reflection_negatives/` (42 frames) has **0/42** human labels. Given 85% agreement and 8.6%
contamination on EMPTY-V2, R9's reflection AUC 0.9214 and R10's head-to-head dAUC +0.1263 must stay
**out of the paper** and be treated as provisional until the same pass is run on them.

## R14 — ALL presence results HUMAN-VERIFIED (2026-08-15). R9/R10/R13 are now citable.
Both staged negative sets fully human-labelled via `ui/verify_negatives.py`:
**EMPTY-V2 120/120** (83% model-human agreement) and **reflection 42/42** (88%).

### Final numbers — human-verified negatives, thin768 @ 0.5, CIs cluster-bootstrapped by source video
| negative set | n | videos | AUC | CI95 | FP@R.90 | FP@area>=.01 |
|---|---|---|---|---|---|---|
| **EMPTY-V2 (empty frames)** | 97 | 52 | **0.9070** | [0.826, 0.958] | 0.186 | 0.155 |
| **reflection** | 36 | 29 | **0.9064** | [0.806, 0.956] | 0.278 | 0.222 |
| old SEG-TEST empty-tank (paper's 0.794) | 19 | **2** | descriptive only | — | — | — |

**1. The paper's 0.794 becomes 0.907 [0.826, 0.958] across 52 recordings.** Human-verified,
detector-independent, multi-video. The old figure was a single-recording artifact.

**2. The NULL RESULT is now airtight: 0.9070 vs 0.9064.** Empty frames and reflections are equally
hard for the model — a near-exact tie on human labels. Neither "reflections are the dominant
false-positive source" (the paper's original framing) nor "the failure mode is backwards" (my
withdrawn claim) is supported. State the null; drop both framings.

**3. Head-to-head survives human verification** (human-verified reflection negatives, detector
training sessions excluded → 30 frames / 24 videos):
| arm | AUC | CI95 | FP@R.90 |
|---|---|---|---|
| mask area (**zero-shot** on this camera) | **0.9126** | [0.821, 0.962] | 0.267 |
| CLIP detector `p_visible` (**in-domain**, 1,519 Right_Left training frames) | 0.7989 | [0.598, 0.882] | 0.700 |

Paired **ΔAUC +0.1340, CI95 [+0.024, +0.308] — still excludes 0.** The wide CI (24 videos) is the
honest weakness; claim the ordering, not the magnitude. The R10 selection-bias caveat still applies:
these negatives come from extractor-selected clips, so the effect size remains an upper bound.

### My labelling erred in the FLATTERING direction on both sets
| set | my labels | human labels | shift |
|---|---|---|---|
| EMPTY-V2 | 0.9170 | 0.9070 | −0.0100 |
| reflection | 0.9214 | 0.9064 | −0.0150 |
Every AI-verified figure I produced was optimistic. The mechanism (documented in R13-FINAL) is a
**shared blind spot**: frames where I missed the animal are disproportionately frames where the
segmenter also missed it, so they masquerade as unusually clean negatives. **AI verification of a
negative set is systematically biased toward the model being evaluated** — it cannot substitute for
human labels, and the bias has a predictable sign.

### Bonus — intra-rater reliability, measured by accident
A UI bug (`first_unreviewed` wrapping to 0 on a completed set + the set dropdown resetting on reload)
sent a session intended for the reflection set back over EMPTY-V2, re-labelling all 120 frames. That
produced an unintended **test-retest**: the same human, the same frames, twice, independently.
**118/120 = 98.3% self-consistency** (differing only on `empty_0014`, `empty_0015`).
This bounds label noise: the human agrees with themselves at 98.3% while agreeing with the model at
83%, so **the model-human gap is model error, not rater instability**. Both UI bugs are fixed.

## R15 — VLM-250: the behaviour labels are only MODERATELY self-consistent (2026-08-15)
UNBLOCKED by a fresh API key. `src/vlm_reliability.py --run` re-ran the structured extractor over the
frozen 250-clip / 140-video sample using a **disjoint set of input frames** (detector ranks
N_KEEP..2*N_KEEP instead of the top N_KEEP). 250/250 succeeded, **$0.165**. Analysis:
`src/vlm_reliability_stats.py` → `data/vlm_reliability_stats.json`.

**What this measures: frame-sampling sensitivity — does a label survive being shown different clear
frames from the same clip. It is CONSISTENCY, not accuracy.** Consistency upper-bounds accuracy; a
perfectly consistent extractor can still be consistently wrong. Never write "accurate" for these.

| field | raw agreement | Cohen's κ | κ CI95 (by video) |
|---|---|---|---|
| **behavior (7-class)** | 0.652 | **0.552** | [0.472, 0.624] |
| posture | 0.684 | 0.510 | [0.423, 0.594] |
| location | 0.632 | 0.511 | [0.426, 0.591] |
| context | 0.756 | 0.585 | [0.482, 0.676] |
| activity | 0.752 | 0.592 | [0.507, 0.675] |
| present | 0.848 | 0.413 | [0.229, 0.579] |
| body_color | 0.868 | 0.744 | [0.658, 0.819] |
| color_or_texture_change | 1.000 | 1.000 | — **ARTIFACT, see below** |

**1. THE HEADLINE CAVEAT. Every behavioural result in this paper — activity budget, circadian
profile, human-presence stimulus response, kinematics × behaviour — is grouped by a label with
κ ≈ 0.55 ("moderate" on Landis–Koch).** Change which frames the model sees and roughly a third of
behaviour labels change. This must appear in the paper's limitations; it was previously unmeasured
and simply assumed.

**2. Rare classes are the least stable — and they are exactly the classes carrying the fewest videos:**
Exploration 72.4% · Resting 71.2% · Human-interaction 60.7% · Crawling 53.8% · Reaching 47.6% ·
**Swimming/jetting 42.9%**. So the classes with the widest kinematic CIs also have the least reliable
labels; treat per-class claims about swimming/reaching with corresponding caution.

**3. This STRENGTHENS R12 rather than undermining it.** Label noise that is independent of kinematics
causes regression dilution — it biases a group-difference test **toward the null**. Finding
KW p=3.5e-06, ε²=0.224 *despite* labels at κ=0.55 means the true separation is likely larger than
measured, not smaller. CAVEAT: this holds only if the mislabelling is independent of motion; if fast
clips are preferentially labelled "swimming", noise could inflate instead. Not currently testable.

**4. `color_or_texture_change` is a DEAD FIELD, and its κ=1.000 is an artifact.** Its value is
**100% determined by the greyscale gate** (IR clips forced to `uncertain` = 151; colour clips `none`
= 99) — the perfect agreement measures the determinism of a preprocessing rule, not model judgement.
More damning: across **99 colour clips the model reported a colour/texture change exactly zero times**.
The field carries no information and should be dropped from the schema or redesigned. The stats script
now detects and flags gate-determined fields rather than reporting them as excellent reliability.

**5. `present` has high raw agreement (84.8%) but low κ (0.413)** because the class is imbalanced
(~87% present). Report κ, not raw agreement, for this field. Abstention (`uncertain` behaviour) ran at
17.6% in condition B.

## PAPER INTEGRATION LOG — R13/R14/R15 folded into the .tex (2026-08-15)
What is now cited in `OCEANS_2026/octopus_behaviour_pipeline.tex`, and what is deliberately not.

**Cited (human-verified only, from `data/presence_human_verified.json` / `data/vlm_reliability_stats.json`):**
- Sec. V-B, new Table `tab:presence` + three run-in paragraphs: EMPTY-V2 **0.907 [0.826,0.958]**
  (97 frames / 52 videos), reflection **0.906 [0.806,0.956]** (36 / 29), head-to-head mask area
  **0.913 [0.821,0.962]** vs CLIP probe **0.799 [0.598,0.882]** (30 / 24), paired **ΔAUC +0.134
  [+0.024,+0.308]**, plus deployed-gate FP 15.5% (empty) / 22.2% (reflection).
- The NULL RESULT is stated as a null ("equally hard", no ordering asserted in either direction) and
  the "reflections are the dominant false-positive source" framing is removed everywhere it appeared
  (abstract, Sec. V-B). Abstract + contribution 5 updated because the headline genuinely changed.
- Sec. III-C's "63% not present" anecdote is REPLACED by a pointer to the head-to-head.
- Limitations: R15 κ=0.552 [0.472,0.624] (raw 0.652), stated as **consistency, not accuracy**, with
  the regression-dilution upside (and its independence caveat) and the rare-class instability
  (swimming/jetting 43%, n=7; reaching 48%, n=21). Abstract carries κ=0.55 as a bound.
- Sec. III-C notes `color_or_texture_change` never fired across the 99 colour clips.

**Refused / deliberately NOT written:**
- Every AI-labelled figure (0.9170, 0.9214, 0.9315, 0.8053, +0.1263) — superseded by R14.
- R10's **combined rank-product gate** negative (ΔAUC −0.0129 [−0.032,+0.009]): it has no
  human-verified re-measurement, so it stays in PAPER_NOTES. Re-run it on the human labels if the
  paper should claim "the two signals are redundant".
- The withdrawn "failure mode is backwards" claim, and the unprovenanced "17/50".
- R15's κ=1.000 for `color_or_texture_change` (gate artifact — only the "never fired" fact is cited).

**Page count:** the draft is now **7 pages** (was 6). The added table + presence prose + κ limitation
is ~1.15 columns; nothing was cut to recover it, per the instruction to keep caveats over pages.
The 7th page holds the limitations tail, conclusion, ethics statement and references.

**Also changed:** `src/benchmarks.py` prints the SEG-TEST presence row as "vs 19 empty, 2 recordings"
and its caption now says the row is superseded — so the auto-generated table can no longer be read as
a population estimate. Regenerate with `--latex-from paper_current`.

## R16 — combined-gate negative CONFIRMED on human labels (2026-08-15)
The paper-integration pass correctly refused to cite R10's pre-registered combined-gate negative,
because R14 re-scored only the two single-signal arms on human labels — citing the AI-labelled
Δ alongside human-verified arms would have mixed label provenances inside one comparison. Re-run now
on the same human-verified frames (122 positives, 30 reflection negatives / 24 videos, detector
training sessions excluded), same pre-registered variant (rank-product, fixed before the first run):

| arm | AUC | FP@recall .90 |
|---|---|---|
| mask area | 0.9126 | 0.267 |
| CLIP detector | 0.7989 | 0.700 |
| combined (rank-product) | 0.9120 | 0.333 |

**Paired ΔAUC (combined − mask area) = −0.0050, CI95 [−0.057, +0.032] — includes 0.**
Kill criterion met on human labels exactly as it was on AI labels (−0.0129 [−0.032, +0.009]).
**NEGATIVE CONFIRMED: the two presence signals are redundant; mask area alone is the gate.** Adding
the detector does not help and costs FP@R.90 (0.267 → 0.333). Now citable with a single label
provenance throughout.

## R17 — Uncertainty for the BEHAVIOURAL findings (day-clustered). Two of three hold; one is soft.
The paper's vision results all carry CIs cluster-bootstrapped by source video, while the behavioural
findings — the activity budget, circadian profile and stimulus response — were bare point estimates,
and the most quotable claim in the paper ("human presence nearly doubles movement") had **no test at
all**. `src/behaviour_uncertainty.py` fixes that. The independent unit is the **recording day** (7 of
them), not the clip: clips within a day share lighting, the animal's state and whether a human was in
the room. n=7 clusters is few, and the resulting widths are the honest measure of what one week supports.

### Stimulus response — STRONGEST result in the behavioural section
| metric | human present | absent | paired diff | CI95 (by day) | days higher | sign test |
|---|---|---|---|---|---|---|
| mean motion | 0.0856 | 0.0406 | **+0.0450** | [+0.0279, +0.0610] | **7/7** | p=0.0078 |
| arousal index | 0.6574 | 0.4585 | **+0.1989** | [+0.1018, +0.3097] | **7/7** | p=0.0078 |
Consistent in direction on **every single recording day**; exact paired sign test p=0.0078 (the floor
at n=7). This claim is now the best-supported behavioural finding, not the weakest.

### Circadian — holds, and the magnitude is larger than reported
Afternoon (13–19h) **37.5%** vs overnight (00–05h) **2.8%** = **13.4× ratio**; peak hour 17:00 at
**46.4%** (paper says ~45%, confirmed); dawn bump at 05–06h confirmed (10.4%, 13.7%).
Afternoon > overnight on **4/4** days with enough exposure in both windows. Per-day spread at the peak
hour is wide, though: **[1, 26, 36, 59, 67, 74, 88]%** — the direction is robust, the level is not.

**Two computation traps, both hit on the first attempt** (worth keeping, both silently produce
plausible-looking numbers):
1. `video_timeline` is a clip **offset** (mm:ss–mm:ss), not a clock time. Parsing it as an hour makes
   every hour look identical (~95–100% everywhere). Absolute clock time comes from the index
   `segment` (HHMMSS) + the window's `start_sec`.
2. The exposure denominator must be restricted to the **same recording days** as the numerator. The
   index holds **13,342** extracted clips but only 3,205 were behaviourally analysed; off-date clips in
   the denominator deflate the rate. Here it is only a 2.6% effect (12,995 of 13,342 are on-date), but
   the *shape* of the curve is the published claim, so exposure must be matched, not assumed benign.

### Activity budget — the SOFT finding; report intervals, not point estimates
| behaviour | share | CI95 (by day) | per-day range |
|---|---|---|---|
| Exploration / manipulation | 41.2% | [31.9, 50.9] | 21.5–56.9% |
| Resting / stationary | 32.5% | [20.2, 51.0] | **16.2–73.1%** |
| Human / enrichment interaction | 13.7% | [7.9, 18.0] | 3.3–34.6% |
| Reaching out of water | 9.0% | [5.7, 11.5] | 2.1–12.5% |
| Crawling | 2.2% | [0.8, 4.1] | 0.0–7.7% |
| Swimming / jetting | 1.4% | [0.4, 2.3] | 0.0–2.8% |
**Resting ranges 16%–73% across seven days.** Quoting "33% resting" as a characteristic of the animal
is not supportable from one week; the paper must give the interval or drop the precision. This is the
clearest quantitative argument for extending to the harvested ~209-day corpus.

## R18 — Related work: HideAndSeg, and the two motivations the paper was missing (2026-08-19)
Two load-bearing claims were absent from v1 and are now in **v2**
(`OCEANS_2026/octopus_behaviour_pipeline_v2.tex`; v1 frozen as the fallback):

**1. There is no public annotated octopus segmentation dataset.** Never stated, so the entire
teacher→student architecture was unmotivated. Independently confirmed by HideAndSeg: *"the absence of
large-scale, publicly available, annotated video datasets for octopus segmentation, which prevents
standard supervised fine-tuning and evaluation."* They assert it three times as their own motivation
and release neither data nor code, so **the claim remains true after their paper** and is citable.

**2. Foundation models fail out of the box here — now quantified** (all pre-existing measurements):
OWLv2 231/232 at thr 0.10 with no class separation (vs the human outcome that 166/232 actually
contained the animal); zero-shot CLIP presence scoring abandoned as unreliable (the working detector
is a *trained* probe on frozen features); GD+SAM2 per-frame IR bright-tool bleed 11.8%→6.5% and
colour background bleed 15.5%→5.8% once fixed by confidence-seeded video propagation; GroundingDINO
13% IR clip acceptance; reflection seed confidence ~0.50 vs 0.74–0.89. **NOTE: the OWLv2 figure was
in AGENTS.md but had never been mirrored into this ledger — now recorded here** (artefacts:
`data/hard_negatives/_detector_verify.json`, `review_decisions.csv`).

### HideAndSeg (arXiv:2511.04426, Nov 2025) — closest prior art, verified against their own tables
de Aguiar, Andrade, Santos & Gois (UFABC). SAM2 + YOLOv11, 148 videos / 366,514 frames of juvenile
*O. insularis*, three Brazilian sites, handheld colour GoPro/Canon, natural habitat.

**Their headline DICE 0.9677 / IoU 0.9383 is a single-prompted-frame number, not a video result.**
Verified directly from their tables, not inferred:
- Their column header reads **"Supervised Metrics (first frame)"**, and frame 1 is *always* prompted
  ("uniformly sampled... beginning with the first frame").
- In their Table 2, DICE and IoU are **bit-identical including standard deviations** (0.9677±0.0191 /
  0.9383±0.0349) across the 5-, 10- and 20-frame conditions. A 4× change in prompt count cannot leave
  a genuine video metric unchanged to four decimals. They read this as "saturation"; it is
  insensitivity — the metric only ever looks at frame 1.
- In their Table 1, adding a second annotation frame changes DICE/IoU by **exactly zero**
  (0.9405/0.8965 in both rows), which cannot affect the frame-1 mask.
- **Not one frame beyond frame 1 is compared to a human mask**, in a paper about video propagation
  whose own figures show propagation failing three ways.
- Their unsupervised DICE_t is **anti-correlated with quality on their own data**: the config with the
  highest DICE_t (0.9747) has the worst supervised DICE (0.6057).

**So 0.9383 vs our 0.6415 is not a like-for-like comparison** — ~17 prompted frames vs 122 human masks
across five fully held-out videos, clear daylight colour vs dim aquarium IR with glass reflections,
*O. insularis* juveniles vs adult *O. vulgaris*. v2 states their number openly and explains the
protocol difference rather than letting a reviewer make the naive comparison.

**Credit where due, and worth citing positively:** their core insight is correct — per-frame
re-detection breaks the failure mode where a propagation tracker, once contaminated, never recovers.
Reproduce or contrast that, do not re-derive it.

**They leave our territory open** (verified by full-text search): no aquarium footage, **no infrared at
all**, no distillation (they deploy ~250M params and hit a compute wall, solved by prompting *less*
rather than shrinking), no skeleton/pose, and **no behaviour analysis** — they stop at masks and call
it "the first step towards automating the detection of animal behavior."

## R19 — TEACHER vs HUMAN masks: the missing measurement. The student BEATS the per-frame teacher (2026-08-20)
Every segmentation IoU we had was student-vs-human (SEG-TEST) or student-vs-teacher (agreement). The
**zero-shot teacher itself was never scored against human masks**, so "the ceiling is teacher-label
quality" was a hypothesis, not a measurement. `src/eval_teacher_masks.py` closes it. Both arms see the
SAME single frame and the SAME human mask; the frozen set is *imported* from `benchmarks.py` (not
re-derived), and the student reproduced its published **0.6415 exactly**, which validates the harness.

Teacher = GroundingDINO-tiny box → SAM2 (box + pos/neg points) → largest blob, **per frame**.
Student = `octo_seg_thin768_lraspp.pt` (the `paper_current` model). 122 frames / 5 holdout videos.

| arm | IoU mean | IoU median |
|---|---|---|
| teacher (zero-shot, per-frame) | **0.3740** | 0.3049 |
| **student (ours)** | **0.6415** | 0.7193 |

**Paired Δ (teacher − student) = −0.2675, CI95 [−0.313, −0.136]**, bootstrap **clustered by source
video**, excludes 0. Per-frame wins: student 82 / teacher 35.

### The conditional breakdown is the actual result — the headline alone misleads
| subset | n | teacher | student |
|---|---|---|---|
| all frames | 122 | 0.3740 | **0.6415** |
| teacher detected something | 92 | 0.4960 | **0.6289** |
| teacher found NOTHING (scored 0) | 30 (25%) | 0.0000 | **0.6803** |
| **teacher conf ≥ 0.60 (its own `MIN_SEED_CONF` gate)** | 21 | **0.7259** | 0.6574 |
| teacher conf < 0.60 (gate would reject) | 101 (83%) | 0.3009 | **0.6382** |

**1. The teacher is high-precision / low-recall; the student is uniformly competent.** When GroundingDINO
clears its own 0.60 gate the teacher is genuinely BETTER than the student (0.726 vs 0.657) — but that
happens on only 21 of 122 frames. It finds nothing at all on 25%, and 83% of frames fall below the gate
(median conf 0.424). Distillation converted a **sparse, high-quality** signal into **dense, reliable**
coverage. That is the distillation working, not failing.

**2. "Teacher-label quality is the ceiling" SURVIVES, sharpened.** The labels the student actually
trained on come from confident seed frames (+ propagation), and the teacher at that operating point
scores **0.726**. The student sits at 0.6415 → **~0.08 of headroom, not a lot**. So the plateau is
consistent with label quality, and more clips will not move it.

**3. The conf≥0.60 subset is not merely "easy frames".** The student scores 0.657 there vs 0.6415
overall — essentially flat. So the teacher's 0.726 on that subset is a real quality signal, not an
artifact of those frames being easier.

### Caveats — state these with the numbers
- **Per-frame teacher, not propagated-label quality.** The training labels used SAM2 *video propagation*
  seeded by the clip's most-confident frame, which Phase 0 showed beats per-frame box→SAM. The 0.374 row
  is the per-frame teacher (the apples-to-apples arm, since the student is per-frame too) and must NOT be
  quoted as the quality of the labels the student learned from — those are better. Measuring propagated
  labels on SEG-TEST needs 122 clips × 40 frames = **4,880 GD calls (~3.4 h locally)**; not run yet.
- **5 clusters is a weak bootstrap.** The CI [−0.313, −0.136] is wide; claim the ordering, not the size.
- Human-labelled frames were chosen for *labelling*, not for GD confidence, so the low median conf is
  expected and is not evidence the teacher is broken in its real pipeline.
- `2026-02-21/183003` is bad for BOTH (teacher 0.296 / student 0.310) — it is also the source of 18 of
  SEG-TEST's 19 empty-tank negatives.
- Per camera, the student's lead is smallest on Right_Front (0.553 vs 0.415) and largest on Right_Back
  (0.782 vs 0.463).

Raw: `data/teacher_vs_human_masks.json` (includes per-frame IoU, conf and areas).

### R19 folded into the paper (v2, 2026-08-20)
- **v2 is the live draft** (`octopus_behaviour_pipeline_v2.tex`); v1 stays frozen at 7 pages.
- New `Table \ref{tab:teacher}` in Sec. Segmentation Student: 3 rows (all 122 / teacher-found-nothing 30 /
  seed-conf>=0.60 21), teacher vs student mask IoU, with the lower-bound caveat in the caption itself.
- New paragraph **"What the teacher alone would give"** carrying the paired Δ −0.267 [−0.313,−0.136],
  the 25%-no-detection fact, the reversal at conf>=0.60, and the ~0.08 headroom → plateau attributed to
  label quality not corpus size.
- The Sec. III "Why distil rather than prompt" passage now points at the table, upgrading its
  "usable only when gated" claim from qualitative to measured.
- **Rebuild: 8 pages, 0 errors, 0 overfull, 0 undefined refs** (unchanged page count — the table fit).
- NOT added: the OWLv2 / VLM-presence / caption base-vs-LoRA rows as a combined "zero-shot" table. Those
  arms are scored on DIFFERENT sets, so pooling them into one table would break the identical-set rule.
  OWLv2 and zero-shot CLIP already appear as prose in "Why distil rather than prompt"; the caption
  base→LoRA numbers are already prose in Sec. Distillation.

## R20 — HideAndSeg read in FULL TEXT (2026-08-20). R18 verified; one correction; the positioning
Read from arXiv HTML (2511.04426v1), not from the abstract. **All four R18 claims confirmed verbatim.**

### Their numbers, exact (Table 1 = SAM2 alone; Table 2 = full automated pipeline, test set)
| model / condition | annotated frames | DICE_t | NC_t | DICE | IoU |
|---|---|---|---|---|---|
| Small, 1st frame, pos clicks | 1 | **0.9747** | 10.65 | **0.6057** | 0.5330 |
| Small, 1st frame, pos+neg | 1 | 0.9671 | 4.05 | 0.7990 | 0.7266 |
| Small, + additional frame | 2 | 0.9667 | 5.86 | **0.7990** | **0.7266** |
| Large, 1st frame, pos clicks | 1 | 0.9688 | 2.97 | 0.9129 | 0.8592 |
| Large, 1st frame, pos+neg | 1 | 0.9664 | 2.30 | 0.9405 | 0.8965 |
| Large, + additional frame | 2 | 0.9672 | 2.17 | **0.9405** | **0.8965** |
| Manual (pipeline) | 2 | 0.9695 | 2.75 | 0.8997 | 0.8582 |
| YOLO | 5 | 0.9709 | 2.16 | **0.9677±0.0191** | **0.9383±0.0349** |
| YOLO | 10 | 0.9709 | 2.23 | **0.9677±0.0191** | **0.9383±0.0349** |
| YOLO | 20 | 0.9706 | 2.21 | **0.9677±0.0191** | **0.9383±0.0349** |

Confirmed: (1) *"we performed manual segmentation on the initial frame of all videos"* + tables labelled
**"Supervised Metrics (first frame)"** — supervised DICE/IoU is **frame 1 only, averaged over videos**;
(2) DICE/IoU **bit-identical to 4 dp incl. SD across 5/10/20** annotated frames; (3) the "+ additional
frame" rows change DICE/IoU by **exactly zero** in both Small and Large; (4) DICE_t is **anti-correlated**
with quality — the best DICE_t (0.9747) is the worst DICE (0.6057).

### CORRECTION to R18
R18 wrote "~17 prompted frames". **The paper never reports the test-set VIDEO count** — it splits by
FRAMES (train 212,924 / val 56,288 / test 36,079 of 366,514). ~15 videos is an *inference* from the 10%
frame share, not a stated figure. Say "N unreported (inferable ≈15 videos)" — that is the sharper
criticism anyway: the denominator of their headline metric is not given.

### Where we are genuinely ahead (all verified, use these)
1. **Evaluation rigour.** Ours: 122 human masks / **5 fully held-out videos**, split by source video,
   holdout excluded from every training source, CIs **cluster-bootstrapped by video**, frozen sets,
   negative types never pooled. Theirs: frame 1 only, no CIs, N unreported, and the headline metric is
   provably insensitive (unchanged across a 4x prompt change). **Their metric does not measure video
   propagation — which is what their paper is about.** Their own figures show propagation failing.
2. **Deployed model size: ~249M (YOLOv11-l 25.3M + SAM2.1-hiera-large 224M) vs our 3.2M — ~78x smaller**,
   and they make **no inference-speed or real-time claim** at all. Their compute wall is solved by
   prompting *less*, not by shrinking. Distillation is our axis, uncontested.
3. **Presence/absence.** They **discard** frames without a visible octopus (366,514 kept of 564,755 =
   35% dropped) — the pipeline *assumes* presence. We must *decide* it, and we measure it on
   human-verified negatives (AUC 0.907 empty / 0.906 reflection) incl. a stated null result.
4. **Modality.** Dim aquarium **IR + glass reflections** (a camera we quantify and reject) vs their
   natural daylight colour, **no IR at all**.
5. **Downstream behaviour.** They stop at masks, explicitly *"the first step towards automating the
   detection of animal behavior."* We have the 7-class ethogram, 3,083 present clips, activity budget,
   exposure-normalised circadian, stimulus response with day-clustered CIs (R17), skeleton kinematics,
   and label reliability (R15 kappa 0.552).

### Where THEY are ahead — state it, do not hide it
1. **External validity.** 148 videos / 366,514 frames / **3 sites / 7 expeditions (2022-24) / wild
   multi-individual** *O. insularis*. We are **one individual (Nity), one tank**. Their generalisation
   claim is stronger than ours and no amount of our harvesting fixes single-animal scope.
2. **Occlusion re-identification** via per-frame re-detection — the correct insight (a contaminated
   propagation tracker never recovers). Qualitative only (Fig. 5, one video, no numbers), but right.
   Cite positively; contrast, do not re-derive.
3. Prompted SAM2-large is genuinely strong (IoU 0.8965 on frame 1).

### The framing that makes all three numbers coherent (use this in the paper)
The field's numbers differ mainly by **how much prompting is assumed at inference**:
**0.9383** (theirs, box-prompted, frame 1) > **0.726** (our teacher where it clears its own 0.60 seed
gate, R19) > **0.6415** (our student, unprompted, 122 held-out frames). Read that way our R19 result
*corroborates* rather than contradicts them: a confident/prompted foundation model does give excellent
masks; the unsolved problem is **unprompted, dense, deployable coverage** — which is exactly the gap a
distilled student fills. This lets us cite their number honestly without conceding a loss.

## R21 — OUT-OF-DOMAIN probe on wild YouTube footage: both presence signals collapse (2026-08-20)
The paper's biggest external-validity weakness is scope (one animal, one tank). This measures it
instead of conceding it. `src/eval_ood_youtube.py`, all footage **Creative Commons Attribution
(reuse allowed)**, verified per video before download; nothing redistributed.

**Data.** 5 positive videos (453 frames @1 fps): `oqj6BMI0qCU` *O. vulgaris* in seagrass (SAME species
as Nity), `0r1pLGA_cVI` temperate Ireland, `7saS5FPM60s` *O. laqueus* (OIST Reiter Unit),
`E33VznOE_PY` mimic octopus, `0wacRRF4BO4` wonderpus. 3 reef negatives (521 frames):
`aCw4GQxNZnY`, `_pLZqfXFuaU`, `E1k5P-E01J4`. Domain shift: daylight colour, open water, camera
motion, other species, no tank glass, no IR.

### Result — near chance, and the detector is the worse of the two
| signal | OOD AUC | FP on reef (deployed gate) | fire-rate on positives |
|---|---|---|---|
| CLIP+MLP detector `p_visible` (gate 0.60) | **0.565** | **88.5%** | 88.1% |
| segmenter mask area (gate 0.01) | **0.605** | 34.2% | 41.9% |

**In-domain presence AUC 0.907 (R14) → 0.605 out of domain.** The detector's median `p_visible` on
**coral-reef footage with no octopus is 1.000** — full confidence, wrong answer. Note the in-domain
ordering is PRESERVED (mask area > detector, R14/R16): the segmenter degrades less badly.

### The failure mode is mislocation, not silence — verified by eye
Not asserted; the top-scoring frames were inspected (`data/youtube_ood/_inspect/`):
- `aCw4GQxNZnY` t=11s: the mask covers **62% of the frame — the entire coral bommie**. Genuine FP.
- `oqj6BMI0qCU` t=122s: an *O. vulgaris* is clearly visible centre-frame, detector `p=1.00`, and the
  mask sits on the **surrounding seagrass, missing the animal**.
- `7saS5FPM60s` t=99s: the animal is the dark shape centre-frame; the mask covers the bright substrate
  around it and **excludes the octopus** — effectively inverted.
This is the SAME failure as in-domain ("fails by *mislocating* a right-sized blob",
SEGMENTATION_LOG), amplified. **Hypothesis (consistent with the evidence, not proven):** the student
learned a tank-specific figure/ground prior — under IR the animal is the mid-tone textured mass on a
dark den background; in daylight it is often DARK on BRIGHT sand, so the polarity flips and the model
selects the background. Testable by a brightness-polarity ablation; not run.
- `E33VznOE_PY` (mimic octopus): segmenter fires on **0%** of frames — a total miss, not a mislocation.

### Caveats — mandatory with these numbers
- **NOT a benchmark.** No GT masks, so no IoU. Frame-level presence is unlabelled, so the positive
  fire-rate is a **lower bound on recall** (the animal is often off-screen), not recall.
- **Negatives are only PRESUMED octopus-free.** The highest-scoring frames were human-checked (above)
  and are genuine FPs, but not exhaustively. Same discipline as the 232-frame hard-negative lesson.
- **5 positive / 3 negative videos** — far too few clusters for a CI; no CIs computed. Claim the
  direction, never a magnitude.
- Frame-level AUC across videos is a video-level proxy, biased by how often the animal is visible.

### What it is good for
The **limitations / external validity** section, upgraded from "generalisation is untested" to
"generalisation was tested and it fails, in this specific way". It also supports the HideAndSeg
complementarity claim (R20): they train on wild multi-site footage, and we now have direct evidence
that a tank-trained student does not transfer there — so the two lines really are addressing
different problems, not competing on one.

## R22 — Trained probe vs ZERO-SHOT CLIP: the probe does essentially all the work (2026-08-20)
The paper reported the detector at $96.8\%$ internal accuracy with **no baseline**, so it could not
attribute performance to the probe rather than to CLIP's features. `src/eval_zeroshot_vs_probe.py`
closes that. Detection only — mask area is a separate section's concern.

**The set matters most here.** EMPTY-V2's **120 human-labelled frames / 60 videos** (23
`octopus_present`, 97 `empty`), sampled at **uniform random timestamps over whole source videos**, so
it is **detector-INDEPENDENT**, and leak-free (`empty_negatives.py` excludes thin768's 142 training
videos *and* the CLIP detector's sessions). Contrast the 232 mined hard negatives, which were selected
at `p_visible >= 0.70` — their p_visible spans only 0.81–1.0, so any detector AUC on them (0.7805) is
a **selection artifact**, not a measurement. Do not use that set to score the detector.

| arm (identical 120 frames) | AUC | CI95 | FP@R.90 |
|---|---|---|---|
| **trained probe** (`clip_mlp_hardneg_v2`) | **0.7450** | [0.564, 0.890] | 0.856 |
| **zero-shot CLIP** (same backbone, 5+5 prompt ensemble) | **0.4500** | [0.259, 0.643] | 0.959 |

**Paired ΔAUC = +0.2950, CI95 [+0.069, +0.528], clustered by source video — excludes 0.**
Both arms share the frozen CLIP ViT-B/32 **and** the letterbox preprocessing, so the gap isolates the
probe. Zero-shot is at chance and its **median score is HIGHER on empty frames (0.182) than on frames
containing the animal (0.129)**. Zero-shot got a deliberately generous prompt ensemble (5 octopus + 5
empty, stored in `data/zeroshot_vs_probe.json`) so the margin is not a strawman artefact.
=> "zero-shot CLIP was abandoned as unreliable" now has a NUMBER behind it.

### Caveats (carry them)
- **23 positives.** All intervals wide. Claim the ordering, never the magnitude.
- **Do NOT compare 0.745 to the 96.8%** — different metric AND a harder, unbiased set.
- Frame-level; the deployed gate acts on 20 s windows (>50% of frames), so this is a per-frame proxy.
- At the shipped threshold (p>=0.60) the probe's recall on these unbiased positives is **0.609** with
  FP 0.175 — lower recall than curated-positive evaluations imply. Noted, not yet folded into the paper.

### Deferred, deliberately (belongs to the segmentation/presence section, not detection)
Mask area on these same 23 unbiased positives scores **0.7064** [0.544, 0.854] vs the **0.907** the
paper reports. Same negatives, same model, same threshold — only the POSITIVES differ (paper uses
human-masked "octopus definitely present" frames from extractor-selected clips; these are random
moments). Not a contradiction, a harder question, and the one deployment actually faces. The probe's
and mask area's CIs overlap heavily here, so **this set cannot rank them** — it only separates both
from zero-shot.

### Paper integration
Added to Sec. III-A "Visibility Detection" as **"What the probe adds over CLIP alone"** (prose, no
table, to save space). **COST: the paper went 8 -> 9 pages.** 0 errors, 0 overfull. Page 8 was already
nearly full — even 13 lines tipped it. T2 (demote the activity budget) and T3 (retire the
enrichment contrast) both REMOVE text, so doing them next would pay the page back.

## R23 — 235B frame-draw ENSEMBLE, and the first human comparison (2026-08-21)
Two artefacts: an ensemble that removes frame-sampling variance, and 298 human labels scoring it.

### The ensemble — `src/ensemble_235b.py` + `src/ensemble_235b_vote.py`
5 independent passes per clip, interleaved-uniform sampling: frames at `DENSE_FPS=2.5`, pass *p*
takes `step=n/10, offset=(p-1)*step/5`, i.e. **10 frames exactly 2 s apart**, the 5 passes tiling the
2 s gap (0.0/0.4/0.8/1.2/1.6 s) and **disjoint by construction**. Verified: 5/5 distinct draws,
union = all 50 candidate frames.
- **Why not the old top-6-by-p_visible:** measured on 33 clips it covers a median 0.79 of the clip
  but leaves a **median 7-frame (7 s) gap** between adjacent frames sent, inside which a whole crawl
  or jet happens unobserved (7/33 clips saw <50%). Uniform 2 s spacing caps the gap at ~3 frames.
- **No CLIP scoring**: frame choice is purely temporal, so the detector no longer decides what the
  VLM sees (removing that confound). The `PRESENT_MIN` veto went with it — it never fired anyway
  (`absent=0` across 2,818 single-pass calls).
- Cost/latency measured: `tokens ~= 290 + 386n`; 6 frames $0.000603/call, 10 frames $0.001043 (1.73x).

### INTER-PASS AGREEMENT (1,144 clips with all 5 passes)
| field | mean pairwise agreement | Fleiss kappa | pairwise Cohen kappa (10 pairs) |
|---|---|---|---|
| ethogram (9 values) | 0.729 | **0.634** | 0.634 (0.617–0.653) |
| presence (binary) | 0.910 | **0.818** | 0.818 (0.803–0.837) |
**The passes are exchangeable** — Cohen kappa spans only 0.617–0.653 across all 10 pairs, which is
the precondition for majority voting to reduce variance rather than bias it.
**The argument for ensembling is that a single pass is an arbitrary draw**: 38% of clips split, and
the vote differs from pass 1 on ~15%. Do NOT frame the 0.552->0.634 change as an ensembling gain —
frame count, frame rate and selection rule all changed at once, so it is not a controlled comparison.

### Voting rules (learned the hard way)
`uncertain` is an **abstention**, discarded from both tallies but the clip is **never dropped** — it
previously counted as PRESENT, because per-pass presence is derived as `"not present" not in label`
and "uncertain" does not contain that string (230 of 5,671 passes). Ties are broken **deterministically**
(prefer pass 1, else sorted) and flagged: `Counter.most_common(1)` had been resolving 49 clips by
FILE-READ ORDER. Low-vote and tied clips are marked `low_confidence`, not discarded.

### HUMAN COMPARISON — `src/eval_human_vs_ensemble.py` -> `data/human_vs_ensemble_results.json`
Two frozen rounds, blind-capable UI (`ui/label_ethogram.py`, port 8021). **Both rounds came back
100% `assisted`** (the model's verdict was on screen), so **these are AGREEMENT, not accuracy** —
anchoring inflates them and the paper's "validation against a human ethologist" gap stays OPEN.

| | v1 | v2 |
|---|---|---|
| clips / source videos | 98 / 40 | **200 / 88** |
| median s per clip | 8.8 | 4.9 |
| presence agreement / kappa | 0.816 / 0.630 | **0.850 / 0.659** |
| model false positives | 18 (26.5% of model-present) | **18 (13.0%)** |
| model false negatives | **0** | **12** |
| behaviour exact match | 38/50 = 76.0% | 76/120 = **63.3%** |

**1. Presence errors are mostly but NOT only one-directional.** v1 showed 18 FP / 0 FN; v2's larger,
more camera-diverse sample found **12 FN**, so "the model never misses a visible animal" is withdrawn.

**2. The false positives are almost entirely ONE CAMERA** (v2): Right_Left **45%** FP, Right_Back 15%,
**Right_Top (IR) 8%**, Right_Front 3%, Right_Right 0%. IR was unmeasurable in v1 (3 clips available)
and is FINE. The presence problem is reflections, not a general failure.

**3. CORRECTION to an earlier read — the vote margin DOES predict correctness.** v1 looked flat
(0.739 unanimous vs 0.737 at <=3/5) on n=19; v2 with n=47 gives **0.726 unanimous / 0.864 at 4-of-5 /
0.426 at <=3/5**. So `low_confidence` is a usable gate on which labels enter an aggregate.

**4. The model over-calls "Reaching out of water"** — 16 of v2's 44 behaviour errors point into it
(from Human-interaction 8, Exploration 5, Resting 3). v1 had suggested a Resting<->Reaching confusion;
the larger sample localises it to Reaching as a sink.

### Caveats to carry
- 100% assisted => agreement only. A blind round is still needed for an accuracy claim.
- Both samples over-sample rare classes and contested clips, so the raw rates are NOT corpus rates;
  per-class and per-margin figures are the valid ones until post-stratified.
- Effective sample size is the VIDEO count (40 and 88), not the clip count.

## R24 — OWLv2 as a presence filter: the paper's stated REASON was wrong (2026-08-22)

Re-measured from `data/hard_negatives/review_decisions.csv` (232 frames the CLIP+MLP probe called
confident-visible, human-adjudicated: 166 octopus / 66 genuine hard negatives).

| quantity | value |
|---|---|
| OWLv2 max-score AUC, octopus vs verified hardneg | **0.759** |
| octopus score median (IQR) | 0.255 (0.209–0.367) |
| hardneg score median (IQR) | 0.193 (0.140–0.243) |
| frames scoring ≥0.10 | 231/232 (99.6%) |
| best Youden J | 0.378 @ thr 0.230 → recall 68.1%, negatives passed 30.3% |
| thr for ≥95% octopus recall | 0.155 → passes 69.7% of negatives (46/66) |

**The correction.** v2 of the paper said OWLv2's "scores never separated the two classes---no
operating point existed". The first half is FALSE: AUC 0.759 is well above chance. The conclusion
("useless as an auto-filter") is correct and now quantified: the distributions overlap so heavily
that no threshold is usable — at its own optimum it still passes 30% of the negatives while
discarding 32% of the real animals. Paper text fixed; AGENTS.md fixed with a do-not-repeat note.

**Why this matters beyond the one sentence.** It is the same shape as R19 (teacher vs human masks):
a foundation model that is *informative but badly calibrated for this footage* reads as "useless"
if you only ever check its default threshold. In both cases the honest finding is about the
**operating point**, not the model's blindness — and in both cases stating it as blindness makes a
claim a reviewer can trivially refute by running the model themselves. Reported the weaker,
survivable version.

Also measured, and worth keeping: the CLIP+MLP probe scores only AUC **0.653** on these same
frames. That is not a fair comparison in the probe's favour — the set is *by construction* its own
confident false positives, i.e. maximally adversarial to it — so it must not be quoted as
"OWLv2 beats our probe". It is quoted here only to note that OWLv2's ranking is not the weakest
signal on this set, which is itself an argument against the old "never separated" phrasing.

## R25 — the ethogram training set was 36% smaller than the data we had paid for (2026-08-22)

Caught by the question "why are we not using all the clips that we completely processed?".

`ensemble_235b_vote.py` was run ONCE, when the 5-pass ensemble was ~3,444 clips in, and never
re-run as the ensemble continued to ~5,222. Every downstream consumer reads the voted file, so the
dataset builder was training on a stale snapshot.

| | clips |
|---|---|
| clips the ensemble touched | 5,222 |
| clips with all 5 passes complete | 5,003 |
| clips in the voted file | 3,444 |
| **fully processed but never voted** | **1,568** |

Re-running the vote: **3,444 → 5,222 records (+52%)**; trainable after filters **2,978 → 4,673
(+57%)**. Cameras lost worst were `Right_Top` (652), `Right_Front` (442), `Right_Back` (356) — i.e.
the three real den angles, not the reflection camera.

**Cost of the miss, had it shipped:** the model would have trained on 64% of the labels the 235B
passes were paid for, and the paper would have reported a training-set size that understated the
data. With ~60 source videos the plateau diagnosis (see the segmentation arc, R19) turns on whether
the ceiling is data or label quality — reporting a data-limited result on 36% less data than
available would have pointed that diagnosis the wrong way.

**Generalisable defect, worth stating as a rule:** any *derived* artifact of a long resumable job
(vote files, merged indices, snapshots) is stale the moment the job advances. The ensemble was
correctly resumable; the vote was not re-derived. Fix applied: the builder now re-runs the vote
itself rather than trusting an existing file, so the two cannot drift.

Vote-quality figures on the full set (5,218 clips with genuine sampling variation, ≥5 votes):
unanimous on ethogram 3,330 (63.8%), split 1,888 (36.2%); the majority vote differs from the
single pass-1 label on **705 clips (13.5%)** — which is the direct measurement of what the
5-pass ensemble buys over one pass.

## R26 — the validator caught two dataset bugs before training (2026-08-22)

Ran `validate_ethogram_dataset.py` on the frozen v1 set. It failed, twice, and both would have
produced a publishable-looking but wrong result.

**(a) 29 source videos spanned two trainable splits — a video-level leak.**
Not a bug in the split function: `video_split()` assigns one split per video and each run was
internally clean. The leak came from RESUMABILITY. Features are cached per clip and `manifest.jsonl`
is append-only, so the re-run after the re-vote (R25) appended rows for the new clips and kept the
old ones — but the split is a GLOBAL greedy decision over the whole clip set. Run 1 saw 2,978 clips,
run 2 saw 4,673, so **29 of the 58 videos present in both runs were assigned to a different split**.
Only the concatenation was broken, which is why nothing errored.

Rule extracted: *resume the expensive per-clip work, never a global decision.* Splits are now
recomputed from scratch every run and the manifest rewritten atomically (ms, leak-free by
construction), with an in-code assertion that no video spans two splits. Same bug class also
silently truncated `human_secondary.jsonl` to only the current session's rows — fixed by rebuilding
it from the full manifest.

**(b) 147 of 6,945 clips on disk (2.1%) are truncated to under 15s.**
Byte-range extraction failures that `extract_clip` accepted because it validates FILE SIZE (>10 KB)
— and one truncated clip is 3.6 MB at 0.49s, so size is simply the wrong test. This is the fifth
variant of the `pcm_alaw` failure. **8 had reached the training set, 4 carrying a behaviour label,
and one of those was in TEST**: a 0.4s clip labelled `Exploration / manipulation`, which the model
could never get right and which would have been reported as a genuine failure.

My first pass at this audit prefiltered by file size and found 98; that undercount was the same
error I was auditing. A full ffprobe sweep of all 6,945 gives 147. Distribution: 95 unreadable /
13 <1s / 20 1-5s / 19 5-15s. By camera: Right_Top 60, Right_Right 56, Right_Front 24.

Fix: `MIN_FRAMES=37` (≈15s at DENSE_FPS 2.5). The corpus is cleanly bimodal — 4,665 clips have 46+
frames, the 8 bad ones have <37, nothing in between — so the threshold isolates exactly the
truncated files at a cost of 0.17%. Enforced both in the feature loop and retroactively at
finalize, since resume would otherwise never revisit rows written before the guard existed.

**Frozen v1: 4,665 clips / 206 videos.** train 3,019 (133 vid) / val 655 (35) / test 740 (34) /
human_secondary 251 (99). Majority baseline 45.8%. Motion channel validated: Locomotion vs Resting
AUC **0.714** on `motion_disp` alone, so the two motion columns carry real signal and rungs 2-3 of
the ladder are worth running. Soft targets: 70.8% unanimous, the rest teach uncertainty.

## R27 — ethogram classifier: the rung ladder (2026-08-22)

Frozen v1 (4,665 clips / 206 videos), splits by source video, 3 seeds per rung, mean ± std.
Metric is **macro-F1**, never accuracy: the classes run 45.8% to 3%, and accuracy rewards collapse.

| rung | model | params | val macro-F1 | TEST macro-F1 |
|---|---|---|---|---|
| — | majority class (`No octopus`) | — | — | **0.1004** |
| 0 | mean-pooled CLIP → linear | 3,078 | 0.4842 ± 0.0050 | 0.4000 ± 0.0143 |
| 1 | mean\|std\|max pooling → MLP | 398,086 | 0.5531 ± 0.0090 | 0.4739 ± 0.0373 |
| 2 | + motion summary stats | 399,634 | 0.5521 ± 0.0115 | 0.4802 ± 0.0300 |
| 3 | full [10,514] → BiGRU + attention | 266,891 | 0.5439 ± 0.0050 | **0.5129 ± 0.0110** |

**1. The previous classifier's failure mode is gone.** That model hit 45% val accuracy with per-class
F1 ≈ 0 outside the majority classes — collapse. Here EVERY class scores 0.29–0.75 at every rung,
including rung 0, which deliberately reproduces the old pooled-CLIP-and-linear recipe. So the fix was
NOT architecture: it was the 6-class merge, the reliability filtering, soft targets, and 4,665 clips.
Also note rung 0's 0.400 vs a 0.100 majority baseline — the old "45%" was never the failure it read
as, because it was compared against an accuracy baseline instead of macro-F1.

**2. Only ONE jump on the ladder is real: rung 0 → 1 (+0.074 test, +0.069 val).** Richer pooling and
a non-linear head help. Rungs 1 → 2 → 3 differ by 0.006 and 0.033 with stds of 0.030–0.037, i.e.
within ~1 std, and **val and test disagree on the ordering** (val ranks rung 3 LAST, test ranks it
first). When the two holdouts disagree, video-level noise dominates at 35 val / 34 test videos. Per
the interpretation rule fixed before running: **the ceiling is video diversity, not architecture** —
the same wall segmentation hit at IoU ~0.47 across every model size. Do NOT claim rung 3 wins.

**3. The motion channels bought nothing measurable** (rung 1 → 2: 0.4739 → 0.4802, well inside
noise), despite scoring AUC 0.714 for Locomotion-vs-Resting in isolation. So the signal is real but
largely REDUNDANT with what CLIP appearance already encodes. Worth stating plainly since the motion
channels were added specifically to fix the pooled-feature critique — the critique was right about
pooling (rung 0 → 1) and wrong about needing explicit motion.

**4. Rung 3's variance is 3× lower (±0.011 vs ±0.037).** Not a headline win but the reason to prefer
it if one model must be picked: sequence modelling makes the result reproducible across seeds.

**5. The dominant error is the two STATIC classes collapsing into each other.**
Confusion matrix (rung 3, 3 seeds pooled): `Resting → No octopus` 16.2%, `No octopus → Resting`
13.7%. **That single confusion is 20.2% of ALL errors.** Physically sensible: a motionless animal in
a den and an empty tank differ only in appearance, and both motion channels read ~0 (medians are
identical, 0.0198 vs 0.0198 — the one place the motion feature CANNOT help). This is the concrete
target for the next iteration, and it argues the segmentation area-gate would help more than any
classifier change, since a mask distinguishes them directly.

**6. `Reaching out of water` behaves as a sink class:** recall 0.81 but precision 0.35 — it absorbs
121 `No octopus` and 84 `Exploration` samples. Per-class precision/recall must be reported, not just
F1, or this reads as the best-performing class.

Caveat carried forward: `Human / enrichment interaction` has n=40 test clips on 7 videos; its F1
(0.31–0.37) is not a reliable per-class figure.

## R28 — the class-weighting bug: rare classes were sinks (2026-08-22)

Chasing "how do we make the 6-class ethogram classifier better", the first real defect was in the
LOSS, not the architecture. v1 used full inverse-frequency class weights (`1/count`). Train counts
run 1,419 (`No octopus`) to 141 (`Human`), so rare classes received 6-10x the weight and became
**sinks**: `Reaching out of water` reached recall 0.81 at **precision 0.35** — 65% of everything it
predicted was wrong, absorbing 121 `No octopus` and 84 `Exploration` samples.

Note the self-defeat: over-weighting a rare class is meant to protect macro-F1, but macro-F1 charges
for precision too, so the weighting cost the very metric it was there to defend.

Sweep of the weight exponent `CW_POWER`, rung 3, 3 seeds each:

| CW_POWER | val macro-F1 | TEST macro-F1 | `Reaching` precision |
|---|---|---|---|
| 1.0 (v1) | 0.5439 ± 0.005 | 0.5129 ± 0.011 | 0.35 |
| **0.5 (sqrt)** | **0.5555 ± 0.010** | **0.5298 ± 0.007** | 0.43 |
| 0.25 | 0.5400 ± 0.005 | 0.5417 ± 0.005 | 0.48 |
| 0.0 (off) | 0.5381 ± 0.015 | 0.5581 ± 0.014 | 0.53 |

`Reaching` precision rises monotonically as the weighting is removed (0.35 → 0.43 → 0.48 → 0.53),
which is clean mechanistic confirmation that inverse-frequency weighting created the sink.

**METHODOLOGICAL TRAP, recorded because I nearly walked into it.** `CW_POWER=0.0` gives the best
TEST macro-F1 (0.5581, +0.045). It is NOT claimable: val does not select it, and picking a
hyperparameter by test score is test-set selection. **Selecting on val gives 0.5 → TEST 0.5298,
+0.0169 over v1** (stds 0.007/0.011, so ~1.5σ). That is the honest number. The val/test optimum
disagreeing again is the same video-level-noise signature as the rung ladder (35 val / 34 test
videos) — and it is exactly why the rule has to be fixed before looking, not after.

Per-class at the selected `CW_POWER=0.5` (rung 3, 3 seeds pooled):

| class | recall | precision | F1 |
|---|---|---|---|
| No octopus | 0.73 | 0.86 | 0.79 |
| Locomotion (crawl/swim) | 0.58 | 0.56 | 0.57 |
| Reaching out of water | 0.72 | 0.43 | 0.54 |
| Exploration / manipulation | 0.54 | 0.51 | 0.53 |
| Resting / stationary | 0.44 | 0.44 | 0.44 |
| Human / enrichment interaction | 0.27 | 0.37 | 0.31 |

`No octopus` ↔ `Resting` remains the dominant confusion (22.3% of all errors) and is untouched by
weighting, as expected — it is a label/signal problem (45% of IR `No octopus` labels are wrong
against the human, R26/R27), not a class-balance problem. Two independent defects, two fixes.

## R29 — teacher vs student vs human: the STUDENT is the weak link (2026-08-22)

`src/eval_ethogram_human.py`. The ladder scores the student against its own teacher, i.e. measures
reproduction, not correctness. 455 human labels make the real question answerable. Populations are
**never pooled** — 102 human-labelled clips are in the train split, so scoring there measures
memorisation and is diagnostic only.

Rung 3, CW_POWER=0.5, 3 seeds, probabilities averaged before argmax.

| population | clips/videos | teacher vs human | student vs human | student vs teacher |
|---|---|---|---|---|
| **human_secondary** | 251 / 99 | **72.5%**, macro-F1 **0.6569** | 60.6%, macro-F1 0.4917 | 0.5635 |
| test | 30 / 19 | 86.7%, 0.5139 | 60.0%, 0.2636 | 0.2721 |
| val | 25 / 18 | 92.0%, 0.4298 | 72.0%, 0.3150 | 0.3214 |

Agreement CIs on the 251-clip population: teacher [66.7, 77.7] vs student [54.4, 66.4] — **disjoint**.
test/val macro-F1 are computed on 25-30 clips over 6 classes; too noisy to quote.

Where student and teacher disagree, who does the human back?

| population | disagreements | backs STUDENT | backs TEACHER | neither |
|---|---|---|---|---|
| human_secondary | 86 | 19 | **49** | 18 |
| test | 11 | 1 | 9 | 1 |
| val | 6 | 0 | 5 | 1 |

**THIS REVERSES R27's headline.** R27 concluded "the ceiling is video diversity" (later narrowed to
"given frozen CLIP features"). Neither labels nor data quantity is the binding constraint:

1. The teacher beats the student on the SAME clips, 0.657 vs 0.492 macro-F1, disjoint intervals. The
   student has ~0.165 macro-F1 of headroom against labels already in hand.
2. On disagreements the human sides with the teacher 2.6:1 — the student's deviations are mostly the
   student being wrong, not the teacher being noisy.
3. So the signal IS in the footage; a 235B VLM recovers it and a frozen-CLIP embedding + 267K-param
   head does not. **The constraint is the student's representation/capacity.**

Consequences for the plan: the backbone swap (R30, running) becomes the first experiment, and buying
teacher labels for the 1,160 harvested clips drops down the list — more of the same labels does not
fix a student already behind the labels it has.

**CONFOUND, recorded because it favours this conclusion.** The hint the labelling UI shows is the
ENSEMBLE's verdict, and all 455 labels are `assisted`, so the annotator was anchored toward the
teacher and `teacher vs human` is inflated by an unknown amount. `student vs human` is NOT anchored
(student predictions were never displayed), so the comparison is biased in the teacher's favour and
the true gap is smaller than 12 points. The direction likely survives — large gap, 2.6:1 split — but
the magnitude must not be quoted as clean. The blind round on the reserved test videos is the fix and
this result raises its value.

Full write-up with all tables: `RESULTS_ETHOGRAM.md` §6.4.

## R30 — representation swap, in flight (2026-08-22)

`src/extract_backbone_feats.py`. R27's ladder varied the HEAD on frozen CLIP; the representation was
never varied, so "the ceiling is video diversity" was always conditional. R29 then showed the student
is the weak link, making this the right experiment. Same clips, same frames (the manifest's recorded
`frames_used`), same two motion channels appended → the backbone is the only free variable and rungs
1-3 stay architecturally identical.

| backbone | kind | params | hypothesis |
|---|---|---|---|
| CLIP ViT-B/32 | image-text | 88M | baseline: test macro-F1 0.5298 |
| DINOv2-base | image, self-supervised | 87M | is CLIP's *appearance* encoding the limit? |
| VideoMAE-base | **video-native** | 86M | is *time* the limit? |

The video arm is sharper: CLIP has no notion of motion, and the hand-computed motion channels meant
to compensate bought +0.006 (R27).

**Trap that would have inverted the result.** transformers 5.12 expects VideoMAE attention biases as
`attention.{query,key,value}.bias`; the checkpoint stores `q_bias`/`v_bias` (no `k_bias` — zero by
design). 36 of 196 tensors therefore came out FRESHLY INITIALISED, announced only via a generic
"MISSING … consider training on your downstream task" warning. It does not crash and the features look
plausible — a partly uninitialised encoder would have produced a **fake negative for the video
backbone** and the wrong conclusion about whether time helps. Patched (24 tensors: query+value over 12
layers) after verifying the attention WEIGHTS load bitwise-identical, so only biases were affected.

Status 2026-08-22: DINOv2 761/4,665 (0 failures), ~22 clips/min on MPS, ETA ~3 h; VideoMAE queued.

## R31 — the representation WAS the constraint: three frozen backbones compared (2026-08-22)

`src/extract_backbone_feats.py` + `src/train_ethogram.py --backbone`. Same clips, same frames (the
manifest's recorded `frames_used`), same motion channels, same splits/seeds/rungs, `CW_POWER=0.5`.
The refactor was regression-tested first: the CLIP path reproduces R28's 0.5298 exactly.

| backbone | rung 1 | rung 2 | rung 3 |
|---|---|---|---|
| CLIP ViT-B/32 (512d) | 0.5096 ±0.030 | 0.5368 ±0.015 | 0.5298 ±0.007 |
| DINOv2-base (768d) | 0.6006 ±0.004 | 0.5772 ±0.014 | 0.5781 ±0.001 |
| VideoMAE-base (768d) | 0.5883 ±0.005 | 0.6057 ±0.011 | 0.6016 ±0.022 |

Val-selected: CLIP rung3 → **0.5298**; DINOv2 rung2 → **0.5772 (+0.047)**; VideoMAE rung1 →
**0.5883 (+0.059)**.

**Robust to the selection rule** — the WORST rung of either new backbone beats the BEST rung of CLIP
(DINOv2 0.577-0.601, VideoMAE 0.588-0.606 vs CLIP 0.510-0.537; no overlap). That is what the R27 rung
ladder lacked, and it confirms R29's diagnosis with a fix rather than a diagnosis.

**But the win is NOT clearly about time.** VideoMAE leads, yet DINOv2 -- an image model -- is within
noise of it, and VideoMAE's best rung is 1, which pools over time and discards order. So the gain
reads as "a better self-supervised representation", not "a video model sees motion". Weaker than the
hypothesis, and it is the claim the data supports.

## R32 — combining backbones: ensemble beats fusion (2026-08-22)

`src/train_ethogram_fusion.py`.

| approach | params | val | TEST macro-F1 |
|---|---|---|---|
| best single (DINOv2 rung2) | 598K | 0.5784 | 0.5772 |
| FUSION (concat all 3 → one MLP, width 6150) | 1.59M | 0.5784 | 0.5960 ±0.028 |
| ensemble, val-selected rungs, soft | 3×~600K | 0.5969 | 0.6177 |
| **ensemble, ALL-MLP (rung 2), hard** | 3×598K | **0.6039** | **0.6172** |
| ensemble, ALL-MLP, soft | 3×598K | 0.6025 | 0.6183 |

Val-selected winner: the all-MLP ensemble. Val margin over the best single backbone **+0.0255 vs seed
std 0.0081** — clears noise, unlike the head sweep.

**Fusion lost**, as predicted: 6,150-wide input and 1.59M params against 133 training videos landed at
val 0.5784, identical to the best single backbone, with the worst test variance. More capacity keeps
losing on this data. **Homogeneous all-MLP members beat each backbone's individually-best rung**
(0.6039 vs 0.5969 val) and are simpler to deploy. Members agree on only **60% of test clips** — that
disagreement is why voting helps.

**ACCURACY vs MACRO-F1, since the two are easily confused:** the ensemble is **71.4% accuracy**
(528/740) at macro-F1 0.6183. Majority baseline is 43.1% accuracy / 0.1004 macro-F1. Accuracy is
reported only alongside — a model predicting `No octopus` always scores 43% accuracy and 0.10 macro-F1,
which is exactly the collapse macro-F1 exists to catch.

Per class (soft vote): No octopus 0.88 · Exploration 0.67 · Locomotion 0.65 · Resting 0.58 ·
Reaching 0.56 · Human interaction 0.37. `Resting` improved from 0.44 (R28) — the No-octopus↔Resting
confusion that was 22% of all errors is genuinely better with stronger representations.

## R33 — four things that did NOT help, measured (2026-08-22)

Recorded because negative results are what stop the same ideas being retried.

1. **Head capacity/architecture: ≈0.** `src/sweep_ethogram_head.py`, hidden {256,512,1024} ×
   dropout {0.3,0.5}. VideoMAE: val gain +0.0017 vs seed std 0.0040. DINOv2: the frozen 256/0.4 head
   IS the val-best. Test spread across the grid is 0.033 while val resolves 0.002 — val cannot
   distinguish these configs, so any "winner" is luck. **More capacity actively hurts**: 1024 hidden is
   the worst config on both backbones.
2. **Upsampling ≈ loss weighting.** `BALANCE` ∈ {none, weight, upsample, both}. Upsample val 0.5775 vs
   weight 0.5753 — margin +0.0023 vs seed std 0.0026, inside noise.
3. **Less balancing keeps winning.** `BALANCE=none` gives the best TEST macro-F1 (0.6103), best
   accuracy (70.4%) AND the best F1 on the weakest class (`Human` 0.44 vs 0.37 weighted) — balancing
   hurt the class it was meant to protect. Consistent with the CW_POWER sweep (1.0→0.5→0 all improved
   test). NOT adopted: val ranks `none` WORST (0.5648), so taking it would be test-set selection. Val
   and test disagree systematically on this axis across three experiments — a limitation to report.
4. **Feature-space augmentation is NEGATIVE.** mixup {0.2,0.4}, Gaussian noise {0.05,0.1}, and the
   combination all lose, monotonically with strength, and val/test AGREE for once (baseline val 0.5753
   / test 0.6057; mixup 0.4 → 0.5667/0.5896; noise 0.1 → 0.5450/0.5481). The model is not
   overfitting-limited, so a regulariser only corrupts frozen features. mixup does lift the weakest
   class slightly (0.37→0.39 as alpha rises) at everyone else's expense — the same recall-for-precision
   trade class weighting made.

**The pattern across R31-R33: everything that adds real information helps; everything that reshuffles
existing information does not.** representation +0.087 › class-weight tempering +0.017 › head ≈ 0 ›
upsampling ≈ 0 › augmentation negative.

## R34 — has the ensemble closed the gap to its teacher? Half of it (2026-08-22)

R29 re-run with the 3-backbone ensemble, same populations, same protocol.

| | agreement | macro-F1 vs human |
|---|---|---|
| Teacher (5-pass 235B) | 72.5% | 0.6569 |
| Student, single CLIP (R29) | 60.6% | 0.4917 |
| **Student, 3-backbone ensemble** | **66.9%** | **0.5755** |

On `human_secondary` (251 clips / 99 videos). Disagreement split improved from **49:19** in the
teacher's favour to **36:22**. So the representation work closed roughly half the gap, and **headroom
remains against labels already in hand** — we are not at the label ceiling, which is why more
teacher-labelled clips is still not the first lever.

Same anchoring confound as R29 (all human labels `assisted`, hint = the teacher's verdict), so
teacher-vs-human is inflated and the true gap is smaller. `test`/`val` populations (30/25 clips) show
the ensemble worse; macro-F1 over 6 classes at that n is dominated by classes with 1-3 examples and
should not be acted on.

**Next, running:** V-JEPA-2 (ViT-L, 326M, 1024d) as a fourth ensemble member — the cheapest test of
the only lever that has worked.

## R35 — mask features: the controls killed a +0.043 "gain" (2026-08-23)

`src/extract_mask_feats.py` + `src/eval_mask_features.py`. Ten segmentation-geometry channels (area,
centroid x/y, bbox w/h, elongation, solidity, masked-motion, centroid displacement, validity) from
`octo_seg_thin768_lraspp.pt`, appended to VideoMAE rung 2. Extraction: 4,665/4,665, 0 failures,
1,623 IR clips zeroed with `valid=0`.

**PRE-RUN VALIDATION was strong** — each channel separates the class it was designed for, on the
2,454 clips with a valid mask block:

| test | via | AUC | existing feature |
|---|---|---|---|
| No-octopus vs Resting | `area` | **0.772** | ~0.50 (motion medians identical) |
| Locomotion vs Resting | `masked_motion` | **0.819** | 0.714 (whole-frame motion) |
| Reaching vs others | `centroid_y` (height) | **0.804** | none exists |
| Locomotion vs Resting | `centroid_disp` | 0.707 | 0.714 |

(`centroid_y` prints as AUC 0.196 in the raw output — image y increases DOWNWARD, so Reaching's low
cy = animal high in frame, exactly as predicted. The direction convention was wrong, not the feature.)

**HEADLINE, WHICH IS NOT CLAIMABLE:** without mask val 0.5753 / TEST 0.6057 → with mask val 0.6200 /
**TEST 0.6491 (+0.0434)**, val margin +0.0447 vs seed std 0.0028.

**Why it fails — the built-in controls:**

| subset | n | macro-F1 without → with |
|---|---|---|
| segmenter SAW the video | 540 | 0.600 → 0.664 (**+0.064**) |
| segmenter UNSEEN | 200 | 0.306 → 0.280 (**−0.025**) |
| IR only (zeroed block, control) | 232 | 0.381 → 0.359 (−0.022) |

The whole gain sits on the 540 clips from the 11 test videos the segmenter trained on. The IR control
correctly shows no gain, so the comparison itself is sound — the effect IS the masks, and the masks
only help where the segmenter has already seen the footage.

**The mechanism predictions also failed.** `Reaching` +0.120 (matches the `cy` prediction), but
`Resting` **−0.010** despite being the main target, and the No-octopus↔Resting confusion rose from
**18.8% → 25.4% of errors** — the one number the experiment was built to move went the wrong way.
Meanwhile `Human interaction`, designated the NO-MECHANISM control, gained **+0.080**, the
second-largest jump. Gains where no mechanism was predicted and absence where one was is the signature
of a cause other than the claimed one.

**Caveat on the negative too:** the unseen subset has `Locomotion` and `Reaching` at 1 clip each and
`Human` at 2, so its macro-F1 is dominated by single-example classes. It shows NO DEMONSTRATED GAIN,
not harm.

**To make this conclusive:** retrain the segmenter with all 34 ethogram test videos held out
(`--holdout-videos`, the discipline the segmentation work already adopted after being burned by this
exact leak in 2026-08-09). Until then the strong pre-run AUCs stay a promising prior, not a result.

**Process note worth keeping:** the leak was foreseen and the split built in BEFORE running. That is
the only reason a +0.043 was not reported as a win — the raw numbers pass the seed-noise test cleanly.
