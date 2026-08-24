# Octopus Segmentation — Running Work Log

Detailed chronological log of the tiny-segmenter effort (teacher auto-labeling → student training).
Newest entries at the bottom. Numbers are honest, including the failures. See `SEGMENTATION_PLAN.md`
for the design.

Compute: a rented **A100-40GB** box, reached by SSH from a CPU-only workstation. Env: `~/seg-venv`. Dataset + weights live on the A100 during the run, pulled back at the end.

---

## 2026-07-22/23 — Session 1

### Phase 1a — balanced sampler (`src/sample_seg_clips.py`)
- Joined the 3,986 on-disk clips to behaviour labels in the index; dropped `octopus not present`
  (311) + Right_Left (reflections) + 33 not-in-index. Colour-first.
- Selected **1,824 present colour clips** (Right_Front 604 / Right_Back 540 / Right_Right 680),
  water-filled across behaviours. Manifest: `src/dataset_seg/sample_v1/sample_manifest.json`.

### Env setup on A100
- Installed ffmpeg + venv (torch cu124, transformers, opencv, `sam2` built `SAM2_BUILD_CUDA=0` — no nvcc).
- Warmed HF cache: GroundingDINO-tiny + SAM2.1-hiera-tiny. Setup script `scratchpad/a100_setup.sh`.
- Transfer: one-time rsync of the 1,824 clips (~10.5 GB) over the internal VPC to `~/seg_clips/`.

### Phase 1 — auto-labeling (`auto_segment.py`, GroundingDINO+SAM2 teacher)
- Sharded 3× by camera (parallel, ~9.5 GB GPU). ~13 s/clip.
- **Data-quality finding:** 504 of 731 Right_Right source clips are **0-byte on disk** (pre-existing
  extraction bug, not a transfer issue). auto_segment correctly rejected them as `no_frames`.
- Result: **1,104 clips accepted → 4,412 (image, mask) pairs** across **77 source videos**.
  Front 1,912 / Back 1,864 / Right 636. Mask area: median 2.9%, healthy (spot-checks clean).
  Rejected 243 low-confidence (reflections/ambiguous), which is the seed-conf gate working.

### Phase 2 — train tiny segmenter (`src/train_segmenter.py`)
- Split BY SOURCE VIDEO (62 train / 15 val), BCE+Dice, IoU@0.5. Aug: flip + brightness only (weak).
- **From-scratch TinyUNet sweep (256², 40 ep):**
  | base_ch | params | best val IoU |
  |---|---|---|
  | 8  | 0.121 M | 0.398 |
  | 16 | 0.483 M | 0.438 |
  | 32 | 1.927 M | 0.474 |
- **LR-ASPP MobileNetV3 (ImageNet-pretrained, 256², 60 ep):** train loss → 0.18, **val IoU 0.447**.

### Diagnosis (why v1 stalls at ~0.47, bar is 0.85)
- Cleaning big-GT masks (area>0.20) barely moved val (0.474→0.480) → **not** label noise.
- **Train IoU 0.684 / Val IoU 0.474** → a **generalization gap**, not underfitting.
- Failure mode on the 31% of val frames with IoU<0.3: GT is a normal ~3% octopus, model predicts a
  normal ~4% blob, **but in the wrong place** (only 10% predict empty) → *mislocation*.
- From-scratch U-Net and pretrained LR-ASPP hit the **same** ceiling → bottleneck is
  **data diversity + augmentation, not architecture**. Only 62 train videos; aug was flip+brightness only.
- **Decision:** highest-leverage cheap fix = strong spatial augmentation (affine/translate/scale) to
  teach position-invariance. Then, if short, add IR Right_Top (1,391 clips) for video diversity.
  (No `octopus_clips_auto` on this box; colour clips essentially exhausted.)

### Experiment: strong augmentation (affine/translate/scale + flips + color jitter + noise)
- U-Net ch32 + aug: **0.469** (was 0.474) — no change.
- LR-ASPP + aug: **0.473** (was 0.447) — +0.026, marginal.
- **Verdict: aug does NOT close the gap.** LR-ASPP still drives train loss to ~0.18 while val
  stays ~0.47 → confirms a genuine video-diversity generalization gap, not something aug fixes.

### Per-video val IoU (aug LR-ASPP) — diversity gap confirmed
- Range **0.00 → 0.82** across the 15 val videos, broad gradient (worst: 116-frame Right_Back @0.28;
  best 0.66–0.82). Not one pathological video → the model does ok on train-like videos, fails on
  dissimilar ones. Classic limited-diversity overfitting (only 62 train videos). macro 0.455 / micro 0.473.

### Decision: add IR Right_Top data (more video diversity + it's a real deployment camera)
- Colour clips are exhausted; the available diversity is IR Right_Top (1,391 clips). Right_Top is the
  BIGGEST deployment camera, so an IR-capable model is wanted (not a compromise) — supersedes the
  earlier colour-only-v1 call given the generalization evidence.
- Sampled **653 present IR clips** (`sample_seg_clips.py --cameras Right_Top --target 800`), rsynced to
  the A100, distributed round-robin into 4 shards, auto-labeling in parallel (~35 min).
- NEXT: merge IR pairs + colour v1 → dataset v2; retrain aug LR-ASPP on v2; eval PER-CAMERA (check IR
  doesn't tank colour, and measure IR quality — watch for the known IR over-segmentation on bright tools).

### IR auto-labeling — FAILED the quality bar (confirms plan's IR caution)
- 653 present IR clips → only **87 accepted (13%)**: GroundingDINO isn't confident on greyscale IR,
  so the seed-conf gate rejects 86%.
- Accepted IR masks **over-segment**: area median 8.5% / mean 14.7% / 27.5% >20% (vs colour 2.9%) —
  SAM2 grabs bright metal tools/pipes. `auto_segment.py` lacks the Phase-0 IR fix (point/negative prompts).
- Hard-filtered to octopus-range area (0.012–0.13) → only 189 of 345 IR pairs survived → v2 = 4,412
  colour + 189 IR. v2 retrain (aug LR-ASPP): val IoU ~0.47 — IR neither helped nor hurt colour.
- **Verdict: IR unusable without the Phase-0 IR fix. Defer to v2 as the plan said.**

### Presence-gate eval — THE deployment test (`scratchpad/eval_presence.py`) — v1 FAILS
- Ran best colour model (aug LR-ASPP) on 300 present val frames vs held-out negatives
  (60 Right_Left reflections + 60 octopus-absent colour clips).
- Mask-area medians: **present 0.027, reflection 0.039, absent 0.017**.
- **AUC (area separates present vs neg) = 0.496 — essentially RANDOM.** vs reflection 0.418 (worse than
  chance — reflections get BIGGER masks than real octopus). Threshold sweep: reflection-FP ≥ present-recall
  at every threshold. **v1 is NOT a usable presence gate.**
- **ROOT CAUSE (key insight): the model was trained ONLY on octopus-present frames** — every one of the
  4,412 masks contains an octopus, so it learned to ALWAYS emit an octopus-shaped blob, including on
  reflections/empty tank. It was never shown a negative. This also explains the "blob in the wrong place"
  mislocation and why arch/aug/IR couldn't move val IoU.

### Decision: add NEGATIVE (empty-mask) frames to training → v3
- Extract frames from reflection + octopus-absent clips, pair with EMPTY masks, add to training so the
  model learns "no octopus → no mask". This directly targets the presence-gate goal (and should sharpen
  localization). Keep the eval's seg_neg set held-out (distinct clips). Retrain, re-run presence eval.

### v3 built + training
- v2 (colour+189 IR) final: **val IoU 0.492** — marginal vs colour-only, IR confirmed not helpful.
- Built **v3 = 5,800 pairs** (4,412 positives + 1,388 empty-mask negatives from 350 reflection/absent
  clips, 24% neg). Training aug LR-ASPP, 60 ep (~65 s/ep — bigger set + CPU-bound aug).
- **Infra note (bug hit + fixed):** the background waiter scripts used `pgrep -c -f build_v3.py` /
  `train_segmenter.py`, which **match pgrep's own command line** (self-match) → count always ≥1 → the
  chained automation stalled at step 1 and never launched v3, and an earlier waiter never fired.
  Fix: `pgrep -cf "[t]rain_segmenter"` (bracket trick). Relaunched clean.
- v3 val IoU will NOT be comparable to v1 (val now contains easy negatives) — the metric that matters
  is the **presence-eval AUC** on the held-out seg_neg set.

### v3 RESULT — negatives fix the presence gate (the deployment win) ✅
v3 (aug LR-ASPP, 4,412 pos + 1,388 neg). Presence eval on held-out negatives (distinct clips):

| metric | v1 (no negs) | **v3 (with negs)** |
|---|---|---|
| AUC present vs all-neg | 0.496 (random) | **0.860** |
| AUC vs **reflections** | 0.418 | **0.991** |
| AUC vs absent (empty tank) | 0.575 | 0.725 |
| reflection mask area (median/mean) | 0.039 / 0.065 | **0.000 / 0.000** |

At op-threshold area≥0.01: **present-recall 0.88, reflection-FP 0.00**, absent-FP 0.52.
**Reflections — the #1 extraction false-positive (esp. Right_Left) — are essentially solved:** the
model emits ZERO mask on glass reflections. This is the mask-gate payoff the plan wanted, and it beats
the CLIP gate (which fires at p=1.0 on the same reflections).

### Honest bottom line
- **Deployment payoff (reflection-rejecting presence gate): achieved.** v3 is usable to clean extraction.
- **Mask pixel-quality bar (IoU 0.85): NOT met** — present-frame masks ~0.5 IoU (ok for coarse
  body-area / masked-motion, not precise). Ceiling is data diversity (62 colour videos), not model/aug.
- **Empty-tank "absent" discrimination (0.725): decent, improvable** with more absent negatives.
- Key insight that unlocked it: **train with negatives** (v1's fatal flaw was positives-only).

### Deliverables pulled back to the repo (before A100 cleanup)
- `weights/seg/octo_seg_*.pt` — all 8 checkpoints (sweep + aug + v2 + **v3 = deployable**).
- `data/dataset_seg/{v1,v3}` — the 4,412-pos mask dataset (+ v3 negatives). [gitignored, local]
- `results/segmentation/` — all train/eval logs + diagnostic overlays.
- `src/eval_presence.py`, `src/build_v3.py` — eval + negatives-dataset scripts.

### A100 cleaned up (2026-07-23)
- All segmentation artifacts deleted from the A100 box (home 31G → 28K): dataset_seg, seg_clips*,
  ir_shard*, seg-venv, weights_seg, all scripts + logs, and the HF/torch/pip/.nv caches I created.
- Left intact (not ours / shared): default dotfiles, `~/.ssh` (SSH access key), and the apt system
  packages (ffmpeg, python3-pip, python3-venv). GPU idle, 189G free. All artifacts already pulled to the repo.

### Recommended next (needs more data — user offered)
- More DISTINCT colour videos (diversity, not volume) → raises present-mask IoU + absent-case AUC.
- A small human-verified mask val set (~100–200) for trustworthy numbers.
- Implement the Phase-0 IR fix (point/negative prompts) before using IR.
- Then wire `segment_octopus` (area≥~0.01 gate) into extraction and A/B vs the CLIP gate.

---

## 2026-07-24 — Session 2: new diverse data + move to Modal

### New harvest dataset (`data/harvest_clips/`) — exactly the diversity we needed
- **392 clean colour clips** (Right_Front 224 + Right_Back 168, **0 zero-byte**), from **~204 distinct
  (date,segment,camera) source videos / 180 sessions** — vs the old 62 train videos, ~3× the diversity.
- Date span **2025-09-17 → 2026-02-20** (~6 months); **111 of 112 dates are NEW** (old set was 7 days in
  Feb 2026). Low redundancy (mostly ≤2 clips/video).
- Caveats: low-motion skew (resting-heavy; motion median 0.0004), presence via the CLIP gate (Front/Back
  so low reflection risk), and these are RAW clips — still need the teacher to auto-label them into masks.

### Ported the pipeline to Modal (`src/modal_seg.py`) — A100, serverless
- Reason: repeated label→train→eval as data arrives; Modal gives a reproducible image, a persistent
  Volume, and trivial parallel auto-labeling via `.map()` (no manual SSH/pgrep sharding).
- App `octo-seg`: A100 image (ffmpeg + torch cu124 + transformers + sam2 `SAM2_BUILD_CUDA=0` + opencv),
  Volume `octo-seg-data` at /data, functions `auto_label(shard)` / `train` / `presence_eval` reusing the
  exact logic from `auto_segment.py` / `train_segmenter.py` / `segment_octopus.py` (added to the image).
- Client installed here via a venv + get-pip (no system pip). Workspace `amera`.
- Uploaded to the Volume: `harvest_clips` (raw), `dataset_seg/v3` (positives+negatives), `seg_neg` (120
  held-out negatives for the presence eval).
- **v4 plan:** auto-label the 392 harvest clips → merge with v3 → train aug LR-ASPP → presence eval.
  Run: `modal run src/modal_seg.py`.

### v4 RESULT — diversity ↑, but teacher-label quality is the new ceiling
- Ran end-to-end on Modal A100 (image build + parallel auto-label + train + eval, ~50 min).
- **Training videos 77 → 170** (~2× diversity — the goal). But the 392 harvest clips yielded only
  **532 mask pairs**: GroundingDINO's conf gate **rejected most** (harvest is resting/still/camouflaged
  octopus — the detector isn't confident on those). v4 = 6,332 pairs (4,944 pos + 1,388 neg).
- **Mask val IoU 0.490** (v3 0.529) — but on a HARDER val (34 videos vs 15), so ~flat, not a regression.
- **Presence AUC 0.66** BUT **confounded**: the v4 eval used HARVEST frames as positives (intrinsically
  hard/resting; model median area only 0.6%), whereas v3's 0.86 used the clearer original positives. Not
  apples-to-apples. Reflection rejection **held** (neg mask area → 0.0) — the robust property survives.
- **Conclusion: raw video diversity alone isn't the fix.** Both teacher and student are weak on the
  resting/camouflaged regime that dominates real footage. Real levers now: (1) better teacher labels on
  hard frames — the Phase-0 point/negative-prompt fix or lower conf + human verification; (2) a small
  human-verified mask set. More auto-harvested clips just re-hit the same GroundingDINO gate.
- Artifacts pulled: `weights/seg/octo_seg_v4_lraspp.pt`, `data/dataset_seg/harvest/` (532 pairs).
  Modal Volume `octo-seg-data` left in place (more harvest data is coming from the harvester run).

---

## 2026-07-26 — Session 3: Phase-0 teacher fix (point/negative prompts)

Goal: raise teacher-label quality (the diagnosed ceiling — see the failure report) by giving SAM2 a
"what is NOT octopus" cue, and accept the resting/camouflaged frames GroundingDINO was rejecting.

### `auto_segment.py` changes
- **`build_prompts()`** — seed SAM2 with box + POSITIVE points inside the box (centre + interior grid,
  anchors the whole body) AND NEGATIVE points on the brightest regions OUTSIDE the box (metal tools /
  pipes on IR, specular reflections) + frame corners (background). Falls back to box-only on any error.
- **`--min-seed-conf`** flag (was hard-coded 0.60) — lower it to accept camouflaged/resting frames;
  route those through human-verify. **`--no-points`** for A/B; **`--debug-dir/--debug-n`** dump seed
  overlays (box=yellow, +pts=green, -pts=red, mask=green).
- New default is points-ON.

### Local A/B smoke test (2 clips, fps=1, min_seed_conf=0.25, MPS)
| clip | conf | recipe | mask area (median) | good frames |
|---|---|---|---|---|
| Right_Front (colour) | 0.49 | box-only | 0.0406 | 4/20 |
| Right_Front (colour) | 0.49 | **box+points** | **0.0384** | **6/20** |
| Right_Top (IR) | 0.808 | box-only | 0.0424 (holey mask) | 20/20 |
| Right_Top (IR) | 0.808 | **box+points** | 0.0447 (solid body) | 20/20 |

- Colour: points → tighter area (less background bleed) + more area-consistent frames.
- IR: points **fill the octopus body** (box-only left holes); negatives landed correctly on the bright
  specular light + white tools + corners (verified in overlays), none on the octopus. Clean cases not regressed.
- The colour clip (conf 0.49) would have been **rejected by the old 0.60 gate** — now usable.
- Overlays: `scratchpad/seg_ab/{Right_Front,Right_Top}_{box,pts}.png`.
- Caveat: smoke test didn't hit a severe tool-bleed frame; dataset-level validation (mask-area distribution
  + downstream IoU) needs a GPU rebuild. Recipe is sound and non-destructive.

### Next
- Run at scale on GPU: `auto_segment.py --min-seed-conf 0.45 --debug-dir ...` over colour+IR, rebuild
  dataset (points-on), retrain, compare mask IoU + IR acceptance vs the old box-only v1/v2.
- Route the newly-accepted low-conf frames through a human-verify UI before training (Change 4).

### Colab A100 A/B run (2026-07-26) — Phase-0 validated at GPU scale
- Ran `auto_segment.py` (points-on vs `--no-points`) on **27 clips** (Front 8 / Back 3 / Right 8 / Top 8 IR)
  from the Drive `octopus_clips_verified.zip`, on a Colab **A100** (~3–4 s/clip, both recipes). fps=1,
  min_seed_conf=0.30.
- **100% accepted** (27/27) at conf≥0.30 — incl. IR clips at conf 0.44–0.59 that the old 0.60 gate rejected.
- Mean mask area (median-per-clip), box → points:
  | camera | box | points |
  |---|---|---|
  | Right_Front | 0.102 | 0.100 |
  | Right_Back | 0.0595 | 0.0604 |
  | Right_Right | 0.045 | 0.0486 |
  | **Right_Top (IR)** | **0.0664** | **0.0602** (−9%, less tool-bleed) |
  | overall | 0.0698 | 0.0686 |
- Overlays (`results/segmentation/phase0_colab_ab_montage.jpg`, full set on Drive
  `GSOC-Catrobat/seg_phase0_out/`): negatives land on bright lights/tools, positives on the octopus;
  points fill the octopus body (row 02 IR) and don't regress clean cases.
- **Verdict:** non-regressing + correctly-steering + accepts hard frames. IR area reduction is the right
  direction. This sample was fairly clean (no severe box-only tool-bleed case), so the *dataset-level* IoU
  win still needs a full rebuild + retrain. NOTE: the Drive zip is a partial set (mostly Right_Left); a
  full rebuild needs the complete clip set on the GPU box.

### Diversity retrain on Modal (2026-08-07) — the video-diversity fix, measured
Goal: close the plateau by attacking its diagnosed root cause — only ~62 training videos (one week).
Used the **diverse-footage harvest** output (see DATA_PLAN.md / `modal_harvest.py`): the harvest processed
all **1,769 Nity colour videos** and produced **530 clips from 276 distinct source videos across 149 dates**
(sitting on the Modal volume `octopus-harvest-vol`, `sidraj` profile).

- **New app `src/modal_seg_train.py`** (A10G, computes on the volume so no ~20 GB clip transfer): `autolabel`
  (GD+SAM2 teacher, HF models cached to the volume) + `train` (accepts a comma-sep `--ds` list → symlink-merges
  datasets before training). Also fixed `auto_segment.py` filename collisions across resumes (key on
  date/segment, not the run-local index `i` which resets when done-clips are filtered out first).
- **Auto-label** (min_seed_conf 0.60): **178 clips accepted → 732 (image,mask) pairs**; 345 rejected `low_conf`
  (colour clips where GD isn't confident on resting/camouflaged octopus — the 0.60 gate was tuned to kill
  reflections, moot here since Right_Left was excluded from the harvest). Rejected clips write NO manifest row,
  so a later lower-conf pass re-processes exactly those (recoverable). ~2 h on the A10G.
- **Three-way retrain (LR-ASPP, 3.218M params, 60 ep, split BY SOURCE VIDEO):**
  | dataset | videos | pairs | train frames | **best val IoU** |
  |---|---|---|---|---|
  | old v1 (the prior plateau) | 62 | 4,412 | ~3,500 | 0.468 (soft same-week val) |
  | harvest new-only | 100 | 732 | 588 | **0.245** — overfits (val peaks ep2 then declines as loss falls) |
  | **old + new merged** | **176** | **5,144** | 4,300 | **0.494** — best, on a HARDER diverse-date val |
- **Findings:**
  - **Diversity helped, modestly but robustly.** +0.026 over the old plateau AND it's measured on 35 held-out
    videos spanning diverse dates (the old 0.468 was soft same-week val). The merged model holds ~0.49 across
    genuinely varied footage → far more robust even if the headline moved little.
  - **New-only failed for the opposite reason from the old plateau** — 588 frames is too few, so it *overfits*
    (val IoU declines while train loss keeps dropping). Diversity without enough frames doesn't help.
  - **The ceiling is now teacher-label quality, not data.** Merged val plateaus flat at ~0.49 with NO
    overfitting (loss 1.06→0.28, val stays ~0.49). A student can't exceed the noisy GD+SAM2 masks it learns
    from. To go higher: a small HUMAN-verified mask val set (to measure TRUE IoU, not IoU-vs-noisy-teacher) +
    cleaner teacher labels — NOT more clips.
- **Deployable:** `weights/seg/octo_seg_merged_lraspp.pt` (new best positives mask model). On the volume at
  `/weights/octo_seg_merged_lraspp.pt`. Presence/negatives variant on the merged set: not yet retrained.
- **Recoverable next data:** the 345 low-conf clips (lower-conf pass) + the ~1,391 IR clips (needs Phase-0 IR fix).

### Human-verified GT + the real levers (2026-08-09) — resolution & loss, measured leak-free
Auto-labeling was shown unreliable on this footage (GroundingDINO grabs cloth/pipes, CLIP says octopus
everywhere, motion catches TVs/people — every automatic localizer has a distractor; montages in
`results/segmentation/`). So built a **human click-to-SAM2 labeler** (`ui/seg_label.py`, port 8015): motion
pre-seed → SAM2 image-predictor mask → click to refine → save verified frame. Produced **412 positive +
87 negative** human pairs over **~35 videos** → `data/dataset_seg_human/`.
- **Clean human-only model = val IoU 0.454** (its own by-video split) — NOT better than noisy-teacher.
  Confirmed label quality was never the ceiling.
- **Eval-review UI** (`ui/seg_eval_review.py`, port 8016): step through held-out val frames, pred (green) vs
  GT (red), per-frame IoU, sort worst-first, flag bad GT.
- **Found the REAL levers by looking at failures** (pred vs GT): (1) **input resolution** — every model trained
  at 256², where the median octopus (2.5% of frame) is ~40×40px and tentacles are 1–2px = unrepresentable;
  (2) **loss** — symmetric Dice+BCE doesn't punish the under-segmentation (missed tentacles/small octopus =
  false negatives). Added **512² training** + **Focal-Tversky loss** (β>α penalizes FN) to `train_segmenter.py`
  (`--in-size --loss focal_tversky`).
- **LEAKAGE CAUGHT:** first human-val comparison was contaminated — `old_hq` (HQ re-label) shares the same
  2026-02 source videos as the human labels, so every hqfull model had trained on the val videos (0.71 was
  train-on-test). Added `--holdout-videos` to force test videos out of ALL training sources.
- **CLEAN leak-free result** (5 human test videos excluded from all training, eval on human masks):
  | model | mean IoU | median | misses | areaErr |
  |---|---|---|---|---|
  | human-only (256, Dice+BCE) | 0.466 | 0.517 | 9/122 | 1.11% |
  | **clean 512 + Focal-Tversky (+human+HQ auto)** | **0.608** | **0.666** | **6/122** | **1.07%** |
  → **+0.14 mean (+30% rel), leak-free.** The 512-res + Tversky levers are REAL (clean 0.608 ≥ leaked 0.596).
  Best model: **`weights/seg/octo_seg_clean512tv_lraspp.pt`** (LR-ASPP 3.2M, in_size 512, focal_tversky).
- **Honest ceiling read:** ~0.61 mean / 0.67 median on a genuinely hard small-camouflaged-object task with a
  3.2M student. areaErr ~1% = size is accurate; residual IoU gap is thin-tentacle boundary + ~6 hard misses
  (small/resting octopus). For the project's needs (presence + area/posture) this is usable. Further IoU would
  need a bigger student (breaks deploy-size constraint) or 768² (diminishing). Next: presence-gate AUC eval
  (using the 87 negatives) head-to-head vs the CLIP gate.
- **Human-only vs blend (same clean held-out test, 2026-08-09):** volume beats purity.
  | model | train data | mean IoU | median | misses |
  |---|---|---|---|---|
  | human-only 256/Dice | 290 clean frames | 0.466 | 0.517 | 9/122 |
  | human-only 512/Tversky | 290 clean frames | 0.505 | 0.577 | 9/122 |
  | **BLEND 512/Tversky** | **3450 frm (8% human + 92% HQ auto)** | **0.608** | **0.666** | **6/122** |
  Config fixes (512+Tversky) generalize (+0.04 on human-only too), but 290 clean frames overfit — the blend's
  12× volume wins. Making human-only competitive would need ~150+ human videos. **Deployable best = the blend
  `octo_seg_clean512tv_lraspp.pt`.** All levers now tested: labels(volume wins) / diversity(helped) /
  resolution 256→512(helped) / loss Dice→Tversky(helped) / leakage(caught via --holdout-videos).

### HQ teacher upgrade (2026-08-07) — the label-quality ceiling, attacked
Since the merged plateau is teacher-label-quality-bound, upgraded the teacher: **GroundingDINO-base + SAM2-large**
(`auto_segment.py --gd-model base --sam2-model large`, parametrized). Validated on 15 harvest clips (clean masks
on diverse new footage; first-15 acceptance 53% but the *full* run settled to ~29%, so GD-base recovers little
extra data vs tiny — the win is mask QUALITY, not quantity).
- **Key evidence the ceiling is label noise (not model):** the merged model (trained on TINY-teacher masks,
  val IoU 0.494 vs tiny masks) scores **IoU 0.697 / median 0.80 vs HQ-teacher masks** on 32 harvest frames.
  CAVEAT: those frames are from clips in the merged TRAIN set (split-membership not reconstructed) → optimistic,
  not a clean test. But the direction is unambiguous — measured against cleaner masks the "error" nearly halves,
  i.e. most of the 0.49 gap was tiny-teacher noise. True quality is likely ~0.7.
- **HQ harvest re-label DONE:** 530 clips → **185 accepted / 740 pairs** (vs tiny 178/732 — same count, so the
  win is mask QUALITY not quantity; GD-base full-run acceptance ~35%, the first-15 53% was a lucky sample).
- **Merged HQ retrain (old-tiny + harvest-HQ):** val IoU **0.508** vs 0.494 (merged tiny) = **only +0.014**.
  Diagnostic, not disappointing: HQ masks cover just the harvest 740/5,152 = **14% of pairs**; the old-tiny bulk
  (86%) dominates train AND val, so we improved 14% of labels and got a proportional bump. **The old-tiny bulk
  is now the binding constraint.** Model `weights/seg/octo_seg_merged_hq_lraspp.pt`.
- **HQ re-label of the 1,104 old v1 clips DONE:** 85% acceptance (old close-up footage is easy for GD-base) →
  **3,991 HQ pairs** → `/data/dataset_seg_old_hq`.
- **Fully-HQ merged retrain (old-HQ 3,991 + harvest-HQ 740 = 4,731 pairs, 177 videos, ALL clean labels):**
  **val IoU 0.506** (Dice 0.620, **areaErr 0.0143** — the lowest of any run). Plateaus flat, no overfitting.

### HONEST NEGATIVE RESULT (2026-08-08) — HQ labels did NOT move held-out IoU; the metric is the issue
Held-out val IoU across teacher quality: tiny **0.494** → 14%-HQ **0.508** → **100%-HQ 0.506**. **Flat.**
- **My "ceiling is label quality" hypothesis was WRONG.** The "0.49→0.70 vs HQ masks" evidence that motivated the
  HQ chain was **train leakage** (those frames were in the model's train set). The clean held-out HQ-vs-HQ number
  is ~0.51, same as tiny. Teacher-label quality is NOT the binding ceiling.
- **The real ceiling is the tiny student's generalization** on this hard, diverse task: it makes **right-SIZED
  but imperfectly-LOCALIZED masks** (areaErr only 1.4%, but pixel-IoU 0.50 / Dice 0.62). This is the R3
  "right-sized blob, wrong place" mode — not fixable by cleaner labels or more data (both tried).
- **Likely the metric is wrong for the goal.** pixel-IoU-vs-teacher-masks caps ~0.5 (teacher masks aren't perfect
  GT either), but the PROJECT needs presence + body-AREA (posture) + masked-motion — and **area is accurate to
  1.4%**. So the model may already be adequate for its actual downstream use.
- **Models (all local `weights/seg/`):** `octo_seg_hqfull_lraspp.pt` (cleanest labels, likely best qualitative
  masks), `octo_seg_merged_hq_lraspp.pt`, `octo_seg_merged_lraspp.pt`. Datasets on volume: `dataset_seg_old_hq`,
  `dataset_seg_harvest_hq`.
- **Real next levers (NOT more teacher/data — those are exhausted):** (1) a small **human-verified mask val set**
  to get TRUE quality vs the teacher-proxy; (2) **evaluate on the downstream task** — does the seg area-gate beat
  the CLIP presence gate on reflections/empties, and is the area/posture signal usable in the behaviour pipeline?
  (3) if pixel-boundary quality genuinely must rise: a **temporal** student (uses motion — the R3 "needs temporal
  features" insight), since per-frame localization is the failure mode. Bigger student is off the table (deploy
  size constraint).

## Skeleton tracking v2 (2026-08-13) — measured on a fixed 10-clip eval set
Downstream of segmentation: the anatomical skeleton pipeline (`src/skeleton/`) turns masks into
mantle/head/arm graphs + per-arm kinematics. Skeletonization phases 1-3 (mask-prep tuning +0.17 arms;
best-frame seeding +3.67 median arms) were followed by **tracking v2** — all changes measured
before/after on a FIXED 10-clip eval set (`src/skel_eval_tracking.py`, all 6 behaviours, metrics in
`src/skeleton/track_metrics.py`: teleport rate / fragmentation / coverage / in-mask / arm-count std).

| run | teleport | arms | verdict |
|---|---|---|---|
| baseline (chain, centroid prior) | 14.85% | 5.1 | — |
| flow (DIS per-node prior, unchanged gates) | 15.71% | 4.9 | ❌ better prior + same gates admits MORE noise |
| **flow2 (flow + tightened gates when prior validates)** | **14.27%** | 4.9 | ✅ adopted (default) |
| global tracklet association (2 tunings) | 15.07 / 14.35% | 4.2–4.3 | ❌ negative: wins big on easy clips, breaks on hard, drops arms; kept opt-in `method='global'` |

**Phase 3 (adopted): occlusion-honest states.** Every node: `detected | fitted | occluded`;
`compute_motion` emits rows only from evidence-backed samples. Key finding: **occluded_frac = 41.7%**
— nearly half of arm samples were evidence-free holds; the tracker's smoothness was largely inertia
(teleport_confident 16.5% > overall 14.3% because held nodes' fake stillness flattered the average).
Kinematics (`batch_skeleton_motion.py`) now gate on states and report `occluded_frac` per clip.
Chain: seg mask (EMA) -> fixed union bbox -> per-frame detect -> best-seed bidirectional chain with
flow prior -> global-consistent-ish IDs -> state-gated smoothed motion -> `kinematics` in
behaviour_records.json. UIs: 8017 (3-way viewer + trails), 8018 (phase results).

## Skeleton-extraction accuracy phases (2026-08-14) — frozen 50-frame benchmark
All measured on `data/skel_bench50/frames.json` (50 fixed frames, human-GT + model masks), metric =
arms/frame with a tip-correctness guard (selected tips must land near true silhouette protrusions).
| phase | change | model-mask arms | note |
|---|---|---|---|
| baseline | — | 2.82 | tip-match 0.876 |
| B | selection floors 2.5x/0.30, prefix 0.70 | 3.65 | old floors discarded real curled arms |
| C | prep: bin_thresh 96, spur width_factor 0.35 | 4.16 | blur@112 erased thin arms; spur rule ate short ones |
| head fix | anatomical head = neck on mantle->crown line | — | head plausible 9% -> 96% |
| hysteresis | prob-field masks (weak-thr grid) | 4.24 | NULL (+0.08, noise): model truly blind to tentacles |
| **D** | **seg retrain thin768: 768^2 + Tversky beta 0.8** | **4.64** | internal val IoU 0.584 (best seg model); tip-match -0.027 (in tolerance) |
Ceilings: clean-GT-mask extraction 6.15, 2D silhouette ~7, biology 8. thin768 is now the skeleton
pipeline's default mask model (`octo_seg_thin768_lraspp.pt`); clean512tv remains for presence/area uses.
Remaining levers: more thin-structure seg gains (bigger backbone / distill), or learned RGB keypoints
(the only way past the silhouette ceiling).
