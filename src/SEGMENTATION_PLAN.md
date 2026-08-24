# Octopus Segmentation Plan — a tiny mask model for cleaner extraction + richer behaviour

**Status:** proposed (2026-07-21). Follows the GroundingDINO + SAM spike (validated below).
**Design constraint (hard):** the *deployed* model must be the **smallest model that still gives
good masks** — single class (octopus vs background), low input resolution, runs on the Mac at
gate speed. GroundingDINO/SAM are the *teacher/auto-labeler only*, never the deployed model.

## Why segmentation (two payoffs)

1. **Better EXTRACTION** — the current gate (CLIP `p_visible` + whole-frame motion) over-extracts:
   it fires at `p_visible=1.0` on human reflections in the glass (esp. Right_Left), and the motion
   gate counts IR-lamp flicker anywhere in frame. A mask fixes both:
   - **Presence:** "is there an octopus-shaped mask, and how big?" beats a whole-frame classifier —
     reflections get a thin/absent mask.
   - **Masked motion:** count changed pixels *inside the octopus only* → flicker/people/reflections
     move zero octopus-pixels and can't trigger a clip.
2. **Richer RECORDS (free at gate time)** — the mask byproducts drop straight into each behavioural
   record: **body area (posture spread), octopus-only colour, colour-change, masked motion** — the
   exact signals the affect model is missing (colour-change "needs animal segmentation").

## Evidence from the spike (2026-07-21)

Ran GroundingDINO-tiny (transformers, ~690 MB, CPU) + SAM-vit-base (~375 MB, CPU) on Mac:
- **Detection** grounds "octopus" on colour AND IR: scores 0.47–0.83; reflections score low/scattered
  (a useful negative signal). One model covers all cameras.
- **Masks** (box→SAM): **surgically clean on colour** (Right_Back climb-out tracked tightly, area
  ~5–9% of frame). **IR works on the animal but over-segments onto bright metal tools/pipes** (the
  octopus glows white and SAM grabs connected bright regions in the loose box) — must be fixed in Phase 0.
- **Speed:** ~5 s/frame on CPU → teacher-only. Confirms we must distill a tiny fast student.
- Demo artifacts: `data/segmentation_demo/{Right_Back,Right_Top}_*_seg.mp4` + stills.

## The smallest-model strategy (the point of this plan)

Because the task is one-class and the target is a large blob, we can be aggressive on size:
- **Input resolution:** 256×256 (maybe 320²), not full frame. Masks upsample fine.
- **Architecture candidates, smallest-first** (pick by the size/IoU curve, Phase 2):
  | Candidate | ~Params | ~INT8 size | Notes |
  |-----------|---------|-----------|-------|
  | Custom compact U-Net (few down/up blocks) | 0.3–1 M | ~0.5–1 MB | likely enough for 1 class; smallest |
  | LR-ASPP / MobileNetV3-small seg head | ~1–2 M | ~1–3 MB | strong efficient baseline |
  | YOLO11n-seg / YOLOv8n-seg | ~3–3.4 M | ~6–7 MB | easy training/eval, larger |
- **Shrink further:** distill from SAM masks (soft targets), INT8 quantize, prune. Report the whole
  **IoU-vs-size curve** and pick the smallest model above the quality bar — don't just ship the first.
- **Success bar:** mask **IoU ≥ 0.85** vs verified val on colour cameras (≥0.75 IR post-fix); mean
  body-area error < ~15%; inference **≥ current gate throughput** (1 fps whole-video scan) on Mac CPU,
  real-time on MPS.

## Phases

### Phase 0 — Harden the auto-labeler  *(prerequisite; ~small)*
- Fix IR over-segmentation: SAM **point prompt** at the box's brightest octopus region + **negative
  points** on tool/pipe blobs; or post-filter masks by compactness/solidity; evaluate **SAM2** (video
  mask propagation — label one frame, track the clip) as an alternative that may also cut cost.
- Mask-quality gate: area bounds, single dominant connected component, mask∩box IoU.
- Multi-box union for sprawled arms; box-confidence threshold ≈ 0.4.
- **Deliverable:** `src/auto_segment.py` — clip/frame → clean mask + quality score. No deployed model yet.

### Phase 1 — Build the mask dataset  *(offline auto-labeling)*
- Sample frames from present-on-disk clips; **colour cameras first**, IR after Phase 0 fix.
- Run `auto_segment.py`; keep only high-quality masks; **balance across cameras + behaviours**
  (over-sample rare Swimming/Crawling).
- **Human verification** of a subset via a small accept/reject/erase UI → clean **val set** (this is
  also the ground truth we've otherwise been missing).
- Split **by source video** (no leakage).
- **Deliverable:** `src/dataset_seg/vN/` — images + binary masks + train/val split + snapshot.json.

### Phase 2 — Train the tiny segmenter  *(the deliverable model)*
- Train candidates smallest-first on the auto-masks; distill from SAM soft masks where it helps.
- Evaluate **IoU vs size** on the verified val (per camera); quantize INT8; re-measure.
- Pick the **smallest model clearing the success bar**.
- **Deliverable:** `src/train_segmenter.ipynb` (or `.py`) + `weights/octo_seg_vN.*` + an inference
  module `src/segment_octopus.py` (`mask, area = segment(frame)`).

### Phase 3 — Integrate into extraction  *(the payoff)*
- New **presence gate**: mask present & area ∈ [min,max] — replaces/augments CLIP `p_visible`.
- **Masked motion**: changed pixels within the mask — replaces whole-frame motion.
- Enrich each record with `body_area`, `octopus_colour`, `colour_change`, `masked_motion`.
- **A/B vs current extractor** on a held-out video set: does it cut the reflection/flicker false
  positives while holding recall? (This is the acceptance test.)
- **Deliverable:** mask-based gate wired into `extract_octopus_clips.py` / `local_pipeline.py` behind a flag.

### Phase 4 — Feed the behaviour analysis
- Add area / octopus-colour / masked-motion channels to `analyze_behaviour.py`; arousal model uses
  **real posture-spread**; **colour-change becomes measurable** (octopus-only pixels) on colour cameras.

## Risks / open questions
- **Camouflage** — segmentation is weakest exactly when she blends into rock (itself a behaviour).
  Acceptable: the gate keeps visible/active clips; note it as a known limitation, don't over-fit to it.
- **IR bright-object confusion** — Phase 0 must solve it or IR stays colour-blind (shape/area only).
- **Tiny-model quality floor** — if <1 M params can't clear IoU 0.85, step up one tier; the curve decides.
- **Disk is tight** (~6 GB free) — SAM2/checkpoints + the frame dataset; prune caches first.

## Decisions needed before Phase 0/2
1. **Auto-labeler:** SAM-vit-base (have it, box prompt) vs **SAM2** (video propagation, may cut cost + fix IR). *Rec: try SAM2.*
2. **Deployed architecture family:** custom compact U-Net (smallest) vs LR-ASPP/MobileNetV3 vs YOLO-seg-nano. *Rec: benchmark U-Net + LR-ASPP, hold YOLO-nano as the fallback.*
3. **Scope of v1:** colour cameras only (ship faster, clean) vs colour+IR from the start. *Rec: colour-first v1, IR in v2.*
