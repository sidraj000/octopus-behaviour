# Data-Generation Plan — more diverse footage for better caption + segmentation students

**Status:** proposed (2026-07-23). Goal: raise BOTH the caption student and the segmentation
student past their current ceilings by generating **more diverse training data**. Neither is
architecture-limited — both are **footage-diversity-limited**, and it's the *same* limit.

## The shared diagnosis (why both models are stuck)

- **Segmentation:** mask IoU plateaus ~0.47 (bar 0.85). A100 run diagnosed it as a
  **video-diversity generalization gap** — only ~62 distinct *training videos* (train IoU 0.68 /
  val 0.47; it fails by *mislocating* a right-sized blob). Not fixable by architecture/aug — needs
  more distinct verified videos. (`src/SEGMENTATION_LOG.md`.)
- **Captioning:** student v1 is decent (emb-sim 0.834) but trained on the same narrow footage, so it
  generalizes to *new* scenes no better than its scene coverage allows.
- **Root cause (shared):** the entire corpus — 13,342 clips, 252 distinct (date,segment) videos — comes
  from **only 7 dates, all in one week (2026-02-20 … 02-28)**. Lots of clips, one week of scenes.
  More clips from that week does NOT help; **more distinct days does.**

## The fix: generate data along the axis that's missing (calendar/scene diversity)

Harvest footage from **many more distinct days across the full recording timeline** (different
lighting, den layouts, times of day, behaviours), auto-label it with the existing teachers, and
retrain both students. Same pipeline feeds both models.

## Phases

### Phase A — Survey + harvest diverse footage  *(the root fix)*
- **Survey the server** (`repo.octopus-intelligence.org`) for ALL available dates/segments beyond the
  7 we've used — enumerate the calendar, not just download. (creds in `.env`, den cameras only.)
- **Harvest a temporally-spread sample**: prioritize *new days* over more clips/day; spread across
  weeks/months and across the circadian cycle (day + night/IR). Den cameras:
  Right_Front / Right_Back / Right_Right (colour) + Right_Top (IR). Drop Right_Left (reflections).
- **Respect the throttle** (memory `server-throttling-sustained-load`): concurrency **2–3**, back off
  when it collapses to ~KB/s. Run harvest as a long background job; log + resume.
- **Target:** get to **≥150–250 distinct source videos** spanning **≥30+ distinct days** (from 62 today).
  Video *count/diversity* is the metric, not clip count.

### Phase B — Extract clips with the IMPROVED presence gate
- Run `extract_octopus_clips.py` on the new footage, but swap the CLIP presence gate for the **v3
  segmentation presence gate** (`weights/seg/octo_seg_v3_lraspp.pt`, AUC 0.86 / 0.99 vs reflections,
  0% reflection-FP). Cleaner clips in one pass — fewer reflection/IR-noise false positives than CLIP.
- This also ships the documented "wire v3 gate into extraction" step and A/Bs it in production.

### Phase C — Auto-label the new clips on the A100  *(both teachers)*
Get new clips to the GPU box (rsync, as done for the first batch), then:
- **Masks (segmentation):** `auto_segment.py` (GroundingDINO→SAM2). **Sample for VIDEO diversity** —
  cap frames-per-video (keep `N_PER_CLIP` small) and maximize distinct videos, since that's the
  bottleneck. Confidence-gate (0.60) as before.
- **Captions:** `caption_openrouter.py` (Qwen3-VL-235B teacher) over new present clips → `caption_235b`.
- **IR Phase-0 fix (unlocks ~1,391 IR clips):** point-prompt + negative-points (or SAM2 refinement) so
  Right_Top masks stop grabbing bright tools; only then fold IR into the mask set.

### Phase D — Build human-verified VAL sets  *(currently missing — do in parallel)*
Without these we cannot tell whether more data actually helped (the A100 run had no verified masks).
- **Segmentation val:** ~200–300 human-verified masks (accept/reject/erase UI), **held out BY VIDEO**.
  This is the honest IoU yardstick and the definition of "better."
- **Captioning val:** extend `data/caption_training_set.json` (already has human A/B labels) to a solid
  held-out set of verified captions across the NEW scenes.

### Phase E — Retrain + measure
- **Segmenter:** retrain on the larger, more-diverse mask set (`train_segmenter.py`), sweep `--base-ch`
  for the IoU-vs-size curve, evaluate on the Phase-D verified val. **Bar: IoU ≥ 0.85 on held-out videos.**
- **Caption student:** rebuild the snapshot (`build_caption_dataset.py`) on the enlarged set, retrain
  the QLoRA (`train_caption_student_qwen3vl.ipynb`), eval emb-sim / rougeL on the verified val — must
  beat v1's 0.834 / 0.455 on NEW scenes.

### Phase F — Close the loop
Better segmenter → better presence gate (Phase B) → cleaner clips → better data. Re-run when the next
footage batch arrives (retrain-from-base on the cumulative snapshot, per `TRAINING_PLAN.md` Option A).

## Targets (the definition of done)
| Model | Now | Target |
|-------|-----|--------|
| Distinct training videos | ~62 | ≥150–250, ≥30+ days |
| Segmentation mask IoU (held-out video) | 0.47 | **≥0.85** (colour) |
| Segmentation presence gate | AUC 0.86 | hold ≥0.85, 0% reflection-FP |
| Caption student (new scenes) | 0.834 emb-sim | **> 0.834** on verified val |
| Verified val sets | none (seg) / partial (cap) | seg ~200–300 masks; cap held-out |

## Risks / open questions
- **Server may not have many more distinct days** — Phase A survey answers this first; if the archive
  really is ~1 week, diversity must come from harder augmentation + the IR unlock, and the IoU bar may
  need revisiting. Survey before committing to harvest volume.
- **Throttling** caps harvest speed (2–3 concurrency) — budget days, not hours, for a big pull.
- **Labeling cost/time**: masks on A100 ~13 s/clip; 235B captions ~$0.0006/clip. Both cheap; the
  gate is harvest bandwidth.
- **Where harvest runs:** locally (has `.env` server creds) → rsync new clips to A100 for labeling,
  same path as the first batch.

## Decisions needed
1. **Harvest volume** — how many new days to pull (survey first, then pick). *Rec: aim ~30–50 days spread across the archive.*
2. **Colour-first or colour+IR** — IR needs the Phase-0 fix. *Rec: colour first (fast win), IR unlock in parallel.*
3. **Verified val sets** — who labels (you, via the review UIs). *Rec: build the seg val set first; it's the missing yardstick.*
