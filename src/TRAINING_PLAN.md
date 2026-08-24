# Caption-student training plan (Qwen3-VL-2B ← Qwen3-VL-235B)

Distill the 235B teacher captions into the **smallest** student that still captions well.
Caption-only (no ethogram label — that gets its own model later). Designed so that when
more clips are extracted, we **retrain from base on the bigger cumulative dataset** rather
than resuming (Option A: no catastrophic forgetting, reproducible, cheap at this scale).

## Locked decisions
- **Task**: one-sentence caption only. No ethogram label in the target.
- **Loss**: causal-LM cross-entropy over the caption tokens (prompt + image tokens masked, `label=-100`).
- **Student**: `Qwen3-VL-2B-Instruct`, QLoRA 4-bit. Same family as the teacher (best distillation
  alignment) + smallest tier + strongest multi-frame handling. **Fallback**: `Qwen2.5-VL-3B` if
  Qwen3-VL QLoRA tooling misbehaves on Colab.
- **Teacher captions**: prefer `caption_235b`, else `caption` (so the target is the 235B teacher
  wherever it exists).
- **Data**: present-only, CLIP-embedding dedup, split train/val **by source video** (no leakage),
  best-N CLAHE frames per clip — *identical* to what the teacher saw at caption time.
- **Continue-when-more-data**: rebuild the versioned dataset snapshot (now bigger) → retrain from base.

## Phases
- **Phase 0 — tooling smoke test (Colab, ~10 min)**: load `Qwen3-VL-2B-Instruct` in 4-bit + run ONE
  LoRA training step. If it fails (arch/PEFT/bitsandbytes), drop to `Qwen2.5-VL-3B`. Do this before
  building anything around the model.
- **Phase 1 — dataset builder (local, `build_caption_dataset.py`)**: index → present clips → dedup →
  split-by-video → extract best-N CLAHE frames → emit `dataset/vN/` (frames + `train.jsonl` +
  `val.jsonl` + `snapshot.json`) + a zip for Colab. **Run after captioning finishes.**
- **Phase 2 — training notebook (Colab, rework `train_caption_student.ipynb`)**: load student 4-bit +
  LoRA, read manifest, build chat examples (image(s) + describe-prompt → assistant caption), QLoRA
  train (loss on assistant tokens only), save adapter (zipped) tagged with the snapshot version.
- **Phase 3 — eval (Colab/local)**: generate on the held-out val split; caption quality =
  embedding-similarity + ROUGE vs teacher, plus eyeball via `compare_base_lora.py`. Base vs LoRA.
- **Phase 4 — continue workflow**: new clips → caption → rebuild dataset (v+1) → retrain from base →
  re-eval. Every adapter records which snapshot version it trained on.

## Dataset snapshot format (`src/dataset/vN/`)
- `frames/<clipid>_fNN.jpg` — the best-N enhanced frames per kept clip.
- `train.jsonl` / `val.jsonl` — one line per clip: `{"clip_path","caption","frames":[...]}`.
- `snapshot.json` — version, counts, dedup threshold, caption source, val fraction, and the list of
  clip_paths (so the snapshot is fully reproducible/auditable).

## Open knobs (defaults; tune later)
- Frames per clip `N_KEEP` = 6 (matches teacher). Dedup threshold ≈ 0.93 (within-video). Val = 10%.
