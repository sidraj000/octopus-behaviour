# Benchmark suite

Three **frozen** benchmarks. Every improvement claim in this project — and every number in the
OCEANS 2026 paper — is measured here, before and after. Run them all with one command:

```bash
venv/bin/python3 src/benchmarks.py --suite all --tag <run-name> --latex
```
Results append to `data/benchmarks.json` (keyed by tag); `--latex` writes
`OCEANS_2026/assets/benchmarks_table.tex` for the paper.

## Non-negotiable rules
1. **Frozen sets.** The frame/clip lists are committed and must not be regenerated to suit a result.
2. **Split by SOURCE VIDEO, never by frame.** Frames from one recording are near-duplicates; a
   frame-level split leaks. We shipped this bug once (an apparent 0.49→0.70 gain evaporated under a
   video-level holdout) — see `SEGMENTATION_LOG.md`.
3. **The holdout videos are excluded from every training source**, not just the one being trained.
4. **Report the metric that can't be gamed** (see the tip-F1 note below), and report negatives.
5. **A leakage check must account for every row of the manifest it reads.** Print
   `rows parsed / rows total` and fail loudly on any unparsed row. `data/frames/manifest.csv` mixes two
   filename conventions — `2026-04-05_2015_Right_Front_...` (date_HHMM) and
   `p0.50_2026-02-20_095420_Right_Back_...` (date_HHMMSS) — so a regex written for one silently skips
   the other. A check matching only the 6-digit form ignored **8,658 of 11,224 rows (77%)** and
   under-reported the detector's training set as 24 videos / 5 Right_Left / 1 overlap when the truth is
   **32 videos / 11 Right_Left / 5 overlapping videos**. A leakage check that quietly matches nothing
   looks exactly like a leakage check that passes.
6. **Different arms of a head-to-head must be scored on the IDENTICAL set.** If leakage forces frames
   out for one arm, drop them for every arm and restate, rather than comparing arm A on 34 frames with
   arm B on 28.

---

## SEG-TEST — segmentation quality (leak-free, human-verified)
| | |
|---|---|
| Data | `data/dataset_seg_human/` — 412 human masks + 87 empty-mask negatives, 35 source videos |
| Test split | 5 held-out source videos → **122 frames** (+19 negatives), excluded from all training |
| Labels | human, via click-to-SAM2 (`ui/seg_label.py`, port 8015): a human clicks the octopus, SAM2 masks it, the human accepts/corrects |
| Metrics | mask **IoU** (mean, median); **body-area error** (%) — the quantity the behaviour analysis actually consumes; **presence AUC** — does mask area separate present frames from empty ones |

Holdout videos: `2026-02-21/150002`, `2026-02-21/183003`, `2026-02-22/153002`,
`2026-02-22/190003`, `2026-02-23/170003`.

## REFL-24 — reflection rejection (presence, leak-free by construction)
| | |
|---|---|
| Data | `data/reflection_negatives/` — Right_Left frames, **≤2 per clip, one per source video** |
| Current set | **REFL-34** — 34 confident negatives / 27 source videos (segmenter-only arms) |
| Head-to-head set | **REFL-28** — 28 frames / 22 videos: REFL-34 minus the 5 recording **sessions** present in the CLIP detector's training manifest (2026-02-20 at 0954/1724/1754/1824/1854). Required whenever the detector is one of the arms, and applied to **every** arm so the sets stay identical |
| Why leak-free | `thin768` trained on `/dataset_seg_thin768` = 4,965 images, **0 Right_Left** (asserted file-by-file, not assumed). The camera is excluded by construction in `auto_segment.py` and absent from the human label set |
| Metrics | **FP rate at fixed present-recall (0.90/0.80)** and at the deployed gate (`area ≥ 0.01`) — headline; **AUC** secondary, with CI **cluster-bootstrapped by source video** |
| Runner | `src/eval_reflection_presence.py` (sampler: `src/reflection_negatives.py`) |

> **Never pool negative types.** Empty-tank negatives (same cameras as the positives) and reflection
> negatives measure different failure modes and are always reported as separate rows. Pooling them
> silently redefines the metric.

> **The leakage unit is the recording SESSION (`date/segment`), not the camera.** Two cameras in one
> session record the same scene, lighting and animal state at the same instant. Excluding only the
> *Right_Left* sessions from the detector's training manifest leaves 4 further sessions it had already
> seen through another camera — we made that mistake once (33 frames instead of 28) before correcting.
> Sessions are matched on `(date, HHMM)` because the manifest mixes two filename conventions; that
> normalisation over-excludes if two recordings start in the same minute, which is the safe direction.

> **SEG-TEST's 19 empty-tank negatives come from only 2 source videos — 18 of them from
> `2026-02-21/183003` alone.** Any presence number computed against them is effectively a single-video
> estimate: report it descriptively, never with an AUC or a CI, and never order it against a
> many-video estimate. Fixing this (negatives drawn from many videos) is the highest-value repair
> available to the presence benchmark. Count n in **videos** at every stage, negatives included.

> **Reflection frames are NOT automatically negatives.** Review of 30 frames found **10% unmistakably
> contain the octopus** (up to 20% including ambiguous cases) — the animal spreads on the glass beside
> its own mirror image. Every frame must be reviewed before it is scored, and ambiguous frames are
> excluded rather than counted as empty. This is the same trap as the 2026-06 hard-negative mining,
> where 166 of 232 assumed-negative frames contained the animal.
> *Provenance caveat:* the current pilot labels were produced by an AI vision model, not a human, and
> are staged for human confirmation.

## SKEL-50 — per-frame skeleton quality
| | |
|---|---|
| Data | `data/skel_bench50/frames.json` — **50 frozen frames** across 20 source videos, each with image + human mask + model mask |
| Metrics | **arm-tip F1** (headline), tip precision, tip recall, head error (body radii), arms/frame (descriptive only) |
| Ground truth for tips | protrusions of the **human** mask (`finger_tips`, `min_prominence=1.8`, `min_len_frac=0.10`) — mean 5.7 / median 6 / max 8 per frame; a predicted tip matches a GT protrusion within 5% of the image diagonal, greedy 1-1 (so a duplicate arm cannot match the same protrusion twice) |

> **The GT detector was itself validated — and was initially wrong.** With the library default
> (`min_len_frac=0.06`) two peaks only 6% of the contour apart counted as separate arms, giving
> **mean 9.7 / max 14** protrusions per human mask; the 8-cap then bound in **80%** of frames, so
> recall was being scored against padded, partly-fictional arms. Requiring peaks ≥10% of the contour
> apart (closer peaks belong to the *same* arm) fixed it. Effect on the shipped skeleton config:
> tip-F1 **0.441 → 0.539** (precision 0.685→0.722, recall 0.380→0.502). Lesson worth keeping: when a
> metric is introduced, measure the metric's own ground truth before trusting any ranking it produces.
| Head GT | human clicks on the eyes via `ui/skel_static_viewer.py` (port 8019) → `data/skel_bench50/head_gt.json`; head error is reported in **body radii** so it is pose/scale independent |

> **Why tip-F1 and not "arms per frame".** Arm count is not a score: *fewer* can be *better*.
> Anti-mess quality gates removed ~1.3 duplicate/tangle arms per frame and the count fell
> 4.80 → 3.48 while the output got visibly cleaner. Precision alone is equally gameable (emit one
> obvious arm). F1 against the human mask's protrusions penalises **both** spurious arms and missed
> arms, so it is the number to optimise and to publish. Arm count stays as a descriptive statistic.

## TRACK-10 — temporal tracking quality
| | |
|---|---|
| Data | 10 frozen clips (`EVAL_CLIPS` in `src/skel_eval_tracking.py`) spanning all 6 behaviour classes, 3 colour cameras, 3 dates |
| Metrics | **teleport rate** (per-node steps jumping > 4× that node's median step — proxy for identity swaps), **teleport-confident** (same, restricted to evidence-backed samples), **occluded fraction** (share of arm samples that are evidence-free holds), **coverage**, **fragmentation**, in-mask fraction, arm-count stability |

> **Why occluded-fraction matters.** Naive tracking looked smooth because 42% of arm samples were
> held with no evidence. Kinematics are computed only from `detected`/`fitted` samples; this metric
> keeps that honesty visible.

---

## Current results
See `data/benchmarks.json` (authoritative, tagged per run) and the summary table in
`PAPER_NOTES.md`. Reproduce any row with:
```bash
venv/bin/python3 src/benchmarks.py --suite seg,skel,track --ckpt weights/seg/<model>.pt --tag <name>
```
