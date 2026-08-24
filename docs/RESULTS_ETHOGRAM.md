# Ethogram classification — results

Every number here is reproducible from a committed script on a frozen dataset. Provenance is given
per table (script + tag). Cross-references `Rn` point at the running ledger in `PAPER_NOTES.md`.

**Metric convention.** Macro-F1 throughout, never accuracy. The six classes run 45.8% to 3% of the
corpus, so accuracy rewards collapsing onto `No octopus`. Accuracy is reported only alongside, to
show the gap. Per-class figures always carry **precision and recall separately**, because one class
here looks strong on F1 while being a sink (see §5).

**Splits are by SOURCE VIDEO, never by clip.** A held-out clip from a training video is not held out
in any useful sense — same tank, same lighting, same animal, minutes apart. The video count is the
real sample size, so it is reported next to every clip count.

---

## 1. The task

Six classes, in one head. `No octopus` is **one of the classes**, not a presence filter in front of a
behaviour classifier, because that is how the system is deployed: a clip arrives and the question is
what is happening in it, with "no animal" a legitimate answer.

    No octopus · Resting/stationary · Exploration/manipulation ·
    Locomotion (crawl/swim) · Reaching out of water · Human/enrichment interaction

Two deliberate reductions from the 7-class sheet (`src/ethogram_list_v2.json`):

- **`Crawling` + `Swimming/jetting` → `Locomotion (crawl/swim)`.** The teacher cannot separate them
  (5 of 40 behaviour errors were Swimming→Crawling, all one direction) and Swimming had 42 clips
  across 14 videos, too few for a per-class F1 under a video-level split. Training-time mapping only —
  the extraction prompt and every stored record keep the 7-class vocabulary, so the merge is
  reversible and earlier κ figures stay comparable.
- **`Colour change / defensive` dropped**: 1–3 clips corpus-wide. Unlearnable, and reporting an F1
  on it would be noise with a decimal point.

---

## 2. Dataset — frozen v1

`src/build_ethogram_dataset.py --version v1` → `src/dataset_etho/v1/`

| split | clips | videos | No octopus | Exploration | Human | Locomotion | Reaching | Resting |
|---|---|---|---|---|---|---|---|---|
| train | 3,019 | 133 | 1,419 | 495 | 141 | 212 | 215 | 537 |
| val | 655 | 35 | 284 | 128 | 25 | 45 | 51 | 122 |
| test | 740 | 34 | 319 | 141 | 40 | 52 | 69 | 119 |
| human_secondary | 251 | 99 | 95 | 25 | 23 | 44 | 32 | 32 |

- **Majority-class baseline: 45.8% accuracy, macro-F1 0.1004.**
- Labels are the **5-pass Qwen3-VL-235B ensemble** majority vote, kept as **soft targets** (the
  normalised vote distribution) and trained with KL divergence. 70.8% of clips are unanimous after
  the merge; the other 29.2% teach uncertainty rather than false confidence.
- Features per clip: **[10, 514]** — 512 frozen CLIP ViT-B/32 (letterboxed) + 2 motion channels
  (`motion_inst` vs 0.4 s earlier, `motion_disp` vs the previous sampled frame). Frames are the
  deterministic pass-1 interleaved grid at 2.5 fps.

### 2.1 Label-reliability filtering, measured not assumed

From the human labels, label reliability depends on the camera **and the direction**:

| camera | model says ABSENT → human agrees | model says PRESENT |
|---|---|---|
| Right_Left | 36/36 = 100% | **wrong 45%** |
| Right_Right | 8/8 = 100% | 0% FP |
| Right_Front | 18/20 = 90% | 3% FP |
| Right_Back | 7/8 = 88% | 15% FP |
| Right_Top | 11/20 = 55% | 8% FP |

Two opposite failure modes, both physically sensible: `Right_Left` hallucinates presence from
tank-glass reflections but is perfect when it says absent; `Right_Top` (IR) misses animals in dim
footage but is reliable when it does see one. So the filter is **directional**, not a blanket camera
exclusion — dropping `Right_Left` outright would have discarded the most reliable hard negatives in
the corpus.

    Right_Left PRESENT  → excluded (329 clips)
    Right_Left ABSENT   → kept, full weight
    Right_Top  ABSENT   → kept at weight 0.5
    everything else     → kept, full weight

### 2.2 Two dataset bugs the validator caught before training (R26)

`src/validate_ethogram_dataset.py` failed twice. Both would have produced a publishable-looking but
wrong result, and neither raised an error on its own.

1. **A video-level leak: 29 videos spanned two trainable splits.** Not a bug in the split function —
   it assigns one split per video and each run was internally clean. The leak came from
   **resumability**: features cache per clip and the manifest is append-only, but the split is a
   *global* greedy decision over the whole clip set, so a run over 2,978 clips and a run over 4,673
   disagreed on 29 of the 58 videos they shared. Rule extracted and enforced in code: *resume the
   expensive per-clip work, never a global decision.* Splits are now recomputed every run with an
   assertion that no video spans two splits.
2. **147 of 6,945 clips on disk (2.1%) are truncated to under 15 s** — byte-range extraction failures
   that passed a **file-size** validity check (one truncated clip is 3.6 MB at 0.49 s). 8 had reached
   the dataset, 4 carrying a behaviour label, **one in the test split**: a 0.4 s clip labelled
   `Exploration / manipulation` that no model could get right and that would have been reported as a
   genuine failure. `MIN_FRAMES=37` isolates them exactly — the corpus is bimodal, 4,665 clips at 46+
   frames and the 8 bad ones under 37, nothing between.

### 2.3 The training set was 36% short (R25)

`ensemble_235b_vote.py` had been run **once**, early, and never re-derived as the resumable ensemble
advanced. Every consumer reads the vote file, so the builder was training on a stale snapshot.

| | clips |
|---|---|
| clips the ensemble touched | 5,222 |
| clips with all 5 passes complete | 5,003 |
| clips in the vote file | 3,444 |
| **fully processed, never voted** | **1,568** |

Re-voting: 3,444 → 5,222 records (+52%); trainable **2,978 → 4,673 (+57%)**. Losses were concentrated
in `Right_Top`/`Front`/`Back` — the three real den angles. `check_vote_fresh()` now refuses to build
from a stale vote. **General defect worth stating: any derived artifact of a long resumable job (vote
files, merged indices, snapshots) is stale the moment the job advances.**

### 2.4 What the 5-pass ensemble buys over one pass

Measured on the full 5,218 clips with genuine sampling variation and ≥5 votes:

| quantity | value |
|---|---|
| unanimous on ethogram | 3,330 (63.8%) |
| split | 1,888 (36.2%) |
| **majority vote differs from a single pass-1 label** | **705 clips (13.5%)** |

---

## 3. Does the motion channel carry signal?

Checked before training anything that depends on it, since two rungs rest on it.

| class | n | median `motion_disp` | IQR |
|---|---|---|---|
| Human / enrichment interaction | 229 | 0.1105 | 0.0797–0.1476 |
| Reaching out of water | 368 | 0.0561 | 0.0330–0.1003 |
| Exploration / manipulation | 789 | 0.0475 | 0.0277–0.0871 |
| Locomotion (crawl/swim) | 353 | 0.0353 | 0.0216–0.0642 |
| No octopus | 2,117 | 0.0198 | 0.0154–0.0379 |
| Resting / stationary | 810 | 0.0198 | 0.0165–0.0261 |

**Locomotion vs Resting on `motion_disp` alone: AUC 0.714.** The channel carries real signal.

Note the last two rows: `No octopus` and `Resting` have **identical** motion medians (0.0198). This
is the one place the motion feature cannot help, and it predicts the dominant confusion in §5.

---

## 4. The rung ladder (R27)

`src/train_ethogram.py`, 3 seeds per rung, mean ± std. Each rung isolates one hypothesis so a gain is
attributable rather than asserted. `CW_POWER=1.0` (full inverse-frequency class weights) here.

| rung | model | params | val macro-F1 | TEST macro-F1 |
|---|---|---|---|---|
| — | majority class | — | — | 0.1004 |
| 0 | mean-pooled CLIP → linear | 3,078 | 0.4842 ± 0.0050 | 0.4000 ± 0.0143 |
| 1 | mean\|std\|max pooling → MLP | 398,086 | 0.5531 ± 0.0090 | 0.4739 ± 0.0373 |
| 2 | + motion summary stats | 399,634 | 0.5521 ± 0.0115 | 0.4802 ± 0.0300 |
| 3 | full [10,514] → BiGRU + attention | 266,891 | 0.5439 ± 0.0050 | 0.5129 ± 0.0110 |

**Finding 1 — the previous classifier's failure mode is gone.** The earlier behaviour classifier
scored 45% val accuracy with per-class F1 ≈ 0 outside the majority classes: collapse. Here every
class scores 0.29–0.75 at **every** rung, including rung 0, which deliberately reproduces the old
pooled-CLIP-and-linear recipe. So the fix was not architecture — it was the 6-class merge, the
reliability filtering, soft targets, and 4,665 clips. Rung 0's 0.400 against a 0.100 macro-F1 baseline
also shows the old "45%" was never the failure it read as: it was compared against an *accuracy*
baseline.

**Finding 2 — only one jump on the ladder is real: rung 0 → 1** (+0.074 test, +0.069 val). Richer
pooling and a non-linear head help. Rungs 1 → 2 → 3 differ by 0.006 and 0.033 against stds of
0.030–0.037, and **val and test disagree on the ordering** (val ranks rung 3 last, test ranks it
first). That disagreement is what video-level noise looks like at 35 val / 34 test videos. **Rung 3
is not claimed as a win.**

**Finding 3 — the motion channels bought nothing measurable** (rung 1 → 2: +0.006, inside noise),
despite AUC 0.714 in isolation. The signal is real but largely redundant with what CLIP appearance
already encodes. Recorded plainly because the channels were added specifically to answer the
pooled-feature critique: that critique was right about pooling and wrong about needing explicit
motion.

**Finding 4 — rung 3's variance is 3× lower** (±0.011 vs ±0.037). Not a headline result, but the
reason to prefer it if one model must ship: sequence modelling makes the outcome reproducible across
seeds.

### 4.1 Scope of "the ceiling is video diversity"

Every rung is a **head on frozen CLIP ViT-B/32**; nothing backpropagates into the backbone. The
ladder varied the head, not the representation. So the supported claim is *"the ceiling is video
diversity **given frozen CLIP features**"*, and a representation swap is required before the broader
version can be stated. That experiment is in flight (§7).

---

## 5. The class-weighting bug (R28)

The first real defect was in the **loss**, not the architecture. v1 used full inverse-frequency class
weights (`1/count`). Train counts run 1,419 (`No octopus`) to 141 (`Human`), so rare classes received
6–10× the weight and became **sinks**.

This is self-defeating: over-weighting a rare class is meant to protect macro-F1, but macro-F1 charges
for precision too, so the weighting cost the very metric it existed to defend.

| CW_POWER | val macro-F1 | TEST macro-F1 | `Reaching` precision |
|---|---|---|---|
| 1.0 (v1) | 0.5439 ± 0.005 | 0.5129 ± 0.011 | 0.35 |
| **0.5 (sqrt) — selected** | **0.5555 ± 0.010** | **0.5298 ± 0.007** | 0.43 |
| 0.25 | 0.5400 ± 0.005 | 0.5417 ± 0.005 | 0.48 |
| 0.0 (off) | 0.5381 ± 0.015 | 0.5581 ± 0.014 | 0.53 |

`Reaching` precision rises monotonically as the weighting is removed — mechanistic confirmation of
the diagnosis rather than a lucky hyperparameter.

> **Selection discipline.** `CW_POWER=0.0` gives the best TEST score (0.5581, +0.045) and is **not
> claimable**: val does not select it, and choosing a hyperparameter by test score is test-set
> selection. The reportable result is the val-selected 0.5 → **TEST 0.5298, +0.0169 over v1** (~1.5σ).

### 5.1 Per-class, at the selected configuration

Rung 3, `CW_POWER=0.5`, 3 seeds pooled, 740 test clips / 34 videos.

| class | recall | precision | F1 | test n |
|---|---|---|---|---|
| No octopus | 0.73 | 0.86 | **0.79** | 319 |
| Locomotion (crawl/swim) | 0.58 | 0.56 | 0.57 | 52 |
| Reaching out of water | 0.72 | **0.43** | 0.54 | 69 |
| Exploration / manipulation | 0.54 | 0.51 | 0.53 | 141 |
| Resting / stationary | 0.44 | 0.44 | 0.44 | 119 |
| Human / enrichment interaction | 0.27 | 0.37 | 0.31 | 40 |

- **`Reaching out of water` is still a sink** even after tempering: recall 0.72 at precision 0.43. On
  F1 alone it looks like the third-best class. It must not be reported without precision.
- **`Human / enrichment interaction`** has 40 test clips on 7 videos; its F1 is not a reliable
  per-class figure and is flagged rather than quoted.

### 5.2 The dominant confusion

**`No octopus` ↔ `Resting / stationary` is 22.3% of all errors** (20.2% at `CW_POWER=1.0`), and class
weighting does not touch it — correctly, since it is not a class-balance problem.

Physically sensible: a motionless animal in a den and an empty tank differ only in appearance, and
both motion channels read ≈0 for each (identical medians, §3).

Per-camera breakdown (rung 3, 3 seeds pooled):

| camera | test n | accuracy | static-class confusions |
|---|---|---|---|
| Right_Top (IR) | 696 | 0.57 | 104/429 = 24% |
| Right_Right | 246 | 0.59 | 34/123 = 28% |
| Right_Back | 552 | 0.49 | 28/270 = 10% |
| Right_Front | 516 | 0.52 | 23/282 = 8% |
| Right_Left | 210 | **0.98** | 0/210 = 0% |

IR carries **55% of all static-class confusions**. `Right_Left` at 0.98 is a near-free camera — by the
directional filter it is ~100% `No octopus` — which inflates aggregate figures and is a reason to
report per-camera numbers.

---

## 6. Human label rounds — 452 labels, and one negative result

Three blind-by-construction rounds via `ui/label_ethogram.py`. The model's answer, its vote margin and
the sampling stratum are never sent to the browser; the verdict sits behind an explicit request, and
each label records whether it was consulted (`assisted`).

### 6.1 Round 3: was IR presence the problem? No.

Motivated by §5.2: on the 20 IR clips reviewed in rounds 1–2 where the ensemble said `No octopus`, the
human agreed only **11/20 = 55%**, and all 9 disagreements were exactly `Resting`. On colour cameras
the same check was 72/75 = 96%. That suggested the dominant confusion was **label noise**, not model
error, and that fixing IR labels was the largest available gain.

`src/sample_ir_presence.py` drew 200 clips in three mixed, shuffled groups — a target arm, a
reverse-direction arm, and a **colour control with a pre-registered expectation of ~96%**. The mix is
methodological: a round made only of clips the ensemble called empty would drag the labeller's prior
toward "empty" and measure that drift instead of the labels. 151 were completed.

| group | n | human agrees with ensemble | 95% CI (Wilson) |
|---|---|---|---|
| IR, ensemble said empty | 90 | **87.8%** | [79.4%, 93.0%] |
| colour, ensemble said empty (control) | 29 | 86.2% | [69.4%, 94.5%] |
| IR, ensemble said present | 32 | 71.9% | [54.6%, 84.4%] |

**The hypothesis is refuted.** IR `No octopus` labels are ~88% correct, not 55%, and the intervals do
not overlap — so the prior estimate was not noise around the same value, it was drawn from a
different population. Checking composition confirms it: the earlier 20 clips were 45% split-vote,
against 17% in a representative draw. **The old sample was enriched for hard cases.**

**And IR is not worse than colour** (87.8% vs 86.2%, indistinguishable). The `No octopus` class
carries **~12–14% label noise across all cameras**, not an IR-specific defect. Consequences:

- Relabelling the 501 IR clips would have been wasted work.
- The `No octopus` ↔ `Resting` confusion is therefore **mostly a genuine model/signal limitation, not
  bad labels.**
- One new lead: **IR *present* calls are wrong ~28% of the time** (vs 8% in prior notes), while the
  behaviour, once presence is confirmed, matches the human **95.7%** (22/23). On IR the weakness is
  *finding* the animal, not naming what it does.

### 6.2 Two caveats that bound §6.1

- **All 151 labels came back `assisted`**, median 2.0 s per clip, so they measure *agreement* and
  87.8% is an **upper bound** on true label accuracy. Cause was a UI defect: the "always show hint"
  toggle persisted under a fixed `always_v2` localStorage key, so a v2 setting silently carried into
  v3 and the round was never blind. Fixed — the key is per-round and defaults off.
- **The control missed its pre-registered value** (86.2% vs ~96%, CI top 94.5%). By the rule fixed
  before looking, the *absolute levels* are held loosely; the *relative* finding (IR ≈ colour) is
  measured under one protocol and is the more trustworthy part.

### 6.3 Status of the 452 labels as an evaluation set

Usable as a **clearly-caveated secondary** only, for two independent reasons: every label is
`assisted` (agreement, not accuracy), and their videos overlap training — 97 of 99 `human_secondary`
videos also appear in train/val/test, so clip-level overlap is present even though no clip is shared.
The 34 **test** videos were reserved for a genuinely blind round precisely so that a primary
human-accuracy figure can exist; only 29 of 740 test clips have been touched. **That round has not
been run, and it is the only route to an accuracy (rather than agreement) claim.**

---

## 6.4 Teacher vs student vs human — the label ceiling, and who is actually the weak link

`src/eval_ethogram_human.py`. The ladder scores the student against its own teacher, which measures
**teacher reproduction, not correctness**. With 455 human labels the better question is answerable,
provided the populations are kept apart. **They are never pooled**: 102 human-labelled clips are in
the *train* split, so scoring there measures memorisation and is computed for diagnosis only.

Rung 3, `CW_POWER=0.5`, 3 seeds with probabilities averaged before the argmax.

| population | clips / videos | teacher vs human | student vs human | student vs teacher |
|---|---|---|---|---|
| **human_secondary** (best powered) | 251 / 99 | **72.5%** agree, macro-F1 **0.6569** | 60.6% agree, macro-F1 0.4917 | macro-F1 0.5635 |
| test (clean, tiny) | 30 / 19 | 86.7% agree, macro-F1 0.5139 | 60.0% agree, macro-F1 0.2636 | macro-F1 0.2721 |
| val | 25 / 18 | 92.0% agree, macro-F1 0.4298 | 72.0% agree, macro-F1 0.3150 | macro-F1 0.3214 |

Agreement CIs (Wilson): human_secondary teacher [66.7%, 77.7%] vs student [54.4%, 66.4%] — **the
intervals do not overlap.** The test and val macro-F1 figures are computed on 25–30 clips across 6
classes and are too noisy to quote; only the 251-clip population is well powered.

**Where the student and the teacher disagree, who does the human back?** This is the only direct
evidence on whether the student's "errors" are errors at all.

| population | disagreements | human backs STUDENT | human backs TEACHER | neither |
|---|---|---|---|---|
| human_secondary | 86 | 19 | **49** | 18 |
| test | 11 | 1 | 9 | 1 |
| val | 6 | 0 | 5 | 1 |

**This reverses the earlier conclusion.** §4 reported the ceiling as video diversity, later narrowed
to "given frozen CLIP features". The evidence now says neither labels nor data quantity is the binding
constraint:

1. **The teacher is far better than the student on the same clips** — 0.657 vs 0.492 macro-F1,
   non-overlapping agreement intervals. The student has not saturated its teacher; roughly 0.165
   macro-F1 of headroom exists against labels we already have.
2. **On disagreements the human sides with the teacher 2.6:1.** The student's deviations are mostly
   the student being wrong, not the teacher being noisy.
3. So the information **is present in the footage** — a 235B VLM recovers it. What cannot recover it
   is a frozen-CLIP embedding plus a 267K-parameter head. **The binding constraint is the student's
   representation and capacity**, which is precisely what §7 tests and what makes that experiment the
   right next one rather than more data.

### 6.4.1 The confound, stated because it favours this conclusion

The hint the labelling UI shows is **the ensemble's own verdict** — i.e. the teacher's. Every one of
the 455 labels was recorded `assisted`, so **the human was anchored toward the teacher**, and
`teacher vs human` is inflated by an unknown amount. `student vs human` is *not* anchored (student
predictions were never shown), so the comparison is biased in the teacher's favour and the true gap
is smaller than 12 points.

The qualitative conclusion likely survives — the gap is large and the disagreement split is 2.6:1 —
but the magnitude should not be quoted as if clean. The blind round on the reserved test videos
(§6.3) is the fix, and this result raises its value: it is now the difference between "the student is
0.165 behind its teacher" and "we do not know how far behind it is".

Per §4.1, the ladder never varied the backbone, so the data-ceiling conclusion is conditional.
`src/extract_backbone_feats.py` re-extracts features over the **same clips, the same frames** (the
manifest's recorded `frames_used` indices), with the **same two motion channels** appended, so the
backbone is the only free variable and rungs 1–3 stay architecturally identical.

| backbone | kind | params | tests |
|---|---|---|---|
| CLIP ViT-B/32 | image–text | 88M | baseline (§4, §5) |
| DINOv2-base | image, self-supervised | 87M | is CLIP's *appearance* encoding the limit? |
| VideoMAE-base | **video-native** | 86M | is *time* the limit? |

The video arm is the more decisive of the two: CLIP has no notion of motion at all, and the
hand-computed motion channels that were supposed to compensate bought +0.006.

### 7.1 Result — the representation WAS the constraint

All at `CW_POWER=0.5`, 3 seeds, identical splits/seeds/rung definitions. The refactor that made the
trainer backbone-agnostic was regression-tested first: the CLIP path reproduces §5's **0.5298
exactly**, so any movement below is real rather than an artifact of the rewrite.

| backbone | rung 1 | rung 2 | rung 3 |
|---|---|---|---|
| CLIP ViT-B/32 (512d) | 0.5096 ± 0.030 | 0.5368 ± 0.015 | 0.5298 ± 0.007 |
| DINOv2-base (768d) | **0.6006** ± 0.004 | 0.5772 ± 0.014 | 0.5781 ± 0.001 |
| VideoMAE-base (768d) | 0.5883 ± 0.005 | **0.6057** ± 0.011 | 0.6016 ± 0.022 |

Val-selected: CLIP rung 3 → 0.5298 · DINOv2 rung 2 → **0.5772 (+0.047)** · VideoMAE rung 1 →
**0.5883 (+0.059)**.

**Robust to the selection rule**, which the §4 rung ladder was not: the *worst* rung of either new
backbone beats the *best* rung of CLIP (DINOv2 0.577–0.601, VideoMAE 0.588–0.606 vs CLIP
0.510–0.537 — no overlap). So the conclusion does not depend on how the winner is picked.

**But the win is NOT clearly about time.** VideoMAE leads on the val-selected figure, yet DINOv2 — an
*image* model — is within noise of it, and VideoMAE's best rung is 1, which pools over time and
discards order. The supported claim is "a better self-supervised representation", not "a video model
sees motion". That is weaker than the hypothesis this experiment set out to test.

### 7.2 Combining backbones — ensemble beats fusion

| approach | params | val | TEST macro-F1 |
|---|---|---|---|
| best single (DINOv2 rung 2) | 598 K | 0.5784 | 0.5772 |
| fusion: concat all 3 → one MLP (width 6,150) | 1.59 M | 0.5784 | 0.5960 ± 0.028 |
| ensemble, val-selected rungs, soft vote | 3 × ~600 K | 0.5969 | 0.6177 |
| **ensemble, all-MLP (rung 2), hard vote** | 3 × 598 K | **0.6039** | **0.6172** |
| ensemble, all-MLP, soft vote | 3 × 598 K | 0.6025 | 0.6183 |

Val-selected winner: the all-MLP ensemble, **+0.0255 on val against a seed std of 0.0081** — clears
noise, unlike the head sweep (§7.4).

**Fusion lost**, as predicted: a 6,150-wide input and 1.59 M params against 133 training videos landed
at *exactly* the best single backbone's val score, i.e. the extra two representations bought nothing
once they had to share one head, and it carries the worst test variance. Homogeneous all-MLP members
beat each backbone's individually-best rung (0.6039 vs 0.5969 val) and are simpler to deploy. The
members agree on only **60% of test clips** — that disagreement is why voting helps.

### 7.3 Accuracy vs macro-F1, since the two are easily confused

The ensemble is **71.4% accuracy** (528/740 clips) at **macro-F1 0.6183**. Majority baseline: 43.1%
accuracy / 0.1004 macro-F1. Accuracy is the more intuitive number and is reported alongside, but
macro-F1 stays the headline: a model predicting `No octopus` on everything scores 43% accuracy and
0.10 macro-F1, which is exactly the collapse macro-F1 exists to catch.

| class | recall | precision | F1 | test n |
|---|---|---|---|---|
| No octopus | 0.85 | 0.92 | **0.88** | 319 |
| Exploration / manipulation | 0.66 | 0.67 | 0.67 | 141 |
| Locomotion (crawl/swim) | 0.67 | 0.62 | 0.65 | 52 |
| Resting / stationary | 0.55 | 0.61 | 0.58 | 119 |
| Reaching out of water | 0.70 | **0.47** | 0.56 | 69 |
| Human / enrichment interaction | 0.38 | 0.37 | 0.37 | 40 |

`Resting` improved from 0.44 (§5.1) — the `No octopus` ↔ `Resting` confusion that was 22% of all
errors is genuinely better with stronger representations. `Reaching` remains a sink (precision 0.47).

### 7.4 Four things that did NOT help, measured

Recorded because each sounds obviously worth trying, and without the numbers they get retried.

1. **Head capacity/architecture ≈ 0.** hidden {256,512,1024} × dropout {0.3,0.5}: VideoMAE val gain
   +0.0017 vs seed std 0.0040; on DINOv2 the frozen 256/0.4 head *is* the val-best. Test spread across
   the grid is 0.033 while val resolves 0.002 — val cannot distinguish these configs, so any "winner"
   is luck. **More capacity actively hurts**: 1024 hidden is the worst config on both backbones.
2. **Upsampling ≈ loss weighting.** `BALANCE` ∈ {none, weight, upsample, both}: upsample val 0.5775 vs
   weight 0.5753, margin +0.0023 against seed std 0.0026.
3. **Less balancing keeps winning.** `BALANCE=none` gives the best test macro-F1 (0.6103), best
   accuracy (0.7036) *and* the best F1 on the weakest class (`Human` 0.44 vs 0.37 weighted) —
   balancing hurt the class it was meant to protect. Consistent with the `CW_POWER` sweep
   (1.0 → 0.5 → 0 all improved test). **NOT adopted**: val ranks `none` *worst* (0.5648), so taking it
   would be test-set selection. Val and test disagree systematically on this axis across three
   separate experiments — a limitation to report, not a number to pick from.
4. **Feature-space augmentation is negative.** mixup {0.2,0.4}, Gaussian noise {0.05,0.1}, and the
   combination all lose, monotonically with strength, and val/test **agree** for once (baseline
   0.5753/0.6057; mixup 0.4 → 0.5667/0.5896; noise 0.1 → 0.5450/0.5481). The model is not
   overfitting-limited, so a regulariser only corrupts frozen features. mixup does lift the weakest
   class slightly (0.37 → 0.39 as alpha rises) at everyone else's expense — the same
   recall-for-precision trade class weighting made.

**The pattern across §7: everything that adds real information helps; everything that reshuffles
existing information does not.** representation +0.087 › class-weight tempering +0.017 › head ≈ 0 ›
upsampling ≈ 0 › augmentation negative. That ordering is what motivates the mask-feature experiment
(§7.5) over any further tuning.

### 7.5 In flight — segmentation geometry as ethogram features

`src/extract_mask_feats.py` → `src/eval_mask_features.py`. The only untried lever that adds *real*
information: CLIP/DINOv2/VideoMAE are appearance encoders that never explicitly localise the animal,
so nothing in the stack knows where the octopus is, how big it is, or whether it moved. Ten channels
per frame (area, centroid x/y, bbox w/h, elongation, solidity, masked-motion, centroid displacement,
validity) from `octo_seg_thin768_lraspp.pt`.

It targets measured weaknesses, not generic ones: `No octopus` ↔ `Resting` is 22% of all errors and
their whole-frame motion medians are **identical** (0.0198 vs 0.0198), so the existing motion channels
provably cannot separate them, while mask area separates present from absent by **10–25×** in the
smoke test (0.001–0.004 on reflection clips vs 0.034–0.073 on real animals). It also uses the accurate
part of a mediocre model: SEG-TEST mask IoU is 0.6415 but **area error is ~1%**.

Two constraints carried into the evaluation rather than left to prose: the segmenter trained on **11
of the 34 ethogram test videos**, so gains are reported split by `seg_seen_video` with the unseen
subset as the honest number; and IR (35% of the corpus) is **zeroed with `valid=0`** because the
colour-trained segmenter over-segments bright tools there, which also gives a free negative control —
a "gain" appearing on IR would prove the effect is not the masks.

**One trap logged, because it would have inverted the result.** transformers 5.12 expects VideoMAE's
attention biases as `attention.{query,key,value}.bias`, while the checkpoint stores `q_bias`/`v_bias`
(no `k_bias` — zero by design). 36 of 196 tensors therefore came out **freshly initialised**,
announced only through a generic "MISSING … consider training on your downstream task" warning. It
does not crash and the features look plausible — a partly uninitialised encoder would have produced a
**fake negative for the video backbone** and the wrong conclusion about whether time helps. Patched
(24 tensors), after verifying the attention *weights* load bitwise-identical so only biases were
affected.

---

## 8. Open items

1. **DONE (§7):** representation swap → **macro-F1 0.5298 → 0.6172, accuracy 71.4%**. V-JEPA-2 (ViT-L,
   326M, 1024d) is extracting as a fourth ensemble member; mask features (§7.5) are the live
   experiment.
2. **Blind human round on the 34 reserved test videos** (§6.3) — the only unanchored human figure.
   §6.4 raises its value: the teacher-vs-student gap is currently measured against labels the
   annotator produced while looking at the teacher's answer.
3. **Video diversity** — 1,160 harvested clips over 593 source videos, unlabelled, would take the
   corpus from 206 to ~700 videos (~$4 at 5 passes). **Deprioritised by §6.4**: more teacher-labelled
   data does not address a student that is already 0.165 macro-F1 behind the teacher it has.
4. **`Human / enrichment interaction`** is starved: 141 train clips on 24 videos, F1 0.31.
5. **`Reaching out of water`** remains a sink at precision 0.43 even after tempering; cause not yet
   diagnosed beyond class weighting.

---

## 9. Provenance

| artifact | path |
|---|---|
| dataset builder | `src/build_ethogram_dataset.py` |
| dataset validator | `src/validate_ethogram_dataset.py` |
| trainer / ladder | `src/train_ethogram.py` → `data/ethogram_ladder_v1.json` |
| backbone features | `src/extract_backbone_feats.py` → `src/dataset_etho/v1/feats_<backbone>/` |
| backbone ladders | `data/ethogram_ladder_{dinov2,videomae,clip_cw05}.json` |
| fusion vs ensemble | `src/train_ethogram_fusion.py` → `data/ethogram_fusion.json` |
| head sweep | `src/sweep_ethogram_head.py` → `data/ethogram_head_sweep_*.json` |
| human eval (3-way) | `src/eval_ethogram_human.py` → `data/ethogram_human_eval.json` |
| mask features | `src/extract_mask_feats.py` → `src/dataset_etho/v1/feats_mask/` |
| mask comparison | `src/eval_mask_features.py` → `data/ethogram_mask_features.json` |
| frozen dataset | `src/dataset_etho/v1/{manifest.jsonl, features.npz, snapshot.json}` |
| ensemble passes | `data/ensemble_235b/pass{1..5}.jsonl` |
| ensemble vote | `src/ensemble_235b_vote.py` → `data/ensemble_235b_voted.json` |
| labelling UI | `ui/label_ethogram.py` (port 8021) |
| label rounds | `data/human_behaviour_labels{,_v2,_v3}.json` + `data/human_eval_sample_v*.json` |
| IR round sampler | `src/sample_ir_presence.py` |
| backbone extractor | `src/extract_backbone_feats.py` |
| running ledger | `PAPER_NOTES.md` R25–R28 |
