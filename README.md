# Octopus behaviour pipeline

Code and labels for **"From Footage to Ethogram: A Deployable Pipeline for Continuous
Behavioural Monitoring of a Captive Octopus"** (OCEANS 2026 Monterey).

The paper itself is not distributed here - see **Paper** below.

A cascade of small local models turns raw aquarium video into a behavioural time series. A
235B-parameter vision–language model is used only as an **offline teacher**; nothing large runs at
inference. On 892 h of footage from one adult *Octopus vulgaris*, cheap gates discard 60% of videos
without a full decode and only ~60 h is ever decoded — because 42.7% of the clips that survive
still contain no animal.

    src/      the pipeline: extraction, teacher labelling, the three students, benchmarks
    ui/       labelling and review apps (FastAPI)
    data/     the ethogram and caption labels  ← in this repo
    docs/     PAPER_NOTES.md (the R1-R35 ledger), BENCHMARKS.md, and the
              ethogram and segmentation logs

## Where things live

**Models and the large datasets are on Google Drive:**
**https://drive.google.com/drive/folders/1e-8iJRsXDho5sXnxGj3_ylKNPvnxwI1P?usp=sharing**

| artifact | location |
|---|---|
| **ethogram labels** (5,222 clips, full vote distributions) | **this repo** — `data/ethogram/` |
| **captions** (~25,000 across 5 passes) | **this repo** — `data/captions/` |
| **human labels** (456) | **this repo** — `data/human_labels/` |
| trained models (5, ~1.8 GB) | [**Drive**](https://drive.google.com/drive/folders/1e-8iJRsXDho5sXnxGj3_ylKNPvnxwI1P?usp=sharing) — `models/`, each with a card in `MODELS.md` |
| cached backbone features (5 × `.npz`, 317 MB) | [**Drive**](https://drive.google.com/drive/folders/1e-8iJRsXDho5sXnxGj3_ylKNPvnxwI1P?usp=sharing) — `datasets/ethogram_dataset_v1/` |
| presence-classification frames (9,596 labelled frames, 8.1 GB) | [**Drive**](https://drive.google.com/drive/folders/1e-8iJRsXDho5sXnxGj3_ylKNPvnxwI1P?usp=sharing) |
| the 20 s video clips (~46 GB) | not distributed — reconstructible, see below |

The features are on Drive rather than here because they are 52–83 MB each and GitHub blocks files
over 100 MB. They matter: **without them the headline model cannot be retrained**, since it needs
DINOv2, VideoMAE and two mask-cropped views, not just CLIP.

## Read these three things before using the data

**1. The behaviour labels are TEACHER labels.** All 5,222 come from a 5-pass Qwen3-VL-235B ensemble,
not from a human. The full vote distribution is kept rather than only the majority, because the
margin predicts human agreement (0.73 unanimous / 0.86 at four-of-five / 0.43 at three-or-fewer).
Treat them as a strong automatic annotation, not ground truth.

**2. The human labels measure AGREEMENT, not accuracy.** All 456 were collected with the model's
suggestion visible on screen. Every record carries an `assisted` flag and every one is `true`. A
blind round is specified but was not run before release, so the ethogram numbers should be read as
*teacher reproduction*.

**3. One animal, one tank.** Every measurement is a single adult *O. vulgaris*. Cross-animal
generalisation is untested and is the paper's principal limitation.

## Results

| model | measured on | result |
|---|---|---|
| ethogram classifier (6 classes incl. absence) | 740 clips / 34 held-out videos | **macro-F1 0.665, accuracy 75.4%** (majority baseline 0.100 / 43.1%) |
| segmentation, 3.2 M params, no prompt | 122 human masks / 5 held-out videos | **IoU 0.642**, mean area error ~1% |
| caption student, 4-bit on-device | held-out val | emb-sim 0.702 → **0.834**, ROUGE-L 0.269 → **0.455** |
| presence probe vs zero-shot CLIP | 120 frames at uniform random timestamps | **AUC 0.745 vs 0.450** |
| our segmenter vs its zero-shot teacher | same 122 human masks | **0.642 vs 0.374** |

The largest single gain in behaviour classification came from changing the frozen **representation**
(+0.087), not the classifier — and specifically from how much of the frame the animal occupies.
Head capacity, class rebalancing, upsampling and feature-space augmentation all measured ≈0 or
negative. `docs/PAPER_NOTES.md` has the numbers that kill each one.

## Protocol rules

Enforced in code, not documented as intentions, because the pipeline broke each at least once:

1. **Splits are by source video, never by clip.** A held-out clip from a training video is not held
   out. `src/validate_ethogram_dataset.py` asserts it and exits non-zero.
2. **Frozen evaluation sets are never regenerated to suit a result.** Figure sources are pinned
   alongside the paper rather than read live; pointing a figure at a scratch directory silently
   changed a published figure once.
3. **Negatives of different kinds are never pooled** — reflections, empty tank and infrared fail
   differently and averaging them hides which.
4. **Holdout videos are excluded from *every* training source**, not merely the final stage.
5. **Derived artifacts of resumable jobs go stale.** A vote file left un-rederived silently cost 36%
   of the training set; `check_vote_fresh()` now refuses to build from one.

## Reproducing

```bash
python3 -m venv venv && ./venv/bin/pip install -r src/requirements.txt
cp src/.env.example .env      # OPENROUTER_API_KEY for the teacher; OCTOPUS_USER/PASS for footage

# the classifier needs the cached features -- download them from Drive first:
#   Drive/datasets/ethogram_dataset_v1/features_*.npz  ->  src/dataset_etho/v1/
./venv/bin/python3 src/validate_ethogram_dataset.py --version v1   # checks before training
./venv/bin/python3 src/train_ethogram.py --version v1              # the rung ladder
./venv/bin/python3 src/train_ethogram_fusion.py                    # the 5-member ensemble
./venv/bin/python3 src/benchmarks.py --tag mytag                   # frozen suites
```

Apple Silicon: CLIP and the students run on MPS; **GroundingDINO must run on CPU** (deformable
attention is unstable on MPS). OpenAI CLIP needs `setuptools<81` for `pkg_resources`.

## Recovering the video clips

Clips are not distributed, but every label carries its provenance. `data/ethogram/manifest.jsonl`
keys on `clip_path`, and the clip index records the source `video_url` plus `start_sec`/`end_sec`
for each. Re-extraction with `ffmpeg -ss/-to -c:v copy` reproduces the same footage — **verified,
but not bit-identical**: container framing shifts by a few KB depending on keyframe boundaries and
ffmpeg version. Labels and cached features are unaffected; re-deriving features from pixels would
be equivalent rather than exactly comparable.

## Experimental record

`docs/PAPER_NOTES.md` is the chronological ledger (R1–R35) with provenance for every
measurement — **including the failed experiments and the conclusions that were retracted.** Two
worth knowing: "teacher-label quality is the segmentation ceiling" was wrong (its supporting
evidence was train leakage, and clean held-out IoU is flat across teacher quality), and a +0.043
gain from mask-geometry features turned out to live entirely on videos the segmenter had trained on.

## Models

Five models, all on [Drive](https://drive.google.com/drive/folders/1e-8iJRsXDho5sXnxGj3_ylKNPvnxwI1P?usp=sharing)
under `models/`, each with a card in the Drive `MODELS.md` stating what it trained on, what it scores on which
frozen set, and **what it must not be used for**:

| model | size | task |
|---|---|---|
| `octo-presence-clip-probe-v1.pt` | 0.6 MB | is an octopus visible in this frame |
| `octo-mask-lraspp-3.2M-v1.pt` | 12.5 MB | segmentation, no prompt at inference |
| `octo-ethogram-ensemble-v1.pt` | 32 MB | 6-class behaviour, absence included |
| `octo-caption-qwen3vl2b-lora-v1/` | 78 MB | caption adapter (needs the base model) |
| `octo-caption-qwen3vl2b-4bit-v1/` | 1.7 GB | same, quantised, ~3 s/clip on a laptop |

Two caveats carried on the cards: the mask model is **colour-only** and must not be used on infrared
(35% of the corpus), and the ethogram model **reproduces the VLM teacher, not ground truth**.

Three further checkpoints are deliberately withheld with reasons recorded in the Drive `models/manifest.json` — most
notably a retrained presence gate that is much better on diverse footage (FPR 0.485 → 0.243) but
forgot its original domain (recall 0.985 → 0.903).

## Licence

- **Code** (`src/`, `ui/`) — Apache-2.0, see `LICENSE`.
- **Data and labels** (`data/`) — CC-BY-4.0, see `LICENSE-DATA`. Please cite the paper.
(The paper is not in this repository; see **Paper** above.)

## Paper

The LaTeX source and PDF are deliberately **not** published here. Under the IEEE copyright
agreement the Version of Record may not be posted online at all, and the accepted version is
permitted only on an author's personal or employer site, an institutional or funder repository,
arXiv or TechRxiv - and IEEE's conference policy states that accepted papers "must be removed from
any other third-party servers." GitHub is a third-party server, so hosting it here would have to be
undone at acceptance. It is simpler not to start.

Please cite:

    S. Raj, K. M. Patel, H. Kumar, R. Thakur, S. Pachnanda, H. P. Singh,
    H. Burgsteiner, and W. Slany, "From Footage to Ethogram: A Deployable
    Pipeline for Continuous Behavioural Monitoring of a Captive Octopus," in
    Proc. OCEANS 2026 Monterey, 2026.

`docs/PAPER_NOTES.md` is the full experimental record behind it, including the failed experiments
and the retracted conclusions, and every number the paper reports is reproducible from `src/`.

## Ethics

All footage is observational, from cameras already installed for husbandry, with no intervention or
manipulation of the animal.
