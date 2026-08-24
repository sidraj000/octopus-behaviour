"""save_ethogram_model.py — persist the paper's headline ethogram model as a usable checkpoint.

WHY THIS EXISTS. `train_ethogram.py` and `train_ethogram_fusion.py` both train in memory, report
metrics, and discard the weights. So the paper's headline result -- the five-member soft-vote
ensemble at 0.665 macro-F1 / 75.4% accuracy -- existed only as a number in a JSON file. Nobody
could load it, deploy it, or check it without retraining from cached features, and a paper that
claims to release models has to actually have one.

WHAT IT SAVES. One checkpoint holding all five members, plus everything needed to run them:
  * five `mlp_256_64`-style heads (rung 2: mean|std|max pooling over the backbone's time axis)
  * the backbone each head expects, and that backbone's feature dim
  * the class list and the vote rule (soft = mean of member softmaxes)
  * the metrics measured for THIS checkpoint, so the file cannot drift from its reported numbers

SEEDS. The paper reports a mean over three seeds. A checkpoint has to be one concrete model, so
this saves all three seeds per member and averages their softmax at inference -- which is what the
reported number actually measured, rather than picking the luckiest seed and quoting the mean.

Usage: venv/bin/python3 src/save_ethogram_model.py
       -> weights/ethogram_ensemble_v1.pt  (+ prints the metrics it was saved with)
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent

import train_ethogram as T

BACKBONES = ["clip", "dinov2", "videomae", "dinov2crop", "videomaecrop"]
RUNG = 2
SEEDS = 3
OUT = REPO / "weights" / "ethogram_ensemble_v1.pt"


def main():
    data, classes = {}, None
    for b in BACKBONES:
        man, X, cls, D = T.load("v1", b)
        data[b] = {"X": X, "D": D, "man": {r["clip"]: r for r in man}}
        classes = cls
    common = set.intersection(*(set(d["man"]) for d in data.values()))
    man = [r for r in next(iter(data.values()))["man"].values() if r["clip"] in common]
    te = [r for r in man if r["split"] == "test"]
    Hte = np.array([r["label_idx"] for r in te])
    n_cls = len(classes)
    print(f"\nclips {len(man)}  test {len(te)} from {len({r['video'] for r in te})} videos")

    members, probs = [], []
    for b in BACKBONES:
        rows = [data[b]["man"][r["clip"]] for r in man]
        acc, states = None, []
        for s in range(SEEDS):
            out = T.run_one(RUNG, rows, data[b]["X"], classes, s, extra_rows=te, D=data[b]["D"])
            p = np.asarray(out["extra_probs"], np.float32)
            acc = p if acc is None else acc + p
            states.append({k: v.cpu() for k, v in out["state_dict"].items()}
                          if "state_dict" in out else None)
        probs.append(acc / SEEDS)
        members.append({"backbone": b, "feat_dim": data[b]["D"], "rung": RUNG,
                        "seeds": SEEDS, "state_dicts": states})
        print(f"  {b:<14} D={data[b]['D']:<5} {SEEDS} seeds")

    P = np.array(probs)                       # [members, N, C]
    pred = P.mean(0).argmax(-1)               # soft vote, the val-selected rule
    f1, per = T.macro_f1(pred, Hte, n_cls)
    acc = float((pred == Hte).mean())
    print(f"\nsoft-vote ensemble: macro-F1 {f1:.4f}   accuracy {acc:.4f}   ({int((pred==Hte).sum())}/{len(Hte)})")

    if any(m["state_dicts"][0] is None for m in members):
        print("\nWARNING: run_one does not return state_dict, so weights could not be captured.\n"
              "         Add it to the returned dict in train_ethogram.run_one, then re-run.\n"
              "         Refusing to write a checkpoint that contains no weights.")
        sys.exit(1)

    torch.save({
        "kind": "ethogram_soft_vote_ensemble",
        "classes": classes,
        "members": members,
        "vote": "soft: mean of member softmaxes, then argmax",
        "seed_policy": f"{SEEDS} seeds per member, probabilities averaged before argmax",
        "cw_power": T.CW_POWER,
        "feature_layout": "per member: [T, D+2] = backbone dim D then 2 motion channels; "
                          "rung 2 pools mean|std|max over T for both blocks",
        "dataset": "src/dataset_etho/v1 (4,665 clips / 204 videos, splits by source video)",
        "metrics_at_save": {"test_macro_f1": round(f1, 4), "test_accuracy": round(acc, 4),
                            "per_class": {classes[c]: per[c] for c in per},
                            "n_test_clips": len(te),
                            "n_test_videos": len({r["video"] for r in te})},
        "caveats": [
            "trained against 5-pass VLM labels, so this reproduces the teacher, not ground truth",
            "all human labels used for validation were collected with the model's answer visible "
            "(agreement, not accuracy)",
            "requires cached backbone features; see src/extract_backbone_feats.py",
        ],
    }, OUT)
    print(f"\nwrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
