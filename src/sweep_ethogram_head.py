"""sweep_ethogram_head.py — tune the MLP head on the winning backbone.

The head was held at hidden=256 / dropout=0.4 / depth=1 for the whole backbone comparison so the
representation was the only free variable. Right for that experiment, but it means the head was sized
for CLIP's 512-dim features and never re-tuned: rung 1 on a 768-dim backbone hands it 2,304 inputs to
squeeze into 256 units. The backbone question is settled (R30), so the head is now worth tuning.

SELECTION DISCIPLINE, and why the grid is deliberately small. Val is 35 SOURCE VIDEOS. Sweeping a
large grid and reporting the val-best test score overfits val and inflates the result -- the same trap
as picking CW_POWER on test (R28), one level removed. So:

  * the grid is 6 configs, not 60
  * selection is on val, and the val-selected config's test score is reported ONCE
  * the FULL grid is printed, so the spread is visible and a lucky pick is obvious
  * a config only counts as better if it clears the val-best by more than that config's own seed std;
    otherwise the honest verdict is "no better than the frozen head", which is a real outcome

Baseline for reference: the frozen head (256/0.4/1) at the val-selected rung.

Usage: venv/bin/python3 src/sweep_ethogram_head.py --backbone videomae --rung 1
"""
import argparse, itertools, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent

import train_ethogram as T

GRID = {"hidden": [256, 512, 1024], "dropout": [0.3, 0.5], "depth": [1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="videomae")
    ap.add_argument("--rung", type=int, default=1)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    man, X, classes, D = T.load(a.version, a.backbone)
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    # the frozen head, so the sweep is measured against what it is replacing
    frozen = {"hidden": 256, "dropout": 0.4, "depth": 1}
    if frozen not in combos:
        combos.insert(0, frozen)
    print(f"backbone={a.backbone} rung={a.rung}  configs={len(combos)}  seeds={a.seeds}  "
          f"CW_POWER={T.CW_POWER}\n")
    print(f"{'hidden':>7}{'drop':>6}{'depth':>6}{'params':>10}{'val macro-F1':>20}{'TEST macro-F1':>20}")

    rows = []
    for c in combos:
        T.MLP_HIDDEN, T.MLP_DROPOUT, T.MLP_DEPTH = c["hidden"], c["dropout"], c["depth"]
        runs = [T.run_one(a.rung, man, X, classes, s, D=D) for s in range(a.seeds)]
        v = np.array([r["val_f1"] for r in runs]); t = np.array([r["test_f1"] for r in runs])
        row = {**c, "params": runs[0]["n_params"], "val": float(v.mean()), "val_std": float(v.std()),
               "test": float(t.mean()), "test_std": float(t.std()),
               "is_frozen_head": c == frozen}
        rows.append(row)
        tag = "  <- frozen head" if row["is_frozen_head"] else ""
        print(f"{c['hidden']:>7}{c['dropout']:>6}{c['depth']:>6}{row['params']:>10}"
              f"{v.mean():>13.4f} ±{v.std():.4f}{t.mean():>13.4f} ±{t.std():.4f}{tag}", flush=True)

    base = next(r for r in rows if r["is_frozen_head"])
    best = max(rows, key=lambda r: r["val"])
    print(f"\nfrozen head      : val {base['val']:.4f}  TEST {base['test']:.4f} ±{base['test_std']:.4f}")
    print(f"val-selected     : hidden={best['hidden']} dropout={best['dropout']} depth={best['depth']}"
          f"  val {best['val']:.4f}  TEST {best['test']:.4f} ±{best['test_std']:.4f}")
    gain_val = best["val"] - base["val"]
    print(f"\nval gain over the frozen head: {gain_val:+.4f}  (that config's seed std {best['val_std']:.4f})")
    if best["is_frozen_head"]:
        print("  -> the frozen head IS the val-best; tuning bought nothing")
    elif gain_val <= best["val_std"]:
        print("  -> gain is inside one seed std: NOT a real improvement. Report the frozen head and "
              "say the head is not the bottleneck.")
    else:
        print(f"  -> exceeds its own seed std, so a real val gain. TEST moves "
              f"{base['test']:.4f} -> {best['test']:.4f} ({best['test']-base['test']:+.4f}).")
    spread = max(r["test"] for r in rows) - min(r["test"] for r in rows)
    print(f"\ntest spread across the whole grid: {spread:.4f} "
          f"({min(r['test'] for r in rows):.4f}-{max(r['test'] for r in rows):.4f})")
    print("  A spread much larger than the val gain means val cannot resolve these configs and the\n"
          "  'winner' is largely luck -- treat the head as tuned-out, not improved.")

    out = a.out or str(REPO / "data" / f"ethogram_head_sweep_{a.backbone}_rung{a.rung}.json")
    Path(out).write_text(json.dumps({"backbone": a.backbone, "rung": a.rung, "cw_power": T.CW_POWER,
                                     "grid": GRID, "rows": rows}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
