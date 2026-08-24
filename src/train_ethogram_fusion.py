"""train_ethogram_fusion.py — combine CLIP + DINOv2 + VideoMAE.

R30 measured each frozen backbone separately: CLIP 0.5298, DINOv2 0.5772, VideoMAE 0.5883 (all
val-selected, CW_POWER=0.5). The three are trained on different objectives -- image-text alignment,
self-supervised image, masked video -- so their errors need not coincide, and combining them is the
obvious next question. Two ways to do it, which fail differently:

  FUSION    concatenate the pooled per-backbone vectors and train ONE head.
            Can learn cross-backbone interactions, but triples the input width against 133 training
            videos, so it is the option that overfits.

  ENSEMBLE  train each backbone separately and average the softmax.
            Cannot learn interactions, but adds no parameters and reduces variance. On small data
            this frequently matches or beats fusion, so it is the honest baseline that fusion has to
            beat to justify itself.

Both are reported. If fusion does not clearly beat ensemble, ensemble wins on simplicity.

THREE THINGS THAT WOULD SILENTLY CORRUPT THIS, handled explicitly:

  1. SCALE MISMATCH. CLIP features are L2-normalised; DINOv2 CLS tokens and VideoMAE pooled tokens are
     not, and their magnitudes differ by an order of magnitude. Concatenating raw would let the
     largest-magnitude block dominate the LayerNorm and effectively discard the others -- which would
     look like "fusion does not help" when the real cause is preprocessing. Each block is standardised
     separately.
  2. LEAKAGE VIA THE SCALER. The mean/std are fit on TRAIN ROWS ONLY and applied to val/test. Fitting
     on everything leaks test statistics into training, which on this dataset would be a repeat of the
     video-leak class of bug (R26).
  3. DIFFERENT SEQUENCE LENGTHS. CLIP and DINOv2 emit T=10 (one per sampled frame), VideoMAE T=8 (one
     per temporal token). Fusion therefore pools each backbone over ITS OWN time axis before
     concatenating, which sidesteps resampling entirely. Since rungs 1-2 (pooled) already beat rung 3
     (sequence) on both new backbones, nothing is given up by fusing at the pooled level.

Motion channels are appended ONCE, not three times -- they are byte-identical across the three
feature sets (copied from the CLIP build), so including them per block would triple-count them.

Usage: venv/bin/python3 src/train_ethogram_fusion.py
"""
import argparse, collections, json, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent

import train_ethogram as T

BACKBONES = ["clip", "dinov2", "videomae", "dinov2crop", "videomaecrop"]
# the val-selected rung per backbone from R30, used for the ENSEMBLE arm
BEST_RUNG = {"clip": 3, "dinov2": 2, "videomae": 1, "dinov2crop": 2, "videomaecrop": 2}


def pooled(X, rows, D):
    """mean | std | max over the backbone's own time axis -> [N, 3D]. Motion excluded here."""
    seq = np.stack([X[r["clip"]] for r in rows])[:, :, :D]
    return np.concatenate([seq.mean(1), seq.std(1), seq.max(1)], 1).astype(np.float32)


def motion_block(X, rows, D):
    """The 2 motion channels, pooled the same way -> [N, 6]. Appended once for the whole fusion."""
    mot = np.stack([X[r["clip"]] for r in rows])[:, :, D:]
    return np.concatenate([mot.mean(1), mot.std(1), mot.max(1)], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--out", default=str(REPO / "data" / "ethogram_fusion.json"))
    a = ap.parse_args()

    # ---- load all three, and keep ONLY clips present in all of them ----
    data, classes = {}, None
    for b in BACKBONES:
        man, X, cls, D = T.load(a.version, b)
        data[b] = {"X": X, "D": D, "man": {r["clip"]: r for r in man}}
        classes = cls
    common = set.intersection(*(set(d["man"]) for d in data.values()))
    man = [r for r in next(iter(data.values()))["man"].values() if r["clip"] in common]
    print(f"\nclips common to all {len(BACKBONES)} backbones: {len(man)}")
    tr = [r for r in man if r["split"] == "train"]
    va = [r for r in man if r["split"] == "val"]
    te = [r for r in man if r["split"] == "test"]
    print(f"  train {len(tr)} / val {len(va)} / test {len(te)} clips "
          f"({len({r['video'] for r in tr})}/{len({r['video'] for r in va})}/"
          f"{len({r['video'] for r in te})} videos)")
    n_cls = len(classes)
    results = {"version": a.version, "cw_power": T.CW_POWER, "n_clips": len(man), "arms": {}}

    # ================= ARM 1: FUSION =================
    def build(rows, stats=None):
        blocks, fit = [], {}
        for b in BACKBONES:
            v = pooled(data[b]["X"], rows, data[b]["D"])
            if stats is None:
                mu, sd = v.mean(0, keepdims=True), v.std(0, keepdims=True) + 1e-6
                fit[b] = (mu, sd)
            else:
                mu, sd = stats[b]
            blocks.append((v - mu) / sd)                  # per-block standardisation
        blocks.append(motion_block(data[BACKBONES[0]]["X"], rows, data[BACKBONES[0]]["D"]))
        return np.concatenate(blocks, 1).astype(np.float32), (fit if stats is None else stats)

    Xtr, stats = build(tr)                                 # scaler fit on TRAIN ONLY
    Xva, _ = build(va, stats)
    Xte, _ = build(te, stats)
    print(f"\nfused input width: {Xtr.shape[1]}  "
          f"(= 3 pooling stats x [{' + '.join(str(data[b]['D']) for b in BACKBONES)}] + 6 motion)")

    Ytr = np.stack([np.asarray(r["soft"], np.float32) for r in tr])
    Htr = np.array([r["label_idx"] for r in tr]); Wtr = np.array([r.get("weight", 1.0) for r in tr], np.float32)
    Hva = np.array([r["label_idx"] for r in va]); Hte = np.array([r["label_idx"] for r in te])
    cnt = collections.Counter(Htr.tolist())
    cw = np.array([(1.0 / max(1, cnt.get(c, 0))) ** T.CW_POWER for c in range(n_cls)], np.float32)
    cw = cw / cw.sum() * n_cls
    sw = Wtr * cw[Htr]

    vs, ts = [], []
    for seed in range(a.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        model = T.MLP(Xtr.shape[1], n_cls, hidden=a.hidden, p=a.dropout, depth=1)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
        t = torch.from_numpy
        xtr, ytr, wtr = t(Xtr), t(Ytr), t(sw)
        best, best_state, bad = -1, None, 0
        for ep in range(T.EPOCHS):
            model.train()
            perm = torch.randperm(len(xtr))
            for i in range(0, len(xtr), 64):
                bi = perm[i:i + 64]
                logp = torch.log_softmax(model(xtr[bi]), -1)
                loss = ((ytr[bi] * (torch.log(ytr[bi].clamp_min(1e-8)) - logp)).sum(-1) * wtr[bi]).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                f1, _ = T.macro_f1(model(t(Xva)).argmax(-1).numpy(), Hva, n_cls)
            if f1 > best:
                best, bad = f1, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= T.PATIENCE:
                    break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            f1t, _ = T.macro_f1(model(t(Xte)).argmax(-1).numpy(), Hte, n_cls)
        vs.append(best); ts.append(f1t)
    fusion = {"val": float(np.mean(vs)), "val_std": float(np.std(vs)),
              "test": float(np.mean(ts)), "test_std": float(np.std(ts)),
              "n_params": sum(p.numel() for p in model.parameters()), "width": int(Xtr.shape[1])}
    results["arms"]["fusion"] = fusion
    print(f"\nFUSION   ({fusion['n_params']:,} params)  val {fusion['val']:.4f} ±{fusion['val_std']:.4f}"
          f"   TEST {fusion['test']:.4f} ±{fusion['test_std']:.4f}")

    # ================= ARM 2: ENSEMBLE OF SEPARATELY-TRAINED HEADS =================
    # Two rung policies, because they answer different questions:
    #   "selected" -- each backbone at its own val-selected rung (clip=3/BiGRU, dinov2=2, videomae=1).
    #                 The strongest members, but heterogeneous.
    #   "mlp"      -- ALL three at rung 2, so every member is the same MLP and the only difference is
    #                 the representation. Cleaner attribution and the simpler thing to deploy.
    # And two vote rules: SOFT (average the probabilities) and HARD (majority over argmaxes, ties
    # broken by summed probability). Hard voting discards confidence; with 4 members a 2-2 tie is now
    # possible and is resolved by summed probability, which makes hard voting partly soft in exactly
    # the ambiguous cases -- another reason to prefer soft unless hard clearly wins on val.
    def ensemble(policy):
        per, pte, pva = {}, [], []
        for b in BACKBONES:
            rung = BEST_RUNG[b] if policy == "selected" else 2
            Xb, Db = data[b]["X"], data[b]["D"]
            acc, pv, pt = None, [], []
            for seed in range(a.seeds):
                out = T.run_one(rung, [data[b]["man"][r["clip"]] for r in man], Xb, classes, seed,
                                extra_rows=te + va, D=Db)
                p = np.asarray(out["extra_probs"], np.float32)
                acc = p if acc is None else acc + p
                pv.append(out["val_f1"]); pt.append(out["test_f1"])
            avg = acc / a.seeds
            pte.append(avg[:len(te)]); pva.append(avg[len(te):])
            per[b] = {"rung": rung, "val": round(float(np.mean(pv)), 4),
                      "test": round(float(np.mean(pt)), 4)}
            print(f"  [{policy}] {b:<10} rung {rung}  val {np.mean(pv):.4f}  TEST {np.mean(pt):.4f}")
        pte, pva = np.array(pte), np.array(pva)                      # [3, N, C]
        soft_te, soft_va = pte.mean(0).argmax(-1), pva.mean(0).argmax(-1)

        def hard(p):
            votes = p.argmax(-1)                                     # [3, N]
            out = np.zeros(votes.shape[1], int)
            for i in range(votes.shape[1]):
                c = collections.Counter(votes[:, i])
                top = max(c.values())
                cand = [k for k, v in c.items() if v == top]
                out[i] = cand[0] if len(cand) == 1 else int(max(cand, key=lambda k: p[:, i, k].sum()))
            return out, votes

        hard_te, votes_te = hard(pte)
        hard_va, _ = hard(pva)
        f1_soft_te, _ = T.macro_f1(soft_te, Hte, n_cls)
        f1_soft_va, _ = T.macro_f1(soft_va, Hva, n_cls)
        f1_hard_te, _ = T.macro_f1(hard_te, Hte, n_cls)
        f1_hard_va, _ = T.macro_f1(hard_va, Hva, n_cls)
        unan = int((votes_te[0] == votes_te[1]).sum() and 0) or int(
            sum(1 for i in range(votes_te.shape[1]) if len(set(votes_te[:, i])) == 1))
        print(f"  [{policy}] SOFT vote  val {f1_soft_va:.4f}  TEST {f1_soft_te:.4f}")
        print(f"  [{policy}] HARD vote  val {f1_hard_va:.4f}  TEST {f1_hard_te:.4f}   "
              f"(all 3 members agreed on {unan}/{votes_te.shape[1]} = {unan/votes_te.shape[1]:.0%} "
              f"of test clips)")
        return {"per_backbone": per, "soft": {"val": round(f1_soft_va, 4), "test": round(f1_soft_te, 4)},
                "hard": {"val": round(f1_hard_va, 4), "test": round(f1_hard_te, 4)},
                "unanimous_frac": round(unan / votes_te.shape[1], 4), "n_params": 0}

    print()
    ens_sel = ensemble("selected")
    print()
    ens_mlp = ensemble("mlp")
    results["arms"]["ensemble_selected"] = ens_sel
    results["arms"]["ensemble_mlp"] = ens_mlp
    best_single = max(ens_sel["per_backbone"].values(), key=lambda x: x["val"])
    results["best_single"] = best_single
    f1_va = max(ens_sel["soft"]["val"], ens_sel["hard"]["val"],
                ens_mlp["soft"]["val"], ens_mlp["hard"]["val"])
    f1_te = max([e[k]["test"] for e in (ens_sel, ens_mlp) for k in ("soft", "hard")
                 if e[k]["val"] == f1_va] or [0])

    # ================= verdict =================
    print("\n" + "=" * 72)
    print(f"best SINGLE backbone (val-selected): val {best_single['val']:.4f}  TEST {best_single['test']:.4f}")
    print(f"FUSION                             : val {fusion['val']:.4f}  TEST {fusion['test']:.4f}")
    print(f"ENSEMBLE (best of soft/hard x policy): val {f1_va:.4f}  TEST {f1_te:.4f}")
    arms = {"single": best_single["val"], "fusion": fusion["val"], "ensemble": f1_va}
    pick = max(arms, key=arms.get)
    print(f"\nval selects: {pick.upper()}")
    if pick != "single":
        margin = arms[pick] - best_single["val"]
        print(f"  val margin over the best single backbone: {margin:+.4f}  "
              f"(fusion seed std {fusion['val_std']:.4f})")
        if margin <= fusion["val_std"]:
            print("  -> inside one seed std: NOT a real gain. Combining does not beat the best single\n"
                  "     backbone, and the simpler model should ship.")
        else:
            print("  -> exceeds seed noise, so a real val gain.")
    Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
