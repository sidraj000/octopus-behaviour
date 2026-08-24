"""benchmarks.py — the project's formal benchmark suite (single runner, paper-ready output).

Three frozen benchmarks, one command, one JSON + one LaTeX table:

  SEG-TEST   segmentation quality on human-verified masks, video-level holdout (leak-free)
             -> IoU mean/median, body-area error, presence AUC (uses the human negatives)
  SKEL-50    per-frame skeleton quality on 50 frozen frames (image + human mask + model mask)
             -> tip PRECISION / RECALL / F1 vs the human mask's protrusions, head error, arm count
  TRACK-10   temporal tracking on 10 frozen clips
             -> teleport rate (all + confident-only), occluded fraction, coverage, fragmentation

WHY TIP-F1 (the metric fix): "arms per frame" is not a score — a *lower* count can mean *better*
output (our anti-mess gates removed ~1.3 duplicate/tangle arms per frame and the count dropped from
4.80 to 3.48 while quality rose). Precision alone is equally gameable (predict one obvious arm).
F1 against the human mask's protrusions punishes BOTH over-detection (tangle) and under-detection
(missed arms), so it is the number the paper should report; arm count stays as a descriptive stat.

Usage:
  venv/bin/python3 src/benchmarks.py --suite all            # everything (slow: ~25 min)
  venv/bin/python3 src/benchmarks.py --suite seg,skel       # pick suites
  venv/bin/python3 src/benchmarks.py --suite skel --refine  # skeleton on SAM2-refined masks
  venv/bin/python3 src/benchmarks.py --suite skel --tag anti_mess_gates   # name the run
Results append to data/benchmarks.json (keyed by tag); --latex writes the paper table.
"""
import argparse, json, math, sys, time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "skeleton"))

RESULTS = REPO / "data" / "benchmarks.json"
# SEG-TEST's 19 empty-tank negatives come from 2 source videos (18 from 2026-02-21/183003).
# Frozen property of the benchmark, documented in BENCHMARKS.md; printed with the row so the
# number is never read as a many-recording estimate.
SEG_TEST_NEG_VIDEOS = 2
DS = REPO / "data" / "dataset_seg_human"
BENCH50 = REPO / "data" / "skel_bench50" / "frames.json"
HEAD_GT = REPO / "data" / "skel_bench50" / "head_gt.json"

# Frozen holdout: these source videos are excluded from ALL training (leakage guard).
HOLDOUT_VIDEOS = {"2026-02-21/150002", "2026-02-21/183003", "2026-02-22/153002",
                  "2026-02-22/190003", "2026-02-23/170003"}
MATCH_FRAC = 0.05          # tip match radius as a fraction of the image diagonal
MAX_GT_TIPS = 8            # safety cap; with GT_MIN_LEN_FRAC below it never binds (max observed = 8)
# GT protrusion detector params. VALIDATED on the 50 human masks: the library default
# (min_len_frac=0.06) counted contour bumps as arms -> mean 9.7, max 14 protrusions, and the 8-cap
# bound in 80% of frames, so recall was measured against a padded, partly-fictional 8 tips.
# Requiring peaks to be >=10% of the contour apart (two closer peaks are the SAME arm) gives
# mean 5.7 / median 6 / max 8 — biologically sane for a 2-D silhouette.
GT_MIN_PROMINENCE = 1.8
GT_MIN_LEN_FRAC = 0.10


def _source_video(clip):
    p = Path(clip); return f"{p.parent.parent.name}/{p.parent.name}"


def _manifest(source):
    rows = [json.loads(l) for l in open(DS / "manifest.jsonl") if l.strip()]
    return [r for r in rows if r.get("source") == source and r.get("image")]


# ── SEG-TEST ────────────────────────────────────────────────────────────────────────
def run_seg(ckpt, fusion="none"):
    from segment_octopus import OctoSegmenter, _largest_blob
    from temporal_fusion import fused_prob, prob_to_mask
    S = OctoSegmenter(str(ckpt))
    n_align_fail = 0
    pos = [r for r in _manifest("human") if _source_video(r["clip"]) in HOLDOUT_VIDEOS]
    neg = [r for r in _manifest("negative") if _source_video(r["clip"]) in HOLDOUT_VIDEOS]
    ious, aerr, pos_area, neg_area = [], [], [], []
    for r in pos:
        img = cv2.imread(str(DS / r["image"])); gt = cv2.imread(str(DS / r["mask"]), 0) > 127
        if fusion == "none":
            pred, _ = S.segment(img)
        else:
            pr, info = fused_prob(S, r["clip"], r.get("seed_frame"), img, mode=fusion)
            n_align_fail += (not info["ok"])
            pred = prob_to_mask(pr, img.shape, _largest_blob)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_NEAREST) > 0
        inter = (pred & gt).sum(); union = (pred | gt).sum()
        ious.append(inter / union if union else 1.0)
        aerr.append(abs(pred.mean() - gt.mean()))
        pos_area.append(float(pred.mean()))
    for r in neg:
        img = cv2.imread(str(DS / r["image"]))
        if fusion == "none":
            pred, _ = S.segment(img)
        else:
            pr, info = fused_prob(S, r["clip"], r.get("seed_frame"), img, mode=fusion)
            n_align_fail += (not info["ok"])
            pred = prob_to_mask(pr, img.shape, _largest_blob)
        neg_area.append(float(pred.mean()))
    auc = None
    if pos_area and neg_area:                       # rank-AUC of mask area separating present/empty
        lab = np.r_[np.ones(len(pos_area)), np.zeros(len(neg_area))]
        sc = np.r_[pos_area, neg_area]
        order = np.argsort(sc); ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc) + 1)
        n1 = lab.sum(); n0 = len(lab) - n1
        auc = float((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return {"n_frames": len(pos), "n_negatives": len(neg), "fusion": fusion,
            "align_fail": n_align_fail,
            "n_videos": len({_source_video(r["clip"]) for r in pos}),
            "iou_mean": round(float(np.mean(ious)), 4), "iou_median": round(float(np.median(ious)), 4),
            "area_err_pct": round(float(np.mean(aerr)) * 100, 3),
            "presence_auc": None if auc is None else round(auc, 4)}


# ── SKEL-50 ─────────────────────────────────────────────────────────────────────────
def _match(pred_tips, gt_tips, r):
    """Greedy 1-1 nearest matching within radius r -> (n_matched)."""
    used, n = set(), 0
    for p in pred_tips:
        best, bj = None, -1
        for j, g in enumerate(gt_tips):
            if j in used:
                continue
            d = math.hypot(p[0] - g[0], p[1] - g[1])
            if d <= r and (best is None or d < best):
                best, bj = d, j
        if bj >= 0:
            used.add(bj); n += 1
    return n


def run_skel(ckpt, refine=False, fusion="none"):
    from segment_octopus import OctoSegmenter, _largest_blob
    from skel_bench50 import skeleton_paths, AFTER
    from skel_phaseA_loss import finger_tips
    from skel_head_fix import full_graph
    S = OctoSegmenter(str(ckpt))
    frames = json.load(open(BENCH50))
    # bench50 rows carry no seed_frame — recover it by joining image -> label manifest
    seed_by_image = {r["image"]: r.get("seed_frame")
                     for r in (_manifest("human") + _manifest("negative"))}
    n_align_fail = 0
    head_gt = json.load(open(HEAD_GT)) if HEAD_GT.exists() else {}
    P, R, F, counts, head_err = [], [], [], [], []
    for f in frames:
        img = cv2.imread(str(DS / f["image"]))
        gtm = (cv2.imread(str(DS / f["mask"]), 0) > 127).astype(np.uint8) * 255
        gt_tips = finger_tips(gtm, min_prominence=GT_MIN_PROMINENCE,
                              min_len_frac=GT_MIN_LEN_FRAC)[:MAX_GT_TIPS]
        if fusion == "none":
            mm, _ = S.segment(img)
        else:
            from temporal_fusion import fused_prob, prob_to_mask
            pr, info = fused_prob(S, f["clip"], seed_by_image.get(f["image"]), img, mode=fusion)
            n_align_fail += (not info["ok"])
            mm = prob_to_mask(pr, img.shape, _largest_blob)
        if refine and mm.any():
            from mask_refine import sam2_refine
            mm = sam2_refine(img, mm, largest_blob=_largest_blob)
        m255 = (mm.astype(np.uint8)) * 255
        paths = skeleton_paths(m255, AFTER)
        pred_tips = [tuple(p[-1]) for p in paths]
        counts.append(len(pred_tips))
        r = MATCH_FRAC * math.hypot(*gtm.shape)
        nm = _match(pred_tips, gt_tips, r)
        p = nm / len(pred_tips) if pred_tips else (1.0 if not gt_tips else 0.0)
        rc = nm / len(gt_tips) if gt_tips else 1.0
        P.append(p); R.append(rc); F.append(0.0 if p + rc == 0 else 2 * p * rc / (p + rc))
        # head error in body radii (only where a human click exists)
        key = str(DS / f["image"])
        if key in head_gt:
            nodes, _ = full_graph(m255)
            hd = next((n for n in (nodes or []) if n.get("is_head")), None)
            if hd is not None:
                dt = cv2.distanceTransform(m255, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                rr = max(float(dt.max()), 1.0)
                gx, gy = head_gt[key]
                head_err.append(math.hypot(hd["x"] - gx, hd["y"] - gy) / rr)
    out = {"n_frames": len(frames), "fusion": fusion, "align_fail": n_align_fail,
           "tip_precision": round(float(np.mean(P)), 4),
           "tip_recall": round(float(np.mean(R)), 4),
           "tip_f1": round(float(np.mean(F)), 4),
           "arms_per_frame": round(float(np.mean(counts)), 2),
           "refined": bool(refine)}
    if head_err:
        out["head_err_radii_mean"] = round(float(np.mean(head_err)), 3)
        out["head_hit_at_0.75R"] = round(float(np.mean(np.array(head_err) <= 0.75)), 3)
        out["n_head_gt"] = len(head_err)
    return out


# ── TRACK-10 ────────────────────────────────────────────────────────────────────────
def run_track(ckpt, fps=3.0):
    from segment_octopus import OctoSegmenter
    from skel_eval_tracking import EVAL_CLIPS, run_tracker
    from track_metrics import sequence_metrics, summarize
    S = OctoSegmenter(str(ckpt))
    per = {}
    for rel in EVAL_CLIPS:
        graphs, crops = run_tracker(REPO / "src" / rel, S, fps)
        if graphs:
            per[rel] = sequence_metrics(graphs, crops, len(crops))
    s = summarize(per) if per else {}
    s["n_clips"] = len(per)
    return s


LATEX = r"""% auto-generated by src/benchmarks.py — do not hand-edit
% regenerate without re-running the models: src/benchmarks.py --latex-from <tag>
\begin{table}[t]
\caption{Frozen benchmark suite, deployed configuration (%CKPT%). All splits are by
SOURCE VIDEO; the segmentation test videos are excluded from every training source.
Mask IoU comes from the same model and test set as the presence row, but that row's
19 negatives come from only two recordings; it is retained for continuity and is
superseded by the multi-recording human-verified negative sets reported in the text.
Tip-F1 scores arm recovery against the human mask's protrusions, so both missed and
spurious arms are penalised; arm count is descriptive only.}
\label{tab:benchmarks}
\centering
\small
\begin{tabular}{@{}lc@{}}
\toprule
Metric & Value \\
%ROWS%
\bottomrule
\end{tabular}
\end{table}
"""


def to_latex(res):
    rows = []
    if "seg" in res:
        s = res["seg"]
        rows += [r"\midrule",
                 f"\\multicolumn{{2}}{{@{{}}l}}{{\\textbf{{SEG-TEST}} --- {s['n_frames']} human-mask frames, "
                 f"{s['n_videos']} held-out videos}} \\\\",
                 f"\\quad mask IoU (mean / median) & {s['iou_mean']:.3f} / {s['iou_median']:.3f} \\\\",
                 f"\\quad body-area error & {s['area_err_pct']:.2f}\\% \\\\"]
        if s.get("presence_auc") is not None:
            # SEG-TEST's empty-tank negatives come from only 2 source videos (18/19 from one) —
            # see BENCHMARKS.md. Always print the recording count so the row cannot be read as
            # a population estimate; the powered replacement is data/presence_human_verified.json.
            rows.append(f"\\quad presence AUC (vs {s['n_negatives']} empty, "
                        f"{SEG_TEST_NEG_VIDEOS} recordings) & {s['presence_auc']:.3f} \\\\")
    if "skel" in res:
        k = res["skel"]
        rows += [r"\midrule",
                 f"\\multicolumn{{2}}{{@{{}}l}}{{\\textbf{{SKEL-50}} --- {k['n_frames']} frozen frames}} \\\\",
                 f"\\quad arm-tip F1 & {k['tip_f1']:.3f} \\\\",
                 f"\\quad tip precision / recall & {k['tip_precision']:.3f} / {k['tip_recall']:.3f} \\\\",
                 f"\\quad arms per frame (descriptive) & {k['arms_per_frame']:.2f} \\\\"]
        if "head_err_radii_mean" in k:
            rows.append(f"\\quad head error (body radii) & {k['head_err_radii_mean']:.2f} \\\\")
    if "track" in res:
        t = res["track"]
        rows += [r"\midrule",
                 f"\\multicolumn{{2}}{{@{{}}l}}{{\\textbf{{TRACK-10}} --- {t.get('n_clips',0)} frozen clips}} \\\\",
                 f"\\quad teleport rate & {t.get('teleport_rate',0):.3f} \\\\",
                 f"\\quad occluded fraction & {t.get('occluded_frac',0):.3f} \\\\",
                 f"\\quad coverage & {t.get('coverage',0):.3f} \\\\"]
    ckpt = (res.get("_meta") or {}).get("ckpt", "")
    return LATEX.replace("%ROWS%", "\n".join(rows)).replace("%CKPT%", ckpt.replace("_", r"\_"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all", help="all | comma list of seg,skel,track")
    ap.add_argument("--ckpt", default=str(REPO / "weights" / "seg" / "octo_seg_thin768_lraspp.pt"))
    ap.add_argument("--refine", action="store_true", help="SAM2-refine masks (skel suite)")
    ap.add_argument("--fusion", default="none", choices=["none", "ema", "flow"],
                    help="test-time temporal fusion of the probability map (default none = shipped)")
    ap.add_argument("--tag", default="current")
    ap.add_argument("--latex", action="store_true", help="also write the paper table")
    ap.add_argument("--latex-from", default=None, metavar="TAG",
                    help="regenerate the paper table from an existing tag in data/benchmarks.json "
                         "(no models are run — for table-formatting changes only)")
    args = ap.parse_args()
    if args.latex_from:
        res = json.load(open(RESULTS))[args.latex_from]
        p = REPO / "OCEANS_2026" / "assets" / "benchmarks_table.tex"
        p.write_text(to_latex(res)); print(f"-> {p}  [from tag '{args.latex_from}']")
        return
    suites = ["seg", "skel", "track"] if args.suite == "all" else \
        [s.strip() for s in args.suite.split(",") if s.strip()]

    res, t0 = {}, time.perf_counter()
    if "seg" in suites:
        print("SEG-TEST …", flush=True); res["seg"] = run_seg(args.ckpt, args.fusion); print(" ", res["seg"], flush=True)
    if "skel" in suites:
        print("SKEL-50 …", flush=True); res["skel"] = run_skel(args.ckpt, args.refine, args.fusion); print(" ", res["skel"], flush=True)
    if "track" in suites:
        print("TRACK-10 …", flush=True); res["track"] = run_track(args.ckpt); print(" ", res["track"], flush=True)
    res["_meta"] = {"tag": args.tag, "ckpt": Path(args.ckpt).name, "refine": args.refine,
                    "fusion": args.fusion,
                    "elapsed_s": round(time.perf_counter() - t0, 1)}

    all_res = json.load(open(RESULTS)) if RESULTS.exists() else {}
    all_res[args.tag] = res
    json.dump(all_res, open(RESULTS, "w"), indent=1)
    print(f"\n-> {RESULTS} [{args.tag}]  ({res['_meta']['elapsed_s']}s)")
    if args.latex:
        p = REPO / "OCEANS_2026" / "assets" / "benchmarks_table.tex"
        p.write_text(to_latex(res)); print(f"-> {p}")


if __name__ == "__main__":
    main()
