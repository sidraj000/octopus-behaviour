"""skel_eval_tracking.py — fixed-set tracking evaluation harness (Tracking v2, Phase 0+).

Runs the current tracker over the FIXED 10-clip eval set below, computes track_metrics per clip,
stores the run under a --tag in data/skel_diag/tracking_evals.json, and (when a --baseline tag is
present) writes the baseline-vs-current comparison chart + per-clip rows for the port-8018 UI.
Also renders a per-clip TRAIL image (tip paths, ID-consistent colours) — identity breaks show up
as colour discontinuities.

Usage:
  venv/bin/python3 src/skel_eval_tracking.py --tag baseline
  venv/bin/python3 src/skel_eval_tracking.py --tag flow --baseline baseline
"""
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "skeleton"))
from segment_octopus import OctoSegmenter
from segment_to_skeleton import segment_masks, union_bbox, grey_crops
from seg_skeleton_pipeline import DEFAULT_CKPT
from skeleton import branch_color
import multi_frame as MF
from track_metrics import sequence_metrics, summarize

OUT = HERE.parent / "data" / "skel_diag"
EVALS = OUT / "tracking_evals.json"

# Fixed, committed eval set: all 6 behaviours, 3 colour cameras, 3 dates. Do not reorder.
EVAL_CLIPS = [
    "octopus_clips_verified/2026-02-20/165422/Right_Right_1680-1700.mp4",   # exploration
    "octopus_clips_verified/2026-02-20/172422/Right_Back_0001-0021.mp4",    # human interaction
    "octopus_clips_verified/2026-02-20/172422/Right_Front_0204-0224.mp4",   # resting
    "octopus_clips_verified/2026-02-20/165422/Right_Back_1661-1681.mp4",    # reaching out
    "octopus_clips_verified/2026-02-20/175422/Right_Right_0129-0149.mp4",   # crawling
    "octopus_clips_verified/2026-02-21/133002/Right_Front_0674-0694.mp4",   # swimming/jetting
    "octopus_clips_verified/2026-02-20/175422/Right_Right_0021-0041.mp4",   # exploration
    "octopus_clips_verified/2026-02-20/172422/Right_Front_1685-1705.mp4",   # human interaction
    "octopus_clips_verified/2026-02-22/150002/Right_Back_1493-1513.mp4",    # resting
    "octopus_clips_verified/2026-02-20/165422/Right_Back_1681-1701.mp4",    # reaching out
]


def trail_image(graphs, shape, path):
    """Tip trails over the whole sequence, ID-consistent colours + final skeleton frame."""
    h, w = shape
    canvas = np.full((h, w, 3), 18, np.uint8)
    tips = {}
    for pos in sorted(graphs):
        nodes, _ = graphs[pos]
        for n in nodes:
            if n.get("is_tip"):
                tips.setdefault(n["branch_id"], []).append((pos, n["x"], n["y"]))
    for a, seq in tips.items():
        col = tuple(int(v) for v in branch_color(a)[::-1])
        pts = np.array([[x, y] for _, x, y in seq], np.float32)
        for i in range(1, len(pts)):
            f = 0.25 + 0.75 * i / len(pts)                       # fade older segments
            c = tuple(int(v * f) for v in col)
            cv2.line(canvas, tuple(np.rint(pts[i - 1]).astype(int)),
                     tuple(np.rint(pts[i]).astype(int)), c, 2, cv2.LINE_AA)
        if len(pts):
            cv2.circle(canvas, tuple(np.rint(pts[-1]).astype(int)), 5, col, -1, cv2.LINE_AA)
    last = graphs[max(graphs)] if graphs else None
    if last:
        from seg_skeleton_pipeline import _draw_skeleton
        _draw_skeleton(canvas, last[0], last[1], 1)
    s = min(1.0, 640.0 / max(h, w))
    if s < 1.0:
        canvas = cv2.resize(canvas, (int(w * s), int(h * s)))
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])


def run_tracker(clip_path, S, fps=3.0, use_flow=True, method="chain"):
    masks, src_fps, step, smalls = segment_masks(str(clip_path), S, fps, 0.004, keep_small=720)
    pm = [(k, m) for k, m in enumerate(masks) if m is not None]
    if len(pm) < 4:
        return None, None
    bbox = union_bbox([m for _, m in pm])
    y0, y1, x0, x1 = bbox
    crops = [(m[y0:y1, x0:x1].astype(np.uint8)) * 255 for _, m in pm]
    greys = grey_crops(smalls, pm[0][1].shape, bbox, [k for k, _ in pm]) if use_flow else None
    graphs = MF.tracked_sequence(crops, 3, 8, 2, 1024, seed="best", greys=greys, method=method)
    return graphs, crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="name for this run (e.g. baseline, flow, global)")
    ap.add_argument("--baseline", default="", help="tag to compare against")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--no-flow", action="store_true", help="disable the optical-flow motion prior")
    ap.add_argument("--method", default="chain", choices=["chain", "global"],
                    help="arm-identity method: chain (greedy per-frame) or global (tracklet linking)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    S = OctoSegmenter(str(DEFAULT_CKPT))
    per_clip = {}
    t0 = time.perf_counter()
    for i, rel in enumerate(EVAL_CLIPS):
        p = HERE.parent / "src" / rel
        graphs, crops = run_tracker(p, S, args.fps, use_flow=not args.no_flow, method=args.method)
        if not graphs:
            print(f"  [{i+1}/10] {rel[-40:]}  SKIP", flush=True)
            continue
        m = sequence_metrics(graphs, crops, len(crops))
        per_clip[rel] = m
        trail_image(graphs, crops[0].shape, OUT / f"trail_{args.tag}_{i:02d}.jpg")
        print(f"  [{i+1}/10] {rel[-40:]}  tele {m['teleport_rate']*100:.1f}%  frag {m['fragmentation']}"
              f"  cov {m['coverage']}  arms {m['arm_count_mean']}", flush=True)

    evals = json.load(open(EVALS)) if EVALS.exists() else {}
    evals[args.tag] = {"per_clip": per_clip, "summary": summarize(per_clip)}
    json.dump(evals, open(EVALS, "w"), indent=1)
    s = evals[args.tag]["summary"]
    print(f"\n[{args.tag}] " + "  ".join(f"{k}={v}" for k, v in s.items()))
    print(f"({time.perf_counter()-t0:.0f}s)")

    # ---- comparison chart + 8018 rows ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    base = evals.get(args.baseline) if args.baseline else None
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), facecolor="#111")
    metric_keys = [("teleport_rate", "teleport rate", 100), ("fragmentation", "fragmentation", 1),
                   ("arm_count_mean", "mean arms", 1)]
    clips = [c for c in EVAL_CLIPS if c in per_clip]
    xs = np.arange(len(clips))
    for ax, (mk, title, scale) in zip(axs, metric_keys):
        ax.set_facecolor("#111")
        cur = [per_clip[c][mk] * scale for c in clips]
        if base:
            bl = [base["per_clip"].get(c, {}).get(mk, 0) * scale for c in clips]
            ax.bar(xs - 0.2, bl, 0.4, color="#ff7a5c", label=args.baseline)
            ax.bar(xs + 0.2, cur, 0.4, color="#4ea3ff", label=args.tag)
        else:
            ax.bar(xs, cur, 0.55, color="#4ea3ff", label=args.tag)
        ax.set_title(title, color="#eee", fontsize=11); ax.tick_params(colors="#aaa")
        ax.set_xlabel("clip", color="#999"); ax.legend(fontsize=8, facecolor="#222", labelcolor="#ddd")
    fig.suptitle(f"Tracking eval — {args.tag}" + (f" vs {args.baseline}" if base else " (baseline)"),
                 color="#eee")
    plt.tight_layout()
    plt.savefig(OUT / "chart.png", dpi=130, facecolor="#111")
    plt.savefig(HERE.parent / "results" / "segmentation" / f"tracking_eval_{args.tag}.png",
                dpi=130, facecolor="#111")

    rows = []
    for i, c in enumerate(clips):
        left = (base["per_clip"].get(c, {}).get("teleport_rate", 0) * 100) if base \
            else per_clip[c]["fragmentation"]
        rows.append({"file": f"trail_{args.tag}_{EVAL_CLIPS.index(c):02d}.jpg",
                     "left_arms": round(left, 1),
                     "right_arms": round(per_clip[c]["teleport_rate"] * 100, 1)})
    meta = {"title": f"Tracking eval — {args.tag}",
            "left": (f"{args.baseline} teleport %" if base else "fragmentation"),
            "right": f"{args.tag} teleport %"}
    json.dump({"meta": meta, "rows": rows}, open(OUT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
