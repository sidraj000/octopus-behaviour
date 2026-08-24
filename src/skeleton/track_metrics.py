"""track_metrics.py — quantitative tracking-quality metrics for a skeleton sequence.

Session lesson: judge tracking changes on numbers computed over a FIXED eval set, not vibes.
All metrics are GT-free proxies computed from the tracker's own output:

  teleport_rate    fraction of per-node steps that jump > k x that node's median step
                   (unsmoothed) — re-associations / identity swaps show up as teleports
  fragmentation    mean number of contiguous presence runs per persistent arm id
                   (1.0 = every arm tracked in one unbroken run)
  coverage         tracked frames / present frames
  in_mask          fraction of skeleton polyline points inside the mask
  arm_count_std    stability of the per-frame arm count

Input is the `{frame_pos: (nodes, edges)}` dict produced by multi_frame.tracked_sequence and the
aligned list of crop masks.
"""
import math
from collections import defaultdict

import numpy as np

from multi_frame import node_key


def sequence_metrics(graphs, crop_masks, n_present, k_teleport=4.0, floor_px=6.0):
    order = sorted(graphs)
    series = defaultdict(list)          # persistent node key -> [(pos, x, y, state)]
    arm_presence = defaultdict(list)    # arm id -> [pos, ...]
    arm_counts = []
    inmask_num = inmask_den = 0
    n_occ = n_nodes = 0
    for pos in order:
        nodes, edges = graphs[pos]
        arm_ids = {n["branch_id"] for n in nodes if n["branch_id"] > 0}
        arm_counts.append(len(arm_ids))
        for a in arm_ids:
            arm_presence[a].append(pos)
        for n in nodes:
            st = n.get("state", "detected")
            n_nodes += 1; n_occ += (st == "occluded")
            series[node_key(n)].append((pos, float(n["x"]), float(n["y"]), st))
        m = np.asarray(crop_masks[pos])
        m = m > 0
        h, w = m.shape
        for e in edges:
            p = np.asarray(e.get("polyline") or [])
            if len(p) == 0:
                continue
            xi = np.clip(np.rint(p[:, 0]).astype(int), 0, w - 1)
            yi = np.clip(np.rint(p[:, 1]).astype(int), 0, h - 1)
            inmask_num += int((m[yi, xi]).sum()); inmask_den += len(p)

    tele = steps = tele_c = steps_c = 0
    for key, seq in series.items():
        seq.sort()
        if len(seq) < 4:
            continue
        d, conf = [], []
        for (p0, x0, y0, s0), (p1, x1, y1, s1) in zip(seq, seq[1:]):
            d.append(math.hypot(x1 - x0, y1 - y0) / max(1, p1 - p0))
            conf.append(s0 != "occluded" and s1 != "occluded")
        thr = max(k_teleport * float(np.median(d)), floor_px)
        tele += sum(1 for v in d if v > thr); steps += len(d)
        tele_c += sum(1 for v, c in zip(d, conf) if c and v > thr)
        steps_c += sum(conf)

    pos_index = {p: i for i, p in enumerate(order)}
    frags = []
    for a, ps in arm_presence.items():
        idxs = sorted(pos_index[p] for p in ps)
        frags.append(1 + sum(1 for i0, i1 in zip(idxs, idxs[1:]) if i1 - i0 > 1))

    return {
        "coverage": round(len(order) / max(1, n_present), 3),
        "teleport_rate": round(tele / max(1, steps), 4),
        "teleport_confident": round(tele_c / max(1, steps_c), 4),
        "occluded_frac": round(n_occ / max(1, n_nodes), 4),
        "fragmentation": round(float(np.mean(frags)), 2) if frags else 0.0,
        "in_mask": round(inmask_num / max(1, inmask_den), 4),
        "arm_count_mean": round(float(np.mean(arm_counts)), 2) if arm_counts else 0.0,
        "arm_count_std": round(float(np.std(arm_counts)), 2) if arm_counts else 0.0,
        "n_tracked": len(order),
    }


def summarize(per_clip):
    """Mean of each metric over clips -> one comparable row per run."""
    keys = ["coverage", "teleport_rate", "teleport_confident", "occluded_frac",
            "fragmentation", "in_mask", "arm_count_mean", "arm_count_std"]
    return {k: round(float(np.mean([m[k] for m in per_clip.values()])), 4) for k in keys}
