"""merge_harvest.py — fold a pulled harvest run (cbox / Modal / local) into the canonical
harvest ledger + clip index.

The harvester writes ONE ledger keyed by video_url plus a clips index. Runs happen on
different boxes (Modal, cbox, local), so this merges those per-run outputs into a single
canonical pair under data/:

    data/harvest_ledger_all.json      — every video ever probed/scanned, keyed by video_url
    data/harvest_clips_index.json     — every harvested clip, in octopus_clips_verified entry format

DELIBERATELY does NOT touch data/octopus_clips_verified.json, and there is no flag to make it.
That index backs the paper's frozen benchmark sets; harvested clips are a different sampling
regime (visibility-only gate, 2 clips/video, no motion requirement) and pooling them silently
would change reported denominators. If you ever do want them pooled, that is a deliberate,
separate step that must be followed by re-running src/benchmarks.py.

Merge rules
  - new video_url            -> inserted
  - existing video_url       -> kept, UNLESS the incoming record is strictly more informative
                                (a real status replacing "failed", or more clips found)
  - clip entries             -> deduped on (video_url, start_sec, end_sec)

Usage
  venv/bin/python3 src/merge_harvest.py --incoming data/harvest_cbox
  venv/bin/python3 src/merge_harvest.py --incoming harvest_dl/harvest --clips-root data/harvest_clips
  venv/bin/python3 src/merge_harvest.py --incoming data/harvest_cbox --dry-run
"""
import argparse, collections, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CANON_LEDGER = REPO / "data" / "harvest_ledger_all.json"
CANON_INDEX = REPO / "data" / "harvest_clips_index.json"

# a status we would rather keep than overwrite, best first
STATUS_RANK = {"clips": 3, "scanned_empty": 2, "probed_empty": 1, "failed": 0}


def load_json(p, default):
    p = Path(p)
    if not p.exists():
        return default
    with open(p) as f:
        return json.load(f)


def better(new, old):
    """True if `new` is strictly more informative than `old` for the same video_url."""
    rn, ro = STATUS_RANK.get(new.get("status"), -1), STATUS_RANK.get(old.get("status"), -1)
    if rn != ro:
        return rn > ro
    # same status -> prefer the one that found more clips, then the one that scanned more
    if new.get("n_clips", 0) != old.get("n_clips", 0):
        return new.get("n_clips", 0) > old.get("n_clips", 0)
    return (new.get("scanned_sec") or 0) > (old.get("scanned_sec") or 0)


def clip_key(c):
    return (c.get("video_url"), c.get("start_sec"), c.get("end_sec"))


def summarize(ledger, label):
    by_status = collections.Counter(v.get("status") for v in ledger.values())
    by_coll = collections.Counter(v.get("collection") for v in ledger.values())
    dates = {v.get("date") for v in ledger.values() if v.get("date")}
    clips = sum(v.get("n_clips", 0) or 0 for v in ledger.values())
    dur = sum(v.get("duration") or 0 for v in ledger.values())
    scanned = sum(v.get("scanned_sec") or 0 for v in ledger.values())
    print(f"\n=== {label} ===")
    print(f"  videos            : {len(ledger)}")
    print(f"  distinct dates    : {len(dates)}")
    print(f"  clips harvested   : {clips}")
    print(f"  footage seen      : {dur/3600:.1f} h total, {scanned/3600:.1f} h actually decoded")
    print(f"  by status         : {dict(by_status)}")
    for c, n in by_coll.most_common():
        print(f"      {c}: {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incoming", required=True,
                    help="pulled harvest dir (contains harvest_ledger.json + harvest_clips_index.json)")
    ap.add_argument("--ledger", default=str(CANON_LEDGER))
    ap.add_argument("--index", default=str(CANON_INDEX))
    ap.add_argument("--clips-root", default=None,
                    help="if given, verify each clip_path exists under this root and report gaps")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    inc_dir = Path(args.incoming)
    inc_ledger = load_json(inc_dir / "harvest_ledger.json", None)
    if inc_ledger is None:
        sys.exit(f"no harvest_ledger.json in {inc_dir}")

    canon = load_json(args.ledger, {})
    summarize(canon, f"canonical BEFORE ({Path(args.ledger).name})")
    summarize(inc_ledger, f"incoming ({inc_dir})")

    added = updated = skipped = 0
    for url, rec in inc_ledger.items():
        if url not in canon:
            canon[url] = rec; added += 1
        elif better(rec, canon[url]):
            canon[url] = rec; updated += 1
        else:
            skipped += 1

    # rebuild the clip index from the merged ledger (ledger is the source of truth), then
    # union in any pre-existing index entries whose video is no longer in the ledger.
    clips, seen = [], set()
    for rec in canon.values():
        for c in rec.get("clips", []) or []:
            k = clip_key(c)
            if k in seen:
                continue
            seen.add(k); clips.append(c)
    old_index = load_json(args.index, {})
    orphans = 0
    for c in (old_index.get("clips", []) if isinstance(old_index, dict) else old_index) or []:
        k = clip_key(c)
        if k not in seen:
            seen.add(k); clips.append(c); orphans += 1

    print(f"\n=== merge ===")
    print(f"  ledger: +{added} new, {updated} updated (more informative), {skipped} unchanged")
    print(f"  index : {len(clips)} clips" + (f" (incl. {orphans} kept from the old index)" if orphans else ""))

    if args.clips_root:
        root = Path(args.clips_root)
        missing = [c["clip_path"] for c in clips
                   if c.get("clip_path") and not (root / c["clip_path"]).exists()]
        print(f"  on disk: {len(clips)-len(missing)}/{len(clips)} present under {root}")
        if missing:
            print(f"  MISSING {len(missing)} (first 5): " + ", ".join(missing[:5]))

    summarize(canon, "canonical AFTER")

    if args.dry_run:
        print("\n[dry-run] nothing written.")
        return

    Path(args.ledger).parent.mkdir(parents=True, exist_ok=True)
    with open(args.ledger, "w") as f:
        json.dump(canon, f, indent=1)
    with open(args.index, "w") as f:
        json.dump({"description": "harvested clips (probe-first stream scan, visibility gate); "
                                  "NOT pooled into octopus_clips_verified.json",
                   "n_clips": len(clips), "clips": clips}, f, indent=1)
    print(f"\nwrote {args.ledger}\nwrote {args.index}")


if __name__ == "__main__":
    main()
