"""ensemble_235b.py — 5 interleaved frame-draws per clip, so presence and ethogram can be VOTED.

WHY. R15 measured the failure mode directly: re-run the same model on a DISJOINT set of frames from
the same clip and ~1/3 of behaviour labels change (behaviour kappa 0.552; `present` worst at 0.413).
That is variance from frame choice, not from the model's weights. Voting over independent draws
attacks it, and the vote margin is a better per-clip confidence than the model's self-reported
`confidence` field.

SAMPLING — interleaved uniform, 10 frames 2 s apart, 5 phase-shifted passes.
  Frames are extracted at DENSE_FPS (2.5 -> one every 0.4 s, 50 per 20 s clip). Pass p takes
  `step = n/10` and `offset = (p-1)*step/5`, i.e. for a 20 s clip:

      pass 1 : 0.0 2.0 4.0 ... 18.0      pass 4 : 1.2 3.2 5.2 ... 19.2
      pass 2 : 0.4 2.4 4.4 ... 18.4      pass 5 : 1.6 3.6 5.6 ... 19.6
      pass 3 : 0.8 2.8 4.8 ... 18.8

  The five passes are DISJOINT and tile the 2 s gap evenly. Offsets are spread across the whole
  interval rather than being 0.2 s apart, because passes separated by 0.2 s would see nearly
  identical images and the vote would be unanimous by construction rather than by evidence.

  Why not the old top-6-by-p_visible: measured on 33 clips it covers a median 0.79 of the clip but
  leaves a median 7-FRAME (7 s) gap between adjacent frames sent, inside which an entire crawl or
  jet happens unobserved (7/33 clips saw <50% of the clip). Uniform 2 s spacing caps the gap.

NO CLIP SCORING, DELIBERATELY. Frame choice is now purely temporal, so the detector plays no part:
the VLM stops being shown only the frames Stage 1 already liked, which removes that confound. The
`PRESENT_MIN` veto goes with it — it never fired anyway (`absent=0` across 2,818 calls in the
single-pass run) — so presence is now purely the VLM's judgement.

IDENTICAL PROMPT EVERY PASS. `caption_openrouter.build_prompt()`, unchanged. Asking passes 2..N for
the label alone would change the task framing and break exchangeability with pass 1, so every pass
emits a caption too; only pass 1's is canonical (the others cost ~30 completion tokens each).

RESUMABILITY / NEW CLIPS -- what this is built around:
  * State is APPEND-ONLY JSONL, one file per pass. Resumption reads the keys already present. A kill
    mid-write can only damage the final line, which is skipped on read. Nothing rewrites a large
    json, so there is no truncation window.
  * The work-list is RECOMPUTED from disk every round, in a loop that runs until nothing is left. So
    clips that arrive while the job runs (e.g. from `refetch_clips.py`) are picked up and get all N
    passes rather than being silently left with fewer votes.
  * Per (clip, pass) attempts are capped and persisted, so a permanently failing clip cannot spin
    the loop forever. A round that makes no progress stops the loop.

NOTE: this is NOT comparable with `data/local_235b_labels.json`, which used top-6 by p_visible at
1 fps. That stays a separate reference labelling, not a sixth vote.

Usage
  venv/bin/python3 src/ensemble_235b.py --passes 5 --limit 20 --max-rounds 1   # smoke
  venv/bin/python3 src/ensemble_235b.py --passes 5 --workers 8
Then:  venv/bin/python3 src/ensemble_235b_vote.py
"""
import argparse, collections, datetime, glob, json, os, subprocess, sys, tempfile, threading, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent

import caption_openrouter as C

INDEX = REPO / "src" / "octopus_clips_verified.json"
ROOTS = [REPO / "src" / "octopus_clips_verified", REPO / "data" / "octopus_clips_verified"]
OUTDIR = REPO / "data" / "ensemble_235b"
ATTEMPTS = OUTDIR / "attempts.json"

DENSE_FPS = 2.5     # 0.4 s between candidate frames -> the minimum rate the 5 offsets need
N_DRAW = 10         # frames per call, 2 s apart on a 20 s clip
MAX_ATTEMPTS = 3

# Provider latency on the 10-image calls degraded badly under sustained load (measured: a single
# call went from ~10 s to >100 s, while FAILURES stayed flat at 15 -- i.e. no 429s, so we are
# latency-bound, not rate-capped). Two consequences:
#   * 120 s was cutting off slow-but-successful calls, throwing away an answer we had already paid
#     for AND booking it as a failure (a requests.Timeout is not one of the retried status codes).
#   * Because the workers are idle waiting on I/O rather than being throttled, CONCURRENCY is the
#     right lever here -- the opposite of the earlier reduction, which was made on the mistaken
#     belief that throttling caused the failures (it was 929 zero-byte clips).
# Watch for 429s: if failures start climbing with concurrency, we have hit a real rate cap and this
# should come back down.
C.REQUEST_TIMEOUT = 300

io_lock = threading.Lock()
att_lock = threading.Lock()
state = collections.Counter()

# Credit exhaustion is not a per-clip failure and must not be treated as one. Three times in one day
# the balance ran out and the runner kept going: every call returned HTTP 402 in under a second, each
# one counted as a clip failure, and the attempt counter climbed until cells hit MAX_ATTEMPTS and were
# skipped PERMANENTLY. The last occurrence burned 2,955 cells that way (ok=0, fail=2850, usage frozen)
# while the supervisor dutifully restarted it. So a 402 now aborts the whole run instead: nothing is
# retried, no attempts are spent, and the work resumes untouched once credits are topped up.
CREDITS_EXHAUSTED = threading.Event()


def is_credit_error(exc):
    s = str(exc)
    return "402" in s or "requires more credits" in s.lower()


def rel3(p):
    return "/".join(str(p).strip("/").split("/")[-3:])


def atomic_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_jsonl(path):
    """Tolerates a truncated final line -- the only damage an append can suffer."""
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("key"):
                out[r["key"]] = r
    return out


def append_jsonl(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def extract_frames_at(clip_path, tmp, fps):
    """Same filter chain as caption_openrouter.extract_frames, at an explicit fps."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip_path),
                    "-vf", f"fps={fps},scale='min(1280,iw)':-2", "-q:v", "3", f"{tmp}/f_%04d.jpg"],
                   capture_output=True)
    return sorted(str(p) for p in Path(tmp).glob("f_*.jpg"))


def interleaved_draw(n, k, p, n_passes):
    """k frames evenly spaced across n, phase-shifted by pass so the passes tile the gap evenly.

    n=50 (20 s @2.5 fps), k=10, n_passes=5 -> step 5 frames (2.0 s), offsets 0,1,2,3,4 frames
    (0.0,0.4,0.8,1.2,1.6 s). Generalises to any clip length.
    """
    if n <= k:
        return list(range(n))
    step = n / k
    off = (p - 1) * step / n_passes
    idx = sorted({min(n - 1, int(round(off + i * step))) for i in range(k)})
    return idx


def disk_clips():
    d = json.load(open(INDEX))
    entries = [x for x in (d if isinstance(d, list) else d.get("clips", [])) if isinstance(x, dict)]
    found = {}
    for r in ROOTS:
        for f in glob.glob(str(r) + "/**/*.mp4", recursive=True):
            found.setdefault(rel3(f), Path(f))
    out = {}
    for e in entries:
        cp = e.get("clip_path")
        if cp and rel3(cp) in found:
            out[rel3(cp)] = {"path": found[rel3(cp)], "camera": e.get("camera"), "date": e.get("date")}
    return out


def run_one(key, meta, p, n_passes, prompt):
    """One (clip, pass). Returns a record dict, or None on failure."""
    with tempfile.TemporaryDirectory() as tmp:
        frames = extract_frames_at(meta["path"], tmp, DENSE_FPS)
        if not frames:
            return None
        pick = interleaved_draw(len(frames), N_DRAW, p, n_passes)
        imgs = [C.b64_image(frames[i]) for i in pick]
        try:
            raw = C.call_openrouter(imgs, prompt)
        except Exception as ex:
            if is_credit_error(ex):
                CREDITS_EXHAUSTED.set()          # abort the run; do not spend an attempt
            return None
    cap, etho = C.parse(raw)
    return {"key": key, "pass": p, "camera": meta["camera"], "date": meta["date"],
            "dense_fps": DENSE_FPS, "frames_available": len(frames),
            "frames_used": pick, "n_sent": len(imgs),
            # <= N_DRAW candidates means every pass draws the SAME frames, so unanimity there is an
            # artifact of the sampler, not evidence. The vote script reports those separately.
            "sampling_varies": len(frames) > N_DRAW,
            "caption": cap, "ethogram": etho,
            "present": ("not present" not in str(etho).lower()),
            "at": datetime.datetime.now().isoformat(timespec="seconds")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="cap cells per round (smoke test)")
    ap.add_argument("--max-rounds", type=int, default=40)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    attempts = json.load(open(ATTEMPTS)) if ATTEMPTS.exists() else {}
    prompt = C.build_prompt()
    print(f"model={C.OR_MODEL}  passes={args.passes}  {N_DRAW} frames @ {DENSE_FPS} fps "
          f"(interleaved uniform, no CLIP scoring)", flush=True)

    t0 = time.time()
    for rnd in range(1, args.max_rounds + 1):
        clips = disk_clips()                       # RECOMPUTED: picks up newly downloaded clips
        done = {p: read_jsonl(OUTDIR / f"pass{p}.jsonl") for p in range(1, args.passes + 1)}
        todo = [(k, m, p) for k, m in clips.items() for p in range(1, args.passes + 1)
                if k not in done[p] and attempts.get(f"{k}|{p}", 0) < MAX_ATTEMPTS]
        have = sum(len(v) for v in done.values())
        print(f"\n-- round {rnd}: {len(clips)} clips on disk | {have}/{len(clips)*args.passes} "
              f"cells done | {len(todo)} to do", flush=True)
        if not todo:
            print("nothing left to do."); break
        if args.limit:
            todo = todo[:args.limit]
        before = have

        def work(item):
            k, m, p = item
            if CREDITS_EXHAUSTED.is_set():
                return                            # no attempt spent, no failure recorded
            with att_lock:
                attempts[f"{k}|{p}"] = attempts.get(f"{k}|{p}", 0) + 1
            rec = run_one(k, m, p, args.passes, prompt)
            with io_lock:
                if rec is None:
                    state["fail"] += 1
                else:
                    append_jsonl(OUTDIR / f"pass{p}.jsonl", rec)
                    state["ok"] += 1
                n = state["ok"] + state["fail"]
                if n % 50 == 0:
                    atomic_write(ATTEMPTS, attempts)
                    el = time.time() - t0
                    print(f"   [{n}] ok={state['ok']} fail={state['fail']} "
                          f"| {n/max(el,1)*60:.1f} cells/min", flush=True)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, todo))
        atomic_write(ATTEMPTS, attempts)
        if CREDITS_EXHAUSTED.is_set():
            print("\nCREDITS EXHAUSTED (HTTP 402) -- stopping. No attempts were spent on the "
                  "remaining cells; top up and re-run to resume exactly here.", flush=True)
            sys.exit(3)

        after = sum(len(read_jsonl(OUTDIR / f"pass{p}.jsonl")) for p in range(1, args.passes + 1))
        if after <= before:
            print("no progress this round -- stopping to avoid a spin."); break

    print(f"\nDONE. ok={state['ok']} fail={state['fail']} in {(time.time()-t0)/60:.1f} min")
    for p in range(1, args.passes + 1):
        print(f"  pass{p}: {len(read_jsonl(OUTDIR / f'pass{p}.jsonl'))} clips")
    print(f"-> {OUTDIR}")


if __name__ == "__main__":
    main()
