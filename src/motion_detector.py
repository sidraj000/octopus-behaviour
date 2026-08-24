"""
Motion detector — frame differencing to measure activity over time.

Streams frames via ffmpeg and computes mean absolute pixel difference
between consecutive frames (grayscale). Returns a per-second motion
score in [0, 1], where 1 = maximum change between frames.

Works on local files and remote HTTP streams.
"""

import logging
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_GRAY_SIZE = 224


def _stream_gray_frames(source: str, fps: float = 1.0):
    """
    Yield (timestamp_sec, H×W uint8 grayscale frame) via ffmpeg pipe.
    source: local path or http(s) URL (auth already embedded).
    """
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", source,
        "-vf", f"fps={fps},scale={_GRAY_SIZE}:{_GRAY_SIZE},format=gray",
        "-f", "image2pipe",
        "-vcodec", "rawvideo",
        "-pix_fmt", "gray",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_size = _GRAY_SIZE * _GRAY_SIZE
    interval = 1.0 / fps
    ts = 0.0
    FRAME_TIMEOUT = 30  # seconds — kill ffmpeg if no frame arrives within this

    def _read_frame():
        return proc.stdout.read(frame_size)

    while True:
        buf = [None]
        t = threading.Thread(target=lambda: buf.__setitem__(0, _read_frame()), daemon=True)
        t.start()
        t.join(timeout=FRAME_TIMEOUT)
        if t.is_alive():
            proc.kill()
            break
        raw = buf[0]
        if raw is None or len(raw) < frame_size:
            break
        yield ts, np.frombuffer(raw, dtype=np.uint8).reshape((_GRAY_SIZE, _GRAY_SIZE))
        ts += interval
    proc.stdout.close()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def scan_motion(
    source: str,
    fps: float = 1.0,
    smooth_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-frame motion score via absolute frame differencing.

    Parameters
    ----------
    source : local file path or authenticated remote URL
    fps    : frames per second to sample (1.0 is sufficient for 30-min videos)
    smooth_window : rolling average window in frames

    Returns
    -------
    timestamps   : float32 array, seconds from start
    motion_scores: float32 array in [0, 1], normalised mean abs diff
    """
    t0 = time.perf_counter()
    log.info("Motion scan: %s  (%.1f fps)", Path(source).name if not source.startswith("http") else source[-40:], fps)

    timestamps, raw_scores = [], []
    prev_frame = None

    for ts, frame in _stream_gray_frames(source, fps):
        if prev_frame is not None:
            diff = np.mean(np.abs(frame.astype(np.int16) - prev_frame.astype(np.int16)))
            raw_scores.append(float(diff))
            timestamps.append(ts)
        prev_frame = frame

    if not raw_scores:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    scores = np.array(raw_scores, dtype=np.float32)
    # normalise to [0, 1] relative to the max in this video
    max_val = scores.max()
    if max_val > 0:
        scores /= max_val

    # smooth
    if smooth_window > 1 and len(scores) >= smooth_window:
        kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
        scores = np.convolve(scores, kernel, mode="same")

    log.info(
        "Motion scan done: %d frames in %.1fs  |  mean=%.3f  max=%.3f",
        len(scores), time.perf_counter() - t0, scores.mean(), scores.max(),
    )
    return np.array(timestamps, dtype=np.float32), scores


def scan_motion_area(
    source: str,
    fps: float = 1.0,
    pix_thresh: int = 25,
    mask_timestamp: bool = True,
    brightness_norm: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Motion as the FRACTION OF PIXELS that genuinely changed — absolute, not
    normalised. Robust alternative to scan_motion() for low-motion / flickering
    footage (e.g. IR-lamp pulsing, ticking timestamp).

    For each consecutive frame pair:
      1. optional per-frame brightness de-mean (removes uniform flicker)
      2. abs pixel diff, with the bottom-right timestamp region masked out
      3. count pixels whose change exceeds `pix_thresh` grey-levels
      4. score = changed_pixels / total_pixels   (in [0, 1])

    Unlike scan_motion(), the result is NOT divided by the per-video max, so the
    threshold has a fixed physical meaning ("X% of the frame actually moved").

    Parameters
    ----------
    source          : local file path or authenticated remote URL
    fps             : frames per second to sample
    pix_thresh      : per-pixel grey-level change to count as "moved" (0-255)
    mask_timestamp  : zero out the burned-in datetime (bottom-right corner)
    brightness_norm : subtract each frame's mean before diffing (kills global flicker)

    Returns
    -------
    timestamps    : float32 seconds from start
    motion_frac   : float32 in [0, 1], absolute fraction of changed pixels
    """
    t0 = time.perf_counter()
    timestamps, scores = [], []
    prev = None

    for ts, frame in _stream_gray_frames(source, fps):
        f = frame.astype(np.float32)
        if brightness_norm:
            f = f - f.mean()
        if prev is not None:
            diff = np.abs(f - prev)
            if mask_timestamp:
                h, w = diff.shape
                diff[int(h * 0.88):, int(w * 0.60):] = 0.0   # burned-in datetime
            scores.append(float((diff > pix_thresh).mean()))
            timestamps.append(ts)
        prev = f

    scores = np.array(scores, dtype=np.float32)
    log.info(
        "Area-motion scan: %d frames in %.1fs  |  mean=%.4f  max=%.4f  (pix_thresh=%d)",
        len(scores), time.perf_counter() - t0,
        scores.mean() if len(scores) else 0.0, scores.max() if len(scores) else 0.0,
        pix_thresh,
    )
    return np.array(timestamps, dtype=np.float32), scores
