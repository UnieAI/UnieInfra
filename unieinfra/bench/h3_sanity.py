# Copyright (c) 2026 LightSeek Foundation
# Copyright (c) 2026 UnieAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""The cheap tier of MiniMax-H3 output verification.

TensorRT-LLM's ``visual_gen/ENGINEERING_CRITERIA.md`` §5.1 splits diffusion
verification in two: a **sanity** check that runs broadly ("content was
generated, valid shape, non-black frame") and a **quality gate** (LPIPS or
VBench against a reference) that runs where quality matters. This is the
first tier only; it answers "did the engine produce a video and a soundtrack
at all", not "is it the right video".

It exists because the failure it catches is invisible to every other signal.
A run that OOMs mid-denoise still exits 0 under ``sglang generate``; a run
that denoises garbage still writes a structurally valid mp4 of the right
duration, resolution and frame rate. Neither a timing number nor ``ffprobe``
would separate those from a good run -- only the pixels and the samples do.

Decoding goes through ``ffmpeg`` rather than imageio: imageio's pyav plugin
raises ``AttributeError: 'VideoCodecContext' object has no attribute 'close'``
against the ``av`` version SGLang's diffusion extra pins.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np

# Decoded at a small size on purpose: this tier asks whether there are pixels
# and whether they move, and neither question needs full resolution.
PROBE_W, PROBE_H = 168, 96
AUDIO_RATE = 16000


def _ffmpeg(path: str, args: list[str]) -> bytes:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, *args, "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {proc.stderr.decode()[:400]}")
    return proc.stdout


def video_stats(path: str) -> dict:
    raw = _ffmpeg(
        path,
        [
            "-map",
            "0:v",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-s",
            f"{PROBE_W}x{PROBE_H}",
        ],
    )
    frames = np.frombuffer(raw, dtype=np.uint8)
    n = frames.size // (PROBE_W * PROBE_H)
    if n == 0:
        raise RuntimeError(f"{path} decoded to zero video frames")
    frames = frames[: n * PROBE_W * PROBE_H].reshape(n, PROBE_H * PROBE_W)
    means = frames.mean(axis=1)
    return {
        "frames": n,
        "luma_min": float(means.min()),
        "luma_max": float(means.max()),
        "luma_mean": float(means.mean()),
        "black_frames": int((means < 1.0).sum()),
        # A frame of uniform colour has zero spatial variance: structurally a
        # video, visually nothing.
        "spatial_std": float(frames.std(axis=1).mean()),
        # Zero means every frame is identical -- a still, not a video.
        "motion": float(np.abs(np.diff(frames.astype(np.int16), axis=0)).mean()),
    }


def audio_stats(path: str) -> dict:
    raw = _ffmpeg(
        path, ["-map", "0:a", "-f", "s16le", "-ac", "1", "-ar", str(AUDIO_RATE)]
    )
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0:
        raise RuntimeError(f"{path} decoded to zero audio samples")
    return {
        "seconds": float(samples.size / AUDIO_RATE),
        "rms": float(np.sqrt((samples**2).mean())),
        "peak": float(np.abs(samples).max()),
    }


def check(path: str, *, require_audio: bool = True) -> list[str]:
    """Return the list of failures; empty means the sanity tier passed."""
    failures = []
    v = video_stats(path)
    print(
        f"VIDEO  frames={v['frames']} luma {v['luma_min']:.1f}..{v['luma_max']:.1f} "
        f"(mean {v['luma_mean']:.1f})  spatial_std={v['spatial_std']:.1f}  "
        f"motion={v['motion']:.2f}  black={v['black_frames']}"
    )
    if v["black_frames"] == v["frames"]:
        failures.append("every frame is black")
    if v["spatial_std"] < 1.0:
        failures.append(f"frames are flat (spatial_std {v['spatial_std']:.2f})")
    if v["motion"] < 0.01:
        failures.append(f"no motion between frames (motion {v['motion']:.4f})")

    if require_audio:
        a = audio_stats(path)
        print(f"AUDIO  {a['seconds']:.2f} s  rms={a['rms']:.4f}  peak={a['peak']:.4f}")
        if a["rms"] < 1e-4:
            failures.append(f"audio is silent (rms {a['rms']:.6f})")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="generated .mp4")
    ap.add_argument(
        "--no-audio",
        action="store_true",
        help="skip the soundtrack check (H3 always has one)",
    )
    args = ap.parse_args()

    failures = check(args.path, require_audio=not args.no_audio)
    if failures:
        print("SANITY FAILED: " + "; ".join(failures), file=sys.stderr)
        return 1
    print("SANITY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
