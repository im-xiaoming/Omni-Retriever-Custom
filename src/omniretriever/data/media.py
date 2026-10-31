"""Lightweight media loaders shared between training and inference.

The loaders intentionally do not import heavy frameworks at module-import time;
all heavy dependencies (``av``, ``librosa``) are imported lazily inside the
function bodies so users who only need text encoding can avoid the cost.

Video decoding moved from ``decord`` to PyAV downstream of the release; see the
note on :func:`load_video_frames`. ``decord`` is no longer imported anywhere.
"""

from __future__ import annotations

import functools
import os
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np

# librosa cannot read the AAC track inside an MP4 through libsndfile, so it falls
# back to audioread and warns about it -- twice per clip, several lines each. The
# fallback is the correct path and its output is what every existing embedding was
# built on, so this silences the noise and changes nothing else. Worth doing: on
# Colab every one of those lines is streamed to the browser, and a full run makes
# tens of thousands of them.
warnings.filterwarnings("ignore", message="PySoundFile failed.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*__audioread_load.*", category=FutureWarning)

# ADDED downstream. A single extraction pass touches the same clip three times,
# once for ``video``, once for ``av`` and once for ``tv``, and decoding is the
# slowest step in the whole pipeline. Memoising the loaders turns that into one
# decode per clip as long as the three calls land close together, which is what
# the record-major batch order in ``omniretriever.cli`` arranges.
#
# The entries are small -- video frames come back already resized, so an 8-frame
# 224 px clip is 1.2 MB, and an 8 s mono waveform at 16 kHz is 512 KB -- so a few
# dozen of them cost tens of megabytes. Callers must not mutate what they get
# back: every hit hands out the same array.
VIDEO_CACHE_SIZE = 32
AUDIO_CACHE_SIZE = 32


# --------------------------------------------------------------------------- #
# Video                                                                       #
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=VIDEO_CACHE_SIZE)
def load_video_frames(
    path: str | os.PathLike,
    *,
    num_frames: int = 8,
    resolution: int = 224,
    sampling: str = "uniform",
    center_crop: bool = True,
) -> np.ndarray:
    """Load ``num_frames`` frames from a video as a ``(N, H, W, 3)`` ``uint8`` array.

    Args:
        path: path to the input video file.
        num_frames: number of frames to sample.
        resolution: target square resolution for each frame.
        sampling: ``"uniform"`` (default, evenly spaced) or ``"random"`` for a
            random subset of ``num_frames`` frames.
        center_crop: centre-crop and resize to ``resolution``. With ``False``
            the frames come back at full size, as training hands them to the
            processor, and ``resolution`` is ignored.

    Returns:
        A NumPy array of shape ``(num_frames, resolution, resolution, 3)``
        (full frame size without ``center_crop``), ``dtype=uint8``, RGB.

    Note:
        CHANGED downstream. The released code decoded with ``decord``, which
        hangs forever inside its native reader on a large share of this
        benchmark's clips -- 561 of 3515 in a full sweep, concentrated in the
        newest source ids. The hang happens while constructing ``VideoReader``,
        in C, so no Python-side timeout can interrupt it. PyAV reads every one
        of those files (562 of the 567 decord could not handle; the other five
        decode to zero frames under either library and are genuinely broken),
        and it is already a dependency because the audio loader falls back to
        it.

        The two libraries disagree on how many frames a clip holds, because
        decord trusts the container index while this counts frames it actually
        decoded. Uniform sampling spreads its indices over that total, so the
        selected frames -- and every ``video``, ``av`` and ``tv`` embedding --
        differ from a decord-based run. Do not mix embeddings across the two.

        Frames are held at full resolution until the sample is picked, so peak
        memory scales with clip length rather than ``num_frames``. The
        benchmark's p99 clip is 16 s, which is comfortable; very long inputs
        would not be.
    """
    frames, _ = _sample_video(str(path), num_frames, sampling)
    return _resize_frames(frames, resolution) if center_crop else frames


def video_sample_fps(path: str | os.PathLike, *, num_frames: int = 8,
                     sampling: str = "uniform") -> float:
    """Rate at which :func:`load_video_frames` samples the clip, frames per second.

    ADDED downstream. Training computes it as ``num_frames / duration`` with
    ``duration = decoded frames / average frame rate`` and passes it to the
    processor (``data_qwen.py::video_decord``); shares the decode cache.
    """
    return _sample_video(str(path), num_frames, sampling)[1]


@functools.lru_cache(maxsize=VIDEO_CACHE_SIZE)
def _sample_video(path: str, num_frames: int, sampling: str) -> tuple[np.ndarray, float]:
    """Decode ``path`` and return its sampled full-size frames and the sampling fps."""
    import av

    with av.open(path) as container:
        if not container.streams.video:
            raise RuntimeError(f"{path!r} contains no video stream.")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = float(stream.average_rate or stream.guessed_rate or 0)
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]

    total = len(frames)
    if total == 0:
        raise RuntimeError(f"Video {path!r} has zero frames.")

    if sampling == "uniform":
        idx = np.linspace(0, total - 1, num=num_frames, dtype=int)
    elif sampling == "random":
        idx = np.sort(
            np.random.default_rng().choice(total, size=min(num_frames, total), replace=False)
        )
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling!r}")

    duration = total / rate if rate > 0 else float(total)
    return np.stack([frames[i] for i in idx]), len(idx) / duration


def _resize_frames(frames: np.ndarray, resolution: int) -> np.ndarray:
    """Centre-crop and resize a batch of frames to ``(resolution, resolution)``."""
    from PIL import Image

    out = np.empty((frames.shape[0], resolution, resolution, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        w, h = img.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        img = img.crop((left, top, left + side, top + side)).resize(
            (resolution, resolution), Image.BICUBIC
        )
        out[i] = np.asarray(img)
    return out


# --------------------------------------------------------------------------- #
# Audio                                                                       #
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=AUDIO_CACHE_SIZE)
def load_audio_waveform(
    path: str | os.PathLike,
    *,
    duration_sec: int = 8,
    sample_rate: int = 16_000,
    pad_mode: str = "zero",
    crop: str = "centre",
) -> np.ndarray:
    """Load a fixed-duration mono waveform from a video or audio container.

    Args:
        path: path to the source file. Video containers (``.mp4``, ``.mov``,
            ``.webm``) are decoded for their audio track; pure audio files
            (``.wav``, ``.flac``, ``.mp3``) are loaded directly.
        duration_sec: target duration in seconds. Longer waveforms are cropped
            (see ``crop``), shorter ones are padded (see ``pad_mode``).
        sample_rate: target sample rate (Hz).
        pad_mode: how to pad short clips. One of ``"zero"`` (silence) or
            ``"loop"`` (repeat the source until it fills the window).
        crop: ``"centre"`` or ``"start"`` (keep the first ``duration_sec``
            seconds, as training does).

    Returns:
        A NumPy array of shape ``(duration_sec * sample_rate,)``, ``dtype=float32``.
    """
    path = str(path)
    target_len = int(duration_sec * sample_rate)

    waveform = np.asarray(_decode_waveform(path, sample_rate), dtype=np.float32)

    if waveform.shape[0] >= target_len:
        if crop not in ("centre", "start"):
            raise ValueError(f"Unknown crop: {crop!r}")
        start = (waveform.shape[0] - target_len) // 2 if crop == "centre" else 0
        return waveform[start:start + target_len]

    # Pad.
    if pad_mode == "zero":
        out = np.zeros(target_len, dtype=np.float32)
        out[: waveform.shape[0]] = waveform
        return out
    if pad_mode == "loop":
        n = (target_len // max(waveform.shape[0], 1)) + 1
        return np.tile(waveform, n)[:target_len]

    raise ValueError(f"Unknown pad_mode: {pad_mode!r}")


def _decode_waveform(path: str, sample_rate: int) -> np.ndarray:
    """Decode a file to a mono waveform at ``sample_rate``.

    ``librosa`` is tried first: it is fast for plain audio files, which
    ``libsndfile`` reads directly. It cannot handle the compressed tracks inside
    video containers (AAC in MP4, for instance), and then falls back to
    ``audioread``, which needs an ffmpeg binary on PATH. When that is missing we
    decode with PyAV, which links its own ffmpeg libraries and so needs nothing
    installed system-wide.
    """
    import librosa

    try:
        waveform, _ = librosa.load(path, sr=sample_rate, mono=True)
        return waveform
    except Exception as exc:  # noqa: BLE001 - librosa surfaces backend errors as many types
        librosa_error = exc

    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            f"Could not decode audio from {path!r}. librosa failed "
            f"({type(librosa_error).__name__}: {librosa_error}) and PyAV is not "
            "installed. Install ffmpeg and put it on PATH, or 'pip install av'."
        ) from exc

    return _decode_waveform_pyav(av, path, sample_rate)


def _decode_waveform_pyav(av, path: str, sample_rate: int) -> np.ndarray:
    """Decode the first audio stream of ``path`` to mono float32 at ``sample_rate``."""
    with av.open(path) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise RuntimeError(f"{path!r} contains no audio stream.")

        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout="mono", rate=sample_rate
        )
        chunks = [
            resampled.to_ndarray().reshape(-1)
            for frame in container.decode(stream)
            for resampled in resampler.resample(frame)
        ]
        # Drain whatever the resampler is still buffering.
        chunks.extend(resampled.to_ndarray().reshape(-1) for resampled in resampler.resample(None))

    if not chunks:
        raise RuntimeError(f"Decoded no audio samples from {path!r}.")
    return np.concatenate(chunks)
