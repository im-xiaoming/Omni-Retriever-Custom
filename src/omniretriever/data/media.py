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
    timestamps: tuple[float, float] | None = None,
) -> np.ndarray:
    """Load ``num_frames`` frames from a video as a ``(N, H, W, 3)`` ``uint8`` array.

    Args:
        path: path to the input video file.
        num_frames: number of frames to sample.
        resolution: target square resolution for each frame.
        sampling: ``"uniform"`` (default, evenly spaced) or ``"random"`` for a
            random subset of ``num_frames`` frames.
        timestamps: optional ``(start, end)`` in seconds. Sampling is then
            confined to that window instead of the whole file, matching what
            ``training/qwenvl/data/data_qwen.py`` does with the ``timestamps``
            field of a manifest record. Must be a tuple: this function is
            memoised, so the argument has to be hashable.

    Returns:
        A NumPy array of shape ``(num_frames, resolution, resolution, 3)``,
        ``dtype=uint8``, RGB.

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
        would not be -- pass ``timestamps`` for those, which bounds both the
        decode and the memory by the window rather than the file.
    """
    import av

    path = str(path)
    with av.open(path) as container:
        if not container.streams.video:
            raise RuntimeError(f"{path!r} contains no video stream.")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if timestamps is None:
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
            idx = _sample_indices(len(frames), num_frames, sampling, path, timestamps)
        else:
            # Two passes over the window: count, then convert only the sampled
            # frames. One pass held every frame at full resolution, which is
            # tens of GB for a minutes-long HT-Step step; the frames picked are
            # the same either way.
            total = sum(1 for _ in _window_frames(container, stream, *timestamps))
            idx = _sample_indices(total, num_frames, sampling, path, timestamps)
            keep = set(idx.tolist())
            frames = {i: frame.to_ndarray(format="rgb24")
                      for i, frame in enumerate(_window_frames(container, stream, *timestamps))
                      if i in keep}

    return _resize_frames(np.stack([frames[i] for i in idx]), resolution)


def _sample_indices(total, num_frames, sampling, path, timestamps) -> np.ndarray:
    """Indices of the frames to keep out of ``total`` decoded ones."""
    if total == 0:
        window = "" if timestamps is None else f" within {timestamps[0]}-{timestamps[1]}s"
        raise RuntimeError(f"Video {path!r} has zero frames{window}.")

    if sampling == "uniform":
        return np.linspace(0, total - 1, num=num_frames, dtype=int)
    if sampling == "random":
        return np.sort(
            np.random.default_rng().choice(total, size=min(num_frames, total), replace=False)
        )
    raise ValueError(f"Unknown sampling strategy: {sampling!r}")


def _window_frames(container, stream, start: float, end: float):
    """Yield the frames presented between ``start`` and ``end`` seconds, not yet converted to RGB.

    Seeking lands on the keyframe at or before the target, never exactly on it,
    so the frames before ``start`` still have to be dropped by timestamp. The
    loop stops at the first frame past ``end`` rather than filtering to the end
    of the file, which is what keeps a 13 s window out of a 10 min video cheap.
    """
    time_base = stream.time_base
    if time_base is None:
        raise RuntimeError("Stream has no time base; cannot honour timestamps.")

    container.seek(max(int(start / time_base), 0), stream=stream)

    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        seconds = float(frame.pts * time_base)
        if seconds < start:
            continue
        if seconds > end:
            break
        yield frame


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
    timestamps: tuple[float, float] | None = None,
) -> np.ndarray:
    """Load a fixed-duration mono waveform from a video or audio container.

    Args:
        path: path to the source file. Video containers (``.mp4``, ``.mov``,
            ``.webm``) are decoded for their audio track; pure audio files
            (``.wav``, ``.flac``, ``.mp3``) are loaded directly.
        duration_sec: target duration in seconds. Longer waveforms are centre-
            cropped, shorter ones are padded (see ``pad_mode``).
        sample_rate: target sample rate (Hz).
        pad_mode: how to pad short clips. One of ``"zero"`` (silence) or
            ``"loop"`` (repeat the source until it fills the window).
        timestamps: optional ``(start, end)`` in seconds, applied before the
            crop/pad above. Needed when the source is a long container holding
            many segments; pass ``None`` for a file that is already one segment.
            Must be a tuple, since this function is memoised.

    Returns:
        A NumPy array of shape ``(duration_sec * sample_rate,)``, ``dtype=float32``.
    """
    path = str(path)

    if timestamps is None:
        waveform = np.asarray(_decode_waveform(path, sample_rate), dtype=np.float32)
    else:
        # Decode only the event's window instead of the whole track and slicing
        # it afterwards: a YouCookII event is ~10 s out of a 5-10 min video.
        waveform = load_audio_segment(path, *timestamps, sample_rate=sample_rate)

    return fit_waveform(waveform, int(duration_sec * sample_rate), pad_mode=pad_mode)


def fit_waveform(waveform: np.ndarray, target_len: int, pad_mode: str = "zero") -> np.ndarray:
    """Centre-crop or pad ``waveform`` to exactly ``target_len`` samples.

    The single fixed-duration policy shared by inference and
    ``training/qwenvl/data/data_qwen.py``, so an event's audio is cut the same
    way on both sides.
    """
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.shape[0] >= target_len:
        start = (waveform.shape[0] - target_len) // 2
        return waveform[start:start + target_len]

    if pad_mode == "zero":
        out = np.zeros(target_len, dtype=np.float32)
        out[: waveform.shape[0]] = waveform
        return out
    if pad_mode == "loop":
        n = (target_len // max(waveform.shape[0], 1)) + 1
        return np.tile(waveform, n)[:target_len]

    raise ValueError(f"Unknown pad_mode: {pad_mode!r}")


def load_audio_segment(
    path: str | os.PathLike,
    start: float | None = None,
    end: float | None = None,
    *,
    sample_rate: int = 16_000,
) -> np.ndarray:
    """Cut the audio of one event out of a video (or audio) container.

    Returns the mono float32 waveform between ``start`` and ``end`` seconds at
    ``sample_rate``, uncropped and unpadded. ``None`` for either bound means the
    start / end of the track. Channels are averaged, as ``librosa.load(mono=True)``
    does.

    Only the window is decoded: the container is seeked to ``start`` and decoding
    stops at the first frame past ``end``, so the cost scales with the event
    rather than the video.
    """
    import av

    path = str(path)
    chunks: list[np.ndarray] = []
    first_time = None
    with av.open(path) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise RuntimeError(f"{path!r} contains no audio stream.")
        time_base = stream.time_base
        # Keep the channel layout and average afterwards (librosa's downmix)
        # rather than let the resampler apply its own mono mix coefficients.
        resampler = av.audio.resampler.AudioResampler(format="fltp", rate=sample_rate)

        if start:
            # Lands on the frame at or before ``start``; the overshoot is trimmed below.
            container.seek(max(int(start / time_base), 0), stream=stream)

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_start = float(frame.pts * time_base)
            if end is not None and frame_start >= end:
                break
            if start is not None and frame_start + frame.samples / frame.sample_rate <= start:
                continue
            if first_time is None:
                first_time = frame_start
            chunks.extend(r.to_ndarray().mean(axis=0) for r in resampler.resample(frame))
        chunks.extend(r.to_ndarray().mean(axis=0) for r in resampler.resample(None))

    if not chunks:
        raise RuntimeError(f"Decoded no audio samples from {path!r} within {start}-{end}s.")
    waveform = np.concatenate(chunks).astype(np.float32)

    begin = start if start is not None else first_time
    offset = max(int(round((begin - first_time) * sample_rate)), 0)
    waveform = waveform[offset:]
    if end is not None:
        waveform = waveform[: max(int(round((end - begin) * sample_rate)), 0)]
    if waveform.shape[0] == 0:
        raise RuntimeError(f"Decoded no audio samples from {path!r} within {start}-{end}s.")
    return waveform


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
