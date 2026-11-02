"""Pre-cut clips: one event of a long video stored as the exact model input the training loader would build.

``cut_clip`` replays ``training/qwenvl/data/data_qwen.py::video_decord`` (decord, ``frame_idx`` = linspace over
``[round(start*fps), min(round(end*fps), n-1)]``, ``video_length`` = ``min(end, n/fps) - start``) and the video-only
audio branch (``load_audio_segment`` over the same window, centre crop to ``audio_sec`` as ``fit_waveform``), then
resizes the frames with the image processor's own ``smart_resize`` + ``resize`` so the loader can call the processor
with ``do_resize=False`` and get the same pixels. ``load_clip`` reads it back.

Clip file (``np.savez_compressed``): ``frames`` uint8 (F, H, W, 3), ``video_length`` float, ``wav`` int16 (at most
``audio_sec`` s, empty when the video has no readable audio), ``audio_ok`` bool, ``src_fps``, ``frame_idx``,
``window``, ``total_frames``.
"""

from __future__ import annotations

import numpy as np

from omniretriever.data.media import load_audio_segment

INT16 = 32767.0


class PastEnd(ValueError):
    """The window starts at or after the end of the file (or decord cannot place a frame in it)."""


def frame_indices(total: int, fps: float, start: float, end: float, n_frames: int):
    """``(frame_idx, video_length)`` exactly as data_qwen.video_decord with ``timestamps=(start, end)`` and
    ``video_min_frames == video_max_frames == n_frames``."""
    video_length = min(end, total / fps) - start
    start_idx = round(start * fps)
    end_idx = min(round(end * fps), total - 1)
    if video_length <= 0 or start_idx > end_idx:
        raise PastEnd(f"window {start}-{end} past the end ({total / fps:.2f}s)")
    return np.linspace(start_idx, end_idx, n_frames, dtype=int), video_length


def resize_frames(frames: np.ndarray, image_processor, min_pixels: int, max_pixels: int) -> np.ndarray:
    """The resize step of Qwen2VLImageProcessor._preprocess, applied ahead of time to uint8 frames."""
    from transformers.image_transforms import resize
    from transformers.image_utils import ChannelDimension
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    h, w = frames.shape[1:3]
    factor = image_processor.patch_size * image_processor.merge_size
    rh, rw = smart_resize(h, w, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    return np.stack([resize(f, size=(rh, rw), resample=image_processor.resample,
                            input_data_format=ChannelDimension.LAST) for f in frames]).astype(np.uint8)


def cut_clip(video_path: str, start: float, end: float, image_processor, *, n_frames: int = 8,
             min_pixels: int = 50176, max_pixels: int = 50176, audio_sec: float = 8.0,
             sample_rate: int = 16000) -> dict:
    """Arrays of one clip (see the module docstring). ``end`` past the file is clamped; raises PastEnd."""
    from decord import VideoReader

    vr = VideoReader(video_path, num_threads=1)
    total, fps = len(vr), vr.get_avg_fps()
    duration = total / fps
    if start >= duration:
        raise PastEnd(f"start {start} >= length {duration:.2f}s")
    window = (float(start), float(min(end, duration)))
    idx, video_length = frame_indices(total, fps, *window, n_frames)
    frames = resize_frames(np.asarray(vr.get_batch(idx).asnumpy()), image_processor, min_pixels, max_pixels)
    try:
        wav = load_audio_segment(video_path, *window, sample_rate=sample_rate)
        target = int(audio_sec * sample_rate)
        if wav.shape[0] > target:                       # fit_waveform's centre crop; padding is left to the loader
            off = (wav.shape[0] - target) // 2
            wav = wav[off:off + target]
        wav16, audio_ok = np.clip(np.round(wav * INT16), -32768, 32767).astype(np.int16), True
    except Exception:                                   # no audio stream / nothing decoded: the loader uses silence
        wav16, audio_ok = np.zeros(0, np.int16), False
    return {"frames": frames, "video_length": np.float64(video_length), "wav": wav16,
            "audio_ok": np.bool_(audio_ok), "src_fps": np.float64(fps), "frame_idx": idx.astype(np.int64),
            "window": np.asarray(window, np.float64), "total_frames": np.int64(total)}


def load_clip(path: str) -> dict:
    """``frames`` (F, H, W, 3) uint8, ``video_length`` float, ``wav`` float32 or None (no audio)."""
    with np.load(path) as z:
        wav = z["wav"].astype(np.float32) / INT16 if bool(z["audio_ok"]) else None
        return {"frames": z["frames"], "video_length": float(z["video_length"]), "wav": wav,
                "n_frames": int(len(z["frame_idx"])), "window": z["window"].tolist()}
