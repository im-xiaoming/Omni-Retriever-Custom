"""Modality-specific encoding routines.

Each ``encode_*`` function takes a (single string or list-of-strings) input,
builds the appropriate multimodal prompt for WAVE-7B, runs a single forward
pass with the all-layer fusion head, and returns L2-normalised embeddings of
shape ``(N, D)`` with ``D = 3584``.

Inputs are batched automatically when a list is provided; users should chunk
manually if memory is tight.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from omniretriever.data.media import load_audio_waveform, load_video_frames, video_sample_fps

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public entry points                                                         #
# --------------------------------------------------------------------------- #


def encode_text(backbone, processor, text, config):
    """Encode a string (or list of strings) into the shared embedding space."""
    end = _turn_end(processor, config)
    texts = [t + end for t in _as_list(text)]
    inputs = processor(text=texts, padding=True, return_tensors="pt").to(_device(backbone))
    return _forward_and_normalise(backbone, inputs, config)


def encode_video(backbone, processor, video_path, config):
    """Encode video file(s) using the visual stream only."""
    paths = _as_list(video_path)
    frames = _frames(paths, config)
    suffix = _instruction(config, "video") + _turn_end(processor, config)
    prompts = [_video_prompt(processor, suffix)] * len(paths)
    inputs = processor(
        text=prompts,
        videos=frames,
        **_video_kwargs(config, paths),
        padding=True,
        return_tensors="pt",
    ).to(_device(backbone))
    return _forward_and_normalise(backbone, inputs, config)


def encode_audio(backbone, processor, audio_path, config):
    """Encode audio file(s) using the audio stream only."""
    paths = _as_list(audio_path)
    waveforms = _waveforms(paths, config)
    suffix = _instruction(config, "audio") + _turn_end(processor, config)
    prompts = [_audio_prompt(processor, suffix)] * len(paths)
    inputs = processor(
        text=prompts,
        audio=waveforms,
        sampling_rate=config.audio_sample_rate,
        padding=True,
        return_tensors="pt",
    )
    inputs = _apply_beats_audio_slots(inputs, processor, backbone, config).to(_device(backbone))
    return _forward_and_normalise(
        backbone, inputs, config, input_raw_wav=_raw_wav(waveforms, backbone)
    )


def encode_av(backbone, processor, clip_path, config):
    """Encode multimodal clip(s) using both visual and audio streams."""
    paths = _as_list(clip_path)
    frames = _frames(paths, config)
    waveforms = _waveforms(paths, config)
    # With use_audio_in_video the processor rewrites the plain video placeholder
    # into interleaved video/audio chunks itself, so the prompt stays the
    # video-only one and must not carry a separate audio placeholder.
    #
    # The layout here matches every other path -- media, then filler, then the
    # turn-end token -- but only because the backbone's M-RoPE index has been
    # patched. Unpatched, it over-counts the interleaved chunks and drops any
    # trailing token, which used to force the filler in front of the placeholder
    # and left this the one path that could not be given the token the fusion
    # head is trained to pool. See scripts/patch_wave_rope.py.
    _require_rope_patch(backbone)
    suffix = _instruction(config, "av") + _turn_end(processor, config)
    prompts = [_video_prompt(processor, suffix)] * len(paths)
    inputs = processor(
        text=prompts,
        videos=frames,
        audio=waveforms,
        sampling_rate=config.audio_sample_rate,
        **_video_kwargs(config, paths, use_audio_in_video=True),
        use_audio_in_video=True,
        padding=True,
        return_tensors="pt",
    )
    inputs = _apply_beats_audio_slots(inputs, processor, backbone, config).to(_device(backbone))
    return _forward_and_normalise(
        backbone,
        inputs,
        config,
        use_audio_in_video=True,
        input_raw_wav=_raw_wav(waveforms, backbone),
    )


# --------------------------------------------------------------------------- #
# Dual-modal encoders (added downstream, not part of the original release)     #
# --------------------------------------------------------------------------- #
#
# NOTE: everything from here to the "Internals" banner was written for this
# checkout; the released OmniRetriever code ships only the four encoders above.
# Without them the `tv` and `at` query sides are missing, so four of the twelve
# benchmark directions (a2tv, tv2a, v2at, at2v) cannot be scored at all.
#
# The prompt layout follows the training-time recipe in
# ``training/qwenvl/data/data_qwen.py::_prepare_submodal_input``: one media
# placeholder first, the caption after it, and no separate audio placeholder
# whenever a video is present. What it does *not* copy is the generic
# "Please describe the video." filler that upstream substitutes when a
# combination carries no caption, since both combinations here always do.
#
# Caveat worth knowing before trusting the numbers: upstream builds that prompt
# through ``apply_chat_template`` plus ``replace_multimodal_special_tokens``,
# while the released ``encode_*`` helpers hand the processor bare placeholder
# tokens. These two encoders follow the released inference convention so their
# output shares a space with the other four, not the training convention.


def encode_tv(backbone, processor, video_path, text, config):
    """Encode video and text jointly into the shared embedding space.

    Args:
        video_path: one path, or a sequence of paths, to video files.
        text: the matching caption, or a sequence of captions of equal length.

    Returns:
        Tensor of shape ``(N, D)``.
    """
    paths, texts = _pair(video_path, text, "video_path", "text")
    frames = _frames(paths, config)
    end = _turn_end(processor, config)
    prompts = [_video_prompt(processor, t + end) for t in texts]
    inputs = processor(
        text=prompts,
        videos=frames,
        **_video_kwargs(config, paths),
        padding=True,
        return_tensors="pt",
    ).to(_device(backbone))
    return _forward_and_normalise(backbone, inputs, config)


def encode_at(backbone, processor, audio_path, text, config):
    """Encode audio and text jointly into the shared embedding space.

    Args:
        audio_path: one path, or a sequence of paths, to audio files.
        text: the matching caption, or a sequence of captions of equal length.

    Returns:
        Tensor of shape ``(N, D)``.
    """
    paths, texts = _pair(audio_path, text, "audio_path", "text")
    waveforms = _waveforms(paths, config)
    end = _turn_end(processor, config)
    prompts = [_audio_prompt(processor, t + end) for t in texts]
    inputs = processor(
        text=prompts,
        audio=waveforms,
        sampling_rate=config.audio_sample_rate,
        padding=True,
        return_tensors="pt",
    )
    inputs = _apply_beats_audio_slots(inputs, processor, backbone, config).to(_device(backbone))
    return _forward_and_normalise(
        backbone, inputs, config, input_raw_wav=_raw_wav(waveforms, backbone)
    )


def _pair(media, text, media_name: str, text_name: str) -> tuple[list, list]:
    """Normalise a (media, text) argument pair into two equal-length lists."""
    media_list, text_list = _as_list(media), _as_list(text)
    if len(media_list) != len(text_list):
        raise ValueError(
            f"{media_name} and {text_name} must have the same length; "
            f"got {len(media_list)} and {len(text_list)}."
        )
    return media_list, text_list


# --------------------------------------------------------------------------- #
# Internals                                                                   #
# --------------------------------------------------------------------------- #


# Filler the training pipeline appends whenever a modality combination carries
# no caption of its own; see
# ``training/qwenvl/data/data_qwen.py::_prepare_submodal_input``. ADDED to the
# inference path downstream: the released ``encode_*`` helpers passed the bare
# placeholder with no text at all, which is not a shape the backbone ever saw in
# training. Table S2 of the paper counts it too, describing the joint AV input as
# "video + audio + prompt".
MEDIA_INSTRUCTION = "Please describe the video."


_ROPE_PATCH_MARKER = "PATCHED: `st` over-counts"
_rope_patch_checked: set = set()


def _require_rope_patch(backbone) -> None:
    """Fail early, and legibly, when the AV path would hit the M-RoPE bug.

    The unpatched ``get_rope_index`` drops any token after the interleaved
    audio/video block, and the failure surfaces deep inside the forward pass as
    a bare shape mismatch. Checking the source here turns that into a sentence
    naming the fix.
    """
    import inspect

    key = type(backbone).__name__
    if key in _rope_patch_checked:
        return
    try:
        source = inspect.getsource(backbone.get_rope_index)
    except (OSError, TypeError, AttributeError):  # not introspectable; let it run
        _rope_patch_checked.add(key)
        return
    if _ROPE_PATCH_MARKER not in source:
        raise RuntimeError(
            "The WAVE-7B remote code is unpatched, so the av path cannot carry a "
            "trailing token and the forward pass would fail on a shape mismatch. "
            "Run `python scripts/patch_wave_rope.py <WAVE-7B dir>` once against the "
            "model directory, or pass --instruction-paths without 'av' together "
            "with --no-turn-end to fall back to the released layout."
        )
    _rope_patch_checked.add(key)


def _turn_end(processor, config) -> str:
    """The token training put at the end of every prompt, or an empty string.

    ADDED downstream, and the single most consequential prompt difference found
    against ``training/qwenvl/data/data_qwen.py``. Upstream builds each prompt
    with ``apply_chat_template`` and then keeps everything after
    ``"<|im_start|>user
"``, which leaves the turn's closing ``<|im_end|>`` on
    the end. The released ``encode_*`` helpers never add it.

    That matters because the fusion head reads a fixed final position,
    ``hidden_states[:, -1, :]``. In training that position is always the same
    token; at inference it is whatever the prompt happens to end with, and it
    differs per modality -- the caption's last word for text, ``<|vision_eos|>``
    for video, a full stop for audio. The embedding is being read out of a
    position the model was never trained to write it to.

    Not applied on the ``av`` path: as with the filler, the backbone's M-RoPE
    index cannot place any trailing token once audio is interleaved into the
    video stream, and it fails on a shape mismatch of exactly one token.
    """
    return processor.tokenizer.eos_token if config.append_turn_end else ""


def _instruction(config, path: str) -> str:
    """The filler for one encoder, empty when that path opts out."""
    return config.media_instruction if path in config.instruction_paths else ""


def _video_prompt(processor, suffix: str = "") -> str:
    """Video placeholder, followed by ``suffix`` (a caption or the filler)."""
    return (
        processor.vision_bos_token + processor.video_token + processor.vision_eos_token + suffix
    )


def _audio_prompt(processor, suffix: str = "") -> str:
    """Audio placeholder, followed by ``suffix`` (a caption or the filler)."""
    return processor.audio_bos_token + processor.audio_token + processor.audio_eos_token + suffix


def _frames(paths, config) -> list:
    """Sampled frames of every clip, as ``load_video_frames`` returns them."""
    return [load_video_frames(p, num_frames=config.video_max_frames,
                              resolution=config.video_resolution,
                              center_crop=config.video_center_crop) for p in paths]


def _waveforms(paths, config) -> list:
    """Fixed-duration waveform of every clip."""
    return [load_audio_waveform(p, duration_sec=config.audio_duration_sec,
                                sample_rate=config.audio_sample_rate,
                                crop=config.audio_crop) for p in paths]


def _video_kwargs(config, paths=(), use_audio_in_video: bool = False) -> dict:
    """Processor kwargs: the visual token budget and the clip's sampling rate.

    Training (``training/train.sh``) pins every frame at ``min_pixels =
    max_pixels = 224 * 224`` and gives the processor full frames, so the aspect
    ratio is kept and an 8-frame clip is about 256 tokens. Without a ``size``
    dict the processor uses its own range, which rescales a 224 px crop up to
    336 px (576 tokens) and leaves full-size frames at ~2300 tokens.
    ``pin_video_resolution`` holds frames at ``video_resolution ** 2`` pixels;
    the default 336 px scored best on 300 records. ``max_pixels`` is ignored by
    this processor version; only a ``size`` dict takes effect.

    With ``video_clip_fps`` the processor also gets the clip's real sampling
    rate and, on the ``av`` path, the training ``seconds_per_chunk``.
    """
    kwargs = {}
    if config.pin_video_resolution:
        # min_pixels = max_pixels, as train.sh sets them
        pixels = config.video_resolution * config.video_resolution
        kwargs["size"] = {"shortest_edge": pixels, "longest_edge": pixels}
    if config.video_clip_fps and paths:
        # ADDED downstream: the clip's real sampling rate, as training passes it
        # (data_qwen.py::video_decord), instead of the processor's 2 fps.
        # The transformers processor takes one fps for the whole batch
        fps = {video_sample_fps(p, num_frames=config.video_max_frames) for p in paths}
        if len(fps) > 1:
            raise ValueError("video_clip_fps needs one clip per batch (or clips of one fps).")
        kwargs["fps"] = fps.pop()
        if use_audio_in_video:
            # Training: seconds_per_chunk = 2 * second_per_grid, second_per_grid = 2 / fps
            kwargs["seconds_per_chunk"] = 2.0 * 2.0 / kwargs["fps"]
    return kwargs


def _apply_beats_audio_slots(inputs, processor, backbone, config):
    """Give every audio frame the second token slot the BEATs branch needs.

    ADDED downstream; the released code does not do this. With ``use_beats`` on
    and ``beats_only`` off, the WAVE forward pass interleaves one BEATs vector
    after every whisper vector, so the feature block it scatters into the token
    stream is twice as long as the audio placeholder the processor expands. The
    scatter is a ``masked_scatter``, which consumes as many rows as there are
    slots and silently drops the rest, so half the audio timeline never reaches
    the model. Upstream avoids this by doubling the placeholder after expansion
    (``training/qwenvl/data/data_qwen.py::_prepare_submodal_input``); this is the
    same edit applied to the already-tokenised ids.

    Set ``InferenceConfig.duplicate_audio_tokens = False`` to reproduce the
    released behaviour.
    """
    if not config.duplicate_audio_tokens or not _beats_interleaves(backbone):
        return inputs

    audio_token_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_token)
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    ids_rows, mask_rows = [], []
    for ids, mask in zip(inputs["input_ids"], inputs["attention_mask"]):
        repeats = torch.where(ids == audio_token_id, 2, 1)
        ids_rows.append(torch.repeat_interleave(ids, repeats))
        mask_rows.append(torch.repeat_interleave(mask, repeats))

    # The processor left-pads the text stream and the fusion head pools the
    # final position, so the re-padding has to stay on the left.
    width = max(row.numel() for row in ids_rows)
    inputs["input_ids"] = torch.stack(
        [F.pad(row, (width - row.numel(), 0), value=pad_token_id) for row in ids_rows]
    )
    inputs["attention_mask"] = torch.stack(
        [F.pad(row, (width - row.numel(), 0), value=0) for row in mask_rows]
    )
    return inputs


def _beats_interleaves(backbone) -> bool:
    """True when the forward pass emits two feature rows per audio frame."""
    from omniretriever.models.beats_adaptor import find_beats_host

    try:
        host = find_beats_host(backbone)
    except AttributeError:
        return False
    return bool(getattr(host, "use_beats", False)) and not bool(getattr(host, "beats_only", False))


def _raw_wav(waveforms, backbone) -> list:
    """Waveforms for the BEATs branch, which reads raw audio rather than mel features.

    The remote code iterates ``input_raw_wav`` and feeds each entry to
    ``BEATs.extract_features``, so it must be a list of 1-D tensors, one per
    sample, at the model's audio sample rate. Leaving it out raises a TypeError
    inside the forward pass.
    """
    device = _device(backbone)
    return [torch.as_tensor(w, dtype=torch.float32).reshape(-1).to(device) for w in waveforms]


def _forward_and_normalise(backbone, inputs, config, **forward_kwargs) -> torch.Tensor:
    """Run a single forward pass and return (optionally L2-normalised) embeddings."""
    # ``pred_embeds=True`` is what makes the forward pass return embeddings and
    # stop there. Without it the WAVE-7B remote code falls through to its
    # contrastive training branch, which calls torch.distributed all_gather and
    # fails outside a launched process group.
    outputs = backbone(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
        pred_embeds=True,
        **forward_kwargs,
    )
    # The all-layer fusion head concatenates the last-token hidden state of every
    # layer and projects it through ``classify_linear``. Upstream reads the
    # result off ``mllm_embeds``; ``text_embeds`` is only populated when label or
    # positive-pair inputs are supplied, which the encode_* helpers never do.
    embeddings = getattr(outputs, "mllm_embeds", None)
    if embeddings is None:
        raise RuntimeError(
            "Backbone did not return an 'mllm_embeds' field. Verify that the LoRA "
            "adapter is applied and that the WAVE-7B remote code is at the "
            "version expected by this release."
        )

    if config.normalize:
        embeddings = F.normalize(embeddings, p=2, dim=-1)

    return embeddings


def _as_list(x) -> list:
    if isinstance(x, (str, Path)):
        return [str(x)]
    if isinstance(x, torch.Tensor):
        return [x]
    return [str(item) if isinstance(item, Path) else item for item in x]


def _device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device
