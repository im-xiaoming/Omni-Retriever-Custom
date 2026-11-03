"""Shared setup of the HT-Step LoRA scripts (probe_htstep.py, eval_htstep.py, export_step_emb.py).

``data_args`` mirrors the flags of training/train.sh (8 frames, 50176 pixels, 8 s audio, BEATs, all_layer), and
``load_model`` the model setup of training/qwenvl/train/train_qwen.py for a pretrained adapter with LORA_ONLY.
"""

from __future__ import annotations

import hashlib
import os
import sys
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(REPO_ROOT, "training"), os.path.join(REPO_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
try:
    import deepspeed  # noqa: F401  (import before transformers, as trainer.py does)
except Exception:
    pass

import torch

HEADS = ("classify_linear", "beats_ln", "beats_proj")


def data_args(processor, manifest: str = "", run_test: bool = True):
    return SimpleNamespace(
        dataset_use=manifest, omni_processor=processor, video_max_frames=8, video_min_frames=8, base_interval=0.5,
        max_pixels=50176, min_pixels=50176, image_max_frame_pixels=2073600, image_min_frame_pixels=784,
        fixed_audio_duration=8.0, pred_embeds=True, train_classify=True, classify_type="all_layer", use_beats=True,
        beats_only=False, use_tuple_infonce=False, run_test=run_test, do_sample=False, num_sample=1,
        feature_size=128, chunk_length=30, hop_length=160, sampling_rate=16000)


def load_processor(wave_path: str):
    from qwenvl.data.processing_qwen2_5_omni import Qwen2_5OmniProcessor
    return Qwen2_5OmniProcessor.from_pretrained(wave_path)


def load_model(wave_path: str, beats_path: str, adapter: str, trainable: bool = False,
               attn: str = "sdpa", temperature: float = 0.01):
    """WAVE-7B (bf16) + BEATs + adapter on the GPU. ``trainable``: only lora_A / lora_B require grad (LORA_ONLY)."""
    from peft import PeftModel
    from qwenvl.model.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig
    from qwenvl.model.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration

    cfg = Qwen2_5OmniThinkerConfig.from_pretrained(wave_path)
    if hasattr(cfg, "text_config"):
        for k, d in (("pad_token_id", 151643), ("bos_token_id", 151644), ("eos_token_id", 151645)):
            if getattr(cfg.text_config, k, None) is None:
                setattr(cfg.text_config, k, getattr(cfg, k, d))
    cfg.train_classify, cfg.classify_type, cfg.sim_temperature = True, "all_layer", temperature
    cfg.use_tuple_infonce = False
    cfg.audio_config.beats_path, cfg.audio_config.beats_only = beats_path, False
    os.environ["BEATS_PATH"] = beats_path
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        wave_path, config=cfg, torch_dtype=torch.bfloat16, attn_implementation=attn)
    model.beats.load_state_dict(torch.load(beats_path, map_location="cpu", weights_only=False)["model"])
    model.requires_grad_(False)
    if trainable:
        model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    for name, p in model.named_parameters():
        p.requires_grad_(trainable and ("lora_A" in name or "lora_B" in name))
    return model.cuda()


def trainable_report(model) -> dict:
    groups = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            g = "LoRA" if "lora_" in name else next((h for h in HEADS if h in name), "other")
            groups[g] = groups.get(g, 0) + p.numel()
    return groups


def head_sha(model) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        if any(k in name for k in HEADS):
            h.update(name.encode())
            h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def git_commit() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""
