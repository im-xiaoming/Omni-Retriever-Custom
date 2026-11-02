# Copyright (2026) Tsinghua University, Tencent Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adopted from https://github.com/bytedance/video-SALMONN-2. The original license is located at 'third-party-license/video-salmonn-2.txt'.
# Adopted from https://github.com/QwenLM/Qwen2.5-VL. The original license is located at 'third-party-license/qwenvl.txt'.


import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    model_base: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    tune_mm_audio: bool = field(default=False)
    tune_mm_qformer: bool = field(default=False)
    use_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    lora_ckpt: str = field(default="No")
    lora_init_only: bool = field(default=False, metadata={"help": "If True, load LoRA + modules_to_save weights from `lora_ckpt` as initialization but DO NOT resume the trainer state (fresh global_step=0, fresh optimizer, fresh cosine LR cycle)."})
    lora_only: bool = field(default=False, metadata={"help": "If True, after loading the pretrained adapter, freeze EVERYTHING except lora_A and lora_B. classify_linear, beats_ln and beats_proj will be restored from the adapter checkpoint but kept frozen. Requires lora_ckpt != 'No' and use_lora=True. When False, those heads are trained too (with train_classify / tune_beats_proj)."})
    train_classify: bool = field(default=False)
    classify_dim: int = field(default=24)
    classify_type: str = field(default="last_layer")
    use_beats: bool = field(default=False)
    beats_only: bool = field(default=False)
    tune_beats_proj: bool = field(default=False)
    use_siglip_loss: bool = field(default=False, metadata={"help": "Use SigLIP sigmoid loss instead of softmax contrastive loss"})
    siglip_bias: float = field(default=-10.0, metadata={"help": "Learnable bias init for SigLIP loss"})
    use_tuple_infonce: bool = field(default=False, metadata={"help": "Use TupleInfoNCE loss for multimodal contrastive learning"})
    tuple_aug_weight: float = field(default=1.0, metadata={"help": "Weight for full-modal augmented contrastive loss (objective 1)"})
    tuple_single_weight: float = field(default=1.0, metadata={"help": "Weight for single-modal to full-modal alignment loss (objective 2)"})
    tuple_bi_weight: float = field(default=1.0, metadata={"help": "Weight for bi-modal to full-modal alignment loss (objective 3)"})
    hn_audio_only: bool = field(default=False, metadata={"help": "If True, the L_T hard-negative cycle is disabled and only audio-shuffle hard-negs are used every step (no text-shuffle, no video-shuffle)."})

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=600)
    video_min_frames: Optional[int] = field(default=1)
    base_interval: float = field(default=0.5)
    max_pixels: int = field(default=176400)
    min_pixels: int = field(default=784)
    image_max_frame_pixels: int = field(default=2073600)
    image_min_frame_pixels: int = field(default=784)
    run_test: bool = field(default=False)
    do_sample: bool = field(default=False)
    num_sample: int = field(default=1)
    feature_size: int = field(default=128)
    chunk_length: int = field(default=30)
    hop_length: int = field(default=160)
    sampling_rate: int = field(default=16000)
    pred_embeds: bool = field(default=False)
    fixed_audio_duration: float = field(default=0, metadata={"help": "Fixed audio duration in seconds for consistent batching. 0=variable length."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    pred_rank: int = field(default=0)
    temperature: float = field(default=0.01)
    use_type_sampler: bool = field(default=False)
    warmup_ratio: float = field(default=0.03)
    past_index: int = field(default=-1)
    # HT-Step LoRA (OMNI_HTSTEP_LORA.md section 4); all off by default, so YouCookII runs are unchanged
    grad_cache_chunk: int = field(default=0, metadata={"help": "> 0: GradCache with chunks of this many items over "
                                  "the per-device batch, pairwise loss computed in the trainer (qwenvl/train/grad_cache.py)"})
    mask_same_text: bool = field(default=True, metadata={"help": "grad_cache only: drop negatives with the same text"})
    eval_manifest: str = field(default="", metadata={"help": "retrieval validation manifest scored during training"})
    eval_every_frac: float = field(default=0.25, metadata={"help": "validate every this fraction of an epoch"})
    early_stop_patience: int = field(default=3, metadata={"help": "stop after this many evals without a better R@1"})
    track_manifest: str = field(default="", metadata={"help": "forgetting track (benchmark subset), scored every epoch"})
    hf_run_repo: str = field(default="", metadata={"help": "HF repo to upload the best LoRA and metrics.json to"})

