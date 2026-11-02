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
# Adopted from https://github.com/QwenLM/Qwen2.5-VL. The original license is located at 'third-party-license/qwenvl25.txt'.
# Adopted from https://github.com/huggingface/transformers. The original license is located at 'third-party-license/transformers.txt'.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
import random
import copy
from torch.utils.data import DataLoader
from transformers import TrainerCallback

# Configure logging early so that INFO messages are displayed on rank 0
_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if _rank == 0 else logging.WARNING,
    handlers=[logging.StreamHandler(sys.stdout)],
)
if _rank == 0:
    transformers.logging.set_verbosity_info()
else:
    transformers.logging.set_verbosity_warning()

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, WhisperFeatureExtractor
from transformers import AutoConfig
from qwenvl.model.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig
from qwenvl.train.trainer import QwenVLTrainer

try:
    from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False

from tqdm import tqdm
import torch.distributed as dist
import yaml

from qwenvl.data.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from qwenvl.model.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel
from qwenvl.train.utils import KeywordsStoppingCriteria

import time

local_rank = None

def collate_fn(batch):
    if batch[0] is None:
        return None
    if batch[0]["input_raw_wav"] is not None:
        batch[0]["input_raw_wav"] = [batch[0]["input_raw_wav"]]
    return batch[0]

def is_rank_zero():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", None))
    if rank is not None:
        try:
            return int(rank) == 0
        except ValueError:
            pass
    return local_rank == 0 or local_rank is None

def rank0_print(*args, **kwargs):
    if is_rank_zero():
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)

class VramCallback(TrainerCallback):
    """GPU memory (GB) once the model + adapter sit on the GPU (train begin, before any optimiser state) and at every
    log, plus the wall time of every optimiser step; rewritten to <output_dir>/vram.json at each log so a run killed
    by OOM keeps what it measured. Reads counters only: training is unchanged."""

    def __init__(self, output_dir):
        self.path, self.doc, self.t = os.path.join(output_dir, "vram.json"), {"log": [], "step_seconds": []}, None

    @staticmethod
    def _mem():
        gb = lambda b: round(b / 2 ** 30, 3)  # noqa: E731
        return {"allocated_gb": gb(torch.cuda.memory_allocated()), "max_allocated_gb": gb(torch.cuda.max_memory_allocated()),
                "max_reserved_gb": gb(torch.cuda.max_memory_reserved())}

    def _write(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.doc, f, indent=1)

    def on_train_begin(self, args, state, control, **kwargs):
        p = torch.cuda.get_device_properties(0)
        self.doc.update(gpu=p.name, total_gb=round(p.total_memory / 2 ** 30, 2),
                        batch_size=args.per_device_train_batch_size, after_load=self._mem())
        self.t = time.time()
        self._write()

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        self.doc["step_seconds"].append(round(now - self.t, 3))
        self.t = now

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.doc["log"].append({"step": state.global_step, "loss": (logs or {}).get("loss"), **self._mem()})
        self._write()

    def on_train_end(self, args, state, control, **kwargs):
        self.doc["end"] = self._mem()
        self._write()


def apply_liger_kernel_to_qwen2_5_vl(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
) -> None:
    if not HAS_LIGER:
        print("Liger kernel is not installed. Using standard PyTorch implementations.")
        return

    print("Applying Liger kernels to Qwen2.5 model...")

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from qwenvl.model.qwen2_5_omni import modeling_qwen2_5_omni

    if rope:
        modeling_qwen2_5_omni.apply_multimodal_rotary_pos_emb = liger_multimodal_rotary_pos_emb
    if rms_norm:
        modeling_qwen2_5_omni.Qwen2RMSNorm = LigerRMSNorm
    if swiglu:
        modeling_qwen2_5_omni.Qwen2MLP = LigerSwiGLUMLP


def _is_modules_to_save(module) -> bool:
    """True for peft's ModulesToSaveWrapper (a head listed in ``modules_to_save``)."""
    return hasattr(module, "original_module") and hasattr(module, "modules_to_save")


def _set_head_trainable(module, name: str, trainable: bool, under_peft: bool) -> None:
    """Set a head (classify_linear, beats_ln, beats_proj) trainable or frozen.

    Under peft the head is a ModulesToSaveWrapper: the forward pass runs its
    ``modules_to_save`` copy, and ``save_pretrained`` writes only that copy, so
    that is the one to train. ``original_module`` is never used and stays frozen.
    """
    if _is_modules_to_save(module):
        module.original_module.requires_grad_(False)
        module.modules_to_save.requires_grad_(trainable)
        return
    if trainable and under_peft:
        # A plain module under peft trains, but the adapter checkpoint never
        # stores it: the trained weights would be lost at save time.
        raise ValueError(
            f"{name} is not in the adapter's modules_to_save, so its trained weights would "
            "not be saved. Use an adapter that lists it (the released omniretriever-7b "
            "does), or start a fresh LoRA with --lora_ckpt No."
        )
    module.requires_grad_(trainable)


def set_model(model_args, model):
    # Once the pretrained adapter is loaded, ``model`` is a PeftModel, and
    # PeftModel forwards unknown attributes down to the wrapped Thinker. There
    # ``model.model`` resolves to the *whole* Thinker rather than its LLM, so
    # freezing it froze classify_linear / beats_ln / beats_proj as well -- and
    # only parameters named "lora" were switched back on afterwards. Resolve the
    # Thinker explicitly so every call below touches the module it names.
    under_peft = isinstance(model, PeftModel)
    thinker = model.get_base_model() if under_peft else model

    if model_args.tune_mm_vision:
        thinker.visual.requires_grad_(True)
    else:
        thinker.visual.requires_grad_(False)

    if model_args.tune_mm_mlp:
        thinker.visual.merger.requires_grad_(True)
    else:
        thinker.visual.merger.requires_grad_(False)

    if model_args.tune_mm_audio:
        thinker.audio_tower.requires_grad_(True)
    else:
        thinker.audio_tower.requires_grad_(False)

    if model_args.tune_mm_qformer:
        thinker.audio_tower.ln_post.requires_grad_(True)
        thinker.audio_tower.proj.requires_grad_(True)
    else:
        thinker.audio_tower.ln_post.requires_grad_(False)
        thinker.audio_tower.proj.requires_grad_(False)

    if model_args.tune_mm_llm:
        if model_args.use_lora:
            raise Exception("tune_mm_llm is not supported when use_lora is True")
        thinker.model.requires_grad_(True)
        thinker.lm_head.requires_grad_(True)
    else:
        # The LLM only. Its LoRA matrices are frozen here too and switched back
        # on by name in train().
        thinker.model.requires_grad_(False)
        thinker.lm_head.requires_grad_(False)

    # Heads last, so nothing above can re-freeze them.
    if model_args.train_classify:
        _set_head_trainable(thinker.classify_linear, "classify_linear", True, under_peft)
    if model_args.use_beats:
        for name in ("beats_ln", "beats_proj"):
            _set_head_trainable(getattr(thinker, name), name, model_args.tune_beats_proj, under_peft)

def train(attn_implementation="flash_attention_2"):
    global local_rank

    seed = int(os.environ.get("TRAIN_SEED", 2025))   # HT-Step LoRA runs set 42 (OMNI_HTSTEP_LORA.md section 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args, remaining_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    if remaining_args:
        logging.warning(f"Unrecognized arguments passed to parser: {remaining_args}")
    data_args.train_classify = model_args.train_classify
    model_args.temperature = training_args.temperature
    data_args.use_beats = model_args.use_beats
    data_args.beats_only = model_args.beats_only
    data_args.use_tuple_infonce = model_args.use_tuple_infonce

    assert model_args.classify_type in ["last_layer", "all_layer"], f"classify_type {model_args.classify_type} is not supported"

    training_args.remove_unused_columns = False

    apply_liger_kernel_to_qwen2_5_vl()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    rank0_print(f"[1/5] Loading processor & tokenizer from {model_args.model_base}...")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_args.model_base)
    data_args.omni_processor = processor
    tokenizer = processor.tokenizer
    tokenizer.model_max_length = training_args.model_max_length

    rank0_print(f"[2/5] Building dataset and dataloader from {data_args.dataset_use}...")
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    if training_args.grad_cache_chunk > 0:
        if model_args.use_tuple_infonce:
            raise ValueError("grad_cache_chunk > 0 trains the pairwise loss: set use_tuple_infonce False")
        from qwenvl.train.grad_cache import InstanceCollator
        data_module["data_collator"] = InstanceCollator()

    if not data_args.run_test:
        rank0_print(f"[3/5] Loading base model config & weights from {model_args.model_name_or_path}...")
        model_config = Qwen2_5OmniThinkerConfig.from_pretrained(model_args.model_base, cache_dir=training_args.cache_dir,)
        if hasattr(model_config, "text_config"):
            if getattr(model_config.text_config, "pad_token_id", None) is None:
                model_config.text_config.pad_token_id = getattr(model_config, "pad_token_id", 151643)
            if getattr(model_config.text_config, "bos_token_id", None) is None:
                model_config.text_config.bos_token_id = getattr(model_config, "bos_token_id", 151644)
            if getattr(model_config.text_config, "eos_token_id", None) is None:
                model_config.text_config.eos_token_id = getattr(model_config, "eos_token_id", 151645)
        if model_args.train_classify:
            model_config.train_classify = model_args.train_classify
            model_config.classify_type = model_args.classify_type
            model_config.sim_temperature = model_args.temperature
            model_config.use_tuple_infonce = model_args.use_tuple_infonce
            model_config.tuple_aug_weight = model_args.tuple_aug_weight
            model_config.tuple_single_weight = model_args.tuple_single_weight
            model_config.tuple_bi_weight = model_args.tuple_bi_weight
            model_config.hn_audio_only = getattr(model_args, "hn_audio_only", False)

        if model_args.use_beats:
            beats_path = os.environ.get("BEATS_PATH")
            if not beats_path:
                raise ValueError(
                    "use_beats=True requires the BEATS_PATH environment variable "
                    "to point at the BEATs checkpoint (e.g. "
                    "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt)."
                )
            model_config.audio_config.beats_path = beats_path
            model_config.audio_config.beats_only = model_args.beats_only

        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=model_config, cache_dir=training_args.cache_dir, torch_dtype=(torch.bfloat16 if training_args.bf16 else None), attn_implementation=attn_implementation)

        if model_args.use_beats:
            rank0_print(f"[4/5] Loading BEATs audio encoder checkpoint from {model_config.audio_config.beats_path}...")
            beats_ckpt = torch.load(model_config.audio_config.beats_path, map_location='cpu', weights_only=False)
            model.beats.load_state_dict(beats_ckpt['model'])

        model.requires_grad_(False)

        if training_args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            if training_args.gradient_checkpointing_kwargs is None:
                training_args.gradient_checkpointing_kwargs={"use_reentrant": False}
            else:
                training_args.gradient_checkpointing_kwargs["use_reentrant"] = False

        rank0_print("[5/5] Configuring model adapters / LoRA...")
        if model_args.lora_ckpt != "No":
            rank0_print(f"Loading pretrained LoRA checkpoint from {model_args.lora_ckpt}...")
            model = PeftModel.from_pretrained(model, model_args.lora_ckpt, is_trainable=True)

        set_model(model_args, model)

        if model_args.use_lora and model_args.lora_ckpt == "No":
            module_to_save = []
            if model_args.tune_mm_vision:
                module_to_save.append("visual")
            if model_args.tune_mm_mlp:
                module_to_save.append("visual.merger")
            if model_args.tune_mm_audio:
                module_to_save.append("audio")
            if model_args.tune_mm_qformer:
                module_to_save.append("audio_tower.ln_post")
                module_to_save.append("audio_tower.proj")
            if model_args.train_classify:
                module_to_save.append("classify_linear")
            if model_args.use_beats:
                if model_args.tune_beats_proj:
                    module_to_save.append("beats_ln")
                    module_to_save.append("beats_proj")
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=r"model\.layers\.(\d+)\.self_attn\.(q|k|v)_proj",
                lora_dropout=model_args.lora_dropout,
                bias=model_args.lora_bias,
                task_type="CAUSAL_LM",
                modules_to_save=module_to_save,
            )
            model = get_peft_model(model, lora_config)

        for k, v in model.named_parameters():
            if "lora" in k:
                v.requires_grad_(True)

        # A head wrapped by get_peft_model above was set trainable before it was
        # wrapped, so its unused original_module is still trainable. Keep only
        # the modules_to_save copy that forward uses and the adapter stores.
        for module in model.modules():
            if _is_modules_to_save(module):
                module.original_module.requires_grad_(False)

        # ── LoRA-only mode ─────────────────────────────────────────────────
        # Re-freeze everything except lora_A / lora_B so that classify_linear,
        # beats_ln and beats_proj (restored from the pretrained adapter above)
        # are kept frozen and only the LoRA delta weights are updated.
        if model_args.lora_only:
            if model_args.lora_ckpt == "No":
                raise ValueError(
                    "lora_only=True requires lora_ckpt to point at the pretrained "
                    "OmniRetriever adapter so that classify_linear, beats_ln and "
                    "beats_proj can be restored before being frozen."
                )
            if not model_args.use_lora:
                raise ValueError("lora_only=True requires use_lora=True.")

            # Pass 1: freeze everything
            for p in model.parameters():
                p.requires_grad_(False)

            # Pass 2: unfreeze only lora_A and lora_B
            for name, p in model.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    p.requires_grad_(True)

            # ── Sanity assertions ──────────────────────────────────────────
            # Any trainable param whose name does not contain "lora" is a bug.
            unexpected_trainable = [
                name for name, p in model.named_parameters()
                if p.requires_grad and "lora" not in name
            ]
            assert len(unexpected_trainable) == 0, (
                f"[lora_only] Unexpected trainable parameters found (should be empty): "
                f"{unexpected_trainable}"
            )

            # classify_linear, beats_ln, beats_proj must be strictly frozen.
            for head_name in ("classify_linear", "beats_ln", "beats_proj"):
                leaky = [
                    name for name, p in model.named_parameters()
                    if head_name in name and p.requires_grad
                ]
                assert len(leaky) == 0, (
                    f"[lora_only] {head_name} has requires_grad=True but must be "
                    f"frozen in lora_only mode. Offending params: {leaky}"
                )

            rank0_print("[lora_only] Assertions passed: only lora_A/lora_B are trainable.")
        # ───────────────────────────────────────────────────────────────────

        # ── Trainable parameter report (rank-0 only) ───────────────────────
        if is_rank_zero():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            trainable_pct = 100.0 * trainable_params / total_params if total_params > 0 else 0.0
            print("\n" + "=" * 70, flush=True)
            print("  TRAINABLE PARAMETER REPORT", flush=True)
            print("=" * 70, flush=True)
            print(f"  Total parameters    : {total_params:,}", flush=True)
            print(f"  Trainable params    : {trainable_params:,}", flush=True)
            print(f"  Trainable %         : {trainable_pct:.4f}%", flush=True)
            print("-" * 70, flush=True)
            print("  Trainable parameter list:", flush=True)
            trainable_cnt = 0
            for name, p in model.named_parameters():
                if p.requires_grad:
                    print(f"    {name}  {list(p.shape)}", flush=True)
                    trainable_cnt += 1
            print(f"  (Total trainable tensors: {trainable_cnt})", flush=True)
            print("-" * 70, flush=True)
            print("  Trainable parameters by group:", flush=True)
            by_group = {}
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if "lora_" in name:
                    group = "LoRA"
                else:
                    group = next((h for h in ("classify_linear", "beats_ln", "beats_proj") if h in name), "other")
                by_group[group] = by_group.get(group, 0) + p.numel()
            for group, count in by_group.items():
                print(f"    {group:<16} {count:>14,}", flush=True)
            print("=" * 70 + "\n", flush=True)
        
        class LossProgressCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if state.is_world_process_zero and logs and "loss" in logs:
                    msg = (f"[Step {state.global_step}/{state.max_steps}] "
                           f"loss={logs['loss']:.4f} "
                           f"obj1={logs.get('obj1_aug',0):.4f} "
                           f"obj2={logs.get('obj2_single',0):.4f} "
                           f"obj3={logs.get('obj3_bi',0):.4f} "
                           f"grad={logs.get('grad_norm',0):.4f} "
                           f"lr={logs.get('learning_rate',0):.2e}")
                    try:
                        from tqdm import tqdm
                        tqdm.write(msg)
                    except Exception:
                        print(msg, flush=True)

        # skip_deepspeed_load is only meaningful when we're starting from an
        # external lora_ckpt AND have no local checkpoint to resume from. If
        # output_dir already has checkpoints, we must let DeepSpeed load them
        # (otherwise optimizer / RNG state wouldn't be restored).
        _local_ckpts_for_skip = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        # Init-only deliberately starts a fresh optimizer/scheduler. Existing
        # checkpoints may have been created with a different set of trainable
        # parameters (for example before lora_only was enabled).
        _skip_ds_load = (
            model_args.lora_ckpt != "No"
            and (
                getattr(model_args, "lora_init_only", False)
                or len(_local_ckpts_for_skip) == 0
            )
        )
        callbacks = [LossProgressCallback()]
        if torch.cuda.is_available() and is_rank_zero():
            callbacks.append(VramCallback(training_args.output_dir))
        if training_args.eval_manifest:
            from qwenvl.train.retrieval_eval import RetrievalEvalCallback, eval_dataset
            n_train = len(data_module["train_dataset"])
            per_step = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
            every = max(1, int(-(-n_train // per_step) * training_args.eval_every_frac))
            track = (eval_dataset(training_args.track_manifest, data_args, tokenizer)
                     if training_args.track_manifest else None)
            callbacks.append(RetrievalEvalCallback(
                model, eval_dataset(training_args.eval_manifest, data_args, tokenizer), training_args.output_dir,
                every, training_args.early_stop_patience, track, training_args.hf_run_repo, training_args.run_name))
            rank0_print(f"[eval] {training_args.eval_manifest}: every {every} optimiser steps")
        trainer = QwenVLTrainer(
            model=model, processing_class=tokenizer, args=training_args,
            callbacks=callbacks,
            skip_deepspeed_load=_skip_ds_load,
            **data_module,
        )

        # Priority order:
        # (1) If output_dir already has its own checkpoints (e.g. from a prior
        #     crashed/killed run of THIS very job), always resume from the
        #     latest local checkpoint — regardless of how the run was kicked
        #     off (init-only or resume-from-external). The local ckpt already
        #     contains all post-init LoRA training + optimizer + scheduler +
        #     RNG state, so resuming there preserves progress across restarts.
        # (2) Otherwise, if an external lora_ckpt was given with init_only,
        #     start a fresh trainer state from those weights (cosine LR from 0).
        # (3) Otherwise, if an external lora_ckpt was given without init_only,
        #     resume the trainer state from it as well.
        # (4) Pure cold start.
        rank0_print(f"Starting training loop (epochs: {training_args.num_train_epochs}, batch_size: {training_args.per_device_train_batch_size}, grad_accum: {training_args.gradient_accumulation_steps}, lr: {training_args.learning_rate})...")
        local_ckpts = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        if local_ckpts and not getattr(model_args, "lora_init_only", False):
            logging.info(f"Local checkpoint(s) found in {training_args.output_dir} "
                         f"({len(local_ckpts)} ckpts); resuming from the latest.")
            trainer.train(resume_from_checkpoint=True)
        elif model_args.lora_ckpt != "No" and getattr(model_args, "lora_init_only", False):
            # INIT-ONLY: model weights already loaded via PeftModel.from_pretrained.
            # Fresh global_step=0 with a fresh optimizer + cosine schedule.
            logging.info(f"Init-only from {model_args.lora_ckpt} -- starting fresh trainer state.")
            trainer.train()
        elif model_args.lora_ckpt != "No":
            # Resume from external lora_ckpt: model weights already loaded via
            # PeftModel.from_pretrained, skip DeepSpeed model loading but
            # restore trainer state/scheduler/rng.
            logging.info(f"Resuming trainer state from {model_args.lora_ckpt}")
            trainer.train(resume_from_checkpoint=model_args.lora_ckpt)
        else:
            trainer.train()
            
    else:
        model_config = Qwen2_5OmniThinkerConfig.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir,)
        if model_args.train_classify:
            model_config.train_classify = model_args.train_classify
            model_config.classify_type = model_args.classify_type
            model_config.sim_temperature = model_args.temperature
        if model_args.use_beats:
            beats_path = os.environ.get("BEATS_PATH")
            if not beats_path:
                raise ValueError(
                    "use_beats=True requires the BEATS_PATH environment variable "
                    "to point at the BEATs checkpoint (e.g. "
                    "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt)."
                )
            model_config.audio_config.beats_path = beats_path
            model_config.audio_config.beats_only = model_args.beats_only

        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=model_config, cache_dir=training_args.cache_dir, torch_dtype=(torch.bfloat16 if training_args.bf16 else None), attn_implementation=attn_implementation)

        if model_args.use_beats:
            beats_ckpt = torch.load(model_config.audio_config.beats_path, map_location='cpu', weights_only=False)
            model.beats.load_state_dict(beats_ckpt['model'])

        if model_args.lora_ckpt != "No":
            model = PeftModel.from_pretrained(model, model_args.lora_ckpt)
            model = model.merge_and_unload()

        model.eval()
        model.cuda()

        pred_rank = training_args.pred_rank

        os.makedirs(os.path.join(training_args.output_dir, training_args.run_name), exist_ok=True)
        result = []
        test_data = data_module["train_dataset"]
        loader = DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            num_workers=training_args.dataloader_num_workers,
            collate_fn=collate_fn,
        )
        all_time = 0.0
        cnt = 0
        for inputs in tqdm(loader, desc=f"RANK {pred_rank}"):
            if inputs:
                res_i = {
                    "video": inputs.pop("video", None),
                    "image": inputs.pop("image", None),
                    "prompt": inputs.pop("prompt", None),
                    "ref": inputs.pop("ref", None),
                    "audio": inputs.pop("audio", None),
                    "use_audio": inputs.pop("use_audio", False),
                }

                new_inputs = {}
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        new_inputs[k] = v.to(model.device)
                    elif (k == 'video_second_per_grid' or k == 'pos_video_second_per_grid') and v is not None:
                        new_inputs[k] = torch.tensor([v], device=model.device)
                    elif v is None:
                        continue
                    elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                        new_inputs[k] = [it.to(model.device) for it in v]
                    else:
                        new_inputs[k] = v

                inputs = new_inputs
                inputs["types"] = inputs.pop("type")

                if data_args.pred_embeds:
                    inputs["pred_embeds"] = True
                    if "seg_video" not in inputs or inputs["seg_video"] is None:
                        with torch.no_grad():
                            start = time.time()
                            outputs = model(**inputs)
                            end = time.time()
                            print("time: ", end - start)
                            all_time += end - start
                            cnt += 1
                        
                        mllm_embeds = outputs.mllm_embeds
                        text_embeds = outputs.text_embeds

                        res_i["mllm_embeds"] = mllm_embeds.cpu() if mllm_embeds is not None else None
                        res_i["text_embeds"] = text_embeds.cpu() if text_embeds is not None else None
                    
                    else:
                        with torch.no_grad():
                            outputs = model(**inputs)
                        
                        mllm_embeds = outputs.mllm_embeds
                        text_embeds = outputs.text_embeds

                        seg_video = inputs["seg_video"]
                        seg_video_grid_thw = inputs["seg_video_grid_thw"]
                        ori_input_ids = inputs["input_ids"].tolist()
                        start_idx = torch.where(inputs["input_ids"][0] == 151652)[0]
                        end_idx = torch.where(inputs["input_ids"][0] == 151653)[0]

                        all_mllm_embeds = [mllm_embeds]
                        for i in range(seg_video.size(0)):
                            new_ids = ori_input_ids[0][:start_idx + 1] + [151656] * (seg_video[i].size(0) // 4) + ori_input_ids[0][end_idx:]
                            inputs["input_ids"] = torch.tensor([new_ids], dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device)
                            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
                            inputs["pixel_values_videos"] = seg_video[i]
                            inputs["video_grid_thw"] = seg_video_grid_thw[i]

                            with torch.no_grad():
                                outputs = model(**inputs)
                            
                            all_mllm_embeds.append(outputs.mllm_embeds)
                        
                        all_mllm_embeds = F.normalize(torch.cat(all_mllm_embeds, dim=0), dim=-1)
                        text_embeds = F.normalize(text_embeds, dim=-1)
                        t2v_sims = text_embeds @ all_mllm_embeds.t()
                        breakpoint()

                else:
                    with torch.no_grad():
                        outputs = model(**inputs)

                    v2t_sims = outputs.v2t_sims
                    pred_idx = torch.argmax(v2t_sims, dim=1)
                    pred = inputs["all_names"][pred_idx]
                    res_i["pred"] = pred

                result.append(res_i)

        if data_args.pred_embeds:
            torch.save(result, os.path.join(training_args.output_dir, training_args.run_name, f"test_results_rank{pred_rank}.bin"))
            print(os.path.join(training_args.output_dir, training_args.run_name, f"test_results_rank{pred_rank}.bin"))
            print(f"Total Time: {all_time}, Samples: {cnt}, Avg: {all_time / cnt if cnt > 0 else 0}")
        else:
            with open(os.path.join(training_args.output_dir, training_args.run_name, f"test_results_rank{pred_rank}.json"), "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(os.path.join(training_args.output_dir, training_args.run_name, f"test_results_rank{pred_rank}.json"))

        return

if __name__ == "__main__":
    train(attn_implementation=os.environ.get("ATTN_IMPL", "sdpa"))
