#!/usr/bin/env python3
"""
probe_htstep.py
===============
VRAM / speed probe of the HT-Step LoRA run and the choice of the GradCache chunk K (OMNI_HTSTEP_LORA_COLAB.md 6.2,
cell 4), on the training GPU, with the pre-cut clips:

1. memory once WAVE-7B + BEATs + the pt adapter (LORA_ONLY) sit on the GPU;
2. for K = 1, 2, 4, 8: 3 optimiser steps on a batch of 32 with GradCache chunks of K (gradient checkpointing on, bf16,
   same-text mask), ``max_memory_allocated`` / ``max_memory_reserved`` and seconds per step (step 1 = warm-up, not
   timed); OOM is recorded and the next K runs;
3. inference at batch 1 on 50 val items (AV + text, what validation runs): peak memory and ms per item;
4. K = the largest one with ``max_memory_reserved`` < 90 % of the GPU memory; time of one epoch with it, of one val
   pass, the eval rule of the protocol (every 1/2 epoch if one val pass > 15 % of an epoch, else every 1/4), the time
   of the whole run and the number of Colab sessions;
5. ``<out>/chunk.json`` (read by the train cell) and ``<out>/vram_probe.json``.

    python scripts/probe_htstep.py --train-manifest <clips>/meta/train_omni_htstep.jsonl \\
        --val-manifest <clips>/meta/val_omni_htstep.jsonl --clip-root <clips> --out <run dir>
"""

import argparse
import json
import math
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import htstep_common as hc  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

CHUNKS = (1, 2, 4, 8)
BATCH = 32
GB = 2 ** 30


def gb(b) -> float:
    return round(b / GB, 2)


def dataset_of(manifest, processor, n):
    """The first ``n`` records of ``manifest`` as a LazySupervisedDataset (CLIP_ROOT resolves their clips)."""
    from qwenvl.data.data_qwen import LazySupervisedDataset
    ds = LazySupervisedDataset(tokenizer=processor.tokenizer, data_args=hc.data_args(processor, manifest))
    ds.list_data_dict = ds.list_data_dict[:n]
    return ds


def train_chunk(model, items, k, steps, temperature):
    from qwenvl.data.data_qwen import DataCollatorForOmniDataset
    from qwenvl.train.grad_cache import grad_cache_step
    from qwenvl.train.retrieval_eval import _to_device

    model.get_base_model().gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
    collate = DataCollatorForOmniDataset()
    p0 = next(model.parameters())
    prep = lambda b: _to_device(b, p0.device, p0.dtype)          # noqa: E731
    auto = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times = []
    try:
        for s in range(steps):
            batch = [items[(s * BATCH + j) % len(items)] for j in range(BATCH)]
            torch.cuda.synchronize()
            t0 = time.time()
            grad_cache_step(model, batch, collate, prep, k, temperature, True, autocast=auto)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if s:
                times.append(time.time() - t0)
        res = {"status": "ok", "sec_per_step": round(float(np.mean(times)), 2)}
    except torch.cuda.OutOfMemoryError:
        res = {"status": "OOM"}
    res.update(max_allocated_gb=gb(torch.cuda.max_memory_allocated()), max_reserved_gb=gb(torch.cuda.max_memory_reserved()))
    del opt
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return res


@torch.no_grad()
def inference(model, items):
    from qwenvl.data.data_qwen import DataCollatorForOmniDataset
    from qwenvl.train.retrieval_eval import _to_device

    model.eval()
    collate = DataCollatorForOmniDataset()
    p0 = next(model.parameters())
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for it in items:
        b = _to_device(collate([it]), p0.device, p0.dtype)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(**b, output_hidden_states=True, return_dict=True, pred_embeds=True)
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    model.train()
    return {"ms_per_item": round(1000 * float(np.mean(ts[1:])), 1), "max_allocated_gb": gb(torch.cuda.max_memory_allocated()),
            "max_reserved_gb": gb(torch.cuda.max_memory_reserved())}


def plan(results: dict, total_gb: float, n_train: int, n_val: int, load_ms: float, val_ms: float,
         epochs: int = 2, session_h: float = 12.0, workers: int = 4) -> dict:
    """K, eval fraction and time estimates from the probe table. ``load_ms``: loader time per item (CPU); training
    loads with ``workers`` processes (train.sh --dataloader_num_workers 4), validation in the main process."""
    fit = [k for k, r in results.items() if r["status"] == "ok" and r["max_reserved_gb"] < 0.9 * total_gb]
    if not fit:
        return {"grad_cache_chunk": None, "why": "no K fits in 90 % of the GPU memory"}
    k = max(fit)
    steps_epoch = math.ceil(n_train / BATCH)
    sec_step = max(results[k]["sec_per_step"], load_ms * BATCH / workers / 1000)
    epoch_h = sec_step * steps_epoch / 3600
    val_h = n_val * (val_ms + load_ms) / 1000 / 3600
    frac = 0.5 if val_h > 0.15 * epoch_h else 0.25
    total_h = epochs * epoch_h + (epochs / frac + 1) * val_h          # + the pt baseline pass
    return {"grad_cache_chunk": k, "eval_every_frac": frac, "optimizer_steps_per_epoch": steps_epoch,
            "sec_per_step_with_loader": round(sec_step, 2), "epoch_hours": round(epoch_h, 2), "val_pass_hours": round(val_h, 2),
            "val_pass_share_of_epoch": round(val_h / epoch_h, 3), "run_hours_max": round(total_h, 1),
            "colab_sessions": math.ceil(total_h / session_h), "session_hours_assumed": session_h,
            "rule": "K = largest with max_memory_reserved < 90 % of the GPU; eval every 1/2 epoch if one val pass "
                    "> 15 % of an epoch, else every 1/4"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    env = os.environ.get
    p.add_argument("--wave", default=env("WAVE_PATH"))
    p.add_argument("--beats", default=env("BEATS_PATH"))
    p.add_argument("--adapter", default=env("LORA_CKPT"))
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--clip-root", default=env("CLIP_ROOT"))
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--session-hours", type=float, default=12.0)
    p.add_argument("--out", required=True)
    p.add_argument("--attn", default=env("ATTN_IMPL", "sdpa"))
    a = p.parse_args()
    os.environ["CLIP_ROOT"] = a.clip_root
    os.makedirs(a.out, exist_ok=True)

    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / GB
    import peft
    import transformers
    info = {"gpu": props.name, "vram_gb": round(total, 2), "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__, "commit": hc.git_commit(),
            "attn": a.attn, "python": platform.python_version()}
    print(json.dumps(info), flush=True)

    processor = hc.load_processor(a.wave)
    model = hc.load_model(a.wave, a.beats, a.adapter, trainable=True, attn=a.attn)
    after_load = {"allocated_gb": gb(torch.cuda.memory_allocated()), "max_reserved_gb": gb(torch.cuda.max_memory_reserved())}
    groups = hc.trainable_report(model)
    print(f"CHECK lora_only {'OK' if groups == {'LoRA': 6881280} else 'FAIL'} {groups} | after load {after_load}", flush=True)

    t0 = time.time()
    ds = dataset_of(a.train_manifest, processor, BATCH * a.steps)
    items = [x for x in (ds._get_item(i) for i in range(len(ds))) if x is not None]
    load_ms = 1000 * (time.time() - t0) / max(len(ds), 1)
    if len(items) < len(ds):
        sys.exit(f"CHECK clips FAIL: {len(ds) - len(items)} of {len(ds)} train items did not load (CLIP_ROOT {a.clip_root})")
    results = {}
    for k in CHUNKS:
        r = results[k] = train_chunk(model, items, k, a.steps, 0.01)
        print(f"K={k}: {r['status']:4s} alloc {r['max_allocated_gb']:6.2f} GB  reserved {r['max_reserved_gb']:6.2f} GB"
              f"  {r.get('sec_per_step', float('nan')):7.2f} s/step", flush=True)
    vds = dataset_of(a.val_manifest, processor, 50)
    inf = inference(model, [x for x in (vds._get_item(i) for i in range(len(vds))) if x is not None])
    n_train = sum(1 for _ in open(a.train_manifest, encoding="utf-8"))
    n_val = sum(1 for _ in open(a.val_manifest, encoding="utf-8"))
    choice = plan(results, total, n_train, n_val, load_ms, inf["ms_per_item"], session_h=a.session_hours)
    rec = {"info": info, "after_load": after_load, "trainable": groups, "train": results, "eval_batch1": inf,
           "loader_ms_per_item": round(load_ms, 1), "n_train": n_train, "n_val": n_val, **choice}
    with open(os.path.join(a.out, "vram_probe.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print(f"\nVRAM (GB, GPU {total:.1f}): after load {after_load['allocated_gb']} | eval batch 1 {inf['max_allocated_gb']} "
          f"(reserved {inf['max_reserved_gb']}), {inf['ms_per_item']} ms/item")
    print("K  status  alloc GB  reserved GB  s/step")
    for k, r in results.items():
        print(f"{k:<2} {r['status']:6s}  {r['max_allocated_gb']:8.2f}  {r['max_reserved_gb']:11.2f}  "
              f"{r.get('sec_per_step', float('nan')):6.2f}")
    print(json.dumps(choice, indent=1), flush=True)
    if choice["grad_cache_chunk"] is None:
        sys.exit("CHECK chunk FAIL: " + choice["why"])
    with open(os.path.join(a.out, "chunk.json"), "w", encoding="utf-8") as f:
        json.dump({k: rec[k] for k in ("grad_cache_chunk", "eval_every_frac", "epoch_hours", "val_pass_hours",
                                       "run_hours_max", "colab_sessions")} | {"gpu": props.name}, f, indent=1)
    print(f"CHECK chunk OK K={choice['grad_cache_chunk']} eval every {choice['eval_every_frac']} epoch -> "
          f"{os.path.join(a.out, 'chunk.json')}")


if __name__ == "__main__":
    main()
