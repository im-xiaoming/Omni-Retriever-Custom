#!/usr/bin/env python3
"""
probe_htstep.py
===============
VRAM / speed probe of the HT-Step LoRA run (plan OMNI_HTSTEP_LORA.md section 6), on one GPU (L4 / A100):

1. clip equivalence: the AV embedding of the pt model from the pre-cut clip vs from the mp4 + ``timestamps``,
   for ``--equiv-items`` items; every cosine must be >= 0.999 (else the script stops), plus the mean pixel / audio
   difference of the model inputs;
2. train steps: for every configuration of the table (GradCache on/off x gradient checkpointing x chunk K),
   5 optimiser steps after one warm-up step, ``max_memory_allocated`` / ``max_memory_reserved`` and seconds per
   step; OOM is recorded and the next configuration runs;
3. inference at batch 1 on 50 items: peak VRAM and ms per item for AV alone and AV + text;
4. estimates: optimiser steps per epoch, time of one / two epochs, time of one validation pass;
5. ``<out>/vram_probe_<gpu>.json`` + a printed table.

The train / val manifests are those of ``convert_htstep.py finalize`` (a partial precut is enough: the probe only
reads the items whose clips exist under ``$CLIP_ROOT``); ``--mp4-root`` holds the mp4s of the equivalence items.
"""

import argparse
import copy
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import htstep_common as hc  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

CONFIGS = ([("gc", True, k) for k in (1, 2, 4, 8)] + [("no_gc", True, k) for k in (1, 2, 4)]
           + [("gc", False, k) for k in (1, 2)])          # (mode, gradient checkpointing, chunk / batch K)
BATCH = 32


def items_with_clips(manifest: str, clip_root: str):
    out = []
    for line in open(manifest, encoding="utf-8"):
        r = json.loads(line)
        if os.path.exists(os.path.join(clip_root, r["clip"])):
            out.append(r)
    return out


def dataset_of(records, processor, tokenizer, tmp: str, name: str, drop_clip: bool = False):
    from qwenvl.data.data_qwen import LazySupervisedDataset
    path = os.path.join(tmp, f"{name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            r = dict(r)
            if drop_clip:
                r.pop("clip")
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return LazySupervisedDataset(tokenizer=tokenizer, data_args=hc.data_args(processor, path, run_test=True))


def equivalence(model, processor, tok, records, tmp, mp4_root, n):
    from qwenvl.train.retrieval_eval import encode
    recs = [r for r in records if os.path.exists(os.path.join(mp4_root, r["video"]))][:n]
    if len(recs) < n:
        sys.exit(f"CHECK clip_equivalence FAIL: only {len(recs)} items have their mp4 under {mp4_root}")
    os.environ["VIDEO_ROOT"] = mp4_root
    ds_clip = dataset_of(recs, processor, tok, tmp, "eq_clip")
    ds_mp4 = dataset_of(recs, processor, tok, tmp, "eq_mp4", drop_clip=True)
    pix, aud = [], []
    for i in range(len(recs)):
        a, b = ds_clip._get_item(i), ds_mp4._get_item(i)
        if a is None or b is None:
            sys.exit(f"CHECK clip_equivalence FAIL: item {recs[i]['id']} did not load (clip {a is not None}, "
                     f"mp4 {b is not None})")
        if a["pixel_values_videos"].shape != b["pixel_values_videos"].shape:
            sys.exit(f"CHECK clip_equivalence FAIL: {recs[i]['id']} pixel shapes {tuple(a['pixel_values_videos'].shape)}"
                     f" vs {tuple(b['pixel_values_videos'].shape)}")
        pix.append(float((a["pixel_values_videos"] - b["pixel_values_videos"]).abs().mean()))
        aud.append(float((a["input_raw_wav"] - b["input_raw_wav"]).abs().mean()))
    ea, eb = encode(model, ds_clip), encode(model, ds_mp4)
    cos = (ea["av"] * eb["av"]).sum(1)
    cos_t = (ea["text"] * eb["text"]).sum(1)
    res = {"n": len(recs), "cos_av_min": float(cos.min()), "cos_av_mean": float(cos.mean()),
           "cos_text_min": float(cos_t.min()), "pixel_abs_diff_mean": float(np.mean(pix)),
           "pixel_abs_diff_max": float(np.max(pix)), "wav_abs_diff_mean": float(np.mean(aud)),
           "ids_below": [recs[i]["id"] for i in np.flatnonzero(cos < 0.999)]}
    ok = res["cos_av_min"] >= 0.999
    print(f"CHECK clip_equivalence {'OK' if ok else 'FAIL'} {json.dumps(res)}", flush=True)
    if not ok:
        sys.exit(1)
    return res


def train_config(model, items, mode, ckpt, k, steps, temperature):
    from qwenvl.data.data_qwen import DataCollatorForOmniDataset
    from qwenvl.train.grad_cache import full_batch_step, grad_cache_step
    from qwenvl.train.retrieval_eval import _to_device

    thinker = model.get_base_model()
    if ckpt:
        thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        thinker.gradient_checkpointing_disable()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
    collate = DataCollatorForOmniDataset()
    p0 = next(model.parameters())
    prep = lambda b: _to_device(b, p0.device, p0.dtype)       # noqa: E731
    auto = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
    bs = BATCH if mode == "gc" else k
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times, losses = [], []
    try:
        for s in range(steps + 1):
            batch = [items[(s * bs + j) % len(items)] for j in range(bs)]
            torch.cuda.synchronize()
            t0 = time.time()
            if mode == "gc":
                out = grad_cache_step(model, batch, collate, prep, k, temperature, True, autocast=auto)
            else:
                out = full_batch_step(model, batch, collate, prep, temperature, True, autocast=auto)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if s:                                            # step 0 is the warm-up
                times.append(time.time() - t0)
            losses.append(float(out["loss"]))
        res = {"status": "ok", "sec_per_step": round(float(np.mean(times)), 3), "loss": losses,
               "items_per_step": bs}
    except torch.cuda.OutOfMemoryError:
        res = {"status": "OOM", "items_per_step": bs}
    res.update(max_allocated_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
               max_reserved_gb=round(torch.cuda.max_memory_reserved() / 2 ** 30, 2))
    del opt
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return res


@torch.no_grad()
def inference(model, items, n=50):
    from qwenvl.data.data_qwen import DataCollatorForOmniDataset
    from qwenvl.train.retrieval_eval import _to_device

    model.eval()
    collate = DataCollatorForOmniDataset()
    p0 = next(model.parameters())
    out = {}
    for name, drop_text in (("av", True), ("av_text", False)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for it in items[:n]:
            b = collate([it])
            if drop_text:
                b["label_ids"], b["label_attention_mask"] = None, None
            b = _to_device(b, p0.device, p0.dtype)
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(**b, output_hidden_states=True, return_dict=True, pred_embeds=True)
            torch.cuda.synchronize()
            ts.append(time.time() - t0)
        out[name] = {"ms_per_item": round(1000 * float(np.mean(ts[1:])), 1),
                     "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
    out["text_ms_per_item"] = round(out["av_text"]["ms_per_item"] - out["av"]["ms_per_item"], 1)
    model.train()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    env = os.environ.get
    p.add_argument("--wave", default=env("WAVE_PATH"))
    p.add_argument("--beats", default=env("BEATS_PATH"))
    p.add_argument("--adapter", default=env("LORA_CKPT"))
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--clip-root", default=env("CLIP_ROOT"))
    p.add_argument("--mp4-root", required=True, help="mp4s of the equivalence items")
    p.add_argument("--n-train", type=int, default=0, help="train steps of the full run (default: count the manifest)")
    p.add_argument("--n-val", type=int, default=0)
    p.add_argument("--equiv-items", type=int, default=20)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--out", default="probe")
    p.add_argument("--attn", default=env("ATTN_IMPL", "sdpa"))
    a = p.parse_args()
    os.environ["CLIP_ROOT"] = a.clip_root
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "tmp")
    os.makedirs(tmp, exist_ok=True)

    gpu = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    if props.total_memory < 20 * 2 ** 30 or not torch.cuda.is_bf16_supported():
        sys.exit(f"{gpu}: không nạp được bf16, dùng L4 / A100")
    import peft
    import transformers
    info = {"gpu": gpu, "vram_gb": round(props.total_memory / 2 ** 30, 2), "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__, "commit": hc.git_commit(),
            "attn": a.attn, "python": platform.python_version()}
    print(json.dumps(info), flush=True)

    processor = hc.load_processor(a.wave)
    tok = processor.tokenizer
    model = hc.load_model(a.wave, a.beats, a.adapter, trainable=True, attn=a.attn)
    groups = hc.trainable_report(model)
    print(f"CHECK lora_only {'OK' if set(groups) == {'LoRA'} and groups['LoRA'] == 6881280 else 'FAIL'} {groups}",
          flush=True)
    sha0 = hc.head_sha(model)

    train_recs = items_with_clips(a.train_manifest, a.clip_root)
    val_recs = items_with_clips(a.val_manifest, a.clip_root)
    print(f"{len(train_recs)} train / {len(val_recs)} val items have clips", flush=True)
    eq = equivalence(model, processor, tok, train_recs + val_recs, tmp, a.mp4_root, a.equiv_items)

    ds = dataset_of(train_recs[:BATCH * (a.steps + 1)], processor, tok, tmp, "probe_train")
    items = [x for x in (ds._get_item(i) for i in range(len(ds))) if x is not None]
    results = {}
    for mode, ckpt, k in CONFIGS:
        name = f"{mode}_ckpt{'on' if ckpt else 'off'}_k{k}"
        r = train_config(model, items, mode, ckpt, k, a.steps, 0.01)
        results[name] = r
        print(f"{name:22s} {r['status']:4s} alloc {r['max_allocated_gb']:6.2f} GB  reserved "
              f"{r['max_reserved_gb']:6.2f} GB  {r.get('sec_per_step', float('nan')):7.2f} s/step", flush=True)
    sha1 = hc.head_sha(model)
    print(f"CHECK heads_unchanged {'OK' if sha0 == sha1 else 'FAIL'} {sha0[:16]} -> {sha1[:16]}", flush=True)
    inf = inference(model, items)

    n_train = a.n_train or sum(1 for _ in open(a.train_manifest, encoding="utf-8"))
    n_val = a.n_val or sum(1 for _ in open(a.val_manifest, encoding="utf-8"))
    steps_epoch = -(-n_train // BATCH)
    val_min = n_val * inf["av_text"]["ms_per_item"] / 60000
    est = {}
    for name, r in results.items():
        if r["status"] != "ok":
            continue
        per_epoch = r["sec_per_step"] * steps_epoch * BATCH / r["items_per_step"] / 3600
        est[name] = {"epoch_h": round(per_epoch, 2), "two_epochs_h": round(2 * per_epoch, 2),
                     "val_eval_share_of_epoch": round(val_min / 60 / per_epoch, 3)}
    rec = {"info": info, "trainable": groups, "clip_equivalence": eq, "train": results, "inference_batch1": inf,
           "n_train_steps": n_train, "n_val_steps": n_val, "optimizer_steps_per_epoch": steps_epoch,
           "val_eval_minutes": round(val_min, 1), "estimates": est,
           "eval_every_rule": "eval every 1/2 epoch if one val pass > 15% of an epoch (of the configuration used), "
                              "else every 1/4 epoch",
           "heads_sha_before_after": [sha0, sha1]}
    out = os.path.join(a.out, f"vram_probe_{gpu.replace(' ', '_').replace('/', '_')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print("\nconfig                  status  alloc GB  reserved GB  s/step  epoch h  2 epochs h  val/epoch")
    for name, r in results.items():
        e = est.get(name, {})
        print(f"{name:22s}  {r['status']:5s}  {r['max_allocated_gb']:8.2f}  {r['max_reserved_gb']:11.2f}  "
              f"{r.get('sec_per_step', float('nan')):6.2f}  {e.get('epoch_h', float('nan')):7.2f}  "
              f"{e.get('two_epochs_h', float('nan')):10.2f}  {e.get('val_eval_share_of_epoch', float('nan')):9.3f}")
    print(f"inference batch 1: {json.dumps(inf)}\nval pass {val_min:.1f} min ({n_val} items) -> {out}", flush=True)


if __name__ == "__main__":
    main()
