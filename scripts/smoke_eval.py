#!/usr/bin/env python3
"""
smoke_eval.py
=============
Eval part of the GPU smoke test (docs/htstep/NHAT_KY_SMOKE.md): load WAVE-7B + BEATs + an adapter (a checkpoint of
the smoke run), encode the first ``--n`` steps of a manifest at batch 1 with the training loader and the
``pred_embeds`` path (text + audio-video in one forward, ``retrieval_eval.encode``), and write peak VRAM, ms per item
and the t2v / v2t scores of those items to ``--out``.

    python scripts/smoke_eval.py --wave $WAVE --beats $BEATS --adapter <run>/checkpoint-20 \\
        --manifest data/htstep/smoke/smoke_val.jsonl --video-root /content/htstep_smoke/videos --out eval.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import htstep_common as hc  # noqa: E402

import torch  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wave", required=True)
    p.add_argument("--beats", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--video-root", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.environ["VIDEO_ROOT"] = a.video_root
    from qwenvl.train.retrieval_eval import encode, eval_dataset, score

    gb = lambda b: round(b / 2 ** 30, 3)  # noqa: E731
    processor = hc.load_processor(a.wave)
    model = hc.load_model(a.wave, a.beats, a.adapter, trainable=False, attn=a.attn)
    after_load = gb(torch.cuda.max_memory_allocated())
    ds = eval_dataset(a.manifest, hc.data_args(processor), processor.tokenizer)
    encode(model, ds, max_items=1)                           # warm-up (kernels, decoder), not timed
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    emb = encode(model, ds, max_items=a.n)
    torch.cuda.synchronize()
    sec = time.time() - t0
    n_ok = len(emb["ids"])
    out = {"adapter": a.adapter, "manifest": a.manifest, "items": a.n, "encoded": n_ok, "failed": emb["failed"],
           "after_load_gb": after_load, "eval_peak_allocated_gb": gb(torch.cuda.max_memory_allocated()),
           "eval_peak_reserved_gb": gb(torch.cuda.max_memory_reserved()),
           "ms_per_item": round(1000 * sec / max(n_ok, 1), 1), "scores": score(emb) if n_ok > 1 else None}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
