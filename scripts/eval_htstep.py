#!/usr/bin/env python3
"""
eval_htstep.py
==============
HT-Step ``val_seen`` with the pre-cut clips, for the pt adapter (0-shot) or a LoRA-only checkpoint
(OMNI_HTSTEP_LORA_COLAB.md section 5): the same encoding and scoring as validation during training
(``qwenvl.train.retrieval_eval``: batch 1, N -> N on the diagonal, T->AV and AV->T, R@1/5/10, NDCG@10, MRR, plus R@1
multi-positive = any item with the same step text counts).

    python scripts/eval_htstep.py --manifest <clips>/meta/val_omni_htstep.jsonl --adapter <pt dir> --tag pt --out <dir>
    python scripts/eval_htstep.py --manifest ... --lora <run>/best --pt-adapter <pt dir> --tag best --out <dir>

Writes ``<out>/metrics_htstep_<tag>.json`` and ``<out>/val_embeds_<tag>.npz``.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import htstep_common as hc  # noqa: E402

import torch  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    env = os.environ.get
    p.add_argument("--wave", default=env("WAVE_PATH"))
    p.add_argument("--beats", default=env("BEATS_PATH"))
    p.add_argument("--manifest", required=True)
    p.add_argument("--clip-root", default=env("CLIP_ROOT"))
    p.add_argument("--adapter", default="", help="full adapter dir (pt)")
    p.add_argument("--lora", default="", help="LoRA-only checkpoint (<run>/best); needs --pt-adapter")
    p.add_argument("--pt-adapter", default=env("LORA_CKPT"))
    p.add_argument("--tag", default="pt")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.environ["CLIP_ROOT"] = a.clip_root
    os.makedirs(a.out, exist_ok=True)
    from qwenvl.data.data_qwen import LazySupervisedDataset
    from qwenvl.train.retrieval_eval import encode, materialize_adapter, save_embeddings, score

    adapter = a.adapter or materialize_adapter(a.pt_adapter, a.lora, os.path.join(a.out, f"adapter_{a.tag}"))
    processor = hc.load_processor(a.wave)
    model = hc.load_model(a.wave, a.beats, adapter).eval()
    after_load = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    torch.cuda.reset_peak_memory_stats()
    ds = LazySupervisedDataset(tokenizer=processor.tokenizer, data_args=hc.data_args(processor, a.manifest))
    t0 = time.time()
    emb = encode(model, ds)
    m = {"tag": a.tag, "adapter": adapter, "manifest": a.manifest, "manifest_sha256": hc.file_sha(a.manifest),
         **score(emb), "seconds": round(time.time() - t0, 1),
         "vram_gb": {"after_load": after_load, "eval_peak_allocated": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                     "eval_peak_reserved": round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)},
         "failed_ids": emb["failed"][:50]}
    save_embeddings(emb, os.path.join(a.out, f"val_embeds_{a.tag}.npz"))
    with open(os.path.join(a.out, f"metrics_htstep_{a.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1)
    print(f"HT-Step val_seen ({a.tag}): N = {m['n']}, failed {m['failed']}")
    for d in ("t2v", "v2t"):
        print(f"  {d}: " + "  ".join(f"{k} {m[d][k]:.2f}" for k in ("R@1", "R@5", "R@10", "NDCG@10", "MRR", "R@1_multi")))
    print(f"  VRAM {m['vram_gb']} | {m['seconds']:.0f} s -> {a.out}")
    if m["failed"]:
        sys.exit(f"CHECK val_items FAIL: {m['failed']} items did not load, e.g. {m['failed_ids'][:5]}")


if __name__ == "__main__":
    main()
