#!/usr/bin/env python3
"""
export_step_emb.py
==================
Omni text vectors of the 4958 HT-Step steps in the layout UniAV's ``dataset.omni_emb_file`` reads
(docs/htstep/UNIAV_XU_LY_DU_LIEU.md 9.1; plan 4.6): ``ids`` = global_step_index sorted, ``emb`` (4958, 3584)
float16, raw (no mean removed, not normalised: UniAV L2-normalises itself), ``text`` = ``"<activity>: <headline>"``.

The vector is the text embedding of training and validation: the caption tokenised as the loader does
(``text + "<|im_end|>"``) through the LLM, last-layer hidden state of the last token
(modeling_qwen2_5_omni.py:3049-3058) -- the ``text_embeds`` that the pairwise loss and val R@1 use. Batch 1.

    python scripts/export_step_emb.py --adapter <pt dir> --tag pt --out omni_step_emb_pt.npz
    python scripts/export_step_emb.py --lora <run>/best --pt-adapter <pt dir> --tag best --out omni_step_emb_best.npz
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import htstep_common as hc  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


@torch.no_grad()
def text_embeddings(model, tokenizer, texts):
    thinker = model.get_base_model()
    dev = next(model.parameters()).device
    out = []
    for t in texts:
        ids = tokenizer(t + "<|im_end|>", padding=True, padding_side="left", return_tensors="pt")["input_ids"].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = thinker.model(inputs_embeds=thinker.get_input_embeddings()(ids),
                              attention_mask=torch.ones_like(ids), return_dict=True)[0][:, -1, :]
        out.append(h.float().cpu())
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    env = os.environ.get
    p.add_argument("--wave", default=env("WAVE_PATH"))
    p.add_argument("--beats", default=env("BEATS_PATH"))
    p.add_argument("--adapter", default="")
    p.add_argument("--lora", default="")
    p.add_argument("--pt-adapter", default=env("LORA_CKPT"))
    p.add_argument("--taxonomy", default=env("HTSTEP_TAXONOMY", "D:/Học/KL/Data/Htstep/htstep/data/taxonomy.csv"),
                   help="taxonomy.csv of github.com/facebookresearch/htstep")
    p.add_argument("--tag", default="pt")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    adapter = a.adapter
    if a.lora:
        from qwenvl.train.retrieval_eval import materialize_adapter
        adapter = materialize_adapter(a.pt_adapter, a.lora, os.path.join(os.path.dirname(os.path.abspath(a.out)),
                                                                         f"adapter_{a.tag}"))
    with open(a.taxonomy, newline="", encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["global_step_index"]))
    texts = [f"{r['activity']}: {r['headline'].strip()}" for r in rows]
    processor = hc.load_processor(a.wave)
    model = hc.load_model(a.wave, a.beats, adapter).eval()
    emb = text_embeddings(model, processor.tokenizer, texts)
    np.savez(a.out, ids=np.array([int(r["global_step_index"]) for r in rows]), emb=emb.astype(np.float16),
             text=np.array(texts, dtype=str))
    print(f"{len(texts)} steps, emb {emb.shape} ({a.tag}) -> {a.out}  sha256 {hc.file_sha(a.out)}")


if __name__ == "__main__":
    main()
