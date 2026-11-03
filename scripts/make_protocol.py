#!/usr/bin/env python3
"""
make_protocol.py
================
The protocol of the HT-Step LoRA run (OMNI_HTSTEP_LORA_COLAB.md section 4): sha256 of the manifests and the split, the
fixed values of section 2, model / adapter sources and sha256, the git commit, K and the eval fraction of the probe.
Written once by the train notebook (cell 4, default ``configs/omni_htstep_lora_protocol.json``, the notebook puts it
in the run folder on Drive); every training session
checks it first (``check``).

    python scripts/make_protocol.py write --grad-cache-chunk 4 --eval-every-frac 0.25 --data-dir <clips>/meta --out <p>
    python scripts/make_protocol.py check --data-dir <clips>/meta --protocol <p>          # before every train session
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO_ROOT, "configs", "omni_htstep_lora_protocol.json")
DATA = os.path.join(REPO_ROOT, "data", "htstep")
HASHED = ("train_omni_htstep.jsonl", "val_omni_htstep.jsonl", "htstep_split_omni.json", "items_train.jsonl",
          "items_val.jsonl", "dropped_steps.json")
VALUES = {   # OMNI_HTSTEP_LORA_COLAB.md section 2; the agent does not change them
    "site": "Colab Pro: precut on a CPU runtime, train / eval on one A100 (bf16)",
    "method": "LoRA continued from the pt adapter (PeftModel.from_pretrained is_trainable=True); r=16, alpha=32, "
              "dropout 0.05 from adapter_config.json; q/k/v of the 28 LLM layers",
    "trained": "LORA_ONLY=True: lora_A / lora_B only; classify_linear, beats_ln, beats_proj = pt, frozen",
    "precision": "bf16, no quantisation",
    "loss": "pairwise symmetric InfoNCE text <-> audio+video, USE_TUPLE_INFONCE=False, temperature 0.01",
    "batch": "effective 32 with GradCache (--grad_cache_chunk K from the probe)",
    "negatives": "pairs (i, j != i) with the same step text masked in both directions (--mask_same_text True)",
    "augment": "off (video, audio, text)",
    "preprocessing": "8 frames linspace over [start, end], max_pixels = min_pixels = 50176, audio 16 kHz centre-cut / "
                     "zero-padded to 8 s, classify_type all_layer, BEATs on; pre-cut clips (scripts/precut_htstep.py)",
    "validation": "HT-Step val_seen, every step of every val video with an mp4; checkpoint = best t2v R@1 (paper "
                  "protocol, diagonal); no dev",
    "data": "train: every subset == train video with a readable mp4, val: every val_seen video with a readable mp4, "
            "no dev; 6MBctYaMU8U (YouCook2) out; UniAV step rule (Full + Partial, end > start, first 48 in json order); "
            "text f'{activity}: {headline.strip()}' from taxonomy.csv; id f'{video_id}#{step_idx}'; AV1 -> lossless H.264 "
            "before cutting, empty video track out, start past the end out, end past the end clamped",
    "libraries": "decord==0.6.0, av==19.0.2, transformers==4.51.3 (requirements-htstep.txt)",
    "checkpoints": "every 200 optimiser steps and after every eval (LoRA + optimizer + scheduler + RNG + data position), "
                   "last + best (LoRA tensors only, heads = pt)",
    "benchmark": "OmniRetriever-Bench 12 directions, pt once and the best checkpoint once; forgetting track t2av/av2t "
                 "on 1000 fixed records (seed 0), logged only",
    "hyper": {"lr": 1e-4, "schedule": "cosine", "warmup_ratio": 0.03, "weight_decay": 0.01, "optimizer": "AdamW",
              "betas": [0.9, 0.95], "max_epochs": 2, "eval": "every 1/4 epoch (1/2 if one val pass > 15% of an "
              "epoch in the probe)", "early_stop": "3 evals without a better R@1", "seed": 42},
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", REPO_ROOT, *args], capture_output=True, text=True).stdout.strip()


def hf_sha(repo, path):
    from huggingface_hub import HfApi
    f = HfApi().get_paths_info(repo, [path], expand=True)[0]
    return f.lfs.sha256 if f.lfs else f.blob_id


def write(a):
    doc = {
        "values": VALUES, "eval_every_frac": a.eval_every_frac, "grad_cache_chunk": a.grad_cache_chunk,
        "files": {n: sha256(os.path.join(a.data_dir, n)) for n in HASHED},
        "model": {"repo": "nguyenminh04/omni-model", "beats": "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
                  "beats_sha256": hf_sha("nguyenminh04/omni-model", "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")},
        "adapter": {"repo": "nguyenminh04/omni-adapters", "file": "omniretriever-7b/adapter_model.safetensors",
                    "sha256": hf_sha("nguyenminh04/omni-adapters", "omniretriever-7b/adapter_model.safetensors"),
                    "author_repo": "YunzeLiu/OmniRetriever-7B",
                    "author_sha256": hf_sha("YunzeLiu/OmniRetriever-7B", "adapter_model.safetensors")},
        "git": {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD")},
    }
    doc["adapter"]["same_as_author"] = doc["adapter"]["sha256"] == doc["adapter"]["author_sha256"]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    print(f"{a.out}\nsha256 {sha256(a.out)}")


def check(a):
    doc = json.load(open(a.protocol, encoding="utf-8"))
    bad = {n: s for n, s in doc["files"].items() if sha256(os.path.join(a.data_dir, n)) != s}
    head = git("rev-parse", "HEAD")
    if head != doc["git"]["commit"]:
        print(f"note: code at {head[:10]}, protocol written at {doc['git']['commit'][:10]}")
    print(f"CHECK protocol {'FAIL' if bad else 'OK'} sha256 {sha256(a.protocol)[:16]} "
          f"({len(doc['files'])} files{', changed: ' + ', '.join(bad) if bad else ''})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--eval-every-frac", type=float, required=True)
    w.add_argument("--grad-cache-chunk", type=int, required=True)
    w.add_argument("--data-dir", default=DATA)
    w.add_argument("--out", default=OUT)
    c = sub.add_parser("check")
    c.add_argument("--protocol", default=OUT)
    c.add_argument("--data-dir", default=DATA)
    a = p.parse_args()
    write(a) if a.cmd == "write" else check(a)
