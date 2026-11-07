#!/usr/bin/env python3
"""
eval_bench.py
=============
OmniRetriever-Bench, 12 directions, for the pt adapter or a LoRA-only checkpoint (plan OMNI_HTSTEP_LORA.md 4.5):
the pipeline of GUIDE_BENCHMARK.md (``omniretriever.cli extract`` -> ``scripts/score_embeddings.py``).

    python scripts/eval_bench.py --adapter <pt dir> --tag pt --out <dir>
    python scripts/eval_bench.py --lora <run>/best --pt-adapter <pt dir> --tag best --out <dir>
    python scripts/eval_bench.py --make-track <out>/track_bench1000.jsonl      # forgetting-track manifest

The benchmark here has N records (GUIDE: 3510; the paper's Table 4: 3782): the summary records N, and numbers are
only comparable with the paper at the same N.
"""
import argparse
import json
import os
import random
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT = "<video>\nPlease describe the video."


def make_track(manifest: str, media_root: str, out: str, n: int = 1000, seed: int = 0) -> None:
    """``n`` fixed benchmark records (seed 0) as training-loader records (video + its .wav, no timestamps)."""
    recs = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
    pick = sorted(random.Random(seed).sample(range(len(recs)), min(n, len(recs))))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for i in pick:
            r = recs[i]
            f.write(json.dumps({
                "id": f"bench_{r['id']}", "type": "retrieval",
                "conversations": [{"from": "human", "value": PROMPT}, {"from": "gpt", "value": r["text"]}],
                "video": os.path.join(media_root, r["video"]), "audio": os.path.join(media_root, r["audio"]),
                "text": r["text"]}, ensure_ascii=False) + "\n")
    print(f"{len(pick)} of {len(recs)} benchmark records -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    env = os.environ.get
    p.add_argument("--manifest", default=env("BENCH_MANIFEST", ""))
    p.add_argument("--media-root", default=env("BENCH_MEDIA_ROOT", ""))
    p.add_argument("--base-model", default=env("WAVE_PATH"))
    p.add_argument("--adapter", default="", help="full adapter dir (pt)")
    p.add_argument("--lora", default="", help="LoRA-only checkpoint (<run>/best); needs --pt-adapter")
    p.add_argument("--pt-adapter", default=env("LORA_CKPT"))
    p.add_argument("--tag", default="pt")
    p.add_argument("--out", default="bench")
    p.add_argument("--make-track", default="")
    a = p.parse_args()
    if a.make_track:
        return make_track(a.manifest, a.media_root, a.make_track)

    os.makedirs(a.out, exist_ok=True)
    adapter = a.adapter
    if a.lora:
        # retrieval_eval imports omniretriever (src/) at module level, not only the training package
        sys.path[:0] = [os.path.join(REPO_ROOT, "training"), os.path.join(REPO_ROOT, "src")]
        from qwenvl.train.retrieval_eval import materialize_adapter
        adapter = materialize_adapter(a.pt_adapter, a.lora, os.path.join(a.out, f"adapter_{a.tag}"))
    n = sum(1 for l in open(a.manifest, encoding="utf-8") if l.strip())
    emb, met = os.path.join(a.out, f"embeddings_{a.tag}.npz"), os.path.join(a.out, f"metrics_{a.tag}.json")
    envp = dict(os.environ, PYTHONPATH=os.path.join(REPO_ROOT, "src") + os.pathsep + env("PYTHONPATH", ""))
    # omniretriever's loader finds BEATs only under WAVE_HOME; the train notebook sets BEATS_PATH (= WAVE_HOME/<pt>)
    envp.setdefault("WAVE_HOME", os.path.dirname(env("BEATS_PATH")) if env("BEATS_PATH") else
                    os.path.dirname(a.base_model))
    subprocess.run([sys.executable, "-m", "omniretriever.cli", "extract", a.manifest, "--media-root", a.media_root,
                    "--base-model", a.base_model, "--adapter", adapter, "--output", emb, "--device", "cuda",
                    "--dtype", "bfloat16"], check=True, env=envp)
    subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "score_embeddings.py"), emb, "--output", met],
                   check=True, env=envp)
    summary = {"tag": a.tag, "adapter": adapter, "n_records": n, "paper_n": 3782,
               "comparable_with_paper_table4": n == 3782, "metrics": json.load(open(met, encoding="utf-8"))}
    with open(os.path.join(a.out, f"bench_{a.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"bench {a.tag}: N = {n} (paper 3782) -> {met}")


if __name__ == "__main__":
    main()
