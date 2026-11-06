#!/usr/bin/env python3
"""
eval_htstep_av.py
=================
Step retrieval on HT-Step ``val_seen`` with the encoders of ``extract_htstep_av.py``, for several adapters in one
pass: every video is fetched and every window decoded once, and each adapter encodes it in turn.

For every step of a ``val_seen`` video (``end > start``):

    av    encode_av(mp4, timestamps=(start, end))        8 frames + audio of the window, from the mp4
    text  encode_text(f"{activity}: {headline}")         the step text of training (scripts/convert_htstep.py)

Scoring, per adapter, over the N steps of all videos (N -> N, the diagonal is the match):

    T->AV / AV->T   R@1/5/10, NDCG@10, MRR, median rank (omniretriever.evaluation.metrics.all_metrics)
    R@1 multi       the top hit counts if it has the same step text (the same step occurs in several videos)

Videos come from ``nguyenminh04/htstep-videos`` (tar range read, see extract_htstep_av.TarLocator) or from
``nguyenminh04/htstep-val-seen-videos`` (``part1/<id>.mp4`` / ``part2/<id>.mp4``, private: needs HF_TOKEN).
Per-video results go to ``<out>/videos/<id>.npz``, so a rerun resumes; ``<out>/metrics.json`` holds the scores.

    python scripts/eval_htstep_av.py --annotations annotations.json --base-model $WAVE_HOME/WAVE-7B \\
        --adapter pt=adapters/omniretriever-7b --adapter ft=adapters/checkpoint-full --out eval_val --device cuda
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_htstep_av import TarLocator  # noqa: E402  (also puts src/ on sys.path)

from omniretriever.data.htstep_io import range_get  # noqa: E402

VAL_REPO = "nguyenminh04/htstep-val-seen-videos"


class VideoSource:
    """mp4 of a video id: the tars of htstep-videos first, else the val_seen video repo."""

    def __init__(self, index_cache):
        from huggingface_hub import HfApi

        self.tars = TarLocator(index_cache)
        files = HfApi().list_repo_files(VAL_REPO, repo_type="dataset")
        self.val = {os.path.basename(f)[:-4]: f for f in files if f.endswith(".mp4")}

    def __contains__(self, video_id):
        return video_id in self.val or video_id in self.tars

    def fetch(self, video_id, out):
        if video_id in self.val:
            from omniretriever.data.htstep_io import hf_get
            hf_get(VAL_REPO, self.val[video_id], out)
        else:
            range_get(*self.tars[video_id], out)


def encode_video(model, backbone, tags, mp4, steps, activity):
    """{tag: (av rows, text rows)} for the steps of one video; a step that fails for one adapter is dropped for all."""
    av = {t: [] for t in tags}
    tx = {t: [] for t in tags}
    kept = []
    for seg_idx, step in steps:
        window = tuple(map(float, step["segment"]))
        text = f"{activity}: {step['headline'].strip()}"
        try:
            got = {}
            for t in tags:
                backbone.set_adapter(t)
                got[t] = (model.encode_av(mp4, timestamps=window).float().reshape(-1).cpu().numpy(),
                          model.encode_text(text).float().reshape(-1).cpu().numpy())
        except Exception as ex:  # one bad window must not lose the video
            print(f"  step {seg_idx} {window} FAIL {type(ex).__name__}: {str(ex)[:120]}", flush=True)
            continue
        for t in tags:
            av[t].append(got[t][0])
            tx[t].append(got[t][1])
        kept.append((seg_idx, step["id"], window, text))
    arrays = {"seg_idx": np.asarray([k[0] for k in kept], np.int64),
              "step_id": np.asarray([k[1] for k in kept], np.int64),
              "segment": np.asarray([k[2] for k in kept], np.float64).reshape(-1, 2),
              "text": np.asarray([k[3] for k in kept], dtype=str)}
    for t in tags:
        arrays[f"av_{t}"] = np.stack(av[t]).astype(np.float16) if kept else np.zeros((0, 3584), np.float16)
        arrays[f"text_{t}"] = np.stack(tx[t]).astype(np.float16) if kept else np.zeros((0, 3584), np.float16)
    return arrays


def multi_r1(sim, labels):
    """Share of queries whose top hit has the same label (same step text)."""
    return float(np.mean(labels[sim.argmax(1)] == labels))


def score(out, tags):
    from omniretriever.evaluation.metrics import all_metrics

    parts = [np.load(os.path.join(out, "videos", f)) for f in sorted(os.listdir(os.path.join(out, "videos")))]
    texts = np.concatenate([p["text"] for p in parts])
    res = {"videos": len(parts), "steps": int(len(texts)), "unique_step_texts": int(len(set(texts.tolist())))}
    for t in tags:
        av = np.concatenate([p[f"av_{t}"] for p in parts]).astype(np.float32)
        tx = np.concatenate([p[f"text_{t}"] for p in parts]).astype(np.float32)
        av /= np.linalg.norm(av, axis=1, keepdims=True)
        tx /= np.linalg.norm(tx, axis=1, keepdims=True)
        sim = tx @ av.T
        res[t] = {"t2av": {**all_metrics(sim), "R@1_multi": multi_r1(sim, texts)},
                  "av2t": {**all_metrics(sim.T), "R@1_multi": multi_r1(sim.T, texts)}}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--subset", default="val_seen")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", action="append", required=True, metavar="TAG=DIR",
                    help="full adapter directory; repeat for several (the first is loaded with the model)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--index-cache", default=None, help="tar index cache (default <out>/_tar_index.csv)")
    ap.add_argument("--work", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many videos (0 = all)")
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()

    adapters = dict(x.split("=", 1) for x in a.adapter)
    tags = list(adapters)
    os.makedirs(os.path.join(a.out, "videos"), exist_ok=True)

    if not a.score_only:
        ann = json.load(open(a.annotations, encoding="utf-8"))
        vids = sorted(v for v, d in ann.items() if d["subset"] == a.subset)
        src = VideoSource(a.index_cache or os.path.join(a.out, "_tar_index.csv"))
        src.tars.prefetch([v for v in vids if v not in src.val])
        have = [v for v in vids if v in src]
        todo = [v for v in have if not os.path.exists(os.path.join(a.out, "videos", f"{v}.npz"))]
        print(f"{a.subset}: {len(vids)} videos, {len(have)} with an mp4, {len(todo)} to do", flush=True)
        with open(os.path.join(a.out, "videos_without_mp4.txt"), "w") as f:
            f.write("".join(v + "\n" for v in vids if v not in src))

        from omniretriever import OmniRetriever
        model = OmniRetriever.from_pretrained(a.base_model, adapters[tags[0]], device=a.device, dtype=a.dtype)
        backbone = model._backbone
        if tags[0] != "default":
            backbone.load_adapter(adapters[tags[0]], adapter_name=tags[0])
        for t in tags[1:]:
            backbone.load_adapter(adapters[t], adapter_name=t)
        backbone.eval()

        work = a.work or os.path.join(a.out, "_tmp")
        os.makedirs(work, exist_ok=True)
        for n, video_id in enumerate(todo):
            if a.limit and n >= a.limit:
                break
            steps = [(k, s) for k, s in enumerate(ann[video_id]["annotations"]) if s["segment"][1] > s["segment"][0]]
            mp4 = os.path.join(work, f"{video_id}.mp4")
            t0 = time.time()
            try:
                src.fetch(video_id, mp4)
                arrays = encode_video(model, backbone, tags, mp4, steps, ann[video_id]["activity"])
            except Exception as ex:
                print(f"{video_id}: FAIL {type(ex).__name__}: {str(ex)[:200]}", flush=True)
                continue
            finally:
                if os.path.exists(mp4):
                    os.remove(mp4)
            tmp = os.path.join(a.out, "videos", f"{video_id}.tmp.npz")
            np.savez(tmp, **arrays)
            os.replace(tmp, os.path.join(a.out, "videos", f"{video_id}.npz"))
            print(f"[{n + 1}/{len(todo)}] {video_id}: {len(arrays['step_id'])}/{len(steps)} steps, "
                  f"{time.time() - t0:.0f} s", flush=True)

    res = score(a.out, tags)
    with open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"{res['videos']} videos, {res['steps']} steps ({res['unique_step_texts']} distinct step texts)")
    print("%-6s %-5s %7s %7s %7s %7s %9s" % ("tag", "dir", "R@1", "R@5", "R@10", "MRR", "R@1multi"))
    for t in tags:
        for d in ("t2av", "av2t"):
            m = res[t][d]
            print("%-6s %-5s %7.4f %7.4f %7.4f %7.4f %9.4f" % (t, d, m["R@1"], m["R@5"], m["R@10"], m["MRR"],
                                                             m["R@1_multi"]))


if __name__ == "__main__":
    main()
