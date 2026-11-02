#!/usr/bin/env python3
"""
precut_htstep.py
================
Cuts every HT-Step step of ``items_{train,val}.jsonl`` (scripts/convert_htstep.py split) into a clip file once, so the
GPU machine only downloads small clips (plan OMNI_HTSTEP_LORA.md section 3). Runs on a CPU runtime.

Work unit = one source: a video tar of ``nguyenminh04/htstep-videos`` (all its steps), or one folder of
``nguyenminh04/htstep-val-seen-videos``. For each unit: download -> cut every step (src/omniretriever/data/clip.py,
the same frames / window / audio as data_qwen.video_decord + load_audio_segment) -> pack the clips of the unit into
``shards/clips_<unit>.tar`` (members ``clips/<video_id>/<step_idx>.npz``) -> upload it (``--hf-repo``) -> delete the
mp4s and the clips. ``state/<unit>.json`` marks a finished unit, so a rerun resumes; ``index/<unit>.jsonl`` holds one
line per step: ``{id, status, window, video_length, src_fps, video_sec, bytes}`` with status ``ok`` / ``past_end`` /
``unreadable``. At the end ``precut_index.jsonl`` (all units) is written and uploaded.

    python scripts/precut_htstep.py --items data/htstep/items_train.jsonl data/htstep/items_val.jsonl \\
        --processor /content/WAVE_HOME/WAVE-7B --work /content/precut --hf-repo <team>/htstep-omni-clips --jobs 4

``--units 1000 part1`` limits the run to some units (the probe needs one tar).
"""

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


def unit_of(src: dict) -> str:
    if "tar" in src:                                    # videos/htstep_videos_1000.tar -> 1000
        return os.path.basename(src["tar"])[len("htstep_videos_"):-len(".tar")]
    return src["file"].split("/")[0]                   # part1 / part2


def read_items(paths):
    units = defaultdict(list)
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                it = json.loads(line)
                units[unit_of(it["source"])].append(it)
    return units


def fetch(unit: str, items, mp4_dir: str) -> dict:
    """video id -> local mp4 path of every video of the unit (downloaded now)."""
    from huggingface_hub import hf_hub_download

    os.makedirs(mp4_dir, exist_ok=True)
    src = items[0]["source"]
    out = {}
    if "tar" in src:
        tar = hf_hub_download(src["repo"], src["tar"], repo_type="dataset", local_dir=os.path.join(mp4_dir, "_dl"))
        want = {it["source"]["member"]: it["video_id"] for it in items}
        with tarfile.open(tar) as tf:
            for m in tf.getmembers():
                if m.name in want:
                    vid = want[m.name]
                    m.name = os.path.basename(m.name)
                    tf.extract(m, mp4_dir, filter="data")
                    out[vid] = os.path.join(mp4_dir, m.name)
        os.remove(tar)
    else:
        for f in sorted({it["source"]["file"] for it in items}):
            p = hf_hub_download(src["repo"], f, repo_type="dataset", local_dir=mp4_dir)
            out[os.path.basename(f)[:-4]] = p
    return out


def upload(path: str, path_in_repo: str, repo: str, tries: int = 6) -> None:
    """upload_file with retries: several workers commit to the same repo and HF rejects concurrent commits."""
    from huggingface_hub import HfApi

    for k in range(tries):
        try:
            HfApi().upload_file(path_or_fileobj=path, path_in_repo=path_in_repo, repo_id=repo, repo_type="dataset",
                                commit_message=f"{path_in_repo} (precut_htstep.py)")
            return
        except Exception as e:
            if k == tries - 1:
                raise
            print(f"upload {path_in_repo} failed ({type(e).__name__}), retry {k + 1}", flush=True)
            time.sleep(10 * 2 ** k)


def run_unit(unit: str, items, a) -> dict:
    import numpy as np
    from transformers import Qwen2VLImageProcessor
    from omniretriever.data.clip import PastEnd, cut_clip

    t0 = time.time()
    ip = Qwen2VLImageProcessor.from_pretrained(a.processor)
    mp4_dir = os.path.join(a.work, "mp4", unit)
    clip_root = os.path.join(a.work, "clips_tmp", unit)
    paths = fetch(unit, items, mp4_dir)
    t_dl = time.time() - t0
    rows = []
    for it in sorted(items, key=lambda r: (r["video_id"], r["step_idx"])):
        row = {"id": it["id"], "status": "ok"}
        path = paths.get(it["video_id"])
        try:
            if path is None:
                raise FileNotFoundError("mp4 not in the source")
            c = cut_clip(path, *it["segment"], ip, n_frames=a.frames, min_pixels=a.min_pixels,
                         max_pixels=a.max_pixels, audio_sec=a.audio_sec)
            rel = f"clips/{it['video_id']}/{it['step_idx']}.npz"
            dst = os.path.join(clip_root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.savez_compressed(dst, **c)
            row.update(window=c["window"].tolist(), video_length=float(c["video_length"]),
                       src_fps=float(c["src_fps"]), video_sec=float(c["total_frames"] / c["src_fps"]),
                       audio_ok=bool(c["audio_ok"]), shape=list(c["frames"].shape), bytes=os.path.getsize(dst))
        except PastEnd as e:
            row.update(status="past_end", why=str(e))
        except Exception as e:                          # broken / undecodable file
            row.update(status="unreadable", why=f"{type(e).__name__}: {e}"[:300])
        rows.append(row)
    shard = os.path.join(a.work, "shards", f"clips_{unit}.tar")
    os.makedirs(os.path.dirname(shard), exist_ok=True)
    with tarfile.open(shard, "w") as tf:
        if os.path.isdir(os.path.join(clip_root, "clips")):
            tf.add(os.path.join(clip_root, "clips"), arcname="clips")
    shard_bytes = os.path.getsize(shard)
    if a.hf_repo:
        upload(shard, f"clips/clips_{unit}.tar", a.hf_repo)
        if not a.keep_shards:
            os.remove(shard)
    if not a.keep_clips:
        shutil.rmtree(clip_root, ignore_errors=True)
    shutil.rmtree(mp4_dir, ignore_errors=True)
    os.makedirs(os.path.join(a.work, "index"), exist_ok=True)
    idx_file = os.path.join(a.work, "index", f"{unit}.jsonl")
    with open(idx_file, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if a.hf_repo:                                       # after the shard: an index on HF means the unit is complete
        upload(idx_file, f"index/{unit}.jsonl", a.hf_repo)
    st = {"unit": unit, "items": len(rows), "ok": sum(r["status"] == "ok" for r in rows),
          "past_end": sum(r["status"] == "past_end" for r in rows),
          "unreadable": sum(r["status"] == "unreadable" for r in rows), "shard_bytes": shard_bytes,
          "seconds": round(time.time() - t0, 1), "download_seconds": round(t_dl, 1)}
    os.makedirs(os.path.join(a.work, "state"), exist_ok=True)
    with open(os.path.join(a.work, "state", f"{unit}.json"), "w", encoding="utf-8") as f:
        json.dump(st, f)
    return st


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", nargs="+", required=True)
    p.add_argument("--processor", required=True, help="WAVE-7B folder (its preprocessor_config.json)")
    p.add_argument("--work", required=True)
    p.add_argument("--hf-repo", default="", help="team HF dataset repo (private) for the clip shards")
    p.add_argument("--units", nargs="*", default=[], help="only these units (e.g. 1000 part1)")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--frames", type=int, default=8)              # train.sh --video_min/max_frames 8
    p.add_argument("--min-pixels", type=int, default=50176)      # train.sh --min_pixels / --max_pixels
    p.add_argument("--max-pixels", type=int, default=50176)
    p.add_argument("--audio-sec", type=float, default=8.0)       # train.sh --fixed_audio_duration 8
    p.add_argument("--keep-clips", action="store_true", help="keep the clip files under work/clips_tmp")
    p.add_argument("--keep-shards", action="store_true")
    a = p.parse_args()
    done_hf = set()
    if a.hf_repo:
        from huggingface_hub import HfApi, hf_hub_download
        HfApi().create_repo(a.hf_repo, repo_type="dataset", private=True, exist_ok=True)
        files = HfApi().list_repo_files(a.hf_repo, repo_type="dataset")
        done_hf = {f[len("index/"):-len(".jsonl")] for f in files if f.startswith("index/") and f.endswith(".jsonl")}
        os.makedirs(os.path.join(a.work, "index"), exist_ok=True)
        for u in done_hf:                               # resume on a fresh disk: the index of every finished unit
            if not os.path.exists(os.path.join(a.work, "index", f"{u}.jsonl")):
                shutil.copy(hf_hub_download(a.hf_repo, f"index/{u}.jsonl", repo_type="dataset"),
                            os.path.join(a.work, "index", f"{u}.jsonl"))

    units = read_items(a.items)
    todo = sorted(u for u in units if (not a.units or u in a.units) and u not in done_hf
                  and not os.path.exists(os.path.join(a.work, "state", f"{u}.json")))
    print(f"{len(units)} units, {len(todo)} to cut ({sum(len(units[u]) for u in todo)} steps)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(run_unit, u, units[u], a): u for u in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                st = fu.result()
                print(f"[{i}/{len(todo)}] {st['unit']}: {st['ok']}/{st['items']} ok, past_end {st['past_end']}, "
                      f"unreadable {st['unreadable']}, shard {st['shard_bytes'] / 1e6:.0f} MB, {st['seconds']:.0f}s "
                      f"(download {st['download_seconds']:.0f}s) | {(time.time() - t0) / 60:.1f} min", flush=True)
            except Exception as e:
                print(f"[{i}/{len(todo)}] {futs[fu]} FAILED: {type(e).__name__}: {e}", flush=True)

    idx_dir, rows, done = os.path.join(a.work, "index"), [], 0
    for u in sorted(units):
        f = os.path.join(idx_dir, f"{u}.jsonl")
        if os.path.exists(f):
            done += 1
            rows += [json.loads(l) for l in open(f, encoding="utf-8")]
    states = [json.load(open(os.path.join(a.work, "state", f), encoding="utf-8"))
              for f in sorted(os.listdir(os.path.join(a.work, "state")))] if os.path.isdir(
        os.path.join(a.work, "state")) else []          # this session's units (timings)
    out = os.path.join(a.work, "precut_index.jsonl")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in sorted(rows, key=lambda r: r["id"]):
            f.write(json.dumps(r) + "\n")
    summary = {"units_done": done, "units_total": len(units), "items": len(rows),
               "ok": sum(r["status"] == "ok" for r in rows), "past_end": sum(r["status"] == "past_end" for r in rows),
               "unreadable": sum(r["status"] == "unreadable" for r in rows),
               "clip_gb": round(sum(r.get("bytes", 0) for r in rows) / 1e9, 2),
               "mean_seconds_per_unit_this_session": round(sum(s["seconds"] for s in states) / max(len(states), 1), 1)}
    with open(os.path.join(a.work, "precut_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "units": states}, f, indent=1)
    if a.hf_repo:
        for name in ("precut_index.jsonl", "precut_summary.json"):
            upload(os.path.join(a.work, name), name, a.hf_repo)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
