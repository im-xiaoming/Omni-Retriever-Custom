#!/usr/bin/env python3
"""
download_htstep.py
==================
Downloads the HT-Step mp4s of the split (``data/htstep/items_{train,val}.jsonl``) into ``--video-root/<id>.mp4``
so that training/train.sh reads them like the YouCook2 videos (docs/htstep/TAI_VIDEO.md):

* train: HTTP range read of the member inside its tar of ``nguyenminh04/htstep-videos`` (offset / size from
  ``data/htstep/tar_index.csv``, ``range_get`` of scripts/bench_precut_one_tar.py);
* val: the file of ``nguyenminh04/htstep-val-seen-videos`` (the split already takes that copy for the 162 videos that
  are in both repos);
* ffprobe of every file. AV1 (decord 0.6.0 cannot decode it) is converted to lossless H.264
  (``-c:v libx264 -qp 0``, frames passed through, audio copied); frame count, fps and length must be the same
  before and after, else the video is dropped. A file without video frames is dropped. decord then opens the
  final file and its frame count / fps / length (what data_qwen.video_decord sees) go to the log.

``<video-root>/download_log.jsonl``: one line per video and attempt (status ``ok`` / ``empty_video_track`` /
``transcode_mismatch`` / ``decord_unreadable`` / ``download_error``, why, probe before / after). A rerun skips
the videos whose last line is final (everything but ``download_error``) and whose mp4 is still there, and does not
download again a raw file that already has the right size. ``convert_htstep.py manifest`` reads this log.

    python scripts/download_htstep.py --video-root /content/htstep/videos --n-train-videos 20 --n-val-videos 5

``--n-*-videos 0`` = every video of that split. With n > 0 the first n videos by id are taken among the readable
ones of ``docs/htstep/tai_video/codecs.jsonl`` (h264 / vp9 / AV1, no empty track), the train set with at least one
AV1 video so that the conversion runs.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FINAL = ("ok", "empty_video_track", "transcode_mismatch", "decord_unreadable")
TRANSCODE = ["-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast",
             "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-c:a", "copy"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def pick(videos, codecs, n, need_av1):
    """First n video ids (sorted) among the readable ones; one AV1 first when ``need_av1``. n == 0: all."""
    if not n:
        return sorted(videos)
    ok = lambda v: codecs.get(v, {}).get("video") in ("h264", "vp9", "av1") and bool(codecs[v].get("pix_fmt"))  # noqa: E731
    cand = sorted(v for v in videos if ok(v))
    av1 = [v for v in cand if codecs[v]["video"] == "av1"][:1] if need_av1 else []
    return sorted(av1 + [v for v in cand if v not in av1][:n - len(av1)])


def probe(path):
    """codec / pix_fmt / avg fps / frame count (packets, no decoding) / length of the first video stream."""
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=codec_name,pix_fmt,avg_frame_rate,nb_read_packets,duration", "-show_entries",
                        "format=duration", "-of", "json", path], capture_output=True, text=True, timeout=600)
    j = json.loads(o.stdout or "{}")
    s = (j.get("streams") or [{}])[0]
    return {"codec": s.get("codec_name"), "pix_fmt": s.get("pix_fmt"), "fps": s.get("avg_frame_rate"),
            "frames": int(s.get("nb_read_packets") or 0),
            "duration": float(s.get("duration") or j.get("format", {}).get("duration") or 0), "err": o.stderr[-200:]}


def same_video(a, b) -> bool:
    """Frame count and fps equal, lengths within one frame."""
    num, den = (int(x) for x in (a["fps"] or "0/1").split("/"))
    frame = den / num if num else 0.1
    return a["frames"] == b["frames"] and a["fps"] == b["fps"] and abs(a["duration"] - b["duration"]) <= frame


def decord_info(path):
    from decord import VideoReader
    vr = VideoReader(path, num_threads=1)
    n, fps = len(vr), float(vr.get_avg_fps())
    vr[n // 2]                                             # one frame really decodes
    return {"frames": n, "fps": fps, "length": n / fps}


def fetch(vid, src, idx, raw):
    """Raw mp4 to ``raw`` (kept if it already has the right size). Returns its expected size or None."""
    if "tar" in src:
        from bench_precut_one_tar import range_get
        m = idx[(src["tar"], src["member"])]
        size = int(m["size"])
        if not (os.path.exists(raw) and os.path.getsize(raw) == size):
            range_get(src["tar"], int(m["offset_data"]), size, raw)
        return size
    from huggingface_hub import hf_hub_download
    for k in range(5):
        try:
            p = hf_hub_download(src["repo"], src["file"], repo_type="dataset", local_dir=os.path.dirname(raw) + "/_hf")
            shutil.move(p, raw)
            return os.path.getsize(raw)
        except Exception:
            if k == 4:
                raise
            time.sleep(5 * 2 ** k)


def one(vid, split, src, idx, root):
    out, raw = os.path.join(root, f"{vid}.mp4"), os.path.join(root, "_raw", f"{vid}.mp4")
    row = {"video_id": vid, "split": split, "source": src.get("tar", src.get("file")), "transcoded": False}
    t0 = time.time()
    try:
        fetch(vid, src, idx, raw)
    except Exception as e:
        return {**row, "status": "download_error", "why": f"{type(e).__name__}: {e}"[:300]}
    before = probe(raw)
    row["probe"] = before
    if not before["codec"] or before["frames"] == 0:
        os.remove(raw)
        return {**row, "status": "empty_video_track", "why": f"no video frames ({before['err'].strip()[:120]})"}
    if before["codec"] == "av1":
        tmp = out + ".x264.mp4"
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, *TRANSCODE, tmp], capture_output=True, text=True)
        after = probe(tmp) if r.returncode == 0 else None
        row.update(transcoded=True, probe_after=after)
        if after is None or not same_video(before, after):
            for p in (raw, tmp):
                if os.path.exists(p):
                    os.remove(p)
            return {**row, "status": "transcode_mismatch", "why": (r.stderr[-300:] if r.returncode else
                                                                    f"before {before} after {after}")}
        os.replace(tmp, out)
        os.remove(raw)
    else:
        os.replace(raw, out)
    try:
        row.update(decord_info(out))
    except Exception as e:
        os.remove(out)
        return {**row, "status": "decord_unreadable", "why": f"{type(e).__name__}: {e}"[:300]}
    return {**row, "status": "ok", "bytes": os.path.getsize(out), "seconds": round(time.time() - t0, 1)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video-root", required=True)
    p.add_argument("--n-train-videos", type=int, default=20)
    p.add_argument("--n-val-videos", type=int, default=5)
    p.add_argument("--items-dir", default=os.path.join(REPO_ROOT, "data", "htstep"))
    p.add_argument("--tar-index", default=os.path.join(REPO_ROOT, "data", "htstep", "tar_index.csv"))
    p.add_argument("--codecs", default=os.path.join(REPO_ROOT, "docs", "htstep", "tai_video", "codecs.jsonl"))
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args()
    root = a.video_root
    os.makedirs(os.path.join(root, "_raw"), exist_ok=True)
    codecs = {r["video_id"]: r for r in read_jsonl(a.codecs)}
    idx = {(r["tar_path"], r["member_name"]): r for r in csv.DictReader(open(a.tar_index, encoding="utf-8"))}
    todo = []
    for split, n in (("train", a.n_train_videos), ("val", a.n_val_videos)):
        src = {it["video_id"]: it["source"] for it in read_jsonl(os.path.join(a.items_dir, f"items_{split}.jsonl"))}
        if split == "val":                                 # val: the files of htstep-val-seen-videos
            src = {v: s for v, s in src.items() if "file" in s}
        todo += [(v, split, src[v]) for v in pick(src, codecs, n, need_av1=split == "train")]
    log_path = os.path.join(root, "download_log.jsonl")
    last = {r["video_id"]: r for r in read_jsonl(log_path)} if os.path.exists(log_path) else {}
    skip = {v for v, r in last.items() if r["status"] in FINAL
            and (r["status"] != "ok" or os.path.exists(os.path.join(root, f"{v}.mp4")))}
    run = [t for t in todo if t[0] not in skip]
    print(f"{len(todo)} videos ({sum(t[1] == 'train' for t in todo)} train / {sum(t[1] == 'val' for t in todo)} val), "
          f"{len(todo) - len(run)} already done", flush=True)
    lock = threading.Lock()
    with ThreadPoolExecutor(a.jobs) as ex, open(log_path, "a", encoding="utf-8") as f:
        for i, r in enumerate(ex.map(lambda t: one(*t, idx, root), run), 1):
            with lock:
                f.write(json.dumps(r) + "\n")
                f.flush()
            print(f"[{i}/{len(run)}] {r['video_id']} ({r['split']}): {r['status']}"
                  f"{' (AV1 -> H.264)' if r['transcoded'] else ''} {r.get('why', '')[:120]}", flush=True)
    final = {r["video_id"]: r for r in read_jsonl(log_path)}
    want = {t[0] for t in todo}
    from collections import Counter
    print(json.dumps({"status": Counter(final[v]["status"] for v in want if v in final),
                      "transcoded_ok": sum(final[v]["transcoded"] and final[v]["status"] == "ok" for v in want if v in final),
                      "log": log_path}, indent=1))


if __name__ == "__main__":
    main()
