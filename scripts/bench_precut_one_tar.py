#!/usr/bin/env python3
"""
bench_precut_one_tar.py
=======================
Measures two ways of getting the clips of ONE tar of ``nguyenminh04/htstep-videos`` (docs/htstep/TAI_VIDEO.md
section 3). Both cut with ``omniretriever.data.clip.cut_clip`` (the frames / window / audio of data_qwen.py, the npz of
plan section 3) and write the clips with ``np.savez_compressed`` as scripts/precut_htstep.py does.

* ``a``: download the whole tar (``hf_hub_download``) -> per needed video: copy its bytes out of the tar (offset from
  ``data/htstep/tar_index.csv``) to a temp mp4 -> cut every step -> delete the mp4; delete the tar at the end.
* ``b``: per needed video: HTTP range read of exactly its bytes of the tar -> temp mp4 -> cut -> delete; videos not
  needed are never read.

``--jobs`` worker processes cut (and, in ``b``, download) one video each at a time. A thread samples the bytes under
the work folder every 0.25 s (disk peak). Clips are kept until the end to measure their size, then deleted
(``--keep-clips`` keeps them); mode ``b`` checks that its clip arrays equal mode ``a``'s.

    python -X utf8 scripts/bench_precut_one_tar.py --processor <WAVE-7B> --work <tmp> --out bench_4007.json
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
REPO = "nguyenminh04/htstep-videos"
_IP = None


def _init(processor: str) -> None:
    global _IP
    from transformers import Qwen2VLImageProcessor
    _IP = Qwen2VLImageProcessor.from_pretrained(processor)


def peak_rss_mb() -> float:
    try:
        import resource                                   # Linux: KB
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        import psutil
        return psutil.Process().memory_info().peak_wset / 2 ** 20


def range_get(tar: str, off: int, size: int, out: str, tries: int = 5) -> None:
    """Bytes [off, off + size) of the tar on HF to ``out``; the CDN URL is resolved per call (signed URLs expire)."""
    import requests
    from huggingface_hub import hf_hub_url
    url = hf_hub_url(REPO, tar, repo_type="dataset")
    tok = os.environ.get("HF_TOKEN")
    auth = {"Authorization": f"Bearer {tok}"} if tok else {}
    for k in range(tries):
        try:
            r = requests.head(url, allow_redirects=False, timeout=60, headers=auth)
            loc, hdr = (r.headers["Location"], {}) if r.is_redirect else (url, auth)
            with requests.get(loc, headers={**hdr, "Range": f"bytes={off}-{off + size - 1}"}, stream=True,
                              timeout=120) as g:
                g.raise_for_status()
                assert g.status_code == 206, f"HTTP {g.status_code}, not a range answer"
                with open(out, "wb") as f:
                    for ch in g.iter_content(1 << 22):
                        f.write(ch)
            if os.path.getsize(out) == size:
                return
            raise IOError(f"got {os.path.getsize(out)} bytes, want {size}")
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * 2 ** k)


def copy_member(tar_path: str, off: int, size: int, out: str) -> None:
    with open(tar_path, "rb") as src, open(out, "wb") as dst:
        src.seek(off)
        left = size
        while left:
            b = src.read(min(left, 1 << 24))
            dst.write(b)
            left -= len(b)


def arrays_sha(c: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(c):
        h.update(k.encode())
        h.update(c[k].tobytes())
    return h.hexdigest()[:16]


def one_video(mode: str, vid: str, steps, member: dict, tar: str, tar_path: str, work: str, a_frames: int) -> dict:
    """Fetch one mp4, cut all its steps, delete the mp4. Runs in a worker process."""
    import numpy as np
    from omniretriever.data.clip import PastEnd, cut_clip

    mp4 = os.path.join(work, "mp4", f"{vid}.mp4")
    t0 = time.time()
    try:
        if mode == "a":
            copy_member(tar_path, member["offset"], member["size"], mp4)
        else:
            range_get(tar, member["offset"], member["size"], mp4)
    except Exception as e:
        return {"video_id": vid, "fetch_error": f"{type(e).__name__}: {e}"[:300], "steps": []}
    t_fetch = time.time() - t0
    rows, c0, w0 = [], time.process_time(), time.time()
    for it in steps:
        s0, row = time.time(), {"id": it["id"], "status": "ok"}
        try:
            c = cut_clip(mp4, *it["segment"], _IP, n_frames=a_frames)
            dst = os.path.join(work, "clips", vid, f"{it['step_idx']}.npz")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.savez_compressed(dst, **c)
            row.update(bytes=os.path.getsize(dst), sha=arrays_sha(c), video_length=float(c["video_length"]),
                       frames_shape=list(c["frames"].shape), wav_len=int(len(c["wav"])), audio_ok=bool(c["audio_ok"]))
        except PastEnd as e:
            row.update(status="past_end", why=str(e))
        except Exception as e:
            row.update(status="unreadable", why=f"{type(e).__name__}: {e}"[:300])
        row["seconds"] = round(time.time() - s0, 3)
        rows.append(row)
    out = {"video_id": vid, "mp4_bytes": member["size"], "fetch_seconds": round(t_fetch, 2),
           "cut_seconds": round(time.time() - w0, 2), "cut_cpu_seconds": round(time.process_time() - c0, 2),
           "steps": rows, "peak_rss_mb": round(peak_rss_mb(), 1)}
    os.remove(mp4)
    return out


class DiskPeak(threading.Thread):
    def __init__(self, root: str):
        super().__init__(daemon=True)
        self.root, self.peak, self.stop = root, 0, threading.Event()

    def run(self):
        while not self.stop.is_set():
            tot = 0
            for d, _, fs in os.walk(self.root):
                for f in fs:
                    try:
                        tot += os.path.getsize(os.path.join(d, f))
                    except OSError:
                        pass
            self.peak = max(self.peak, tot)
            time.sleep(0.25)


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q / 100 * len(xs)))], 3) if xs else None


def run_mode(mode: str, tar: str, videos: dict, members: dict, a) -> dict:
    from huggingface_hub import hf_hub_download

    work = os.path.join(a.work, mode)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(os.path.join(work, "mp4"))
    mon = DiskPeak(work)
    mon.start()
    t0 = time.time()
    tar_path, t_dl = "", 0.0
    if mode == "a":
        tar_path = hf_hub_download(REPO, tar, repo_type="dataset", local_dir=os.path.join(work, "dl"))
        t_dl = time.time() - t0
        print(f"[a] tar downloaded: {os.path.getsize(tar_path) / 1e9:.2f} GB in {t_dl:.0f}s", flush=True)
    t1 = time.time()
    res = []
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(a.processor,)) as ex:
        futs = [ex.submit(one_video, mode, v, videos[v], members[v], tar, tar_path, work, a.frames)
                for v in sorted(videos, key=lambda v: -members[v]["size"])]   # big first: shorter tail
        for i, f in enumerate(futs, 1):
            r = f.result()
            res.append(r)
            print(f"[{mode}] {i}/{len(futs)} {r['video_id']}: {len(r['steps'])} steps, fetch {r.get('fetch_seconds')}s, "
                  f"cut {r.get('cut_seconds')}s | {time.time() - t0:.0f}s", flush=True)
    t_cut_wall = time.time() - t1
    if tar_path:
        os.remove(tar_path)
    mon.stop.set()
    mon.join()
    steps = [s for r in res for s in r["steps"]]
    ok = [s for s in steps if s["status"] == "ok"]
    clip_bytes = sum(s["bytes"] for s in ok)
    vids_ok = [r for r in res if "cut_seconds" in r]
    out = {
        "mode": mode, "jobs": a.jobs, "videos": len(res), "steps": len(steps),
        "status": {k: sum(s["status"] == k for s in steps) for k in ("ok", "past_end", "unreadable")},
        "fetch_errors": [r for r in res if "fetch_error" in r],
        "wall_seconds": round(time.time() - t0, 1),
        "tar_download_seconds": round(t_dl, 1),
        "cut_phase_wall_seconds": round(t_cut_wall, 1),
        "fetch_seconds_sum": round(sum(r.get("fetch_seconds", 0) for r in res), 1),
        "mp4_bytes": sum(r["mp4_bytes"] for r in vids_ok),
        "cut_seconds_sum": round(sum(r["cut_seconds"] for r in vids_ok), 1),
        "cut_cpu_seconds_sum": round(sum(r["cut_cpu_seconds"] for r in vids_ok), 1),
        "cut_seconds_per_video": {q: pct([r["cut_seconds"] for r in vids_ok], q) for q in (50, 90, 100)},
        "cut_seconds_per_step": {"mean": round(sum(s["seconds"] for s in steps) / max(len(steps), 1), 3),
                                 **{f"p{q}": pct([s["seconds"] for s in steps], q) for q in (50, 90, 99, 100)}},
        "disk_peak_bytes": mon.peak,
        "clip_bytes": clip_bytes, "clip_bytes_per_step": round(clip_bytes / max(len(ok), 1)),
        "peak_rss_mb_worker_max": max((r.get("peak_rss_mb", 0) for r in res), default=0),
        "per_video": [{k: r[k] for k in r if k != "steps"} for r in res],
        "per_step": steps,
    }
    if not a.keep_clips:
        shutil.rmtree(work, ignore_errors=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tar", default="videos/htstep_videos_4007.tar")
    p.add_argument("--items", nargs="+", default=[os.path.join(REPO_ROOT, "data/htstep/items_train.jsonl"),
                                                  os.path.join(REPO_ROOT, "data/htstep/items_val.jsonl")])
    p.add_argument("--tar-index", default=os.path.join(REPO_ROOT, "data/htstep/tar_index.csv"))
    p.add_argument("--processor", required=True, help="folder with WAVE-7B preprocessor_config.json")
    p.add_argument("--work", required=True)
    p.add_argument("--modes", nargs="+", default=["a", "b"])
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--keep-clips", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="only the first N needed videos (smoke test)")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    videos = defaultdict(list)
    for path in a.items:
        for line in open(path, encoding="utf-8"):
            it = json.loads(line)
            if it["source"].get("tar") == a.tar:
                videos[it["video_id"]].append(it)
    members, all_members = {}, 0
    with open(a.tar_index, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["tar_path"] == a.tar:
                all_members += 1
                if r["video_id"] in videos:
                    members[r["video_id"]] = {"offset": int(r["offset_data"]), "size": int(r["size"])}
    assert set(members) == set(videos), "a needed video is missing from the tar index"
    if a.limit:
        videos = {v: videos[v] for v in sorted(videos)[:a.limit]}
    for v in videos:
        videos[v].sort(key=lambda it: it["step_idx"])
    import psutil
    host = {"platform": platform.platform(), "cpu_logical": os.cpu_count(),
            "cpu_physical": psutil.cpu_count(logical=False), "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1),
            "ram_free_gb": round(psutil.virtual_memory().available / 2 ** 30, 1),
            "disk_free_gb": round(shutil.disk_usage(os.path.dirname(os.path.abspath(a.work)) or ".").free / 2 ** 30, 1)}
    print(json.dumps({"tar": a.tar, "members": all_members, "needed_videos": len(videos),
                      "needed_steps": sum(map(len, videos.values())), "host": host}), flush=True)
    os.makedirs(a.work, exist_ok=True)
    result = {"tar": a.tar, "members": all_members, "needed_videos": len(videos),
              "needed_bytes": sum(m["size"] for m in members.values()),
              "needed_steps": sum(map(len, videos.values())), "host": host, "modes": {}}
    for m in a.modes:
        result["modes"][m] = run_mode(m, a.tar, videos, members, a)
        r = result["modes"][m]
        print(json.dumps({k: v for k, v in r.items() if k not in ("per_video", "per_step")}), flush=True)
    if {"a", "b"} <= set(result["modes"]):
        sa = {s["id"]: s.get("sha") for s in result["modes"]["a"]["per_step"]}
        sb = {s["id"]: s.get("sha") for s in result["modes"]["b"]["per_step"]}
        result["a_equals_b"] = {"steps": len(sa), "differ": sorted(k for k in sa if sa[k] != sb.get(k))}
        print("a_equals_b", json.dumps(result["a_equals_b"]), flush=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
