#!/usr/bin/env python3
"""
extract_htstep_av.py
====================
Joint audio-video embedding (``OmniRetriever.encode_av``) of every annotated step segment of HT-Step videos.

Inputs are only ``--ids``, ``--annotations`` and the HF dataset ``nguyenminh04/htstep-videos``: no file derived
earlier (tar_index.csv, items_*.jsonl, ...) is read. Where an mp4 sits is found fresh: the tar's headers are read by HTTP range
(512 bytes each), cached per tar in ``<out>/_tar_index.csv``. ``manifest.json`` of the repo picks the tar to scan,
and the tars it does not list (it is stale) are scanned for the videos it does not name.

For each video id of ``--ids`` (one per line), in file order:

    mp4 by HTTP range read from its tar
    -> ``encode_av(mp4, timestamps=segment)`` for every step of ``annotations.json`` with ``end > start``: 8 frames
    and the audio of that window, both taken from the mp4 -> ``<out>/<video_id>.npz`` -> the mp4 is deleted.

Each ``.npz`` holds one row per step, in ``annotations.json`` order:

    emb       (n, 3584) float16   L2-normalised encode_av vector
    step_id   (n,) int64          the step ``id`` (global_step_index of taxonomy.csv)
    seg_idx   (n,) int64          position of the step in the video's ``annotations`` list
    segment   (n, 2) float64      [start, end] in seconds, as annotated
    partial   (n,) str            "Full Match" / "Partial Match"

Steps that fail (window past the end of the file, no frames, ...) are left out of the ``.npz`` and listed in
``<out>/_failed_steps.jsonl``; videos not in annotations.json or in no tar of the repo go to
``<out>/_missing_videos.txt``. A video whose
``.npz`` already exists is skipped, so a rerun resumes.

    python scripts/extract_htstep_av.py --ids ../../UniAV/extracted_video_ids.txt \\
        --annotations ../../UniAV/data/htstep/data/annotations.json \\
        --base-model $WAVE_HOME/WAVE-7B --adapter ../adapters/omniretriever-7b --out ../output/htstep_av \\
        --device cuda --limit 1
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from omniretriever.data.htstep_io import VIDEOS_REPO, range_get  # noqa: E402

INDEX_FIELDS = ["tar_path", "member_name", "video_id", "offset_data", "size"]


class RangeReader:
    """Byte ranges of one tar of ``repo``. The HF URL redirects to a signed CDN URL; it is resolved once per tar
    (HEAD) and reused through one HTTP session, instead of one redirect per 512-byte header."""

    def __init__(self, tar, repo=VIDEOS_REPO):
        import requests
        from huggingface_hub import hf_hub_url

        self.session = requests.Session()
        url = hf_hub_url(repo, tar, repo_type="dataset")
        r = self.session.head(url, allow_redirects=False, timeout=60)
        self.url = r.headers["Location"] if r.is_redirect else url

    def __call__(self, off, size):
        r = self.session.get(self.url, headers={"Range": f"bytes={off}-{off + size - 1}"}, timeout=60)
        r.raise_for_status()
        if r.status_code != 206:
            raise IOError(f"HTTP {r.status_code}, not a range answer")
        return r.content


def scan_tar(tar, repo=VIDEOS_REPO):
    """Rows of every regular .mp4 member of ``tar``, read from its 512-byte headers (no data downloaded)."""
    read = RangeReader(tar, repo)
    rows, off = [], 0
    while True:
        h = read(off, 512)
        if len(h) < 512 or h == bytes(512):           # end-of-archive block
            return rows
        name = h[0:100].rstrip(b"\0").decode("utf-8")
        size = int(h[124:136].rstrip(b"\0 ").decode() or "0", 8)
        if h[156:157] in (b"0", b"\0") and name.endswith(".mp4"):
            rows.append({"tar_path": tar, "member_name": name, "video_id": os.path.basename(name)[:-4],
                         "offset_data": off + 512, "size": size})
        off += 512 + (size + 511) // 512 * 512


class TarLocator:
    """video id -> (tar, offset, size) in the repo.

    ``manifest.json`` says which tar holds a video, but it is stale (95 of the 113 tars on 2026-11-06), so the tar
    list comes from the repo itself: a video the manifest does not name is looked for in the tars the manifest does
    not list. A tar's headers are scanned once and cached in ``cache_path``; cached rows are trusted as they are.
    """

    def __init__(self, cache_path, repo=VIDEOS_REPO):
        from huggingface_hub import HfApi, hf_hub_download

        self.repo, self.cache_path = repo, cache_path
        manifest = json.load(open(hf_hub_download(repo, "manifest.json", repo_type="dataset"), encoding="utf-8"))
        self.tar_of = {v: tar for tar, shard in manifest["shards"].items() for v in shard["videos"]}
        tars = {f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".tar")}
        self.unlisted = sorted(tars - set(manifest["shards"]))
        self.rows, self.scanned = {}, set()
        if os.path.exists(cache_path):
            with open(cache_path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    self._add(r["tar_path"], r)
        print(f"{len(tars)} tars in {repo} ({len(self.unlisted)} not in manifest.json), "
              f"{len(self.scanned)} already scanned", flush=True)

    def _add(self, tar, r):
        self.rows[r["video_id"]] = (tar, int(r["offset_data"]), int(r["size"]))
        self.tar_of[r["video_id"]] = tar
        self.scanned.add(tar)

    def _store(self, tar, rows, seconds):
        new = not os.path.exists(self.cache_path)
        with open(self.cache_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
        for r in rows:
            self._add(tar, r)
        self.scanned.add(tar)
        print(f"  scanned {tar}: {len(rows)} mp4 in {seconds:.0f} s", flush=True)

    def _scan(self, tar):
        t = time.time()
        self._store(tar, scan_tar(tar, self.repo), time.time() - t)

    def prefetch(self, video_ids, workers=16):
        """Scan, in parallel, every tar these videos need (the manifest's tars and, for the videos it does not
        name, the unlisted ones) that is not cached yet."""
        from concurrent.futures import ThreadPoolExecutor

        need = {self.tar_of[v] for v in video_ids if v in self.tar_of and v not in self.rows}
        if any(v not in self.tar_of for v in video_ids):
            need |= set(self.unlisted)
        need = sorted(need - self.scanned)
        if not need:
            return
        t = time.time()
        with ThreadPoolExecutor(min(workers, len(need))) as ex:
            for tar, rows in zip(need, ex.map(lambda x: scan_tar(x, self.repo), need)):
                self._store(tar, rows, time.time() - t)

    def __contains__(self, video_id):
        if video_id not in self.tar_of:
            for tar in self.unlisted:
                if tar not in self.scanned:
                    self._scan(tar)
        return video_id in self.tar_of

    def __getitem__(self, video_id):
        if video_id not in self.rows:
            self._scan(self.tar_of[video_id])
        return self.rows[video_id]


def append_line(path, line):
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def extract_video(model, mp4, steps, video_id, failed_log):
    """encode_av of every step window of one video -> dict of arrays (rows of the steps that worked)."""
    rows = []
    for seg_idx, step in steps:
        start, end = map(float, step["segment"])
        t = time.time()
        try:
            z = model.encode_av(mp4, timestamps=(start, end))
        except Exception as ex:  # one bad window must not lose the rest of the video
            append_line(failed_log, json.dumps({"video_id": video_id, "seg_idx": seg_idx, "step_id": step["id"],
                                                "segment": [start, end], "error": f"{type(ex).__name__}: {ex}"[:300]}))
            print(f"  step {seg_idx} [{start:.1f}, {end:.1f}] FAIL {type(ex).__name__}: {str(ex)[:120]}", flush=True)
            continue
        rows.append((z.float().reshape(-1).cpu().numpy(), step["id"], seg_idx, (start, end), step.get("partial", "")))
        print(f"  step {seg_idx} [{start:.1f}, {end:.1f}] ok {time.time() - t:.0f} s", flush=True)
    return {
        "emb": np.stack([r[0] for r in rows]).astype(np.float16) if rows else np.zeros((0, 3584), np.float16),
        "step_id": np.asarray([r[1] for r in rows], np.int64),
        "seg_idx": np.asarray([r[2] for r in rows], np.int64),
        "segment": np.asarray([r[3] for r in rows], np.float64).reshape(-1, 2),
        "partial": np.asarray([r[4] for r in rows], dtype=str),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", required=True, help="text file, one video id per line")
    ap.add_argument("--annotations", required=True, help="HT-Step annotations.json")
    ap.add_argument("--base-model", required=True, help="WAVE-7B directory (rope-patched)")
    ap.add_argument("--adapter", required=True, help="full adapter directory")
    ap.add_argument("--out", required=True, help="output directory, one .npz per video")
    ap.add_argument("--work", default=None, help="where the temporary mp4 goes (default <out>/_tmp)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16", "float16"))
    ap.add_argument("--limit", type=int, default=0, help="stop after this many extracted videos (0 = all)")
    a = ap.parse_args()

    ids = [line.strip() for line in open(a.ids, encoding="utf-8") if line.strip()]
    ann = json.load(open(a.annotations, encoding="utf-8"))
    work = a.work or os.path.join(a.out, "_tmp")
    os.makedirs(work, exist_ok=True)
    index = TarLocator(os.path.join(a.out, "_tar_index.csv"))
    missing_log, failed_log = os.path.join(a.out, "_missing_videos.txt"), os.path.join(a.out, "_failed_steps.jsonl")
    known_missing = set(open(missing_log).read().split()) if os.path.exists(missing_log) else set()

    todo = [v for v in ids if not os.path.exists(os.path.join(a.out, f"{v}.npz"))]
    print(f"{len(ids)} ids, {len(ids) - len(todo)} already done, {len(todo)} to do", flush=True)
    index.prefetch([v for v in todo if v in ann])

    from omniretriever import OmniRetriever
    model = OmniRetriever.from_pretrained(a.base_model, a.adapter, device=a.device, dtype=a.dtype)

    done = 0
    for video_id in todo:
        if a.limit and done >= a.limit:
            break
        if video_id not in ann or video_id not in index:
            if video_id not in known_missing:
                append_line(missing_log, video_id)
                known_missing.add(video_id)
            print(f"{video_id}: not in {'annotations.json' if video_id not in ann else VIDEOS_REPO}, skipped", flush=True)
            continue
        steps = [(k, s) for k, s in enumerate(ann[video_id]["annotations"]) if s["segment"][1] > s["segment"][0]]
        tar, off, size = index[video_id]
        mp4 = os.path.join(work, f"{video_id}.mp4")
        t = time.time()
        range_get(tar, off, size, mp4)
        print(f"{video_id}: {len(steps)} steps, mp4 {size / 1e6:.0f} MB fetched in {time.time() - t:.0f} s", flush=True)
        try:
            arrays = extract_video(model, mp4, steps, video_id, failed_log)
        finally:
            os.remove(mp4)
        tmp = os.path.join(a.out, f"{video_id}.tmp.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, os.path.join(a.out, f"{video_id}.npz"))
        done += 1
        print(f"{video_id}: {len(arrays['emb'])}/{len(steps)} steps saved, {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
