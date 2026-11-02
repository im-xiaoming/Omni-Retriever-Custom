#!/usr/bin/env python3
"""
probe_codecs_htstep.py
======================
Codec of every mp4 the HT-Step split uses, read over HTTP without downloading the videos: ffprobe opens the member
inside its tar through ffmpeg's ``subfile`` protocol (start / end from ``data/htstep/tar_index.csv``), or the file of
htstep-val-seen-videos, and only fetches the container header. Then ``ffmpeg -frames:v 1`` decodes one frame of the
videos whose pixel format ffprobe cannot tell. -> ``codecs.jsonl`` (one line per video) + a count on stdout.
Needs ffprobe / ffmpeg on PATH (Linux; docs/htstep/TAI_VIDEO.md section 3.3).

    python scripts/probe_codecs_htstep.py --out codecs.jsonl --threads 16
"""
import argparse
import csv
import json
import os
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests

WL = ["-protocol_whitelist", "file,http,https,tcp,tls,crypto,subfile"]   # subfile over https is off by default
from huggingface_hub import hf_hub_url

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cdn(repo: str, path: str) -> str:
    url = hf_hub_url(repo, path, repo_type="dataset")
    r = requests.head(url, allow_redirects=False, timeout=60)
    return r.headers["Location"] if r.is_redirect else url


def probe(job) -> dict:
    vid, src, off, size = job
    out = {"video_id": vid, "source": src.get("tar") or src["file"]}
    for k in range(4):
        try:
            loc = cdn(src["repo"], src.get("tar") or src["file"])
            url = f"subfile,,start,{off},end,{off + size},,:{loc}" if "tar" in src else loc
            o = subprocess.run(["ffprobe", "-v", "error", *WL, "-show_entries", "stream=codec_type,codec_name,pix_fmt,width,height",
                                "-show_entries", "format=duration", "-of", "json", url],
                               capture_output=True, text=True, timeout=180)
            j = json.loads(o.stdout or "{}")
            vs = [s for s in j.get("streams", []) if s["codec_type"] == "video"]
            out.update(video=vs[0]["codec_name"] if vs else None, pix_fmt=vs[0].get("pix_fmt") if vs else None,
                       wh=[vs[0].get("width"), vs[0].get("height")] if vs else None,
                       audio=[s["codec_name"] for s in j.get("streams", []) if s["codec_type"] == "audio"],
                       duration=float(j.get("format", {}).get("duration", 0) or 0), err=o.stderr.strip()[:200])
            if vs and not vs[0].get("pix_fmt"):           # decode one frame: is the stream decodable at all?
                d = subprocess.run(["ffmpeg", "-v", "error", *WL, "-i", url, "-frames:v", "1", "-f", "null", "-"],
                                   capture_output=True, text=True, timeout=300)
                out["decode_one_frame"] = {"rc": d.returncode, "err": d.stderr.strip()[-200:]}
            return out
        except Exception as e:
            out["err"] = f"{type(e).__name__}: {e}"[:200]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default=os.path.join(REPO_ROOT, "data/htstep/htstep_split_omni.json"))
    p.add_argument("--tar-index", default=os.path.join(REPO_ROOT, "data/htstep/tar_index.csv"))
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    idx = {(r["tar_path"], r["member_name"]): r for r in csv.DictReader(open(a.tar_index, encoding="utf-8"))}
    split = json.load(open(a.split, encoding="utf-8"))
    jobs = []
    for sp in ("train", "val"):
        for vid, v in sorted(split[f"{sp}_videos"].items()):
            s = v["source"]
            if "tar" in s:
                r = idx[(s["tar"], s["member"])]
                jobs.append((vid, s, int(r["offset_data"]), int(r["size"])))
            else:
                jobs.append((vid, s, 0, 0))
    done = set()
    if os.path.exists(a.out):                              # resume: keep the probed lines, redo the failed ones
        rows = [json.loads(l) for l in open(a.out, encoding="utf-8")]
        rows = [r for r in rows if r.get("video") and not r.get("decode_one_frame", {}).get("rc")]
        done = {r["video_id"] for r in rows}
        with open(a.out, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    jobs = [j for j in jobs if j[0] not in done]
    print(len(jobs), "to probe", flush=True)
    with ThreadPoolExecutor(a.threads) as ex, open(a.out, "a", encoding="utf-8") as f:
        for i, r in enumerate(ex.map(probe, jobs), 1):
            f.write(json.dumps(r) + "\n")
            if i % 500 == 0:
                f.flush()
                print(i, flush=True)
    rows = [json.loads(l) for l in open(a.out, encoding="utf-8")]
    print(json.dumps({"videos": len(rows),
                      "codec": Counter(f"{r.get('video')}/{r.get('pix_fmt')}" for r in rows).most_common(),
                      "decode_one_frame_failed": sum(r.get("decode_one_frame", {}).get("rc", 0) != 0 for r in rows),
                      "probe_errors": sum(bool(r.get("err")) and r.get("video") is None for r in rows)}, indent=1))


if __name__ == "__main__":
    main()
