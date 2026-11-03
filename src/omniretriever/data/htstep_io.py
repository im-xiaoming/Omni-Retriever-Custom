"""Getting HT-Step mp4s and clip shards in and out of HF (scripts/precut_htstep.py, docs/htstep/TAI_VIDEO.md section 5).

* ``range_get``: bytes of one member of a ``nguyenminh04/htstep-videos`` tar by HTTP range (offset / size from
  ``data/htstep/tar_index.csv``); the CDN URL is resolved per call (signed URLs expire) and the token never goes to the
  CDN; answer must be ``206`` with the exact byte count; retry ``5 * 2**k``, 5 tries.
* ``hf_get``: one file of an HF dataset repo (the ``htstep-val-seen-videos`` mp4s), same retries.
* ``probe`` / ``same_video`` / ``transcode``: ffprobe of the first video stream, and AV1 -> lossless H.264
  (decord 0.6.0 cannot decode AV1) with the frame count / avg fps / length checked before and after.
* ``upload``: ``upload_file`` with retries.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time

VIDEOS_REPO = "nguyenminh04/htstep-videos"
TRANSCODE = ["-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast",
             "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-c:a", "copy"]


def _retry(fn, tries: int = 5):
    for k in range(tries):
        try:
            return fn()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * 2 ** k)


def range_get(tar: str, off: int, size: int, out: str, repo: str = VIDEOS_REPO, tries: int = 5) -> None:
    """Bytes [off, off + size) of ``tar`` in ``repo`` to ``out``."""
    import requests
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo, tar, repo_type="dataset")
    tok = os.environ.get("HF_TOKEN")
    auth = {"Authorization": f"Bearer {tok}"} if tok else {}

    def once():
        r = requests.head(url, allow_redirects=False, timeout=60, headers=auth)
        loc, hdr = (r.headers["Location"], {}) if r.is_redirect else (url, auth)
        with requests.get(loc, headers={**hdr, "Range": f"bytes={off}-{off + size - 1}"}, stream=True,
                          timeout=120) as g:
            g.raise_for_status()
            if g.status_code != 206:
                raise IOError(f"HTTP {g.status_code}, not a range answer")
            with open(out, "wb") as f:
                for ch in g.iter_content(1 << 22):
                    f.write(ch)
        if os.path.getsize(out) != size:
            raise IOError(f"got {os.path.getsize(out)} bytes, want {size}")

    _retry(once, tries)


def hf_get(repo: str, path: str, out: str, tries: int = 5) -> None:
    """One file of a dataset repo to ``out``."""
    from huggingface_hub import hf_hub_download

    tmp = os.path.join(os.path.dirname(os.path.abspath(out)), "_hf")

    def once():
        shutil.move(hf_hub_download(repo, path, repo_type="dataset", local_dir=tmp), out)

    try:
        _retry(once, tries)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def probe(path: str) -> dict:
    """codec / pix_fmt / avg fps / frame count (packets, no decoding) / length of the first video stream."""
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=codec_name,pix_fmt,avg_frame_rate,nb_read_packets,duration", "-show_entries",
                        "format=duration", "-of", "json", path], capture_output=True, text=True, timeout=600)
    j = json.loads(o.stdout or "{}")
    s = (j.get("streams") or [{}])[0]
    return {"codec": s.get("codec_name"), "pix_fmt": s.get("pix_fmt"), "fps": s.get("avg_frame_rate"),
            "frames": int(s.get("nb_read_packets") or 0),
            "duration": float(s.get("duration") or j.get("format", {}).get("duration") or 0), "err": o.stderr[-200:]}


def same_video(a: dict, b: dict) -> bool:
    """Frame count and avg fps equal, lengths within one frame."""
    num, den = (int(x) for x in (a["fps"] or "0/1").split("/"))
    frame = den / num if num else 0.1
    return a["frames"] == b["frames"] and a["fps"] == b["fps"] and abs(a["duration"] - b["duration"]) <= frame


def transcode(src: str, dst: str, before: dict) -> tuple:
    """AV1 ``src`` -> lossless H.264 ``dst``. (ok, why, probe after); ``dst`` removed when not ok."""
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, *TRANSCODE, dst],
                       capture_output=True, text=True)
    after = probe(dst) if r.returncode == 0 else None
    if after is not None and same_video(before, after):
        return True, "", after
    if os.path.exists(dst):
        os.remove(dst)
    return False, (r.stderr[-300:] if r.returncode else f"before {before} after {after}"), after


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def upload(path: str, path_in_repo: str, repo: str, tries: int = 6) -> None:
    from huggingface_hub import HfApi

    _retry(lambda: HfApi().upload_file(path_or_fileobj=path, path_in_repo=path_in_repo, repo_id=repo,
                                       repo_type="dataset", commit_message=f"{path_in_repo} (precut_htstep.py)"), tries)
