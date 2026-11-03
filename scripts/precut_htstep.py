#!/usr/bin/env python3
"""
precut_htstep.py
================
Cuts every HT-Step step of ``items_{train,val}.jsonl`` (scripts/convert_htstep.py split) into a clip file once, on a
CPU runtime, so the GPU machine only downloads the clips (OMNI_HTSTEP_LORA_COLAB.md section 3, design of
docs/htstep/TAI_VIDEO.md section 5). The work unit is one video:

    HTTP range read of its mp4 inside the tar of nguyenminh04/htstep-videos (offset / size from tar_index.csv), or the
    file of nguyenminh04/htstep-val-seen-videos -> ffprobe -> AV1: lossless H.264 (frame count / avg fps / length
    checked, else ``transcode_mismatch``); no video frames: ``empty_video_track`` -> ``cut_clip`` of every step
    (src/omniretriever/data/clip.py: ``clips/<video_id>/<step_idx>.npz``; ``past_end`` / ``unreadable`` per step)
    -> the temp mp4s are deleted.

Finished videos go into the open shard; at ``--shard-gb`` it is packed into ``shards/clips_NNNN.tar`` with
``shards/clips_NNNN.manifest.jsonl`` (one line per step of its videos: id, status, sha256, bytes, window, ...), the
tar is uploaded to ``--hf-repo``, then the manifest, then ``state/done.jsonl`` / ``state/failed.jsonl``; the local
copies are deleted after that. ``--hf-repo`` is the user's own private dataset repo (``check-repo``: it exists, is
private, is in the account of HF_TOKEN, the token can write); it is the only repo written to, ``nguyenminh04/*`` is
only read. The uploaded manifests are the truth: a rerun (new Colab session) downloads them and
skips their videos, so a video is in exactly one shard; the shard open when a session dies is cut again. Shard names
are never reused. ``fetch_error`` videos (network) go to ``state/failed.jsonl`` and are tried again by the next run.

    python scripts/precut_htstep.py run --hf-repo <user>/htstep-omni-clips --processor <WAVE-7B dir> --work /content/precut
    python scripts/precut_htstep.py verify --hf-repo <user>/htstep-omni-clips --out /content/precut/precut_index.jsonl
    python scripts/precut_htstep.py check-transcode --n 5 --work /content/tc
    python scripts/precut_htstep.py fetch --hf-repo <user>/htstep-omni-clips --out /content/htstep_clips   # GPU side
    python scripts/precut_htstep.py check-repo --hf-repo <user>/htstep-omni-clips   # exists, private, own account, Write token

``verify``: every item id has exactly one line (``ok`` with its sha256, or the reason it is out), no id in two shards,
every manifest has its tar; prints the count per reason and writes ``precut_index.jsonl`` for ``convert_htstep.py
finalize``. ``check-transcode``: for ``--n`` AV1 videos, the frames decord reads from the H.264 copy must equal pixel
for pixel the frames ffmpeg decodes from the AV1 file. ``fetch``: downloads + extracts every shard (free disk checked
first), checks the sha256 of every clip, and downloads ``meta/``.
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from omniretriever.data import htstep_io as io  # noqa: E402

DATA = os.path.join(REPO_ROOT, "data", "htstep")
SHARD_RE = re.compile(r"^shards/(clips_\d{4,})\.manifest\.jsonl$")
VIDEO_FAIL = ("empty_video_track", "transcode_mismatch", "unreadable")
_IP = None


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_items(paths):
    """video id -> {split, source, items} from items_<split>.jsonl files."""
    out = {}
    for p in paths:
        split = os.path.basename(p)[len("items_"):-len(".jsonl")]
        for it in read_jsonl(p):
            out.setdefault(it["video_id"], {"split": split, "source": it["source"], "items": []})["items"].append(it)
    return out


def auto_jobs() -> int:
    """min(nproc, RAM GB / 2.5)."""
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    return max(1, min(os.cpu_count() or 1, int(ram_gb / 2.5)))


def next_shard(used: set, start: int = 0) -> str:
    n = start
    while f"clips_{n:04d}" in used:
        n += 1
    return f"clips_{n:04d}"


# ------------------------------------------------------------------------------------------------------ worker
def _init(processor: str) -> None:
    global _IP
    from transformers import Qwen2VLImageProcessor
    _IP = Qwen2VLImageProcessor.from_pretrained(processor)


def cut_video(vid: str, v: dict, tar_entry, a) -> dict:
    """One video -> its clips under ``a.work/clips/<vid>/``; the video row with one row per step."""
    import numpy as np
    from omniretriever.data.clip import PastEnd, cut_clip

    t0 = time.time()
    tmp = os.path.join(a.work, "tmp", vid)
    os.makedirs(tmp, exist_ok=True)
    raw, h264 = os.path.join(tmp, "raw.mp4"), os.path.join(tmp, "h264.mp4")
    row = {"video_id": vid, "split": v["split"], "source": v["source"].get("tar", v["source"].get("file")),
           "status": "ok", "why": "", "transcoded": False}
    try:
        try:
            if "tar" in v["source"]:
                io.range_get(v["source"]["tar"], tar_entry[0], tar_entry[1], raw, repo=v["source"]["repo"])
            else:
                io.hf_get(v["source"]["repo"], v["source"]["file"], raw)
        except Exception as e:
            return {**row, "status": "fetch_error", "why": f"{type(e).__name__}: {e}"[:300], "steps": []}
        row["mp4_bytes"] = os.path.getsize(raw)
        before = row["probe"] = io.probe(raw)
        path = raw
        if not before["codec"] or before["frames"] == 0:
            row.update(status="empty_video_track", why=f"no video frames ({before['err'].strip()[:120]})")
        elif before["codec"] == "av1":
            t1 = time.time()
            ok, why, after = io.transcode(raw, h264, before)
            row.update(transcoded=True, probe_after=after, transcode_seconds=round(time.time() - t1, 1))
            if ok:
                path = h264
                os.remove(raw)
            else:
                row.update(status="transcode_mismatch", why=why)
        steps = []
        for it in sorted(v["items"], key=lambda r: r["step_idx"]):
            s = {"id": it["id"], "video_id": vid, "split": v["split"], "status": row["status"], "why": row["why"]}
            if row["status"] == "ok":
                rel = f"clips/{vid}/{it['step_idx']}.npz"
                try:
                    c = cut_clip(path, *it["segment"], _IP, n_frames=a.frames, min_pixels=a.pixels,
                                 max_pixels=a.pixels, audio_sec=a.audio_sec)
                    dst = os.path.join(a.work, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    np.savez_compressed(dst, **c)
                    s.update(clip=rel, sha256=io.file_sha256(dst), bytes=os.path.getsize(dst),
                             window=c["window"].tolist(), video_length=float(c["video_length"]),
                             src_fps=float(c["src_fps"]), audio_ok=bool(c["audio_ok"]), src_transcoded=row["transcoded"])
                except PastEnd as e:
                    s.update(status="past_end", why=str(e))
                except Exception as e:                    # broken / undecodable file
                    s.update(status="unreadable", why=f"{type(e).__name__}: {e}"[:300])
            steps.append(s)
        if row["status"] == "ok" and all(s["status"] == "unreadable" for s in steps):
            row.update(status="unreadable", why=steps[0]["why"])
        return {**row, "steps": steps, "seconds": round(time.time() - t0, 1)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------ shards
class Shard:
    """Finished videos waiting to be packed; ``flush`` packs, uploads (tar, then manifest) and cleans up."""

    def __init__(self, work: str, used: set, upload):
        self.work, self.used, self.upload = work, used, upload
        self.videos, self.bytes = [], 0

    def add(self, r: dict) -> None:
        self.videos.append(r)
        self.bytes += sum(s.get("bytes", 0) for s in r["steps"])

    def flush(self):
        """(shard name, rows) or None when empty. Local files are removed only after both uploads returned."""
        if not self.videos:
            return None
        name = next_shard(self.used, len(self.used))
        self.used.add(name)
        d = os.path.join(self.work, "shards")
        os.makedirs(d, exist_ok=True)
        tar, man = os.path.join(d, f"{name}.tar"), os.path.join(d, f"{name}.manifest.jsonl")
        rows = [{**s, "shard": name} for r in self.videos for s in r["steps"]]
        with tarfile.open(tar, "w") as tf:
            for s in rows:
                if s["status"] == "ok":
                    tf.add(os.path.join(self.work, s["clip"]), arcname=s["clip"])
        write_jsonl(rows, man)
        self.upload(tar, f"shards/{name}.tar")
        self.upload(man, f"shards/{name}.manifest.jsonl")
        os.remove(tar)
        os.remove(man)
        for r in self.videos:
            shutil.rmtree(os.path.join(self.work, "clips", r["video_id"]), ignore_errors=True)
        videos, self.videos, self.bytes = self.videos, [], 0
        return name, rows, videos


def hf_manifests(repo: str, out_dir: str) -> dict:
    """shard name -> its manifest rows, for every manifest uploaded to ``repo``."""
    from huggingface_hub import HfApi, hf_hub_download

    out = {}
    for f in sorted(HfApi().list_repo_files(repo, repo_type="dataset")):
        m = SHARD_RE.match(f)
        if m:
            out[m.group(1)] = read_jsonl(hf_hub_download(repo, f, repo_type="dataset", local_dir=out_dir))
    return out


def hf_state(repo: str, name: str, out_dir: str) -> list:
    from huggingface_hub import hf_hub_download
    try:
        return read_jsonl(hf_hub_download(repo, f"state/{name}", repo_type="dataset", local_dir=out_dir))
    except Exception:                                   # first run: no state yet
        return []


def check_clip_repo(repo: str, api=None) -> None:
    """The only repo this pipeline writes to: it must already exist (the user creates it), be a private dataset of the
    token's own account and the token must be able to write. Anything else stops before a byte is written."""
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    me = api.whoami()
    if repo.split("/")[0] != me["name"]:
        sys.exit(f"CHECK clip_repo FAIL: {repo} is not in the account of HF_TOKEN ({me['name']})")
    try:
        info = api.repo_info(repo, repo_type="dataset")
    except Exception as e:
        sys.exit(f"CHECK clip_repo FAIL: dataset repo {repo} not found ({type(e).__name__}); create it first")
    if not info.private:
        sys.exit(f"CHECK clip_repo FAIL: {repo} is public; make it private (repo Settings -> Change visibility)")
    role = me.get("auth", {}).get("accessToken", {}).get("role")
    if role == "read":
        sys.exit("CHECK clip_repo FAIL: HF_TOKEN is a Read token; make a Write token")
    print(f"CHECK clip_repo OK {repo} (private, account {me['name']}, token {role})", flush=True)


def cmd_check_repo(a) -> None:
    check_clip_repo(a.hf_repo)


def cmd_run(a) -> None:
    check_clip_repo(a.hf_repo)
    meta = os.path.join(a.work, "hf")
    videos = read_items(a.items)
    with open(a.tar_index, newline="", encoding="utf-8") as f:
        tar_idx = {(r["tar_path"], r["member_name"]): (int(r["offset_data"]), int(r["size"])) for r in csv.DictReader(f)}
    manifests = hf_manifests(a.hf_repo, meta)
    done = {r["video_id"]: name for name, rows in manifests.items() for r in rows}
    done_rows = {r["video_id"]: r for r in hf_state(a.hf_repo, "done.jsonl", meta)}
    for vid, name in done.items():                      # a session that died between manifest and state upload
        done_rows.setdefault(vid, {"video_id": vid, "shard": name, "recovered_from_manifest": True})
    failed = {r["video_id"]: r for r in hf_state(a.hf_repo, "failed.jsonl", meta) if r["video_id"] not in done}
    todo = deque(v for v in sorted(videos) if v not in done)
    jobs = a.jobs or auto_jobs()
    n_steps = sum(len(videos[v]["items"]) for v in todo)
    print(f"{len(videos)} videos ({sum(len(v['items']) for v in videos.values())} steps): {len(done)} done in "
          f"{len(manifests)} shards, {len(failed)} to retry (fetch_error), {len(todo)} to cut ({n_steps} steps) "
          f"| {jobs} workers | disk free {shutil.disk_usage(a.work).free / 2 ** 30:.0f} GB", flush=True)
    state_dir = os.path.join(a.work, "state")

    def push_state():
        write_jsonl(sorted(done_rows.values(), key=lambda r: r["video_id"]), os.path.join(state_dir, "done.jsonl"))
        write_jsonl(sorted(failed.values(), key=lambda r: r["video_id"]), os.path.join(state_dir, "failed.jsonl"))
        for n in ("done.jsonl", "failed.jsonl"):
            io.upload(os.path.join(state_dir, n), f"state/{n}", a.hf_repo)

    def flush():
        out = shard.flush()
        if out:
            name, rows, vids = out
            for r in vids:
                done_rows[r["video_id"]] = {
                    **{k: r[k] for k in ("video_id", "split", "status", "why", "transcoded")}, "shard": name,
                    "steps": len(r["steps"]), "ok": sum(s["status"] == "ok" for s in r["steps"]),
                    "seconds": r.get("seconds"), "transcode_seconds": r.get("transcode_seconds")}
                failed.pop(r["video_id"], None)
            push_state()
            print(f"  -> {name}: {len(vids)} videos, {sum(s['status'] == 'ok' for s in rows)} clips uploaded", flush=True)

    shard = Shard(a.work, set(manifests), lambda p, q: io.upload(p, q, a.hf_repo))
    t0, steps_done, n_done = time.time(), 0, 0
    with ProcessPoolExecutor(jobs, initializer=_init, initargs=(a.processor,)) as ex:
        pending = {}
        try:
            while todo or pending:
                while todo and len(pending) < 2 * jobs:
                    if shutil.disk_usage(a.work).free < a.min_free_gb * 2 ** 30:
                        if pending:
                            break                       # running videos free their temp mp4 when they finish
                        flush()
                        if shutil.disk_usage(a.work).free < a.min_free_gb * 2 ** 30:
                            sys.exit(f"disk free < {a.min_free_gb} GB with nothing running: stop")
                    vid = todo.popleft()
                    src = videos[vid]["source"]
                    entry = tar_idx[(src["tar"], src["member"])] if "tar" in src else None
                    pending[ex.submit(cut_video, vid, videos[vid], entry, a)] = vid
                fin, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fu in fin:
                    vid = pending.pop(fu)
                    try:
                        r = fu.result()
                    except Exception as e:              # worker crash (e.g. killed for memory): try again next run
                        r = {"video_id": vid, "split": videos[vid]["split"], "status": "fetch_error",
                             "why": f"worker: {type(e).__name__}: {e}"[:300], "steps": []}
                    n_done += 1
                    steps_done += len(videos[vid]["items"])
                    if r["status"] == "fetch_error":
                        failed[vid] = {k: r[k] for k in ("video_id", "split", "status", "why")}
                    else:
                        shard.add(r)
                    rate = steps_done / (time.time() - t0)
                    ok = sum(s["status"] == "ok" for s in r["steps"])
                    print(f"[{n_done}/{n_done + len(todo) + len(pending)}] {vid} ({r['split']}): {r['status']} "
                          f"{ok}/{len(videos[vid]['items'])} clips{' AV1->H.264' if r.get('transcoded') else ''}"
                          f"{' ' + r['why'][:80] if r['why'] else ''} | shard {shard.bytes / 1e9:.2f} GB | "
                          f"{(time.time() - t0) / 60:.0f} min, ETA {(n_steps - steps_done) / max(rate, 1e-9) / 3600:.1f} h",
                          flush=True)
                    if shard.bytes >= a.shard_gb * 1e9:
                        flush()
            flush()
        except KeyboardInterrupt:                       # stop button: keep what is finished
            print("interrupted: uploading the finished videos", flush=True)
            for fu in pending:
                fu.cancel()
            flush()
            raise
    push_state()
    print(json.dumps({"session_videos": n_done, "session_steps": steps_done, "minutes": round((time.time() - t0) / 60, 1),
                      "failed_to_retry": len(failed), "shards": len(shard.used)}, indent=1), flush=True)


# ------------------------------------------------------------------------------------------------------ verify
def index_rows(items: dict, manifests: dict, failed: set):
    """(precut index rows sorted by id, problems). ``items`` id -> item, ``manifests`` shard -> rows,
    ``failed`` video ids with fetch_error."""
    seen, problems = {}, []
    for name, rows in sorted(manifests.items()):
        for r in rows:
            if r["id"] not in items:
                problems.append(f"{r['id']} in {name} is not an item")
            elif r["id"] in seen:
                problems.append(f"{r['id']} in two shards: {seen[r['id']].get('shard')} and {name}")
            else:
                seen[r["id"]] = {**r, "shard": name}
            if r["status"] == "ok" and not r.get("sha256"):
                problems.append(f"{r['id']} ok without sha256 in {name}")
    out = []
    for i, it in sorted(items.items()):
        if i in seen:
            out.append(seen[i])
        else:
            st = "fetch_error" if it["video_id"] in failed else "missing"
            out.append({"id": i, "video_id": it["video_id"], "split": it["split"], "status": st, "why": ""})
            if st == "missing":
                problems.append(f"{i} not cut yet")
    return out, problems


def cmd_verify(a) -> None:
    from huggingface_hub import HfApi

    items = {}
    for p in a.items:
        split = os.path.basename(p)[len("items_"):-len(".jsonl")]
        items.update({it["id"]: {**it, "split": split} for it in read_jsonl(p)})
    meta = os.path.join(os.path.dirname(os.path.abspath(a.out)), "hf")
    manifests = hf_manifests(a.hf_repo, meta)
    files = set(HfApi().list_repo_files(a.hf_repo, repo_type="dataset"))
    failed = {r["video_id"] for r in hf_state(a.hf_repo, "failed.jsonl", meta)}
    rows, problems = index_rows(items, manifests, failed)
    problems += [f"shards/{n}.tar missing on HF" for n in manifests if f"shards/{n}.tar" not in files]
    write_jsonl(rows, a.out)
    table = Counter((r["split"], r["status"]) for r in rows)
    summary = {"items": len(rows), "shards": len(manifests),
               "by_split_status": {s: {k[1]: n for k, n in sorted(table.items()) if k[0] == s} for s in ("train", "val")},
               "videos_transcoded_ok": len({r["video_id"] for r in rows if r.get("src_transcoded")}),
               "clip_gb": round(sum(r.get("bytes", 0) for r in rows) / 1e9, 2), "problems": problems[:50],
               "n_problems": len(problems)}
    with open(os.path.splitext(a.out)[0] + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    if summary["by_split_status"]["train"].get("fetch_error") or summary["by_split_status"]["val"].get("fetch_error"):
        print("fetch_error: chạy lại `run` để thử tải lại các video này", flush=True)
    print(f"CHECK precut_integrity {'FAIL' if problems else 'OK'} -> {a.out}")
    sys.exit(1 if problems else 0)


# ------------------------------------------------------------------------------------------------------ transcode check
def ffmpeg_frames(path: str, idx, w: int, h: int, pix_fmt: str):
    """Frames ``idx`` of ``path`` decoded by the ffmpeg CLI: rgb24 with decord's conversion (``scale=W:H`` filter,
    vf_scale default flags of FFmpeg 4: bilinear, sink format rgb24) or raw yuv420p."""
    import numpy as np
    sel = "+".join(f"eq(n\\,{int(i)})" for i in idx)
    vf = f"select='{sel}'" + (f",scale={w}:{h}:flags=bilinear,format=rgb24" if pix_fmt == "rgb24" else ",format=yuv420p")
    o = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-map", "0:v:0", "-vf", vf,
                        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
                       capture_output=True, check=True)
    per = w * h * 3 if pix_fmt == "rgb24" else w * h * 3 // 2
    return np.frombuffer(o.stdout, np.uint8).reshape(-1, per)


def cmd_check_transcode(a) -> None:
    import numpy as np
    from decord import VideoReader

    videos = read_items(a.items)
    codecs = [r for r in read_jsonl(a.codecs) if r["video"] == "av1" and r["video_id"] in videos]
    pick = sorted(codecs, key=lambda r: r["duration"])[:a.n]
    with open(a.tar_index, newline="", encoding="utf-8") as f:
        tar_idx = {(r["tar_path"], r["member_name"]): (int(r["offset_data"]), int(r["size"])) for r in csv.DictReader(f)}
    os.makedirs(a.work, exist_ok=True)
    res, ok_all = [], True
    for c in pick:
        vid, src = c["video_id"], videos[c["video_id"]]["source"]
        raw, h264 = os.path.join(a.work, f"{vid}.av1.mp4"), os.path.join(a.work, f"{vid}.h264.mp4")
        if "tar" in src:
            io.range_get(src["tar"], *tar_idx[(src["tar"], src["member"])], raw, repo=src["repo"])
        else:
            io.hf_get(src["repo"], src["file"], raw)
        before = io.probe(raw)
        t0 = time.time()
        ok, why, _ = io.transcode(raw, h264, before)
        r = {"video_id": vid, "frames": before["frames"], "fps": before["fps"], "duration": before["duration"],
             "transcode_ok": ok, "transcode_seconds": round(time.time() - t0, 1), "why": why}
        if ok:
            vr = VideoReader(h264, num_threads=1)
            idx = np.linspace(0, len(vr) - 1, 8, dtype=int)
            dec = vr.get_batch(idx).asnumpy()
            h, w = dec.shape[1:3]
            dec = dec.reshape(len(idx), -1)
            ref = ffmpeg_frames(raw, idx, w, h, "rgb24")              # decoded straight from the AV1 file
            yuv_av1, yuv_h264 = ffmpeg_frames(raw, idx, w, h, "yuv420p"), ffmpeg_frames(h264, idx, w, h, "yuv420p")
            same = ref.shape == dec.shape
            r.update(decord_frames=len(vr), frame_idx=idx.tolist(), wh=[w, h], rgb_equal=same and bool((ref == dec).all()),
                     rgb_max_abs_diff=int(np.abs(ref.astype(int) - dec).max()) if same else None,
                     yuv_equal=yuv_av1.shape == yuv_h264.shape and bool((yuv_av1 == yuv_h264).all()))
        # yuv_equal is a diagnostic: rgb_equal False with yuv_equal True = a colour-conversion difference, not the codec
        r["ok"] = bool(ok and r.get("rgb_equal") and r.get("decord_frames") == before["frames"])
        ok_all &= r["ok"]
        res.append(r)
        print(json.dumps(r), flush=True)
        for p in (raw, h264):
            if os.path.exists(p):
                os.remove(p)
    ffv = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    with open(os.path.join(a.work, "check_transcode.json"), "w", encoding="utf-8") as f:
        json.dump({"ffmpeg": ffv, "videos": res, "ok": ok_all}, f, indent=1)
    print(f"{ffv}\nCHECK transcode_pixels {'OK' if ok_all and len(res) == a.n else 'FAIL'} ({len(res)} videos)")
    sys.exit(0 if ok_all and len(res) == a.n else 1)


# ------------------------------------------------------------------------------------------------------ fetch (GPU side)
def cmd_fetch(a) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    os.makedirs(a.out, exist_ok=True)
    tars = {f.path: f.size for f in api.list_repo_tree(a.hf_repo, repo_type="dataset", path_in_repo="shards")
            if f.path.endswith(".tar")}
    done_path = os.path.join(a.out, "fetched.json")
    fetched = json.load(open(done_path, encoding="utf-8")) if os.path.exists(done_path) else {}
    left = {p: s for p, s in tars.items() if p not in fetched}
    need = sum(left.values()) + max(left.values(), default=0) + a.margin_gb * 2 ** 30
    free = shutil.disk_usage(a.out).free
    print(f"{len(tars)} shards, {sum(tars.values()) / 2 ** 30:.1f} GB; {len(left)} to fetch: need "
          f"{need / 2 ** 30:.1f} GB, free {free / 2 ** 30:.1f} GB", flush=True)
    if free < need:
        sys.exit(f"CHECK disk FAIL: free {free / 2 ** 30:.1f} GB < {need / 2 ** 30:.1f} GB")
    dl = os.path.join(a.out, "_dl")
    bad = []
    for i, p in enumerate(sorted(left), 1):
        tar = hf_hub_download(a.hf_repo, p, repo_type="dataset", local_dir=dl)
        man = read_jsonl(hf_hub_download(a.hf_repo, p[:-4] + ".manifest.jsonl", repo_type="dataset", local_dir=dl))
        with tarfile.open(tar) as tf:
            tf.extractall(a.out, filter="data")
        os.remove(tar)
        ok = [r for r in man if r["status"] == "ok"]
        bad += [r["id"] for r in ok if io.file_sha256(os.path.join(a.out, r["clip"])) != r["sha256"]]
        fetched[p] = {"clips": len(ok)}
        with open(done_path, "w", encoding="utf-8") as f:
            json.dump(fetched, f)
        print(f"[{i}/{len(left)}] {p}: {len(ok)} clips, sha256 mismatches so far {len(bad)}", flush=True)
    for f in api.list_repo_files(a.hf_repo, repo_type="dataset"):
        if f.startswith("meta/"):
            hf_hub_download(a.hf_repo, f, repo_type="dataset", local_dir=a.out)
    shutil.rmtree(dl, ignore_errors=True)
    n = sum(v["clips"] for v in fetched.values())
    print(f"CHECK clips_sha256 {'FAIL' if bad else 'OK'} {n} clips in {len(fetched)} shards"
          f"{', mismatched: ' + ', '.join(bad[:10]) if bad else ''} -> {a.out}")
    sys.exit(1 if bad else 0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    items = [os.path.join(DATA, "items_train.jsonl"), os.path.join(DATA, "items_val.jsonl")]
    r = sub.add_parser("run")
    r.add_argument("--hf-repo", required=True, help="the user's private HF dataset repo for the clip shards")
    r.add_argument("--processor", required=True, help="WAVE-7B folder (its preprocessor_config.json)")
    r.add_argument("--work", required=True)
    r.add_argument("--jobs", type=int, default=0, help="0 = min(nproc, RAM GB / 2.5)")
    r.add_argument("--shard-gb", type=float, default=1.5)
    r.add_argument("--min-free-gb", type=float, default=10)
    r.add_argument("--frames", type=int, default=8)              # train.sh --video_min/max_frames 8
    r.add_argument("--pixels", type=int, default=50176)          # train.sh --min_pixels = --max_pixels
    r.add_argument("--audio-sec", type=float, default=8.0)       # train.sh --fixed_audio_duration 8
    v = sub.add_parser("verify")
    v.add_argument("--hf-repo", required=True)
    v.add_argument("--out", required=True, help="precut_index.jsonl")
    t = sub.add_parser("check-transcode")
    t.add_argument("--n", type=int, default=5)
    t.add_argument("--work", required=True)
    t.add_argument("--codecs", default=os.path.join(REPO_ROOT, "docs", "htstep", "tai_video", "codecs.jsonl"))
    f = sub.add_parser("fetch")
    f.add_argument("--hf-repo", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--margin-gb", type=float, default=5)
    k = sub.add_parser("check-repo")
    k.add_argument("--hf-repo", required=True)
    for s in (r, v, t):
        s.add_argument("--items", nargs="+", default=items)
    for s in (r, t):
        s.add_argument("--tar-index", default=os.path.join(DATA, "tar_index.csv"))
    a = p.parse_args()
    {"run": cmd_run, "verify": cmd_verify, "check-transcode": cmd_check_transcode, "fetch": cmd_fetch,
     "check-repo": cmd_check_repo}[a.cmd](a)


if __name__ == "__main__":
    main()
