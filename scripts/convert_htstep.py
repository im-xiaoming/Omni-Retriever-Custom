#!/usr/bin/env python3
"""
convert_htstep.py
=================
HT-Step -> the Omni manifests, with the data rules of UniAV (docs/htstep/UNIAV_XU_LY_DU_LIEU.md section 12,
plan OMNI_HTSTEP_LORA.md section 2). Two steps:

``split`` (local, no GPU, no video read)
    Picks videos and steps and writes, under ``--out-dir`` (default ``data/htstep``):

    * ``htstep_split_omni.json``  train / val videos with their mp4 source, excluded videos and why, the rules,
      sha256 of annotations.json / taxonomy.csv, counts;
    * ``items_train.jsonl`` / ``items_val.jsonl``  one record per step: what ``precut_htstep.py`` cuts.

``smoke`` (local or Colab; GPU smoke test of docs/htstep/NHAT_KY_SMOKE.md)
    A very small train / val set without precut: ``--limit-videos 30 5`` videos (train: one tar, val: tar sources),
    h264 / vp9 only (``docs/htstep/tai_video/codecs.jsonl``: AV1 and empty video tracks dropped), window clamped to
    the ffprobe duration; ``smoke_train.jsonl`` / ``smoke_val.jsonl`` in the ``finalize`` record format without
    ``clip`` (the loader cuts the mp4 itself). ``--video-root`` downloads exactly those mp4s by HTTP range.

``finalize`` (after ``precut_htstep.py``)
    Reads the precut index (window actually cut, ``video_length``, status of every item) and writes
    ``train_omni_htstep.jsonl`` / ``val_omni_htstep.jsonl`` in the record format of the YouCookII manifests
    (``scripts/convert_youcookii.py``) plus the HT-Step fields and ``clip``; ``dropped_steps.json`` lists every
    step precut could not use.

Rules (section 2.2 of the plan):

* video: ``subset == "train"`` -> train, ``subset == "val_seen"`` -> val; an mp4 on HF; not a YouCook2 video;
* step: every ``partial``; ``segment[1] > segment[0]``; the first 48 in json order; ``step_idx`` = position in
  that filtered list (UniAV's ``labels``, ``libs/datasets/htstep.py:106,125`` of htstep-artc);
* text: ``f"{activity}: {headline.strip()}"`` of the taxonomy row whose ``global_step_index`` is the step ``id``
  (not the ``headline`` field someone added to the local annotations.json);
* window (precut, from the mp4 itself): ``end`` past the end of the file is clamped to it; ``start >= length``
  drops the step (``past_end``).

Usage
-----
    python scripts/convert_htstep.py split
    python scripts/convert_htstep.py finalize --precut-index <clips>/precut_index.jsonl
    python scripts/convert_htstep.py smoke --limit-videos 30 5 [--video-root /content/htstep_smoke/videos]
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT = "<video>\nPlease describe the video."            # same as scripts/convert_youcookii.py
MAX_STEPS = 48                                            # UniAV num_classes (htstep-artc libs/core/config.py:25)
SPLITS = {"train": "train", "val": "val_seen"}            # manifest split -> annotations.json subset
VIDEOS_REPO = "nguyenminh04/htstep-videos"
VAL_VIDEOS_REPO = "nguyenminh04/htstep-val-seen-videos"
BATCH = 32
RULES = {
    "video": "subset == 'train' -> train, subset == 'val_seen' -> val; mp4 on HF (tar_index.csv of htstep-videos, "
             "htstep-val-seen-videos part1/part2; val prefers htstep-val-seen-videos); YouCook2 ids excluded "
             "(yc2_overlap); mp4 unreadable at precut -> excluded",
    "step": "keep Full Match and Partial Match; segment[1] > segment[0]; first 48 in json order; "
            "step_idx = position after that filter",
    "text": "f'{activity}: {headline.strip()}' from the taxonomy row with global_step_index == id",
    "window": "from the mp4 (decord frames / avg fps): end > length -> clamped; start >= length -> step dropped "
              "(past_end)",
    "dev": "none",
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: List[dict], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return sha256(path)


def load_taxonomy(path: str) -> Dict[int, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {int(r["global_step_index"]): r for r in csv.DictReader(f)}


def step_text(row: dict) -> str:
    return f"{row['activity']}: {row['headline'].strip()}"


def video_steps(annotations: List[dict]) -> List[dict]:
    """UniAV's step filter, in its order: end > start, then the first 48 (htstep-artc htstep.py:106)."""
    return [a for a in annotations if a["segment"][1] > a["segment"][0]][:MAX_STEPS]


def mp4_sources(tar_index: str, val_files: List[str]) -> Dict[str, List[dict]]:
    """video id -> every place its mp4 is on HF, val repo first."""
    out: Dict[str, List[dict]] = {}
    for f in sorted(val_files):
        name = f.split("/")[-1]
        if f.count("/") == 1 and name.endswith(".mp4") and name.count(".") == 1:   # part1/<id>.mp4, not <id>.f605.mp4
            out.setdefault(name[:-4], []).append({"repo": VAL_VIDEOS_REPO, "file": f})
    with open(tar_index, newline="", encoding="utf-8") as f:
        for r in sorted(csv.DictReader(f), key=lambda r: (r["tar_path"], r["member_name"])):
            if r["member_name"].endswith(".mp4"):
                out.setdefault(r["video_id"], []).append({"repo": VIDEOS_REPO, "tar": r["tar_path"],
                                                          "member": r["member_name"]})
    return out


def select(ann: dict, tax: Dict[int, dict], sources: Dict[str, List[dict]], yc2: set):
    """(videos {split: {id: source}}, items {split: [item]}, excluded {id: reason}, counts)."""
    videos = {s: {} for s in SPLITS}
    items = {s: [] for s in SPLITS}
    excluded: Dict[str, str] = {}
    counts = {s: Counter() for s in SPLITS}
    subset_split = {v: k for k, v in SPLITS.items()}
    for vid in sorted(ann):
        v = ann[vid]
        split = subset_split.get(v["subset"].lower())
        if split is None:
            continue
        c = counts[split]
        c["videos_in_json"] += 1
        c["steps_in_json"] += len(v["annotations"])
        steps = video_steps(v["annotations"])
        c["steps_after_step_filter"] += len(steps)
        if not steps:
            excluded[vid] = "no_step"
            continue
        if vid in yc2:
            excluded[vid] = "yc2_overlap"
            c["videos_yc2_overlap"] += 1
            c["steps_yc2_overlap"] += len(steps)
            continue
        if vid not in sources:
            excluded[vid] = "no_mp4"
            c["videos_no_mp4"] += 1
            c["steps_no_mp4"] += len(steps)
            continue
        src = sources[vid][0]                     # val repo first (mp4_sources), as section 2.2 asks
        videos[split][vid] = {"source": src, "n_sources": len(sources[vid])}
        for i, a in enumerate(steps):
            row = tax[int(a["id"])]               # KeyError = an id outside the taxonomy: must not happen
            items[split].append({
                "id": f"{vid}#{i}", "video_id": vid, "step_idx": i, "global_step_index": int(a["id"]),
                "partial": a["partial"], "segment": [float(a["segment"][0]), float(a["segment"][1])],
                "text": step_text(row), "source": src,
            })
        c["videos_kept"] += 1
        c["steps_kept"] += len(steps)
    return videos, items, excluded, {s: dict(c) for s, c in counts.items()}


def same_text_pairs(texts: List[str], batch: int = BATCH, draws: int = 1000, seed: int = 0) -> float:
    """Mean number of pairs (i < j) with the same text in a random batch of ``batch`` items."""
    rng = random.Random(seed)
    tot = 0
    for _ in range(draws):
        c = Counter(rng.sample(texts, batch))
        tot += sum(n * (n - 1) // 2 for n in c.values())
    return tot / draws


def length_stats(lengths) -> dict:
    x = np.asarray(lengths, dtype=float)
    if not len(x):
        return {"n": 0}
    out = {f"p{q}": round(float(np.percentile(x, q)), 2) for q in (0, 10, 50, 90, 99, 100)}
    out.update({"n": int(len(x)), **{f">{t}s": round(100 * float((x > t).mean()), 2) for t in (8, 30, 60)}})
    return out


def describe(items: List[dict], seg_key: str) -> dict:
    texts = [it["text"] for it in items]
    part = Counter(it["partial"] for it in items)
    return {
        "videos": len({it["video_id"] for it in items}), "steps": len(items),
        "partial": dict(part), "full_match_pct": round(100 * part["Full Match"] / max(len(items), 1), 2),
        "length_sec": length_stats([it[seg_key][1] - it[seg_key][0] for it in items]),
        "distinct_text": len(set(texts)),
        f"same_text_pairs_per_batch{BATCH}": round(same_text_pairs(texts), 3) if len(texts) >= BATCH else None,
    }


def hf_val_files() -> List[str]:
    from huggingface_hub import HfApi
    return HfApi().list_repo_files(VAL_VIDEOS_REPO, repo_type="dataset")


def verify_tar_index(tars_json: str) -> dict:
    """The tar index is still right if HF holds the same tars with the same sizes (a tar is never edited in
    place without its size changing in practice; a changed tar must be re-indexed)."""
    from huggingface_hub import HfApi
    want = json.load(open(tars_json, encoding="utf-8"))["tars"]
    have = {f.path: f.size for f in HfApi().list_repo_tree(VIDEOS_REPO, repo_type="dataset", path_in_repo="videos")
            if f.path.endswith(".tar")}
    diff = sorted(t for t in set(want) | set(have) if want.get(t, {}).get("size") != have.get(t))
    return {"tars_index": len(want), "tars_hf": len(have), "differ": diff}


def cmd_split(a) -> None:
    ann_path, tax_path = os.path.join(a.htstep_dir, "annotations.json"), os.path.join(a.htstep_dir, "taxonomy.csv")
    ann = json.load(open(ann_path, encoding="utf-8"))
    tax = load_taxonomy(tax_path)
    check = verify_tar_index(a.tar_index_tars) if not a.no_verify else {"skipped": True}
    if check.get("differ"):
        sys.exit(f"CHECK tar_index FAIL: tars differ from HF: {check['differ'][:10]} -- rebuild tar_index.csv")
    print(f"CHECK tar_index {'OK' if not check.get('skipped') else 'SKIPPED'} {check}")
    val_files = json.load(open(a.val_files, encoding="utf-8")) if a.val_files else hf_val_files()
    sources = mp4_sources(a.tar_index, val_files)
    yc2 = set(json.load(open(a.yc2, encoding="utf-8"))["database"])
    videos, items, excluded, counts = select(ann, tax, sources, yc2)

    split_doc = {
        "rules": RULES,
        "sha256": {"annotations.json": sha256(ann_path), "taxonomy.csv": sha256(tax_path),
                   "tar_index.csv": sha256(a.tar_index)},
        "tar_index_check": check,
        "train_videos": videos["train"], "val_videos": videos["val"],
        "excluded": dict(sorted(excluded.items())),
        "counts": counts,
        "items": {s: describe(items[s], "segment") for s in SPLITS},
    }
    if a.uniav_feat_index and os.path.isfile(a.uniav_feat_index):
        idx = json.load(open(a.uniav_feat_index, encoding="utf-8"))
        uniav = {os.path.basename(n)[:-4] for t in idx.get("nguyenminh04/htstep-feats-k400", {}).values()
                 for n, size in t if n.endswith(".npz") and size > 0}
        uniav_train = {v for v in uniav if ann.get(v, {}).get("subset") == "train"}
        omni = set(videos["train"])
        split_doc["uniav_train_overlap"] = {   # report only (plan 2.3)
            "uniav_train_videos_with_npz": len(uniav_train), "omni_train_in_uniav": len(omni & uniav_train),
            "omni_train_not_in_uniav": len(omni - uniav_train), "uniav_not_in_omni": len(uniav_train - omni)}
    out = a.out_dir
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "htstep_split_omni.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(split_doc, f, indent=1, ensure_ascii=False)
    for s in SPLITS:
        sha = write_jsonl(items[s], os.path.join(out, f"items_{s}.jsonl"))
        print(f"items_{s}.jsonl: {len(items[s])} steps / {len(videos[s])} videos  sha256 {sha}")
    print(json.dumps({"counts": counts, "excluded": dict(Counter(excluded.values())),
                      "items": split_doc["items"], "uniav": split_doc.get("uniav_train_overlap")},
                     indent=1, ensure_ascii=False))
    print("htstep_split_omni.json sha256", sha256(os.path.join(out, "htstep_split_omni.json")))


def build_record(it: dict, window: List[float], video_length: float) -> dict:
    """The record of scripts/convert_youcookii.py::build_record plus the HT-Step fields and ``clip``."""
    return {
        "id": it["id"], "type": "retrieval",
        "conversations": [{"from": "human", "value": PROMPT}, {"from": "gpt", "value": it["text"]}],
        "video": f"{it['video_id']}.mp4", "timestamps": [float(window[0]), float(window[1])],
        "text": it["text"],
        "video_id": it["video_id"], "step_idx": it["step_idx"], "global_step_index": it["global_step_index"],
        "partial": it["partial"], "segment": it["segment"], "window": [float(window[0]), float(window[1])],
        "video_length": float(video_length),
        "clip": f"clips/{it['video_id']}/{it['step_idx']}.npz",
    }


def cmd_finalize(a) -> None:
    pre = {r["id"]: r for r in read_jsonl(a.precut_index)}
    dropped, stats = {}, {}
    for s in SPLITS:
        items = read_jsonl(os.path.join(a.out_dir, f"items_{s}.jsonl"))
        recs, st = [], Counter()
        for it in items:
            p = pre.get(it["id"])
            status = p["status"] if p else "not_cut"
            st[status] += 1
            if status != "ok":
                dropped[it["id"]] = status
                continue
            recs.append(build_record(it, p["window"], p["video_length"]))
        recs.sort(key=lambda r: (r["video_id"], r["step_idx"]))
        sha = write_jsonl(recs, os.path.join(a.out_dir, f"{s}_omni_htstep.jsonl"))
        stats[s] = {"status": dict(st), "clamped_end": sum(r["window"][1] < r["segment"][1] for r in recs),
                    **describe(recs, "window"), "sha256": sha}
        print(f"{s}_omni_htstep.jsonl: {len(recs)} records / {stats[s]['videos']} videos  sha256 {sha}")
    with open(os.path.join(a.out_dir, "dropped_steps.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"steps": dict(sorted(dropped.items())), "stats": stats}, f, indent=1, ensure_ascii=False)
    print(json.dumps(stats, indent=1, ensure_ascii=False))


def decodable(row: Optional[dict]) -> bool:
    """decord (pip, Linux) reads h264 / vp9 with a pixel format; AV1 and empty video tracks fail
    (docs/htstep/TAI_VIDEO.md 3.2-3.3)."""
    return bool(row) and row.get("video") in ("h264", "vp9") and bool(row.get("pix_fmt"))


def smoke_window(segment: List[float], duration: float) -> Optional[List[float]]:
    """Rule 2.2 with the ffprobe duration: end clamped to it, None (past_end) when start >= duration."""
    if segment[0] >= duration:
        return None
    return [float(segment[0]), float(min(segment[1], duration))]


def smoke_select(items: Dict[str, List[dict]], codecs: Dict[str, dict], train_tar: str, n_train: int, n_val: int):
    """{split: [record]} and the dropped step ids: the first ``n`` decodable videos by id (train: of ``train_tar``,
    val: with a tar source, so that every mp4 comes by range read)."""
    want = {"train": lambda it: it["source"].get("tar") == train_tar, "val": lambda it: "tar" in it["source"]}
    out, dropped = {}, {}
    for sp, n in (("train", n_train), ("val", n_val)):
        vids = sorted({it["video_id"] for it in items[sp] if want[sp](it) and decodable(codecs.get(it["video_id"]))})[:n]
        keep = set(vids)
        recs = []
        for it in items[sp]:
            if it["video_id"] not in keep:
                continue
            w = smoke_window(it["segment"], codecs[it["video_id"]]["duration"])
            if w is None:
                dropped[it["id"]] = "past_end"
                continue
            r = build_record(it, w, w[1] - w[0])
            del r["clip"]
            recs.append(r)
        out[sp] = sorted(recs, key=lambda r: (r["video_id"], r["step_idx"]))
    return out, dropped


def cmd_smoke(a) -> None:
    items = {s: read_jsonl(os.path.join(a.items_dir, f"items_{s}.jsonl")) for s in SPLITS}
    codecs = {r["video_id"]: r for r in read_jsonl(a.codecs)}
    recs, dropped = smoke_select(items, codecs, a.train_tar, *a.limit_videos)
    for s, rs in recs.items():
        sha = write_jsonl(rs, os.path.join(a.out_dir, f"smoke_{s}.jsonl"))
        print(f"smoke_{s}.jsonl: {len(rs)} steps / {len({r['video_id'] for r in rs})} videos  sha256 {sha}")
    print("dropped (past_end):", dropped)
    if not a.video_root:
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bench_precut_one_tar import range_get
    idx = {(r["tar_path"], r["member_name"]): r for r in csv.DictReader(open(a.tar_index, encoding="utf-8"))}
    src = {it["video_id"]: it["source"] for s in SPLITS for it in items[s]}
    os.makedirs(a.video_root, exist_ok=True)
    for vid in sorted({r["video_id"] for rs in recs.values() for r in rs}):
        m = idx[(src[vid]["tar"], src[vid]["member"])]
        out = os.path.join(a.video_root, f"{vid}.mp4")
        if not (os.path.exists(out) and os.path.getsize(out) == int(m["size"])):
            range_get(src[vid]["tar"], int(m["offset_data"]), int(m["size"]), out)
        print(f"  {vid}.mp4 {int(m['size']) / 1e6:.1f} MB", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--htstep-dir", default="D:/Học/KL/Data/Htstep/htstep/data")
    s.add_argument("--tar-index", default="D:/Học/KL/Data/Htstep/pseudocap/metadata/tar_index.csv")
    s.add_argument("--tar-index-tars", default="D:/Học/KL/Data/Htstep/pseudocap/metadata/tar_index_tars.json")
    s.add_argument("--no-verify", action="store_true", help="skip the HF size check of the tar index")
    s.add_argument("--val-files", default="", help="json list of htstep-val-seen-videos files (default: ask HF)")
    s.add_argument("--yc2", default="D:/Học/KL/Data/YouCookII/metadata/youcookii_annotations_uemr.json")
    s.add_argument("--uniav-feat-index", default="D:/Học/KL/Data/Htstep/khao_sat/feat_tar_index.json")
    s.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "data", "htstep"))
    f = sub.add_parser("finalize")
    f.add_argument("--precut-index", required=True)
    f.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "data", "htstep"))
    m = sub.add_parser("smoke")
    m.add_argument("--limit-videos", nargs=2, type=int, default=[30, 5], metavar=("TRAIN", "VAL"))
    m.add_argument("--train-tar", default="videos/htstep_videos_4007.tar")
    m.add_argument("--codecs", default=os.path.join(REPO_ROOT, "docs", "htstep", "tai_video", "codecs.jsonl"))
    m.add_argument("--tar-index", default=os.path.join(REPO_ROOT, "data", "htstep", "tar_index.csv"))
    m.add_argument("--items-dir", default=os.path.join(REPO_ROOT, "data", "htstep"))
    m.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "data", "htstep", "smoke"))
    m.add_argument("--video-root", default="", help="download the mp4s here (HTTP range); empty = manifests only")
    a = p.parse_args()
    {"split": cmd_split, "finalize": cmd_finalize, "smoke": cmd_smoke}[a.cmd](a)


if __name__ == "__main__":
    main()
