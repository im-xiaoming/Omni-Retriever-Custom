#!/usr/bin/env python3
"""
inventory_htstep_videos.py
==========================
Inventory of the HT-Step mp4s on HF, for docs/htstep/TAI_VIDEO.md section 1. Uses the HF API and tar headers only
(HTTP range reads of 512 bytes), never downloads a video.

* sizes of every tar of ``nguyenminh04/htstep-videos`` and every file of ``nguyenminh04/htstep-val-seen-videos``;
* the tar index (``pseudocap/metadata/tar_index.csv``) is still right: same tars, same sizes as on HF, and the header
  of the LAST member of every tar, read again now, has the name / offset / size of the index (a tar that differs
  must be re-indexed with pseudocap/tools/build_tar_index.py before cutting);
* per tar: members used for train / val (rule 2.2, i.e. ``data/htstep/htstep_split_omni.json``) and members not
  needed, by reason (val copy taken from htstep-val-seen-videos, YouCook2, subset outside train / val_seen, no step);
  bytes that really have to be downloaded.

    python -I -X utf8 scripts/inventory_htstep_videos.py --out docs/htstep/tai_video
"""

import argparse
import csv
import json
import os
import sys
import tarfile
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEOS_REPO = "nguyenminh04/htstep-videos"
VAL_REPO = "nguyenminh04/htstep-val-seen-videos"
PSEUDO = "D:/Học/KL/Data/Htstep/pseudocap"


def last_header_ok(read, rows) -> bool:
    """The 512-byte header just before the last member's data names that member with that size."""
    r = max(rows, key=lambda r: int(r["offset_data"]))
    ti = tarfile.TarInfo.frombuf(read(int(r["offset_data"]) - 512, 512), "utf-8", "surrogateescape")
    return (ti.name.endswith(r["member_name"]) or len(r["member_name"]) > 99) and ti.size == int(r["size"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default=os.path.join(REPO_ROOT, "data/htstep/htstep_split_omni.json"))
    p.add_argument("--annotations", default="D:/Học/KL/Data/Htstep/htstep/data/annotations.json")
    p.add_argument("--tar-index", default=f"{PSEUDO}/metadata/tar_index.csv")
    p.add_argument("--tars-json", default=f"{PSEUDO}/metadata/tar_index_tars.json")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    sys.path.insert(0, f"{PSEUDO}/tools")
    from build_tar_index import range_reader               # the pseudo-caption pipeline's HTTP range reader
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfApi, hf_hub_url

    api = HfApi()
    rev = api.dataset_info(VIDEOS_REPO).sha
    tars = {f.path: f.size for f in api.list_repo_tree(VIDEOS_REPO, "videos", repo_type="dataset", revision=rev)
            if f.path.endswith(".tar")}
    vrev = api.dataset_info(VAL_REPO).sha
    vfiles = {f.path: f.size for f in api.list_repo_tree(VAL_REPO, repo_type="dataset", recursive=True, revision=vrev)
              if hasattr(f, "size") and f.path.endswith(".mp4")}

    old = json.load(open(a.tars_json, encoding="utf-8"))
    index = defaultdict(list)
    with open(a.tar_index, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            index[r["tar_path"]].append(r)
    size_diff = sorted(t for t in set(tars) | set(old["tars"]) if old["tars"].get(t, {}).get("size") != tars.get(t))

    def check(t):
        return t, last_header_ok(range_reader(hf_hub_url(VIDEOS_REPO, t, repo_type="dataset", revision=rev)), index[t])
    with ThreadPoolExecutor(16) as ex:
        hdr = dict(ex.map(check, sorted(t for t in tars if t in index)))
    index_check = {"revision_now": rev, "revision_index": old["revision"], "tars_hf": len(tars),
                   "tars_index": len(old["tars"]), "size_differ": size_diff,
                   "last_header_differ": sorted(t for t, ok in hdr.items() if not ok),
                   "members_index": sum(len(v) for v in index.values()),
                   "end_le_size": all(max(int(r["offset_data"]) + int(r["size"]) for r in index[t]) <= tars[t]
                                      for t in index if t in tars)}
    print(json.dumps(index_check, indent=1), flush=True)

    split = json.load(open(a.split, encoding="utf-8"))
    ann = json.load(open(a.annotations, encoding="utf-8"))
    used_tar = {}                                           # video id -> train / val, only when its source is a tar
    used_val_files = {}
    for sp in ("train", "val"):
        for vid, v in split[f"{sp}_videos"].items():
            src = v["source"]
            if "tar" in src:
                used_tar[vid] = (sp, src["tar"])
            else:
                used_val_files[src["file"]] = sp
    excluded = split["excluded"]

    def reason(vid, t):
        if vid in used_tar and used_tar[vid][1] == t:
            return used_tar[vid][0]
        if vid in split["val_videos"]:
            return "val_copy_in_val_repo"                  # rule 2.2: take the htstep-val-seen-videos copy
        if vid in split["train_videos"]:
            return "train_copy_elsewhere"
        if vid not in ann:
            return "not_in_annotations"
        sub = ann[vid]["subset"].lower()
        if sub not in ("train", "val_seen"):
            return f"subset_{sub}"
        return excluded.get(vid, "unknown")

    per_tar, tot = [], Counter()
    for t in sorted(tars):
        c, b = Counter(), Counter()
        for r in index.get(t, []):
            k = reason(r["video_id"], t)
            c[k] += 1
            b[k] += int(r["size"])
        need = b["train"] + b["val"]
        row = {"tar": t, "tar_bytes": tars[t], "members": sum(c.values()), "need_bytes": need,
               "need_frac": round(need / tars[t], 4), **{f"n_{k}": n for k, n in sorted(c.items())},
               **{f"bytes_{k}": n for k, n in sorted(b.items())}}
        per_tar.append(row)
        for k, n in c.items():
            tot[f"n_{k}"] += n
            tot[f"bytes_{k}"] += b[k]
    val_used = {f: s for f, s in vfiles.items() if f in used_val_files}
    val_unused = {f: s for f, s in vfiles.items() if f not in used_val_files}
    dup_val = sum(v["n_sources"] > 1 for v in split["val_videos"].values())
    summary = {
        "htstep_videos": {"revision": rev, "tars": len(tars), "bytes": sum(tars.values()),
                          "tar_bytes_min_med_max": [min(tars.values()), sorted(tars.values())[len(tars) // 2],
                                                    max(tars.values())]},
        "val_repo": {"revision": vrev, "mp4_files": len(vfiles), "bytes": sum(vfiles.values()),
                     "used_files": len(val_used), "used_bytes": sum(val_used.values()),
                     "unused_files": sorted(val_unused), "unused_bytes": sum(val_unused.values())},
        "members_by_reason": dict(sorted(tot.items())),
        "val_videos_with_two_copies": dup_val,
        "tars_with_needed_member": sum(r["need_bytes"] > 0 for r in per_tar),
        "need_bytes": {"from_tars": tot["bytes_train"] + tot["bytes_val"], "from_val_repo": sum(val_used.values()),
                       "total": tot["bytes_train"] + tot["bytes_val"] + sum(val_used.values())},
        "all_bytes": sum(tars.values()) + sum(vfiles.values()),
        "split_videos": {"train": len(split["train_videos"]), "val": len(split["val_videos"]),
                         "val_from_tar": sum(1 for v, (s, _) in used_tar.items() if s == "val"),
                         "val_from_val_repo": len(used_val_files)},
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "inventory_per_tar.csv"), "w", newline="", encoding="utf-8") as f:
        head = ["tar", "tar_bytes", "members", "need_bytes", "need_frac"]
        keys = head + sorted({k for r in per_tar for k in r} - set(head))
        w = csv.DictWriter(f, keys, restval=0, lineterminator="\n")
        w.writeheader()
        w.writerows(per_tar)
    with open(os.path.join(a.out, "inventory.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"index_check": index_check, "summary": summary,
                   "val_repo_files": dict(sorted(vfiles.items()))}, f, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
