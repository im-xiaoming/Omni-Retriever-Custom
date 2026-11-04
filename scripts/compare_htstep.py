#!/usr/bin/env python3
"""
compare_htstep.py
=================
best vs pt on HT-Step ``val_seen`` with a paired bootstrap over the queries (numpy only, CPU, seconds): from the
``val_embeds_{pt,best}.npz`` that ``eval_htstep.py`` writes, the diagonal rank of every query against the full gallery
(``omniretriever.evaluation.metrics.per_query_rank``), then ``--n`` resamples of the query set (seed 0) of the
difference best - pt of R@1 / R@5 / R@10 / MRR, T->AV and AV->T. The gallery is never resampled. Vectors are the
float16 copies of the npz, so the point values can differ from ``metrics_htstep_*.json`` in the second decimal.

    python scripts/compare_htstep.py --pt <eval>/val_embeds_pt.npz --best <eval>/val_embeds_best.npz --out <eval>/compare_pt_best.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from omniretriever.evaluation.metrics import per_query_rank  # noqa: E402

KS = (1, 5, 10)


def per_query(npz: dict) -> dict:
    """direction -> metric -> (N,) per-query value in %."""
    sim = npz["text"].astype(np.float32) @ npz["av"].astype(np.float32).T
    out = {}
    for d, s in (("t2v", sim), ("v2t", sim.T)):
        r = per_query_rank(s)
        out[d] = {**{f"R@{k}": 100.0 * (r <= k) for k in KS}, "MRR": 100.0 / r}
    return out


def compare(pt: dict, best: dict, n: int = 2000, seed: int = 0) -> dict:
    if list(pt["ids"]) != list(best["ids"]):
        raise ValueError("pt and best were not encoded on the same items in the same order")
    a, b = per_query(pt), per_query(best)
    idx = np.random.default_rng(seed).integers(0, len(pt["ids"]), size=(n, len(pt["ids"])))
    out = {"n_queries": len(pt["ids"]), "n_boot": n, "seed": seed}
    for d in a:
        out[d] = {}
        for m in a[d]:
            diff = b[d][m] - a[d][m]
            boot = diff[idx].mean(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            out[d][m] = {"pt": round(float(a[d][m].mean()), 2), "best": round(float(b[d][m].mean()), 2),
                         "diff": round(float(diff.mean()), 2), "ci95": [round(float(lo), 2), round(float(hi), 2)],
                         "significant": bool(lo > 0 or hi < 0)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pt", required=True)
    p.add_argument("--best", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=2000)
    a = p.parse_args()
    with np.load(a.pt) as pt, np.load(a.best) as best:
        res = compare(dict(pt), dict(best), n=a.n)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"best - pt on val_seen, N = {res['n_queries']}, paired bootstrap {res['n_boot']} (95 % CI):")
    for d in ("t2v", "v2t"):
        for m, r in res[d].items():
            print(f"  {d} {m:5s} pt {r['pt']:6.2f} | best {r['best']:6.2f} | diff {r['diff']:+6.2f} "
                  f"[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]{' *' if r['significant'] else ''}")
    print(f"(* = CI does not contain 0) -> {a.out}")


if __name__ == "__main__":
    main()
