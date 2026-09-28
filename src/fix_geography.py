"""Repair missing region coordinate proxies in an existing feature cache.

The candidate pool is held fixed for an isolated feature experiment. Regions
without their own advertisements use the most frequent historically selected
city as a distance proxy. New feature generation applies the same rule.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts"


def main():
    changes = {}
    for mode in ["eval", "benchmark"]:
        path = A / f"{mode}_history.joblib"
        hist, hf = joblib.load(path)
        children = {}
        for (sl, il), p in hist.loc_probs.items():
            children.setdefault(sl, []).append((p, il))
        changed = []
        for sl, values in children.items():
            center = hist.search_centers.get(sl, {})
            if center and np.isfinite(float(center.get("item_latitude", 0))):
                continue
            mainloc = max(values)[1]
            proxy = hist.centers.get(mainloc, {})
            if proxy:
                hist.search_centers[sl] = proxy
                changed.append(sl)
        joblib.dump((hist, hf), path, compress=0)
        changes[mode] = len(changed)
        fpath = A / f"{mode}_features.npz"
        if not fpath.exists():
            continue
        arrays = dict(np.load(fpath))
        X = arrays["X"]
        qi = arrays["query_indices"]
        ii = arrays["item_indices"]
        queries = pd.read_parquet(
            A / "eval_queries.parquet"
            if mode == "eval"
            else ROOT / "data/benchmark_queries.parquet"
        )
        bounds = np.r_[0, np.flatnonzero(qi[1:] != qi[:-1]) + 1, len(qi)]
        for q, rec in enumerate(queries.to_dict("records")):
            if rec["search_location_id"] not in changed:
                continue
            start, end = bounds[q : q + 2]
            ix = ii[start:end]
            same, lp, dist = hist.location_scores(rec)
            geo = np.maximum(np.maximum(same, lp), 0.35 * np.exp(-dist / 50))[ix]
            block = X[start:end]
            block[:, 12] = np.log1p(dist[ix])
            block[:, 13] = dist[ix] < 10
            block[:, 14] = dist[ix] < 50
            cm = np.maximum(block[:, 9], 0.7 * block[:, 38])
            block[:, 33] = block[:, 1] * (0.15 + 0.85 * geo)
            block[:, 34] = block[:, 5] * (0.15 + 0.85 * geo)
            block[:, 35] = block[:, 1] * (0.2 + 0.8 * cm) * (0.25 + 0.75 * geo)
            block[:, 36] = block[:, 5] * (0.2 + 0.8 * cm) * (0.25 + 0.75 * geo)
        np.savez(fpath, **arrays)
    (ROOT / "reports/geography_fix.json").write_text(
        json.dumps(changes, indent=2), encoding="utf-8"
    )
    print(changes, flush=True)


if __name__ == "__main__":
    main()
