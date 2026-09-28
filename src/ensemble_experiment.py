"""Compare light CB6/CB8 ensembles on existing held-out candidate pools.

No retraining, benchmark reads or feature-X loading is needed. Only ordered
candidate positions, held-out labels and saved probabilities are consumed.
The selected model is never changed; an improving ensemble gets its own scores
and a reproducible component recipe for the parent pipeline to consider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]


def memory_guard() -> float:
    available = psutil.virtual_memory().available / 2**30
    if available < 3.0:
        raise MemoryError(f"Less than 3 GiB free RAM: {available:.2f} GiB")
    return float(available)


def top(scores, k=50):
    # Match pipeline.top's boundary/tie convention so baseline comparison is
    # exact, rather than introducing a different tie-breaking experiment.
    k = min(k, len(scores))
    if k == 0:
        return np.empty(0, np.int64)
    chosen = np.argpartition(scores, len(scores) - k)[-k:]
    return chosen[np.lexsort((chosen, -scores[chosen]))]


def run(artifacts: Path, reports: Path) -> dict:
    initial_memory = memory_guard()
    with np.load(artifacts / "eval_features.npz", allow_pickle=False) as archive:
        qi, ii = archive["query_indices"], archive["item_indices"]
    queries = pd.read_parquet(artifacts / "eval_queries.parquet")
    item_ids = pd.read_parquet(
        artifacts / "eval_items.parquet", columns=["item_id"]
    ).item_id.to_numpy()
    targets = json.loads((artifacts / "eval_targets.json").read_text(encoding="utf-8"))
    p6 = np.load(artifacts / "catboost_depth6_scores.npy", mmap_mode="r")
    p8 = np.load(artifacts / "catboost_depth8_scores.npy", mmap_mode="r")
    if p6.shape != p8.shape or p6.ndim != 1 or len(p6) != len(qi) or len(ii) != len(qi):
        raise ValueError("Candidate arrays and model scores are not aligned")
    if not np.all(qi[:-1] <= qi[1:]):
        raise ValueError("Expected candidates grouped by query")
    counts = np.bincount(qi, minlength=len(queries))
    offsets = np.r_[0, np.cumsum(counts)]
    if len(counts) != len(queries) or np.any(counts == 0):
        raise ValueError("Every eval query must have one candidate group")
    if np.any(ii < 0) or np.any(ii >= len(item_ids)):
        raise ValueError("Invalid item positions")
    validation = queries.split.to_numpy() == "validation"
    cold = queries.cold_text.to_numpy(dtype=bool)
    relevant = [set(targets[str(int(key))]) for key in queries.qkey]
    ceiling = np.asarray(
        [
            len(set(item_ids[ii[start:end]]) & rel) / len(rel)
            for start, end, rel in zip(offsets[:-1], offsets[1:], relevant)
        ]
    )

    def evaluate(scores):
        recalls = np.zeros(len(queries), np.float64)
        for q, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            prediction = set(item_ids[ii[start:end][top(scores[start:end])]])
            recalls[q] = len(prediction & relevant[q]) / len(relevant[q])
        return recalls

    def metrics(values, base=None):
        result = {
            "validation_recall50": float(values[validation].mean()),
            "cold_recall50": float(values[validation & cold].mean()),
            "warm_recall50": float(values[validation & ~cold].mean()),
            "ranker_train_recall50": float(values[~validation].mean()),
        }
        if base is not None:
            differences = (values - base)[validation]
            standard_error = float(differences.std(ddof=1) / np.sqrt(len(differences)))
            result.update(
                {
                    "delta_vs_cb6": float(differences.mean()),
                    "queries_improved": int(np.sum(differences > 1e-12)),
                    "queries_worsened": int(np.sum(differences < -1e-12)),
                    "paired_delta_standard_error": standard_error,
                }
            )
        return result

    base6, base8 = evaluate(p6), evaluate(p8)
    results = {
        "catboost_depth6": metrics(base6),
        "catboost_depth8": metrics(base8, base6),
    }
    best_name, best_values, best_recipe = None, base6, None
    best_validation = float(base6[validation].mean())
    best_scores = None
    recipes = {}

    def consider(name, scores, recipe):
        nonlocal best_name, best_values, best_recipe, best_validation, best_scores
        memory_guard()
        values = evaluate(scores)
        results[name], recipes[name] = metrics(values, base6), recipe
        current = results[name]["validation_recall50"]
        print(name, json.dumps(results[name]), flush=True)
        if current > best_validation + 1e-12:
            best_name, best_values, best_recipe, best_validation = (
                name,
                values,
                recipe,
                current,
            )
            best_scores = np.array(scores, copy=True)

    for weight6 in [0.25, 0.50, 0.75]:
        consider(
            f"probability_cb6_{weight6:.2f}",
            weight6 * p6 + (1 - weight6) * p8,
            {
                "kind": "probability_average",
                "components": {
                    "catboost_depth6": weight6,
                    "catboost_depth8": 1 - weight6,
                },
            },
        )
    # An average in log-odds space behaves differently near 0/1 probabilities.
    clipped6, clipped8 = np.clip(p6, 1e-7, 1 - 1e-7), np.clip(p8, 1e-7, 1 - 1e-7)
    logits = 0.5 * (
        np.log(clipped6 / (1 - clipped6)) + np.log(clipped8 / (1 - clipped8))
    )
    del clipped6, clipped8
    consider(
        "logit_average_50_50",
        logits,
        {
            "kind": "logit_average",
            "components": {"catboost_depth6": 0.5, "catboost_depth8": 0.5},
            "clip": 1e-7,
        },
    )
    del logits
    rank6 = np.zeros(len(qi), np.int32)
    rank8 = np.zeros(len(qi), np.int32)
    for start, end in zip(offsets[:-1], offsets[1:]):
        for p, rank in [(p6, rank6), (p8, rank8)]:
            order = np.lexsort((np.arange(end - start), -p[start:end]))
            rank[start + order] = np.arange(1, end - start + 1)
    for constant in [20, 60]:
        fused = 0.5 / (constant + rank6) + 0.5 / (constant + rank8)
        consider(
            f"rrf_constant_{constant}",
            fused,
            {
                "kind": "reciprocal_rank_fusion",
                "components": {"catboost_depth6": 0.5, "catboost_depth8": 0.5},
                "constant": constant,
            },
        )
        del fused
    report = {
        "validation_queries": int(validation.sum()),
        "cold_validation_queries": int((validation & cold).sum()),
        "candidate_pairs": len(qi),
        "initial_free_memory_gb": initial_memory,
        "final_free_memory_gb": memory_guard(),
        "candidate_recall_ceiling": float(ceiling[validation].mean()),
        "results": results,
        "recipes": recipes,
        "improving_ensemble": best_name,
        "ordered_query_keys_sha256": hashlib.sha256(
            "\n".join(queries.qkey.astype(str)).encode()
        ).hexdigest(),
        "ordered_item_ids_sha256": hashlib.sha256(
            "\n".join(item_ids).encode()
        ).hexdigest(),
        "note": "Selected model unchanged. Validation differences share the same candidate union; this is a hyperparameter experiment, not an independent test.",
    }
    if best_name is not None:
        np.save(artifacts / "cb_ensemble_scores.npy", best_scores)
        metadata = {
            "name": best_name,
            "recipe": best_recipe,
            "scores": "cb_ensemble_scores.npy",
            "validation_metrics": metrics(best_values, base6),
            "ordered_query_keys_sha256": report["ordered_query_keys_sha256"],
            "ordered_item_ids_sha256": report["ordered_item_ids_sha256"],
        }
        (artifacts / "cb_ensemble.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    reports.mkdir(exist_ok=True)
    (reports / "ensemble_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    report = run(args.artifacts, args.reports)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
