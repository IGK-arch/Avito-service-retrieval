"""Independent grouped LambdaRank comparison on our fixed query holdout.

This code compares models using the generated candidate features and labels.
LambdaRank is a standard grouped ranking objective:
https://lightgbm.readthedocs.io/en/stable/Parameters.html#objective
Query-level score ranks/gaps are generic retrieval metadata. Numeric query/item
indices are used for grouping and joins and are never prediction features.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from common import FEATURES, recall
from ranking_features import (
    SOURCES as SOURCES,
)
from ranking_features import (
    TRANSFORMS as TRANSFORMS,
)
from ranking_features import (
    augment_features as augment_features,
)
from ranking_features import (
    feature_names as feature_names,
)
from ranking_features import (
    query_boundaries as query_boundaries,
)
from ranking_features import (
    reciprocal_scores as reciprocal_scores,
)
from ranking_features import (
    standardize_scores as standardize_scores,
)

ROOT = Path(__file__).resolve().parents[1]
A, R = ROOT / "artifacts", ROOT / "reports"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def log(message):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} LGBM {message}'
    print(line, flush=True)
    with (R / "lgbm_run.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def available_memory_bytes():
    """Windows allocations need commit capacity as well as physical free RAM.

    GetPerformanceInfo fields are documented at Microsoft's Win32 API page:
    https://learn.microsoft.com/windows/win32/api/psapi/ns-psapi-performance_information
    Other platforms use psutil's physical available memory directly.
    """
    available = int(psutil.virtual_memory().available)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PerformanceInformation(ctypes.Structure):
            _fields_ = (
                [("cb", wintypes.DWORD)]
                + [
                    (name, ctypes.c_size_t)
                    for name in [
                        "CommitTotal",
                        "CommitLimit",
                        "CommitPeak",
                        "PhysicalTotal",
                        "PhysicalAvailable",
                        "SystemCache",
                        "KernelTotal",
                        "KernelPaged",
                        "KernelNonpaged",
                        "PageSize",
                    ]
                ]
                + [
                    (name, wintypes.DWORD)
                    for name in ["HandleCount", "ProcessCount", "ThreadCount"]
                ]
            )

        info = PerformanceInformation()
        info.cb = ctypes.sizeof(info)
        function = ctypes.windll.psapi.GetPerformanceInfo
        function.argtypes = [ctypes.POINTER(PerformanceInformation), wintypes.DWORD]
        function.restype = wintypes.BOOL
        if function(ctypes.byref(info), info.cb):
            commit_available = max(
                0, int(info.CommitLimit - info.CommitTotal) * int(info.PageSize)
            )
            available = min(available, commit_available)
    return available


def wait_for_memory(minimum_gb=4.0):
    last_message = 0.0
    while available_memory_bytes() < minimum_gb * 2**30:
        if time.time() - last_message > 60:
            log(
                f"Waiting for physical/commit memory: {available_memory_bytes() / 2**30:.2f} GB usable"
            )
            last_message = time.time()
        time.sleep(10)


def summarize(per_query, queries):
    validation = queries.split.eq("validation").to_numpy()
    cold = queries.cold_text.to_numpy()
    return {
        "validation_recall50": float(per_query[validation].mean()),
        "cold_recall50": float(per_query[validation & cold].mean()),
        "warm_recall50": float(per_query[validation & ~cold].mean()),
        "ranker_train_recall50": float(per_query[~validation].mean()),
    }


def compare_ensembles():
    """Compare a few fixed margin/rank mixtures after standalone models finish."""
    wait_for_memory()
    with np.load(A / "eval_features.npz") as archive:
        qi, ii = archive["query_indices"], archive["item_indices"]
    queries = pd.read_parquet(A / "eval_queries.parquet")
    items = pd.read_parquet(A / "eval_items.parquet", columns=["item_id"])
    targets = json.loads((A / "eval_targets.json").read_text(encoding="utf-8"))
    lgbm_results = json.loads((R / "lgbm_metrics.json").read_text(encoding="utf-8"))[
        "models"
    ]
    lgbm_name = max(
        lgbm_results, key=lambda name: lgbm_results[name]["validation_recall50"]
    )
    cb_results = json.loads((R / "model_metrics.json").read_text(encoding="utf-8"))
    if (R / "catboost_depth9_metrics.json").exists():
        cb_results["catboost_depth9"] = json.loads(
            (R / "catboost_depth9_metrics.json").read_text(encoding="utf-8")
        )
    cb_name = max(cb_results, key=lambda name: cb_results[name]["validation_recall50"])
    lb = np.load(A / f"{lgbm_name}_scores.npy", mmap_mode="r")
    cb = np.load(A / f"{cb_name}_scores.npy", mmap_mode="r")
    if lb.shape != cb.shape or len(lb) != len(qi):
        raise ValueError("Model scores and candidate rows are not aligned")
    lb_per, _ = recall(qi, ii, lb, queries, items, targets)
    cb_per, _ = recall(qi, ii, cb, queries, items, targets)
    results = {
        "base_models": {"lightgbm": lgbm_name, "catboost": cb_name},
        "results": {
            lgbm_name: summarize(lb_per, queries),
            cb_name: summarize(cb_per, queries),
        },
        "recipes": {},
        "note": "Exploratory mixtures on the same validation; standalone selected model unchanged.",
    }
    validation = queries.split.eq("validation").to_numpy()
    best = float(lb_per[validation].mean())
    best_scores, best_name = None, None
    zlb, zcb = standardize_scores(lb, qi), standardize_scores(cb, qi)
    configurations = [
        (
            f"query_zscore_lgbm_{weight:.2f}",
            weight * zlb + (1 - weight) * zcb,
            {"kind": "query_zscore", "lightgbm_weight": weight},
        )
        for weight in [0.5, 0.75, 0.9]
    ]
    rlb, rcb = reciprocal_scores(lb, qi), reciprocal_scores(cb, qi)
    configurations.append(
        (
            "rrf_lgbm_0.75",
            0.75 * rlb + 0.25 * rcb,
            {"kind": "reciprocal_rank", "lightgbm_weight": 0.75, "constant": 60.0},
        )
    )
    for name, scores, recipe in configurations:
        per, _ = recall(qi, ii, scores, queries, items, targets)
        difference = (per - lb_per)[validation]
        results["results"][name] = {
            **summarize(per, queries),
            "delta_vs_lgbm": float(difference.mean()),
            "paired_delta_standard_error": float(
                difference.std(ddof=1) / np.sqrt(len(difference))
            ),
            "queries_improved": int((difference > 1e-12).sum()),
            "queries_worsened": int((difference < -1e-12).sum()),
        }
        results["recipes"][name] = {
            **recipe,
            "lightgbm_model": lgbm_name,
            "catboost_model": cb_name,
        }
        log(f'Ensemble {name}: {results["results"][name]}')
        if results["results"][name]["validation_recall50"] > best + 1e-12:
            best = results["results"][name]["validation_recall50"]
            best_name, best_scores = name, scores
    results["improving_ensemble"] = best_name
    if best_name is not None:
        np.save(A / "lgbm_cb_ensemble_scores.npy", best_scores)
        write_json(
            A / "lgbm_cb_ensemble.json",
            {
                "name": best_name,
                "recipe": results["recipes"][best_name],
                "metrics": results["results"][best_name],
            },
        )
    write_json(R / "lgbm_cb_ensemble_metrics.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--variant", choices=["all", "base", "depth9"], default="all")
    parser.add_argument("--ensemble-only", action="store_true")
    args = parser.parse_args()
    if args.ensemble_only:
        print(json.dumps(compare_ensembles(), indent=2), flush=True)
        return
    import lightgbm as lgb

    started = time.time()
    wait_for_memory()
    path = A / "eval_features.npz"
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(
            "Root must confirm completed and corrected eval_features before this run"
        )
    log(
        f"Loading completed features; available RAM {psutil.virtual_memory().available / 2**30:.2f} GB"
    )
    with np.load(path) as data:
        X = data["X"]
        qi, ii, labels = data["query_indices"], data["item_indices"], data["labels"]
    X[:, 15:17] = 0  # Match our CatBoost baseline: artificial history signal disabled.
    queries = pd.read_parquet(A / "eval_queries.parquet")
    items = pd.read_parquet(A / "eval_items.parquet", columns=["item_id"])
    targets = json.loads((A / "eval_targets.json").read_text(encoding="utf-8"))
    boundaries = query_boundaries(qi)
    assert len(boundaries) - 1 == len(
        queries
    ), "Missing/noncontiguous feature query groups"
    assert np.array_equal(
        np.unique(qi), np.arange(len(queries))
    ), "Unexpected feature query indices"
    augmented = augment_features(X, qi)
    del X
    X = augmented
    del augmented
    gc.collect()
    log(
        f"Augmented features: {X.shape}; RAM free {psutil.virtual_memory().available / 2**30:.2f} GB"
    )
    metadata = {
        "base_features": FEATURES,
        "feature_names": feature_names(),
        "sources": SOURCES,
        "transforms": TRANSFORMS,
        "history_columns_disabled": [15, 16],
        "gap_reference": "current query candidate pool; 50th if available, else last",
        "query_indices_are_features": False,
        "api": "rank_experiments.augment_features(X, query_indices)",
        "lightgbm_version": lgb.__version__,
        "features_file_bytes": path.stat().st_size,
        "features_file_mtime_ns": path.stat().st_mtime_ns,
        "candidate_pool": "fixed pool generated before coordinate-center correction; distance features corrected",
    }
    write_json(A / "lgbm_transform.json", metadata)
    train_queries = queries.split.eq("ranker_train").to_numpy()
    is_train = train_queries[qi]
    rng = np.random.default_rng(20260928)
    hard = (X[:, 1] > 0.5) & ((X[:, 10] > 0) | (X[:, 11] > 0.05) | (X[:, 12] < 4))
    selected = is_train & ((labels > 0) | hard | (rng.random(len(labels)) < 0.20))
    del hard, is_train
    assert not np.any(queries.split.eq("validation").to_numpy()[qi[selected]])
    X_train, y_train = X[selected], labels[selected]
    _, group_sizes = np.unique(qi[selected], return_counts=True)
    assert int(group_sizes.sum()) == len(X_train)
    retrieved = np.add.reduceat(labels.astype(np.float32), boundaries[:-1])
    target_counts = np.asarray(
        [len(set(targets[str(int(key))])) for key in queries.qkey]
    )
    ceiling = retrieved / target_counts
    results = {
        "candidate_ceiling": summarize(ceiling, queries),
        "models": {},
        "training_rows": int(selected.sum()),
        "training_positives": int(y_train.sum()),
        "training_queries": int(len(group_sizes)),
        "seed": 20260928,
        "augmentation": metadata,
        "evaluation": "1500 external validation groups; no eval_set/early stopping uses validation",
        "limitations": [
            "Selection among fixed models on this validation is exploratory.",
            "Local corpus includes heldout positives and can underrepresent benchmark competitor density.",
        ],
    }
    if args.variant == "depth9" and (R / "lgbm_metrics.json").exists():
        previous = json.loads((R / "lgbm_metrics.json").read_text(encoding="utf-8"))
        # Keep earlier measured configurations unchanged when adding this run.
        if (
            previous["augmentation"]["features_file_mtime_ns"]
            != path.stat().st_mtime_ns
        ):
            raise RuntimeError("Existing comparison used a different features artifact")
        results["models"].update(previous["models"])
    log(
        f"Training {len(group_sizes)} queries / {len(X_train)} rows / {int(y_train.sum())} positives"
    )
    parameters = dict(
        objective="lambdarank",
        n_estimators=800,
        learning_rate=0.045,
        reg_lambda=5.0,
        min_child_samples=30,
        n_jobs=args.threads,
        lambdarank_truncation_level=50,
        random_state=20260928,
        verbosity=-1,
        metric="None",
    )
    configurations = [
        dict(name="lgbm31", parameters=dict(num_leaves=31)),
        dict(name="lgbm63", parameters=dict(num_leaves=63)),
        dict(
            name="lgbm_depth9",
            parameters=dict(
                max_depth=9,
                num_leaves=127,
                n_estimators=1200,
                learning_rate=0.035,
                reg_lambda=10.0,
                reg_alpha=0.05,
                min_child_samples=40,
                max_bin=127,
                colsample_bytree=0.9,
                subsample=0.85,
                subsample_freq=1,
            ),
        ),
    ]
    configurations = [
        config
        for config in configurations
        if args.variant == "all"
        or (args.variant == "base" and config["name"] != "lgbm_depth9")
        or (args.variant == "depth9" and config["name"] == "lgbm_depth9")
    ]
    for config in configurations:
        wait_for_memory()
        name = config["name"]
        model_parameters = {**parameters, **config["parameters"]}
        model_started = time.time()
        log(f"{name}: fit started with {args.threads} threads")
        model = lgb.LGBMRanker(**model_parameters)
        last_report = [time.time()]

        def report_progress(environment):
            if time.time() - last_report[0] >= 60:
                log(
                    f'{name}: iteration {environment.iteration+1}/{model_parameters["n_estimators"]}, elapsed {time.time()-model_started:.1f}s'
                )
                write_json(
                    R / "lgbm_progress.json",
                    {
                        "model": name,
                        "iteration": environment.iteration + 1,
                        "elapsed_seconds": time.time() - model_started,
                    },
                )
                last_report[0] = time.time()

        model.fit(
            X_train,
            y_train,
            group=group_sizes,
            feature_name=feature_names(),
            callbacks=[report_progress],
        )
        model.booster_.save_model(str(A / f"{name}_model.txt"))
        log(
            f"{name}: fit finished in {time.time()-model_started:.1f}s; predicting all query pools"
        )
        scores = np.empty(len(X), dtype=np.float32)
        for start in range(0, len(X), 250000):
            scores[start : start + 250000] = model.booster_.predict(
                X[start : start + 250000], num_threads=args.threads
            ).astype(np.float32)
        per_query, _ = recall(qi, ii, scores, queries, items, targets)
        np.save(A / f"{name}_scores.npy", scores)
        np.save(A / f"{name}_recall_per_query.npy", per_query)
        results["models"][name] = {
            **summarize(per_query, queries),
            "elapsed_seconds": time.time() - model_started,
            "num_leaves": model_parameters["num_leaves"],
            "max_depth": model_parameters.get("max_depth", -1),
            "parameters": model_parameters,
        }
        write_json(R / "lgbm_metrics.json", results)
        best = max(
            results["models"],
            key=lambda key: results["models"][key]["validation_recall50"],
        )
        write_json(
            A / "lgbm_selected_model.json",
            {
                "name": best,
                "model": f"{best}_model.txt",
                "transform": "lgbm_transform.json",
                "feature_names": feature_names(),
                "parameters": {
                    **results["models"][best]["parameters"],
                    "num_leaves": results["models"][best]["num_leaves"],
                },
            },
        )
        log(f'{name}: {results["models"][name]}')
        del model, scores
        gc.collect()
    results["elapsed_seconds"] = time.time() - started
    write_json(R / "lgbm_metrics.json", results)
    log(f'Comparison completed in {results["elapsed_seconds"]:.1f}s')


if __name__ == "__main__":
    main()
