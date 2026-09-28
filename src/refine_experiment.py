"""Test learned Russian embeddings and explicit remote-service context.

This is our implementation of a grouped candidate selector. Fixed model
parameters are measured on the same item/text-disjoint development split.
"""

import gc
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
from scipy.stats import rankdata

from pipeline import TINY_FEATURES, A, R, log, recall
from rank_experiments import augment_features, feature_names, wait_for_memory
from remote_features import build_remote_features

REMOTE_NAMES = [
    "query_online",
    "query_remote_wording",
    "query_nationwide",
    "item_remote",
    "item_nationwide",
    "query_nonlocal_prior",
    "remote_aware_bm25",
]


def remote_items_frame(path):
    """Read short remote text prefixes without a ~1 GB Arrow string buffer."""
    items = pd.read_parquet(
        path, columns=["item_id", "item_location_id", "item_microcat_id"]
    )
    columns = ["item_title_raw", "item_infm_params_text", "item_description_raw"]
    texts = {column: [] for column in columns}
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=4096, columns=columns, use_threads=False
    ):
        for column in columns:
            values = batch.column(column).to_pylist()
            if column == "item_title_raw":
                texts[column].extend(values)
            else:
                texts[column].extend(
                    value[:1000] if isinstance(value, str) else value
                    for value in values
                )
            del values
    for column in columns:
        assert len(texts[column]) == len(items), "Remote text/corpus row order mismatch"
        items[column] = texts[column]
    return items


def model_features(X, qi, ii, queries, mode, use_tiny):
    # Category probabilities are already saved; loading the unused history
    # object would consume memory while other retrieval jobs are running.
    interactions = pd.read_parquet(
        A
        / ("fit_interactions.parquet" if mode == "eval" else "all_interactions.parquet")
    )
    items = remote_items_frame(A / "eval_items.parquet")
    prob = np.load(A / f"{mode}_category_probabilities.npy")
    classes = np.load(A / "category_classes.npy")
    bundle = build_remote_features(
        queries, items, interactions, microcat_probabilities=prob, microcat_ids=classes
    )
    del interactions, items, prob
    gc.collect()
    X[:, 15:17] = 0
    base = augment_features(X[:, :41], qi)
    qf = bundle.query_flags[qi]
    iflags = bundle.item_flags[ii]
    prior = bundle.query_remote_prior[qi]
    # Explicit distance-free services can expose a high text score even when
    # the geography-adjusted score is small. This remains a learned feature.
    openness = np.maximum(np.max(qf, axis=1), prior.ravel())
    remote_score = X[:, 1] * np.maximum(openness, X[:, 33] / np.maximum(X[:, 1], 1e-6))
    extra = np.column_stack([qf, iflags, prior, remote_score]).astype(np.float32)
    names = feature_names() + REMOTE_NAMES
    if use_tiny:
        tiny = X[:, 41:45]
        bounds = np.r_[0, np.flatnonzero(qi[1:] != qi[:-1]) + 1, len(qi)]
        aux = np.empty((len(qi), 6), np.float32)
        for start, end in zip(bounds[:-1], bounds[1:]):
            for j, col in enumerate([0, 2]):
                s = tiny[start:end, col]
                ordered = np.sort(s)[::-1]
                aux[start:end, 3 * j] = 1 - (rankdata(-s, method="min") - 1) / max(
                    len(s) - 1, 1
                )
                aux[start:end, 3 * j + 1] = s - ordered[0]
                aux[start:end, 3 * j + 2] = s - ordered[min(49, len(s) - 1)]
        extra = np.column_stack([extra, tiny, aux]).astype(np.float32)
        names += TINY_FEATURES + [
            "tiny_pool_rank",
            "tiny_gap_best",
            "tiny_gap50",
            "tiny_geo_pool_rank",
            "tiny_geo_gap_best",
            "tiny_geo_gap50",
        ]
    # Disk-backed output avoids an additional anonymous 1.5-GB allocation while
    # source X and 62-column augmentation coexist under Windows commit limits.
    result = np.lib.format.open_memmap(
        A / f'{mode}_refined_{"tiny" if use_tiny else "remote"}_features.npy',
        mode="w+",
        dtype=np.float32,
        shape=(len(X), len(names)),
    )
    for start in range(0, len(X), 250000):
        result[start : start + 250000, : base.shape[1]] = base[start : start + 250000]
        result[start : start + 250000, base.shape[1] :] = extra[start : start + 250000]
    result.flush()
    assert result.shape[1] == len(names), (result.shape, len(names))
    return result, names


def train(use_tiny=False):
    wait_for_memory(minimum_gb=4.0)
    log(
        f"Remote refinement loading: free RAM {psutil.virtual_memory().available/2**30:.2f} GiB"
    )
    name = "refined_tiny_depth9" if use_tiny else "refined_remote_depth9"
    path = A / ("eval_tiny_features.npz" if use_tiny else "eval_features.npz")
    data = np.load(path)
    X = data["X"]
    qi = data["query_indices"]
    ii = data["item_indices"]
    y = data["labels"]
    queries = pd.read_parquet(A / "eval_queries.parquet")
    baseX = X
    X, names = model_features(X, qi, ii, queries, "eval", use_tiny)
    log(
        f"Remote refinement features: {X.shape}; free RAM {psutil.virtual_memory().available/2**30:.2f} GiB"
    )
    rng = np.random.default_rng(20260928)
    tr = queries.split.to_numpy()[qi] == "ranker_train"
    hard = (baseX[:, 1] > 0.5) & (
        (baseX[:, 10] > 0) | (baseX[:, 11] > 0.05) | (baseX[:, 12] < 4)
    )
    take = tr & ((y > 0) | hard | (rng.random(len(y)) < 0.20))
    group = np.unique(qi[take], return_counts=True)[1]
    # The old source matrix is no longer needed after the sampling mask.
    del baseX, hard, data
    gc.collect()
    wait_for_memory(minimum_gb=4.0)
    params = dict(
        objective="lambdarank",
        n_estimators=1200,
        learning_rate=0.035,
        num_leaves=127,
        max_depth=9,
        reg_lambda=10,
        reg_alpha=0.05,
        min_child_samples=40,
        max_bin=127,
        colsample_bytree=0.9,
        subsample=0.85,
        subsample_freq=1,
        lambdarank_truncation_level=50,
        n_jobs=6,
        verbosity=-1,
        random_state=20260928,
    )
    log(f"Training {name}: {take.sum()} pairs, {X.shape[1]} features")
    start = time.time()
    model = lgb.LGBMRanker(**params)
    last_report = [time.time()]

    def progress(environment):
        if time.time() - last_report[0] >= 60:
            log(
                f'{name}: iteration {environment.iteration+1}/{params["n_estimators"]}, elapsed {time.time()-start:.1f}s; RAM {psutil.virtual_memory().available/2**30:.2f} GiB'
            )
            last_report[0] = time.time()

    model.fit(X[take], y[take], group=group, feature_name=names, callbacks=[progress])
    model.booster_.save_model(str(A / f"{name}.txt"))
    log(f"{name}: fit complete, scoring candidate pools")
    score = np.empty(len(X), np.float32)
    for chunk in range(0, len(X), 250000):
        score[chunk : chunk + 250000] = model.booster_.predict(
            X[chunk : chunk + 250000], num_threads=6
        ).astype(np.float32)
    np.save(A / f"{name}_scores.npy", score)
    items = pd.read_parquet(A / "eval_items.parquet", columns=["item_id"])
    targets = json.loads((A / "eval_targets.json").read_text(encoding="utf-8"))
    per, _ = recall(qi, ii, score, queries, items, targets)
    val = queries.split.to_numpy() == "validation"
    cold = queries.cold_text.to_numpy()
    result = dict(
        name=name,
        validation_recall50=float(per[val].mean()),
        cold_recall50=float(per[val & cold].mean()),
        warm_recall50=float(per[val & ~cold].mean()),
        train_recall50=float(per[~val].mean()),
        runtime_seconds=time.time() - start,
        parameters=params,
        features=names,
        use_tiny=use_tiny,
    )
    (R / f"{name}_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    log(
        f'{name}: val={result["validation_recall50"]:.6f}; cold={result["cold_recall50"]:.6f}'
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--use-tiny", action="store_true")
    args = p.parse_args()
    train(args.use_tiny)
