"""A depth-nine selector experiment requested by the user.

Greater capacity is paired with regularization and feature/row subsampling.
Validation labels are used only to measure Recall, never for fitting.
"""

import json
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from pipeline import FEATURES, A, R, log, recall


def main():
    d = np.load(A / "eval_features.npz")
    X = d["X"]
    y = d["labels"]
    qi = d["query_indices"]
    ii = d["item_indices"]
    X[:, 15:17] = 0
    queries = pd.read_parquet(A / "eval_queries.parquet")
    items = pd.read_parquet(A / "eval_items.parquet", columns=["item_id"])
    targets = json.loads((A / "eval_targets.json").read_text(encoding="utf-8"))
    tr = queries.split.to_numpy()[qi] == "ranker_train"
    rng = np.random.default_rng(20260928)
    hard = (X[:, 1] > 0.5) & ((X[:, 10] > 0) | (X[:, 11] > 0.05) | (X[:, 12] < 4))
    take = tr & ((y > 0) | hard | (rng.random(len(y)) < 0.20))
    params = dict(
        iterations=1200,
        depth=9,
        learning_rate=0.035,
        loss_function="Logloss",
        l2_leaf_reg=15,
        random_strength=0.5,
        bootstrap_type="Bernoulli",
        subsample=0.85,
        rsm=0.9,
        border_count=128,
        random_seed=20260928,
        thread_count=6,
        verbose=150,
        allow_writing_files=False,
    )
    log(f"Training requested depth9 CatBoost: {take.sum()} pairs")
    start = time.time()
    model = CatBoostClassifier(**params)
    model.fit(Pool(X[take], y[take], feature_names=FEATURES))
    model.save_model(str(A / "catboost_depth9.cbm"))
    scores = model.predict_proba(X)[:, 1]
    np.save(A / "catboost_depth9_scores.npy", scores)
    per, _ = recall(qi, ii, scores, queries, items, targets)
    val = queries.split.to_numpy() == "validation"
    cold = queries.cold_text.to_numpy()
    result = dict(
        validation_recall50=float(per[val].mean()),
        cold_recall50=float(per[val & cold].mean()),
        warm_recall50=float(per[val & ~cold].mean()),
        train_recall50=float(per[~val].mean()),
        runtime_seconds=time.time() - start,
        parameters=params,
    )
    (R / "catboost_depth9_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    log(f"CatBoost depth9 result: {result}")


if __name__ == "__main__":
    main()
