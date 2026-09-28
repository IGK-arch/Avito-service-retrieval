"""Experimental refit of an individual selector, separate from the submission.

Selection uses development Recall@50. Refit can then use all 6000 development
queries. Benchmark query ids never enter model features or historical rules.
This script was not used for the submitted ensemble. Use submit_best.py to
reproduce the answer.csv reported in README.md.
"""

import argparse
import gc
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from pipeline import FEATURES, ROOT, A, R, log, recall
from prepare_data import normalize_text, query_keys
from rank_experiments import augment_features, feature_names
from refine_experiment import model_features


def candidates():
    result = []
    if (R / "model_metrics.json").exists():
        for name, m in json.loads((R / "model_metrics.json").read_text()).items():
            result.append(
                dict(
                    name=name,
                    type="catboost",
                    score=m["validation_recall50"],
                    use_tiny=False,
                    parameters=dict(
                        iterations=700,
                        depth=int(name[-1]),
                        learning_rate=0.07,
                        loss_function="Logloss",
                        random_seed=20260928,
                        thread_count=6,
                        l2_leaf_reg=5,
                        verbose=150,
                        allow_writing_files=False,
                    ),
                )
            )
    if (R / "catboost_depth9_metrics.json").exists():
        m = json.loads((R / "catboost_depth9_metrics.json").read_text())
        result.append(
            dict(
                name="catboost_depth9",
                type="catboost",
                score=m["validation_recall50"],
                parameters=m["parameters"],
                use_tiny=False,
            )
        )
    if (R / "lgbm_metrics.json").exists():
        for name, m in json.loads((R / "lgbm_metrics.json").read_text())[
            "models"
        ].items():
            p = m["parameters"].copy()
            p["num_leaves"] = m["num_leaves"]
            result.append(
                dict(
                    name=name,
                    type="lgbm",
                    score=m["validation_recall50"],
                    parameters=p,
                    use_tiny=False,
                )
            )
    for name in ["refined_remote_depth9", "refined_tiny_depth9"]:
        path = R / f"{name}_metrics.json"
        if path.exists():
            m = json.loads(path.read_text())
            result.append(
                dict(
                    name=name,
                    type="refined",
                    score=m["validation_recall50"],
                    parameters=m["parameters"],
                    use_tiny=m["use_tiny"],
                )
            )
    return result


def transform(X, qi, ii, queries, mode, selected):
    X[:, 15:17] = 0
    if selected["type"] == "catboost":
        return X[:, :41], FEATURES
    if selected["type"] == "lgbm":
        return augment_features(X[:, :41], qi), feature_names()
    return model_features(X, qi, ii, queries, mode, selected["use_tiny"])


def refit(selected):
    suffix = "_tiny" if selected["use_tiny"] else ""
    d = np.load(A / f"eval{suffix}_features.npz")
    X = d["X"]
    qi = d["query_indices"]
    ii = d["item_indices"]
    y = d["labels"]
    queries = pd.read_parquet(A / "eval_queries.parquet")
    rng = np.random.default_rng(20260928)
    hard = (X[:, 1] > 0.5) & ((X[:, 10] > 0) | (X[:, 11] > 0.05) | (X[:, 12] < 4))
    take = (y > 0) | hard | (rng.random(len(y)) < 0.20)
    X, names = transform(X, qi, ii, queries, "eval", selected)
    log(f'Final refit {selected["name"]}: {take.sum()} pairs, {len(names)} features')
    if selected["type"] == "catboost":
        model = CatBoostClassifier(**selected["parameters"])
        model.fit(Pool(X[take], y[take], feature_names=names))
        model.save_model(str(A / "final_selected.cbm"))
    else:
        model = lgb.LGBMRanker(**selected["parameters"])
        model.fit(
            X[take],
            y[take],
            group=np.unique(qi[take], return_counts=True)[1],
            feature_name=names,
        )
        model.booster_.save_model(str(A / "final_selected.txt"))
    selected["features"] = names
    selected["training_queries"] = len(queries)
    selected["training_pairs"] = int(take.sum())
    (A / "final_selection.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    log("Final refit complete")


def reserve_history(queries, items, qi, ii, scores, X, base, legal_ids):
    t = pd.read_parquet(A / "all_interactions.parquet")
    t = t[t.item_id.isin(legal_ids)].copy()
    t["key"] = query_keys(t)
    exact = t.groupby("key")["item_id"].agg(lambda x: list(dict.fromkeys(x)))
    t["text"] = t.search_query.map(normalize_text)
    groups = {s: g for s, g in t.groupby("text")}
    qkeys = query_keys(queries)
    bounds = np.r_[0, np.flatnonzero(qi[1:] != qi[:-1]) + 1, len(qi)]
    itemids = items.item_id.to_numpy()
    out = []
    counts = []
    for q, rec in enumerate(queries.to_dict("records")):
        reserved = []
        key = qkeys.iloc[q]
        text = normalize_text(rec["search_query"])
        if key in exact.index:
            reserved += exact.loc[key][:25]
        if text in groups:
            group = groups[text]
            start, end = bounds[q : q + 2]
            ids = itemids[ii[start:end]]
            block = X[start:end]
            known = set(
                group.loc[
                    group.search_location_id == rec["search_location_id"], "item_id"
                ]
            )
            compatible = (
                (block[:, 10] > 0)
                | (block[:, 11] > 0.1)
                | (block[:, 12] < np.log1p(50))
            )
            known |= set(group.item_id) & set(ids[compatible])
            order = ids[np.argsort(-scores[start:end])]
            extra = [
                i for i in order if i in known and i not in reserved and i in legal_ids
            ]
            reserved += extra[: max(0, 10 - len(reserved))]
        out.append(list(dict.fromkeys(reserved + base[q]))[:50])
        counts.append(len(reserved))
    (R / "history_reservation.json").write_text(
        json.dumps(
            dict(
                queries_with_reserved=sum(x > 0 for x in counts),
                mean_reserved=float(np.mean(counts)),
                max_reserved=max(counts),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def predict(selected, history=True):
    suffix = "_tiny" if selected["use_tiny"] else ""
    d = np.load(A / f"benchmark{suffix}_features.npz")
    X = d["X"]
    qi = d["query_indices"]
    ii = d["item_indices"]
    queries = pd.read_parquet(ROOT / "data/benchmark_queries.parquet")
    items = pd.read_parquet(A / "eval_items.parquet", columns=["item_id"])
    legal_ids = set(
        pd.read_parquet(
            ROOT / "data/benchmark_items.parquet", columns=["item_id"]
        ).item_id
    )
    transformed, names = transform(X, qi, ii, queries, "benchmark", selected)
    assert names == selected["features"], "Final feature schema changed"
    if selected["type"] == "catboost":
        m = CatBoostClassifier()
        m.load_model(str(A / "final_selected.cbm"))
        score = m.predict_proba(transformed)[:, 1]
    else:
        m = lgb.Booster(model_file=str(A / "final_selected.txt"))
        score = m.predict(transformed, num_threads=6)
    del transformed
    gc.collect()
    score[~items.item_id.isin(legal_ids).to_numpy()[ii]] = -1e12
    _, pred = recall(qi, ii, score, queries, items, {}, 50)
    base = [[iid for iid in row if iid in legal_ids] for row in pred]
    pd.DataFrame(
        {"query_id": queries.query_id, "answer": [" ".join(row) for row in base]}
    ).to_csv(ROOT / "answer_without_history.csv", index=False, encoding="utf-8")
    if history:
        pred = reserve_history(queries, items, qi, ii, score, X, base, legal_ids)
    else:
        pred = base
    pd.DataFrame(
        {"query_id": queries.query_id, "answer": [" ".join(row) for row in pred]}
    ).to_csv(ROOT / "answer.csv", index=False, encoding="utf-8")
    log(f'answer.csv generated by {selected["name"]}; {len(queries)} rows')

    # The standalone validator is run by the caller, avoiding argparse collision.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selector")
    p.add_argument("--predict-only", action="store_true")
    p.add_argument("--no-history", action="store_true")
    args = p.parse_args()
    if args.predict_only:
        selected = json.loads((A / "final_selection.json").read_text())
    else:
        choices = candidates()
        selected = (
            next(x for x in choices if x["name"] == args.selector)
            if args.selector
            else max(choices, key=lambda x: x["score"])
        )
        refit(selected)
    predict(selected, not args.no_history)


if __name__ == "__main__":
    main()
