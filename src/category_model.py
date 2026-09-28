"""Learn query intent with aggregated multinomial/complement Naive Bayes.

Training examples are never expanded into duplicate sparse text vectors. A
microcategory-by-query count matrix multiplies HistoryIndex.matrix, then a
second small channel contributes query filter text. Held-out item/text events
are absent from fit_interactions. Final benchmark counts use all_interactions.
Models are plain dictionaries, so joblib loading does not depend on __main__.
"""

import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from history import norm

ROOT = Path(__file__).resolve().parents[1]
FILTER_WEIGHT = 0.35


def transform(model, queries):
    texts = queries.search_query.fillna("").map(norm).tolist()
    text_matrix = normalize(
        sparse.hstack(
            [
                model["word"].transform(texts) * 0.8,
                model["char"].transform(texts) * 0.6,
            ],
            format="csr",
        )
    )
    filters = queries.search_infm_params_text.fillna("").map(norm).tolist()
    filter_matrix = model["filter"].transform(filters) * model["filter_weight"]
    return sparse.hstack([text_matrix, filter_matrix], format="csr", dtype=np.float32)


def predict_proba(model, queries, batch_size=256):
    """Rows follow queries order; columns follow model['classes'] exactly."""
    matrix = transform(model, queries)
    output = np.empty((len(queries), len(model["classes"])), dtype=np.float32)
    for start in range(0, len(queries), batch_size):
        scores = matrix[start : start + batch_size] @ model["weights"].T
        scores = np.asarray(scores, dtype=np.float32)
        scores += model["intercept"]
        output[start : start + batch_size] = softmax(
            scores / model["temperature"], axis=1
        )
    return output


def aggregate_counts(interactions, history, filter_vectorizer, classes):
    """Aggregate label counts before multiplying text features (memory <2GB)."""
    texts = interactions.search_query.fillna("").map(norm)
    unique_texts = texts.unique().tolist()
    text_map = {text: i for i, text in enumerate(unique_texts)}
    # Reuse the cached matrix where possible; only final full-fit unseen texts
    # need another word/character transform in the same fixed vocabulary.
    known = np.asarray([history.textmap.get(t, -1) for t in unique_texts])
    if (known >= 0).all():
        text_matrix = history.matrix[known]
    else:
        text_matrix = normalize(
            sparse.hstack(
                [
                    history.word.transform(unique_texts) * 0.8,
                    history.char.transform(unique_texts) * 0.6,
                ],
                format="csr",
            )
        )
    class_map = {int(c): i for i, c in enumerate(classes)}
    labels = interactions.item_microcat_id.map(class_map).to_numpy(np.int32)
    text_ids = texts.map(text_map).to_numpy(np.int32)
    ones = np.ones(len(interactions), dtype=np.float32)
    incidence = sparse.csr_matrix(
        (ones, (labels, text_ids)), shape=(len(classes), len(unique_texts))
    )
    text_counts = incidence @ text_matrix
    filters = interactions.search_infm_params_text.fillna("").map(norm)
    unique_filters = filters.unique().tolist()
    filter_map = {text: i for i, text in enumerate(unique_filters)}
    filter_ids = filters.map(filter_map).to_numpy(np.int32)
    filter_incidence = sparse.csr_matrix(
        (ones, (labels, filter_ids)), shape=(len(classes), len(unique_filters))
    )
    filter_counts = filter_incidence @ filter_vectorizer.transform(unique_filters)
    counts = sparse.hstack(
        [text_counts, filter_counts * FILTER_WEIGHT], format="csr", dtype=np.float32
    ).toarray()
    return counts, np.bincount(labels, minlength=len(classes)).astype(np.float32)


def fit_weights(counts, class_counts, kind, alpha):
    if kind == "multinomial":
        values = counts + np.float32(alpha)
        weights = np.log(values)
        weights -= np.log(values.sum(axis=1, dtype=np.float64)).astype(np.float32)[
            :, None
        ]
        intercept = np.log(
            (class_counts + 1) / (class_counts.sum() + len(class_counts))
        )
    else:
        values = counts.sum(axis=0, keepdims=True) - counts + np.float32(alpha)
        weights = -np.log(values)
        weights += np.log(values.sum(axis=1, dtype=np.float64)).astype(np.float32)[
            :, None
        ]
        intercept = np.zeros(len(class_counts), dtype=np.float32)
    return weights.astype(np.float32, copy=False), intercept.astype(np.float32)


def category_metrics(probabilities, labels, mask):
    rows = np.flatnonzero(mask)
    order = np.argsort(-probabilities[rows], axis=1)
    result = {"queries": int(len(rows))}
    for k in [1, 5, 10]:
        recalls = [
            len(set(order[i, :k]).intersection(labels[row])) / len(labels[row])
            for i, row in enumerate(rows)
        ]
        result[f"microcat_recall_at_{k}"] = float(np.mean(recalls)) if recalls else 0.0
    return result


def main():
    started = time.time()
    out = ROOT / "artifacts"
    cached = joblib.load(out / "eval_history.joblib")
    history, cached_query_features = (
        cached if isinstance(cached, tuple) else (cached, None)
    )
    eval_queries = pd.read_parquet(out / "eval_queries.parquet")
    targets = json.loads((out / "eval_targets.json").read_text(encoding="utf-8"))
    items = pd.read_parquet(
        out / "eval_items.parquet", columns=["item_id", "item_microcat_id"]
    )
    fit = pd.read_parquet(
        out / "fit_interactions.parquet",
        columns=["search_query", "search_infm_params_text", "item_microcat_id"],
    )
    fit_texts = set(fit.search_query.fillna("").map(norm))
    assert (
        set(history.texts) == fit_texts
    ), "History cache is stale: rebuild it for current fit_interactions"
    eval_texts = eval_queries.search_query.fillna("").map(norm).tolist()
    if (
        cached_query_features is not None
        and cached_query_features["texts"] != eval_texts
    ):
        cached_query_features = None  # Recompute if a cache used another row order.
    all_fit = pd.read_parquet(
        out / "all_interactions.parquet",
        columns=["search_query", "search_infm_params_text", "item_microcat_id"],
    )
    classes = np.sort(all_fit.item_microcat_id.unique())
    class_map = {int(c): i for i, c in enumerate(classes)}
    item_categories = dict(zip(items.item_id, items.item_microcat_id))
    labels = [
        {class_map[int(item_categories[i])] for i in targets[str(int(q))]}
        for q in eval_queries.qkey
    ]
    is_validation = eval_queries.split.eq("validation").to_numpy()
    masks = {
        "validation": is_validation,
        "validation_cold": is_validation & eval_queries.cold_text.to_numpy(),
        "validation_warm": is_validation & ~eval_queries.cold_text.to_numpy(),
    }
    filter_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), max_features=20000, dtype=np.float32, sublinear_tf=True
    )
    # Empty filters are valid zero vectors; nonempty training filters define vocab.
    filters = fit.search_infm_params_text.fillna("").map(norm).unique().tolist()
    filter_vectorizer.fit(filters)
    base = dict(
        word=history.word,
        char=history.char,
        filter=filter_vectorizer,
        classes=classes,
        filter_weight=FILTER_WEIGHT,
        temperature=1.0,
    )
    counts, class_counts = aggregate_counts(fit, history, filter_vectorizer, classes)
    counts_shape, counts_bytes = list(counts.shape), counts.nbytes
    print(f"Aggregated {counts.shape}; elapsed {time.time()-started:.1f}s", flush=True)
    # Nearest-query microcategory voting is the independent existing baseline.
    val_indices = np.flatnonzero(is_validation)
    nearest = (
        cached_query_features["microcat"][val_indices]
        if cached_query_features is not None
        else history.query_features(eval_queries.iloc[val_indices])["microcat"]
    )
    nearest_aligned = np.zeros((len(eval_queries), len(classes)), dtype=np.float32)
    for j, c in enumerate(history.microcats):
        nearest_aligned[val_indices, class_map[int(c)]] = nearest[:, j]
    metrics = {
        "nearest_history": {
            name: category_metrics(nearest_aligned, labels, mask)
            for name, mask in masks.items()
        }
    }
    best, best_probs, best_score = None, None, -1.0
    for kind, alpha in [
        ("multinomial", 0.1),
        ("multinomial", 1.0),
        ("complement", 0.1),
        ("complement", 1.0),
    ]:
        weights, intercept = fit_weights(counts, class_counts, kind, alpha)
        model = {
            **base,
            "weights": weights,
            "intercept": intercept,
            "kind": kind,
            "alpha": alpha,
        }
        probabilities = predict_proba(model, eval_queries)
        key = f"{kind}_alpha_{alpha}"
        metrics[key] = {
            name: category_metrics(probabilities, labels, mask)
            for name, mask in masks.items()
        }
        score = metrics[key]["validation"]["microcat_recall_at_5"]
        print(key, metrics[key]["validation"], flush=True)
        if score > best_score:
            best, best_probs, best_score = model, probabilities, score
    # This small fixed-default comparison is exploratory local selection; scores
    # are not claimed to be unbiased after selection. No benchmark labels exist.
    joblib.dump(best, out / "eval_category_model.joblib", compress=3)
    np.save(out / "eval_category_probabilities.npy", best_probs)
    np.save(out / "category_classes.npy", classes)
    np.save(out / "eval_category_query_keys.npy", eval_queries.qkey.to_numpy(np.uint64))
    selected = {
        "kind": best["kind"],
        "alpha": best["alpha"],
        "temperature": best["temperature"],
    }
    del counts, best_probs, weights, model, best
    gc.collect()
    full_counts, full_class_counts = aggregate_counts(
        all_fit, history, filter_vectorizer, classes
    )
    weights, intercept = fit_weights(
        full_counts, full_class_counts, selected["kind"], selected["alpha"]
    )
    final = {**base, **selected, "weights": weights, "intercept": intercept}
    benchmark = pd.read_parquet(ROOT / "data/benchmark_queries.parquet")
    benchmark_probs = predict_proba(final, benchmark)
    joblib.dump(final, out / "final_category_model.joblib", compress=3)
    np.save(out / "benchmark_category_probabilities.npy", benchmark_probs)
    np.save(
        out / "benchmark_category_query_ids.npy", benchmark.query_id.to_numpy(dtype=str)
    )
    metrics["selected"] = selected
    metrics["classes"] = classes.astype(int).tolist()
    metrics["files"] = {
        "eval_order": "eval_category_query_keys.npy (eval_queries file order)",
        "benchmark_order": "benchmark_category_query_ids.npy (benchmark_queries file order)",
        "columns": "category_classes.npy",
        "api": "predict_proba(joblib.load(model_path), queries)",
    }
    metrics["elapsed_seconds"] = time.time() - started
    metrics["aggregate_counts_shape"] = counts_shape
    metrics["aggregate_counts_bytes"] = counts_bytes
    metrics["limitations"] = [
        "Microcat recall is a proxy, not item Recall@50.",
        "Four fixed settings compared on local validation; selected score is exploratory.",
        "Final conditional counts use all train; text/filter vocabulary remains fixed from fit.",
    ]
    (ROOT / "reports/category_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(
        "Finished category comparison", selected, metrics["elapsed_seconds"], flush=True
    )


if __name__ == "__main__":
    main()
