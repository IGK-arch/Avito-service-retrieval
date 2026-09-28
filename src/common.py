"""Shared paths, feature contract and deterministic retrieval utilities.

Keep this module lightweight: exporting a saved model does not require loading
the sparse retriever, tokenizers or neural encoder. Feature order is a model
contract and must remain unchanged for the already submitted checkpoints.
"""

import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts"
R = ROOT / "reports"

# Text scores, followed by intent, geography, history, item metadata, coverage,
# filters and interactions between the channels. Columns 15/16 are deliberately
# disabled in the trained selectors; see docs/validation.md for the reason.
FEATURES = [
    "bm25",
    "bm25_relative",
    "char",
    "char_relative",
    "dense",
    "dense_relative",
    "title_bm25",
    "title_relative",
    "params_bm25",
    "microcat_probability",
    "same_location",
    "location_probability",
    "log_distance",
    "near_10km",
    "near_50km",
    "history_click",
    "item_popularity",
    "rating",
    "log_reviews",
    "log_price",
    "phone_hidden",
    "message_forbidden",
    "category_match",
    "query_tokens",
    "title_tokens",
    "title_coverage",
    "description_coverage",
    "params_coverage",
    "exact_title",
    "exact_description",
    "filter_coverage",
    "rating_filter_satisfied",
    "has_filters",
    "geo_bm25",
    "geo_dense",
    "microcat_bm25",
    "microcat_dense",
    "supervised_microcat",
    "supervised_microcat_relative",
    "supervised_microcat_rank",
    "microcat_confidence",
]


def log(message):
    """Mirror progress to the console and the local experiment journal."""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    R.mkdir(parents=True, exist_ok=True)
    with (R / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def top(scores, k):
    """Select the largest scores; use corpus position to order score ties."""
    k = min(k, len(scores))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    indices = np.argpartition(scores, len(scores) - k)[-k:]
    return indices[np.lexsort((indices, -scores[indices]))]


def recall(qi, ii, scores, queries, items, targets, k=50):
    """Macro Recall@k for contiguous query groups and unique positive item IDs.

    An empty targets dictionary is permitted at inference. Item/query positions
    only align arrays; they are never model features.
    """
    result, predictions = [], []
    boundaries = np.r_[0, np.flatnonzero(qi[1:] != qi[:-1]) + 1, len(qi)]
    item_ids = items.item_id.to_numpy()
    for q, start, end in zip(range(len(queries)), boundaries[:-1], boundaries[1:]):
        selected = ii[start:end][top(scores[start:end], k)]
        ids = item_ids[selected].tolist()
        predictions.append(ids)
        if targets:
            relevant = set(targets[str(int(queries.iloc[q].qkey))])
            result.append(len(relevant.intersection(ids)) / len(relevant))
    return np.array(result), predictions


def select_legal_candidates(
    query_indices, item_indices, scores, item_ids, legal_ids, k=50
):
    """Select unique corpus IDs, excluding validation-only items before top-k.

    All arrays must already follow the same contiguous query groups. Sorting is
    deterministic even for identical model scores, using corpus position.
    """
    if not (len(query_indices) == len(item_indices) == len(scores)):
        raise ValueError("Candidate arrays have different lengths")
    if np.any(query_indices[1:] < query_indices[:-1]):
        raise ValueError("Candidate query groups must be sorted")
    is_legal = np.asarray([iid in legal_ids for iid in item_ids], dtype=bool)
    bounds = np.r_[
        0, np.flatnonzero(query_indices[1:] != query_indices[:-1]) + 1, len(scores)
    ]
    predictions = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        positions = np.arange(start, end)
        positions = positions[is_legal[item_indices[positions]]]
        order = np.lexsort((item_indices[positions], -scores[positions]))
        selected = item_ids[item_indices[positions[order[:k]]]].tolist()
        if len(selected) != len(set(selected)):
            raise ValueError("Duplicate item positions inside a query candidate pool")
        predictions.append(selected)
    return predictions
