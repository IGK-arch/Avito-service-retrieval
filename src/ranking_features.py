"""Query-relative score features and ensemble calibration.

These transformations use scores and current candidate pools, without labels.
Query and item positions only align rows and never become prediction features.
"""

import numpy as np

from common import FEATURES

SOURCES = [
    "bm25",
    "geo_bm25",
    "title_bm25",
    "char",
    "dense",
    "microcat_probability",
    "supervised_microcat",
]
TRANSFORMS = ["percentile", "gap_best", "gap_50"]


def feature_names():
    return FEATURES + [
        f"pool_{transform}_{source}" for source in SOURCES for transform in TRANSFORMS
    ]


def query_boundaries(query_indices):
    if len(query_indices) and np.any(query_indices[1:] < query_indices[:-1]):
        raise ValueError("Query rows must be contiguous and sorted")
    return np.r_[
        0,
        np.flatnonzero(query_indices[1:] != query_indices[:-1]) + 1,
        len(query_indices),
    ]


def augment_features(X, query_indices):
    """Return float32 [base features, query-relative ranks/gaps], preserving rows.

    Rank ties receive their shared minimum rank. Percentile is 1 for rank 1 and
    0 for the last possible rank; score gaps compare with the best and 50th item
    of the current query pool. This uses no relevance labels. One output array
    is allocated directly, avoiding an additional large features hstack copy.
    """
    if X.shape[1] != len(FEATURES):
        raise ValueError(f"Expected {len(FEATURES)} base columns, got {X.shape[1]}")
    if len(X) != len(query_indices):
        raise ValueError("Feature/query rows differ")
    boundaries = query_boundaries(query_indices)
    result = np.empty((len(X), len(feature_names())), dtype=np.float32)
    result[:, : len(FEATURES)] = X
    columns = [FEATURES.index(name) for name in SOURCES]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        n = end - start
        if not n:
            continue
        for j, column in enumerate(columns):
            scores = np.asarray(X[start:end, column], dtype=np.float32)
            scores = np.where(np.isfinite(scores), scores, np.float32(-1e20))
            order = np.argsort(-scores, kind="stable")
            ordered = scores[order]
            new_value = np.r_[True, ordered[1:] != ordered[:-1]]
            sorted_ranks = np.maximum.accumulate(
                np.where(new_value, np.arange(n) + 1, 0)
            )
            ranks = np.empty(n, dtype=np.float32)
            ranks[order] = sorted_ranks
            target = len(FEATURES) + j * len(TRANSFORMS)
            result[start:end, target] = 1.0 - (ranks - 1.0) / max(n - 1, 1)
            result[start:end, target + 1] = scores - ordered[0]
            result[start:end, target + 2] = scores - ordered[min(49, n - 1)]
    return result


def standardize_scores(scores, query_indices):
    """Calibrate unrelated model margins within each query, without labels."""
    result = np.empty(len(scores), dtype=np.float32)
    boundaries = query_boundaries(query_indices)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        values = np.asarray(scores[start:end], dtype=np.float32)
        mean = values.mean(dtype=np.float64)
        deviation = values.std(dtype=np.float64)
        result[start:end] = (values - mean) / max(float(deviation), 1e-6)
    return result


def reciprocal_scores(scores, query_indices, constant=60.0):
    result = np.empty(len(scores), dtype=np.float32)
    for start, end in zip(
        query_boundaries(query_indices)[:-1], query_boundaries(query_indices)[1:]
    ):
        values = scores[start:end]
        order = np.lexsort((np.arange(end - start), -values))
        ranks = np.empty(end - start, dtype=np.float32)
        ranks[order] = np.arange(1, end - start + 1)
        result[start:end] = 1.0 / (constant + ranks)
    return result
