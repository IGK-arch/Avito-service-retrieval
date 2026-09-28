"""Generic remote-service signals and train-only nonlocal microcategory rates.

These are weak numeric features; they never remove an advertisement or override
the learned model. No query/item identifiers or held-out positives are encoded.
An explicit online service can have a provider in a different city, so locality
should be conditional on service wording rather than universally mandatory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

QUERY_FLAG_NAMES = [
    "query_online_wording",
    "query_remote_wording",
    "query_nationwide_wording",
]
ITEM_FLAG_NAMES = ["item_remote_wording", "item_nationwide_wording"]
PRIOR_NAMES = ["query_remote_prior"]

# ё is converted to е first. Require the adjective/adverb stem "удаленн";
# matching the broader "удален" would confuse "удаление" (removal) with remote.
ONLINE = re.compile(r"\b(?:он[\s-]?лайн|online)\b", re.IGNORECASE)
REMOTE = re.compile(
    r"\b(?:удаленн[а-я]*|удаленк[а-я]*|дистанцион[а-я]*)\b|\b(?:по|через|в)\s+интернет[а-я]*\b",
    re.IGNORECASE,
)
NATIONWIDE = re.compile(
    r"\b(?:по\s+всей|во\s+всей|вся|всей|по)\s+росси[а-я]*\b"
    r"|\b(?:люб[а-я]*\s+(?:точк[а-я]*|город[а-я]*)|все\s+город[а-я]*)\s+(?:в\s+)?росси[а-я]*\b",
    re.IGNORECASE,
)
TOKENS = re.compile(r"\w+", re.UNICODE)


def _text(value: Any, max_chars: int | None = None) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        if pd.isna(value):
            return ""
        value = str(value)
    if max_chars is not None:
        value = value[:max_chars]
    return value.lower().replace("ё", "е")


def _query_key(value: Any) -> str:
    return " ".join(TOKENS.findall(_text(value)))


@dataclass
class RemoteFeatureBundle:
    query_flags: np.ndarray
    item_flags: np.ndarray
    query_remote_prior: np.ndarray
    microcat_rates: dict[int, float]
    microcat_counts: dict[int, int]
    global_prior: float
    stats: dict[str, Any]

    @property
    def names(self) -> dict[str, list[str]]:
        return {
            "query_flags": QUERY_FLAG_NAMES.copy(),
            "item_flags": ITEM_FLAG_NAMES.copy(),
            "query_remote_prior": PRIOR_NAMES.copy(),
        }

    def __iter__(self):
        """Convenience unpacking: qflags, iflags, qprior, names = result."""
        yield self.query_flags
        yield self.item_flags
        yield self.query_remote_prior
        yield self.names


def build_remote_features(
    queries: pd.DataFrame,
    items: pd.DataFrame,
    interactions: pd.DataFrame,
    microcat_probabilities=None,
    microcat_ids=None,
    smoothing: float = 50.0,
    description_chars: int = 1000,
    params_chars: int = 1000,
) -> RemoteFeatureBundle:
    """Build generic text flags and smooth train-only remote-location priors.

    ``interactions`` must be the fit-only interaction table for evaluation, and
    all training interactions for final inference. All labels enter only the
    generic category-level nonlocal rate; the caller controls split hygiene.

    A city search is identified conservatively by a search-location ID observed
    as an advertisement-location ID in corpus or fit interactions. Region-only
    IDs are excluded from rate estimation: their city inequality is expected
    and does not by itself imply remote delivery.

    Optional ``microcat_probabilities`` is shape (Nquery, Nmicrocat), aligned
    with ``microcat_ids``. It can be history votes or a train-only classifier.
    Then query_remote_prior is the expected nonlocal rate under that category
    distribution. Without it, use the fit exact-text category mixture, with
    the global city-search rate as fallback for unseen texts. The bundle also
    returns the category-rate dictionary for callers to combine themselves.
    """
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")
    if description_chars < 0 or params_chars < 0:
        raise ValueError("Text truncation lengths must be non-negative")
    query_flags = np.zeros((len(queries), 3), np.float32)
    query_params = queries.get(
        "search_infm_params_text", pd.Series("", index=queries.index)
    )
    for position, (query, params) in enumerate(zip(queries.search_query, query_params)):
        text = _text(query) + " " + _text(params, params_chars)
        query_flags[position] = (
            bool(ONLINE.search(text)),
            bool(REMOTE.search(text)),
            bool(NATIONWIDE.search(text)),
        )
    item_flags = np.zeros((len(items), 2), np.float32)
    item_params = items.get("item_infm_params_text", pd.Series("", index=items.index))
    descriptions = items.get("item_description_raw", pd.Series("", index=items.index))
    for position, (title, params, description) in enumerate(
        zip(items.item_title_raw, item_params, descriptions)
    ):
        # Truncate each long field before concatenating; only one small temporary
        # string exists per item and no giant joined-text array is retained.
        text = " ".join(
            (
                _text(title),
                _text(params, params_chars),
                _text(description, description_chars),
            )
        )
        item_flags[position] = (
            bool(ONLINE.search(text) or REMOTE.search(text)),
            bool(NATIONWIDE.search(text)),
        )

    needed = ["search_location_id", "item_location_id", "item_microcat_id"]
    missing = [name for name in needed if name not in interactions]
    if missing:
        raise ValueError(f"Missing fit-interaction columns: {missing}")
    observed_item_locations = set(
        pd.to_numeric(items.item_location_id, errors="coerce").dropna()
    )
    observed_item_locations.update(
        pd.to_numeric(interactions.item_location_id, errors="coerce").dropna()
    )
    search_location = pd.to_numeric(interactions.search_location_id, errors="coerce")
    item_location = pd.to_numeric(interactions.item_location_id, errors="coerce")
    microcat = pd.to_numeric(interactions.item_microcat_id, errors="coerce")
    is_city = search_location.isin(observed_item_locations)
    valid = is_city & search_location.notna() & item_location.notna() & microcat.notna()
    train_rates = pd.DataFrame(
        {
            "microcat": microcat.loc[valid].astype(np.int64),
            "nonlocal": (search_location.loc[valid] != item_location.loc[valid]).astype(
                np.float32
            ),
        }
    )
    # A fallback is only used for an entirely empty fit subset. In normal data
    # the global prior is measured from hundreds of thousands of city searches.
    global_prior = float(train_rates["nonlocal"].mean()) if len(train_rates) else 0.10
    aggregate = train_rates.groupby("microcat", sort=True)["nonlocal"].agg(
        ["sum", "count"]
    )
    microcat_rates = {
        int(category): float(
            (row["sum"] + smoothing * global_prior) / (row["count"] + smoothing)
        )
        for category, row in aggregate.iterrows()
    }
    microcat_counts = {
        int(category): int(row["count"]) for category, row in aggregate.iterrows()
    }
    prior = np.full(len(queries), global_prior, np.float32)
    if microcat_probabilities is not None:
        if microcat_ids is None:
            raise ValueError("microcat_ids are required with microcat_probabilities")
        if microcat_probabilities.shape != (len(queries), len(microcat_ids)):
            raise ValueError("Query microcategory probabilities have the wrong shape")
        rates = np.asarray(
            [
                microcat_rates.get(int(category), global_prior)
                for category in microcat_ids
            ],
            np.float32,
        )
        distribution = microcat_probabilities
        if sparse.issparse(distribution):
            if (
                np.any(distribution.data < 0)
                or not np.isfinite(distribution.data).all()
            ):
                raise ValueError(
                    "Category probabilities must be finite and nonnegative"
                )
        else:
            distribution = np.asarray(distribution, np.float32)
            if np.any(distribution < 0) or not np.isfinite(distribution).all():
                raise ValueError(
                    "Category probabilities must be finite and nonnegative"
                )
        weighted = np.asarray(distribution @ rates).ravel()
        row_sum = np.asarray(distribution.sum(axis=1)).ravel()
        np.divide(weighted, row_sum, out=prior, where=row_sum > 0)
        prior_source = "provided_train_only_microcategory_probabilities"
    elif "search_query" in interactions:
        # This contains no item-specific score. It averages generic smoothed
        # category rates for texts observed in fit, leaving cold texts at prior.
        text_keys = interactions.search_query.map(_query_key)
        transferred = microcat.map(microcat_rates).fillna(global_prior)
        text_mean = transferred.groupby(text_keys, sort=False).mean()
        prior = (
            queries.search_query.map(_query_key)
            .map(text_mean)
            .fillna(global_prior)
            .to_numpy(np.float32)
        )
        prior_source = "fit_exact_text_category_mixture_or_global_fallback"
    else:
        prior_source = "global_city_search_fallback"
    prior = np.clip(prior, 0.0, 1.0).reshape(-1, 1)
    stats = {
        "fit_rows": int(len(interactions)),
        "city_fit_rows": int(valid.sum()),
        "excluded_region_or_missing_rows": int((~valid).sum()),
        "microcategories_with_city_evidence": len(microcat_rates),
        "global_nonlocal_prior": global_prior,
        "smoothing": smoothing,
        "prior_source": prior_source,
        "description_chars": description_chars,
        "params_chars": params_chars,
        "query_flag_counts": dict(
            zip(QUERY_FLAG_NAMES, query_flags.sum(axis=0).astype(int).tolist())
        ),
        "item_flag_counts": dict(
            zip(ITEM_FLAG_NAMES, item_flags.sum(axis=0).astype(int).tolist())
        ),
    }
    return RemoteFeatureBundle(
        query_flags,
        item_flags,
        prior,
        microcat_rates,
        microcat_counts,
        global_prior,
        stats,
    )
