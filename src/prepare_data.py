"""Prepare an item-disjoint holdout with a benchmark-like query distribution.

Sample unique normalized query texts, then one full query context per text.
Length and empty-filter strata follow the unlabeled benchmark distribution.
Remove all interactions with held-out positive items and with intentionally cold
texts. Advertisement features remain visible, as at inference time. Benchmark
query IDs and unknown benchmark labels are never used as model features.
"""

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEARCH = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
SEED = 20260928
# Fixed, interpretable bins; their weights are learned from unlabeled queries.
LENGTH_BOUNDS = [0, 14, 21, 29, 39, float("inf")]
LENGTH_LABELS = ["0-14", "15-21", "22-29", "30-39", "40+"]


def normalize_text(value):
    """Use the same spelling normalization for sampling and leakage checks."""
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip()


def query_metadata(df):
    text = df.search_query.fillna("").map(normalize_text)
    length_bin = pd.cut(
        text.str.len(), bins=LENGTH_BOUNDS, labels=LENGTH_LABELS, include_lowest=True
    ).astype(str)
    empty_filter = df.search_infm_params_text.fillna("").map(normalize_text).eq("")
    return pd.DataFrame(
        {
            "normalized_query": text,
            "query_stratum": length_bin + "|empty=" + empty_filter.astype(str),
        },
        index=df.index,
    )


def query_keys(df):
    x = df[SEARCH].fillna("").astype(str)
    for field in ["search_query", "search_infm_params_text"]:
        x[field] = x[field].map(normalize_text)
    # Hashing a tuple of input features is an index operation, not a prediction.
    return pd.util.hash_pandas_object(x, index=False).astype("uint64")


def proportional_counts(counts, total):
    """Largest-remainder apportionment preserves exactly the requested size."""
    weights = counts.astype(float) / counts.sum() * total
    result = np.floor(weights).astype(int)
    remainder = total - int(result.sum())
    for cell in (weights - result).sort_values(ascending=False).index[:remainder]:
        result.loc[cell] += 1
    return result


def sample_groups(eligible, benchmark_meta, rng, n_validation, n_ranker):
    """Uniform texts inside each stratum; a random context for the chosen text.

    Sampling raw interaction rows would overweight common short queries. A text
    can occur in both filter strata, so a global used-text set prevents overlap
    between strata and between ranker training and validation.
    """
    distribution = benchmark_meta.query_stratum.value_counts().sort_index()
    requested = {
        "validation": proportional_counts(distribution, n_validation),
        "ranker_train": proportional_counts(distribution, n_ranker),
    }
    pools = {
        cell: part.groupby("normalized_query", sort=False)["qkey"].agg(list).to_dict()
        for cell, part in eligible.groupby("query_stratum", sort=False)
    }
    used_texts, records, shortages = set(), [], []
    # Draw scarce strata first: the same text can be eligible with/without filters.
    cell_order = sorted(
        distribution.index,
        key=lambda cell: len(pools.get(cell, {}))
        / max(1, sum(int(v.loc[cell]) for v in requested.values())),
    )
    for cell in cell_order:
        pool = pools.get(cell, {})
        available = [text for text in pool if text not in used_texts]
        wanted = sum(int(v.loc[cell]) for v in requested.values())
        count = min(wanted, len(available))
        drawn = (
            rng.choice(available, size=count, replace=False).tolist() if count else []
        )
        rng.shuffle(drawn)
        if count < wanted:
            shortages.append({"stratum": cell, "requested": wanted, "available": count})
        # If a cell is short, divide it proportionally; global fallback fills
        # the remaining slots and the deviation is exposed in split statistics.
        n_dev = min(
            count,
            int(round(count * int(requested["validation"].loc[cell]) / max(1, wanted))),
        )
        for i, text in enumerate(drawn):
            # np.random.choice on Python ints spanning signed/unsigned 64-bit
            # can infer float64 and round a query hash. Sample only the position.
            key = int(pool[text][int(rng.integers(len(pool[text])))])
            records.append((key, "validation" if i < n_dev else "ranker_train"))
            used_texts.add(text)

    for name, size in [("validation", n_validation), ("ranker_train", n_ranker)]:
        missing = size - sum(split == name for _, split in records)
        if missing <= 0:
            continue
        fallback = eligible.loc[~eligible.normalized_query.isin(used_texts)]
        pool = (
            fallback.groupby("normalized_query", sort=False)["qkey"].agg(list).to_dict()
        )
        available = list(pool)
        drawn = rng.choice(
            available, size=min(missing, len(available)), replace=False
        ).tolist()
        for text in drawn:
            key = int(pool[text][int(rng.integers(len(pool[text])))])
            records.append((key, name))
            used_texts.add(text)
    return records, distribution.to_dict(), shortages


def main():
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    train = pd.read_parquet(ROOT / "data/train.parquet")
    train["qkey"] = query_keys(train)
    train["normalized_query"] = train.search_query.fillna("").map(normalize_text)
    pairs = train[["qkey", "item_id"]].drop_duplicates()
    groups = pairs.groupby("qkey", sort=False)["item_id"].agg(list)
    queries = train.drop_duplicates("qkey")[["qkey"] + SEARCH].set_index("qkey")
    benchmark_queries = pd.read_parquet(ROOT / "data/benchmark_queries.parquet")
    benchmark_meta = query_metadata(benchmark_queries)
    rng = np.random.default_rng(SEED)
    # Extremely broad groups would disproportionately enlarge a small local test.
    eligible = queries.loc[groups.index[groups.map(len).between(1, 15)]].reset_index()
    eligible = pd.concat([eligible, query_metadata(eligible)], axis=1)
    n_total = min(6000, eligible.normalized_query.nunique())
    desired_dev = min(1500, n_total // 4)
    records, benchmark_strata, shortages = sample_groups(
        eligible, benchmark_meta, rng, desired_dev, n_total - desired_dev
    )
    # Preserve exact hashes for pandas .loc; mixed Python integers can otherwise
    # be inferred as float64 by an intermediate index conversion.
    chosen = np.asarray([key for key, _ in records], dtype=np.uint64)
    split = dict(records)
    n_dev = sum(name == "validation" for _, name in records)
    held = set(i for k in chosen for i in groups.loc[k])
    bench = pd.read_parquet(ROOT / "data/benchmark_items.parquet")
    item_cols = [c for c in bench.columns]
    extra = train[train.item_id.isin(held)].drop_duplicates("item_id")[item_cols]
    corpus = (
        pd.concat([bench, extra], ignore_index=True)
        .drop_duplicates("item_id")
        .reset_index(drop=True)
    )
    corpus.to_parquet(out / "eval_items.parquet", index=False)
    q = queries.loc[chosen].reset_index()
    q = pd.concat([q, query_metadata(q)], axis=1)
    q["split"] = [split[int(k)] for k in q.qkey]
    # Benchmark text overlap is observed without looking at benchmark labels.
    # Match its roughly 63% unseen-text share in BOTH supervised/evaluation splits.
    desired_cold_fraction = float(
        (~benchmark_meta.normalized_query.isin(set(train.normalized_query))).mean()
    )
    base_fit_mask = ~train.item_id.isin(held)
    remaining_texts = set(train.loc[base_fit_mask, "normalized_query"])
    # Some texts become cold automatically when all their positive items are held.
    # Include these first so the reported warm/cold flags reflect actual fit data.
    q["cold_text"] = ~q.normalized_query.isin(remaining_texts)
    forced_cold_counts, target_cold_counts = {}, {}
    for name in ["validation", "ranker_train"]:
        indices = q.index[q.split.eq(name)]
        desired = int(round(len(indices) * desired_cold_fraction))
        forced = int(q.loc[indices, "cold_text"].sum())
        forced_cold_counts[name], target_cold_counts[name] = forced, desired
        possible = q.index[q.split.eq(name) & ~q.cold_text].to_numpy()
        extra_cold = max(0, desired - forced)
        if extra_cold:
            q.loc[
                rng.choice(
                    possible, size=min(extra_cold, len(possible)), replace=False
                ),
                "cold_text",
            ] = True
    cold_texts = set(q.loc[q.cold_text, "normalized_query"])
    assert (
        q.normalized_query.is_unique
    ), "A normalized text must appear in only one holdout group"
    q.to_parquet(out / "eval_queries.parquet", index=False)
    targets = {str(int(k)): groups.loc[k] for k in chosen}
    (out / "eval_targets.json").write_text(json.dumps(targets), encoding="utf-8")
    fit = train.loc[
        base_fit_mask & ~train.normalized_query.isin(cold_texts),
        SEARCH
        + [
            "item_id",
            "item_microcat_id",
            "item_location_id",
            "item_category_id",
            "item_latitude",
            "item_longitude",
        ],
    ]
    assert not set(fit.item_id).intersection(
        held
    ), "Held-out item interactions leaked into fit"
    fit_texts = set(fit.search_query.fillna("").map(normalize_text))
    assert not fit_texts.intersection(
        cold_texts
    ), "Cold text interactions leaked into fit"
    assert (
        q.loc[~q.cold_text, "normalized_query"].isin(fit_texts).all()
    ), "Warm text missing from fit"
    fit.to_parquet(out / "fit_interactions.parquet", index=False)
    # The final model can use all supplied training interactions, including known
    # benchmark advertisements, after hyperparameters have been selected.
    train[
        SEARCH
        + [
            "item_id",
            "item_microcat_id",
            "item_location_id",
            "item_category_id",
            "item_latitude",
            "item_longitude",
        ]
    ].to_parquet(out / "all_interactions.parquet", index=False)
    stats = dict(
        seed=SEED,
        train_rows=len(train),
        unique_pairs=len(pairs),
        unique_query_keys=len(groups),
        corpus_size=len(corpus),
        validation_queries=n_dev,
        ranker_train_queries=len(q) - n_dev,
        heldout_items=len(held),
        fit_rows=len(fit),
        validation_cold_queries=int(q.loc[q.split == "validation", "cold_text"].sum()),
        ranker_train_cold_queries=int(
            q.loc[q.split == "ranker_train", "cold_text"].sum()
        ),
        benchmark_cold_text_fraction=desired_cold_fraction,
        target_cold_counts=target_cold_counts,
        forced_cold_counts=forced_cold_counts,
        unique_holdout_texts=q.normalized_query.nunique(),
        benchmark_strata=benchmark_strata,
        validation_strata=q.loc[q.split == "validation", "query_stratum"]
        .value_counts()
        .to_dict(),
        ranker_train_strata=q.loc[q.split == "ranker_train", "query_stratum"]
        .value_counts()
        .to_dict(),
        stratum_shortages=shortages,
        excluded_broad_query_groups=int((groups.map(len) > 15).sum()),
        query_positive_counts=groups.loc[chosen].map(len).describe().to_dict(),
    )
    (ROOT / "reports/split_statistics.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
