"""End-to-end offline retrieval experiments and submission generation.

This module builds candidate pools and compares CatBoost baselines. Its refit
and predict commands are historical baseline experiments. The submitted result
is exported by submit_best.py; see README.md for its exact reproduction path.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
import argparse
import gc
import hashlib
import json
import re
import time

import joblib
import numpy as np
import pandas as pd

from common import FEATURES, ROOT, A, R, log, recall, top
from history import HistoryIndex, norm
from lexical import (
    SparseConfig,
    SparseRetriever,
    _bm25_field,
    clean_text,
    normalize_tokens,
)
from load_items import load_items


def build():
    items = load_items(A / "eval_items.parquet")
    cfg = SparseConfig(
        use_char=True,
        description_max_chars=1800,
        word_max_features=400000,
        char_max_features=180000,
        query_params_weight=0,
    )
    log("Building field BM25 + title character index")
    index = SparseRetriever(items, cfg)
    index.save(A / "sparse_index.joblib")
    (A / "sparse_index.json").write_text(
        json.dumps(
            {
                "items": len(items),
                "id_sha256": hashlib.sha256(
                    "\n".join(items.item_id).encode()
                ).hexdigest(),
                "stats": index.build_stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"Index built: {index.build_stats}")


TINY_FEATURES = ["tiny_cosine", "tiny_relative", "tiny_geo", "tiny_microcat_geo"]


def features(mode, use_tiny=False):
    log(f"Preparing features: {mode}")
    items = load_items(A / "eval_items.parquet")
    queries = pd.read_parquet(
        A / "eval_queries.parquet"
        if mode == "eval"
        else ROOT / "data/benchmark_queries.parquet"
    )
    cache = A / f"{mode}_history.joblib"
    if cache.exists():
        hist, hf = joblib.load(cache)
    else:
        interactions = pd.read_parquet(
            A
            / (
                "fit_interactions.parquet"
                if mode == "eval"
                else "all_interactions.parquet"
            )
        )
        hist = HistoryIndex(interactions, items)
        hf = hist.query_features(queries)
        joblib.dump((hist, hf), cache, compress=0)
        del interactions
    log("Historical query neighbours and microcategory transfer ready")
    assert (
        hf["texts"] == queries.search_query.fillna("").map(norm).tolist()
    ), "Stale history query order"
    assert hist.n_items == len(items) and all(
        hist.item_map[iid] == i for i, iid in enumerate(items.item_id)
    ), "Stale history corpus"
    gc.collect()
    index = SparseRetriever.load(A / "sparse_index.joblib", mmap_mode="r")
    assert np.array_equal(
        index.item_ids, items.item_id.to_numpy()
    ), "Stale sparse index row order"
    category_prob = np.load(A / f"{mode}_category_probabilities.npy")
    category_classes = np.load(A / "category_classes.npy")
    assert category_prob.shape == (
        len(queries),
        len(category_classes),
    ), "Category probability dimensions"
    cat_rowids = np.load(
        A
        / (
            "eval_category_query_keys.npy"
            if mode == "eval"
            else "benchmark_category_query_ids.npy"
        ),
        allow_pickle=False,
    )
    assert np.array_equal(
        cat_rowids, queries["qkey" if mode == "eval" else "query_id"].to_numpy()
    ), "Stale category query order"
    c_map = {v: i for i, v in enumerate(category_classes)}
    c_item = np.array([c_map.get(v, -1) for v in items.item_microcat_id])
    records = queries.to_dict("records")
    qw = index._query_matrix(records, "word")
    qc = index._query_matrix(records, "char")
    qp = index.word_vectorizer.transform(
        queries.search_infm_params_text.fillna("")
    ).tocsr()
    qp.data.fill(1.0)
    titles = items.item_title_raw.fillna("").map(clean_text).tolist()
    descriptions = (
        items.item_description_raw.fillna("")
        .map(lambda x: clean_text(x, 1800))
        .tolist()
    )
    params = items.item_infm_params_text.fillna("").map(clean_text).tolist()
    title_tokens = [set(normalize_tokens(x)) for x in titles]
    desc_tokens = [set(normalize_tokens(x)) for x in descriptions]
    param_tokens = [set(normalize_tokens(x)) for x in params]
    title_matrix = _bm25_field(index.word_vectorizer.transform(titles), 0.35, 1.2)
    param_matrix = _bm25_field(index.word_vectorizer.transform(params), 0.4, 1.2)
    # Weight title/parameters by corpus IDF, the same convention as field BM25.
    df = np.bincount(index.word_matrix.indices, minlength=index.word_matrix.shape[1])
    idf = np.log1p((len(items) - df + 0.5) / (df + 0.5)).astype(np.float32)
    title_matrix = title_matrix.multiply(idf).tocsr()
    param_matrix = param_matrix.multiply(idf).tocsr()
    numeric = []
    for name, transform in [
        ("item_rating", False),
        ("item_rating_reviews_count", True),
        ("item_price", True),
    ]:
        vals = (
            pd.to_numeric(items[name], errors="coerce").fillna(0).to_numpy(np.float32)
        )
        numeric.append(np.log1p(np.maximum(vals, 0)) if transform else vals)
    flags = (
        items[["item_is_phone_hidden", "item_is_message_forbidden"]]
        .fillna(False)
        .to_numpy(np.float32)
    )
    category = items.item_category_id.to_numpy()
    # All needed metadata is now in compact arrays. Release the untruncated
    # Parquet descriptions before accumulating millions of candidate pairs.
    items = items[["item_id"]].copy()
    log("Corpus text and numeric features ready")
    # SciPy CSR @ CSC would reconvert the complete corpus on every batch.
    # Build the inverted CSR orientation once for all query batches.
    word_t = index.word_matrix.T.tocsr()
    char_t = index.char_matrix.T.tocsr()
    title_t = title_matrix.T.tocsr()
    param_t = param_matrix.T.tocsr()
    del title_matrix, param_matrix, params, index
    gc.collect()
    # Encoding may run concurrently with CPU preparation. A manifest appears
    # only after np.save has completed, so a partially written array is not read.
    qpath = A / (
        "eval_queries_embeddings.npy"
        if mode == "eval"
        else "benchmark_queries_embeddings.npy"
    )
    while not (
        (A / "item_embeddings.json").exists() and qpath.with_suffix(".json").exists()
    ):
        time.sleep(5)
    qe = np.load(qpath)
    ie = np.load(A / "item_embeddings.npy", mmap_mode="r")
    assert len(ie) == len(items), "Embedding/corpus row mismatch"
    for path, frame, col in [
        (A / "item_embeddings.json", items, "item_id"),
        (qpath.with_suffix(".json"), queries, "qkey" if mode == "eval" else "query_id"),
    ]:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        actual = hashlib.sha256("\n".join(frame[col].astype(str)).encode()).hexdigest()
        if "id_sha256" in manifest:
            assert manifest["id_sha256"] == actual, f"Stale embedding order: {path}"
    if use_tiny:
        tqp = A / (
            "tiny_eval_queries_embeddings.npy"
            if mode == "eval"
            else "tiny_benchmark_queries_embeddings.npy"
        )
        log("Waiting for locally trained Russian encoder embeddings")
        while not ((A / "tiny_item_embeddings.npy").exists() and tqp.exists()):
            time.sleep(5)
        # The encoder process publishes all arrays before its training report.
        while not (R / "tiny_encoder_validation.json").exists():
            time.sleep(5)
        tqe = np.load(tqp)
        tie = np.load(A / "tiny_item_embeddings.npy", mmap_mode="r")
        assert len(tie) == len(items) and len(tqe) == len(
            queries
        ), "Tiny embedding row count"
    rows = []
    indices = []
    qindices = []
    metrics = {}
    target = (
        json.loads((A / "eval_targets.json").read_text(encoding="utf-8"))
        if mode == "eval"
        else {}
    )
    channels = [
        "bm25",
        "char",
        "dense",
        "bm25_geo",
        "dense_geo",
        "hybrid_rule",
        "candidate_union",
    ]
    if use_tiny:
        channels += ["tiny_dense", "tiny_dense_geo", "tiny_hybrid_rule"]
    baselines = {k: [] for k in channels}
    itemids = items.item_id.to_numpy()
    for start in range(0, len(queries), 32):
        stop = min(start + 32, len(queries))
        word = (qw[start:stop] @ word_t).toarray()
        chars = (qc[start:stop] @ char_t).toarray()
        title_score = (qw[start:stop] @ title_t).toarray()
        param_score = (qp[start:stop] @ param_t).toarray()
        # BLAS performs exhaustive cosine retrieval, with bounded query batching.
        dense = qe[start:stop] @ ie.T
        if use_tiny:
            tiny = tqe[start:stop] @ tie.T
        for j, qi in enumerate(range(start, stop)):
            rec = records[qi]
            w = word[j]
            c = chars[j]
            d = dense[j]
            ts = title_score[j]
            ps = param_score[j]
            wr = w / max(float(w.max()), 1e-6)
            cr = c / max(float(c.max()), 1e-6)
            dr = d / max(float(d.max()), 1e-6)
            tr = ts / max(float(ts.max()), 1e-6)
            same, lp, dist = hist.location_scores(rec)
            geo = np.maximum(same, lp)
            # Nearby cities remain candidates; coarse region ids use the learned
            # search-location to item-location distribution instead of equality.
            geo = np.maximum(geo, 0.35 * np.exp(-dist / 50))
            m = np.zeros(len(items), np.float32)
            valid = hist.item_m >= 0
            m[valid] = hf["microcat"][qi, hist.item_m[valid]]
            cm = np.zeros(len(items), np.float32)
            validc = c_item >= 0
            cm[validc] = category_prob[qi, c_item[validc]]
            cmr = cm / max(float(cm.max()), 1e-6)
            cranks = np.argsort(np.argsort(-category_prob[qi])) + 1
            cmrank = np.zeros(len(items), np.float32)
            cmrank[validc] = 1.0 / cranks[c_item[validc]]
            combined_m = np.maximum(m, 0.7 * cmr)
            clicks = hist.click_scores(hf, qi)
            wg = wr * (0.15 + 0.85 * geo)
            dg = dr * (0.15 + 0.85 * geo)
            wm = wr * (0.2 + 0.8 * combined_m) * (0.25 + 0.75 * geo)
            dm = dr * (0.2 + 0.8 * combined_m) * (0.25 + 0.75 * geo)
            rule = (
                (0.48 * wr + 0.18 * cr + 0.34 * dr)
                * (0.15 + 0.85 * geo)
                * (0.4 + 0.6 * combined_m)
            )
            lists = [
                top(w, 250),
                top(c, 150),
                top(d, 250),
                top(wg, 300),
                top(dg, 250),
                top(wm, 200),
                top(dm, 150),
                top(rule, 150),
            ]
            if use_tiny:
                td = tiny[j]
                tdr = td / max(float(td.max()), 1e-6)
                tg = tdr * (0.15 + 0.85 * geo)
                tm = tdr * (0.2 + 0.8 * combined_m) * (0.25 + 0.75 * geo)
                tiny_rule = (
                    (0.45 * wr + 0.10 * cr + 0.45 * tdr)
                    * (0.15 + 0.85 * geo)
                    * (0.4 + 0.6 * combined_m)
                )
                lists += [top(td, 200), top(tg, 450), top(tm, 250), top(tiny_rule, 150)]
            candidates = np.unique(np.concatenate(lists + [np.flatnonzero(clicks > 0)]))
            toks = set(normalize_tokens(rec["search_query"]))
            nt = max(len(toks), 1)
            filt = set(normalize_tokens(rec["search_infm_params_text"]))
            fnoise = set(
                normalize_tokens(
                    "Вид услуги Тип услуги Рейтинг пользователя звезды и выше"
                )
            )
            filt -= fnoise
            literal = " ".join(clean_text(rec["search_query"]).split())
            tc = np.array(
                [len(toks & title_tokens[i]) / nt for i in candidates], np.float32
            )
            dc = np.array(
                [len(toks & desc_tokens[i]) / nt for i in candidates], np.float32
            )
            pc = np.array(
                [len(toks & param_tokens[i]) / nt for i in candidates], np.float32
            )
            fc = np.array(
                [len(filt & param_tokens[i]) / max(len(filt), 1) for i in candidates],
                np.float32,
            )
            exact = np.array(
                [float(bool(literal) and literal in titles[i]) for i in candidates],
                np.float32,
            )
            exactd = np.array(
                [
                    float(bool(literal) and literal in descriptions[i])
                    for i in candidates
                ],
                np.float32,
            )
            ratingfilter = re.search(
                r"Рейтинг пользователя\s+(\d)", str(rec["search_infm_params_text"])
            )
            minimum = int(ratingfilter.group(1)) if ratingfilter else 0
            ix = candidates
            x = np.column_stack(
                [
                    w[ix],
                    wr[ix],
                    c[ix],
                    cr[ix],
                    d[ix],
                    dr[ix],
                    ts[ix],
                    tr[ix],
                    ps[ix],
                    m[ix],
                    same[ix],
                    lp[ix],
                    np.log1p(dist[ix]),
                    dist[ix] < 10,
                    dist[ix] < 50,
                    clicks[ix],
                    hist.pop[ix],
                    numeric[0][ix],
                    numeric[1][ix],
                    numeric[2][ix],
                    flags[ix, 0],
                    flags[ix, 1],
                    (category[ix] == rec["search_category"])
                    | (rec["search_category"] == 0),
                    np.full(len(ix), len(toks)),
                    [len(title_tokens[i]) for i in ix],
                    tc,
                    dc,
                    pc,
                    exact,
                    exactd,
                    fc,
                    numeric[0][ix] >= minimum,
                    np.full(len(ix), bool(rec["search_infm_params_text"])),
                    wg[ix],
                    dg[ix],
                    wm[ix],
                    dm[ix],
                    cm[ix],
                    cmr[ix],
                    cmrank[ix],
                    np.full(len(ix), float(category_prob[qi].max())),
                ]
            ).astype(np.float32)
            if use_tiny:
                x = np.column_stack([x, td[ix], tdr[ix], tg[ix], tm[ix]]).astype(
                    np.float32
                )
            rows.append(x)
            indices.append(ix.astype(np.int32))
            qindices.append(np.full(len(ix), qi, np.int32))
            if mode == "eval":
                relevant = set(target[str(int(rec["qkey"]))])
                for name, scores in [
                    ("bm25", w),
                    ("char", c),
                    ("dense", d),
                    ("bm25_geo", wg),
                    ("dense_geo", dg),
                    ("hybrid_rule", rule),
                ]:
                    predicted = set(itemids[top(scores, 50)])
                    baselines[name].append(len(relevant & predicted) / len(relevant))
                if use_tiny:
                    for name, scores in [
                        ("tiny_dense", td),
                        ("tiny_dense_geo", tg),
                        ("tiny_hybrid_rule", tiny_rule),
                    ]:
                        baselines[name].append(
                            len(relevant & set(itemids[top(scores, 50)]))
                            / len(relevant)
                        )
                baselines["candidate_union"].append(
                    len(relevant & set(itemids[ix])) / len(relevant)
                )
        if start % 320 == 0:
            log(f"Features {stop}/{len(queries)}")
    suffix = "_tiny" if use_tiny else ""
    shape = (
        sum(len(row) for row in rows),
        len(FEATURES) + (len(TINY_FEATURES) if use_tiny else 0),
    )
    # A disk-backed matrix avoids a second gigabyte anonymous allocation when
    # concatenating all query blocks on machines with a tight commit limit.
    x = np.lib.format.open_memmap(
        A / f"{mode}{suffix}_X.npy", mode="w+", dtype=np.float32, shape=shape
    )
    cursor = 0
    for row in rows:
        x[cursor : cursor + len(row)] = row
        cursor += len(row)
    del (
        rows,
        descriptions,
        desc_tokens,
        title_tokens,
        param_tokens,
        word_t,
        char_t,
        title_t,
        param_t,
    )
    gc.collect()
    x.flush()
    ii = np.concatenate(indices)
    qi = np.concatenate(qindices)
    labels = np.zeros(len(ii), np.uint8)
    if mode == "eval":
        for q in range(len(queries)):
            mask = qi == q
            labels[mask] = np.isin(
                itemids[ii[mask]], target[str(int(queries.iloc[q].qkey))]
            )
        validation = queries.split.to_numpy() == "validation"
        metrics = {
            name: dict(
                all=float(np.mean(v)),
                validation=float(np.mean(np.array(v)[validation])),
                cold_validation=float(
                    np.mean(np.array(v)[validation & queries.cold_text.to_numpy()])
                ),
            )
            for name, v in baselines.items()
        }
        suffix = "_tiny" if use_tiny else ""
        (R / f"baseline{suffix}_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        pd.DataFrame(baselines).to_parquet(
            A / f"baseline{suffix}_per_query.parquet", index=False
        )
    # Uncompressed NPZ trades disk space for much faster iterative experiments.
    suffix = "_tiny" if use_tiny else ""
    np.savez(
        A / f"{mode}{suffix}_features.npz",
        X=x,
        item_indices=ii,
        query_indices=qi,
        labels=labels,
    )
    (A / f"{mode}{suffix}_feature_names.json").write_text(
        json.dumps(FEATURES + (TINY_FEATURES if use_tiny else [])), encoding="utf-8"
    )
    log(f"Features complete: {x.shape}; metrics={metrics}")


def train():
    from catboost import CatBoostClassifier, Pool

    data = np.load(A / "eval_features.npz")
    X = data["X"]
    qi = data["query_indices"]
    ii = data["item_indices"]
    y = data["labels"]
    # Item-disjoint positives necessarily have no click history. Letting a model
    # use this fact would teach an artificial negative popularity effect. These
    # channels are instead applied explicitly at inference for exact historical
    # query texts and compatible geography; never use query ids as features.
    X[:, 15:17] = 0
    queries = pd.read_parquet(A / "eval_queries.parquet")
    items = pd.read_parquet(A / "eval_items.parquet", columns=["item_id"])
    targets = json.loads((A / "eval_targets.json").read_text(encoding="utf-8"))
    tr = queries.split.to_numpy()[qi] == "ranker_train"
    # Keep all positives and hard candidates, plus a reproducible sample of easy
    # negatives; this bounds training cost without changing validation ranking.
    rng = np.random.default_rng(20260928)
    hard = (X[:, 1] > 0.5) & ((X[:, 10] > 0) | (X[:, 11] > 0.05) | (X[:, 12] < 4))
    take = tr & ((y > 0) | hard | (rng.random(len(y)) < 0.20))
    log(f"Training CatBoost on {take.sum()} pairs; {y[take].sum()} positives")
    pool = Pool(X[take], y[take], feature_names=FEATURES)
    results = {}
    for name, depth in [("catboost_depth6", 6), ("catboost_depth8", 8)]:
        model = CatBoostClassifier(
            iterations=700,
            depth=depth,
            learning_rate=0.07,
            loss_function="Logloss",
            random_seed=20260928,
            thread_count=8,
            l2_leaf_reg=5,
            verbose=100,
            allow_writing_files=False,
        )
        model.fit(pool)
        scores = model.predict_proba(X)[:, 1]
        per, pred = recall(qi, ii, scores, queries, items, targets)
        vm = queries.split.to_numpy() == "validation"
        cold = queries.cold_text.to_numpy()
        results[name] = dict(
            validation_recall50=float(per[vm].mean()),
            cold_recall50=float(per[vm & cold].mean()),
            warm_recall50=float(per[vm & ~cold].mean()),
            train_recall50=float(per[~vm].mean()),
        )
        model.save_model(str(A / f"{name}.cbm"))
        np.save(A / f"{name}_scores.npy", scores)
        log(f"{name}: {results[name]}")
    best = max(results, key=lambda n: results[n]["validation_recall50"])
    (A / "selected_model.json").write_text(
        json.dumps({"name": best, "features": FEATURES}), encoding="utf-8"
    )
    (R / "model_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    log(f"Selected model {best}")


def refit():
    """After selection, use both development parts to fit the final selector."""
    from catboost import CatBoostClassifier, Pool

    selected = json.loads((A / "selected_model.json").read_text(encoding="utf-8"))
    depth = 6 if selected["name"].endswith("6") else 8
    data = np.load(A / "eval_features.npz")
    X = data["X"]
    y = data["labels"]
    X[:, 15:17] = 0
    rng = np.random.default_rng(20260928)
    hard = (X[:, 1] > 0.5) & ((X[:, 10] > 0) | (X[:, 11] > 0.05) | (X[:, 12] < 4))
    take = (y > 0) | hard | (rng.random(len(y)) < 0.20)
    log(f"Refitting selected model on all development queries: {take.sum()} pairs")
    model = CatBoostClassifier(
        iterations=700,
        depth=depth,
        learning_rate=0.07,
        loss_function="Logloss",
        random_seed=20260928,
        thread_count=8,
        l2_leaf_reg=5,
        verbose=100,
        allow_writing_files=False,
    )
    model.fit(Pool(X[take], y[take], feature_names=FEATURES))
    model.save_model(str(A / "final_model.cbm"))
    log("Final model saved")


def predict():
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(A / "final_model.cbm"))
    data = np.load(A / "benchmark_features.npz")
    X = data["X"].copy()
    X[:, 15:17] = 0
    scores = model.predict_proba(X)[:, 1]
    queries = pd.read_parquet(ROOT / "data/benchmark_queries.parquet")
    items = pd.read_parquet(A / "eval_items.parquet", columns=["item_id"])
    # Evaluation extras must never enter the submission, even if scores are high.
    corpus = set(
        pd.read_parquet(
            ROOT / "data/benchmark_items.parquet", columns=["item_id"]
        ).item_id
    )
    legal = items.item_id.isin(corpus).to_numpy()[data["item_indices"]]
    scores[~legal] = -1e9
    _, pred = recall(
        data["query_indices"], data["item_indices"], scores, queries, items, {}, 50
    )
    base = [list(i for i in ids if i in corpus) for ids in pred]
    pd.DataFrame(
        {"query_id": queries.query_id, "answer": [" ".join(ids) for ids in base]}
    ).to_csv(ROOT / "answer_without_history.csv", index=False, encoding="utf-8")
    # Historical positive pairs are available training evidence, never manual
    # benchmark answers. Reserve a small number with exact text and compatible
    # geography; full input-feature matches receive precedence.
    from prepare_data import normalize_text, query_keys

    interactions = pd.read_parquet(A / "all_interactions.parquet")
    interactions = interactions[interactions.item_id.isin(corpus)].copy()
    interactions["key"] = query_keys(interactions)
    exact = interactions.groupby("key")["item_id"].agg(lambda x: list(dict.fromkeys(x)))
    interactions["text"] = interactions.search_query.map(normalize_text)
    textgroups = {text: part for text, part in interactions.groupby("text")}
    qkeys = query_keys(queries)
    ii = data["item_indices"]
    qi = data["query_indices"]
    combined = []
    reserved_counts = []
    for q, rec in enumerate(queries.to_dict("records")):
        reserved = []
        key = qkeys.iloc[q]
        if key in exact.index:
            reserved.extend(exact.loc[key][:25])
        text = normalize_text(rec["search_query"])
        if text in textgroups:
            group = textgroups[text]
            # Exact search geography is evidence even for coarse region codes.
            known = set(
                group.loc[
                    group.search_location_id == rec["search_location_id"], "item_id"
                ]
            )
            mask = qi == q
            ids = items.item_id.to_numpy()[ii[mask]]
            compatible = (
                (X[mask, 10] > 0) | (X[mask, 11] > 0.10) | (X[mask, 12] < np.log1p(50))
            )
            known.update(set(group.item_id) & set(ids[compatible]))
            modelorder = ids[np.argsort(-scores[mask])]
            additional = [i for i in modelorder if i in known and i not in reserved]
            reserved.extend(additional[: max(0, 10 - len(reserved))])
        result = list(dict.fromkeys(reserved + base[q]))[:50]
        combined.append(result)
        reserved_counts.append(len(reserved))
    output = pd.DataFrame(
        {"query_id": queries.query_id, "answer": [" ".join(ids) for ids in combined]}
    )
    output.to_csv(ROOT / "answer.csv", index=False, encoding="utf-8")
    log(f"Wrote answer.csv: {len(output)} queries")
    (R / "history_reservation.json").write_text(
        json.dumps(
            dict(
                queries_with_reserved=sum(n > 0 for n in reserved_counts),
                mean_reserved=float(np.mean(reserved_counts)),
                max_reserved=max(reserved_counts),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


def prepare_history(mode):
    items = pd.read_parquet(A / "eval_items.parquet")
    queries = pd.read_parquet(
        A / "eval_queries.parquet"
        if mode == "eval"
        else ROOT / "data/benchmark_queries.parquet"
    )
    interactions = pd.read_parquet(
        A
        / ("fit_interactions.parquet" if mode == "eval" else "all_interactions.parquet")
    )
    log(f"Building {mode} history transfer index")
    hist = HistoryIndex(interactions, items)
    hf = hist.query_features(queries)
    joblib.dump((hist, hf), A / f"{mode}_history.joblib", compress=0)
    log(f"{mode} history cached")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step", choices=["build", "history", "features", "train", "refit", "predict"]
    )
    parser.add_argument("--mode", choices=["eval", "benchmark"], default="eval")
    parser.add_argument(
        "--use-tiny",
        action="store_true",
        help="Expand candidates with the locally trained encoder",
    )
    args = parser.parse_args()
    if args.step == "build":
        build()
    elif args.step == "history":
        prepare_history(args.mode)
    elif args.step == "features":
        features(args.mode, args.use_tiny)
    elif args.step == "train":
        train()
    elif args.step == "refit":
        refit()
    else:
        predict()


if __name__ == "__main__":
    main()
