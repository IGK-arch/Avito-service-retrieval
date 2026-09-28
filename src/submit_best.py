"""Export the best completed validation experiment, without further tuning.

This uses our saved LightGBM and CatBoost models, the same 90/10 per-query
standardized-score mixture measured on the held-out queries, and our benchmark
candidate features. No query_id is a feature and no external API is used.
Run from the project root: py -3.12 src/submit_best.py
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from common import FEATURES, ROOT, A, R, log, select_legal_candidates
from ranking_features import (
    augment_features,
    feature_names,
    standardize_scores,
)
from validate_answer import file_sha256, validate_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "answer.csv")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--artifacts-dir", type=Path, default=A)
    parser.add_argument("--checkpoints-dir", type=Path, default=ROOT / "checkpoints")
    args = parser.parse_args()
    artifacts = args.artifacts_dir
    checkpoints = args.checkpoints_dir
    config = json.loads((ROOT / "configs/submission.json").read_text(encoding="utf-8"))
    recipe = config["ensemble"]
    assert recipe["recipe"]["kind"] == "query_zscore"
    assert recipe["recipe"]["lightgbm_model"] == "lgbm_depth9"
    assert recipe["recipe"]["catboost_model"] == "catboost_depth6"
    weight = float(recipe["recipe"]["lightgbm_weight"])
    queries_path = args.data_dir / "benchmark_queries.parquet"
    items_path = args.data_dir / "benchmark_items.parquet"
    queries = pd.read_parquet(queries_path)
    # A row-order mapping is sufficient for exporting cached candidate features.
    # The fast reproduction bundle contains no descriptions or answer lists.
    mapping_path = artifacts / "eval_item_ids.parquet"
    if not mapping_path.exists():
        mapping_path = artifacts / "eval_items.parquet"
    items = pd.read_parquet(mapping_path, columns=["item_id"])
    legal = set(pd.read_parquet(items_path, columns=["item_id"]).item_id)
    # These manifests were checked when the candidate features were generated.
    query_order = np.load(
        artifacts / "benchmark_category_query_ids.npy", allow_pickle=False
    )
    assert np.array_equal(
        query_order, queries.query_id.to_numpy()
    ), "Query cache order changed"
    index_metadata = json.loads(
        (artifacts / "item_embeddings.json").read_text(encoding="utf-8")
    )
    item_hash = hashlib.sha256("\n".join(items.item_id).encode()).hexdigest()
    assert item_hash == index_metadata["id_sha256"], "Corpus cache order changed"
    manifest_path = artifacts / "reproduction_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, metadata in manifest["files"].items():
            if file_sha256(artifacts / name) != metadata["sha256"]:
                raise ValueError(f"Reproduction artifact checksum mismatch: {name}")
    with np.load(artifacts / "benchmark_features.npz") as cache:
        X = cache["X"]
        qi = cache["query_indices"]
        ii = cache["item_indices"]
    assert X.shape == (len(qi), len(FEATURES))
    assert len(ii) == len(qi) and np.all((ii >= 0) & (ii < len(items)))
    assert np.array_equal(np.unique(qi), np.arange(len(queries)))
    # Historical item clicks are zeroed just as in validation model training.
    X[:, 15:17] = 0
    log(
        f"Export best validated ensemble: {len(queries)} queries, {len(qi)} candidate pairs"
    )
    catboost = CatBoostClassifier()
    cb_path = checkpoints / "catboost_depth6.cbm"
    lb_path = checkpoints / "lgbm_depth9_model.txt"
    for name, path in [("catboost", cb_path), ("lightgbm", lb_path)]:
        if file_sha256(path) != config["model_sha256"][name]:
            raise ValueError(f"Submitted checkpoint checksum mismatch: {name}")
    catboost.load_model(str(cb_path))
    assert list(catboost.feature_names_) == FEATURES
    cb_scores = catboost.predict_proba(X, thread_count=6)[:, 1]
    lb_features = augment_features(X, qi)
    del X
    gc.collect()
    lightgbm = lgb.Booster(model_file=str(lb_path))
    assert lightgbm.feature_name() == feature_names()
    lb_scores = np.empty(len(qi), dtype=np.float32)
    for start in range(0, len(qi), 250000):
        end = min(start + 250000, len(qi))
        lb_scores[start:end] = lightgbm.predict(lb_features[start:end], num_threads=6)
        log(f"Best ensemble prediction: {end}/{len(qi)} pairs")
    del lb_features
    gc.collect()
    scores = weight * standardize_scores(lb_scores, qi) + (
        1 - weight
    ) * standardize_scores(cb_scores, qi)
    assert np.isfinite(scores).all()
    np.save(artifacts / "benchmark_best_ensemble_scores.npy", scores)
    # Our validation corpus also contains held-out positives. They are excluded
    # here: only the original benchmark corpus is legal in a submission.
    predictions = select_legal_candidates(
        qi, ii, scores, items.item_id.to_numpy(), legal, k=50
    )
    answer = pd.DataFrame(
        {"query_id": queries.query_id, "answer": [" ".join(ids) for ids in predictions]}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".csv.tmp")
    # Fix CRLF explicitly: the submitted Windows CSV must match byte-for-byte
    # when reproduced on Linux/macOS as well.
    answer.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\r\n")
    validation = validate_answer(temporary, queries_path, items_path)
    assert validation["valid"], validation
    actual_hash = file_sha256(temporary)
    if actual_hash != config["answer_sha256"]:
        raise ValueError(f"Predictions differ from submitted answer: {actual_hash}")
    temporary.replace(args.output)
    validation["answer_path"] = str(args.output.resolve())
    validation["matches_submitted_answer"] = True
    (R / "answer_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    provenance = {
        "ensemble": recipe,
        "model_sha256": {
            "catboost": file_sha256(cb_path),
            "lightgbm": file_sha256(lb_path),
        },
        "answer_sha256": actual_hash,
        "public_recall50": config["public_recall50"],
        "refit": False,
        "additional_history_reservation": False,
        "note": "Uses the exact saved models whose mixture was evaluated; reported recall is local validation, not platform score.",
    }
    (R / "submission_best.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(
        f'{args.output.name} ready, valid and identical to submitted answer: {len(answer)} rows; local Recall@50 {recipe["metrics"]["validation_recall50"]:.7f}'
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
