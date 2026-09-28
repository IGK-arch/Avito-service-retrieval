"""Analyze saved held-out retrieval/ranker results without rerunning models.

This consumes eval_features.npz, selected model scores, eval_targets.json and
the corresponding query/item Parquets. Geography/filter context is displayed
beside missed relevant items and top distractors. Labels are observed positives;
"distractor" here means absent from those labels, not proven irrelevant.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = [
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
]


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def compact(value, limit=220) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return "—"
    return " ".join(str(value).split())[:limit].replace("|", "\\|")


def feature_dict(row: np.ndarray | None, names: list[str]) -> dict:
    return (
        {} if row is None else {name: float(value) for name, value in zip(names, row)}
    )


def error_tags(
    target_features: dict,
    false_features: dict,
    target_rank: int | None,
    query_location,
    target_location,
    false_location,
    target_microcat=None,
    false_microcat=None,
) -> list[str]:
    """Heuristic descriptions of observed failures, not causal diagnoses."""
    if target_rank is None:
        return ["candidate miss"]
    tags = []
    if 51 <= target_rank <= 100:
        tags.append("close rank ≥50")
    if false_location != target_location or target_features.get(
        "same_location", 0
    ) != false_features.get("same_location", 0):
        tags.append("location diff")
    target_lex = target_features.get("bm25_relative", 0.0)
    if target_lex < 0.20 or (
        target_features.get("title_coverage", 0.0) < 0.25
        and target_lex + 0.25 < false_features.get("bm25_relative", 0.0)
    ):
        tags.append("low lexical")
    target_dense = target_features.get("dense_relative", 0.0)
    # Generic encoder scores often lie around .3-.4 even for true positives;
    # .65 would label almost every item "weak". Prefer a low floor or a clear
    # deficit to the observed distractor, rather than an arbitrary high floor.
    if target_dense < 0.15 or target_dense + 0.15 < false_features.get(
        "dense_relative", 0.0
    ):
        tags.append("low semantic")
    if (
        target_microcat is not None
        and false_microcat is not None
        and target_microcat != false_microcat
    ):
        tags.append("microcat diff")
    return tags or ["other ranking"]


def analyze(
    artifacts: Path,
    reports: Path,
    output: Path,
    scores_path: Path | None = None,
    maximum_examples: int = 40,
    model_name_override: str | None = None,
) -> dict:
    selected = read_json(artifacts / "selected_model.json", {})
    model_name = model_name_override or selected.get("name", "unknown")
    names = selected.get("features", DEFAULT_FEATURES)
    if scores_path is None:
        if model_name == "unknown":
            raise FileNotFoundError(
                "selected_model.json is required unless --scores is supplied"
            )
        scores_path = artifacts / f"{model_name}_scores.npy"
    with np.load(artifacts / "eval_features.npz", allow_pickle=False) as archive:
        features = archive["X"]
        query_indices = archive["query_indices"]
        item_indices = archive["item_indices"]
    scores = np.load(scores_path, allow_pickle=False).reshape(-1)
    queries = pd.read_parquet(artifacts / "eval_queries.parquet")
    # Full descriptions are unnecessary for analysis and can occupy hundreds
    # of MB; titles, parameter filters and geographic fields are sufficient.
    wanted = [
        "item_id",
        "item_title_raw",
        "item_infm_params_text",
        "item_location_id",
        "item_category_id",
        "item_microcat_id",
        "item_rating",
        "item_rating_reviews_count",
    ]
    items = pd.read_parquet(artifacts / "eval_items.parquet", columns=wanted)
    targets = read_json(artifacts / "eval_targets.json")
    if targets is None:
        raise FileNotFoundError(artifacts / "eval_targets.json")
    n_pairs = len(scores)
    if (
        features.shape != (n_pairs, len(names))
        or len(query_indices) != n_pairs
        or len(item_indices) != n_pairs
    ):
        raise ValueError("Feature, score and candidate arrays are not aligned")
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite model scores")
    if np.any(query_indices < 0) or np.any(query_indices >= len(queries)):
        raise ValueError("Invalid query positions in saved candidates")
    if np.any(item_indices < 0) or np.any(item_indices >= len(items)):
        raise ValueError("Invalid item positions in saved candidates")
    if not items.item_id.is_unique:
        raise ValueError("eval_items must have unique item_id")
    item_ids = items.item_id.to_numpy()
    item_map = {item_id: pos for pos, item_id in enumerate(item_ids)}
    counts = np.bincount(query_indices, minlength=len(queries))
    offsets = np.r_[0, np.cumsum(counts)]
    # Usually already grouped by query. The fallback handles independently
    # saved/interleaved candidate rows without silently mixing query scores.
    order = (
        None
        if np.all(query_indices[:-1] <= query_indices[1:])
        else np.argsort(query_indices, kind="stable")
    )
    recalls, ceilings, cold_recalls, warm_recalls = [], [], [], []
    tags = Counter()
    examples = []
    top1_distractors = 0
    positive_features, negative_features = [], []
    missed_total = 0
    duplicate_candidate_queries = 0
    for query_pos, query in queries.iterrows():
        if query.get("split", "validation") != "validation":
            continue
        key = str(int(query.qkey)) if "qkey" in queries else str(query.query_id)
        relevant = set(targets[key])
        if not relevant:
            raise ValueError(f"No held-out positives for query {key}")
        if any(item_id not in item_map for item_id in relevant):
            raise ValueError(f"Positive item absent from eval corpus for query {key}")
        start, end = offsets[query_pos : query_pos + 2]
        pair_positions = np.arange(start, end) if order is None else order[start:end]
        local_items, local_scores = item_indices[pair_positions], scores[pair_positions]
        ranked = np.lexsort((local_items, -local_scores))
        ranked_pairs = pair_positions[ranked]
        ranked_items = local_items[ranked]
        if len(set(ranked_items.tolist())) != len(ranked_items):
            duplicate_candidate_queries += 1
        ranks = {
            item_ids[item]: rank
            for rank, item in reversed(list(enumerate(ranked_items, start=1)))
        }
        pair_for_item = {
            item_ids[item]: int(pair) for item, pair in zip(local_items, pair_positions)
        }
        predicted = set(item_ids[ranked_items[:50]])
        hits = len(predicted & relevant)
        recall = hits / len(relevant)
        recalls.append(recall)
        ceilings.append(len(set(ranks) & relevant) / len(relevant))
        (cold_recalls if bool(query.get("cold_text", False)) else warm_recalls).append(
            recall
        )
        false_pair = next(
            (
                int(pair)
                for pair in ranked_pairs
                if item_ids[item_indices[pair]] not in relevant
            ),
            None,
        )
        top_pair = int(ranked_pairs[0]) if len(ranked_pairs) else None
        if top_pair is not None and item_ids[item_indices[top_pair]] not in relevant:
            top1_distractors += 1
        for item_id in relevant:
            if item_id in pair_for_item:
                positive_features.append(features[pair_for_item[item_id]])
        if false_pair is not None:
            negative_features.append(features[false_pair])
        false_item = (
            items.iloc[item_indices[false_pair]] if false_pair is not None else None
        )
        false_features = feature_dict(
            None if false_pair is None else features[false_pair], names
        )
        missed = sorted(
            relevant - predicted, key=lambda iid: (ranks.get(iid, 10**9), iid)
        )
        for item_id in missed:
            missed_total += 1
            target_item = items.iloc[item_map[item_id]]
            target_pair = pair_for_item.get(item_id)
            target_features = feature_dict(
                None if target_pair is None else features[target_pair], names
            )
            categories = error_tags(
                target_features,
                false_features,
                ranks.get(item_id),
                query.search_location_id,
                target_item.item_location_id,
                None if false_item is None else false_item.item_location_id,
                target_item.item_microcat_id,
                None if false_item is None else false_item.item_microcat_id,
            )
            tags.update(categories)
            if len(examples) < maximum_examples:
                examples.append(
                    {
                        "query": query,
                        "key": key,
                        "recall": recall,
                        "relevant_count": len(relevant),
                        "target": target_item,
                        "target_rank": ranks.get(item_id),
                        "target_features": target_features,
                        "false": false_item,
                        "false_features": false_features,
                        "top_is_false": top_pair is not None
                        and item_ids[item_indices[top_pair]] not in relevant,
                        "categories": categories,
                    }
                )

    def mean(values):
        return None if not values else float(np.mean(values))

    summary = {
        "model": model_name,
        "validation_queries": len(recalls),
        "recall50": mean(recalls),
        "cold_recall50": mean(cold_recalls),
        "warm_recall50": mean(warm_recalls),
        "candidate_recall_ceiling": mean(ceilings),
        "missed_relevant_items": missed_total,
        "top1_absent_from_labels_queries": top1_distractors,
        "duplicate_candidate_queries": duplicate_candidate_queries,
        "error_tags": dict(tags),
    }
    lines = [
        "# Анализ ошибок на отложенных запросах",
        "",
        "Отчёт рассчитан по сохранённым кандидатам и оценкам модели; тестовая разметка не используется. "
        "«Отвлекающее объявление» означает отсутствие в наблюдаемых положительных примерах. "
        "Позиция top-1 служит для сравнения текстов; целевая метрика зависит только от попадания в top-50.",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    baseline_metrics = read_json(reports / "baseline_metrics.json", {})
    model_metrics = read_json(reports / "model_metrics.json", {})
    if baseline_metrics or model_metrics:
        lines += [
            "## Сравнение алгоритмов",
            "",
            "| Алгоритм | Validation Recall@50 | Cold Recall@50 |",
            "|---|---:|---:|",
        ]
        for name, values in baseline_metrics.items():
            lines.append(
                f"| {compact(name)} | {values.get('validation', '—')} | {values.get('cold_validation', '—')} |"
            )
        for name, values in model_metrics.items():
            lines.append(
                f"| {compact(name)} | {values.get('validation_recall50', '—')} | {values.get('cold_recall50', '—')} |"
            )
        lines.append("")
    if positive_features and negative_features:
        positive = np.mean(np.stack(positive_features), axis=0)
        negative = np.mean(np.stack(negative_features), axis=0)
        lines += [
            "## Признаки положительных примеров и первого отвлекающего кандидата",
            "",
            "Положительные примеры взяты только из union-кандидатов. Сравнение показывает наблюдаемую связь; оно не доказывает причинность.",
            "",
            "| Признак | Положительные | Первое отвлекающее |",
            "|---|---:|---:|",
        ]
        for name in [
            "bm25_relative",
            "dense_relative",
            "title_coverage",
            "same_location",
            "location_probability",
            "history_click",
            "item_popularity",
            "rating_filter_satisfied",
        ]:
            if name in names:
                i = names.index(name)
                lines.append(f"| {name} | {positive[i]:.4f} | {negative[i]:.4f} |")
        lines.append("")
    lines += [
        "## Типы ошибок",
        "",
        "Метки эвристические и могут пересекаться. `candidate miss` — релевантного объявления нет в union. "
        "`close rank ≥50` — его позиция 51–100. `location diff` — у сравниваемых объявлений различается география. "
        "`low lexical` — слабый BM25 или покрытие заголовка; `low semantic` — очень низкая относительная cosine similarity либо заметное отставание от отвлекающего кандидата. "
        "`microcat diff` — у сравниваемых кандидатов различаются подкатегории.",
        "",
        "| Тип | Пропущенные релевантные объявления |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in tags.most_common())
    lines += ["", "## Примеры", ""]
    for number, example in enumerate(examples, start=1):
        query = example["query"]
        lines += [
            f"### {number}. {compact(query.search_query)}",
            "",
            f"Query key: `{example['key']}`; cold={bool(query.get('cold_text', False))}; Recall@50={example['recall']:.3f}; "
            f"положительных={example['relevant_count']}; search_location_id={query.search_location_id}; search_category={query.search_category}.",
            "",
            f"Фильтры: {compact(query.search_infm_params_text, 350)}.",
            "",
            f"Типы: {', '.join(example['categories'])}. Пропущенное объявление: позиция {example['target_rank'] if example['target_rank'] is not None else 'отсутствует в кандидатах'}. "
            f"Top-1 отсутствует в разметке: {example['top_is_false']}.",
            "",
            "| Роль | item_id | Заголовок | Локация | Микрокатегория | Параметры |",
            "|---|---|---|---:|---:|---|",
        ]
        for role, item in [
            ("Положительное", example["target"]),
            ("Первое отвлекающее", example["false"]),
        ]:
            if item is not None:
                lines.append(
                    f"| {role} | {item.item_id} | {compact(item.item_title_raw)} | {item.item_location_id} | {item.item_microcat_id} | {compact(item.item_infm_params_text)} |"
                )
        lines += ["", "| Признак | Положительное | Отвлекающее |", "|---|---:|---:|"]
        for name in [
            "bm25_relative",
            "dense_relative",
            "title_coverage",
            "description_coverage",
            "same_location",
            "location_probability",
            "microcat_probability",
            "supervised_microcat_relative",
            "supervised_microcat_rank",
            "filter_coverage",
            "rating_filter_satisfied",
            "history_click",
        ]:
            true_value = example["target_features"].get(name)
            false_value = example["false_features"].get(name)
            lines.append(
                f"| {name} | {'—' if true_value is None else f'{true_value:.4f}'} | {'—' if false_value is None else f'{false_value:.4f}'} |"
            )
        lines.append("")
    if not examples:
        lines.append(
            "Пропущенных релевантных объявлений в validation top-50 не найдено."
        )
    lines += [
        "",
        "## Интерпретация и следующие проверки",
        "",
        "При `candidate miss` помогает расширение retrieval-каналов; исправление ранжирования уже не вернёт отсутствующее объявление. "
        "Ошибки на границе top-50 указывают на настройку смеси признаков/модели. Географические ошибки требуют отдельной проверки "
        "регионального поиска и соседних городов. Низкое текстовое сходство требует анализа синонимов и описаний. "
        "В item-disjoint split истории положительных объявлений исключены: связь history_click/popularity с меткой может отличаться "
        "от финального корпуса, где часть объявлений известна по train.",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--model-name", help="Label an explicitly supplied score file")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/error_analysis.md"
    )
    parser.add_argument("--max-examples", type=int, default=40)
    args = parser.parse_args()
    summary = analyze(
        args.artifacts,
        args.reports,
        args.output,
        args.scores,
        args.max_examples,
        args.model_name,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
