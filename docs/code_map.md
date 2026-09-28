# Карта кода

| Модуль | Ответственность |
| --- | --- |
| `common.py` | Пути, контракт признаков, логи, Recall, отбор допустимых IDs |
| `extract_data.py` | Извлечение трёх Parquet из исходного или вложенного ZIP |
| `prepare_data.py` | Split, исключение interactions и подготовка корпуса |
| `lexical.py` | BM25 по полям, char TF-IDF, разреженный индекс |
| `history.py` | Перенос намерения и географии из обучающих запросов |
| `category_model.py` | MNB/ComplementNB и метрики микро-категорий |
| `load_items.py` | Batch чтение с ранней обрезкой описаний |
| `semantic.py` | Локальный MiniLM, cosine retrieval, manifests |
| `pipeline.py` | Объединение кандидатов, признаки, CatBoost baselines |
| `ranking_features.py` | Rank/gap признаки и стандартизация score |
| `rank_experiments.py` | LightGBM и сравнение ансамблей |
| `submit_best.py` | Точное воспроизведение отправленного CSV |
| `validate_answer.py` | Требования CSV, SHA и JSON отчёт |
| `package_reproduction.py` | Упаковка/проверка release-архива признаков |
| `analyze_errors.py` | Анализ сохранённых validation предсказаний |

## Дополнительные эксперименты

| Модуль | Статус относительно сабмита |
| --- | --- |
| `catboost_deep.py` | Depth 9 измерена, хуже depth 6, не выбрана |
| `ensemble_experiment.py` | Смесь CB6/CB8 измерена, не выбрана |
| `fix_geography.py` | Историческое исправление готового distance feature cache |
| `remote_features.py` | Online/remote признаки, не используются |
| `refine_experiment.py` | Remote/Tiny selector, финальная оценка не завершена |
| `tiny_encoder.py` | Собственное contrastive обучение ruBERT Tiny, не используется |
| `finalize.py` | Экспериментальное refit отдельных моделей, не использовано |

`pipeline.py refit/predict` относятся к раннему CatBoost baseline и создают
другой ответ. Для проверки отправленного решения запускается
**`submit_best.py`**. Данные и тяжёлые кэши исключены из Git;
архив точного воспроизведения передаётся проверяющему отдельно.
