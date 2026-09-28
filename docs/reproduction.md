# Воспроизведение

## Точное повторение отправленного файла

Каноническая команда — `python src/submit_best.py`. Она использует рецепт
`configs/submission.json`, два собственных checkpoint и замороженный пул
признаков. Готовый `answer.csv` не читается.

Локально подготовлен `reproduction-cache-v1.zip`, около 147 MB.
Архив передаётся проверяющему отдельно вместе с решением. Содержимое:

| Файл | Назначение |
| --- | --- |
| `benchmark_features.npz` | 2 420 904 пары query–item, 41 численный признак, позиции строк |
| `eval_item_ids.parquet` | Соответствие позиции корпуса строковому item_id |
| `benchmark_category_query_ids.npy` | Контроль порядка запросов |
| `item_embeddings.json` | Контроль порядка корпуса и provenance MiniLM |
| `reproduction_manifest.json` | SHA-256 и размеры каждого файла |

Это вычисленный кэш, не готовые top-50 списки. Он сохраняет значения
признаков исходного запуска и экономит повторный BM25/encoding.
После установки `requirements.txt`, размещения benchmark Parquet в `data/`
и получения архива:

```bash
python src/package_reproduction.py extract --archive reproduction-cache-v1.zip
python src/submit_best.py --output answer_reproduced.csv
python src/validate_answer.py --answer answer_reproduced.csv
```

Скрипт проверяет все SHA кэша, моделей и результата; при изменении
предсказаний итоговый файл не заменяется. Пути можно задать через
`--data-dir`, `--artifacts-dir`, `--checkpoints-dir`.
Работа проверена на Python 3.12 и Windows. Переносимые пути и CRLF CSV
предусмотрены для Linux/macOS; отдельная проверка на этих ОС не проводилась.

Интернет нужен для первоначального получения репозитория, библиотек и данных.
Архив воспроизведения предоставляется локальным файлом.
Применение моделей полностью офлайн. Для изолированной машины можно заранее
перенести зависимости как wheels. GPU для экспорта не нужен.

## Исследование с исходными Parquet

Полный путь ниже предназначен для повторения стадий и новых экспериментов.
Сохранённые checkpoint и кэш выше точно повторяют отправленный результат.
Новое обучение и encoding на другом оборудовании могут дать другой CSV;
он не заменяет автоматически зафиксированный сабмит.

Окружение обучения: Python 3.12, Windows, RAM 32 GB, RTX 4060 8 GB.
Установите `requirements-training.txt`. В исходном запуске использовался
`torch==2.7.1+cu118`; подходящий GPU wheel устанавливается из официального
CUDA 11.8 индекса PyTorch. Для CPU подходит `torch==2.7.1`.

Заранее разместите официальную
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, revision
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, в
`models/multilingual-MiniLM-L12-v2`. Код использует `local_files_only=True`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; загрузки модели во время запуска нет.

Стадии выполняются последовательно, чтобы не исчерпать Windows commit limit:

```bash
python src/extract_data.py --archive NLP_avito_interns.zip
python src/prepare_data.py
python src/pipeline.py build
python src/pipeline.py history --mode eval
python src/pipeline.py history --mode benchmark
python src/category_model.py
python src/semantic.py encode-bundle --input artifacts/eval_items.parquet \
  --output artifacts --eval-queries artifacts/eval_queries.parquet \
  --benchmark-queries data/benchmark_queries.parquet \
  --model-path models/multilingual-MiniLM-L12-v2 --device cuda --half
python src/pipeline.py features --mode eval
python src/pipeline.py features --mode benchmark
python src/pipeline.py train
python src/catboost_deep.py
python src/rank_experiments.py --threads 6
python src/rank_experiments.py --ensemble-only
```

Для CPU уберите `--half` и задайте `--device cpu`. В PowerShell для
многострочных команд вместо `\` используется backtick; можно также записать
каждую команду одной строкой. Новые модели пишутся в `artifacts/`, а не
в зафиксированные `checkpoints/`.

В исходном опыте географический fallback исправлен после генерации validation
пула: пул сохранён, его distance/geo признаки исправлены. Это отражено
в `reports/geography_fix.json` и `lgbm_metrics.json`. Новая генерация сразу
использует исправление и может поменять пул. Её метрики нельзя считать
тем же измерением; для точного сабмита используется frozen benchmark cache.

## Разработка

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check src tests
python -m black --check src tests
python -m pytest
python -m compileall -q src
```

Тесты проверяют границы групп, tie ordering, короткие пулы, фильтрацию
validation-only items, строковые IDs, повторы и целостность архива.
Полный replay с совпадением SHA является отдельной интеграционной проверкой.
