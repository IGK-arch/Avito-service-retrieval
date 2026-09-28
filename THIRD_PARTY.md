# Open-source компоненты

Версии экспорта: `requirements.txt`, обучения: `requirements-training.txt`,
проверки кода: `requirements-dev.txt`. Применение решения не обращается
к внешним inference API.

| Компонент | Использование | Лицензия / источник |
| --- | --- | --- |
| NumPy 1.26.4 | Массивы и top-k | [BSD-3-Clause](https://github.com/numpy/numpy) |
| pandas 2.3.0 | Таблицы и CSV | [BSD-3-Clause](https://github.com/pandas-dev/pandas) |
| PyArrow 18.0.0 | Parquet | [Apache-2.0](https://github.com/apache/arrow) |
| CatBoost 1.2.7 | Классификационный селектор | [Apache-2.0](https://github.com/catboost/catboost) |
| LightGBM 4.6.0 | LambdaRank | [MIT](https://github.com/microsoft/LightGBM) |
| SciPy 1.15.2 | Разреженные матрицы | [BSD-3-Clause](https://github.com/scipy/scipy) |
| scikit-learn 1.7.0 | TF-IDF, нормализация | [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn) |
| NLTK 3.9.1 | Russian Snowball | [Apache-2.0](https://github.com/nltk/nltk) |
| joblib 1.5.1 | Локальные кэши | [BSD-3-Clause](https://github.com/joblib/joblib) |
| psutil 6.1.0 | Контроль памяти | [BSD-3-Clause](https://github.com/giampaolo/psutil) |
| PyTorch 2.7.1 | Encoding, исследование Tiny | [BSD-style](https://github.com/pytorch/pytorch) |
| Transformers 4.48.2 | Локальные модели | [Apache-2.0](https://github.com/huggingface/transformers) |
| Sentence Transformers 4.1.0 | Frozen MiniLM | [Apache-2.0](https://github.com/UKPLab/sentence-transformers) |
| Black 24.8.0, Ruff 0.12.0, pytest 8.3.3 | Формат и проверки | MIT, MIT, MIT |

Базовая модель сабмита:
[`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2),
revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, Apache-2.0.
Модель не дообучалась; 128 токенов, 384 измерения, нормализованные embeddings.

Для отдельного эксперимента:
[`cointegrated/rubert-tiny2`](https://huggingface.co/cointegrated/rubert-tiny2),
revision `e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae`, MIT.
Собственные CatBoost/LightGBM checkpoints обучены на предоставленных данных.
