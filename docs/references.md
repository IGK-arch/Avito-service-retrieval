# Источники общих идей

По просьбе автора выполнен обзор публичных описаний и исходных текстов.
Готовые ответы, веса, обученные на задаче, и notebook outputs не использовались.
Чужой код не запускался, не импортировался и не переносился в pipeline.

| Работа | Просмотренный commit | Направления проверки |
| --- | --- | --- |
| [boomchik93/AvitoDS_Bootcamp_TestTask](https://github.com/boomchik93/AvitoDS_Bootcamp_TestTask) | `956b6c8f3d2726f1f6f9b95114c5d6dd4f20743f` | Поля, география, supervised intent, ранги, grouped ranking |
| [alyaalyo/avito_nlp_item_for_queries](https://github.com/alyaalyo/avito_nlp_item_for_queries) | `5bc951c107bc8a3b1a04eb82ba3af146a9358700` | Региональные центры, query-relative признаки, contrastive encoder |

BM25, char TF-IDF, MiniLM и history transfer уже были реализованы до обзора.
Дальнейшие собственные проверки включили исправление собственного
географического fallback, признаки рангов/разниц и LightGBM LambdaRank.
Для отдельного Tiny эксперимента использовалась официальная базовая модель
cointegrated, собственный PyTorch loop и разрешённые train interactions.
Этот эксперимент не вошёл в отправленный ответ.

Веса и гиперпараметры выбирались по собственным измерениям. Оценки внешних
работ не заявляются как результат этой работы: group holdout и исключение
всех held-out item interactions измеряют разные условия обобщения.
