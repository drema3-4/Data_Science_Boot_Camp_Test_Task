# Avito Data Science Bootcamp Test Task

Проект решает задачу информационного поиска по статьям справки Avito: по текстовому запросу нужно вернуть релевантные `article_id` в формате `answer.csv`.

Основная идея решения - построить модульный retrieval pipeline вокруг Qdrant: отдельно вынесены загрузка и обработка данных, конфигурация моделей, построение коллекции, поиск кандидатов, rerank и оценка качества на `calibration.f`.

## Подход

### 1. Обработка HTML статей

Исходные статьи лежат в `src/data_sources/raw/articles.f`. В них поле `body` содержит HTML.

Обработка выполняется в `src/data/processing.py`:

- HTML парсится через `BeautifulSoup`;
- удаляются служебные теги `script`, `style`, `noscript`;
- из HTML извлекается чистый текст через `get_text(separator=" ", strip=True)`;
- дополнительно строится лексическая версия текста: нижний регистр, выделение слов и чисел регулярным выражением, объединение токенов пробелами.

В итоге для каждой статьи получается структура `BaseProcessedArticle`:

- `article_id` - идентификатор статьи;
- `title` - заголовок;
- `body_plain` - очищенный текст статьи;
- `body_lexical` - нормализованный текст для BM25.

### 2. Гибридный поиск

Для каждой статьи строятся несколько векторных представлений:

- `title_sparse` - BM25 sparse-вектор по заголовку;
- `body_sparse` - BM25 sparse-вектор по тексту статьи;
- `title_dense` - dense-вектор по заголовку;
- `body_dense` - dense-вектор по тексту статьи.

Схема коллекции описана в `src/indexing/collection_schema.py`.

Sparse-вектора строятся моделью:

```text
Qdrant/bm25
```

Dense-вектора строятся моделью:

```text
intfloat/multilingual-e5-small
```

Для dense-модели используется размерность `384`. Модель подходит для мультиязычного семантического поиска и хорошо работает с русскоязычными запросами.

### 3. Объединение сигналов

Поиск кандидатов выполняется в `src/retrieval/retrievers.py`.

Для каждого поля создаётся отдельный запрос в Qdrant. Затем результаты объединяются через RRF:

```text
Reciprocal Rank Fusion
```

Вес полей задаётся в схеме коллекции:

- заголовок получает больший вес, потому что часто содержит наиболее точное описание проблемы;
- текст статьи получает меньший вес, но даёт больше контекста.

### 4. Rerank

После candidate generation используется reranker:

```text
Qwen/Qwen3-Reranker-0.6B
```

Он получает пару:

```text
(query, document)
```

и пересортировывает кандидатов по более точной cross-encoder оценке. Это особенно полезно, когда BM25/dense retrieval уже нашёл правильные статьи, но поставил их не на первые позиции.

## Архитектура

Проект разделён на независимые слои.

### `src/app/bootstrap.py`

Точка сборки зависимостей.

Здесь задаются:

- пути к данным;
- URL Qdrant;
- конфигурации sparse/dense/reranker моделей;
- фабрика моделей;
- схема коллекции;
- Qdrant store;
- search pipeline;
- evaluator.

Идея файла: остальной код не должен вручную собирать все зависимости. Скрипты запуска просто вызывают готовые builder-функции.

### `src/data/`

Работа с данными:

- `loaders.py` - загрузка `.f` файлов через pandas/feather;
- `processing.py` - очистка HTML и подготовка текстовых представлений;
- `types.py` - dataclass-структуры для статей и calibration samples.

### `src/models/`

Обёртки над ML-моделями:

- `configs.py` - dataclass-конфиги моделей;
- `embedders.py` - sparse/dense embedders на FastEmbed;
- `rerankers.py` - CrossEncoder reranker;
- `factory.py` - фабрика, которая создаёт модель по её конфигу.

### `src/indexing/`

Индексация в Qdrant:

- `collection_schema.py` - описание коллекции и vector fields;
- `point_builder.py` - построение `PointStruct` для Qdrant;
- `qdrant_store.py` - создание коллекции и загрузка points.

### `src/retrieval/`

Поиск:

- `retrievers.py` - hybrid retriever поверх Qdrant;
- `pipeline.py` - общий search pipeline: retrieval -> optional rerank -> финальный top-k;
- `types.py` - структура ответа pipeline.

### `src/evaluation/`

Оценка качества:

- `metrics.py` - `recall@k`, `MAP@k`;
- `evaluator.py` - запуск поиска по calibration samples и расчёт метрик.

### `src/scripts/`

Скрипты для запуска:

- `index_articles.py` - построение и загрузка коллекции в Qdrant;
- `evaluate.py` - расчёт метрик на `calibration.f`;
- `search_debug.py` - ручная проверка поиска на одном запросе.

## Структура проекта

```text
.
├── pyproject.toml
├── poetry.lock
├── README.md
└── src
    ├── app
    │   └── bootstrap.py
    ├── data
    │   ├── loaders.py
    │   ├── processing.py
    │   └── types.py
    ├── data_sources
    │   └── raw
    │       ├── articles.f
    │       ├── calibration.f
    │       ├── test.f
    │       └── candidate_statement_ru.md
    ├── evaluation
    │   ├── evaluator.py
    │   └── metrics.py
    ├── indexing
    │   ├── collection_schema.py
    │   ├── point_builder.py
    │   └── qdrant_store.py
    ├── models
    │   ├── configs.py
    │   ├── embedders.py
    │   ├── factory.py
    │   └── rerankers.py
    ├── retrieval
    │   ├── pipeline.py
    │   ├── retrievers.py
    │   └── types.py
    ├── scripts
    │   ├── evaluate.py
    │   ├── index_articles.py
    │   └── search_debug.py
    ├── runs
    │   └── ...
    └── docker-compose.yml
```

`src/qdrant_storage/` - локальное хранилище Qdrant. Его можно пересоздать повторной индексацией, поэтому это не часть логики решения.

## Стек

- Python `3.12.12`;
- pandas;
- pyarrow / feather;
- BeautifulSoup4;
- Qdrant;
- qdrant-client;
- FastEmbed;
- sentence-transformers;
- PyTorch;
- tqdm;
- Docker / Docker Compose;
- Poetry.

## Как запустить проект

Ниже команды приведены для запуска из корня проекта.

### 1. Установить зависимости

```powershell
poetry install
```

Если используется уже созданное локальное окружение `.venv`, можно запускать скрипты через:

```powershell
.\.venv\Scripts\python.exe
```

### 2. Запустить Qdrant

Docker Compose файл лежит в `src/docker-compose.yml`, поэтому запускать Qdrant удобнее из папки `src`:

```powershell
cd src
docker compose up -d
cd ..
```

Qdrant будет доступен по адресу:

```text
http://127.0.0.1:6333
```

### 3. Построить индекс

```powershell
poetry run python src/scripts/index_articles.py
```

Скрипт:

- загружает `articles.f`;
- очищает HTML;
- строит sparse и dense vectors;
- создаёт коллекцию `base_collection`;
- загружает points в Qdrant.

Первый запуск может быть дольше, потому что модели скачиваются и кэшируются локально.

### 4. Проверить поиск вручную

```powershell
poetry run python src/scripts/search_debug.py
```

Скрипт запускает pipeline на одном тестовом запросе и печатает найденные `article_id`, score и title.

### 5. Посчитать метрики на calibration.f

```powershell
poetry run python src/scripts/evaluate.py
```

Скрипт:

- загружает `calibration.f`;
- запускает поиск по каждому запросу;
- считает `recall@10`, `MAP@10`, `candidate_recall@50`;
- сохраняет результат в `src/runs/<timestamp>_base/metrics.json`.

## Как получить answer.csv

Финальный файл для платформы должен иметь две колонки:

```text
query_id,answer
```

В колонке `answer` должен лежать список из найденных `article_id`, записанных одной строкой через пробел:

```text
query_id,answer
1,1909 4396 4403 2695 4433 2408 4219 4218 4217 4216
```

В текущей структуре проекта уже есть все основные компоненты для генерации ответов: загрузка данных, search pipeline и Qdrant index. Если отдельный скрипт генерации ответов ещё не вынесен в `src/scripts/make_answers.py`, его логика должна быть такой:

1. загрузить `test.f`;
2. собрать `SearchPipeline` через `build_search_pipeline(with_reranker=True)`;
3. для каждого `query_text` получить top-10 результатов;
4. взять `article_id` из результатов;
5. сохранить CSV с колонками `query_id` и `answer`.

Ожидаемый запуск такого скрипта:

```powershell
poetry run python src/scripts/make_answers.py
```

Ожидаемый результат:

```text
src/answers/base_model_answers/answer.csv
```

Важно: перед генерацией `answer.csv` Qdrant должен быть запущен, а коллекция должна быть предварительно проиндексирована через `src/scripts/index_articles.py`.

## Метрики

Сохранённые прогоны:

```text
src/runs/20260719_201906_base/metrics.json
src/runs/20260720_032002_section_chunked/metrics.json
```

Результаты:

| Прогон | Схема | Параметры поиска | `recall@10` | `map@10` | `candidate_recall@50` |
|---|---|---|---:|---:|---:|
| `20260719_201906_base` | base | `candidate_limit=50` | `0.3333333333` | `0.1944444444` | `1.0` |
| `20260720_032002_section_chunked` | section_chunked | `candidate_limit=150`, `prefetch_limit=200`, `final_limit=10` | `0.8241666667` | `0.4985039683` | - |

Интерпретация:

- базовый прогон показал `MAP@10 = 0.1944` при `candidate_recall@50 = 1.0`, то есть нужные статьи часто находились среди кандидатов, но не всегда поднимались в финальный top-10;
- section chunking заметно улучшил качество: `recall@10` вырос с `0.3333` до `0.8242`, а `MAP@10` - с `0.1944` до `0.4985`;
- основной выигрыш, вероятно, связан с тем, что длинные статьи стали представлены более точными секционными фрагментами, и запросы перестали теряться внутри большого `body`.

## Анализ ошибок и возможные улучшения

По текущим метрикам видно, что candidate generation работает сильнее, чем финальная сортировка. Это значит, что правильные документы часто находятся, но не всегда оказываются достаточно высоко в top-10.

Что можно улучшить дальше:

- подобрать веса `title_sparse`, `body_sparse`, `title_dense`, `body_dense` на `calibration.f`;
- отдельно сравнить качество без reranker и с reranker;
- продолжить эксперименты с chunking: размер секций, overlap, способ агрегации chunk-level результатов обратно в article-level выдачу;
- добавить сохранение подробного error analysis: запрос, ground truth, top-10, позиции релевантных статей;
- расширить evaluation скрипт, чтобы он сохранял не только aggregate metrics, но и per-query результаты;
- вынести генерацию `answer.csv` в отдельный воспроизводимый скрипт.

## Воспроизводимость

Чтобы воспроизвести результат:

1. установить зависимости;
2. запустить Qdrant;
3. выполнить индексацию;
4. запустить evaluation или генерацию ответов.

При повторном запуске важно использовать те же:

- версии зависимостей из `poetry.lock`;
- модели;
- параметры схемы коллекции;
- параметры поиска `candidate_limit`, `prefetch_limit`, `final_limit`;
- reranker config.

## Ключевые файлы для проверки

- `src/app/bootstrap.py` - как собирается весь pipeline;
- `src/indexing/collection_schema.py` - какие вектора создаются;
- `src/retrieval/retrievers.py` - как работает hybrid retrieval;
- `src/retrieval/pipeline.py` - как retrieval объединён с reranker;
- `src/evaluation/metrics.py` - как считаются метрики;
- `src/scripts/index_articles.py` - как воспроизвести индекс;
- `src/scripts/evaluate.py` - как воспроизвести метрики.
