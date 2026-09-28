"""Local sparse retrieval for short Russian service queries.

The index contains only corpus features.  No relevance labels or query IDs are
used here, so the same object can be used for a validation corpus and benchmark.
Word retrieval is BM25 summed over separately normalized title, parameters and
description fields.  Optional title character TF-IDF covers misspellings and
words which were absent from the vocabulary.  All search is local/CPU-only.

Example::

    index = SparseRetriever(items_df, SparseConfig(use_char=True))
    result = index.query_batch(queries_df, top_k=150, channel="word")
    # result.indices are zero-based positions in items_df, not item_id values.
    candidate_ids = index.item_ids[result.indices[0]]
    word_feature = index.score_pairs(queries_df, query_positions, item_positions)

Use ``channel='hybrid'`` to add word and character scores after dividing each
query's scores by its channel maximum.  For a learned downstream ranker, request
the two raw channels separately with ``score_pairs``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from nltk.stem.snowball import RussianStemmer
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

_TOKEN_RE = re.compile(r"[a-zа-я0-9]+", re.IGNORECASE)
_RUSSIAN_RE = re.compile(r"[а-я]")
_STEMMER = RussianStemmer()


def clean_text(value: Any, max_chars: int | None = None) -> str:
    """Convert nulls safely and preserve all non-null text as text."""
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ""
    text = str(value).lower().replace("ё", "е")
    return text if max_chars is None else text[:max_chars]


@lru_cache(maxsize=300_000)
def _stem_token(token: str) -> str:
    # A cached light stemmer is considerably faster than running a morphology
    # parser for every occurrence in 189k long descriptions. English/model
    # names and numbers are kept intact, which matters for repair services.
    return _STEMMER.stem(token) if _RUSSIAN_RE.search(token) else token


def normalize_tokens(value: Any) -> list[str]:
    """Tokenize, unify е/ё and lightly stem Russian inflections."""
    return [_stem_token(token) for token in _TOKEN_RE.findall(clean_text(value))]


def normalize_query(value: Any) -> str:
    """A deterministic query key; useful for train-only query statistics."""
    return " ".join(normalize_tokens(value))


@dataclass(frozen=True)
class WordAnalyzer:
    """Pickle-safe sklearn analyzer; bigrams cannot collide with unigrams."""

    max_ngram: int = 2

    def __call__(self, value: Any) -> list[str]:
        tokens = normalize_tokens(value)
        features = ["w:" + token for token in tokens]
        if self.max_ngram >= 2:
            features.extend("b:" + a + "_" + b for a, b in zip(tokens, tokens[1:]))
        return features


@dataclass
class SparseConfig:
    # Titles are compact and precise; descriptions add paraphrases but also
    # advertising boilerplate. Separate BM25 length normalization prevents a
    # long description from reducing the importance of a short exact title.
    title_weight: float = 3.0
    params_weight: float = 1.0
    description_weight: float = 0.8
    title_b: float = 0.35
    params_b: float = 0.40
    description_b: float = 0.75
    k1: float = 1.2
    bigram_weight: float = 0.8
    word_ngrams: int = 2
    word_min_df: int = 1
    word_max_df: float = 0.98
    word_max_features: int | None = 500_000
    description_max_chars: int = 2_500
    query_params_weight: float = 0.15
    use_char: bool = False
    char_ngram_range: tuple[int, int] = (3, 5)
    char_min_df: int = 2
    char_max_features: int | None = 300_000
    hybrid_char_weight: float = 0.35
    batch_size: int = 32


@dataclass
class RetrievalResult:
    """Rows correspond to input queries; -1 denotes an unfilled position."""

    indices: np.ndarray
    scores: np.ndarray

    def __iter__(self):
        yield self.indices
        yield self.scores


def _records(data: Any) -> list[Mapping[str, Any]]:
    """Accept pandas frames, dictionaries and plain query strings."""
    if hasattr(data, "to_dict") and hasattr(data, "columns"):
        return data.to_dict("records")
    if isinstance(data, Mapping):
        return [data]
    if isinstance(data, str):
        return [{"search_query": data}]
    return [{"search_query": row} if isinstance(row, str) else row for row in data]


def _bm25_field(counts: sparse.csr_matrix, b: float, k1: float) -> sparse.csr_matrix:
    """Replace counts in place by the field's BM25 term-frequency factor."""
    counts = counts.tocsr().astype(np.float32, copy=False)
    lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    average = float(lengths.mean()) if lengths.size else 0.0
    if average <= 0:
        return counts
    normalizer = k1 * (1.0 - b + b * lengths / average)
    # repeat allocates one float per nonzero, not a dense document-term array.
    row_norm = np.repeat(normalizer, np.diff(counts.indptr))
    counts.data = (counts.data * (k1 + 1.0) / (counts.data + row_norm)).astype(
        np.float32
    )
    return counts


def _top_sparse(matrix: sparse.csr_matrix, top_k: int) -> RetrievalResult:
    """Extract top rows without materializing an Nquery x Nitem dense array."""
    matrix = matrix.tocsr()
    result_ids = np.full((matrix.shape[0], top_k), -1, dtype=np.int32)
    result_scores = np.zeros((matrix.shape[0], top_k), dtype=np.float32)
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row : row + 2]
        values = matrix.data[start:end]
        ids = matrix.indices[start:end]
        positive = values > 0
        values, ids = values[positive], ids[positive]
        take = min(top_k, len(values))
        if take == 0:
            continue
        chosen = np.argpartition(values, len(values) - take)[-take:]
        # Ties use the corpus position as a deterministic secondary order.
        chosen = chosen[np.lexsort((ids[chosen], -values[chosen]))]
        result_ids[row, :take] = ids[chosen]
        result_scores[row, :take] = values[chosen]
    return RetrievalResult(result_ids, result_scores)


def _row_max_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    matrix = matrix.tocsr()
    maximum = np.asarray(matrix.max(axis=1).toarray()).ravel()
    inverse = np.zeros_like(maximum, dtype=np.float32)
    np.divide(1.0, maximum, out=inverse, where=maximum > 0)
    return sparse.diags(inverse, format="csr") @ matrix


class SparseRetriever:
    """BM25 field index plus an optional title-character cosine index.

    Item positions remain aligned with the input frame, including after save /
    load. The index never reads any labels or hard-codes any benchmark queries.
    """

    def __init__(self, items: Any = None, config: SparseConfig | None = None):
        self.config = config or SparseConfig()
        self.item_ids = np.empty(0, dtype=object)
        self.word_vectorizer: CountVectorizer | None = None
        self.char_vectorizer: TfidfVectorizer | None = None
        self.word_matrix: sparse.csr_matrix | None = None
        self.char_matrix: sparse.csr_matrix | None = None
        self.build_stats: dict[str, Any] = {}
        if items is not None:
            self.fit(items)

    def fit(self, items: Any) -> "SparseRetriever":
        """Build once from the corpus. No large intermediate dense matrix."""
        started = time.monotonic()
        # Avoid copying ratings, decimal coordinates and other unused corpus
        # columns into Python dictionaries when a large DataFrame is supplied.
        if hasattr(items, "columns"):
            keep = [
                name
                for name in (
                    "item_id",
                    "item_title_raw",
                    "item_infm_params_text",
                    "item_description_raw",
                )
                if name in items.columns
            ]
            rows = _records(items[keep])
        else:
            rows = _records(items)
        if not rows:
            raise ValueError("Cannot build an index on an empty item corpus")
        cfg = self.config
        self.item_ids = np.asarray([str(row["item_id"]) for row in rows], dtype=object)
        titles = [clean_text(row.get("item_title_raw")) for row in rows]
        params = [clean_text(row.get("item_infm_params_text")) for row in rows]
        descriptions = [
            clean_text(row.get("item_description_raw"), cfg.description_max_chars)
            for row in rows
        ]
        del rows
        self.word_vectorizer = CountVectorizer(
            analyzer=WordAnalyzer(cfg.word_ngrams),
            dtype=np.float32,
            min_df=cfg.word_min_df,
            max_df=cfg.word_max_df,
            max_features=cfg.word_max_features,
        )
        # The combined document is used only to learn document frequencies and
        # one shared vocabulary. Field-specific counts below carry the boosts.
        combined = self.word_vectorizer.fit_transform(
            f"{title} {param} {description}"
            for title, param, description in zip(titles, params, descriptions)
        ).tocsr()
        document_frequency = np.bincount(combined.indices, minlength=combined.shape[1])
        n_items = len(titles)
        idf = np.log1p(
            (n_items - document_frequency + 0.5) / (document_frequency + 0.5)
        ).astype(np.float32)
        del combined
        # Bigram evidence is useful for compounds such as "баня на дровах",
        # but a single rare bigram should not overwhelm all unigram evidence.
        if cfg.bigram_weight != 1.0:
            names = self.word_vectorizer.get_feature_names_out()
            idf[
                np.fromiter((name.startswith("b:") for name in names), dtype=bool)
            ] *= cfg.bigram_weight
        merged: sparse.csr_matrix | None = None
        for texts, boost, b in (
            (titles, cfg.title_weight, cfg.title_b),
            (params, cfg.params_weight, cfg.params_b),
            (descriptions, cfg.description_weight, cfg.description_b),
        ):
            if boost <= 0:
                continue
            field = _bm25_field(self.word_vectorizer.transform(texts), b, cfg.k1)
            field.data *= np.float32(boost)
            merged = field if merged is None else merged + field
            del field
        if merged is None:
            raise ValueError("At least one text field must have positive weight")
        self.word_matrix = merged.multiply(idf).tocsr().astype(np.float32)
        self.word_matrix.eliminate_zeros()
        self.word_matrix.sort_indices()
        self.char_matrix, self.char_vectorizer = None, None
        if cfg.use_char:
            self.char_vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=cfg.char_ngram_range,
                min_df=min(cfg.char_min_df, n_items),
                max_features=cfg.char_max_features,
                dtype=np.float32,
                sublinear_tf=True,
                norm="l2",
            )
            self.char_matrix = self.char_vectorizer.fit_transform(titles).tocsr()
            self.char_matrix.sort_indices()
        self.build_stats = {
            "items": n_items,
            "word_features": self.word_matrix.shape[1],
            "word_nnz": self.word_matrix.nnz,
            "char_features": (
                0 if self.char_matrix is None else self.char_matrix.shape[1]
            ),
            "char_nnz": 0 if self.char_matrix is None else self.char_matrix.nnz,
            "seconds": round(time.monotonic() - started, 3),
            "word_csr_mb": round(
                sum(
                    x.nbytes
                    for x in (
                        self.word_matrix.data,
                        self.word_matrix.indices,
                        self.word_matrix.indptr,
                    )
                )
                / 2**20,
                2,
            ),
        }
        return self

    build = fit

    def _query_matrix(
        self, rows: Sequence[Mapping[str, Any]], channel: str
    ) -> sparse.csr_matrix:
        if self.word_matrix is None or self.word_vectorizer is None:
            raise RuntimeError("Call fit(items) before retrieval")
        query_texts = [clean_text(row.get("search_query")) for row in rows]
        if channel == "char":
            if self.char_vectorizer is None:
                raise ValueError(
                    "Character index is disabled; set SparseConfig(use_char=True)"
                )
            return self.char_vectorizer.transform(query_texts).tocsr()
        if channel != "word":
            raise ValueError("Raw query matrix channel must be 'word' or 'char'")
        query = self.word_vectorizer.transform(query_texts).tocsr()
        # Queries are short: one occurrence is enough. Repeated user words must
        # not drown out the rest of the query.
        query.data.fill(1.0)
        if self.config.query_params_weight > 0:
            filters = self.word_vectorizer.transform(
                [clean_text(row.get("search_infm_params_text")) for row in rows]
            ).tocsr()
            filters.data.fill(self.config.query_params_weight)
            query = query + filters
        return query.astype(np.float32)

    def query_batch(
        self,
        queries: Any,
        top_k: int = 50,
        channel: str = "word",
        batch_size: int | None = None,
    ) -> RetrievalResult:
        """Return corpus row positions and scores for each input query.

        Channels: word = raw BM25; char = cosine; hybrid = max-normalized
        BM25 + hybrid_char_weight * max-normalized cosine. Empty/out-of-
        vocabulary queries can have fewer than K hits; padded positions are -1.
        """
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if channel not in {"word", "char", "hybrid"}:
            raise ValueError("channel must be 'word', 'char', or 'hybrid'")
        rows = _records(queries)
        width = min(top_k, len(self.item_ids))
        output_ids = np.full((len(rows), width), -1, dtype=np.int32)
        output_scores = np.zeros((len(rows), width), dtype=np.float32)
        if not rows or width == 0:
            return RetrievalResult(output_ids, output_scores)
        batch_size = batch_size or self.config.batch_size
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(rows), batch_size):
            part = rows[start : start + batch_size]
            if channel in {"word", "hybrid"}:
                scores = (self._query_matrix(part, "word") @ self.word_matrix.T).tocsr()
            if channel == "char":
                scores = (self._query_matrix(part, "char") @ self.char_matrix.T).tocsr()
            elif channel == "hybrid":
                char_scores = (
                    self._query_matrix(part, "char") @ self.char_matrix.T
                ).tocsr()
                scores = _row_max_normalize(
                    scores
                ) + self.config.hybrid_char_weight * _row_max_normalize(char_scores)
            result = _top_sparse(scores, width)
            output_ids[start : start + len(part)] = result.indices
            output_scores[start : start + len(part)] = result.scores
        return RetrievalResult(output_ids, output_scores)

    def query_one(
        self, query: Any, top_k: int = 50, channel: str = "word"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single-query convenience API, returning valid positions only."""
        result = self.query_batch(query, top_k, channel)
        if len(result.indices) != 1:
            raise ValueError("query_one expects exactly one query")
        valid = result.indices[0] >= 0
        return result.indices[0, valid], result.scores[0, valid]

    def score_pairs(
        self,
        queries: Any,
        query_indices: Iterable[int],
        item_indices: Iterable[int],
        channel: str = "word",
        batch_size: int = 20_000,
    ) -> np.ndarray:
        """Raw dot-product features for arbitrary aligned (query,item) pairs.

        This scores candidates without allocating a complete query-item score
        matrix. ``query_indices`` refer to rows of queries; ``item_indices`` to
        the original corpus rows. Ask for word and char separately when ranking.
        """
        query_indices = np.asarray(query_indices, dtype=np.int64)
        item_indices = np.asarray(item_indices, dtype=np.int64)
        if query_indices.shape != item_indices.shape or query_indices.ndim != 1:
            raise ValueError("Pair arrays must be one-dimensional and the same length")
        rows = _records(queries)
        q = self._query_matrix(rows, channel)
        items = self.word_matrix if channel == "word" else self.char_matrix
        if np.any(query_indices < 0) or np.any(query_indices >= len(rows)):
            raise IndexError("query_indices outside query rows")
        if np.any(item_indices < 0) or np.any(item_indices >= len(self.item_ids)):
            raise IndexError("item_indices outside corpus rows")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        output = np.zeros(len(item_indices), dtype=np.float32)
        for start in range(0, len(item_indices), batch_size):
            end = start + batch_size
            dot = q[query_indices[start:end]].multiply(items[item_indices[start:end]])
            output[start:end] = np.asarray(dot.sum(axis=1)).ravel()
        return output

    def save(self, path: str | Path) -> None:
        """Persist the local index; compress=0 allows faster load/mmap."""
        joblib.dump(self, str(path), compress=0)

    @staticmethod
    def load(path: str | Path, mmap_mode: str | None = None) -> "SparseRetriever":
        return joblib.load(str(path), mmap_mode=mmap_mode)
