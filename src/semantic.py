"""Offline semantic candidate generation with multilingual MiniLM.

The encoder is deliberately loaded only from local files.  The model is an
open-source sentence-transformers model and no external inference API is used.
Embeddings are L2-normalized, so their dot product is cosine similarity.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

# Set these before importing Hugging Face libraries: even optional metadata
# probes must not trigger network requests when the solution is reproduced.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

import numpy as np

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


def resolve_model_path(
    model_path: str | Path | None = None, cache_dir: str | Path | None = None
) -> Path:
    """Find the pinned local snapshot or an explicitly supplied model folder.

    For reproduction on another computer, copy the model snapshot and pass
    --model-path.  No download fallback is provided intentionally.
    """
    if model_path is not None:
        path = Path(model_path).expanduser().resolve()
    else:
        root = Path(
            cache_dir
            or os.environ.get("HF_HUB_CACHE", "")
            or Path.home() / ".cache" / "huggingface" / "hub"
        )
        path = (
            root
            / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
            / "snapshots"
            / MODEL_REVISION
        ).resolve()
    required = (
        "modules.json",
        "config.json",
        "tokenizer.json",
        "1_Pooling/config.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if not (
        (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()
    ):
        missing.append("model.safetensors or pytorch_model.bin")
    if missing:
        raise FileNotFoundError(f"Incomplete local model at {path}: {missing}")
    return path


def _text(value) -> str:
    # Parquet text columns can contain either None, pd.NA, or NaN.
    if value is None:
        return ""
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).split())


def item_texts(items, description_chars: int = 600) -> list[str]:
    """Title first, then service parameters and the start of the description.

    Descriptions often contain contacts or repeated boilerplate.  A short
    prefix captures service details without swamping the title; tokenizer
    truncation to 128 tokens is still the actual model input limit.
    """
    titles = items["item_title_raw"].tolist()
    params = items["item_infm_params_text"].tolist()
    descriptions = items["item_description_raw"].tolist()
    return [
        ". ".join(
            part for part in (_text(t), _text(p), _text(d)[:description_chars]) if part
        )
        for t, p, d in zip(titles, params, descriptions)
    ]


def query_texts(queries, include_filters: bool = False) -> list[str]:
    """Short queries are embedded as written; rating filters are numeric.

    Search filters can optionally be appended for a controlled experiment.
    Default retrieval uses only semantic intent, with location/rating/category
    handled as separate supervised features by the downstream solution.
    """
    texts = [_text(q) for q in queries["search_query"].tolist()]
    if include_filters:
        filters = queries["search_infm_params_text"].tolist()
        texts = [
            ". ".join(x for x in (q, _text(f)) if x) for q, f in zip(texts, filters)
        ]
    return texts


class SemanticEncoder:
    """Local sentence-transformer with deterministic inference and batching."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        device: str = "auto",
        max_seq_length: int = 128,
        half: bool = False,
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_path = resolve_model_path(model_path, cache_dir)
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.model = SentenceTransformer(
            str(self.model_path),
            device=self.device,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model.max_seq_length = max_seq_length
        self.model.eval()
        if half and self.device.startswith("cuda"):
            self.model.half()
        self.max_seq_length = max_seq_length
        self.half = half and self.device.startswith("cuda")

    def encode_texts(
        self, texts: Sequence[str], batch_size: int = 128, show_progress: bool = True
    ) -> np.ndarray:
        if not texts:
            return np.empty(
                (0, self.model.get_sentence_embedding_dimension()), dtype=np.float32
            )
        # encode normalizes after pooling; float32 output keeps later dot
        # products accurate even if the transformer itself uses half precision.
        values = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        values = np.asarray(values, dtype=np.float32)
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        return np.ascontiguousarray(values)

    def encode_items(
        self,
        items,
        batch_size: int = 128,
        description_chars: int = 600,
        show_progress: bool = True,
    ) -> np.ndarray:
        return self.encode_texts(
            item_texts(items, description_chars), batch_size, show_progress
        )

    def encode_queries(
        self,
        queries,
        batch_size: int = 128,
        include_filters: bool = False,
        show_progress: bool = True,
    ) -> np.ndarray:
        return self.encode_texts(
            query_texts(queries, include_filters), batch_size, show_progress
        )


def retrieve_topk(
    query_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    k: int = 200,
    device: str = "auto",
    query_batch_size: int = 128,
    item_batch_size: int = 32768,
) -> tuple[np.ndarray, np.ndarray]:
    """Return row indices and cosine scores, both shaped (queries, k).

    Chunking both axes bounds memory even for a large training corpus.  Each
    chunk's top-k is merged exactly, so this is exhaustive cosine retrieval
    rather than an approximate nearest-neighbour search.
    """
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    q = np.asarray(query_embeddings, dtype=np.float32)
    x = np.asarray(item_embeddings, dtype=np.float32)
    if q.ndim != 2 or x.ndim != 2 or q.shape[1] != x.shape[1]:
        raise ValueError("Embeddings must be 2-D matrices of equal dimension")
    k = min(max(0, k), len(x))
    result_indices = np.empty((len(q), k), dtype=np.int64)
    result_scores = np.empty((len(q), k), dtype=np.float32)
    if k == 0:
        return result_indices, result_scores
    with torch.inference_mode():
        for q_start in range(0, len(q), query_batch_size):
            q_batch = torch.as_tensor(
                q[q_start : q_start + query_batch_size], device=device
            )
            best_scores = torch.empty((len(q_batch), 0), device=device)
            best_indices = torch.empty(
                (len(q_batch), 0), dtype=torch.long, device=device
            )
            for x_start in range(0, len(x), item_batch_size):
                x_batch = torch.as_tensor(
                    x[x_start : x_start + item_batch_size], device=device
                )
                scores = q_batch @ x_batch.T
                scores, indices = torch.topk(scores, min(k, len(x_batch)), dim=1)
                indices += x_start
                combined_scores = torch.cat((best_scores, scores), dim=1)
                combined_indices = torch.cat((best_indices, indices), dim=1)
                best_scores, order = torch.topk(
                    combined_scores, min(k, combined_scores.shape[1]), dim=1
                )
                best_indices = combined_indices.gather(1, order)
            end = q_start + len(q_batch)
            result_indices[q_start:end] = best_indices.cpu().numpy()
            result_scores[q_start:end] = best_scores.cpu().numpy()
    return result_indices, result_scores


def pair_scores(
    query_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    query_indices: np.ndarray,
    item_indices: np.ndarray,
    batch_size: int = 100000,
) -> np.ndarray:
    """Cosine scores for aligned candidate pairs, without a full score matrix."""
    qi = np.asarray(query_indices, dtype=np.int64)
    ii = np.asarray(item_indices, dtype=np.int64)
    if qi.shape != ii.shape:
        raise ValueError("query_indices and item_indices must have the same shape")
    original_shape = qi.shape
    qi, ii = qi.ravel(), ii.ravel()
    scores = np.empty(len(qi), dtype=np.float32)
    for start in range(0, len(qi), batch_size):
        end = start + batch_size
        scores[start:end] = np.einsum(
            "ij,ij->i", query_embeddings[qi[start:end]], item_embeddings[ii[start:end]]
        )
    return scores.reshape(original_shape)


def smoke_test(encoder: SemanticEncoder) -> dict:
    texts = [
        "автоподбор",
        "Автоподбор и осмотр автомобиля перед покупкой",
        "баня на дровах",
        "Аренда русской бани на дровах",
        "монтаж видеодомофонов",
    ]
    embeddings = encoder.encode_texts(texts, batch_size=5, show_progress=False)
    indices, scores = retrieve_topk(
        embeddings,
        embeddings,
        k=3,
        device=encoder.device,
        query_batch_size=2,
        item_batch_size=2,
    )
    # Self similarities and a brute-force comparison catch encoding/retrieval
    # mistakes; the actual quality comparison belongs to the held-out split.
    dense = embeddings @ embeddings.T
    expected = np.take_along_axis(dense, indices, axis=1)
    assert np.all(np.isfinite(embeddings))
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5)
    assert np.allclose(expected, scores, atol=1e-5)
    assert np.allclose(
        pair_scores(embeddings, embeddings, np.arange(5), np.arange(5)), 1, atol=1e-5
    )
    return {
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "model_path": str(encoder.model_path),
        "device": encoder.device,
        "shape": list(embeddings.shape),
        "max_seq_length": encoder.max_seq_length,
        "half": encoder.half,
        "cosine_matrix": np.round(dense, 4).tolist(),
        "retrieval_check": "passed",
        "pair_score_check": "passed",
    }


def save_embeddings(
    output: str | Path,
    embeddings: np.ndarray,
    frame,
    input_path: str | Path,
    encoder: SemanticEncoder,
    kind: str,
    description_chars: int = 600,
    include_filters: bool = False,
    runtime_seconds: float | None = None,
    batch_size: int = 128,
) -> dict:
    """Save a cache plus a row-order/model manifest for reproducible reuse."""
    import hashlib

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, embeddings)
    id_column = (
        "item_id" if kind == "items" else "query_id" if "query_id" in frame else "qkey"
    )
    manifest = {
        "input": str(Path(input_path).resolve()),
        "rows": len(frame),
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "model_path": str(encoder.model_path),
        "max_seq_length": encoder.max_seq_length,
        "description_chars": description_chars,
        "include_filters": include_filters,
        "device": encoder.device,
        "half": encoder.half,
        "batch_size": batch_size,
        "runtime_seconds": runtime_seconds,
    }
    if id_column in frame:
        manifest["id_column"] = id_column
        manifest["id_sha256"] = hashlib.sha256(
            "\n".join(frame[id_column].astype(str)).encode()
        ).hexdigest()
    if runtime_seconds:
        manifest["rows_per_second"] = len(frame) / runtime_seconds
    output.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["smoke", "encode-items", "encode-queries", "encode-bundle"]
    )
    parser.add_argument("--input", help="Parquet source; row order is preserved")
    parser.add_argument("--output", help="Output .npy embeddings")
    parser.add_argument("--model-path")
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--description-chars", type=int, default=600)
    parser.add_argument("--include-filters", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument(
        "--eval-queries", help="Evaluation queries Parquet for encode-bundle"
    )
    parser.add_argument(
        "--benchmark-queries", help="Benchmark queries Parquet for encode-bundle"
    )
    args = parser.parse_args()
    started = time.perf_counter()
    encoder = SemanticEncoder(
        args.model_path, args.cache_dir, args.device, args.max_seq_length, args.half
    )
    print(
        f"Model loaded in {time.perf_counter() - started:.1f}s on {encoder.device}",
        flush=True,
    )
    if args.command == "smoke":
        print(json.dumps(smoke_test(encoder), ensure_ascii=False, indent=2))
        return
    if not args.input or not args.output:
        parser.error("encode commands require --input and --output")
    import pandas as pd

    if args.command == "encode-bundle":
        if not args.eval_queries or not args.benchmark_queries:
            parser.error(
                "encode-bundle also requires --eval-queries and --benchmark-queries"
            )
        folder = Path(args.output)
        jobs = [
            ("items", args.input, folder / "item_embeddings.npy"),
            ("queries", args.eval_queries, folder / "eval_queries_embeddings.npy"),
            (
                "queries",
                args.benchmark_queries,
                folder / "benchmark_queries_embeddings.npy",
            ),
        ]
    else:
        kind = "items" if args.command == "encode-items" else "queries"
        jobs = [(kind, args.input, Path(args.output))]
    for kind, source, output in jobs:
        step_started = time.perf_counter()
        frame = pd.read_parquet(source)
        print(f"Encoding {len(frame)} {kind} from {source}", flush=True)
        if kind == "items":
            embeddings = encoder.encode_items(
                frame, args.batch_size, args.description_chars
            )
        else:
            embeddings = encoder.encode_queries(
                frame, args.batch_size, args.include_filters
            )
        elapsed = time.perf_counter() - step_started
        manifest = save_embeddings(
            output,
            embeddings,
            frame,
            source,
            encoder,
            kind,
            args.description_chars,
            args.include_filters,
            elapsed,
            args.batch_size,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        del embeddings, frame
    print(
        f"Total encode process runtime: {time.perf_counter() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
