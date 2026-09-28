"""Train our offline Russian service retriever from an official public base.

Only ``download-base`` accesses the network, to fetch cointegrated/rubert-tiny2
from its publisher's Hugging Face repository. Training, evaluation and encoding
use local files. This is a separate, original PyTorch implementation; no weights
or implementation from another competition solution are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import unicodedata
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
BASE_REPO = "cointegrated/rubert-tiny2"
BASE_REVISION = "e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae"
SEED = 20260928


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_base(folder):
    """Fetch config/tokenizer/one weight file, pin revision, record provenance.

    No remote Python files are fetched or executed. The model card recommends
    normalized CLS pooling; it declares an MIT license and 29.4M parameters.
    """
    import requests

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        f"https://huggingface.co/api/models/{BASE_REPO}/revision/{BASE_REVISION}",
        timeout=60,
    )
    response.raise_for_status()
    info = response.json()
    revision = info["sha"]
    assert (
        revision == BASE_REVISION
    ), "Publisher revision differs from pinned official base"
    names = {entry["rfilename"] for entry in info["siblings"]}
    weight = (
        "model.safetensors" if "model.safetensors" in names else "pytorch_model.bin"
    )
    wanted = [
        name
        for name in (
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.txt",
            "special_tokens_map.json",
            "README.md",
            "LICENSE",
            weight,
        )
        if name in names
    ]
    assert (
        "config.json" in wanted and weight in wanted
    ), "Incomplete official model repository"
    files = {}
    for name in wanted:
        target = folder / name
        url = f"https://huggingface.co/{BASE_REPO}/resolve/{revision}/{name}"
        print(f"Downloading official base file: {name}", flush=True)
        with requests.get(url, stream=True, timeout=120) as stream:
            stream.raise_for_status()
            with target.open("wb") as output:
                for chunk in stream.iter_content(chunk_size=1024 * 1024):
                    output.write(chunk)
        files[name] = {
            "bytes": target.stat().st_size,
            "sha256": file_hash(target),
            "url": url,
        }
    provenance = {
        "source": BASE_REPO,
        "revision": revision,
        "license": "MIT",
        "model_card": f"https://huggingface.co/{BASE_REPO}/blob/{revision}/README.md",
        "pooling": "CLS token, L2 normalized (publisher model card)",
        "files": files,
    }
    write_json(folder / "base_provenance.json", provenance)
    print(json.dumps(provenance, ensure_ascii=False, indent=2), flush=True)


def text(value):
    import pandas as pd

    return "" if value is None or pd.isna(value) else " ".join(str(value).split())


def query_normalization(value):
    """Same Unicode/spelling normalization as our independently created split."""
    value = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")
    return re.sub(r"\s+", " ", value).strip()


def item_texts(frame, description_chars=400):
    return [
        ". ".join(
            part
            for part in (
                text(title),
                text(params),
                text(description)[:description_chars],
            )
            if part
        )
        for title, params, description in zip(
            frame.item_title_raw,
            frame.item_infm_params_text,
            frame.item_description_raw,
        )
    ]


def load_local(folder, device):
    # Even an accidentally incomplete folder must fail locally, not download.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(folder), local_files_only=True, trust_remote_code=False
    )
    model = AutoModel.from_pretrained(
        str(folder), local_files_only=True, trust_remote_code=False
    ).to(device)
    return tokenizer, model


def cls_vectors(model, tokens):
    import torch.nn.functional as F

    # Float32 normalization and loss improve numerical stability under AMP.
    values = model(**tokens).last_hidden_state[:, 0].float()
    return F.normalize(values, p=2, dim=1)


def prepare_pairs(args):
    import numpy as np
    import pandas as pd

    fit = pd.read_parquet(args.fit, columns=["search_query", "item_id"])
    fit["query_norm"] = fit.search_query.fillna("").map(query_normalization)
    original_rows = len(fit)
    fit = fit.drop_duplicates(["query_norm", "item_id"])
    # Cap common phrases before epoch sampling to keep a broad service mix.
    fit = fit.sample(frac=1, random_state=SEED)
    fit = fit[fit.groupby("query_norm").cumcount() < args.cap_per_query]
    items = pd.read_parquet(
        args.train_items,
        columns=[
            "item_id",
            "item_title_raw",
            "item_infm_params_text",
            "item_description_raw",
        ],
    )
    items = items.drop_duplicates("item_id")
    items = items[items.item_id.isin(fit.item_id)].copy()
    items["encoder_text"] = item_texts(items, args.description_chars)
    fit = fit.merge(
        items[["item_id", "encoder_text"]],
        on="item_id",
        how="left",
        validate="many_to_one",
    )
    assert (
        not fit.encoder_text.isna().any()
    ), "Fit positives missing original item texts"
    assert not fit.query_norm.eq("").any(), "Empty normalized fit query"
    # Explicit audit of the strict item/cold-text exclusions, in addition to
    # relying on fit_interactions produced by the main preparation pipeline.
    audit = {}
    if Path(args.eval_queries).is_file() and Path(args.targets).is_file():
        queries = pd.read_parquet(args.eval_queries)
        targets = json.loads(Path(args.targets).read_text(encoding="utf-8"))
        held_items = set().union(*(set(values) for values in targets.values()))
        cold = set(
            queries.loc[queries.cold_text, "search_query"].map(query_normalization)
        )
        assert not set(fit.item_id).intersection(
            held_items
        ), "Held item leaked into encoder training"
        assert not set(fit.query_norm).intersection(
            cold
        ), "Cold text leaked into encoder training"
        audit = {
            "held_item_overlap": 0,
            "cold_text_overlap": 0,
            "held_items": len(held_items),
            "cold_texts": len(cold),
        }
    query_ids, _ = pd.factorize(fit.query_norm)
    item_ids, _ = pd.factorize(fit.item_id)
    stats = {
        "input_rows": original_rows,
        "deduplicated_capped_pairs": len(fit),
        "unique_query_texts": int(fit.query_norm.nunique()),
        "unique_items": int(fit.item_id.nunique()),
        "cap_per_query": args.cap_per_query,
        "fit_path": str(Path(args.fit).resolve()),
        "fit_sha256": file_hash(args.fit),
        "pair_order_sha256": hashlib.sha256(
            "\n".join(fit.query_norm + "\t" + fit.item_id).encode()
        ).hexdigest(),
        "leakage_audit": audit,
    }
    return (
        fit.search_query.fillna("").map(text).tolist(),
        fit.encoder_text.tolist(),
        np.asarray(query_ids),
        np.asarray(item_ids),
        stats,
    )


def train(args):
    import numpy as np
    import torch
    import torch.nn.functional as F

    started = time.perf_counter()
    # A completion marker from an earlier run must not unblock a consumer of
    # newly trained weights until their new embedding caches have been audited.
    previous_marker = ROOT / "reports/tiny_encoder_validation.json"
    if previous_marker.is_file():
        previous_marker.unlink()
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = (
        "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    )
    if device == "auto":
        device = "cpu"
    queries, documents, query_ids, item_ids, pair_stats = prepare_pairs(args)
    print(json.dumps(pair_stats, ensure_ascii=False, indent=2), flush=True)
    tokenizer, model = load_local(Path(args.base), device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    per_epoch = min(args.max_pairs, len(queries))
    steps_epoch = math.ceil(per_epoch / args.batch_size)
    steps_total = steps_epoch * args.epochs
    warmup = max(1, int(0.05 * steps_total))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.1, (steps_total - step) / max(1, steps_total - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    rng = np.random.default_rng(SEED)
    global_step = 0
    epoch_stats = []
    for epoch in range(args.epochs):
        selected = rng.permutation(len(queries))[:per_epoch]
        total_loss = 0.0
        recent_losses = []
        epoch_started = time.perf_counter()
        for start in range(0, len(selected), args.batch_size):
            indices = selected[start : start + args.batch_size]
            q_tokens = tokenizer(
                [queries[i] for i in indices],
                max_length=args.max_query_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            d_tokens = tokenizer(
                [documents[i] for i in indices],
                max_length=args.max_item_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            q_tokens = {name: values.to(device) for name, values in q_tokens.items()}
            d_tokens = {name: values.to(device) for name, values in d_tokens.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda" if amp else "cpu", dtype=torch.float16, enabled=amp
            ):
                q_vectors = cls_vectors(model, q_tokens)
                d_vectors = cls_vectors(model, d_tokens)
            logits = (q_vectors @ d_vectors.T) / args.temperature
            qids = torch.as_tensor(query_ids[indices], device=device)
            iids = torch.as_tensor(item_ids[indices], device=device)
            # Same query may have several legitimate positives, and different
            # queries may click the same item: neither is an in-batch negative.
            false_negative = (qids[:, None] == qids[None, :]) | (
                iids[:, None] == iids[None, :]
            )
            false_negative.fill_diagonal_(False)
            logits = logits.masked_fill(false_negative, -10000.0)
            labels = torch.arange(len(indices), device=device)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite contrastive loss; stop before saving a broken model"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_value = float(loss.detach())
            total_loss += loss_value * len(indices)
            recent_losses.append(loss_value)
            global_step += 1
            if global_step % 50 == 0 or start + len(indices) == len(selected):
                record = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "steps_total": steps_total,
                    "epoch_pairs_done": start + len(indices),
                    "epoch_pairs": len(selected),
                    "mean_recent_loss": float(np.mean(recent_losses)),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "runtime_seconds": time.perf_counter() - started,
                }
                print(json.dumps(record, ensure_ascii=False), flush=True)
                with log_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                recent_losses = []
        elapsed = time.perf_counter() - epoch_started
        epoch_stats.append(
            {
                "epoch": epoch + 1,
                "pairs": len(selected),
                "mean_loss": total_loss / len(selected),
                "runtime_seconds": elapsed,
            }
        )
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(folder, safe_serialization=True)
    tokenizer.save_pretrained(folder)
    provenance_path = Path(args.base) / "base_provenance.json"
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.is_file()
        else {"source": BASE_REPO}
    )
    report = {
        "architecture": "Shared-weight BERT bi-encoder, normalized CLS pooling",
        "implementation": "Original local PyTorch loop; no reference solution code/weights",
        "base_provenance": provenance,
        "seed": SEED,
        "device": device,
        "epochs": args.epochs,
        "epoch_pair_limit": args.max_pairs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "loss": "Query-to-item in-batch cross-entropy; duplicate queries/items masked",
        "query_text": "search_query only",
        "item_text": "title + parameters + first description chars",
        "description_chars": args.description_chars,
        "max_query_length": args.max_query_length,
        "max_item_length": args.max_item_length,
        "pooling": "CLS",
        "dimension": model.config.hidden_size,
        "pair_stats": pair_stats,
        "epoch_stats": epoch_stats,
        "runtime_seconds": time.perf_counter() - started,
        "weights_sha256": file_hash(folder / "model.safetensors"),
    }
    write_json(folder / "training_report.json", report)
    write_json(Path(args.log).with_suffix(".summary.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.encode_after:
        encode_bundle(args, tokenizer, model, device)


def encode_texts(tokenizer, model, texts, device, batch_size, max_length):
    import numpy as np
    import torch

    result = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    # Similar lengths reduce padding. Restore original row order after encoding.
    order = np.argsort(
        np.fromiter((len(value) for value in texts), dtype=np.int64), kind="stable"
    )
    model.eval()
    amp = device.startswith("cuda")
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            tokens = tokenizer(
                [texts[i] for i in selected],
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokens = {name: values.to(device) for name, values in tokens.items()}
            with torch.autocast(
                device_type="cuda" if amp else "cpu", dtype=torch.float16, enabled=amp
            ):
                vectors = cls_vectors(model, tokens)
            result[selected] = vectors.cpu().numpy()
            if start % (batch_size * 100) == 0 or start + len(selected) == len(order):
                print(f"Encode {start + len(selected)}/{len(order)} rows", flush=True)
    assert np.isfinite(result).all()
    return result


def encode_bundle(args, tokenizer=None, model=None, device=None):
    import numpy as np
    import pandas as pd
    import torch

    torch.set_num_threads(args.cpu_threads)
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available() and args.device == "auto"
            else args.device
        )
        if device == "auto":
            device = "cpu"
    if model is None:
        tokenizer, model = load_local(Path(args.output), device)
    folder = Path(args.cache_output)
    folder.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            "items",
            args.corpus,
            "tiny_item_embeddings.npy",
            "item_id",
            args.max_item_length,
        ),
        (
            "queries",
            args.eval_queries,
            "tiny_eval_queries_embeddings.npy",
            "qkey",
            args.max_query_length,
        ),
        (
            "queries",
            args.benchmark_queries,
            "tiny_benchmark_queries_embeddings.npy",
            "query_id",
            args.max_query_length,
        ),
    ]
    for kind, source, name, id_column, max_length in jobs:
        started = time.perf_counter()
        frame = pd.read_parquet(source)
        if kind == "items" and args.expected_item_rows:
            assert (
                len(frame) == args.expected_item_rows
            ), "Corpus rows changed since split/embedding preparation"
        texts = (
            item_texts(frame, args.description_chars)
            if kind == "items"
            else frame.search_query.fillna("").map(text).tolist()
        )
        values = encode_texts(
            tokenizer, model, texts, device, args.encode_batch_size, max_length
        )
        norm_error = float(np.max(np.abs(np.linalg.norm(values, axis=1) - 1)))
        assert norm_error < 1e-5, "Embedding normalization failed"
        path = folder / name
        np.save(path, values)
        report = {
            "input": str(Path(source).resolve()),
            "rows": len(frame),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "id_column": id_column,
            "id_sha256": hashlib.sha256(
                "\n".join(frame[id_column].astype(str)).encode()
            ).hexdigest(),
            "embeddings_sha256": file_hash(path),
            "model_path": str(Path(args.output).resolve()),
            "weights_sha256": file_hash(Path(args.output) / "model.safetensors"),
            "pooling": "CLS, L2 normalized",
            "max_length": max_length,
            "description_chars": args.description_chars,
            "query_filters": False,
            "device": device,
            "amp": device.startswith("cuda"),
            "norm_max_abs_error": norm_error,
            "finite_check": "passed",
            "runtime_seconds": time.perf_counter() - started,
        }
        write_json(path.with_suffix(".json"), report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        del frame, values, texts
    validate_caches(args)


def validate_caches(args):
    """Independently audit all caches; publish the marker only after success."""
    import datetime

    import numpy as np
    import pandas as pd

    folder = Path(args.cache_output)
    config = json.loads((Path(args.output) / "config.json").read_text(encoding="utf-8"))
    dimension = config["hidden_size"]
    weights_hash = file_hash(Path(args.output) / "model.safetensors")
    jobs = [
        (args.corpus, "tiny_item_embeddings.npy", "item_id", args.expected_item_rows),
        (args.eval_queries, "tiny_eval_queries_embeddings.npy", "qkey", 6000),
        (
            args.benchmark_queries,
            "tiny_benchmark_queries_embeddings.npy",
            "query_id",
            2452,
        ),
    ]
    report = {
        "status": "passed",
        "model_path": str(Path(args.output).resolve()),
        "weights_sha256": weights_hash,
        "shapes": {},
        "caches": {},
    }
    for source, name, id_column, expected_rows in jobs:
        ids = pd.read_parquet(source, columns=[id_column])[id_column].astype(str)
        values = np.load(folder / name, mmap_mode="r")
        assert len(ids) == expected_rows and values.shape == (
            expected_rows,
            dimension,
        ), name
        assert values.dtype == np.float32, (name, "dtype")
        maximum_error = 0.0
        for start in range(0, len(values), 8192):
            batch = values[start : start + 8192]
            assert np.isfinite(batch).all(), (name, "non-finite")
            maximum_error = max(
                maximum_error, float(np.max(np.abs(np.linalg.norm(batch, axis=1) - 1)))
            )
        assert maximum_error < 1e-5, (name, "L2 norm", maximum_error)
        id_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        embedding_hash = file_hash(folder / name)
        manifest = json.loads(
            (folder / name).with_suffix(".json").read_text(encoding="utf-8")
        )
        assert manifest["id_sha256"] == id_hash, (name, "row order")
        assert manifest["embeddings_sha256"] == embedding_hash, (name, "file integrity")
        assert manifest["weights_sha256"] == weights_hash, (name, "model version")
        report["shapes"][name] = list(values.shape)
        report["caches"][name] = {
            "shape": list(values.shape),
            "dtype": "float32",
            "id_column": id_column,
            "id_sha256": id_hash,
            "embeddings_sha256": embedding_hash,
            "finite_check": "passed",
            "norm_max_abs_error": maximum_error,
            "runtime_seconds": manifest["runtime_seconds"],
        }
    report["completed_at_utc"] = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    marker = ROOT / "reports/tiny_encoder_validation.json"
    temporary_marker = marker.with_suffix(".tmp")
    write_json(temporary_marker, report)
    os.replace(temporary_marker, marker)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["download-base", "train", "encode", "validate"]
    )
    parser.add_argument("--base", default=str(ROOT / "models/rubert-tiny2-base"))
    parser.add_argument("--output", default=str(ROOT / "models/tiny-avito-eval"))
    parser.add_argument(
        "--fit", default=str(ROOT / "artifacts/fit_interactions.parquet")
    )
    parser.add_argument("--train-items", default=str(ROOT / "data/train.parquet"))
    parser.add_argument("--targets", default=str(ROOT / "artifacts/eval_targets.json"))
    parser.add_argument(
        "--eval-queries", default=str(ROOT / "artifacts/eval_queries.parquet")
    )
    parser.add_argument(
        "--benchmark-queries", default=str(ROOT / "data/benchmark_queries.parquet")
    )
    parser.add_argument("--corpus", default=str(ROOT / "artifacts/eval_items.parquet"))
    parser.add_argument("--cache-output", default=str(ROOT / "artifacts"))
    parser.add_argument("--expected-item-rows", type=int, default=195084)
    parser.add_argument("--log", default=str(ROOT / "reports/tiny_training.jsonl"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=120000)
    parser.add_argument("--cap-per-query", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--description-chars", type=int, default=400)
    parser.add_argument("--max-query-length", type=int, default=48)
    parser.add_argument("--max-item-length", type=int, default=128)
    parser.add_argument("--encode-after", action="store_true")
    args = parser.parse_args()
    if args.command == "download-base":
        download_base(args.base)
    elif args.command == "train":
        train(args)
    elif args.command == "validate":
        validate_caches(args)
    else:
        encode_bundle(args)


if __name__ == "__main__":
    main()
