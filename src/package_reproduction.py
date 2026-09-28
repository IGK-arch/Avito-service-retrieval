"""Package or restore the exact candidate features used for the submission.

The archive is a computed feature cache, not a table of ready-made answers.
The exporter still runs both trained models and selects their top 50 results.
Large caches are published as a GitHub Release asset, outside Git history.
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from common import ROOT, A, log
from validate_answer import file_sha256

CACHE_FILES = [
    "benchmark_features.npz",
    "eval_item_ids.parquet",
    "benchmark_category_query_ids.npy",
    "item_embeddings.json",
]
MANIFEST = "reproduction_manifest.json"


def create(archive: Path) -> None:
    """Freeze the aligned feature/mapping files and their independent hashes."""
    pd.read_parquet(A / "eval_items.parquet", columns=["item_id"]).to_parquet(
        A / "eval_item_ids.parquet", index=False
    )
    config = json.loads((ROOT / "configs/submission.json").read_text(encoding="utf-8"))
    metadata = {
        "version": 1,
        "purpose": "Frozen candidate features for exact offline model inference",
        "contains_ready_answers": False,
        "expected_answer_sha256": config["answer_sha256"],
        "base_feature_count": config["candidate_feature_count"],
        "files": {
            name: {"sha256": file_sha256(A / name), "bytes": (A / name).stat().st_size}
            for name in CACHE_FILES
        },
    }
    (A / MANIFEST).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(
        temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=4
    ) as bundle:
        for name in CACHE_FILES + [MANIFEST]:
            log(f"Packaging reproduction artifact: {name}")
            bundle.write(A / name, f"artifacts/{name}")
    temporary.replace(archive)
    log(f"Reproduction bundle ready: {archive.name}, {archive.stat().st_size:,} bytes")


def extract(archive: Path, destination: Path) -> None:
    """Restore only documented cache files, checking paths and all checksums."""
    destination = destination.resolve()
    expected = {f"artifacts/{name}" for name in CACHE_FILES + [MANIFEST]}
    with zipfile.ZipFile(archive) as bundle:
        if set(bundle.namelist()) != expected or len(bundle.namelist()) != len(
            expected
        ):
            raise ValueError("Unexpected or duplicate files in reproduction archive")
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Archive path escapes destination: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    artifacts = destination / "artifacts"
    manifest = json.loads((artifacts / MANIFEST).read_text(encoding="utf-8"))
    for name, metadata in manifest["files"].items():
        if file_sha256(artifacts / name) != metadata["sha256"]:
            raise ValueError(f"Checksum mismatch after extraction: {name}")
    log("Reproduction cache extracted; all checksums match")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["create", "extract"])
    parser.add_argument(
        "--archive", type=Path, default=ROOT / "dist/reproduction-cache-v1.zip"
    )
    parser.add_argument("--destination", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "create":
        create(args.archive)
    else:
        extract(args.archive, args.destination)


if __name__ == "__main__":
    main()
