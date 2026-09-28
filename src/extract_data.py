"""Extract only the three task Parquets, including from the nested dataset.zip.

Only fixed basenames are written to the selected data directory. Unrelated
archive files are ignored, and no in-memory copy of the whole dataset is made.
"""

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

from common import ROOT, log

DATA_FILES = {"train.parquet", "benchmark_queries.parquet", "benchmark_items.parquet"}


def extract_parquets(archive: Path, destination: Path) -> bool:
    with zipfile.ZipFile(archive) as bundle:
        matches = {
            name: [entry for entry in bundle.namelist() if Path(entry).name == name]
            for name in DATA_FILES
        }
        if not all(matches.values()):
            return False
        if any(len(entries) != 1 for entries in matches.values()):
            raise ValueError("Ambiguous duplicate Parquet basenames in the archive")
        for name in sorted(DATA_FILES):
            log(f"Extracting dataset: {name}")
            with (
                bundle.open(matches[name][0]) as source,
                (destination / name).open("wb") as output,
            ):
                shutil.copyfileobj(source, output, length=1024 * 1024)
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "NLP_avito_interns.zip")
    parser.add_argument("--destination", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    if extract_parquets(args.archive, args.destination):
        return
    with zipfile.ZipFile(args.archive) as outer:
        nested = [name for name in outer.namelist() if Path(name).name == "dataset.zip"]
        if len(nested) != 1:
            raise ValueError("Expected three Parquets or a single nested dataset.zip")
        with tempfile.TemporaryDirectory(dir=args.destination) as temporary:
            path = Path(temporary) / "dataset.zip"
            with outer.open(nested[0]) as source, path.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if not extract_parquets(path, args.destination):
                raise ValueError(
                    "The nested archive does not contain all three task Parquets"
                )
    log("All task datasets extracted")


if __name__ == "__main__":
    main()
