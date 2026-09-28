"""Read advertisement text in bounded batches, truncating before concatenation.

Long Parquet descriptions can require a gigabyte temporary Arrow buffer even
when only their prefix is used. This loader preserves row/id order and all
non-description fields while bounding that temporary allocation.
"""

import pandas as pd
import pyarrow.parquet as pq


def load_items(path, columns=None, description_chars=1800, batch_size=2048):
    file = pq.ParquetFile(path)
    wanted = list(columns or file.schema_arrow.names)
    texts = [
        c
        for c in wanted
        if c in ["item_title_raw", "item_description_raw", "item_infm_params_text"]
    ]
    other = [c for c in wanted if c not in texts]
    frame = (
        pd.read_parquet(path, columns=other)
        if other
        else pd.DataFrame(index=range(file.metadata.num_rows))
    )
    collected = {c: [] for c in texts}
    for batch in file.iter_batches(
        batch_size=batch_size, columns=texts, use_threads=False
    ):
        for c in texts:
            values = batch.column(batch.schema.get_field_index(c)).to_pylist()
            if c == "item_description_raw":
                values = [(x or "")[:description_chars] for x in values]
            collected[c].extend(values)
    for c, values in collected.items():
        frame[c] = values
    return frame[wanted]
