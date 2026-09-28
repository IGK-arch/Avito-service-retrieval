"""Transfer training evidence to new query texts and new advertisements.

Nearest training query texts vote for service microcategories. Historical
search/item location pairs also teach the model region-to-city relationships.
No benchmark labels or manually specified query-specific answers are used.
"""

import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def norm(text):
    return " ".join(re.findall(r"[\w]+", str(text or "").lower().replace("ё", "е")))


class HistoryIndex:
    def __init__(self, interactions, items):
        self.n_items = len(items)
        self.item_map = {x: i for i, x in enumerate(items.item_id)}
        self.microcats = np.sort(interactions.item_microcat_id.unique())
        self.mmap = {x: i for i, x in enumerate(self.microcats)}
        t = interactions.copy()
        t["text"] = t.search_query.fillna("").map(norm)
        self.texts = t.text.unique().tolist()
        self.textmap = {x: i for i, x in enumerate(self.texts)}
        self.word = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=180000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=150000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        # Independent normalized word/character channels tolerate Russian endings
        # and small misspellings while retaining exact uncommon service names.
        self.matrix = normalize(
            sparse.hstack(
                [
                    self.word.fit_transform(self.texts) * 0.8,
                    self.char.fit_transform(self.texts) * 0.6,
                ],
                format="csr",
            )
        )
        qt = t.text.map(self.textmap).to_numpy()
        mi = t.item_microcat_id.map(self.mmap).to_numpy()
        self.votes = sparse.csr_matrix(
            (np.ones(len(t), np.float32), (qt, mi)),
            shape=(len(self.texts), len(self.microcats)),
        )
        self.votes = normalize(self.votes, norm="l1")
        # Pair evidence only applies to advertisements actually present in corpus.
        in_corpus = t.item_id.isin(self.item_map)
        ct = t.loc[in_corpus]
        self.clicks = defaultdict(dict)
        for (text, iid), n in ct.groupby(["text", "item_id"]).size().items():
            self.clicks[text][self.item_map[iid]] = int(n)
        pop = t.item_id.value_counts()
        self.pop = np.log1p(items.item_id.map(pop).fillna(0).to_numpy(dtype=np.float32))
        loc = t.groupby(["search_location_id", "item_location_id"]).size()
        sums = t.groupby("search_location_id").size()
        self.loc_probs = {
            (int(a), int(b)): float(c / sums.loc[a]) for (a, b), c in loc.items()
        }
        # Coordinate medians are robust to multiple advertisements in one city.
        xy = t[["item_location_id", "item_latitude", "item_longitude"]].copy()
        xy["item_latitude"] = pd.to_numeric(xy.item_latitude, errors="coerce")
        xy["item_longitude"] = pd.to_numeric(xy.item_longitude, errors="coerce")
        self.centers = (
            xy.groupby("item_location_id")[["item_latitude", "item_longitude"]]
            .median()
            .to_dict("index")
        )
        self.search_centers = {}
        for sl, group in t.groupby("search_location_id"):
            mainloc = group.item_location_id.value_counts().index[0]
            self.search_centers[int(sl)] = self.centers.get(
                int(sl), self.centers.get(int(mainloc), {})
            )
        self.item_m = np.array([self.mmap.get(x, -1) for x in items.item_microcat_id])
        self.item_locs = items.item_location_id.to_numpy(np.int64)
        self.unique_locs, self.loc_inverse = np.unique(
            self.item_locs, return_inverse=True
        )
        self.item_lat = (
            pd.to_numeric(items.item_latitude, errors="coerce")
            .fillna(0)
            .to_numpy(np.float32)
        )
        self.item_lon = (
            pd.to_numeric(items.item_longitude, errors="coerce")
            .fillna(0)
            .to_numpy(np.float32)
        )

    def query_features(self, queries, k=25):
        texts = queries.search_query.fillna("").map(norm).tolist()
        mat = normalize(
            sparse.hstack(
                [self.word.transform(texts) * 0.8, self.char.transform(texts) * 0.6],
                format="csr",
            )
        )
        votes = np.zeros((len(texts), len(self.microcats)), np.float32)
        neighbours = []
        for start in range(0, len(texts), 64):
            sim = (mat[start : start + 64] @ self.matrix.T).toarray()
            ix = np.argpartition(sim, -min(k, sim.shape[1]), axis=1)[:, -k:]
            for j, ids in enumerate(ix):
                scores = sim[j, ids]
                positive = scores > 0.10
                ids = ids[positive]
                scores = scores[positive]
                # A steep kernel gives exact and near-exact service phrases most
                # influence and limits broad-word false semantic transfers.
                weights = scores**5
                row = (
                    (sparse.csr_matrix(weights[None, :]) @ self.votes[ids])
                    .toarray()
                    .ravel()
                )
                if row.sum():
                    row /= row.sum()
                exact = self.textmap.get(texts[start + j])
                if exact is not None:
                    row = 0.75 * self.votes[exact].toarray().ravel() + 0.25 * row
                votes[start + j] = row
                neighbours.append((ids, scores))
        return dict(texts=texts, microcat=votes, neighbours=neighbours)

    def location_scores(self, query):
        sl = int(query["search_location_id"])
        # Look up ~2,600 locations, then gather to advertisements. Avoid a Python
        # dictionary lookup for each of 200k advertisements and each query.
        compact = np.array(
            [self.loc_probs.get((sl, int(x)), 0.0) for x in self.unique_locs],
            np.float32,
        )
        probs = compact[self.loc_inverse]
        same = (self.item_locs == sl).astype(np.float32)
        center = self.search_centers.get(sl, {})
        lat = float(center.get("item_latitude", 0.0) or 0.0)
        lon = float(center.get("item_longitude", 0.0) or 0.0)
        # Fast equirectangular approximation; sufficient for locality features.
        distance = (
            np.sqrt(
                (self.item_lat - lat) ** 2
                + ((self.item_lon - lon) * np.cos(lat * np.pi / 180)) ** 2
            )
            * 111
        )
        if not lat:
            distance[:] = 20000
        return same, probs, distance

    def click_scores(self, qfeatures, qi):
        result = np.zeros(self.n_items, np.float32)
        for ii, n in self.clicks.get(qfeatures["texts"][qi], {}).items():
            result[ii] = np.log1p(n)
        ids, similarities = qfeatures["neighbours"][qi]
        for t, score in zip(ids, similarities):
            for ii, n in self.clicks.get(self.texts[t], {}).items():
                result[ii] += float(score**5) * np.log1p(n) * 0.3
        return result
