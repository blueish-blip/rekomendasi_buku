"""Memuat artefak hasil pipeline dan menyusun kembali objek model (tanpa melatih ulang).

Dipakai oleh dashboard Streamlit agar aplikasi cepat: model cukup dilatih sekali
lewat `python -m src.pipeline`.
"""
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from . import config as C
from . import models as M
from .data_prep import Index


def _new(cls, **attrs):
    obj = cls.__new__(cls)
    obj.__dict__.update(attrs)
    return obj


@dataclass
class Bundle:
    meta: dict
    catalog: pd.DataFrame
    index: Index
    R: sparse.csr_matrix
    train: pd.DataFrame
    test: pd.DataFrame
    users: pd.DataFrame
    rules: pd.DataFrame
    models: dict = field(default_factory=dict)
    evals: dict = field(default_factory=dict)
    eda: dict = field(default_factory=dict)

    # ------------------------------------------------------------- utilitas
    def history(self, user_id):
        """Riwayat latih user: (indeks buku, rating)."""
        u = self.index.u2i[user_id]
        row = self.R[u]
        return row.indices.copy(), row.data.copy()

    def test_items(self, user_id):
        t = self.test[self.test["user_id"] == user_id]
        return dict(zip(t["book_id"], t["rating"]))

    def recommend(self, model_key, item_idx, ratings, n=C.TOP_N):
        mdl = self.models[model_key]
        scores = mdl.score_history(item_idx, ratings)
        top = M.top_n(scores, exclude=item_idx, n=n)
        out = self.catalog.iloc[top][["book_id", "title", "author", "year", "publisher", "img_m",
                                      "n_rating_train", "mean_rating_train"]].copy()
        out.insert(0, "skor", np.round(np.asarray(scores)[top], 4))
        out["pred_rating_svd"] = np.round(self.predict_rating(item_idx, ratings, top), 2)
        return out.reset_index(drop=True)

    def recommend_user(self, model_key, user_id, n=C.TOP_N):
        """Rekomendasi untuk user yang ada di data latih (memakai parameter hasil latih)."""
        u = self.index.u2i[user_id]
        item_idx, _ = self.history(user_id)
        scores = self.models[model_key].score_users(np.array([u]))[0]
        top = M.top_n(scores, exclude=item_idx, n=n)
        out = self.catalog.iloc[top][["book_id", "title", "author", "year", "publisher", "img_m",
                                      "n_rating_train", "mean_rating_train"]].copy()
        out.insert(0, "skor", np.round(scores[top], 4))
        svd = self.models["svd"]
        out["pred_rating_svd"] = np.round(svd.predict_many(np.full(len(top), u), top), 2)
        return out.reset_index(drop=True)

    def predict_rating(self, item_idx, ratings, target_idx):
        svd = self.models["svd"]
        if len(item_idx) == 0:
            return np.clip(svd.mu + svd.bi[target_idx], 1, 10)
        return svd.predict_history(item_idx, ratings, target_idx)


def load(model_dir=None, processed_dir=None):
    mdir = model_dir or C.MODEL_DIR
    pdir = processed_dir or C.PROCESSED_DIR
    meta = json.loads((mdir / "meta.json").read_text())
    catalog = pd.read_csv(mdir / "catalog.csv.gz", dtype={"book_id": str})
    users_idx = pd.read_csv(mdir / "users_index.csv.gz")["user_id"].to_numpy()
    index = Index(users_idx, catalog["book_id"].to_numpy())
    R = sparse.load_npz(mdir / "R_train.npz").tocsr()
    train = pd.read_csv(pdir / "train.csv.gz", dtype={"book_id": str})
    test = pd.read_csv(pdir / "test.csv.gz", dtype={"book_id": str})
    users = pd.read_csv(pdir / "users_model.csv.gz")
    rules = pd.read_csv(mdir / "rules.csv.gz", dtype={"consequent_id": str, "antecedent_ids": str})

    # --- rekonstruksi model
    C_, m_ = np.load(mdir / "popularity_meta.npy")
    pop = _new(M.PopularityRecommender, C=float(C_), m=float(m_), R=R,
               count=catalog["n_rating_train"].to_numpy(float),
               mean=catalog["mean_rating_train"].to_numpy(float),
               wr=catalog["weighted_rating"].to_numpy(float))
    cb = _new(M.ContentBasedRecommender, X=sparse.load_npz(mdir / "cb_tfidf.npz").tocsr(), R=R)
    S = sparse.load_npz(mdir / "itemcf_sim.npz").tocsr()
    icf = _new(M.ItemCFRecommender, R=R, S=S, ST=S.T.tocsr(),
               user_mean=np.load(mdir / "itemcf_user_mean.npy"))
    z = np.load(mdir / "funksvd.npz")
    svd = _new(M.FunkSVD, mu=float(z["mu"]), bu=z["bu"], bi=z["bi"], P=z["P"], Q=z["Q"],
               k=z["P"].shape[1], reg=meta["svd_params"]["reg"], history=z["history"].tolist(), R=R)
    psvd = _new(M.PureSVDRecommender, V=np.load(mdir / "puresvd_V.npy"), R=R)
    psvd.k = psvd.V.shape[1]
    h = meta["hybrid"]
    hybrid = M.SwitchingHybrid(pop, cb, psvd, h["threshold"], h["alpha_low"], h["alpha_high"], R=R)

    # rule recommender dari tabel aturan yang tersimpan
    rule_items = np.array(sorted(set(rules["consequent_id"]) |
                                 {b for a in rules["antecedent_ids"] for b in a.split("|")}))
    pos = {b: k for k, b in enumerate(rule_items)}
    rr = pd.DataFrame({"antecedent": rules["antecedent_ids"].map(lambda a: tuple(pos[b] for b in a.split("|"))),
                       "consequent": rules["consequent_id"].map(pos), "confidence": rules["confidence"]})
    rule_rec = M.RuleRecommender().fit(rr, rule_items, index, pop)

    b = Bundle(meta=meta, catalog=catalog, index=index, R=R, train=train, test=test, users=users,
               rules=rules)
    b.models = {"pop": pop, "rule_rec": rule_rec, "cb": cb, "icf": icf, "svd": svd, "psvd": psvd,
                "hybrid": hybrid}
    for name in ["rating", "topn", "by_activity"]:
        b.evals[name] = pd.read_csv(mdir / f"eval_{name}.csv")
    for f in mdir.glob("tuning_*.csv"):
        b.evals[f.stem] = pd.read_csv(f)
    for f in pdir.glob("eda_*.csv"):
        b.eda[f.stem.replace("eda_", "")] = pd.read_csv(f)
    return b
