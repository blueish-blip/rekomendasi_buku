"""Tahap 4: evaluasi model.

- Prediksi rating : RMSE dan MAE pada data uji.
- Top-N           : Precision@K, Recall@K, Hit rate, dan Coverage.
"""
import numpy as np
import pandas as pd

from . import config as C


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def relevant_sets(test_u, test_i, test_r, threshold=C.RELEVANT_THRESHOLD):
    """Kumpulan buku relevan (rating uji >= threshold) untuk tiap user."""
    m = np.asarray(test_r) >= threshold
    df = pd.DataFrame({"u": np.asarray(test_u)[m], "i": np.asarray(test_i)[m]})
    return df.groupby("u")["i"].apply(set).to_dict()


def evaluate_topn(model, R_train, rel, k=C.TOP_N, batch=1000):
    """Hitung Precision@K dan Recall@K untuk setiap user yang punya buku relevan di data uji.

    Kandidat = semua buku yang BELUM ada di riwayat latih user tersebut.
    """
    users = np.array(sorted(rel))
    n_hist = np.diff(R_train.indptr)
    out, recommended = [], set()
    for s in range(0, len(users), batch):
        ub = users[s:s + batch]
        S = np.asarray(model.score_users(ub), dtype=np.float64)
        sub = R_train[ub]
        rr = np.repeat(np.arange(len(ub)), np.diff(sub.indptr))
        S[rr, sub.indices] = -np.inf                                   # buang buku yang sudah dibaca
        top = np.argpartition(-S, k - 1, axis=1)[:, :k]
        for row, u in enumerate(ub):
            items = top[row]
            recommended.update(items.tolist())
            hits = len(rel[u].intersection(items.tolist()))
            out.append((u, n_hist[u], hits, len(rel[u])))
    df = pd.DataFrame(out, columns=["user", "n_train", "hits", "n_relevant"])
    df["precision"] = df["hits"] / k
    df["recall"] = df["hits"] / df["n_relevant"]
    df["hit"] = (df["hits"] > 0).astype(int)
    coverage = len(recommended) / R_train.shape[1]
    return df, coverage


def summarize_topn(per_user, coverage, name, k=C.TOP_N):
    return {"model": name, f"Precision@{k}": per_user["precision"].mean(),
            f"Recall@{k}": per_user["recall"].mean(), f"HitRate@{k}": per_user["hit"].mean(),
            "Coverage": coverage, "n_user": len(per_user)}


ACTIVITY_BINS = [0, 3, 5, 10, 20, np.inf]
ACTIVITY_LABELS = ["2", "3-4", "5-9", "10-19", "20+"]


def by_activity(per_user, name):
    """Rata-rata Precision/Recall per kelompok jumlah rating latih user."""
    g = pd.cut(per_user["n_train"], ACTIVITY_BINS, right=False, labels=ACTIVITY_LABELS)
    res = per_user.groupby(g, observed=True).agg(precision=("precision", "mean"),
                                                  recall=("recall", "mean"),
                                                  hit=("hit", "mean"), n_user=("user", "size"))
    res["model"] = name
    return res.reset_index().rename(columns={"n_train": "aktivitas"})
