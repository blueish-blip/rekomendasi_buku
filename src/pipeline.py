"""Pipeline lengkap: cleaning -> filter/split -> tuning -> fit -> evaluasi -> simpan artefak.

Jalankan dari folder proyek:
    python -m src.pipeline            (tuning + fit + evaluasi + simpan)
    python -m src.pipeline --fast     (lewati tuning SVD, pakai parameter di config.py)

Hasilnya (folder models/ dan data/processed/) dipakai oleh dashboard Streamlit.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse

from . import config as C
from . import data_prep as D
from . import evaluation as E
from . import models as M


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# Persiapan data
# ============================================================================

def prepare():
    books, users, ratings, summary = D.load_clean()
    model_df = D.build_model_data(ratings)
    train, test = D.train_test_split_per_user(model_df)
    index = D.Index.from_frame(train)
    test = test[test["user_id"].isin(index.users)].reset_index(drop=True)
    summary.update({
        "model_ratings": len(model_df), "model_users": int(model_df["user_id"].nunique()),
        "model_books": int(model_df["book_id"].nunique()), "train_ratings": len(train),
        "test_ratings": len(test),
        "model_sparsity_pct": float(100 * (1 - len(model_df) / (model_df["user_id"].nunique()
                                                                 * model_df["book_id"].nunique()))),
    })
    return dict(books=books, users=users, ratings=ratings, summary=summary,
                model_df=model_df, train=train, test=test, index=index)


def books_in_order(books, index):
    return books.set_index("book_id").loc[index.items].reset_index()


def encode(df, index):
    u, i = index.encode(df)
    return u, i, df["rating"].to_numpy(dtype=np.float64)


# ============================================================================
# Tuning (memakai data VALIDASI yang dipisah dari data latih)
# ============================================================================

def validation_split(train):
    tr2, va = D.train_test_split_per_user(train, seed=7)
    idx2 = D.Index.from_frame(tr2)
    va = va[va["user_id"].isin(idx2.users) & va["book_id"].isin(idx2.items)].reset_index(drop=True)
    return tr2, va, idx2


def tune_svd(tr2, va, idx2, grid=None):
    grid = grid or [dict(n_factors=k, reg=reg) for k in (10, 20, 50) for reg in (0.05, 0.15)]
    u, i, r = encode(tr2, idx2)
    vu, vi, vr = encode(va, idx2)
    rows = []
    for g in grid:
        p = {**C.SVD_PARAMS, **g}
        m = M.FunkSVD(**p).fit(u, i, r, len(idx2.users), len(idx2.items))
        pred = m.predict_many(vu, vi)
        rows.append({**{k: p[k] for k in ("n_factors", "reg", "lr", "n_epochs")},
                     "RMSE_validasi": E.rmse(vr, pred), "MAE_validasi": E.mae(vr, pred)})
        log(f"  SVD {g}: RMSE val = {rows[-1]['RMSE_validasi']:.4f}")
    return pd.DataFrame(rows).sort_values("RMSE_validasi").reset_index(drop=True)


def tune_hybrid(tr2, va, idx2, books, alphas=(1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0),
                thresholds=(3, 5, 10, 20), k_svd=50):
    """Pilih alpha_low, alpha_high, dan threshold yang memaksimalkan Precision@10 validasi."""
    R2 = M.build_matrix(tr2, idx2)
    cb = M.ContentBasedRecommender().fit(books_in_order(books, idx2), R2)
    cf = M.PureSVDRecommender(k_svd).fit(R2)
    vu, vi, vr = encode(va, idx2)
    rel = E.relevant_sets(vu, vi, vr)
    per_alpha, act = {}, []
    for a in alphas:
        pu, _ = E.evaluate_topn(M.WeightedHybrid(cb, cf, a), R2, rel)
        per_alpha[a] = pu.set_index("user")
        b = E.by_activity(pu, f"alpha={a:.1f}")
        b["alpha"] = a
        act.append(b)
    act = pd.concat(act, ignore_index=True)
    n_train = per_alpha[alphas[0]]["n_train"]
    best, grid_rows = None, []
    for t in thresholds:
        low = n_train < t
        for al in alphas:
            for ah in alphas:
                prec = np.where(low, per_alpha[al]["precision"], per_alpha[ah]["precision"]).mean()
                grid_rows.append({"threshold": t, "alpha_low": al, "alpha_high": ah, "precision@10": prec})
                if best is None or prec > best["precision@10"] + 1e-12:
                    best = grid_rows[-1]
    grid = pd.DataFrame(grid_rows).sort_values("precision@10", ascending=False).reset_index(drop=True)
    return best, grid, act


# ============================================================================
# Fit semua model pada data latih
# ============================================================================

def fit_all(data, hybrid_params, svd_params=None):
    books, train, test, index, ratings = data["books"], data["train"], data["test"], data["index"], data["ratings"]
    R = M.build_matrix(train, index)
    u, i, r = encode(train, index)
    out = {"R": R}
    log("fit Popularitas"); out["pop"] = M.PopularityRecommender().fit(R)
    log("fit Content-Based"); out["cb"] = M.ContentBasedRecommender().fit(books_in_order(books, index), R)
    log("fit Item-based CF"); out["icf"] = M.ItemCFRecommender().fit(R)
    log("fit Bias baseline"); out["bias"] = M.BiasBaseline().fit(u, i, r, len(index.users), len(index.items))
    log("fit Funk SVD (+-30 detik)")
    out["svd"] = M.FunkSVD(**(svd_params or C.SVD_PARAMS)).fit(u, i, r, len(index.users), len(index.items), R=R)
    log("fit PureSVD"); out["psvd"] = M.PureSVDRecommender(hybrid_params["k_svd"]).fit(R)
    out["hybrid"] = M.SwitchingHybrid(out["pop"], out["cb"], out["psvd"], hybrid_params["threshold"],
                                      hybrid_params["alpha_low"], hybrid_params["alpha_high"], R=R)
    log("Apriori / association rules")
    # transaksi tanpa pasangan (user, buku) data uji -> tidak bocor ke evaluasi
    key = ratings["user_id"].astype(str) + "|" + ratings["book_id"]
    test_key = set(test["user_id"].astype(str) + "|" + test["book_id"])
    X, rule_items = M.build_transactions(ratings[~key.isin(test_key)])
    freq = M.apriori(X)
    rules = M.association_rules(freq)
    out["rules"], out["rule_items"], out["n_transactions"] = rules, rule_items, X.shape[0]
    out["rule_rec"] = M.RuleRecommender().fit(rules, rule_items, index, out["pop"])
    return out


# ============================================================================
# Evaluasi pada data uji
# ============================================================================

def evaluate_all(fitted, data):
    index, test, train = data["index"], data["test"], data["train"]
    R = fitted["R"]
    tu, ti, tr = encode(test, index)
    rating_rows = [
        ("Rata-rata global", np.full(len(tr), train["rating"].mean())),
        ("Baseline bias (mu + b_u + b_i)", fitted["bias"].predict_many(tu, ti)),
        ("Item-based CF", np.clip([fitted["icf"].predict(a, b) for a, b in zip(tu, ti)], 1, 10)),
        ("SVD (Matrix Factorization)", fitted["svd"].predict_many(tu, ti)),
    ]
    rating_metrics = pd.DataFrame([{"model": n, "RMSE": E.rmse(tr, p), "MAE": E.mae(tr, p)}
                                   for n, p in rating_rows])
    rel = E.relevant_sets(tu, ti, tr)
    topn, act, per_user_all = [], [], {}
    for key in ["pop", "rule_rec", "cb", "icf", "svd", "psvd", "hybrid"]:
        mdl = fitted[key]
        pu, cov = E.evaluate_topn(mdl, R, rel)
        topn.append(E.summarize_topn(pu, cov, mdl.name))
        act.append(E.by_activity(pu, mdl.name))
        per_user_all[key] = pu
        log(f"  Top-N {mdl.name}: P@10 = {topn[-1]['Precision@10']:.4f}")
    return rating_metrics, pd.DataFrame(topn), pd.concat(act, ignore_index=True)


# ============================================================================
# Ringkasan EDA (disimpan agar dashboard ringan)
# ============================================================================

def eda_tables(data):
    books, users, ratings = data["books"], data["users"], data["ratings"]
    t = {}
    t["rating_dist"] = ratings.assign(r=ratings["rating"].round().astype(int))["r"].value_counts().sort_index() \
        .rename_axis("rating").reset_index(name="jumlah")
    ex = ratings[ratings["rating"] > 0]
    agg = ratings.groupby("book_id").agg(n_interaksi=("rating", "size"))
    agg = agg.join(ex.groupby("book_id").agg(n_rating=("rating", "size"), rata2=("rating", "mean")))
    agg = agg.join(books.set_index("book_id")[["title", "author", "year", "publisher"]])
    t["top_books"] = agg.sort_values("n_interaksi", ascending=False).head(50).reset_index()
    t["top_rated"] = agg[agg["n_rating"] >= 50].sort_values("rata2", ascending=False).head(30).reset_index()
    t["top_authors"] = ratings.merge(books[["book_id", "author"]], on="book_id") \
        .groupby("author").size().sort_values(ascending=False).head(20).rename("n_interaksi").reset_index()
    a = users["age"].dropna().astype(int)
    t["age_dist"] = a.value_counts().sort_index().rename_axis("usia").reset_index(name="jumlah")
    active = ratings["user_id"].unique()
    t["countries"] = users[users["user_id"].isin(active)]["country"].value_counts().head(15) \
        .rename_axis("negara").reset_index(name="jumlah_user")
    uc = ex.groupby("user_id").size()
    bc = ex.groupby("book_id").size()
    bins = [1, 2, 3, 5, 10, 20, 50, 100, np.inf]
    labels = ["1", "2", "3-4", "5-9", "10-19", "20-49", "50-99", "100+"]
    t["user_activity"] = pd.cut(uc, bins, right=False, labels=labels).value_counts().sort_index() \
        .rename_axis("jumlah_rating").reset_index(name="jumlah_user")
    t["book_popularity"] = pd.cut(bc, bins, right=False, labels=labels).value_counts().sort_index() \
        .rename_axis("jumlah_rating").reset_index(name="jumlah_buku")
    yr = books["year"].dropna().astype(int)
    t["year_dist"] = yr[yr >= 1950].value_counts().sort_index().rename_axis("tahun").reset_index(name="jumlah_buku")
    return t


# ============================================================================
# Simpan artefak untuk dashboard
# ============================================================================

def save_artifacts(data, fitted, hybrid_params, evals, tuning, eda):
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    index, books = data["index"], data["books"]
    mdir, pdir = C.MODEL_DIR, C.PROCESSED_DIR

    # --- katalog buku model (urutan = kolom matriks) + statistik popularitas
    pop = fitted["pop"]
    cat = books_in_order(books, index)
    cat["n_rating_train"] = pop.count.astype(int)
    cat["mean_rating_train"] = np.round(pop.mean, 3)
    cat["weighted_rating"] = np.round(pop.wr, 4)
    cat.to_csv(mdir / "catalog.csv.gz", index=False)
    pd.DataFrame({"user_id": index.users}).to_csv(mdir / "users_index.csv.gz", index=False)
    data["train"].to_csv(pdir / "train.csv.gz", index=False)
    data["test"].to_csv(pdir / "test.csv.gz", index=False)
    u = data["users"]
    u[u["user_id"].isin(index.users)].to_csv(pdir / "users_model.csv.gz", index=False)

    # --- parameter model
    sparse.save_npz(mdir / "R_train.npz", fitted["R"])
    sparse.save_npz(mdir / "cb_tfidf.npz", fitted["cb"].X)
    sparse.save_npz(mdir / "itemcf_sim.npz", fitted["icf"].S)
    np.save(mdir / "itemcf_user_mean.npy", fitted["icf"].user_mean)
    s = fitted["svd"]
    np.savez(mdir / "funksvd.npz", mu=s.mu, bu=s.bu, bi=s.bi, P=s.P, Q=s.Q, history=np.array(s.history))
    np.save(mdir / "puresvd_V.npy", fitted["psvd"].V)
    np.save(mdir / "popularity_meta.npy", np.array([pop.C, pop.m]))

    # --- association rules (+ judul agar mudah dibaca)
    rules, items = fitted["rules"].copy(), fitted["rule_items"]
    title = books.set_index("book_id")["title"]
    rules["antecedent_ids"] = rules["antecedent"].map(lambda a: "|".join(items[j] for j in a))
    rules["consequent_id"] = rules["consequent"].map(lambda c: items[c])
    rules["jika_membaca"] = rules["antecedent"].map(lambda a: " + ".join(title[items[j]] for j in a))
    rules["maka_rekomendasikan"] = rules["consequent"].map(lambda c: title[items[c]])
    rules["panjang"] = rules["antecedent"].map(len) + 1
    rules.drop(columns=["antecedent", "consequent"]).sort_values("lift", ascending=False) \
        .to_csv(mdir / "rules.csv.gz", index=False)

    # --- hasil evaluasi & tuning
    rating_metrics, topn, act = evals
    rating_metrics.to_csv(mdir / "eval_rating.csv", index=False)
    topn.to_csv(mdir / "eval_topn.csv", index=False)
    act.to_csv(mdir / "eval_by_activity.csv", index=False)
    for name, df in tuning.items():
        df.to_csv(mdir / f"tuning_{name}.csv", index=False)
    for name, df in eda.items():
        df.to_csv(pdir / f"eda_{name}.csv", index=False)

    meta = {"summary": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v))
                        for k, v in data["summary"].items()},
            "hybrid": hybrid_params, "svd_params": C.SVD_PARAMS, "top_n": C.TOP_N,
            "relevant_threshold": C.RELEVANT_THRESHOLD, "n_transactions": int(fitted["n_transactions"]),
            "rules_params": {"top_books": C.RULES_TOP_BOOKS, "min_support": C.RULES_MIN_SUPPORT,
                             "min_confidence": C.RULES_MIN_CONFIDENCE, "max_len": C.RULES_MAX_LEN},
            "filter": {"min_book": C.MIN_RATINGS_PER_BOOK, "min_user": C.MIN_RATINGS_PER_USER,
                       "test_size": C.TEST_SIZE},
            "created": time.strftime("%Y-%m-%d %H:%M")}
    (mdir / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"artefak tersimpan di {mdir} dan {pdir}")


# ============================================================================

def run(fast=False):
    log("1. cleaning + filter + split")
    data = prepare()
    log(f"   {data['summary']['model_ratings']:,} rating | {data['summary']['model_users']:,} user | "
        f"{data['summary']['model_books']:,} buku")
    tr2, va, idx2 = validation_split(data["train"])
    tuning = {}
    svd_params = dict(C.SVD_PARAMS)
    if not fast:
        log("2a. tuning Funk SVD (validasi)")
        tuning["svd"] = tune_svd(tr2, va, idx2)
        best = tuning["svd"].iloc[0]
        svd_params.update(n_factors=int(best["n_factors"]), reg=float(best["reg"]))
    log("2b. tuning hybrid switching (validasi)")
    best, grid, act = tune_hybrid(tr2, va, idx2, data["books"])
    hybrid_params = {"threshold": int(best["threshold"]), "alpha_low": float(best["alpha_low"]),
                     "alpha_high": float(best["alpha_high"]), "k_svd": 50,
                     "precision_validasi": float(best["precision@10"])}
    tuning["hybrid_grid"], tuning["hybrid_alpha_activity"] = grid, act
    log(f"   terpilih: {hybrid_params}")
    log("3. fit semua model")
    fitted = fit_all(data, hybrid_params, svd_params)
    log("4. evaluasi pada data uji")
    evals = evaluate_all(fitted, data)
    print(evals[0].round(4).to_string(index=False))
    print(evals[1].round(4).to_string(index=False))
    log("5. simpan artefak")
    C.SVD_PARAMS.update(svd_params)
    save_artifacts(data, fitted, hybrid_params, evals, tuning, eda_tables(data))
    return data, fitted, evals


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="lewati tuning Funk SVD")
    run(fast=ap.parse_args().fast)
