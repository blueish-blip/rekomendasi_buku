"""Tahap 3: model-model sistem rekomendasi.

Semua model rekomendasi punya antarmuka yang sama:

    model.score_history(item_idx, ratings) -> skor untuk SEMUA buku
        dipakai untuk satu user berdasarkan riwayatnya (bisa user lama atau user baru)
    model.score_users(user_idx)            -> matriks skor (banyak user sekaligus)
        dipakai untuk evaluasi; user_idx = nomor baris pada matriks latih

Skor lebih tinggi = lebih direkomendasikan. Buku yang sudah dibaca dibuang
oleh fungsi `top_n` di bawah.
"""
import re

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from . import config as C


def build_matrix(df, index):
    """Matriks rating user x buku (sparse CSR) dari DataFrame (user_id, book_id, rating)."""
    u, i = index.encode(df)
    return sparse.csr_matrix((df["rating"].to_numpy(dtype=np.float64), (u, i)),
                             shape=(len(index.users), len(index.items)))


def top_n(scores, exclude=None, n=C.TOP_N):
    """Ambil n indeks dengan skor tertinggi, setelah membuang indeks di `exclude`."""
    s = np.asarray(scores, dtype=np.float64).copy()
    if exclude is not None and len(exclude):
        s[np.asarray(exclude, dtype=int)] = -np.inf
    n = min(n, int(np.isfinite(s).sum()))
    if n <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-s, n - 1)[:n]
    return idx[np.argsort(-s[idx], kind="stable")]


# ============================================================================
# 0. Baseline: popularitas (weighted rating ala IMDb)
# ============================================================================

class PopularityRecommender:
    """WR_i = v/(v+m) * R_i + m/(v+m) * C

    v = jumlah rating buku i, R_i = rata-rata rating buku i,
    C = rata-rata rating seluruh buku, m = jumlah rating minimum yang 'dipercaya'.
    Buku dengan sedikit rating ditarik ke arah rata-rata global C, sehingga
    buku yang hanya dinilai 10 oleh 1 orang tidak langsung di peringkat atas.
    """

    name = "Popularitas"

    def fit(self, R, m_quantile=C.POPULARITY_M_QUANTILE):
        v = np.asarray((R > 0).sum(axis=0)).ravel().astype(float)
        s = np.asarray(R.sum(axis=0)).ravel()
        mean_i = np.divide(s, v, out=np.zeros_like(s), where=v > 0)
        self.C = s.sum() / v.sum()
        self.m = float(np.quantile(v[v > 0], m_quantile))
        self.count, self.mean, self.R = v, mean_i, R
        self.wr = v / (v + self.m) * mean_i + self.m / (v + self.m) * self.C
        return self

    def score_history(self, item_idx=None, ratings=None):
        return self.wr.copy()

    def score_users(self, user_idx):
        return np.tile(self.wr, (len(user_idx), 1))


# ============================================================================
# 1. Content-based: TF-IDF (judul + penulis + penerbit) + cosine similarity
# ============================================================================

def _token(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def book_text(books):
    """Gabungkan atribut buku menjadi satu 'dokumen'.

    Penulis dan penerbit dijadikan satu token utuh (mis. 'author_jkrowling')
    supaya nama tidak terpecah menjadi kata-kata umum. Token penulis diulang 2x
    agar bobotnya lebih besar daripada penerbit.
    """
    a = "author_" + books["author"].map(_token)
    p = "pub_" + books["publisher"].map(_token)
    return books["title"].fillna("") + " " + a + " " + a + " " + p


class ContentBasedRecommender:
    name = "Content-Based"

    def fit(self, books_in_order, R):
        """books_in_order: DataFrame buku yang urutannya sama dengan kolom matriks R."""
        self.vectorizer = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True,
                                          token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b")
        self.X = normalize(self.vectorizer.fit_transform(book_text(books_in_order)))  # item profile
        self.R = R
        return self

    def _weights(self, ratings):
        """Bobot user profile: rating dikurangi rata-rata user, hanya yang positif (disukai)."""
        r = np.asarray(ratings, dtype=float)
        w = r - r.mean()
        if (w > 0).any():
            return np.clip(w, 0, None)
        return np.ones_like(r)          # semua rating sama -> bobot sama

    def user_profile(self, item_idx, ratings):
        w = self._weights(ratings)
        prof = sparse.csr_matrix(w) @ self.X[np.asarray(item_idx)]
        return normalize(prof)

    def score_history(self, item_idx, ratings):
        if len(item_idx) == 0:
            return np.zeros(self.X.shape[0])
        return np.asarray((self.user_profile(item_idx, ratings) @ self.X.T).todense()).ravel()

    def score_users(self, user_idx):
        rows = self.R[user_idx]
        W = rows.copy().tocsr()
        # bobot per baris: rating - rata2 user, diklip >= 0
        for k in range(W.shape[0]):
            a, b = W.indptr[k], W.indptr[k + 1]
            W.data[a:b] = self._weights(rows.data[a:b])
        P = normalize(W @ self.X)                       # user profile (n_user x vocab)
        return np.asarray((P @ self.X.T).todense())     # cosine (n_user x n_item)

    def similar_items(self, i, n=C.TOP_N):
        s = np.asarray((self.X[i] @ self.X.T).todense()).ravel()
        idx = top_n(s, exclude=[i], n=n)
        return idx, s[idx]


# ============================================================================
# 2. Item-based Collaborative Filtering (memory-based)
# ============================================================================

class ItemCFRecommender:
    """Kemiripan antar buku = adjusted cosine.

    Rating dikurangi dulu rata-rata user (mean-centering), lalu dihitung cosine
    antar kolom (buku). Hanya k tetangga terdekat per buku yang disimpan.
    """

    name = "Item-based CF"

    def fit(self, R, k=C.ITEMCF_NEIGHBORS, shrink=10.0):
        self.R = R.tocsr()
        n_items = R.shape[1]
        cnt = np.diff(self.R.indptr)
        self.user_mean = np.asarray(self.R.sum(axis=1)).ravel() / np.maximum(cnt, 1)
        Rc = self.R.copy()
        Rc.data = Rc.data - np.repeat(self.user_mean, cnt)            # mean-centering
        Xi = Rc.T.tocsr()                                             # buku x user
        norms = np.sqrt(np.asarray(Xi.multiply(Xi).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        B = (self.R > 0).astype(np.float64)
        co = (B.T @ B).toarray()                                      # jml user yang menilai keduanya
        S = (Xi @ Xi.T).toarray() / np.outer(norms, norms)
        S *= co / (co + shrink)                                       # shrinkage: kecilkan sim dari irisan kecil
        np.fill_diagonal(S, 0.0)
        # simpan hanya k tetangga dengan kemiripan positif tertinggi
        rows, cols, vals = [], [], []
        for i in range(n_items):
            s = S[i]
            nb = np.argpartition(-s, k)[:k]
            nb = nb[s[nb] > 0]
            rows.append(np.full(len(nb), i)); cols.append(nb); vals.append(s[nb])
        self.S = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                   shape=(n_items, n_items))
        self.ST = self.S.T.tocsr()
        return self

    # -- prediksi rating (untuk RMSE/MAE) --------------------------------------
    def predict(self, u, i):
        """r_hat(u,i) = mean_u + sum_j sim(i,j)(r_uj - mean_u) / sum_j |sim(i,j)|"""
        row = self.R[u]
        nb = self.S[i]
        common, ia, ib = np.intersect1d(nb.indices, row.indices, return_indices=True)
        if len(common) == 0:
            return self.user_mean[u]
        sims, r = nb.data[ia], row.data[ib]
        return self.user_mean[u] + np.dot(sims, r - self.user_mean[u]) / np.abs(sims).sum()

    # -- skor Top-N ------------------------------------------------------------
    def _liked_weights(self, ratings):
        r = np.asarray(ratings, dtype=float)
        liked = r >= r.mean()
        return liked.astype(float)

    def score_history(self, item_idx, ratings):
        """Skor buku i = jumlah kemiripan i dengan buku-buku yang disukai user."""
        if len(item_idx) == 0:
            return np.zeros(self.S.shape[0])
        w = np.zeros(self.S.shape[0])
        w[np.asarray(item_idx)] = self._liked_weights(ratings)
        return self.ST @ w

    def score_users(self, user_idx):
        rows = self.R[user_idx].tocsr().copy()
        for k in range(rows.shape[0]):
            a, b = rows.indptr[k], rows.indptr[k + 1]
            rows.data[a:b] = self._liked_weights(rows.data[a:b])
        return np.asarray((rows @ self.S).todense())

    def similar_items(self, i, n=C.TOP_N):
        row = self.S[i]
        order = np.argsort(-row.data)[:n]
        return row.indices[order], row.data[order]


# ============================================================================
# 3. Model-based CF: Matrix Factorization (Funk SVD dengan bias)
# ============================================================================

class BiasBaseline:
    """r_hat = mu + b_u + b_i  (diestimasi dengan ALS + regularisasi)."""

    name = "Baseline bias (mu + b_u + b_i)"

    def fit(self, u, i, r, n_users, n_items, reg_u=15.0, reg_i=10.0, n_iter=10):
        self.mu = r.mean()
        bu, bi = np.zeros(n_users), np.zeros(n_items)
        for _ in range(n_iter):
            bi = np.bincount(i, r - self.mu - bu[u], n_items) / (reg_i + np.bincount(i, minlength=n_items))
            bu = np.bincount(u, r - self.mu - bi[i], n_users) / (reg_u + np.bincount(u, minlength=n_users))
        self.bu, self.bi = bu, bi
        return self

    def predict_many(self, u, i):
        return np.clip(self.mu + self.bu[u] + self.bi[i], 1, 10)


class FunkSVD:
    """r_hat(u,i) = mu + b_u + b_i + p_u . q_i

    Parameter dicari dengan Stochastic Gradient Descent yang meminimalkan
        sum (r_ui - r_hat_ui)^2 + reg * (b_u^2 + b_i^2 + ||p_u||^2 + ||q_i||^2)
    HANYA pada rating yang teramati (sel kosong tidak dianggap 0).
    """

    name = "SVD (Matrix Factorization)"

    def __init__(self, n_factors=50, n_epochs=25, lr=0.005, reg=0.05, init_std=0.1,
                 seed=C.RANDOM_STATE, verbose=False):
        self.k, self.n_epochs, self.lr, self.reg = n_factors, n_epochs, lr, reg
        self.init_std, self.seed, self.verbose = init_std, seed, verbose

    def fit(self, u, i, r, n_users, n_items, R=None):
        rng = np.random.default_rng(self.seed)
        self.mu = float(r.mean())
        bu, bi = np.zeros(n_users), np.zeros(n_items)
        P = rng.normal(0, self.init_std, (n_users, self.k))
        Q = rng.normal(0, self.init_std, (n_items, self.k))
        lr, reg, mu = self.lr, self.reg, self.mu
        u_l, i_l, r_l = u.tolist(), i.tolist(), r.tolist()
        order = np.arange(len(r))
        self.history = []
        for ep in range(self.n_epochs):
            rng.shuffle(order)
            sq = 0.0
            for t in order.tolist():
                uu, ii, rr = u_l[t], i_l[t], r_l[t]
                pu, qi = P[uu].copy(), Q[ii].copy()   # salin: update p dan q pakai nilai lama
                err = rr - (mu + bu[uu] + bi[ii] + pu @ qi)
                sq += err * err
                bu[uu] += lr * (err - reg * bu[uu])
                bi[ii] += lr * (err - reg * bi[ii])
                P[uu] = pu + lr * (err * qi - reg * pu)
                Q[ii] = qi + lr * (err * pu - reg * qi)
            self.history.append(np.sqrt(sq / len(r)))
            if self.verbose:
                print(f"epoch {ep + 1:2d}  RMSE latih = {self.history[-1]:.4f}")
        self.bu, self.bi, self.P, self.Q = bu, bi, P, Q
        self.R = R
        return self

    def predict_many(self, u, i):
        pred = self.mu + self.bu[u] + self.bi[i] + np.einsum("ij,ij->i", self.P[u], self.Q[i])
        return np.clip(pred, 1, 10)

    def score_users(self, user_idx):
        return self.mu + self.bu[user_idx][:, None] + self.bi[None, :] + self.P[user_idx] @ self.Q.T

    def fold_in(self, item_idx, ratings, reg_b=5.0, reg_p=1.0):
        """Estimasi b_u dan p_u untuk user BARU (tidak ikut dilatih) dari beberapa rating.

        b_u = rata-rata residual (dengan shrinkage); p_u = ridge regression
        p_u = (Q_I^T Q_I + reg_p * I)^-1 Q_I^T (r - mu - b_i - b_u)
        """
        item_idx = np.asarray(item_idx, dtype=int)
        r = np.asarray(ratings, dtype=float)
        res = r - self.mu - self.bi[item_idx]
        bu = res.sum() / (reg_b + len(r))
        Qi = self.Q[item_idx]
        pu = np.linalg.solve(Qi.T @ Qi + reg_p * np.eye(self.k), Qi.T @ (res - bu))
        return bu, pu

    def predict_history(self, item_idx, ratings, target_idx):
        """Prediksi rating buku target untuk user baru (lewat fold-in)."""
        bu, pu = self.fold_in(item_idx, ratings)
        t = np.asarray(target_idx, dtype=int)
        return np.clip(self.mu + bu + self.bi[t] + self.Q[t] @ pu, 1, 10)

    def score_history(self, item_idx, ratings):
        if len(item_idx) == 0:
            return self.mu + self.bi.copy()
        bu, pu = self.fold_in(item_idx, ratings)
        return self.mu + bu + self.bi + self.Q @ pu


class PureSVDRecommender:
    """Truncated SVD untuk Top-N (Cremonesi, Koren & Turrin, 2010).

    R ~ U S V^T dengan sel kosong diisi 0, lalu skor user u = r_u V V^T.
    Berbeda dengan Funk SVD (yang hanya belajar dari rating teramati dan
    dioptimalkan untuk RMSE), di sini sel kosong DIANGGAP sinyal: buku yang
    tidak dibaca user kemungkinan memang kurang menarik baginya. Karena itu
    PureSVD jauh lebih baik untuk menyusun peringkat Top-N.
    Skor juga bisa dihitung untuk user baru tanpa melatih ulang (cukup r_u).
    """

    name = "PureSVD (Truncated SVD)"

    def __init__(self, n_factors=50):
        self.k = n_factors

    def fit(self, R):
        self.R = R.tocsr().astype(np.float64)
        _, self.sigma, Vt = svds(self.R, k=self.k, random_state=C.RANDOM_STATE)
        self.V = Vt.T
        return self

    def score_history(self, item_idx, ratings):
        r = np.zeros(self.V.shape[0])
        r[np.asarray(item_idx, dtype=int)] = ratings
        return self.V @ (self.V.T @ r)

    def score_users(self, user_idx):
        return (self.R[user_idx] @ self.V) @ self.V.T


# ============================================================================
# 4. Hybrid: weighted + switching
# ============================================================================

def minmax_rows(S):
    """Normalisasi min-max per user (per baris) agar skor dua model setara skalanya."""
    S = np.atleast_2d(np.asarray(S, dtype=np.float64))
    lo, hi = S.min(axis=1, keepdims=True), S.max(axis=1, keepdims=True)
    return (S - lo) / np.where(hi > lo, hi - lo, 1.0)


class WeightedHybrid:
    """skor = alpha * skor_CB + (1 - alpha) * skor_CF   (keduanya dinormalisasi 0-1)."""

    def __init__(self, cb, cf, alpha):
        self.cb, self.cf, self.alpha = cb, cf, alpha
        self.name = f"Weighted (alpha={alpha:.1f})"

    def score_history(self, item_idx, ratings):
        a = minmax_rows(self.cb.score_history(item_idx, ratings))
        b = minmax_rows(self.cf.score_history(item_idx, ratings))
        return (self.alpha * a + (1 - self.alpha) * b).ravel()

    def score_users(self, user_idx):
        return (self.alpha * minmax_rows(self.cb.score_users(user_idx))
                + (1 - self.alpha) * minmax_rows(self.cf.score_users(user_idx)))


class SwitchingHybrid:
    """Hybrid switching berdasarkan banyaknya riwayat rating user (n):

        n = 0              -> Popularitas              (cold-start total)
        1 <= n < threshold -> Weighted dengan alpha_low
        n >= threshold     -> Weighted dengan alpha_high
    alpha = porsi Content-Based. Nilai threshold dan kedua alpha dipilih dari data
    validasi; hasilnya: makin banyak riwayat user, makin besar porsi Collaborative.
    """

    name = "Hybrid Switching"

    def __init__(self, popularity, cb, cf, threshold, alpha_low, alpha_high, R=None):
        self.pop, self.threshold, self.R = popularity, threshold, R
        self.low = WeightedHybrid(cb, cf, alpha_low)
        self.high = WeightedHybrid(cb, cf, alpha_high)

    def choose(self, n_hist):
        """Kembalikan (model yang dipakai, penjelasan dalam bahasa manusia)."""
        if n_hist == 0:
            return self.pop, "Belum ada riwayat -> Popularitas (cold-start)"
        if n_hist < self.threshold:
            m, why = self.low, f"Riwayat {n_hist} buku (< {self.threshold})"
        else:
            m, why = self.high, f"Riwayat {n_hist} buku (>= {self.threshold})"
        a = m.alpha
        return m, f"{why} -> {a:.0%} Content-Based + {1 - a:.0%} Collaborative (PureSVD)"

    def score_history(self, item_idx, ratings):
        return self.choose(len(item_idx))[0].score_history(item_idx, ratings)

    def score_users(self, user_idx):
        user_idx = np.asarray(user_idx)
        n_hist = np.diff(self.R.indptr)[user_idx]
        out = np.empty((len(user_idx), self.R.shape[1]))
        low = n_hist < self.threshold
        if low.any():
            out[low] = self.low.score_users(user_idx[low])
        if (~low).any():
            out[~low] = self.high.score_users(user_idx[~low])
        return out


# ============================================================================
# 5. Rule-based: Association Rules dengan algoritma Apriori
# ============================================================================

def build_transactions(ratings, top_books=C.RULES_TOP_BOOKS, min_items=2):
    """Setiap user = satu 'transaksi' berisi buku yang pernah ia interaksikan.

    Di sini rating 0 (implicit) IKUT dipakai, karena yang dilihat hanya
    'pernah membaca/berinteraksi', bukan nilai ratingnya.
    """
    top = ratings["book_id"].value_counts().head(top_books).index
    t = ratings[ratings["book_id"].isin(top)][["user_id", "book_id"]].drop_duplicates()
    sizes = t.groupby("user_id")["book_id"].transform("size")
    t = t[sizes >= min_items]
    users = np.sort(t["user_id"].unique())
    items = np.sort(t["book_id"].unique())
    ui = pd.Series(np.arange(len(users)), index=users)
    ii = pd.Series(np.arange(len(items)), index=items)
    X = sparse.csr_matrix((np.ones(len(t)), (ui[t["user_id"]].to_numpy(), ii[t["book_id"]].to_numpy())),
                          shape=(len(users), len(items)))
    return X, items


def apriori(X, min_support=C.RULES_MIN_SUPPORT, max_len=C.RULES_MAX_LEN):
    """Apriori: cari itemset yang support-nya >= min_support, naik level demi level.

    Prinsip Apriori: jika sebuah itemset tidak frequent, semua supersetnya pasti
    tidak frequent -> kandidat level k+1 hanya dibentuk dari itemset frequent level k.
    """
    n = X.shape[0]
    Xc = X.tocsc().astype(bool)
    sup1 = np.asarray(X.sum(axis=0)).ravel() / n
    freq = {1: {(i,): sup1[i] for i in np.where(sup1 >= min_support)[0]}}
    cols = {i: Xc[:, [i]] for (i,) in freq[1]}
    k = 1
    while k < max_len and freq[k]:
        prev = sorted(freq[k])
        prev_set = set(prev)
        cand = set()
        for a in range(len(prev)):
            for b in range(a + 1, len(prev)):
                x, y = prev[a], prev[b]
                if x[:-1] != y[:-1]:
                    continue
                c = x + (y[-1],)
                # pruning: semua subset ukuran k harus frequent
                if all(tuple(c[:j] + c[j + 1:]) in prev_set for j in range(len(c))):
                    cand.add(c)
        level = {}
        for c in cand:
            m = cols[c[0]]
            for j in c[1:]:
                m = m.multiply(cols[j])
            s = m.sum() / n
            if s >= min_support:
                level[c] = s
        k += 1
        freq[k] = level
    return {s: v for lvl in freq.values() for s, v in lvl.items()}


def association_rules(freq, min_confidence=C.RULES_MIN_CONFIDENCE):
    """Bentuk aturan {antecedent} -> {consequent} (consequent 1 buku).

    support    = P(A n B)
    confidence = P(B | A) = support(A n B) / support(A)
    lift       = confidence / support(B)     (> 1 berarti asosiasi positif)
    """
    rows = []
    for itemset, sup in freq.items():
        if len(itemset) < 2:
            continue
        for j, cons in enumerate(itemset):
            ante = itemset[:j] + itemset[j + 1:]
            conf = sup / freq[ante]
            if conf >= min_confidence:
                rows.append((ante, cons, sup, freq[ante], freq[(cons,)], conf, conf / freq[(cons,)]))
    return pd.DataFrame(rows, columns=["antecedent", "consequent", "support", "support_A",
                                       "support_B", "confidence", "lift"])


class RuleRecommender:
    """Rekomendasi dari association rules: jika semua buku di antecedent sudah
    dibaca user, buku consequent mendapat skor = confidence aturan (ambil yang
    tertinggi bila ada beberapa aturan). Popularitas dipakai sebagai pemecah seri.
    """

    name = "Rule-Based (Association Rules)"

    def fit(self, rules, rule_items, index, popularity):
        n_items = len(index.items)
        to_model = np.array([index.i2i.get(b, -1) for b in rule_items])
        keep = []
        for k, (ante, cons) in enumerate(zip(rules["antecedent"], rules["consequent"])):
            if to_model[cons] >= 0 and all(to_model[a] >= 0 for a in ante):
                keep.append(k)
        r = rules.iloc[keep].reset_index(drop=True)
        rows = np.repeat(np.arange(len(r)), r["antecedent"].map(len).to_numpy())
        cols = np.concatenate([to_model[list(a)] for a in r["antecedent"]]) if len(r) else np.array([], int)
        self.A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(r), n_items))
        self.ante_len = r["antecedent"].map(len).to_numpy()
        self.cons = to_model[r["consequent"].to_numpy()]
        self.conf = r["confidence"].to_numpy()
        self.tiebreak = 1e-6 * popularity.count / popularity.count.max()
        self.R_train = popularity.R
        return self

    def _score_fired(self, fired):
        out = np.zeros((fired.shape[0], self.A.shape[1]))
        for c in np.unique(self.cons):
            m = self.cons == c
            out[:, c] = (fired[:, m] * self.conf[m]).max(axis=1)
        return out + self.tiebreak

    def score_history(self, item_idx, ratings=None):
        h = np.zeros(self.A.shape[1])
        h[np.asarray(item_idx, dtype=int)] = 1
        fired = (self.A @ h >= self.ante_len)[None, :]
        return self._score_fired(fired).ravel()

    def score_users(self, user_idx):
        H = (self.R_train[user_idx] > 0).astype(np.float64)
        fired = np.asarray((H @ self.A.T).todense()) >= self.ante_len[None, :]
        return self._score_fired(fired)
