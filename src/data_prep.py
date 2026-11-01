"""Tahap 1-2: membaca, membersihkan, memfilter, dan membagi data Book-Crossing."""
import html

import numpy as np
import pandas as pd

from . import config as C

# ============================================================================
# 1. Membaca data mentah
# ============================================================================

def read_raw(name):
    """Baca salah satu CSV mentah ('books', 'users', atau 'ratings').

    File Book-Crossing memakai pemisah ';', encoding latin-1, dan tanda kutip
    yang di-escape dengan backslash (\\"). Tanpa escapechar, 3 baris buku
    kolomnya bergeser dan 19 baris hilang.
    """
    return pd.read_csv(
        C.RAW_FILES[name], sep=";", encoding="latin-1", quotechar='"',
        escapechar="\\", dtype=str, keep_default_na=False, na_values=["NULL", ""],
    )


def normalize_isbn(s):
    """Seragamkan ISBN: huruf besar dan hanya karakter 0-9 / X."""
    return s.astype(str).str.upper().str.replace(r"[^0-9X]", "", regex=True)


# ============================================================================
# 2. Membersihkan tiap tabel
# ============================================================================

def _clean_text(s):
    return (s.fillna("").astype(str)
             .map(html.unescape)                 # &amp; -> &
             .str.replace(r'^"+|"+$', "", regex=True)
             .str.replace(r"\s+", " ", regex=True)
             .str.strip())


def clean_books(books):
    b = books.rename(columns={
        "ISBN": "isbn", "Book-Title": "title", "Book-Author": "author",
        "Year-Of-Publication": "year", "Publisher": "publisher",
        "Image-URL-S": "img_s", "Image-URL-M": "img_m", "Image-URL-L": "img_l",
    }).copy()
    b["isbn"] = normalize_isbn(b["isbn"])
    for col in ["title", "author", "publisher"]:
        b[col] = _clean_text(b[col])
    b.loc[b["author"] == "", "author"] = "Unknown"
    b.loc[b["publisher"] == "", "publisher"] = "Unknown"
    b["year"] = pd.to_numeric(b["year"], errors="coerce")
    b.loc[(b["year"] < C.YEAR_MIN) | (b["year"] > C.YEAR_MAX), "year"] = np.nan
    b["year"] = b["year"].astype("Int64")
    b = b.drop_duplicates("isbn", keep="first")
    return b[["isbn", "title", "author", "year", "publisher", "img_m", "img_l"]].reset_index(drop=True)


def clean_users(users):
    u = users.rename(columns={"User-ID": "user_id", "Location": "location", "Age": "age"}).copy()
    u["user_id"] = u["user_id"].astype(int)
    u["location"] = u["location"].fillna("").str.strip().str.lower()
    u["age"] = pd.to_numeric(u["age"], errors="coerce")
    u.loc[(u["age"] < C.AGE_MIN) | (u["age"] > C.AGE_MAX), "age"] = np.nan
    u["age"] = u["age"].astype("Int64")
    # negara = bagian terakhir dari "kota, provinsi, negara"
    country = u["location"].str.split(",").str[-1].str.strip().str.replace(r'[."]', "", regex=True)
    country = country.where(~country.isin(["", "n/a", "na", "-", "?", "none", "unknown"]))
    u["country"] = country
    return u[["user_id", "location", "country", "age"]]


def clean_ratings(ratings, books):
    r = ratings.rename(columns={"User-ID": "user_id", "ISBN": "isbn", "Book-Rating": "rating"}).copy()
    r["user_id"] = r["user_id"].astype(int)
    r["isbn"] = normalize_isbn(r["isbn"])
    r["rating"] = r["rating"].astype(int)
    n0 = len(r)
    r = r[r["isbn"].isin(books["isbn"])]                 # buang ISBN yang tak ada di katalog
    r = r.drop_duplicates(["user_id", "isbn"], keep="last")
    r["type"] = np.where(r["rating"] > 0, "explicit", "implicit")
    r.attrs["dropped_unknown_isbn"] = n0 - len(r)
    return r.reset_index(drop=True)


# ============================================================================
# 2b. Menggabungkan edisi: satu judul bisa punya banyak ISBN
# ============================================================================

_GENERIC_SUBTITLE = (r"\s*:\s*(a|an)?\s*(novel|memoir|novella|thriller|mystery|romance|"
                     r"novel of suspense|love story|story|stories)\s*$")


def work_key(title, author):
    """Kunci 'karya': judul + penulis yang dinormalisasi.

    Contoh: 'The Lovely Bones: A Novel' (Alice Sebold) dan 'The Lovely Bones'
    (ALICE SEBOLD) -> 'lovely bones|alicesebold'. Isi tanda kurung, subjudul
    generik (': A Novel'), dan kata sandang awal dibuang.
    """
    t = title.str.lower()
    t = t.str.replace(r"\(.*?\)|\[.*?\]", "", regex=True)
    t = t.str.replace(_GENERIC_SUBTITLE, "", regex=True).str.strip()
    t = t.str.replace(r"^(the|a|an)\s+", "", regex=True)
    t = t.str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    a = author.str.lower().str.replace(r"[^a-z]", "", regex=True)
    return t + "|" + a


def merge_editions(books, ratings):
    """Ganti ISBN dengan `book_id` = ISBN edisi yang paling banyak dinilai pada karya tsb.

    - books  -> satu baris per book_id (metadata dari edisi representatif + jumlah edisi)
    - ratings-> isbn diganti book_id; jika user menilai >1 edisi, ambil rata-rata
               rating eksplisitnya (rating 0 hanya dipakai jika tak ada yang eksplisit).
    """
    b = books.copy()
    b["work"] = work_key(b["title"], b["author"])
    n_rating = ratings["isbn"].value_counts()
    b["n_rating"] = b["isbn"].map(n_rating).fillna(0).astype(int)
    b = b.sort_values(["work", "n_rating"], ascending=[True, False])
    rep = b.groupby("work")["isbn"].first()
    b["book_id"] = b["work"].map(rep)
    n_ed = b.groupby("book_id")["isbn"].size().rename("n_editions")
    works = (b[b["isbn"] == b["book_id"]]
             .drop(columns=["isbn", "n_rating"])
             .merge(n_ed, left_on="book_id", right_index=True)
             [["book_id", "title", "author", "year", "publisher", "n_editions", "img_m", "img_l"]]
             .reset_index(drop=True))

    r = ratings.merge(b[["isbn", "book_id"]], on="isbn", how="inner")
    r["explicit"] = r["rating"].where(r["rating"] > 0)
    agg = r.groupby(["user_id", "book_id"]).agg(rating=("explicit", "mean"), n=("rating", "size")).reset_index()
    agg["rating"] = agg["rating"].fillna(0)
    agg["type"] = np.where(agg["rating"] > 0, "explicit", "implicit")
    return works, agg[["user_id", "book_id", "rating", "type"]]


def load_clean():
    """Jalankan semua langkah cleaning; kembalikan (books, users, ratings, ringkasan).

    books   : satu baris per judul (book_id), edisi-edisi sudah digabung
    ratings : user_id, book_id, rating (0 = implicit, 1-10 = explicit; bisa desimal
              bila user menilai >1 edisi), type
    """
    books_raw, users_raw, ratings_raw = read_raw("books"), read_raw("users"), read_raw("ratings")
    books_isbn = clean_books(books_raw)
    users = clean_users(users_raw)
    ratings_isbn = clean_ratings(ratings_raw, books_isbn)
    books, ratings = merge_editions(books_isbn, ratings_isbn)
    summary = {
        "books_raw": len(books_raw), "users_raw": len(users_raw), "ratings_raw": len(ratings_raw),
        "isbn_clean": len(books_isbn), "books_clean": len(books), "users_clean": len(users),
        "ratings_isbn_known": len(ratings_isbn), "ratings_clean": len(ratings),
        "ratings_dropped_unknown_isbn": len(ratings_raw) - len(ratings_isbn),
        "explicit": int((ratings["rating"] > 0).sum()), "implicit": int((ratings["rating"] == 0).sum()),
        "age_missing_pct": float(users["age"].isna().mean() * 100),
        "year_missing": int(books["year"].isna().sum()),
    }
    return books, users, ratings, summary


# ============================================================================
# 3. Data untuk pemodelan: filter + split
# ============================================================================

def build_model_data(ratings, min_book=C.MIN_RATINGS_PER_BOOK, min_user=C.MIN_RATINGS_PER_USER):
    """Ambil rating eksplisit, lalu filter buku dan user yang terlalu jarang.

    Urutan: (1) buku dengan >= min_book rating, (2) user dengan >= min_user rating
    pada buku-buku tersebut. Rating 0 (implicit) TIDAK dipakai di sini karena
    0 berarti 'tidak menilai', bukan 'tidak suka'.
    """
    ex = ratings[ratings["rating"] > 0][["user_id", "book_id", "rating"]]
    bc = ex["book_id"].value_counts()
    ex = ex[ex["book_id"].isin(bc[bc >= min_book].index)]
    uc = ex["user_id"].value_counts()
    ex = ex[ex["user_id"].isin(uc[uc >= min_user].index)]
    return ex.reset_index(drop=True)


def train_test_split_per_user(df, test_size=C.TEST_SIZE, seed=C.RANDOM_STATE):
    """Split per user: tiap user menyisihkan ~20% ratingnya (minimal 1) sebagai data uji.

    Dengan begitu setiap user di data uji juga punya riwayat di data latih,
    sehingga rekomendasi untuk user tersebut bisa dievaluasi.
    """
    d = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    d["_rank"] = d.groupby("user_id").cumcount()
    n_user = d.groupby("user_id")["book_id"].transform("size")
    n_test = np.maximum(1, np.floor(n_user * test_size)).astype(int)
    is_test = d["_rank"] < n_test
    train = d[~is_test].drop(columns="_rank").reset_index(drop=True)
    test = d[is_test].drop(columns="_rank").reset_index(drop=True)
    # buang baris uji yang bukunya tidak muncul di data latih (model tak mengenalnya)
    test = test[test["book_id"].isin(train["book_id"])].reset_index(drop=True)
    return train, test


class Index:
    """Pemetaan user_id / isbn <-> nomor baris/kolom matriks."""

    def __init__(self, users, items):
        self.users = np.asarray(users)
        self.items = np.asarray(items)
        self.u2i = {u: i for i, u in enumerate(self.users)}
        self.i2i = {it: i for i, it in enumerate(self.items)}

    @classmethod
    def from_frame(cls, df):
        return cls(np.sort(df["user_id"].unique()), np.sort(df["book_id"].unique()))

    def encode(self, df):
        return df["user_id"].map(self.u2i).to_numpy(), df["book_id"].map(self.i2i).to_numpy()
