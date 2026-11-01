"""Konfigurasi proyek: lokasi folder dan parameter utama.

Semua angka penting (ambang filter, ukuran Top-N, hyperparameter) dikumpulkan
di sini supaya mudah diubah dan konsisten antara notebook, pipeline, dan dashboard.
"""
from pathlib import Path

# ---------------------------------------------------------------- lokasi folder
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

RAW_FILES = {
    "books": RAW_DIR / "BX-Books.csv",
    "users": RAW_DIR / "BX-Users.csv",
    "ratings": RAW_DIR / "BX-Book-Ratings.csv",
}

# ---------------------------------------------------------------- cleaning
AGE_MIN, AGE_MAX = 5, 100          # usia di luar rentang ini dianggap tidak valid
YEAR_MIN, YEAR_MAX = 1800, 2006    # data dikumpulkan tahun 2004 (Ziegler dkk., 2005)

# ---------------------------------------------------------------- filter data model
MIN_RATINGS_PER_BOOK = 10   # buku minimal punya 10 rating eksplisit
MIN_RATINGS_PER_USER = 3    # user minimal punya 3 rating eksplisit (agar bisa di-split)
TEST_SIZE = 0.2             # 20% rating tiap user disisihkan untuk data uji
RANDOM_STATE = 42

# ---------------------------------------------------------------- rekomendasi & evaluasi
TOP_N = 10
RELEVANT_THRESHOLD = 8      # rating >= 8 dianggap "disukai" (relevan) saat evaluasi Top-N
POPULARITY_M_QUANTILE = 0.5 # m pada weighted rating = median jumlah rating per buku

# ---------------------------------------------------------------- model
ITEMCF_NEIGHBORS = 50       # simpan 50 tetangga terdekat per buku

SVD_PARAMS = dict(n_factors=20, n_epochs=30, lr=0.005, reg=0.15, init_std=0.1)  # hasil tuning validasi

# ---------------------------------------------------------------- association rules
RULES_TOP_BOOKS = 300       # hanya 300 buku dengan interaksi terbanyak
RULES_MIN_SUPPORT = 0.005
RULES_MIN_CONFIDENCE = 0.2
RULES_MAX_LEN = 3
