# Sistem Rekomendasi Buku — Hybrid Switching (Book-Crossing)

Proyek Analisis Data Lanjut: membangun sistem rekomendasi buku dengan Python, lalu
menyajikannya dalam dashboard **Streamlit**.

**Model utama:** *Hybrid Switching* yang menggabungkan **Content-Based** (TF-IDF judul,
penulis, penerbit) dan **Collaborative Filtering** (PureSVD). Bobotnya berpindah sesuai
banyaknya riwayat user. Model pembanding: Popularitas, Rule-Based (Apriori),
Item-based CF, dan Funk SVD.

---

## 1. Struktur folder

```
rekomendasi-buku/
├── data/
│   ├── raw/              ← taruh 3 CSV Book-Crossing di sini
│   └── processed/        ← hasil cleaning, data latih/uji, ringkasan EDA
├── models/               ← model terlatih + hasil evaluasi (dipakai dashboard)
├── notebooks/
│   ├── 01_Data_Cleaning.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Pemodelan.ipynb
│   └── 04_Evaluasi_dan_Hybrid.ipynb
├── src/
│   ├── config.py         ← semua parameter (ambang filter, Top-N, hyperparameter)
│   ├── data_prep.py      ← membaca, membersihkan, menggabungkan edisi, filter, split
│   ├── models.py         ← semua model rekomendasi
│   ├── evaluation.py     ← RMSE, MAE, Precision/Recall@K, coverage
│   ├── pipeline.py       ← menjalankan semuanya dari awal sampai simpan artefak
│   └── artifacts.py      ← memuat model untuk dashboard
├── app/
│   ├── Beranda.py        ← halaman utama dashboard
│   ├── utils.py
│   └── pages/            ← 6 halaman dashboard
├── .streamlit/config.toml
└── requirements.txt
```

## 2. Instalasi (sekali saja)

Buka terminal (Anaconda Prompt / Command Prompt / Terminal) di folder proyek:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Jika memakai Anaconda, cukup `pip install -r requirements.txt` di environment Anda.

## 3. Menjalankan dashboard (langsung)

Model yang sudah dilatih **sudah disertakan** di folder `models/` dan `data/processed/`,
jadi dashboard bisa langsung dijalankan:

```bash
streamlit run app/Beranda.py
```

Browser akan terbuka di `http://localhost:8501`. Jalankan perintah dari **folder proyek**,
bukan dari dalam folder `app/`.

## 4. Mengulang seluruh proses dari data mentah

1. Salin `BX-Books.csv`, `BX-Users.csv`, `BX-Book-Ratings.csv` ke `data/raw/`.
2. Pilih salah satu:
   - **Belajar langkah demi langkah:** buka notebook 01 → 04 secara berurutan
     (`jupyter lab`, lalu buka folder `notebooks/`). Notebook 04 menyimpan model untuk dashboard.
   - **Sekali jalan:** `python -m src.pipeline` (±3–4 menit; `--fast` untuk melewati tuning SVD).
3. Jalankan dashboard seperti di langkah 3.

## 5. Alur pengerjaan

| Tahap | Isi | Di mana |
|---|---|---|
| 1. Cleaning | `escapechar` agar baris tidak bergeser; usia di luar 5–100 & tahun di luar 1800–2006 → kosong; buang rating ber-ISBN asing; **gabungkan edisi** (232 ribu judul dari 271 ribu ISBN) | notebook 01, `data_prep.py` |
| 2. EDA | 63% rating = 0 (implicit); rating eksplisit menumpuk di 7–10; long-tail ekstrem; pembaca setia pada penulis/seri | notebook 02 |
| 3. Filter & split | rating 1–10; buku ≥10 & user ≥3 rating → 130.559 rating, 11.867 user, 5.656 buku; split 80/20 **per user** | notebook 03 |
| 4. Model | Popularitas · Apriori · Content-Based · Item-CF · Funk SVD · PureSVD · Hybrid | notebook 03, `models.py` |
| 5. Tuning | Funk SVD (k, λ) dan hybrid (τ, α) dipilih di **data validasi** | notebook 03–04 |
| 6. Evaluasi | RMSE/MAE dan Precision/Recall/HitRate@10, coverage, per kelompok user | notebook 04 |
| 7. Dashboard | 6 halaman Streamlit | `app/` |

## 6. Aturan hybrid switching (hasil tuning)

| Riwayat user (n) | Model |
|---|---|
| n = 0 | Popularitas (*weighted rating*) |
| 1 ≤ n < 20 | 40% Content-Based + 60% PureSVD |
| n ≥ 20 | 20% Content-Based + 80% PureSVD |

## 7. Hasil utama (data uji)

| Model | RMSE | Precision@10 | Recall@10 | HitRate@10 | Coverage |
|---|---|---|---|---|---|
| Popularitas | – | 0,0035 | 0,017 | 3,2% | 0,4% |
| Rule-Based (Apriori) | – | 0,0112 | 0,062 | 10,1% | 2,5% |
| Content-Based | – | 0,0192 | 0,103 | 15,5% | 93,6% |
| Item-based CF | 1,736 | 0,0159 | 0,078 | 12,3% | 88,4% |
| Funk SVD | **1,558** | 0,0005 | 0,002 | 0,5% | 1,4% |
| PureSVD | – | 0,0159 | 0,081 | 13,0% | 13,1% |
| **Hybrid Switching** | – | **0,0226** | **0,120** | **17,7%** | 81,6% |

Pembanding RMSE: rata-rata global 1,776; baseline bias (μ + b_u + b_i) 1,578.

**Temuan penting:**
- Hybrid terbaik untuk Top-10 (+18% precision dibanding Content-Based saja).
- Funk SVD paling akurat memprediksi **nilai** rating, tetapi hampir gagal menyusun **peringkat**
  Top-10. Karena itu PureSVD yang dipakai di dalam hybrid, sedangkan Funk SVD dipakai untuk
  menampilkan prediksi rating.
- Precision terlihat kecil karena model memilih 10 dari ±5.650 buku dan rata-rata user hanya
  punya ±2 buku relevan di data uji. Tebakan acak ≈ 0,00036, jadi hybrid ±60× lebih baik dari acak.

## 8. Halaman dashboard

| Halaman | Isi |
|---|---|
| Beranda | ringkasan data, alur kerja, aturan hybrid |
| Eksplorasi Data | distribusi rating, long-tail, buku/penulis populer, profil user |
| Rekomendasi User | pilih user → riwayat, alasan pemilihan model, Top-N, tanda ✅ jika cocok dengan data uji, tampilan *mixed*, perbandingan model |
| Simulasi User Baru | pilih buku favorit & beri rating → lihat hybrid berpindah model secara langsung |
| Buku Mirip | buku serupa menurut Item-CF, Content-Based, dan Association Rules |
| Association Rules | filter aturan berdasarkan support / confidence / lift |
| Evaluasi Model | semua metrik, analisis per kelompok user, hasil tuning |

## 9. Masalah umum

- **`Artefak model belum ada`** → jalankan `python -m src.pipeline` (perlu CSV di `data/raw/`).
- **`ModuleNotFoundError: src`** → jalankan perintah dari folder proyek, bukan dari subfolder.
- **`FileNotFoundError: BX-Books.csv`** → periksa nama dan lokasi file di `data/raw/`.
- Mengubah parameter (misalnya ambang filter) → edit `src/config.py`, lalu jalankan ulang pipeline.

## 10. Referensi

- Ziegler, C.-N., McNee, S. M., Konstan, J. A., & Lausen, G. (2005). *Improving recommendation lists through topic diversification.* WWW '05. (sumber dataset Book-Crossing)
- Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix factorization techniques for recommender systems.* IEEE Computer, 42(8).
- Cremonesi, P., Koren, Y., & Turrin, R. (2010). *Performance of recommender algorithms on top-N recommendation tasks.* RecSys '10. (PureSVD)
- Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). *Item-based collaborative filtering recommendation algorithms.* WWW '01.
- Agrawal, R., & Srikant, R. (1994). *Fast algorithms for mining association rules.* VLDB '94. (Apriori)
- Materi *Recommendation System* — Statistical Data Mining, Dr. Muhammad Ahsan, ITS.
