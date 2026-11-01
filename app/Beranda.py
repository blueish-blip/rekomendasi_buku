"""Halaman utama dashboard.  Jalankan:  streamlit run app/Beranda.py"""
import streamlit as st

import utils as U

U.setup("Sistem Rekomendasi Buku")
U.sidebar_note()
b = U.get_bundle()
s, h = b.meta["summary"], b.meta["hybrid"]

st.title("📚 Sistem Rekomendasi Buku — Book-Crossing")
st.markdown(
    "Dashboard ini menyajikan sistem rekomendasi **hybrid switching** yang menggabungkan "
    "*Content-Based* dan *Collaborative Filtering*, lengkap dengan pembanding Rule-Based, "
    "Item-based CF, dan Matrix Factorization."
)

c = st.columns(4)
c[0].metric("Rating (setelah cleaning)", f"{s['ratings_clean']:,}")
c[1].metric("Rating eksplisit (1–10)", f"{s['explicit']:,}",
            help="Rating 0 = implicit (berinteraksi tapi tidak menilai)")
c[2].metric("Judul buku unik", f"{s['books_clean']:,}",
            help=f"Dari {s['isbn_clean']:,} ISBN; edisi-edisi satu judul digabung")
c[3].metric("User", f"{s['users_clean']:,}")

c = st.columns(4)
c[0].metric("Data model: rating", f"{s['model_ratings']:,}",
            help=f"Buku ≥{b.meta['filter']['min_book']} rating, user ≥{b.meta['filter']['min_user']} rating")
c[1].metric("Data model: user × buku", f"{s['model_users']:,} × {s['model_books']:,}")
c[2].metric("Kekosongan matriks", f"{s['model_sparsity_pct']:.2f}%")
topn = b.evals["topn"].set_index("model")
c[3].metric("Precision@10 Hybrid", f"{topn.loc['Hybrid Switching', 'Precision@10']:.4f}",
            delta=f"{(topn.loc['Hybrid Switching', 'Precision@10'] / topn.loc['Content-Based', 'Precision@10'] - 1):+.0%} vs Content-Based")

st.divider()
left, right = st.columns([1.1, 1])
with left:
    st.subheader("Alur pengerjaan")
    st.graphviz_chart(f"""
    digraph {{
      rankdir=TB; node [shape=box, style="rounded,filled", fillcolor="#eef4fc", color="#2a78d6",
                        fontname="Helvetica", fontsize=11]; edge [color="#9a9893"];
      a [label="1. Data mentah\\n3 CSV Book-Crossing"];
      b [label="2. Cleaning\\nencoding, ISBN, usia, tahun,\\npenggabungan edisi"];
      c [label="3. EDA"];
      d [label="4. Filter & split\\nbuku ≥{b.meta['filter']['min_book']}, user ≥{b.meta['filter']['min_user']} · 80/20 per user"];
      e [label="5. Model\\nPopularitas · Rules · CB ·\\nItem-CF · Funk SVD · PureSVD"];
      f [label="6. Tuning di data validasi\\n(threshold & bobot hybrid)"];
      g [label="7. Evaluasi di data uji\\nRMSE · MAE · Precision@10"];
      h [label="8. Dashboard Streamlit", fillcolor="#fde5db", color="#eb6834"];
      a->b->c->d->e->f->g->h;
    }}""")
with right:
    st.subheader("Aturan hybrid switching")
    st.markdown(f"""
| Riwayat user (n) | Model yang dipakai |
|---|---|
| n = 0 | **Popularitas** (cold-start total) |
| 1 ≤ n < {h['threshold']} | **{h['alpha_low']:.0%} Content-Based + {1 - h['alpha_low']:.0%} PureSVD** |
| n ≥ {h['threshold']} | **{h['alpha_high']:.0%} Content-Based + {1 - h['alpha_high']:.0%} PureSVD** |

Ambang **{h['threshold']}** dan kedua bobot dipilih otomatis dari data **validasi**
(bukan data uji). Polanya: makin banyak riwayat user, makin besar porsi
*Collaborative Filtering*.
""")
    st.info("**Kenapa PureSVD, bukan Funk SVD, di dalam hybrid?**  Funk SVD paling akurat "
            "memprediksi *nilai* rating (RMSE terendah), tetapi buruk untuk menyusun *peringkat* "
            "Top-10. PureSVD dirancang untuk Top-N. Funk SVD tetap dipakai untuk menampilkan "
            "**prediksi rating** di setiap rekomendasi.")

st.divider()
st.subheader("Isi dashboard")
st.markdown("""
- **📊 Eksplorasi Data** — distribusi rating, aktivitas user, buku & penulis terpopuler.
- **👤 Rekomendasi User** — pilih user, lihat riwayat, alasan pemilihan model, dan Top-10; cek apakah rekomendasi cocok dengan data uji.
- **✨ Simulasi User Baru** — pilih beberapa buku favorit dan lihat hybrid berpindah model secara langsung.
- **🔍 Buku Mirip** — buku serupa menurut Item-based CF, Content-Based, dan Association Rules.
- **🔗 Association Rules** — aturan "jika membaca A maka membaca B" (support, confidence, lift).
- **📈 Evaluasi Model** — perbandingan semua model dan hasil tuning.
""")
