import plotly.express as px
import streamlit as st

import utils as U

U.setup("Evaluasi Model")
U.sidebar_note()
b = U.get_bundle()
ev, h, meta = b.evals, b.meta["hybrid"], b.meta
K = meta["top_n"]

st.title("📈 Evaluasi Model")
st.markdown(
    f"Data model dibagi **per user**: ±80% rating untuk latih, ±20% untuk uji "
    f"({meta['summary']['train_ratings']:,} vs {meta['summary']['test_ratings']:,} rating). "
    f"Pemilihan hyperparameter memakai **data validasi** yang dipisah dari data latih, "
    f"sehingga data uji hanya dipakai sekali untuk angka akhir di halaman ini."
)

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediksi rating", f"🏆 Top-{K}", "👥 Menurut aktivitas user",
                                  "🔧 Tuning"])

# ---------------------------------------------------------------- RMSE / MAE
with tab1:
    st.markdown("Seberapa dekat **prediksi nilai rating** dengan rating sebenarnya di data uji (skala 1–10).")
    r = ev["rating"].copy()
    c1, c2 = st.columns([1.3, 1])
    with c1:
        long = r.melt(id_vars="model", var_name="metrik", value_name="nilai")
        fig = px.bar(long, x="nilai", y="model", color="metrik", barmode="group", orientation="h",
                     color_discrete_map={"RMSE": U.BLUE, "MAE": U.AQUA},
                     title="RMSE dan MAE di data uji (lebih kecil = lebih baik)", text_auto=".3f")
        fig.update_layout(yaxis_title="", legend_title_text="", height=380)
        U.show_chart(fig)
    with c2:
        U.show_table(r.round(4))
        st.markdown(r"""
- $\text{RMSE}=\sqrt{\tfrac1n\sum (r-\hat r)^2}$ — menghukum kesalahan besar.
- $\text{MAE}=\tfrac1n\sum |r-\hat r|$ — rata-rata selisih absolut.

**Funk SVD** memberi error terendah. Peningkatan dari baseline bias kecil karena data sangat
jarang dan rating menumpuk di 7–10; sebagian besar variasi sudah dijelaskan oleh bias user/buku.
""")

# ---------------------------------------------------------------- Top-N
with tab2:
    t = ev["topn"].copy()
    st.markdown(
        f"Untuk setiap user di data uji, model menyusun Top-{K} dari buku yang **belum** ada di riwayat "
        f"latihnya. Buku **relevan** = buku di data uji yang diberi rating ≥ {meta['relevant_threshold']}. "
        f"Dievaluasi pada {int(t['n_user'].iloc[0]):,} user.")
    metric = st.radio("Metrik", [f"Precision@{K}", f"Recall@{K}", f"HitRate@{K}", "Coverage"], horizontal=True)
    tt = t.sort_values(metric)
    fig = px.bar(tt, x=metric, y="model", orientation="h", color="model", color_discrete_map=U.MODEL_COLORS,
                 text_auto=".4f", title=f"{metric} per model")
    fig.update_layout(showlegend=False, yaxis_title="", height=400)
    U.show_chart(fig)
    show = t.drop(columns="n_user").copy()
    U.show_table(show.round(4))
    st.markdown(f"""
**Cara membaca:**
- **Precision@{K}** — dari {K} rekomendasi, berapa proporsi yang relevan.
- **Recall@{K}** — dari semua buku relevan user, berapa yang berhasil masuk Top-{K}.
- **HitRate@{K}** — proporsi user yang minimal 1 buku relevannya masuk Top-{K}.
- **Coverage** — proporsi katalog yang pernah direkomendasikan ke minimal satu user (keberagaman).

Angka precision terlihat kecil karena model memilih 10 dari **±5.650 buku** dan rata-rata user hanya
punya ±2 buku relevan di data uji. Tebakan acak menghasilkan Precision@10 ≈ 0,00036, jadi hybrid
±60 kali lebih baik dari acak. Yang penting adalah **perbandingan antar model**.

**Temuan utama:**
1. **Hybrid Switching terbaik** di precision, recall, dan hit rate.
2. **Funk SVD** paling akurat memprediksi rating, tetapi hampir gagal di Top-N — model yang
   dioptimalkan untuk RMSE cenderung mengunggulkan buku niche dengan rating tinggi. Untuk peringkat,
   **PureSVD** (sel kosong dianggap sinyal) jauh lebih baik (Cremonesi dkk., 2010).
3. **Popularitas** hampir tidak personal dan coverage-nya sangat kecil (selalu buku yang sama).
4. **Content-Based** kuat karena pembaca Book-Crossing setia pada penulis/seri.
""")

# ---------------------------------------------------------------- per aktivitas
with tab3:
    a = ev["by_activity"].copy()
    order = ["2", "3-4", "5-9", "10-19", "20+"]
    a["aktivitas"] = a["aktivitas"].astype(str)
    models = st.multiselect("Model", a["model"].unique().tolist(),
                            default=["Hybrid Switching", "Content-Based", "PureSVD (Truncated SVD)",
                                     "Item-based CF", "Popularitas"])
    a = a[a["model"].isin(models)]
    fig = px.line(a, x="aktivitas", y="precision", color="model", markers=True,
                  category_orders={"aktivitas": order}, color_discrete_map=U.MODEL_COLORS,
                  title=f"Precision@{K} menurut jumlah rating latih user")
    fig.update_traces(line=dict(width=2), marker=dict(size=8))
    fig.update_layout(xaxis_title="jumlah rating di data latih", yaxis_title=f"Precision@{K}",
                      legend_title_text="")
    U.show_chart(fig)
    st.markdown("Semua model membaik saat riwayat user bertambah. Untuk user dengan riwayat sedikit, "
                "Content-Based cukup kuat; untuk user aktif, model Collaborative mengejar bahkan melampaui. "
                "Pola ini yang mendasari **switching**.")
    n_users = a.drop_duplicates("aktivitas").set_index("aktivitas")["n_user"].reindex(order)
    st.caption("Jumlah user per kelompok: " + ", ".join(f"{k}: {int(v):,}" for k, v in n_users.dropna().items()))

# ---------------------------------------------------------------- tuning
with tab4:
    st.subheader("Hybrid: bobot α (porsi Content-Based) per kelompok user — data validasi")
    act = ev["tuning_hybrid_alpha_activity"].copy()
    act["aktivitas"] = act["aktivitas"].astype(str)
    piv = act.pivot(index="alpha", columns="aktivitas", values="precision")[
        [c for c in ["2", "3-4", "5-9", "10-19", "20+"] if c in set(act["aktivitas"])]]
    fig = px.imshow(piv.sort_index(ascending=False), text_auto=".4f", aspect="auto",
                    color_continuous_scale=U.SEQ,
                    labels=dict(x="jumlah rating latih user", y="α (porsi Content-Based)", color="Precision"),
                    title="Precision@10 validasi untuk setiap α dan kelompok user")
    U.show_chart(fig)
    st.markdown(f"""
- α = 1 → Content-Based saja; α = 0 → PureSVD saja. Nilai di antaranya **selalu lebih baik** dari keduanya.
- Bobot terbaik bergeser ke Collaborative ketika riwayat user banyak.
- Kombinasi terpilih: **threshold = {h['threshold']}**, **α_low = {h['alpha_low']}**, **α_high = {h['alpha_high']}**
  (Precision@10 validasi = {h['precision_validasi']:.4f}).
""")
    g = ev["tuning_hybrid_grid"].copy()
    best_t = g.sort_values("precision@10", ascending=False).groupby("threshold").head(1).sort_values("threshold")
    same = g[g["alpha_low"] == g["alpha_high"]].sort_values("precision@10", ascending=False).head(1)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Kombinasi terbaik per threshold**")
        U.show_table(best_t.round(4))
    with c2:
        st.markdown("**Tanpa switching (satu α untuk semua user)**")
        U.show_table(same[["alpha_low", "precision@10"]].rename(columns={"alpha_low": "alpha"}).round(4))
        gain = h["precision_validasi"] / same["precision@10"].iloc[0] - 1
        st.caption(f"Switching menambah {gain:+.1%} dibanding satu bobot tetap. Kenaikan terbesar "
                   "datang dari penggabungan CB + CF itu sendiri.")
    if "tuning_svd" in ev:
        st.subheader("Funk SVD: tuning jumlah faktor laten dan regularisasi (data validasi)")
        U.show_table(ev["tuning_svd"].round(4))
