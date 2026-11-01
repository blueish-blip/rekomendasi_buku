import numpy as np
import streamlit as st

import utils as U

U.setup("Simulasi User Baru")
U.sidebar_note()
b = U.get_bundle()
h = b.meta["hybrid"]
cat = b.catalog

st.title("✨ Simulasi User Baru (cold-start)")
st.markdown(
    "Anggap Anda pengguna baru. Pilih buku yang pernah Anda baca dan beri rating — "
    "lihat bagaimana **hybrid switching** berpindah model sesuai banyaknya riwayat: "
    f"**0 buku → Popularitas**, **1–{h['threshold'] - 1} buku → bobot α = {h['alpha_low']:.1f}**, "
    f"**≥ {h['threshold']} buku → bobot α = {h['alpha_high']:.1f}** (α = porsi Content-Based)."
)

# urutkan pilihan dari yang paling populer agar mudah dicari
order = np.argsort(-cat["n_rating_train"].to_numpy())
labels = {int(i): U.book_label(cat.iloc[i]) for i in order}

PRESETS = {
    "Penggemar Harry Potter": ("Harry Potter", "title"),
    "Penggemar John Grisham": ("John Grisham", "author"),
    "Penggemar Tolkien": ("Tolkien", "author"),
    "Penggemar Stephen King": ("Stephen King", "author"),
}
st.write("Contoh cepat:")
cols = st.columns(len(PRESETS) + 1)
for col, (name, (q, field)) in zip(cols, PRESETS.items()):
    if col.button(name):
        m = cat[cat[field].str.contains(q, case=False, na=False)]
        m = m.sort_values("n_rating_train", ascending=False).head(4)
        st.session_state["picked"] = [int(i) for i in m.index]
if cols[-1].button("Kosongkan"):
    st.session_state["picked"] = []

picked = st.multiselect("Buku yang pernah dibaca (ketik untuk mencari)", options=list(labels),
                        format_func=lambda i: labels[i], key="picked")

ratings = []
if picked:
    st.markdown("**Beri rating (1–10):**")
    grid = st.columns(min(len(picked), 4))
    for k, i in enumerate(picked):
        with grid[k % len(grid)]:
            ratings.append(st.slider(U.short(cat.iloc[i]["title"], 40), 1, 10, 9, key=f"r_{i}"))

item_idx = np.array(picked, dtype=int)
ratings = np.array(ratings, dtype=float)
_, reason = b.models["hybrid"].choose(len(item_idx))

st.divider()
c1, c2 = st.columns([1, 2])
c1.metric("Jumlah buku di riwayat", len(item_idx))
c2.success(f"**Keputusan hybrid:** {reason}")
if 0 < len(item_idx) < h["threshold"]:
    st.caption(f"Tambah {h['threshold'] - len(item_idx)} buku lagi agar porsi Collaborative Filtering naik.")

n = st.slider("Jumlah rekomendasi", 5, 20, 10)
rec = b.recommend("hybrid", item_idx, ratings, n=n)
U.show_table(U.rec_table(rec),
             column_config={"skor": st.column_config.ProgressColumn("skor", min_value=0, max_value=1,
                                                                    format="%.3f")})
if len(item_idx) == 0:
    st.caption("Skor popularitas = weighted rating (rata-rata rating yang 'ditarik' ke rata-rata global "
               "bila jumlah rating buku masih sedikit).")
else:
    st.caption("Prediksi rating dihitung dengan *fold-in* Funk SVD: vektor selera user baru diestimasi "
               "dari rating yang Anda berikan tanpa melatih ulang model.")

with st.expander("Bandingkan dengan Content-Based saja dan PureSVD saja"):
    if len(item_idx) == 0:
        st.write("Pilih minimal satu buku dulu.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Content-Based saja**")
            r = b.recommend("cb", item_idx, ratings, n=n)
            U.show_table(r[["title", "author"]].rename(columns={"title": "judul", "author": "penulis"}))
        with c2:
            st.markdown("**PureSVD (Collaborative) saja**")
            r = b.recommend("psvd", item_idx, ratings, n=n)
            U.show_table(r[["title", "author"]].rename(columns={"title": "judul", "author": "penulis"}))
        st.caption("Content-Based cenderung memberi penulis/seri yang sama; Collaborative memberi "
                   "buku lain yang dibaca orang berselera mirip. Hybrid menggabungkan keduanya.")
