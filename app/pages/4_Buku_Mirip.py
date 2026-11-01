import numpy as np
import streamlit as st

import utils as U

U.setup("Buku Mirip")
U.sidebar_note()
b = U.get_bundle()
cat = b.catalog

st.title("🔍 Buku Mirip")
st.markdown("Bandingkan tiga cara mencari buku serupa untuk satu judul.")

order = np.argsort(-cat["n_rating_train"].to_numpy())
labels = {int(i): U.book_label(cat.iloc[i]) for i in order}
i = st.selectbox("Pilih buku", list(labels), format_func=lambda k: labels[k])
book = cat.iloc[i]

c = st.columns(4)
c[0].metric("Penulis", U.short(book["author"], 28))
c[1].metric("Tahun", U.fmt_year(book["year"]))
c[2].metric("Jumlah rating (latih)", int(book["n_rating_train"]))
c[3].metric("Rata-rata rating", f"{book['mean_rating_train']:.2f}")
st.caption(f"Penerbit: {book['publisher']} · edisi yang digabung: {int(book['n_editions'])}")

n = st.slider("Jumlah buku", 5, 20, 10)


def to_frame(idx, sims, col):
    df = cat.iloc[idx][["title", "author"]].copy()
    df[col] = np.round(sims, 3)
    return df.rename(columns={"title": "judul", "author": "penulis"})


c1, c2 = st.columns(2)
with c1:
    st.subheader("👥 Item-based CF")
    st.caption("Mirip karena **dinilai serupa oleh user yang sama** (adjusted cosine, 50 tetangga).")
    idx, sims = b.models["icf"].similar_items(i, n=n)
    if len(idx):
        U.show_table(to_frame(idx, sims, "kemiripan"))
    else:
        st.write("Tidak ada tetangga dengan kemiripan positif.")
with c2:
    st.subheader("📖 Content-Based")
    st.caption("Mirip karena **judul, penulis, dan penerbit** serupa (TF-IDF + cosine).")
    X = b.models["cb"].X
    s = np.asarray((X[i] @ X.T).todense()).ravel()
    s[i] = -1
    idx = np.argsort(-s)[:n]
    U.show_table(to_frame(idx, s[idx], "cosine"))

st.subheader("🔗 Association Rules yang memuat buku ini")
bid = book["book_id"]
r = b.rules[b.rules["antecedent_ids"].str.split("|").map(lambda a: bid in a)]
if r.empty:
    st.write("Buku ini tidak termasuk dalam aturan asosiasi "
             f"(aturan hanya dibentuk untuk {b.meta['rules_params']['top_books']} buku paling sering dibaca).")
else:
    r = r.sort_values(["confidence", "lift"], ascending=False).head(n)
    show = r[["jika_membaca", "maka_rekomendasikan", "support", "confidence", "lift"]].copy()
    show[["support", "confidence", "lift"]] = show[["support", "confidence", "lift"]].round(3)
    U.show_table(show.rename(columns={"jika_membaca": "jika membaca", "maka_rekomendasikan": "maka rekomendasikan"}))

st.info("Perhatikan perbedaannya: Content-Based hampir selalu memberi **penulis/seri yang sama**, "
        "sedangkan Item-based CF bisa memberi **penulis lain** yang disukai pembaca yang sama "
        "(*serendipity*). Inilah alasan keduanya digabung dalam hybrid.")
