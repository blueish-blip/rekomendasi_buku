import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import utils as U

U.setup("Rekomendasi User")
U.sidebar_note()
b = U.get_bundle()
h = b.meta["hybrid"]

st.title("👤 Rekomendasi untuk User")
st.caption("Riwayat = data latih. Buku di data uji disembunyikan dari model; jika muncul di "
           "rekomendasi, berarti model berhasil 'menebak' buku yang memang dibaca user.")

# ---------------------------------------------------------------- pilih user
n_hist = np.diff(b.R.indptr)
groups = {
    f"Riwayat sedikit (< {h['threshold']} buku)": n_hist < h["threshold"],
    f"Riwayat banyak (≥ {h['threshold']} buku)": n_hist >= h["threshold"],
    "Semua user": np.ones_like(n_hist, dtype=bool),
}
test_users = set(b.test.loc[b.test["rating"] >= b.meta["relevant_threshold"], "user_id"])
with st.sidebar:
    st.header("Pilih user")
    g = st.radio("Kelompok", list(groups), index=1)
    only_test = st.checkbox("Hanya user yang punya buku disukai di data uji", value=True)
    cand = b.index.users[groups[g]]
    if only_test:
        cand = np.array([u for u in cand if u in test_users])
    if st.button("🎲 User acak"):
        st.session_state["uid"] = int(np.random.default_rng().choice(cand))
    default = st.session_state.get("uid", int(cand[0]))
    if default not in set(cand.tolist()):
        default = int(cand[0])
    uid = st.selectbox(f"User-ID ({len(cand):,} user)", cand.tolist(),
                       index=int(np.where(cand == default)[0][0]))

item_idx, ratings = b.history(uid)
test_items = b.test_items(uid)
prof = b.users.set_index("user_id").loc[uid]
mdl, reason = b.models["hybrid"].choose(len(item_idx))

c = st.columns(4)
c[0].metric("User-ID", uid)
c[1].metric("Negara", str(prof["country"]).title() if pd.notna(prof["country"]) else "-")
c[2].metric("Usia", "-" if pd.isna(prof["age"]) else int(prof["age"]))
c[3].metric("Rating di data latih / uji", f"{len(item_idx)} / {len(test_items)}")
st.success(f"**Keputusan hybrid switching:** {reason}")

# ---------------------------------------------------------------- riwayat
with st.expander(f"📖 Riwayat membaca ({len(item_idx)} buku, data latih)", expanded=False):
    hist = b.catalog.iloc[item_idx][["title", "author", "year"]].copy()
    hist["rating user"] = ratings
    hist["tahun"] = hist["year"].map(U.fmt_year)
    hist = hist.drop(columns="year").sort_values("rating user", ascending=False)
    U.show_table(hist.rename(columns={"title": "judul", "author": "penulis"}))
    top_auth = hist["author"].value_counts().head(5)
    st.caption("Penulis paling sering dibaca: " + ", ".join(f"{a} ({n})" for a, n in top_auth.items()))

n = st.slider("Jumlah rekomendasi (Top-N)", 5, 30, 10)

tab1, tab2, tab3 = st.tabs(["⭐ Hybrid Switching", "🧩 Tampilan Mixed", "⚖️ Bandingkan semua model"])

with tab1:
    rec = b.recommend_user("hybrid", uid, n=n)
    hits = rec["book_id"].isin(test_items).sum()
    st.markdown(f"**Top-{n} rekomendasi** — {hits} buku cocok dengan data uji user ini.")
    U.show_table(U.rec_table(rec, test_items),
                 column_config={"skor": st.column_config.ProgressColumn("skor", min_value=0, max_value=1,
                                                                        format="%.3f")})
    st.caption("Skor = gabungan berbobot skor Content-Based dan PureSVD (masing-masing dinormalisasi 0–1). "
               "Prediksi rating = perkiraan rating yang akan diberikan user (Funk SVD, skala 1–10).")

with tab2:
    st.markdown("Contoh **mixed hybrid**: beberapa metode ditampilkan berdampingan sebagai baris terpisah.")
    rows = [("📖 Karena Anda membaca penulis/judul serupa", "cb"),
            ("👥 Pembaca dengan selera serupa juga membaca", "psvd"),
            ("🔗 Sering dibaca bersama (association rules)", "rule_rec"),
            ("🔥 Paling populer", "pop")]
    for title, key in rows:
        st.markdown(f"**{title}**")
        r = b.recommend_user(key, uid, n=5)
        cols = st.columns(5)
        for col, (_, x) in zip(cols, r.iterrows()):
            with col.container(border=True):
                mark = " ✅" if x["book_id"] in test_items else ""
                st.markdown(f"**{U.short(x['title'], 55)}**{mark}")
                st.caption(f"{x['author']} · {U.fmt_year(x['year'])}")

with tab3:
    st.markdown(f"Berapa buku di Top-{n} tiap model yang cocok dengan data uji user ini?")
    res, lists = [], {}
    for name, key in U.MODEL_KEYS.items():
        r = b.recommend_user(key, uid, n=n)
        lists[name] = r
        res.append({"model": name, "cocok dengan data uji": int(r["book_id"].isin(test_items).sum())})
    res = pd.DataFrame(res)
    fig = px.bar(res, x="cocok dengan data uji", y="model", orientation="h", color="model",
                 color_discrete_map=U.MODEL_COLORS, title=f"Jumlah 'hit' di Top-{n} (user {uid})")
    fig.update_layout(showlegend=False, yaxis_title="", height=360, xaxis=dict(dtick=1))
    U.show_chart(fig)
    pick = st.selectbox("Lihat daftar dari model", list(lists))
    U.show_table(U.rec_table(lists[pick], test_items))
    st.caption("Satu user hanya contoh; perbandingan yang sah ada di halaman **Evaluasi Model** "
               "(rata-rata ribuan user).")

with st.expander("🔎 Buku di data uji user ini (disembunyikan dari model)"):
    t = b.catalog.set_index("book_id").loc[list(test_items), ["title", "author"]].copy()
    t["rating"] = [test_items[k] for k in t.index]
    U.show_table(t.sort_values("rating", ascending=False).rename(columns={"title": "judul", "author": "penulis"}))
