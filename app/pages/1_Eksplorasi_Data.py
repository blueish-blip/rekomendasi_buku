import plotly.express as px
import streamlit as st

import utils as U

U.setup("Eksplorasi Data")
U.sidebar_note()
b = U.get_bundle()
e, s = b.eda, b.meta["summary"]

st.title("📊 Eksplorasi Data")
st.caption("Seluruh data setelah cleaning (sebelum difilter untuk pemodelan).")

# ---------------------------------------------------------------- rating
st.subheader("1. Distribusi rating")
rd = e["rating_dist"].copy()
rd["jenis"] = rd["rating"].map(lambda r: "Implicit (0)" if r == 0 else "Eksplisit (1–10)")
c1, c2 = st.columns([1.6, 1])
with c1:
    fig = px.bar(rd, x="rating", y="jumlah", color="jenis",
                 color_discrete_map={"Implicit (0)": U.GRAY, "Eksplisit (1–10)": U.BLUE},
                 title="Jumlah rating per nilai")
    fig.update_layout(xaxis=dict(dtick=1), legend_title_text="", bargap=0.15)
    U.show_chart(fig)
with c2:
    imp = s["implicit"] / (s["implicit"] + s["explicit"])
    st.metric("Porsi rating 0 (implicit)", f"{imp:.0%}")
    st.markdown(
        "- Rating **0** berarti user berinteraksi tanpa memberi nilai, **bukan** rating buruk.\n"
        "- Rating eksplisit menumpuk di **7–10** (rata-rata ±7,6), jadi wajar bila model "
        "memakai **bias user & bias buku**.\n"
        "- Rating 0 hanya dipakai untuk *association rules*; model prediksi rating hanya "
        "memakai rating 1–10.")

# ---------------------------------------------------------------- long tail
st.subheader("2. Long-tail: kebanyakan user & buku hanya punya sedikit rating")
c1, c2 = st.columns(2)
with c1:
    fig = px.bar(e["user_activity"], x="jumlah_rating", y="jumlah_user",
                 title="User menurut jumlah rating eksplisit", color_discrete_sequence=[U.BLUE])
    U.show_chart(fig)
with c2:
    fig = px.bar(e["book_popularity"], x="jumlah_rating", y="jumlah_buku",
                 title="Buku menurut jumlah rating eksplisit", color_discrete_sequence=[U.BLUE])
    U.show_chart(fig)
st.info(f"Karena sangat jarang, data model difilter menjadi buku ≥{b.meta['filter']['min_book']} rating "
        f"dan user ≥{b.meta['filter']['min_user']} rating → **{s['model_ratings']:,} rating**, "
        f"{s['model_users']:,} user, {s['model_books']:,} buku. Matriksnya tetap "
        f"**{s['model_sparsity_pct']:.2f}% kosong** — inilah alasan diperlukan hybrid.")

# ---------------------------------------------------------------- buku & penulis
st.subheader("3. Buku dan penulis terpopuler")
c1, c2 = st.columns(2)
with c1:
    n = st.slider("Jumlah buku", 5, 30, 15, key="nb")
    tb = e["top_books"].head(n).copy()
    tb["judul"] = tb["title"].map(lambda t: U.short(t, 45))
    fig = px.bar(tb.iloc[::-1], x="n_interaksi", y="judul", orientation="h",
                 hover_data={"author": True, "rata2": ":.2f", "n_rating": True, "judul": False},
                 title="Buku dengan interaksi terbanyak", color_discrete_sequence=[U.BLUE])
    fig.update_layout(height=120 + 26 * n, yaxis_title="")
    U.show_chart(fig)
with c2:
    ta = e["top_authors"].head(15)
    fig = px.bar(ta.iloc[::-1], x="n_interaksi", y="author", orientation="h",
                 title="Penulis dengan interaksi terbanyak", color_discrete_sequence=[U.AQUA])
    fig.update_layout(height=120 + 26 * 15, yaxis_title="")
    U.show_chart(fig)

with st.expander("Buku dengan rata-rata rating tertinggi (minimal 50 rating eksplisit)"):
    tr = e["top_rated"][["title", "author", "n_rating", "rata2"]].rename(
        columns={"title": "judul", "author": "penulis", "n_rating": "jml rating", "rata2": "rata-rata"})
    tr["rata-rata"] = tr["rata-rata"].round(2)
    U.show_table(tr)

# ---------------------------------------------------------------- user
st.subheader("4. Profil user")
c1, c2 = st.columns(2)
with c1:
    fig = px.bar(e["age_dist"], x="usia", y="jumlah", title="Distribusi usia user",
                 color_discrete_sequence=[U.BLUE])
    fig.update_layout(bargap=0.05)
    U.show_chart(fig)
    st.caption(f"Usia kosong/tidak valid: {s['age_missing_pct']:.0f}% user "
               "(usia < 5 atau > 100 dianggap tidak valid).")
with c2:
    cn = e["countries"].head(12)
    fig = px.bar(cn.iloc[::-1], x="jumlah_user", y="negara", orientation="h",
                 title="Negara asal user aktif (12 teratas)", color_discrete_sequence=[U.BLUE])
    fig.update_layout(yaxis_title="")
    U.show_chart(fig)

st.subheader("5. Tahun terbit")
fig = px.bar(e["year_dist"], x="tahun", y="jumlah_buku", title="Jumlah judul per tahun terbit (≥1950)",
             color_discrete_sequence=[U.BLUE])
fig.update_layout(bargap=0.05)
U.show_chart(fig)
