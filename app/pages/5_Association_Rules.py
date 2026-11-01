import plotly.express as px
import streamlit as st

import utils as U

U.setup("Association Rules")
U.sidebar_note()
b = U.get_bundle()
rp = b.meta["rules_params"]
rules = b.rules

st.title("🔗 Association Rules (Rule-Based)")
st.markdown(
    f"Setiap user diperlakukan sebagai satu **transaksi** berisi buku yang pernah ia baca "
    f"(rating eksplisit maupun implicit). Algoritma **Apriori** mencari kombinasi buku yang sering "
    f"muncul bersama pada {b.meta['n_transactions']:,} transaksi dan {rp['top_books']} buku "
    f"paling populer (min support = {rp['min_support']}, min confidence = {rp['min_confidence']})."
)
with st.expander("Arti support, confidence, dan lift"):
    st.markdown(r"""
Untuk aturan **A → B**:

- **Support** $= P(A \cap B)$ — proporsi user yang membaca A **dan** B.
- **Confidence** $= P(B \mid A) = \dfrac{\text{support}(A \cap B)}{\text{support}(A)}$ — dari pembaca A, berapa persen juga membaca B.
- **Lift** $= \dfrac{\text{confidence}}{\text{support}(B)}$ — berapa kali lebih mungkin membaca B jika sudah membaca A,
  dibanding peluang dasar. **Lift > 1** = asosiasi positif.

Confidence saja bisa menipu: jika B memang sangat populer, confidence tinggi walau A dan B tidak berhubungan. Karena itu lift juga perlu dilihat.
""")

with st.sidebar:
    st.header("Filter aturan")
    min_sup = st.slider("Support minimum", float(rules["support"].min()), float(rules["support"].max()),
                        float(rules["support"].min()), step=0.001, format="%.3f")
    min_conf = st.slider("Confidence minimum", 0.0, 1.0, 0.5, 0.05)
    min_lift = st.slider("Lift minimum", 1.0, float(rules["lift"].max()), 1.0, 0.5)
    length = st.multiselect("Panjang itemset", [2, 3], default=[2, 3],
                            help="2 = {A}→{B}; 3 = {A, B}→{C}")
    q = st.text_input("Cari judul (opsional)")

f = rules[(rules["support"] >= min_sup) & (rules["confidence"] >= min_conf) & (rules["lift"] >= min_lift)
          & rules["panjang"].isin(length)]
if q:
    f = f[f["jika_membaca"].str.contains(q, case=False) | f["maka_rekomendasikan"].str.contains(q, case=False)]

c = st.columns(4)
c[0].metric("Aturan lolos filter", f"{len(f):,}")
c[1].metric("Total aturan", f"{len(rules):,}")
c[2].metric("Median lift (terfilter)", f"{f['lift'].median():.1f}" if len(f) else "-")
c[3].metric("Confidence tertinggi", f"{f['confidence'].max():.2f}" if len(f) else "-")

if len(f):
    sample = f.sort_values("lift", ascending=False).head(3000)
    fig = px.scatter(sample, x="support", y="confidence", color="lift", color_continuous_scale=U.SEQ,
                     hover_data={"jika_membaca": True, "maka_rekomendasikan": True, "lift": ":.1f",
                                 "support": ":.4f", "confidence": ":.2f"},
                     title="Support vs confidence (warna = lift)")
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="white")))
    U.show_chart(fig)

    sort = st.radio("Urutkan menurut", ["lift", "confidence", "support"], horizontal=True)
    show = f.sort_values(sort, ascending=False).head(200)[
        ["jika_membaca", "maka_rekomendasikan", "support", "confidence", "lift"]].copy()
    show[["support", "confidence", "lift"]] = show[["support", "confidence", "lift"]].round(4)
    U.show_table(show.rename(columns={"jika_membaca": "jika membaca", "maka_rekomendasikan": "maka rekomendasikan"}),
                 height=480)
    st.caption("Aturan dengan lift sangat tinggi umumnya **buku dalam satu seri** "
               "(mis. Stephanie Plum, Harry Potter, Lord of the Rings).")
else:
    st.warning("Tidak ada aturan yang lolos filter. Longgarkan filternya.")
