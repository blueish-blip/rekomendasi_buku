"""Fungsi bersama untuk semua halaman dashboard."""
import inspect
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import artifacts  # noqa: E402
from src import config as C  # noqa: E402

# Palet (urutan tetap) - biru = utama, oranye = sorotan (model hybrid)
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
GRAY = "#9a9893"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

MODEL_COLORS = {
    "Popularitas": GRAY,
    "Rule-Based (Association Rules)": YELLOW,
    "Content-Based": AQUA,
    "Item-based CF": VIOLET,
    "SVD (Matrix Factorization)": "#e87ba4",
    "PureSVD (Truncated SVD)": BLUE,
    "Hybrid Switching": ORANGE,
}

MODEL_KEYS = {
    "Hybrid Switching": "hybrid",
    "Content-Based": "cb",
    "Item-based CF": "icf",
    "PureSVD (Truncated SVD)": "psvd",
    "SVD (Matrix Factorization)": "svd",
    "Rule-Based (Association Rules)": "rule_rec",
    "Popularitas": "pop",
}

def _has_width_param(fn):
    """Streamlit versi baru memakai width="stretch"; versi lama memakai use_container_width=True."""
    try:
        return "width" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


_NEW_WIDTH = _has_width_param(st.plotly_chart) and _has_width_param(st.dataframe)


def setup(title, icon="📚"):
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")


@st.cache_resource(show_spinner="Memuat model...")
def get_bundle():
    try:
        return artifacts.load()
    except FileNotFoundError:
        st.error("Artefak model belum ada. Jalankan dulu dari folder proyek:\n\n"
                 "`python -m src.pipeline`")
        st.stop()


def show_chart(fig):
    fig.update_layout(template="plotly_white", font=dict(size=13),
                      margin=dict(l=10, r=10, t=50, b=10), hoverlabel=dict(font_size=13))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    if _NEW_WIDTH:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def show_table(df, **kw):
    kw.setdefault("hide_index", True)
    if _NEW_WIDTH:
        st.dataframe(df, width="stretch", **kw)
    else:
        st.dataframe(df, use_container_width=True, **kw)


def fmt_year(y):
    try:
        return str(int(float(y)))
    except (TypeError, ValueError):
        return "-"


def short(s, n=60):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def book_label(row):
    return f"{short(row['title'], 70)} — {row['author']}"


def rec_table(rec, test_items=None):
    """Tabel rekomendasi siap tampil. test_items: dict book_id -> rating di data uji."""
    df = rec.copy()
    out = df[["title", "author"]].copy()
    out.insert(0, "#", range(1, len(df) + 1))
    out["tahun"] = df["year"].map(fmt_year)
    out["skor"] = df["skor"]
    out["prediksi rating (SVD)"] = df["pred_rating_svd"]
    out["rating rata-rata"] = df["mean_rating_train"].round(2)
    out["jml rating"] = df["n_rating_train"].astype(int)
    if test_items is not None:
        out["ada di data uji?"] = df["book_id"].map(
            lambda b: f"✅ dirating {test_items[b]:.0f}" if b in test_items else "")
    return out.rename(columns={"title": "judul", "author": "penulis"})


def sidebar_note():
    with st.sidebar:
        st.caption("Proyek ADL · Sistem Rekomendasi Buku (Book-Crossing)")
