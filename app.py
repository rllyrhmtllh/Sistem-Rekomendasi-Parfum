import base64
import math
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from recommender import (
    get_similar_perfumes,
    has_vocabulary_overlap,
    load_artifacts,
    preprocess_text,
    recommend_by_preference,
    recommend_hybrid,
)
from image_utils import PLACEHOLDER_IMAGE, get_product_image_url

BG_IMAGE_PATH = Path(__file__).parent / "image-background" / "bg-image.jpg"
PLACEHOLDER_VALUES = {"", "0", "000", "null", "nan", "none", "n/a", "unknown"}
HOME_PAGE_SIZE = 6


@st.cache_data(show_spinner=False)
def get_background_image_data() -> str:
    encoded = base64.b64encode(BG_IMAGE_PATH.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


BACKGROUND_IMAGE = get_background_image_data()

st.set_page_config(
    page_title="Rekomendasi Parfum - Content-Based & Hybrid Filtering",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Poppins:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif;
    }}

    .stApp {{
        background:
            linear-gradient(rgba(42, 26, 12, 0.52), rgba(42, 26, 12, 0.64)),
            url('{BACKGROUND_IMAGE}') center center / cover no-repeat fixed;
    }}

    .main .block-container {{
        background: linear-gradient(180deg, rgba(30, 18, 8, 0.72), rgba(18, 10, 4, 0.82));
        border: 1px solid rgba(243, 210, 137, 0.58);
        border-radius: 24px;
        padding: 1.25rem 1.35rem 1.4rem 1.35rem;
        backdrop-filter: blur(12px);
        box-shadow:
            0 12px 30px rgba(0, 0, 0, 0.32),
            inset 0 0 0 1px rgba(255, 239, 201, 0.08);
    }}

    .hero {{
        padding: 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(14, 11, 8, 0.82), rgba(53, 32, 13, 0.84), rgba(114, 75, 19, 0.86));
        color: #f5eede;
        margin-bottom: 1.1rem;
        border: 1px solid rgba(243, 210, 137, 0.30);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }}

    .hero-eyebrow {{
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #ffe3a6;
        margin-bottom: 0.55rem;
    }}

    .hero h1 {{
        font-family: 'Playfair Display', serif;
        font-size: 2.15rem;
        margin-bottom: 0.35rem;
        color: #f4d78e;
    }}

    .hero p {{
        font-size: 0.98rem;
        color: #e8e0cf;
        margin: 0;
        max-width: 860px;
    }}

    .page-intro {{
        margin: 0 0 1rem 0;
        padding: 1rem 1.15rem;
        border-radius: 18px;
        background: rgba(246, 232, 204, 0.10);
        border: 1px solid rgba(243, 210, 137, 0.24);
    }}

    .page-intro strong {{
        color: #f8d98f;
    }}

    .page-intro p {{
        color: #f8f1df;
        margin: 0.2rem 0 0 0;
        font-size: 0.95rem;
    }}

    .metric-card {{
        background: rgba(251, 245, 234, 0.90);
        border: 1px solid rgba(212, 175, 55, 0.55);
        border-radius: 16px;
        padding: 1rem 1.05rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
        min-height: 126px;
    }}

    .metric-label {{
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #a8842f;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }}

    .metric-value {{
        font-family: 'Playfair Display', serif;
        color: #2d1d09;
        font-size: 1.7rem;
        line-height: 1.1;
        margin-bottom: 0.35rem;
    }}

    .metric-note {{
        color: #5f4a23;
        font-size: 0.86rem;
        line-height: 1.5;
    }}

    .perfume-card {{
        background: rgba(251, 245, 234, 0.90);
        border-radius: 16px;
        padding: 0;
        overflow: hidden;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.16);
        border: 1px solid rgba(212, 175, 55, 0.55);
        margin-bottom: 1.2rem;
        transition: transform 0.15s ease;
        backdrop-filter: blur(4px);
    }}

    .perfume-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.14);
    }}

    .card-body {{
        padding: 0.9rem 1.1rem 1.1rem 1.1rem;
    }}

    .card-brand {{
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 0.72rem;
        color: #a8842f;
        font-weight: 600;
        margin-bottom: 0.1rem;
    }}

    .card-title {{
        font-family: 'Playfair Display', serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #2d1d09;
        margin-bottom: 0.35rem;
        line-height: 1.25;
    }}

    .badge {{
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 500;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
    }}

    .badge-gender {{
        background: rgba(35, 43, 50, 0.95);
        color: #ffffff;
    }}

    .badge-accord {{
        background: rgba(245, 233, 201, 0.96);
        color: #7a5a17;
    }}

    .score-bar-bg {{
        background: #f0f0f0;
        border-radius: 999px;
        height: 8px;
        width: 100%;
        margin-top: 0.5rem;
        overflow: hidden;
    }}

    .score-bar-fill {{
        background: linear-gradient(90deg, #d4af37, #a8842f);
        height: 100%;
        border-radius: 999px;
    }}

    .score-label {{
        font-size: 0.75rem;
        color: #666666;
        margin-top: 0.2rem;
    }}

    .card-link a {{
        display: inline-block;
        margin-top: 0.6rem;
        font-size: 0.82rem;
        font-weight: 600;
        color: #875410;
        text-decoration: none;
        border-bottom: 1px solid #875410;
    }}

    .perfume-card a img {{
        display: block;
        border: 2px solid rgba(212, 175, 55, 0.82);
        box-sizing: border-box;
        transition: transform 0.18s ease, filter 0.18s ease, box-shadow 0.18s ease;
        cursor: pointer;
    }}

    .perfume-card a:hover img {{
        transform: scale(1.02);
        filter: saturate(1.08) brightness(1.03);
        box-shadow: 0 0 0 2px rgba(212, 175, 55, 0.38);
    }}

    .section-title {{
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        color: #fff2cf;
        margin: 0.4rem 0 0.65rem 0;
        border-left: 4px solid #d4af37;
        padding-left: 0.6rem;
        text-shadow: 0 2px 12px rgba(0, 0, 0, 0.5);
    }}

    .section-copy {{
        color: #f7efdd;
        font-size: 0.94rem;
        margin: 0 0 1rem 0;
    }}

    .sidebar-panel {{
        padding: 0.85rem 0.95rem;
        margin-bottom: 0.85rem;
        border-radius: 16px;
        background: rgba(255, 248, 233, 0.72);
        border: 1px solid rgba(168, 132, 47, 0.20);
    }}

    .sidebar-panel h3 {{
        margin: 0;
        font-size: 1rem;
    }}

    .sidebar-panel p {{
        margin: 0.35rem 0 0 0;
        font-size: 0.85rem;
        line-height: 1.45;
    }}

    .stTabs [data-testid="stTab"] {{
        background: rgba(47, 30, 13, 0.54);
        border: 1px solid rgba(243, 210, 137, 0.38);
        border-radius: 12px 12px 0 0;
        color: #f6ead1;
        padding: 0.55rem 0.9rem;
        margin-right: 0.35rem;
    }}

    .stTabs [data-testid="stTab"][aria-selected="true"] {{
        background: linear-gradient(180deg, rgba(116, 79, 26, 0.9), rgba(75, 47, 15, 0.92));
        border-color: rgba(243, 210, 137, 0.85);
        color: #fff3d2;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.24);
    }}

    .stButton > button[kind="secondary"] {{
        background: linear-gradient(180deg, rgba(65, 42, 16, 0.92), rgba(43, 28, 11, 0.94));
        color: #f8e6bd;
        border: 1px solid rgba(233, 194, 103, 0.48);
        border-radius: 10px;
        min-height: 2.25rem;
        padding: 0.2rem 0.5rem;
        font-size: 0.82rem;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.14);
    }}

    .stButton > button[kind="secondary"]:hover {{
        border-color: rgba(243, 210, 137, 0.82);
        color: #fff3d2;
    }}

    .stButton > button[kind="secondary"]:disabled {{
        background: linear-gradient(180deg, rgba(65, 42, 16, 0.55), rgba(43, 28, 11, 0.58));
        color: rgba(248, 230, 189, 0.62);
        border-color: rgba(233, 194, 103, 0.22);
    }}

    .stTextInput label, .stRadio label, .stToggle label, .stSelectbox label, .stSlider label {{
        color: #ffffff !important;
    }}

    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] strong,
    div[data-testid="stAlert"] p,
    div[data-testid="stInfo"] p,
    div[data-testid="stWarning"] p,
    div[data-testid="stError"] p,
    div[data-testid="stSuccess"] p,
    .stWrite p {{
        color: #ffffff !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Memuat data dan indeks, mohon tunggu...")
def get_resources():
    df, tfidf_vectorizer, tfidf_matrix, sbert_embeddings = load_artifacts()
    return df, tfidf_vectorizer, tfidf_matrix, sbert_embeddings


@st.cache_resource(show_spinner="Memuat model semantic jika tersedia...")
def get_sbert_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_selection_dataset():
    csv_path = Path(__file__).parent / "data" / "fra_cleaned.csv"
    if not csv_path.exists():
        return pd.DataFrame(columns=["Brand", "Perfume"])

    return pd.read_csv(csv_path, sep=";", encoding="latin1", engine="python")


def init_session_state():
    defaults = {
        "item_result": None,
        "item_label": "",
        "preference_result": None,
        "preference_query": "",
        "preference_score_col": "hybrid_score",
        "home_page_number": 1,
        "home_gender_filter": "Semua",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def validate_user_query(query: str, tfidf_vectorizer=None):
    text = str(query or "").strip()
    if not text:
        return False, "Masukkan deskripsi aroma terlebih dahulu."

    normalized = preprocess_text(text)
    tokens = normalized.split()
    if len(tokens) < 2:
        return False, "Deskripsi aroma terlalu singkat. Coba tulis minimal dua kata aroma yang kamu sukai."

    short_token_ratio = sum(1 for token in tokens if len(token) < 3) / len(tokens)
    if short_token_ratio >= 0.6:
        return False, "Deskripsi aroma kurang jelas. Coba gunakan kata seperti vanilla, citrus, oud, rose, woody, atau musk."

    # Panjang token & rasio token pendek tidak cukup untuk menyaring input yang
    # sama sekali tidak berhubungan dengan aroma (mis. "tai kucing" tetap lolos
    # dua pengecekan di atas). Di sinilah dicek apakah setidaknya satu kata
    # benar-benar dikenali sebagai istilah aroma di vocabulary TF-IDF, yang
    # dibangun dari notes parfum asli pada dataset.
    if tfidf_vectorizer is not None and not has_vocabulary_overlap(normalized, tfidf_vectorizer):
        return False, (
            "Deskripsi aroma tidak dikenali sistem. Coba gunakan istilah aroma yang lebih umum, "
            "misalnya vanilla, citrus, oud, rose, woody, musk, amber, atau floral."
        )

    return True, ""


def is_placeholder_text(value):
    if value is None or pd.isna(value):
        return True

    text = str(value).strip()
    if text.lower() in PLACEHOLDER_VALUES:
        return True

    if re.fullmatch(r"\d+([.,]\d+)?", text):
        return True

    return False


def clean_display_text(value, fallback="Tidak tersedia"):
    if is_placeholder_text(value):
        return fallback

    return str(value).strip()


def render_hero(active_page: str):
    if active_page == "Beranda":
        eyebrow = "Beranda"
        description = (
            "Eksplorasi parfum dengan rating tertimbang terbaik terlebih dahulu, "
            "lalu pindah ke halaman pencarian saat kamu ingin hasil yang lebih personal."
        )
    else:
        eyebrow = "Pencarian dan Info"
        description = (
            "Cari parfum serupa berdasarkan item, jelaskan preferensi aroma secara bebas, "
            "dan pahami bagaimana sistem rekomendasi ini bekerja."
        )

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-eyebrow">{eyebrow}</div>
            <h1>Sistem Rekomendasi Parfum</h1>
            <p>
                Content-Based dan Hybrid Filtering menggunakan TF-IDF, Sentence-BERT (SBERT),
                dan weighted rating, dibangun dari dataset Fragrantica.com.
                {description}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, note: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_gender_options(dataframe: pd.DataFrame):
    return ["Semua"] + sorted(
        dataframe["Gender"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda series: ~series.str.lower().isin(PLACEHOLDER_VALUES)]
        .unique()
        .tolist()
    )


def build_best_rated_df(dataframe: pd.DataFrame, gender_filter: str):
    candidate_df = dataframe.loc[
        ~dataframe["Perfume"].astype(str).str.strip().str.lower().isin(PLACEHOLDER_VALUES)
        & ~dataframe["Brand"].astype(str).str.strip().str.lower().isin(PLACEHOLDER_VALUES)
    ].copy()

    if gender_filter != "Semua":
        candidate_df = candidate_df[candidate_df["Gender"] == gender_filter].copy()

    return (
        candidate_df.sort_values(["weighted_score", "Rating Value"], ascending=[False, False])
        .reset_index(drop=True)
    )


def build_pagination_items(current_page: int, total_pages: int, window_size: int = 6):
    if total_pages <= window_size + 2:
        return list(range(1, total_pages + 1))

    start_page = max(1, current_page - (window_size // 2))
    end_page = start_page + window_size - 1

    if end_page > total_pages:
        end_page = total_pages
        start_page = max(1, end_page - window_size + 1)

    items = []

    if start_page > 1:
        items.append(1)
        if start_page > 2:
            items.append("left-ellipsis")

    items.extend(range(start_page, end_page + 1))

    if end_page < total_pages:
        if end_page < total_pages - 1:
            items.append("right-ellipsis")
        items.append(total_pages)

    return items


def render_home_pagination(total_items: int, current_page: int, page_size: int):
    total_pages = max(1, math.ceil(total_items / page_size))
    page_start = ((current_page - 1) * page_size) + 1 if total_items else 0
    page_end = min(current_page * page_size, total_items)

    st.markdown(
        f'<p class="section-copy">Menampilkan <strong>{page_start}-{page_end}</strong> dari <strong>{total_items}</strong> parfum. Halaman <strong>{current_page}</strong> dari <strong>{total_pages}</strong>.</p>',
        unsafe_allow_html=True,
    )

    pagination_items = build_pagination_items(current_page, total_pages, window_size=6)
    width_spec = [0.95] + [0.48 if isinstance(item, int) else 0.22 for item in pagination_items] + [0.95]
    cols = st.columns(width_spec)
    next_page = None

    with cols[0]:
        if st.button("Previous", key=f"home_prev_{current_page}", disabled=current_page == 1, use_container_width=True):
            next_page = current_page - 1

    for idx, item in enumerate(pagination_items, start=1):
        with cols[idx]:
            if isinstance(item, int):
                is_active = item == current_page
                if st.button(
                    str(item),
                    key=f"home_page_{item}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                    disabled=is_active,
                ):
                    next_page = item
            else:
                st.markdown(
                    '<div style="text-align:center;color:#f8e9c8;font-weight:600;padding-top:0.45rem;">...</div>',
                    unsafe_allow_html=True,
                )

    with cols[-1]:
        if st.button(
            "Next",
            key=f"home_next_{current_page}",
            disabled=current_page == total_pages,
            use_container_width=True,
        ):
            next_page = current_page + 1

    if next_page is not None and next_page != current_page:
        st.session_state.home_page_number = next_page
        st.rerun()


def get_valid_perfume_options():
    valid_choice_df = load_selection_dataset()[["Brand", "Perfume"]].copy()
    valid_choice_df = valid_choice_df.dropna(subset=["Brand", "Perfume"])
    valid_choice_df["Brand"] = valid_choice_df["Brand"].astype(str).str.strip()
    valid_choice_df["Perfume"] = valid_choice_df["Perfume"].astype(str).str.strip()
    valid_choice_df = valid_choice_df[
        ~valid_choice_df["Brand"].str.lower().isin(PLACEHOLDER_VALUES)
    ]
    valid_choice_df = valid_choice_df[
        ~valid_choice_df["Perfume"].str.lower().isin(PLACEHOLDER_VALUES)
    ]
    valid_choice_df = valid_choice_df.drop_duplicates(subset=["Brand", "Perfume"], keep="first")

    dropdown_labels = []
    label_to_selection = {}
    for _, row in valid_choice_df.iterrows():
        brand = row["Brand"]
        perfume = row["Perfume"]
        label = f"{brand} - {perfume}"
        if label not in label_to_selection:
            dropdown_labels.append(label)
            label_to_selection[label] = {"Brand": brand, "Perfume": perfume}

    sorted_labels = sorted(dropdown_labels)
    return sorted_labels, label_to_selection


def render_perfume_cards(result_df, score_col="similarity_score", n_cols=3, score_label="Kecocokan"):
    if result_df is None or result_df.empty:
        st.info("Parfum tidak ditemukan atau belum ada hasil yang cocok.")
        return

    result_df = result_df.copy()
    result_df = result_df[
        ~result_df["Perfume"].astype(str).str.strip().str.fullmatch(r"\d+([.,]\d+)?")
    ].reset_index(drop=True)

    if result_df.empty:
        st.info("Parfum tidak ditemukan atau belum ada hasil yang cocok.")
        return

    max_score = result_df[score_col].max() if result_df[score_col].max() > 0 else 1
    rows = [result_df.iloc[i:i + n_cols] for i in range(0, len(result_df), n_cols)]

    for chunk in rows:
        cols = st.columns(n_cols)
        for col, (_, row) in zip(cols, chunk.iterrows()):
            with col:
                img_url = get_product_image_url(row.get("url", ""))
                product_url = row.get("url") or "#"
                score_pct = max(0, min(100, round(row[score_col] / max_score * 100)))

                brand_text = clean_display_text(row.get("Brand", ""), fallback="Brand tidak tersedia").replace("-", " ")
                perfume_text = clean_display_text(row.get("Perfume", ""), fallback=brand_text).replace("-", " ")
                gender_text = clean_display_text(row.get("Gender", ""), fallback="Umum")
                accords = [row.get("mainaccord1", ""), row.get("mainaccord2", "")]
                accord_badges = "".join(
                    f'<span class="badge badge-accord">{clean_display_text(accord, fallback="")}</span>'
                    for accord in accords
                    if clean_display_text(accord, fallback="")
                )

                st.markdown(
                    f"""
                    <div class="perfume-card">
                        <a href="{product_url}" target="_blank" rel="noopener noreferrer">
                            <img src="{img_url}" style="width:100%;height:220px;object-fit:cover;"
                                 onerror="this.onerror=null;this.src='{PLACEHOLDER_IMAGE}';" />
                        </a>
                        <div class="card-body">
                            <div class="card-brand">{brand_text}</div>
                            <div class="card-title">{brand_text} {perfume_text}</div>
                            <span class="badge badge-gender">{gender_text}</span>
                            {accord_badges}
                            <div class="score-bar-bg">
                                <div class="score-bar-fill" style="width:{score_pct}%;"></div>
                            </div>
                            <div class="score-label">{score_label}: {score_pct}%</div>
                            <div class="card-link">
                                <a href="{product_url}" target="_blank" rel="noopener noreferrer">Lihat di Fragrantica -></a>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_home_page(dataframe: pd.DataFrame, gender_filter: str):
    if st.session_state.home_gender_filter != gender_filter:
        st.session_state.home_gender_filter = gender_filter
        st.session_state.home_page_number = 1

    best_rated_df = build_best_rated_df(dataframe, gender_filter=gender_filter)
    total_items = len(best_rated_df)
    total_pages = max(1, math.ceil(total_items / HOME_PAGE_SIZE))
    current_page = min(max(1, st.session_state.home_page_number), total_pages)
    st.session_state.home_page_number = current_page

    start_idx = (current_page - 1) * HOME_PAGE_SIZE
    end_idx = start_idx + HOME_PAGE_SIZE
    visible_best_rated_df = best_rated_df.iloc[start_idx:end_idx].reset_index(drop=True)

    total_perfumes = len(dataframe)
    total_brands = dataframe["Brand"].dropna().astype(str).str.strip().nunique()
    active_gender = gender_filter if gender_filter != "Semua" else "Semua kategori"

    st.markdown('<div class="section-title">Parfum dengan Rating Terbaik</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Daftar ini diurutkan berdasarkan weighted score lalu diperkuat oleh rating, sehingga parfum populer dan konsisten lebih menonjol.</p>',
        unsafe_allow_html=True,
    )
    render_perfume_cards(
        visible_best_rated_df,
        score_col="weighted_score",
        n_cols=3,
        score_label="Rating Tertimbang",
    )

    if total_items > HOME_PAGE_SIZE:
        render_home_pagination(total_items, current_page, HOME_PAGE_SIZE)


def render_search_page(
    df: pd.DataFrame,
    tfidf_vectorizer,
    tfidf_matrix,
    sbert_embeddings,
    sbert_model,
    gender_filter: str,
    top_n: int,
):
    st.markdown(
        """
        <div class="page-intro">
            <strong>Pencarian dibagi menjadi dua mode utama.</strong>
            <p>
                Gunakan tab pertama jika sudah punya parfum acuan, atau tab kedua jika ingin
                menulis preferensi aroma secara bebas. Tab terakhir menjelaskan metodologi sistem.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        ["Cari Berdasarkan Parfum", "Berdasarkan Preferensi Aroma", "Tentang Sistem"]
    )

    with tab1:
        st.markdown('<div class="section-title">Temukan Parfum yang Mirip</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-copy">Pilih satu parfum favoritmu, lalu sistem akan mencarikan parfum lain dengan karakteristik yang serupa.</p>',
            unsafe_allow_html=True,
        )

        dropdown_labels, label_to_selection = get_valid_perfume_options()
        with st.form("item_based_form", clear_on_submit=False):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                selected_label = st.selectbox("Pilih parfum", dropdown_labels, index=0)
            with col_b:
                method = st.radio("Metode", ["Sentence-BERT", "TF-IDF"], index=0, horizontal=True)
            submitted = st.form_submit_button("Cari Rekomendasi", type="primary")

        if submitted:
            selected_brand = label_to_selection[selected_label]["Brand"]
            selected_perfume = label_to_selection[selected_label]["Perfume"]
            with st.spinner("Menghitung kemiripan parfum..."):
                if method == "TF-IDF":
                    result = get_similar_perfumes(
                        selected_perfume,
                        tfidf_matrix,
                        df,
                        top_n=top_n,
                        is_sparse=True,
                        brand=selected_brand,
                    )
                else:
                    result = get_similar_perfumes(
                        selected_perfume,
                        sbert_embeddings,
                        df,
                        top_n=top_n,
                        is_sparse=False,
                        brand=selected_brand,
                    )

                if gender_filter != "Semua" and result is not None:
                    result = result[result["Gender"] == gender_filter].reset_index(drop=True)

            st.session_state.item_result = result
            st.session_state.item_label = selected_label

        if st.session_state.item_result is not None:
            st.markdown(
                f'<div class="section-title">Hasil Rekomendasi untuk "{st.session_state.item_label}"</div>',
                unsafe_allow_html=True,
            )
            render_perfume_cards(st.session_state.item_result, score_col="similarity_score")

    with tab2:
        st.markdown('<div class="section-title">Ceritakan Aroma yang Kamu Suka</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-copy">Contoh input: "vanilla amber citrus sweet", "fresh citrus aquatic marine", atau "oud rose spicy oriental".</p>',
            unsafe_allow_html=True,
        )

        with st.form("preference_form", clear_on_submit=False):
            query = st.text_input(
                "Deskripsi preferensi aroma",
                placeholder="mis. vanilla amber citrus sweet",
            )
            use_hybrid = st.toggle("Gunakan hybrid: gabungan kemiripan dan rating", value=True)
            alpha = 0.7
            if use_hybrid:
                alpha = st.slider(
                    "Bobot kemiripan konten (alpha)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                )
            submitted = st.form_submit_button("Cari Rekomendasi", type="primary")

        if submitted:
            is_valid, message = validate_user_query(query, tfidf_vectorizer=tfidf_vectorizer)
            if not is_valid:
                st.warning(message)
                st.session_state.preference_result = None
            else:
                with st.spinner("Mencari parfum yang sesuai..."):
                    if use_hybrid:
                        result = recommend_hybrid(
                            query,
                            tfidf_vectorizer,
                            tfidf_matrix,
                            df,
                            gender_filter=gender_filter,
                            top_n=top_n,
                            alpha=alpha,
                        )
                        score_col = "hybrid_score"
                    else:
                        result = recommend_by_preference(
                            query,
                            sbert_model,
                            sbert_embeddings,
                            df,
                            gender_filter=gender_filter,
                            top_n=top_n,
                            tfidf_vectorizer=tfidf_vectorizer,
                            tfidf_matrix=tfidf_matrix,
                        )
                        score_col = "similarity_score"

                if result is None or len(result) == 0:
                    st.warning(
                        "Tidak ditemukan parfum yang relevan dengan deskripsi tersebut. "
                        "Coba gunakan istilah aroma yang lebih umum, misalnya vanilla, citrus, oud, "
                        "rose, woody, musk, amber, atau floral."
                    )
                    st.session_state.preference_result = None
                    st.session_state.preference_query = query
                else:
                    st.session_state.preference_result = result
                    st.session_state.preference_query = query
                    st.session_state.preference_score_col = score_col

        if st.session_state.preference_result is not None:
            st.markdown(
                f'<div class="section-title">Hasil Rekomendasi untuk "{st.session_state.preference_query}"</div>',
                unsafe_allow_html=True,
            )
            render_perfume_cards(
                st.session_state.preference_result,
                score_col=st.session_state.preference_score_col,
            )

    with tab3:
        st.markdown('<div class="section-title">Tentang Sistem Rekomendasi Ini</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="section-copy">
                Aplikasi ini merupakan frontend dari penelitian skripsi mengenai sistem rekomendasi
                parfum berbasis <strong>Content-Based Filtering</strong> dan <strong>Hybrid Filtering</strong>,
                menggunakan <strong>Fragrantica.com Fragrance Dataset</strong> (gabungan <code>fra_cleaned.csv</code>
                dan <code>fra_perfumes.csv</code>). Katalog yang aktif dicari saat ini berisi
                <strong>{len(df):,} parfum</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            **1. Data & Cleaning**
            - `fra_cleaned.csv` (atribut: notes, brand, gender, rating, dst.) digabungkan dengan
              `fra_perfumes.csv` (deskripsi teks) melalui kolom `url` yang dinormalisasi terlebih dahulu.
            - Baris duplikat (berdasarkan URL ternormalisasi) dan baris tanpa `Perfume`/`Brand` dibuang.
            - Notes `Top`, `Middle`, `Base` digabung menjadi `combined_notes`, lalu dibersihkan (huruf kecil,
              hanya karakter alfabet) untuk dipakai TF-IDF.
            - Kalimat deskripsi yang membocorkan kategori aroma (pola *"is a(n) ... fragrance"*) dinetralkan
              sebelum deskripsi dipakai sebagai input Sentence-BERT, supaya model tidak "curang" mengandalkan
              kalimat yang menyebutkan label kategorinya sendiri.

            **2. Feature Engineering**
            - **TF-IDF**: `stop_words="english"`, `max_features=8000`, `min_df=3`, `max_df=0.85`,
              `ngram_range=(1, 2)`, `sublinear_tf=True` — dilatih pada notes parfum (`combined_notes_clean`).
            - **Sentence-BERT** (`all-MiniLM-L6-v2`): encode deskripsi parfum yang sudah dibersihkan dari
              kebocoran kategori.
            - Reduksi dimensi TF-IDF (TruncatedSVD/LSA) sempat diuji di tahap eksperimen notebook, namun versi
              yang dipakai aplikasi ini adalah TF-IDF & embedding SBERT mentah (tanpa reduksi), sesuai isi
              `features_cache.pkl` yang di-*load* aplikasi.

            **3. Skema Evaluasi (Train-Test Split Level Item)**
            - Karena sistem ini *content-based* tanpa data interaksi user-item, evaluasi dilakukan dengan
              membagi katalog parfum menjadi data **train** (katalog) dan **test** (dianggap parfum yang belum
              pernah dilihat/*unseen items*, dipakai sebagai query) pada 4 rasio: **90:10, 80:20, 70:30, 60:40**.
            - Relevansi didefinisikan sebagai kemiripan minimal 2 *main accord* antara parfum query dan parfum
              kandidat di katalog train.
            - **Catalog produksi**: skema ini murni untuk mengukur generalisasi model secara *offline*. Untuk
              deployment, katalog train dan test dari skema **90:10** digabungkan kembali sehingga seluruh data
              tersedia sebagai katalog pencarian di aplikasi ini (bukan hanya sebagian).

            **4. Mode Rekomendasi**
            - **Item-based** (*"Cari Berdasarkan Parfum"*): mencari parfum lain paling mirip dengan satu parfum
              acuan, via TF-IDF atau Sentence-BERT, dengan opsi filter brand & gender.
            - **Preference-based** (*"Berdasarkan Preferensi Aroma"*): menerima deskripsi aroma bebas dari
              pengguna, dicocokkan lewat Sentence-BERT (fallback otomatis ke TF-IDF bila SBERT tidak tersedia
              karena keterbatasan memori server).
            - **Hybrid**: skor kemiripan konten (TF-IDF) digabung dengan *weighted rating* — rumus mirip
              *Bayesian average* (`Rating Value` & `Rating Count`, kuantil rating 0.70) — lewat parameter
              `alpha` yang bisa diatur pengguna (`skor = alpha × similarity + (1 − alpha) × weighted_rating`).
            - **Validasi input**: deskripsi aroma bebas dicek dulu terhadap vocabulary TF-IDF (istilah aroma
              nyata dari dataset) dan ambang similarity minimum, sehingga input yang tidak relevan sama sekali
              (mis. kata acak/tidak berhubungan dengan aroma) akan ditolak dengan pesan peringatan, bukan
              tetap dipaksakan menampilkan rekomendasi.

            **5. Hasil Evaluasi (skema 90:10, dasar pemilihan model untuk deployment)**
            """
        )

        eval_90_10 = pd.DataFrame(
            [
                {"Model": "TF-IDF", "Precision@10": 0.7493, "nDCG@10": 0.7603, "HitRate@10": 1.0000, "BrandDiversity@10": 0.9177},
                {"Model": "Sentence-BERT", "Precision@10": 0.5573, "nDCG@10": 0.5732, "HitRate@10": 0.9800, "BrandDiversity@10": 0.6107},
                {"Model": "Hybrid (TF-IDF + Rating)", "Precision@10": 0.7520, "nDCG@10": 0.7608, "HitRate@10": 1.0000, "BrandDiversity@10": 0.9167},
                {"Model": "Hybrid (SBERT + Rating)", "Precision@10": 0.5413, "nDCG@10": 0.5571, "HitRate@10": 0.9633, "BrandDiversity@10": 0.6203},
                {"Model": "Random Baseline", "Precision@10": 0.3450, "nDCG@10": 0.3449, "HitRate@10": 0.9400, "BrandDiversity@10": 0.9850},
            ]
        )
        st.dataframe(eval_90_10, hide_index=True, use_container_width=True)

        st.markdown(
            """
            - **Hybrid (TF-IDF + Rating)** sedikit mengungguli TF-IDF murni dan menjadi dasar pemilihan skema
              90:10 untuk deployment. Pendekatan berbasis Sentence-BERT konsisten lebih rendah performanya
              dibanding TF-IDF pada seluruh skema split yang diuji (P@10 sekitar 0.54–0.56 vs 0.71–0.75).
            - **Recall@10 sengaja tidak dijadikan acuan utama** — nilainya kecil (≈0.001) bukan karena model
              buruk, melainkan karena katalog train berjumlah puluhan ribu item sedangkan K hanya 10, sehingga
              Precision@10 dan nDCG@10 lebih representatif untuk skema evaluasi ini.
            - Pencarian nilai `alpha` terbaik (berdasarkan nDCG@10) untuk skema 90:10 menghasilkan **alpha = 0.9**,
              sedangkan slider `alpha` di aplikasi ini memakai nilai default **0.7**. Pengguna tetap bisa
              mengatur slider tersebut secara manual bila ingin mereplikasi hasil eksperimen paling optimal.
            """
        )

    st.caption(
        "Kredit citra: Fragrantica.com Fragrance Dataset. Tampilan gambar produk diambil dari og:image "
        "halaman produk terkait, dengan gambar placeholder bila tidak tersedia."
    )


init_session_state()

try:
    df, tfidf_vectorizer, tfidf_matrix, sbert_embeddings = get_resources()
    sbert_model = get_sbert_model()
    DATA_READY = True
except FileNotFoundError as error:
    DATA_READY = False
    LOAD_ERROR = str(error)

if not DATA_READY:
    render_hero("Beranda")
    st.error(
        "Data belum siap. "
        + LOAD_ERROR
        + "\n\nJalankan `python prepare_data.py` terlebih dahulu (lihat README.md)."
    )
    st.stop()

GENDER_OPTIONS = build_gender_options(df)

with st.sidebar:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] h3,
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] li,
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] strong,
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] b,
        section[data-testid="stSidebar"] * {
            color: #000000 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-panel">
            <h3>Navigasi Halaman</h3>
            <p>Pilih halaman utama yang ingin kamu lihat. Filter pencarian tetap berlaku di seluruh aplikasi.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_page = st.radio(
        "Navigasi",
        ["Beranda", "Pencarian & Info"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Pengaturan Pencarian")
    gender_filter = st.selectbox("Filter gender", GENDER_OPTIONS, index=0)
    top_n = st.slider("Jumlah rekomendasi", min_value=1, max_value=20, value=10, step=1)
    st.markdown("---")
    st.markdown(
        "**Tentang metode:**\n\n"
        "- **TF-IDF**: kemiripan berbasis kata kunci notes parfum\n"
        "- **Sentence-BERT**: kemiripan berbasis makna deskripsi parfum\n"
        "- **Hybrid**: kombinasi kemiripan konten dan rating tertimbang"
    )
    st.markdown("---")
    st.caption("Skripsi Sistem Rekomendasi Parfum | Streamlit App")

render_hero(current_page)

if current_page == "Beranda":
    render_home_page(df, gender_filter=gender_filter)
else:
    render_search_page(
        df=df,
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        sbert_embeddings=sbert_embeddings,
        sbert_model=sbert_model,
        gender_filter=gender_filter,
        top_n=top_n,
    )

st.info(f"Total data parfum yang digunakan: **{len(df):,}** entri.")