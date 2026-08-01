import re
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_artifacts():
    """Load dataframe bersih & feature cache (tfidf, sbert) hasil prepare_data.py."""
    df_path = os.path.join(DATA_DIR, "df_clean.pkl")
    features_path = os.path.join(DATA_DIR, "features_cache.pkl")

    if not os.path.exists(df_path) or not os.path.exists(features_path):
        raise FileNotFoundError(
            "Artefak data tidak ditemukan. Jalankan `python prepare_data.py` "
            "terlebih dahulu untuk membuat data/df_clean.pkl dan "
            "data/features_cache.pkl."
        )

    df = pd.read_pickle(df_path)
    with open(features_path, "rb") as f:
        cache = pickle.load(f)

    tfidf_vectorizer = cache["tfidf_vectorizer"]
    tfidf_matrix = cache["tfidf_matrix"]
    sbert_embeddings = cache["sbert_embeddings"]

    return df, tfidf_vectorizer, tfidf_matrix, sbert_embeddings


DISPLAY_COLS = ["Perfume", "Brand", "Gender", "mainaccord1", "mainaccord2", "url", "Rating Value"]

# Skor similarity minimum agar sebuah hasil dianggap benar-benar relevan.
# Kalau similarity tertinggi dari seluruh kandidat berada di bawah ambang ini,
# artinya query tidak punya kemiripan nyata dengan parfum manapun di dataset
# (mis. input random/tidak relevan seperti "tai kucing"), sehingga fungsi
# rekomendasi berbasis preferensi akan mengembalikan None alih-alih tetap
# memaksakan top_n hasil dengan skor mendekati nol.
MIN_PREFERENCE_SIMILARITY = 0.08


def has_vocabulary_overlap(query_clean: str, tfidf_vectorizer) -> bool:
    """Cek apakah token hasil preprocessing query punya overlap dengan vocabulary TF-IDF.

    Vocabulary TF-IDF dibangun dari notes/deskripsi parfum asli, sehingga ini
    dipakai sebagai sinyal awal murah untuk menolak input yang sama sekali
    tidak berhubungan dengan istilah aroma (mis. kata acak, bahasa lain yang
    tidak relevan, atau typo total) sebelum masuk ke perhitungan similarity.
    """
    if tfidf_vectorizer is None:
        return True

    tokens = [t for t in query_clean.split() if t]
    if not tokens:
        return False

    vocabulary = getattr(tfidf_vectorizer, "vocabulary_", None)
    if not vocabulary:
        return True  # fail-open kalau vectorizer belum ter-fit dengan benar

    return any(token in vocabulary for token in tokens)


def get_similar_perfumes(
    perfume_name,
    feature_matrix,
    df,
    top_n=10,
    is_sparse=True,
    brand=None,
    restrict_candidates_to_brand=False,
):
    """Rekomendasi item-based: parfum lain yang mirip dengan 1 parfum acuan.

    `Perfume` tidak unik di seluruh dataset (banyak brand memakai nama yang sama,
    mis. "amber queen" ada di beberapa brand berbeda). Karena itu, jika `brand`
    diberikan, ia SELALU dipakai untuk mengidentifikasi baris parfum acuan yang
    benar (Perfume + Brand) -- terlepas dari apakah `restrict_candidates_to_brand`
    aktif. Tanpa ini, `perfume_name` yang ambigu bisa membuat fungsi diam-diam
    mengambil baris dari brand yang salah sebagai acuan, sehingga rekomendasi
    yang dihasilkan tidak relevan dengan parfum yang sebenarnya dipilih user.

    `restrict_candidates_to_brand` mengontrol hal yang berbeda: apakah kandidat
    hasil rekomendasi dibatasi hanya ke brand yang sama dengan acuan, atau
    mencakup seluruh katalog (default).
    """
    if brand is not None:
        brand_norm = str(brand).strip().lower()
        ref_matches = df.index[
            (df["Perfume"] == perfume_name)
            & (df["Brand"].astype(str).str.strip().str.lower() == brand_norm)
        ]
    else:
        brand_norm = None
        ref_matches = df.index[df["Perfume"] == perfume_name]

    if len(ref_matches) == 0:
        return None

    idx = ref_matches[0]

    candidate_df = df.copy()
    if restrict_candidates_to_brand and brand_norm is not None:
        candidate_df = candidate_df[
            candidate_df["Brand"].astype(str).str.strip().str.lower() == brand_norm
        ].copy()

    candidate_idx = candidate_df.index.tolist()
    query_vec = feature_matrix[idx] if is_sparse else feature_matrix[idx].reshape(1, -1)
    sims = cosine_similarity(query_vec, feature_matrix[candidate_idx]).flatten()

    top_idx = sims.argsort()[::-1]
    top_idx = [candidate_idx[i] for i in top_idx if candidate_idx[i] != idx][:top_n]

    cols = [c for c in DISPLAY_COLS if c in df.columns]
    result = df.iloc[top_idx][cols].copy()
    result["similarity_score"] = [sims[candidate_idx.index(i)] for i in top_idx]
    return result.reset_index(drop=True)


def recommend_by_preference(
    favorite_notes,
    sbert_model,
    sbert_embeddings,
    df,
    gender_filter=None,
    top_n=10,
    tfidf_vectorizer=None,
    tfidf_matrix=None,
):
    """Rekomendasi berbasis deskripsi bebas preferensi aroma (mis. 'vanilla amber citrus').

    Jika model SBERT tidak tersedia karena keterbatasan memori, fungsi ini otomatis fall back
    ke TF-IDF agar rekomendasi tetap dapat dijalankan.
    """
    query_clean = preprocess_text(favorite_notes)

    # Query yang sama sekali tidak mengandung istilah aroma (mis. kata acak/tidak
    # relevan) ditolak lebih awal, tanpa perlu menjalankan similarity SBERT/TF-IDF.
    if tfidf_vectorizer is not None and not has_vocabulary_overlap(query_clean, tfidf_vectorizer):
        return None

    if sbert_model is not None and sbert_embeddings is not None:
        query_vec = sbert_model.encode([query_clean])
        sims = cosine_similarity(query_vec, sbert_embeddings).flatten()
    elif tfidf_vectorizer is not None and tfidf_matrix is not None:
        query_vec = tfidf_vectorizer.transform([query_clean])
        sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    else:
        raise ValueError("Tidak ada metode similarity yang tersedia untuk rekomendasi preferensi.")

    if sims.size == 0 or sims.max() < MIN_PREFERENCE_SIMILARITY:
        return None

    candidate_df = df.copy()
    candidate_df["similarity_score"] = sims

    if gender_filter and gender_filter != "Semua":
        candidate_df = candidate_df[candidate_df["Gender"] == gender_filter]

    cols = [c for c in DISPLAY_COLS if c in df.columns] + ["similarity_score"]
    result = candidate_df.sort_values("similarity_score", ascending=False).head(top_n)
    return result[cols].reset_index(drop=True)


def recommend_hybrid(favorite_notes, tfidf_vectorizer, tfidf_matrix, df, gender_filter=None, top_n=10, alpha=0.7):
    """Rekomendasi hybrid: kemiripan konten (TF-IDF) dikombinasikan dengan weighted rating."""
    query_clean = preprocess_text(favorite_notes)

    # Query yang sama sekali tidak mengandung istilah aroma (mis. kata acak/tidak
    # relevan) ditolak lebih awal, tanpa perlu menjalankan similarity TF-IDF.
    if not has_vocabulary_overlap(query_clean, tfidf_vectorizer):
        return None

    query_vec = tfidf_vectorizer.transform([query_clean])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()

    if sims.size == 0 or sims.max() < MIN_PREFERENCE_SIMILARITY:
        return None

    candidate_df = df.copy()
    candidate_df["similarity_score"] = sims

    if gender_filter and gender_filter != "Semua":
        candidate_df = candidate_df[candidate_df["Gender"] == gender_filter]

    sim_norm = (candidate_df["similarity_score"] - candidate_df["similarity_score"].min()) / (
        candidate_df["similarity_score"].max() - candidate_df["similarity_score"].min() + 1e-9
    )
    rating_norm = (candidate_df["weighted_score"] - candidate_df["weighted_score"].min()) / (
        candidate_df["weighted_score"].max() - candidate_df["weighted_score"].min() + 1e-9
    )
    candidate_df["hybrid_score"] = alpha * sim_norm + (1 - alpha) * rating_norm

    cols = [c for c in DISPLAY_COLS if c in df.columns] + ["similarity_score", "weighted_score", "hybrid_score"]
    result = candidate_df.sort_values("hybrid_score", ascending=False).head(top_n)
    return result[cols].reset_index(drop=True)