"""Feature engineering for entity resolution pairs.

Four feature families:
  1. String similarity (~8 features)
  2. Embedding cosine (~2 features)
  3. Structured / numeric (~6 features)
  4. Interaction (~2 features)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from er.normalize import normalize

# ── Feature name registry (column order in the output matrix) ──────────────

FEATURE_NAMES: list[str] = [
    # String
    "jaro_winkler_raw",
    "jaro_winkler_norm",
    "token_sort_ratio",
    "token_set_ratio",
    "partial_ratio",
    "levenshtein_norm",
    "trigram_jaccard",
    "numeric_token_overlap",
    "length_ratio",
    # Embedding
    "emb_name_cosine",
    "emb_record_cosine",
    # Structured
    "price_abs_diff",
    "price_log_ratio",
    "price_rel_diff",
    "manufacturer_exact",
    "manufacturer_jw",
    "has_price_both",
    "has_manufacturer_both",
    # Interaction
    "high_name_sim_price_mismatch",
    "disagreeing_fields",
]


# ── String features ────────────────────────────────────────────────────────

def _trigrams(s: str) -> set[str]:
    if len(s) < 3:
        return {s} if s else set()
    return {s[i : i + 3] for i in range(len(s) - 2)}


def _numeric_tokens(s: str) -> set[str]:
    return {t for t in s.split() if t.isdigit()}


def string_features(title_a: str, title_b: str) -> list[float]:
    """Compute 9 string-similarity features between two titles."""
    norm_a = normalize(title_a)
    norm_b = normalize(title_b)

    # Jaro-Winkler on raw and normalized
    jw_raw = JaroWinkler.similarity(title_a or "", title_b or "")
    jw_norm = JaroWinkler.similarity(norm_a, norm_b)

    # Fuzzy ratios (0-100 scale, normalize to 0-1)
    tsr = fuzz.token_sort_ratio(norm_a, norm_b) / 100.0
    tsetr = fuzz.token_set_ratio(norm_a, norm_b) / 100.0
    pr = fuzz.partial_ratio(norm_a, norm_b) / 100.0

    # Normalized Levenshtein (0 = identical, 1 = completely different) -> similarity
    max_len = max(len(norm_a), len(norm_b), 1)
    lev_dist = Levenshtein.distance(norm_a, norm_b)
    lev_norm = 1.0 - lev_dist / max_len

    # Trigram Jaccard
    tri_a = _trigrams(norm_a)
    tri_b = _trigrams(norm_b)
    if tri_a or tri_b:
        tri_jac = len(tri_a & tri_b) / len(tri_a | tri_b)
    else:
        tri_jac = 0.0

    # Numeric token overlap (Jaccard of digit tokens)
    num_a = _numeric_tokens(norm_a)
    num_b = _numeric_tokens(norm_b)
    if num_a or num_b:
        num_overlap = len(num_a & num_b) / len(num_a | num_b)
    else:
        num_overlap = 0.0

    # Length ratio (shorter / longer)
    len_a, len_b = len(norm_a), len(norm_b)
    length_ratio = min(len_a, len_b) / max(len_a, len_b) if max(len_a, len_b) > 0 else 0.0

    return [jw_raw, jw_norm, tsr, tsetr, pr, lev_norm, tri_jac, num_overlap, length_ratio]


# ── Embedding features ─────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def embedding_features(
    emb_name_a: np.ndarray,
    emb_name_b: np.ndarray,
    emb_record_a: np.ndarray,
    emb_record_b: np.ndarray,
) -> list[float]:
    """Compute 2 embedding cosine similarity features."""
    return [
        _cosine_sim(emb_name_a, emb_name_b),
        _cosine_sim(emb_record_a, emb_record_b),
    ]


# ── Structured features ───────────────────────────────────────────────────

def structured_features(
    price_a: float | None,
    price_b: float | None,
    manufacturer_a: str | None,
    manufacturer_b: str | None,
) -> list[float]:
    """Compute 6 structured features from price and manufacturer fields."""
    # Price features — NaN propagates naturally for LightGBM
    has_price = 1.0 if _not_null(price_a) and _not_null(price_b) else 0.0

    if has_price:
        pa, pb = float(price_a), float(price_b)
        price_abs = abs(pa - pb)
        eps = 1e-6
        price_log = abs(np.log(max(pa, eps) / max(pb, eps)))
        price_rel = price_abs / max(pa, pb, eps)
    else:
        price_abs = np.nan
        price_log = np.nan
        price_rel = np.nan

    # Manufacturer features
    has_mfr = 1.0 if _not_null(manufacturer_a) and _not_null(manufacturer_b) else 0.0

    if has_mfr:
        ma = normalize(str(manufacturer_a))
        mb = normalize(str(manufacturer_b))
        mfr_exact = 1.0 if ma == mb else 0.0
        mfr_jw = JaroWinkler.similarity(ma, mb)
    else:
        mfr_exact = np.nan
        mfr_jw = np.nan

    return [price_abs, price_log, price_rel, mfr_exact, mfr_jw, has_price, has_mfr]


def _not_null(val) -> bool:
    if val is None:
        return False
    if isinstance(val, float) and np.isnan(val):
        return False
    if isinstance(val, str) and val.strip().lower() in ("", "nan"):
        return False
    return True


# ── Interaction features ───────────────────────────────────────────────────

def interaction_features(string_feats: list[float], struct_feats: list[float]) -> list[float]:
    """Compute 2 interaction features from string and structured features.

    Args:
        string_feats: output of string_features() (9 values)
        struct_feats: output of structured_features() (7 values)
    """
    jw_norm = string_feats[1]  # jaro_winkler_norm
    price_rel = struct_feats[2]  # price_rel_diff

    # High name similarity but price mismatch
    high_sim_price_mismatch = 1.0 if (jw_norm > 0.9 and _safe_gt(price_rel, 0.5)) else 0.0

    # Count of disagreeing structured fields
    mfr_exact = struct_feats[3]
    has_price = struct_feats[5]
    has_mfr = struct_feats[6]
    disagreeing = 0
    if has_price and _safe_gt(price_rel, 0.3):
        disagreeing += 1
    if has_mfr and _not_null(mfr_exact) and mfr_exact < 1.0:
        disagreeing += 1

    return [high_sim_price_mismatch, float(disagreeing)]


def _safe_gt(val, threshold: float) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    return float(val) > threshold


# ── Embedding precomputation ──────────────────────────────────────────────

def serialize_record(title: str, manufacturer, price) -> str:
    """Serialize a record for embedding: 'title | manufacturer | price'."""
    parts = [
        str(title) if _not_null(title) else "",
        str(manufacturer) if _not_null(manufacturer) else "",
        str(price) if _not_null(price) else "",
    ]
    return " | ".join(parts)


def precompute_embeddings(
    df: pd.DataFrame,
    model,
    text_col: str = "title",
    structured_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute name and record embeddings for all rows in a table.

    Returns:
        (name_embeddings, record_embeddings) — both shape (len(df), embed_dim)
    """
    structured_cols = structured_cols or ["manufacturer", "price"]

    names = df[text_col].fillna("").tolist()
    records = [
        serialize_record(row[text_col], *[row.get(c) for c in structured_cols])
        for _, row in df.iterrows()
    ]

    name_embs = model.encode(names, show_progress_bar=False, convert_to_numpy=True)
    record_embs = model.encode(records, show_progress_bar=False, convert_to_numpy=True)
    return name_embs, record_embs


# ── Pair feature assembly ─────────────────────────────────────────────────

def compute_pair_features(
    row_a: pd.Series,
    row_b: pd.Series,
    emb_name_a: np.ndarray,
    emb_name_b: np.ndarray,
    emb_record_a: np.ndarray,
    emb_record_b: np.ndarray,
) -> np.ndarray:
    """Compute the full feature vector for a single pair."""
    sf = string_features(str(row_a.get("title", "")), str(row_b.get("title", "")))
    ef = embedding_features(emb_name_a, emb_name_b, emb_record_a, emb_record_b)
    stf = structured_features(
        row_a.get("price"), row_b.get("price"),
        row_a.get("manufacturer"), row_b.get("manufacturer"),
    )
    intf = interaction_features(sf, stf)
    return np.array(sf + ef + stf + intf, dtype=np.float64)


def compute_features(
    pairs: pd.DataFrame,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    embeddings_a: tuple[np.ndarray, np.ndarray],
    embeddings_b: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Compute feature matrix for all candidate pairs.

    Args:
        pairs: DataFrame with columns [ltable_id, rtable_id]
        df_a: Table A, indexed by id column
        df_b: Table B, indexed by id column
        embeddings_a: (name_embs, record_embs) for table A, row-aligned with df_a
        embeddings_b: (name_embs, record_embs) for table B, row-aligned with df_b

    Returns:
        np.ndarray of shape (len(pairs), len(FEATURE_NAMES))
    """
    name_embs_a, rec_embs_a = embeddings_a
    name_embs_b, rec_embs_b = embeddings_b

    # Build index maps: id -> positional index in the dataframe
    idx_a = {row_id: i for i, row_id in enumerate(df_a.index)}
    idx_b = {row_id: i for i, row_id in enumerate(df_b.index)}

    n_pairs = len(pairs)
    n_feats = len(FEATURE_NAMES)
    X = np.empty((n_pairs, n_feats), dtype=np.float64)

    for i, (_, pair) in enumerate(pairs.iterrows()):
        aid = pair["ltable_id"]
        bid = pair["rtable_id"]

        pos_a = idx_a[aid]
        pos_b = idx_b[bid]

        X[i] = compute_pair_features(
            df_a.iloc[pos_a],
            df_b.iloc[pos_b],
            name_embs_a[pos_a],
            name_embs_b[pos_b],
            rec_embs_a[pos_a],
            rec_embs_b[pos_b],
        )

    return X
