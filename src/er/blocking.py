"""TF-IDF cosine blocking for candidate pair generation.

Generates candidate pairs from the full A x B cartesian product using
TF-IDF cosine similarity on normalized titles. Also measures the blocking
recall ceiling against a set of labeled true matches.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from er.config import BlockingConfig
from er.normalize import normalize

logger = logging.getLogger(__name__)


def generate_candidates(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    config: BlockingConfig,
    text_col: str = "title",
) -> pd.DataFrame:
    """Generate candidate pairs via TF-IDF cosine blocking.

    Vectorizes normalized titles from both tables, computes pairwise cosine
    similarity on the sparse TF-IDF matrix, and keeps pairs above the
    configured threshold.

    Args:
        df_a: Table A with an id column and text_col.
        df_b: Table B with an id column and text_col.
        config: Blocking configuration (method, threshold).
        text_col: Column name containing the text to block on.

    Returns:
        DataFrame with columns [ltable_id, rtable_id] — candidate pairs.
    """
    if len(df_a) == 0 or len(df_b) == 0:
        return pd.DataFrame({"ltable_id": pd.Series(dtype=int), "rtable_id": pd.Series(dtype=int)})

    titles_a = df_a[text_col].fillna("").apply(normalize).tolist()
    titles_b = df_b[text_col].fillna("").apply(normalize).tolist()

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3))
    all_titles = titles_a + titles_b
    tfidf = vectorizer.fit_transform(all_titles)

    tfidf_a = tfidf[: len(titles_a)]
    tfidf_b = tfidf[len(titles_a) :]

    # Sparse cosine similarity: (n_a, n_b) matrix
    # Normalize rows to unit length, then dot product = cosine similarity
    tfidf_a_norm = _l2_normalize_sparse(tfidf_a)
    tfidf_b_norm = _l2_normalize_sparse(tfidf_b)
    sim_matrix = tfidf_a_norm.dot(tfidf_b_norm.T)

    # Extract pairs above threshold
    if not isinstance(sim_matrix, csr_matrix):
        sim_matrix = csr_matrix(sim_matrix)

    ids_a = df_a.index.tolist()
    ids_b = df_b.index.tolist()

    rows, cols = sim_matrix.nonzero()
    scores = np.array(sim_matrix[rows, cols]).flatten()

    mask = scores >= config.threshold
    pair_rows = rows[mask]
    pair_cols = cols[mask]

    candidates = pd.DataFrame({
        "ltable_id": [ids_a[r] for r in pair_rows],
        "rtable_id": [ids_b[c] for c in pair_cols],
    })

    logger.info(
        "Blocking: %d x %d = %d full, %d candidates after threshold %.2f (%.1f%% reduction)",
        len(df_a),
        len(df_b),
        len(df_a) * len(df_b),
        len(candidates),
        config.threshold,
        (1 - len(candidates) / (len(df_a) * len(df_b))) * 100,
    )

    return candidates


def recall_ceiling(candidates: pd.DataFrame, labels: pd.DataFrame) -> float:
    """Compute blocking recall ceiling.

    Measures: |true matches surviving block| / |total true matches|.
    This is computed against the labeled pairs only, not exhaustive A x B.

    Args:
        candidates: DataFrame with [ltable_id, rtable_id] from the blocker.
        labels: DataFrame with [ltable_id, rtable_id, label] — the union of
                train+valid+test labeled pairs.

    Returns:
        Recall ceiling as a float in [0, 1].
    """
    true_matches = labels[labels["label"] == 1]
    if len(true_matches) == 0:
        return 0.0

    candidate_set = set(zip(candidates["ltable_id"], candidates["rtable_id"]))
    true_set = set(zip(true_matches["ltable_id"], true_matches["rtable_id"]))

    survived = len(true_set & candidate_set)
    ceiling = survived / len(true_set)

    logger.info(
        "Blocking recall ceiling: %d / %d true matches survived (%.3f)",
        survived,
        len(true_set),
        ceiling,
    )

    return ceiling


def _l2_normalize_sparse(matrix: csr_matrix) -> csr_matrix:
    """L2-normalize each row of a sparse matrix in place."""
    matrix = csr_matrix(matrix, copy=True, dtype=np.float64)
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1))
    norms = np.asarray(norms).flatten()
    norms[norms == 0] = 1.0  # avoid division by zero
    # Divide each row by its norm
    diag = csr_matrix(np.diag(1.0 / norms))
    return diag.dot(matrix)
