"""Tests for TF-IDF cosine blocking."""

import pandas as pd

from er.blocking import generate_candidates, recall_ceiling
from er.config import BlockingConfig


def _make_tables():
    """Small fixture: 4 records in A, 4 in B, 2 true matches."""
    df_a = pd.DataFrame({
        "title": [
            "adobe photoshop cs3",
            "microsoft word 2007",
            "angry birds game",
            "norton antivirus 2023",
        ],
    }, index=[0, 1, 2, 3])

    df_b = pd.DataFrame({
        "title": [
            "photoshop cs3 adobe",
            "microsoft office word 2007 upgrade",
            "candy crush saga mobile",
            "norton antivirus plus 2023 edition",
        ],
    }, index=[10, 11, 12, 13])

    labels = pd.DataFrame({
        "ltable_id": [0, 1, 2, 3],
        "rtable_id": [10, 11, 12, 13],
        "label":     [1,  1,  0,  1],
    })

    return df_a, df_b, labels


class TestGenerateCandidates:
    def test_returns_dataframe_with_correct_columns(self):
        df_a, df_b, _ = _make_tables()
        config = BlockingConfig(threshold=0.01)
        cands = generate_candidates(df_a, df_b, config)
        assert "ltable_id" in cands.columns
        assert "rtable_id" in cands.columns

    def test_low_threshold_keeps_most_pairs(self):
        df_a, df_b, _ = _make_tables()
        config = BlockingConfig(threshold=0.01)
        cands = generate_candidates(df_a, df_b, config)
        # Very low threshold should keep many pairs (but not necessarily all,
        # since zero-overlap pairs have cosine 0.0)
        assert len(cands) > 0

    def test_high_threshold_reduces_pairs(self):
        df_a, df_b, _ = _make_tables()
        low = generate_candidates(df_a, df_b, BlockingConfig(threshold=0.01))
        high = generate_candidates(df_a, df_b, BlockingConfig(threshold=0.5))
        assert len(high) <= len(low)

    def test_true_matches_survive_low_threshold(self):
        df_a, df_b, labels = _make_tables()
        config = BlockingConfig(threshold=0.01)
        cands = generate_candidates(df_a, df_b, config)
        cand_set = set(zip(cands["ltable_id"], cands["rtable_id"]))
        # The adobe photoshop pair should survive
        assert (0, 10) in cand_set

    def test_uses_correct_ids(self):
        df_a, df_b, _ = _make_tables()
        config = BlockingConfig(threshold=0.01)
        cands = generate_candidates(df_a, df_b, config)
        # IDs should come from the index, not positional
        assert all(lid in [0, 1, 2, 3] for lid in cands["ltable_id"])
        assert all(rid in [10, 11, 12, 13] for rid in cands["rtable_id"])

    def test_empty_tables(self):
        df_a = pd.DataFrame({"title": []}, index=pd.Index([], dtype=int))
        df_b = pd.DataFrame({"title": []}, index=pd.Index([], dtype=int))
        config = BlockingConfig(threshold=0.1)
        cands = generate_candidates(df_a, df_b, config)
        assert len(cands) == 0


class TestRecallCeiling:
    def test_perfect_ceiling(self):
        _, _, labels = _make_tables()
        # Candidates contain all true match pairs
        true_matches = labels[labels["label"] == 1]
        cands = true_matches[["ltable_id", "rtable_id"]].copy()
        ceiling = recall_ceiling(cands, labels)
        assert ceiling == 1.0

    def test_zero_ceiling(self):
        _, _, labels = _make_tables()
        # Candidates contain none of the true match pairs
        cands = pd.DataFrame({"ltable_id": [999], "rtable_id": [999]})
        ceiling = recall_ceiling(cands, labels)
        assert ceiling == 0.0

    def test_partial_ceiling(self):
        _, _, labels = _make_tables()
        # Only one of three true matches survives
        cands = pd.DataFrame({"ltable_id": [0], "rtable_id": [10]})
        ceiling = recall_ceiling(cands, labels)
        assert abs(ceiling - 1 / 3) < 1e-6

    def test_no_true_matches(self):
        labels = pd.DataFrame({
            "ltable_id": [0, 1],
            "rtable_id": [10, 11],
            "label": [0, 0],
        })
        cands = pd.DataFrame({"ltable_id": [0], "rtable_id": [10]})
        ceiling = recall_ceiling(cands, labels)
        assert ceiling == 0.0

    def test_with_real_blocker(self):
        """End-to-end: generate candidates then measure ceiling."""
        df_a, df_b, labels = _make_tables()
        config = BlockingConfig(threshold=0.05)
        cands = generate_candidates(df_a, df_b, config)
        ceiling = recall_ceiling(cands, labels)
        # With a low threshold on these similar titles, ceiling should be decent
        assert ceiling > 0.0
