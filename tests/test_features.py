"""Tests for feature engineering."""

import numpy as np

from er.features import (
    FEATURE_NAMES,
    _cosine_sim,
    _not_null,
    _numeric_tokens,
    _trigrams,
    compute_pair_features,
    embedding_features,
    interaction_features,
    serialize_record,
    string_features,
    structured_features,
)


class TestStringFeatures:
    def test_identical_titles(self):
        feats = string_features("adobe photoshop cs3", "adobe photoshop cs3")
        # All similarity scores should be high (near 1.0)
        for f in feats:
            assert f >= 0.9, f"Expected high similarity, got {f}"

    def test_different_titles(self):
        feats = string_features("microsoft word 2007", "angry birds game")
        jw_norm = feats[1]
        assert jw_norm < 0.5

    def test_empty_strings(self):
        feats = string_features("", "")
        assert len(feats) == 9
        # Length ratio should be 0 (both empty)
        assert feats[8] == 0.0

    def test_feature_count(self):
        feats = string_features("foo", "bar")
        assert len(feats) == 9

    def test_similar_product_names(self):
        feats = string_features(
            "motu digital performer 5 digital audio software",
            "motu digital performer dp5 software",
        )
        token_set = feats[3]  # token_set_ratio
        assert token_set > 0.7

    def test_numeric_token_overlap(self):
        feats = string_features("version 5.0 pro", "version 5.0 standard")
        num_overlap = feats[7]
        assert num_overlap > 0.0  # "5" and "0" should overlap


class TestTrigrams:
    def test_normal(self):
        assert _trigrams("abc") == {"abc"}
        assert _trigrams("abcd") == {"abc", "bcd"}

    def test_short(self):
        assert _trigrams("ab") == {"ab"}
        assert _trigrams("") == set()


class TestNumericTokens:
    def test_extracts_digits(self):
        assert _numeric_tokens("version 5 pro 2007") == {"5", "2007"}

    def test_no_digits(self):
        assert _numeric_tokens("hello world") == set()


class TestEmbeddingFeatures:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        feats = embedding_features(v, v, v, v)
        assert len(feats) == 2
        assert abs(feats[0] - 1.0) < 1e-6
        assert abs(feats[1] - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        feats = embedding_features(a, b, a, b)
        assert abs(feats[0]) < 1e-6

    def test_zero_vector(self):
        z = np.zeros(3)
        v = np.array([1.0, 2.0, 3.0])
        feats = embedding_features(z, v, z, v)
        assert feats[0] == 0.0


class TestCosineSim:
    def test_parallel(self):
        assert abs(_cosine_sim(np.array([1, 0]), np.array([2, 0])) - 1.0) < 1e-6

    def test_antiparallel(self):
        assert abs(_cosine_sim(np.array([1, 0]), np.array([-1, 0])) + 1.0) < 1e-6


class TestStructuredFeatures:
    def test_same_price(self):
        feats = structured_features(19.99, 19.99, "adobe", "adobe")
        price_abs = feats[0]
        mfr_exact = feats[3]
        assert abs(price_abs) < 0.01
        assert mfr_exact == 1.0

    def test_different_price(self):
        feats = structured_features(100.0, 50.0, None, None)
        price_abs = feats[0]
        price_rel = feats[2]
        assert price_abs == 50.0
        assert price_rel == 0.5

    def test_missing_price(self):
        feats = structured_features(None, 50.0, "acme", "acme")
        has_price = feats[5]
        assert has_price == 0.0
        assert np.isnan(feats[0])  # price_abs should be NaN

    def test_missing_manufacturer(self):
        feats = structured_features(10.0, 10.0, "adobe", None)
        has_mfr = feats[6]
        assert has_mfr == 0.0
        assert np.isnan(feats[3])  # mfr_exact should be NaN

    def test_nan_manufacturer(self):
        feats = structured_features(10.0, 10.0, "adobe", float("nan"))
        assert feats[6] == 0.0  # has_manufacturer_both

    def test_feature_count(self):
        feats = structured_features(10.0, 20.0, "a", "b")
        assert len(feats) == 7


class TestNotNull:
    def test_none(self):
        assert _not_null(None) is False

    def test_nan(self):
        assert _not_null(float("nan")) is False

    def test_nan_string(self):
        assert _not_null("nan") is False

    def test_empty_string(self):
        assert _not_null("") is False

    def test_valid(self):
        assert _not_null("adobe") is True
        assert _not_null(19.99) is True
        assert _not_null(0) is True


class TestInteractionFeatures:
    def test_high_sim_price_mismatch(self):
        # jw_norm=0.95 (index 1), price_rel=0.6 (index 2 of struct)
        sf = [0.0, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        stf = [50.0, 0.5, 0.6, 1.0, 1.0, 1.0, 1.0]
        feats = interaction_features(sf, stf)
        assert feats[0] == 1.0  # high_name_sim_price_mismatch

    def test_no_mismatch(self):
        sf = [0.0, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        stf = [1.0, 0.01, 0.05, 1.0, 1.0, 1.0, 1.0]  # price_rel=0.05 < 0.5
        feats = interaction_features(sf, stf)
        assert feats[0] == 0.0

    def test_disagreeing_fields(self):
        sf = [0.0] * 9
        stf = [50.0, 0.5, 0.5, 0.0, 0.3, 1.0, 1.0]  # price disagree + mfr disagree
        feats = interaction_features(sf, stf)
        assert feats[1] == 2.0


class TestSerializeRecord:
    def test_full_record(self):
        assert serialize_record("photoshop", "adobe", 599.99) == "photoshop | adobe | 599.99"

    def test_missing_manufacturer(self):
        assert serialize_record("photoshop", None, 599.99) == "photoshop |  | 599.99"

    def test_all_missing(self):
        assert serialize_record(None, None, None) == " |  | "


class TestComputePairFeatures:
    def test_output_shape(self):
        import pandas as pd

        row_a = pd.Series({"title": "adobe photoshop", "price": 599.99, "manufacturer": "adobe"})
        row_b = pd.Series(
            {"title": "adobe photoshop cs3", "price": 599.99, "manufacturer": "adobe"}
        )
        emb = np.random.randn(64)
        feats = compute_pair_features(row_a, row_b, emb, emb, emb, emb)
        assert feats.shape == (len(FEATURE_NAMES),)

    def test_feature_names_count(self):
        assert len(FEATURE_NAMES) == 20
