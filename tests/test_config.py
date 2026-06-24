"""Tests for config loading."""

import pytest

from er.config import load_config


def test_load_amazon_google_config():
    config = load_config("configs/amazon_google.yaml")
    assert config.dataset.name == "amazon_google"
    assert config.dataset.id_col == "id"
    assert config.dataset.text_col == "title"
    assert "manufacturer" in config.dataset.structured_cols
    assert config.model.type == "lightgbm"
    assert config.blocking.method == "tfidf_cosine"
    assert config.pipeline.random_seed == 42


def test_load_missing_config():
    with pytest.raises(FileNotFoundError):
        load_config("configs/nonexistent.yaml")
