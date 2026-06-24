"""LightGBM model training with early stopping."""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np

from er.config import ModelConfig

logger = logging.getLogger(__name__)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    config: ModelConfig,
    random_seed: int = 42,
) -> lgb.Booster:
    """Train a LightGBM binary classifier with early stopping.

    Args:
        X_train: Training feature matrix (n_train, n_features).
        y_train: Training labels (0/1).
        X_valid: Validation feature matrix.
        y_valid: Validation labels.
        config: Model configuration (params, early_stopping_rounds).
        random_seed: Random seed for reproducibility.

    Returns:
        Trained LightGBM Booster.
    """
    params = dict(config.params)
    params.setdefault("objective", "binary")
    params.setdefault("metric", "binary_logloss")
    params.setdefault("verbosity", -1)
    params.setdefault("seed", random_seed)

    # Auto-compute scale_pos_weight if not set or null
    if params.get("scale_pos_weight") is None:
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        if n_pos > 0:
            params["scale_pos_weight"] = n_neg / n_pos
            logger.info(
                "Auto scale_pos_weight: %d neg / %d pos = %.2f",
                n_neg, n_pos, params["scale_pos_weight"],
            )
        else:
            raise ValueError(
                "No positive examples in training data. "
                "Check that the training split contains both labels."
            )

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    callbacks = [
        lgb.early_stopping(config.early_stopping_rounds),
        lgb.log_evaluation(period=100),
    ]

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=params.pop("n_estimators", 1000),
        valid_sets=[valid_data],
        valid_names=["valid"],
        callbacks=callbacks,
    )

    logger.info("Training complete: %d iterations, best score %.4f",
                booster.best_iteration, booster.best_score["valid"]["binary_logloss"])

    return booster


def save_model(booster: lgb.Booster, path: str | Path) -> None:
    """Save a trained model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))
    logger.info("Model saved to %s", path)


def load_model(path: str | Path) -> lgb.Booster:
    """Load a trained model from disk."""
    return lgb.Booster(model_file=str(path))
