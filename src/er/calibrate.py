"""Probability calibration and threshold selection."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, precision_recall_curve

logger = logging.getLogger(__name__)


class CalibratedModel:
    """Wraps a raw scorer with isotonic calibration.

    Takes raw scores from model.predict() and maps them to calibrated
    probabilities via isotonic regression.
    """

    def __init__(self, model, isotonic: IsotonicRegression):
        self.model = model
        self.isotonic = isotonic

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Raw uncalibrated scores from the underlying model."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated probabilities."""
        raw = self.predict_raw(X)
        return self.isotonic.transform(raw)


def calibrate(
    model,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
) -> tuple[CalibratedModel, float]:
    """Fit isotonic calibration and select the max-F1 threshold.

    Args:
        model: Trained model with a .predict(X) method returning scores.
        X_valid: Validation feature matrix.
        y_valid: Validation labels (0/1).

    Returns:
        (calibrated_model, threshold) where threshold maximizes F1 on
        the calibrated validation scores.
    """
    raw_scores = model.predict(X_valid)

    # Fit isotonic regression: maps raw scores -> calibrated probabilities
    isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    isotonic.fit(raw_scores, y_valid)

    calibrated = CalibratedModel(model, isotonic)
    cal_probs = calibrated.predict_proba(X_valid)

    # Select threshold that maximizes F1
    threshold = select_threshold_max_f1(cal_probs, y_valid)

    preds = (cal_probs >= threshold).astype(int)
    f1 = f1_score(y_valid, preds)

    logger.info("Calibration complete: threshold=%.3f, valid F1=%.4f", threshold, f1)

    return calibrated, threshold


def select_threshold_max_f1(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    """Find the threshold that maximizes F1 score.

    Uses the precision-recall curve to efficiently sweep thresholds.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)

    # precision_recall_curve returns n+1 precision/recall values but n thresholds
    # The last precision/recall pair (1.0, 0.0) has no corresponding threshold
    precision = precision[:-1]
    recall = recall[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores, nan=0.0)

    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])
