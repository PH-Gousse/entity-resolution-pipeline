"""Evaluation harness: metrics, plots, naive baseline comparison."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use("Agg")  # non-interactive backend for CI/headless
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate(
    calibrated_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    output_dir: str | Path,
    feature_names: list[str] | None = None,
    raw_model=None,
) -> dict:
    """Run full evaluation and write metrics + plots.

    Args:
        calibrated_model: Model with predict_proba(X) returning calibrated probabilities.
        X_test: Test feature matrix.
        y_test: Test labels (0/1).
        threshold: Decision threshold (from calibration).
        output_dir: Directory for metrics.json and plot PNGs.
        feature_names: Feature names for importance plot.
        raw_model: Raw LightGBM Booster for feature importance. If None, skips importance plot.

    Returns:
        Metrics dict (also written to output_dir/metrics.json).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probs = calibrated_model.predict_proba(X_test)
    preds = (probs >= threshold).astype(int)

    # Core metrics
    metrics = compute_metrics(y_test, preds, probs, threshold)

    # Naive baseline: best single string feature with validation-tuned threshold
    # (computed externally and passed via metrics if available)

    # Write metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics written to %s", metrics_path)

    # Generate plots
    plot_pr_curve(y_test, probs, output_dir / "pr_curve.png")
    plot_confusion_matrix(y_test, preds, output_dir / "confusion_matrix.png")

    if raw_model is not None:
        raw_scores = calibrated_model.predict_raw(X_test)
        plot_calibration_curve(y_test, raw_scores, probs, output_dir / "calibration_curve.png")

    if raw_model is not None and feature_names is not None:
        plot_feature_importance(raw_model, feature_names, output_dir / "feature_importance.png")

    logger.info(
        "Evaluation complete: F1=%.4f, P=%.4f, R=%.4f, AUC-PR=%.4f",
        metrics["f1"], metrics["precision"], metrics["recall"], metrics["average_precision"],
    )

    return metrics


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    """Compute all evaluation metrics."""
    # Clip probabilities for metrics that require [0, 1] range
    y_prob_clipped = np.clip(y_prob, 0.0, 1.0)

    # Handle single-class edge case for AUC metrics
    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        auc_roc = float("nan")
        ap = float("nan")
    else:
        auc_roc = float(roc_auc_score(y_true, y_prob))
        ap = float(average_precision_score(y_true, y_prob))

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": auc_roc,
        "average_precision": ap,
        "brier_score": float(brier_score_loss(y_true, y_prob_clipped)),
        "n_test": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_predicted_positive": int(y_pred.sum()),
    }


def evaluate_naive_baseline(
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    feature_idx: int,
    feature_name: str,
) -> dict:
    """Evaluate naive baseline: single feature with val-tuned threshold.

    Args:
        X_test: Test features.
        y_test: Test labels.
        X_valid: Validation features (for threshold tuning).
        y_valid: Validation labels.
        feature_idx: Column index of the feature to use as the sole predictor.
        feature_name: Name of the feature for reporting.

    Returns:
        Metrics dict for the baseline.
    """
    from er.calibrate import select_threshold_max_f1

    valid_scores = X_valid[:, feature_idx]
    threshold = select_threshold_max_f1(valid_scores, y_valid)

    test_scores = X_test[:, feature_idx]
    preds = (test_scores >= threshold).astype(int)

    metrics = compute_metrics(y_test, preds, test_scores, threshold)
    metrics["baseline_feature"] = feature_name
    metrics["baseline_type"] = "single_feature"

    logger.info(
        "Naive baseline (%s): F1=%.4f, threshold=%.3f",
        feature_name, metrics["f1"], threshold,
    )

    return metrics


# ── Plots ──────────────────────────────────────────────────────────────────

def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, path: Path) -> None:
    """Precision-Recall curve with average precision."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, linewidth=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("PR curve saved to %s", path)


def plot_calibration_curve(
    y_true: np.ndarray,
    raw_scores: np.ndarray,
    calibrated_probs: np.ndarray,
    path: Path,
) -> None:
    """Calibration curve: before and after isotonic regression."""
    brier_before = brier_score_loss(y_true, np.clip(raw_scores, 0, 1))
    brier_after = brier_score_loss(y_true, calibrated_probs)

    fig, ax = plt.subplots(figsize=(7, 5))

    for scores, label, color in [
        (np.clip(raw_scores, 0, 1), f"Before (Brier={brier_before:.4f})", "tab:blue"),
        (calibrated_probs, f"After isotonic (Brier={brier_after:.4f})", "tab:orange"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_true, scores, n_bins=10, strategy="uniform")
        ax.plot(mean_pred, frac_pos, "s-", color=color, label=label)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Calibration curve saved to %s", path)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-match", "Match"])
    ax.set_yticklabels(["Non-match", "Match"])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", path)


def plot_feature_importance(model, feature_names: list[str], path: Path) -> None:
    """LightGBM feature importance (gain-based)."""
    importance = model.feature_importance(importance_type="gain")
    sorted_idx = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(8, max(5, len(feature_names) * 0.3)))
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        importance[sorted_idx],
        color="tab:green",
    )
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Feature importance saved to %s", path)
