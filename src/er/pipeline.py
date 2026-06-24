"""Pipeline orchestrator — chains blocking, features, training, calibration, evaluation."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from er.blocking import generate_candidates, recall_ceiling
from er.calibrate import calibrate
from er.config import Config
from er.evaluate import evaluate, evaluate_naive_baseline
from er.features import FEATURE_NAMES, compute_features, precompute_embeddings
from er.train import save_model, train_model

logger = logging.getLogger(__name__)


def load_data(config: Config) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Load tables and splits, setting the id column as index."""
    id_col = config.dataset.id_col

    df_a = pd.read_csv(config.dataset.table_a).set_index(id_col)
    df_b = pd.read_csv(config.dataset.table_b).set_index(id_col)
    train = pd.read_csv(config.dataset.train)
    valid = pd.read_csv(config.dataset.valid)
    test = pd.read_csv(config.dataset.test)

    logger.info(
        "Data loaded: A=%d, B=%d, train=%d, valid=%d, test=%d",
        len(df_a), len(df_b), len(train), len(valid), len(test),
    )
    return df_a, df_b, train, valid, test


def run_blocking(
    config: Config,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Run the blocker and measure recall ceiling."""
    t0 = time.time()
    candidates = generate_candidates(df_a, df_b, config.blocking, config.dataset.text_col)

    # Recall ceiling against all labeled pairs
    all_labels = pd.concat([train, valid, test], ignore_index=True)
    ceiling = recall_ceiling(candidates, all_labels)

    logger.info("Blocking: %.1fs, %d candidates, recall ceiling=%.3f",
                time.time() - t0, len(candidates), ceiling)
    return candidates, ceiling


def run_features(
    config: Config,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute features for train/valid/test splits.

    Returns:
        (X_train, y_train, X_valid, y_valid, X_test, y_test)
    """
    artifacts_dir = Path(config.pipeline.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    emb_path_a = artifacts_dir / "embeddings_a.npz"
    emb_path_b = artifacts_dir / "embeddings_b.npz"

    if emb_path_a.exists() and emb_path_b.exists():
        logger.info("Loading cached embeddings")
        data_a = np.load(emb_path_a)
        embs_a = (data_a["name"], data_a["record"])
        data_b = np.load(emb_path_b)
        embs_b = (data_b["name"], data_b["record"])
    else:
        logger.info("Precomputing embeddings (first run)...")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(config.pipeline.embedding_model)

        t0 = time.time()
        embs_a = precompute_embeddings(
            df_a, model, config.dataset.text_col, config.dataset.structured_cols,
        )
        embs_b = precompute_embeddings(
            df_b, model, config.dataset.text_col, config.dataset.structured_cols,
        )
        logger.info("Embeddings computed in %.1fs", time.time() - t0)

        np.savez(emb_path_a, name=embs_a[0], record=embs_a[1])
        np.savez(emb_path_b, name=embs_b[0], record=embs_b[1])
        logger.info("Embeddings cached to %s", artifacts_dir)

    t0 = time.time()
    X_train = compute_features(train, df_a, df_b, embs_a, embs_b)
    y_train = train["label"].values.astype(np.float64)

    X_valid = compute_features(valid, df_a, df_b, embs_a, embs_b)
    y_valid = valid["label"].values.astype(np.float64)

    X_test = compute_features(test, df_a, df_b, embs_a, embs_b)
    y_test = test["label"].values.astype(np.float64)

    logger.info("Features computed in %.1fs: %d features x (%d train, %d valid, %d test)",
                time.time() - t0, X_train.shape[1], len(X_train), len(X_valid), len(X_test))

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def run_train(
    config: Config,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
):
    """Train model, calibrate, and save artifacts.

    Returns:
        (calibrated_model, threshold, raw_booster)
    """
    artifacts_dir = Path(config.pipeline.artifacts_dir)

    t0 = time.time()
    booster = train_model(X_train, y_train, X_valid, y_valid,
                          config.model, config.pipeline.random_seed)
    logger.info("Training: %.1fs", time.time() - t0)

    save_model(booster, artifacts_dir / "model.txt")

    t0 = time.time()
    calibrated_model, threshold = calibrate(booster, X_valid, y_valid)
    logger.info("Calibration: %.1fs, threshold=%.3f", time.time() - t0, threshold)

    return calibrated_model, threshold, booster


def run_evaluate(
    config: Config,
    calibrated_model,
    threshold: float,
    booster,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    blocking_ceiling: float | None = None,
) -> dict:
    """Run evaluation and naive baseline comparison."""
    output_dir = Path(config.pipeline.output_dir)

    # Main model evaluation
    metrics = evaluate(
        calibrated_model, X_test, y_test, threshold, output_dir,
        feature_names=FEATURE_NAMES, raw_model=booster,
    )

    # Naive baseline: token_sort_ratio (index 2 in FEATURE_NAMES)
    tsr_idx = FEATURE_NAMES.index("token_sort_ratio")
    baseline = evaluate_naive_baseline(
        X_test, y_test, X_valid, y_valid,
        feature_idx=tsr_idx, feature_name="token_sort_ratio",
    )

    # Combine results
    results = {
        "model": metrics,
        "baseline": baseline,
    }
    if blocking_ceiling is not None:
        results["blocking_recall_ceiling"] = blocking_ceiling

    # Write combined results
    import json
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    logger.info("Full results written to %s", results_path)

    return results


def run_pipeline(config: Config) -> dict:
    """Run the full pipeline end-to-end."""
    logger.info("Starting pipeline: %s", config.dataset.name)
    t_start = time.time()

    np.random.seed(config.pipeline.random_seed)

    # 1. Load data
    df_a, df_b, train, valid, test = load_data(config)

    # 2. Blocking + recall ceiling
    _, blocking_ceiling = run_blocking(config, df_a, df_b, train, valid, test)

    # 3. Features
    X_train, y_train, X_valid, y_valid, X_test, y_test = run_features(
        config, df_a, df_b, train, valid, test,
    )

    # 4. Train + calibrate
    calibrated_model, threshold, booster = run_train(
        config, X_train, y_train, X_valid, y_valid,
    )

    # 5. Evaluate
    results = run_evaluate(
        config, calibrated_model, threshold, booster,
        X_test, y_test, X_valid, y_valid, blocking_ceiling,
    )

    elapsed = time.time() - t_start
    logger.info("Pipeline complete in %.1fs", elapsed)

    # Print summary
    m = results["model"]
    b = results["baseline"]
    print(f"\n{'=' * 50}")
    print(f"  Dataset: {config.dataset.name}")
    print(f"  Blocking recall ceiling: {results.get('blocking_recall_ceiling', 'N/A'):.3f}")
    print(f"  Model F1:    {m['f1']:.4f}  (P={m['precision']:.4f}, R={m['recall']:.4f})")
    print(f"  Baseline F1: {b['f1']:.4f}  ({b['baseline_feature']})")
    print(f"  AUC-PR:      {m['average_precision']:.4f}")
    print(f"  Brier score: {m['brier_score']:.4f}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'=' * 50}\n")

    return results


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
