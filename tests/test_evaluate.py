"""Tests for the evaluation harness."""

import json

import numpy as np

from er.evaluate import compute_metrics, evaluate, evaluate_naive_baseline


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])
        metrics = compute_metrics(y_true, y_pred, y_prob, threshold=0.5)
        assert metrics["f1"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        metrics = compute_metrics(y_true, y_pred, y_prob, threshold=0.5)
        assert metrics["f1"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_required_keys(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_prob = np.array([0.2, 0.8, 0.6, 0.4])
        metrics = compute_metrics(y_true, y_pred, y_prob, threshold=0.5)
        expected_keys = {
            "threshold", "precision", "recall", "f1",
            "auc_roc", "average_precision", "brier_score",
            "n_test", "n_positive", "n_predicted_positive",
        }
        assert set(metrics.keys()) == expected_keys

    def test_counts(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_prob = np.array([0.1, 0.6, 0.9, 0.8, 0.3])
        metrics = compute_metrics(y_true, y_pred, y_prob, threshold=0.5)
        assert metrics["n_test"] == 5
        assert metrics["n_positive"] == 3
        assert metrics["n_predicted_positive"] == 3

    def test_metrics_are_json_serializable(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_prob = np.array([0.1, 0.9, 0.6, 0.8])
        metrics = compute_metrics(y_true, y_pred, y_prob, threshold=0.5)
        serialized = json.dumps(metrics)
        assert isinstance(serialized, str)


class TestEvaluate:
    def test_writes_metrics_json(self, tmp_path):
        class FakeModel:
            def predict_proba(self, X):
                return X[:, 0]
            def predict_raw(self, X):
                return X[:, 0]

        y = np.array([0, 0, 1, 1, 0, 1])
        probs = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
        X = probs.reshape(-1, 1)

        result = evaluate(FakeModel(), X, y, threshold=0.5, output_dir=tmp_path)

        assert (tmp_path / "metrics.json").exists()
        with open(tmp_path / "metrics.json") as f:
            saved = json.load(f)
        assert saved["f1"] == result["f1"]

    def test_writes_plots(self, tmp_path):
        class FakeModel:
            def predict_proba(self, X):
                return X[:, 0]
            def predict_raw(self, X):
                return X[:, 0]

        y = np.array([0, 0, 1, 1, 0, 1])
        probs = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
        X = probs.reshape(-1, 1)

        evaluate(FakeModel(), X, y, threshold=0.5, output_dir=tmp_path, raw_model=True)

        assert (tmp_path / "pr_curve.png").exists()
        assert (tmp_path / "confusion_matrix.png").exists()
        assert (tmp_path / "calibration_curve.png").exists()

    def test_returns_dict(self, tmp_path):
        class FakeModel:
            def predict_proba(self, X):
                return X[:, 0]
            def predict_raw(self, X):
                return X[:, 0]

        y = np.array([0, 1, 0, 1])
        X = np.array([0.1, 0.9, 0.2, 0.8]).reshape(-1, 1)

        result = evaluate(FakeModel(), X, y, threshold=0.5, output_dir=tmp_path)
        assert isinstance(result, dict)
        assert "f1" in result


class TestNaiveBaseline:
    def test_returns_metrics_with_baseline_info(self):
        rng = np.random.RandomState(42)
        # Interleave classes so both splits have positives and negatives
        y = np.array([1, 0] * 50)
        scores = np.where(y == 1, rng.normal(0.7, 0.1, 100), rng.normal(0.3, 0.1, 100))
        X = np.column_stack([np.clip(scores, 0, 1), rng.randn(100)])

        metrics = evaluate_naive_baseline(
            X_test=X[50:], y_test=y[50:],
            X_valid=X[:50], y_valid=y[:50],
            feature_idx=0, feature_name="token_sort_ratio",
        )

        assert "baseline_feature" in metrics
        assert metrics["baseline_feature"] == "token_sort_ratio"
        assert metrics["f1"] > 0.0

    def test_useless_feature_low_f1(self):
        rng = np.random.RandomState(42)
        y = np.array([1, 0] * 50)
        X = np.column_stack([
            np.clip(rng.randn(100), 0, 1),
            rng.randn(100),
        ])

        metrics = evaluate_naive_baseline(
            X_test=X[50:], y_test=y[50:],
            X_valid=X[:50], y_valid=y[:50],
            feature_idx=0, feature_name="random_noise",
        )
        # Random feature should have poor F1
        assert metrics["f1"] < 0.8
