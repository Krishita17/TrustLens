"""Tests for the Explanation Robustness Auditor."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.xai.robustness import ExplanationRobustnessAuditor


FEATURES = ["failed_attempts", "device_trust", "anomaly_score", "auth_method"]
WEIGHTS = np.array([0.8, -0.5, 0.3, 0.1])


def linear_predict(X):
    X = np.atleast_2d(X)
    return (X @ WEIGHTS > 0).astype(int)


def stable_explain(x):
    """Gradient-times-input: smooth, so explanations are robust."""
    return WEIGHTS * np.asarray(x).ravel()


def make_fragile_explain(seed=0):
    """Explanation dominated by noise -> unstable under perturbation."""
    rng = np.random.default_rng(seed)

    def explain(x):
        return WEIGHTS * np.asarray(x).ravel() + rng.normal(0, 5, size=len(WEIGHTS))

    return explain


class TestRobustnessAuditor:
    def test_returns_expected_keys(self):
        auditor = ExplanationRobustnessAuditor(
            linear_predict, stable_explain, FEATURES, epsilon=0.05, n_samples=50)
        result = auditor.audit(np.array([0.6, 0.2, 0.4, 1.0]))
        for key in ["robustness_score", "explanation_sensitivity",
                    "local_lipschitz", "rank_instability",
                    "prediction_stability", "fragility_attack_suspected"]:
            assert key in result

    def test_stable_explanation_is_robust(self):
        auditor = ExplanationRobustnessAuditor(
            linear_predict, stable_explain, FEATURES, epsilon=0.03, n_samples=100)
        result = auditor.audit(np.array([0.6, 0.2, 0.4, 1.0]))
        assert result["robustness_score"] >= 0.85
        assert result["fragility_attack_suspected"] is False

    def test_noisy_explanation_is_less_robust(self):
        stable = ExplanationRobustnessAuditor(
            linear_predict, stable_explain, FEATURES, epsilon=0.05, n_samples=100)
        fragile = ExplanationRobustnessAuditor(
            linear_predict, make_fragile_explain(), FEATURES,
            epsilon=0.05, n_samples=100)
        x = np.array([0.6, 0.2, 0.4, 1.0])
        assert fragile.audit(x)["robustness_score"] < stable.audit(x)["robustness_score"]

    def test_score_is_bounded(self):
        auditor = ExplanationRobustnessAuditor(
            linear_predict, make_fragile_explain(1), FEATURES,
            epsilon=0.1, n_samples=80)
        result = auditor.audit(np.array([0.5, 0.5, 0.5, 0.5]))
        assert 0.0 <= result["robustness_score"] <= 1.0
        assert 0.0 <= result["prediction_stability"] <= 1.0
        assert 0.0 <= result["rank_instability"] <= 1.0

    def test_deterministic_with_seed(self):
        a = ExplanationRobustnessAuditor(
            linear_predict, stable_explain, FEATURES, n_samples=40, random_state=7)
        b = ExplanationRobustnessAuditor(
            linear_predict, stable_explain, FEATURES, n_samples=40, random_state=7)
        x = np.array([0.6, 0.2, 0.4, 1.0])
        assert a.audit(x)["robustness_score"] == b.audit(x)["robustness_score"]
