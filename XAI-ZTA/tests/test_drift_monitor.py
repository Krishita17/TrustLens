"""Tests for the Concept-Drift Monitor."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.zta.drift_monitor import ConceptDriftMonitor


FEATURES = ["device_trust", "failed_attempts", "anomaly_score"]


@pytest.fixture
def reference():
    rng = np.random.default_rng(0)
    return rng.normal(0, 1, size=(2000, 3))


class TestConceptDriftMonitor:
    def test_detect_before_fit_raises(self):
        monitor = ConceptDriftMonitor(FEATURES)
        with pytest.raises(RuntimeError):
            monitor.detect(np.zeros((10, 3)))

    def test_no_drift_on_same_distribution(self, reference):
        rng = np.random.default_rng(1)
        monitor = ConceptDriftMonitor(FEATURES).fit_reference(reference)
        live = rng.normal(0, 1, size=(1000, 3))
        result = monitor.detect(live)
        assert result["severity"] == "stable"
        assert result["drift_detected"] is False

    def test_major_drift_detected(self, reference):
        rng = np.random.default_rng(2)
        monitor = ConceptDriftMonitor(FEATURES).fit_reference(reference)
        live = rng.normal(0, 1, size=(1000, 3))
        live[:, 1] += 2.5  # shove 'failed_attempts' far off its reference
        result = monitor.detect(live)
        assert result["drift_detected"] is True
        assert result["severity"] == "major"
        assert result["per_feature"][0]["feature"] == "failed_attempts"
        assert "RETRAIN" in result["recommendation"]

    def test_prediction_psi_computed(self, reference):
        rng = np.random.default_rng(3)
        monitor = ConceptDriftMonitor(FEATURES).fit_reference(reference)
        live = rng.normal(0, 1, size=(1000, 3))
        ref_pred = np.zeros(2000, dtype=int)
        ref_pred[:200] = 1                     # 10% deny
        live_pred = np.zeros(1000, dtype=int)
        live_pred[:600] = 1                    # 60% deny -> big shift
        result = monitor.detect(live, ref_pred, live_pred)
        assert result["prediction_psi"] is not None
        assert result["prediction_psi"]["band"] == "major"

    def test_shape_mismatch_raises(self, reference):
        monitor = ConceptDriftMonitor(FEATURES).fit_reference(reference)
        with pytest.raises(ValueError):
            monitor.detect(np.zeros((10, 2)))

    def test_constant_feature_is_safe(self):
        ref = np.zeros((500, 3))          # all-constant reference
        monitor = ConceptDriftMonitor(FEATURES).fit_reference(ref)
        result = monitor.detect(np.zeros((100, 3)))
        assert result["severity"] == "stable"
