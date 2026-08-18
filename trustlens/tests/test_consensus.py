"""Tests for the XAI Consensus Engine."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.xai.consensus import XAIConsensusEngine


@pytest.fixture
def features():
    return ["device_trust_score", "failed_attempts", "anomaly_score",
            "auth_method", "location_risk", "patch_level"]


@pytest.fixture
def engine(features):
    return XAIConsensusEngine(features, top_k=3)


class TestXAIConsensusEngine:
    def test_requires_features(self):
        with pytest.raises(ValueError):
            XAIConsensusEngine([])

    def test_weights_must_sum_to_one(self, features):
        with pytest.raises(ValueError):
            XAIConsensusEngine(features, weights={"rank_correlation": 0.5,
                                                  "top_k_overlap": 0.2,
                                                  "sign_agreement": 0.2})

    def test_requires_two_methods(self, engine):
        with pytest.raises(ValueError):
            engine.compute({"shap": {"failed_attempts": 0.5}})

    def test_identical_explanations_give_perfect_consensus(self, engine):
        expl = {"failed_attempts": 0.5, "anomaly_score": 0.3,
                "device_trust_score": -0.2}
        result = engine.compute({"shap": expl, "lime": dict(expl)})
        assert result["xai_consensus_score"] == pytest.approx(1.0, abs=1e-6)
        assert result["low_consensus"] is False

    def test_opposite_explanations_flag_low_consensus(self, engine):
        a = {"failed_attempts": 0.9, "anomaly_score": 0.8, "device_trust_score": 0.7}
        b = {"patch_level": 0.9, "location_risk": 0.8, "auth_method": 0.7}
        result = engine.compute({"shap": a, "lime": b})
        assert result["low_consensus"] is True
        assert "LOW CONSENSUS" in result["verdict"]

    def test_score_is_bounded(self, engine):
        a = {"failed_attempts": 0.4, "anomaly_score": -0.3}
        b = {"failed_attempts": 0.2, "device_trust_score": 0.5}
        c = {"anomaly_score": 0.9}
        result = engine.compute({"shap": a, "lime": b, "anchor": c})
        assert 0.0 <= result["xai_consensus_score"] <= 1.0
        for comp in result["components"].values():
            assert 0.0 <= comp <= 1.0

    def test_pairwise_count(self, engine):
        a = {"failed_attempts": 0.4}
        result = engine.compute({"shap": a, "lime": a, "anchor": a})
        # 3 methods -> 3 unique pairs
        assert len(result["pairwise"]) == 3

    def test_unknown_features_ignored(self, engine):
        result = engine.compute({
            "shap": {"failed_attempts": 0.5, "not_a_feature": 9.9},
            "lime": {"failed_attempts": 0.5},
        })
        assert 0.0 <= result["xai_consensus_score"] <= 1.0

    def test_rankdata_handles_ties(self, engine):
        ranks = engine._rankdata(np.array([1.0, 1.0, 2.0]))
        assert ranks[0] == ranks[1]  # tied values share a rank
