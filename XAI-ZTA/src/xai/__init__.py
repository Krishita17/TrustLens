"""Explainable AI modules: SHAP, LIME, Anchor, cross-method consensus, and
adversarial-robustness auditing for authentication explanations."""

from src.xai.consensus import XAIConsensusEngine
from src.xai.robustness import ExplanationRobustnessAuditor

__all__ = ["XAIConsensusEngine", "ExplanationRobustnessAuditor"]
