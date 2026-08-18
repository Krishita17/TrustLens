"""
XAI Consensus Engine for XAI-ZTA.

Novel research contribution
---------------------------
Prior work (Krishna et al., "The Disagreement Problem in Explainable ML", 2022)
shows that SHAP, LIME, and Anchor frequently *disagree* about which features
drive a decision. For a Zero Trust authentication system this is dangerous: an
analyst who trusts a single explanation may be acting on an artefact of the
explainer rather than on the model's true reasoning.

This module quantifies that disagreement with a single, bounded, auditable
**XAI Consensus Score (XCS)** and flags low-consensus decisions so they can be
escalated to a human reviewer instead of being silently trusted.

The XCS is a weighted blend of three complementary agreement signals:

    XCS = w_rho * mean_pairwise_spearman
        + w_jac * mean_pairwise_jaccard_top_k
        + w_sgn * mean_pairwise_sign_agreement

All three components live in [0, 1] (Spearman is rescaled from [-1, 1]), so the
XCS is itself in [0, 1] where 1.0 == perfect agreement across every method.

The module depends only on numpy, so it adds no heavy runtime dependency and is
fully deterministic for reproducible research.
"""

import logging
from itertools import combinations

import numpy as np

logger = logging.getLogger(__name__)


class XAIConsensusEngine:
    """Measure agreement between multiple XAI explanations of one decision."""

    # Component weights for the composite XAI Consensus Score. Sum to 1.0.
    DEFAULT_WEIGHTS = {
        "rank_correlation": 0.50,   # ordering of feature importances
        "top_k_overlap": 0.30,      # which features make the shortlist
        "sign_agreement": 0.20,     # do methods agree on direction of effect
    }

    # Below this XCS a decision is flagged as "low consensus" for human review.
    DISAGREEMENT_THRESHOLD = 0.60

    def __init__(self, feature_names, weights: dict = None,
                 top_k: int = 5, disagreement_threshold: float = None):
        """
        Initialize the consensus engine.

        Args:
            feature_names: Canonical ordered list of every feature name.
            weights: Optional override of component weights (must sum to 1.0).
            top_k: Size of the shortlist used for the overlap metric.
            disagreement_threshold: XCS below which a case is flagged.
        """
        if not feature_names:
            raise ValueError("feature_names must be a non-empty list")

        self.feature_names = list(feature_names)
        self.index = {name: i for i, name in enumerate(self.feature_names)}
        self.top_k = min(top_k, len(self.feature_names))

        self.weights = dict(weights) if weights else dict(self.DEFAULT_WEIGHTS)
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0):
            raise ValueError(f"weights must sum to 1.0, got {weight_sum:.4f}")

        self.threshold = (
            disagreement_threshold
            if disagreement_threshold is not None
            else self.DISAGREEMENT_THRESHOLD
        )

    # ------------------------------------------------------------------ #
    # Vectorisation
    # ------------------------------------------------------------------ #
    def _to_vector(self, attribution: dict) -> np.ndarray:
        """
        Map a {feature_name: importance} dict onto the canonical feature axis.

        Unmentioned features are treated as zero-importance. This lets us
        compare methods that surface different subsets of features (e.g. Anchor
        only names the features inside its rule).
        """
        vec = np.zeros(len(self.feature_names), dtype=float)
        for name, value in attribution.items():
            if name in self.index:
                vec[self.index[name]] = float(value)
            else:
                logger.debug("Unknown feature '%s' ignored in consensus", name)
        return vec

    # ------------------------------------------------------------------ #
    # Pairwise agreement primitives
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rankdata(values: np.ndarray) -> np.ndarray:
        """Average-rank of values (ties share the mean rank). Pure numpy."""
        order = values.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        # Resolve ties by averaging ranks of equal values.
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        cumulative = np.cumsum(counts)
        start = cumulative - counts
        avg = (start + cumulative - 1) / 2.0
        return avg[inverse]

    def _spearman(self, a: np.ndarray, b: np.ndarray) -> float:
        """Spearman rank correlation of two attribution vectors, in [-1, 1]."""
        ra, rb = self._rankdata(a), self._rankdata(b)
        ra, rb = ra - ra.mean(), rb - rb.mean()
        denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
        if denom == 0:
            return 0.0
        return float((ra * rb).sum() / denom)

    def _top_k_jaccard(self, a: np.ndarray, b: np.ndarray) -> float:
        """Jaccard overlap of the top-k features by absolute importance."""
        top_a = set(np.argsort(-np.abs(a))[: self.top_k])
        top_b = set(np.argsort(-np.abs(b))[: self.top_k])
        union = top_a | top_b
        if not union:
            return 1.0
        return len(top_a & top_b) / len(union)

    def _sign_agreement(self, a: np.ndarray, b: np.ndarray) -> float:
        """Fraction of shared top features whose effect direction agrees."""
        top_a = set(np.argsort(-np.abs(a))[: self.top_k])
        top_b = set(np.argsort(-np.abs(b))[: self.top_k])
        shared = list(top_a & top_b)
        if not shared:
            return 0.0
        agree = sum(1 for i in shared if np.sign(a[i]) == np.sign(b[i]))
        return agree / len(shared)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def compute(self, explanations: dict) -> dict:
        """
        Compute the XAI Consensus Score for one decision.

        Args:
            explanations: Mapping of method name -> {feature_name: importance}.
                e.g. {"shap": {...}, "lime": {...}, "anchor": {...}}.
                At least two methods are required.

        Returns:
            Dictionary with the composite XCS, per-component scores, the full
            pairwise agreement matrix, a low-consensus flag, and a
            human-readable verdict string.
        """
        if len(explanations) < 2:
            raise ValueError("Consensus requires at least two explanation methods")

        methods = list(explanations.keys())
        vectors = {m: self._to_vector(explanations[m]) for m in methods}

        pairwise = []
        rho_scores, jac_scores, sgn_scores = [], [], []
        for m1, m2 in combinations(methods, 2):
            rho = self._spearman(vectors[m1], vectors[m2])
            jac = self._top_k_jaccard(vectors[m1], vectors[m2])
            sgn = self._sign_agreement(vectors[m1], vectors[m2])
            rho_scores.append(rho)
            jac_scores.append(jac)
            sgn_scores.append(sgn)
            pairwise.append({
                "method_a": m1,
                "method_b": m2,
                "spearman": round(rho, 4),
                "top_k_jaccard": round(jac, 4),
                "sign_agreement": round(sgn, 4),
            })

        # Rescale Spearman from [-1, 1] to [0, 1] before blending.
        rho_component = (float(np.mean(rho_scores)) + 1.0) / 2.0
        jac_component = float(np.mean(jac_scores))
        sgn_component = float(np.mean(sgn_scores))

        xcs = (
            self.weights["rank_correlation"] * rho_component
            + self.weights["top_k_overlap"] * jac_component
            + self.weights["sign_agreement"] * sgn_component
        )
        xcs = float(np.clip(xcs, 0.0, 1.0))
        low_consensus = xcs < self.threshold

        return {
            "xai_consensus_score": round(xcs, 4),
            "components": {
                "rank_correlation": round(rho_component, 4),
                "top_k_overlap": round(jac_component, 4),
                "sign_agreement": round(sgn_component, 4),
            },
            "pairwise": pairwise,
            "methods": methods,
            "top_k": self.top_k,
            "low_consensus": low_consensus,
            "verdict": self._verdict(xcs, low_consensus),
        }

    def _verdict(self, xcs: float, low_consensus: bool) -> str:
        """Return a short analyst-facing summary of the consensus result."""
        if low_consensus:
            return (
                f"LOW CONSENSUS (XCS={xcs:.2f}) — explanation methods disagree; "
                f"escalate this decision for human review before acting on it."
            )
        if xcs >= 0.85:
            return f"STRONG CONSENSUS (XCS={xcs:.2f}) — explanations are mutually reinforcing."
        return f"MODERATE CONSENSUS (XCS={xcs:.2f}) — explanations broadly agree."


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    features = [
        "device_trust_score", "failed_attempts", "anomaly_score",
        "auth_method", "location_risk", "patch_level",
    ]
    engine = XAIConsensusEngine(features)

    demo = {
        "shap": {"failed_attempts": 0.42, "anomaly_score": 0.31,
                 "device_trust_score": -0.20, "auth_method": 0.05},
        "lime": {"failed_attempts": 0.38, "anomaly_score": 0.28,
                 "device_trust_score": -0.15, "location_risk": 0.09},
        "anchor": {"failed_attempts": 0.9, "device_trust_score": -0.6},
    }
    import json
    print(json.dumps(engine.compute(demo), indent=2))
