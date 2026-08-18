"""
Explanation Robustness Auditor for XAI-ZTA.

Novel research contribution
---------------------------
Ghorbani et al. ("Interpretation of Neural Networks is Fragile", AAAI 2019)
showed that explanations can be manipulated by imperceptible input
perturbations *without* changing the model's prediction. In a Zero Trust
setting an adversary who can nudge telemetry within noise tolerance could make
a malicious login *look* benign to the analyst while the ALLOW/DENY verdict is
unchanged — an attack on the human-in-the-loop, not on the classifier.

This module audits that risk. For a given decision it samples the L-infinity
epsilon-ball around the input, re-computes the explanation at each sample, and
measures how much the explanation moves relative to how much the input moved.
It reports:

    * explanation_sensitivity  — mean L2 drift of normalised attributions
    * local_lipschitz          — worst-case ||dExpl|| / ||dInput|| in the ball
    * rank_instability         — 1 - mean top-k Jaccard vs. the original ranking
    * prediction_stability     — fraction of samples with the SAME verdict
    * robustness_score         — bounded [0,1] summary (1.0 == robust)

A decision that is *prediction-stable but explanation-unstable* is the exact
signature of an explanation-fragility attack and is flagged accordingly.

Pure numpy; deterministic given a seed.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ExplanationRobustnessAuditor:
    """Stress-test an explanation against bounded adversarial perturbations."""

    def __init__(self, predict_fn, explain_fn, feature_names,
                 epsilon: float = 0.05, n_samples: int = 100,
                 top_k: int = 5, random_state: int = 42):
        """
        Args:
            predict_fn: Callable(X_2d) -> predicted class labels (1d array).
            explain_fn: Callable(x_1d) -> attribution vector aligned with
                feature_names (1d array, one weight per feature).
            feature_names: Ordered feature names.
            epsilon: L-infinity perturbation budget (in scaled feature units).
            n_samples: Number of perturbations sampled inside the ball.
            top_k: Shortlist size for rank-instability.
            random_state: Seed for reproducibility.
        """
        self.predict_fn = predict_fn
        self.explain_fn = explain_fn
        self.feature_names = list(feature_names)
        self.epsilon = float(epsilon)
        self.n_samples = int(n_samples)
        self.top_k = min(top_k, len(self.feature_names))
        self.rng = np.random.default_rng(random_state)

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        """Unit-L2 normalise an attribution vector (zero-safe)."""
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _top_k_set(self, vec: np.ndarray) -> set:
        return set(np.argsort(-np.abs(vec))[: self.top_k])

    def audit(self, instance: np.ndarray) -> dict:
        """
        Run the robustness audit for a single authentication instance.

        Args:
            instance: 1d feature vector (already scaled the way the model expects).

        Returns:
            Dictionary of robustness metrics plus an attack-signature flag.
        """
        instance = np.asarray(instance, dtype=float).ravel()
        base_expl = np.asarray(self.explain_fn(instance), dtype=float).ravel()
        base_norm = self._normalise(base_expl)
        base_pred = int(np.asarray(self.predict_fn(instance.reshape(1, -1))).ravel()[0])
        base_top = self._top_k_set(base_expl)

        expl_drifts = []          # L2 distance between normalised explanations
        lipschitz_ratios = []     # explanation drift / input drift
        jaccard_scores = []       # top-k overlap with the original
        same_pred = 0

        for _ in range(self.n_samples):
            delta = self.rng.uniform(-self.epsilon, self.epsilon, size=instance.shape)
            perturbed = instance + delta
            input_drift = np.linalg.norm(delta)
            if input_drift == 0:
                continue

            pert_expl = np.asarray(self.explain_fn(perturbed), dtype=float).ravel()
            expl_drift = np.linalg.norm(self._normalise(pert_expl) - base_norm)

            expl_drifts.append(expl_drift)
            lipschitz_ratios.append(expl_drift / input_drift)

            top = self._top_k_set(pert_expl)
            union = base_top | top
            jaccard_scores.append(len(base_top & top) / len(union) if union else 1.0)

            pert_pred = int(np.asarray(self.predict_fn(perturbed.reshape(1, -1))).ravel()[0])
            same_pred += int(pert_pred == base_pred)

        n = max(len(expl_drifts), 1)
        sensitivity = float(np.mean(expl_drifts)) if expl_drifts else 0.0
        local_lipschitz = float(np.max(lipschitz_ratios)) if lipschitz_ratios else 0.0
        rank_instability = 1.0 - (float(np.mean(jaccard_scores)) if jaccard_scores else 1.0)
        prediction_stability = same_pred / n

        # Bounded robustness score: explanation drift of sqrt(2) is the maximum
        # possible between two unit vectors, so sensitivity/sqrt(2) in [0,1].
        robustness_score = float(np.clip(1.0 - (sensitivity / np.sqrt(2)), 0.0, 1.0))

        # Attack signature: verdict barely moves but the explanation swings hard.
        fragility_attack = (prediction_stability >= 0.95) and (rank_instability >= 0.40)

        return {
            "epsilon": self.epsilon,
            "n_samples": self.n_samples,
            "base_prediction": base_pred,
            "explanation_sensitivity": round(sensitivity, 4),
            "local_lipschitz": round(local_lipschitz, 4),
            "rank_instability": round(rank_instability, 4),
            "prediction_stability": round(prediction_stability, 4),
            "robustness_score": round(robustness_score, 4),
            "fragility_attack_suspected": bool(fragility_attack),
            "verdict": self._verdict(robustness_score, fragility_attack),
        }

    @staticmethod
    def _verdict(score: float, fragility_attack: bool) -> str:
        if fragility_attack:
            return (
                "FRAGILE EXPLANATION — verdict is stable but the explanation is "
                "highly sensitive to imperceptible input noise; do not present "
                "this explanation to an analyst without a stability warning."
            )
        if score >= 0.85:
            return "ROBUST — explanation is stable under bounded perturbation."
        if score >= 0.60:
            return "MODERATE — explanation drifts under perturbation; interpret with care."
        return "UNSTABLE — explanation is not reliable for this decision."


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Toy demo: a linear model whose explanation is the (fixed) weight vector,
    # so it is robust by construction.
    w = np.array([0.8, -0.5, 0.3, 0.1])
    names = ["failed_attempts", "device_trust", "anomaly_score", "auth_method"]

    def predict(X):
        return (X @ w > 0).astype(int)

    def explain(x):
        return w * x  # simple gradient-times-input attribution

    auditor = ExplanationRobustnessAuditor(predict, explain, names, epsilon=0.05)
    import json
    print(json.dumps(auditor.audit(np.array([0.6, 0.2, 0.4, 1.0])), indent=2))
