"""
Concept-Drift Monitor for XAI-ZTA continuous authentication.

Novel research contribution
---------------------------
"Continuous" authentication implies the *world* is non-stationary: user
behaviour, device fleets, and attacker tactics all shift over time. A trust
model frozen at training time silently decays — yesterday's ALLOW boundary is
wrong today. Worse, an attacker can *induce* drift (slow behavioural poisoning)
to walk the decision boundary toward a malicious profile.

This monitor turns drift into a first-class, auditable ZTA signal. For each
monitored feature it computes the **Population Stability Index (PSI)** between a
reference (training) window and a live window, plus a prediction-distribution
PSI on the model's ALLOW/DENY rate. It emits a per-feature drift table, an
overall drift verdict, and an actionable recommendation (monitor / investigate /
retrain), so the Zero Trust control plane can decide when the model must be
re-verified — an operational realisation of "never trust, always verify"
applied to the model itself.

Pure numpy; deterministic and dependency-light.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ConceptDriftMonitor:
    """Detect distribution shift between a reference and a live feature window."""

    # Standard PSI interpretation bands (Siddiqi, "Credit Risk Scorecards").
    PSI_NO_DRIFT = 0.10        # < 0.10  : insignificant change
    PSI_MODERATE = 0.25        # 0.10-0.25: moderate shift, monitor
    # >= 0.25 : major shift, action required

    def __init__(self, feature_names, n_bins: int = 10, epsilon: float = 1e-6):
        """
        Args:
            feature_names: Ordered names of the features being monitored.
            n_bins: Number of quantile bins used to build distributions.
            epsilon: Small constant to avoid division by / log of zero.
        """
        self.feature_names = list(feature_names)
        self.n_bins = int(n_bins)
        self.epsilon = float(epsilon)
        self._reference = None       # dict: feature_idx -> (bin_edges, ref_pct)

    # ------------------------------------------------------------------ #
    def fit_reference(self, X_reference: np.ndarray) -> "ConceptDriftMonitor":
        """
        Learn the reference distribution (typically the training set).

        Quantile bin edges are frozen from the reference so live data is scored
        against a fixed ruler.
        """
        X = np.asarray(X_reference, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError("X_reference shape does not match feature_names")

        self._reference = {}
        for j in range(X.shape[1]):
            col = X[:, j]
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            edges = np.unique(np.percentile(col, quantiles))
            if len(edges) < 2:  # constant feature — single bin
                edges = np.array([col.min() - 1e-9, col.max() + 1e-9])
            edges[0], edges[-1] = -np.inf, np.inf
            ref_pct = self._bin_percentages(col, edges)
            self._reference[j] = (edges, ref_pct)
        logger.info("Drift reference fitted on %d samples", X.shape[0])
        return self

    def _bin_percentages(self, col: np.ndarray, edges: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(col, bins=edges)
        pct = counts / max(counts.sum(), 1)
        return np.clip(pct, self.epsilon, None)

    @staticmethod
    def _psi(ref_pct: np.ndarray, live_pct: np.ndarray) -> float:
        """Population Stability Index between two binned distributions."""
        return float(np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct)))

    def _band(self, psi: float) -> str:
        if psi < self.PSI_NO_DRIFT:
            return "stable"
        if psi < self.PSI_MODERATE:
            return "moderate"
        return "major"

    # ------------------------------------------------------------------ #
    def detect(self, X_live: np.ndarray,
               ref_predictions: np.ndarray = None,
               live_predictions: np.ndarray = None) -> dict:
        """
        Score a live window against the fitted reference.

        Args:
            X_live: Live feature matrix (same columns as the reference).
            ref_predictions: Optional reference predictions (0/1) for output PSI.
            live_predictions: Optional live predictions (0/1) for output PSI.

        Returns:
            Per-feature PSI table, overall drift verdict, and a recommendation.
        """
        if self._reference is None:
            raise RuntimeError("Call fit_reference() before detect()")

        X = np.asarray(X_live, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError("X_live shape does not match feature_names")

        per_feature = []
        for j, name in enumerate(self.feature_names):
            edges, ref_pct = self._reference[j]
            live_pct = self._bin_percentages(X[:, j], edges)
            psi = self._psi(ref_pct, live_pct)
            per_feature.append({
                "feature": name,
                "psi": round(psi, 4),
                "band": self._band(psi),
            })

        per_feature.sort(key=lambda d: d["psi"], reverse=True)
        max_psi = per_feature[0]["psi"] if per_feature else 0.0
        n_major = sum(1 for f in per_feature if f["band"] == "major")

        prediction_psi = None
        if ref_predictions is not None and live_predictions is not None:
            prediction_psi = self._prediction_psi(ref_predictions, live_predictions)

        overall_band = self._band(max_psi)
        result = {
            "n_features_monitored": len(self.feature_names),
            "max_feature_psi": max_psi,
            "n_major_drift_features": n_major,
            "per_feature": per_feature,
            "prediction_psi": prediction_psi,
            "drift_detected": overall_band != "stable",
            "severity": overall_band,
            "recommendation": self._recommend(overall_band, n_major, prediction_psi),
        }
        return result

    def _prediction_psi(self, ref_pred, live_pred) -> dict:
        ref = np.asarray(ref_pred).ravel()
        live = np.asarray(live_pred).ravel()
        ref_rate = np.clip([1 - ref.mean(), ref.mean()], self.epsilon, None)
        live_rate = np.clip([1 - live.mean(), live.mean()], self.epsilon, None)
        psi = self._psi(ref_rate, live_rate)
        return {
            "psi": round(psi, 4),
            "band": self._band(psi),
            "ref_deny_rate": round(float(ref.mean()), 4),
            "live_deny_rate": round(float(live.mean()), 4),
        }

    def _recommend(self, band: str, n_major: int, prediction_psi) -> str:
        pred_major = prediction_psi and prediction_psi["band"] == "major"
        if band == "major" or pred_major:
            return (
                f"RETRAIN — {n_major} feature(s) show major drift"
                + (" and the model's decision rate has shifted significantly"
                   if pred_major else "")
                + "; re-verify and retrain the trust model before continuing."
            )
        if band == "moderate":
            return "INVESTIGATE — moderate drift detected; increase sampling and review."
        return "MONITOR — distributions are stable; continue routine monitoring."


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    names = ["device_trust", "failed_attempts", "anomaly_score"]

    reference = rng.normal(0, 1, size=(2000, 3))
    monitor = ConceptDriftMonitor(names).fit_reference(reference)

    # Live window with a shifted 'failed_attempts' distribution.
    live = rng.normal(0, 1, size=(1000, 3))
    live[:, 1] += 1.8

    import json
    print(json.dumps(monitor.detect(live), indent=2))
