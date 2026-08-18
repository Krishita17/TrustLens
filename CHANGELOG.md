# Changelog

All notable changes to XAI-ZTA are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/) and this
project adheres to [Semantic Versioning](https://semver.org/).

## [2.0.0] — 2026-08-18

### Added — Explanation-Assurance Layer (novel contribution)
- **XAI Consensus Engine** (`src/xai/consensus.py`) — a bounded **XAI Consensus
  Score (XCS)** measuring agreement between SHAP, LIME, and Anchor via rank
  correlation, top-k Jaccard overlap, and sign agreement. Flags low-consensus
  decisions for human review.
- **Explanation Robustness Auditor** (`src/xai/robustness.py`) — audits
  explanation stability under L∞ ε-bounded adversarial perturbation; detects the
  *fragility-attack* signature (stable verdict + unstable explanation) and
  estimates a local-Lipschitz constant.
- **Concept-Drift Monitor** (`src/zta/drift_monitor.py`) — Population Stability
  Index (PSI) per feature and on the decision rate, with monitor / investigate /
  **retrain** recommendations for the continuous-authentication setting.
- 20 new unit tests (`test_consensus.py`, `test_robustness.py`,
  `test_drift_monitor.py`) — total suite now **59 tests**.

### Added — Security & Supply Chain
- `SECURITY.md` — coordinated vulnerability-disclosure policy.
- `docs/THREAT_MODEL.md` — full **STRIDE** threat model with a Mermaid attack
  tree covering explanation-manipulation, model-extraction-via-explanations, and
  behavioural drift-poisoning threats, each mapped to a shipped detector.
- CI pipeline (`.github/workflows/ci.yml`) — multi-version pytest matrix plus
  **Bandit** SAST and **pip-audit** dependency scanning.
- **CodeQL** analysis (`.github/workflows/codeql.yml`, security-and-quality).
- **Dependabot** (`.github/dependabot.yml`) for pip and GitHub Actions.
- Least-privilege `permissions:` blocks on all workflows.

### Added — Documentation & Diagrams
- `docs/ARCHITECTURE.md` — C4 system-context, component, sequence, and
  assurance-loop diagrams (Mermaid).
- Mermaid architecture diagram in the main README; new comparison tables for the
  assurance metrics.

### Changed
- Author/citation metadata updated to **Krishita Sanjay Choksi** (sole author).
- READMEs refreshed with v2.0 features, badges, and topic tags.

## [1.0.0] — 2026-06
- Initial release: 3 ML models, 3 XAI methods, NIST SP 800-207 policy engine,
  5-page Streamlit dashboard, 39 tests.
