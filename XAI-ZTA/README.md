<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-MLP%20Neural%20Net-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/NIST-SP%20800--207-003366?style=for-the-badge" />
</p>

<h1 align="center">XAI-ZTA</h1>
<h3 align="center">Explainable AI for Zero Trust Continuous Authentication Decisions</h3>

<p align="center">
  <em>When an AI system grants or denies access in a Zero Trust network, can it explain <strong>why</strong> — in a way a human security analyst understands and trusts?</em>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-dashboard">Dashboard</a> •
  <a href="#-ml-models">ML Models</a> •
  <a href="#-xai-methods">XAI Methods</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-citation">Citation</a>
</p>

---

## What is XAI-ZTA?

XAI-ZTA is a complete research system that bridges the gap between **AI-driven security decisions** and **human understanding**. It implements a full Zero Trust Architecture (ZTA) authentication engine with three ML models, three XAI explanation methods, and a professional-grade security analyst dashboard — all aligned with **NIST SP 800-207**.

### Why This Matters

Zero Trust Architecture is rapidly becoming the standard for enterprise security. But when ML models make authentication decisions, security analysts need to understand *why* access was granted or denied. Without explainability:

- Analysts can't verify if the AI is making the right decisions
- Compliance requirements (NIST, HIPAA, GDPR) demand auditable explanations
- False denials frustrate legitimate users without actionable remediation
- False allows go undetected because no one understands what the model missed

XAI-ZTA solves this by generating real-time, human-readable explanations for every single authentication decision.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Krishita17/XAI-ZTA.git
cd XAI-ZTA

# Create virtual environment
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (generates data, trains models, launches dashboard)
python run_pipeline.py
```

The dashboard opens automatically at **http://localhost:8501**.

### Quick Start — Dashboard Only (No Training)

```bash
streamlit run src/dashboard/app.py
```

The dashboard works immediately with synthetic data — no model training required.

---

## Features

### ML Pipeline
| Feature | Description |
|---------|-------------|
| **3 ML Models** | Random Forest, XGBoost, sklearn MLPClassifier Neural Network — trained on 50K synthetic auth events |
| **5-Fold Cross-Validation** | Rigorous model evaluation with stratified splits |
| **Automated Training Pipeline** | One-command pipeline: data generation → feature engineering → training → evaluation |
| **Model Comparison** | Head-to-head accuracy, F1, AUC-ROC, and inference time benchmarks |

### Explainable AI (XAI)
| Feature | Description |
|---------|-------------|
| **SHAP Explanations** | TreeExplainer for RF/XGBoost, KernelExplainer for MLPClassifier |
| **LIME Explanations** | Local linear surrogate models for any classifier |
| **Anchor Rules** | IF-THEN rules with precision and coverage metrics |
| **Counterfactual Analysis** | "What would change the decision?" — actionable remediation |
| **Novel Evaluation Metrics** | Faithfulness, stability, sparsity, comprehensibility, latency |
| **🆕 XAI Consensus Score** | Cross-method agreement (SHAP vs LIME vs Anchor); flags disagreement for human review — `src/xai/consensus.py` |
| **🆕 Robustness Auditing** | Adversarial ε-perturbation test; detects explanation-fragility attacks — `src/xai/robustness.py` |

### Zero Trust Architecture
| Feature | Description |
|---------|-------------|
| **NIST SP 800-207 Compliance** | All 5 ZTA pillars implemented and auditable |
| **Trust Score Engine** | Weighted multi-factor trust computation (device + behavior + network + auth + location) |
| **Policy Engine** | Never trust/always verify, least privilege, continuous validation, micro-segmentation |
| **Compliance Mapping** | Automatic NIST, HIPAA, GDPR tagging for every decision |
| **🆕 Concept-Drift Monitor** | PSI-based distribution-shift detection; recommends retrain when the trust model decays or is drift-poisoned — `src/zta/drift_monitor.py` |

### Dashboard (5 Pages)
| Page | Description |
|------|-------------|
| **Live Monitor** | Real-time auth stream with trust gauges, filtering, and timeline |
| **Explanation Deep Dive** | SHAP vs LIME vs Anchor side-by-side with counterfactuals |
| **Model Comparison** | Performance cards, ROC curves, inference benchmarks |
| **Threat Intelligence** | Risk heatmaps, attack pattern analysis, anomaly detection |
| **Compliance & Audit** | NIST/HIPAA/GDPR reports with CSV/JSON export |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Incoming Auth Request                       │
│         user_id · device · location · auth_method · ...         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ZTA Context Builder                           │
│      Assembles context vector from user + device + network      │
│              src/zta/context_builder.py                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Trust Scorer                                  │
│   trust = 0.30×device + 0.25×behavior + 0.20×network           │
│           + 0.15×auth_method + 0.10×location                   │
│   ALLOW if trust ≥ 0.65, else → ML model for final verdict     │
│              src/zta/trust_scorer.py                            │
└───────────────┬──────────────────────────┬──────────────────────┘
                │                          │
         trust ≥ 0.65                trust < 0.65
                │                          │
                ▼                          ▼
           [ ALLOW ]            ┌──────────────────────┐
                                │   ML Classifier       │
                                │  Random Forest (PRI)  │
                                │  XGBoost / Neural Net │
                                │  src/models/          │
                                └──────────┬───────────┘
                                           │
                                 ┌─────────▼──────────┐
                                 │   XAI Explainer     │
                                 │  SHAP / LIME / Anch │
                                 │  src/xai/           │
                                 └─────────┬───────────┘
                                           │
                                 ┌─────────▼──────────┐
                                 │  Decision Logger    │
                                 │  + Compliance Map   │
                                 │  NIST · HIPAA · GDPR│
                                 └─────────┬───────────┘
                                           │
                                 ┌─────────▼──────────┐
                                 │  Streamlit Dashboard│
                                 │  5-Page Analyst UI  │
                                 └────────────────────┘
```

---

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | C4 context/component views, decision sequence, and the v2.0 explanation-assurance loop (Mermaid) |
| [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) | STRIDE threat model + attack tree for explanation-manipulation & drift-poisoning attacks |
| [`../SECURITY.md`](../SECURITY.md) | Vulnerability disclosure policy and security hardening controls |
| [`../CHANGELOG.md`](../CHANGELOG.md) | Version history (v2.0 assurance layer) |

---

## Project Structure

```
XAI-ZTA/
│
├── README.md                    ← You are here
├── LICENSE                      ← MIT License
├── requirements.txt             ← Python dependencies
├── run_pipeline.py              ← One-command full pipeline runner
├── .env.example                 ← Environment configuration template
├── .gitignore
│
├── data/
│   ├── raw/                     ← Place UNSW-NB15 dataset here (optional)
│   ├── processed/
│   │   └── processed_auth_events.csv    ← 50K rows, 21 features
│   └── synthetic/
│       └── generated_auth_logs.csv      ← 50K rows, 12 features
│
├── src/
│   ├── data/
│   │   ├── loader.py                # Dataset loader with schema validation
│   │   ├── preprocessor.py          # Cleaning, encoding, scaling pipeline
│   │   ├── feature_engineering.py   # ZTA-specific derived features
│   │   └── synthetic_generator.py   # 50K synthetic event generator
│   │
│   ├── models/
│   │   ├── train.py                 # Full training pipeline (3 models)
│   │   ├── evaluate.py              # Accuracy, F1, AUC-ROC, confusion matrix
│   │   ├── random_forest.py         # RF classifier with CV
│   │   ├── xgboost_model.py         # XGBoost with auto class balancing
│   │   ├── neural_net.py            # sklearn MLPClassifier with early stopping
│   │   └── model_utils.py           # joblib save/load helpers
│   │
│   ├── xai/
│   │   ├── shap_explainer.py        # SHAP TreeExplainer + KernelExplainer
│   │   ├── lime_explainer.py        # LIME tabular explainer
│   │   ├── anchor_explainer.py      # Anchor rule-based explanations
│   │   ├── xai_evaluator.py         # Faithfulness / stability / sparsity
│   │   ├── consensus.py             # 🆕 XAI Consensus Score (cross-method agreement)
│   │   └── robustness.py            # 🆕 Adversarial explanation-robustness auditor
│   │
│   ├── zta/
│   │   ├── policy_engine.py         # NIST SP 800-207 policy enforcement
│   │   ├── trust_scorer.py          # Weighted trust score computation
│   │   ├── context_builder.py       # Auth request → context vector
│   │   ├── decision_logger.py       # Audit logging + compliance tags
│   │   └── drift_monitor.py         # 🆕 PSI concept-drift monitor (retrain signal)
│   │
│   └── dashboard/
│       ├── app.py                   # Streamlit entry point (5 pages)
│       ├── components/
│       │   ├── decision_table.py    # Filterable color-coded auth table
│       │   ├── shap_plot.py         # SHAP waterfall + summary plots
│       │   ├── lime_plot.py         # LIME bar chart
│       │   ├── trust_gauge.py       # Real-time trust score gauge
│       │   ├── threat_map.py        # Risk heatmap + radar chart
│       │   ├── model_cards.py       # Model performance cards + ROC
│       │   └── counterfactual.py    # "What-if" remediation panel
│       └── assets/style.css         # Professional cybersecurity theme
│
├── notebooks/                       # 7 Jupyter research notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_shap_analysis.ipynb
│   ├── 05_lime_analysis.ipynb
│   ├── 06_xai_evaluation.ipynb
│   └── 07_user_study_analysis.ipynb
│
├── experiments/
│   ├── configs/                     # Model hyperparameter configs (YAML)
│   ├── results/                     # Trained models + metrics CSVs
│   └── logs/                        # Training logs
│
├── tests/                           # 39 pytest tests (all passing)
│   ├── test_preprocessor.py
│   ├── test_trust_scorer.py
│   ├── test_shap_explainer.py
│   ├── test_lime_explainer.py
│   └── test_policy_engine.py
│
├── paper/                           # IEEE paper materials
│   ├── outline.md
│   ├── references.bib
│   └── figures/
│
└── user_study/                      # Analyst study protocol
    ├── study_protocol.md
    └── results/
```

---

## Installation

### Prerequisites

- **Python 3.9+** (tested with 3.9, 3.10, 3.11, 3.12, 3.14)
- **pip** (comes with Python)
- **Git** (for cloning)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Krishita17/XAI-ZTA.git
cd XAI-ZTA
```

### Step 2 — Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows PowerShell
# venv\Scripts\activate.bat       # Windows CMD
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary><strong>Dependency List (click to expand)</strong></summary>

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24.0 | Numerical computing |
| pandas | ≥2.0.0 | Data manipulation |
| scikit-learn | ≥1.3.0 | ML models, MLPClassifier neural network, and metrics |
| xgboost | ≥1.7.0 | Gradient boosting classifier |
| shap | ≥0.42.0 | SHAP explanations |
| lime | ≥0.2.0.1 | LIME explanations |
| streamlit | ≥1.28.0 | Dashboard framework |
| plotly | ≥5.17.0 | Interactive visualizations |
| joblib | ≥1.3.0 | Model serialization |
| pyyaml | ≥6.0 | Config file parsing |
| python-dotenv | ≥1.0.0 | Environment variables |
| pytest | ≥7.4.0 | Testing framework |
| matplotlib | ≥3.7.0 | Static plots |
| seaborn | ≥0.12.0 | Statistical visualizations |
| imbalanced-learn | ≥0.11.0 | Class imbalance handling |
| reportlab | ≥4.0.0 | PDF report generation |

</details>

### Step 4 — Configure Environment

```bash
cp .env.example .env
# Edit .env if needed — defaults work out of the box
```

---

## Step-by-Step Workflow

### Option A — Full Pipeline (Recommended)

```bash
python run_pipeline.py
```

This runs everything automatically:
1. Generates 50K synthetic authentication events
2. Runs feature engineering (21 features)
3. Trains all 3 ML models with 5-fold CV
4. Runs SHAP/LIME/Anchor explanations
5. Evaluates XAI method quality
6. Launches the dashboard

### Option B — Step by Step

```bash
# 1. Generate synthetic data
python -m src.data.synthetic_generator

# 2. Feature engineering
python -m src.data.feature_engineering

# 3. Train all models
python -m src.models.train

# 4. Generate explanations
python -m src.xai.shap_explainer
python -m src.xai.lime_explainer

# 5. Evaluate XAI methods
python -m src.xai.xai_evaluator

# 6. Launch dashboard
streamlit run src/dashboard/app.py
```

### Option C — Dashboard Only (No Training)

```bash
streamlit run src/dashboard/app.py
```

The dashboard works immediately with synthetic data and simulated models.

---

## Dashboard

The dashboard is a 5-page professional security analyst interface built with Streamlit and Plotly.

### Page 1 — Live Authentication Monitor

- Real-time stream of authentication requests
- Color-coded decision table (green = ALLOW, red = DENY)
- Trust score distribution histogram
- Interactive timeline with anomaly size markers
- Click any request to inspect trust score breakdown

### Page 2 — Explanation Deep Dive

- Select any request (denied or allowed)
- **SHAP tab**: Feature contribution waterfall chart
- **LIME tab**: Local linear approximation weights
- **Anchor tab**: IF-THEN rules with precision/coverage
- **Counterfactual tab**: "What would flip the decision?" with specific remediation steps

### Page 3 — Model Comparison & Benchmarks

- Performance cards for all 3 models (accuracy, F1, AUC-ROC, inference time)
- Interactive ROC curve comparison
- Inference speed benchmark
- XAI method evaluation table

### Page 4 — Threat Intelligence

- Risk heatmap: time-of-day vs location risk
- Attack pattern detection (brute force, credential stuffing, etc.)
- Anomaly analysis scatter plot
- Risk radar: allowed vs denied profiles
- Top risky users table

### Page 5 — Compliance & Audit

- **NIST SP 800-207**: All 5 ZTA principles with compliance status
- **HIPAA**: PHI-adjacent access monitoring
- **GDPR**: Right to explanation, data minimization, pseudonymization
- Full audit log with CSV/JSON export

---

## ML Models

All models are trained on the same 80/20 stratified split with 5-fold cross-validation.

### Random Forest (Primary)

```yaml
n_estimators: 200
max_depth: 15
class_weight: balanced
```

Best for SHAP explanations — `TreeExplainer` is exact and fast (~1ms inference).

### XGBoost (Secondary)

```yaml
learning_rate: 0.05
max_depth: 6
n_estimators: 300
scale_pos_weight: auto  # computed from class imbalance ratio
```

Highest accuracy overall (~2ms inference).

### Neural Network (sklearn MLPClassifier)

```yaml
hidden_layers: [128, 64, 32]
activation: relu
solver: adam
learning_rate: 0.001
max_iter: 100 (with early stopping, patience=10)
```

Uses sklearn's `MLPClassifier` — fast, reliable across all platforms (macOS/Linux/Windows). Uses `KernelExplainer` for SHAP (~0.03ms inference).

### Expected Performance

| Model | Accuracy | F1 Score | AUC-ROC | Inference |
|-------|----------|----------|---------|-----------|
| Random Forest | 0.907 | 0.910 | 0.948 | ~13ms |
| XGBoost | 0.876 | 0.888 | 0.949 | ~0.2ms |
| Neural Network (MLP) | **0.921** | **0.917** | **0.952** | ~0.03ms |

---

## XAI Methods

### SHAP (SHapley Additive exPlanations)

- **TreeExplainer** for Random Forest and XGBoost (exact, polynomial time)
- **KernelExplainer** for Neural Network (model-agnostic, sampling-based)
- Generates per-feature contribution values showing direction and magnitude

### LIME (Local Interpretable Model-agnostic Explanations)

- Creates local linear surrogate models around each decision
- Produces feature weight bar charts ranked by importance
- Model-agnostic — works with any classifier

### Anchor Rules

- Generates IF-THEN rules with precision and coverage guarantees
- Example: `failed_attempts > 3 AND device_trust_score < 0.40 → DENY (precision: 0.97)`
- Falls back to rule-based approximation if `anchor-exp` is not installed

### XAI Evaluation Metrics (Research Contribution)

| Metric | Definition | Target |
|--------|-----------|--------|
| **Faithfulness** | Accuracy drop when top-k features are removed | Higher = more faithful |
| **Stability** | Cosine similarity of explanations across runs | > 0.90 |
| **Sparsity** | Avg features needed per explanation | < 5 features |
| **Comprehensibility** | Analyst Likert ratings (user study) | > 4/7 |
| **Latency** | Time to generate explanation | < 500 ms |

---

## Zero Trust Policy Engine

Aligned with **NIST SP 800-207**, implemented in `src/zta/policy_engine.py`.

### Trust Score Formula

```
trust_score = (0.30 × device_trust_score)
            + (0.25 × user_behavior_score)
            + (0.20 × network_risk_score)
            + (0.15 × auth_method_score)
            + (0.10 × location_score)

ALLOW if trust_score ≥ 0.65
DENY  if trust_score < 0.65
```

### Core ZTA Principles

| Principle | Implementation |
|-----------|---------------|
| Never trust, always verify | Every request re-evaluated independently |
| Least privilege | Role-based access control: deny if role < resource sensitivity |
| Continuous validation | Re-authenticate every 15 minutes based on risk |
| Micro-segmentation | Network segment boundary enforcement |
| Assume breach | Anomaly detection monitors all sessions |

---

## Dataset

### Synthetic Data (Included)

The repository includes **50,000 pre-generated synthetic authentication events** with 21 features. No external download needed.

| File | Rows | Columns | Generator |
|------|------|---------|-----------|
| `data/synthetic/generated_auth_logs.csv` | 50,000 | 12 | `synthetic_generator.py` |
| `data/processed/processed_auth_events.csv` | 50,000 | 21 | `feature_engineering.py` |

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `device_trust_score` | float [0,1] | Device health score |
| `location_risk` | categorical | low / medium / high |
| `time_of_access` | int [0,23] | Hour of day |
| `failed_attempts` | int | Failed logins in last hour |
| `resource_sensitivity` | int [1,5] | Target resource sensitivity |
| `vpn_active` | bool | VPN connection status |
| `patch_level` | int | Days since last security patch |
| `auth_method` | categorical | password / MFA / biometric |
| `anomaly_score` | float [0,1] | Isolation Forest anomaly score |
| `decision` | string | ALLOW / DENY |
| + 11 engineered features | various | Cyclical time, risk tiers, composite scores |

### UNSW-NB15 Real Dataset (Optional)

For research using real network intrusion data, download [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and place files in `data/raw/`.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

**59 tests, all passing** (~3 seconds).

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_preprocessor.py` | 6 | Data cleaning, encoding, scaling, train/test split |
| `test_trust_scorer.py` | 9 | Trust score computation, thresholds, remediation |
| `test_shap_explainer.py` | 6 | SHAP values, JSON serialization, feature ranking |
| `test_lime_explainer.py` | 5 | LIME output, feature weights, text summaries |
| `test_policy_engine.py` | 8 | ZTA principles, trust scoring, full evaluation |
| `test_consensus.py` 🆕 | 9 | XAI Consensus Score, disagreement flagging, bounds |
| `test_robustness.py` 🆕 | 5 | Robustness score, fragility signature, determinism |
| `test_drift_monitor.py` 🆕 | 6 | PSI drift bands, retrain trigger, prediction PSI |

---

## Troubleshooting

<details>
<summary><strong>ModuleNotFoundError: No module named 'src'</strong></summary>

Run all commands from the `XAI-ZTA/` directory:
```bash
cd XAI-ZTA
python -m src.models.train
```
</details>

<details>
<summary><strong>SHAP produces wrong-shaped arrays</strong></summary>

The `_extract_class1_values()` method handles both old and new SHAP API formats automatically. Upgrade SHAP if needed:
```bash
pip install "shap>=0.50.0"
```
</details>

<details>
<summary><strong>anchor-exp fails to install</strong></summary>

```bash
pip install anchor-exp --no-build-isolation
```
If it still fails, the Anchor explainer automatically falls back to rule-based approximation.
</details>

<details>
<summary><strong>Dashboard shows no data</strong></summary>

The dashboard works with synthetic data by default. For trained model data, run the training pipeline first:
```bash
python -m src.models.train
```
</details>

<details>
<summary><strong>Neural network convergence warning</strong></summary>

If you see a `ConvergenceWarning`, increase `max_iter` in `src/models/train.py`. The default (100) with early stopping (patience=10) converges in ~24 iterations on the synthetic dataset.
</details>

---

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| macOS (Intel/Apple Silicon) | Fully supported | Default dev environment |
| Linux (Ubuntu/Debian/Kali) | Fully supported | Use `python3` |
| Windows 10/11 | Fully supported | Use `venv\Scripts\activate` |
| VS Code | Recommended IDE | Built-in Jupyter support |
| Google Colab | Compatible | Upload project files |

---

## Research Questions

1. **RQ1**: How effectively can XAI methods explain ZTA authentication decisions?
2. **RQ2**: Which XAI method best balances explanation quality and real-time performance (< 500 ms)?
3. **RQ3**: Do XAI explanations improve security analyst trust and decision accuracy?

---

## Citation

If you use this project in your research, please cite:

```bibtex
@inproceedings{choksi2026xaizta,
  title     = {{XAI-ZTA}: Explainable and Assured {AI} for Zero Trust
               Continuous Authentication},
  author    = {Choksi, Krishita Sanjay},
  booktitle = {Proceedings of the IEEE Conference on Security and Privacy},
  year      = {2026},
  note      = {https://github.com/Krishita17/XAI-ZTA}
}
```

<details>
<summary><strong>Additional Citations</strong></summary>

```bibtex
@article{moustafa2015unsw,
  title   = {{UNSW-NB15}: A comprehensive data set for network intrusion
             detection systems},
  author  = {Moustafa, Nour and Slay, Jill},
  journal = {MilCIS},
  year    = {2015}
}

@inproceedings{lundberg2017shap,
  title     = {A unified approach to interpreting model predictions},
  author    = {Lundberg, Scott M and Lee, Su-In},
  booktitle = {NeurIPS},
  year      = {2017}
}

@inproceedings{ribeiro2016lime,
  title     = {"Why should I trust you?": Explaining the predictions
               of any classifier},
  author    = {Ribeiro, Marco Tulio and Singh, Sameer and Guestrin, Carlos},
  booktitle = {KDD},
  year      = {2016}
}

@inproceedings{ribeiro2018anchors,
  title     = {Anchors: High-precision model-agnostic explanations},
  author    = {Ribeiro, Marco Tulio and Singh, Sameer and Guestrin, Carlos},
  booktitle = {AAAI},
  year      = {2018}
}
```

</details>

---

## License

MIT License — See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built and maintained by Krishita Sanjay Choksi (<a href="https://github.com/Krishita17">@Krishita17</a>)</strong>
  <br>
  <em>Academic Research Project — IEEE Conference on Security and Privacy</em>
</p>
