# XAI-ZTA Architecture

This document describes the architecture of XAI-ZTA using layered diagrams:
a **system context**, a **component view**, the **decision sequence**, and the
**v2.0 explanation-assurance loop** (consensus, robustness, and drift).

All diagrams are Mermaid and render natively on GitHub.

---

## 1. System Context (C4 Level 1)

```mermaid
flowchart TB
    analyst([Security Analyst])
    subject([Subject / Device])
    admin([ML / Security Engineer])

    subject -->|authentication request| ZTA
    ZTA -->|ALLOW / DENY + explanation| subject
    analyst -->|reviews decisions, explanations, alerts| ZTA
    admin -->|configures policy, retrains on drift alert| ZTA

    subgraph ZTA[XAI-ZTA System]
        engine[Zero Trust Decision Engine]
        xai[Explainability + Assurance Layer]
        dash[Analyst Dashboard]
        engine --- xai --- dash
    end

    ZTA -->|NIST / HIPAA / GDPR audit records| compliance[(Compliance Store)]
```

---

## 2. Component View (C4 Level 2)

```mermaid
flowchart LR
    subgraph DATA[Data Layer]
        gen[synthetic_generator]
        fe[feature_engineering]
        pre[preprocessor]
    end

    subgraph ZTAL[Zero Trust Layer]
        ctx[context_builder]
        trust[trust_scorer]
        policy[policy_engine]
        drift[drift_monitor ⭐]
        loggr[decision_logger]
    end

    subgraph ML[Model Layer]
        rf[random_forest]
        xgb[xgboost_model]
        nn[neural_net]
    end

    subgraph XAI[XAI + Assurance Layer]
        shap[shap_explainer]
        lime[lime_explainer]
        anchor[anchor_explainer]
        cons[consensus ⭐]
        robust[robustness ⭐]
        eval[xai_evaluator]
    end

    subgraph UI[Dashboard]
        app[Streamlit app · 5 pages]
    end

    gen --> fe --> pre --> ctx --> trust --> policy
    policy -->|low trust| ML
    ML --> XAI
    shap & lime & anchor --> cons
    ML --> robust
    pre --> drift
    cons & robust & eval --> app
    policy --> loggr --> app
    drift --> app

    classDef novel fill:#7b2ff7,stroke:#4b1e9e,color:#fff;
    class drift,cons,robust novel;
```

> ⭐ = modules added in **v2.0** (novel contributions). Highlighted in purple.

---

## 3. Decision Sequence (per authentication request)

```mermaid
sequenceDiagram
    autonumber
    participant S as Subject
    participant P as Policy Engine (PEP/PDP)
    participant T as Trust Scorer
    participant M as ML Classifier
    participant X as XAI Layer
    participant A as Consensus + Robustness
    participant L as Audit Log
    participant U as Analyst

    S->>P: auth request (device, location, auth_method, ...)
    P->>T: build context, compute trust score
    alt trust >= 0.65
        T-->>P: ALLOW (fast path)
    else trust < 0.65
        T->>M: classify request
        M->>X: SHAP / LIME / Anchor explanations
        X->>A: cross-method consensus + ε-robustness audit
        A-->>P: verdict + XCS + robustness flags
    end
    P->>L: append decision + compliance tags
    P-->>S: ALLOW / DENY
    L-->>U: dashboard stream (flags escalate low-consensus/fragile cases)
```

---

## 4. Explanation-Assurance Loop (v2.0 novelty)

The core new idea: **don't trust a single explanation, and don't trust a model
forever.** Three feedback controls guard the human-in-the-loop.

```mermaid
flowchart TD
    D[Decision + 3 explanations] --> C{XAI Consensus Score}
    C -->|XCS >= 0.60| R{Robustness audit}
    C -->|XCS < 0.60| ESC1[Escalate: methods disagree]
    R -->|robust| SHOW[Show explanation to analyst]
    R -->|fragile| ESC2[Escalate: explanation is manipulable]

    subgraph BG[Continuous background]
        DR{Concept-drift PSI}
        DR -->|stable| OK[Keep serving]
        DR -->|major| RT[Recommend retrain / re-verify]
    end

    SHOW --> DR
    RT -.-> D

    style ESC1 fill:#c0392b,stroke:#7b241c,color:#fff
    style ESC2 fill:#c0392b,stroke:#7b241c,color:#fff
    style RT fill:#d35400,stroke:#873600,color:#fff
```

---

## 5. Layer Responsibilities

| Layer | Responsibility | Key modules |
|-------|----------------|-------------|
| Data | Generate, engineer, and clean auth events | `synthetic_generator`, `feature_engineering`, `preprocessor` |
| Zero Trust | Build context, score trust, enforce NIST SP 800-207 policy, **detect drift**, log | `context_builder`, `trust_scorer`, `policy_engine`, `drift_monitor` ⭐, `decision_logger` |
| Model | Classify low-trust requests | `random_forest`, `xgboost_model`, `neural_net` |
| XAI + Assurance | Explain, **measure cross-method consensus**, **audit robustness**, evaluate | `shap_explainer`, `lime_explainer`, `anchor_explainer`, `consensus` ⭐, `robustness` ⭐, `xai_evaluator` |
| Presentation | 5-page analyst dashboard | `dashboard/app.py` + components |

---

## 6. Design Principles

1. **Fail closed, explain always** — a DENY still ships a human-readable reason.
2. **Never trust one explanation** — cross-method consensus gates analyst trust.
3. **Never trust a model forever** — drift monitoring makes "always verify"
   apply to the model itself.
4. **Latency is a security property** — the trust-gate fast path keeps p95 under
   the 500 ms real-time budget so the system degrades gracefully under load.
5. **Everything is auditable** — every decision, flag, and metric is logged and
   mapped to NIST / HIPAA / GDPR.
