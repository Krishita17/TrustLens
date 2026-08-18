# Security Policy

TrustLens is a defensive security research project. We take the security of the
code — and the integrity of the security decisions it models — seriously.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.x     | ✅ Active security support |
| 1.x     | ⚠️ Critical fixes only |
| < 1.0   | ❌ Unsupported |

## Reporting a Vulnerability

Please report security issues **privately** — do not open a public issue for an
undisclosed vulnerability.

1. Use GitHub's **[Private vulnerability reporting](https://github.com/Krishita17/TrustLens/security/advisories/new)**
   (Security tab → *Report a vulnerability*).
2. Include: affected file/module, reproduction steps, impact, and any PoC.
3. You will receive an acknowledgement within **72 hours** and a remediation
   timeline within **7 days**.

Please give us a reasonable window to remediate before any public disclosure
(coordinated disclosure, 90 days by default).

## Scope

In scope:

- The Python source under `trustlens/src/` (trust scoring, policy engine, XAI,
  consensus/robustness/drift modules, dashboard).
- CI/CD workflows under `.github/`.
- Dependency and supply-chain issues surfaced by `pip-audit` / Dependabot.

Out of scope:

- The bundled **synthetic** dataset (contains no real user data).
- Denial-of-service via deliberately pathological inputs to the offline
  research pipeline.
- Findings that require a compromised host or physical access.

## Threat Model

This project ships a full **STRIDE threat model**, including attacks that are
specific to the explainability layer (explanation manipulation, model
extraction via explanations, adversarial evasion, and behavioural drift
poisoning). See **[`trustlens/docs/THREAT_MODEL.md`](trustlens/docs/THREAT_MODEL.md)**.

## Security Hardening in This Project

| Control | Where |
|---------|-------|
| Static application security testing (Bandit) | `.github/workflows/ci.yml` |
| Semantic code analysis (CodeQL, security-and-quality) | `.github/workflows/codeql.yml` |
| Dependency vulnerability auditing (pip-audit) | `.github/workflows/ci.yml` |
| Automated dependency updates (Dependabot) | `.github/dependabot.yml` |
| Least-privilege GitHub token scopes | all workflows (`permissions:` blocks) |
| Explanation-integrity checks (consensus + robustness auditing) | `src/xai/consensus.py`, `src/xai/robustness.py` |
| Model-decay detection (concept-drift monitor) | `src/zta/drift_monitor.py` |

## Responsible Use

TrustLens is intended for **research, education, and defensive** security work
(understanding and auditing AI-driven access control). It must not be used to
build systems that make consequential access decisions about real people
without appropriate human oversight, validation, and compliance review.
