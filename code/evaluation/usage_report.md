# HackerRank Orchestrate (September 2026) — Model and Token Usage Report

## 1. Executive Summary

| Metric | Full-Dataset Value | Notes |
|---|---|---|
| **Challenge** | Buy or Wait? | HackerRank Orchestrate (September 2026) |
| **Model Architecture** | Deterministic Quantitative Decision Engine | Zero-inference, rule-based simulator & optimizer |
| **Model Providers** | None | Pure deterministic Python execution |
| **Model Names** | None | N/A |
| **Total Model / API Calls** | 0 | All 250 requests processed locally |
| **Input Tokens** | 0 | No external LLM calls |
| **Output Tokens** | 0 | No external LLM calls |
| **Total Tokens** | 0 | Pure deterministic execution |
| **Average Tokens Per Request** | 0.0 | Calculated over 250 requests |
| **Estimated Total Cost (USD)** | $0.00 | Zero external API charges |
| **Estimated Cost Per Request (USD)** | $0.00 | Zero marginal cost |
| **Runtime Execution Speed** | ~300 ms / request | Deterministic vector/loop balance simulation |
| **Determinism & Reproducibility** | 100% | Guaranteed bitwise identical results on identical data |

---

## 2. Methodology & Architectural Rationale

In strict accordance with challenge guidelines, the decision engine is designed to be fully deterministic, auditable, and reproducible:

1. **Deterministic Financial Forecasting**:
   - Reconstructs user financial position directly from `dataset/financial_profiles.csv` and `dataset/financial_events.csv`.
   - Incorporates strict FX conversion via `dataset/exchange_rates.csv` without network calls.
   - Resolves unstated event amounts from verified image receipts in `dataset/images.csv`.
   - Projects 90-day daily cash-flow trajectories using confirmed salary schedules, recurring commitments, and conservative essential spending baselines.

2. **Feasibility & Safety Evaluator**:
   - Candidate payment plans (full payment, installments, partial payment, wait) are verified by recursively projecting the 90-day balance curve.
   - A plan is feasible if and only if the available balance never falls below the user's `minimum_balance_to_keep`.

3. **Specification-Compliant Ranking & Explanation**:
   - Feasible candidate plans are sorted strictly by the 6 challenge tie-breakers:
     1. Complete by deadline
     2. Require no spending changes
     3. Minimize total payable amount
     4. Start payment earlier
     5. Use fewer payments
     6. Lowest `payment_option_id`
   - Decision explanations are generated deterministically from structured decision facts and financial parameters rather than probabilistic text generation, eliminating hallucinations while guaranteeing natural language fluency.

---

## 3. Production Environment & Secrets Declaration

- **Live APIs**: None.
- **Web Scraping / Banking Links**: None.
- **Secrets / API Keys**: No API keys, tokens, credentials, or confidential environment variables are required or included in this submission package.
