# ai-infra-model-drift

**Systematic model-drift detection on 10 AI-infrastructure & semiconductor equities across 4 macro regimes (2014–2026).**

## Motivation

A master's thesis on FinBERT + SARIMAX forecasting observed a sentiment→return coefficient **sign-flip post-COVID** — but had no framework to measure *when* and *how much* the relationship had drifted. This pipeline builds that instrument: a walk-forward, multi-model, multi-detector drift monitor over the AI-infrastructure stack.

## Methodology

- **Universe** — 10 tickers: NVDA, MU, WDC, LRCX, ASML, AMAT, VRT, EQIX, CIEN, TSM
- **Regimes** (hand-set, see Limitations): R1 baseline (2014–2019) · R2 COVID/ZIRP (2020–2021) · R3 rate hikes (2022–2023) · R4 AI boom (2024–2026)
- **Models** — SARIMAX (statsmodels) · XGBoost tuned with Optuna · LSTM (PyTorch, MPS→CPU fallback), all under **walk-forward validation** (252-day window, 63-day step)
- **Features** — FRED macro (fed funds, 10Y, yield curve, BAA spread), market (VIX, DXY, SOXX, BOTZ), FinBERT sentiment on SEC EDGAR 8-K press releases, EPS surprise
- **Drift detection (4 methods)** — PSI · Kolmogorov–Smirnov · Page-Hinkley (river) · rolling RMSE (90d)
- **Explainability** — SHAP feature importance per ticker × regime

## Key results

**Feature drift (PSI vs. R1 baseline).** `fed_funds` is the most drifted input by far — PSI 18.3 in R2 and 12.4 in R4 (PSI > 0.25 already means "significant"). `vix` follows at 3.2–4.5 across regimes.

**SHAP regime trajectory.** Mean |SHAP| of the SOXX semiconductor index across the 10 tickers grows monotonically: **0.07 (R1) → 3.30 (R2) → 5.90 (R3) → 10.12 (R4)** — by the AI boom, sector beta dominates every model's predictions. Top-3 features shift from {botz, soxx, baa_spread} in R1 to {soxx, yield_curve, botz} in R4.

**Per-regime RMSE (raw, from `predictions.parquet`)** — price units, pooled across tickers; comparable across regimes within a model:

| Model | R1 baseline | R2 COVID/ZIRP | R3 rate hikes | R4 AI boom |
|---|---|---|---|---|
| SARIMAX | **8.9** | **23.8** | 24.7 | **39.8** |
| XGBoost | 15.1 | 26.1 | **21.3** | 54.8 |
| LSTM | 77.9 | 213.7 | 223.0 | 350.4 |

Every model degrades ~4× from R1 to R4; SARIMAX stays the most robust overall. In R3 (rate hikes) XGBoost edges SARIMAX by ~3% (13.69 vs 14.15 rolling RMSE) — effectively tied; both vastly outperform LSTM. Note: `figures/fig5_regime_performance.png` reports the 90-day **rolling** RMSE averaged per regime — a smoothed metric with lower absolute values, but the same model ordering in every regime.

## How to run

```bash
pip install -r requirements.txt

# Optional .env at the repo root:
#   EDGAR_USER_AGENT="your-name your@email.com"   # SEC requires an identifying User-Agent
#   FINNHUB_API_KEY=...                            # optional, not required (earnings text comes from SEC EDGAR)

python drift_pipeline.py   # full pipeline: fetch → models → drift → figures (~45 min)
python run_shap_and_viz.py # light pass: SHAP + visualisations only (requires populated cache/)
```

Outputs land in `output/` (gitignored); curated copies are committed under `results/` and `figures/`.

## Limitations

See [LIMITATIONS.md](LIMITATIONS.md). In particular: **the `sentiment_score` and `surprise_eps` results are artifacts of data sparsity (quarterly events 0-filled to daily), not findings** — the near-zero R4 PSI for sentiment and the all-zero SHAP for EPS surprise reflect the fill scheme, not market behaviour.

## Future work

A Bayesian hierarchical changepoint extension — estimating regime break dates *with uncertainty* instead of hand-setting them — is in progress.
