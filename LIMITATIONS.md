# Limitations

Known limitations of the drift-detection pipeline, in decreasing order of impact on interpretation.

## 1. Sentiment sparsity → PSI/SHAP artifacts

FinBERT sentiment is computed on **quarterly** SEC EDGAR 8-K press releases, then 0-filled to a daily panel. The resulting `sentiment_score` distribution is a point mass at 0 with rare non-zero spikes. Consequences visible in `results/drift_metrics.csv`:

- PSI for `sentiment_score` in R4 is ≈ 7×10⁻¹⁰ (i.e. "no drift") while the KS test on the same pair is highly significant — the PSI buckets are dominated by the zero mass, not by sentiment dynamics.
- SHAP importance for sentiment (0 → 0.10 → 0.32 → 0.20 across R1→R4) mostly tracks *event density*, not signal strength.

**These values are artifacts of the fill scheme and must not be read as "sentiment doesn't drift" or "sentiment doesn't matter".**

## 2. `surprise_eps` SHAP ≈ 0 — same cause

EPS surprise is also a quarterly event 0-filled to daily; its SHAP importance is exactly 0.0 in **all four regimes** (`results/shap_importance.csv`). Same artifact mechanism as §1, more extreme.

## 3. Regimes are hand-set dates, not estimated

The four regime windows (R1 2014–2019, R2 2020–2021, R3 2022–2023, R4 2024–2026) are imposed a priori from macro narrative, not estimated from the data. All regime-conditional results inherit this assumption — a misplaced boundary smears drift across neighbouring regimes. **This is precisely the motivation for the planned Bayesian hierarchical changepoint extension** (break dates estimated with posterior uncertainty).

## 4. Single asset class

All 10 tickers are AI-infrastructure / semiconductor equities, and two of the features (SOXX, BOTZ) are sector ETFs — near-collinear sector proxies. Findings (e.g. the SOXX SHAP trajectory) describe this sector's regime dynamics and do not generalise to other asset classes without re-estimation.

## 5. Walk-forward step widened for compute

The walk-forward step is 63 trading days (one quarter), widened from the original 21 to cut iterations 3×. This is a compute compromise, not a methodological optimum: drift onsets can be detected up to one quarter late, and rolling-RMSE granularity is correspondingly coarser.

## Additional caveats

- RMSE is reported in **price units pooled across tickers**, so high-priced names dominate the level; compare across regimes within a model, not across models' absolute skill on individual names.
- LSTM RMSE (77.9 → 350.4 across regimes) is uncompetitive here — likely under-tuned for this data size rather than evidence against deep models in general.
