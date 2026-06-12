#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini-script : charge les caches existants et fait UNIQUEMENT
  - SHAP par régime
  - les 5 visualisations
  - export drift_metrics.csv

Usage : python run_shap_and_viz.py
Prérequis : avoir déjà tourné drift_pipeline.py au moins une fois
            (cache/master_dataset.parquet et output/predictions.parquet doivent exister).
"""

import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
import shap
from scipy.stats import ks_2samp
from sklearn.inspection import permutation_importance


# ======================================================================
# CONFIG (doit matcher drift_pipeline.py)
# ======================================================================
@dataclass
class Config:
    regimes: Dict[str, Tuple[str, str]] = field(default_factory=lambda: {
        'R1_baseline':   ('2014-01-01', '2019-12-31'),
        'R2_covid_zirp': ('2020-01-01', '2021-12-31'),
        'R3_rate_hikes': ('2022-01-01', '2023-12-31'),
        'R4_ai_boom':    ('2024-01-01', '2026-05-01'),
    })
    fred_series: Dict[str, str] = field(default_factory=lambda: {
        'fed_funds':    'FEDFUNDS',
        'treasury_10y': 'DGS10',
        'yield_curve':  'T10Y2Y',
        'baa_spread':   'BAA10Y',
    })
    market_series: Dict[str, str] = field(default_factory=lambda: {
        'vix':  '^VIX', 'dxy':  'DX-Y.NYB', 'soxx': 'SOXX', 'botz': 'BOTZ',
    })
    tickers: List[str] = field(default_factory=lambda: [
        'NVDA', 'MU', 'WDC', 'LRCX', 'ASML',
        'AMAT', 'VRT', 'EQIX', 'CIEN', 'TSM',
    ])
    output_dir:  Path = field(default_factory=lambda: Path('./output'))
    cache_dir:   Path = field(default_factory=lambda: Path('./cache'))
    figures_dir: Path = field(default_factory=lambda: Path('./figures'))


CFG = Config()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


# ======================================================================
# UTILS
# ======================================================================
def assign_regime(date: pd.Timestamp) -> str:
    """
    Map a date to its macro regime window.

    Parameters
    ----------
    date : pd.Timestamp
        Date to classify.

    Returns
    -------
    str
        Regime key (e.g. 'R1_baseline'), or 'unknown' if outside all windows.
    """
    for name, (start, end) in CFG.regimes.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return name
    return 'unknown'


def attach_sentiment_to_panel(master: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Merge event-level FinBERT sentiment onto the daily panel (backward as-of, per ticker).

    Parameters
    ----------
    master : pd.DataFrame
        Daily panel with 'date' and 'ticker' columns.
    sentiment : pd.DataFrame
        Event-level scores with 'date', 'ticker', 'sentiment_score' columns.

    Returns
    -------
    pd.DataFrame
        Master panel with a 'sentiment_score' column (0.0 where no event is available).
    """
    if sentiment.empty:
        master['sentiment_score'] = 0.0
        return master
    parts = []
    for ticker, group in master.groupby('ticker'):
        sent_t = sentiment[sentiment['ticker'] == ticker][['date', 'sentiment_score']]
        if sent_t.empty:
            group = group.copy()
            group['sentiment_score'] = 0.0
            parts.append(group)
            continue
        merged = pd.merge_asof(
            group.sort_values('date'),
            sent_t.sort_values('date'),
            on='date', direction='backward'
        )
        merged['sentiment_score'] = merged['sentiment_score'].fillna(0.0)
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)


# ======================================================================
# LOAD CACHED DATA
# ======================================================================
def load_master_with_sentiment() -> pd.DataFrame:
    """
    Load the cached master dataset and attach the cached sentiment panel if present.

    Parameters
    ----------
    None

    Returns
    -------
    pd.DataFrame
        Master panel with a 'sentiment_score' column (zeros if no sentiment cache).
    """
    log.info('Loading cached master dataset...')
    master_path = CFG.cache_dir / 'master_dataset.parquet'
    if not master_path.exists():
        raise FileNotFoundError(f'Missing {master_path}. Run drift_pipeline.py first.')
    master = pd.read_parquet(master_path)

    sent_path = CFG.cache_dir / 'sentiment_panel.parquet'
    if sent_path.exists():
        log.info('Loading cached sentiment panel...')
        sentiment = pd.read_parquet(sent_path)
        master = attach_sentiment_to_panel(master, sentiment)
    else:
        master['sentiment_score'] = 0.0
        log.warning('No sentiment cache, using zeros.')

    return master


def load_predictions() -> pd.DataFrame:
    """
    Load cached walk-forward predictions from output/predictions.parquet.

    Parameters
    ----------
    None

    Returns
    -------
    pd.DataFrame
        Long format: 'date', 'ticker', 'actual', 'predicted', 'model'.
    """
    pred_path = CFG.output_dir / 'predictions.parquet'
    if not pred_path.exists():
        raise FileNotFoundError(f'Missing {pred_path}. Run walk-forward in drift_pipeline.py first.')
    log.info('Loading cached predictions...')
    return pd.read_parquet(pred_path)


# ======================================================================
# DRIFT METRICS (rapide)
# ======================================================================
def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between a baseline sample and a comparison sample.

    Parameters
    ----------
    expected : np.ndarray
        Baseline sample defining the quantile buckets.
    actual : np.ndarray
        Comparison sample.
    bins : int, optional
        Number of quantile buckets, by default 10.

    Returns
    -------
    float
        PSI value (rule of thumb: > 0.25 indicates a significant shift).
    """
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    e, _ = np.histogram(expected, bins=breakpoints)
    a, _ = np.histogram(actual, bins=breakpoints)
    e = np.where(e == 0, 1e-6, e) / max(len(expected), 1)
    a = np.where(a == 0, 1e-6, a) / max(len(actual), 1)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_panel_by_regime(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Compute PSI and KS-test for one feature, each regime vs the R1 baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Panel with a 'date' column and the feature column.
    feature : str
        Feature column to test.

    Returns
    -------
    pd.DataFrame
        One row per non-baseline regime: 'feature', 'regime', 'psi',
        'ks_stat', 'ks_p', 'n_baseline', 'n_actual'.
    """
    df = df.copy()
    df['regime'] = df['date'].apply(assign_regime)
    baseline_name = list(CFG.regimes.keys())[0]
    baseline = df[df['regime'] == baseline_name][feature].dropna().values
    rows = []
    for regime in CFG.regimes.keys():
        if regime == baseline_name:
            continue
        actual = df[df['regime'] == regime][feature].dropna().values
        if len(actual) < 30:
            continue
        psi = compute_psi(baseline, actual)
        ks_stat, ks_p = ks_2samp(baseline, actual)
        rows.append({'feature': feature, 'regime': regime,
                     'psi': psi, 'ks_stat': float(ks_stat), 'ks_p': float(ks_p),
                     'n_baseline': len(baseline), 'n_actual': len(actual)})
    return pd.DataFrame(rows)


def rolling_rmse(predictions: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    """
    Rolling RMSE of predictions per (model, ticker) pair.

    Parameters
    ----------
    predictions : pd.DataFrame
        Walk-forward predictions ('date', 'ticker', 'actual', 'predicted', 'model').
    window : int, optional
        Rolling window length in observations, by default 90.

    Returns
    -------
    pd.DataFrame
        Long format: 'date', 'ticker', 'model', 'rolling_rmse'.
    """
    out = []
    for (m, t), g in predictions.groupby(['model', 'ticker']):
        g = g.sort_values('date').reset_index(drop=True)
        sq_err = (g['actual'] - g['predicted']) ** 2
        roll = np.sqrt(sq_err.rolling(window).mean())
        out.append(pd.DataFrame({
            'date': g['date'], 'ticker': t, 'model': m, 'rolling_rmse': roll
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ======================================================================
# SHAP avec fixes empilés + fallback permutation
# ======================================================================
def compute_shap_by_regime(df: pd.DataFrame, ticker: str,
                           features: List[str], target: str = 'Close') -> pd.DataFrame:
    """
    Mean |SHAP| importance per feature for one ticker, fitted separately per regime.

    Parameters
    ----------
    df : pd.DataFrame
        Master panel with 'date', 'ticker', feature and target columns.
    ticker : str
        Ticker to analyse.
    features : List[str]
        Feature column names.
    target : str, optional
        Target column, by default 'Close'.

    Returns
    -------
    pd.DataFrame
        Rows: 'ticker', 'regime', 'feature', 'shap_importance', 'method'
        ('shap', or 'permutation' when the SHAP explainer fails).
    """
    df = df.copy()
    df['regime'] = df['date'].apply(assign_regime)
    sub = df[df['ticker'] == ticker].dropna(subset=features + [target])

    rows = []
    for regime in CFG.regimes.keys():
        rdf = sub[sub['regime'] == regime]
        if len(rdf) < 100:
            continue
        x_data, y_data = rdf[features], rdf[target]
        m = xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=42, verbosity=0, tree_method='hist',
            base_score=0.5,  # force base_score scalaire pour eviter le bug shap
        )
        m.fit(x_data, y_data)

        sample_size = min(200, len(x_data))
        sample = x_data.iloc[:sample_size]

        try:
            # API moderne shap.Explainer (model-agnostic, contourne le bug TreeExplainer)
            explainer = shap.Explainer(m.predict, sample)
            shap_vals = explainer(sample).values
            mean_abs = np.abs(shap_vals).mean(axis=0)
            method = 'shap'
        except Exception as e:
            log.warning(f'  SHAP failed for {ticker} {regime}, fallback to permutation: {e}')
            perm = permutation_importance(m, sample, y_data.iloc[:sample_size],
                                          n_repeats=5, random_state=42, n_jobs=-1)
            mean_abs = np.abs(perm.importances_mean)
            method = 'permutation'

        for f, imp in zip(features, mean_abs):
            rows.append({'ticker': ticker, 'regime': regime,
                         'feature': f, 'shap_importance': float(imp),
                         'method': method})
    return pd.DataFrame(rows)


# ======================================================================
# VIZ
# ======================================================================
def plot_distribution_shift(df: pd.DataFrame, feature: str, save_as: str):
    """
    KDE plot of a feature's distribution per regime, saved as PNG.

    Parameters
    ----------
    df : pd.DataFrame
        Panel with a 'date' column and the feature column.
    feature : str
        Column to plot.
    save_as : str
        Output filename inside CFG.figures_dir.

    Returns
    -------
    None
    """
    df = df.copy()
    df['regime'] = df['date'].apply(assign_regime)
    fig, ax = plt.subplots(figsize=(10, 5))
    palette = sns.color_palette('viridis', n_colors=len(CFG.regimes))
    for color, regime in zip(palette, CFG.regimes.keys()):
        data = df[df['regime'] == regime][feature].dropna()
        if len(data) > 10:
            sns.kdeplot(data, label=regime, ax=ax, color=color, linewidth=2)
    ax.set_title(f'Distribution Shift: {feature} across regimes', fontsize=13)
    ax.set_xlabel(feature)
    ax.legend()
    plt.tight_layout()
    plt.savefig(CFG.figures_dir / save_as, dpi=150)
    plt.close()


def plot_rolling_rmse(rmse_df: pd.DataFrame, save_as: str):
    """
    Time series of rolling RMSE averaged across tickers, with regime shading, saved as PNG.

    Parameters
    ----------
    rmse_df : pd.DataFrame
        Output of rolling_rmse() ('date', 'ticker', 'model', 'rolling_rmse').
    save_as : str
        Output filename inside CFG.figures_dir.

    Returns
    -------
    None
    """
    avg = rmse_df.groupby(['date', 'model'])['rolling_rmse'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    for model in avg['model'].unique():
        m = avg[avg['model'] == model]
        ax.plot(m['date'], m['rolling_rmse'], label=model, linewidth=1.5)
    for _, (start, end) in CFG.regimes.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.07)
    ax.set_title('Rolling RMSE (90-day) - model degradation across regimes', fontsize=13)
    ax.set_xlabel('Date'); ax.set_ylabel('RMSE')
    ax.legend()
    plt.tight_layout()
    plt.savefig(CFG.figures_dir / save_as, dpi=150)
    plt.close()


def plot_shap_drift_heatmap(shap_df: pd.DataFrame, save_as: str):
    """
    Heatmap of mean |SHAP| per feature x regime, saved as PNG.

    Parameters
    ----------
    shap_df : pd.DataFrame
        Output of compute_shap_by_regime() stacked over tickers.
    save_as : str
        Output filename inside CFG.figures_dir.

    Returns
    -------
    None
    """
    pivot = shap_df.groupby(['feature', 'regime'])['shap_importance'].mean().unstack()
    cols = [c for c in CFG.regimes.keys() if c in pivot.columns]
    pivot = pivot[cols]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.4 * len(pivot))))
    sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.2f',
                cbar_kws={'label': 'Mean |SHAP|'}, ax=ax)
    ax.set_title('Feature importance drift across regimes (SHAP)', fontsize=13)
    plt.tight_layout()
    plt.savefig(CFG.figures_dir / save_as, dpi=150)
    plt.close()


def plot_regime_performance_heatmap(rmse_df: pd.DataFrame, save_as: str):
    """
    Heatmap of mean rolling RMSE per model x regime, saved as PNG.

    Parameters
    ----------
    rmse_df : pd.DataFrame
        Output of rolling_rmse() ('date', 'ticker', 'model', 'rolling_rmse').
    save_as : str
        Output filename inside CFG.figures_dir.

    Returns
    -------
    None
    """
    df = rmse_df.copy()
    df['regime'] = df['date'].apply(assign_regime)
    pivot = df.groupby(['model', 'regime'])['rolling_rmse'].mean().unstack()
    cols = [c for c in CFG.regimes.keys() if c in pivot.columns]
    pivot = pivot[cols]
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(pivot, cmap='RdYlGn_r', annot=True, fmt='.2f',
                cbar_kws={'label': 'Avg RMSE'}, ax=ax)
    ax.set_title('Model x Regime performance matrix', fontsize=13)
    plt.tight_layout()
    plt.savefig(CFG.figures_dir / save_as, dpi=150)
    plt.close()


# ======================================================================
# MAIN
# ======================================================================
def main():
    """
    Run the light pipeline: load caches, drift metrics, rolling RMSE, SHAP, figures.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    log.info('=' * 70)
    log.info('SHAP + VIZ ONLY - using cached predictions')
    log.info('=' * 70)

    CFG.figures_dir.mkdir(parents=True, exist_ok=True)
    CFG.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load caches
    master = load_master_with_sentiment()
    predictions = load_predictions()
    log.info(f'  master:      {len(master):,} rows')
    log.info(f'  predictions: {len(predictions):,} rows')

    feature_cols = (
        list(CFG.fred_series.keys())
        + list(CFG.market_series.keys())
        + ['surprise_eps', 'sentiment_score']
    )
    feature_cols = [c for c in feature_cols if c in master.columns]
    master[feature_cols] = master[feature_cols].fillna(0)

    # 2. Drift metrics + rolling RMSE
    log.info('\n[1/4] Drift detection (recompute, fast)...')
    drift_results = []
    for feat in ['sentiment_score', 'fed_funds', 'vix']:
        if feat in master.columns:
            drift_results.append(drift_panel_by_regime(master, feat))
    drift_df = pd.concat(drift_results, ignore_index=True) if drift_results else pd.DataFrame()
    if not drift_df.empty:
        drift_df.to_csv(CFG.output_dir / 'drift_metrics.csv', index=False)
        log.info(f'\n{drift_df.to_string(index=False)}')

    rmse_df = rolling_rmse(predictions, window=90)
    rmse_df.to_parquet(CFG.output_dir / 'rolling_rmse.parquet')

    # 3. SHAP
    log.info('\n[2/4] SHAP explainability (this is the slow part)...')
    shap_results = []
    for ticker in CFG.tickers:
        log.info(f'  SHAP: {ticker}')
        sd = compute_shap_by_regime(master, ticker, feature_cols)
        if not sd.empty:
            shap_results.append(sd)
    shap_df = pd.concat(shap_results, ignore_index=True) if shap_results else pd.DataFrame()
    if not shap_df.empty:
        shap_df.to_csv(CFG.output_dir / 'shap_importance.csv', index=False)
        n_shap = (shap_df['method'] == 'shap').sum()
        n_perm = (shap_df['method'] == 'permutation').sum()
        log.info(f'  computed via SHAP: {n_shap}, via permutation fallback: {n_perm}')

    # 4. Viz
    log.info('\n[3/4] Generating visualizations...')
    if 'sentiment_score' in master.columns:
        plot_distribution_shift(master, 'sentiment_score', 'fig1_sentiment_drift.png')
    if 'fed_funds' in master.columns:
        plot_distribution_shift(master, 'fed_funds', 'fig2_fedfunds_drift.png')
    if not rmse_df.empty:
        plot_rolling_rmse(rmse_df, 'fig3_rolling_rmse.png')
        plot_regime_performance_heatmap(rmse_df, 'fig5_regime_performance.png')
    if not shap_df.empty:
        plot_shap_drift_heatmap(shap_df, 'fig4_shap_drift.png')

    log.info('\n[4/4] Done.')
    log.info('=' * 70)
    log.info(f'  Output:  {CFG.output_dir.resolve()}')
    log.info(f'  Figures: {CFG.figures_dir.resolve()}')
    log.info('=' * 70)


if __name__ == '__main__':
    main()