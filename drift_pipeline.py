#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Drift Detection Pipeline — AI Infrastructure Stocks (2014-2026)
====================================================================
A production-grade pipeline comparing SARIMAX, XGBoost, and LSTM models
on AI infrastructure stocks, with walk-forward validation and 4-method
drift detection (PSI, KS, Page-Hinkley, Rolling RMSE).

Author: Jean Trèves
Date: May 2026
Hardware: Apple M1 Pro (MPS backend for PyTorch)
"""

# ======================================================================
# 1. IMPORTS
# ======================================================================
# noinspection PyPep8Naming
import os
import sys
import time
import logging
import warnings
import importlib.util
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Time series & stats
import yfinance as yf
from pandas_datareader import data as pdr
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import ks_2samp

# ML
import xgboost as xgb
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Deep learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# NLP
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# Explainability
import shap

# Drift detection
from river.drift import PageHinkley

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Config secrets — chargés depuis .env (jamais commité)
from dotenv import load_dotenv
load_dotenv()

# On demande la variable nommée 'FINNHUB_API_KEY'
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')



# ======================================================================
# 2. CONFIGURATION
# ======================================================================
@dataclass
class Config:
    """Centralized configuration for the entire pipeline."""

    # noinspection PyTypeChecker
    tickers: List[str] = field(default_factory=lambda: [
        'NVDA', 'MU', 'WDC', 'LRCX', 'ASML',
        'AMAT', 'VRT', 'EQIX', 'CIEN', 'TSM',
    ])

    start_date: str = '2014-01-01'
    end_date: str = '2026-05-01'

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
        'vix':  '^VIX',
        'dxy':  'DX-Y.NYB',
        'soxx': 'SOXX',
        'botz': 'BOTZ',
    })

    walk_forward_window: int = 252
    walk_forward_step: int = 63       # était 21 → 3× moins d'itérations walk-forward
    optuna_trials: int = 50
    lstm_epochs: int = 15             # était 30 → premier fit plus rapide
    lstm_hidden: int = 32             # était 64 → réseau plus léger
    lstm_seq_len: int = 10            # était 20 → séquences plus courtes

    output_dir: Path = field(default_factory=lambda: Path('./output'))
    cache_dir: Path = field(default_factory=lambda: Path('./cache'))
    figures_dir: Path = field(default_factory=lambda: Path('./figures'))

    device: str = 'mps' if torch.backends.mps.is_available() else 'cpu'

    def __post_init__(self):
        for d in [self.output_dir, self.cache_dir, self.figures_dir]:
            d.mkdir(parents=True, exist_ok=True)


CFG = Config()


# ======================================================================
# 3. LOGGING
# ======================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(CFG.output_dir / 'pipeline.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)
log.info("FINNHUB_API_KEY chargée : %s", "OK" if FINNHUB_API_KEY else "MANQUANTE")


# ======================================================================
# 4. UTILITY: ROBUST TIMEZONE STRIPPING
# ======================================================================
def strip_tz(idx: pd.Index) -> pd.Index:
    """
    Remove timezone info from a DatetimeIndex.

    Parameters
    ----------
    idx : pd.Index
        Index to strip; converted to DatetimeIndex if not already.

    Returns
    -------
    pd.Index
        Timezone-naive DatetimeIndex.
    """
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        return idx.tz_localize(None)
    return idx


# ======================================================================
# 5. DYNAMIC CRAWLER LOADING (graceful fallback)
# ======================================================================
def _load_crawler_function():
    """
    Attempt to import write_crawl_results from a local crawler module.

    Returns
    -------
    callable or None
        The write_crawl_results function if found, else None.
    """
    try:
        spec = importlib.util.find_spec('crawler')
        if spec is None:
            return None
        crawler_module = importlib.import_module('crawler')
        return getattr(crawler_module, 'write_crawl_results', None)
    except Exception as e:
        log.debug(f'  crawler import failed: {e}')
        return None


_CRAWLER_FN = _load_crawler_function()


# ======================================================================
# 6. DATA COLLECTION
# ======================================================================
def fetch_stock_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV prices via yfinance, with parquet cache.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. 'NVDA').
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        Columns: date, Open, High, Low, Close, Volume, ticker.
        Empty DataFrame if fetch fails.
    """
    cache_path = CFG.cache_dir / f'prices_{ticker}.parquet'
    if cache_path.exists():
        log.info(f'  [cache hit] prices for {ticker}')
        return pd.read_parquet(cache_path)

    log.info(f'  Fetching prices: {ticker}')
    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()

    df.index = strip_tz(df.index)
    df = df.reset_index().rename(columns={'Date': 'date'})
    df['ticker'] = ticker
    df.to_parquet(cache_path)
    return df


def fetch_earnings_surprise(ticker: str) -> pd.DataFrame:
    """
    Fetch quarterly earnings surprises via yfinance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.

    Returns
    -------
    pd.DataFrame
        Columns: date, ticker, epsActual, epsEstimate, surprise_eps.
        Empty DataFrame if no data available.
    """
    cache_path = CFG.cache_dir / f'earnings_{ticker}.parquet'
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    try:
        t = yf.Ticker(ticker)
        eh = t.earnings_history
        if eh is None or eh.empty:
            log.warning(f'  No earnings history for {ticker}')
            return pd.DataFrame()

        eh = eh.reset_index()
        date_col = eh.columns[0]
        eh['date'] = pd.to_datetime(eh[date_col], errors='coerce')
        eh['date'] = eh['date'].dt.tz_localize(None) if eh['date'].dt.tz is not None else eh['date']

        if 'epsActual' in eh.columns and 'epsEstimate' in eh.columns:
            eh['surprise_eps'] = eh['epsActual'] - eh['epsEstimate']
        else:
            return pd.DataFrame()

        eh['ticker'] = ticker
        eh = eh[['date', 'ticker', 'epsActual', 'epsEstimate', 'surprise_eps']]
        n_before = len(eh)
        eh = eh.dropna(subset=['date'])
        log.info('  dropna earnings %s : %d lignes supprimées (%d → %d)',
                 ticker, n_before - len(eh), n_before, len(eh))
        eh.to_parquet(cache_path)
        return eh
    except Exception as e:
        log.warning(f'  Earnings fetch failed for {ticker}: {e}')
        return pd.DataFrame()


def fetch_macro_variables(start: str, end: str) -> pd.DataFrame:
    """
    Fetch FRED macro variables and market index prices.

    Parameters
    ----------
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        Columns: date, fed_funds, treasury_10y, yield_curve, baa_spread,
        vix, dxy, soxx, botz. Forward-filled then dropna applied.
    """
    cache_path = CFG.cache_dir / 'macro.parquet'
    if cache_path.exists():
        log.info('  [cache hit] macro variables')
        return pd.read_parquet(cache_path)

    log.info('  Fetching FRED macro variables...')
    macro_dfs = []
    for name, code in CFG.fred_series.items():
        try:
            s = pdr.DataReader(code, 'fred', start, end)
            s.columns = [name]
            macro_dfs.append(s)
        except Exception as e:
            log.warning(f'  FRED {code} failed: {e}')

    log.info('  Fetching market indices...')
    for name, ticker in CFG.market_series.items():
        try:
            hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
            if hist.empty:
                continue
            s = hist['Close'].copy()
            s.index = strip_tz(s.index)
            s.name = name
            macro_dfs.append(s.to_frame())
        except Exception as e:
            log.warning(f'  Market {ticker} failed: {e}')

    if not macro_dfs:
        return pd.DataFrame()

    macro = pd.concat(macro_dfs, axis=1).ffill()
    n_before = len(macro)
    macro = macro.dropna()
    log.info('  dropna macro : %d lignes supprimées (%d → %d)',
             n_before - len(macro), n_before, len(macro))
    macro.index.name = 'date'
    macro = macro.reset_index()
    macro['date'] = pd.to_datetime(macro['date'])
    if hasattr(macro['date'].dt, 'tz') and macro['date'].dt.tz is not None:
        macro['date'] = macro['date'].dt.tz_localize(None)
    macro.to_parquet(cache_path)
    return macro


# ======================================================================
# 7. SENTIMENT EXTRACTION
# ======================================================================
class FinBERTAnalyzer:
    """
    FinBERT-based sentiment scorer for financial text.

    Loads the yiyanghkust/finbert-tone model and scores arbitrary-length
    text by chunking into 510-token segments and averaging weighted scores.
    """

    LABEL_WEIGHTS = {'Neutral': 0, 'Positive': 1, 'Negative': -1}
    MAX_CHUNK = 510

    def __init__(self):
        log.info('Loading FinBERT model...')
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.model = BertForSequenceClassification.from_pretrained(
            'yiyanghkust/finbert-tone', num_labels=3
        )
        device_idx = 0 if torch.backends.mps.is_available() else -1
        self.pipe = pipeline(
            'sentiment-analysis',
            model=self.model,
            tokenizer=self.tokenizer,
            device=device_idx,
        )

    def score_text(self, text: str) -> float:
        """
        Compute a weighted sentiment score for a financial text passage.

        Parameters
        ----------
        text : str
            Raw financial text (earnings press release, 8-K body, etc.).

        Returns
        -------
        float
            Score in [-1, 1]: positive > 0, negative < 0, neutral ≈ 0.
            Returns 0.0 for empty or very short input.
        """
        if not text or len(text.strip()) < 10:
            return 0.0

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = [
            token_ids[i:i + self.MAX_CHUNK]
            for i in range(0, len(token_ids), self.MAX_CHUNK)
        ]

        scores = []
        for chunk in chunks:
            chunk_text = self.tokenizer.decode(chunk, skip_special_tokens=True)
            try:
                out = self.pipe(chunk_text, truncation=True, max_length=512)[0]
                w = self.LABEL_WEIGHTS.get(out['label'], 0)
                scores.append(w * out['score'])
            except Exception as e:
                log.debug(f'  chunk scoring failed: {e}')

        return float(np.mean(scores)) if scores else 0.0


# ======================================================================
# 7b. SEC EDGAR — FETCH EARNINGS TEXT (8-K press releases)
# ======================================================================
import requests, re, json

EDGAR_HEADERS = {
    # SEC exige un User-Agent identifiant — définir EDGAR_USER_AGENT dans .env
    "User-Agent": os.getenv("EDGAR_USER_AGENT", "research-bot contact@example.com"),
    "Accept-Encoding": "gzip, deflate",
}

_EDGAR_CIK_CACHE: Dict[str, str] = {}

def _get_cik(ticker: str) -> Optional[str]:
    """
    Resolve a stock ticker to its SEC EDGAR CIK number.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. 'NVDA').

    Returns
    -------
    str or None
        Zero-padded 10-digit CIK string, or None if lookup fails.
    """
    if ticker in _EDGAR_CIK_CACHE:
        return _EDGAR_CIK_CACHE[ticker]
    try:
        r = requests.get(
            f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom"
            f"&startdt=2014-01-01&enddt=2014-12-31&forms=8-K",
            headers=EDGAR_HEADERS, timeout=10
        )
        # Méthode plus fiable : company_tickers.json
        r2 = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=EDGAR_HEADERS, timeout=15
        )
        data = r2.json()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                _EDGAR_CIK_CACHE[ticker] = cik
                return cik
    except Exception as e:
        log.debug(f"  CIK lookup failed for {ticker}: {e}")
    return None


def _get_8k_filings(cik: str, year: int, quarter: int) -> List[dict]:
    """
    Return 8-K filing metadata for a company/quarter via EDGAR submissions API.

    Parameters
    ----------
    cik : str
        Zero-padded 10-digit CIK string.
    year : int
        Calendar year.
    quarter : int
        Quarter index (1–4). Maps to the approximate earnings filing window.

    Returns
    -------
    list of dict
        Each dict contains 'date' (str) and 'accessionNumber' (str).
        Empty list if the request fails or no filings found.
    """
    # Q1 earnings → filed in April/May, Q2 → Jul/Aug, Q3 → Oct/Nov, Q4 → Jan/Feb
    q_windows = {
        1: (f"{year}-03-01", f"{year}-05-31"),
        2: (f"{year}-06-01", f"{year}-08-31"),
        3: (f"{year}-09-01", f"{year}-11-30"),
        4: (f"{year}-11-01", f"{year+1}-02-28"),
    }
    start_dt, end_dt = q_windows[quarter]
    try:
        url = (
            f"https://efts.sec.gov/LATEST/search-index?q=%22earnings%22"
            f"&dateRange=custom&startdt={start_dt}&enddt={end_dt}"
            f"&forms=8-K&entity={cik}"
        )
        # Approche plus robuste via submissions API
        sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        r = requests.get(sub_url, headers=EDGAR_HEADERS, timeout=15)
        data = r.json()
        filings = data.get("filings", {}).get("recent", {})
        forms   = filings.get("form", [])
        dates   = filings.get("filingDate", [])
        acc_nos = filings.get("accessionNumber", [])

        results = []
        for form, date_str, acc in zip(forms, dates, acc_nos):
            if form != "8-K":
                continue
            if start_dt <= date_str <= end_dt:
                results.append({"date": date_str, "accessionNumber": acc})
        return results
    except Exception as e:
        log.debug(f"  8-K listing failed CIK={cik}: {e}")
        return []


def _fetch_8k_text(cik: str, accession_number: str) -> str:
    """
    Download and strip HTML from a single 8-K filing document.

    Parameters
    ----------
    cik : str
        Zero-padded 10-digit CIK string.
    accession_number : str
        EDGAR accession number (e.g. '0001234567-24-000001').

    Returns
    -------
    str
        Plain-text press release body, truncated to 8 000 chars.
        Empty string on fetch failure.
    """
    acc_clean = accession_number.replace("-", "")
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{acc_clean}/{accession_number}-index.htm"
    )
    try:
        r = requests.get(index_url, headers=EDGAR_HEADERS, timeout=15)
        # Find the primary document (ex99.htm or similar)
        doc_links = re.findall(r'href="([^"]+\.htm)"', r.text, re.IGNORECASE)
        # Préférer les exhibits (ex99) qui contiennent le press release
        primary = next(
            (l for l in doc_links if re.search(r'ex.?99|press|earnings', l, re.I)),
            doc_links[0] if doc_links else None
        )
        if primary is None:
            return ""
        if not primary.startswith("http"):
            primary = "https://www.sec.gov" + primary

        r2 = requests.get(primary, headers=EDGAR_HEADERS, timeout=15)
        # Strip HTML tags
        text = re.sub(r'<[^>]+>', ' ', r2.text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Garder seulement les 8000 premiers caractères (suffisant pour FinBERT)
        return text[:8000]
    except Exception as e:
        log.debug(f"  8-K text fetch failed {accession_number}: {e}")
        return ""


def fetch_earnings_text_edgar(ticker: str, year: int, quarter: int) -> str:
    """
    Fetch earnings press release text from SEC EDGAR (official, free).

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    year : int
        Calendar year.
    quarter : int
        Quarter index (1–4).

    Returns
    -------
    str
        Raw text of the 8-K earnings press release (≤ 8 000 chars).
        Empty string if no filing found.
    """
    cik = _get_cik(ticker)
    if not cik:
        log.debug(f"  No CIK found for {ticker}")
        return ""

    filings = _get_8k_filings(cik, year, quarter)
    if not filings:
        log.debug(f"  No 8-K found for {ticker} {year}Q{quarter}")
        return ""

    # Prendre le premier filing (le plus récent dans la fenêtre)
    acc = filings[0]["accessionNumber"]
    text = _fetch_8k_text(cik, acc)
    log.info(f"  EDGAR {ticker} {year}Q{quarter}: {len(text)} chars")
    return text


def fetch_earnings_text_crawler_fallback(ticker: str, year: int, quarter: int) -> str:
    """
    Fetch earnings text via the local crawler module if available.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    year : int
        Calendar year.
    quarter : int
        Quarter index (1–4).

    Returns
    -------
    str
        Concatenated body text from crawler results.
        Empty string if crawler is unavailable or the call fails.
    """
    if _CRAWLER_FN is None:
        return ''
    company_map = {
        'NVDA': 'Nvidia', 'MU': 'Micron', 'WDC': 'Western Digital',
        'LRCX': 'Lam Research', 'ASML': 'ASML', 'AMAT': 'Applied Materials',
        'VRT': 'Vertiv', 'EQIX': 'Equinix', 'CIEN': 'Ciena', 'TSM': 'TSMC',
    }
    company = company_map.get(ticker, ticker)
    q_map = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth'}
    query = f'{company} {q_map[quarter]} quarter {year} earnings results'
    try:
        df = _CRAWLER_FN([query], 5)
        if df is not None and not df.empty and 'body' in df.columns:
            return ' '.join(df['body'].astype(str).tolist())
    except Exception as e:
        log.debug(f'  crawler call failed: {e}')
    return ''


def build_sentiment_panel(tickers: List[str], analyzer: FinBERTAnalyzer) -> pd.DataFrame:
    """
    Build a quarterly FinBERT sentiment panel for all tickers.

    Parameters
    ----------
    tickers : list of str
        Stock ticker symbols to score.
    analyzer : FinBERTAnalyzer
        Loaded FinBERT analyzer instance.

    Returns
    -------
    pd.DataFrame
        Columns: date, ticker, year, quarter, sentiment_score, text_length.
        One row per (ticker, quarter). Cached to parquet on first run.
    """
    cache_path = CFG.cache_dir / 'sentiment_panel.parquet'
    if cache_path.exists():
        log.info('  [cache hit] sentiment panel')
        return pd.read_parquet(cache_path)

    rows = []
    quarters = [(y, q) for y in range(2014, 2027) for q in range(1, 5)]
    month_map = {1: 3, 2: 6, 3: 9, 4: 12}

    for ticker in tickers:
        log.info(f'  Sentiment scoring: {ticker}')
        for year, q in quarters:
            text = fetch_earnings_text_edgar(ticker, year, q)
            if not text:
                text = fetch_earnings_text_crawler_fallback(ticker, year, q)
            if not text:
                continue

            score = analyzer.score_text(text)
            quarter_end = pd.Timestamp(f'{year}-{month_map[q]:02d}-28')
            rows.append({
                'date': quarter_end,
                'ticker': ticker,
                'year': year,
                'quarter': q,
                'sentiment_score': score,
                'text_length': len(text),
            })
            time.sleep(0.5)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_parquet(cache_path)
    return df


# ======================================================================
# 8. DATA ASSEMBLY
# ======================================================================
def build_master_dataset(tickers: List[str]) -> pd.DataFrame:
    """
    Assemble prices, earnings surprises, and macro variables into one panel.

    Parameters
    ----------
    tickers : list of str
        Stock ticker symbols to include.

    Returns
    -------
    pd.DataFrame
        Daily panel with columns: date, ticker, OHLCV, macro features,
        epsActual, epsEstimate, surprise_eps. Sorted by (ticker, date).
        Cached to parquet on first run.

    Raises
    ------
    RuntimeError
        If no price data could be fetched for any ticker.
    """
    log.info('Building master dataset...')
    cache_path = CFG.cache_dir / 'master_dataset.parquet'
    if cache_path.exists():
        log.info('  [cache hit] master dataset')
        return pd.read_parquet(cache_path)

    price_dfs = [fetch_stock_prices(t, CFG.start_date, CFG.end_date) for t in tickers]
    price_dfs = [d for d in price_dfs if not d.empty]
    if not price_dfs:
        raise RuntimeError('No price data fetched.')
    all_prices = pd.concat(price_dfs, ignore_index=True)

    earnings_dfs = []
    for t in tickers:
        e = fetch_earnings_surprise(t)
        if not e.empty:
            earnings_dfs.append(e)
    all_earnings = pd.concat(earnings_dfs, ignore_index=True) if earnings_dfs else pd.DataFrame()

    macro = fetch_macro_variables(CFG.start_date, CFG.end_date)

    log.info('Merging components...')
    df = all_prices.merge(macro, on='date', how='left')
    if not all_earnings.empty:
        df = df.merge(all_earnings, on=['date', 'ticker'], how='left')
        df['surprise_eps'] = df.groupby('ticker')['surprise_eps'].ffill()

    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    macro_cols = [c for c in (list(CFG.fred_series.keys()) + list(CFG.market_series.keys())) if c in df.columns]
    df[macro_cols] = df[macro_cols].ffill()
    n_before = len(df)
    df = df.dropna(subset=macro_cols)
    log.info('  dropna macro_cols : %d lignes supprimées (%d → %d)',
             n_before - len(df), n_before, len(df))

    df.to_parquet(cache_path)
    log.info(f'  master dataset: {len(df):,} rows × {df.shape[1]} cols')
    return df


def attach_sentiment_to_panel(master: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill quarterly sentiment scores onto the daily master panel.

    Parameters
    ----------
    master : pd.DataFrame
        Daily panel returned by build_master_dataset.
    sentiment : pd.DataFrame
        Quarterly sentiment panel returned by build_sentiment_panel.

    Returns
    -------
    pd.DataFrame
        master with an added 'sentiment_score' column (float, default 0.0).
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


def assign_regime(date: pd.Timestamp) -> str:
    """
    Map a date to one of the four macro regimes defined in CFG.

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


# ======================================================================
# 9. MODELS
# ======================================================================
class SARIMAXModel:
    """
    Baseline SARIMAX(0, 2, 1) model per ticker.

    Wraps statsmodels SARIMAX with a sklearn-style fit/predict interface.
    Order follows the thesis methodology (d=2 for price-level stationarity).
    """

    def __init__(self, order=(0, 2, 1)):
        self.order = order
        self.results = None

    # noinspection PyPep8Naming
    def fit(self, y: pd.Series, X: pd.DataFrame) -> "SARIMAXModel":
        """
        Fit SARIMAX on training data.

        Parameters
        ----------
        y : pd.Series
            Target series (e.g. Close prices).
        X : pd.DataFrame
            Exogenous features aligned with y.

        Returns
        -------
        SARIMAXModel
            Fitted instance (for chaining).
        """
        self.results = SARIMAX(
            y, exog=X, order=self.order,
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
        return self

    # noinspection PyPep8Naming
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Forecast n steps ahead using test exogenous features.

        Parameters
        ----------
        X : pd.DataFrame
            Exogenous features for the forecast horizon (n rows).

        Returns
        -------
        np.ndarray
            Forecast values, shape (n,).
        """
        n = len(X)
        return self.results.forecast(steps=n, exog=X).values


class XGBoostModel:
    """
    XGBoost regressor with Optuna Bayesian hyperparameter optimisation.

    Uses an 80/20 internal time split for trial evaluation; best params are
    then used to retrain on the full training window.
    """

    def __init__(self):
        self.model = None
        self.best_params = None

    # noinspection PyPep8Naming
    @staticmethod
    def _objective(trial, X, y):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 600),
            'max_depth':        trial.suggest_int('max_depth', 3, 8),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
            'random_state':     42,
            'verbosity':        0,
            'tree_method':      'hist',
        }
        split = int(0.8 * len(X))
        m = xgb.XGBRegressor(**params)
        m.fit(X.iloc[:split], y.iloc[:split])
        pred = m.predict(X.iloc[split:])
        return float(np.sqrt(mean_squared_error(y.iloc[split:], pred)))

    # noinspection PyPep8Naming
    def fit(self, X: pd.DataFrame, y: pd.Series, n_trials: Optional[int] = None) -> "XGBoostModel":
        """
        Run Optuna study and fit final XGBoost on best hyperparameters.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series
            Target values.
        n_trials : int, optional
            Number of Optuna trials. Defaults to CFG.optuna_trials (50).

        Returns
        -------
        XGBoostModel
            Fitted instance (for chaining).
        """
        n_trials = n_trials or CFG.optuna_trials
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda t: self._objective(t, X, y), n_trials=n_trials, show_progress_bar=False)
        self.best_params = study.best_params
        self.model = xgb.XGBRegressor(**self.best_params, random_state=42, verbosity=0, tree_method='hist')
        self.model.fit(X, y)
        return self

    # noinspection PyPep8Naming
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions from the fitted XGBoost model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with the same columns as training data.

        Returns
        -------
        np.ndarray
            Predicted values, shape (n_samples,).
        """
        return self.model.predict(X)


class LSTMRegressor(nn.Module):
    """
    Two-layer LSTM with a linear output head for price regression.

    Parameters
    ----------
    n_features : int
        Number of input features per time step.
    hidden : int, optional
        Hidden state size, by default 64.
    """

    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM and linear head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, n_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch,).
        """
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMModel:
    """
    LSTM forecaster with sklearn-style interface, running on MPS or CPU.

    Handles sequence construction, feature scaling, and incremental
    fine-tuning across walk-forward windows.
    """

    def __init__(self, seq_len: Optional[int] = None,
                 hidden: Optional[int] = None,
                 epochs: Optional[int] = None):
        self.seq_len = seq_len or CFG.lstm_seq_len
        self.hidden = hidden or CFG.lstm_hidden
        self.epochs = epochs or CFG.lstm_epochs
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.net = None
        self.device = torch.device(CFG.device)

    # noinspection PyPep8Naming
    def _make_sequences(self, X, y):
        x_seq, y_seq = [], []
        for i in range(len(X) - self.seq_len):
            x_seq.append(X[i:i + self.seq_len])
            y_seq.append(y[i + self.seq_len])
        return np.asarray(x_seq), np.asarray(y_seq)

    # noinspection PyPep8Naming
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMModel":
        """
        Train the LSTM from scratch on the given window.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (standardised internally).
        y : pd.Series
            Target series (standardised internally).

        Returns
        -------
        LSTMModel
            Fitted instance (for chaining).
        """
        x_scaled = self.scaler_x.fit_transform(X.values)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
        x_seq, y_seq = self._make_sequences(x_scaled, y_scaled)

        x_tensor = torch.tensor(x_seq, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(self.device)
        loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=64, shuffle=True)

        self.net = LSTMRegressor(X.shape[1], self.hidden).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        self.net.train()
        for ep in range(self.epochs):
            losses = []
            for xb, yb in loader:
                opt.zero_grad()
                prediction = self.net(xb)
                loss = loss_fn(prediction, yb)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if (ep + 1) % 10 == 0:
                log.debug(f'  LSTM epoch {ep+1}/{self.epochs} | loss={np.mean(losses):.4f}')
        return self

    def fit_incremental(self, X: pd.DataFrame, y: pd.Series, epochs: int = 3) -> "LSTMModel":
        """
        Fine-tune the existing network on a new walk-forward window.

        Reuses current weights with a reduced learning rate (3e-4). Falls
        back to full training if the network has not been initialised yet.

        Parameters
        ----------
        X : pd.DataFrame
            New window feature matrix.
        y : pd.Series
            New window target series.
        epochs : int, optional
            Number of fine-tuning epochs, by default 3.

        Returns
        -------
        LSTMModel
            Updated instance (for chaining).
        """
        if self.net is None:
            return self.fit(X, y)

        x_scaled = self.scaler_x.transform(X.values)
        y_scaled = self.scaler_y.transform(y.values.reshape(-1, 1)).flatten()
        x_seq, y_seq = self._make_sequences(x_scaled, y_scaled)

        if len(x_seq) == 0:
            return self

        x_tensor = torch.tensor(x_seq, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(self.device)
        loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=64, shuffle=True)

        opt = torch.optim.Adam(self.net.parameters(), lr=3e-4)  # lr réduit pour fine-tuning
        loss_fn = nn.MSELoss()

        self.net.train()
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                opt.step()
        return self

    # noinspection PyPep8Naming
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for a feature matrix.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix; must have at least seq_len + 1 rows.

        Returns
        -------
        np.ndarray
            Predicted values (inverse-scaled), shape (n_samples,).
            First seq_len values are zero-padded.
        """
        x_scaled = self.scaler_x.transform(X.values)
        if len(x_scaled) <= self.seq_len:
            return np.zeros(len(x_scaled))
        x_seq = np.stack([x_scaled[i:i + self.seq_len] for i in range(len(x_scaled) - self.seq_len)])
        x_tensor = torch.tensor(x_seq, dtype=torch.float32).to(self.device)
        self.net.eval()
        with torch.no_grad():
            predictions = self.net(x_tensor).cpu().numpy()
        predictions = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        return np.concatenate([np.zeros(self.seq_len), predictions])


# ======================================================================
# 10. WALK-FORWARD VALIDATION
# ======================================================================
def walk_forward_validate(df: pd.DataFrame, ticker: str, model_class,
                          features: List[str], target: str = 'Close') -> pd.DataFrame:
    """
    Run walk-forward cross-validation for one ticker and one model class.

    Slides a training window of CFG.walk_forward_window days forward by
    CFG.walk_forward_step days at each step.

    Parameters
    ----------
    df : pd.DataFrame
        Master panel (all tickers combined).
    ticker : str
        Ticker to filter and validate.
    model_class : type
        One of SARIMAXModel, XGBoostModel, or LSTMModel.
    features : list of str
        Column names used as exogenous/feature inputs.
    target : str, optional
        Column name to forecast, by default 'Close'.

    Returns
    -------
    pd.DataFrame
        Columns: date, ticker, actual, predicted, model.
        Empty DataFrame if insufficient data.
    """
    sub = df[df['ticker'] == ticker].sort_values('date').reset_index(drop=True)
    sub = sub.dropna(subset=features + [target])
    n = len(sub)
    if n < CFG.walk_forward_window + CFG.walk_forward_step:
        return pd.DataFrame()

    records = []

    # LSTM warm-start : une seule instance réutilisée, fine-tuning incrémental
    lstm_instance = None

    # Calculer le nombre total de steps pour le logging
    total_steps = len(range(CFG.walk_forward_window, n - CFG.walk_forward_step, CFG.walk_forward_step))

    for idx, i in enumerate(range(CFG.walk_forward_window, n - CFG.walk_forward_step, CFG.walk_forward_step)):
        train = sub.iloc[i - CFG.walk_forward_window:i]
        test = sub.iloc[i:i + CFG.walk_forward_step]

        # Progress logging toutes les 5 itérations
        if idx % 5 == 0:
            log.info(f'    {model_class.__name__} [{ticker}] step {idx+1}/{total_steps}')

        try:
            if model_class is SARIMAXModel:
                m = model_class().fit(train[target], train[features])

            elif model_class is XGBoostModel:
                m = model_class().fit(train[features], train[target], n_trials=10)

            elif model_class is LSTMModel:
                if lstm_instance is None:
                    log.info(f'    LSTM [{ticker}] initial full training ({CFG.lstm_epochs} epochs)...')
                    lstm_instance = LSTMModel()
                    lstm_instance.fit(train[features], train[target])
                    log.info(f'    LSTM [{ticker}] initial training done, switching to incremental')
                else:
                    lstm_instance.fit_incremental(train[features], train[target], epochs=3)
                m = lstm_instance

            else:
                raise ValueError(f'Unknown model: {model_class}')

            pred = m.predict(test[features])
            actual = test[target].values
            for d, a, p in zip(test['date'].values, actual, pred):
                records.append({
                    'date': pd.Timestamp(d),
                    'ticker': ticker,
                    'actual': float(a),
                    'predicted': float(p),
                    'model': model_class.__name__,
                })
        except Exception as e:
            log.warning(f'  walk-forward fail {ticker} idx={i} {model_class.__name__}: {e}')

    return pd.DataFrame(records)

# ======================================================================
# 11. DRIFT DETECTION
# ======================================================================
def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Compute the Population Stability Index between two distributions.

    PSI < 0.1: no significant shift; 0.1–0.25: moderate; > 0.25: major.

    Parameters
    ----------
    expected : np.ndarray
        Reference (baseline) distribution samples.
    actual : np.ndarray
        Current distribution samples to compare.
    bins : int, optional
        Number of quantile-based bins, by default 10.

    Returns
    -------
    float
        PSI value (non-negative).
    """
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    e, _ = np.histogram(expected, bins=breakpoints)
    a, _ = np.histogram(actual, bins=breakpoints)
    e = np.where(e == 0, 1e-6, e) / max(len(expected), 1)
    a = np.where(a == 0, 1e-6, a) / max(len(actual), 1)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_test(expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
    """
    Kolmogorov-Smirnov 2-sample test for distribution equality.

    Parameters
    ----------
    expected : np.ndarray
        Reference distribution samples.
    actual : np.ndarray
        Current distribution samples.

    Returns
    -------
    tuple of (float, float)
        (statistic, p_value). Low p-value indicates significant drift.
    """
    stat, p = ks_2samp(expected, actual)
    return float(stat), float(p)


def page_hinkley_drift(series: np.ndarray, threshold: float = 50.0,
                       min_instances: int = 30) -> List[int]:
    """
    Detect gradual drift points using the Page-Hinkley test.

    Parameters
    ----------
    series : np.ndarray
        Univariate time series of residuals or errors.
    threshold : float, optional
        Detection threshold (higher = less sensitive), by default 50.0.
    min_instances : int, optional
        Minimum observations before drift can be flagged, by default 30.

    Returns
    -------
    list of int
        Indices where drift was detected.
    """
    ph = PageHinkley(threshold=threshold, min_instances=min_instances)
    drift_points = []
    for i, v in enumerate(series):
        ph.update(float(v))
        if ph.drift_detected:
            drift_points.append(i)
    return drift_points


def rolling_rmse(predictions: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    """
    Compute rolling RMSE per (model, ticker) pair.

    Parameters
    ----------
    predictions : pd.DataFrame
        Walk-forward predictions with columns: date, ticker, actual,
        predicted, model.
    window : int, optional
        Rolling window size in trading days, by default 90.

    Returns
    -------
    pd.DataFrame
        Columns: date, ticker, model, rolling_rmse.
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


def drift_panel_by_regime(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Compute PSI and KS drift metrics for each regime vs the baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Master panel containing the feature column and a 'date' column.
    feature : str
        Column name to analyse (e.g. 'fed_funds', 'sentiment_score').

    Returns
    -------
    pd.DataFrame
        Columns: feature, regime, psi, ks_stat, ks_p, n_baseline, n_actual.
        One row per non-baseline regime.
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
        ks_stat, ks_p = ks_test(baseline, actual)
        rows.append({
            'feature': feature, 'regime': regime,
            'psi': psi, 'ks_stat': ks_stat, 'ks_p': ks_p,
            'n_baseline': len(baseline), 'n_actual': len(actual),
        })
    return pd.DataFrame(rows)


# ======================================================================
# 12. SHAP EXPLAINABILITY OVER TIME
# ======================================================================
def compute_shap_by_regime(df: pd.DataFrame, ticker: str,
                           features: List[str], target: str = 'Close') -> pd.DataFrame:
    """
    Train one XGBoost per regime and compute mean |SHAP| feature importance.

    Falls back to permutation importance if SHAP TreeExplainer fails.

    Parameters
    ----------
    df : pd.DataFrame
        Master panel (all tickers).
    ticker : str
        Ticker to analyse.
    features : list of str
        Feature columns to include in XGBoost and SHAP.
    target : str, optional
        Target column, by default 'Close'.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, regime, feature, shap_importance.
        One row per (regime, feature) combination.
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
        m = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                             random_state=42, verbosity=0, tree_method='hist',
                             base_score=0.5)  # ← force base_score à un float simple
        m.fit(x_data, y_data)

        sample = x_data.iloc[:min(500, len(x_data))]

        try:
            # API moderne SHAP — contourne le bug de parsing TreeExplainer
            explainer = shap.Explainer(m.predict, sample)
            shap_vals = explainer(sample).values
        except Exception as e:
            log.warning(f'  SHAP fallback (permutation) for {ticker} {regime}: {e}')
            # Fallback ultime : permutation importance
            from sklearn.inspection import permutation_importance
            perm = permutation_importance(m, sample, y_data.iloc[:len(sample)],
                                          n_repeats=5, random_state=42, n_jobs=-1)
            mean_abs = perm.importances_mean
            for f, imp in zip(features, mean_abs):
                rows.append({'ticker': ticker, 'regime': regime,
                             'feature': f, 'shap_importance': float(abs(imp))})
            continue

        mean_abs = np.abs(shap_vals).mean(axis=0)
        for f, imp in zip(features, mean_abs):
            rows.append({'ticker': ticker, 'regime': regime,
                         'feature': f, 'shap_importance': float(imp)})
    return pd.DataFrame(rows)

# ======================================================================
# 13. VISUALIZATIONS
# ======================================================================
def plot_distribution_shift(df: pd.DataFrame, feature: str, save_as: str) -> None:
    """
    Plot KDE of feature distribution per macro regime.

    Parameters
    ----------
    df : pd.DataFrame
        Master panel with 'date' and the feature column.
    feature : str
        Column to visualise.
    save_as : str
        Filename (relative to CFG.figures_dir) for the saved PNG.
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


def plot_rolling_rmse(rmse_df: pd.DataFrame, save_as: str) -> None:
    """
    Plot 90-day rolling RMSE averaged across tickers for each model.

    Parameters
    ----------
    rmse_df : pd.DataFrame
        Output of rolling_rmse(): columns date, model, rolling_rmse.
    save_as : str
        Filename (relative to CFG.figures_dir) for the saved PNG.
    """
    avg = rmse_df.groupby(['date', 'model'])['rolling_rmse'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    for model in avg['model'].unique():
        m = avg[avg['model'] == model]
        ax.plot(m['date'], m['rolling_rmse'], label=model, linewidth=1.5)
    for _, (start, end) in CFG.regimes.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.07)
    ax.set_title('Rolling RMSE (90-day) — model degradation across regimes', fontsize=13)
    ax.set_xlabel('Date'); ax.set_ylabel('RMSE')
    ax.legend()
    plt.tight_layout()
    plt.savefig(CFG.figures_dir / save_as, dpi=150)
    plt.close()


def plot_shap_drift_heatmap(shap_df: pd.DataFrame, save_as: str) -> None:
    """
    Plot a heatmap of mean |SHAP| importance by (feature × regime).

    Parameters
    ----------
    shap_df : pd.DataFrame
        Output of compute_shap_by_regime(): columns feature, regime,
        shap_importance.
    save_as : str
        Filename (relative to CFG.figures_dir) for the saved PNG.
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


def plot_regime_performance_heatmap(rmse_df: pd.DataFrame, save_as: str) -> None:
    """
    Plot a heatmap of average RMSE by (model × regime).

    Parameters
    ----------
    rmse_df : pd.DataFrame
        Output of rolling_rmse(): columns date, model, ticker, rolling_rmse.
    save_as : str
        Filename (relative to CFG.figures_dir) for the saved PNG.
    """
    df = rmse_df.copy()
    df['regime'] = df['date'].apply(assign_regime)
    pivot = df.groupby(['model', 'regime'])['rolling_rmse'].mean().unstack()
    cols = [c for c in CFG.regimes.keys() if c in pivot.columns]
    pivot = pivot[cols]
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(pivot, cmap='RdYlGn_r', annot=True, fmt='.2f',
                cbar_kws={'label': 'Avg RMSE'}, ax=ax)
    ax.set_title('Model × Regime performance matrix', fontsize=13)
    plt.tight_layout()
    plt.savefig(CFG.figures_dir / save_as, dpi=150)
    plt.close()


# ======================================================================
# 14. MAIN PIPELINE
# ======================================================================
def main() -> None:
    """Run the full drift detection pipeline end to end."""
    log.info('=' * 70)
    log.info('ML DRIFT DETECTION PIPELINE — AI INFRASTRUCTURE STOCKS')
    log.info('=' * 70)
    log.info(f'  Tickers: {", ".join(CFG.tickers)}')
    log.info(f'  Period:  {CFG.start_date} → {CFG.end_date}')
    log.info(f'  Device:  {CFG.device}')
    log.info(f'  Crawler: {"available" if _CRAWLER_FN else "not available"}')

    # --- Step 1: master dataset ---
    log.info('\n[1/6] Building master dataset...')
    master = build_master_dataset(CFG.tickers)

    # --- Step 2: sentiment ---
    log.info('\n[2/6] Computing sentiment scores...')
    analyzer = FinBERTAnalyzer()
    sentiment_panel = build_sentiment_panel(CFG.tickers, analyzer)
    master = attach_sentiment_to_panel(master, sentiment_panel)

    # --- Step 3: feature set ---
    feature_cols = (
        list(CFG.fred_series.keys())
        + list(CFG.market_series.keys())
        + ['surprise_eps', 'sentiment_score']
    )
    feature_cols = [c for c in feature_cols if c in master.columns]
    master[feature_cols] = master[feature_cols].fillna(0)
    log.info(f'  Features: {feature_cols}')

    # --- Step 4: walk-forward predictions ---
    log.info('\n[3/6] Walk-forward validation...')
    predictions_all = []
    for ticker in CFG.tickers:
        log.info(f'  Ticker: {ticker}')
        for ModelCls in [SARIMAXModel, XGBoostModel, LSTMModel]:
            preds = walk_forward_validate(master, ticker, ModelCls, feature_cols)
            if not preds.empty:
                predictions_all.append(preds)
                log.info(f'    {ModelCls.__name__}: {len(preds)} predictions')

    predictions = pd.concat(predictions_all, ignore_index=True) if predictions_all else pd.DataFrame()
    if not predictions.empty:
        predictions.to_parquet(CFG.output_dir / 'predictions.parquet')

    # --- Step 5: drift detection ---
    log.info('\n[4/6] Drift detection...')
    drift_results = []
    for feat in ['sentiment_score', 'fed_funds', 'vix']:
        if feat in master.columns:
            drift_results.append(drift_panel_by_regime(master, feat))
    drift_df = pd.concat(drift_results, ignore_index=True) if drift_results else pd.DataFrame()
    if not drift_df.empty:
        drift_df.to_csv(CFG.output_dir / 'drift_metrics.csv', index=False)
        log.info(f'\n{drift_df.to_string(index=False)}')

    rmse_df = rolling_rmse(predictions, window=90) if not predictions.empty else pd.DataFrame()
    if not rmse_df.empty:
        rmse_df.to_parquet(CFG.output_dir / 'rolling_rmse.parquet')

    # --- Step 6: SHAP ---
    log.info('\n[5/6] SHAP explainability...')
    shap_results = []
    for ticker in CFG.tickers:
        sd = compute_shap_by_regime(master, ticker, feature_cols)
        if not sd.empty:
            shap_results.append(sd)
    shap_df = pd.concat(shap_results, ignore_index=True) if shap_results else pd.DataFrame()
    if not shap_df.empty:
        shap_df.to_csv(CFG.output_dir / 'shap_importance.csv', index=False)

    # --- Step 7: visualizations ---
    log.info('\n[6/6] Generating visualizations...')
    if 'sentiment_score' in master.columns:
        plot_distribution_shift(master, 'sentiment_score', 'fig1_sentiment_drift.png')
    if 'fed_funds' in master.columns:
        plot_distribution_shift(master, 'fed_funds', 'fig2_fedfunds_drift.png')
    if not rmse_df.empty:
        plot_rolling_rmse(rmse_df, 'fig3_rolling_rmse.png')
        plot_regime_performance_heatmap(rmse_df, 'fig5_regime_performance.png')
    if not shap_df.empty:
        plot_shap_drift_heatmap(shap_df, 'fig4_shap_drift.png')

    log.info('\n' + '=' * 70)
    log.info('PIPELINE COMPLETE.')
    log.info(f'  Output dir:  {CFG.output_dir.resolve()}')
    log.info(f'  Figures:     {CFG.figures_dir.resolve()}')
    log.info('=' * 70)


if __name__ == '__main__':
    main()