# Drift Detection — Semi-conducteurs & IA Infrastructure

## But du projet
Détection de drift sur 10 actions AI-infra/semi-conducteurs (NVDA, MU, WDC, LRCX,
ASML, AMAT, VRT, EQIX, CIEN, TSM) via modèles hybrides walk-forward :
SARIMAX · XGBoost (Optuna) · LSTM (MPS/CPU).

NLP financier : FinBERT sur earnings press releases (SEC EDGAR 8-K).
4 méthodes de détection : PSI · KS-test · Page-Hinkley · Rolling RMSE (90j).
Période : 2014-01-01 → 2026-05-01 · 4 régimes macro.

## Stack

| Couche | Librairies |
|--------|------------|
| Data | pandas, numpy, yfinance, pandas-datareader |
| Stats / TS | statsmodels (SARIMAX), scipy, pmdarima |
| ML | xgboost, optuna, scikit-learn, river |
| Deep Learning | torch (MPS → CPU fallback), transformers (FinBERT) |
| Explainability | shap |
| Visualisation | matplotlib, seaborn |
| Config | python-dotenv, os.getenv |

## Commandes

```bash
# Activer l'environnement
source .venv/bin/activate

# Pipeline complet (fetch + modèles + drift + figures)
python drift_pipeline.py

# Pipeline léger : SHAP + visualisations seulement (requiert cache/ peuplé)
python script2.py

# Installer les dépendances
pip install -r requirements_drift.txt

# Formater / linter
ruff format .
ruff check .
```

## Variables d'environnement

Créer `.env` à la racine (jamais commité) :
```
FINNHUB_API_KEY=<votre_clé>
RISK_FREE_RATE=0.04
```

Chargé en tête de pipeline :
```python
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("FINNHUB_API_KEY")
```

## Conventions de code

### Typage — stricts
Type hints sur toutes les fonctions publiques, retour `None` explicite.

### Docstrings — format NumPy
```python
def calcul_sharpe(returns: pd.Series, rf: float = 0.04) -> float:
    """
    Calcule le ratio de Sharpe annualisé.

    Parameters
    ----------
    returns : pd.Series
        Log-returns quotidiens.
    rf : float, optional
        Taux sans risque annualisé, by default 0.04.

    Returns
    -------
    float
        Sharpe annualisé (×252).
    """
```

### Naming
- `snake_case` pour variables, fonctions, fichiers
- `UPPER_CASE` pour les constantes
- Préfixe `_` pour les helpers internes

### Logging — jamais de print()
```python
import logging
logger = logging.getLogger(__name__)
logger.info("message")  # jamais print()
```

### Clés API — jamais en dur
```python
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("FINNHUB_API_KEY")  # jamais "sk-..."
```

### Validation des données — avant toute analyse
```python
logger.info("Shape : %s", df.shape)
logger.info("dtypes : %s", df.dtypes.to_dict())
logger.info("NaN : %s", df.isna().sum().to_dict())
logger.info("Date range : %s → %s", df.index.min(), df.index.max())
```

### dropna() — jamais silencieux
```python
n_before = len(df)
df = df.dropna(subset=["close"])
logger.info("dropna : %d lignes supprimées (%d → %d)",
            n_before - len(df), n_before, len(df))
```

### Finance — calculs standardisés
```python
# Log returns
log_returns = np.log(prices / prices.shift(1))

# Annualisation ×252
TRADING_DAYS = 252
annual_vol    = daily_vol * np.sqrt(TRADING_DAYS)
annual_return = daily_return * TRADING_DAYS

# Sharpe — rf toujours explicite
RF = float(os.getenv("RISK_FREE_RATE", "0.04"))
sharpe = (annual_return - RF) / annual_vol
```
