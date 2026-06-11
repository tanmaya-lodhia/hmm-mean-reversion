# data.py  —  SPX price download and HMM feature construction

import numpy as np
import pandas as pd
import yfinance as yf

from config import HMM_VOL_WINDOW


def fetch_spx(start, end):
    """Download SPX daily OHLCV and compute log returns."""
    raw = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)
    # yfinance ≥0.2 returns MultiIndex columns when multi-ticker; flatten defensively
    raw.columns = [col[0] if isinstance(col, tuple) else col for col in raw.columns]
    raw.index   = pd.to_datetime(raw.index).tz_localize(None)
    raw["log_return"] = np.log(raw["Close"] / raw["Close"].shift(1))
    return raw.dropna(subset=["log_return"])


def build_features(df):
    """Feature matrix for the HMM: daily log return + rolling vol."""
    df = df.copy()
    df["rolling_vol"] = df["log_return"].rolling(HMM_VOL_WINDOW).std()
    df = df.dropna(subset=["rolling_vol"])
    X  = df[["log_return", "rolling_vol"]].values
    return X, df
