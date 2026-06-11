# universe.py  —  Russell 2000 tickers, price data, and earnings dates

import os
import time
import pickle
import urllib.request

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from config import (
    MAX_TICKERS, BATCH_SIZE, BATCH_PAUSE,
    PRICE_CACHE, EARNINGS_CACHE,
)


# Russell 2000 ticker list

_FALLBACK_TICKERS = [
    "ACAD","AEHR","AEIS","AGIO","AGYS","AAOI","ALRM","AMKR","AMPH","ANGI",
    "APPF","APPN","ARCT","AROW","ARWR","ATEC","ATNI","ATRC","AVNS","AXNX",
    "AXSM","BBIO","BCPC","BEAM","BHVN","BOOT","BPMC","BTAI","BYND","CARA",
    "CARS","CASH","CBPO","CCRN","CDMO","CELC","CHGG","CHWY","CLFD","CLNE",
    "CLRB","CLSK","CLOV","CMCO","CMPS","CNMD","COGT","COOP","CORT","CPRX",
    "CRDF","CRDO","CRSP","CRVL","CSWC","CSWI","CTMX","CUTR","CVAC","CVCO",
    "CVNA","DAWN","DOCN","DY","EGBN","ENSG","EPRT","ESNT","FFIN","FN",
    "FORM","FULT","GATX","GBCI","GKOS","GH","GVA","HQY","HL","HOMB",
    "IDCC","IBP","INDB","IONQ","JBTM","JXN","KRG","KTOS","KRYS","LUMN",
    "MDGL","MGY","MOGA","MOD","NE","NJR","NBHC","NBTB","NPO","NXT",
    "OFG","OKLO","ONB","ORA","PACW","PCVX","PL","PLXS","PNFP","POR",
    "PRAX","PRIM","PTGX","QBTS","RMBS","RHP","RIG","ROAD","SANM","SATS",
    "SITM","SLAB","SM","SMTC","SPXC","SR","STRL","SWX","TMHC","TEX",
    "TRNO","TTMI","TXNM","UBSI","UEC","UMBF","VAL","VLY","VSAT","VIAV",
    "WTS","ZWS","AGX","CNX","CNR","CMC","CDE","AROC","BE","ESE",
    "ENS","FSS","CWST","CYTK","EAT","FLR","GTLS","IBOC",
    "BTSG","CWAN","AHR","FCFS",
]


def get_russell2000_tickers():
    """Try iShares IWM holdings → GitHub list → hardcoded fallback."""
    try:
        url = (
            "https://www.ishares.com/us/products/239710/"
            "ishares-russell-2000-etf/1467271812596.ajax"
            "?fileType=csv&fileName=IWM_holdings&dataType=fund"
        )
        df = pd.read_csv(url, skiprows=9)
        tickers = df["Ticker"].dropna().tolist()
        tickers = [
            str(t).strip().replace(".", "-") for t in tickers
            if str(t).strip() and str(t).strip() != "nan" and len(str(t).strip()) <= 5
        ]
        if len(tickers) > 100:
            print(f"      {len(tickers)} tickers from iShares IWM.")
            return tickers[:MAX_TICKERS]
    except Exception:
        pass

    try:
        url = (
            "https://raw.githubusercontent.com/rreichel3/"
            "US-Stock-Symbols/main/russell2000/russell2000_tickers.txt"
        )
        with urllib.request.urlopen(url, timeout=10) as f:
            tickers = [line.decode().strip() for line in f if line.strip()]
        tickers = [t.replace(".", "-") for t in tickers if t]
        if len(tickers) > 100:
            print(f"      {len(tickers)} tickers from GitHub fallback.")
            return tickers[:MAX_TICKERS]
    except Exception:
        pass

    print("      Using hardcoded fallback list.")
    return _FALLBACK_TICKERS[:MAX_TICKERS]


# Price data (multi-year, cached)

def load_price_cache():
    if os.path.exists(PRICE_CACHE):
        try:
            with open(PRICE_CACHE, "rb") as f:
                data = pickle.load(f)
            print(f"      {len(data)} tickers loaded from {PRICE_CACHE}.")
            print(f"      Delete '{PRICE_CACHE}' to force a fresh download.")
            return data
        except Exception as e:
            print(f"      Cache load failed ({e}) — downloading fresh.")
    return None


def save_price_cache(price_data):
    with open(PRICE_CACHE, "wb") as f:
        pickle.dump(price_data, f)
    print(f"      Price cache saved to '{PRICE_CACHE}'.")


def download_price_data(tickers, start, end):
    try:
        from curl_cffi import requests as cffi_requests
        session     = cffi_requests.Session(impersonate="chrome")
        use_session = True
        print("      Using curl_cffi (browser impersonation).")
    except ImportError:
        session     = None
        use_session = False
        print("      curl_cffi not found — pip install curl_cffi to reduce rate limits.")

    batches    = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    price_data = {}

    print(f"      {len(tickers)} tickers in {len(batches)} batches of {BATCH_SIZE}.")
    for i, batch in enumerate(batches, 1):
        print(f"      Batch {i}/{len(batches)}...", end=" ", flush=True)
        try:
            kwargs = dict(
                start=start, end=end, interval="1d",
                group_by="ticker", auto_adjust=True,
                progress=False, threads=False,
            )
            if use_session:
                kwargs["session"] = session

            raw   = yf.download(batch, **kwargs)
            added = 0
            for ticker in batch:
                try:
                    df = raw[ticker].copy() if len(batch) > 1 else raw.copy()
                    df = df.dropna(subset=["Close"])
                    if not df.empty:
                        price_data[ticker] = df
                        added += 1
                except Exception:
                    pass
            print(f"{added} OK  (total {len(price_data)})")
        except Exception as e:
            print(f"FAILED: {e}")

        if i < len(batches):
            time.sleep(BATCH_PAUSE)

    print(f"      Downloaded {len(price_data)} / {len(tickers)} tickers.")
    return price_data


# Earnings dates (cached per ticker)

def load_earnings_cache():
    if os.path.exists(EARNINGS_CACHE):
        try:
            with open(EARNINGS_CACHE, "rb") as f:
                data = pickle.load(f)
            print(f"      Earnings dates loaded for {len(data)} tickers ({EARNINGS_CACHE}).")
            return data
        except Exception:
            pass
    return {}


def save_earnings_cache(cache):
    with open(EARNINGS_CACHE, "wb") as f:
        pickle.dump(cache, f)
    print(f"      Earnings cache saved ({len(cache)} tickers).")


def fetch_earnings_dates(tickers, existing_cache=None):
    """
    Download historical earnings dates from yfinance and cache them.
    Returns a dict: {ticker: [sorted list of tz-naive Timestamps]}.

    Limitation: yfinance typically covers only ~2 years of earnings history,
    so the catalyst filter may miss some pre-2023 earnings for this universe.
    """
    cache   = existing_cache or {}
    missing = [t for t in tickers if t not in cache]

    if not missing:
        print("      All earnings dates already cached.")
        return cache

    print(f"      Fetching earnings dates for {len(missing)} tickers "
          f"(~{len(missing)//2} seconds)...")
    for ticker in tqdm(missing, unit="ticker"):
        try:
            ed = yf.Ticker(ticker).earnings_dates
            if ed is not None and not ed.empty:
                # Normalize to tz-naive dates for straightforward comparison
                dates = pd.to_datetime(ed.index).tz_localize(None).normalize()
                cache[ticker] = sorted(set(dates.tolist()))
            else:
                cache[ticker] = []
        except Exception:
            cache[ticker] = []

    save_earnings_cache(cache)
    return cache
