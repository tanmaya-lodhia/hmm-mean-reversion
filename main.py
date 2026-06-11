"""
HMM regime-tagged mean reversion short.

Shorts Russell 2000 stocks up 10%+ intraday with no earnings nearby,
entering at the signal-day close. Tags every trade with the SPX regime
(bull / high_vol / bear) from a walk-forward HMM and splits performance
by regime.

Usage: python -X utf8 main.py [--start YYYY-MM-DD --end YYYY-MM-DD]
"""

import argparse
import pandas as pd

from config import START_DATE, END_DATE, SPX_TRAIN_START
from data import fetch_spx, build_features
from regime import RegimeHMM, walk_forward_regimes
from universe import (
    get_russell2000_tickers,
    load_price_cache, save_price_cache, download_price_data,
    load_earnings_cache, fetch_earnings_dates,
)
from backtest import (
    find_daily_gainers, run_all_trades,
    stats_by_regime, print_stats_table,
)
from plot import plot_all


def main(start=START_DATE, end=END_DATE):
    print(f"\n{'='*62}")
    print(f"  HMM Regime-Tagged Mean Reversion Short")
    print(f"  Period: {start}  →  {end}")
    print(f"{'='*62}")

    # 1. SPX + walk-forward HMM regime labels
    # SPX history starts at SPX_TRAIN_START so the first refit has several
    # years of training data. Each block of trading days is labelled by an
    # HMM fit only on data BEFORE that block (out-of-sample labels).
    print(f"\n[1/5] Walk-forward HMM on SPX (training from {SPX_TRAIN_START})...")
    spx_raw   = fetch_spx(SPX_TRAIN_START, end)
    X, spx_all = build_features(spx_raw)

    first_oos_idx = int((spx_all.index < start).sum())
    print(f"      {first_oos_idx} training days before {start}, "
          f"{len(spx_all) - first_oos_idx} OOS days to label.")

    regime_raw = walk_forward_regimes(X, spx_all.index, first_oos_idx)

    # Diagnostic: model fit on pre-backtest training data only
    hmm_train = RegimeHMM().fit(X[:first_oos_idx])
    print(f"\n  Training-period regime summary (fit ≤ {start}):")
    print(hmm_train.summary(X[:first_oos_idx]))
    print(f"\n  Training-period transition matrix:")
    print(hmm_train.transition_matrix())

    print(f"\n  Walk-forward (OOS) regime distribution:")
    print(regime_raw.value_counts().to_string())

    # OOS slice of SPX for the chart, with unlagged labels
    spx_df = spx_all.iloc[first_oos_idx:].copy()
    spx_df["regime"] = regime_raw

    # Lag regime by 1 day so each signal date gets yesterday's known regime
    regime_lagged = regime_raw.shift(1)
    regime_lookup = {
        ts: val
        for ts, val in regime_lagged.items()
        if pd.notna(val)
    }

    # 2. Russell 2000 price data
    print("\n[2/5] Loading Russell 2000 price data...")
    tickers    = get_russell2000_tickers()
    price_data = load_price_cache()
    if price_data is None:
        price_data = download_price_data(tickers, start, end)
        if not price_data:
            print("ERROR: No price data downloaded. Check your internet connection.")
            return
        save_price_cache(price_data)

    # 3. Earnings dates
    print("\n[3/5] Earnings date cache...")
    earnings_cache = load_earnings_cache()
    earnings_cache = fetch_earnings_dates(tickers, existing_cache=earnings_cache)

    # 4. Signals + trades
    print("\n[4/5] Finding gainer signals and simulating trades...")
    all_index    = sorted(set(idx for df in price_data.values() for idx in df.index))
    trading_days = [
        d for d in all_index
        if start <= d.strftime("%Y-%m-%d") <= end
    ]
    print(f"      {len(trading_days)} trading days in range.")

    daily_gainers = find_daily_gainers(price_data, trading_days)
    total_signals = sum(len(v) for v in daily_gainers.values())
    print(f"      {total_signals} raw gainer signals across {len(daily_gainers)} days.")

    trades = run_all_trades(daily_gainers, price_data, earnings_cache, regime_lookup)

    if not trades:
        print("\n  No trades generated. "
              "Try lowering MIN_GAIN_PCT or MIN_DOLLAR_VOLUME in config.py.")
        return

    df = pd.DataFrame(trades)
    df.to_csv("trades.csv", index=False)
    print(f"      Saved → trades.csv")

    # 5. Results
    print("\n[5/5] Performance by regime...")
    stats = stats_by_regime(df)

    for label in ["ALL", "bull", "high_vol", "bear"]:
        if label in stats:
            print_stats_table(label, stats[label])

    # Regime distribution of trades
    print(f"\n  Regime distribution of trades:")
    counts = df["regime"].value_counts()
    for r, c in counts.items():
        print(f"    {r:<10} {c:>5}  ({c/len(df)*100:.1f}%)")

    plot_all(df, spx_df, stats, start, end)

    print(f"\n{'='*62}")
    print(f"  Complete. See results.png and trades.csv.")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HMM regime-tagged mean reversion short on Russell 2000"
    )
    parser.add_argument("--start", default=START_DATE, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=END_DATE,   help="End date YYYY-MM-DD")
    args = parser.parse_args()
    main(start=args.start, end=args.end)
