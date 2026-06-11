# backtest.py  —  signal generation, trade simulation, and per-regime statistics

import numpy as np
import pandas as pd

from config import (
    MIN_GAIN_PCT, MIN_PRICE, MIN_DOLLAR_VOLUME,
    TOP_N_GAINERS, MAX_HOLD_DAYS, STOP_LOSS_PCT, EARNINGS_WINDOW,
)


# ─────────────────────────────────────────────────────────────
# 1. Signal generation
# ─────────────────────────────────────────────────────────────

def find_daily_gainers(price_data, trading_days):
    """
    For each trading day, find stocks with intraday gain ≥ MIN_GAIN_PCT
    (measured open→close) that also clear the minimum price and volume hurdles.
    Returns {date: [list of gainer dicts]}, sorted by % gain descending, capped at TOP_N.
    """
    daily_gainers = {}
    for date in trading_days:
        gainers = []
        for ticker, df in price_data.items():
            if date not in df.index:
                continue
            row = df.loc[date]
            try:
                o = float(row["Open"])
                c = float(row["Close"])
                v = float(row["Volume"])
            except Exception:
                continue
            if o <= 0:
                continue
            pct        = (c - o) / o * 100
            dollar_vol = c * v
            if pct >= MIN_GAIN_PCT and c >= MIN_PRICE and dollar_vol >= MIN_DOLLAR_VOLUME:
                gainers.append({
                    "ticker":     ticker,
                    "date":       date,
                    "pct_change": round(pct, 2),
                    "close":      round(c, 4),
                })
        gainers.sort(key=lambda x: x["pct_change"], reverse=True)
        if gainers:
            daily_gainers[date] = gainers[:TOP_N_GAINERS]
    return daily_gainers


def near_earnings(ticker, date, earnings_cache, window_days=EARNINGS_WINDOW):
    """
    Return True if `date` is within ±window_days calendar days of any known
    earnings release for this ticker.
    """
    dates = earnings_cache.get(ticker, [])
    if not dates:
        return False
    d = pd.Timestamp(date).normalize()
    for ed in dates:
        if abs((d - pd.Timestamp(ed)).days) <= window_days:
            return True
    return False


# ─────────────────────────────────────────────────────────────
# 2. Trade simulation
# ─────────────────────────────────────────────────────────────

def simulate_trade(ticker, signal_date, price_data):
    """
    Entry:  signal-day Close (market-on-close order) — captures the full
            overnight/premarket fade before next-day open.
    Stop:   exit at stop_price if any subsequent bar's High ≥ entry × 1.15.
    Exit:   first bar whose close is higher than the prior close
            (prev_close anchored to entry price on the first bar).
    P&L:    (entry − exit) / entry × 100  (positive = profit for short).
    """
    df = price_data.get(ticker)
    if df is None:
        return None

    if signal_date not in df.index:
        return None

    entry_price = float(df.loc[signal_date, "Close"])
    if entry_price <= 0:
        return None

    future = df[df.index > signal_date].head(MAX_HOLD_DAYS + 1)
    if future.empty:
        return None

    stop_price  = entry_price * (1 + STOP_LOSS_PCT / 100)
    prev_close  = entry_price   # first comparison is vs. signal-day close
    exit_price  = None
    exit_date   = None
    exit_reason = "max_hold"

    for i in range(len(future)):
        row        = future.iloc[i]
        curr_high  = float(row["High"])
        curr_close = float(row["Close"])
        bar_date   = future.index[i].strftime("%Y-%m-%d")

        # Stop loss — checked against intraday high before close comparison
        if curr_high >= stop_price:
            exit_price  = stop_price
            exit_date   = bar_date
            exit_reason = "stop_loss"
            break

        if i >= MAX_HOLD_DAYS:
            exit_price  = curr_close
            exit_date   = bar_date
            exit_reason = "max_hold"
            break

        if curr_close > prev_close:
            exit_price  = curr_close
            exit_date   = bar_date
            exit_reason = "rebound"
            break

        prev_close = curr_close

    if exit_price is None or exit_price <= 0:
        return None

    pnl_pct = (entry_price - exit_price) / entry_price * 100

    return {
        "ticker":      ticker,
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "entry_date":  signal_date.strftime("%Y-%m-%d"),   # same day — MOC order
        "entry_price": round(entry_price, 4),
        "exit_price":  round(exit_price, 4),
        "exit_date":   exit_date,
        "exit_reason": exit_reason,
        "pnl_pct":     round(pnl_pct, 4),
        "win":         pnl_pct > 0,
    }


# ─────────────────────────────────────────────────────────────
# 3. Full backtest loop
# ─────────────────────────────────────────────────────────────

def run_all_trades(daily_gainers, price_data, earnings_cache, regime_lookup):
    """
    Iterate over all signal days and gainers. Apply earnings filter. Simulate each trade.
    Attach the lagged SPX regime label to each trade record.

    `regime_lookup` is a dict {Timestamp: regime_str} where regime is already
    lagged by 1 day relative to the signal date (no look-ahead).
    """
    trades           = []
    skipped_earnings = 0
    no_data          = 0

    for date in sorted(daily_gainers):
        regime = regime_lookup.get(pd.Timestamp(date), "unknown")

        for g in daily_gainers[date]:
            ticker = g["ticker"]

            if near_earnings(ticker, date, earnings_cache):
                skipped_earnings += 1
                continue

            trade = simulate_trade(ticker, date, price_data)
            if trade is None:
                no_data += 1
                continue

            trade["signal_gain_pct"] = g["pct_change"]
            trade["regime"]          = regime if (regime and str(regime) != "nan") else "unknown"
            trades.append(trade)

    print(f"      Trades simulated:           {len(trades)}")
    print(f"      Skipped (near earnings):    {skipped_earnings}")
    print(f"      Skipped (no future data):   {no_data}")
    return trades


# ─────────────────────────────────────────────────────────────
# 4. Statistics
# ─────────────────────────────────────────────────────────────

def _max_drawdown(pnl_series):
    """Peak-to-trough drawdown on a running cumulative P&L series."""
    cum  = pnl_series.cumsum()
    peak = cum.cummax()
    return float((cum - peak).min())


def compute_stats(subset):
    if subset is None or subset.empty:
        return None
    wins   = subset[subset["win"]]
    losses = subset[~subset["win"]]
    n      = len(subset)
    wr     = len(wins) / n * 100
    aw     = float(wins["pnl_pct"].mean())   if len(wins)   > 0 else 0.0
    al     = float(losses["pnl_pct"].mean()) if len(losses) > 0 else 0.0
    ap     = float(subset["pnl_pct"].mean())
    gross_profit = wins["pnl_pct"].sum()
    gross_loss   = abs(losses["pnl_pct"].sum())
    pf           = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    std    = float(subset["pnl_pct"].std())
    sharpe = ap / std if std > 0 else 0.0   # per-trade Sharpe (not annualised)
    mdd    = _max_drawdown(subset.sort_values("entry_date")["pnl_pct"])

    kelly = 0.0
    if al != 0 and wr > 0:
        W     = wr / 100
        R     = abs(aw / al)
        kelly = W - (1 - W) / R

    exit_reasons = subset["exit_reason"].value_counts().to_dict()

    return dict(
        n=n, wins=len(wins), losses=len(losses),
        win_rate=wr, avg_pnl=ap, avg_win=aw, avg_loss=al,
        profit_factor=pf, sharpe=sharpe, max_drawdown=mdd,
        kelly=kelly,
        total_pnl=float(subset["pnl_pct"].sum()),
        max_win=float(subset["pnl_pct"].max()),
        max_loss=float(subset["pnl_pct"].min()),
        exit_reasons=exit_reasons,
    )


def stats_by_regime(df):
    """Compute stats for ALL trades and per-regime subsets."""
    result = {"ALL": compute_stats(df)}
    for regime in ["bull", "high_vol", "bear", "unknown"]:
        sub = df[df["regime"] == regime]
        if not sub.empty:
            result[regime] = compute_stats(sub)
    return result


def print_stats_table(label, s):
    if not s:
        print(f"  {label}: no trades.")
        return
    print(f"\n  ── {label} ({s['n']} trades) " + "─" * 34)
    print(f"  {'Win rate':<38} {s['win_rate']:>8.1f}%")
    print(f"  {'Avg P&L per trade':<38} {s['avg_pnl']:>8.2f}%")
    print(f"  {'Total P&L (sum, no compounding)':<38} {s['total_pnl']:>8.2f}%")
    print(f"  {'Avg win':<38} {s['avg_win']:>8.2f}%")
    print(f"  {'Avg loss':<38} {s['avg_loss']:>8.2f}%")
    print(f"  {'Profit factor':<38} {s['profit_factor']:>8.2f}")
    print(f"  {'Per-trade Sharpe (not annualised)':<38} {s['sharpe']:>8.3f}")
    print(f"  {'Max drawdown (cum. P&L basis)':<38} {s['max_drawdown']:>8.2f}%")
    print(f"  {'Kelly criterion':<38} {s['kelly']:>8.2%}")
    print(f"  {'Best single trade':<38} {s['max_win']:>8.2f}%")
    print(f"  {'Worst single trade':<38} {s['max_loss']:>8.2f}%")
    reasons = s.get("exit_reasons", {})
    if reasons:
        print(f"  Exit breakdown:")
        for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {r:<18} {c:>4}  ({c/s['n']*100:.0f}%)")
