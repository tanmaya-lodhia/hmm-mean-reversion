# analyze_trades.py — conditioning-feature analysis on existing trades.csv
#
# Enriches each trade with features computable from the cached price data
# (volume ratio, prior run-up, extension above 20d MA, hold days) and reports
# avg P&L / win rate / profit factor across buckets of each feature.
# Pure analysis — does not change the strategy.

import pickle
import numpy as np
import pandas as pd

from config import PRICE_CACHE


def profit_factor(pnl):
    gp = pnl[pnl > 0].sum()
    gl = abs(pnl[pnl <= 0].sum())
    return gp / gl if gl > 0 else float("inf")


def bucket_report(df, col, buckets, labels):
    """Print avg P&L, win rate, PF, n for each bucket of `col`."""
    df = df.copy()
    df["bucket"] = pd.cut(df[col], bins=buckets, labels=labels)
    print(f"\n  ── by {col} " + "─" * (48 - len(col)))
    print(f"  {'bucket':<22}{'n':>6}{'win%':>8}{'avgP&L':>9}{'PF':>7}")
    for b in labels:
        sub = df[df["bucket"] == b]
        if len(sub) == 0:
            print(f"  {b:<22}{0:>6}")
            continue
        pnl = sub["pnl_pct"]
        print(f"  {b:<22}{len(sub):>6}{(sub['win'].mean()*100):>7.1f}%"
              f"{pnl.mean():>8.2f}%{profit_factor(pnl):>7.2f}")


def main():
    trades = pd.read_csv("trades.csv", parse_dates=["signal_date", "exit_date"])
    with open(PRICE_CACHE, "rb") as f:
        price_data = pickle.load(f)

    print(f"{len(trades)} trades loaded.")

    # ── Enrich with price-history features ─────────────────────
    vol_ratio  = []   # signal-day volume / 20d avg volume (excl. signal day)
    runup_5d   = []   # % gain over the 5 sessions before signal day
    ext_20dma  = []   # signal-day close vs 20d MA (%)
    hold_days  = []

    for _, t in trades.iterrows():
        df = price_data.get(t["ticker"])
        feats = (np.nan, np.nan, np.nan)
        if df is not None and t["signal_date"] in df.index:
            i = df.index.get_loc(t["signal_date"])
            if i >= 21:
                win20   = df.iloc[i-20:i]
                avg_vol = float(win20["Volume"].mean())
                v       = float(df.iloc[i]["Volume"])
                vr      = v / avg_vol if avg_vol > 0 else np.nan

                c_now  = float(df.iloc[i]["Close"])
                c_5ago = float(df.iloc[i-5]["Close"])
                ru     = (c_now / c_5ago - 1) * 100 if c_5ago > 0 else np.nan

                ma20 = float(win20["Close"].mean())
                ext  = (c_now / ma20 - 1) * 100 if ma20 > 0 else np.nan
                feats = (vr, ru, ext)
        vol_ratio.append(feats[0])
        runup_5d.append(feats[1])
        ext_20dma.append(feats[2])
        hold_days.append((t["exit_date"] - t["signal_date"]).days)

    trades["vol_ratio"] = vol_ratio
    trades["runup_5d"]  = runup_5d
    trades["ext_20dma"] = ext_20dma
    trades["hold_days"] = hold_days
    trades["dow"]       = trades["signal_date"].dt.day_name()

    n_feat = trades["vol_ratio"].notna().sum()
    print(f"{n_feat} trades enriched with price-history features.\n")
    print("=" * 62)
    print("  BASELINE")
    print("=" * 62)
    pnl = trades["pnl_pct"]
    print(f"  n={len(trades)}  win={trades['win'].mean()*100:.1f}%  "
          f"avg={pnl.mean():+.2f}%  PF={profit_factor(pnl):.2f}")

    # ── Feature buckets ─────────────────────────────────────────
    bucket_report(
        trades, "signal_gain_pct",
        buckets=[10, 12.5, 15, 20, 30, 1000],
        labels=["10-12.5%", "12.5-15%", "15-20%", "20-30%", ">30%"],
    )
    bucket_report(
        trades, "entry_price",
        buckets=[5, 10, 20, 50, 100000],
        labels=["$5-10", "$10-20", "$20-50", ">$50"],
    )
    bucket_report(
        trades, "vol_ratio",
        buckets=[0, 2, 5, 10, 20, 100000],
        labels=["<2x", "2-5x", "5-10x", "10-20x", ">20x"],
    )
    bucket_report(
        trades, "runup_5d",
        buckets=[-100, 0, 10, 25, 50, 100000],
        labels=["negative", "0-10%", "10-25%", "25-50%", ">50%"],
    )
    bucket_report(
        trades, "ext_20dma",
        buckets=[-100, 5, 15, 30, 50, 100000],
        labels=["<5%", "5-15%", "15-30%", "30-50%", ">50%"],
    )

    # Day of week
    print(f"\n  ── by signal day of week " + "─" * 30)
    print(f"  {'day':<22}{'n':>6}{'win%':>8}{'avgP&L':>9}{'PF':>7}")
    for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        sub = trades[trades["dow"] == d]
        if len(sub) == 0:
            continue
        pnl = sub["pnl_pct"]
        print(f"  {d:<22}{len(sub):>6}{(sub['win'].mean()*100):>7.1f}%"
              f"{pnl.mean():>8.2f}%{profit_factor(pnl):>7.2f}")

    # Hold duration
    print(f"\n  ── by hold duration (calendar days) " + "─" * 19)
    print(f"  {'days':<22}{'n':>6}{'win%':>8}{'avgP&L':>9}{'PF':>7}")
    for lo, hi, lab in [(0, 1, "1 day"), (2, 4, "2-4 days"),
                        (5, 9, "5-9 days"), (10, 99, "10+ days")]:
        sub = trades[(trades["hold_days"] >= lo) & (trades["hold_days"] <= hi)]
        if len(sub) == 0:
            continue
        pnl = sub["pnl_pct"]
        print(f"  {lab:<22}{len(sub):>6}{(sub['win'].mean()*100):>7.1f}%"
              f"{pnl.mean():>8.2f}%{profit_factor(pnl):>7.2f}")

    # ── Best/worst interaction: spike size × volume ratio ───────
    print(f"\n  ── spike size × volume ratio (avg P&L %, n) " + "─" * 10)
    t2 = trades.dropna(subset=["vol_ratio"]).copy()
    t2["spike_b"] = pd.cut(t2["signal_gain_pct"], [10, 15, 25, 1000],
                           labels=["10-15%", "15-25%", ">25%"])
    t2["vol_b"]   = pd.cut(t2["vol_ratio"], [0, 3, 10, 100000],
                           labels=["<3x", "3-10x", ">10x"])
    pivot_pnl = t2.pivot_table(values="pnl_pct", index="spike_b",
                               columns="vol_b", aggfunc="mean", observed=False)
    pivot_n   = t2.pivot_table(values="pnl_pct", index="spike_b",
                               columns="vol_b", aggfunc="count", observed=False)
    print("  avg P&L:")
    print(pivot_pnl.round(2).to_string())
    print("  n:")
    print(pivot_n.fillna(0).astype(int).to_string())

    trades.to_csv("trades_enriched.csv", index=False)
    print(f"\nSaved enriched trades → trades_enriched.csv")


if __name__ == "__main__":
    main()
