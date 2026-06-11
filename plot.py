# plot.py  —  multi-panel regime attribution chart

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

DARK_BG  = "#0d0d0d"
PANEL_BG = "#161616"

REGIME_COLORS = {
    "bull":     "#00e5ff",
    "high_vol": "#ff9f43",
    "bear":     "#ff4d6d",
    "unknown":  "#888888",
    "ALL":      "#a29bfe",
}


def _style(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.xaxis.label.set_color("#aaaaaa")
    ax.yaxis.label.set_color("#aaaaaa")
    ax.title.set_color("#eeeeee")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a2a")


def _equity_curve(ax, subset, color, label):
    """Draw cumulative P&L curve with shaded fill."""
    if subset.empty:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center",
                transform=ax.transAxes, color="#888888", fontsize=10)
        return
    s   = subset.sort_values("entry_date").reset_index(drop=True)
    cum = s["pnl_pct"].cumsum()
    ax.plot(s.index, cum, color=color, linewidth=1.6)
    ax.fill_between(s.index, cum, 0, where=cum >= 0,  alpha=0.12, color=color)
    ax.fill_between(s.index, cum, 0, where=cum <  0,  alpha=0.12, color="#ff4444")
    ax.axhline(0, color="#444444", linewidth=0.7, linestyle="--")
    final = float(cum.iloc[-1])
    ax.annotate(f"{final:+.1f}%", xy=(len(cum)-1, final),
                xytext=(5, 0), textcoords="offset points",
                color=color, fontsize=8, va="center")


def plot_all(df, spx_df, stats, start_date, end_date, out_file="results.png"):
    regimes = ["bull", "high_vol", "bear"]

    fig = plt.figure(figsize=(20, 17), dpi=150)
    fig.suptitle(
        f"HMM Regime-Tagged Mean Reversion Short  ·  {start_date} → {end_date}",
        fontsize=14, fontweight="bold", color="#ffffff", y=0.995,
    )
    fig.patch.set_facecolor(DARK_BG)
    gs = fig.add_gridspec(4, 3, hspace=0.55, wspace=0.38)

    # ── Row 0: per-regime equity curves ──────────────────────
    for col, regime in enumerate(regimes):
        ax    = fig.add_subplot(gs[0, col])
        _style(ax)
        color = REGIME_COLORS[regime]
        sub   = df[df["regime"] == regime]
        n     = len(sub)
        _equity_curve(ax, sub, color, regime)
        ax.set_title(f"Equity — {regime.replace('_', ' ').title()} ({n} trades)", fontsize=9)
        ax.set_xlabel("Trade #", fontsize=8)
        ax.set_ylabel("Cum. P&L %", fontsize=8)

    # ── Row 1 col 0: overall equity curve ────────────────────
    ax_all = fig.add_subplot(gs[1, 0])
    _style(ax_all)
    _equity_curve(ax_all, df, REGIME_COLORS["ALL"], "ALL")
    ax_all.set_title(f"Overall Equity ({len(df)} trades)", fontsize=9)
    ax_all.set_xlabel("Trade #", fontsize=8)
    ax_all.set_ylabel("Cum. P&L %", fontsize=8)

    # ── Row 1 col 1: win rate by regime ──────────────────────
    ax_wr = fig.add_subplot(gs[1, 1])
    _style(ax_wr)
    r_labels  = ["ALL"] + regimes
    win_rates = [
        stats[r]["win_rate"] if (r in stats and stats[r]) else 0
        for r in r_labels
    ]
    colors_wr = [REGIME_COLORS[r] for r in r_labels]
    bars      = ax_wr.bar(r_labels, win_rates, color=colors_wr, edgecolor=DARK_BG, width=0.55)
    ax_wr.axhline(50, color="#555555", linewidth=0.9, linestyle="--", label="50% breakeven")
    ax_wr.set_ylim(0, 112)
    ax_wr.set_title("Win Rate by Regime", fontsize=9)
    ax_wr.set_ylabel("Win Rate %", fontsize=8)
    ax_wr.legend(fontsize=7, labelcolor="#cccccc", framealpha=0)
    for bar, val in zip(bars, win_rates):
        ax_wr.text(
            bar.get_x() + bar.get_width() / 2, val + 1.5,
            f"{val:.1f}%", ha="center", color="#ffffff", fontsize=8, fontweight="bold",
        )

    # ── Row 1 col 2: avg P&L by regime ───────────────────────
    ax_ap = fig.add_subplot(gs[1, 2])
    _style(ax_ap)
    avg_pnls  = [
        stats[r]["avg_pnl"] if (r in stats and stats[r]) else 0
        for r in r_labels
    ]
    n_trades  = [
        stats[r]["n"] if (r in stats and stats[r]) else 0
        for r in r_labels
    ]
    colors_ap = ["#00cc66" if v > 0 else "#ff4444" for v in avg_pnls]
    ax_ap.bar(r_labels, avg_pnls, color=colors_ap, edgecolor=DARK_BG, width=0.55)
    ax_ap.axhline(0, color="#444444", linewidth=0.8, linestyle="--")
    ax_ap.set_title("Avg P&L per Trade by Regime", fontsize=9)
    ax_ap.set_ylabel("Avg P&L %", fontsize=8)
    for i, (v, n) in enumerate(zip(avg_pnls, n_trades)):
        offset = 0.04 if v >= 0 else -0.18
        ax_ap.text(i, v + offset, f"{v:+.2f}%\nn={n}",
                   ha="center", color="#ffffff", fontsize=7)

    # ── Row 2 cols 0-1: SPX with regime shading ──────────────
    ax_spx = fig.add_subplot(gs[2, 0:2])
    _style(ax_spx)
    if spx_df is not None and "regime" in spx_df.columns:
        spx_df = spx_df.copy()
        spx_df.index = pd.to_datetime(spx_df.index)
        price = spx_df["Close"].astype(float)
        ax_spx.plot(spx_df.index, price, color="#dddddd", linewidth=0.8, zorder=2)

        y_min, y_max = float(price.min()), float(price.max())
        for regime in regimes:
            color = REGIME_COLORS[regime]
            mask  = spx_df["regime"] == regime
            ax_spx.fill_between(
                spx_df.index, y_min, y_max,
                where=mask, alpha=0.15, color=color, zorder=1,
            )

        patches = [
            mpatches.Patch(color=REGIME_COLORS[r], alpha=0.5, label=r.replace("_", " ").title())
            for r in regimes
        ]
        ax_spx.legend(handles=patches, fontsize=7, labelcolor="#cccccc", framealpha=0)
        ax_spx.set_xlim(spx_df.index[0], spx_df.index[-1])
        ax_spx.set_ylim(y_min * 0.97, y_max * 1.03)
    ax_spx.set_title("SPX Price  ·  HMM Regime Background", fontsize=9)
    ax_spx.set_ylabel("SPX Close", fontsize=8)

    # ── Row 2 col 2: Sharpe by regime ────────────────────────
    ax_sh = fig.add_subplot(gs[2, 2])
    _style(ax_sh)
    sharpes    = [
        stats[r]["sharpe"] if (r in stats and stats[r]) else 0
        for r in r_labels
    ]
    colors_sh  = ["#00cc66" if v > 0 else "#ff4444" for v in sharpes]
    ax_sh.bar(r_labels, sharpes, color=colors_sh, edgecolor=DARK_BG, width=0.55)
    ax_sh.axhline(0, color="#444444", linewidth=0.8, linestyle="--")
    ax_sh.set_title("Per-Trade Sharpe by Regime", fontsize=9)
    ax_sh.set_ylabel("Sharpe", fontsize=8)
    for i, v in enumerate(sharpes):
        ax_sh.text(i, v + (0.002 if v >= 0 else -0.008),
                   f"{v:.3f}", ha="center", color="#ffffff", fontsize=8)

    # ── Row 3: P&L distributions per regime ──────────────────
    for col, regime in enumerate(regimes):
        ax    = fig.add_subplot(gs[3, col])
        _style(ax)
        color = REGIME_COLORS[regime]
        sub   = df[df["regime"] == regime]["pnl_pct"]
        if not sub.empty:
            ax.hist(sub, bins=25, color=color, edgecolor=DARK_BG, alpha=0.85)
            mv = float(sub.mean())
            ax.axvline(0,  color="#ff4444", linewidth=1.2, linestyle="--", label="Zero")
            ax.axvline(mv, color="#ffff66", linewidth=1.2, linestyle="--",
                       label=f"Avg {mv:+.2f}%")
            ax.legend(fontsize=7, labelcolor="#cccccc", framealpha=0)
        ax.set_title(f"P&L Dist — {regime.replace('_', ' ').title()}", fontsize=9)
        ax.set_xlabel("P&L %", fontsize=8)
        ax.set_ylabel("Frequency", fontsize=8)

    # ── Summary stats box ─────────────────────────────────────
    s = stats.get("ALL")
    if s:
        box_lines = [
            f"ALL  n={s['n']}",
            f"Win rate:     {s['win_rate']:.1f}%",
            f"Avg P&L:      {s['avg_pnl']:+.2f}%",
            f"Sharpe:       {s['sharpe']:.3f}",
            f"Profit factor:{s['profit_factor']:.2f}",
            f"Max DD:       {s['max_drawdown']:.2f}%",
            f"Kelly:        {s['kelly']:.2%}",
        ]
        fig.text(
            0.986, 0.015, "\n".join(box_lines),
            ha="right", va="bottom", color="#cccccc",
            fontsize=7.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1e1e1e", edgecolor="#444444"),
        )

    # ── Footnote ──────────────────────────────────────────────
    fig.text(
        0.012, 0.004,
        "Strategy: short Russell 2000 intraday gainers ≥10% with no recent earnings  |  "
        "Entry: next-day open  |  Stop: +20%  |  Exit: first up-close or 10-day max  |  "
        "Regime: 3-state HMM on SPX log return + 21d vol  |  Not financial advice.",
        fontsize=6.2, color="#666666", ha="left", va="bottom",
    )

    plt.savefig(out_file, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"  Chart saved → {out_file}")
