# ─────────────────────────────────────────────────────────────
# config.py  —  all tunable parameters in one place
# ─────────────────────────────────────────────────────────────

# Backtest window
START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"

# ── Signal filters ────────────────────────────────────────────
MIN_GAIN_PCT      = 10.0      # minimum open→close intraday gain (%)
MIN_PRICE         = 5.0       # minimum close price ($)
MIN_DOLLAR_VOLUME = 500_000   # minimum dollar volume on signal day
TOP_N_GAINERS     = 10        # cap: top N gainers per day considered

# Catalyst filter (earnings proximity)
# Skip any stock whose signal date is within ±EARNINGS_WINDOW calendar days
# of a known earnings release. We use yfinance earnings_dates as the source.
# Limitation: non-earnings catalysts (FDA, M&A, analyst actions) are NOT filtered.
EARNINGS_WINDOW = 3

# ── Trade execution ───────────────────────────────────────────
MAX_HOLD_DAYS = 10     # forced exit after this many bars
STOP_LOSS_PCT = 15.0   # stop loss: exit if daily High ≥ entry × (1 + SL%)

# ── HMM ──────────────────────────────────────────────────────
HMM_N_STATES   = 3    # bull / high_vol / bear
HMM_VOL_WINDOW = 21   # rolling std window for volatility feature (trading days)
HMM_RANDOM_SEED = 42

# ── Universe ──────────────────────────────────────────────────
MAX_TICKERS = 400
BATCH_SIZE  = 50
BATCH_PAUSE = 5        # seconds between yfinance download batches

# ── Cache files (written into the project directory) ──────────
PRICE_CACHE    = "price_cache.pkl"
EARNINGS_CACHE = "earnings_cache.pkl"
