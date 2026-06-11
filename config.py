# All tunable parameters in one place.

START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"

# Signal filters
MIN_GAIN_PCT      = 10.0      # minimum open->close intraday gain (%)
MIN_PRICE         = 5.0
MIN_DOLLAR_VOLUME = 500_000
TOP_N_GAINERS     = 10        # per day

# Skip stocks within +/- this many calendar days of an earnings release.
# Non-earnings catalysts (FDA, M&A, analyst actions) are NOT filtered.
EARNINGS_WINDOW = 3

# Trade execution
MAX_HOLD_DAYS = 10
STOP_LOSS_PCT = 15.0   # exit if daily High >= entry * (1 + SL%)

# HMM
HMM_N_STATES    = 3    # bull / high_vol / bear
HMM_VOL_WINDOW  = 21   # rolling std window (trading days)
HMM_RANDOM_SEED = 42

# Walk-forward: SPX history starts earlier than the backtest so the first
# refit has training data. Refit every REFIT_EVERY trading days on an
# expanding window; each block is labelled by a model fit only on data
# before that block.
SPX_TRAIN_START = "2015-01-01"
REFIT_EVERY     = 63   # ~quarterly

# Universe / downloads
MAX_TICKERS = 400
BATCH_SIZE  = 50
BATCH_PAUSE = 5

PRICE_CACHE    = "price_cache.pkl"
EARNINGS_CACHE = "earnings_cache.pkl"
