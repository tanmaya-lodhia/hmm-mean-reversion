import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from config import HMM_N_STATES, HMM_RANDOM_SEED, REFIT_EVERY


class RegimeHMM:
    """3-state Gaussian HMM on (log_return, rolling_vol). States named by
    mean return: highest = bull, middle = high_vol, lowest = bear."""

    def __init__(self, n_states=HMM_N_STATES, random_seed=HMM_RANDOM_SEED):
        self.n_states = n_states
        self.scaler   = StandardScaler()
        self.model    = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=random_seed,
        )
        self.state_map = {}

    def fit(self, X):
        self.model.fit(self.scaler.fit_transform(X))
        self._label_states(X)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def predict_named(self, X):
        return [self.state_map[s] for s in self.predict(X)]

    def regime_series(self, X, index):
        return pd.Series(self.predict_named(X), index=index, name="regime")

    def _label_states(self, X):
        raw    = self.predict(X)
        means  = {s: X[raw == s, 0].mean() for s in range(self.n_states)}
        ranked = sorted(means, key=means.get, reverse=True)
        self.state_map = {
            ranked[0]: "bull",
            ranked[1]: "high_vol",
            ranked[2]: "bear",
        }

    def summary(self, X):
        raw  = self.predict(X)
        rows = []
        for s in range(self.n_states):
            mask = raw == s
            rows.append({
                "state":       self.state_map[s],
                "count":       int(mask.sum()),
                "pct":         f"{mask.mean()*100:.1f}%",
                "mean_return": f"{X[mask, 0].mean()*100:.4f}%",
                "mean_vol":    f"{X[mask, 1].mean()*100:.4f}%",
            })
        return pd.DataFrame(rows).set_index("state")

    def transition_matrix(self):
        labels = [self.state_map[i] for i in range(self.n_states)]
        return pd.DataFrame(self.model.transmat_, index=labels, columns=labels).round(4)


def walk_forward_regimes(X, index, first_oos_idx, refit_every=REFIT_EVERY):
    """Label index[first_oos_idx:] out-of-sample: refit the HMM at the start
    of each block on all data before it, decode through the block, keep only
    the block's labels.

    The label for day t still uses day t's own return/vol, so trade
    attribution has to lag it by one day (done in main.py).
    """
    labels   = []
    n        = len(X)
    n_blocks = (n - first_oos_idx + refit_every - 1) // refit_every
    print(f"      Walk-forward: {n_blocks} refits "
          f"({refit_every} trading days each, expanding window)...")

    block_no = 0
    for t in range(first_oos_idx, n, refit_every):
        block_no += 1
        t_end = min(t + refit_every, n)

        hmm = RegimeHMM()
        hmm.fit(X[:t])
        labels.extend(hmm.predict_named(X[:t_end])[t:t_end])

        if block_no % 5 == 0 or t_end == n:
            print(f"        refit {block_no}/{n_blocks}  "
                  f"(train<={index[t-1].date()}, label->{index[t_end-1].date()})")

    return pd.Series(labels, index=index[first_oos_idx:], name="regime")
