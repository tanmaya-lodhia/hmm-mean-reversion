# regime.py  —  Gaussian HMM wrapper for market regime detection

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from config import HMM_N_STATES, HMM_RANDOM_SEED


class RegimeHMM:
    """
    3-state Gaussian HMM fitted on (log_return, rolling_vol).
    States are labelled by mean log return: highest → 'bull', middle → 'high_vol',
    lowest → 'bear'.

    NOTE: the model is fit on the full backtest window (in-sample). The regime
    label used in trade attribution is lagged by one day in main.py to avoid
    any forward-looking bias in per-trade tagging.
    """

    def __init__(self, n_states=HMM_N_STATES, random_seed=HMM_RANDOM_SEED):
        self.n_states  = n_states
        self.scaler    = StandardScaler()
        self.model     = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=random_seed,
            verbose=False,
        )
        self.state_map = {}

    def fit(self, X):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        self._label_states(X)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def predict_named(self, X):
        return [self.state_map[s] for s in self.predict(X)]

    def regime_series(self, X, index):
        """Return a pd.Series of named regime labels aligned to `index`."""
        return pd.Series(self.predict_named(X), index=index, name="regime")

    def _label_states(self, X):
        raw   = self.predict(X)
        means = {s: X[raw == s, 0].mean() for s in range(self.n_states)}
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
        return pd.DataFrame(
            self.model.transmat_, index=labels, columns=labels
        ).round(4)
