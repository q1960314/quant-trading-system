"""
Alpha decay curve: measures how signal profitability erodes with execution delay.

Critical for short-term strategies (2-3 day holding). Quantifies the cost of
delayed execution from T-close signal to T+1 open, T+1 close, T+2 open, etc.
"""
import pandas as pd
import numpy as np


class AlphaDecayAnalyzer:
    def __init__(self, delays=(0, 1, 2, 3, 5)):
        self.delays = delays

    def analyze(self, signal_scores: pd.Series,
                daily_returns: pd.Series) -> dict:
        results = {}
        baseline = None
        for delay in self.delays:
            fwd_ret = daily_returns.shift(-delay)
            aligned = pd.concat([signal_scores, fwd_ret], axis=1).dropna()
            if len(aligned) < 10:
                results[delay] = {'mean_return': 0, 'hit_rate': 0, 'decay_pct': 0}
                continue
            top = aligned[aligned.iloc[:, 0] >= aligned.iloc[:, 0].quantile(0.8)]
            mean_ret = top.iloc[:, 1].mean()
            hit_rate = (top.iloc[:, 1] > 0).mean()
            if delay == 0:
                baseline = mean_ret
            decay = mean_ret / baseline if baseline and baseline != 0 else 1.0
            results[delay] = {
                'mean_return': mean_ret,
                'hit_rate': hit_rate,
                'decay_pct': decay,
            }
        return results

    def plot_data(self, results: dict) -> dict:
        return {d: r['decay_pct'] for d, r in results.items()}
