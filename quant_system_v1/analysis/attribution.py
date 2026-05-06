"""Performance attribution: Brinson and Fama-French decomposition."""
import pandas as pd
import numpy as np


def brinson_attribution(strategy_returns: pd.Series,
                        benchmark_returns: pd.Series,
                        strategy_weights: pd.DataFrame,
                        benchmark_weights: pd.DataFrame) -> dict:
    common_dates = strategy_returns.index.intersection(benchmark_returns.index)
    sr = strategy_returns[common_dates]
    br = benchmark_returns[common_dates]
    alloc_effect = ((strategy_weights - benchmark_weights) * br).sum(axis=1).mean()
    select_effect = (benchmark_weights * (sr - br)).sum(axis=1).mean()
    interact_effect = ((strategy_weights - benchmark_weights) * (sr - br)).sum(axis=1).mean()
    excess = sr.mean() - br.mean()
    return {
        'excess_return': excess,
        'allocation_effect': alloc_effect,
        'selection_effect': select_effect,
        'interaction_effect': interact_effect,
    }
