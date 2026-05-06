"""Strategy correlation analysis."""
import pandas as pd
import numpy as np


def strategy_correlation(daily_returns: pd.DataFrame) -> pd.DataFrame:
    return daily_returns.corr()


def diversification_ratio(corr_matrix: pd.DataFrame) -> float:
    n = len(corr_matrix)
    if n <= 1:
        return 1.0
    avg_corr = (corr_matrix.values.sum() - n) / (n * (n - 1))
    return 1.0 / max(1.0 + avg_corr * (n - 1), 0.01)
