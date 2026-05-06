"""Coarse grid search over parameter space."""
import itertools
import pandas as pd
import numpy as np
from typing import Dict, List, Callable
from utils.logger import get_logger

logger = get_logger("grid_search")


def grid_search(param_grid: Dict[str, List],
                backtest_fn: Callable[[Dict], dict],
                metric: str = 'sharpe_ratio',
                top_n: int = 10) -> pd.DataFrame:
    keys = list(param_grid.keys())
    results = []
    total = int(np.prod([len(v) for v in param_grid.values()]))
    for i, values in enumerate(itertools.product(*param_grid.values())):
        params = dict(zip(keys, values))
        if i % 10 == 0:
            logger.info(f"Grid search: {i+1}/{total}")
        try:
            metrics = backtest_fn(params)
            metrics['params'] = params
            results.append(metrics)
        except Exception as e:
            logger.warning(f"Grid search failed for {params}: {e}")
    df = pd.DataFrame(results)
    if metric in df.columns:
        df = df.sort_values(metric, ascending=False)
    return df.head(top_n)
