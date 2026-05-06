"""
Factor synthesis: combine multiple factors into a composite.

Methods:
- Equal weight: mean of all factor z-scores
- IC-weighted: weight by historical |IC|
- Stepwise regression: forward selection of effective factors
"""
import pandas as pd
import numpy as np
from typing import Dict, List


def equal_weight(factor_dict: Dict[str, pd.Series]) -> pd.Series:
    composites = []
    for name, series in factor_dict.items():
        z = (series - series.mean()) / series.std()
        composites.append(z.rename(name))
    return pd.concat(composites, axis=1).mean(axis=1)


def ic_weighted(factor_dict: Dict[str, pd.Series],
                ic_dict: Dict[str, float]) -> pd.Series:
    total = sum(abs(v) for v in ic_dict.values())
    if total == 0:
        return equal_weight(factor_dict)
    result = pd.Series(0.0, index=next(iter(factor_dict.values())).index)
    for name, series in factor_dict.items():
        w = abs(ic_dict.get(name, 0)) / total
        z = (series - series.mean()) / series.std()
        result = result.add(z * w, fill_value=0)
    return result


def stepwise_selection(factor_dict: Dict[str, pd.Series],
                       forward_returns: pd.Series,
                       min_improvement: float = 0.001) -> List[str]:
    selected = []
    best_ic = 0.0
    remaining = set(factor_dict.keys())
    while remaining:
        best_candidate = None
        best_candidate_ic = best_ic
        for name in remaining:
            test_factors = selected + [name]
            composite = equal_weight({n: factor_dict[n] for n in test_factors})
            ic = composite.corr(forward_returns)
            if abs(ic) > best_candidate_ic:
                best_candidate_ic = abs(ic)
                best_candidate = name
        if best_candidate and (best_candidate_ic - best_ic) > min_improvement:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            best_ic = best_candidate_ic
        else:
            break
    return selected
