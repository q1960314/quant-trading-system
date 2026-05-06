"""
Factor evaluation system: IC analysis, group backtest, decay, turnover.

Evaluates each factor across dimensions:
1. IC (Information Coefficient): predictive power for forward returns
2. ICIR: stability of IC over time
3. Group backtest: long-short performance by factor quintile
4. Decay: how factor performance decays with lag
5. Turnover: portfolio turnover within each quintile
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List
from utils.logger import get_logger

logger = get_logger("factor_eval")


@dataclass
class FactorEvalResult:
    factor_name: str
    factor_type: str
    ic_mean: float = 0.0
    ic_std: float = 0.0
    icir: float = 0.0
    ic_win_rate: float = 0.0
    top_ann_return: float = 0.0
    bottom_ann_return: float = 0.0
    long_short_sharpe: float = 0.0
    ic_after_neutral: float = 0.0
    eval_date: str = ""
    passed: bool = False

    def summary(self) -> str:
        return (
            f"{self.factor_name}: IC={self.ic_mean:.4f}, ICIR={self.icir:.2f}, "
            f"WinRate={self.ic_win_rate:.1%}, Top={self.top_ann_return:.1%}, "
            f"Bot={self.bottom_ann_return:.1%}, LS_Sharpe={self.long_short_sharpe:.2f}, "
            f"Pass={self.passed}"
        )


class FactorEvaluator:
    def __init__(self, min_obs=100, icir_threshold=0.3):
        self.min_obs = min_obs
        self.icir_threshold = icir_threshold

    def ic_analysis(self, factor_values: pd.Series,
                    forward_returns: pd.Series) -> dict:
        aligned = pd.concat([factor_values, forward_returns], axis=1).dropna()
        if len(aligned) < self.min_obs:
            return {'ic_mean': 0, 'ic_std': 0, 'icir': 0, 'ic_win_rate': 0}
        # Daily cross-sectional IC
        if isinstance(aligned.index, pd.MultiIndex):
            ic_series = aligned.groupby(level=0).apply(
                lambda g: g.iloc[:, 0].corr(g.iloc[:, 1])
            )
        else:
            ic_series = pd.Series([aligned.iloc[:, 0].corr(aligned.iloc[:, 1])])
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        icir = ic_mean / ic_std if ic_std > 0 else 0
        ic_win_rate = (ic_series > 0).mean()
        return {'ic_mean': ic_mean, 'ic_std': ic_std, 'icir': icir,
                'ic_win_rate': ic_win_rate, 'ic_series': ic_series}

    def ic_decay(self, factor_values: pd.Series,
                 returns: pd.Series,
                 lags: List[int] = None) -> dict:
        if lags is None:
            lags = [1, 3, 5, 10, 20]
        result = {}
        for lag in lags:
            fwd_ret = returns.shift(-lag)
            ic_info = self.ic_analysis(factor_values, fwd_ret)
            result[lag] = ic_info['ic_mean']
        return result

    def group_backtest(self, factor_values: pd.Series,
                       returns: pd.Series, n_groups: int = 5) -> dict:
        aligned = pd.concat([
            factor_values.rename('factor'), returns.rename('return'),
        ], axis=1).dropna()
        if len(aligned) < n_groups * self.min_obs:
            return {}
        if isinstance(aligned.index, pd.MultiIndex):
            aligned['group'] = aligned.groupby(level=0)['factor'].transform(
                lambda x: pd.qcut(x, n_groups, labels=False, duplicates='drop')
            )
            group_returns = aligned.groupby(['group', aligned.index.get_level_values(0)])['return'].mean()
        else:
            aligned['group'] = pd.qcut(aligned['factor'], n_groups, labels=False, duplicates='drop')
            group_returns = aligned.groupby('group')['return'].mean()
        group_ann = group_returns.groupby('group').mean() * 252
        return {
            'top_return': group_ann.iloc[-1] if len(group_ann) > 0 else 0,
            'bottom_return': group_ann.iloc[0] if len(group_ann) > 0 else 0,
            'group_returns': group_ann.to_dict(),
        }

    def evaluate(self, factor_name: str, factor_type: str,
                 factor_values: pd.Series, returns: pd.Series) -> FactorEvalResult:
        ic_info = self.ic_analysis(factor_values, returns.shift(-1))
        gb = self.group_backtest(factor_values, returns)
        result = FactorEvalResult(
            factor_name=factor_name, factor_type=factor_type,
            ic_mean=ic_info['ic_mean'], ic_std=ic_info['ic_std'],
            icir=ic_info['icir'], ic_win_rate=ic_info['ic_win_rate'],
            top_ann_return=gb.get('top_return', 0),
            bottom_ann_return=gb.get('bottom_return', 0),
            long_short_sharpe=self._ls_sharpe(factor_values, returns),
            eval_date=pd.Timestamp.now().strftime('%Y-%m-%d'),
            passed=ic_info['icir'] >= self.icir_threshold,
        )
        logger.info(result.summary())
        return result

    def _ls_sharpe(self, factor_values, returns):
        gb = self.group_backtest(factor_values, returns)
        if not gb:
            return 0.0
        top, bot = gb.get('top_return', 0), gb.get('bottom_return', 0)
        denom = max(abs(bot), 1e-6)
        return (top - bot) / denom
