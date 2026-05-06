"""Historical stress scenarios and custom stress testing."""
import pandas as pd
import numpy as np
from typing import Dict, List, Callable

SCENARIOS = {
    '2015_crash': ('2015-06-12', '2015-09-15', '2015股灾1.0+2.0，3个月跌45%'),
    '2016_circuit_breaker': ('2016-01-04', '2016-01-07', '熔断机制，4天跌15%'),
    '2018_bear': ('2018-01-01', '2018-12-31', '全年熊市，跌25%'),
    '2020_covid': ('2020-02-03', '2020-02-03', '疫情恐慌，单日3000+跌停'),
    '2024_microcap': ('2024-01-15', '2024-02-07', '微盘股流动性危机'),
}


class StressTester:
    def __init__(self, scenarios: dict = None):
        self.scenarios = scenarios or SCENARIOS

    def run_historical(self, backtest_fn: Callable, strategy_name: str) -> pd.DataFrame:
        results = []
        for name, (start, end, desc) in self.scenarios.items():
            try:
                r = backtest_fn(start_date=start, end_date=end)
                results.append({
                    'scenario': name, 'description': desc,
                    'start': start, 'end': end,
                    'total_return': r.get('total_return', 0),
                    'max_drawdown': r.get('max_drawdown', 0),
                    'sharpe_ratio': r.get('sharpe_ratio', 0),
                })
            except Exception as e:
                results.append({
                    'scenario': name, 'description': desc,
                    'start': start, 'end': end,
                    'total_return': None, 'max_drawdown': None,
                    'sharpe_ratio': None, 'error': str(e),
                })
        return pd.DataFrame(results)

    def run_custom(self, backtest_fn: Callable,
                   shocks: List[dict]) -> pd.DataFrame:
        results = []
        for shock in shocks:
            try:
                r = backtest_fn()
                r['scenario'] = shock['name']
                r['shock'] = shock['shock']
                results.append(r)
            except Exception as e:
                results.append({'scenario': shock['name'], 'error': str(e)})
        return pd.DataFrame(results)
