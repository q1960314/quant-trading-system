"""Walk-Forward Analysis: rolling train/validation to prevent overfitting."""
import pandas as pd
from typing import Dict, List, Callable
from dateutil.relativedelta import relativedelta
from .grid_search import grid_search
from utils.logger import get_logger

logger = get_logger("walk_forward")


def walk_forward(param_grid: Dict[str, List],
                 backtest_fn: Callable,
                 start_date: str,
                 end_date: str,
                 train_months: int = 12,
                 valid_months: int = 3,
                 metric: str = 'sharpe_ratio') -> pd.DataFrame:
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    results = []
    while current + relativedelta(months=train_months + valid_months) <= end:
        train_start = current.strftime('%Y-%m-%d')
        train_end = (current + relativedelta(months=train_months)).strftime('%Y-%m-%d')
        test_start = train_end
        test_end = (current + relativedelta(months=train_months + valid_months)).strftime('%Y-%m-%d')

        def bt_train(params):
            p = params.copy()
            p['start_date'] = train_start
            p['end_date'] = train_end
            return backtest_fn(p)

        top_params = grid_search(param_grid, bt_train, metric=metric, top_n=3)
        best_params = top_params.iloc[0]['params'] if len(top_params) > 0 else {}

        def bt_test(params):
            p = params.copy()
            p['start_date'] = test_start
            p['end_date'] = test_end
            return backtest_fn(p)

        test_result = bt_test(best_params)
        results.append({
            'train_start': train_start, 'train_end': train_end,
            'test_start': test_start, 'test_end': test_end,
            'best_params': best_params,
            'test_sharpe': test_result.get('sharpe_ratio', 0),
            'test_return': test_result.get('total_return', 0),
            'test_max_dd': test_result.get('max_drawdown', 0),
        })
        logger.info(f"WF {train_start}->{test_end}: Sharpe={test_result.get('sharpe_ratio', 0):.2f}")
        current += relativedelta(months=valid_months)
    return pd.DataFrame(results)
