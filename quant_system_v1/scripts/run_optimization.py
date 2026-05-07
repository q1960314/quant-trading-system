"""Walk-Forward optimization on board strategy."""
import sys, os, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
from optimizer.walk_forward import walk_forward
from backtest.vectorized_engine import VectorizedBacktestEngine
import strategy.board_strategy
from strategy import StrategyRegistry


def backtest_fn(params):
    engine = VectorizedBacktestEngine(
        initial_capital=params.get('initial_capital', 5000),
        start_date=params['start_date'], end_date=params['end_date'],
        max_hold_stocks=params.get('max_hold_stocks', 5),
        single_stock_position=params.get('single_stock_position', 0.15),
    )
    s = StrategyRegistry.get('打板策略')
    r = engine.run(s)
    return {'sharpe_ratio': r.sharpe_ratio, 'total_return': r.total_return,
            'max_drawdown': r.max_drawdown, 'total_trades': r.total_trades}


param_grid = {
    'max_hold_stocks': [3, 5],
    'single_stock_position': [0.10, 0.15, 0.20],
}

print("Walk-Forward Optimization: 打板策略")
print(f"Params: {param_grid}")
print(f"Train=2mo, Valid=1mo, 2024-10-01 ~ 2024-12-31")
results = walk_forward(param_grid, backtest_fn, start_date='2024-10-01',
                        end_date='2024-12-31', train_months=2, valid_months=1)
print(results.to_string())
results.to_csv('walk_forward_results.csv', index=False)
print("\nSaved: walk_forward_results.csv")

if not results.empty:
    best = results.loc[results['test_sharpe'].idxmax()]
    print(f"\nBest window: {best['train_start']}→{best['test_end']} Sharpe={best['test_sharpe']:.2f}")
    print(f"Best params: {best['best_params']}")
