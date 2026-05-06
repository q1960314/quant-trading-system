"""Bayesian optimization using Optuna TPE sampler."""
import pandas as pd
from typing import Dict, Callable
from utils.logger import get_logger

logger = get_logger("bayesian_opt")


def bayesian_optimize(param_space: Dict[str, tuple],
                      backtest_fn: Callable[[Dict], dict],
                      n_trials: int = 50,
                      direction: str = 'maximize',
                      metric: str = 'sharpe_ratio') -> Dict:
    try:
        import optuna
    except ImportError:
        logger.error("Optuna not installed. Run: pip install optuna")
        return {'best_params': {}, 'best_value': 0, 'study': None}

    def objective(trial):
        params = {}
        for name, spec in param_space.items():
            low, high = spec[0], spec[1]
            dtype = spec[2] if len(spec) > 2 else 'float'
            if dtype == 'int':
                params[name] = trial.suggest_int(name, int(low), int(high))
            elif dtype == 'float':
                params[name] = trial.suggest_float(name, low, high)
            elif dtype == 'categorical':
                params[name] = trial.suggest_categorical(name, high)
        result = backtest_fn(params)
        return result.get(metric, 0)

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials)
    return {'best_params': study.best_params, 'best_value': study.best_value, 'study': study}
