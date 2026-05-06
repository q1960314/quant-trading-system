"""
Train LightGBM model on 2022-2023 data, validate on 2024.

Usage: python scripts/train_ml_model.py
"""
import os, sys, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)
import warnings; warnings.filterwarnings('ignore')
import pandas as pd

from backtest.vectorized_engine import VectorizedBacktestEngine
from ml.predictor import MLPredictor


def main():
    # Load training data (2022-09 to 2023-12)
    print("Loading training data (2022-09 ~ 2023-12)...", flush=True)
    train_engine = VectorizedBacktestEngine(
        initial_capital=5000, start_date="2022-09-01", end_date="2023-12-31",
        max_hold_stocks=5,
    )
    train_engine.load_data()
    print(f"Training data: {len(train_engine.codes)} stocks x {len(train_engine.dates)} days")

    # Prepare features and labels
    print("Computing factors...", flush=True)
    predictor = MLPredictor()
    X, y = predictor.prepare_data(train_engine.df)
    print(f"Features: {X.shape[1]} columns, {X.shape[0]} rows")
    print(f"Target distribution: positive={y.sum()} ({y.mean()*100:.1f}%)")

    # Train
    print("Training LightGBM...", flush=True)
    t0 = time.time()
    auc = predictor.train(X, y)
    elapsed = time.time() - t0
    print(f"Training complete: AUC={auc:.4f}, time={elapsed:.1f}s")

    if auc:
        predictor.save("20260507")
        print("Model saved!")

        # Quick backtest on validation period
        print("\nValidating on 2024 Q4...", flush=True)
        val_engine = VectorizedBacktestEngine(
            initial_capital=5000, start_date="2024-10-08", end_date="2024-11-29",
            max_hold_stocks=5, single_stock_position=0.2,
        )
        val_engine.load_data()
        val_engine.constraint_matrix = val_engine.build_constraint_matrix()

        from strategy.ml_strategy import MLPredictionStrategy
        strategy = MLPredictionStrategy()
        # Force load the trained model
        strategy._predictor = predictor
        result = val_engine._run_with_data(strategy)
        print(f"ML Strategy: Return={result.total_return*100:+.2f}% "
              f"Sharpe={result.sharpe_ratio:.2f} Trades={result.total_trades} "
              f"Win={result.win_rate:.1f}%")
    else:
        print("Training failed!")


if __name__ == '__main__':
    main()
