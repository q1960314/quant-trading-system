"""LightGBM-based next-day direction predictor."""
import os, sys, pickle
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from factor_lib.registry import FactorRegistry

logger = get_logger("ml")


class MLPredictor:
    def __init__(self, model_dir=None):
        base = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = model_dir or os.path.join(base, '..', 'evolution_output', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        self.model = None
        self.feature_names = []

    def prepare_data(self, df, factor_names=None):
        if factor_names is None:
            factor_names = FactorRegistry.list_all()
        factor_df = FactorRegistry.compute(df, factor_names)
        X = factor_df.dropna(axis=1, thresh=int(len(factor_df) * 0.5))
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        df['_fwd_ret'] = df.groupby('ts_code')['close'].transform(
            lambda x: x.pct_change().shift(-1)
        )
        y = (df['_fwd_ret'] > 0).astype(int)
        if self.feature_names is None or len(self.feature_names) == 0:
            self.feature_names = X.columns.tolist()
        return X, y

    def train(self, X, y, valid_frac=0.2):
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("lightgbm required: pip install lightgbm")
            return None
        split = int(len(X) * (1 - valid_frac))
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_valid, y_valid = X.iloc[split:], y.iloc[split:]
        self.model = lgb.LGBMClassifier(
            objective='binary', metric='auc', n_estimators=200,
            max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42, verbose=-1,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        auc = self.model.best_score_['valid_0']['auc']
        logger.info(f"LightGBM trained: AUC={auc:.4f}, features={len(self.feature_names)}")
        return auc

    def predict(self, X):
        if self.model is None:
            return pd.Series(0.5, index=X.index, name='ml_probability')
        # Align features with training columns
        X_aligned = pd.DataFrame(index=X.index)
        for col in self.feature_names:
            if col in X.columns:
                X_aligned[col] = X[col]
            else:
                X_aligned[col] = 0.0
        X_aligned = X_aligned.fillna(0)
        return pd.Series(self.model.predict_proba(X_aligned)[:, 1], index=X.index,
                         name='ml_probability')

    def save(self, version=None):
        if self.model is None:
            return
        v = version or pd.Timestamp.now().strftime('%Y%m%d')
        path = os.path.join(self.model_dir, f'model_{v}.pkl')
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'features': self.feature_names}, f)
        logger.info(f"Model saved: {path}")

    def load(self, version=None):
        if version is None:
            files = sorted([f for f in os.listdir(self.model_dir) if f.endswith('.pkl')])
            if not files:
                return False
            path = os.path.join(self.model_dir, files[-1])
        else:
            path = os.path.join(self.model_dir, f'model_{version}.pkl')
        if not os.path.exists(path):
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.feature_names = data['features']
        logger.info(f"Model loaded: {path}")
        return True
