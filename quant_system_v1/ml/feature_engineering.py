"""Auto feature engineering: cross features, lag features, aggregation."""
import pandas as pd
import numpy as np
from itertools import combinations


def generate_cross_features(df, factor_cols, top_n=50):
    added = 0
    for c1, c2 in combinations(factor_cols[:min(len(factor_cols), 15)], 2):
        if added >= top_n:
            break
        name = f'{c1}_x_{c2}'
        df[name] = df[c1].fillna(0) * df[c2].fillna(0)
        added += 1
    return df


def generate_lag_features(df, factor_cols, lags=(1, 2, 5)):
    for col in factor_cols:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df.groupby('ts_code')[col].shift(lag)
    return df


def generate_agg_features(df, factor_cols):
    for col in factor_cols:
        if 'industry' in df.columns:
            df[f'{col}_ind_mean'] = df.groupby(['trade_date', 'industry'])[col].transform('mean')
    return df
