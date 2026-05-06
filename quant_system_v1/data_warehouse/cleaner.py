"""Data cleaning: handle missing values, outliers, price adjustment alignment."""
import pandas as pd
import numpy as np


def clean_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[col] = df.groupby('ts_code')[col].transform(
                lambda x: x.replace(0, np.nan).ffill()
            )
    price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in df.columns]
    if price_cols:
        df = df.dropna(subset=price_cols, how='all')
    if 'pct_chg' in df.columns:
        df = df[df['pct_chg'].abs() < 21]
    return df


def align_prices(df: pd.DataFrame, price_adj: str = 'front') -> pd.DataFrame:
    return df
