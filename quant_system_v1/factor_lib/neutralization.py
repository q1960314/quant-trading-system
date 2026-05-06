"""Factor neutralization: remove confounding effects from industry, size, Barra factors."""
import pandas as pd
import numpy as np


def neutralize_by_industry(factor_values, industry_map):
    df = pd.concat([factor_values.rename('factor'), industry_map.rename('industry')], axis=1)
    industry_mean = df.groupby('industry')['factor'].transform('mean')
    return df['factor'] - industry_mean


def neutralize_by_size(factor_values, market_cap):
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels required: pip install statsmodels")
    df = pd.concat([
        factor_values.rename('factor'), np.log(market_cap).rename('ln_cap'),
    ], axis=1).dropna()
    if len(df) < 10:
        return factor_values
    X = sm.add_constant(df['ln_cap'])
    y = df['factor']
    model = sm.OLS(y, X).fit()
    residuals = y - model.predict(X)
    return residuals


def neutralize_barra(factor_values, barra_factors):
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels required: pip install statsmodels")
    df = pd.concat([factor_values.rename('factor'), barra_factors], axis=1).dropna()
    if len(df) < 10:
        return factor_values
    y = df['factor']
    X = df.drop(columns=['factor'])
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return y - model.predict(X)
