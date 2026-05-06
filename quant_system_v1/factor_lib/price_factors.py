"""价量因子组: 动量/反转/波动率/振幅/量比"""
import pandas as pd; import numpy as np
from .registry import FactorRegistry, FactorBase

@FactorRegistry.register
class Momentum20d(FactorBase):
    name="momentum_20d"; category="price"; desc="20日收益率"
    def compute(self, df):
        if 'close' not in df.columns: return pd.Series(np.nan, index=df.index)
        return df.groupby('ts_code')['close'].transform(lambda x: x/x.shift(20)-1)

@FactorRegistry.register
class Momentum60d(FactorBase):
    name="momentum_60d"; category="price"; desc="60日收益率"
    def compute(self, df):
        if 'close' not in df.columns: return pd.Series(np.nan, index=df.index)
        return df.groupby('ts_code')['close'].transform(lambda x: x/x.shift(60)-1)

@FactorRegistry.register
class Reversal5d(FactorBase):
    name="reversal_5d"; category="price"; desc="5日反转"
    def compute(self, df):
        if 'close' not in df.columns: return pd.Series(np.nan, index=df.index)
        return -df.groupby('ts_code')['close'].transform(lambda x: x/x.shift(5)-1)

@FactorRegistry.register
class Volatility20d(FactorBase):
    name="volatility_20d"; category="price"; desc="20日波动率"
    def compute(self, df):
        if 'pct_chg' not in df.columns: return pd.Series(np.nan, index=df.index)
        return df.groupby('ts_code')['pct_chg'].transform(lambda x: x.rolling(20).std())

@FactorRegistry.register
class Amplitude14d(FactorBase):
    name="amplitude_14d"; category="price"; desc="14日振幅"
    def compute(self, df):
        if 'high' not in df.columns or 'low' not in df.columns: return pd.Series(np.nan, index=df.index)
        pre = df['pre_close'] if 'pre_close' in df.columns else df.groupby('ts_code')['close'].shift(1)
        return ((df['high'] - df['low']) / pre.replace(0, np.nan)).rolling(14).mean()

@FactorRegistry.register
class GapRatio(FactorBase):
    name="gap_ratio"; category="price"; desc="开盘缺口率"
    def compute(self, df):
        if 'open' not in df.columns: return pd.Series(np.nan, index=df.index)
        pre = df['pre_close'] if 'pre_close' in df.columns else df.groupby('ts_code')['close'].shift(1)
        return (df['open'] - pre) / pre.replace(0, np.nan)

@FactorRegistry.register
class VolRatio5d(FactorBase):
    name="vol_ratio_5d"; category="volume"; desc="5日量比"
    def compute(self, df):
        if 'vol' not in df.columns: return pd.Series(np.nan, index=df.index)
        ma = df.groupby('ts_code')['vol'].transform(lambda x: x.rolling(5).mean())
        return df['vol'] / ma.replace(0, np.nan)

@FactorRegistry.register
class AmountChange5d(FactorBase):
    name="amount_change_5d"; category="volume"; desc="成交额5日变化"
    def compute(self, df):
        if 'amount' not in df.columns: return pd.Series(np.nan, index=df.index)
        return df.groupby('ts_code')['amount'].transform(lambda x: x/x.shift(5)-1)

@FactorRegistry.register
class VolumeTrend(FactorBase):
    name="volume_trend"; category="volume"; desc="量价趋势"
    def compute(self, df):
        if 'vol' not in df.columns or 'close' not in df.columns: return pd.Series(np.nan, index=df.index)
        vc = df.groupby('ts_code')['vol'].transform(lambda x: x.diff(5))
        pc = df.groupby('ts_code')['close'].transform(lambda x: x.diff(5))
        return np.sign(vc) * np.sign(pc)

@FactorRegistry.register
class TurnoverRate(FactorBase):
    name="turnover_rate"; category="volume"; desc="换手率"
    def compute(self, df):
        if 'turnover_rate' in df.columns: return df['turnover_rate']
        if 'turnover_ratio' in df.columns: return df['turnover_ratio']
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register
class VolumeShrink(FactorBase):
    name="volume_shrink"; category="volume"; desc="缩量比(量/20日均量)"
    def compute(self, df):
        if 'vol' not in df.columns: return pd.Series(np.nan, index=df.index)
        ma = df.groupby('ts_code')['vol'].transform(lambda x: x.rolling(20).mean())
        return 1 - df['vol'] / ma.replace(0, np.nan)  # 正值=缩量

# ---- v2 新增因子 ----

@FactorRegistry.register
class IntradayAmplitude(FactorBase):
    name="intraday_amplitude"; category="price"; desc="日内振幅(high-low)/open"
    def compute(self, df):
        if 'high' not in df.columns or 'low' not in df.columns: return pd.Series(np.nan, index=df.index)
        return (df['high'] - df['low']) / df['open'].replace(0, np.nan)

@FactorRegistry.register
class HighLowPosition(FactorBase):
    name="high_low_position"; category="price"; desc="收盘在日内位置(close-low)/(high-low)"
    def compute(self, df):
        if 'high' not in df.columns or 'low' not in df.columns: return pd.Series(np.nan, index=df.index)
        rng = df['high'] - df['low']
        return (df['close'] - df['low']) / rng.replace(0, np.nan)

@FactorRegistry.register
class VolumePriceCorr(FactorBase):
    name="volume_price_corr"; category="volume"; desc="量价10日相关性"
    def compute(self, df):
        if 'close' not in df.columns or 'vol' not in df.columns: return pd.Series(np.nan, index=df.index)
        return df.groupby('ts_code').apply(
            lambda g: g['close'].rolling(10).corr(g['vol'])
        ).reset_index(level=0, drop=True)

@FactorRegistry.register
class ReturnSkewness(FactorBase):
    name="return_skewness"; category="price"; desc="20日收益率偏度"
    def compute(self, df):
        if 'pct_chg' not in df.columns: return pd.Series(np.nan, index=df.index)
        return df.groupby('ts_code')['pct_chg'].transform(lambda x: x.rolling(20).skew())

@FactorRegistry.register
class UpDaysRatio(FactorBase):
    name="up_days_ratio"; category="price"; desc="20日上涨天数占比"
    def compute(self, df):
        if 'pct_chg' not in df.columns: return pd.Series(np.nan, index=df.index)
        return df.groupby('ts_code')['pct_chg'].transform(
            lambda x: x.rolling(20).apply(lambda y: (y > 0).sum() / 20)
        )

@FactorRegistry.register
class DailyVolumeRatio(FactorBase):
    name="daily_volume_ratio"; category="volume"; desc="当日成交量/前日成交量"
    def compute(self, df):
        if 'vol' not in df.columns: return pd.Series(np.nan, index=df.index)
        prev = df.groupby('ts_code')['vol'].shift(1)
        return df['vol'] / prev.replace(0, np.nan)
