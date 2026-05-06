"""技术指标因子组: MACD/RSI/ATR/KDJ/CCI/MA交叉/布林"""
import pandas as pd; import numpy as np
from .registry import FactorRegistry, FactorBase

@FactorRegistry.register
class RSI14(FactorBase):
    name="rsi_14"; category="technical"; desc="14日RSI"
    def compute(self, df):
        if 'close' not in df.columns: return pd.Series(np.nan, index=df.index)
        delta = df.groupby('ts_code')['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - 100/(1+rs)

@FactorRegistry.register
class MACD(FactorBase):
    name="macd"; category="technical"; desc="MACD柱/收盘价"
    def compute(self, df):
        if 'close' not in df.columns: return pd.Series(np.nan, index=df.index)
        ema12 = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        ema26 = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
        dif = ema12 - ema26
        dea = dif.groupby(df['ts_code']).transform(lambda x: x.ewm(span=9, adjust=False).mean())
        return 2*(dif-dea) / df['close']

@FactorRegistry.register
class ATR14(FactorBase):
    name="atr_14"; category="technical"; desc="14日ATR/收盘价"
    def compute(self, df):
        if 'high' not in df.columns or 'low' not in df.columns: return pd.Series(np.nan, index=df.index)
        pre = df.groupby('ts_code')['close'].shift(1)
        tr = np.maximum(df['high']-df['low'],
               np.maximum(abs(df['high']-pre), abs(df['low']-pre)))
        return tr.rolling(14).mean() / df['close']

@FactorRegistry.register
class KDJ_K(FactorBase):
    name="kdj_k"; category="technical"; desc="KDJ-K值"
    def compute(self, df):
        if 'high' not in df.columns or 'low' not in df.columns: return pd.Series(np.nan, index=df.index)
        low9 = df.groupby('ts_code')['low'].transform(lambda x: x.rolling(9).min())
        high9 = df.groupby('ts_code')['high'].transform(lambda x: x.rolling(9).max())
        rsv = (df['close']-low9)/(high9-low9).replace(0,np.nan)*100
        return rsv.rolling(3).mean()  # K = RSV的3日均值

@FactorRegistry.register
class CCI14(FactorBase):
    name="cci_14"; category="technical"; desc="14日CCI"
    def compute(self, df):
        if 'high' not in df.columns or 'low' not in df.columns: return pd.Series(np.nan, index=df.index)
        tp = (df['high']+df['low']+df['close'])/3
        ma = tp.rolling(14).mean()
        md = tp.rolling(14).apply(lambda x: np.abs(x-x.mean()).mean())
        return (tp-ma)/(0.015*md.replace(0,np.nan))

@FactorRegistry.register
class MA5CrossMA20(FactorBase):
    name="ma5_cross_ma20"; category="technical"; desc="5/20均线偏离%"
    def compute(self, df):
        if 'close' not in df.columns: return pd.Series(np.nan, index=df.index)
        ma5 = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
        ma20 = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
        return (ma5/ma20.replace(0,np.nan)-1)*100

@FactorRegistry.register
class BollWidth(FactorBase):
    name="boll_width"; category="technical"; desc="布林带宽度"
    def compute(self, df):
        if 'close' not in df.columns: return pd.Series(np.nan, index=df.index)
        std20 = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).std())
        ma20 = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
        return std20/ma20.replace(0,np.nan)

# ---- v2 新增因子 ----

@FactorRegistry.register
class ADX14(FactorBase):
    name="adx_14"; category="technical"; desc="14日ADX趋向指标"
    def compute(self, df):
        if 'high' not in df.columns or 'low' not in df.columns: return pd.Series(np.nan, index=df.index)
        pre = df.groupby('ts_code')['close'].shift(1)
        tr = np.maximum(df['high'] - df['low'],
               np.maximum(abs(df['high'] - pre), abs(df['low'] - pre)))
        atr14 = tr.rolling(14).mean()
        up = df['high'] - df.groupby('ts_code')['high'].shift(1)
        dn = df.groupby('ts_code')['low'].shift(1) - df['low']
        plus_dm = np.where((up > dn) & (up > 0), up, 0)
        minus_dm = np.where((dn > up) & (dn > 0), dn, 0)
        plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / atr14.replace(0, np.nan)
        minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / atr14.replace(0, np.nan)
        dx = abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan) * 100
        return dx.rolling(14).mean()

@FactorRegistry.register
class BollPosition(FactorBase):
    name="boll_position"; category="technical"; desc="布林带位置(close-lower)/(upper-lower)"
    def compute(self, df):
        if 'close' not in df.columns: return pd.Series(np.nan, index=df.index)
        ma20 = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
        std20 = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).std())
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        return (df['close'] - lower) / (upper - lower).replace(0, np.nan)

@FactorRegistry.register
class MAConvergence(FactorBase):
    name="ma_convergence"; category="technical"; desc="均线粘合度(MA5/MA10/MA20/MA60极差/MA20)"
    def compute(self, df):
        if 'close' not in df.columns: return pd.Series(np.nan, index=df.index)
        g = df.groupby('ts_code')['close']
        ma5 = g.transform(lambda x: x.rolling(5).mean())
        ma10 = g.transform(lambda x: x.rolling(10).mean())
        ma20 = g.transform(lambda x: x.rolling(20).mean())
        ma60 = g.transform(lambda x: x.rolling(60).mean())
        mas = pd.concat([ma5, ma10, ma20, ma60], axis=1)
        rng = mas.max(axis=1) - mas.min(axis=1)
        return -rng / ma20.replace(0, np.nan)  # 负值=粘合

@FactorRegistry.register
class EMASlope(FactorBase):
    name="ema_slope"; category="technical"; desc="EMA12斜率(5日变动)"
    def compute(self, df):
        if 'close' not in df.columns: return pd.Series(np.nan, index=df.index)
        ema12 = df.groupby('ts_code')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        return ema12 / ema12.groupby(df['ts_code']).shift(5).replace(0, np.nan) - 1

@FactorRegistry.register
class WR14(FactorBase):
    name="wr_14"; category="technical"; desc="14日威廉指标"
    def compute(self, df):
        if 'high' not in df.columns or 'low' not in df.columns: return pd.Series(np.nan, index=df.index)
        h14 = df.groupby('ts_code')['high'].transform(lambda x: x.rolling(14).max())
        l14 = df.groupby('ts_code')['low'].transform(lambda x: x.rolling(14).min())
        return (h14 - df['close']) / (h14 - l14).replace(0, np.nan) * -100
