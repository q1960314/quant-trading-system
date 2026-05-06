"""板块+宏观因子组: 行业相对强度/资金流入/市场状态/利率"""
import pandas as pd; import numpy as np
from .registry import FactorRegistry, FactorBase

@FactorRegistry.register
class SectorRelativeStrength(FactorBase):
    name="sector_rs"; category="sector"; desc="板块相对强度(个股/板块涨幅比)"
    def compute(self, df):
        """需要有 sector_return 列，否则返回中性值"""
        if 'sector_return' in df.columns and 'pct_chg' in df.columns:
            return df['pct_chg'] - df['sector_return']
        return pd.Series(0, index=df.index)

@FactorRegistry.register
class IndustryFundRank(FactorBase):
    name="industry_fund_rank"; category="sector"; desc="行业资金流入排名(越小越好)"
    def compute(self, df):
        if 'industry_fund_rank' in df.columns: return -df['industry_fund_rank']  # 负号=高排名得分高
        return pd.Series(0, index=df.index)

@FactorRegistry.register
class MarketCapGroup(FactorBase):
    name="market_cap_group"; category="sector"; desc="市值分组(0小盘/1中盘/2大盘)"
    def compute(self, df):
        fmc = df['float_market_cap'] if 'float_market_cap' in df.columns else pd.Series(50, index=df.index)
        return pd.cut(fmc, bins=[0,50,200,99999], labels=[0,1,2]).astype(float)

@FactorRegistry.register
class IndexCorrelation(FactorBase):
    name="index_corr"; category="sector"; desc="个股与大盘20日相关性"
    def compute(self, df):
        if 'pct_chg' not in df.columns: return pd.Series(np.nan, index=df.index)
        return df.groupby('ts_code')['pct_chg'].transform(
            lambda x: x.rolling(20).corr(df.loc[x.index, 'pct_chg'].mean() if 'index_pct' not in df.columns else df['index_pct']))

@FactorRegistry.register
class MarketRegime(FactorBase):
    name="market_regime"; category="macro"; desc="市场状态: 1牛市/0震荡/-1熊市"
    def compute(self, df):
        """基于MA排列简单判断"""
        if 'close' not in df.columns or 'trade_date' not in df.columns: return pd.Series(0, index=df.index)
        ma5 = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
        ma20 = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
        regime = pd.Series(0, index=df.index)
        regime[ma5 > ma20 * 1.05] = 1
        regime[ma5 < ma20 * 0.95] = -1
        return regime

@FactorRegistry.register
class ShiborLevel(FactorBase):
    name="shibor_level"; category="macro"; desc="隔夜拆借利率水平"
    def compute(self, df):
        """需要外部注入宏观数据，此处返回中性"""
        return pd.Series(0.015, index=df.index)  # 默认1.5%

@FactorRegistry.register
class PmiTrend(FactorBase):
    name="pmi_trend"; category="macro"; desc="PMI趋势(>50扩张)"
    def compute(self, df):
        return pd.Series(0, index=df.index)  # 默认中性，需加载宏观数据后覆盖

# ---- v2 新增因子 ----

@FactorRegistry.register
class SectorRPS(FactorBase):
    name="sector_rps"; category="sector"; desc="板块RPS强度(20日涨幅排名)"
    def compute(self, df):
        if 'sector_return' not in df.columns: return pd.Series(0, index=df.index)
        return df.groupby('trade_date')['sector_return'].transform(
            lambda x: x.rank(pct=True)
        ) if 'trade_date' in df.columns else pd.Series(0.5, index=df.index)

@FactorRegistry.register
class SectorFundFlow(FactorBase):
    name="sector_fund_flow"; category="sector"; desc="板块资金流向(万元,正=流入)"
    def compute(self, df):
        if 'sector_fund_flow' in df.columns: return df['sector_fund_flow']
        return pd.Series(0, index=df.index)

@FactorRegistry.register
class IndexBreadth(FactorBase):
    name="index_breadth"; category="macro"; desc="指数宽度(收盘>MA20的比例)"
    def compute(self, df):
        if 'close' not in df.columns or 'trade_date' not in df.columns:
            return pd.Series(0.5, index=df.index)
        ma20 = df.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
        df['_above'] = (df['close'] > ma20).astype(int)
        result = df.groupby('trade_date')['_above'].transform('mean')
        df.drop(columns=['_above'], inplace=True, errors='ignore')
        return result

@FactorRegistry.register
class NewHighRatio(FactorBase):
    name="new_high_ratio"; category="sector"; desc="创20日新高比例"
    def compute(self, df):
        if 'high' not in df.columns: return pd.Series(0, index=df.index)
        h20 = df.groupby('ts_code')['high'].transform(lambda x: x.rolling(20).max())
        new_high = (df['high'] >= h20.shift(1)).astype(int)
        df['_new_high'] = new_high
        result = df.groupby('trade_date')['_new_high'].transform('mean') if 'trade_date' in df.columns else pd.Series(0, index=df.index)
        df.drop(columns=['_new_high'], inplace=True, errors='ignore')
        return result

@FactorRegistry.register
class LimitUpRatio(FactorBase):
    name="limit_up_ratio"; category="macro"; desc="全市场涨停占比(情绪)"
    def compute(self, df):
        if 'limit_status' not in df.columns or 'trade_date' not in df.columns:
            return pd.Series(0, index=df.index)
        return df.groupby('trade_date')['limit_status'].transform(
            lambda x: (x == 'U').sum() / max(len(x), 1)
        )

@FactorRegistry.register
class FundFlowStrength(FactorBase):
    name="fund_flow_strength"; category="sector"; desc="个股资金流向/流通市值"
    def compute(self, df):
        if 'net_amount' not in df.columns: return pd.Series(0, index=df.index)
        fmc = df['float_market_cap'] if 'float_market_cap' in df.columns else pd.Series(100, index=df.index)
        return df['net_amount'] / fmc.replace(0, np.nan) / 10000
