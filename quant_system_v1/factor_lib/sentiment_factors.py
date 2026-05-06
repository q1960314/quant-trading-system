"""情绪因子组: 机构/游资/北向/融资/涨停强度"""
import pandas as pd; import numpy as np
from .registry import FactorRegistry, FactorBase

@FactorRegistry.register
class InstNetBuy(FactorBase):
    name="inst_net_buy"; category="sentiment"; desc="机构净买入(万元)"
    def compute(self, df):
        if 'inst_buy' in df.columns and 'inst_sell' in df.columns:
            return df['inst_buy'] - df['inst_sell'].fillna(0)
        if 'inst_buy' in df.columns: return df['inst_buy']
        return pd.Series(0, index=df.index)

@FactorRegistry.register
class BoardHeight(FactorBase):
    name="board_height"; category="sentiment"; desc="连板高度"
    def compute(self, df):
        return df['up_down_times'] if 'up_down_times' in df.columns else pd.Series(0, index=df.index)

@FactorRegistry.register
class OrderStrength(FactorBase):
    name="order_strength"; category="sentiment"; desc="封单强度(封单/流通市值)"
    def compute(self, df):
        if 'order_amount' not in df.columns: return pd.Series(0, index=df.index)
        fmc = df['float_market_cap'].replace(0, np.nan) if 'float_market_cap' in df.columns else 100
        return df['order_amount']/fmc/10000

@FactorRegistry.register
class BreakLimitScore(FactorBase):
    name="break_limit_score"; category="sentiment"; desc="炸板惩罚(负分)"
    def compute(self, df):
        return -df['break_limit_times'] if 'break_limit_times' in df.columns else pd.Series(0, index=df.index)

@FactorRegistry.register
class NorthBoundChange(FactorBase):
    name="north_bound_change"; category="sentiment"; desc="北向资金净流入"
    def compute(self, df):
        if 'net_amount' in df.columns: return df['net_amount']
        return pd.Series(0, index=df.index)

@FactorRegistry.register
class MarginRatio(FactorBase):
    name="margin_ratio"; category="sentiment"; desc="融资余额占比"
    def compute(self, df):
        return df['margin_ratio'] if 'margin_ratio' in df.columns else pd.Series(0, index=df.index)

@FactorRegistry.register
class LimitUpCountMarket(FactorBase):
    name="limit_up_count"; category="sentiment"; desc="全市场涨停家数(情绪温度)"
    def compute(self, df):
        if 'limit_status' not in df.columns: return pd.Series(0, index=df.index)
        return df.groupby('trade_date')['limit_status'].transform(
            lambda x: (x=='U').sum()) if 'trade_date' in df.columns else pd.Series(0, index=df.index)

# ---- v2 新增因子 ----

@FactorRegistry.register
class HotRank(FactorBase):
    name="hot_rank"; category="sentiment"; desc="同花顺热榜排名(越小越好)"
    def compute(self, df):
        if 'hot_rank' in df.columns: return -df['hot_rank'].rank(pct=True)
        return pd.Series(0, index=df.index)

@FactorRegistry.register
class TopListNetBuy(FactorBase):
    name="top_list_net_buy"; category="sentiment"; desc="龙虎榜净买入(万元)"
    def compute(self, df):
        if 'lhb_net_buy' in df.columns: return df['lhb_net_buy']
        if 'lhb_buy' in df.columns and 'lhb_sell' in df.columns:
            return df['lhb_buy'] - df['lhb_sell']
        return pd.Series(0, index=df.index)

@FactorRegistry.register
class CapitalTrace(FactorBase):
    name="capital_trace"; category="sentiment"; desc="游资参与标记(0/1)"
    def compute(self, df):
        if 'capital_name' in df.columns:
            known = ['华鑫', '财通', '上塘', '作手', '方新侠', '炒股养家', '赵老哥', '章盟主']
            pattern = '|'.join(known)
            return df['capital_name'].astype(str).str.contains(pattern).astype(float)
        return pd.Series(0, index=df.index)

@FactorRegistry.register
class LimitUpStrength(FactorBase):
    name="limit_up_strength"; category="sentiment"; desc="涨停封板时长占比(越大越强)"
    def compute(self, df):
        if 'first_time' not in df.columns or 'last_time' not in df.columns:
            return pd.Series(0, index=df.index)
        return 1 - (df['last_time'] - df['first_time']) / 240  # 240分钟交易时间

@FactorRegistry.register
class BidAskSpread(FactorBase):
    name="bid_ask_spread"; category="sentiment"; desc="买卖价差率(越小越好)"
    def compute(self, df):
        if 'bid1' in df.columns and 'ask1' in df.columns:
            return -abs(df['ask1'] - df['bid1']) / df['close'].replace(0, np.nan)
        return pd.Series(0, index=df.index)
