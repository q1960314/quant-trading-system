"""
数据适配器 — 统一不同数据源的字段差异
输入：任意数据源的 DataFrame
输出：标准字段的 DataFrame
"""
import pandas as pd
import numpy as np


class DataAdapter:
    """把不同源的字段映射到标准名，补充计算列"""

    # 日线字段映射：标准名 → [可能的源字段名列表]
    DAILY_MAP = {
        'ts_code': ['ts_code', 'code', 'symbol_code'],
        'trade_date': ['trade_date', 'date', 'tradeDate', 'datetime'],
        'open': ['open', 'open_price', 'openPrice'],
        'high': ['high', 'high_price', 'highPrice'],
        'low': ['low', 'low_price', 'lowPrice'],
        'close': ['close', 'close_price', 'closePrice'],
        'vol': ['vol', 'volume', 'trade_volume'],
        'amount': ['amount', 'trade_amount', 'totalAmount'],
        'pct_chg': ['pct_chg', 'pctChg', 'change_pct', 'changePercent'],
        'turnover_ratio': ['turnover_ratio', 'turn', 'turnoverRate'],
        'pre_close': ['pre_close', 'preClose', 'prevClose'],
    }

    LIMIT_MAP = {
        'ts_code': ['ts_code', 'code'],
        'trade_date': ['trade_date', 'date'],
        'name': ['name', 'stock_name'],
        'close': ['close', 'price'],
        'pct_chg': ['pct_chg', 'pctChg'],
        'limit_status': ['limit_status', 'limit', 'limitStatus'],
        'order_amount': ['order_amount', 'fd_amount', 'orderAmount'],
        'float_market_cap': ['float_market_cap', 'float_mv', 'floatMarketCap'],
        'up_down_times': ['up_down_times', 'up_stat', 'upDownTimes'],
        'break_limit_times': ['break_limit_times', 'open_times', 'breakTimes'],
        'first_limit_time': ['first_limit_time', 'first_time', 'firstLimitTime'],
        'turnover_ratio': ['turnover_ratio', 'turn', 'turnoverRate'],
    }

    @classmethod
    def normalize_daily(cls, df):
        """标准化日线数据"""
        if df is None or df.empty:
            return df
        df = cls._remap(df, cls.DAILY_MAP)

        # 确保类型一致
        for col in ['open', 'high', 'low', 'close', 'vol', 'amount', 'turnover_ratio', 'pct_chg']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'trade_date' in df.columns:
            df['trade_date'] = df['trade_date'].astype(str).str.replace('-', '')

        # 补 pre_close
        if 'pre_close' not in df.columns or df['pre_close'].isna().all():
            if 'close' in df.columns and 'ts_code' in df.columns:
                df = df.sort_values(['ts_code', 'trade_date'])
                df['pre_close'] = df.groupby('ts_code')['close'].shift(1)

        # 补 pct_chg
        if 'pct_chg' not in df.columns or df['pct_chg'].isna().all():
            if 'pre_close' in df.columns and 'close' in df.columns:
                df['pct_chg'] = np.where(df['pre_close'] > 0,
                                          (df['close'] - df['pre_close']) / df['pre_close'] * 100, 0)

        return df.reset_index(drop=True)

    @classmethod
    def normalize_limit_list(cls, df):
        """标准化涨跌停列表"""
        if df is None or df.empty:
            return df
        df = cls._remap(df, cls.LIMIT_MAP)

        if 'trade_date' in df.columns:
            df['trade_date'] = df['trade_date'].astype(str).str.replace('-', '')

        for col in ['order_amount', 'float_market_cap', 'up_down_times', 'break_limit_times', 'turnover_ratio']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 标准化 limit_status
        if 'limit_status' in df.columns:
            df['limit_status'] = df['limit_status'].apply(cls._norm_limit)
        elif 'pct_chg' in df.columns:
            df['limit_status'] = np.where(df['pct_chg'] >= 9.5, 'U',
                                   np.where(df['pct_chg'] <= -9.5, 'D', ''))

        return df.reset_index(drop=True)

    @staticmethod
    def _remap(df, mapping):
        """按映射表重命名列"""
        rename = {}
        for std_name, aliases in mapping.items():
            for alias in aliases:
                if alias in df.columns and std_name not in df.columns:
                    rename[alias] = std_name
                    break
        if rename:
            df = df.rename(columns=rename)
        return df

    @staticmethod
    def _norm_limit(val):
        s = str(val).upper()
        if s in ('U', '涨停', '1', 'TRUE', 'LIMIT_UP'):
            return 'U'
        if s in ('D', '跌停', '-1', 'LIMIT_DOWN'):
            return 'D'
        return ''
