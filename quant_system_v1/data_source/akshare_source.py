"""
AKShare 免费数据源
pip install akshare
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .base import DataSourceBase


class AKShareSource(DataSourceBase):
    name = "akshare"
    priority = 2

    def __init__(self):
        self._available = False
        try:
            import akshare as ak
            self.ak = ak
            self._available = True
        except ImportError:
            self.ak = None
            print("[AKShare] 未安装，请运行: pip install akshare")

    def ping(self):
        return self._available

    # ========== 股票列表 ==========
    def get_stock_list(self, market="主板"):
        if not self._available:
            return pd.DataFrame()
        try:
            df = self.ak.stock_info_a_code_name()
            df = df.rename(columns={'code': 'symbol', 'name': 'name'})

            # 按代码前缀映射板块 → ts_code
            def _make_ts_code(sym):
                if sym.startswith(('60', '00')):
                    return f"{sym}.SH" if sym.startswith('6') else f"{sym}.SZ"
                elif sym.startswith('30'):
                    return f"{sym}.SZ"
                elif sym.startswith('68'):
                    return f"{sym}.SH"
                elif sym.startswith(('8', '4')):
                    return f"{sym}.BJ"
                return f"{sym}.SZ"

            df['ts_code'] = df['symbol'].apply(_make_ts_code)
            df['market'] = df['ts_code'].apply(self._detect_market)

            # 按板块过滤
            if market and market != "全部":
                df = df[df['market'] == market]

            return df.reset_index(drop=True)
        except Exception as e:
            print(f"[AKShare] 获取股票列表失败: {e}")
            return pd.DataFrame()

    # ========== 日线行情 ==========
    def get_daily(self, ts_code, start_date, end_date):
        if not self._available:
            return pd.DataFrame()
        try:
            symbol = ts_code.split('.')[0]
            start = self._fmt_date(start_date)
            end = self._fmt_date(end_date)

            df = self.ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                          start_date=start, end_date=end,
                                          adjust="qfq")
            if df is None or df.empty:
                return pd.DataFrame()

            df = df.rename(columns={
                '日期': 'trade_date', '开盘': 'open', '收盘': 'close',
                '最高': 'high', '最低': 'low', '成交量': 'vol',
                '成交额': 'amount', '换手率': 'turnover_ratio',
                '涨跌幅': 'pct_chg',
            })
            df['ts_code'] = ts_code
            df['trade_date'] = df['trade_date'].astype(str).str.replace('-', '')
            if 'pre_close' not in df.columns:
                df['pre_close'] = df.groupby('ts_code')['close'].shift(1)
            df['vol'] = df['vol'].astype(float)
            df['amount'] = df['amount'].astype(float)  # AKShare amount 单位是元，转千元匹配 Tushare
            df['amount'] = df['amount'] / 1000

            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = df[col].astype(float)

            return df.reset_index(drop=True)
        except Exception as e:
            print(f"[AKShare] 获取 {ts_code} 日线失败: {e}")
            return pd.DataFrame()

    # ========== 涨跌停列表 ==========
    def get_limit_list(self, trade_date):
        if not self._available:
            return pd.DataFrame()
        try:
            date_str = self._fmt_date(trade_date)
            df = self.ak.stock_zt_pool_em(date=date_str)
            if df is None or df.empty:
                return pd.DataFrame()

            df = df.rename(columns={
                '代码': 'ts_code', '名称': 'name',
                '最新价': 'close', '涨跌幅': 'pct_chg',
                '封单金额': 'order_amount',
                '流通市值': 'float_market_cap',
                '换手率': 'turnover_ratio',
                '连板数': 'up_down_times',
                '炸板次数': 'break_limit_times',
                '首次封板时间': 'first_limit_time',
            })
            df['trade_date'] = date_str.replace('-', '') if '-' in date_str else date_str
            df['limit_status'] = 'U'
            df['ts_code'] = df['ts_code'].apply(self._normalize_code)

            for col in ['order_amount', 'float_market_cap', 'up_down_times', 'break_limit_times']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            if 'order_amount' in df.columns:
                df['order_amount'] = df['order_amount'] / 10000  # 元 → 万元

            if 'float_market_cap' in df.columns:
                df['float_market_cap'] = df['float_market_cap'] / 1e8  # 元 → 亿

            return df.reset_index(drop=True)
        except Exception as e:
            print(f"[AKShare] 获取 {trade_date} 涨停列表失败: {e}")
            return pd.DataFrame()

    # ========== 实时行情 ==========
    def get_spot(self):
        if not self._available:
            return pd.DataFrame()
        try:
            df = self.ak.stock_zh_a_spot_em()
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.rename(columns={
                '代码': 'ts_code', '名称': 'name',
                '最新价': 'close', '涨跌幅': 'pct_chg',
                '成交量': 'vol', '成交额': 'amount',
                '换手率': 'turnover_ratio', '量比': 'volume_ratio',
                '今开': 'open', '最高': 'high', '最低': 'low',
            })
            df['ts_code'] = df['ts_code'].apply(self._normalize_code)
            for col in ['close', 'open', 'high', 'low', 'vol', 'amount', 'pct_chg', 'turnover_ratio']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            print(f"[AKShare] 获取实时行情失败: {e}")
            return pd.DataFrame()

    # ========== 龙虎榜 ==========
    def get_top_list(self, trade_date):
        if not self._available:
            return pd.DataFrame()
        try:
            date_str = self._fmt_date(trade_date)
            df = self.ak.stock_lhb_detail_em(date=date_str)
            if df is None or df.empty:
                return pd.DataFrame()
            return df
        except Exception:
            return pd.DataFrame()

    # ========== 交易日历 ==========
    def get_trade_calendar(self):
        if not self._available:
            return pd.DataFrame()
        try:
            df = self.ak.tool_trade_date_hist_sina()
            if df is not None and not df.empty:
                df = df.rename(columns={'trade_date': 'cal_date'})
                df['is_open'] = 1
                return df
        except Exception:
            pass
        return pd.DataFrame()

    # ========== 辅助 ==========
    @staticmethod
    def _detect_market(code):
        sym = code.split('.')[0]
        if sym.startswith(('60', '00')): return "主板"
        if sym.startswith('30'): return "创业板"
        if sym.startswith('68'): return "科创板"
        if sym.startswith(('8', '4')): return "北交所"
        return "主板"

    @staticmethod
    def _normalize_code(code):
        code = str(code)
        if '.' in code:
            return code
        if code.startswith(('60', '68')):
            return f"{code}.SH"
        elif code.startswith(('30', '00')):
            return f"{code}.SZ"
        elif code.startswith(('8', '4')):
            return f"{code}.BJ"
        return f"{code}.SZ"

    @staticmethod
    def _fmt_date(d):
        if isinstance(d, (datetime,)):
            return d.strftime('%Y%m%d')
        s = str(d)
        s = s.replace('-', '')
        if len(s) == 8:
            return s
        return datetime.now().strftime('%Y%m%d')
