"""
百度证券宝 Baostock 免费数据源
pip install baostock
优势：财务报表数据结构完整，适合基本面因子
"""
import pandas as pd
import numpy as np
from datetime import datetime
from .base import DataSourceBase


class BaostockSource(DataSourceBase):
    name = "baostock"
    priority = 4

    def __init__(self):
        self._available = False
        self._logged_in = False
        try:
            import baostock as bs
            self.bs = bs
            self._available = True
        except ImportError:
            self.bs = None

    def ping(self):
        return self._available

    def _login(self):
        if not self._available or self._logged_in:
            return True
        try:
            lg = self.bs.login()
            self._logged_in = (lg.error_code == '0')
            return self._logged_in
        except Exception:
            return False

    def _logout(self):
        if self._logged_in:
            try:
                self.bs.logout()
            except Exception:
                pass
            self._logged_in = False

    def get_daily(self, ts_code, start_date, end_date):
        if not self._login():
            return pd.DataFrame()
        try:
            symbol = ts_code.split('.')[0]
            # baostock 格式: sh.600519 或 sz.000001
            prefix = "sh" if ts_code.endswith('.SH') or symbol.startswith(('6', '68')) else "sz"
            bs_code = f"{prefix}.{symbol}"

            start = self._fmt(start_date)
            end = self._fmt(end_date)

            rs = self.bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount,turn,pctChg",
                start_date=start, end_date=end,
                frequency="d", adjustflag="2"  # 前复权
            )
            if rs.error_code != '0':
                return pd.DataFrame()

            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'turnover_ratio', 'pct_chg'])
            df['trade_date'] = df['trade_date'].str.replace('-', '')
            df['ts_code'] = ts_code
            for col in ['open', 'high', 'low', 'close', 'vol', 'amount', 'turnover_ratio', 'pct_chg']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['amount'] = df['amount'] / 1000  # 元 → 千元（匹配Tushare）
            if 'close' in df.columns:
                df['pre_close'] = df.groupby('ts_code')['close'].shift(1)
            return df.reset_index(drop=True)
        except Exception as e:
            print(f"[Baostock] 获取 {ts_code} 日线失败: {e}")
            return pd.DataFrame()

    def get_stock_list(self, market="主板"):
        if not self._login():
            return pd.DataFrame()
        try:
            rs = self.bs.query_stock_basic()
            if rs.error_code != '0':
                return pd.DataFrame()
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=['code', 'code_name', 'ipoDate', 'outDate', 'type', 'status'])
            df = df[df['status'] == '1']
            df = df.rename(columns={'code': 'ts_code', 'code_name': 'name'})
            df['ts_code'] = df['ts_code'].apply(lambda x: f"{x}.{'SH' if x.startswith('sh') else 'SZ'}".replace('sh.', '').replace('sz.', ''))
            df['symbol'] = df['ts_code'].str.split('.').str[0]

            # 按market过滤
            if market != "全部":
                sym = df['symbol']
                if market == "主板":
                    df = df[sym.str.startswith(('60', '00'))]
                elif market == "创业板":
                    df = df[sym.str.startswith('30')]
                elif market == "科创板":
                    df = df[sym.str.startswith('68')]

            return df.reset_index(drop=True)
        except Exception as e:
            print(f"[Baostock] 获取股票列表失败: {e}")
            return pd.DataFrame()

    def get_financial(self, ts_code, start_date, end_date):
        """获取财务三表数据（利润表/资产负债表/现金流量表）"""
        if not self._login():
            return pd.DataFrame()
        try:
            symbol = ts_code.split('.')[0]
            prefix = "sh" if ts_code.endswith('.SH') or symbol.startswith(('6', '68')) else "sz"
            bs_code = f"{prefix}.{symbol}"

            start = self._fmt(start_date)
            end = self._fmt(end_date)

            all_dfs = []
            for table in ['profit', 'balance', 'cash']:
                rs = self.bs.query_growth_data(bs_code, year=start[:4], quarter=1)
                if rs.error_code == '0':
                    rows = []
                    while rs.next():
                        rows.append(rs.get_row_data())
                    if rows:
                        df = pd.DataFrame(rows)
                        df['ts_code'] = ts_code
                        all_dfs.append(df)

            if not all_dfs:
                return pd.DataFrame()
            return pd.concat(all_dfs, ignore_index=True)
        except Exception as e:
            print(f"[Baostock] 获取 {ts_code} 财务数据失败: {e}")
            return pd.DataFrame()

    def get_trade_calendar(self):
        """从baostock获取交易日历"""
        if not self._login():
            return pd.DataFrame()
        try:
            rs = self.bs.query_trade_dates(start_date="2010-01-01", end_date="2030-12-31")
            if rs.error_code != '0':
                return pd.DataFrame()
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if rows:
                df = pd.DataFrame(rows, columns=['calendar_date', 'is_trading_day'])
                df = df[df['is_trading_day'] == '1']
                df = df.rename(columns={'calendar_date': 'cal_date'})
                df['is_open'] = 1
                return df[['cal_date', 'is_open']]
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def get_limit_list(self, trade_date):
        return pd.DataFrame()  # Baostock无此接口

    def get_spot(self):
        return pd.DataFrame()  # Baostock无实时

    @staticmethod
    def _fmt(d):
        if isinstance(d, datetime):
            return d.strftime('%Y-%m-%d')
        s = str(d)
        if len(s) == 8:
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return s
