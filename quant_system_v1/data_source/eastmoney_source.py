"""
东方财富 免费HTTP数据源
无需token，直接HTTP请求，适合实时行情 + 历史K线
"""
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
from .base import DataSourceBase


class EastMoneySource(DataSourceBase):
    name = "eastmoney"
    priority = 3

    KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    SPOT_URL = "https://push2.eastmoney.com/api/qt/stock/get"

    def ping(self):
        try:
            r = requests.get("https://push2.eastmoney.com/api/qt/stock/get",
                           params={"secid": "1.600519", "fields": "f43"}, timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def _secid(self, ts_code):
        """000001.SZ → 0.000001 或 600519.SH → 1.600519"""
        code = ts_code.split('.')[0]
        if ts_code.endswith('.SH'):
            return f"1.{code}"
        return f"0.{code}"

    def get_daily(self, ts_code, start_date, end_date):
        try:
            secid = self._secid(ts_code)
            params = {
                "secid": secid,
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "klt": "101",  # 日线
                "fqt": "1",    # 前复权
                "beg": self._fmt(start_date),
                "end": self._fmt(end_date),
                "lmt": "10000",
            }
            r = requests.get(self.KLINE_URL, params=params, timeout=30)
            data = r.json()
            if not data or data.get('data') is None:
                return pd.DataFrame()
            klines = data['data'].get('klines', [])
            if not klines:
                return pd.DataFrame()

            rows = []
            for k in klines:
                parts = k.split(',')
                rows.append({
                    'trade_date': parts[0].replace('-', ''),
                    'open': float(parts[1]),
                    'close': float(parts[2]),
                    'high': float(parts[3]),
                    'low': float(parts[4]),
                    'vol': float(parts[5]),
                    'amount': float(parts[6]) / 1000,  # 元→千元
                    'pct_chg': float(parts[8]) if len(parts) > 8 else 0,
                    'turnover_ratio': float(parts[10]) if len(parts) > 10 else 0,
                })
            df = pd.DataFrame(rows)
            df['ts_code'] = ts_code
            if 'close' in df.columns:
                df['pre_close'] = df['close'].shift(1)
            return df
        except Exception as e:
            print(f"[EastMoney] 获取 {ts_code} 日线失败: {e}")
            return pd.DataFrame()

    def get_spot(self):
        """全市场实时行情，单次HTTP请求"""
        try:
            all_data = []
            # 分页获取：沪深两市分别请求
            for market_id in [1, 0]:  # 1=沪市, 0=深市
                params = {
                    "secid": f"{market_id}.000001",
                    "fields": "f2,f3,f4,f5,f6,f7,f8,f10,f12,f14,f15,f16,f17,f18,f20,f21",
                    "pn": "1",
                    "pz": "5000",
                }
                try:
                    # 用板块接口批量获取
                    url = "https://push2.eastmoney.com/api/qt/clist/get"
                    clist_params = {
                        "pn": "1", "pz": "5000",
                        "fs": f"m:{'1' if market_id == 1 else '0'}+t:2,m:{'1' if market_id == 1 else '0'}+t:6,m:{'1' if market_id == 1 else '0'}+t:13,m:{'1' if market_id == 1 else '0'}+t:80",
                        "fields": "f2,f3,f4,f5,f6,f7,f8,f10,f12,f14,f15,f16,f17,f18,f20,f21,f22,f23,f24,f25"
                    }
                    r = requests.get(url, params=clist_params, timeout=30)
                    data = r.json()
                    if data and data.get('data') and data['data'].get('diff'):
                        for item in data['data']['diff']:
                            code = item.get('f12', '')
                            all_data.append({
                                'ts_code': f"{code}.SH" if market_id == 1 else f"{code}.SZ",
                                'name': item.get('f14', ''),
                                'close': item.get('f2', 0),
                                'pct_chg': item.get('f3', 0),
                                'open': item.get('f17', 0),
                                'high': item.get('f15', 0),
                                'low': item.get('f16', 0),
                                'vol': item.get('f5', 0),
                                'amount': item.get('f6', 0),
                                'turnover_ratio': item.get('f8', 0),
                                'volume_ratio': item.get('f10', 0),
                                'pe_ttm': item.get('f9', 0),
                                'float_market_cap': item.get('f20', 0) / 1e8 if item.get('f20') else 0,
                            })
                except Exception:
                    continue
            return pd.DataFrame(all_data) if all_data else pd.DataFrame()
        except Exception as e:
            print(f"[EastMoney] 获取实时行情失败: {e}")
            return pd.DataFrame()

    def get_stock_list(self, market="主板"):
        """从东方财富获取股票列表"""
        try:
            all_data = []
            for market_id in [1, 0]:
                url = "https://push2.eastmoney.com/api/qt/clist/get"
                params = {
                    "pn": "1", "pz": "5000",
                    "fs": f"m:{'1' if market_id == 1 else '0'}+t:2,m:{'1' if market_id == 1 else '0'}+t:6,m:{'1' if market_id == 1 else '0'}+t:13,m:{'1' if market_id == 1 else '0'}+t:80",
                    "fields": "f12,f14"
                }
                r = requests.get(url, params=params, timeout=30)
                data = r.json()
                if data and data.get('data') and data['data'].get('diff'):
                    for item in data['data']['diff']:
                        code = str(item.get('f12', ''))
                        prefix = "SH" if market_id == 1 else "SZ"
                        all_data.append({
                            'ts_code': f"{code}.{prefix}",
                            'symbol': code,
                            'name': item.get('f14', ''),
                        })
            return pd.DataFrame(all_data)
        except Exception as e:
            print(f"[EastMoney] 获取股票列表失败: {e}")
            return pd.DataFrame()

    def get_limit_list(self, trade_date):
        """东方财富无涨跌停列表API，返回空（由AKShare覆盖）"""
        return pd.DataFrame()

    @staticmethod
    def _fmt(d):
        if isinstance(d, datetime):
            return d.strftime('%Y%m%d')
        return str(d).replace('-', '')
