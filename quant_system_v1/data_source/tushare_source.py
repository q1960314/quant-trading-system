"""
Tushare 第三方数据源 — 严格按教程初始化
"""
import pandas as pd
from datetime import datetime
from .base import DataSourceBase

TOKEN = "123396cf48bacd87370d4541fe4c2c51bcacb43f32f66890650dd5bb907a"
BASE_URL = "http://jiaoch.site"


class TushareSource(DataSourceBase):
    name = "tushare"
    priority = 1

    def __init__(self, token="", api_url=""):
        self.token = token or TOKEN
        self.api_url = api_url or BASE_URL
        self._pro = None
        self._available = False
        self._connect()

    def _connect(self):
        try:
            import tushare as ts
            self._pro = ts.pro_api(self.token)
            self._pro._DataApi__token = self.token
            self._pro._DataApi__http_url = self.api_url
            # 快速验证
            df = self._pro.stock_basic(exchange='', list_status='L', fields='ts_code', limit=1)
            self._available = df is not None and not df.empty
            print(f"[Tushare] {'ok' if self._available else 'fail'} | {self.api_url}")
        except Exception as e:
            print(f"[Tushare] fail: {e}")
            self._available = False

    def ping(self):
        return self._available

    @staticmethod
    def _fmt(d):
        if isinstance(d, (datetime, pd.Timestamp)):
            return d.strftime('%Y%m%d')
        return str(d).replace('-', '')[:8]

    def _try(self, method, *args, **kwargs):
        """调用 pro 对象的方法，失败返回空 DataFrame"""
        try:
            fn = getattr(self._pro, method)
            df = fn(*args, **kwargs)
            return df if df is not None and not df.empty else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    # ---- 个股 ----
    def stock_basic(self, market="主板"):
        df = self._try('stock_basic', exchange='', list_status='L',
                       fields='ts_code,symbol,name,area,industry,list_date,market')
        if df.empty: return df
        m = {"主板": ["主板","MainBoard","上交所主板","深交所主板"],
             "创业板": ["创业板","ChiNext","深交所创业板"],
             "科创板": ["科创板","STAR","上交所科创板"],
             "北交所": ["北交所","BSE","北京证券交易所"]}
        if market in m: df = df[df['market'].isin(m[market])]
        return df.reset_index(drop=True)

    def daily(self, ts_code=None, start_date=None, end_date=None):
        """日线行情。不传ts_code=全市场，传trade_date=单日"""
        if ts_code is None and start_date is None and end_date is None:
            return pd.DataFrame()
        if ts_code is not None:
            return self._try('daily', ts_code=ts_code, start_date=self._fmt(start_date), end_date=self._fmt(end_date))
        # 按日期拉全市场
        if isinstance(start_date, pd.Timestamp):
            return self._try('daily', trade_date=start_date.strftime('%Y%m%d'))
        return self._try('daily', trade_date=str(start_date).replace('-', '')[:8])

    def daily_basic(self, ts_code, start_date, end_date):
        return self._try('daily_basic', ts_code=ts_code, start_date=self._fmt(start_date), end_date=self._fmt(end_date))

    def moneyflow(self, ts_code, start_date, end_date):
        return self._try('moneyflow', ts_code=ts_code, start_date=self._fmt(start_date), end_date=self._fmt(end_date))

    def stk_limit(self, trade_date):
        return self._try('stk_limit', trade_date=self._fmt(trade_date))

    def top_list(self, trade_date):
        return self._try('top_list', trade_date=self._fmt(trade_date))

    def top_inst(self, trade_date):
        return self._try('top_inst', trade_date=self._fmt(trade_date))

    def fina_indicator(self, ts_code, start_date, end_date):
        return self._try('fina_indicator', ts_code=ts_code, start_date=self._fmt(start_date), end_date=self._fmt(end_date))

    def income(self, ts_code, start_date, end_date):
        return self._try('income', ts_code=ts_code, start_date=self._fmt(start_date), end_date=self._fmt(end_date))

    def balancesheet(self, ts_code, start_date, end_date):
        return self._try('balancesheet', ts_code=ts_code, start_date=self._fmt(start_date), end_date=self._fmt(end_date))

    def cashflow(self, ts_code, start_date, end_date):
        return self._try('cashflow', ts_code=ts_code, start_date=self._fmt(start_date), end_date=self._fmt(end_date))

    def forecast(self, ts_code, start_date, end_date):
        return self._try('forecast', ts_code=ts_code, start_date=self._fmt(start_date), end_date=self._fmt(end_date))

    def express(self, ts_code, start_date, end_date):
        return self._try('express', ts_code=ts_code, start_date=self._fmt(start_date), end_date=self._fmt(end_date))

    def suspend_d(self, trade_date=None):
        if trade_date: return self._try('suspend_d', trade_date=self._fmt(trade_date))
        return self._try('suspend_d', start_date='20200101', end_date='20301231')

    def stk_holdertrade(self, ts_code=None, start_date=None, end_date=None):
        return self._try('stk_holdertrade', ts_code=ts_code or '', start_date=self._fmt(start_date), end_date=self._fmt(end_date))

    def stk_holdernumber(self, ts_code, start_date, end_date):
        return self._try('stk_holdernumber', ts_code=ts_code, start_date=self._fmt(start_date), end_date=self._fmt(end_date))

    def share_float(self, start_date=None, end_date=None):
        return self._try('share_float', start_date=self._fmt(start_date), end_date=self._fmt(end_date))

    def block_trade(self, start_date=None, end_date=None):
        return self._try('block_trade', start_date=self._fmt(start_date), end_date=self._fmt(end_date))

    def pledge_stat(self, ts_code=None):
        return self._try('pledge_stat', ts_code=ts_code or '')

    def hk_hold(self, ts_code, start_date, end_date):
        return self._try('hk_hold', ts_code=ts_code, start_date=self._fmt(start_date), end_date=self._fmt(end_date))

    def concept_detail(self, ts_code):
        return self._try('concept_detail', ts_code=ts_code)

    # ---- 指数 ----
    def index_basic(self, market='SSE'):
        return self._try('index_basic', market=market)

    def index_daily(self, ts_code, start_date, end_date):
        return self._try('index_daily', ts_code=ts_code, start_date=self._fmt(start_date), end_date=self._fmt(end_date))

    def index_weight(self, index_code, trade_date):
        return self._try('index_weight', index_code=index_code, trade_date=self._fmt(trade_date))

    def index_dailybasic(self, ts_code, trade_date):
        return self._try('index_dailybasic', trade_date=self._fmt(trade_date))

    def index_classify(self, level='L1', src='SW'):
        return self._try('index_classify', level=level, src=src)

    # ---- 板块 ----
    def sw_daily(self, ts_code, start_date, end_date):
        return self._try('sw_daily', ts_code=ts_code, start_date=self._fmt(start_date), end_date=self._fmt(end_date))

    def ths_index(self):
        return self._try('ths_index')

    def ths_daily(self, ts_code, start_date, end_date):
        return self._try('ths_daily', ts_code=ts_code, start_date=self._fmt(start_date), end_date=self._fmt(end_date))

    def ths_member(self, ts_code):
        return self._try('ths_member', ts_code=ts_code)

    def dc_index(self, trade_date=None):
        return self._try('dc_index', trade_date=self._fmt(trade_date) if trade_date else '')

    def dc_daily(self, ts_code, start_date, end_date):
        return self._try('dc_daily', ts_code=ts_code, start_date=self._fmt(start_date), end_date=self._fmt(end_date))

    def dc_member(self, ts_code, trade_date):
        return self._try('dc_member', ts_code=ts_code, trade_date=self._fmt(trade_date))

    def moneyflow_ind_dc(self, trade_date):
        return self._try('moneyflow_ind_dc', trade_date=self._fmt(trade_date))

    def moneyflow_mkt_dc(self, trade_date):
        return self._try('moneyflow_mkt_dc', trade_date=self._fmt(trade_date))

    # ---- 情绪 ----
    def limit_list_d(self, trade_date):
        return self._try('limit_list_d', trade_date=self._fmt(trade_date))

    def limit_step(self, trade_date):
        return self._try('limit_step', trade_date=self._fmt(trade_date))

    def hsgt_top10(self, trade_date):
        return self._try('hsgt_top10', trade_date=self._fmt(trade_date))

    def margin(self, trade_date=None):
        if trade_date: return self._try('margin', trade_date=self._fmt(trade_date))
        return self._try('margin')

    def margin_detail(self, trade_date):
        return self._try('margin_detail', trade_date=self._fmt(trade_date))

    def ths_hot(self, trade_date):
        return self._try('ths_hot', trade_date=self._fmt(trade_date))

    def dc_hot(self, trade_date):
        return self._try('dc_hot', trade_date=self._fmt(trade_date))

    def hm_detail(self, trade_date=None):
        if trade_date: return self._try('hm_detail', trade_date=self._fmt(trade_date))
        return self._try('hm_detail')

    def stk_auction(self, trade_date=None):
        if trade_date: return self._try('stk_auction', trade_date=self._fmt(trade_date))
        return self._try('stk_auction')

    # ---- 宏观 ----
    def shibor(self):
        return self._try('shibor')

    def shibor_lpr(self):
        return self._try('shibor_lpr')

    def cn_m(self):
        return self._try('cn_m')

    def cn_pmi(self):
        return self._try('cn_pmi')

    def cn_cpi(self):
        return self._try('cn_cpi')

    def cn_ppi(self):
        return self._try('cn_ppi')

    def trade_cal(self, start_date='20100101', end_date='20301231'):
        return self._try('trade_cal', exchange='', start_date=start_date, end_date=end_date)

    # ---- DataSourceBase 兼容 ----
    def get_daily(self, ts_code, start_date, end_date):
        return self.daily(ts_code, start_date, end_date)

    def get_stock_list(self, market="主板"):
        return self.stock_basic(market)

    def get_limit_list(self, trade_date):
        return self.stk_limit(trade_date)

    def get_daily_basic(self, ts_code, start_date, end_date):
        return self.daily_basic(ts_code, start_date, end_date)

    def get_financial(self, ts_code, start_date, end_date):
        return self.fina_indicator(ts_code, start_date, end_date)

    def get_moneyflow(self, ts_code, start_date, end_date):
        return self.moneyflow(ts_code, start_date, end_date)

    def get_top_list(self, trade_date):
        return self.top_list(trade_date)

    def get_trade_calendar(self):
        return self.trade_cal()

    def get_spot(self):
        return pd.DataFrame()
