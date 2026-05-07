from datetime import datetime, timedelta
import pandas as pd

class TradeCalendar:
    """A股交易日历，优先从数据源加载，失败则用简易估算"""
    def __init__(self):
        self._dates = set()

    def load_from_source(self, source):
        """从数据源加载官方交易日历"""
        try:
            cal = source.get_trade_calendar()
            if cal is not None and not cal.empty:
                if 'cal_date' in cal.columns:
                    self._dates = set(pd.to_datetime(cal['cal_date']).dt.date)
                    return len(self._dates)
                if 'trade_date' in cal.columns:
                    self._dates = set(pd.to_datetime(cal['trade_date']).dt.date)
                    return len(self._dates)
        except Exception:
            pass
        return 0

    def is_trade_day(self, d=None):
        if d is None:
            d = datetime.now().date()
        if isinstance(d, datetime):
            d = d.date()
        if self._dates:
            return d in self._dates
        return d.weekday() < 5  # 周一至周五

    def prev_trade_day(self, d=None):
        if d is None:
            d = datetime.now().date()
        if isinstance(d, datetime):
            d = d.date()
        d = d - timedelta(days=1)
        while not self.is_trade_day(d):
            d = d - timedelta(days=1)
        return d

    def next_trade_day(self, d=None):
        if d is None:
            d = datetime.now().date()
        if isinstance(d, datetime):
            d = d.date()
        d = d + timedelta(days=1)
        while not self.is_trade_day(d):
            d = d + timedelta(days=1)
        return d

    def trade_days_between(self, start, end):
        if isinstance(start, str):
            start = datetime.strptime(start, "%Y%m%d").date() if len(start) == 8 else datetime.strptime(start, "%Y-%m-%d").date()
        if isinstance(end, str):
            end = datetime.strptime(end, "%Y%m%d").date() if len(end) == 8 else datetime.strptime(end, "%Y-%m-%d").date()
        d = start
        result = []
        while d <= end:
            if self.is_trade_day(d):
                result.append(d)
            d += timedelta(days=1)
        return result
