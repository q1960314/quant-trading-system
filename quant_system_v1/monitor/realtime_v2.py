"""
实时监控模块 v2 — 自动发现候选 + 信号生成 + 企微推送

增强点:
1. 盘前: 从最新数据自动生成监控列表（调用选股器）
2. 盘中: 实时信号检测（涨停/放量/异动）
3. 推送: 企微 Webhook 推送告警
4. 盘后: 当日监控总结 + 明日候选
"""
import time
import json
import requests
from datetime import datetime
import pandas as pd
from utils.logger import get_logger
from utils.calendar import TradeCalendar
from utils.helpers import get_limit_ratio
from data_source.manager import DataSourceManager
from config.settings import TUSHARE_TOKEN, TUSHARE_API_URL, WECHAT_ROBOT_URL, WECHAT_ROBOT_ENABLED

logger = get_logger("monitor_v2")


class RealtimeMonitorV2:
    def __init__(self, strategy_names=None):
        self.strategy_names = strategy_names or ["打板策略"]
        self.mgr = DataSourceManager(token=TUSHARE_TOKEN, api_url=TUSHARE_API_URL)
        self.cal = TradeCalendar()
        self.watch_list = []
        self.alerted = set()
        self.history = []
        logger.info(f"监控v2: {self.strategy_names}")

    def auto_generate_watchlist(self):
        """盘前自动生成监控列表"""
        from stock_picker.picker_v2 import DailyStockPickerV2
        picker = DailyStockPickerV2(self.strategy_names)
        df, trade_date = picker.get_latest_data()
        if df.empty:
            logger.warning("无法加载数据，无法生成监控列表")
            return []
        result = picker.pick(df, trade_date)
        if result.empty:
            return []
        self.watch_list = result['ts_code'].tolist()
        logger.info(f"自动生成监控列表: {len(self.watch_list)} stocks")
        self._send_wechat(f"[盘前] {trade_date} 今日监控 {len(self.watch_list)}只: {','.join(self.watch_list[:5])}...")
        return self.watch_list

    def run(self, interval=60):
        self.auto_generate_watchlist()
        if not self.watch_list:
            logger.warning("监控列表为空")
            return
        logger.info(f"开始监控 {len(self.watch_list)} stocks, interval={interval}s")
        while True:
            try:
                now = datetime.now()
                if not self.cal.is_trade_day(now):
                    time.sleep(3600)
                    continue
                t = now.time()
                in_trading = (
                    (datetime.strptime("09:30", "%H:%M").time() <= t <= datetime.strptime("11:30", "%H:%M").time()) or
                    (datetime.strptime("13:00", "%H:%M").time() <= t <= datetime.strptime("15:00", "%H:%M").time())
                )
                if not in_trading:
                    if t > datetime.strptime("15:00", "%H:%M").time():
                        self.alerted.clear()
                    time.sleep(60)
                    continue
                spot = self.mgr.get_spot()
                if spot is None or spot.empty:
                    time.sleep(10)
                    continue
                self._check_signals(spot, now)
                time.sleep(interval)
            except KeyboardInterrupt:
                logger.info("监控停止")
                break
            except Exception as e:
                logger.error(f"监控异常: {e}")
                time.sleep(30)

    def _check_signals(self, spot, now):
        if not self.watch_list:
            return
        for code in self.watch_list:
            if code in self.alerted:
                continue
            row = spot[spot['ts_code'] == code]
            if row.empty:
                continue
            row = row.iloc[0]
            price = row.get('close', 0)
            pct = row.get('pct_chg', 0)
            vol_ratio = row.get('volume_ratio', 1)
            if price <= 0:
                continue
            signals = []
            lr = get_limit_ratio(code)
            if 'pre_close' in row and row['pre_close'] > 0:
                limit_up = row['pre_close'] * (1 + lr)
                if price >= limit_up * 0.99:
                    signals.append(('LIMIT_UP', f"涨停 {price:.2f}"))
                limit_down = row['pre_close'] * (1 - lr)
                if price <= limit_down * 1.01:
                    signals.append(('LIMIT_DOWN', f"跌停!!! {price:.2f}"))
            if vol_ratio > 3:
                signals.append(('VOL_SURGE', f"放量 {vol_ratio:.1f}x"))
            if pct > 7:
                signals.append(('SURGE', f"拉升 +{pct:.1f}%"))
            if pct < -5:
                signals.append(('DUMP', f"急跌 {pct:.1f}%"))
            if signals:
                msg = f"[{now.strftime('%H:%M:%S')}] {code} {row.get('name','')} | " + " | ".join(s[1] for s in signals)
                logger.warning(msg)
                self.alerted.add(code)
                self.history.append({'time': now.isoformat(), 'code': code, 'signals': [s[0] for s in signals]})
                self._send_wechat(msg)

    def daily_summary(self):
        """盘后总结"""
        if not self.history:
            return "今日无告警"
        df = pd.DataFrame(self.history)
        summary = f"[盘后总结] 今日告警 {len(df)} 次, 涉及 {df['code'].nunique()} 只股票"
        logger.info(summary)
        self._send_wechat(summary)
        self.history.clear()
        return summary

    def _send_wechat(self, msg):
        if not WECHAT_ROBOT_ENABLED or not WECHAT_ROBOT_URL:
            return
        try:
            resp = requests.post(WECHAT_ROBOT_URL, json={
                "msgtype": "text", "text": {"content": msg}
            }, timeout=5)
            if resp.status_code != 200:
                logger.warning(f"企微推送失败: {resp.status_code}")
        except Exception as e:
            logger.warning(f"企微推送异常: {e}")
