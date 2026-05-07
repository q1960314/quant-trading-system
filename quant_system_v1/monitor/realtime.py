"""
实时监控模块 — 盘中监控候选标的
"""
import time
from datetime import datetime
import pandas as pd
from utils.logger import get_logger
from utils.calendar import TradeCalendar
from utils.helpers import detect_market, get_limit_ratio
from data_source.manager import DataSourceManager
from config.settings import TUSHARE_TOKEN, TUSHARE_API_URL

logger = get_logger("monitor")


class RealtimeMonitor:
    def __init__(self, strategy_name="打板策略", watch_list=None):
        self.strategy_name = strategy_name
        self.watch_list = watch_list or []
        self.mgr = DataSourceManager(token=TUSHARE_TOKEN, api_url=TUSHARE_API_URL)
        self.cal = TradeCalendar()
        self.alerted_today = set()
        logger.info(f"监控初始化: {strategy_name} | 标的: {len(self.watch_list)}只")

    def run(self, interval=60):
        """主循环"""
        logger.info(f"开始实时监控，刷新间隔: {interval}秒")
        while True:
            try:
                now = datetime.now()

                # 只在交易时间运行
                if not self.cal.is_trade_day(now):
                    logger.info(f"{now.date()} 非交易日，休眠...")
                    time.sleep(3600)
                    continue

                t = now.time()
                morning_start = datetime.strptime("09:30", "%H:%M").time()
                morning_end = datetime.strptime("11:30", "%H:%M").time()
                afternoon_start = datetime.strptime("13:00", "%H:%M").time()
                afternoon_end = datetime.strptime("15:00", "%H:%M").time()

                in_trading = ((morning_start <= t <= morning_end) or
                              (afternoon_start <= t <= afternoon_end))

                if not in_trading:
                    # 收盘后重置告警记录
                    if t > afternoon_end:
                        self.alerted_today.clear()
                    time.sleep(60)
                    continue

                # 获取实时行情
                spot = self.mgr.get_spot()
                if spot is None or spot.empty:
                    logger.debug("实时行情获取失败，重试...")
                    time.sleep(10)
                    continue

                self._check_alerts(spot, now)
                time.sleep(interval)

            except KeyboardInterrupt:
                logger.info("监控已停止")
                break
            except Exception as e:
                logger.error(f"监控异常: {e}")
                time.sleep(30)

    def _check_alerts(self, spot, now):
        """检查告警条件"""
        if not self.watch_list:
            return

        for code in self.watch_list:
            if code in self.alerted_today:
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

            # 告警条件
            reasons = []

            # 涨停
            lr = get_limit_ratio(code)
            if 'pre_close' in row and row['pre_close'] > 0:
                limit_up_price = row['pre_close'] * (1 + lr)
                if price >= limit_up_price * 0.995:
                    reasons.append(f"触及涨停: {price:.2f}")

            # 放量
            if vol_ratio > 3:
                reasons.append(f"放量: 量比{vol_ratio:.1f}")

            # 大幅拉升
            if pct > 7:
                reasons.append(f"大幅拉升: {pct:.1f}%")

            # 跌停风险
            if 'pre_close' in row and row['pre_close'] > 0:
                limit_down = row['pre_close'] * (1 - lr)
                if price <= limit_down * 1.005:
                    reasons.append(f"触及跌停: {price:.2f}")

            if reasons:
                msg = f"[{now.strftime('%H:%M:%S')}] {code} {row.get('name','')} | {'; '.join(reasons)}"
                logger.info(f"⚠️ {msg}")
                self.alerted_today.add(code)
