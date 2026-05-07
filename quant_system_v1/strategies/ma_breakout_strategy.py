"""
均线突破策略 (MA Crossover Breakout Strategy)

基于5日/20日均线交叉 + 成交量确认的中低频A股策略。
策略文档: _shared/knowledge/strategy_TASK-20260506-AUTO001.md

⚠️ 本策略仅供研究回测，不构成投资建议。实盘交易需用户书面确认 (R001)。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
#  参数定义
# ═══════════════════════════════════════════════════
@dataclass
class MABreakoutParams:
    """均线突破策略可调参数"""

    # 均线参数
    fast_ma_period: int = 5          # 短期均线周期 [3, 10]
    slow_ma_period: int = 20         # 长期均线周期 [10, 60]
    volume_ratio: float = 1.5        # 成交量放大倍数阈值 [1.2, 3.0]

    # 止盈止损
    stop_loss_pct: float = 0.05      # 固定止损比例 [0.03, 0.10]
    profit_target_1: float = 0.15    # 第一止盈目标 [0.10, 0.25]
    trailing_trigger: float = 0.08   # 移动止盈触发盈利阈值 [0.05, 0.15]
    max_holding_days: int = 20       # 最大持仓天数 [10, 40]

    # 仓位与风控
    single_stock_limit: float = 0.10 # 单票仓位上限 [0.05, 0.20]
    max_positions: int = 10          # 最大持仓标的数 [5, 20]
    max_total_position: float = 0.80 # 总仓位上限（持仓市值占总资金比例） [0.50, 0.95]
    industry_limit: float = 0.30     # 行业集中度上限 [0.20, 0.50]
    max_daily_drawdown: float = 0.03 # 日度最大回撤阈值 [0.02, 0.05]
    max_monthly_drawdown: float = 0.08  # 月度最大回撤阈值 [0.05, 0.15]

    # 波动率过滤
    atr_period: int = 14            # ATR 周期
    max_volatility: float = 0.08    # 最大允许波动率（ATR/Close）

    # 上市天数要求
    min_listing_days: int = 60      # 最低上市交易天数

    # 连续止损熔断
    consecutive_loss_limit: int = 3       # 连续止损次数触发熔断
    consecutive_loss_cooldown: int = 5    # 熔断冷却交易日数

    # 涨停不追入
    limit_up_threshold: float = 9.5       # 涨停判定阈值 (%)


# ═══════════════════════════════════════════════════
#  信号枚举
# ═══════════════════════════════════════════════════
class Signal(Enum):
    NONE = "none"
    GOLDEN_CROSS = "golden_cross"   # 金叉买入信号
    DEATH_CROSS = "death_cross"     # 死叉卖出信号


class PositionAction(Enum):
    OPEN = "open"       # 开仓
    CLOSE = "close"     # 平仓
    PARTIAL_CLOSE = "partial_close"  # 部分平仓


# ═══════════════════════════════════════════════════
#  持仓记录
# ═══════════════════════════════════════════════════
@dataclass
class Position:
    """单个标的持仓"""
    ts_code: str
    entry_date: str           # 入场日期 YYYYMMDD
    entry_price: float        # 入场价（次日开盘价）
    shares: int               # 持仓股数
    total_cost: float         # 总成本（含佣金估算）
    trailing_activated: bool = False   # 移动止盈是否已激活
    peak_price: float = 0.0           # 持仓期间最高价
    partial_closed: bool = False       # 是否已部分止盈

    @property
    def cost_price(self) -> float:
        return self.total_cost / self.shares if self.shares > 0 else 0.0

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """未实现盈亏比例"""
        if self.cost_price <= 0:
            return 0.0
        return (current_price - self.cost_price) / self.cost_price

    def holding_days(self, current_date: str) -> int:
        """持仓天数"""
        try:
            d0 = datetime.strptime(self.entry_date, "%Y%m%d")
            d1 = datetime.strptime(current_date, "%Y%m%d")
            return (d1 - d0).days
        except ValueError:
            return 0


# ═══════════════════════════════════════════════════
#  策略核心
# ═══════════════════════════════════════════════════
class MABreakoutStrategy:
    """均线突破策略引擎"""

    def __init__(self, params: MABreakoutParams | None = None):
        self.params = params or MABreakoutParams()
        # 持仓字典 ts_code → Position
        self.positions: Dict[str, Position] = {}
        # 总资金（由回测器/实盘引擎设置，用于计算总仓位比例）
        self.total_capital: float = 0.0
        # 风控状态
        self.consecutive_losses: int = 0
        self.loss_cooldown_until: str = ""   # YYYYMMDD
        self.daily_drawdown_triggered: bool = False
        self.monthly_liquidated: bool = False
        self.monthly_high_watermark: float = 1.0
        self.current_date: str = ""
        # 净值追踪
        self.nav_history: List[Dict] = []
        # 行业映射 ts_code → industry
        self.industry_map: Dict[str, str] = {}
        # 成交记录
        self.trades: List[Dict] = []

    # ─────────────── 数据准备 ───────────────
    @staticmethod
    def prepare_features(df: pd.DataFrame, params: MABreakoutParams | None = None) -> pd.DataFrame:
        """
        在日线数据上计算策略所需的全部特征列。

        输入 df 须包含标准列: ts_code, trade_date, open, high, low, close, vol
        输出 df 新增列: ma_fast, ma_slow, vol_ma, volume_ratio, atr, volatility,
                        golden_cross, death_cross, ma_slow_slope
        """
        p = params or MABreakoutParams()
        df = df.copy()
        df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

        # 均线
        df['ma_fast'] = df.groupby('ts_code')['close'].transform(
            lambda s: s.rolling(p.fast_ma_period, min_periods=p.fast_ma_period).mean()
        )
        df['ma_slow'] = df.groupby('ts_code')['close'].transform(
            lambda s: s.rolling(p.slow_ma_period, min_periods=p.slow_ma_period).mean()
        )

        # 均线斜率：MA20[t] > MA20[t-1] 即为正斜率
        df['ma_slow_slope'] = df.groupby('ts_code')['ma_slow'].transform(
            lambda s: s.diff() > 0
        )

        # 成交量均线 & 比值
        df['vol_ma'] = df.groupby('ts_code')['vol'].transform(
            lambda s: s.rolling(p.slow_ma_period, min_periods=p.slow_ma_period).mean()
        )
        df['volume_ratio'] = df['vol'] / df['vol_ma']

        # ATR (Average True Range)
        df['tr'] = _calc_true_range(df)
        df['atr'] = df.groupby('ts_code')['tr'].transform(
            lambda s: s.rolling(p.atr_period, min_periods=p.atr_period).mean()
        )
        df['volatility'] = df['atr'] / df['close']

        # 金叉 / 死叉信号
        df['golden_cross'] = (
            (df['ma_fast'] > df['ma_slow']) &
            (df.groupby('ts_code')['ma_fast'].shift(1) <=
             df.groupby('ts_code')['ma_slow'].shift(1))
        )
        df['death_cross'] = (
            (df['ma_fast'] < df['ma_slow']) &
            (df.groupby('ts_code')['ma_fast'].shift(1) >=
             df.groupby('ts_code')['ma_slow'].shift(1))
        )

        # 上一个交易日日期（用于判断上市天数）
        df['prev_close'] = df.groupby('ts_code')['close'].shift(1)

        return df

    # ─────────────── 入场判断 ───────────────
    def check_entry(self, row: pd.Series, listing_days: int = 999,
                    is_limit_up: bool = False) -> bool:
        """
        判断某日某标的是否满足入场条件。

        Parameters
        ----------
        row : 包含 prepare_features 计算后的特征列
        listing_days : 该标的已上市交易天数
        is_limit_up : 当日是否涨停
        """
        p = self.params

        # 1. 金叉
        if not row.get('golden_cross', False):
            return False

        # 2. 成交量放大
        if pd.isna(row.get('volume_ratio')) or row['volume_ratio'] < p.volume_ratio:
            return False

        # 3. 趋势方向：MA20 斜率为正
        if not row.get('ma_slow_slope', False):
            return False

        # 4. 价格位置：收盘价在 MA20 上方
        if pd.isna(row.get('ma_slow')) or row['close'] <= row['ma_slow']:
            return False

        # 5. 波动率过滤
        if pd.isna(row.get('volatility')) or row['volatility'] >= p.max_volatility:
            return False

        # 6. 上市时间
        if listing_days < p.min_listing_days:
            return False

        # 7. 涨停不追入
        if is_limit_up:
            return False

        # 8. 已有持仓不重复开仓
        if row['ts_code'] in self.positions:
            return False

        # 9. 持仓数量上限
        if len(self.positions) >= p.max_positions:
            return False

        # 10. 总仓位比例上限（不超过80%）
        if self.total_capital > 0:
            current_position_cost = sum(pos.total_cost for pos in self.positions.values())
            position_ratio = current_position_cost / self.total_capital
            if position_ratio >= p.max_total_position:
                logger.info(
                    f"[总仓位限制] 当前仓位比例 {position_ratio:.1%} 已达上限 "
                    f"{p.max_total_position:.0%}，禁止新开仓"
                )
                return False

        # 11. 风控状态：熔断 / 日度回撤 / 月度清仓
        if self._is_blocked():
            return False

        # 12. 行业集中度检查
        if not self._check_industry_limit(row.get('ts_code', '')):
            return False

        return True

    # ─────────────── 出场判断 ───────────────
    def check_exit(self, ts_code: str, row: pd.Series) -> List[Tuple[PositionAction, str, float]]:
        """
        检查某持仓是否应出场。

        Returns
        -------
        list of (action, reason, quantity_ratio)
            action: OPEN/CLOSE/PARTIAL_CLOSE
            reason: 出场原因
            quantity_ratio: 平仓比例（1.0=全部, 0.5=半仓等）
        """
        if ts_code not in self.positions:
            return []

        pos = self.positions[ts_code]
        p = self.params
        current_price = row['close']
        pnl_pct = pos.unrealized_pnl_pct(current_price)
        days = pos.holding_days(row['trade_date'])
        actions = []

        # 更新持仓期间最高价
        if current_price > pos.peak_price:
            pos.peak_price = current_price

        # 激活移动止盈
        if pnl_pct >= p.trailing_trigger and not pos.trailing_activated:
            pos.trailing_activated = True
            logger.info(f"[移动止盈激活] {ts_code} 盈利 {pnl_pct:.1%}")

        # 1. 固定止损
        if pnl_pct <= -p.stop_loss_pct:
            actions.append((PositionAction.CLOSE, "固定止损", 1.0))
            return actions  # 止损优先，直接返回

        # 2. 死叉
        if row.get('death_cross', False):
            actions.append((PositionAction.CLOSE, "死叉", 1.0))
            return actions

        # 3. 移动止盈：收盘价跌破 MA5
        if pos.trailing_activated:
            ma_fast = row.get('ma_fast', None)
            if pd.notna(ma_fast) and current_price < ma_fast:
                actions.append((PositionAction.CLOSE, "移动止盈（跌破MA5）", 1.0))
                return actions

        # 4. 固定止盈：盈利达15%，平50%仓位
        if pnl_pct >= p.profit_target_1 and not pos.partial_closed:
            actions.append((PositionAction.PARTIAL_CLOSE, f"固定止盈{p.profit_target_1:.0%}", 0.5))
            pos.partial_closed = True

        # 5. 时间止损
        if days >= p.max_holding_days:
            actions.append((PositionAction.CLOSE, "时间止损", 1.0))
            return actions

        return actions

    # ─────────────── 执行交易 ───────────────
    def open_position(self, ts_code: str, entry_date: str,
                      entry_price: float, capital: float,
                      commission_rate: float = 0.0003) -> Optional[Position]:
        """开仓"""
        p = self.params
        # 计算可买入股数（单票仓位上限）
        max_amount = capital * p.single_stock_limit
        shares = int(max_amount / (entry_price * (1 + commission_rate)) / 100) * 100  # 整手
        if shares <= 0:
            logger.warning(f"[开仓失败] {ts_code} 资金不足或单票限额过低")
            return None

        total_cost = shares * entry_price * (1 + commission_rate)
        pos = Position(
            ts_code=ts_code,
            entry_date=entry_date,
            entry_price=entry_price,
            shares=shares,
            total_cost=total_cost,
            peak_price=entry_price,
        )
        self.positions[ts_code] = pos
        self.trades.append({
            'ts_code': ts_code, 'date': entry_date, 'action': 'buy',
            'price': entry_price, 'shares': shares, 'reason': '金叉入场',
        })
        logger.info(f"[开仓] {ts_code} @ {entry_price:.2f} × {shares}股, 成本 {total_cost:.0f}")
        return pos

    def close_position(self, ts_code: str, exit_date: str,
                       exit_price: float, reason: str,
                       ratio: float = 1.0,
                       commission_rate: float = 0.0003) -> Optional[Dict]:
        """平仓（支持部分平仓）"""
        if ts_code not in self.positions:
            return None

        pos = self.positions[ts_code]
        close_shares = int(pos.shares * ratio / 100) * 100  # 整手
        if close_shares <= 0:
            close_shares = 100 if pos.shares >= 100 else pos.shares

        proceeds = close_shares * exit_price * (1 - commission_rate)
        pnl = proceeds - pos.total_cost * (close_shares / pos.shares)
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price

        self.trades.append({
            'ts_code': ts_code, 'date': exit_date, 'action': 'sell',
            'price': exit_price, 'shares': close_shares,
            'reason': reason, 'pnl': pnl, 'pnl_pct': pnl_pct,
        })

        logger.info(
            f"[平仓] {ts_code} @ {exit_price:.2f} × {close_shares}股, "
            f"原因={reason}, 盈亏={pnl:+.0f} ({pnl_pct:+.1%})"
        )

        # 更新剩余持仓
        remaining = pos.shares - close_shares
        if remaining <= 0 or ratio >= 1.0:
            del self.positions[ts_code]
            # 连续止损统计
            if reason in ("固定止损", "时间止损"):
                self.consecutive_losses += 1
                if self.consecutive_losses >= self.params.consecutive_loss_limit:
                    # 计算熔断到期日
                    try:
                        d = datetime.strptime(exit_date, "%Y%m%d")
                        self.loss_cooldown_until = (d + timedelta(
                            days=self.params.consecutive_loss_cooldown * 2  # 粗估日历日
                        )).strftime("%Y%m%d")
                    except ValueError:
                        pass
                    logger.warning(
                        f"[熔断触发] 连续{self.consecutive_losses}笔止损，"
                        f"暂停开仓至 {self.loss_cooldown_until}"
                    )
            else:
                self.consecutive_losses = 0
        else:
            pos.shares = remaining
            pos.total_cost = pos.total_cost * (remaining / (remaining + close_shares))

        return {'ts_code': ts_code, 'pnl': pnl, 'pnl_pct': pnl_pct, 'reason': reason}

    # ─────────────── 风控检查 ───────────────
    def _is_blocked(self) -> bool:
        """是否被风控阻止开仓"""
        # 熔断检查
        if self.loss_cooldown_until and self.current_date <= self.loss_cooldown_until:
            return True
        # 日度回撤
        if self.daily_drawdown_triggered:
            return True
        # 月度清仓
        if self.monthly_liquidated:
            return True
        return False

    def _check_industry_limit(self, ts_code: str) -> bool:
        """行业集中度检查"""
        industry = self.industry_map.get(ts_code, "unknown")
        if not self.positions:
            return True
        industry_count = sum(
            1 for pos in self.positions.values()
            if self.industry_map.get(pos.ts_code, "unknown") == industry
        )
        # 简化：用持仓数量占比近似资金占比
        if industry_count >= max(1, self.params.max_positions * self.params.industry_limit):
            return False
        return True

    def update_risk_state(self, current_nav: float, current_date: str) -> List[str]:
        """
        每日更新风控状态。

        Parameters
        ----------
        current_nav : 当前组合净值
        current_date : 当前日期 YYYYMMDD

        Returns
        -------
        风控事件列表
        """
        self.current_date = current_date
        events = []

        # 重置日度标记（每个新交易日重置）
        self.daily_drawdown_triggered = False

        # 月度清仓检查
        if self.monthly_liquidated:
            # 月初重置
            if len(current_date) == 8 and current_date[6:8] == "01":
                self.monthly_liquidated = False
                events.append("月度清仓解除")

        # 更新高水位
        if current_nav > self.monthly_high_watermark:
            self.monthly_high_watermark = current_nav

        # 月度回撤检查
        monthly_dd = (self.monthly_high_watermark - current_nav) / self.monthly_high_watermark
        if monthly_dd >= self.params.max_monthly_drawdown:
            self.monthly_liquidated = True
            events.append(f"月度回撤 {monthly_dd:.1%} 超限，清仓并暂停交易至下月初")

        # 日度回撤检查（简化：用 NAV 变化）
        if len(self.nav_history) > 0:
            prev_nav = self.nav_history[-1].get('nav', current_nav)
            daily_dd = (prev_nav - current_nav) / prev_nav if prev_nav > 0 else 0
            if daily_dd >= self.params.max_daily_drawdown:
                self.daily_drawdown_triggered = True
                events.append(f"日度回撤 {daily_dd:.1%} 超限，当日禁止开新仓")

        self.nav_history.append({'date': current_date, 'nav': current_nav})
        return events


# ═══════════════════════════════════════════════════
#  回测引擎（简易版）
# ═══════════════════════════════════════════════════
class MABreakoutBacktester:
    """均线突破策略简易回测引擎"""

    def __init__(self, strategy: MABreakoutStrategy, initial_capital: float = 1_000_000):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.daily_results: List[Dict] = []

    def run(self, data: pd.DataFrame, industry_map: Dict[str, str] | None = None) -> pd.DataFrame:
        """
        运行回测。

        Parameters
        ----------
        data : 标准日线数据，需先经 prepare_features 计算特征
        industry_map : ts_code → 行业映射

        Returns
        -------
        每日净值与交易记录 DataFrame
        """
        strat = self.strategy
        p = strat.params

        if industry_map:
            strat.industry_map = industry_map

        # 按日期遍历
        dates = sorted(data['trade_date'].unique())

        for date in dates:
            day_data = data[data['trade_date'] == date].copy()

            # ── 设置总资金 ──
            if strat.total_capital <= 0:
                strat.total_capital = float(self.initial_capital)

            strat.current_date = date

            # ── 先处理出场 ──
            for ts_code in list(strat.positions.keys()):
                stock_rows = day_data[day_data['ts_code'] == ts_code]
                if stock_rows.empty:
                    continue
                row = stock_rows.iloc[0]
                exits = strat.check_exit(ts_code, row)

                # 处理跌停：记录但次日执行
                is_limit_down = row.get('pct_chg', 0) <= -p.limit_up_threshold

                for action, reason, ratio in exits:
                    exit_price = row['open']  # 次日开盘执行，简化用当日开盘价
                    if is_limit_down and reason not in ("固定止损",):
                        # 跌停时暂不执行非紧急出场
                        continue
                    result = strat.close_position(ts_code, date, exit_price, reason, ratio)
                    if result:
                        self.cash += result['pnl'] + strat.positions.get(  # 归还本金
                            ts_code, Position(ts_code, date, exit_price, 0, 0)
                        ).total_cost * ratio  # 简化

            # ── 再处理入场 ──
            for _, row in day_data.iterrows():
                # 获取上市天数（简化：用数据中该标的出现的天数）
                listing_days = len(data[data['ts_code'] == row['ts_code']]['trade_date'].unique())
                is_limit_up = row.get('pct_chg', 0) >= p.limit_up_threshold

                if strat.check_entry(row, listing_days=listing_days, is_limit_up=is_limit_up):
                    # 次日开盘价执行，简化用次日 open（这里用当日 open 近似）
                    entry_price = row['open']
                    strat.open_position(row['ts_code'], date, entry_price, self.cash)

            # ── 计算当日净值 ──
            position_value = sum(
                pos.shares * day_data[day_data['ts_code'] == code].iloc[0]['close']
                if not day_data[day_data['ts_code'] == code].empty else 0
                for code, pos in strat.positions.items()
            )
            total_nav = self.cash + position_value

            # 风控更新
            nav_ratio = total_nav / self.initial_capital
            strat.update_risk_state(nav_ratio, date)

            self.daily_results.append({
                'date': date,
                'nav': total_nav,
                'cash': self.cash,
                'position_value': position_value,
                'num_positions': len(strat.positions),
            })

        return pd.DataFrame(self.daily_results)

    def summary(self) -> Dict:
        """输出回测摘要"""
        if not self.daily_results:
            return {}

        df = pd.DataFrame(self.daily_results)
        nav = df['nav']
        total_return = (nav.iloc[-1] / nav.iloc[0]) - 1
        max_dd = ((nav.cummax() - nav) / nav.cummax()).max()
        sharpe = (nav.pct_change().mean() / nav.pct_change().std()) * np.sqrt(252) if nav.pct_change().std() > 0 else 0

        return {
            'total_return': f"{total_return:.2%}",
            'max_drawdown': f"{max_dd:.2%}",
            'sharpe_ratio': f"{sharpe:.2f}",
            'total_trades': len(self.strategy.trades),
            'final_nav': nav.iloc[-1],
        }


# ═══════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════
def _calc_true_range(df: pd.DataFrame) -> pd.Series:
    """计算 True Range"""
    high = df['high']
    low = df['low']
    prev_close = df.groupby('ts_code')['close'].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def is_main_board(ts_code: str) -> bool:
    """判断是否为A股主板标的"""
    if '.' not in ts_code:
        code = ts_code
    else:
        code = ts_code.split('.')[0]

    # 沪市主板: 600/601/603 开头
    # 深市主板: 000/001 开头
    return code.startswith(('600', '601', '603', '000', '001'))


def strip_st_stocks(stock_list: pd.DataFrame) -> pd.DataFrame:
    """排除 ST/*ST 标的"""
    if 'name' in stock_list.columns:
        mask = ~stock_list['name'].str.contains(r'ST', case=False, na=False)
        return stock_list[mask]
    return stock_list


# ═══════════════════════════════════════════════════
#  便捷入口
# ═══════════════════════════════════════════════════
def create_strategy(**overrides) -> MABreakoutStrategy:
    """快速创建策略实例，可覆盖默认参数"""
    params = MABreakoutParams(**overrides)
    return MABreakoutStrategy(params)


if __name__ == "__main__":
    # 演示：用随机数据跑通完整流程
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    np.random.seed(42)
    dates = pd.date_range("20230101", periods=120, freq="B")
    ts_code = "600519.SH"

    df_demo = pd.DataFrame({
        'ts_code': ts_code,
        'trade_date': dates.strftime("%Y%m%d"),
        'open': 1800 + np.cumsum(np.random.randn(120) * 5),
        'high': 1810 + np.cumsum(np.random.randn(120) * 5),
        'low': 1790 + np.cumsum(np.random.randn(120) * 5),
        'close': 1805 + np.cumsum(np.random.randn(120) * 5),
        'vol': np.random.randint(10000, 50000, 120).astype(float),
        'amount': np.random.randint(1e7, 5e7, 120).astype(float),
        'pct_chg': np.random.randn(120) * 2,
        'turnover_ratio': np.random.rand(120) * 3,
        'pre_close': 1800 + np.cumsum(np.random.randn(120) * 5),
    })
    # 保证 high >= close >= low
    df_demo['high'] = df_demo[['high', 'close', 'open']].max(axis=1) + 5
    df_demo['low'] = df_demo[['low', 'close', 'open']].min(axis=1) - 5

    strat = create_strategy()
    df_feat = MABreakoutStrategy.prepare_features(df_demo)
    bt = MABreakoutBacktester(strat, initial_capital=1_000_000)
    result = bt.run(df_feat)
    print("\n=== 回测摘要 ===")
    for k, v in bt.summary().items():
        print(f"  {k}: {v}")
