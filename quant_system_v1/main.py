#!/usr/bin/env python
"""
量化交易系统 v1 — 主入口
用法:
  python main.py fetch                    # 全量数据抓取
  python main.py fetch --mode incremental # 增量数据抓取
  python main.py backtest                 # 回测（默认打板策略）
  python main.py backtest --strategy 缩量潜伏策略
  python main.py pick                     # 每日选股
  python main.py monitor                  # 实时监控
"""
import sys
import os
import argparse

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import STRATEGY_TYPE, INIT_CAPITAL, START_DATE, END_DATE
from utils.logger import get_logger
from strategy import StrategyRegistry
from data_source.fetcher import DataFetcher
from backtest import BacktestEngine
from stock_picker import DailyStockPicker

# 加载策略（触发注册）
import strategy.board_strategy
import strategy.shrink_volume
import strategy.sector_rotation
import strategy.ma_breakout_wrapper

logger = get_logger("main")


def cmd_fetch(args):
    """数据抓取"""
    fetcher = DataFetcher()
    if args.mode == 'incremental':
        fetcher.fetch_incremental()
    else:
        fetcher.fetch_full(start=args.start, end=args.end, markets=args.markets)


def cmd_backtest(args):
    """回测"""
    strategy_name = args.strategy or STRATEGY_TYPE
    if strategy_name not in StrategyRegistry.list_all():
        logger.error(f"未知策略: {strategy_name}。可用: {StrategyRegistry.list_all()}")
        return

    if args.engine == 'vectorized':
        from backtest.vectorized_engine import VectorizedBacktestEngine
        from config.settings import CONSTRAINT_CONFIG
        engine = VectorizedBacktestEngine(
            initial_capital=args.capital or INIT_CAPITAL,
            start_date=args.start or START_DATE,
            end_date=args.end or END_DATE,
            constraint_config=CONSTRAINT_CONFIG,
        )
        strategy = StrategyRegistry.get(strategy_name)
        result = engine.run(strategy)
    else:
        engine = BacktestEngine(
            strategy_name=strategy_name,
            initial_capital=args.capital or INIT_CAPITAL,
            start_date=args.start or START_DATE,
            end_date=args.end or END_DATE,
        )
        df = engine.load_data()
        result = engine.run(df)

    # 导出报告
    if args.export and result.trades is not None and not result.trades.empty:
        import pandas as pd
        fname = f"回测报告_{strategy_name}_{args.start}_{args.end}.xlsx"
        try:
            with pd.ExcelWriter(fname, engine='openpyxl') as w:
                import pandas as pd
                result.trades.to_excel(w, sheet_name='交易记录', index=False)
                if not result.daily_capital.empty:
                    result.daily_capital.to_excel(w, sheet_name='资金曲线', index=False)
                pd.DataFrame([{
                    '策略': strategy_name, '初始资金': result.initial_capital,
                    '最终资金': f"{result.final_capital:,.0f}",
                    '总收益率': f"{result.total_return*100:.2f}%",
                    '年化收益率': f"{result.annual_return*100:.2f}%",
                    '最大回撤': f"{result.max_drawdown*100:.2f}%",
                    '夏普比率': f"{result.sharpe_ratio:.2f}",
                    '交易次数': result.total_trades,
                    '胜率': f"{result.win_rate:.1f}%",
                    '盈亏比': f"{result.profit_factor:.2f}",
                }]).to_excel(w, sheet_name='核心指标', index=False)
            logger.info(f"回测报告已导出: {fname}")
        except Exception as e:
            logger.warning(f"报告导出失败: {e}")


def cmd_pick(args):
    """每日选股"""
    picker = DailyStockPicker(strategy_name=args.strategy or STRATEGY_TYPE)
    df_latest, trade_date = picker.get_latest_data()
    picker.pick(df_latest, trade_date)


def cmd_monitor(args):
    """实时监控"""
    from monitor.realtime import RealtimeMonitor
    mon = RealtimeMonitor(strategy_name=args.strategy or STRATEGY_TYPE)
    mon.run(interval=args.interval or 60)


def main():
    parser = argparse.ArgumentParser(description="量化交易系统 v1")
    sub = parser.add_subparsers(dest='command')

    # fetch
    p = sub.add_parser('fetch', help='数据抓取')
    p.add_argument('--mode', choices=['full', 'incremental'], default='full')
    p.add_argument('--start', help='开始日期 YYYY-MM-DD')
    p.add_argument('--end', help='结束日期 YYYY-MM-DD')
    p.add_argument('--markets', nargs='*', default=['主板'], help='板块列表')

    # backtest
    p = sub.add_parser('backtest', help='回测')
    p.add_argument('--strategy', help='策略名称')
    p.add_argument('--start', help='开始日期')
    p.add_argument('--end', help='结束日期')
    p.add_argument('--capital', type=int, help='初始资金')
    p.add_argument('--engine', choices=['event', 'vectorized'], default='vectorized',
                   help='回测引擎: event(事件驱动) / vectorized(向量化)')
    p.add_argument('--export', action='store_true', default=True, help='导出Excel报告')

    # pick
    p = sub.add_parser('pick', help='每日选股')
    p.add_argument('--strategy', help='策略名称')

    # monitor
    p = sub.add_parser('monitor', help='实时监控')
    p.add_argument('--strategy', help='策略名称')
    p.add_argument('--interval', type=int, default=60, help='刷新间隔(秒)')

    args = parser.parse_args()

    print("=" * 60)
    print("  量化交易系统 v1")
    print(f"  可用策略: {StrategyRegistry.list_all()}")
    print("=" * 60)

    commands = {'fetch': cmd_fetch, 'backtest': cmd_backtest, 'pick': cmd_pick, 'monitor': cmd_monitor}
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
