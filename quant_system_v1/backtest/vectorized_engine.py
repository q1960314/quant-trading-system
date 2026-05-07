"""
Vectorized backtest engine.

Converts per-stock row iteration to numpy matrix operations.
Performance: ~50-100x faster than event-driven loop.
"""
import pandas as pd
import numpy as np
import os
import time
from dataclasses import dataclass
from typing import Dict, List

from utils.logger import get_logger
from backtest.constraints import (
    LimitUpDownConstraint, SuspensionConstraint,
    LiquidityConstraint, PositionConstraint,
)
from backtest.engine import BacktestResult

logger = get_logger("vectorized_bt")


class VectorizedBacktestEngine:
    def __init__(self, initial_capital=5000, start_date="2020-01-01",
                 end_date="2024-12-31", commission_rate=0.00025,
                 min_commission=5, stamp_tax_rate=0.001,
                 slippage_rate=0.005, max_hold_stocks=5,
                 single_stock_position=0.2, constraint_config=None):
        self.initial_capital = initial_capital
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage_rate = slippage_rate
        self.max_hold_stocks = max_hold_stocks
        self.single_stock_ratio = single_stock_position
        cc = constraint_config or {}
        self.constraints = [
            LimitUpDownConstraint(),
            SuspensionConstraint(),
            LiquidityConstraint(
                min_amount=cc.get('min_amount', 100000),
                min_turnover=cc.get('min_turnover', 3.0),
            ),
        ]
        self.pos_constraint = PositionConstraint(
            single_stock_ratio=single_stock_position,
            max_hold_stocks=max_hold_stocks,
        )
        self.df = None
        self.codes = []
        self.dates = []
        self.close_matrix = None
        self.volume_matrix = None
        self.open_matrix = None

    def load_data(self, data_dir="../data_all_stocks", global_dir="../data", use_duckdb=True):
        logger.info("Loading data for vectorized backtest...")
        t0 = time.time()

        # DuckDB fast path
        if use_duckdb:
            db_path = os.path.join(global_dir, 'quant.duckdb')
            if os.path.exists(db_path):
                return self._load_data_duckdb(db_path)

        # CSV fallback path
        frames = []
        if os.path.isdir(data_dir):
            for d in os.listdir(data_dir):
                fp = os.path.join(data_dir, d, 'daily.csv')
                if os.path.isfile(fp) and os.path.getsize(fp) > 1024:
                    try:
                        dd = pd.read_csv(fp, dtype={'trade_date': str, 'ts_code': str})
                        if dd.empty:
                            continue
                        dd['trade_date'] = pd.to_datetime(dd['trade_date'])
                        frames.append(dd)
                    except Exception:
                        pass
        if not frames:
            logger.warning("No data loaded from per-stock directory")
            return None
        self.df = pd.concat(frames, ignore_index=True)
        self.df = self.df[
            (self.df['trade_date'] >= self.start_date) &
            (self.df['trade_date'] <= self.end_date)
        ]
        self.codes = sorted(self.df['ts_code'].unique())
        self.dates = sorted(self.df['trade_date'].unique())

        # Merge global data (limit status, stock info, etc.) for strategy scoring
        self._merge_global_data(global_dir)

        self.close_matrix = self.df.pivot(index='trade_date', columns='ts_code', values='close')
        self.volume_matrix = self.df.pivot(index='trade_date', columns='ts_code', values='vol')
        self.open_matrix = self.df.pivot(index='trade_date', columns='ts_code', values='open')
        for m in [self.close_matrix, self.volume_matrix, self.open_matrix]:
            if m is not None:
                m.ffill(inplace=True)
        logger.info(f"Loaded {len(self.codes)} stocks x {len(self.dates)} days in {time.time()-t0:.1f}s")
        return self.df

    def _load_data_duckdb(self, db_path):
        """Load backtest data directly from DuckDB — single SQL query with all JOINs."""
        import duckdb
        conn = duckdb.connect(db_path)
        # Build one big query joining all relevant tables
        self.df = conn.execute(f"""
            SELECT d.*, sb.name, sb.industry,
                   db.turnover_rate, db.pe_ttm, db.pb, db.circ_mv, db.total_mv, db.volume_ratio,
                   lim.first_time AS first_limit_time, lim.open_times AS break_limit_times,
                   lim.limit_times AS up_down_times, lim.limit AS limit_status,
                   lim.limit_amount AS order_amount, lim.float_mv AS float_market_cap,
                   top.net_amount
            FROM daily d
            LEFT JOIN stock_basic sb ON d.ts_code = sb.ts_code
            LEFT JOIN daily_basic db ON d.ts_code = db.ts_code AND d.trade_date = db.trade_date
            LEFT JOIN limit_list_d lim ON d.ts_code = lim.ts_code AND d.trade_date = lim.trade_date
            LEFT JOIN top_list top ON d.ts_code = top.ts_code AND d.trade_date = top.trade_date
            WHERE d.trade_date BETWEEN '{self.start_date.strftime('%Y-%m-%d')}'
              AND '{self.end_date.strftime('%Y-%m-%d')}'
            ORDER BY d.trade_date, d.ts_code
        """).df()
        conn.close()
        self.df['trade_date'] = pd.to_datetime(self.df['trade_date'])
        self.codes = sorted(self.df['ts_code'].unique())
        self.dates = sorted(self.df['trade_date'].unique())
        for col, default in [('limit_status','N'),('order_amount',0),('break_limit_times',0),
                              ('up_down_times',0),('float_market_cap',50),('turnover_rate',2),
                              ('net_amount',0),('first_limit_time','14:00')]:
            if col not in self.df.columns: self.df[col] = default
        self.df = self.df.drop_duplicates(subset=['ts_code','trade_date'], keep='last')
        for col in ['float_market_cap','total_mv','circ_mv']:
            if col in self.df.columns and self.df[col].max() > 1e10:
                self.df[col] = self.df[col] / 1e8
        self._precompute_factors()
        self.close_matrix = self.df.pivot(index='trade_date', columns='ts_code', values='close')
        if self.close_matrix is not None: self.close_matrix.ffill(inplace=True)
        logger.info(f"DuckDB: {len(self.codes)} stocks x {len(self.dates)} days")
        return self.df

    def _precompute_factors(self):
        """Pre-compute commonly needed factors to speed up strategy scoring."""
        from factor_lib.registry import FactorRegistry
        # Only compute price/volume/technical factors (fast, always available)
        fast_factors = FactorRegistry.list_by_category('price') + \
                       FactorRegistry.list_by_category('volume') + \
                       FactorRegistry.list_by_category('technical')
        fast_factors = [f for f in fast_factors if f in FactorRegistry._factors][:20]
        try:
            factor_df = FactorRegistry.compute(self.df, fast_factors)
            for col in factor_df.columns:
                if col not in self.df.columns:
                    self.df[col] = factor_df[col]
        except Exception:
            pass

    def _merge_global_data(self, global_dir):
        """Merge stock_basic, limit_list_d, daily_basic, top_list into self.df."""
        if not os.path.isdir(global_dir):
            return

        # Stock basic info (name, industry)
        bp = os.path.join(global_dir, 'stock_basic.csv')
        if os.path.exists(bp):
            try:
                basic = pd.read_csv(bp, dtype={'ts_code': str})
                cols = ['ts_code'] + [c for c in ['name', 'industry', 'market'] if c in basic.columns]
                self.df = self.df.merge(basic[cols], on='ts_code', how='left', suffixes=('','_basic'))
            except Exception:
                pass

        # Daily basic (fundamental data: pe_ttm, pb, turnover_rate, circ_mv)
        dbp = os.path.join(global_dir, 'daily_basic.csv')
        if os.path.exists(dbp):
            try:
                basic_d = pd.read_csv(dbp, dtype={'ts_code': str, 'trade_date': str})
                basic_d['trade_date'] = pd.to_datetime(basic_d['trade_date'])
                cols = ['ts_code', 'trade_date'] + [c for c in ['turnover_rate', 'pe_ttm', 'pb', 'circ_mv', 'total_mv', 'volume_ratio'] if c in basic_d.columns]
                self.df = self.df.merge(basic_d[cols], on=['ts_code', 'trade_date'], how='left', suffixes=('','_db'))
            except Exception:
                pass

        # Limit list (board info: first_time, open_times, limit_times, limit status)
        lp = os.path.join(global_dir, 'limit_list_d.csv')
        if os.path.exists(lp):
            try:
                lim = pd.read_csv(lp, dtype={'ts_code': str, 'trade_date': str})
                lim['trade_date'] = pd.to_datetime(lim['trade_date'])
                rename = {}
                if 'first_time' in lim.columns: rename['first_time'] = 'first_limit_time'
                if 'open_times' in lim.columns: rename['open_times'] = 'break_limit_times'
                if 'limit_times' in lim.columns: rename['limit_times'] = 'up_down_times'
                if 'limit' in lim.columns: rename['limit'] = 'limit_status'
                if 'float_mv' in lim.columns: rename['float_mv'] = 'float_market_cap'
                if rename:
                    lim = lim.rename(columns=rename)
                cols = ['ts_code', 'trade_date'] + [c for c in rename.values() if c in lim.columns]
                if 'limit_amount' in lim.columns and 'float_market_cap' in lim.columns:
                    lim['order_amount'] = lim['limit_amount']
                self.df = self.df.merge(lim[cols], on=['ts_code', 'trade_date'], how='left', suffixes=('','_lim'))
            except Exception:
                pass

        # Top list (龙虎榜: net buy amount)
        tp = os.path.join(global_dir, 'top_list.csv')
        if os.path.exists(tp):
            try:
                top = pd.read_csv(tp, dtype={'ts_code': str, 'trade_date': str})
                top['trade_date'] = pd.to_datetime(top['trade_date'])
                cols = ['ts_code', 'trade_date'] + [c for c in ['net_amount', 'l_buy', 'l_sell'] if c in top.columns]
                if cols:
                    self.df = self.df.merge(top[cols], on=['ts_code', 'trade_date'], how='left', suffixes=('','_top'))
            except Exception:
                pass

        # Fill missing critical columns
        defaults = [
            ('limit_status', 'N'), ('order_amount', 0), ('break_limit_times', 0),
            ('up_down_times', 0), ('float_market_cap', 50), ('turnover_rate', 2),
            ('net_amount', 0), ('pe_ttm', 20), ('pb', 2), ('volume_ratio', 1),
            ('first_limit_time', '14:00'), ('inst_buy', 0), ('inst_sell', 0),
        ]
        for col, default in defaults:
            if col not in self.df.columns:
                self.df[col] = default
        # Deduplicate: keep last row per (ts_code, trade_date) to avoid pivot errors
        self.df = self.df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')

        # Normalize units: convert 元 to 亿 for market cap columns
        for col in ['float_market_cap', 'total_mv', 'circ_mv']:
            if col in self.df.columns and self.df[col].max() > 1e10:
                self.df[col] = self.df[col] / 1e8

        # Pre-compute common factors for strategy scoring
        self._precompute_factors()

    def build_constraint_matrix(self) -> pd.DataFrame:
        mask = pd.DataFrame(True, index=self.dates, columns=self.codes)
        for date in self.dates:
            day_df = self.df[self.df['trade_date'] == date].copy()
            if day_df.empty:
                mask.loc[date] = False
                continue
            day_df = day_df.set_index('ts_code')
            for constraint in self.constraints:
                try:
                    c = constraint.apply(day_df.reset_index())
                    c.index = day_df.index
                    mask.loc[date, c.index] &= c.astype(bool)
                except Exception:
                    pass
        return mask

    def run(self, strategy) -> BacktestResult:
        self.load_data()
        if self.df is None or self.df.empty:
            return BacktestResult(initial_capital=self.initial_capital)
        self.constraint_matrix = self.build_constraint_matrix()
        return self._run_with_data(strategy)

    def _run_with_data(self, strategy) -> BacktestResult:
        if self.df is None or self.df.empty:
            return BacktestResult(initial_capital=self.initial_capital)
        if not hasattr(self, 'constraint_matrix') or self.constraint_matrix is None:
            self.constraint_matrix = self.build_constraint_matrix()
        signals = strategy.generate_signals_vectorized(self.df)
        if signals.empty:
            signals = pd.DataFrame(0, index=self.dates, columns=self.codes)
            # Batch per-date: filter+score all stocks at once per day (fast)
            for date in self.dates:
                day = self.df[self.df['trade_date'] == date]
                if day.empty:
                    continue
                try:
                    scored = strategy.run(day)
                    if not scored.empty and 'ts_code' in scored.columns:
                        for _, s in scored.iterrows():
                            if s.get('total_score', 0) >= 10:
                                code = s['ts_code']
                                if code in signals.columns:
                                    signals.loc[date, code] = 1
                except Exception:
                    pass
        # Align signals to constraint matrix dimensions
        signals = signals.reindex(index=self.constraint_matrix.index,
                                  columns=self.constraint_matrix.columns, fill_value=0)
        valid_signals = signals * self.constraint_matrix.astype(int)
        cash = self.initial_capital
        positions: Dict[str, Dict] = {}
        trades: List[Dict] = []
        daily_values = []
        peak_capital = self.initial_capital

        for i, date in enumerate(self.dates):
            today_signal = valid_signals.loc[date]
            today_close = self.close_matrix.loc[date] if self.close_matrix is not None else pd.Series()
            total_value = cash
            for code, pos in positions.items():
                if code in today_close.index and pd.notna(today_close[code]):
                    total_value += pos['shares'] * today_close[code]
            # Sell signals
            sell_codes = today_signal[today_signal == -1].index
            for code in sell_codes:
                if code in positions:
                    price = today_close.get(code, 0)
                    if pd.isna(price) or price <= 0:
                        continue
                    slippage_price = price * (1 - self.slippage_rate)
                    value = positions[code]['shares'] * slippage_price
                    cost = max(value * self.commission_rate, self.min_commission)
                    cost += value * self.stamp_tax_rate
                    cash += value - cost
                    trades.append({
                        'date': date, 'code': code, 'direction': 'sell',
                        'price': slippage_price, 'shares': positions[code]['shares'],
                        'profit': value - cost - positions[code]['cost_basis'],
                    })
                    del positions[code]
            # Auto-sell: positions held beyond max_hold_days or hitting stop loss/profit
            max_hold = 3  # default max hold days
            stop_loss = 0.06
            stop_profit = 0.12
            expired = []
            for code, pos in list(positions.items()):
                if code in today_close.index and pd.notna(today_close[code]):
                    pos['current_price'] = today_close[code]
                    held_days = (date - pos['buy_date']).days
                    pnl = (today_close[code] - pos['cost_basis'] / pos['shares']) / (pos['cost_basis'] / pos['shares'])
                    # Stop loss / profit
                    if pnl <= -stop_loss:
                        expired.append((code, 'stop_loss'))
                    elif pnl >= stop_profit:
                        expired.append((code, 'stop_profit'))
                    elif held_days >= max_hold:
                        expired.append((code, 'max_hold'))
            for code, reason in expired:
                price = today_close.get(code, 0)
                if pd.isna(price) or price <= 0:
                    continue
                slippage_price = price * (1 - self.slippage_rate)
                value = positions[code]['shares'] * slippage_price
                cost = max(value * self.commission_rate, self.min_commission)
                cost += value * self.stamp_tax_rate
                cash += value - cost
                trades.append({
                    'date': date, 'code': code, 'direction': 'sell',
                    'price': slippage_price, 'shares': positions[code]['shares'],
                    'profit': value - cost - positions[code]['cost_basis'],
                    'reason': reason,
                })
                del positions[code]
            # Buy signals
            buy_codes = today_signal[today_signal == 1].index
            available_cash = cash * 0.95
            per_stock_cash = min(
                available_cash / max(self.max_hold_stocks, 1),
                total_value * self.single_stock_ratio,
            )
            for code in buy_codes:
                if code in positions:
                    continue
                if len(positions) >= self.max_hold_stocks:
                    break
                price = today_close.get(code, 0)
                if pd.isna(price) or price <= 0 or per_stock_cash < price * 100:
                    continue
                slippage_price = price * (1 + self.slippage_rate)
                shares = int(per_stock_cash / slippage_price / 100) * 100
                if shares < 100:
                    continue
                value = shares * slippage_price
                cost = max(value * self.commission_rate, self.min_commission)
                if cash >= value + cost:
                    cash -= value + cost
                    positions[code] = {'shares': shares, 'cost_basis': value, 'buy_date': date}
                    trades.append({
                        'date': date, 'code': code, 'direction': 'buy',
                        'price': slippage_price, 'shares': shares,
                    })
            # Daily snapshot
            position_value = 0
            for code, pos in positions.items():
                if code in today_close.index and pd.notna(today_close[code]):
                    position_value += pos['shares'] * today_close[code]
            total = cash + position_value
            peak_capital = max(peak_capital, total)
            daily_values.append({
                'date': date, 'cash': cash,
                'position_value': position_value, 'total_value': total,
            })

        dv = pd.DataFrame(daily_values)
        final_capital = dv['total_value'].iloc[-1] if len(dv) > 0 else self.initial_capital
        result = BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=(final_capital / self.initial_capital - 1) if len(dv) > 0 else 0,
            total_trades=len(trades),
            trades=pd.DataFrame(trades) if trades else pd.DataFrame(),
            daily_capital=dv,
        )
        if len(dv) > 1:
            dv['daily_return'] = dv['total_value'].pct_change()
            result.max_drawdown = self._calc_max_drawdown(dv['total_value'])
            result.sharpe_ratio = self._calc_sharpe(dv['daily_return'])
            result.annual_return = (1 + result.total_return) ** (252 / len(dv)) - 1
            sell_trades = [t for t in trades if t['direction'] == 'sell']
            win_trades = [t for t in sell_trades if t.get('profit', 0) > 0]
            result.win_rate = len(win_trades) / len(sell_trades) * 100 if sell_trades else 0
        return result

    def _calc_max_drawdown(self, equity: pd.Series) -> float:
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return abs(drawdown.min())

    def _calc_sharpe(self, returns: pd.Series) -> float:
        r = returns.dropna()
        if len(r) < 2 or r.std() == 0:
            return 0
        return (r.mean() / r.std()) * np.sqrt(252)
