"""Data validation: completeness checks against trade calendar."""
import pandas as pd


def check_completeness(daily_df: pd.DataFrame, trade_cal: pd.DataFrame,
                       start: str, end: str) -> dict:
    expected_dates = set(trade_cal[
        (trade_cal['cal_date'] >= start) &
        (trade_cal['cal_date'] <= end) &
        (trade_cal['is_open'] == 1)
    ]['cal_date'].values)
    if daily_df.empty:
        return {'total_stocks': 0, 'avg_completeness': 0, 'missing_dates': 0}
    result = {}
    for code in daily_df['ts_code'].unique():
        stock_dates = set(daily_df[daily_df['ts_code'] == code]['trade_date'].values)
        result[code] = len(stock_dates & expected_dates) / len(expected_dates) if expected_dates else 0
    return {
        'total_stocks': len(result),
        'avg_completeness': sum(result.values()) / len(result) if result else 0,
        'stocks_below_80pct': sum(1 for v in result.values() if v < 0.8),
    }
