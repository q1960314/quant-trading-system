"""Cross-strategy candidate ranking with resonance bonus."""
import pandas as pd
import numpy as np


def merge_candidates(strategy_picks: dict, resonance_bonus: float = 0.1,
                     top_n: int = 20) -> pd.DataFrame:
    all_candidates = []
    for sname, df in strategy_picks.items():
        if df.empty:
            continue
        df = df.copy()
        df['strategy'] = sname
        score_col = 'total_score' if 'total_score' in df.columns else df.columns[0]
        score_min, score_max = df[score_col].min(), df[score_col].max()
        if score_max > score_min:
            df['norm_score'] = (df[score_col] - score_min) / (score_max - score_min)
        else:
            df['norm_score'] = 0.5
        all_candidates.append(df[['ts_code', 'norm_score', 'strategy']])
    if not all_candidates:
        return pd.DataFrame(columns=['ts_code', 'combined_score', 'strategy_count', 'strategies'])
    merged = pd.concat(all_candidates)
    agg = merged.groupby('ts_code').agg(
        combined_score=('norm_score', 'mean'),
        strategy_count=('norm_score', 'count'),
        strategies=('strategy', lambda x: ','.join(sorted(set(x)))),
    ).reset_index()
    agg['combined_score'] = agg['combined_score'] * (1 + resonance_bonus * (agg['strategy_count'] - 1))
    return agg.sort_values('combined_score', ascending=False).head(top_n).reset_index(drop=True)
