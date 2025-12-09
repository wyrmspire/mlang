import pandas as pd
import numpy as np

path = 'data/processed/smart_reverse_trades.parquet'
df = pd.read_parquet(path)

print(f"Total Intervals: {len(df)}")
p_follow = 0.43 # Threshold used
p_fade = 0.39   # Threshold used

# 1. Follow Only
# If p_long > p_follow -> PnL Long
# If p_short > p_follow -> PnL Short
# Tie-break? Max freq.
# df columns: time, p_long, p_short, pnl_long, pnl_short, outcome_long, outcome_short

# Vectorized Analysis
# Best Prob
best_prob = df[['p_long', 'p_short']].max(axis=1)
long_best = df['p_long'] >= df['p_short']

# Follow
mask_follow = best_prob > p_follow
pnl_follow = np.where(long_best, df['pnl_long'], df['pnl_short'])
total_follow = pnl_follow[mask_follow].sum()
count_follow = mask_follow.sum()

# Fade Losers (Prob < p_fade)
# If p_long < p_fade -> Short (Take Short PnL)
# If p_short < p_fade -> Long (Take Long PnL)
# Prioritize? If both bad? 
# Logic in script: 
# If p_long < p_fade: Fade Long (Go Short).
# If p_short < p_fade: Fade Short (Go Long).
# Both? Add both? (Script likely added both to hybrid_pnl or distinct).
# Script logic:
# if pl < low: fade_losers_pnl += pnl_short
# if ps < low: fade_losers_pnl += pnl_long

mask_fade_long = df['p_long'] < p_fade
pnl_fade_long = df['pnl_short'][mask_fade_long].sum()
count_fade_long = mask_fade_long.sum()

mask_fade_short = df['p_short'] < p_fade
pnl_fade_short = df['pnl_long'][mask_fade_short].sum()
count_fade_short = mask_fade_short.sum()

total_fade = pnl_fade_long + pnl_fade_short
count_fade = count_fade_long + count_fade_short

# Hybrid
total_hybrid = total_follow + total_fade
count_hybrid = count_follow + count_fade

# Fade Winners (Inverse of Follow)
# If Follow would go Long, we go Short.
pnl_fade_winners_arr = np.where(long_best, df['pnl_short'], df['pnl_long'])
total_fade_winners = pnl_fade_winners_arr[mask_follow].sum()

print("-" * 30)
print(f"1. Follow Only (> {p_follow}):")
print(f"   Trades: {count_follow} | PnL: {total_follow:.2f}")
print("-" * 30)
print(f"2. Fade Losers (< {p_fade}):")
print(f"   Trades: {count_fade} | PnL: {total_fade:.2f}")
print("-" * 30)
print(f"3. Hybrid (Follow + Fade Losers):")
print(f"   Trades: {count_hybrid} | PnL: {total_hybrid:.2f}")
print("-" * 30)
print(f"4. Fade Winners (Contrarian):")
print(f"   Trades: {count_follow} | PnL: {total_fade_winners:.2f}")
