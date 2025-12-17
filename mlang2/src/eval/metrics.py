"""
Trade Metrics
Expectancy, win rate, payoff ratio, drawdown.
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from src.datasets.trade_record import TradeRecord
from src.datasets.decision_record import DecisionRecord


@dataclass
class TradeMetrics:
    """Comprehensive trade metrics."""
    total_trades: int
    wins: int
    losses: int
    timeouts: int
    win_rate: float
    
    # PnL
    total_pnl: float
    avg_pnl: float
    avg_win: float
    avg_loss: float
    
    # Risk-adjusted
    payoff_ratio: float      # avg_win / avg_loss
    expectancy: float        # (WR * avg_win) - ((1-WR) * avg_loss)
    profit_factor: float     # gross_win / gross_loss
    
    # Drawdown
    max_drawdown: float
    max_drawdown_pct: float
    
    # Additional
    avg_bars_held: float
    avg_r_multiple: float


def compute_trade_metrics(trades: List[TradeRecord]) -> TradeMetrics:
    """Compute metrics from trade records."""
    if not trades:
        return TradeMetrics(
            total_trades=0, wins=0, losses=0, timeouts=0, win_rate=0,
            total_pnl=0, avg_pnl=0, avg_win=0, avg_loss=0,
            payoff_ratio=0, expectancy=0, profit_factor=0,
            max_drawdown=0, max_drawdown_pct=0,
            avg_bars_held=0, avg_r_multiple=0
        )
    
    # Basic counts
    wins = [t for t in trades if t.outcome == 'WIN']
    losses = [t for t in trades if t.outcome == 'LOSS']
    timeouts = [t for t in trades if t.outcome == 'TIMEOUT']
    
    win_count = len(wins)
    loss_count = len(losses)
    total = len(trades)
    
    win_rate = win_count / total if total > 0 else 0
    
    # PnL
    total_pnl = sum(t.pnl_dollars for t in trades)
    avg_pnl = total_pnl / total if total > 0 else 0
    
    avg_win = np.mean([t.pnl_dollars for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t.pnl_dollars) for t in losses]) if losses else 0
    
    # Risk-adjusted
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    gross_win = sum(t.pnl_dollars for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl_dollars for t in losses)) if losses else 0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float('inf')
    
    # Drawdown
    equity_curve = np.cumsum([t.pnl_dollars for t in trades])
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = peak - equity_curve
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0
    max_dd_pct = max_dd / np.max(peak) if np.max(peak) > 0 else 0
    
    # Additional
    avg_bars = np.mean([t.bars_held for t in trades])
    avg_r = np.mean([t.r_multiple for t in trades if t.r_multiple != 0])
    
    return TradeMetrics(
        total_trades=total,
        wins=win_count,
        losses=loss_count,
        timeouts=len(timeouts),
        win_rate=win_rate,
        total_pnl=total_pnl,
        avg_pnl=avg_pnl,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=payoff_ratio,
        expectancy=expectancy,
        profit_factor=profit_factor,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        avg_bars_held=avg_bars,
        avg_r_multiple=avg_r if not np.isnan(avg_r) else 0,
    )


def compute_from_records(records: List[DecisionRecord]) -> TradeMetrics:
    """Compute metrics from decision records using counterfactual outcomes."""
    if not records:
        return compute_trade_metrics([])
    
    # Convert counterfactual outcomes to simple format
    class SimpleRecord:
        def __init__(self, r: DecisionRecord):
            self.outcome = r.cf_outcome
            self.pnl_dollars = r.cf_pnl_dollars
            self.bars_held = r.cf_bars_held
            self.r_multiple = 0  # Not tracked in decision records
    
    simple = [SimpleRecord(r) for r in records if r.cf_outcome in ['WIN', 'LOSS', 'TIMEOUT']]
    
    # Reuse computation
    return compute_trade_metrics(simple)
