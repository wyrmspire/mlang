"""
Account
Position and PnL tracking.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd

from src.sim.execution import Fill
from src.sim.costs import CostModel, DEFAULT_COSTS


@dataclass
class Position:
    """Active position."""
    direction: str
    entry_price: float
    size: int
    entry_bar: int
    entry_time: Optional[pd.Timestamp] = None
    
    # Tracking
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0


@dataclass
class TradeRecord:
    """Completed trade record."""
    direction: str
    entry_price: float
    exit_price: float
    size: int
    entry_bar: int
    exit_bar: int
    entry_time: Optional[pd.Timestamp] = None
    exit_time: Optional[pd.Timestamp] = None
    
    # Outcome
    outcome: str = ""  # 'WIN', 'LOSS', 'TIMEOUT'
    pnl: float = 0.0
    gross_pnl: float = 0.0
    commission: float = 0.0
    
    # Analytics
    bars_held: int = 0
    mae: float = 0.0
    mfe: float = 0.0
    r_multiple: float = 0.0  # PnL / initial risk


class Account:
    """
    Trading account with position and PnL tracking.
    """
    
    def __init__(
        self,
        starting_balance: float = 50000.0,
        costs: CostModel = None
    ):
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.costs = costs or DEFAULT_COSTS
        
        self.positions: List[Position] = []
        self.trades: List[TradeRecord] = []
        
        # Running stats
        self.realized_pnl = 0.0
        self.peak_balance = starting_balance
        self.max_drawdown = 0.0
    
    def open_position(
        self,
        fill: Fill,
        stop_loss: float = None,
        take_profit: float = None,
        time: pd.Timestamp = None
    ) -> Position:
        """Open new position from fill."""
        position = Position(
            direction=fill.direction,
            entry_price=fill.fill_price,
            size=fill.size,
            entry_bar=fill.fill_bar,
            entry_time=time,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self.positions.append(position)
        return position
    
    def close_position(
        self,
        position: Position,
        fill: Fill,
        outcome: str = "",
        mae: float = 0.0,
        mfe: float = 0.0,
        time: pd.Timestamp = None
    ) -> TradeRecord:
        """Close position and record trade."""
        # Calculate PnL
        gross_pnl = self.costs.calculate_pnl(
            position.entry_price,
            fill.fill_price,
            position.direction,
            position.size,
            include_commission=False
        )
        
        commission = self.costs.calculate_commission(position.size, round_trip=True)
        net_pnl = gross_pnl - commission
        
        # Calculate R-multiple if we have stop loss
        r_multiple = 0.0
        if position.stop_loss:
            initial_risk = abs(position.entry_price - position.stop_loss) * self.costs.point_value * position.size
            if initial_risk > 0:
                r_multiple = net_pnl / initial_risk
        
        # Determine outcome if not provided
        if not outcome:
            if net_pnl > 0:
                outcome = 'WIN'
            elif net_pnl < 0:
                outcome = 'LOSS'
            else:
                outcome = 'BREAKEVEN'
        
        # Create trade record
        trade = TradeRecord(
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=fill.fill_price,
            size=position.size,
            entry_bar=position.entry_bar,
            exit_bar=fill.fill_bar,
            entry_time=position.entry_time,
            exit_time=time,
            outcome=outcome,
            pnl=net_pnl,
            gross_pnl=gross_pnl,
            commission=commission,
            bars_held=fill.fill_bar - position.entry_bar,
            mae=mae,
            mfe=mfe,
            r_multiple=r_multiple,
        )
        
        # Update account
        self.trades.append(trade)
        self.positions.remove(position)
        self.balance += net_pnl
        self.realized_pnl += net_pnl
        
        # Update drawdown tracking
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        drawdown = self.peak_balance - self.balance
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        return trade
    
    def get_equity(self, current_price: float) -> float:
        """Get current equity (balance + unrealized)."""
        unrealized = 0.0
        for pos in self.positions:
            unrealized += self.costs.calculate_pnl(
                pos.entry_price,
                current_price,
                pos.direction,
                pos.size,
                include_commission=False
            )
        return self.balance + unrealized
    
    def has_open_position(self) -> bool:
        """Check if any position is open."""
        return len(self.positions) > 0
    
    def get_stats(self) -> dict:
        """Get account statistics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'max_drawdown': 0.0,
            }
        
        wins = sum(1 for t in self.trades if t.outcome == 'WIN')
        total = len(self.trades)
        
        return {
            'total_trades': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': wins / total if total > 0 else 0.0,
            'total_pnl': self.realized_pnl,
            'avg_pnl': self.realized_pnl / total if total > 0 else 0.0,
            'max_drawdown': self.max_drawdown,
            'final_balance': self.balance,
        }
