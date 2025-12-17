"""
Cost Model
Fees, slippage, and tick rounding.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.config import TICK_SIZE, POINT_VALUE, COMMISSION_PER_SIDE, DEFAULT_SLIPPAGE_TICKS


@dataclass
class CostModel:
    """
    Trading cost model for realistic simulation.
    """
    commission_per_side: float = COMMISSION_PER_SIDE  # Per contract per side
    slippage_ticks: float = DEFAULT_SLIPPAGE_TICKS    # Average slippage
    tick_size: float = TICK_SIZE                       # MES = 0.25
    point_value: float = POINT_VALUE                   # MES = $5
    
    def round_to_tick(self, price: float, direction: str = 'nearest') -> float:
        """
        Round price to valid tick.
        
        Args:
            price: Raw price
            direction: 'nearest', 'up', or 'down'
        """
        if direction == 'up':
            return np.ceil(price / self.tick_size) * self.tick_size
        elif direction == 'down':
            return np.floor(price / self.tick_size) * self.tick_size
        else:
            return round(price / self.tick_size) * self.tick_size
    
    def apply_slippage(
        self,
        price: float,
        direction: str,
        order_type: str = 'MARKET'
    ) -> float:
        """
        Apply slippage to fill price.
        
        Slippage is adverse: 
        - BUY market fills ABOVE quoted price
        - SELL market fills BELOW quoted price
        
        Limit orders have no slippage (fill at limit or better).
        """
        if order_type == 'LIMIT':
            return price
        
        slippage_points = self.slippage_ticks * self.tick_size
        
        if direction == 'LONG':
            # Buying - slip up
            return self.round_to_tick(price + slippage_points, 'up')
        else:
            # Selling - slip down
            return self.round_to_tick(price - slippage_points, 'down')
    
    def calculate_commission(self, contracts: int, round_trip: bool = True) -> float:
        """Calculate commission in dollars."""
        sides = 2 if round_trip else 1
        return contracts * self.commission_per_side * sides
    
    def calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        direction: str,
        contracts: int,
        include_commission: bool = True
    ) -> float:
        """
        Calculate trade PnL in dollars.
        
        Args:
            entry_price: Entry fill price
            exit_price: Exit fill price
            direction: 'LONG' or 'SHORT'
            contracts: Number of contracts
            include_commission: Whether to subtract commission
        """
        if direction == 'LONG':
            points = exit_price - entry_price
        else:
            points = entry_price - exit_price
        
        gross_pnl = points * self.point_value * contracts
        
        if include_commission:
            commission = self.calculate_commission(contracts, round_trip=True)
            return gross_pnl - commission
        
        return gross_pnl
    
    def calculate_risk(
        self,
        entry_price: float,
        stop_price: float,
        contracts: int
    ) -> float:
        """Calculate risk in dollars (not including commission)."""
        points = abs(entry_price - stop_price)
        return points * self.point_value * contracts


# Default cost model
DEFAULT_COSTS = CostModel()
