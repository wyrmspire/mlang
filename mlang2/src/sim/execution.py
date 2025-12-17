"""
Order Execution
Order types and execution logic.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
import pandas as pd


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


@dataclass
class Order:
    """
    Order representation.
    """
    order_type: OrderType
    direction: str          # 'LONG' or 'SHORT'
    price: Optional[float]  # Limit/stop price (None for market)
    size: int = 1
    expiry_bars: int = 15   # Bars until expiry (0 = GTC)
    
    # Tracking
    order_id: str = ""
    created_bar: int = 0
    status: OrderStatus = OrderStatus.PENDING
    
    def is_expired(self, current_bar: int) -> bool:
        """Check if order has expired."""
        if self.expiry_bars == 0:
            return False  # GTC
        return (current_bar - self.created_bar) >= self.expiry_bars


@dataclass
class Fill:
    """
    Execution fill.
    """
    order: Order
    fill_price: float
    fill_bar: int
    fill_time: Optional[pd.Timestamp] = None
    slippage: float = 0.0
    
    @property
    def direction(self) -> str:
        return self.order.direction
    
    @property
    def size(self) -> int:
        return self.order.size


def process_order(
    order: Order,
    bar: pd.Series,
    bar_idx: int,
    costs = None
) -> Optional[Fill]:
    """
    Process a single order against a bar.
    
    Returns Fill if order executes, None otherwise.
    """
    from src.sim.costs import DEFAULT_COSTS
    costs = costs or DEFAULT_COSTS
    
    if order.status != OrderStatus.PENDING:
        return None
    
    # Check expiry
    if order.is_expired(bar_idx):
        order.status = OrderStatus.EXPIRED
        return None
    
    if order.order_type == OrderType.MARKET:
        # Market orders fill at open with slippage
        fill_price = costs.apply_slippage(
            bar['open'],
            order.direction,
            'MARKET'
        )
        order.status = OrderStatus.FILLED
        return Fill(
            order=order,
            fill_price=fill_price,
            fill_bar=bar_idx,
            slippage=abs(fill_price - bar['open'])
        )
    
    elif order.order_type == OrderType.LIMIT:
        # Limit order - check if touched
        if order.direction == 'LONG':
            # Buy limit fills if low <= limit
            if bar['low'] <= order.price:
                # Fill at limit or better (open if gap down)
                fill_price = min(order.price, bar['open']) if bar['open'] <= order.price else order.price
                order.status = OrderStatus.FILLED
                return Fill(
                    order=order,
                    fill_price=fill_price,
                    fill_bar=bar_idx
                )
        else:
            # Sell limit fills if high >= limit
            if bar['high'] >= order.price:
                fill_price = max(order.price, bar['open']) if bar['open'] >= order.price else order.price
                order.status = OrderStatus.FILLED
                return Fill(
                    order=order,
                    fill_price=fill_price,
                    fill_bar=bar_idx
                )
    
    elif order.order_type == OrderType.STOP:
        # Stop order - check if triggered
        if order.direction == 'LONG':
            # Buy stop triggers if high >= stop
            if bar['high'] >= order.price:
                fill_price = max(order.price, bar['open'])
                fill_price = costs.apply_slippage(fill_price, order.direction, 'MARKET')
                order.status = OrderStatus.FILLED
                return Fill(
                    order=order,
                    fill_price=fill_price,
                    fill_bar=bar_idx,
                    slippage=abs(fill_price - order.price)
                )
        else:
            # Sell stop triggers if low <= stop
            if bar['low'] <= order.price:
                fill_price = min(order.price, bar['open'])
                fill_price = costs.apply_slippage(fill_price, order.direction, 'MARKET')
                order.status = OrderStatus.FILLED
                return Fill(
                    order=order,
                    fill_price=fill_price,
                    fill_bar=bar_idx,
                    slippage=abs(fill_price - order.price)
                )
    
    return None


def process_orders(
    orders: List[Order],
    bar: pd.Series,
    bar_idx: int,
    costs = None
) -> List[Fill]:
    """Process multiple orders, return all fills."""
    fills = []
    for order in orders:
        fill = process_order(order, bar, bar_idx, costs)
        if fill:
            fills.append(fill)
    return fills
