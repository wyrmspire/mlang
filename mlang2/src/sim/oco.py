"""
OCO (One-Cancels-Other) Order Logic
Bracket orders with entry, stop loss, and take profit.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from enum import Enum
import pandas as pd

from src.sim.execution import Order, OrderType, OrderStatus, Fill
from src.sim.bar_fill_model import BarFillEngine, BarFillConfig, DEFAULT_FILL_ENGINE
from src.sim.costs import CostModel, DEFAULT_COSTS


class OCOStatus(Enum):
    PENDING_ENTRY = "PENDING_ENTRY"   # Waiting for entry fill
    ACTIVE = "ACTIVE"                  # Entry filled, SL/TP pending
    CLOSED_TP = "CLOSED_TP"           # Closed by take profit
    CLOSED_SL = "CLOSED_SL"           # Closed by stop loss
    CLOSED_TIMEOUT = "CLOSED_TIMEOUT" # Closed by max bars
    CANCELLED = "CANCELLED"           # Entry expired/cancelled


class OCOReference(Enum):
    """What the OCO bracket is referenced from."""
    PRICE = "PRICE"              # Raw price level
    EMA_5M = "EMA_5M"            # 5-minute 200 EMA
    EMA_15M = "EMA_15M"          # 15-minute 200 EMA
    VWAP_SESSION = "VWAP_SESSION"
    VWAP_WEEKLY = "VWAP_WEEKLY"
    LEVEL_1H = "LEVEL_1H"        # Nearest 1h S/R
    LEVEL_4H = "LEVEL_4H"        # Nearest 4h S/R


@dataclass
class OCOConfig:
    """
    OCO bracket configuration.
    """
    direction: str = "LONG"         # 'LONG' or 'SHORT'
    
    # Entry
    entry_type: str = "LIMIT"       # 'MARKET', 'LIMIT'
    entry_offset_atr: float = 0.25  # ATR multiplier for limit offset
    
    # Exit
    stop_atr: float = 1.0           # Stop distance in ATR
    tp_multiple: float = 1.4        # Take profit as multiple of risk
    max_bars: int = 200             # Max bars in trade
    
    # OCO reference (for indicator-based levels)
    reference: OCOReference = OCOReference.PRICE
    reference_offset_atr: float = 0.0
    
    # Unique ID
    name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'direction': self.direction,
            'entry_type': self.entry_type,
            'entry_offset_atr': self.entry_offset_atr,
            'stop_atr': self.stop_atr,
            'tp_multiple': self.tp_multiple,
            'max_bars': self.max_bars,
            'reference': self.reference.value,
            'reference_offset_atr': self.reference_offset_atr,
            'name': self.name,
        }
    
    def to_cli_args(self) -> list:
        return [
            '--direction', self.direction,
            '--entry-type', self.entry_type,
            '--entry-offset', str(self.entry_offset_atr),
            '--stop-atr', str(self.stop_atr),
            '--tp-mult', str(self.tp_multiple),
            '--max-bars', str(self.max_bars),
        ]


@dataclass
class OCOBracket:
    """
    Active OCO bracket tracking.
    """
    config: OCOConfig
    
    # Prices (computed at creation)
    entry_price: float = 0.0
    stop_price: float = 0.0
    tp_price: float = 0.0
    
    # State
    status: OCOStatus = OCOStatus.PENDING_ENTRY
    entry_bar: int = 0
    entry_fill: Optional[Fill] = None
    exit_fill: Optional[Fill] = None
    
    # Reference for logging
    reference_value: float = 0.0   # Value of indicator reference at creation
    atr_at_creation: float = 0.0
    
    # Tracking
    bars_in_trade: int = 0
    mae: float = 0.0   # Max Adverse Excursion
    mfe: float = 0.0   # Max Favorable Excursion


def create_oco_bracket(
    config: OCOConfig,
    base_price: float,
    atr: float,
    reference_value: Optional[float] = None,
    costs: CostModel = None
) -> OCOBracket:
    """
    Create OCO bracket with computed price levels.
    
    Args:
        config: OCO configuration
        base_price: Current price or signal bar close
        atr: Current ATR for offset calculations
        reference_value: Value if using indicator reference
        costs: Cost model for tick rounding
    """
    costs = costs or DEFAULT_COSTS
    
    # Use reference value if provided, else base price
    ref = reference_value if reference_value else base_price
    
    if config.direction == 'LONG':
        # LONG: entry below, stop below entry, TP above entry
        entry_price = costs.round_to_tick(
            ref - config.entry_offset_atr * atr, 'down'
        ) if config.entry_type == 'LIMIT' else base_price
        
        stop_price = costs.round_to_tick(
            entry_price - config.stop_atr * atr, 'down'
        )
        
        risk = entry_price - stop_price
        tp_price = costs.round_to_tick(
            entry_price + risk * config.tp_multiple, 'up'
        )
    else:
        # SHORT: entry above, stop above entry, TP below entry
        entry_price = costs.round_to_tick(
            ref + config.entry_offset_atr * atr, 'up'
        ) if config.entry_type == 'LIMIT' else base_price
        
        stop_price = costs.round_to_tick(
            entry_price + config.stop_atr * atr, 'up'
        )
        
        risk = stop_price - entry_price
        tp_price = costs.round_to_tick(
            entry_price - risk * config.tp_multiple, 'down'
        )
    
    return OCOBracket(
        config=config,
        entry_price=entry_price,
        stop_price=stop_price,
        tp_price=tp_price,
        reference_value=ref,
        atr_at_creation=atr,
    )


def process_oco_bar(
    bracket: OCOBracket,
    bar: pd.Series,
    bar_idx: int,
    fill_engine: BarFillEngine = None
) -> Tuple[OCOBracket, Optional[str]]:
    """
    Process one bar for an OCO bracket.
    
    Returns:
        Updated bracket and event ('ENTRY', 'SL', 'TP', 'TIMEOUT', or None)
    """
    fill_engine = fill_engine or DEFAULT_FILL_ENGINE
    
    if bracket.status == OCOStatus.PENDING_ENTRY:
        # Check for entry fill
        if bracket.config.entry_type == 'MARKET':
            # Market entry fills at open
            fill_price = fill_engine.costs.apply_slippage(
                bar['open'], bracket.config.direction, 'MARKET'
            )
            bracket.entry_fill = Fill(
                order=Order(OrderType.MARKET, bracket.config.direction, None),
                fill_price=fill_price,
                fill_bar=bar_idx
            )
            bracket.entry_bar = bar_idx
            bracket.status = OCOStatus.ACTIVE
            bracket.entry_price = fill_price  # Update actual entry
            return (bracket, 'ENTRY')
        
        else:
            # Limit entry
            fill_price = fill_engine.get_limit_entry_fill_price(
                bracket.entry_price,
                bracket.config.direction,
                bar
            )
            if fill_price is not None:
                bracket.entry_fill = Fill(
                    order=Order(OrderType.LIMIT, bracket.config.direction, bracket.entry_price),
                    fill_price=fill_price,
                    fill_bar=bar_idx
                )
                bracket.entry_bar = bar_idx
                bracket.status = OCOStatus.ACTIVE
                bracket.entry_price = fill_price  # May be better than limit
                return (bracket, 'ENTRY')
    
    elif bracket.status == OCOStatus.ACTIVE:
        bracket.bars_in_trade += 1
        
        # Track MAE/MFE
        if bracket.config.direction == 'LONG':
            adverse = bracket.entry_price - bar['low']
            favorable = bar['high'] - bracket.entry_price
        else:
            adverse = bar['high'] - bracket.entry_price
            favorable = bracket.entry_price - bar['low']
        
        bracket.mae = max(bracket.mae, adverse)
        bracket.mfe = max(bracket.mfe, favorable)
        
        # Check timeout
        if bracket.bars_in_trade >= bracket.config.max_bars:
            bracket.status = OCOStatus.CLOSED_TIMEOUT
            return (bracket, 'TIMEOUT')
        
        # Check SL/TP
        result, fill_price = fill_engine.check_exit(
            bracket.config.direction,
            bracket.stop_price,
            bracket.tp_price,
            bar,
            bracket.entry_bar,
            bar_idx
        )
        
        if result == 'SL':
            bracket.status = OCOStatus.CLOSED_SL
            bracket.exit_fill = Fill(
                order=Order(OrderType.STOP, bracket.config.direction, bracket.stop_price),
                fill_price=fill_price,
                fill_bar=bar_idx
            )
            return (bracket, 'SL')
        
        elif result == 'TP':
            bracket.status = OCOStatus.CLOSED_TP
            bracket.exit_fill = Fill(
                order=Order(OrderType.LIMIT, bracket.config.direction, bracket.tp_price),
                fill_price=fill_price,
                fill_bar=bar_idx
            )
            return (bracket, 'TP')
    
    return (bracket, None)
