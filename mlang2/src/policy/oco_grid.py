"""
OCO Grid
Standard variations of OCO configurations for multi-OCO testing.
Now includes stop type variations.
"""

from src.sim.oco import OCOConfig
from src.sim.stop_calculator import StopType, StopConfig


# =============================================================================
# OCO Grid with SMART STOPS (based on actual levels, not just ATR offset)
# =============================================================================

# For Opening Range strategy: use OR levels as stops
OR_OCO_GRID = [
    # LONG: stop below OR low
    {
        'name': 'LONG_OR_1R',
        'direction': 'LONG',
        'stop_type': StopType.RANGE_LOW,
        'tp_multiple': 1.0,
        'atr_padding': 0.25,
    },
    {
        'name': 'LONG_OR_1.5R',
        'direction': 'LONG',
        'stop_type': StopType.RANGE_LOW,
        'tp_multiple': 1.5,
        'atr_padding': 0.25,
    },
    {
        'name': 'LONG_OR_2R',
        'direction': 'LONG',
        'stop_type': StopType.RANGE_LOW,
        'tp_multiple': 2.0,
        'atr_padding': 0.25,
    },
    
    # SHORT: stop above OR high
    {
        'name': 'SHORT_OR_1R',
        'direction': 'SHORT',
        'stop_type': StopType.RANGE_HIGH,
        'tp_multiple': 1.0,
        'atr_padding': 0.25,
    },
    {
        'name': 'SHORT_OR_1.5R',
        'direction': 'SHORT',
        'stop_type': StopType.RANGE_HIGH,
        'tp_multiple': 1.5,
        'atr_padding': 0.25,
    },
    {
        'name': 'SHORT_OR_2R',
        'direction': 'SHORT',
        'stop_type': StopType.RANGE_HIGH,
        'tp_multiple': 2.0,
        'atr_padding': 0.25,
    },
]


# Generic candle-based stops (for non-OR strategies)
CANDLE_OCO_GRID = [
    # LONG: stop below previous 5m candle low
    {
        'name': 'LONG_5M_LOW_1R',
        'direction': 'LONG',
        'stop_type': StopType.CANDLE_LOW,
        'timeframe': '5m',
        'tp_multiple': 1.0,
        'atr_padding': 0.25,
    },
    {
        'name': 'LONG_5M_LOW_2R',
        'direction': 'LONG',
        'stop_type': StopType.CANDLE_LOW,
        'timeframe': '5m',
        'tp_multiple': 2.0,
        'atr_padding': 0.25,
    },
    {
        'name': 'LONG_15M_LOW_1R',
        'direction': 'LONG',
        'stop_type': StopType.CANDLE_LOW,
        'timeframe': '15m',
        'tp_multiple': 1.0,
        'atr_padding': 0.25,
    },
    
    # SHORT: stop above previous 5m candle high
    {
        'name': 'SHORT_5M_HIGH_1R',
        'direction': 'SHORT',
        'stop_type': StopType.CANDLE_HIGH,
        'timeframe': '5m',
        'tp_multiple': 1.0,
        'atr_padding': 0.25,
    },
    {
        'name': 'SHORT_5M_HIGH_2R',
        'direction': 'SHORT',
        'stop_type': StopType.CANDLE_HIGH,
        'timeframe': '5m',
        'tp_multiple': 2.0,
        'atr_padding': 0.25,
    },
]


# =============================================================================
# LEGACY: Simple ATR-based OCO (for backwards compatibility)
# =============================================================================

OCO_GRID_10 = [
    # LONG variants
    OCOConfig(direction="LONG", stop_atr=1.0, tp_multiple=1.0, name="LONG_1ATR_1R"),
    OCOConfig(direction="LONG", stop_atr=1.0, tp_multiple=1.5, name="LONG_1ATR_1.5R"),
    OCOConfig(direction="LONG", stop_atr=1.0, tp_multiple=2.0, name="LONG_1ATR_2R"),
    OCOConfig(direction="LONG", stop_atr=0.5, tp_multiple=1.0, name="LONG_0.5ATR_1R"),
    OCOConfig(direction="LONG", stop_atr=1.5, tp_multiple=2.0, name="LONG_1.5ATR_2R"),
    
    # SHORT variants
    OCOConfig(direction="SHORT", stop_atr=1.0, tp_multiple=1.0, name="SHORT_1ATR_1R"),
    OCOConfig(direction="SHORT", stop_atr=1.0, tp_multiple=1.5, name="SHORT_1ATR_1.5R"),
    OCOConfig(direction="SHORT", stop_atr=1.0, tp_multiple=2.0, name="SHORT_1ATR_2R"),
    OCOConfig(direction="SHORT", stop_atr=0.5, tp_multiple=1.0, name="SHORT_0.5ATR_1R"),
    OCOConfig(direction="SHORT", stop_atr=1.5, tp_multiple=2.0, name="SHORT_1.5ATR_2R"),
]


def get_directional_oco_grid(direction: str):
    """Get legacy OCO grid filtered by direction."""
    return [oco for oco in OCO_GRID_10 if oco.direction == direction]


def get_or_oco_grid(direction: str):
    """Get Opening Range OCO grid filtered by direction."""
    return [cfg for cfg in OR_OCO_GRID if cfg['direction'] == direction]


def get_candle_oco_grid(direction: str):
    """Get candle-based OCO grid filtered by direction."""
    return [cfg for cfg in CANDLE_OCO_GRID if cfg['direction'] == direction]
