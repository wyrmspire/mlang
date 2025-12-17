"""
Mid-Day Reversal Strategy
Modular scanner that looks for reversals during lunch/mid-day.
"""

from src.policy.scanners import Scanner, ScannerResult
from src.features.state import MarketState
from src.features.pipeline import FeatureBundle

class MidDayReversalScanner(Scanner):
    """
    Scanner that triggers for mid-day reversal setups.
    
    Logic:
    1. Must be in RTH (Regular Trading Hours).
    2. Must be Mid-day (11:00 AM - 1:30 PM NY).
    3. Price must show an extreme or RSI must be at an extreme.
    """
    
    def __init__(
        self, 
        start_hour: int = 11, 
        end_hour: int = 13, 
        rsi_extreme: float = 30.0
    ):
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.rsi_extreme = rsi_extreme

    @property
    def scanner_id(self) -> str:
        return f"midday_reversal_{self.start_hour}_{self.end_hour}"

    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScannerResult:
        t = features.time_features
        if not t or not t.is_rth:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # 1. Check time window
        is_midday = self.start_hour <= t.hour_ny <= self.end_hour
        if not is_midday:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        # 2. Check for reversal signal (Simple RSI extreme for now)
        if features.indicators is None:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        rsi = features.indicators.rsi_5m_14
        oversold = rsi <= self.rsi_extreme
        overbought = rsi >= (100 - self.rsi_extreme)
        
        triggered = oversold or overbought
        
        return ScannerResult(
            scanner_id=self.scanner_id,
            triggered=triggered,
            context={
                'hour': t.hour_ny,
                'rsi': rsi,
                'condition': 'oversold' if oversold else 'overbought' if overbought else 'neutral'
            },
            score=1.0 if triggered else 0.0
        )
