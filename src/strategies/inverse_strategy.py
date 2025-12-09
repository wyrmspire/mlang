import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import PROCESSED_DIR
from src.utils.logging_utils import get_logger

logger = get_logger("inverse_strategy")

def run_inverse_backtest():
    logger.info("Running Inverse (Fade) Strategy Backtest...")
    
    trades_path = PROCESSED_DIR / "labeled_rejections_5m.parquet"
    if not trades_path.exists():
        logger.error("Trades file missing.")
        return
        
    trades = pd.read_parquet(trades_path).sort_values('start_time')
    
    # We use the whole dataset or just test? User said "fade all entries".
    # Let's use the same Test split (last 20%) to be comparable, 
    # OR since we aren't using a model (blind fade), we can run on ALL.
    # Let's run on ALL to see robust stats.
    
    logger.info(f"Total Triggers: {len(trades)}")
    
    initial_balance = 50000.0
    risk_per_trade = 300.0
    balance = initial_balance
    
    wins = 0
    losses = 0
    total_pnl = 0.0
    
    for idx, trade in trades.iterrows():
        # Rejection Strategy:
        # SHORT (Price went Up, we Sell). Target = Down. SL = Up (Extreme).
        # Outcome WIN = Price went Down.
        # Outcome LOSS = Price went Up (Hit SL/Extreme).
        
        # MEANING OF INVERSE (FADE THE ENTRY):
        # We see Price went Up. We see Return to Open.
        # Instead of Selling (Rejection), we BUY (Continuation).
        # We Target the Extreme (High).
        # We Stop Out if Price goes Down (Rejection Win).
        
        # So:
        # Unique mapping:
        # Rejection LOSS -> Price hit Extreme -> Inverse WIN.
        # Rejection WIN -> Price hit Rejection Target -> Inverse LOSS.
        
        original_outcome = trade['outcome']
        if original_outcome not in ['WIN', 'LOSS']: continue
        
        pnl = 0.0
        
        if original_outcome == 'LOSS':
            # Inverse WIN
            # We risk $300.
            # Reward? 
            # In Rejection: Risk = |Entry - Extreme|.
            # In Inverse: Target = Extreme. So Reward = |Entry - Extreme|.
            # So Reward = 1.0 * Risk (Distance to Extreme).
            # Wait, Rejection had SL at Extreme. So Risk distance = Extreme.
            # Inverse Target is Extreme. So Reward distance = Risk distance.
            # So Reward = 1R.
            
            pnl = risk_per_trade * 1.0 # 1:1 Reward
            wins += 1
            
        elif original_outcome == 'WIN':
            # Inverse LOSS
            # We Stop Out.
            # Risk = $300.
            pnl = -risk_per_trade
            losses += 1
            
        balance += pnl
        total_pnl += pnl
        
    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    logger.info("--------------------------------------------------")
    logger.info("INVERSE STRATEGY RESULTS (Fading the Rejection)")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Win Rate: {win_rate*100:.2f}%")
    logger.info(f"Total PnL: ${total_pnl:.2f}")
    logger.info(f"Final Balance: ${balance:.2f}")
    logger.info("--------------------------------------------------")

if __name__ == "__main__":
    run_inverse_backtest()
