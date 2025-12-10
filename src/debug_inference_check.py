
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model_inference import ModelInference
from src.api import get_mock_candles

def test_inference():
    print("Initializing ModelInference...")
    try:
        # Initialize engine (loads model)
        # Ensure we have a model file. The user history implies one exists.
        engine = ModelInference(model_name="CNN_Predictive_5m") 
        if engine.model is None:
            print("ERROR: Model failed to load.")
            return

        print("Generating 1000 mock candles 1m...")
        # We need to simulate the API's cache structure
        mock_data = get_mock_candles(bars=1000, timeframe="1m")
        results = mock_data["data"]
        
        df = pd.DataFrame(results)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        print(f"Data shape: {df.shape}")
        
        print("Running inference loop...")
        signals_found = 0
        max_prob = 0.0
        
        # We need at least 200 bars
        for i in range(200, len(df)):
            # Force analyze even if not on 5m boundary just to see if ANYTHING fires
            # The analyze method does its own resampling, so calling it on every minute is fine for testing.
            
            # Note: analyze() requires 200 bars history.
            
            # We must be careful about the 'time' column which analyze expects
            res = engine.analyze(i, df)
            
            if res:
                print(f"Index {i}: SIGNAL FOUND! Prob: {res['signal']['prob']:.4f}")
                signals_found += 1
            
            # To check 'aliveness', we might want to peek at the raw probability even if it creates no signal
            # But the `analyze` method swallows the probability if < THRESHOLD
            # So let's monkey-patch or just trust that if we see NOTHING, the threshold is the issue.
            
            # Actually, let's temporarily modify the threshold in memory
            
        print(f"Total Signals with default threshold ({engine.THRESHOLD}): {signals_found}")

        # Test with lower threshold
        print("\nRetesting with threshold 0.01...")
        engine.THRESHOLD = 0.01
        low_thresh_signals = 0
        
        for i in range(200, len(df), 5): # Check every 5 mins
            res = engine.analyze(i, df)
            if res:
                 p = res['signal']['prob']
                 max_prob = max(max_prob, p)
                 if p > 0.10: # Only print interesting ones
                     print(f"Index {i}: Prob {p:.4f}")
                 low_thresh_signals += 1

        print(f"Max Probability seen: {max_prob:.4f}")
        print(f"Signals > 0.01: {low_thresh_signals}")

    except Exception as e:
        print(f"EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference()
