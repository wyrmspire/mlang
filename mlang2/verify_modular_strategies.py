"""
Verification script for Modular Strategy Discovery.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[0]))

from src.policy.scanners import get_scanner

def test_modular_discovery():
    print("--- Testing Modular Discovery ---")
    try:
        # Try to get the mid_day_reversal scanner
        scanner = get_scanner("midday_reversal", start_hour=11, end_hour=13)
        print(f"Successfully instantiated: {scanner.scanner_id}")
        
        # Check if always scanner still works
        always = get_scanner("always")
        print(f"Successfully instantiated: {always.scanner_id}")
        
    except Exception as e:
        print(f"Error: {e}")
    print("\n")

if __name__ == "__main__":
    test_modular_discovery()
