"""
Verification script for mlang2 Agent Skills.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[0]))

from src.skills.registry import list_available_skills, registry

def test_discovery():
    print("--- Testing Skill Discovery ---")
    print(list_available_skills())
    print("\n")

def test_data_skill():
    print("--- Testing Data Skill (Summary) ---")
    summary = registry.get_skill("get_data_summary")()
    print(summary)
    print("\n")

if __name__ == "__main__":
    test_discovery()
    test_data_skill()
