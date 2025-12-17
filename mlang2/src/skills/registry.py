"""
Skill Registry
Discover and access all available agent skills.
"""

from typing import Dict, List, Callable
import inspect

from src.skills.data import ingest_raw_data, get_data_summary
from src.skills.research import run_research_experiment, run_simple_walkforward
from src.skills.model import train_agent_model, evaluate_model_performance

class SkillRegistry:
    """
    Registry of all available skills.
    Agents can query this to understand what they can do.
    """
    
    def __init__(self):
        self._skills: Dict[str, Dict] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        # Data Skills
        self.register("ingest_raw_data", ingest_raw_data, "Ingests raw JSON data into processed Parquet.")
        self.register("get_data_summary", get_data_summary, "Provides a summary of available data.")
        
        # Research Skills
        self.register("run_experiment", run_research_experiment, "Runs a standard research experiment.")
        self.register("run_walkforward", run_simple_walkforward, "Runs a walk-forward research session.")
        
        # Model Skills
        self.register("train_model", train_agent_model, "Trains a neural model on sharded data.")
        self.register("evaluate_model", evaluate_model_performance, "Evaluates a model's performance.")

    def register(self, name: str, func: Callable, description: str):
        self._skills[name] = {
            "func": func,
            "description": description,
            "signature": str(inspect.signature(func))
        }

    def list_skills(self) -> List[Dict]:
        """Returns a list of all skills with descriptions."""
        return [
            {"name": name, "description": data["description"], "signature": data["signature"]}
            for name, data in self._skills.items()
        ]

    def get_skill(self, name: str) -> Callable:
        if name not in self._skills:
            raise ValueError(f"Skill '{name}' not found.")
        return self._skills[name]["func"]

# Global registry instance
registry = SkillRegistry()

def list_available_skills():
    """Helper function for agents to see what they can do."""
    skills = registry.list_skills()
    output = ["Available Agent Skills in mlang2:"]
    for s in skills:
        output.append(f"- {s['name']}{s['signature']}: {s['description']}")
    return "\n".join(output)
