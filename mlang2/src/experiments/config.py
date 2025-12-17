"""
Experiment Configuration
Central config dataclass for experiments.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from src.features.pipeline import FeatureConfig
from src.labels.labeler import LabelConfig
from src.sim.oco import OCOConfig
from src.sim.bar_fill_model import BarFillConfig
from src.sim.costs import CostModel, DEFAULT_COSTS
from src.models.train import TrainConfig
from src.datasets.schema import DatasetSchema


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.
    
    Single source of truth for all parameters.
    """
    # Identification
    name: str = "experiment"
    description: str = ""
    
    # Data range
    start_date: str = ""
    end_date: str = ""
    timeframe: str = "1m"
    
    # Scanner
    scanner_id: str = "always"
    scanner_params: Dict[str, Any] = field(default_factory=dict)
    
    # Features
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    
    # Labels
    label_config: LabelConfig = field(default_factory=LabelConfig)
    oco_config: OCOConfig = field(default_factory=OCOConfig)
    
    # Simulation
    fill_config: BarFillConfig = field(default_factory=BarFillConfig)
    cost_model: CostModel = field(default_factory=lambda: DEFAULT_COSTS)
    
    # Training
    train_config: TrainConfig = field(default_factory=TrainConfig)
    
    # Schema
    schema: DatasetSchema = field(default_factory=DatasetSchema)
    
    # Reproducibility
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'timeframe': self.timeframe,
            'scanner_id': self.scanner_id,
            'scanner_params': self.scanner_params,
            'feature_config': self.feature_config.to_dict(),
            'oco_config': self.oco_config.to_dict(),
            'fill_config': self.fill_config.to_dict(),
            'train_config': self.train_config.to_dict(),
            'schema': self.schema.to_dict(),
            'seed': self.seed,
        }
    
    def to_cli_args(self) -> List[str]:
        """Generate CLI arguments."""
        args = [
            '--name', self.name,
            '--start-date', self.start_date,
            '--end-date', self.end_date,
            '--timeframe', self.timeframe,
            '--scanner', self.scanner_id,
            '--seed', str(self.seed),
        ]
        args.extend(self.oco_config.to_cli_args())
        return args
    
    def save(self, path: Path):
        """Save config to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        
        config = cls(
            name=data.get('name', 'experiment'),
            description=data.get('description', ''),
            start_date=data.get('start_date', ''),
            end_date=data.get('end_date', ''),
            timeframe=data.get('timeframe', '1m'),
            scanner_id=data.get('scanner_id', 'always'),
            seed=data.get('seed', 42),
        )
        
        # Load nested configs if present
        if 'oco_config' in data:
            oco = data['oco_config']
            config.oco_config = OCOConfig(
                direction=oco.get('direction', 'LONG'),
                tp_multiple=oco.get('tp_multiple', 1.4),
                stop_atr=oco.get('stop_atr', 1.0),
            )
        
        return config
