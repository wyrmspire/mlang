"""
Report Generation
Generate markdown reports from experiment results.
"""

from pathlib import Path
from typing import List
import pandas as pd

from src.experiments.runner import ExperimentResult
from src.config import RESULTS_DIR


def generate_report(
    results: List[ExperimentResult],
    output_path: Path = None
) -> Path:
    """
    Generate markdown report from experiment results.
    """
    output_path = output_path or RESULTS_DIR / "report.md"
    
    lines = [
        "# Experiment Report",
        "",
        f"Generated: {pd.Timestamp.now()}",
        "",
        f"Total experiments: {len(results)}",
        "",
        "## Summary",
        "",
    ]
    
    # Create summary table
    lines.extend([
        "| Name | Records | WIN | LOSS | Best Val Loss | Best Epoch |",
        "|------|---------|-----|------|---------------|------------|",
    ])
    
    for r in results:
        val_loss = f"{r.train_result.best_val_loss:.4f}" if r.train_result else "N/A"
        epoch = str(r.train_result.best_epoch) if r.train_result else "N/A"
        
        lines.append(
            f"| {r.config.name} | {r.total_records} | {r.win_records} | "
            f"{r.loss_records} | {val_loss} | {epoch} |"
        )
    
    lines.extend(["", "## Configuration Details", ""])
    
    for r in results:
        lines.extend([
            f"### {r.config.name}",
            "",
            f"- Fingerprint: `{r.fingerprint}`",
            f"- Scanner: {r.config.scanner_id}",
            f"- Direction: {r.config.oco_config.direction}",
            f"- TP Multiple: {r.config.oco_config.tp_multiple}",
            f"- Stop ATR: {r.config.oco_config.stop_atr}",
            f"- Records: {r.total_records} ({r.win_records}W / {r.loss_records}L)",
            "",
        ])
    
    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Report saved to {output_path}")
    return output_path
