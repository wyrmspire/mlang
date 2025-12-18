"""
Tests for Viz Export Pipeline
"""

import sys
import json
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.viz.schema import VizDecision, VizTrade, VizOCO, VizWindow, VizRun
from src.viz.export import Exporter
from src.viz.config import VizConfig


class TestVizSchema(unittest.TestCase):
    """Test viz schema dataclasses."""
    
    def test_viz_decision_to_dict(self):
        """VizDecision should produce valid JSON-serializable dict."""
        decision = VizDecision(
            decision_id="abc123",
            timestamp="2025-03-17T10:30:00",
            bar_idx=100,
            index=0,
            scanner_id="interval_60",
            action="PLACE_ORDER",
            current_price=5000.0,
            atr=10.0,
            cf_outcome="WIN",
            cf_pnl_dollars=250.0,
        )
        
        d = decision.to_dict()
        
        # Should be JSON-serializable
        json_str = json.dumps(d)
        self.assertIsInstance(json_str, str)
        
        # Key fields should be present
        self.assertEqual(d['decision_id'], 'abc123')
        self.assertEqual(d['action'], 'PLACE_ORDER')

    def test_viz_trade_to_dict(self):
        """VizTrade should produce valid JSON-serializable dict."""
        trade = VizTrade(
            trade_id="trade_001",
            decision_id="abc123",
            direction="LONG",
            entry_price=5000.0,
            exit_price=5014.0,
            pnl_dollars=70.0,
            outcome="WIN",
        )
        
        d = trade.to_dict()
        json_str = json.dumps(d)
        self.assertIsInstance(json_str, str)
        self.assertEqual(d['outcome'], 'WIN')


class TestExporter(unittest.TestCase):
    """Test Exporter class."""
    
    def test_exporter_finalize_creates_files(self):
        """Exporter.finalize() should create manifest, run, decisions, trades files."""
        config = VizConfig(include_windows=False)
        exporter = Exporter(config, run_id="test_run")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "viz_output"
            exporter.finalize(out_dir)
            
            self.assertTrue((out_dir / "manifest.json").exists())
            self.assertTrue((out_dir / "run.json").exists())
            self.assertTrue((out_dir / "decisions.jsonl").exists())
            self.assertTrue((out_dir / "trades.jsonl").exists())

    def test_exporter_decision_trade_link(self):
        """Decisions and trades should be linkable via decision_id."""
        from src.datasets.decision_record import DecisionRecord
        from src.datasets.trade_record import TradeRecord
        from src.policy.actions import Action, SkipReason
        import pandas as pd
        
        config = VizConfig(include_windows=False)
        exporter = Exporter(config, run_id="test_link")
        
        # Add a decision
        decision = DecisionRecord(
            timestamp=pd.Timestamp("2025-03-17 10:30:00"),
            bar_idx=100,
            decision_id="link_test_001",
            scanner_id="test",
            action=Action.PLACE_ORDER,
            skip_reason=SkipReason.NOT_SKIPPED,
            current_price=5000.0,
            atr=10.0,
            cf_outcome="WIN",
            cf_pnl_dollars=100.0,
        )
        exporter.on_decision(decision, None)
        
        # Add a trade
        trade = TradeRecord(
            trade_id="trade_link_001",
            decision_id="link_test_001",
            entry_price=5000.0,
            exit_price=5010.0,
            pnl_dollars=50.0,
            outcome="WIN",
        )
        exporter.on_trade_closed(trade)
        
        # Check linkage
        self.assertEqual(len(exporter.decisions), 1)
        self.assertEqual(len(exporter.trades), 1)
        self.assertEqual(exporter.decisions[0].decision_id, exporter.trades[0].decision_id)


class TestVizConfig(unittest.TestCase):
    """Test VizConfig."""
    
    def test_config_defaults(self):
        """VizConfig should have sensible defaults."""
        config = VizConfig()
        self.assertFalse(config.include_full_series)
        self.assertTrue(config.include_windows)
        self.assertEqual(config.output_format, "jsonl")


if __name__ == "__main__":
    unittest.main()
