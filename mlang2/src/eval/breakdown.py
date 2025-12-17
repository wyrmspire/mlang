"""
Breakdown Analysis
Metrics by setup, time of day, volatility regime.
"""

from typing import List, Dict
from collections import defaultdict

from src.datasets.decision_record import DecisionRecord
from src.eval.metrics import TradeMetrics, compute_from_records


def breakdown_by_scanner(
    records: List[DecisionRecord]
) -> Dict[str, TradeMetrics]:
    """Metrics grouped by scanner/setup type."""
    by_scanner = defaultdict(list)
    
    for r in records:
        by_scanner[r.scanner_id].append(r)
    
    return {k: compute_from_records(v) for k, v in by_scanner.items()}


def breakdown_by_hour(
    records: List[DecisionRecord]
) -> Dict[int, TradeMetrics]:
    """Metrics grouped by hour (NY time)."""
    by_hour = defaultdict(list)
    
    for r in records:
        if r.timestamp:
            hour = r.timestamp.hour
            by_hour[hour].append(r)
    
    return {k: compute_from_records(v) for k, v in sorted(by_hour.items())}


def breakdown_by_day(
    records: List[DecisionRecord]
) -> Dict[int, TradeMetrics]:
    """Metrics grouped by day of week (0=Mon, 4=Fri)."""
    by_day = defaultdict(list)
    
    for r in records:
        if r.timestamp:
            dow = r.timestamp.weekday()
            by_day[dow].append(r)
    
    return {k: compute_from_records(v) for k, v in sorted(by_day.items())}


def breakdown_by_action(
    records: List[DecisionRecord]
) -> Dict[str, dict]:
    """
    Analyze by action taken.
    
    Returns stats for:
    - Trades taken
    - Trades skipped (broken down by skip reason)
    - Skipped good (would have lost)
    - Skipped bad (would have won)
    """
    taken = [r for r in records if r.is_trade()]
    skipped = [r for r in records if r.was_skipped()]
    
    skipped_good = [r for r in skipped if r.is_good_skip()]
    skipped_bad = [r for r in skipped if r.is_bad_skip()]
    
    return {
        'taken': {
            'count': len(taken),
            'metrics': compute_from_records(taken),
        },
        'skipped': {
            'count': len(skipped),
            'good_skips': len(skipped_good),
            'bad_skips': len(skipped_bad),
            'good_skip_rate': len(skipped_good) / len(skipped) if skipped else 0,
        },
        'by_skip_reason': _count_skip_reasons(skipped),
    }


def _count_skip_reasons(records: List[DecisionRecord]) -> Dict[str, int]:
    """Count records by skip reason."""
    counts = defaultdict(int)
    for r in records:
        counts[r.skip_reason.value] += 1
    return dict(counts)


def print_breakdown_summary(
    records: List[DecisionRecord],
    title: str = "Breakdown Summary"
):
    """Print formatted breakdown summary."""
    print(f"\n{'='*50}")
    print(title)
    print('='*50)
    
    # Overall
    overall = compute_from_records(records)
    print(f"\nOverall: {overall.total_trades} records, "
          f"{overall.win_rate:.1%} WR, "
          f"${overall.total_pnl:.2f} PnL")
    
    # By hour
    print("\nBy Hour (NY):")
    by_hour = breakdown_by_hour(records)
    for hour, m in by_hour.items():
        if m.total_trades > 0:
            print(f"  {hour:02d}:00 - {m.total_trades:4d} trades, "
                  f"{m.win_rate:.1%} WR, ${m.total_pnl:7.2f}")
    
    # By action
    print("\nBy Action:")
    by_action = breakdown_by_action(records)
    print(f"  Taken: {by_action['taken']['count']}")
    print(f"  Skipped: {by_action['skipped']['count']} "
          f"({by_action['skipped']['good_skips']} good, "
          f"{by_action['skipped']['bad_skips']} bad)")
