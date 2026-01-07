"""
Performance Integration Module
==============================

Integrates PerformanceTracker with existing risk management components.

Classes:
    - EnhancedPositionAdjuster: PositionAdjuster with performance-based sizing
    - RiskManagerWithPerformance: Complete risk manager with performance tracking

This module bridges:
    - PerformanceTracker (this module)
    - PositionAdjuster (src/risk_management)
    - MarketRegimeDetector (src/risk_management)
    - AdaptiveBlocker (src/risk_management)

Last Updated: 2025-12-03
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from .performance_tracker import (
    PerformanceTracker,
    TradeRecord,
    TradeResult,
    create_performance_tracker
)

# Import risk management components
try:
    from ..risk_management.position_adjuster import PositionAdjuster, PositionAdjustment
    from ..risk_management.market_regime_detector import (
        MarketRegimeDetector,
        MarketRegime,
        RegimeDetection
    )
    from ..risk_management.adaptive_blocker import AdaptiveBlocker, BlockingResult
except ImportError:
    # Fallback - allow standalone testing
    PositionAdjuster = None
    PositionAdjustment = None
    MarketRegimeDetector = None
    MarketRegime = None
    RegimeDetection = None
    AdaptiveBlocker = None
    BlockingResult = None


@dataclass
class EnhancedPositionResult:
    """Result from EnhancedPositionAdjuster with performance metrics."""
    ticker: str
    signal_type: str
    asset_type: str
    base_position: float
    adjusted_position: float
    final_position: float  # After performance adjustment
    regime_multiplier: float
    performance_multiplier: float
    total_multiplier: float
    regime: str
    historical_win_rate: float
    historical_profit_factor: float
    adjustments_applied: List[str]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EnhancedPositionAdjuster:
    """
    Position adjuster that incorporates historical performance metrics.

    Combines:
    - PositionAdjuster: Regime-based position sizing
    - PerformanceTracker: Historical performance analysis

    Position sizing considers:
    1. Market regime (from PositionAdjuster)
    2. Historical win rate for asset/regime/direction
    3. Profit factor history
    4. Kelly Criterion optimization
    """

    def __init__(
        self,
        position_adjuster: Optional[PositionAdjuster] = None,
        performance_tracker: Optional[PerformanceTracker] = None,
        performance_lookback_days: int = 30,
        min_trades_for_adjustment: int = 5,
        max_position_size: float = 0.15,
        min_position_size: float = 0.01
    ):
        """
        Initialize enhanced position adjuster.

        Args:
            position_adjuster: Existing PositionAdjuster instance
            performance_tracker: Existing PerformanceTracker instance
            performance_lookback_days: Days to look back for performance metrics
            min_trades_for_adjustment: Minimum trades needed to apply performance adjustment
            max_position_size: Maximum allowed position size
            min_position_size: Minimum allowed position size
        """
        if PositionAdjuster is not None:
            self.position_adjuster = position_adjuster or PositionAdjuster()
        else:
            self.position_adjuster = None

        self.performance_tracker = performance_tracker or create_performance_tracker()
        self.performance_lookback_days = performance_lookback_days
        self.min_trades_for_adjustment = min_trades_for_adjustment
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size

    def adjust_position_with_performance(
        self,
        ticker: str,
        signal_type: str,
        asset_type: str,
        base_position: float,
        regime: str,
        detection: Optional[Any] = None
    ) -> EnhancedPositionResult:
        """
        Adjust position size using both regime and performance data.

        Args:
            ticker: Asset ticker
            signal_type: 'BUY' or 'SELL'
            asset_type: Asset type (stock, jpy_pair, crude_oil, etc.)
            base_position: Starting position size (0-1)
            regime: Current market regime
            detection: Optional RegimeDetection from MarketRegimeDetector

        Returns:
            EnhancedPositionResult with final position size
        """
        adjustments_applied = []

        # Step 1: Get base regime adjustment from PositionAdjuster
        regime_adjusted = base_position
        regime_multiplier = 1.0

        if self.position_adjuster is not None:
            try:
                pos_adjustment = self.position_adjuster.adjust_position(
                    ticker=ticker,
                    signal_type=signal_type,
                    asset_type=asset_type,
                    base_position=base_position,
                    detection=detection
                )
                regime_adjusted = pos_adjustment.adjusted_position
                regime_multiplier = pos_adjustment.total_multiplier
                adjustments_applied.extend(pos_adjustment.adjustments_applied)
            except Exception as e:
                adjustments_applied.append(f"Regime adjustment error: {e}")
        else:
            adjustments_applied.append("No PositionAdjuster - using base position")

        # Step 2: Get performance metrics for this asset/regime/direction
        perf_metrics = self.performance_tracker.get_performance_metrics(
            asset=ticker,
            regime=regime,
            direction=signal_type,
            lookback_days=self.performance_lookback_days
        )

        historical_win_rate = perf_metrics.get('win_rate', 0.5)
        historical_profit_factor = perf_metrics.get('profit_factor', 1.0)
        total_trades = perf_metrics.get('total_trades', 0)
        kelly_fraction = perf_metrics.get('kelly_fraction', 0)

        # Step 3: Calculate performance-based multiplier
        if total_trades >= self.min_trades_for_adjustment:
            # Use adaptive multiplier from performance tracker
            performance_multiplier = self.performance_tracker.get_adaptive_multiplier(
                asset=ticker,
                regime=regime,
                direction=signal_type,
                base_multiplier=1.0,
                lookback_days=self.performance_lookback_days
            )
            adjustments_applied.append(
                f"Performance: WR={historical_win_rate:.0%}, "
                f"PF={historical_profit_factor:.2f} -> {performance_multiplier:.2f}x"
            )
        else:
            # Not enough trades - use neutral multiplier
            performance_multiplier = 1.0
            adjustments_applied.append(
                f"Insufficient trades ({total_trades}/{self.min_trades_for_adjustment}) - neutral performance multiplier"
            )

        # Step 4: Apply performance multiplier to regime-adjusted position
        final_position = regime_adjusted * performance_multiplier

        # Step 5: Apply Kelly Criterion constraint if available
        if kelly_fraction > 0 and total_trades >= self.min_trades_for_adjustment:
            # Don't exceed Kelly optimal sizing (using quarter-Kelly for safety)
            quarter_kelly = kelly_fraction * 0.25
            if final_position > quarter_kelly:
                final_position = quarter_kelly
                adjustments_applied.append(
                    f"Kelly constraint: capped at {quarter_kelly:.2%} (quarter-Kelly)"
                )

        # Step 6: Clamp to min/max
        final_position = max(self.min_position_size, min(self.max_position_size, final_position))

        total_multiplier = regime_multiplier * performance_multiplier

        if final_position < base_position:
            adjustments_applied.append(f"Final: {base_position:.2%} -> {final_position:.2%} (reduced)")
        elif final_position > base_position:
            adjustments_applied.append(f"Final: {base_position:.2%} -> {final_position:.2%} (increased)")
        else:
            adjustments_applied.append(f"Final: {final_position:.2%} (unchanged)")

        return EnhancedPositionResult(
            ticker=ticker,
            signal_type=signal_type,
            asset_type=asset_type,
            base_position=base_position,
            adjusted_position=regime_adjusted,
            final_position=final_position,
            regime_multiplier=regime_multiplier,
            performance_multiplier=performance_multiplier,
            total_multiplier=total_multiplier,
            regime=regime,
            historical_win_rate=historical_win_rate,
            historical_profit_factor=historical_profit_factor,
            adjustments_applied=adjustments_applied
        )

    def record_trade_outcome(
        self,
        ticker: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        regime: str,
        asset_type: str = "stock",
        position_size: float = 1.0
    ) -> str:
        """
        Record a completed trade for performance tracking.

        Args:
            ticker: Asset ticker
            direction: 'BUY' or 'SELL'
            entry_price: Trade entry price
            exit_price: Trade exit price
            regime: Market regime at entry
            asset_type: Type of asset
            position_size: Position size used

        Returns:
            trade_id: Unique trade identifier
        """
        record = TradeRecord(
            trade_id="",  # Will be auto-generated
            timestamp=datetime.now(),
            asset=ticker,
            direction=direction.upper(),
            entry_price=entry_price,
            exit_price=exit_price,
            regime=regime,
            asset_type=asset_type,
            position_size=position_size
        )

        self.performance_tracker.update_performance(record)
        return record.trade_id


class RiskManagerWithPerformance:
    """
    Complete risk manager integrating all components.

    Combines:
    - MarketRegimeDetector: Detect current market regime
    - AdaptiveBlocker: Block risky signals
    - EnhancedPositionAdjuster: Size positions based on regime + performance
    - PerformanceTracker: Track historical performance

    This is the main entry point for the risk management system.
    """

    def __init__(
        self,
        regime_detector: Optional[Any] = None,
        adaptive_blocker: Optional[Any] = None,
        performance_tracker: Optional[PerformanceTracker] = None,
        storage_path: Optional[str] = None,
        max_position_size: float = 0.15,
        min_position_size: float = 0.01
    ):
        """
        Initialize complete risk manager.

        Args:
            regime_detector: MarketRegimeDetector instance
            adaptive_blocker: AdaptiveBlocker instance
            performance_tracker: PerformanceTracker instance
            storage_path: Path for performance data persistence
            max_position_size: Maximum allowed position size
            min_position_size: Minimum allowed position size
        """
        # Initialize components
        if MarketRegimeDetector is not None:
            self.regime_detector = regime_detector or MarketRegimeDetector()
        else:
            self.regime_detector = None

        if AdaptiveBlocker is not None:
            self.adaptive_blocker = adaptive_blocker or AdaptiveBlocker()
        else:
            self.adaptive_blocker = None

        self.performance_tracker = performance_tracker or create_performance_tracker(
            storage_path=storage_path
        )

        # Create enhanced position adjuster
        position_adjuster = None
        if PositionAdjuster is not None:
            position_adjuster = PositionAdjuster(
                regime_detector=self.regime_detector,
                max_position_size=max_position_size,
                min_position_size=min_position_size
            )

        self.enhanced_adjuster = EnhancedPositionAdjuster(
            position_adjuster=position_adjuster,
            performance_tracker=self.performance_tracker,
            max_position_size=max_position_size,
            min_position_size=min_position_size
        )

        self._current_detection: Optional[Any] = None

    def update_regime(self, market_indicators: Dict[str, float]) -> Optional[Any]:
        """
        Update market regime detection.

        Args:
            market_indicators: Dict with VIX, SPY_return_20d, etc.

        Returns:
            RegimeDetection with current regime
        """
        if self.regime_detector is None:
            return None

        try:
            self._current_detection = self.regime_detector.detect_regime(market_indicators)
            return self._current_detection
        except Exception as e:
            print(f"Error detecting regime: {e}")
            return None

    def evaluate_signal(
        self,
        ticker: str,
        signal_type: str,
        asset_type: str,
        base_position: float = 0.05,
        signal_confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a trading signal through the complete risk management pipeline.

        Args:
            ticker: Asset ticker
            signal_type: 'BUY' or 'SELL'
            asset_type: Asset type
            base_position: Base position size
            signal_confidence: Signal confidence score

        Returns:
            Dict with:
                - blocked: Whether signal was blocked
                - blocking_reason: Reason if blocked
                - final_position: Adjusted position size
                - regime: Current market regime
                - adjustments: List of adjustments applied
                - performance_metrics: Historical performance for this signal type
        """
        result = {
            'ticker': ticker,
            'signal_type': signal_type,
            'asset_type': asset_type,
            'blocked': False,
            'blocking_reason': None,
            'final_position': base_position,
            'regime': None,
            'adjustments': [],
            'performance_metrics': {}
        }

        # Step 1: Get current regime
        regime_str = "SIDEWAYS"  # Default
        if self._current_detection is not None:
            try:
                regime_str = self._current_detection.primary_regime.value
                result['regime'] = regime_str
            except Exception:
                pass

        # Step 2: Check for blocking (Fixes 17 & 18)
        if self.adaptive_blocker is not None:
            try:
                blocking_result = self.adaptive_blocker.evaluate_signal(
                    ticker=ticker,
                    signal_type=signal_type,
                    asset_type=asset_type,
                    detection=self._current_detection
                )
                if blocking_result.blocked:
                    result['blocked'] = True
                    result['blocking_reason'] = blocking_result.reason
                    result['final_position'] = 0.0
                    return result
            except Exception as e:
                result['adjustments'].append(f"Blocking evaluation error: {e}")

        # Step 3: Adjust position with performance data
        try:
            enhanced_result = self.enhanced_adjuster.adjust_position_with_performance(
                ticker=ticker,
                signal_type=signal_type,
                asset_type=asset_type,
                base_position=base_position,
                regime=regime_str,
                detection=self._current_detection
            )

            result['final_position'] = enhanced_result.final_position
            result['adjustments'] = enhanced_result.adjustments_applied
            result['performance_metrics'] = {
                'win_rate': enhanced_result.historical_win_rate,
                'profit_factor': enhanced_result.historical_profit_factor
            }
        except Exception as e:
            result['adjustments'].append(f"Position adjustment error: {e}")

        return result

    def record_trade(
        self,
        ticker: str,
        direction: str,
        entry_price: float,
        exit_price: Optional[float] = None,
        asset_type: str = "stock",
        position_size: float = 1.0,
        blocked: bool = False,
        blocking_reason: Optional[str] = None
    ) -> str:
        """
        Record a trade for performance tracking.

        Args:
            ticker: Asset ticker
            direction: 'BUY' or 'SELL'
            entry_price: Entry price
            exit_price: Exit price (None if trade still open)
            asset_type: Type of asset
            position_size: Position size
            blocked: Whether trade was blocked
            blocking_reason: Reason if blocked

        Returns:
            trade_id: Unique trade identifier
        """
        # Get current regime
        regime_str = "SIDEWAYS"
        regime_confidence = 0.5
        if self._current_detection is not None:
            try:
                regime_str = self._current_detection.primary_regime.value
                regime_confidence = self._current_detection.confidence
            except Exception:
                pass

        record = TradeRecord(
            trade_id="",  # Will be auto-generated
            timestamp=datetime.now(),
            asset=ticker,
            direction=direction.upper(),
            entry_price=entry_price,
            exit_price=exit_price,
            regime=regime_str,
            asset_type=asset_type,
            position_size=position_size,
            regime_confidence=regime_confidence,
            blocked=blocked,
            blocking_reason=blocking_reason
        )

        self.performance_tracker.update_performance(record)
        return record.trade_id

    def close_trade(
        self,
        trade_id: str,
        exit_price: float
    ) -> bool:
        """
        Close an open trade with exit price.

        Args:
            trade_id: Trade ID from record_trade
            exit_price: Exit price

        Returns:
            success: Whether the trade was found and closed
        """
        # Get current regime for exit
        exit_regime = None
        if self._current_detection is not None:
            try:
                exit_regime = self._current_detection.primary_regime.value
            except Exception:
                pass

        return self.performance_tracker.update_trade_exit(
            trade_id=trade_id,
            exit_price=exit_price,
            exit_regime=exit_regime
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'overall': self.performance_tracker.get_performance_metrics(),
            'by_regime': self.performance_tracker.get_regime_performance(),
            'by_direction': self.performance_tracker.get_direction_performance(),
            'by_asset_type': self.performance_tracker.get_asset_type_performance(),
            'by_asset': self.performance_tracker.get_asset_performance()
        }

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive performance report."""
        return self.performance_tracker.generate_report(output_file=output_file)

    def save_performance(self, filepath: Optional[str] = None) -> bool:
        """Save performance data to disk."""
        return self.performance_tracker.save_performance_data(filepath)

    def load_performance(self, filepath: Optional[str] = None) -> bool:
        """Load performance data from disk."""
        return self.performance_tracker.load_performance_data(filepath)


# ==================== Factory Function ====================

def create_risk_manager_with_performance(
    storage_path: Optional[str] = None,
    max_position_size: float = 0.15,
    min_position_size: float = 0.01
) -> RiskManagerWithPerformance:
    """
    Create a fully configured RiskManagerWithPerformance.

    Args:
        storage_path: Path for performance data persistence
        max_position_size: Maximum allowed position size
        min_position_size: Minimum allowed position size

    Returns:
        Configured RiskManagerWithPerformance instance
    """
    return RiskManagerWithPerformance(
        storage_path=storage_path,
        max_position_size=max_position_size,
        min_position_size=min_position_size
    )


# ==================== Test Code ====================

if __name__ == '__main__':
    print("=" * 80)
    print("RISK MANAGER WITH PERFORMANCE - TEST")
    print("=" * 80)

    # Create risk manager
    risk_manager = create_risk_manager_with_performance()

    # Simulate some trades
    print("\n--- Recording Sample Trades ---")

    sample_trades = [
        ("AAPL", "BUY", 150.0, 155.0, "stock"),    # Win
        ("AAPL", "BUY", 155.0, 153.0, "stock"),    # Loss
        ("AAPL", "BUY", 153.0, 160.0, "stock"),    # Win
        ("USDJPY=X", "SELL", 150.0, 148.0, "jpy_pair"),  # Win
        ("USDJPY=X", "SELL", 148.0, 149.0, "jpy_pair"),  # Loss
        ("CL=F", "BUY", 75.0, 72.0, "crude_oil"),   # Loss
        ("BTC-USD", "BUY", 50000.0, 55000.0, "cryptocurrency"),  # Win
    ]

    for ticker, direction, entry, exit_price, asset_type in sample_trades:
        trade_id = risk_manager.record_trade(
            ticker=ticker,
            direction=direction,
            entry_price=entry,
            exit_price=exit_price,
            asset_type=asset_type
        )
        print(f"  Recorded: {ticker} {direction} {entry} -> {exit_price}")

    # Evaluate new signals
    print("\n--- Evaluating New Signals ---")

    test_signals = [
        ("AAPL", "BUY", "stock", 0.05),
        ("USDJPY=X", "SELL", "jpy_pair", 0.05),
        ("CL=F", "BUY", "crude_oil", 0.05),
        ("BTC-USD", "BUY", "cryptocurrency", 0.05),
    ]

    for ticker, signal_type, asset_type, base_pos in test_signals:
        result = risk_manager.evaluate_signal(
            ticker=ticker,
            signal_type=signal_type,
            asset_type=asset_type,
            base_position=base_pos
        )
        print(f"\n  {ticker} {signal_type}:")
        print(f"    Blocked: {result['blocked']}")
        print(f"    Base Position: {base_pos:.2%}")
        print(f"    Final Position: {result['final_position']:.2%}")
        if result['performance_metrics']:
            print(f"    Historical Win Rate: {result['performance_metrics'].get('win_rate', 0):.0%}")

    # Generate report
    print("\n" + "=" * 80)
    print(risk_manager.generate_report())
