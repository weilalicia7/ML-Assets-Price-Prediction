"""
Position Adjuster
=================

Fix 19: Dynamic position adjustment based on market regime.

This module adjusts position sizes based on:
1. Market regime (from MarketRegimeDetector)
2. Regime confidence level
3. Asset type characteristics
4. Historical performance in current regime
5. Kelly Criterion optimization

Position Adjustment Factors:
    - Regime multiplier: Based on current market conditions
    - Confidence multiplier: Higher confidence = larger positions
    - Asset multiplier: Asset-specific adjustments
    - Win rate multiplier: Based on historical performance

Author: Claude Code
Last Updated: 2025-12-03
"""

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from enum import Enum

from .market_regime_detector import MarketRegimeDetector, MarketRegime, RegimeDetection


@dataclass
class PositionAdjustment:
    """Result of position size adjustment."""
    ticker: str
    signal_type: str
    asset_type: str
    base_position: float          # Original position size (0-1)
    adjusted_position: float      # Final position size (0-1)
    total_multiplier: float       # Combined multiplier applied
    regime_multiplier: float      # Regime-based adjustment
    confidence_multiplier: float  # Confidence-based adjustment
    asset_multiplier: float       # Asset-specific adjustment
    win_rate_multiplier: float    # Historical win rate adjustment
    regime: MarketRegime
    regime_confidence: float
    adjustments_applied: List[str]  # List of adjustments made
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class PositionAdjuster:
    """
    Adjusts position sizes based on market regime and other factors.

    Key features:
    - Regime-based sizing: Reduce in volatile/crisis, increase in trending
    - Confidence-based: Scale by regime detection confidence
    - Asset-specific: Different multipliers for different asset types
    - Win rate integration: Scale by historical performance
    """

    # Base regime multipliers for position sizing
    REGIME_MULTIPLIERS = {
        MarketRegime.BULL: 1.2,       # Increase positions in bull
        MarketRegime.RISK_ON: 1.15,   # Slightly larger in risk-on
        MarketRegime.INFLATION: 1.0,  # Neutral
        MarketRegime.SIDEWAYS: 0.8,   # Reduce in sideways
        MarketRegime.DEFLATION: 0.75, # Reduce in deflation
        MarketRegime.BEAR: 0.6,       # Smaller in bear (unless shorting)
        MarketRegime.RISK_OFF: 0.5,   # Much smaller in risk-off
        MarketRegime.VOLATILE: 0.4,   # Small in volatile
        MarketRegime.CRISIS: 0.2,     # Minimal in crisis
    }

    # Regime multipliers for SHORT (SELL) positions
    # Inverted from long - shorts do better in bear markets
    REGIME_MULTIPLIERS_SHORT = {
        MarketRegime.BEAR: 1.1,       # Shorts work well in bear
        MarketRegime.RISK_OFF: 1.0,   # Risk-off good for shorts
        MarketRegime.DEFLATION: 0.9,  # Deflation okay for shorts
        MarketRegime.CRISIS: 0.7,     # Crisis - shorts work but volatile
        MarketRegime.VOLATILE: 0.5,   # Risky for shorts too
        MarketRegime.SIDEWAYS: 0.5,   # Choppy for shorts
        MarketRegime.INFLATION: 0.4,  # Bad for shorts
        MarketRegime.RISK_ON: 0.3,    # Shorts get squeezed
        MarketRegime.BULL: 0.2,       # Worst for shorts
    }

    # Asset-specific base multipliers
    ASSET_BASE_MULTIPLIERS = {
        'stock': 1.0,
        'etf': 1.1,              # Slightly higher for diversified ETFs
        'cryptocurrency': 0.6,   # Lower for volatile crypto
        'commodity': 0.7,        # Lower for volatile commodities
        'forex': 0.8,            # Medium for forex
        'jpy_pair': 0.5,         # Lower for JPY pairs (Fix 17)
        'crude_oil': 0.4,        # Lower for crude oil (Fix 18)
    }

    # Win rate to multiplier mapping
    WIN_RATE_MULTIPLIERS = {
        (0.80, 1.00): 1.5,   # 80%+ = 1.5x
        (0.70, 0.80): 1.3,   # 70-80% = 1.3x
        (0.60, 0.70): 1.1,   # 60-70% = 1.1x
        (0.50, 0.60): 1.0,   # 50-60% = baseline
        (0.40, 0.50): 0.7,   # 40-50% = 0.7x
        (0.30, 0.40): 0.5,   # 30-40% = 0.5x
        (0.00, 0.30): 0.2,   # <30% = 0.2x (near-skip)
    }

    # Confidence to multiplier
    CONFIDENCE_MULTIPLIERS = {
        (0.80, 1.00): 1.2,   # High confidence
        (0.60, 0.80): 1.0,   # Medium confidence
        (0.40, 0.60): 0.8,   # Low-medium confidence
        (0.00, 0.40): 0.5,   # Low confidence
    }

    def __init__(
        self,
        regime_detector: Optional[MarketRegimeDetector] = None,
        historical_win_rates: Optional[Dict[str, float]] = None,
        max_position_size: float = 0.15,
        min_position_size: float = 0.01,
    ):
        """
        Initialize the position adjuster.

        Args:
            regime_detector: Optional pre-configured detector
            historical_win_rates: Dict mapping ticker -> win_rate (0-1)
            max_position_size: Maximum position size (default 15%)
            min_position_size: Minimum position size (default 1%)
        """
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self.historical_win_rates = historical_win_rates or {}
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self._last_detection: Optional[RegimeDetection] = None

    def adjust_position(
        self,
        ticker: str,
        signal_type: str,
        asset_type: str,
        base_position: float,
        win_rate: Optional[float] = None,
        detection: Optional[RegimeDetection] = None,
    ) -> PositionAdjustment:
        """
        Adjust position size based on all factors.

        Args:
            ticker: Asset ticker
            signal_type: 'BUY' or 'SELL'
            asset_type: 'stock', 'jpy_pair', 'crude_oil', etc.
            base_position: Starting position size (0-1)
            win_rate: Optional win rate (uses historical if not provided)
            detection: Optional regime detection (uses stored if not provided)

        Returns:
            PositionAdjustment with final position size
        """
        det = detection or self._last_detection
        adjustments_applied = []

        # Default values if no regime detection
        if not det:
            regime = MarketRegime.SIDEWAYS
            regime_confidence = 0.5
            adjustments_applied.append("No regime detection - using defaults")
        else:
            regime = det.primary_regime
            regime_confidence = det.confidence
            self._last_detection = det

        signal_upper = signal_type.upper()
        asset_lower = asset_type.lower()

        # 1. Regime multiplier
        if signal_upper == 'SELL':
            regime_mult = self.REGIME_MULTIPLIERS_SHORT.get(regime, 0.5)
            adjustments_applied.append(f"Regime (SHORT): {regime.value} -> {regime_mult:.2f}x")
        else:
            regime_mult = self.REGIME_MULTIPLIERS.get(regime, 0.8)
            adjustments_applied.append(f"Regime (LONG): {regime.value} -> {regime_mult:.2f}x")

        # 2. Confidence multiplier
        conf_mult = self._get_confidence_multiplier(regime_confidence)
        adjustments_applied.append(f"Confidence: {regime_confidence:.0%} -> {conf_mult:.2f}x")

        # 3. Asset multiplier
        asset_mult = self.ASSET_BASE_MULTIPLIERS.get(asset_lower, 0.8)
        adjustments_applied.append(f"Asset type: {asset_lower} -> {asset_mult:.2f}x")

        # 4. Win rate multiplier
        actual_win_rate = win_rate or self.historical_win_rates.get(ticker, 0.50)
        win_mult = self._get_win_rate_multiplier(actual_win_rate)
        adjustments_applied.append(f"Win rate: {actual_win_rate:.0%} -> {win_mult:.2f}x")

        # Calculate total multiplier
        total_mult = regime_mult * conf_mult * asset_mult * win_mult

        # Apply to base position
        adjusted = base_position * total_mult

        # Clamp to min/max
        adjusted = max(self.min_position_size, min(self.max_position_size, adjusted))

        if adjusted < base_position:
            adjustments_applied.append(f"Final: {base_position:.2%} -> {adjusted:.2%} (reduced)")
        elif adjusted > base_position:
            adjustments_applied.append(f"Final: {base_position:.2%} -> {adjusted:.2%} (increased)")
        else:
            adjustments_applied.append(f"Final: {adjusted:.2%} (unchanged)")

        return PositionAdjustment(
            ticker=ticker,
            signal_type=signal_type,
            asset_type=asset_type,
            base_position=base_position,
            adjusted_position=adjusted,
            total_multiplier=total_mult,
            regime_multiplier=regime_mult,
            confidence_multiplier=conf_mult,
            asset_multiplier=asset_mult,
            win_rate_multiplier=win_mult,
            regime=regime,
            regime_confidence=regime_confidence,
            adjustments_applied=adjustments_applied,
        )

    def _get_confidence_multiplier(self, confidence: float) -> float:
        """Get multiplier based on regime confidence."""
        for (low, high), mult in self.CONFIDENCE_MULTIPLIERS.items():
            if low <= confidence < high:
                return mult
        return 0.8

    def _get_win_rate_multiplier(self, win_rate: float) -> float:
        """Get multiplier based on historical win rate."""
        for (low, high), mult in self.WIN_RATE_MULTIPLIERS.items():
            if low <= win_rate < high:
                return mult
        return 1.0

    def update_regime(self, detection: RegimeDetection):
        """Update stored regime detection."""
        self._last_detection = detection

    def update_win_rates(self, win_rates: Dict[str, float]):
        """Update historical win rates."""
        self.historical_win_rates.update(win_rates)

    def batch_adjust(
        self,
        positions: List[Dict],
        detection: Optional[RegimeDetection] = None,
    ) -> List[PositionAdjustment]:
        """
        Adjust multiple positions at once.

        Args:
            positions: List of dicts with keys: ticker, signal_type, asset_type, base_position
            detection: Optional regime detection

        Returns:
            List of PositionAdjustments
        """
        results = []
        for pos in positions:
            result = self.adjust_position(
                ticker=pos.get('ticker', ''),
                signal_type=pos.get('signal_type', 'BUY'),
                asset_type=pos.get('asset_type', 'stock'),
                base_position=pos.get('base_position', 0.05),
                win_rate=pos.get('win_rate'),
                detection=detection,
            )
            results.append(result)
        return results

    def get_adjustment_summary(
        self,
        results: List[PositionAdjustment],
    ) -> Dict:
        """Get summary statistics of position adjustments."""
        if not results:
            return {'total': 0}

        total_base = sum(r.base_position for r in results)
        total_adjusted = sum(r.adjusted_position for r in results)

        return {
            'total_positions': len(results),
            'total_base_allocation': total_base,
            'total_adjusted_allocation': total_adjusted,
            'average_multiplier': sum(r.total_multiplier for r in results) / len(results),
            'max_multiplier': max(r.total_multiplier for r in results),
            'min_multiplier': min(r.total_multiplier for r in results),
            'net_change_pct': (total_adjusted - total_base) / total_base * 100 if total_base > 0 else 0,
        }

    def calculate_kelly_position(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.25,  # Quarter-Kelly
    ) -> float:
        """
        Calculate Kelly Criterion position size.

        Formula: f* = (p * b - q) / b
        Where:
            p = probability of winning
            q = probability of losing (1 - p)
            b = win/loss ratio (avg_win / avg_loss)

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning return
            avg_loss: Average losing return (absolute value)
            fraction: Fraction of Kelly to use (default 0.25 = quarter-Kelly)

        Returns:
            Recommended position size (0 to max_position_size)
        """
        if win_rate <= 0 or avg_loss <= 0:
            return 0.0

        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss if avg_loss > 0 else 1.0

        kelly = (p * b - q) / b if b > 0 else 0.0

        # Apply fraction (quarter-Kelly is common)
        kelly *= fraction

        # Clamp to valid range
        return max(0.0, min(self.max_position_size, kelly))


# Convenience function
def create_position_adjuster(
    historical_win_rates: Optional[Dict[str, float]] = None,
    max_position: float = 0.15,
    min_position: float = 0.01,
) -> PositionAdjuster:
    """
    Create a configured position adjuster.

    Args:
        historical_win_rates: Dict of ticker -> win_rate
        max_position: Maximum position size
        min_position: Minimum position size

    Returns:
        Configured PositionAdjuster
    """
    return PositionAdjuster(
        historical_win_rates=historical_win_rates,
        max_position_size=max_position,
        min_position_size=min_position,
    )


if __name__ == '__main__':
    # Test the position adjuster
    print("=" * 80)
    print("POSITION ADJUSTER - TEST")
    print("=" * 80)

    from .market_regime_detector import create_indicators_from_data

    detector = MarketRegimeDetector()
    adjuster = PositionAdjuster(
        regime_detector=detector,
        historical_win_rates={
            'AAPL': 0.72,
            'USDJPY=X': 0.35,  # Low win rate for JPY
            'CL=F': 0.40,     # Low win rate for crude
            'BTC-USD': 0.65,
        },
    )

    # Test in different regimes
    regimes_to_test = [
        ("Bull Market", create_indicators_from_data(
            vix=14, spy_return_20d=0.08, spy_above_200ma=True
        )),
        ("Bear Market", create_indicators_from_data(
            vix=30, spy_return_20d=-0.10, spy_above_200ma=False
        )),
        ("Crisis", create_indicators_from_data(
            vix=45, spy_return_20d=-0.15, high_yield_spread=9.0
        )),
    ]

    test_positions = [
        {'ticker': 'AAPL', 'signal_type': 'BUY', 'asset_type': 'stock', 'base_position': 0.10},
        {'ticker': 'USDJPY=X', 'signal_type': 'SELL', 'asset_type': 'jpy_pair', 'base_position': 0.05},
        {'ticker': 'CL=F', 'signal_type': 'BUY', 'asset_type': 'crude_oil', 'base_position': 0.05},
        {'ticker': 'BTC-USD', 'signal_type': 'BUY', 'asset_type': 'cryptocurrency', 'base_position': 0.05},
    ]

    for regime_name, indicators in regimes_to_test:
        detection = detector.detect_regime(indicators)
        print(f"\n{'='*80}")
        print(f"REGIME: {regime_name} ({detection.primary_regime.value}, {detection.confidence:.0%} confidence)")
        print(f"{'='*80}")

        results = adjuster.batch_adjust(test_positions, detection)

        for result in results:
            print(f"\n{result.ticker} {result.signal_type}:")
            print(f"  Base: {result.base_position:.2%} -> Adjusted: {result.adjusted_position:.2%}")
            print(f"  Total Multiplier: {result.total_multiplier:.2f}x")
            print(f"  Breakdown:")
            print(f"    - Regime: {result.regime_multiplier:.2f}x")
            print(f"    - Confidence: {result.confidence_multiplier:.2f}x")
            print(f"    - Asset: {result.asset_multiplier:.2f}x")
            print(f"    - Win Rate: {result.win_rate_multiplier:.2f}x")

        summary = adjuster.get_adjustment_summary(results)
        print(f"\nSummary:")
        print(f"  Total Base Allocation: {summary['total_base_allocation']:.2%}")
        print(f"  Total Adjusted: {summary['total_adjusted_allocation']:.2%}")
        print(f"  Net Change: {summary['net_change_pct']:+.1f}%")
