"""
Adaptive Blocker
================

Implements adaptive signal blocking based on market regime.

Fix 17: Adaptive blocking for JPY SELL signals
    - JPY SELL has 0% win rate in BULL/RISK_ON markets
    - JPY strengthens (SELL wins) in BEAR/RISK_OFF/CRISIS
    - Block SELL signals in unfavorable regimes

Fix 18: Adaptive blocking for Crude Oil CL=F
    - Crude Oil is a consistent loser in wrong market regimes
    - BUY favorable in: INFLATION, BULL, RISK_ON
    - SELL favorable in: DEFLATION, BEAR
    - Block both in: VOLATILE, CRISIS, SIDEWAYS

Blocking Levels:
    NO_BLOCK: Signal passes through unchanged
    WARNING: Signal passes with warning flag
    REDUCE_POSITION: Signal allowed but position reduced
    BLOCK_SIGNAL: Signal blocked entirely
    FULL_BLOCK: Asset type blocked in current regime

Last Updated: 2025-12-03
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from datetime import datetime

from .market_regime_detector import MarketRegimeDetector, MarketRegime, RegimeDetection


class BlockingLevel(Enum):
    """Signal blocking levels."""
    NO_BLOCK = 'no_block'           # Signal passes unchanged
    WARNING = 'warning'             # Signal passes with warning
    REDUCE_POSITION = 'reduce'      # Position reduced (e.g., 50%)
    BLOCK_SIGNAL = 'block'          # This specific signal blocked
    FULL_BLOCK = 'full_block'       # Asset type blocked entirely


@dataclass
class BlockingResult:
    """Result of adaptive blocking evaluation."""
    ticker: str
    signal_type: str
    blocking_level: BlockingLevel
    position_reduction: float         # 1.0 = no reduction, 0.5 = 50% reduction
    blocked: bool                     # True if signal should not execute
    reason: str
    regime: MarketRegime
    regime_confidence: float
    alternative_action: Optional[str]  # Suggested alternative
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class AdaptiveBlocker:
    """
    Adaptive signal blocker based on market regime.

    IMPORTANT: This is ADAPTIVE blocking, NOT permanent blocking!
    All signals are allowed through with position size adjustments.
    Uses MarketRegimeDetector to determine position reduction:
    - Favorable regime: Full position (1.0x)
    - Neutral regime: Warning + slight reduction (0.8x)
    - Unfavorable regime: Significant reduction (0.3x-0.5x)
    - Very unfavorable: Minimal position (0.1x-0.2x)

    NO signals are fully blocked - just heavily reduced in unfavorable regimes.
    This allows the system to still capture opportunities if regime changes.
    """

    # JPY SELL position multipliers by regime (Fix 17 - ADAPTIVE, not permanent)
    # Historical 0% win rate in BULL/RISK_ON, but we reduce instead of block
    JPY_SELL_RULES = {
        MarketRegime.BEAR: BlockingLevel.NO_BLOCK,           # 1.0x - favorable
        MarketRegime.RISK_OFF: BlockingLevel.NO_BLOCK,       # 1.0x - favorable
        MarketRegime.CRISIS: BlockingLevel.NO_BLOCK,         # 1.0x - JPY safe haven
        MarketRegime.DEFLATION: BlockingLevel.NO_BLOCK,      # 1.0x - favorable
        MarketRegime.INFLATION: BlockingLevel.WARNING,       # 0.8x - neutral
        MarketRegime.SIDEWAYS: BlockingLevel.REDUCE_POSITION,# 0.5x - uncertain
        MarketRegime.VOLATILE: BlockingLevel.REDUCE_POSITION,# 0.5x - risky
        MarketRegime.BULL: BlockingLevel.REDUCE_POSITION,    # 0.3x - unfavorable (was FULL_BLOCK)
        MarketRegime.RISK_ON: BlockingLevel.REDUCE_POSITION, # 0.3x - unfavorable (was FULL_BLOCK)
    }

    # JPY BUY position multipliers by regime
    JPY_BUY_RULES = {
        MarketRegime.BULL: BlockingLevel.NO_BLOCK,           # 1.0x - favorable
        MarketRegime.RISK_ON: BlockingLevel.NO_BLOCK,        # 1.0x - favorable
        MarketRegime.INFLATION: BlockingLevel.NO_BLOCK,      # 1.0x - favorable
        MarketRegime.SIDEWAYS: BlockingLevel.WARNING,        # 0.8x - neutral
        MarketRegime.VOLATILE: BlockingLevel.REDUCE_POSITION,# 0.5x - risky
        MarketRegime.BEAR: BlockingLevel.REDUCE_POSITION,    # 0.5x - unfavorable
        MarketRegime.DEFLATION: BlockingLevel.REDUCE_POSITION,# 0.5x - unfavorable
        MarketRegime.RISK_OFF: BlockingLevel.REDUCE_POSITION,# 0.3x - unfavorable
        MarketRegime.CRISIS: BlockingLevel.REDUCE_POSITION,  # 0.2x - very unfavorable (was FULL_BLOCK)
    }

    # Crude Oil BUY position multipliers by regime (Fix 18 - ADAPTIVE)
    CRUDE_BUY_RULES = {
        MarketRegime.INFLATION: BlockingLevel.NO_BLOCK,      # 1.0x - oil rises with inflation
        MarketRegime.BULL: BlockingLevel.NO_BLOCK,           # 1.0x - favorable
        MarketRegime.RISK_ON: BlockingLevel.NO_BLOCK,        # 1.0x - favorable
        MarketRegime.SIDEWAYS: BlockingLevel.WARNING,        # 0.8x - neutral
        MarketRegime.VOLATILE: BlockingLevel.REDUCE_POSITION,# 0.5x - risky
        MarketRegime.BEAR: BlockingLevel.REDUCE_POSITION,    # 0.4x - unfavorable
        MarketRegime.RISK_OFF: BlockingLevel.REDUCE_POSITION,# 0.4x - unfavorable
        MarketRegime.DEFLATION: BlockingLevel.REDUCE_POSITION,# 0.2x - very unfavorable (was FULL_BLOCK)
        MarketRegime.CRISIS: BlockingLevel.REDUCE_POSITION,  # 0.2x - oil crashes (was FULL_BLOCK)
    }

    # Crude Oil SELL position multipliers by regime (Fix 18 - ADAPTIVE)
    CRUDE_SELL_RULES = {
        MarketRegime.DEFLATION: BlockingLevel.NO_BLOCK,      # 1.0x - favorable
        MarketRegime.BEAR: BlockingLevel.NO_BLOCK,           # 1.0x - favorable
        MarketRegime.CRISIS: BlockingLevel.WARNING,          # 0.8x - can work but volatile
        MarketRegime.RISK_OFF: BlockingLevel.WARNING,        # 0.8x - neutral
        MarketRegime.SIDEWAYS: BlockingLevel.REDUCE_POSITION,# 0.5x - uncertain
        MarketRegime.VOLATILE: BlockingLevel.REDUCE_POSITION,# 0.5x - risky
        MarketRegime.BULL: BlockingLevel.REDUCE_POSITION,    # 0.3x - unfavorable (was BLOCK)
        MarketRegime.RISK_ON: BlockingLevel.REDUCE_POSITION, # 0.2x - very unfavorable (was FULL_BLOCK)
        MarketRegime.INFLATION: BlockingLevel.REDUCE_POSITION,# 0.2x - very unfavorable (was FULL_BLOCK)
    }

    # Position reduction factors by blocking level - NO MORE ZEROS!
    # All positions allowed, just reduced in unfavorable conditions
    POSITION_REDUCTIONS = {
        BlockingLevel.NO_BLOCK: 1.0,       # Full position
        BlockingLevel.WARNING: 0.8,        # Slight caution
        BlockingLevel.REDUCE_POSITION: 0.5,# Significant reduction (default)
        BlockingLevel.BLOCK_SIGNAL: 0.2,   # Heavy reduction (was 0.0)
        BlockingLevel.FULL_BLOCK: 0.1,     # Minimal position (was 0.0)
    }

    # Additional regime-specific fine-tuning for unfavorable conditions
    # Used when REDUCE_POSITION is selected to provide granular control
    UNFAVORABLE_REGIME_REDUCTIONS = {
        # JPY SELL in risk-on environments
        ('jpy_pair', 'SELL', MarketRegime.BULL): 0.3,
        ('jpy_pair', 'SELL', MarketRegime.RISK_ON): 0.3,
        ('jpy_pair', 'SELL', MarketRegime.VOLATILE): 0.4,
        # JPY BUY in safe-haven demand
        ('jpy_pair', 'BUY', MarketRegime.CRISIS): 0.2,
        ('jpy_pair', 'BUY', MarketRegime.RISK_OFF): 0.3,
        # Crude Oil BUY in deflationary/crisis
        ('crude_oil', 'BUY', MarketRegime.DEFLATION): 0.2,
        ('crude_oil', 'BUY', MarketRegime.CRISIS): 0.2,
        ('crude_oil', 'BUY', MarketRegime.BEAR): 0.4,
        # Crude Oil SELL in inflationary/bull
        ('crude_oil', 'SELL', MarketRegime.INFLATION): 0.2,
        ('crude_oil', 'SELL', MarketRegime.RISK_ON): 0.2,
        ('crude_oil', 'SELL', MarketRegime.BULL): 0.3,
    }

    def __init__(self, regime_detector: Optional[MarketRegimeDetector] = None):
        """
        Initialize the adaptive blocker.

        Args:
            regime_detector: Optional pre-configured detector. If None, creates new one.
        """
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self._last_detection: Optional[RegimeDetection] = None

    def evaluate_signal(
        self,
        ticker: str,
        signal_type: str,
        asset_type: str,
        detection: Optional[RegimeDetection] = None,
    ) -> BlockingResult:
        """
        Evaluate a signal for potential blocking.

        Args:
            ticker: Asset ticker (e.g., 'USDJPY=X', 'CL=F')
            signal_type: 'BUY' or 'SELL'
            asset_type: 'jpy_pair', 'crude_oil', 'stock', etc.
            detection: Optional pre-computed regime detection

        Returns:
            BlockingResult with blocking decision
        """
        det = detection or self._last_detection
        if not det:
            # No regime info - be conservative
            return BlockingResult(
                ticker=ticker,
                signal_type=signal_type,
                blocking_level=BlockingLevel.REDUCE_POSITION,
                position_reduction=0.5,
                blocked=False,
                reason="No regime detection available - reducing position by 50%",
                regime=MarketRegime.SIDEWAYS,
                regime_confidence=0.0,
                alternative_action="Wait for regime detection before trading",
            )

        self._last_detection = det
        signal_upper = signal_type.upper()
        asset_lower = asset_type.lower()

        # Route to appropriate evaluation
        if asset_lower == 'jpy_pair':
            return self._evaluate_jpy_signal(ticker, signal_upper, det)
        elif asset_lower == 'crude_oil':
            return self._evaluate_crude_signal(ticker, signal_upper, det)
        else:
            # Other assets - pass through with regime-based position adjustment
            return self._evaluate_general_signal(ticker, signal_upper, asset_lower, det)

    def _evaluate_jpy_signal(
        self,
        ticker: str,
        signal_type: str,
        detection: RegimeDetection,
    ) -> BlockingResult:
        """
        Evaluate JPY pair signal (Fix 17) - ADAPTIVE, never fully blocks.

        Position is adjusted based on regime favorability:
        - Favorable (BEAR/RISK_OFF for SELL): 100% position
        - Neutral: 80% position
        - Unfavorable (BULL/RISK_ON for SELL): 30% position (heavily reduced but NOT blocked)
        """
        regime = detection.primary_regime

        if signal_type == 'SELL':
            rules = self.JPY_SELL_RULES
            base_reason = "JPY SELL"
        else:
            rules = self.JPY_BUY_RULES
            base_reason = "JPY BUY"

        blocking_level = rules.get(regime, BlockingLevel.WARNING)

        # Get base position reduction from blocking level
        position_reduction = self.POSITION_REDUCTIONS[blocking_level]

        # Apply regime-specific fine-tuning for REDUCE_POSITION cases
        regime_key = ('jpy_pair', signal_type, regime)
        if regime_key in self.UNFAVORABLE_REGIME_REDUCTIONS:
            position_reduction = self.UNFAVORABLE_REGIME_REDUCTIONS[regime_key]

        # NEVER block - always allow with position adjustment (Fix 17 is ADAPTIVE)
        blocked = False

        # Build detailed reason based on position reduction
        if position_reduction >= 1.0:
            reason = f"{base_reason} favorable in {regime.value}: full position (100%)"
        elif position_reduction >= 0.8:
            reason = f"{base_reason} cautious in {regime.value}: slight reduction ({position_reduction:.0%})"
        elif position_reduction >= 0.5:
            reason = f"{base_reason} reduced in {regime.value}: moderate reduction ({position_reduction:.0%})"
        else:
            reason = f"{base_reason} heavily reduced in {regime.value}: unfavorable regime ({position_reduction:.0%})"

        # Suggestion for heavily reduced positions
        if position_reduction < 0.5:
            alternative = f"Position heavily reduced to {position_reduction:.0%} - regime may improve"
        else:
            alternative = None

        return BlockingResult(
            ticker=ticker,
            signal_type=signal_type,
            blocking_level=blocking_level,
            position_reduction=position_reduction,
            blocked=blocked,  # NEVER True for adaptive blocking
            reason=reason,
            regime=regime,
            regime_confidence=detection.confidence,
            alternative_action=alternative,
        )

    def _evaluate_crude_signal(
        self,
        ticker: str,
        signal_type: str,
        detection: RegimeDetection,
    ) -> BlockingResult:
        """
        Evaluate Crude Oil signal (Fix 18) - ADAPTIVE, never fully blocks.

        Position is adjusted based on regime favorability:
        - Favorable (INFLATION/BULL for BUY, DEFLATION/BEAR for SELL): 100% position
        - Neutral: 80% position
        - Unfavorable: 20-40% position (heavily reduced but NOT blocked)
        """
        regime = detection.primary_regime

        if signal_type == 'BUY':
            rules = self.CRUDE_BUY_RULES
            base_reason = "Crude Oil BUY"
        else:
            rules = self.CRUDE_SELL_RULES
            base_reason = "Crude Oil SELL"

        blocking_level = rules.get(regime, BlockingLevel.WARNING)

        # Get base position reduction from blocking level
        position_reduction = self.POSITION_REDUCTIONS[blocking_level]

        # Apply regime-specific fine-tuning for REDUCE_POSITION cases
        regime_key = ('crude_oil', signal_type, regime)
        if regime_key in self.UNFAVORABLE_REGIME_REDUCTIONS:
            position_reduction = self.UNFAVORABLE_REGIME_REDUCTIONS[regime_key]

        # NEVER block - always allow with position adjustment (Fix 18 is ADAPTIVE)
        blocked = False

        # Build detailed reason based on position reduction
        if position_reduction >= 1.0:
            reason = f"{base_reason} favorable in {regime.value}: full position (100%)"
        elif position_reduction >= 0.8:
            reason = f"{base_reason} cautious in {regime.value}: slight reduction ({position_reduction:.0%})"
        elif position_reduction >= 0.5:
            reason = f"{base_reason} reduced in {regime.value}: moderate reduction ({position_reduction:.0%})"
        else:
            reason = f"{base_reason} heavily reduced in {regime.value}: unfavorable regime ({position_reduction:.0%})"

        # Suggestion for heavily reduced positions
        if position_reduction < 0.5:
            alternative = f"Position heavily reduced to {position_reduction:.0%} - regime may improve"
        else:
            alternative = None

        return BlockingResult(
            ticker=ticker,
            signal_type=signal_type,
            blocking_level=blocking_level,
            position_reduction=position_reduction,
            blocked=blocked,  # NEVER True for adaptive blocking
            reason=reason,
            regime=regime,
            regime_confidence=detection.confidence,
            alternative_action=alternative,
        )

    def _evaluate_general_signal(
        self,
        ticker: str,
        signal_type: str,
        asset_type: str,
        detection: RegimeDetection,
    ) -> BlockingResult:
        """Evaluate general asset signals with regime-based adjustment."""
        regime = detection.primary_regime

        # General rules - more conservative in volatile/crisis
        if regime == MarketRegime.CRISIS:
            blocking_level = BlockingLevel.REDUCE_POSITION
            position_reduction = 0.3
            reason = f"Crisis regime: reducing {asset_type} position to 30%"
        elif regime == MarketRegime.VOLATILE:
            blocking_level = BlockingLevel.REDUCE_POSITION
            position_reduction = 0.5
            reason = f"Volatile regime: reducing {asset_type} position to 50%"
        elif signal_type == 'SELL' and regime in [MarketRegime.BULL, MarketRegime.RISK_ON]:
            blocking_level = BlockingLevel.WARNING
            position_reduction = 0.7
            reason = f"SELL in {regime.value}: reduced position due to upward trend"
        elif signal_type == 'BUY' and regime in [MarketRegime.BEAR, MarketRegime.RISK_OFF]:
            blocking_level = BlockingLevel.WARNING
            position_reduction = 0.7
            reason = f"BUY in {regime.value}: reduced position due to downward trend"
        else:
            blocking_level = BlockingLevel.NO_BLOCK
            position_reduction = 1.0
            reason = f"Signal allowed in {regime.value} regime"

        return BlockingResult(
            ticker=ticker,
            signal_type=signal_type,
            blocking_level=blocking_level,
            position_reduction=position_reduction,
            blocked=False,
            reason=reason,
            regime=regime,
            regime_confidence=detection.confidence,
            alternative_action=None,
        )

    def update_regime(self, detection: RegimeDetection):
        """Update the stored regime detection."""
        self._last_detection = detection

    def batch_evaluate(
        self,
        signals: List[Dict],
        detection: Optional[RegimeDetection] = None,
    ) -> List[BlockingResult]:
        """
        Evaluate multiple signals at once.

        Args:
            signals: List of dicts with keys: ticker, signal_type, asset_type
            detection: Optional regime detection (uses stored if not provided)

        Returns:
            List of BlockingResults
        """
        results = []
        for sig in signals:
            result = self.evaluate_signal(
                ticker=sig.get('ticker', ''),
                signal_type=sig.get('signal_type', 'BUY'),
                asset_type=sig.get('asset_type', 'stock'),
                detection=detection,
            )
            results.append(result)
        return results

    def get_blocking_summary(
        self,
        results: List[BlockingResult],
    ) -> Dict:
        """Get summary statistics of blocking results."""
        total = len(results)
        if total == 0:
            return {'total': 0, 'blocked': 0, 'reduced': 0, 'passed': 0}

        blocked = sum(1 for r in results if r.blocked)
        reduced = sum(1 for r in results if r.blocking_level == BlockingLevel.REDUCE_POSITION and not r.blocked)
        warned = sum(1 for r in results if r.blocking_level == BlockingLevel.WARNING)
        passed = sum(1 for r in results if r.blocking_level == BlockingLevel.NO_BLOCK)

        return {
            'total': total,
            'blocked': blocked,
            'blocked_pct': blocked / total * 100,
            'reduced': reduced,
            'reduced_pct': reduced / total * 100,
            'warned': warned,
            'warned_pct': warned / total * 100,
            'passed': passed,
            'passed_pct': passed / total * 100,
        }


if __name__ == '__main__':
    # Test the adaptive blocker
    print("=" * 80)
    print("ADAPTIVE BLOCKER - TEST")
    print("=" * 80)

    from .market_regime_detector import create_indicators_from_data

    detector = MarketRegimeDetector()
    blocker = AdaptiveBlocker(detector)

    # Test in different regimes
    regimes_to_test = [
        ("Bull Market", create_indicators_from_data(
            vix=14, spy_return_20d=0.08, spy_return_60d=0.15, spy_above_200ma=True
        )),
        ("Bear Market", create_indicators_from_data(
            vix=30, spy_return_20d=-0.10, spy_return_60d=-0.15, spy_above_200ma=False
        )),
        ("Risk-Off", create_indicators_from_data(
            vix=25, spy_return_20d=-0.03, gold_return_20d=0.04, high_yield_spread=6.0
        )),
        ("Inflation", create_indicators_from_data(
            vix=18, spy_return_20d=0.02, tip_ief_spread=0.04, gold_return_20d=0.03
        )),
    ]

    signals_to_test = [
        {'ticker': 'USDJPY=X', 'signal_type': 'SELL', 'asset_type': 'jpy_pair'},
        {'ticker': 'USDJPY=X', 'signal_type': 'BUY', 'asset_type': 'jpy_pair'},
        {'ticker': 'CL=F', 'signal_type': 'BUY', 'asset_type': 'crude_oil'},
        {'ticker': 'CL=F', 'signal_type': 'SELL', 'asset_type': 'crude_oil'},
    ]

    for regime_name, indicators in regimes_to_test:
        detection = detector.detect_regime(indicators)
        print(f"\n{'='*80}")
        print(f"REGIME: {regime_name} ({detection.primary_regime.value})")
        print(f"{'='*80}")

        for sig in signals_to_test:
            result = blocker.evaluate_signal(
                ticker=sig['ticker'],
                signal_type=sig['signal_type'],
                asset_type=sig['asset_type'],
                detection=detection,
            )
            status = "BLOCKED" if result.blocked else f"{result.position_reduction:.0%} position"
            print(f"\n  {sig['ticker']} {sig['signal_type']}: {status}")
            print(f"    Level: {result.blocking_level.value}")
            print(f"    Reason: {result.reason}")
            if result.alternative_action:
                print(f"    Alternative: {result.alternative_action}")
