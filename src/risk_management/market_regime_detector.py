"""
Market Regime Detector
======================

Fix 19: Dynamic position adjustment based on market regime.

This module detects the current market regime using multiple indicators:
- VIX levels (volatility)
- SPY trend (bull/bear)
- Treasury yields (risk-on/off)
- Dollar strength (USD index)
- Inflation indicators (TIP vs IEF spread)

Market Regimes:
    BULL: Strong upward trend, low volatility
    BEAR: Strong downward trend, elevated volatility
    RISK_ON: Risk appetite high, money flowing to equities
    RISK_OFF: Flight to safety, money flowing to bonds/gold
    VOLATILE: High volatility, uncertain direction
    SIDEWAYS: Range-bound, low conviction
    INFLATION: Rising inflation expectations
    DEFLATION: Falling inflation expectations
    CRISIS: Market stress indicators elevated

Last Updated: 2025-12-03
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import numpy as np


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL = 'bull'                # Strong upward trend
    BEAR = 'bear'                # Strong downward trend
    RISK_ON = 'risk_on'          # High risk appetite
    RISK_OFF = 'risk_off'        # Flight to safety
    VOLATILE = 'volatile'        # High volatility, uncertain
    SIDEWAYS = 'sideways'        # Range-bound market
    INFLATION = 'inflation'      # Rising inflation expectations
    DEFLATION = 'deflation'      # Falling inflation expectations
    CRISIS = 'crisis'            # Market stress/panic


@dataclass
class RegimeIndicators:
    """Raw market indicators for regime detection."""
    vix: float                           # VIX index level
    spy_return_20d: float                # SPY 20-day return
    spy_return_60d: float                # SPY 60-day return
    spy_above_200ma: bool                # SPY above 200-day MA
    treasury_10y: float                  # 10-year treasury yield
    treasury_2y: float                   # 2-year treasury yield
    usd_index_return: float              # USD index return
    tip_ief_spread: float                # TIP vs IEF spread (inflation)
    gold_return_20d: float               # Gold 20-day return
    high_yield_spread: Optional[float]   # High yield credit spread
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class RegimeDetection:
    """Result of market regime detection."""
    primary_regime: MarketRegime
    secondary_regime: Optional[MarketRegime]
    confidence: float                    # 0-1 confidence in detection
    regime_scores: Dict[MarketRegime, float]
    indicators_used: RegimeIndicators
    regime_history: List[Tuple[datetime, MarketRegime]]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MarketRegimeDetector:
    """
    Detects current market regime using multiple indicators.

    The regime detection drives:
    - Fix 17: JPY SELL blocking (block in BULL/RISK_ON, allow in BEAR/RISK_OFF)
    - Fix 18: Crude Oil blocking (BUY in INFLATION/BULL, SELL in DEFLATION/BEAR)
    - Fix 19: Position sizing adjustments based on regime confidence
    """

    # VIX thresholds
    VIX_LOW = 15.0       # Low volatility
    VIX_ELEVATED = 20.0  # Elevated volatility
    VIX_HIGH = 25.0      # High volatility
    VIX_EXTREME = 35.0   # Extreme/crisis volatility

    # Return thresholds
    BULL_THRESHOLD = 0.05    # 5% return indicates bull
    BEAR_THRESHOLD = -0.05   # -5% return indicates bear

    # Inflation thresholds
    INFLATION_THRESHOLD = 0.02   # TIP/IEF spread
    DEFLATION_THRESHOLD = -0.01

    # Yield curve
    INVERTED_THRESHOLD = -0.25   # 2y-10y spread

    def __init__(
        self,
        lookback_days: int = 60,
        regime_history_size: int = 30,
        smoothing_window: int = 5,
    ):
        """
        Initialize the regime detector.

        Args:
            lookback_days: Days of data for regime detection
            regime_history_size: Number of regime changes to track
            smoothing_window: Window for smoothing regime transitions
        """
        self.lookback_days = lookback_days
        self.regime_history_size = regime_history_size
        self.smoothing_window = smoothing_window
        self.regime_history: List[Tuple[datetime, MarketRegime]] = []
        self._last_detection: Optional[RegimeDetection] = None

    def detect_regime(self, indicators: RegimeIndicators) -> RegimeDetection:
        """
        Detect current market regime from indicators.

        Args:
            indicators: Current market indicators

        Returns:
            RegimeDetection with primary/secondary regimes and confidence
        """
        scores = self._calculate_regime_scores(indicators)

        # Sort regimes by score
        sorted_regimes = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        primary = sorted_regimes[0][0]
        primary_score = sorted_regimes[0][1]

        secondary = sorted_regimes[1][0] if len(sorted_regimes) > 1 else None
        secondary_score = sorted_regimes[1][1] if len(sorted_regimes) > 1 else 0.0

        # Calculate confidence based on score difference
        confidence = self._calculate_confidence(primary_score, secondary_score, scores)

        # Update history
        self._update_history(primary, indicators.timestamp)

        detection = RegimeDetection(
            primary_regime=primary,
            secondary_regime=secondary,
            confidence=confidence,
            regime_scores=scores,
            indicators_used=indicators,
            regime_history=list(self.regime_history),
            timestamp=indicators.timestamp,
        )

        self._last_detection = detection
        return detection

    def _calculate_regime_scores(self, ind: RegimeIndicators) -> Dict[MarketRegime, float]:
        """Calculate scores for each regime based on indicators."""
        scores = {regime: 0.0 for regime in MarketRegime}

        # === BULL regime scoring ===
        if ind.spy_return_20d > self.BULL_THRESHOLD:
            scores[MarketRegime.BULL] += 0.3
        if ind.spy_return_60d > self.BULL_THRESHOLD * 2:
            scores[MarketRegime.BULL] += 0.2
        if ind.spy_above_200ma:
            scores[MarketRegime.BULL] += 0.2
        if ind.vix < self.VIX_LOW:
            scores[MarketRegime.BULL] += 0.15
        if ind.vix < self.VIX_ELEVATED:
            scores[MarketRegime.BULL] += 0.15

        # === BEAR regime scoring ===
        if ind.spy_return_20d < self.BEAR_THRESHOLD:
            scores[MarketRegime.BEAR] += 0.3
        if ind.spy_return_60d < self.BEAR_THRESHOLD * 2:
            scores[MarketRegime.BEAR] += 0.2
        if not ind.spy_above_200ma:
            scores[MarketRegime.BEAR] += 0.2
        if ind.vix > self.VIX_HIGH:
            scores[MarketRegime.BEAR] += 0.15
        if ind.gold_return_20d > 0.03:  # Gold rising = flight to safety
            scores[MarketRegime.BEAR] += 0.15

        # === RISK_ON regime scoring ===
        if ind.vix < self.VIX_ELEVATED and ind.spy_return_20d > 0:
            scores[MarketRegime.RISK_ON] += 0.3
        if ind.high_yield_spread is not None and ind.high_yield_spread < 4.0:
            scores[MarketRegime.RISK_ON] += 0.25
        if ind.spy_above_200ma and ind.spy_return_20d > 0.02:
            scores[MarketRegime.RISK_ON] += 0.25
        if ind.gold_return_20d < 0:  # Gold falling = risk appetite
            scores[MarketRegime.RISK_ON] += 0.2

        # === RISK_OFF regime scoring ===
        if ind.vix > self.VIX_ELEVATED and ind.spy_return_20d < 0:
            scores[MarketRegime.RISK_OFF] += 0.3
        if ind.gold_return_20d > 0.02:  # Gold rising
            scores[MarketRegime.RISK_OFF] += 0.25
        if ind.high_yield_spread is not None and ind.high_yield_spread > 5.0:
            scores[MarketRegime.RISK_OFF] += 0.25
        if ind.usd_index_return > 0.02:  # Dollar strengthening
            scores[MarketRegime.RISK_OFF] += 0.2

        # === VOLATILE regime scoring ===
        if ind.vix > self.VIX_HIGH:
            scores[MarketRegime.VOLATILE] += 0.4
        if ind.vix > self.VIX_EXTREME:
            scores[MarketRegime.VOLATILE] += 0.3
        # Mixed signals
        if (ind.spy_return_20d > 0 and ind.spy_return_60d < 0) or \
           (ind.spy_return_20d < 0 and ind.spy_return_60d > 0):
            scores[MarketRegime.VOLATILE] += 0.3

        # === SIDEWAYS regime scoring ===
        if abs(ind.spy_return_20d) < 0.02 and abs(ind.spy_return_60d) < 0.04:
            scores[MarketRegime.SIDEWAYS] += 0.4
        if self.VIX_LOW < ind.vix < self.VIX_ELEVATED:
            scores[MarketRegime.SIDEWAYS] += 0.3
        if abs(ind.gold_return_20d) < 0.02 and abs(ind.usd_index_return) < 0.01:
            scores[MarketRegime.SIDEWAYS] += 0.3

        # === INFLATION regime scoring ===
        if ind.tip_ief_spread > self.INFLATION_THRESHOLD:
            scores[MarketRegime.INFLATION] += 0.4
        if ind.gold_return_20d > 0.03:  # Gold as inflation hedge
            scores[MarketRegime.INFLATION] += 0.3
        if ind.treasury_10y > ind.treasury_2y + 0.5:  # Steep yield curve
            scores[MarketRegime.INFLATION] += 0.3

        # === DEFLATION regime scoring ===
        if ind.tip_ief_spread < self.DEFLATION_THRESHOLD:
            scores[MarketRegime.DEFLATION] += 0.4
        if ind.treasury_10y < 2.0 and ind.spy_return_20d < 0:
            scores[MarketRegime.DEFLATION] += 0.3
        if ind.usd_index_return > 0.03:  # Strong dollar = deflationary
            scores[MarketRegime.DEFLATION] += 0.3

        # === CRISIS regime scoring ===
        if ind.vix > self.VIX_EXTREME:
            scores[MarketRegime.CRISIS] += 0.5
        if ind.spy_return_20d < -0.10:  # 10% drawdown
            scores[MarketRegime.CRISIS] += 0.3
        if ind.high_yield_spread is not None and ind.high_yield_spread > 7.0:
            scores[MarketRegime.CRISIS] += 0.2

        # Normalize scores
        max_score = max(scores.values()) if scores.values() else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

        return scores

    def _calculate_confidence(
        self,
        primary_score: float,
        secondary_score: float,
        all_scores: Dict[MarketRegime, float],
    ) -> float:
        """Calculate confidence in regime detection."""
        if primary_score == 0:
            return 0.0

        # Base confidence from primary score
        confidence = primary_score

        # Boost if clear separation from secondary
        if primary_score > 0 and secondary_score > 0:
            separation = (primary_score - secondary_score) / primary_score
            confidence *= (0.7 + 0.3 * separation)

        # Reduce if too many competing regimes
        high_scoring = sum(1 for s in all_scores.values() if s > 0.5)
        if high_scoring > 2:
            confidence *= 0.8

        return min(confidence, 1.0)

    def _update_history(self, regime: MarketRegime, timestamp: datetime):
        """Update regime history."""
        if not self.regime_history or self.regime_history[-1][1] != regime:
            self.regime_history.append((timestamp, regime))

        # Trim to max size
        if len(self.regime_history) > self.regime_history_size:
            self.regime_history = self.regime_history[-self.regime_history_size:]

    def get_regime_for_asset_type(
        self,
        asset_type: str,
        detection: Optional[RegimeDetection] = None,
    ) -> Tuple[MarketRegime, str]:
        """
        Get the relevant regime for an asset type.

        Different assets care about different regimes:
        - JPY pairs: BULL/BEAR for currency strength
        - Crude Oil: INFLATION/DEFLATION for commodity demand
        - Stocks: RISK_ON/RISK_OFF for equity flows

        Args:
            asset_type: 'jpy_pair', 'crude_oil', 'stock', etc.
            detection: Optional pre-computed detection

        Returns:
            (relevant_regime, explanation)
        """
        det = detection or self._last_detection
        if not det:
            return MarketRegime.SIDEWAYS, "No regime detection available"

        primary = det.primary_regime
        secondary = det.secondary_regime

        if asset_type == 'jpy_pair':
            # JPY SELL only works in BEAR/RISK_OFF
            if primary in [MarketRegime.BEAR, MarketRegime.RISK_OFF, MarketRegime.CRISIS]:
                return primary, "JPY strengthens in risk-off/bear markets"
            elif primary in [MarketRegime.BULL, MarketRegime.RISK_ON]:
                return primary, "JPY weakens in risk-on/bull markets - SELL risky"
            return primary, "Neutral regime for JPY"

        elif asset_type == 'crude_oil':
            # Crude Oil BUY in INFLATION/BULL, SELL in DEFLATION/BEAR
            if primary in [MarketRegime.INFLATION, MarketRegime.BULL]:
                return primary, "Oil rises in inflation/bull - BUY favorable"
            elif primary in [MarketRegime.DEFLATION, MarketRegime.BEAR, MarketRegime.CRISIS]:
                return primary, "Oil falls in deflation/bear - SELL favorable"
            return primary, "Neutral regime for crude oil"

        else:
            return primary, f"General market regime: {primary.value}"

    def is_jpy_sell_favorable(self, detection: Optional[RegimeDetection] = None) -> Tuple[bool, str]:
        """
        Check if current regime is favorable for JPY SELL signals.

        Fix 17: JPY SELL has 0% win rate in certain regimes.
        - BLOCK in: BULL, RISK_ON (JPY weakens, SELL loses)
        - ALLOW in: BEAR, RISK_OFF, CRISIS (JPY strengthens, SELL wins)

        Returns:
            (is_favorable, reason)
        """
        det = detection or self._last_detection
        if not det:
            return False, "No regime detection - blocking JPY SELL by default"

        favorable_regimes = [MarketRegime.BEAR, MarketRegime.RISK_OFF, MarketRegime.CRISIS]
        unfavorable_regimes = [MarketRegime.BULL, MarketRegime.RISK_ON]

        if det.primary_regime in favorable_regimes:
            return True, f"JPY SELL favorable in {det.primary_regime.value} regime"
        elif det.primary_regime in unfavorable_regimes:
            return False, f"JPY SELL blocked in {det.primary_regime.value} regime (0% historical win rate)"
        else:
            # Neutral regimes - allow with caution
            if det.confidence > 0.7:
                return True, f"JPY SELL allowed in {det.primary_regime.value} with caution"
            return False, f"JPY SELL blocked due to low regime confidence ({det.confidence:.0%})"

    def is_crude_oil_signal_favorable(
        self,
        signal_type: str,
        detection: Optional[RegimeDetection] = None,
    ) -> Tuple[bool, str]:
        """
        Check if current regime is favorable for Crude Oil signals.

        Fix 18: Crude Oil CL=F is a consistent loser in wrong regimes.
        - BUY favorable in: INFLATION, BULL
        - SELL favorable in: DEFLATION, BEAR
        - Block both in: VOLATILE, CRISIS, SIDEWAYS

        Args:
            signal_type: 'BUY' or 'SELL'
            detection: Optional pre-computed detection

        Returns:
            (is_favorable, reason)
        """
        det = detection or self._last_detection
        if not det:
            return False, "No regime detection - blocking crude oil by default"

        signal_upper = signal_type.upper()

        if signal_upper == 'BUY':
            favorable = [MarketRegime.INFLATION, MarketRegime.BULL, MarketRegime.RISK_ON]
            if det.primary_regime in favorable:
                return True, f"Crude BUY favorable in {det.primary_regime.value}"
            return False, f"Crude BUY blocked in {det.primary_regime.value}"

        elif signal_upper == 'SELL':
            favorable = [MarketRegime.DEFLATION, MarketRegime.BEAR]
            if det.primary_regime in favorable:
                return True, f"Crude SELL favorable in {det.primary_regime.value}"
            return False, f"Crude SELL blocked in {det.primary_regime.value}"

        return False, "Unknown signal type"

    def get_position_multiplier(self, detection: Optional[RegimeDetection] = None) -> float:
        """
        Get position size multiplier based on current regime.

        Fix 19: Dynamic position adjustment.
        - High confidence, favorable regime: 1.0-1.5x
        - Medium confidence: 0.7-1.0x
        - Low confidence, volatile: 0.3-0.7x
        - Crisis: 0.1-0.3x

        Returns:
            Position multiplier (0.1 to 1.5)
        """
        det = detection or self._last_detection
        if not det:
            return 0.5  # Conservative default

        regime = det.primary_regime
        confidence = det.confidence

        # Base multiplier by regime
        regime_multipliers = {
            MarketRegime.BULL: 1.2,
            MarketRegime.RISK_ON: 1.1,
            MarketRegime.SIDEWAYS: 0.8,
            MarketRegime.BEAR: 0.7,
            MarketRegime.RISK_OFF: 0.6,
            MarketRegime.VOLATILE: 0.5,
            MarketRegime.INFLATION: 0.9,
            MarketRegime.DEFLATION: 0.7,
            MarketRegime.CRISIS: 0.3,
        }

        base = regime_multipliers.get(regime, 0.8)

        # Adjust by confidence
        if confidence > 0.8:
            multiplier = base * 1.2
        elif confidence > 0.6:
            multiplier = base * 1.0
        elif confidence > 0.4:
            multiplier = base * 0.8
        else:
            multiplier = base * 0.6

        return max(0.1, min(1.5, multiplier))


def create_indicators_from_data(
    vix: float = 20.0,
    spy_return_20d: float = 0.0,
    spy_return_60d: float = 0.0,
    spy_above_200ma: bool = True,
    treasury_10y: float = 4.0,
    treasury_2y: float = 4.5,
    usd_index_return: float = 0.0,
    tip_ief_spread: float = 0.0,
    gold_return_20d: float = 0.0,
    high_yield_spread: Optional[float] = None,
) -> RegimeIndicators:
    """
    Create RegimeIndicators from raw data.

    This is a convenience function for creating indicators from
    external data sources like Yahoo Finance.
    """
    return RegimeIndicators(
        vix=vix,
        spy_return_20d=spy_return_20d,
        spy_return_60d=spy_return_60d,
        spy_above_200ma=spy_above_200ma,
        treasury_10y=treasury_10y,
        treasury_2y=treasury_2y,
        usd_index_return=usd_index_return,
        tip_ief_spread=tip_ief_spread,
        gold_return_20d=gold_return_20d,
        high_yield_spread=high_yield_spread,
        timestamp=datetime.now(),
    )


if __name__ == '__main__':
    # Test the regime detector
    print("=" * 80)
    print("MARKET REGIME DETECTOR - TEST")
    print("=" * 80)

    detector = MarketRegimeDetector()

    # Test different market conditions
    test_cases = [
        ("Bull Market", create_indicators_from_data(
            vix=14, spy_return_20d=0.08, spy_return_60d=0.15,
            spy_above_200ma=True, gold_return_20d=-0.02
        )),
        ("Bear Market", create_indicators_from_data(
            vix=30, spy_return_20d=-0.10, spy_return_60d=-0.15,
            spy_above_200ma=False, gold_return_20d=0.05
        )),
        ("Risk-Off", create_indicators_from_data(
            vix=25, spy_return_20d=-0.03, spy_return_60d=-0.05,
            spy_above_200ma=True, gold_return_20d=0.04, high_yield_spread=6.0
        )),
        ("Sideways", create_indicators_from_data(
            vix=17, spy_return_20d=0.01, spy_return_60d=0.02,
            spy_above_200ma=True, gold_return_20d=0.01
        )),
        ("Crisis", create_indicators_from_data(
            vix=45, spy_return_20d=-0.15, spy_return_60d=-0.20,
            spy_above_200ma=False, gold_return_20d=0.08, high_yield_spread=9.0
        )),
    ]

    for name, indicators in test_cases:
        detection = detector.detect_regime(indicators)
        print(f"\n{name}:")
        print(f"  Primary: {detection.primary_regime.value} ({detection.confidence:.0%} confidence)")
        if detection.secondary_regime:
            print(f"  Secondary: {detection.secondary_regime.value}")

        # Test JPY SELL
        jpy_ok, jpy_reason = detector.is_jpy_sell_favorable(detection)
        print(f"  JPY SELL: {'ALLOW' if jpy_ok else 'BLOCK'} - {jpy_reason}")

        # Test Crude Oil
        crude_buy_ok, crude_buy_reason = detector.is_crude_oil_signal_favorable('BUY', detection)
        crude_sell_ok, crude_sell_reason = detector.is_crude_oil_signal_favorable('SELL', detection)
        print(f"  Crude BUY: {'ALLOW' if crude_buy_ok else 'BLOCK'} - {crude_buy_reason}")
        print(f"  Crude SELL: {'ALLOW' if crude_sell_ok else 'BLOCK'} - {crude_sell_reason}")

        # Position multiplier
        mult = detector.get_position_multiplier(detection)
        print(f"  Position Multiplier: {mult:.2f}x")

    print("\n" + "=" * 80)
