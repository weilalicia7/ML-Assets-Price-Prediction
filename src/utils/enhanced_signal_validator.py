"""
Enhanced Signal Validator for China Model
Fixes catastrophic short losses by preventing dangerous shorts

Based on analysis showing:
- BUY signals: +92.9% profit
- SELL signals: -94.3% loss (catastrophic shorts)

Key fixes:
1. Prevent shorting stocks with strong momentum + high volume
2. Higher confidence thresholds for shorts (0.7 vs 0.6)
3. Conservative short sizing (max 30%)
4. Volatility-based position caps
"""

import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime


class EnhancedSignalValidator:
    """
    Validates trading signals with enhanced short protection.

    Prevents catastrophic shorts by identifying dangerous patterns:
    - Strong momentum + high volume (squeeze candidates)
    - Low float + high short interest
    - Recent breakouts with volume confirmation
    - Extreme volatility + upward trend
    """

    def __init__(self,
                 min_confidence_long: float = 0.6,
                 min_confidence_short: float = 0.7,  # Higher bar for shorts
                 max_short_momentum: float = 0.3,    # Don't short if >30% momentum
                 max_short_volatility: float = 0.6): # Don't short if >60% vol

        self.min_confidence_long = min_confidence_long
        self.min_confidence_short = min_confidence_short
        self.max_short_momentum = max_short_momentum
        self.max_short_volatility = max_short_volatility

        # Dangerous sectors for shorts (prone to squeezes)
        self.dangerous_short_sectors = [
            'Real Estate',  # Zhongfu Straits, China Vanke
            'Technology',   # BlueFocus, SenseTime
            'Green Energy', # GCL Technology
        ]

        # Track validation history
        self.validation_history = []

    def validate_signal(self,
                       signal: str,
                       confidence: float,
                       ticker_info: Dict) -> Tuple[str, float, str]:
        """
        Validate a trading signal with enhanced short protection.

        Args:
            signal: 'BUY', 'SELL', or 'HOLD'
            confidence: Model confidence (0-1)
            ticker_info: Dict with momentum, volatility, volume_ratio, sector, etc.

        Returns:
            (validated_signal, adjusted_confidence, reason)
        """

        # BUY signals - apply trend validation (NEW: prevent buying falling knives)
        if signal == 'BUY':
            # Check 1: Is this a dangerous buy (falling knife)?
            is_dangerous, danger_reason = self._is_dangerous_buy(ticker_info)
            if is_dangerous:
                return 'HOLD', confidence * 0.3, f"DANGEROUS_BUY_BLOCKED: {danger_reason}"

            # Check 2: Confidence threshold
            if confidence < self.min_confidence_long:
                return 'HOLD', confidence, f"LOW_CONF_LONG ({confidence:.2f} < {self.min_confidence_long})"

            # Check 3: Classify buy quality for position sizing
            buy_quality = self._classify_buy_quality(ticker_info)
            quality_adjustment = {
                'strong_buy': 1.2,
                'cautious_buy': 0.9,
                'recovery_buy': 0.7
            }.get(buy_quality, 1.0)

            adjusted_conf = confidence * quality_adjustment
            return signal, adjusted_conf, f"VALID_LONG ({buy_quality})"

        # SELL signals - apply strict validation
        if signal == 'SELL':
            # Check 1: Is this a dangerous short?
            if self._is_dangerous_short(ticker_info):
                return 'HOLD', confidence * 0.3, "DANGEROUS_SHORT_BLOCKED"

            # Check 2: Higher confidence threshold for shorts
            if confidence < self.min_confidence_short:
                return 'HOLD', confidence, f"LOW_CONF_SHORT ({confidence:.2f} < {self.min_confidence_short})"

            # Check 3: Momentum too strong
            momentum = ticker_info.get('momentum_20d', 0)
            if momentum > self.max_short_momentum:
                return 'HOLD', confidence * 0.5, f"HIGH_MOMENTUM ({momentum:.1%} > {self.max_short_momentum:.1%})"

            # Check 4: Volatility too high
            volatility = ticker_info.get('volatility', 0)
            if volatility > self.max_short_volatility:
                adjusted_conf = confidence * (self.max_short_volatility / volatility)
                return signal, adjusted_conf, f"VOL_ADJUSTED ({volatility:.1%})"

            # Check 5: Sector check
            sector = ticker_info.get('sector', '')
            if sector in self.dangerous_short_sectors:
                return signal, confidence * 0.7, f"SECTOR_RISK ({sector})"

            return signal, confidence, "VALID_SHORT"

        # HOLD signals pass through
        return signal, confidence, "HOLD_PASSTHROUGH"

    def _is_dangerous_short(self, ticker_info: Dict) -> bool:
        """
        Identify stocks that should NEVER be shorted.

        Patterns that indicate dangerous shorts:
        1. Strong momentum + high volume (squeeze setup)
        2. Recent breakout with volume confirmation
        3. Extreme volatility + upward trend
        4. Low float characteristics
        """

        momentum_20d = ticker_info.get('momentum_20d', 0)
        momentum_5d = ticker_info.get('momentum_5d', 0)
        volume_ratio = ticker_info.get('volume_ratio', 1)
        volatility = ticker_info.get('volatility', 0)
        dist_ma = ticker_info.get('dist_from_ma', 0)

        # Pattern 1: Strong momentum + high volume (CLASSIC SQUEEZE)
        # This caught Zhongfu Straits (+290% loss)
        if momentum_20d > 0.5 and volume_ratio > 2.0:
            return True

        # Pattern 2: Extreme recent momentum (breakout)
        # This caught Winnovation (+77% loss)
        if momentum_5d > 0.15 and volume_ratio > 1.5:
            return True

        # Pattern 3: Far above moving average + high volatility
        # Stock already extended, could squeeze further
        if dist_ma > 0.20 and volatility > 0.40:
            return True

        # Pattern 4: Triple threat - momentum + vol + volume
        if momentum_20d > 0.3 and volatility > 0.50 and volume_ratio > 1.5:
            return True

        # Pattern 5: Parabolic move (extreme short-term momentum)
        if momentum_5d > 0.25:
            return True

        return False

    def _is_dangerous_buy(self, ticker_info: Dict) -> Tuple[bool, str]:
        """
        Identify BUY signals that should be blocked (falling knives).

        Based on PDF analysis showing BUY losses due to:
        1. Buying stocks in strong downtrends (catching falling knives)
        2. No trend confirmation (price below MA20)
        3. High volatility without momentum improvement

        Returns:
            (is_dangerous, reason)
        """

        momentum_20d = ticker_info.get('momentum_20d', 0)
        momentum_5d = ticker_info.get('momentum_5d', 0)
        price_vs_ma20 = ticker_info.get('dist_from_ma', 0)
        volatility = ticker_info.get('volatility', 0)

        # HARD FILTER 1: Never buy strong downtrends (mom_20d < -5%)
        if momentum_20d < -0.05:
            return True, f"Strong downtrend (mom_20d: {momentum_20d:.1%} < -5%)"

        # HARD FILTER 2: Price significantly below MA20 without improving momentum
        if price_vs_ma20 < -0.05 and momentum_5d < momentum_20d:
            return True, f"Falling knife (below MA20: {price_vs_ma20:.1%}, momentum worsening)"

        # HARD FILTER 3: High volatility + downtrend combination
        if volatility > 0.40 and momentum_20d < -0.02:
            return True, f"High vol downtrend (vol: {volatility:.0%}, mom: {momentum_20d:.1%})"

        return False, "Trend OK"

    def _classify_buy_quality(self, ticker_info: Dict) -> str:
        """
        Classify BUY signal quality for position sizing.

        Based on PDF recommendations:
        - strong_buy: Uptrend confirmation (mom_20d > 2%, mom_5d > 1%, price > MA20)
        - cautious_buy: Improving momentum (mom_5d > mom_20d)
        - recovery_buy: Bottoming pattern

        Returns:
            'strong_buy', 'cautious_buy', or 'recovery_buy'
        """

        momentum_20d = ticker_info.get('momentum_20d', 0)
        momentum_5d = ticker_info.get('momentum_5d', 0)
        price_vs_ma20 = ticker_info.get('dist_from_ma', 0)

        # STRONG BUY: Full uptrend confirmation
        if (momentum_20d > 0.02 and
            momentum_5d > 0.01 and
            price_vs_ma20 > 0.01):
            return 'strong_buy'

        # CAUTIOUS BUY: Improving momentum (5d > 20d and 5d positive)
        if (momentum_5d > momentum_20d and
            momentum_5d > 0.02 and
            momentum_20d > -0.02):
            return 'cautious_buy'

        # RECOVERY BUY: Everything else that passed danger checks
        return 'recovery_buy'

    def get_validation_stats(self) -> Dict:
        """Get statistics on signal validation."""
        if not self.validation_history:
            return {}

        total = len(self.validation_history)
        blocked = sum(1 for v in self.validation_history if 'BLOCKED' in v.get('reason', ''))
        adjusted = sum(1 for v in self.validation_history if 'ADJUSTED' in v.get('reason', ''))

        return {
            'total_validated': total,
            'shorts_blocked': blocked,
            'confidence_adjusted': adjusted,
            'block_rate': blocked / total if total > 0 else 0
        }


class BuyStopLossManager:
    """
    Stop-Loss System for BUY Signals.

    Based on PDF recommendations:
    - 5-7% stop-loss range
    - Tighter stops for high volatility
    - Time-based stops for weak trends
    """

    def __init__(self):
        self.stop_levels = {
            'strong_buy': 0.06,    # 6% stop in strong uptrends
            'cautious_buy': 0.05,  # 5% stop in cautious buys
            'recovery_buy': 0.04,  # 4% stop for recovery buys (tighter)
        }
        self.time_stops = {
            'strong_buy': 15,      # 15 days max hold
            'cautious_buy': 10,    # 10 days max hold
            'recovery_buy': 7,     # 7 days max hold
        }

    def calculate_stop_loss(self, buy_quality: str, volatility: float) -> float:
        """
        Calculate stop-loss percentage based on buy quality and volatility.

        Args:
            buy_quality: 'strong_buy', 'cautious_buy', or 'recovery_buy'
            volatility: Annualized volatility

        Returns:
            Stop-loss percentage (e.g., 0.06 = 6%)
        """
        base_stop = self.stop_levels.get(buy_quality, 0.06)

        # Tighten stops for high volatility (PDF recommendation)
        if volatility > 0.40:
            base_stop *= 0.80  # 20% tighter
        elif volatility > 0.30:
            base_stop *= 0.90  # 10% tighter
        elif volatility < 0.15:
            base_stop *= 1.15  # Slightly wider for low vol

        # Ensure within 4-8% range
        return max(0.04, min(base_stop, 0.08))

    def get_max_hold_days(self, buy_quality: str) -> int:
        """Get maximum hold period based on buy quality."""
        return self.time_stops.get(buy_quality, 10)

    def should_exit_trade(self, position: Dict, current_price: float,
                         days_held: int) -> Tuple[bool, str]:
        """
        Check if stop-loss or time-stop triggered.

        Args:
            position: Dict with entry_price, stop_loss_pct, buy_quality
            current_price: Current stock price
            days_held: Number of days since entry

        Returns:
            (should_exit, reason)
        """
        entry_price = position.get('entry_price', current_price)
        stop_loss_pct = position.get('stop_loss_pct', 0.06)
        buy_quality = position.get('buy_quality', 'cautious_buy')

        current_return = (current_price - entry_price) / entry_price

        # Stop-loss check
        if current_return <= -stop_loss_pct:
            return True, f"Stop-loss triggered: {current_return:.1%}"

        # Time-stop check (for non-profitable positions)
        max_days = self.get_max_hold_days(buy_quality)
        if days_held >= max_days and current_return < 0.02:
            return True, f"Time-stop: {days_held} days, return {current_return:.1%}"

        return False, "Hold"


class SmartPositionSizer:
    """
    Dynamic position sizing with conservative short sizing.

    Key principles:
    - Longs can size up to 100% of base
    - Shorts capped at 30% max
    - Volatility-based scaling
    - Momentum penalty for shorts
    """

    def __init__(self,
                 base_position_pct: float = 0.05,  # 5% base position
                 max_long_pct: float = 0.10,       # 10% max long
                 max_short_pct: float = 0.03,      # 3% max short (conservative!)
                 vol_target: float = 0.15):        # 15% target vol

        self.base_position_pct = base_position_pct
        self.max_long_pct = max_long_pct
        self.max_short_pct = max_short_pct
        self.vol_target = vol_target

    def calculate_position_size(self,
                               signal: str,
                               confidence: float,
                               ticker_info: Dict) -> Tuple[float, Dict]:
        """
        Calculate position size with risk controls.

        Args:
            signal: 'BUY' or 'SELL'
            confidence: Adjusted confidence (0-1)
            ticker_info: Dict with volatility, momentum, etc.

        Returns:
            (position_size_pct, sizing_breakdown)
        """

        volatility = ticker_info.get('volatility', 0.25)
        momentum = ticker_info.get('momentum_20d', 0)
        regime = ticker_info.get('regime', 'NORMAL')

        # Start with base size
        base_size = self.base_position_pct

        # Step 1: Volatility scaling (vol targeting)
        vol_scale = self.vol_target / max(volatility, 0.10)
        vol_scale = np.clip(vol_scale, 0.3, 2.0)

        # Step 2: Confidence scaling
        conf_scale = confidence

        # Step 3: Regime adjustment
        regime_scale = self._get_regime_scale(regime)

        # Calculate base adjusted size
        adjusted_size = base_size * vol_scale * conf_scale * regime_scale

        # Step 4: Apply signal-specific limits
        if signal == 'BUY':
            # Enhanced BUY sizing (PDF recommendations)
            buy_quality = ticker_info.get('buy_quality', 'cautious_buy')
            adjusted_size = self._smart_buy_sizing(adjusted_size, ticker_info, buy_quality)
            final_size = min(adjusted_size, self.max_long_pct)

        elif signal == 'SELL':
            # Shorts - apply CONSERVATIVE limits
            adjusted_size = self._conservative_short_sizing(adjusted_size, ticker_info)
            final_size = min(adjusted_size, self.max_short_pct)

        else:
            final_size = 0

        # Build breakdown
        breakdown = {
            'base_size': self.base_position_pct,
            'vol_scale': vol_scale,
            'conf_scale': conf_scale,
            'regime_scale': regime_scale,
            'pre_limit_size': adjusted_size,
            'final_size': final_size,
            'signal': signal,
            'capped': final_size < adjusted_size
        }

        return final_size, breakdown

    def _smart_buy_sizing(self, base_size: float, ticker_info: Dict, buy_quality: str) -> float:
        """
        Enhanced BUY sizing based on PDF recommendations.

        Key sizing rules:
        - strong_buy: Up to 6% (full confidence)
        - cautious_buy: Up to 3% (reduced size)
        - recovery_buy: Up to 2% (minimal size)

        Additional adjustments:
        - High volatility (>35%): 30% reduction
        - Market downtrend: 40% reduction
        """

        volatility = ticker_info.get('volatility', 0.25)
        market_trend = ticker_info.get('market_trend', 'neutral')

        # Base size adjustment by buy quality (PDF Table)
        quality_multipliers = {
            'strong_buy': 1.2,    # Size up for strong trends
            'cautious_buy': 0.8,  # Reduce for cautious buys
            'recovery_buy': 0.5,  # Minimal for recovery plays
        }
        quality_mult = quality_multipliers.get(buy_quality, 0.8)
        size = base_size * quality_mult

        # Volatility penalty (PDF recommendation #5)
        if volatility > 0.35:
            size *= 0.70  # 30% reduction for high vol
        elif volatility > 0.25:
            size *= 0.85  # 15% reduction for moderate vol

        # Market regime adjustment (PDF recommendation #4)
        if market_trend == 'downtrend':
            size *= 0.60  # 40% reduction in downtrends

        # Ensure minimum viable position (1%) and max (8%)
        return max(0.01, min(size, 0.08))

    def _conservative_short_sizing(self, base_size: float, ticker_info: Dict) -> float:
        """
        Apply conservative sizing for short positions.

        Reduces size based on:
        - Momentum (higher momentum = smaller short)
        - Volume (higher volume = smaller short)
        - Volatility (higher vol = smaller short)
        """

        momentum = ticker_info.get('momentum_20d', 0)
        volume_ratio = ticker_info.get('volume_ratio', 1)
        volatility = ticker_info.get('volatility', 0.25)

        size = base_size

        # Momentum penalty (most important)
        if momentum > 0.3:
            size *= 0.3  # 70% reduction for strong momentum
        elif momentum > 0.1:
            size *= 0.5  # 50% reduction for moderate momentum
        elif momentum > 0:
            size *= 0.7  # 30% reduction for any positive momentum

        # Volume penalty
        if volume_ratio > 2.0:
            size *= 0.5  # High volume = more squeeze risk
        elif volume_ratio > 1.5:
            size *= 0.7

        # Volatility cap
        if volatility > 0.50:
            size *= 0.5  # High vol = reduce size
        elif volatility > 0.35:
            size *= 0.7

        return size

    def _get_regime_scale(self, regime: str) -> float:
        """Get position scale based on market regime."""
        regime_scales = {
            'CRISIS': 0.3,      # Minimal positions in crisis
            'HIGH_VOL': 0.5,    # Reduced in high vol
            'ELEVATED': 0.7,    # Slightly reduced
            'NORMAL': 1.0,      # Full sizing
            'LOW_VOL': 1.2,     # Can size up slightly
        }
        return regime_scales.get(regime, 1.0)


class EnhancedRegimeDetector:
    """
    Enhanced regime detection with speculative pattern identification.

    Identifies:
    - Standard volatility regimes (CRISIS, HIGH_VOL, etc.)
    - Speculative patterns (squeeze candidates)
    - Sector-specific regimes
    """

    def __init__(self):
        self.regime_history = []

    def detect_regime(self,
                     volatility: float,
                     volume_ratio: float = 1.0,
                     momentum: float = 0.0,
                     market_breadth: float = 0.5) -> Tuple[str, float, Dict]:
        """
        Detect market regime with enhanced pattern recognition.

        Args:
            volatility: Annualized volatility (e.g., 0.35 = 35%)
            volume_ratio: Current volume / average volume
            momentum: 20-day momentum
            market_breadth: % of stocks above MA (0-1)

        Returns:
            (regime, regime_multiplier, regime_info)
        """

        # Check for speculative pattern first
        if self._is_speculative_pattern(volatility, volume_ratio, momentum):
            return 'SPECULATIVE', 0.3, {
                'warning': 'Speculative pattern detected - reduce short exposure',
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'momentum': momentum
            }

        # Standard volatility-based regime
        if volatility > 0.60:
            regime = 'CRISIS'
            multiplier = 0.3
        elif volatility > 0.45:
            regime = 'HIGH_VOL'
            multiplier = 0.5
        elif volatility > 0.30:
            regime = 'ELEVATED'
            multiplier = 0.7
        elif volatility < 0.15:
            regime = 'LOW_VOL'
            multiplier = 1.15
        else:
            regime = 'NORMAL'
            multiplier = 1.0

        # Adjust for market breadth
        if market_breadth > 0.7:  # Strong bull
            multiplier *= 1.1
        elif market_breadth < 0.3:  # Strong bear
            multiplier *= 0.9

        regime_info = {
            'volatility': volatility,
            'volume_ratio': volume_ratio,
            'momentum': momentum,
            'market_breadth': market_breadth
        }

        return regime, multiplier, regime_info

    def _is_speculative_pattern(self,
                               volatility: float,
                               volume_ratio: float,
                               momentum: float) -> bool:
        """
        Detect speculative/squeeze patterns.

        These patterns are DANGEROUS for shorts:
        - High vol + high volume + positive momentum
        - Classic short squeeze setup
        """

        # Triple threat pattern
        if volatility > 0.50 and volume_ratio > 1.8 and momentum > 0.3:
            return True

        # Volume surge + momentum
        if volume_ratio > 2.5 and momentum > 0.2:
            return True

        # Extreme momentum (parabolic)
        if momentum > 0.5:
            return True

        return False

    def get_regime_for_ticker(self, ticker_data: Dict) -> Tuple[str, float]:
        """
        Get regime for a specific ticker.

        Args:
            ticker_data: Dict with volatility, returns, volume data

        Returns:
            (regime, multiplier)
        """

        vol = ticker_data.get('volatility', 0.25)
        vol_ratio = ticker_data.get('volume_ratio', 1.0)
        mom = ticker_data.get('momentum_20d', 0.0)

        regime, mult, _ = self.detect_regime(vol, vol_ratio, mom)
        return regime, mult


class PerformanceTracker:
    """
    Tracks historical signal performance for accuracy weighting.

    Uses rolling accuracy to adjust confidence:
    - High historical accuracy = trust signals more
    - Low historical accuracy = reduce confidence
    """

    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days
        self.signal_history = []  # List of (date, ticker, signal, outcome)

    def record_signal(self,
                     ticker: str,
                     signal: str,
                     confidence: float,
                     actual_return: Optional[float] = None):
        """Record a signal for tracking."""

        self.signal_history.append({
            'date': datetime.now(),
            'ticker': ticker,
            'signal': signal,
            'confidence': confidence,
            'actual_return': actual_return
        })

        # Keep only recent history
        cutoff = datetime.now()
        self.signal_history = [
            s for s in self.signal_history
            if (cutoff - s['date']).days <= self.lookback_days
        ]

    def get_accuracy_adjustment(self, signal_type: str = 'all') -> float:
        """
        Get accuracy-based adjustment factor.

        Returns:
            Adjustment factor (0.5 to 1.5)
        """

        if not self.signal_history:
            return 1.0

        # Filter by signal type if specified
        if signal_type == 'BUY':
            signals = [s for s in self.signal_history if s['signal'] == 'BUY']
        elif signal_type == 'SELL':
            signals = [s for s in self.signal_history if s['signal'] == 'SELL']
        else:
            signals = self.signal_history

        if not signals:
            return 1.0

        # Calculate accuracy (signals with known outcomes)
        with_outcome = [s for s in signals if s['actual_return'] is not None]

        if len(with_outcome) < 5:  # Need minimum history
            return 1.0

        # Count correct signals
        correct = 0
        for s in with_outcome:
            if s['signal'] == 'BUY' and s['actual_return'] > 0:
                correct += 1
            elif s['signal'] == 'SELL' and s['actual_return'] < 0:
                correct += 1

        accuracy = correct / len(with_outcome)

        # Convert to adjustment factor
        # 50% accuracy = 0.75x, 70% = 1.0x, 90% = 1.25x
        adjustment = 0.5 + accuracy
        return np.clip(adjustment, 0.5, 1.5)

    def get_stats(self) -> Dict:
        """Get performance statistics."""

        total = len(self.signal_history)
        buys = sum(1 for s in self.signal_history if s['signal'] == 'BUY')
        sells = sum(1 for s in self.signal_history if s['signal'] == 'SELL')

        return {
            'total_signals': total,
            'buy_signals': buys,
            'sell_signals': sells,
            'buy_adjustment': self.get_accuracy_adjustment('BUY'),
            'sell_adjustment': self.get_accuracy_adjustment('SELL')
        }


class EnhancedPhaseSystem:
    """
    Integrates all enhanced components into the Phase 1-6 system.

    Flow:
    1. Generate base signal and confidence
    2. Validate signal (block dangerous shorts)
    3. Detect regime
    4. Calculate position size
    5. Apply performance adjustments
    """

    def __init__(self):
        self.validator = EnhancedSignalValidator()
        self.sizer = SmartPositionSizer()
        self.regime_detector = EnhancedRegimeDetector()
        self.performance_tracker = PerformanceTracker()
        self.buy_stop_loss_mgr = BuyStopLossManager()  # NEW: BUY stop-loss manager

    def process_signal(self,
                      ticker: str,
                      base_signal: str,
                      base_confidence: float,
                      ticker_info: Dict) -> Dict:
        """
        Process a trading signal through the enhanced system.

        Args:
            ticker: Stock ticker
            base_signal: Original signal ('BUY', 'SELL', 'HOLD')
            base_confidence: Original confidence (0-1)
            ticker_info: Dict with momentum, volatility, volume, etc.

        Returns:
            Dict with processed signal, confidence, position size, etc.
        """

        # Step 1: Validate signal
        validated_signal, validated_conf, validation_reason = self.validator.validate_signal(
            base_signal, base_confidence, ticker_info
        )

        # Step 2: Detect regime
        regime, regime_mult = self.regime_detector.get_regime_for_ticker(ticker_info)

        # Step 3: Apply regime to confidence
        regime_adjusted_conf = validated_conf * regime_mult

        # Step 4: Apply performance adjustment
        perf_adjustment = self.performance_tracker.get_accuracy_adjustment(validated_signal)
        final_confidence = regime_adjusted_conf * perf_adjustment
        final_confidence = np.clip(final_confidence, 0.1, 0.95)

        # Step 5: Calculate position size
        ticker_info['regime'] = regime

        # NEW: Extract buy quality from validation reason for sizing
        buy_quality = 'cautious_buy'  # Default
        if validated_signal == 'BUY' and 'VALID_LONG' in validation_reason:
            # Extract quality from reason like "VALID_LONG (strong_buy)"
            if 'strong_buy' in validation_reason:
                buy_quality = 'strong_buy'
            elif 'cautious_buy' in validation_reason:
                buy_quality = 'cautious_buy'
            elif 'recovery_buy' in validation_reason:
                buy_quality = 'recovery_buy'
            ticker_info['buy_quality'] = buy_quality

        position_size, sizing_breakdown = self.sizer.calculate_position_size(
            validated_signal, final_confidence, ticker_info
        )

        # NEW: Calculate stop-loss for BUY signals
        stop_loss_pct = None
        max_hold_days = None
        if validated_signal == 'BUY':
            volatility = ticker_info.get('volatility', 0.25)
            stop_loss_pct = self.buy_stop_loss_mgr.calculate_stop_loss(buy_quality, volatility)
            max_hold_days = self.buy_stop_loss_mgr.get_max_hold_days(buy_quality)

        # Build result
        result = {
            'ticker': ticker,
            'original_signal': base_signal,
            'original_confidence': base_confidence,
            'validated_signal': validated_signal,
            'validated_confidence': validated_conf,
            'validation_reason': validation_reason,
            'regime': regime,
            'regime_multiplier': regime_mult,
            'performance_adjustment': perf_adjustment,
            'final_confidence': final_confidence,
            'position_size_pct': position_size,
            'sizing_breakdown': sizing_breakdown,
            'signal_blocked': validated_signal != base_signal,
            'is_dangerous_short': self.validator._is_dangerous_short(ticker_info) if base_signal == 'SELL' else False,
            # NEW: BUY-specific fields
            'is_dangerous_buy': self.validator._is_dangerous_buy(ticker_info)[0] if base_signal == 'BUY' else False,
            'buy_quality': buy_quality if validated_signal == 'BUY' else None,
            'stop_loss_pct': stop_loss_pct,
            'max_hold_days': max_hold_days,
        }

        # Record for tracking
        self.performance_tracker.record_signal(ticker, validated_signal, final_confidence)

        return result

    def get_system_stats(self) -> Dict:
        """Get statistics from all components."""
        return {
            'validator_stats': self.validator.get_validation_stats(),
            'performance_stats': self.performance_tracker.get_stats()
        }


# =============================================================================
# NEW OPTIMIZATION CLASSES (from buy fixing2 PDF)
# =============================================================================

class StopLossEnforcer:
    """
    Enforce stop-losses to prevent large losses like 000981.SZ (-18.6%).

    Based on PDF recommendations:
    - 6% maximum loss (from 5-7% range)
    - Simulates stop-loss impact on historical trades
    """

    def __init__(self, max_loss_pct: float = 0.06):
        self.max_loss_pct = max_loss_pct  # 6% maximum loss

    def enforce_stop_losses(self, active_positions: list, get_price_func=None) -> list:
        """
        Enforce stop-losses on all active positions.

        Args:
            active_positions: List of position dicts with ticker, entry_price, etc.
            get_price_func: Optional function to get current price (ticker -> price)

        Returns:
            List of positions that should be closed
        """
        positions_to_close = []

        for position in active_positions:
            current_price = position.get('current_price')
            if current_price is None and get_price_func:
                current_price = get_price_func(position['ticker'])

            if current_price is None:
                continue

            entry_price = position.get('entry_price', current_price)
            current_return = (current_price - entry_price) / entry_price

            if current_return <= -self.max_loss_pct:
                positions_to_close.append({
                    'ticker': position['ticker'],
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'return_pct': current_return,
                    'reason': f'Stop-loss triggered at {current_return:.1%}'
                })

        return positions_to_close

    def simulate_stop_loss_impact(self, historical_trades: list) -> Tuple[float, list]:
        """
        Simulate how stop-losses would have improved performance.

        Args:
            historical_trades: List of dicts with ticker, return, position_size

        Returns:
            (improved_pnl, saved_losses)
        """
        improved_pnl = 0
        saved_losses = []

        for trade in historical_trades:
            actual_return = trade.get('return', trade.get('actual_ret', 0))
            position_size = trade.get('position_size', 0.05)

            if actual_return < -self.max_loss_pct:
                # Stop-loss would have capped this loss
                saved_loss = abs(actual_return) - self.max_loss_pct
                improved_pnl += saved_loss * position_size
                saved_losses.append({
                    'ticker': trade.get('ticker', 'Unknown'),
                    'original_loss': actual_return,
                    'capped_loss': -self.max_loss_pct,
                    'saved': saved_loss
                })

        return improved_pnl, saved_losses


class WinnerOptimizer:
    """
    Increase position sizing on high-quality winners.

    Based on PDF recommendations:
    - High quality: +15% return potential, 75%+ confidence -> 1.5x size
    - Medium quality: +8% return potential, 65%+ confidence -> 1.0x size
    - Low quality: +2% return potential, 55%+ confidence -> 0.8x size
    """

    def __init__(self):
        self.quality_thresholds = {
            'high_quality': {'min_return': 0.15, 'min_confidence': 0.75, 'size_mult': 1.5},
            'medium_quality': {'min_return': 0.08, 'min_confidence': 0.65, 'size_mult': 1.0},
            'low_quality': {'min_return': 0.02, 'min_confidence': 0.55, 'size_mult': 0.8}
        }

    def optimize_winner_sizing(self, signal_data: Dict, actual_performance: Dict = None) -> float:
        """
        Increase sizing for proven winner patterns.

        Args:
            signal_data: Dict with position_size, confidence, trend_strength, etc.
            actual_performance: Optional dict with return_5d, volume_ratio (for tracking)

        Returns:
            Optimized position size
        """
        base_size = signal_data.get('position_size', 0.05)

        # Check if this matches high-quality winner pattern
        if actual_performance:
            return_5d = actual_performance.get('return_5d', 0)
            volume_ratio = actual_performance.get('volume_ratio', 1)
            trend_strength = signal_data.get('trend_strength', '')

            # High-quality winner pattern: positive 5d return + volume surge + uptrend
            if (return_5d > 0.05 and
                volume_ratio > 1.5 and
                trend_strength == 'strong_uptrend'):
                # Increase size up to 50%, max 8%
                return min(base_size * 1.5, 0.08)

        # Check buy quality from signal data
        buy_quality = signal_data.get('buy_quality', 'cautious_buy')

        quality_multipliers = {
            'strong_buy': 1.3,     # 30% increase for strong buys
            'cautious_buy': 1.0,   # No change
            'recovery_buy': 0.7    # 30% decrease for recovery plays
        }

        multiplier = quality_multipliers.get(buy_quality, 1.0)
        return min(base_size * multiplier, 0.08)

    def identify_winner_patterns(self, historical_winners: list) -> Dict:
        """
        Identify common patterns in winning trades.

        Args:
            historical_winners: List of winning trade dicts

        Returns:
            Dict with common pattern characteristics
        """
        if not historical_winners:
            return {}

        patterns = {
            'avg_pre_momentum': np.mean([w.get('momentum_20d', 0) for w in historical_winners]),
            'avg_volume_surge': np.mean([w.get('volume_ratio', 1) for w in historical_winners]),
            'common_sectors': self._get_common_sectors(historical_winners),
            'breakout_rate': sum(1 for w in historical_winners
                                if w.get('price_vs_high_20d', 0) > 0.95) / len(historical_winners)
        }

        return patterns

    def _get_common_sectors(self, trades: list) -> list:
        """Get most common sectors from trades."""
        sector_counts = {}
        for t in trades:
            sector = t.get('sector', 'Unknown')
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        # Sort by count, return top 3
        sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)
        return [s[0] for s in sorted_sectors[:3]]


class LoserPatternDetector:
    """
    Detect and avoid patterns that lead to losses like 000981.SZ.

    Based on PDF analysis showing common loser patterns:
    - High volatility + downtrend (vol > 35%, mom_20d < -3%)
    - Low volume breakdown (vol_ratio < 0.8, price < MA20)
    - Sector downtrend (sector_momentum < -8%)
    """

    def __init__(self):
        self.loser_patterns = {
            'high_vol_downtrend': {
                'conditions': {'volatility_min': 0.35, 'momentum_20d_max': -0.03},
                'risk_level': 'HIGH'
            },
            'low_volume_breakdown': {
                'conditions': {'volume_ratio_max': 0.8, 'price_vs_ma20_max': -0.05},
                'risk_level': 'MEDIUM'
            },
            'sector_downtrend': {
                'conditions': {'sector_momentum_max': -0.08, 'relative_strength_max': -0.05},
                'risk_level': 'HIGH'
            }
        }

    def detect_loser_risk(self, ticker_data: Dict) -> list:
        """
        Detect high-risk patterns before entering trades.

        Args:
            ticker_data: Dict with volatility, momentum, volume_ratio, etc.

        Returns:
            List of detected risk signals
        """
        risk_signals = []

        volatility = ticker_data.get('volatility', 0)
        momentum_20d = ticker_data.get('momentum_20d', 0)
        volume_ratio = ticker_data.get('volume_ratio', 1)
        price_vs_ma20 = ticker_data.get('dist_from_ma', 0)
        sector_momentum = ticker_data.get('sector_momentum', 0)
        relative_strength = ticker_data.get('relative_strength', 0)

        # Check high_vol_downtrend pattern
        if volatility > 0.35 and momentum_20d < -0.03:
            risk_signals.append({
                'pattern': 'high_vol_downtrend',
                'risk_level': 'HIGH',
                'details': f'Vol: {volatility:.1%}, Mom20d: {momentum_20d:.1%}',
                'suggested_action': 'REDUCE_SIZE'
            })

        # Check low_volume_breakdown pattern
        if volume_ratio < 0.8 and price_vs_ma20 < -0.05:
            risk_signals.append({
                'pattern': 'low_volume_breakdown',
                'risk_level': 'MEDIUM',
                'details': f'VolRatio: {volume_ratio:.1f}x, PriceVsMA: {price_vs_ma20:.1%}',
                'suggested_action': 'MONITOR_CLOSELY'
            })

        # Check sector_downtrend pattern
        if sector_momentum < -0.08 and relative_strength < -0.05:
            risk_signals.append({
                'pattern': 'sector_downtrend',
                'risk_level': 'HIGH',
                'details': f'SectorMom: {sector_momentum:.1%}, RelStr: {relative_strength:.1%}',
                'suggested_action': 'REDUCE_SIZE'
            })

        return risk_signals

    def analyze_historical_losers(self, losing_trades: list) -> Dict:
        """
        Analyze common characteristics of losing trades.

        Args:
            losing_trades: List of losing trade dicts

        Returns:
            Dict with common loser characteristics
        """
        if not losing_trades:
            return {}

        characteristics = {
            'avg_volatility': np.mean([t.get('volatility', 0) for t in losing_trades]),
            'avg_momentum': np.mean([t.get('momentum_20d', 0) for t in losing_trades]),
            'sector_concentration': self._analyze_sector_concentration(losing_trades),
            'avg_volume_ratio': np.mean([t.get('volume_ratio', 1) for t in losing_trades]),
            'avg_loss': np.mean([t.get('return', t.get('actual_ret', 0)) for t in losing_trades])
        }

        return characteristics

    def _analyze_sector_concentration(self, trades: list) -> Dict:
        """Analyze sector concentration in trades."""
        sector_counts = {}
        for t in trades:
            sector = t.get('sector', 'Unknown')
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        total = len(trades)
        return {sector: count/total for sector, count in sector_counts.items()}


class DynamicSizingOptimizer:
    """
    Adjust sizing based on real-time performance feedback.

    Based on PDF recommendations:
    - High performers (60%+ win rate, 8%+ avg return): 1.3x size
    - Medium performers (40%+ win rate, 3%+ avg return): 1.0x size
    - Low performers (<40% win rate or negative avg return): 0.5x size
    """

    def __init__(self):
        self.performance_buckets = {
            'high_performer': {'min_win_rate': 0.6, 'min_avg_return': 0.08, 'size_multiplier': 1.3},
            'medium_performer': {'min_win_rate': 0.4, 'min_avg_return': 0.03, 'size_multiplier': 1.0},
            'low_performer': {'min_win_rate': 0.0, 'min_avg_return': -1.0, 'size_multiplier': 0.5}
        }

        # Track per-ticker performance
        self.ticker_performance = {}

    def record_trade_result(self, ticker: str, return_pct: float, was_winner: bool):
        """Record a trade result for future sizing adjustments."""
        if ticker not in self.ticker_performance:
            self.ticker_performance[ticker] = {'returns': [], 'wins': 0, 'total': 0}

        self.ticker_performance[ticker]['returns'].append(return_pct)
        self.ticker_performance[ticker]['total'] += 1
        if was_winner:
            self.ticker_performance[ticker]['wins'] += 1

    def get_performance_adjusted_size(self, ticker: str, historical_performance: Dict,
                                      base_size: float) -> float:
        """
        Adjust size based on this ticker's historical performance.

        Args:
            ticker: Stock ticker
            historical_performance: Dict with win_rate, avg_return
            base_size: Base position size (e.g., 0.05)

        Returns:
            Adjusted position size
        """
        performance_profile = self._classify_performance(historical_performance)
        multiplier = self.performance_buckets[performance_profile]['size_multiplier']

        return base_size * multiplier

    def _classify_performance(self, performance_data: Dict) -> str:
        """Classify ticker performance level."""
        win_rate = performance_data.get('win_rate', 0)
        avg_return = performance_data.get('avg_return', 0)

        if win_rate >= 0.6 and avg_return >= 0.08:
            return 'high_performer'
        elif win_rate >= 0.4 and avg_return >= 0.03:
            return 'medium_performer'
        else:
            return 'low_performer'

    def get_ticker_performance(self, ticker: str) -> Dict:
        """Get historical performance for a ticker."""
        if ticker not in self.ticker_performance:
            return {'win_rate': 0.5, 'avg_return': 0, 'trades': 0}

        perf = self.ticker_performance[ticker]
        returns = perf['returns']

        return {
            'win_rate': perf['wins'] / perf['total'] if perf['total'] > 0 else 0.5,
            'avg_return': np.mean(returns) if returns else 0,
            'trades': perf['total']
        }


class ChinaBuyOptimizer:
    """
    Complete optimization system for China BUY signals.

    Integrates all PDF recommendations:
    1. Stop-loss enforcement (6% max loss)
    2. Winner sizing optimization (up to 50% increase)
    3. Loser pattern detection (block high-risk patterns)
    4. Dynamic sizing based on performance
    """

    def __init__(self):
        self.stop_loss_enforcer = StopLossEnforcer(max_loss_pct=0.06)
        self.winner_optimizer = WinnerOptimizer()
        self.loser_detector = LoserPatternDetector()
        self.sizing_optimizer = DynamicSizingOptimizer()

    def optimize_buy_signal(self, signal_data: Dict, ticker_data: Dict,
                           historical_performance: Dict = None) -> Dict:
        """
        Apply all optimizations to a BUY signal.

        Args:
            signal_data: Dict with ticker, confidence, position_size, buy_quality
            ticker_data: Dict with volatility, momentum, volume_ratio, etc.
            historical_performance: Optional dict with win_rate, avg_return

        Returns:
            Dict with optimized signal parameters
        """
        ticker = signal_data.get('ticker', 'Unknown')
        base_size = signal_data.get('position_size', 0.05)
        buy_quality = signal_data.get('buy_quality', 'cautious_buy')

        # Step 1: Loser pattern detection - skip if high risk
        risk_signals = self.loser_detector.detect_loser_risk(ticker_data)
        high_risk_count = sum(1 for r in risk_signals if r['risk_level'] == 'HIGH')

        if high_risk_count >= 2:
            # Too many high-risk signals - block trade
            return {
                'ticker': ticker,
                'action': 'BLOCKED',
                'reason': f'High-risk patterns detected: {[r["pattern"] for r in risk_signals]}',
                'optimized_size': 0,
                'risk_signals': risk_signals
            }

        # Step 2: Apply winner optimization
        signal_data_copy = signal_data.copy()
        signal_data_copy['buy_quality'] = buy_quality
        optimized_size = self.winner_optimizer.optimize_winner_sizing(signal_data_copy)

        # Step 3: Apply performance-based sizing (if historical data available)
        if historical_performance:
            perf_adjusted_size = self.sizing_optimizer.get_performance_adjusted_size(
                ticker, historical_performance, optimized_size
            )
            optimized_size = perf_adjusted_size

        # Step 4: Apply risk-based reduction if medium risk detected
        if high_risk_count == 1 or len(risk_signals) > 0:
            optimized_size *= 0.7  # 30% reduction for elevated risk

        # Step 5: Calculate stop-loss
        volatility = ticker_data.get('volatility', 0.25)
        stop_loss_mgr = BuyStopLossManager()
        stop_loss_pct = stop_loss_mgr.calculate_stop_loss(buy_quality, volatility)
        max_hold_days = stop_loss_mgr.get_max_hold_days(buy_quality)

        return {
            'ticker': ticker,
            'action': 'OPTIMIZED',
            'original_size': base_size,
            'optimized_size': optimized_size,
            'size_change': (optimized_size - base_size) / base_size if base_size > 0 else 0,
            'stop_loss_pct': stop_loss_pct,
            'max_hold_days': max_hold_days,
            'risk_signals': risk_signals,
            'buy_quality': buy_quality,
            'optimization_applied': True
        }

    def simulate_optimized_pnl(self, trades: list) -> Dict:
        """
        Simulate P&L with all optimizations applied.

        Args:
            trades: List of trade dicts with ticker, return, position_size, etc.

        Returns:
            Dict with original_pnl, optimized_pnl, improvement
        """
        original_pnl = 0
        optimized_pnl = 0
        blocked_trades = []
        stopped_out_trades = []

        for trade in trades:
            actual_return = trade.get('return', trade.get('actual_ret', 0))
            original_size = trade.get('position_size', 0.05)

            # Original P&L
            original_pnl += actual_return * original_size

            # Build ticker_data for optimization
            ticker_data = {
                'volatility': trade.get('volatility', trade.get('vol', 0.25)),
                'momentum_20d': trade.get('momentum_20d', trade.get('mom_20d', 0)),
                'volume_ratio': trade.get('volume_ratio', trade.get('vol_ratio', 1)),
                'dist_from_ma': trade.get('dist_from_ma', trade.get('dist_ma', 0))
            }

            # Optimize the signal
            signal_data = {
                'ticker': trade.get('ticker', 'Unknown'),
                'position_size': original_size,
                'buy_quality': trade.get('buy_quality', 'cautious_buy')
            }

            result = self.optimize_buy_signal(signal_data, ticker_data)

            if result['action'] == 'BLOCKED':
                blocked_trades.append(trade)
                # Blocked trades contribute 0 to optimized P&L
                continue

            optimized_size = result['optimized_size']
            stop_loss = result['stop_loss_pct']

            # Check if stop-loss would have triggered
            if actual_return < -stop_loss:
                # Capped loss
                optimized_pnl += -stop_loss * optimized_size
                stopped_out_trades.append(trade)
            else:
                optimized_pnl += actual_return * optimized_size

        improvement = optimized_pnl - original_pnl

        return {
            'original_pnl': original_pnl,
            'optimized_pnl': optimized_pnl,
            'improvement': improvement,
            'improvement_pct': improvement / abs(original_pnl) if original_pnl != 0 else 0,
            'trades_blocked': len(blocked_trades),
            'trades_stopped_out': len(stopped_out_trades),
            'blocked_trades': blocked_trades,
            'stopped_out_trades': stopped_out_trades
        }


# =============================================================================
# ADVANCED OPTIMIZATION CLASSES (from buy fix3 PDF recommendations)
# =============================================================================

class RiskAdjustedSizer:
    """
    More aggressive risk-based position sizing.

    Based on PDF buy fix3 recommendations:
    - HIGH_VOL_DOWNSIDE: 0.3x multiplier (very aggressive reduction)
    - LOW_VOLUME_BREAKDOWN: 0.4x multiplier
    - HIGH_VOL_UPSIDE: 1.2x multiplier (size up for vol + uptrend)
    - Default: 1.0x
    """

    def __init__(self):
        self.risk_multipliers = {
            'HIGH_VOL_DOWNSIDE': 0.3,      # Very aggressive - 70% reduction
            'LOW_VOLUME_BREAKDOWN': 0.4,   # Aggressive - 60% reduction
            'HIGH_VOL_UPSIDE': 1.2,        # Size up for momentum
            'SECTOR_WEAKNESS': 0.5,        # 50% reduction
            'NORMAL': 1.0                   # No adjustment
        }

    def classify_risk_pattern(self, ticker_data: Dict) -> str:
        """
        Classify the risk pattern for position sizing.

        Args:
            ticker_data: Dict with volatility, momentum, volume_ratio, etc.

        Returns:
            Risk pattern name
        """
        volatility = ticker_data.get('volatility', 0.25)
        momentum_20d = ticker_data.get('momentum_20d', 0)
        momentum_5d = ticker_data.get('momentum_5d', 0)
        volume_ratio = ticker_data.get('volume_ratio', 1.0)
        dist_from_ma = ticker_data.get('dist_from_ma', 0)
        sector = ticker_data.get('sector', '')

        # Pattern 1: HIGH_VOL_DOWNSIDE (most dangerous)
        # High volatility + downward momentum = falling knife
        if volatility > 0.35 and momentum_20d < -0.03:
            return 'HIGH_VOL_DOWNSIDE'

        # Pattern 2: LOW_VOLUME_BREAKDOWN
        # Price below MA with low volume = lack of buying interest
        if volume_ratio < 0.8 and dist_from_ma < -0.05:
            return 'LOW_VOLUME_BREAKDOWN'

        # Pattern 3: HIGH_VOL_UPSIDE (positive - size up!)
        # High volatility + upward momentum = momentum play
        if volatility > 0.30 and momentum_20d > 0.05 and momentum_5d > 0.03:
            return 'HIGH_VOL_UPSIDE'

        # Pattern 4: SECTOR_WEAKNESS
        # Specific weak sectors (Green Energy, Real Estate during downturns)
        weak_sectors = ['Green Energy', 'Real Estate']
        if sector in weak_sectors and momentum_20d < 0:
            return 'SECTOR_WEAKNESS'

        return 'NORMAL'

    def get_size_multiplier(self, ticker_data: Dict) -> Tuple[float, str]:
        """
        Get position size multiplier based on risk pattern.

        Args:
            ticker_data: Dict with volatility, momentum, etc.

        Returns:
            (multiplier, pattern_name)
        """
        pattern = self.classify_risk_pattern(ticker_data)
        multiplier = self.risk_multipliers.get(pattern, 1.0)
        return multiplier, pattern

    def calculate_risk_adjusted_size(self, base_size: float, ticker_data: Dict) -> Dict:
        """
        Calculate risk-adjusted position size.

        Args:
            base_size: Base position size (e.g., 0.05 = 5%)
            ticker_data: Dict with volatility, momentum, etc.

        Returns:
            Dict with adjusted_size, pattern, multiplier
        """
        multiplier, pattern = self.get_size_multiplier(ticker_data)
        adjusted_size = base_size * multiplier

        # Enforce min/max bounds
        adjusted_size = max(0.01, min(adjusted_size, 0.10))

        return {
            'adjusted_size': adjusted_size,
            'pattern': pattern,
            'multiplier': multiplier,
            'base_size': base_size,
            'size_change_pct': (adjusted_size - base_size) / base_size * 100 if base_size > 0 else 0
        }


class PatternBasedFilter:
    """
    Hard rejection filter for known losing patterns.

    Based on PDF buy fix3 analysis:
    - SENSTEED_PATTERN: vol > 40%, mom < -5%, Industrial/Tech sectors
    - GCL_PATTERN: Green Energy + high vol + negative momentum
    - FALLING_KNIFE: Strong downtrend with worsening momentum
    """

    def __init__(self):
        # Define rejection patterns
        self.rejection_patterns = {
            'SENSTEED_PATTERN': {
                'description': 'High volatility + strong downtrend in Industrial/Tech',
                'conditions': {
                    'volatility_min': 0.40,
                    'momentum_20d_max': -0.05,
                    'sectors': ['Industrial', 'Technology', 'Manufacturing']
                }
            },
            'GCL_PATTERN': {
                'description': 'Green Energy sector weakness',
                'conditions': {
                    'volatility_min': 0.35,
                    'momentum_20d_max': -0.03,
                    'sectors': ['Green Energy', 'Solar', 'Renewable']
                }
            },
            'FALLING_KNIFE': {
                'description': 'Accelerating downtrend',
                'conditions': {
                    'momentum_20d_max': -0.08,
                    'momentum_5d_worsening': True  # 5d < 20d
                }
            },
            'VOLUME_COLLAPSE': {
                'description': 'Price drop with volume collapse',
                'conditions': {
                    'volume_ratio_max': 0.5,
                    'momentum_20d_max': -0.05,
                    'dist_from_ma_max': -0.10
                }
            }
        }

    def should_reject(self, ticker_data: Dict) -> Tuple[bool, str]:
        """
        Check if ticker matches any rejection pattern.

        Args:
            ticker_data: Dict with volatility, momentum, volume_ratio, sector, etc.

        Returns:
            (should_reject, pattern_name or reason)
        """
        volatility = ticker_data.get('volatility', 0.25)
        momentum_20d = ticker_data.get('momentum_20d', 0)
        momentum_5d = ticker_data.get('momentum_5d', 0)
        volume_ratio = ticker_data.get('volume_ratio', 1.0)
        dist_from_ma = ticker_data.get('dist_from_ma', 0)
        sector = ticker_data.get('sector', '')

        # Check SENSTEED_PATTERN
        sensteed_conds = self.rejection_patterns['SENSTEED_PATTERN']['conditions']
        if (volatility >= sensteed_conds['volatility_min'] and
            momentum_20d <= sensteed_conds['momentum_20d_max'] and
            sector in sensteed_conds['sectors']):
            return True, 'SENSTEED_PATTERN: High vol downtrend in sensitive sector'

        # Check GCL_PATTERN
        gcl_conds = self.rejection_patterns['GCL_PATTERN']['conditions']
        if (volatility >= gcl_conds['volatility_min'] and
            momentum_20d <= gcl_conds['momentum_20d_max'] and
            sector in gcl_conds['sectors']):
            return True, 'GCL_PATTERN: Green Energy sector weakness'

        # Check FALLING_KNIFE
        fk_conds = self.rejection_patterns['FALLING_KNIFE']['conditions']
        is_worsening = momentum_5d < momentum_20d
        if (momentum_20d <= fk_conds['momentum_20d_max'] and is_worsening):
            return True, f'FALLING_KNIFE: Accelerating downtrend (mom_20d={momentum_20d:.1%})'

        # Check VOLUME_COLLAPSE
        vc_conds = self.rejection_patterns['VOLUME_COLLAPSE']['conditions']
        if (volume_ratio <= vc_conds['volume_ratio_max'] and
            momentum_20d <= vc_conds['momentum_20d_max'] and
            dist_from_ma <= vc_conds['dist_from_ma_max']):
            return True, 'VOLUME_COLLAPSE: No buying interest'

        return False, 'PASSED'

    def get_pattern_details(self, ticker_data: Dict) -> Dict:
        """
        Get detailed pattern analysis for a ticker.

        Returns:
            Dict with pattern matches and risk assessment
        """
        should_reject, pattern = self.should_reject(ticker_data)

        return {
            'should_reject': should_reject,
            'matched_pattern': pattern if should_reject else None,
            'volatility': ticker_data.get('volatility', 0),
            'momentum_20d': ticker_data.get('momentum_20d', 0),
            'momentum_5d': ticker_data.get('momentum_5d', 0),
            'sector': ticker_data.get('sector', ''),
            'risk_level': 'CRITICAL' if should_reject else 'ACCEPTABLE'
        }


class VolatilityAdjustedStopLoss:
    """
    Dynamic stop-loss based on volatility brackets.

    Based on PDF buy fix3 recommendations:
    - 0-15% vol: 7.8% stop (wider for low vol stocks)
    - 15-25% vol: 6.0% stop (standard)
    - 25-35% vol: 4.8% stop (tighter for elevated vol)
    - 35%+ vol: 3.6% stop (very tight for high vol)

    Rationale: High vol stocks move more, so tighter stops prevent large losses.
    """

    def __init__(self):
        # Volatility brackets and corresponding stop-loss percentages
        self.volatility_brackets = [
            {'vol_min': 0.00, 'vol_max': 0.15, 'stop_loss': 0.078},  # 7.8%
            {'vol_min': 0.15, 'vol_max': 0.25, 'stop_loss': 0.060},  # 6.0%
            {'vol_min': 0.25, 'vol_max': 0.35, 'stop_loss': 0.048},  # 4.8%
            {'vol_min': 0.35, 'vol_max': 1.00, 'stop_loss': 0.036},  # 3.6%
        ]

        # Quality-based adjustments
        self.quality_adjustments = {
            'strong_buy': 1.10,    # 10% wider stops for strong trends
            'cautious_buy': 1.00,  # Standard
            'recovery_buy': 0.85,  # 15% tighter for recovery plays
        }

    def calculate_stop_loss(self, volatility: float, buy_quality: str = 'cautious_buy') -> float:
        """
        Calculate dynamic stop-loss based on volatility.

        Args:
            volatility: Annualized volatility (e.g., 0.35 = 35%)
            buy_quality: 'strong_buy', 'cautious_buy', or 'recovery_buy'

        Returns:
            Stop-loss percentage (e.g., 0.06 = 6%)
        """
        # Find the appropriate bracket
        base_stop = 0.06  # Default 6%

        for bracket in self.volatility_brackets:
            if bracket['vol_min'] <= volatility < bracket['vol_max']:
                base_stop = bracket['stop_loss']
                break

        # Apply quality adjustment
        quality_mult = self.quality_adjustments.get(buy_quality, 1.0)
        adjusted_stop = base_stop * quality_mult

        # Enforce bounds (3% - 10%)
        return max(0.03, min(adjusted_stop, 0.10))

    def get_stop_details(self, volatility: float, buy_quality: str = 'cautious_buy') -> Dict:
        """
        Get detailed stop-loss calculation breakdown.

        Returns:
            Dict with stop_loss, bracket, adjustments
        """
        # Find bracket
        bracket_info = None
        for bracket in self.volatility_brackets:
            if bracket['vol_min'] <= volatility < bracket['vol_max']:
                bracket_info = bracket
                break

        base_stop = bracket_info['stop_loss'] if bracket_info else 0.06
        quality_mult = self.quality_adjustments.get(buy_quality, 1.0)
        final_stop = self.calculate_stop_loss(volatility, buy_quality)

        return {
            'volatility': volatility,
            'volatility_bracket': f"{bracket_info['vol_min']*100:.0f}%-{bracket_info['vol_max']*100:.0f}%" if bracket_info else 'default',
            'base_stop_loss': base_stop,
            'buy_quality': buy_quality,
            'quality_multiplier': quality_mult,
            'final_stop_loss': final_stop,
            'max_loss_per_unit': final_stop
        }


class PerformanceWeightedScorer:
    """
    Score signals based on historical pattern performance.

    Based on PDF buy fix3 analysis showing:
    - HIGH_VOL_UPSIDE patterns: +12% avg return (weight: 1.3)
    - BREAKOUT patterns: +8% avg return (weight: 1.15)
    - NORMAL patterns: +3% avg return (weight: 1.0)
    - RECOVERY patterns: -2% avg return (weight: 0.7)
    """

    def __init__(self):
        # Pattern weights based on historical performance
        self.pattern_weights = {
            'HIGH_VOL_UPSIDE': 1.30,     # Strong performers
            'BREAKOUT': 1.15,             # Good performers
            'UPTREND_CONFIRMED': 1.10,    # Above average
            'NORMAL': 1.00,               # Baseline
            'CAUTIOUS': 0.85,             # Below average
            'RECOVERY': 0.70,             # Weak performers
            'HIGH_VOL_DOWNSIDE': 0.40,    # Poor performers (if not rejected)
        }

        # Track pattern performance for adaptive weighting
        self.pattern_performance = {}

    def classify_pattern(self, ticker_data: Dict) -> str:
        """
        Classify ticker into a performance pattern.

        Args:
            ticker_data: Dict with volatility, momentum, volume_ratio, etc.

        Returns:
            Pattern name
        """
        volatility = ticker_data.get('volatility', 0.25)
        momentum_20d = ticker_data.get('momentum_20d', 0)
        momentum_5d = ticker_data.get('momentum_5d', 0)
        volume_ratio = ticker_data.get('volume_ratio', 1.0)
        dist_from_ma = ticker_data.get('dist_from_ma', 0)

        # HIGH_VOL_UPSIDE: Volatile with positive momentum
        if volatility > 0.30 and momentum_20d > 0.05 and momentum_5d > 0.03:
            return 'HIGH_VOL_UPSIDE'

        # BREAKOUT: Price above MA with volume surge
        if dist_from_ma > 0.03 and volume_ratio > 1.5 and momentum_5d > 0.02:
            return 'BREAKOUT'

        # UPTREND_CONFIRMED: All momentum positive, above MA
        if momentum_20d > 0.02 and momentum_5d > 0.01 and dist_from_ma > 0:
            return 'UPTREND_CONFIRMED'

        # HIGH_VOL_DOWNSIDE: Volatile with negative momentum
        if volatility > 0.35 and momentum_20d < -0.03:
            return 'HIGH_VOL_DOWNSIDE'

        # RECOVERY: Improving momentum from negative
        if momentum_5d > momentum_20d and momentum_20d < 0 and momentum_5d > 0:
            return 'RECOVERY'

        # CAUTIOUS: Mixed signals
        if momentum_20d < 0 or dist_from_ma < -0.03:
            return 'CAUTIOUS'

        return 'NORMAL'

    def score_signal(self, ticker_data: Dict, base_confidence: float) -> Dict:
        """
        Score a signal based on pattern and confidence.

        Args:
            ticker_data: Dict with volatility, momentum, etc.
            base_confidence: Original model confidence (0-1)

        Returns:
            Dict with pattern, weight, adjusted_confidence
        """
        pattern = self.classify_pattern(ticker_data)
        weight = self.pattern_weights.get(pattern, 1.0)

        # Apply weight to confidence
        adjusted_confidence = base_confidence * weight
        adjusted_confidence = np.clip(adjusted_confidence, 0.1, 0.95)

        return {
            'pattern': pattern,
            'weight': weight,
            'base_confidence': base_confidence,
            'adjusted_confidence': adjusted_confidence,
            'confidence_change': (adjusted_confidence - base_confidence) / base_confidence * 100 if base_confidence > 0 else 0
        }

    def record_pattern_performance(self, pattern: str, actual_return: float):
        """
        Record actual performance for adaptive weighting.

        Args:
            pattern: Pattern name
            actual_return: Actual return achieved
        """
        if pattern not in self.pattern_performance:
            self.pattern_performance[pattern] = {'returns': [], 'count': 0}

        self.pattern_performance[pattern]['returns'].append(actual_return)
        self.pattern_performance[pattern]['count'] += 1

    def get_pattern_stats(self) -> Dict:
        """Get performance statistics for each pattern."""
        stats = {}
        for pattern, data in self.pattern_performance.items():
            returns = data['returns']
            if returns:
                stats[pattern] = {
                    'count': data['count'],
                    'avg_return': np.mean(returns),
                    'win_rate': sum(1 for r in returns if r > 0) / len(returns),
                    'current_weight': self.pattern_weights.get(pattern, 1.0)
                }
        return stats


class EnhancedChinaBuyOptimizer:
    """
    Complete ENHANCED optimization system for China BUY signals.

    Integrates all PDF buy fix3 recommendations:
    1. RiskAdjustedSizer - Aggressive position sizing based on risk patterns
    2. PatternBasedFilter - Hard rejection of losing patterns
    3. VolatilityAdjustedStopLoss - Dynamic stops by volatility
    4. PerformanceWeightedScorer - Pattern-based confidence adjustment
    5. Stop-loss enforcement (6% max base, adjusted by volatility)

    This replaces the basic ChinaBuyOptimizer with more aggressive optimizations.
    """

    def __init__(self):
        # New advanced components from PDF
        self.risk_sizer = RiskAdjustedSizer()
        self.pattern_filter = PatternBasedFilter()
        self.vol_stop_loss = VolatilityAdjustedStopLoss()
        self.pattern_scorer = PerformanceWeightedScorer()

        # Legacy components (still useful)
        self.stop_loss_enforcer = StopLossEnforcer(max_loss_pct=0.06)
        self.winner_optimizer = WinnerOptimizer()
        self.loser_detector = LoserPatternDetector()
        self.sizing_optimizer = DynamicSizingOptimizer()

    def optimize_buy_signal(self, signal_data: Dict, ticker_data: Dict,
                           historical_performance: Dict = None) -> Dict:
        """
        Apply ALL optimizations to a BUY signal.

        This is the main entry point for the enhanced optimizer.

        Args:
            signal_data: Dict with ticker, confidence, position_size, buy_quality
            ticker_data: Dict with volatility, momentum, volume_ratio, sector, etc.
            historical_performance: Optional dict with win_rate, avg_return

        Returns:
            Dict with fully optimized signal parameters
        """
        ticker = signal_data.get('ticker', 'Unknown')
        base_confidence = signal_data.get('confidence', 0.5)
        base_size = signal_data.get('position_size', 0.05)
        buy_quality = signal_data.get('buy_quality', 'cautious_buy')

        # =====================================================================
        # STEP 1: Pattern-based rejection (HARD FILTER)
        # =====================================================================
        should_reject, rejection_reason = self.pattern_filter.should_reject(ticker_data)

        if should_reject:
            return {
                'ticker': ticker,
                'action': 'BLOCKED',
                'reason': rejection_reason,
                'optimized_size': 0,
                'optimized_confidence': 0,
                'stop_loss_pct': 0,
                'blocked_by': 'PatternBasedFilter',
                'optimization_applied': True
            }

        # =====================================================================
        # STEP 2: Legacy loser pattern detection (additional safety)
        # =====================================================================
        risk_signals = self.loser_detector.detect_loser_risk(ticker_data)
        high_risk_count = sum(1 for r in risk_signals if r['risk_level'] == 'HIGH')

        if high_risk_count >= 2:
            return {
                'ticker': ticker,
                'action': 'BLOCKED',
                'reason': f'Multiple high-risk patterns: {[r["pattern"] for r in risk_signals]}',
                'optimized_size': 0,
                'optimized_confidence': 0,
                'stop_loss_pct': 0,
                'risk_signals': risk_signals,
                'blocked_by': 'LoserPatternDetector',
                'optimization_applied': True
            }

        # =====================================================================
        # STEP 3: Performance-weighted scoring (adjust confidence)
        # =====================================================================
        score_result = self.pattern_scorer.score_signal(ticker_data, base_confidence)
        adjusted_confidence = score_result['adjusted_confidence']
        pattern = score_result['pattern']

        # =====================================================================
        # STEP 4: Risk-adjusted sizing (aggressive multipliers)
        # =====================================================================
        size_result = self.risk_sizer.calculate_risk_adjusted_size(base_size, ticker_data)
        risk_adjusted_size = size_result['adjusted_size']
        risk_pattern = size_result['pattern']

        # =====================================================================
        # STEP 5: Apply winner optimization (for strong patterns)
        # =====================================================================
        signal_data_copy = signal_data.copy()
        signal_data_copy['buy_quality'] = buy_quality
        signal_data_copy['position_size'] = risk_adjusted_size
        winner_size = self.winner_optimizer.optimize_winner_sizing(signal_data_copy)

        # Take the more conservative of risk_adjusted and winner_optimized
        optimized_size = min(risk_adjusted_size, winner_size)

        # =====================================================================
        # STEP 6: Performance-based adjustment (if historical data available)
        # =====================================================================
        if historical_performance:
            perf_adjusted_size = self.sizing_optimizer.get_performance_adjusted_size(
                ticker, historical_performance, optimized_size
            )
            optimized_size = perf_adjusted_size

        # =====================================================================
        # STEP 7: Risk signal reduction (if any medium risk detected)
        # =====================================================================
        if high_risk_count == 1 or len(risk_signals) > 0:
            optimized_size *= 0.7  # 30% reduction for elevated risk

        # =====================================================================
        # STEP 8: Volatility-adjusted stop-loss (dynamic by vol bracket)
        # =====================================================================
        volatility = ticker_data.get('volatility', 0.25)
        stop_loss_result = self.vol_stop_loss.get_stop_details(volatility, buy_quality)
        stop_loss_pct = stop_loss_result['final_stop_loss']

        # =====================================================================
        # STEP 9: Calculate max hold days based on quality
        # =====================================================================
        max_hold_days = {
            'strong_buy': 15,
            'cautious_buy': 10,
            'recovery_buy': 7
        }.get(buy_quality, 10)

        # =====================================================================
        # Build comprehensive result
        # =====================================================================
        return {
            'ticker': ticker,
            'action': 'OPTIMIZED',

            # Original values
            'original_size': base_size,
            'original_confidence': base_confidence,

            # Optimized values
            'optimized_size': optimized_size,
            'optimized_confidence': adjusted_confidence,

            # Size breakdown
            'size_change_pct': (optimized_size - base_size) / base_size * 100 if base_size > 0 else 0,
            'risk_pattern': risk_pattern,
            'risk_multiplier': size_result['multiplier'],

            # Confidence breakdown
            'pattern': pattern,
            'pattern_weight': score_result['weight'],
            'confidence_change_pct': score_result['confidence_change'],

            # Stop-loss details
            'stop_loss_pct': stop_loss_pct,
            'stop_loss_bracket': stop_loss_result['volatility_bracket'],

            # Risk management
            'max_hold_days': max_hold_days,
            'risk_signals': risk_signals,
            'buy_quality': buy_quality,

            # Metadata
            'optimization_applied': True,
            'blocked_by': None,
            'volatility': volatility
        }

    def simulate_enhanced_pnl(self, trades: list) -> Dict:
        """
        Simulate P&L with ALL enhanced optimizations applied.

        Args:
            trades: List of trade dicts with ticker, return, position_size, etc.

        Returns:
            Dict with original_pnl, optimized_pnl, improvement, details
        """
        original_pnl = 0
        optimized_pnl = 0
        blocked_trades = []
        stopped_out_trades = []

        for trade in trades:
            actual_return = trade.get('return', trade.get('actual_ret', 0))
            original_size = trade.get('position_size', 0.05)

            # Original P&L
            original_pnl += actual_return * original_size

            # Build ticker_data for optimization
            ticker_data = {
                'volatility': trade.get('volatility', trade.get('vol', 0.25)),
                'momentum_20d': trade.get('momentum_20d', trade.get('mom_20d', 0)),
                'momentum_5d': trade.get('momentum_5d', trade.get('mom_5d', 0)),
                'volume_ratio': trade.get('volume_ratio', trade.get('vol_ratio', 1)),
                'dist_from_ma': trade.get('dist_from_ma', trade.get('dist_ma', 0)),
                'sector': trade.get('sector', '')
            }

            # Build signal_data
            signal_data = {
                'ticker': trade.get('ticker', 'Unknown'),
                'confidence': trade.get('confidence', 0.6),
                'position_size': original_size,
                'buy_quality': trade.get('buy_quality', 'cautious_buy')
            }

            # Run through enhanced optimizer
            result = self.optimize_buy_signal(signal_data, ticker_data)

            if result['action'] == 'BLOCKED':
                blocked_trades.append({
                    'ticker': trade.get('ticker'),
                    'reason': result['reason'],
                    'would_have_returned': actual_return,
                    'loss_avoided': actual_return * original_size if actual_return < 0 else 0
                })
                continue

            optimized_size = result['optimized_size']
            stop_loss = result['stop_loss_pct']

            # Check if stop-loss would have triggered
            if actual_return < -stop_loss:
                # Capped loss
                optimized_pnl += -stop_loss * optimized_size
                stopped_out_trades.append({
                    'ticker': trade.get('ticker'),
                    'actual_return': actual_return,
                    'capped_at': -stop_loss,
                    'saved': (abs(actual_return) - stop_loss) * optimized_size
                })
            else:
                optimized_pnl += actual_return * optimized_size

        improvement = optimized_pnl - original_pnl

        return {
            'original_pnl': original_pnl,
            'optimized_pnl': optimized_pnl,
            'improvement': improvement,
            'improvement_pct': improvement / abs(original_pnl) * 100 if original_pnl != 0 else 0,
            'trades_blocked': len(blocked_trades),
            'trades_stopped_out': len(stopped_out_trades),
            'blocked_trades': blocked_trades,
            'stopped_out_trades': stopped_out_trades,
            'total_loss_avoided': sum(t.get('loss_avoided', 0) for t in blocked_trades),
            'total_stop_loss_saved': sum(t.get('saved', 0) for t in stopped_out_trades)
        }

    def get_optimization_summary(self) -> Dict:
        """Get summary of optimization components and their status."""
        return {
            'components': {
                'RiskAdjustedSizer': 'Active - Aggressive position sizing based on risk patterns',
                'PatternBasedFilter': 'Active - Hard rejection of SENSTEED, GCL, FALLING_KNIFE patterns',
                'VolatilityAdjustedStopLoss': 'Active - Dynamic stops by volatility bracket',
                'PerformanceWeightedScorer': 'Active - Pattern-based confidence adjustment',
                'StopLossEnforcer': 'Active - 6% max loss enforcement',
                'WinnerOptimizer': 'Active - Size up for strong patterns',
                'LoserPatternDetector': 'Active - Additional risk detection',
                'DynamicSizingOptimizer': 'Active - Performance-based sizing'
            },
            'risk_multipliers': self.risk_sizer.risk_multipliers,
            'pattern_weights': self.pattern_scorer.pattern_weights,
            'volatility_brackets': [
                {'range': '0-15%', 'stop_loss': '7.8%'},
                {'range': '15-25%', 'stop_loss': '6.0%'},
                {'range': '25-35%', 'stop_loss': '4.8%'},
                {'range': '35%+', 'stop_loss': '3.6%'}
            ]
        }


# =============================================================================
# PRODUCTION DEPLOYMENT - From PDF "buy fix4 on C model"
# Production-ready system with validated +1.9% improvement
# =============================================================================

# Production Configuration
PRODUCTION_CONFIG = {
    'system_parameters': {
        'min_confidence': 0.65,
        'max_daily_signals': 8,
        'max_portfolio_allocation': 0.25,
        'auto_rebalance': True
    },
    'risk_parameters': {
        'high_vol_downsize_multiplier': 0.3,
        'low_volume_downsize_multiplier': 0.4,
        'high_vol_upsize_multiplier': 1.2,
        'base_stop_loss': 0.06
    },
    'performance_targets': {
        'min_improvement': '+1.5%',  # Conservative target
        'max_rejection_rate': 0.6,   # Allow up to 60% rejection
        'risk_adjustment_effectiveness': 'High'
    },
    'monitoring': {
        'performance_checks': 'hourly',
        'alert_threshold': -0.005,
        'auto_adjustment': True
    }
}


class ProductionChinaBuyOptimizer:
    """
    Production-ready China BUY optimizer with validated +1.9% improvement.

    This is the FINAL deployment class that integrates all components from
    the PDF "buy fix4 on C model" for production use.
    """

    def __init__(self):
        # Core optimization components
        self.risk_sizer = RiskAdjustedSizer()
        self.pattern_filter = PatternBasedFilter()
        self.stop_loss_manager = VolatilityAdjustedStopLoss()
        self.performance_scorer = PerformanceWeightedScorer()

        # Enhanced optimizer (combines all components)
        self.enhanced_optimizer = EnhancedChinaBuyOptimizer()

        # Production configuration
        self.production_config = {
            'max_signals_per_day': PRODUCTION_CONFIG['system_parameters']['max_daily_signals'],
            'minimum_confidence': PRODUCTION_CONFIG['system_parameters']['min_confidence'],
            'maximum_portfolio_allocation': PRODUCTION_CONFIG['system_parameters']['max_portfolio_allocation'],
            'auto_rebalance_frequency': 'weekly'
        }

        # Performance tracking
        self.daily_stats = {
            'signals_generated': 0,
            'signals_rejected': 0,
            'total_allocation': 0.0
        }

    def generate_production_signals(self, raw_signals: list, market_conditions: Dict = None) -> list:
        """
        Generate production signals with all optimizations applied.

        This is the main entry point for production signal generation.

        Args:
            raw_signals: List of raw signal dicts from ML model
            market_conditions: Optional dict with market regime info

        Returns:
            List of optimized signal dicts ready for execution
        """
        optimized_signals = []
        total_allocation = 0

        # Reset daily stats
        self.daily_stats = {
            'signals_generated': 0,
            'signals_rejected': 0,
            'total_allocation': 0.0
        }

        for signal in raw_signals:
            # Skip if we've reached maximum signals
            if len(optimized_signals) >= self.production_config['max_signals_per_day']:
                break

            # Skip low confidence signals
            confidence = signal.get('confidence', 0)
            if confidence < self.production_config['minimum_confidence']:
                self.daily_stats['signals_rejected'] += 1
                continue

            ticker = signal.get('ticker', 'Unknown')

            # Build ticker_data from signal (or use provided market data)
            ticker_data = self._extract_ticker_data(signal, market_conditions)

            # Build signal_data for optimizer
            signal_data = {
                'ticker': ticker,
                'confidence': confidence,
                'position_size': signal.get('position_size', 0.05),
                'buy_quality': signal.get('buy_quality', signal.get('signal_quality', 'cautious_buy'))
            }

            # Run through enhanced optimizer
            result = self.enhanced_optimizer.optimize_buy_signal(signal_data, ticker_data)

            if result['action'] == 'BLOCKED':
                self._log_rejection(ticker, result.get('reason', 'Unknown'))
                self.daily_stats['signals_rejected'] += 1
                continue

            # Get optimized parameters
            optimized_size = result['optimized_size']

            # Check portfolio allocation limits
            if total_allocation + optimized_size > self.production_config['maximum_portfolio_allocation']:
                continue

            # Build production signal
            production_signal = {
                **signal,
                'optimized_size': optimized_size,
                'stop_loss_pct': result['stop_loss_pct'],
                'performance_score': result['optimized_confidence'],
                'risk_patterns': result.get('risk_signals', []),
                'signal_pattern': result.get('pattern', 'NORMAL'),
                'risk_pattern': result.get('risk_pattern', 'NORMAL'),
                'portfolio_allocation': total_allocation + optimized_size,
                'optimization_applied': True,
                'max_hold_days': result.get('max_hold_days', 10)
            }

            optimized_signals.append(production_signal)
            total_allocation += optimized_size
            self.daily_stats['signals_generated'] += 1

        self.daily_stats['total_allocation'] = total_allocation

        # Final validation and sorting
        return self._finalize_production_signals(optimized_signals)

    def _extract_ticker_data(self, signal: Dict, market_conditions: Dict = None) -> Dict:
        """Extract ticker data from signal or market conditions."""
        # Try to get from signal first
        ticker_data = {
            'volatility': signal.get('volatility', signal.get('vol', 0.25)),
            'momentum_20d': signal.get('momentum_20d', signal.get('mom_20d', 0)),
            'momentum_5d': signal.get('momentum_5d', signal.get('mom_5d', 0)),
            'volume_ratio': signal.get('volume_ratio', signal.get('vol_ratio', 1.0)),
            'dist_from_ma': signal.get('dist_from_ma', signal.get('dist_ma', 0)),
            'sector': signal.get('sector', signal.get('industry', ''))
        }

        # Override with market conditions if provided
        if market_conditions:
            for key in ticker_data:
                if key in market_conditions:
                    ticker_data[key] = market_conditions[key]

        return ticker_data

    def _finalize_production_signals(self, signals: list) -> list:
        """Final validation and preparation for production."""
        # Sort by performance score (highest first)
        signals.sort(key=lambda x: x.get('performance_score', 0), reverse=True)

        # Add production metadata
        from datetime import datetime
        for signal in signals:
            signal.update({
                'optimization_version': '1.0',
                'deployment_timestamp': datetime.now().isoformat(),
                'expected_improvement': '+1.9% (validated)',
                'risk_level': self._calculate_risk_level(signal.get('risk_patterns', []))
            })

        return signals

    def _calculate_risk_level(self, risk_patterns: list) -> str:
        """Calculate overall risk level from patterns."""
        if not risk_patterns:
            return 'LOW'

        high_risk_count = sum(1 for r in risk_patterns
                             if isinstance(r, dict) and r.get('risk_level') == 'HIGH')

        if high_risk_count >= 2:
            return 'HIGH'
        elif high_risk_count == 1 or len(risk_patterns) > 2:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _log_rejection(self, ticker: str, reason: str):
        """Log pattern rejections for monitoring."""
        print(f"  REJECTED {ticker}: {reason}")

    def get_daily_stats(self) -> Dict:
        """Get daily production statistics."""
        rejection_rate = (self.daily_stats['signals_rejected'] /
                         max(1, self.daily_stats['signals_generated'] +
                             self.daily_stats['signals_rejected']))

        return {
            **self.daily_stats,
            'rejection_rate': rejection_rate,
            'allocation_remaining': (self.production_config['maximum_portfolio_allocation'] -
                                    self.daily_stats['total_allocation'])
        }

    def get_production_config(self) -> Dict:
        """Get current production configuration."""
        return {
            'production_config': self.production_config,
            'global_config': PRODUCTION_CONFIG
        }

    def optimize_buy_signal(self, signal_data: Dict, ticker_data: Dict,
                           historical_performance: Dict = None) -> Dict:
        """
        Backward-compatible method for webapp.py integration.

        This delegates to the enhanced_optimizer's optimize_buy_signal method
        while adding production-level validation and monitoring.

        Args:
            signal_data: Dict with ticker, confidence, position_size, buy_quality
            ticker_data: Dict with volatility, momentum_20d, momentum_5d, volume_ratio, dist_from_ma
            historical_performance: Optional historical performance data

        Returns:
            Dict with optimization results including action, stop_loss_pct, optimized_size, etc.
        """
        # Check minimum confidence threshold
        confidence = signal_data.get('confidence', 0)
        if confidence < self.production_config['minimum_confidence']:
            return {
                'action': 'BLOCKED',
                'reason': f'Low confidence ({confidence:.2%} < {self.production_config["minimum_confidence"]:.2%})',
                'risk_signals': [{'signal': 'LOW_CONFIDENCE', 'risk_level': 'HIGH'}],
                'original_size': signal_data.get('position_size', 0.05),
                'optimized_size': 0,
                'stop_loss_pct': 0.06,
            }

        # Delegate to enhanced optimizer
        result = self.enhanced_optimizer.optimize_buy_signal(
            signal_data, ticker_data, historical_performance
        )

        # Check portfolio allocation limits
        if result.get('action') != 'BLOCKED':
            optimized_size = result.get('optimized_size', 0.05)
            if self.daily_stats['total_allocation'] + optimized_size > self.production_config['maximum_portfolio_allocation']:
                result['action'] = 'BLOCKED'
                result['reason'] = f'Portfolio allocation limit ({self.production_config["maximum_portfolio_allocation"]:.0%}) exceeded'
                result['risk_signals'] = result.get('risk_signals', []) + [
                    {'signal': 'ALLOCATION_LIMIT', 'risk_level': 'HIGH'}
                ]

        # Track statistics
        if result.get('action') == 'BLOCKED':
            self.daily_stats['signals_rejected'] += 1
        else:
            self.daily_stats['signals_generated'] += 1
            self.daily_stats['total_allocation'] += result.get('optimized_size', 0.05)

        return result


class ChinaBuyMonitor:
    """
    Real-time monitoring for China BUY optimization system.

    Tracks performance metrics and triggers alerts when thresholds are breached.
    """

    def __init__(self, production_optimizer: ProductionChinaBuyOptimizer = None):
        self.production_optimizer = production_optimizer

        self.performance_metrics = {
            'signals_generated': 0,
            'signals_rejected': 0,
            'total_pnl': 0.0,
            'rejection_reasons': {}
        }

        self.alert_thresholds = {
            'consecutive_rejections': 5,
            'rejection_rate': 0.7,  # 70% rejection rate
            'performance_drop': -0.005  # -0.5% P&L drop
        }

        self.alerts_triggered = []

    def record_signal(self, signal: Dict, was_rejected: bool, rejection_reason: str = None):
        """Record a signal for monitoring."""
        if was_rejected:
            self.performance_metrics['signals_rejected'] += 1
            if rejection_reason:
                self.performance_metrics['rejection_reasons'][rejection_reason] = \
                    self.performance_metrics['rejection_reasons'].get(rejection_reason, 0) + 1
        else:
            self.performance_metrics['signals_generated'] += 1

    def record_pnl(self, pnl: float):
        """Record P&L for performance tracking."""
        self.performance_metrics['total_pnl'] += pnl

    def check_performance_alerts(self) -> list:
        """Check for performance degradation alerts."""
        alerts = []

        total_signals = (self.performance_metrics['signals_generated'] +
                        self.performance_metrics['signals_rejected'])

        if total_signals > 0:
            # High rejection rate alert
            rejection_rate = self.performance_metrics['signals_rejected'] / total_signals
            if rejection_rate > self.alert_thresholds['rejection_rate']:
                alerts.append(f"High rejection rate: {rejection_rate:.1%}")

            # Performance drop alert
            if self.performance_metrics['total_pnl'] < self.alert_thresholds['performance_drop']:
                alerts.append(f"Performance drop: {self.performance_metrics['total_pnl']:.3%}")

        self.alerts_triggered.extend(alerts)
        return alerts

    def get_monitoring_report(self) -> Dict:
        """Get comprehensive monitoring report."""
        total_signals = (self.performance_metrics['signals_generated'] +
                        self.performance_metrics['signals_rejected'])

        return {
            'optimization_performance': {
                'validated_improvement': '+1.899%',
                'current_pnl': self.performance_metrics['total_pnl'],
                'risk_reduction': 'Significant',
                'deployment_status': 'Production'
            },
            'signal_metrics': {
                'total_processed': total_signals,
                'generated': self.performance_metrics['signals_generated'],
                'rejected': self.performance_metrics['signals_rejected'],
                'rejection_rate': (self.performance_metrics['signals_rejected'] /
                                  max(1, total_signals))
            },
            'rejection_breakdown': self.performance_metrics['rejection_reasons'],
            'alerts': self.alerts_triggered,
            'alert_thresholds': self.alert_thresholds
        }

    def reset_metrics(self):
        """Reset metrics for new monitoring period."""
        self.performance_metrics = {
            'signals_generated': 0,
            'signals_rejected': 0,
            'total_pnl': 0.0,
            'rejection_reasons': {}
        }
        self.alerts_triggered = []


def verify_production_readiness() -> Dict:
    """
    Verify all systems are ready for production deployment.

    Returns:
        Dict with readiness status and details
    """
    checks = {}

    # Test pattern filter
    try:
        pf = PatternBasedFilter()
        test_data = {'volatility': 0.42, 'momentum_20d': -0.08, 'sector': 'Industrial'}
        should_reject, _ = pf.should_reject(test_data)
        checks['pattern_filter'] = should_reject  # Should reject this pattern
    except Exception as e:
        checks['pattern_filter'] = False

    # Test risk sizing
    try:
        rs = RiskAdjustedSizer()
        result = rs.calculate_risk_adjusted_size(0.05, {'volatility': 0.45, 'momentum_20d': -0.06})
        checks['risk_sizing'] = result['adjusted_size'] < 0.05  # Should reduce size
    except Exception as e:
        checks['risk_sizing'] = False

    # Test stop loss
    try:
        sl = VolatilityAdjustedStopLoss()
        result = sl.get_stop_details(0.40, 'cautious_buy')
        checks['stop_loss'] = result['final_stop_loss'] < 0.05  # Tight stop for high vol
    except Exception as e:
        checks['stop_loss'] = False

    # Test integration
    try:
        optimizer = EnhancedChinaBuyOptimizer()
        signal_data = {'ticker': 'TEST', 'confidence': 0.7, 'position_size': 0.05}
        ticker_data = {'volatility': 0.25, 'momentum_20d': 0.05}
        result = optimizer.optimize_buy_signal(signal_data, ticker_data)
        checks['integration'] = result['action'] in ['OPTIMIZED', 'BLOCKED']
    except Exception as e:
        checks['integration'] = False

    # Test production optimizer
    try:
        prod_opt = ProductionChinaBuyOptimizer()
        checks['production_optimizer'] = prod_opt is not None
    except Exception as e:
        checks['production_optimizer'] = False

    all_checks_passed = all(checks.values())

    return {
        'ready': all_checks_passed,
        'reason': "All production checks passed" if all_checks_passed else "Some checks failed",
        'risk_improvement': '+1.9% validated improvement',
        'details': checks
    }


def deploy_china_buy_optimizer() -> ProductionChinaBuyOptimizer:
    """
    Deploy the validated China BUY optimizer to production.

    Returns:
        ProductionChinaBuyOptimizer instance ready for use
    """
    print("Deploying China BUY Optimizer to Production...")

    # Verify deployment readiness
    readiness = verify_production_readiness()
    if not readiness['ready']:
        print(f"Deployment blocked: {readiness['reason']}")
        print(f"Failed checks: {[k for k, v in readiness['details'].items() if not v]}")
        return None

    # Initialize production optimizer
    production_optimizer = ProductionChinaBuyOptimizer()

    print("China BUY Optimizer successfully deployed!")
    print(f"Expected improvement: +1.9% P&L (validated)")
    print(f"Risk reduction: {readiness['risk_improvement']}")

    return production_optimizer


# Convenience function for quick validation
def validate_china_signal(ticker: str,
                         signal: str,
                         confidence: float,
                         momentum_5d: float,
                         momentum_20d: float,
                         volatility: float,
                         volume_ratio: float = 1.0,
                         dist_from_ma: float = 0.0,
                         sector: str = '') -> Dict:
    """
    Quick validation for China model signals.

    Args:
        ticker: Stock ticker
        signal: 'BUY', 'SELL', or 'HOLD'
        confidence: Model confidence (0-1)
        momentum_5d: 5-day momentum (e.g., 0.15 = 15%)
        momentum_20d: 20-day momentum
        volatility: Annualized volatility
        volume_ratio: Current volume / average
        dist_from_ma: Distance from 20-day MA
        sector: Stock sector

    Returns:
        Dict with validation results
    """

    ticker_info = {
        'momentum_5d': momentum_5d,
        'momentum_20d': momentum_20d,
        'volatility': volatility,
        'volume_ratio': volume_ratio,
        'dist_from_ma': dist_from_ma,
        'sector': sector
    }

    system = EnhancedPhaseSystem()
    return system.process_signal(ticker, signal, confidence, ticker_info)


if __name__ == "__main__":
    # Test the enhanced signal validator
    print("=" * 80)
    print("ENHANCED SIGNAL VALIDATOR TEST")
    print("=" * 80)

    # Test cases from actual catastrophic shorts
    test_cases = [
        # Zhongfu Straits - SELL signal, +290% actual (catastrophic)
        {
            'ticker': '000592.SZ',
            'name': 'Zhongfu Straits',
            'signal': 'SELL',
            'confidence': 0.65,
            'momentum_5d': 0.30,
            'momentum_20d': 0.85,  # Very strong momentum
            'volatility': 0.75,
            'volume_ratio': 3.5,   # Very high volume
            'sector': 'Real Estate'
        },
        # Winnovation - SELL signal, +77% actual
        {
            'ticker': '000620.SZ',
            'name': 'Winnovation',
            'signal': 'SELL',
            'confidence': 0.60,
            'momentum_5d': 0.20,
            'momentum_20d': 0.45,
            'volatility': 0.55,
            'volume_ratio': 2.2,
            'sector': 'Technology'
        },
        # BlueFocus - SELL signal, +46% actual
        {
            'ticker': '300058.SZ',
            'name': 'BlueFocus',
            'signal': 'SELL',
            'confidence': 0.62,
            'momentum_5d': 0.12,
            'momentum_20d': 0.35,
            'volatility': 0.48,
            'volume_ratio': 1.8,
            'sector': 'Technology'
        },
        # CCB Bank - BUY signal (should pass)
        {
            'ticker': '0939.HK',
            'name': 'CCB Bank',
            'signal': 'BUY',
            'confidence': 0.72,
            'momentum_5d': 0.03,
            'momentum_20d': 0.08,
            'volatility': 0.22,
            'volume_ratio': 1.1,
            'sector': 'Financial'
        },
        # GCL Technology - BUY signal (should pass with adjustment)
        {
            'ticker': '3800.HK',
            'name': 'GCL Technology',
            'signal': 'BUY',
            'confidence': 0.68,
            'momentum_5d': 0.08,
            'momentum_20d': 0.15,
            'volatility': 0.45,
            'volume_ratio': 1.5,
            'sector': 'Green Energy'
        }
    ]

    system = EnhancedPhaseSystem()

    for case in test_cases:
        ticker_info = {
            'momentum_5d': case['momentum_5d'],
            'momentum_20d': case['momentum_20d'],
            'volatility': case['volatility'],
            'volume_ratio': case['volume_ratio'],
            'dist_from_ma': case['momentum_5d'],  # Approximate
            'sector': case['sector']
        }

        result = system.process_signal(
            case['ticker'],
            case['signal'],
            case['confidence'],
            ticker_info
        )

        print(f"\n{'-'*60}")
        print(f"{case['ticker']} - {case['name']}")
        print(f"  Original: {case['signal']} @ {case['confidence']:.0%}")
        print(f"  Momentum: 5d={case['momentum_5d']:.0%}, 20d={case['momentum_20d']:.0%}")
        print(f"  Vol: {case['volatility']:.0%}, Volume: {case['volume_ratio']:.1f}x")
        print(f"  ")
        print(f"  RESULT:")
        print(f"    Signal: {result['original_signal']} -> {result['validated_signal']}")
        print(f"    Reason: {result['validation_reason']}")
        print(f"    Confidence: {result['original_confidence']:.0%} -> {result['final_confidence']:.0%}")
        print(f"    Position Size: {result['position_size_pct']:.1%}")
        print(f"    Dangerous Short: {result['is_dangerous_short']}")
        if result['signal_blocked']:
            print(f"    *** SIGNAL BLOCKED ***")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    blocked = sum(1 for case in test_cases
                  if system.process_signal(
                      case['ticker'], case['signal'], case['confidence'],
                      {'momentum_5d': case['momentum_5d'], 'momentum_20d': case['momentum_20d'],
                       'volatility': case['volatility'], 'volume_ratio': case['volume_ratio'],
                       'dist_from_ma': case['momentum_5d'], 'sector': case['sector']}
                  )['signal_blocked'])

    sells = sum(1 for case in test_cases if case['signal'] == 'SELL')
    print(f"Total SELL signals: {sells}")
    print(f"Signals blocked: {blocked}")
    print(f"Block rate: {blocked/sells:.0%}" if sells > 0 else "N/A")
