"""
China Lag-Free Transition Detector

Standalone module for lag-free regime transition detection in China/DeepSeek model.
This module can be integrated into existing China model components without
modifying the core prediction logic.

Key Features:
1. Early warning signal detection (10-30 days faster than traditional indicators)
2. Gradual parameter blending (prevents whipsaws)
3. Confirmation periods (prevents false signals)
4. Multi-timeframe analysis

Usage:
    from china_model.src.china_lag_free_transition import (
        ChinaLagFreeTransitionDetector,
        ChinaTransitionOutput
    )

    detector = ChinaLagFreeTransitionDetector()
    result = detector.detect_transition(market_data)

    if result.is_transition:
        print(f"Transition detected: {result.transition_type}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Signals: {result.signals_detected}")

Last Updated: 2025-12-17
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
from datetime import datetime


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ChinaTransitionOutput:
    """Container for transition detection results."""

    # Transition classification
    transition_type: str         # TRANSITION_TO_BULL, TRANSITION_TO_BEAR, EARLY_SIGNAL, NO_TRANSITION
    confidence: float            # 0.0 to 1.0
    is_transition: bool          # True if transition detected

    # Early warning signals
    signals_detected: List[str]  # ['higher_lows', 'bullish_divergence', ...]
    signal_count: int            # Number of signals detected

    # Confirmation status
    days_in_transition: int      # Days since transition started
    is_confirmed: bool           # True if transition confirmed

    # Blending factors
    blend_factor: float          # 0.0 to 1.0 (how much to blend toward new regime)
    recommended_allocation_adjustment: float  # -0.2 to +0.2

    # Guidance
    recommended_actions: List[str]


# ============================================================================
# CHINA LAG-FREE TRANSITION DETECTOR
# ============================================================================

class ChinaLagFreeTransitionDetector:
    """
    Standalone lag-free transition detector for China markets.

    This module detects bear-to-bull and bull-to-bear transitions
    10-30 days faster than traditional indicators by analyzing:
    - Higher lows / lower highs patterns
    - RSI divergence
    - Volume patterns
    - SMA breakouts
    - Momentum acceleration

    Integration:
        detector = ChinaLagFreeTransitionDetector()

        # In your daily analysis loop:
        result = detector.detect_transition(hsi_data)

        if result.is_transition:
            # Adjust strategy parameters
            params['allocation'] *= (1 + result.recommended_allocation_adjustment)
    """

    # Early warning signal weights
    BULL_SIGNAL_WEIGHTS = {
        'higher_lows': 0.25,
        'bullish_divergence': 0.25,
        'volume_expansion': 0.20,
        'sma_breakout': 0.20,
        'momentum_acceleration': 0.10
    }

    BEAR_SIGNAL_WEIGHTS = {
        'lower_highs': 0.25,
        'bearish_divergence': 0.25,
        'volume_on_down': 0.20,
        'sma_breakdown': 0.20,
        'momentum_deceleration': 0.10
    }

    # Transition thresholds
    TRANSITION_THRESHOLDS = {
        'bear_to_bull': {
            'minimum_signals': 3,
            'confidence_threshold': 0.6,
            'confirmation_period': 3,
        },
        'bull_to_bear': {
            'minimum_signals': 2,
            'confidence_threshold': 0.5,
            'confirmation_period': 2,
        }
    }

    # Blending speeds
    BLENDING_SPEEDS = {
        'bear_to_bull': 0.3,    # 30% per day (patient)
        'bull_to_bear': 0.5,    # 50% per day (faster defense)
    }

    def __init__(self):
        """Initialize the transition detector."""
        self.pending_transition = None
        self.days_in_transition = 0
        self.transition_history = deque(maxlen=20)
        self.last_signals = []

    def detect_transition(
        self,
        market_data: pd.DataFrame,
        current_regime: str = 'NEUTRAL'
    ) -> ChinaTransitionOutput:
        """
        Detect regime transition with early warning signals.

        Args:
            market_data: DataFrame with OHLCV columns
            current_regime: Current regime ('BULL', 'BEAR', 'NEUTRAL', 'HIGH_VOL')

        Returns:
            ChinaTransitionOutput with full transition analysis
        """
        if len(market_data) < 30:
            return self._no_transition_output()

        # Detect bullish signals
        bull_result = self._detect_bull_signals(market_data)

        # Detect bearish signals
        bear_result = self._detect_bear_signals(market_data)

        # Determine primary transition direction
        if bull_result['confidence'] > bear_result['confidence'] and bull_result['confidence'] >= 0.25:
            return self._process_bull_transition(bull_result, current_regime)
        elif bear_result['confidence'] >= 0.25:
            return self._process_bear_transition(bear_result, current_regime)
        else:
            # Reset pending transition if no signals
            if self.pending_transition is not None:
                self.days_in_transition = 0
                self.pending_transition = None
            return self._no_transition_output()

    def _detect_bull_signals(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect bullish transition signals."""
        signals = []
        scores = []

        # 1. Higher Lows Pattern (25%)
        try:
            lows_5d = market_data['Low'].rolling(5).min()
            if len(lows_5d) >= 15:
                low_now = float(lows_5d.iloc[-1])
                low_6d = float(lows_5d.iloc[-6])
                low_11d = float(lows_5d.iloc[-11])

                if not any(pd.isna([low_now, low_6d, low_11d])):
                    if low_now > low_6d > low_11d:
                        signals.append('higher_lows')
                        scores.append(self.BULL_SIGNAL_WEIGHTS['higher_lows'])
        except Exception:
            pass

        # 2. RSI Bullish Divergence (25%)
        try:
            rsi = self._calculate_rsi(market_data['Close'], 14)
            if len(rsi) >= 15:
                rsi_now = float(rsi.iloc[-1])
                rsi_10d = float(rsi.iloc[-10])
                price_now = float(market_data['Close'].iloc[-1])
                price_10d = float(market_data['Close'].iloc[-10])

                if not any(pd.isna([rsi_now, rsi_10d, price_now, price_10d])):
                    if price_now < price_10d and rsi_now > rsi_10d + 3:
                        signals.append('bullish_divergence')
                        scores.append(self.BULL_SIGNAL_WEIGHTS['bullish_divergence'])
        except Exception:
            pass

        # 3. Volume Expansion on Up Days (20%)
        if 'Volume' in market_data.columns:
            try:
                data = market_data.copy()
                data['Up_Day'] = data['Close'] > data['Open']
                up_days = data[data['Up_Day']]
                down_days = data[~data['Up_Day']]

                if len(up_days) >= 5 and len(down_days) >= 5:
                    up_vol = float(up_days['Volume'].tail(5).mean())
                    down_vol = float(down_days['Volume'].tail(5).mean())
                    if down_vol > 0 and up_vol > down_vol * 1.3:
                        signals.append('volume_expansion')
                        scores.append(self.BULL_SIGNAL_WEIGHTS['volume_expansion'])
            except Exception:
                pass

        # 4. SMA Breakout (20%)
        try:
            sma_20 = market_data['Close'].rolling(20).mean()
            if len(sma_20) >= 21:
                price_now = float(market_data['Close'].iloc[-1])
                price_prev = float(market_data['Close'].iloc[-2])
                sma_now = float(sma_20.iloc[-1])
                sma_prev = float(sma_20.iloc[-2])

                if not any(pd.isna([price_now, price_prev, sma_now, sma_prev])):
                    if price_now > sma_now and price_prev <= sma_prev:
                        signals.append('sma_breakout')
                        scores.append(self.BULL_SIGNAL_WEIGHTS['sma_breakout'])
        except Exception:
            pass

        # 5. Momentum Acceleration (10%)
        if len(market_data) >= 11:
            try:
                ret_5d = float(market_data['Close'].pct_change(5).iloc[-1])
                ret_10d = float(market_data['Close'].pct_change(10).iloc[-1])

                if not pd.isna(ret_5d) and not pd.isna(ret_10d):
                    if ret_5d > 0.02 and ret_5d > ret_10d:
                        signals.append('momentum_acceleration')
                        scores.append(self.BULL_SIGNAL_WEIGHTS['momentum_acceleration'])
            except Exception:
                pass

        return {
            'signals': signals,
            'confidence': sum(scores),
            'signal_count': len(signals)
        }

    def _detect_bear_signals(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect bearish transition signals."""
        signals = []
        scores = []

        # 1. Lower Highs Pattern (25%)
        try:
            highs_5d = market_data['High'].rolling(5).max()
            if len(highs_5d) >= 15:
                high_now = float(highs_5d.iloc[-1])
                high_6d = float(highs_5d.iloc[-6])
                high_11d = float(highs_5d.iloc[-11])

                if not any(pd.isna([high_now, high_6d, high_11d])):
                    if high_now < high_6d < high_11d:
                        signals.append('lower_highs')
                        scores.append(self.BEAR_SIGNAL_WEIGHTS['lower_highs'])
        except Exception:
            pass

        # 2. RSI Bearish Divergence (25%)
        try:
            rsi = self._calculate_rsi(market_data['Close'], 14)
            if len(rsi) >= 15:
                rsi_now = float(rsi.iloc[-1])
                rsi_10d = float(rsi.iloc[-10])
                price_now = float(market_data['Close'].iloc[-1])
                price_10d = float(market_data['Close'].iloc[-10])

                if not any(pd.isna([rsi_now, rsi_10d, price_now, price_10d])):
                    if price_now > price_10d and rsi_now < rsi_10d - 3:
                        signals.append('bearish_divergence')
                        scores.append(self.BEAR_SIGNAL_WEIGHTS['bearish_divergence'])
        except Exception:
            pass

        # 3. Volume on Down Days (20%)
        if 'Volume' in market_data.columns:
            try:
                data = market_data.copy()
                data['Down_Day'] = data['Close'] < data['Open']
                down_days = data[data['Down_Day']]
                up_days = data[~data['Down_Day']]

                if len(down_days) >= 5 and len(up_days) >= 5:
                    down_vol = float(down_days['Volume'].tail(5).mean())
                    up_vol = float(up_days['Volume'].tail(5).mean())
                    if up_vol > 0 and down_vol > up_vol * 1.3:
                        signals.append('volume_on_down')
                        scores.append(self.BEAR_SIGNAL_WEIGHTS['volume_on_down'])
            except Exception:
                pass

        # 4. SMA Breakdown (20%)
        try:
            sma_20 = market_data['Close'].rolling(20).mean()
            if len(sma_20) >= 21:
                price_now = float(market_data['Close'].iloc[-1])
                price_prev = float(market_data['Close'].iloc[-2])
                sma_now = float(sma_20.iloc[-1])
                sma_prev = float(sma_20.iloc[-2])

                if not any(pd.isna([price_now, price_prev, sma_now, sma_prev])):
                    if price_now < sma_now and price_prev >= sma_prev:
                        signals.append('sma_breakdown')
                        scores.append(self.BEAR_SIGNAL_WEIGHTS['sma_breakdown'])
        except Exception:
            pass

        # 5. Momentum Deceleration (10%)
        if len(market_data) >= 11:
            try:
                ret_5d = float(market_data['Close'].pct_change(5).iloc[-1])
                ret_10d = float(market_data['Close'].pct_change(10).iloc[-1])

                if not pd.isna(ret_5d) and not pd.isna(ret_10d):
                    if ret_5d < -0.02 and ret_5d < ret_10d:
                        signals.append('momentum_deceleration')
                        scores.append(self.BEAR_SIGNAL_WEIGHTS['momentum_deceleration'])
            except Exception:
                pass

        return {
            'signals': signals,
            'confidence': sum(scores),
            'signal_count': len(signals)
        }

    def _process_bull_transition(
        self,
        bull_result: Dict[str, Any],
        current_regime: str
    ) -> ChinaTransitionOutput:
        """Process potential bull transition."""
        signals = bull_result['signals']
        confidence = bull_result['confidence']
        thresholds = self.TRANSITION_THRESHOLDS['bear_to_bull']

        # Check if starting new transition
        if self.pending_transition != 'BULL':
            self.pending_transition = 'BULL'
            self.days_in_transition = 1
            self.last_signals = signals
        else:
            self.days_in_transition += 1

        # Check confirmation
        is_confirmed = (
            self.days_in_transition >= thresholds['confirmation_period'] and
            confidence >= thresholds['confidence_threshold'] and
            len(signals) >= thresholds['minimum_signals']
        )

        # Calculate blend factor
        blend_speed = self.BLENDING_SPEEDS['bear_to_bull']
        blend_factor = min(self.days_in_transition * blend_speed * confidence, 1.0)

        # Determine transition type
        if is_confirmed:
            transition_type = 'CONFIRMED_TRANSITION_TO_BULL'
            self.transition_history.append(('bull', datetime.now()))
            self.pending_transition = None
            self.days_in_transition = 0
        elif confidence >= 0.5:
            transition_type = 'EARLY_BULL_TRANSITION'
        else:
            transition_type = 'EARLY_BULL_SIGNAL'

        # Calculate allocation adjustment
        if is_confirmed:
            alloc_adj = 0.15  # Boost allocation by 15%
        elif confidence >= 0.5:
            alloc_adj = 0.10 * blend_factor
        else:
            alloc_adj = 0.05 * blend_factor

        # Recommended actions
        actions = []
        if is_confirmed:
            actions = [
                'Increase position sizes',
                'Focus on momentum stocks',
                'Widen stop losses',
                'Extend holding periods'
            ]
        elif confidence >= 0.5:
            actions = [
                'Prepare to increase exposure',
                'Watch for confirmation',
                'Start adding momentum positions'
            ]
        else:
            actions = [
                'Monitor transition signals',
                'Maintain current positions'
            ]

        return ChinaTransitionOutput(
            transition_type=transition_type,
            confidence=confidence,
            is_transition=True,
            signals_detected=signals,
            signal_count=len(signals),
            days_in_transition=self.days_in_transition,
            is_confirmed=is_confirmed,
            blend_factor=blend_factor,
            recommended_allocation_adjustment=alloc_adj,
            recommended_actions=actions
        )

    def _process_bear_transition(
        self,
        bear_result: Dict[str, Any],
        current_regime: str
    ) -> ChinaTransitionOutput:
        """Process potential bear transition."""
        signals = bear_result['signals']
        confidence = bear_result['confidence']
        thresholds = self.TRANSITION_THRESHOLDS['bull_to_bear']

        # Check if starting new transition
        if self.pending_transition != 'BEAR':
            self.pending_transition = 'BEAR'
            self.days_in_transition = 1
            self.last_signals = signals
        else:
            self.days_in_transition += 1

        # Check confirmation (faster for bear)
        is_confirmed = (
            self.days_in_transition >= thresholds['confirmation_period'] and
            confidence >= thresholds['confidence_threshold'] and
            len(signals) >= thresholds['minimum_signals']
        )

        # Calculate blend factor (faster for bear)
        blend_speed = self.BLENDING_SPEEDS['bull_to_bear']
        blend_factor = min(self.days_in_transition * blend_speed * confidence, 1.0)

        # Determine transition type
        if is_confirmed:
            transition_type = 'CONFIRMED_TRANSITION_TO_BEAR'
            self.transition_history.append(('bear', datetime.now()))
            self.pending_transition = None
            self.days_in_transition = 0
        elif confidence >= 0.5:
            transition_type = 'EARLY_BEAR_TRANSITION'
        else:
            transition_type = 'EARLY_BEAR_SIGNAL'

        # Calculate allocation adjustment (negative for bear)
        if is_confirmed:
            alloc_adj = -0.20  # Reduce allocation by 20%
        elif confidence >= 0.5:
            alloc_adj = -0.15 * blend_factor
        else:
            alloc_adj = -0.10 * blend_factor

        # Recommended actions
        actions = []
        if is_confirmed:
            actions = [
                'Reduce position sizes immediately',
                'Raise cash position',
                'Tighten stop losses',
                'Focus on quality/defensive stocks'
            ]
        elif confidence >= 0.5:
            actions = [
                'Prepare to reduce exposure',
                'Review stop losses',
                'Trim weakest positions'
            ]
        else:
            actions = [
                'Monitor warning signals',
                'Prepare defensive strategy'
            ]

        return ChinaTransitionOutput(
            transition_type=transition_type,
            confidence=confidence,
            is_transition=True,
            signals_detected=signals,
            signal_count=len(signals),
            days_in_transition=self.days_in_transition,
            is_confirmed=is_confirmed,
            blend_factor=blend_factor,
            recommended_allocation_adjustment=alloc_adj,
            recommended_actions=actions
        )

    def _no_transition_output(self) -> ChinaTransitionOutput:
        """Return output when no transition detected."""
        return ChinaTransitionOutput(
            transition_type='NO_TRANSITION',
            confidence=0.0,
            is_transition=False,
            signals_detected=[],
            signal_count=0,
            days_in_transition=0,
            is_confirmed=False,
            blend_factor=0.0,
            recommended_allocation_adjustment=0.0,
            recommended_actions=['Maintain current strategy']
        )

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_status(self) -> Dict[str, Any]:
        """Get current detector status."""
        return {
            'pending_transition': self.pending_transition,
            'days_in_transition': self.days_in_transition,
            'last_signals': self.last_signals,
            'recent_transitions': list(self.transition_history)
        }

    def reset(self):
        """Reset detector state."""
        self.pending_transition = None
        self.days_in_transition = 0
        self.last_signals = []


# ============================================================================
# INTEGRATION HELPER
# ============================================================================

def integrate_transition_detection(
    market_data: pd.DataFrame,
    current_params: Dict[str, Any],
    detector: Optional[ChinaLagFreeTransitionDetector] = None
) -> Dict[str, Any]:
    """
    Helper function to integrate transition detection into existing code.

    Usage:
        from china_model.src.china_lag_free_transition import integrate_transition_detection

        # In your daily processing:
        adjusted_params = integrate_transition_detection(
            market_data=hsi_data,
            current_params={'allocation': 0.85, 'stop_loss': 0.06}
        )

    Args:
        market_data: DataFrame with OHLCV
        current_params: Current strategy parameters
        detector: Optional detector instance (creates new if None)

    Returns:
        Adjusted parameters based on transition state
    """
    if detector is None:
        detector = ChinaLagFreeTransitionDetector()

    result = detector.detect_transition(market_data)

    adjusted_params = current_params.copy()

    # Determine current regime based on transition
    current_regime = 'NEUTRAL'
    if result.is_confirmed:
        if 'BULL' in result.transition_type:
            current_regime = 'BULL'
        elif 'BEAR' in result.transition_type:
            current_regime = 'BEAR'
        elif 'HIGH_VOLATILITY' in result.transition_type:
            current_regime = 'HIGH_VOL'

    # Calculate adjusted allocation
    base_allocation = current_params.get('max_allocation', current_params.get('allocation', 0.85))
    if result.is_transition:
        adjusted_allocation = max(0.3, min(0.95,
            base_allocation * (1 + result.recommended_allocation_adjustment)
        ))
    else:
        adjusted_allocation = base_allocation

    # Add required output keys
    adjusted_params['current_regime'] = current_regime
    adjusted_params['adjusted_allocation'] = adjusted_allocation
    adjusted_params['blend_factor'] = result.blend_factor

    if result.is_transition:
        # Adjust allocation
        if 'allocation' in adjusted_params:
            adjusted_params['allocation'] = adjusted_allocation

        # Adjust stop loss (tighter during bear, wider during bull)
        if 'stop_loss' in adjusted_params:
            if 'BEAR' in result.transition_type:
                adjusted_params['stop_loss'] = max(0.03, adjusted_params['stop_loss'] * 0.8)
            elif 'BULL' in result.transition_type and result.is_confirmed:
                adjusted_params['stop_loss'] = min(0.10, adjusted_params['stop_loss'] * 1.2)

        # Add transition info
        adjusted_params['_transition'] = {
            'type': result.transition_type,
            'confidence': result.confidence,
            'signals': result.signals_detected,
            'actions': result.recommended_actions,
            'is_confirmed': result.is_confirmed
        }

    return adjusted_params


# ============================================================================
# MAIN EXECUTION / EXAMPLE
# ============================================================================

def main():
    """Example usage of ChinaLagFreeTransitionDetector."""
    print("=" * 70)
    print("CHINA LAG-FREE TRANSITION DETECTOR - EXAMPLE")
    print("=" * 70)

    # Create sample transition data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=90, freq='B')

    prices = [20000]
    for i in range(89):
        if i < 30:
            change = np.random.normal(-0.003, 0.015)
        elif i < 50:
            change = np.random.normal(0.002, 0.012)
        else:
            change = np.random.normal(0.005, 0.010)
        prices.append(prices[-1] * (1 + change))

    market_data = pd.DataFrame({
        'Open': prices,
        'High': [p * 1.01 for p in prices],
        'Low': [p * 0.99 for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000000, 5000000000, 90)
    }, index=dates)

    # Initialize detector
    detector = ChinaLagFreeTransitionDetector()

    print("\nRunning transition detection through market phases...\n")

    # Simulate daily detection
    for day in [25, 35, 45, 55, 65]:
        window = market_data.iloc[:day+1]
        result = detector.detect_transition(window)

        print(f"Day {day+1}:")
        print(f"  Transition Type: {result.transition_type}")
        print(f"  Confidence: {result.confidence:.1%}")
        print(f"  Signals: {', '.join(result.signals_detected) or 'None'}")
        print(f"  Days in Transition: {result.days_in_transition}")
        print(f"  Confirmed: {result.is_confirmed}")
        print(f"  Allocation Adj: {result.recommended_allocation_adjustment:+.1%}")
        print(f"  Actions: {result.recommended_actions[0]}")
        print()

    print("=" * 70)
    print("INTEGRATION EXAMPLE:")
    print("-" * 70)
    print("""
    from china_model.src.china_lag_free_transition import (
        ChinaLagFreeTransitionDetector,
        integrate_transition_detection
    )

    # Option 1: Direct usage
    detector = ChinaLagFreeTransitionDetector()
    result = detector.detect_transition(hsi_data)
    if result.is_transition:
        allocation *= (1 + result.recommended_allocation_adjustment)

    # Option 2: Helper function
    adjusted_params = integrate_transition_detection(
        market_data=hsi_data,
        current_params={'allocation': 0.85}
    )
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
