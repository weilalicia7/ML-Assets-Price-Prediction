"""
Multi-Timeframe Ensemble Module for Phase 5 Dynamic Weighting

Combines signals from multiple timeframes (1h, 4h, 1d, 1w) with configurable
weights and agreement-based weighting adjustments.

Expected improvement: +1-2% consistency in signal generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class TimeframeType(Enum):
    """Supported timeframes for multi-timeframe analysis"""
    HOURLY = '1h'
    FOUR_HOUR = '4h'
    DAILY = '1d'
    WEEKLY = '1w'


@dataclass
class TimeframeSignal:
    """Signal from a specific timeframe"""
    timeframe: str
    direction: float  # -1 to 1 (bearish to bullish)
    strength: float   # 0 to 1
    confidence: float # 0 to 1
    features: Dict[str, float] = None


@dataclass
class MultiTimeframeResult:
    """Combined result from multi-timeframe analysis"""
    combined_signal: float
    combined_confidence: float
    agreement_score: float
    timeframe_signals: Dict[str, TimeframeSignal]
    recommendation: str
    position_multiplier: float


class MultiTimeframeEnsemble:
    """
    Multi-Timeframe Ensemble for signal generation across multiple timeframes.

    Combines signals from 1h, 4h, 1d, and 1w timeframes with configurable
    weights. Adjusts weights based on signal agreement across timeframes.

    Default weights (from Phase 5 specification):
    - 1h: 15% (short-term momentum)
    - 4h: 25% (intraday trends)
    - 1d: 35% (primary signal)
    - 1w: 25% (long-term trend confirmation)
    """

    def __init__(
        self,
        base_ensemble: Any = None,
        timeframe_weights: Dict[str, float] = None,
        agreement_threshold: float = 0.6,
        min_timeframes: int = 2,
        volatility_adjustment: bool = True
    ):
        """
        Initialize Multi-Timeframe Ensemble.

        Args:
            base_ensemble: Base ensemble model for generating signals
            timeframe_weights: Custom weights for each timeframe
            agreement_threshold: Minimum agreement for high confidence (0-1)
            min_timeframes: Minimum number of timeframes required
            volatility_adjustment: Whether to adjust weights based on volatility
        """
        self.base_ensemble = base_ensemble
        self.agreement_threshold = agreement_threshold
        self.min_timeframes = min_timeframes
        self.volatility_adjustment = volatility_adjustment

        # Default timeframe configuration (from Phase 5 spec)
        self.timeframes = timeframe_weights or {
            '1h': {'weight': 0.15, 'periods': 1, 'lookback': 24},
            '4h': {'weight': 0.25, 'periods': 4, 'lookback': 42},
            '1d': {'weight': 0.35, 'periods': 24, 'lookback': 60},
            '1w': {'weight': 0.25, 'periods': 168, 'lookback': 52}
        }

        # Volatility regime adjustments
        self.volatility_adjustments = {
            'low': {'1h': 0.10, '4h': 0.20, '1d': 0.40, '1w': 0.30},
            'normal': {'1h': 0.15, '4h': 0.25, '1d': 0.35, '1w': 0.25},
            'high': {'1h': 0.20, '4h': 0.30, '1d': 0.30, '1w': 0.20},
            'crisis': {'1h': 0.05, '4h': 0.15, '1d': 0.40, '1w': 0.40}
        }

        # Performance tracking per timeframe
        self.timeframe_performance = {
            tf: {'correct': 0, 'total': 0, 'recent_accuracy': []}
            for tf in self.timeframes.keys()
        }

    def resample_data(
        self,
        data: pd.DataFrame,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Resample OHLCV data to target timeframe.

        Args:
            data: DataFrame with OHLCV columns and DatetimeIndex
            timeframe: Target timeframe ('1h', '4h', '1d', '1w')

        Returns:
            Resampled DataFrame
        """
        # Map timeframe to pandas resample rule
        resample_rules = {
            '1h': 'h',
            '4h': '4h',
            '1d': 'D',
            '1w': 'W'
        }

        rule = resample_rules.get(timeframe, 'D')

        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data = data.set_index('date')
            elif 'Date' in data.columns:
                data = data.set_index('Date')

        # Standard OHLCV resampling
        resampled = pd.DataFrame()

        if 'open' in data.columns or 'Open' in data.columns:
            open_col = 'open' if 'open' in data.columns else 'Open'
            high_col = 'high' if 'high' in data.columns else 'High'
            low_col = 'low' if 'low' in data.columns else 'Low'
            close_col = 'close' if 'close' in data.columns else 'Close'

            resampled['open'] = data[open_col].resample(rule).first()
            resampled['high'] = data[high_col].resample(rule).max()
            resampled['low'] = data[low_col].resample(rule).min()
            resampled['close'] = data[close_col].resample(rule).last()

            if 'volume' in data.columns or 'Volume' in data.columns:
                vol_col = 'volume' if 'volume' in data.columns else 'Volume'
                resampled['volume'] = data[vol_col].resample(rule).sum()
        else:
            # Just resample close prices if that's all we have
            for col in data.columns:
                if 'close' in col.lower() or 'price' in col.lower():
                    resampled[col] = data[col].resample(rule).last()
                else:
                    resampled[col] = data[col].resample(rule).mean()

        return resampled.dropna()

    def calculate_timeframe_features(
        self,
        data: pd.DataFrame,
        timeframe: str
    ) -> Dict[str, float]:
        """
        Calculate technical features for a specific timeframe.

        Args:
            data: Resampled OHLCV data
            timeframe: Timeframe identifier

        Returns:
            Dictionary of feature values
        """
        features = {}

        close_col = 'close' if 'close' in data.columns else 'Close'
        close = data[close_col] if close_col in data.columns else data.iloc[:, 0]

        if len(close) < 5:
            return features

        # Momentum features
        features['momentum_5'] = (close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else 0
        features['momentum_10'] = (close.iloc[-1] / close.iloc[-10] - 1) if len(close) >= 10 else 0
        features['momentum_20'] = (close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0

        # Moving averages
        ma_5 = close.rolling(5).mean()
        ma_10 = close.rolling(10).mean()
        ma_20 = close.rolling(20).mean()

        features['price_vs_ma5'] = (close.iloc[-1] / ma_5.iloc[-1] - 1) if not pd.isna(ma_5.iloc[-1]) else 0
        features['price_vs_ma10'] = (close.iloc[-1] / ma_10.iloc[-1] - 1) if not pd.isna(ma_10.iloc[-1]) else 0
        features['price_vs_ma20'] = (close.iloc[-1] / ma_20.iloc[-1] - 1) if not pd.isna(ma_20.iloc[-1]) else 0

        # MA crossover signals
        if not pd.isna(ma_5.iloc[-1]) and not pd.isna(ma_10.iloc[-1]):
            features['ma_crossover'] = 1 if ma_5.iloc[-1] > ma_10.iloc[-1] else -1
        else:
            features['ma_crossover'] = 0

        # Volatility
        returns = close.pct_change().dropna()
        if len(returns) >= 10:
            features['volatility'] = returns.iloc[-10:].std() * np.sqrt(252)
        else:
            features['volatility'] = 0

        # RSI calculation
        if len(returns) >= 14:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.inf)
            rsi = 100 - (100 / (1 + rs))
            features['rsi'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        else:
            features['rsi'] = 50

        # Trend strength (ADX simplified)
        if len(close) >= 14:
            high_col = 'high' if 'high' in data.columns else 'High'
            low_col = 'low' if 'low' in data.columns else 'Low'

            if high_col in data.columns and low_col in data.columns:
                tr = pd.DataFrame({
                    'hl': data[high_col] - data[low_col],
                    'hc': abs(data[high_col] - close.shift(1)),
                    'lc': abs(data[low_col] - close.shift(1))
                }).max(axis=1)
                atr = tr.rolling(14).mean()
                features['trend_strength'] = (atr.iloc[-1] / close.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0
            else:
                features['trend_strength'] = 0
        else:
            features['trend_strength'] = 0

        return features

    def generate_timeframe_signal(
        self,
        data: pd.DataFrame,
        timeframe: str
    ) -> TimeframeSignal:
        """
        Generate trading signal for a specific timeframe.

        Args:
            data: OHLCV data (will be resampled if needed)
            timeframe: Target timeframe

        Returns:
            TimeframeSignal with direction, strength, and confidence
        """
        # Resample data to target timeframe
        resampled = self.resample_data(data, timeframe)

        if len(resampled) < 10:
            return TimeframeSignal(
                timeframe=timeframe,
                direction=0,
                strength=0,
                confidence=0,
                features={}
            )

        # Calculate features
        features = self.calculate_timeframe_features(resampled, timeframe)

        # Generate signal from features
        signal_components = []

        # Momentum signal
        if 'momentum_10' in features:
            momentum_signal = np.tanh(features['momentum_10'] * 10)  # Scale and bound
            signal_components.append(momentum_signal * 0.3)

        # MA crossover signal
        if 'ma_crossover' in features:
            signal_components.append(features['ma_crossover'] * 0.2)

        # Price vs MA signal
        if 'price_vs_ma20' in features:
            ma_signal = np.tanh(features['price_vs_ma20'] * 5)
            signal_components.append(ma_signal * 0.25)

        # RSI signal (mean reversion)
        if 'rsi' in features:
            rsi = features['rsi']
            if rsi > 70:
                rsi_signal = -((rsi - 70) / 30)  # Overbought
            elif rsi < 30:
                rsi_signal = ((30 - rsi) / 30)   # Oversold
            else:
                rsi_signal = 0
            signal_components.append(rsi_signal * 0.25)

        # Combine signals
        if signal_components:
            direction = np.clip(sum(signal_components), -1, 1)
            strength = abs(direction)

            # Confidence based on feature agreement
            positive_signals = sum(1 for s in signal_components if s > 0)
            negative_signals = sum(1 for s in signal_components if s < 0)
            agreement = max(positive_signals, negative_signals) / len(signal_components)
            confidence = agreement * strength
        else:
            direction = 0
            strength = 0
            confidence = 0

        return TimeframeSignal(
            timeframe=timeframe,
            direction=direction,
            strength=strength,
            confidence=confidence,
            features=features
        )

    def calculate_timeframe_agreement(
        self,
        signals: Dict[str, TimeframeSignal]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate agreement score across all timeframes.

        Args:
            signals: Dictionary of timeframe signals

        Returns:
            Tuple of (agreement_score, weight_adjustments)
        """
        if not signals:
            return 0.0, {}

        # Get directions
        directions = [s.direction for s in signals.values() if s.confidence > 0.1]

        if len(directions) < 2:
            return 0.5, {}

        # Calculate agreement
        positive_count = sum(1 for d in directions if d > 0)
        negative_count = sum(1 for d in directions if d < 0)
        neutral_count = sum(1 for d in directions if abs(d) <= 0.1)

        total = len(directions)
        max_agreement = max(positive_count, negative_count)
        agreement_score = max_agreement / total

        # Weight adjustments based on agreement
        weight_adjustments = {}

        if agreement_score >= self.agreement_threshold:
            # High agreement - boost weights of agreeing timeframes
            dominant_direction = 1 if positive_count >= negative_count else -1

            for tf, signal in signals.items():
                if signal.direction * dominant_direction > 0:
                    weight_adjustments[tf] = 1.2  # 20% boost
                else:
                    weight_adjustments[tf] = 0.8  # 20% reduction
        else:
            # Low agreement - reduce overall confidence
            for tf in signals.keys():
                weight_adjustments[tf] = 0.9

        return agreement_score, weight_adjustments

    def get_volatility_regime(self, data: pd.DataFrame) -> str:
        """
        Determine current volatility regime.

        Args:
            data: OHLCV data

        Returns:
            Volatility regime: 'low', 'normal', 'high', or 'crisis'
        """
        close_col = 'close' if 'close' in data.columns else 'Close'
        close = data[close_col] if close_col in data.columns else data.iloc[:, 0]

        if len(close) < 30:
            return 'normal'

        # Calculate realized volatility
        returns = close.pct_change().dropna()
        recent_vol = returns.iloc[-20:].std() * np.sqrt(252)
        historical_vol = returns.std() * np.sqrt(252)

        # Percentile-based regime
        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1

        if vol_ratio < 0.7:
            return 'low'
        elif vol_ratio < 1.3:
            return 'normal'
        elif vol_ratio < 2.0:
            return 'high'
        else:
            return 'crisis'

    def generate_multi_timeframe_signal(
        self,
        ticker: str,
        data: pd.DataFrame,
        use_base_ensemble: bool = False
    ) -> MultiTimeframeResult:
        """
        Generate combined signal from all timeframes.

        Args:
            ticker: Stock ticker symbol
            data: OHLCV data (highest available frequency)
            use_base_ensemble: Whether to use base ensemble for signals

        Returns:
            MultiTimeframeResult with combined signal and metadata
        """
        # Generate signals for each timeframe
        timeframe_signals = {}

        for tf in self.timeframes.keys():
            if use_base_ensemble and self.base_ensemble is not None:
                # Use base ensemble (if integrated)
                try:
                    resampled = self.resample_data(data, tf)
                    signal = self.base_ensemble.predict(resampled)
                    timeframe_signals[tf] = TimeframeSignal(
                        timeframe=tf,
                        direction=signal.get('direction', 0),
                        strength=signal.get('strength', 0),
                        confidence=signal.get('confidence', 0),
                        features=signal.get('features', {})
                    )
                except Exception:
                    timeframe_signals[tf] = self.generate_timeframe_signal(data, tf)
            else:
                timeframe_signals[tf] = self.generate_timeframe_signal(data, tf)

        # Get volatility regime and adjust weights
        volatility_regime = self.get_volatility_regime(data)

        if self.volatility_adjustment:
            current_weights = self.volatility_adjustments.get(
                volatility_regime,
                self.volatility_adjustments['normal']
            )
        else:
            current_weights = {tf: config['weight'] for tf, config in self.timeframes.items()}

        # Calculate agreement and get weight adjustments
        agreement_score, weight_adjustments = self.calculate_timeframe_agreement(timeframe_signals)

        # Apply weight adjustments
        adjusted_weights = {}
        for tf in self.timeframes.keys():
            base_weight = current_weights.get(tf, 0.25)
            adjustment = weight_adjustments.get(tf, 1.0)
            adjusted_weights[tf] = base_weight * adjustment

        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {tf: w / total_weight for tf, w in adjusted_weights.items()}

        # Combine signals
        combined_direction = 0
        combined_confidence = 0

        for tf, signal in timeframe_signals.items():
            weight = adjusted_weights.get(tf, 0.25)
            combined_direction += signal.direction * weight
            combined_confidence += signal.confidence * weight

        # Adjust confidence based on agreement
        combined_confidence *= (0.5 + 0.5 * agreement_score)

        # Generate recommendation
        if combined_confidence < 0.3:
            recommendation = 'HOLD'
            position_multiplier = 0.0
        elif combined_direction > 0.2:
            if combined_confidence > 0.6:
                recommendation = 'STRONG_BUY'
                position_multiplier = 1.0
            else:
                recommendation = 'BUY'
                position_multiplier = 0.7
        elif combined_direction < -0.2:
            if combined_confidence > 0.6:
                recommendation = 'STRONG_SELL'
                position_multiplier = 1.0
            else:
                recommendation = 'SELL'
                position_multiplier = 0.7
        else:
            recommendation = 'HOLD'
            position_multiplier = 0.3

        # Adjust position for volatility regime
        regime_multipliers = {
            'low': 1.2,
            'normal': 1.0,
            'high': 0.6,
            'crisis': 0.2
        }
        position_multiplier *= regime_multipliers.get(volatility_regime, 1.0)

        return MultiTimeframeResult(
            combined_signal=combined_direction,
            combined_confidence=combined_confidence,
            agreement_score=agreement_score,
            timeframe_signals=timeframe_signals,
            recommendation=recommendation,
            position_multiplier=min(position_multiplier, 1.0)
        )

    def update_performance(
        self,
        timeframe: str,
        predicted_direction: float,
        actual_return: float
    ) -> None:
        """
        Update performance tracking for a timeframe.

        Args:
            timeframe: Timeframe identifier
            predicted_direction: Predicted direction (-1 to 1)
            actual_return: Actual return
        """
        if timeframe not in self.timeframe_performance:
            return

        perf = self.timeframe_performance[timeframe]

        # Check if prediction was correct
        correct = (predicted_direction > 0 and actual_return > 0) or \
                  (predicted_direction < 0 and actual_return < 0)

        perf['total'] += 1
        if correct:
            perf['correct'] += 1

        # Track recent accuracy (last 50 predictions)
        perf['recent_accuracy'].append(1 if correct else 0)
        if len(perf['recent_accuracy']) > 50:
            perf['recent_accuracy'].pop(0)

    def get_timeframe_statistics(self) -> Dict[str, Dict]:
        """
        Get performance statistics for each timeframe.

        Returns:
            Dictionary of statistics per timeframe
        """
        stats = {}

        for tf, perf in self.timeframe_performance.items():
            if perf['total'] > 0:
                overall_accuracy = perf['correct'] / perf['total']
                recent_accuracy = np.mean(perf['recent_accuracy']) if perf['recent_accuracy'] else 0

                stats[tf] = {
                    'total_predictions': perf['total'],
                    'correct_predictions': perf['correct'],
                    'overall_accuracy': overall_accuracy,
                    'recent_accuracy': recent_accuracy,
                    'current_weight': self.timeframes[tf]['weight']
                }
            else:
                stats[tf] = {
                    'total_predictions': 0,
                    'correct_predictions': 0,
                    'overall_accuracy': 0,
                    'recent_accuracy': 0,
                    'current_weight': self.timeframes[tf]['weight']
                }

        return stats


class AdaptiveMultiTimeframeEnsemble(MultiTimeframeEnsemble):
    """
    Extended Multi-Timeframe Ensemble with adaptive weight adjustment
    based on historical performance of each timeframe.
    """

    def __init__(
        self,
        base_ensemble: Any = None,
        min_samples_for_adaptation: int = 30,
        adaptation_rate: float = 0.1,
        **kwargs
    ):
        """
        Initialize Adaptive Multi-Timeframe Ensemble.

        Args:
            base_ensemble: Base ensemble model
            min_samples_for_adaptation: Minimum samples before adapting weights
            adaptation_rate: Rate of weight adaptation (0-1)
            **kwargs: Additional arguments for parent class
        """
        super().__init__(base_ensemble=base_ensemble, **kwargs)
        self.min_samples_for_adaptation = min_samples_for_adaptation
        self.adaptation_rate = adaptation_rate

    def adapt_weights(self) -> None:
        """
        Adapt timeframe weights based on historical performance.
        """
        # Check if we have enough samples
        total_samples = sum(
            perf['total'] for perf in self.timeframe_performance.values()
        )

        if total_samples < self.min_samples_for_adaptation * len(self.timeframes):
            return

        # Calculate performance scores
        scores = {}
        for tf, perf in self.timeframe_performance.items():
            if perf['total'] >= self.min_samples_for_adaptation:
                # Use recent accuracy with some weight on overall
                recent = np.mean(perf['recent_accuracy']) if perf['recent_accuracy'] else 0.5
                overall = perf['correct'] / perf['total'] if perf['total'] > 0 else 0.5
                scores[tf] = 0.7 * recent + 0.3 * overall
            else:
                scores[tf] = 0.5  # Neutral score

        # Normalize scores to weights
        total_score = sum(scores.values())
        if total_score > 0:
            target_weights = {tf: score / total_score for tf, score in scores.items()}
        else:
            return

        # Gradually adapt weights
        for tf in self.timeframes:
            current = self.timeframes[tf]['weight']
            target = target_weights.get(tf, current)

            # Apply bounded adaptation
            new_weight = current + self.adaptation_rate * (target - current)
            new_weight = max(0.05, min(0.50, new_weight))  # Bound between 5% and 50%

            self.timeframes[tf]['weight'] = new_weight

        # Re-normalize weights
        total = sum(config['weight'] for config in self.timeframes.values())
        for tf in self.timeframes:
            self.timeframes[tf]['weight'] /= total

    def generate_multi_timeframe_signal(
        self,
        ticker: str,
        data: pd.DataFrame,
        use_base_ensemble: bool = False
    ) -> MultiTimeframeResult:
        """
        Generate signal with automatic weight adaptation.
        """
        # Adapt weights before generating signal
        self.adapt_weights()

        # Use parent method for signal generation
        return super().generate_multi_timeframe_signal(ticker, data, use_base_ensemble)
