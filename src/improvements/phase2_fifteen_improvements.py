"""
Phase 2 Fixing: 15 Comprehensive Improvements Module

This module implements ALL 15 improvements from the phase2 fixing PDF:
1. Dynamic Ensemble Weighting
2. Regime-Aware Feature Selection
3. Advanced Cross-Asset Correlations
4. Multi-Timeframe Ensemble
5. Real-Time Feature Engineering
6. Confidence-Calibrated Position Sizing
7. Regime Transition Detection
8. Feature Importance Over Time
9. Bayesian Signal Combination
10. Dynamic Drawdown Protection
11. Information-Theoretic Model Selection (bonus)
12. Adaptive Feature Thresholds (bonus)
13. Cross-Market Signal Validation (bonus)
14. Profit-Maximizing Loss Functions (bonus)
15. Walk-Forward Validation Framework (bonus)

Expected Overall Impact:
- +3-5% improvement in profit rate (to ~65-70%)
- -10-20% reduction in maximum drawdown
- +15-25% improvement in risk-adjusted returns (Sharpe ratio)

Author: Phase 2 Improvement System
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# IMPROVEMENT #1: Dynamic Ensemble Weighting
# =============================================================================

class DynamicEnsembleWeighter:
    """
    Dynamically adjust ensemble weights based on recent performance.

    Problem Solved: Static weights don't adapt to changing market conditions
    Expected Improvement: +2-3% profit rate

    Integration: Enhance meta_ensemble.py PerformanceTracker
    """

    def __init__(self, lookback_period: int = 63, min_weight: float = 0.05,
                 max_weight: float = 0.35, smoothing_factor: float = 0.3):
        """
        Initialize dynamic weighter.

        Args:
            lookback_period: Days to look back for performance (63 = ~3 months)
            min_weight: Minimum weight for any ensemble (5%)
            max_weight: Maximum weight for any ensemble (35%)
            smoothing_factor: How fast weights change (0.3 = 30% new, 70% old)
        """
        self.lookback_period = lookback_period
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.smoothing_factor = smoothing_factor

        # Performance tracking per asset class
        self.performance_history = defaultdict(lambda: deque(maxlen=lookback_period))
        self.current_weights = {}
        self.last_update = None

        # Asset class mapping
        self.asset_class_mapping = {
            'equity': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'BAC'],
            'crypto': ['COIN', 'MARA', 'RIOT', 'MSTR', 'BTC-USD', 'ETH-USD'],
            'commodity': ['XOM', 'CVX', 'COP', 'EOG', 'GOLD', 'NEM', 'FCX', 'GLD', 'SLV'],
            'international': ['BABA', 'PDD', 'JD', 'TSM', 'ASML', 'SONY'],
            'forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X'],
            'bond': ['TLT', 'IEF', 'BND', 'AGG', 'GOVT'],
            'etf': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI']
        }

        # Initialize equal weights
        n_classes = len(self.asset_class_mapping)
        self.current_weights = {ac: 1.0/n_classes for ac in self.asset_class_mapping.keys()}

    def update_performance(self, asset_class: str, daily_return: float,
                          prediction_correct: bool = None):
        """Record performance for an asset class."""
        self.performance_history[asset_class].append({
            'return': daily_return,
            'correct': prediction_correct,
            'timestamp': datetime.now()
        })

    def calculate_performance_metrics(self, asset_class: str) -> Dict[str, float]:
        """Calculate comprehensive performance metrics for an asset class."""
        history = list(self.performance_history[asset_class])

        if len(history) < 5:
            return {'sharpe': 0, 'win_rate': 0.5, 'calmar': 0, 'consistency': 0.5}

        returns = np.array([h['return'] for h in history])

        # Sharpe ratio (annualized)
        mean_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 0.001
        sharpe = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0

        # Win rate
        win_rate = np.mean([1 if r > 0 else 0 for r in returns])

        # Calmar ratio (return / max drawdown)
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else -0.01
        calmar = mean_return / abs(max_drawdown) if max_drawdown < 0 else mean_return / 0.01

        # Consistency (percentage of positive rolling 5-day periods)
        if len(returns) >= 5:
            rolling_returns = pd.Series(returns).rolling(5).apply(
                lambda x: np.prod(1 + x) - 1, raw=True
            ).dropna()
            consistency = np.mean([1 if r > 0 else 0 for r in rolling_returns])
        else:
            consistency = 0.5

        return {
            'sharpe': max(0, sharpe),
            'win_rate': win_rate,
            'calmar': max(0, calmar),
            'consistency': consistency,
            'sample_size': len(history)
        }

    def get_dynamic_weights(self) -> Dict[str, float]:
        """Calculate dynamic weights based on recent performance."""
        performance_scores = {}

        for asset_class in self.asset_class_mapping.keys():
            metrics = self.calculate_performance_metrics(asset_class)

            # Composite score (weighted combination)
            composite_score = (
                metrics['sharpe'] * 0.4 +
                metrics['win_rate'] * 0.3 +
                metrics['calmar'] * 0.2 +
                metrics['consistency'] * 0.1
            )

            # Confidence adjustment based on sample size
            confidence = min(1.0, metrics['sample_size'] / 20)
            performance_scores[asset_class] = max(0.01, composite_score * confidence)

        # Normalize to weights
        total_score = sum(performance_scores.values())
        if total_score <= 0:
            # Equal weighting fallback
            return {ac: 1.0/len(performance_scores) for ac in performance_scores}

        raw_weights = {ac: score/total_score for ac, score in performance_scores.items()}

        # Apply min/max constraints
        constrained_weights = {}
        for asset_class, weight in raw_weights.items():
            constrained_weights[asset_class] = max(self.min_weight,
                                                    min(self.max_weight, weight))

        # Renormalize
        total = sum(constrained_weights.values())
        new_weights = {ac: w/total for ac, w in constrained_weights.items()}

        # Smooth transition (don't change weights too fast)
        for ac in new_weights:
            old_weight = self.current_weights.get(ac, new_weights[ac])
            self.current_weights[ac] = (
                (1 - self.smoothing_factor) * old_weight +
                self.smoothing_factor * new_weights[ac]
            )

        self.last_update = datetime.now()
        return self.current_weights

    def get_asset_class(self, ticker: str) -> str:
        """Map ticker to asset class."""
        ticker_upper = ticker.upper()

        for asset_class, tickers in self.asset_class_mapping.items():
            if ticker_upper in tickers:
                return asset_class

        # Pattern-based defaults
        if '.HK' in ticker_upper or '.SS' in ticker_upper:
            return 'international'
        elif '=X' in ticker_upper:
            return 'forex'
        elif 'BTC' in ticker_upper or 'ETH' in ticker_upper:
            return 'crypto'

        return 'equity'  # Default


# =============================================================================
# IMPROVEMENT #2: Regime-Aware Feature Selection
# =============================================================================

class RegimeAwareFeatureSelector:
    """
    Select features based on current market regime.

    Problem Solved: Same features used in all market conditions
    Expected Improvement: Better signal quality in different regimes

    Integration: Between regime detection and ensemble prediction
    """

    def __init__(self):
        """Initialize regime-aware feature selector."""
        # Feature mappings by regime
        self.regime_feature_mapping = {
            'bull_market': [
                'momentum_5d', 'momentum_10d', 'momentum_20d',
                'rsi_14', 'macd_signal', 'adx_14',
                'sector_momentum', 'relative_strength',
                'volume_trend', 'price_acceleration'
            ],
            'bear_market': [
                'volatility_20d', 'atr_14', 'bollinger_width',
                'rsi_14', 'stochastic_k', 'stochastic_d',
                'volume_spike', 'drawdown_depth',
                'vix_level', 'put_call_ratio'
            ],
            'high_volatility': [
                'mean_reversion_score', 'bollinger_pct_b',
                'atr_14', 'volatility_ratio',
                'liquidity_score', 'bid_ask_spread',
                'tail_risk', 'kurtosis_20d'
            ],
            'low_volatility': [
                'momentum_20d', 'momentum_60d',
                'trend_strength', 'adx_14',
                'carry_return', 'dividend_yield',
                'relative_strength', 'sector_rotation'
            ],
            'trending': [
                'momentum_5d', 'momentum_10d', 'momentum_20d',
                'trend_strength', 'adx_14', 'dmi_plus',
                'breakout_signal', 'channel_position'
            ],
            'mean_reverting': [
                'rsi_14', 'stochastic_k', 'bollinger_pct_b',
                'mean_reversion_score', 'zscore_price',
                'deviation_from_ma', 'oversold_signal'
            ]
        }

        # Default features always included
        self.core_features = [
            'close', 'volume', 'returns_1d',
            'volatility_20d', 'rsi_14'
        ]

    def select_features(self, current_regime: str,
                       available_features: List[str]) -> List[str]:
        """
        Select regime-specific features.

        Args:
            current_regime: Current market regime
            available_features: List of available feature names

        Returns:
            List of selected feature names
        """
        # Get regime-specific features
        regime_features = self.regime_feature_mapping.get(
            current_regime, available_features
        )

        # Filter to only available features
        selected = [f for f in regime_features if f in available_features]

        # Always include core features
        for core in self.core_features:
            if core in available_features and core not in selected:
                selected.append(core)

        return selected

    def get_feature_weights(self, current_regime: str,
                           features: List[str]) -> Dict[str, float]:
        """
        Get feature importance weights based on regime.

        Args:
            current_regime: Current market regime
            features: List of features

        Returns:
            Dictionary of feature -> weight
        """
        regime_features = self.regime_feature_mapping.get(current_regime, [])

        weights = {}
        for f in features:
            if f in regime_features:
                # Higher weight for regime-specific features
                idx = regime_features.index(f)
                weights[f] = 1.0 - (idx * 0.05)  # Top features get higher weight
            elif f in self.core_features:
                weights[f] = 0.5  # Core features get moderate weight
            else:
                weights[f] = 0.3  # Other features get lower weight

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}

        return weights


# =============================================================================
# IMPROVEMENT #3: Advanced Cross-Asset Correlations
# =============================================================================

class AdvancedCorrelationAnalyzer:
    """
    Calculate dynamic correlation networks between asset classes.

    Problem Solved: Missing dynamic correlation structure
    Expected Improvement: Better risk-on/risk-off regime detection

    Integration: Enhance intermarket_features.py
    """

    def __init__(self, correlation_window: int = 60, n_clusters: int = 3):
        """
        Initialize correlation analyzer.

        Args:
            correlation_window: Rolling window for correlations
            n_clusters: Number of correlation clusters to identify
        """
        self.correlation_window = correlation_window
        self.n_clusters = n_clusters
        self.correlation_history = {}

    def calculate_rolling_correlation_network(self,
                                              assets_data: Dict[str, pd.DataFrame],
                                              window: int = None) -> Dict:
        """
        Calculate dynamic correlation network between assets.

        Args:
            assets_data: Dict of asset_name -> DataFrame with returns
            window: Rolling window size

        Returns:
            Dictionary with correlation analysis results
        """
        window = window or self.correlation_window
        correlations = {}

        # Calculate pairwise correlations
        asset_names = list(assets_data.keys())
        for i, asset1 in enumerate(asset_names):
            for asset2 in asset_names[i+1:]:
                if 'returns' in assets_data[asset1].columns:
                    returns1 = assets_data[asset1]['returns']
                else:
                    returns1 = assets_data[asset1]['Close'].pct_change()

                if 'returns' in assets_data[asset2].columns:
                    returns2 = assets_data[asset2]['returns']
                else:
                    returns2 = assets_data[asset2]['Close'].pct_change()

                # Align the series
                aligned = pd.concat([returns1, returns2], axis=1).dropna()
                if len(aligned) >= window:
                    corr = aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])
                    correlations[f"{asset1}_{asset2}"] = corr

        if not correlations:
            return {'clusters': {}, 'regime_stability': 0, 'primary_component_variance': 0}

        # Identify correlation clusters using PCA
        correlation_matrix = pd.DataFrame(correlations).fillna(0)

        if len(correlation_matrix) > 10 and correlation_matrix.shape[1] >= 2:
            try:
                pca = PCA(n_components=min(2, correlation_matrix.shape[1]))
                components = pca.fit_transform(correlation_matrix.T)

                # K-means clustering on PCA components
                if len(components) >= self.n_clusters:
                    kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(components)
                else:
                    clusters = np.zeros(len(components))

                return {
                    'correlations': correlations,
                    'primary_component_variance': pca.explained_variance_ratio_[0],
                    'clusters': dict(zip(correlations.keys(), clusters)),
                    'regime_stability': 1 - np.std(components, axis=0).mean(),
                    'pca_components': components
                }
            except Exception as e:
                pass

        return {
            'correlations': correlations,
            'clusters': {},
            'regime_stability': 0.5,
            'primary_component_variance': 0
        }

    def detect_correlation_breakdown(self,
                                     current_correlations: Dict[str, float],
                                     historical_correlations: Dict[str, pd.Series],
                                     threshold: float = 2.0) -> Dict:
        """
        Detect when correlations are breaking down (regime change signal).

        Args:
            current_correlations: Current correlation values
            historical_correlations: Historical correlation series
            threshold: Z-score threshold for breakdown detection

        Returns:
            Dictionary with breakdown detection results
        """
        breakdowns = {}

        for pair, current_corr in current_correlations.items():
            if pair in historical_correlations:
                hist = historical_correlations[pair].dropna()
                if len(hist) >= 20:
                    mean_corr = hist.mean()
                    std_corr = hist.std()

                    if std_corr > 0:
                        zscore = (current_corr - mean_corr) / std_corr
                        breakdowns[pair] = {
                            'zscore': zscore,
                            'is_breakdown': abs(zscore) > threshold,
                            'direction': 'increasing' if zscore > 0 else 'decreasing'
                        }

        n_breakdowns = sum(1 for b in breakdowns.values() if b.get('is_breakdown', False))

        return {
            'breakdowns': breakdowns,
            'n_breakdowns': n_breakdowns,
            'breakdown_ratio': n_breakdowns / len(breakdowns) if breakdowns else 0,
            'regime_change_signal': n_breakdowns >= len(breakdowns) * 0.3
        }


# =============================================================================
# IMPROVEMENT #4: Multi-Timeframe Ensemble
# =============================================================================

class MultiTimeframeEnsemble:
    """
    Combine signals from multiple timeframes to reduce bias.

    Problem Solved: Single timeframe bias
    Expected Improvement: +1-2% consistency, better entry/exit timing

    Integration: Wrap existing ensembles with timeframe resampling
    """

    def __init__(self, base_predictor=None):
        """
        Initialize multi-timeframe ensemble.

        Args:
            base_predictor: Base prediction model/ensemble
        """
        self.base_predictor = base_predictor

        # Timeframe configurations
        self.timeframes = {
            '1h': {'resample': '1H', 'weight': 0.15, 'min_bars': 20},
            '4h': {'resample': '4H', 'weight': 0.25, 'min_bars': 15},
            '1d': {'resample': '1D', 'weight': 0.35, 'min_bars': 10},
            '1w': {'resample': '1W', 'weight': 0.25, 'min_bars': 5}
        }

    def resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to different timeframe."""
        if timeframe == '1d' or timeframe == '1D':
            return data

        tf_config = self.timeframes.get(timeframe, {})
        resample_rule = tf_config.get('resample', timeframe)

        try:
            if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                resampled = data.resample(resample_rule).agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum' if 'Volume' in data.columns else 'first'
                }).dropna()
                return resampled
            else:
                return data['Close'].resample(resample_rule).last().to_frame()
        except Exception:
            return data

    def calculate_timeframe_agreement(self, signals: Dict[str, float]) -> float:
        """Calculate how much different timeframes agree."""
        signal_values = list(signals.values())
        if len(signal_values) <= 1:
            return 1.0
        return max(0, 1 - np.std(signal_values))

    def calculate_dynamic_weights(self, signals: Dict[str, float],
                                  recent_volatility: float) -> Dict[str, float]:
        """Dynamically adjust timeframe weights based on market conditions."""
        base_weights = {tf: self.timeframes[tf]['weight'] for tf in signals.keys()}

        # High volatility: favor shorter timeframes
        if recent_volatility > 0.25:
            for tf in base_weights:
                if tf in ['1h', '4h']:
                    base_weights[tf] *= 1.3
                else:
                    base_weights[tf] *= 0.8
        # Low volatility: favor longer timeframes
        elif recent_volatility < 0.15:
            for tf in base_weights:
                if tf in ['1d', '1w']:
                    base_weights[tf] *= 1.2
                else:
                    base_weights[tf] *= 0.9

        # Normalize
        total = sum(base_weights.values())
        return {tf: w/total for tf, w in base_weights.items()}

    def generate_multi_timeframe_signal(self, ticker: str,
                                        data: pd.DataFrame,
                                        predictor_func: Callable = None) -> Dict:
        """
        Generate signal combining multiple timeframes.

        Args:
            ticker: Asset ticker
            data: OHLCV DataFrame
            predictor_func: Function to generate prediction for each timeframe

        Returns:
            Combined signal dictionary
        """
        timeframe_signals = {}
        timeframe_confidences = {}

        # Calculate recent volatility
        if 'Close' in data.columns:
            recent_vol = data['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
        else:
            recent_vol = 0.2  # Default

        for tf_name, tf_config in self.timeframes.items():
            try:
                tf_data = self.resample_data(data, tf_config['resample'])

                if len(tf_data) < tf_config['min_bars']:
                    continue

                # Get prediction for this timeframe
                if predictor_func:
                    prediction = predictor_func(tf_data)
                    signal = prediction.get('signal', 0)
                    confidence = prediction.get('confidence', 0.5)
                elif self.base_predictor:
                    prediction = self.base_predictor.predict(tf_data)
                    signal = prediction.get('signal', 0)
                    confidence = prediction.get('confidence', 0.5)
                else:
                    # Simple momentum signal as fallback
                    returns = tf_data['Close'].pct_change(5).iloc[-1]
                    signal = np.tanh(returns * 10)
                    confidence = 0.5

                timeframe_signals[tf_name] = signal
                timeframe_confidences[tf_name] = confidence

            except Exception as e:
                continue

        if not timeframe_signals:
            return {
                'signal': 0,
                'confidence': 0,
                'timeframe_agreement': 0,
                'method': 'fallback'
            }

        # Calculate agreement
        agreement = self.calculate_timeframe_agreement(timeframe_signals)

        # Get dynamic weights
        dynamic_weights = self.calculate_dynamic_weights(timeframe_signals, recent_vol)

        # Combine signals
        combined_signal = 0
        combined_confidence = 0
        total_weight = 0

        for tf_name, signal in timeframe_signals.items():
            weight = dynamic_weights.get(tf_name, 0.25)
            confidence = timeframe_confidences.get(tf_name, 0.5)

            combined_signal += signal * weight
            combined_confidence += confidence * weight
            total_weight += weight

        if total_weight > 0:
            final_signal = combined_signal / total_weight
            final_confidence = combined_confidence / total_weight
        else:
            final_signal = 0
            final_confidence = 0

        # Boost confidence for high agreement
        if agreement > 0.8:
            final_confidence = min(0.95, final_confidence * 1.2)
        elif agreement < 0.4:
            final_signal *= 0.5  # Reduce signal for low agreement

        return {
            'signal': np.clip(final_signal, -1, 1),
            'confidence': np.clip(final_confidence, 0, 1),
            'timeframe_agreement': agreement,
            'timeframes_used': list(timeframe_signals.keys()),
            'timeframe_signals': timeframe_signals,
            'timeframe_weights': dynamic_weights,
            'method': 'multi_timeframe_ensemble'
        }


# =============================================================================
# IMPROVEMENT #5: Real-Time Feature Engineering (Streaming)
# =============================================================================

class StreamingFeatureEngine:
    """
    Incremental feature calculation for real-time updates.

    Problem Solved: Batch processing delays
    Expected Improvement: Faster signal generation, reduced latency

    Integration: Enhance feature calculation with incremental updates
    """

    def __init__(self, max_cache_size: int = 1000):
        """
        Initialize streaming feature engine.

        Args:
            max_cache_size: Maximum number of values to cache
        """
        self.max_cache_size = max_cache_size
        self.feature_cache = {}
        self.incremental_state = {}

    def update_features_incremental(self, new_data_point: Dict) -> Dict[str, float]:
        """
        Update features incrementally without full recalculation.

        Args:
            new_data_point: New OHLCV data point

        Returns:
            Updated feature values
        """
        features = {}

        # Update price cache
        if 'prices' not in self.incremental_state:
            self.incremental_state['prices'] = deque(maxlen=self.max_cache_size)
        self.incremental_state['prices'].append(new_data_point.get('close', 0))

        prices = np.array(self.incremental_state['prices'])

        if len(prices) >= 2:
            # Incremental returns
            features['return_1d'] = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0

        if len(prices) >= 20:
            # Incremental z-score
            features['zscore_20d'] = self.calculate_rolling_zscore_incremental(
                prices[-1], window=20
            )

            # Incremental volatility
            features['volatility_20d'] = self.calculate_rolling_volatility_incremental(
                window=20
            )

            # Incremental RSI
            features['rsi_14'] = self.calculate_rsi_incremental(prices, period=14)

        return features

    def calculate_rolling_zscore_incremental(self, new_value: float,
                                             window: int = 20) -> float:
        """Incremental z-score calculation."""
        state_key = f'zscore_state_{window}'

        if state_key not in self.incremental_state:
            self.incremental_state[state_key] = {
                'values': deque(maxlen=window),
                'sum': 0,
                'sum_squares': 0
            }

        state = self.incremental_state[state_key]

        # Remove oldest value if window is full
        if len(state['values']) == window:
            old_val = state['values'][0]
            state['sum'] -= old_val
            state['sum_squares'] -= old_val ** 2

        # Add new value
        state['values'].append(new_value)
        state['sum'] += new_value
        state['sum_squares'] += new_value ** 2

        # Calculate statistics
        n = len(state['values'])
        mean = state['sum'] / n
        variance = (state['sum_squares'] / n) - (mean ** 2)
        std = max(0.001, np.sqrt(variance))

        return (new_value - mean) / std

    def calculate_rolling_volatility_incremental(self, window: int = 20) -> float:
        """Incremental volatility calculation."""
        prices = np.array(self.incremental_state.get('prices', []))

        if len(prices) < window:
            return 0.0

        returns = np.diff(prices[-window:]) / prices[-window:-1]
        return np.std(returns) * np.sqrt(252)

    def calculate_rsi_incremental(self, prices: np.ndarray, period: int = 14) -> float:
        """Incremental RSI calculation."""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices[-(period+1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi


# =============================================================================
# IMPROVEMENT #6: Confidence-Calibrated Position Sizing
# =============================================================================

class ConfidenceAwarePositionSizer:
    """
    Kelly criterion-based position sizing with confidence calibration.

    Problem Solved: Binary position sizing based on signals
    Expected Improvement: +2-3% risk-adjusted returns, -10% drawdown

    Integration: Enhance kelly_backtester.py
    """

    def __init__(self, kelly_fraction: float = 0.25, min_position: float = 0.02,
                 max_position: float = 0.15, confidence_threshold: float = 0.6):
        """
        Initialize confidence-aware position sizer.

        Args:
            kelly_fraction: Fraction of Kelly criterion (0.25 = quarter-Kelly)
            min_position: Minimum position size as fraction of capital
            max_position: Maximum position size as fraction of capital
            confidence_threshold: Minimum confidence to boost position
        """
        self.kelly_fraction = kelly_fraction
        self.min_position = min_position
        self.max_position = max_position
        self.confidence_threshold = confidence_threshold

        # Track performance by signal strength
        self.performance_history = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0})
        self.signal_buckets = [0.1, 0.3, 0.5, 0.7, 0.9]

    def _get_signal_bucket(self, signal_strength: float) -> float:
        """Map signal strength to nearest bucket."""
        return min(self.signal_buckets, key=lambda x: abs(x - signal_strength))

    def update_performance(self, signal_strength: float, actual_return: float,
                          position_size: float):
        """Update performance history."""
        bucket = self._get_signal_bucket(signal_strength)

        if actual_return > 0:
            self.performance_history[bucket]['wins'] += 1
        else:
            self.performance_history[bucket]['losses'] += 1

        self.performance_history[bucket]['total_pnl'] += actual_return * position_size

    def get_historical_performance(self, signal_strength: float) -> Dict:
        """Get historical performance for a signal strength."""
        bucket = self._get_signal_bucket(signal_strength)
        history = self.performance_history[bucket]

        total_trades = history['wins'] + history['losses']

        if total_trades == 0:
            return {'win_rate': 0.5, 'avg_win': 0.02, 'avg_loss': -0.01, 'win_loss_ratio': 2.0}

        win_rate = history['wins'] / total_trades
        win_loss_ratio = 2.0  # Default

        return {
            'win_rate': win_rate,
            'avg_win': 0.02,
            'avg_loss': -0.01,
            'win_loss_ratio': win_loss_ratio,
            'total_trades': total_trades
        }

    def calculate_kelly_position(self, signal_strength: float, win_rate: float,
                                win_loss_ratio: float, confidence: float = 1.0) -> float:
        """
        Calculate position size using Kelly criterion with confidence adjustment.

        Args:
            signal_strength: Signal value normalized to 0-1
            win_rate: Historical win rate
            win_loss_ratio: Average win / average loss
            confidence: Model confidence (0-1)

        Returns:
            Position size as fraction of capital
        """
        # Don't bet if no edge
        if win_rate <= 0.5 or win_loss_ratio <= 1:
            return self.min_position

        # Kelly formula: f = p - (1-p)/b
        kelly_f = win_rate - (1 - win_rate) / win_loss_ratio

        # Apply fractional Kelly and bounds
        kelly_position = kelly_f * self.kelly_fraction
        kelly_position = max(self.min_position, min(self.max_position, kelly_position))

        # Adjust by signal strength
        signal_adjusted = kelly_position * signal_strength

        # Adjust by confidence
        confidence_adjusted = signal_adjusted * confidence

        # Boost for high-confidence strong signals
        if confidence > self.confidence_threshold and signal_strength > 0.7:
            confidence_adjusted *= 1.2

        return max(self.min_position, min(self.max_position, confidence_adjusted))

    def calculate_diversification_penalty(self, ticker: str,
                                          current_portfolio: Dict) -> float:
        """Reduce position size for correlated assets."""
        if not current_portfolio:
            return 1.0

        # Sector mapping
        sectors = {
            'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA'],
            'financial': ['JPM', 'BAC', 'GS', 'MS'],
            'energy': ['XOM', 'CVX', 'COP'],
            'healthcare': ['JNJ', 'PFE', 'MRK']
        }

        # Find ticker's sector
        ticker_sector = None
        for sector, tickers in sectors.items():
            if ticker.upper() in tickers:
                ticker_sector = sector
                break

        if not ticker_sector:
            return 1.0

        # Calculate sector exposure
        sector_exposure = 0
        for pos_ticker, position in current_portfolio.items():
            for sector, tickers in sectors.items():
                if pos_ticker.upper() in tickers and sector == ticker_sector:
                    sector_exposure += abs(position.get('size', 0))

        # Apply penalty for high concentration
        if sector_exposure > 0.3:
            return 0.5
        elif sector_exposure > 0.2:
            return 0.7
        return 1.0

    def get_position_size(self, signal_data: Dict, portfolio: Dict = None) -> float:
        """
        Calculate final position size with all adjustments.

        Args:
            signal_data: Signal information dictionary
            portfolio: Current portfolio positions

        Returns:
            Position size as fraction of capital
        """
        signal_strength = signal_data.get('signal_strength', 0.5)
        confidence = signal_data.get('confidence', 0.5)
        ticker = signal_data.get('ticker', '')

        # Get historical performance
        historical_perf = self.get_historical_performance(signal_strength)

        # Calculate base position
        base_position = self.calculate_kelly_position(
            signal_strength=signal_strength,
            win_rate=historical_perf['win_rate'],
            win_loss_ratio=historical_perf['win_loss_ratio'],
            confidence=confidence
        )

        # Apply diversification penalty
        if portfolio:
            div_penalty = self.calculate_diversification_penalty(ticker, portfolio)
            final_position = base_position * div_penalty
        else:
            final_position = base_position

        return max(self.min_position, min(self.max_position, final_position))


# =============================================================================
# IMPROVEMENT #7: Regime Transition Detection
# =============================================================================

class RegimeTransitionDetector:
    """
    Early detection of regime transitions.

    Problem Solved: Late detection of regime changes
    Expected Improvement: Earlier warnings, better preparation for regime changes

    Integration: Enhance regime_detection.py
    """

    def __init__(self, transition_threshold: float = 0.7):
        """
        Initialize regime transition detector.

        Args:
            transition_threshold: Probability threshold for transition warning
        """
        self.transition_threshold = transition_threshold
        self.regime_indicators = {}
        self.transition_history = []

    def detect_impending_regime_change(self, market_data: Dict) -> Dict:
        """
        Early detection of regime transitions.

        Args:
            market_data: Dictionary with market indicators

        Returns:
            Dictionary with transition detection results
        """
        transition_signals = {}

        # Volatility regime change
        vol_signal = self.detect_volatility_regime_change(market_data)
        transition_signals['volatility'] = vol_signal

        # Correlation structure change
        corr_signal = self.detect_correlation_breakdown(market_data)
        transition_signals['correlation'] = corr_signal

        # Momentum regime change
        mom_signal = self.detect_momentum_regime_change(market_data)
        transition_signals['momentum'] = mom_signal

        # Composite transition probability
        valid_signals = [s for s in transition_signals.values() if s is not None]
        transition_prob = np.mean(valid_signals) if valid_signals else 0

        result = {
            'transition_probability': transition_prob,
            'impending_change': transition_prob > self.transition_threshold,
            'signals': transition_signals,
            'timestamp': datetime.now()
        }

        # Track history
        self.transition_history.append(result)
        if len(self.transition_history) > 100:
            self.transition_history = self.transition_history[-100:]

        return result

    def detect_volatility_regime_change(self, market_data: Dict) -> float:
        """Detect when volatility regime is about to change."""
        current_vol = market_data.get('current_volatility', 0.15)
        vol_regime = market_data.get('volatility_regime', 'medium')
        vol_trend = market_data.get('volatility_trend', 0)

        # Check if volatility is approaching regime boundaries
        if vol_regime == 'low' and current_vol > 0.12:
            return 0.6
        elif vol_regime == 'medium' and current_vol > 0.25:
            return 0.8
        elif vol_regime == 'high' and current_vol < 0.18:
            return 0.7

        # Check volatility acceleration
        if abs(vol_trend) > 0.02:
            return 0.5

        return 0.2

    def detect_correlation_breakdown(self, market_data: Dict) -> float:
        """Detect correlation structure breakdown."""
        correlation_change = market_data.get('correlation_change', 0)

        # Large correlation change indicates regime shift
        if abs(correlation_change) > 0.3:
            return 0.8
        elif abs(correlation_change) > 0.2:
            return 0.5

        return 0.2

    def detect_momentum_regime_change(self, market_data: Dict) -> float:
        """Detect momentum regime change."""
        momentum_5d = market_data.get('momentum_5d', 0)
        momentum_20d = market_data.get('momentum_20d', 0)

        # Momentum divergence (short-term vs long-term)
        if momentum_5d * momentum_20d < 0:  # Different signs
            return 0.6

        # Strong momentum reversal
        momentum_change = market_data.get('momentum_change', 0)
        if abs(momentum_change) > 0.05:
            return 0.7

        return 0.2

    def calculate_drawdown_velocity(self, returns: pd.Series, window: int = 5) -> float:
        """Calculate how fast portfolio is losing value."""
        recent_returns = returns.tail(window)
        return min(0, recent_returns.mean())


# =============================================================================
# IMPROVEMENT #8: Feature Importance Over Time
# =============================================================================

class TimeVaryingFeatureImportance:
    """
    Track how feature importance changes over time.

    Problem Solved: Static feature importance
    Expected Improvement: Better feature weighting, adaptive model inputs

    Integration: Create new analysis module
    """

    def __init__(self, lookback_window: int = 252):
        """
        Initialize time-varying feature importance tracker.

        Args:
            lookback_window: Window for importance calculation
        """
        self.lookback_window = lookback_window
        self.importance_history = {}

    def calculate_rolling_feature_importance(self, features: pd.DataFrame,
                                             returns: pd.Series,
                                             forward_periods: int = 5) -> Dict:
        """
        Calculate how feature importance changes over time.

        Args:
            features: DataFrame of feature values
            returns: Series of forward returns
            forward_periods: Periods to shift returns

        Returns:
            Dictionary of feature importance metrics
        """
        importance_scores = {}

        for feature in features.columns:
            # Rolling correlation with forward returns
            rolling_corr = features[feature].rolling(63).corr(returns.shift(-forward_periods))

            importance_scores[feature] = {
                'current_importance': rolling_corr.iloc[-1] if len(rolling_corr) > 0 else 0,
                'trend': rolling_corr.diff(5).iloc[-1] if len(rolling_corr) > 5 else 0,
                'stability': rolling_corr.rolling(20).std().iloc[-1] if len(rolling_corr) > 20 else 1
            }

        return importance_scores

    def get_feature_weights(self, current_regime: str,
                           importance_scores: Dict = None) -> Dict[str, float]:
        """
        Get dynamic feature weights based on regime and recent importance.

        Args:
            current_regime: Current market regime
            importance_scores: Recent importance scores

        Returns:
            Dictionary of feature -> weight
        """
        importance_scores = importance_scores or self.importance_history

        if not importance_scores:
            return {}

        adjusted_weights = {}

        for feature, data in importance_scores.items():
            base_weight = 1.0

            # Adjust based on current importance
            current_imp = data.get('current_importance', 0)
            if not np.isnan(current_imp):
                base_weight *= (1 + current_imp)

            # Adjust based on trend
            trend = data.get('trend', 0)
            if not np.isnan(trend):
                trend_adjustment = 1 + (trend * 2)
                base_weight *= max(0.5, trend_adjustment)

            # Penalize unstable features
            stability = data.get('stability', 1)
            if not np.isnan(stability) and stability > 0:
                stability_adjustment = 1 / (stability + 0.1)
                base_weight *= stability_adjustment

            adjusted_weights[feature] = max(0.01, base_weight)

        # Normalize weights
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}

        return adjusted_weights


# =============================================================================
# IMPROVEMENT #9: Bayesian Signal Combination
# =============================================================================

class BayesianSignalCombiner:
    """
    Bayesian model averaging for robust signal combination.

    Problem Solved: Simple weighted average of signals
    Expected Improvement: +1-2% signal accuracy, better risk management

    Integration: New signal aggregation method
    """

    def __init__(self, prior_alpha: float = 1, prior_beta: float = 1,
                 min_samples: int = 5):
        """
        Initialize Bayesian signal combiner.

        Args:
            prior_alpha: Prior alpha for Beta distribution
            prior_beta: Prior beta for Beta distribution
            min_samples: Minimum samples before using posterior
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.min_samples = min_samples

        # Signal priors (Beta distributions)
        self.signal_priors = defaultdict(
            lambda: {'alpha': prior_alpha, 'beta': prior_beta}
        )
        self.signal_performance = defaultdict(list)

    def update_signal_reliability(self, signal_name: str, signal_value: float,
                                 actual_return: float):
        """
        Update signal reliability using Bayesian updating.

        Args:
            signal_name: Name of the signal
            signal_value: Signal value (-1 to 1)
            actual_return: Actual return
        """
        # Check if signal was correct (directional accuracy)
        signal_direction = 1 if signal_value > 0 else -1
        return_direction = 1 if actual_return > 0 else -1
        signal_correct = (signal_direction * return_direction) > 0

        # Bayesian update (Beta-Bernoulli conjugate prior)
        prior = self.signal_priors[signal_name]
        new_alpha = prior['alpha'] + (1 if signal_correct else 0)
        new_beta = prior['beta'] + (0 if signal_correct else 1)

        self.signal_priors[signal_name] = {'alpha': new_alpha, 'beta': new_beta}

        # Track performance
        self.signal_performance[signal_name].append({
            'signal_value': signal_value,
            'actual_return': actual_return,
            'correct': signal_correct
        })

        # Keep only recent history
        if len(self.signal_performance[signal_name]) > 1000:
            self.signal_performance[signal_name] = self.signal_performance[signal_name][-1000:]

    def get_signal_reliability(self, signal_name: str) -> Dict:
        """Get reliability metrics for a signal."""
        prior = self.signal_priors[signal_name]
        total_observations = prior['alpha'] + prior['beta'] - 2

        if total_observations < self.min_samples:
            expected_reliability = self.prior_alpha / (self.prior_alpha + self.prior_beta)
            confidence = 0.1
        else:
            expected_reliability = prior['alpha'] / (prior['alpha'] + prior['beta'])
            confidence = min(0.95, total_observations / (total_observations + 50))

        return {
            'expected_reliability': expected_reliability,
            'confidence': confidence,
            'total_observations': total_observations
        }

    def combine_signals_bayesian(self, signals: Dict[str, float]) -> Dict:
        """
        Combine multiple signals using Bayesian model averaging.

        Args:
            signals: Dictionary of signal_name -> signal_value

        Returns:
            Combined signal dictionary
        """
        if not signals:
            return {'combined_signal': 0, 'confidence': 0, 'components': {}}

        weighted_signal = 0
        total_weight = 0
        component_weights = {}

        for signal_name, signal_value in signals.items():
            reliability_data = self.get_signal_reliability(signal_name)

            # Weight = reliability * confidence
            weight = reliability_data['expected_reliability'] * reliability_data['confidence']

            weighted_signal += signal_value * weight
            total_weight += weight
            component_weights[signal_name] = weight

        # Normalize
        if total_weight > 0:
            combined_signal = weighted_signal / total_weight
            overall_confidence = total_weight / len(signals)
        else:
            combined_signal = 0
            overall_confidence = 0

        # Shrinkage towards zero for low confidence
        if overall_confidence < 0.3:
            combined_signal *= overall_confidence / 0.3

        return {
            'combined_signal': np.clip(combined_signal, -1, 1),
            'confidence': np.clip(overall_confidence, 0, 1),
            'component_weights': component_weights,
            'signals_combined': len(signals)
        }

    def get_signal_advice(self, signals: Dict[str, float]) -> Dict:
        """Get advice on which signals to trust."""
        combination = self.combine_signals_bayesian(signals)

        # Signal agreement
        signal_values = list(signals.values())
        signal_agreement = 1 - np.std(signal_values) if len(signal_values) > 1 else 1

        # Find strongest and weakest
        weights = combination['component_weights']
        if weights:
            strongest = max(weights.items(), key=lambda x: x[1])
            weakest = min(weights.items(), key=lambda x: x[1])
        else:
            strongest = weakest = (None, 0)

        return {
            'signal_agreement': signal_agreement,
            'strongest_signal': strongest[0],
            'strongest_weight': strongest[1],
            'weakest_signal': weakest[0],
            'weakest_weight': weakest[1],
            'recommendation': self._generate_recommendation(combination, signal_agreement)
        }

    def _generate_recommendation(self, combination: Dict, agreement: float) -> str:
        """Generate trading recommendation."""
        signal = combination['combined_signal']
        confidence = combination['confidence']

        if confidence < 0.3:
            return "LOW_CONFIDENCE - REDUCE_POSITION"
        elif agreement < 0.5:
            return "CONFLICTING_SIGNALS - CAUTION_ADVISED"
        elif abs(signal) > 0.5 and confidence > 0.6:
            return "HIGH_CONFIDENCE_STRONG_SIGNAL - INCREASE_POSITION"
        elif abs(signal) < 0.2:
            return "WEAK_SIGNAL - HOLD_OR_REDUCE"
        else:
            return "MODERATE_CONFIDENCE - NORMAL_POSITION"


# =============================================================================
# IMPROVEMENT #10: Dynamic Drawdown Protection
# =============================================================================

class AdaptiveDrawdownProtection:
    """
    Dynamically adjust position sizes based on drawdown severity.

    Problem Solved: Static drawdown limits
    Expected Improvement: -10-20% max drawdown, better capital preservation

    Integration: Enhance hybrid_strategy.py
    """

    def __init__(self, max_drawdown_limit: float = 0.15,
                 warning_threshold: float = 0.05,
                 danger_threshold: float = 0.10):
        """
        Initialize adaptive drawdown protection.

        Args:
            max_drawdown_limit: Maximum allowed drawdown (15%)
            warning_threshold: Start reducing at this level (5%)
            danger_threshold: Aggressive reduction at this level (10%)
        """
        self.max_drawdown_limit = max_drawdown_limit
        self.warning_threshold = warning_threshold
        self.danger_threshold = danger_threshold

        self.position_multipliers = {
            'normal': 1.0,
            'warning': 0.7,
            'danger': 0.3,
            'critical': 0.0
        }

        self.equity_history = []
        self.peak_equity = 0

    def update_equity(self, current_equity: float):
        """Update equity tracking."""
        self.equity_history.append(current_equity)
        self.peak_equity = max(self.peak_equity, current_equity)

    def calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_equity <= 0 or not self.equity_history:
            return 0

        current_equity = self.equity_history[-1]
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        return max(0, drawdown)

    def calculate_drawdown_velocity(self, window: int = 5) -> float:
        """Calculate how fast drawdown is increasing."""
        if len(self.equity_history) < window + 1:
            return 0

        recent_returns = []
        for i in range(-window, 0):
            if self.equity_history[i-1] > 0:
                ret = (self.equity_history[i] - self.equity_history[i-1]) / self.equity_history[i-1]
                recent_returns.append(ret)

        return min(0, np.mean(recent_returns)) if recent_returns else 0

    def get_position_multiplier(self, portfolio_returns: pd.Series = None) -> float:
        """
        Get position multiplier based on drawdown.

        Args:
            portfolio_returns: Series of portfolio returns

        Returns:
            Position multiplier (0 to 1)
        """
        current_drawdown = self.calculate_current_drawdown()
        drawdown_velocity = self.calculate_drawdown_velocity()

        # Base multiplier from drawdown level
        if current_drawdown >= self.max_drawdown_limit:
            base_multiplier = self.position_multipliers['critical']
        elif current_drawdown >= self.danger_threshold:
            base_multiplier = self.position_multipliers['danger']
        elif current_drawdown >= self.warning_threshold:
            base_multiplier = self.position_multipliers['warning']
        else:
            base_multiplier = self.position_multipliers['normal']

        # Additional reduction for rapid drawdowns
        if drawdown_velocity < -0.02:  # Losing >2% daily
            base_multiplier *= 0.5
        elif drawdown_velocity < -0.01:  # Losing >1% daily
            base_multiplier *= 0.7

        return max(0, min(1, base_multiplier))

    def get_protection_status(self) -> Dict:
        """Get current protection status."""
        current_dd = self.calculate_current_drawdown()
        dd_velocity = self.calculate_drawdown_velocity()
        multiplier = self.get_position_multiplier()

        if current_dd >= self.max_drawdown_limit:
            status = 'CRITICAL - TRADING HALTED'
        elif current_dd >= self.danger_threshold:
            status = 'DANGER - SEVERELY REDUCED'
        elif current_dd >= self.warning_threshold:
            status = 'WARNING - REDUCED'
        else:
            status = 'NORMAL'

        return {
            'status': status,
            'current_drawdown': current_dd,
            'drawdown_velocity': dd_velocity,
            'position_multiplier': multiplier,
            'peak_equity': self.peak_equity,
            'current_equity': self.equity_history[-1] if self.equity_history else 0
        }


# =============================================================================
# IMPROVEMENTS #11-15: Additional Advanced Features
# =============================================================================

class InformationTheoreticModelSelector:
    """
    IMPROVEMENT #11: Information-theoretic model selection.

    Select models based on information content of predictions.
    """

    def __init__(self):
        self.model_entropy_history = defaultdict(list)

    def calculate_prediction_entropy(self, predictions: np.ndarray) -> float:
        """Calculate entropy of model predictions."""
        # Bin predictions
        bins = np.histogram(predictions, bins=10, range=(0, 1))[0]
        probs = bins / bins.sum()
        probs = probs[probs > 0]

        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs))
        return entropy

    def select_model(self, model_predictions: Dict[str, np.ndarray]) -> str:
        """Select model with best information content."""
        entropies = {}

        for model_name, predictions in model_predictions.items():
            entropy = self.calculate_prediction_entropy(predictions)
            # Lower entropy = more decisive predictions
            entropies[model_name] = entropy

        # Select model with lowest entropy (most decisive)
        return min(entropies.items(), key=lambda x: x[1])[0]


class AdaptiveFeatureThresholds:
    """
    IMPROVEMENT #12: Adaptive feature thresholds.

    Dynamically adjust decision thresholds based on recent performance.
    """

    def __init__(self, base_threshold: float = 0.5, adaptation_rate: float = 0.1):
        self.base_threshold = base_threshold
        self.adaptation_rate = adaptation_rate
        self.threshold_history = defaultdict(lambda: base_threshold)

    def update_threshold(self, feature_name: str, was_profitable: bool):
        """Update threshold based on trade outcome."""
        current = self.threshold_history[feature_name]

        if was_profitable:
            # Lower threshold if profitable (allow more trades)
            self.threshold_history[feature_name] = current - self.adaptation_rate * (current - 0.3)
        else:
            # Raise threshold if unprofitable (be more selective)
            self.threshold_history[feature_name] = current + self.adaptation_rate * (0.8 - current)

    def get_threshold(self, feature_name: str) -> float:
        """Get current threshold for a feature."""
        return self.threshold_history.get(feature_name, self.base_threshold)


class CrossMarketSignalValidator:
    """
    IMPROVEMENT #13: Cross-market signal validation.

    Validate signals by checking consistency across correlated markets.
    """

    def __init__(self, min_agreement: float = 0.6):
        self.min_agreement = min_agreement
        self.market_correlations = {}

    def validate_signal(self, primary_signal: float,
                       related_signals: Dict[str, float],
                       correlations: Dict[str, float]) -> Dict:
        """
        Validate signal against related markets.

        Args:
            primary_signal: Signal for primary asset
            related_signals: Signals for related assets
            correlations: Correlations with related assets

        Returns:
            Validation result dictionary
        """
        if not related_signals:
            return {'valid': True, 'agreement': 1.0, 'adjusted_signal': primary_signal}

        agreements = []
        weighted_signal = primary_signal
        total_weight = 1.0

        for asset, signal in related_signals.items():
            corr = correlations.get(asset, 0)

            # Expected signal based on correlation
            expected_direction = np.sign(signal) if corr > 0 else -np.sign(signal)
            actual_direction = np.sign(primary_signal)

            agreement = 1 if expected_direction == actual_direction else 0
            agreements.append(agreement)

            # Weight adjustment
            weight = abs(corr) * 0.3
            weighted_signal += signal * corr * weight
            total_weight += weight

        avg_agreement = np.mean(agreements) if agreements else 1.0
        adjusted_signal = weighted_signal / total_weight

        return {
            'valid': avg_agreement >= self.min_agreement,
            'agreement': avg_agreement,
            'adjusted_signal': adjusted_signal,
            'original_signal': primary_signal
        }


class ProfitMaximizingLoss:
    """
    IMPROVEMENT #14: Profit-maximizing loss functions.

    Custom loss function that weights direction errors by magnitude.
    """

    def __init__(self, direction_weight: float = 0.7, magnitude_weight: float = 0.3):
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight

    def calculate_loss(self, predictions: np.ndarray,
                      actuals: np.ndarray) -> float:
        """
        Calculate profit-maximizing loss.

        Penalizes direction errors more than magnitude errors.
        """
        # Direction loss (binary - did we get direction right?)
        pred_direction = np.sign(predictions - 0.5)
        actual_direction = np.sign(actuals)
        direction_loss = np.mean(pred_direction != actual_direction)

        # Magnitude loss (MSE weighted by actual magnitude)
        magnitude_weights = np.abs(actuals)
        magnitude_loss = np.mean((predictions - actuals) ** 2 * magnitude_weights)

        total_loss = (self.direction_weight * direction_loss +
                     self.magnitude_weight * magnitude_loss)

        return total_loss


class WalkForwardValidator:
    """
    IMPROVEMENT #15: Walk-forward validation framework.

    Proper walk-forward backtesting with expanding/rolling windows.
    """

    def __init__(self, train_size: int = 252, test_size: int = 21,
                 step_size: int = 21, expanding: bool = True):
        """
        Initialize walk-forward validator.

        Args:
            train_size: Initial training window size (252 = 1 year)
            test_size: Test window size (21 = 1 month)
            step_size: Steps between windows (21 = 1 month)
            expanding: If True, use expanding window; else rolling
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.expanding = expanding

    def generate_folds(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train/test folds for walk-forward validation.

        Args:
            data: Full dataset

        Returns:
            List of (train_df, test_df) tuples
        """
        folds = []
        n = len(data)

        start_idx = self.train_size

        while start_idx + self.test_size <= n:
            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, start_idx - self.train_size)

            train_end = start_idx
            test_start = start_idx
            test_end = min(start_idx + self.test_size, n)

            train_df = data.iloc[train_start:train_end]
            test_df = data.iloc[test_start:test_end]

            folds.append((train_df, test_df))

            start_idx += self.step_size

        return folds

    def validate(self, data: pd.DataFrame, model_func: Callable,
                predict_func: Callable) -> Dict:
        """
        Run walk-forward validation.

        Args:
            data: Full dataset
            model_func: Function to train model
            predict_func: Function to make predictions

        Returns:
            Validation results dictionary
        """
        folds = self.generate_folds(data)

        all_predictions = []
        all_actuals = []
        fold_metrics = []

        for i, (train_df, test_df) in enumerate(folds):
            # Train model
            model = model_func(train_df)

            # Make predictions
            predictions = predict_func(model, test_df)
            actuals = test_df['returns'].values if 'returns' in test_df.columns else np.zeros(len(test_df))

            all_predictions.extend(predictions)
            all_actuals.extend(actuals)

            # Calculate fold metrics
            if len(predictions) > 0 and len(actuals) > 0:
                direction_accuracy = np.mean(np.sign(predictions) == np.sign(actuals))
                fold_metrics.append({
                    'fold': i,
                    'direction_accuracy': direction_accuracy,
                    'n_samples': len(predictions)
                })

        # Aggregate metrics
        return {
            'n_folds': len(folds),
            'total_samples': len(all_predictions),
            'overall_direction_accuracy': np.mean(np.sign(all_predictions) == np.sign(all_actuals)),
            'fold_metrics': fold_metrics,
            'mean_fold_accuracy': np.mean([f['direction_accuracy'] for f in fold_metrics])
        }


# =============================================================================
# MASTER INTEGRATION CLASS
# =============================================================================

class Phase2ImprovementSystem:
    """
    Master class integrating all 15 improvements.

    Usage:
        system = Phase2ImprovementSystem()
        system.initialize()

        # Generate enhanced signal
        signal = system.generate_enhanced_signal(ticker, data, portfolio)

        # Get position size
        position = system.get_position_size(signal, portfolio)
    """

    def __init__(self, config: Dict = None):
        """
        Initialize Phase 2 improvement system.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize all improvement components
        self.dynamic_weighter = DynamicEnsembleWeighter()
        self.regime_feature_selector = RegimeAwareFeatureSelector()
        self.correlation_analyzer = AdvancedCorrelationAnalyzer()
        self.multi_timeframe = MultiTimeframeEnsemble()
        self.streaming_features = StreamingFeatureEngine()
        self.position_sizer = ConfidenceAwarePositionSizer()
        self.regime_transition = RegimeTransitionDetector()
        self.feature_importance = TimeVaryingFeatureImportance()
        self.bayesian_combiner = BayesianSignalCombiner()
        self.drawdown_protection = AdaptiveDrawdownProtection()

        # Additional improvements
        self.model_selector = InformationTheoreticModelSelector()
        self.adaptive_thresholds = AdaptiveFeatureThresholds()
        self.cross_market_validator = CrossMarketSignalValidator()
        self.profit_loss = ProfitMaximizingLoss()
        self.walk_forward = WalkForwardValidator()

        self.initialized = False

    def initialize(self):
        """Initialize the system."""
        self.initialized = True
        print("[Phase2ImprovementSystem] Initialized with 15 improvements:")
        print("  1. Dynamic Ensemble Weighting")
        print("  2. Regime-Aware Feature Selection")
        print("  3. Advanced Cross-Asset Correlations")
        print("  4. Multi-Timeframe Ensemble")
        print("  5. Real-Time Feature Engineering")
        print("  6. Confidence-Calibrated Position Sizing")
        print("  7. Regime Transition Detection")
        print("  8. Feature Importance Over Time")
        print("  9. Bayesian Signal Combination")
        print("  10. Dynamic Drawdown Protection")
        print("  11. Information-Theoretic Model Selection")
        print("  12. Adaptive Feature Thresholds")
        print("  13. Cross-Market Signal Validation")
        print("  14. Profit-Maximizing Loss Functions")
        print("  15. Walk-Forward Validation Framework")

    def generate_enhanced_signal(self, ticker: str, data: pd.DataFrame,
                                 portfolio: Dict = None,
                                 market_conditions: Dict = None) -> Dict:
        """
        Generate enhanced signal using all improvements.

        Args:
            ticker: Asset ticker
            data: OHLCV DataFrame
            portfolio: Current portfolio positions
            market_conditions: Current market conditions

        Returns:
            Enhanced signal dictionary
        """
        if not self.initialized:
            self.initialize()

        market_conditions = market_conditions or {}

        # 1. Get asset class and dynamic weights
        asset_class = self.dynamic_weighter.get_asset_class(ticker)
        weights = self.dynamic_weighter.get_dynamic_weights()
        asset_weight = weights.get(asset_class, 0.15)

        # 2. Select regime-appropriate features
        current_regime = market_conditions.get('regime', 'neutral')
        available_features = list(data.columns)
        selected_features = self.regime_feature_selector.select_features(
            current_regime, available_features
        )

        # 3. Generate multi-timeframe signal
        mtf_result = self.multi_timeframe.generate_multi_timeframe_signal(
            ticker, data
        )

        # 4. Check regime transition
        transition_result = self.regime_transition.detect_impending_regime_change(
            market_conditions
        )

        # 5. Prepare signals for Bayesian combination
        signals = {
            'multi_timeframe': mtf_result['signal'],
            'momentum': self._calculate_momentum_signal(data),
            'mean_reversion': self._calculate_mean_reversion_signal(data)
        }

        # 6. Combine signals using Bayesian approach
        bayesian_result = self.bayesian_combiner.combine_signals_bayesian(signals)

        # 7. Get drawdown protection multiplier
        dd_multiplier = self.drawdown_protection.get_position_multiplier()

        # 8. Adjust signal for impending regime change
        final_signal = bayesian_result['combined_signal']
        if transition_result['impending_change']:
            final_signal *= 0.5  # Reduce signal during regime transitions

        # 9. Validate signal across markets (if related signals available)
        related_signals = market_conditions.get('related_signals', {})
        correlations = market_conditions.get('correlations', {})
        if related_signals:
            validation = self.cross_market_validator.validate_signal(
                final_signal, related_signals, correlations
            )
            if not validation['valid']:
                final_signal *= 0.7

        return {
            'ticker': ticker,
            'signal': final_signal,
            'confidence': bayesian_result['confidence'],
            'asset_class': asset_class,
            'asset_weight': asset_weight,
            'timeframe_agreement': mtf_result['timeframe_agreement'],
            'regime_transition_probability': transition_result['transition_probability'],
            'drawdown_multiplier': dd_multiplier,
            'signals_combined': bayesian_result['signals_combined'],
            'method': 'phase2_enhanced'
        }

    def _calculate_momentum_signal(self, data: pd.DataFrame) -> float:
        """Calculate simple momentum signal."""
        if 'Close' not in data.columns or len(data) < 20:
            return 0

        returns_20d = data['Close'].pct_change(20).iloc[-1]
        return np.tanh(returns_20d * 10)

    def _calculate_mean_reversion_signal(self, data: pd.DataFrame) -> float:
        """Calculate mean reversion signal."""
        if 'Close' not in data.columns or len(data) < 20:
            return 0

        close = data['Close']
        sma_20 = close.rolling(20).mean().iloc[-1]
        current = close.iloc[-1]

        deviation = (current - sma_20) / sma_20
        return -np.tanh(deviation * 10)  # Negative = expect reversion

    def get_position_size(self, signal_data: Dict, portfolio: Dict = None) -> float:
        """
        Get position size with all adjustments.

        Args:
            signal_data: Signal data from generate_enhanced_signal
            portfolio: Current portfolio positions

        Returns:
            Position size as fraction of capital
        """
        # Base position from confidence-calibrated sizer
        base_position = self.position_sizer.get_position_size(
            {
                'signal_strength': abs(signal_data.get('signal', 0)),
                'confidence': signal_data.get('confidence', 0.5),
                'ticker': signal_data.get('ticker', '')
            },
            portfolio
        )

        # Apply drawdown multiplier
        dd_multiplier = signal_data.get('drawdown_multiplier', 1.0)
        adjusted_position = base_position * dd_multiplier

        # Apply asset class weight
        asset_weight = signal_data.get('asset_weight', 0.15)
        adjusted_position *= (asset_weight / 0.15)  # Scale relative to baseline

        # Reduce for regime transitions
        transition_prob = signal_data.get('regime_transition_probability', 0)
        if transition_prob > 0.5:
            adjusted_position *= (1 - transition_prob * 0.5)

        return max(0.02, min(0.15, adjusted_position))

    def update_performance(self, ticker: str, actual_return: float,
                          signal_data: Dict):
        """
        Update all performance trackers.

        Args:
            ticker: Asset ticker
            actual_return: Actual return achieved
            signal_data: Signal data used for the trade
        """
        asset_class = signal_data.get('asset_class', 'equity')

        # Update dynamic weighter
        self.dynamic_weighter.update_performance(
            asset_class, actual_return,
            actual_return > 0
        )

        # Update Bayesian combiner
        self.bayesian_combiner.update_signal_reliability(
            'multi_timeframe',
            signal_data.get('signal', 0),
            actual_return
        )

        # Update position sizer
        self.position_sizer.update_performance(
            abs(signal_data.get('signal', 0)),
            actual_return,
            signal_data.get('position_size', 0.05)
        )

        # Update adaptive thresholds
        self.adaptive_thresholds.update_threshold(
            'confidence',
            actual_return > 0
        )

    def get_system_status(self) -> Dict:
        """Get current system status."""
        return {
            'initialized': self.initialized,
            'dynamic_weights': self.dynamic_weighter.current_weights,
            'drawdown_protection': self.drawdown_protection.get_protection_status(),
            'n_tracked_signals': len(self.bayesian_combiner.signal_priors),
            'improvements_active': 15
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_phase2_system(config: Dict = None) -> Phase2ImprovementSystem:
    """Get singleton Phase2 improvement system."""
    global _phase2_system
    if '_phase2_system' not in globals() or _phase2_system is None:
        _phase2_system = Phase2ImprovementSystem(config)
        _phase2_system.initialize()
    return _phase2_system


def apply_all_improvements(ticker: str, data: pd.DataFrame,
                          portfolio: Dict = None,
                          market_conditions: Dict = None) -> Dict:
    """
    Convenience function to apply all improvements.

    Args:
        ticker: Asset ticker
        data: OHLCV DataFrame
        portfolio: Current portfolio positions
        market_conditions: Market conditions

    Returns:
        Enhanced signal with position sizing
    """
    system = get_phase2_system()

    signal = system.generate_enhanced_signal(
        ticker, data, portfolio, market_conditions
    )

    position_size = system.get_position_size(signal, portfolio)
    signal['position_size'] = position_size

    return signal


# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    """Test Phase 2 improvement system."""
    print("=" * 70)
    print("PHASE 2 IMPROVEMENT SYSTEM TEST")
    print("15 Comprehensive Improvements")
    print("=" * 70)

    # Initialize system
    system = Phase2ImprovementSystem()
    system.initialize()

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02 + 0.001))

    data = pd.DataFrame({
        'Open': prices * (1 + np.random.randn(100) * 0.005),
        'High': prices * (1 + np.abs(np.random.randn(100) * 0.01)),
        'Low': prices * (1 - np.abs(np.random.randn(100) * 0.01)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)

    # Test signal generation
    print("\n[TEST] Generating enhanced signal for AAPL...")
    market_conditions = {
        'regime': 'bull_market',
        'current_volatility': 0.18,
        'volatility_regime': 'medium',
        'momentum_5d': 0.02,
        'momentum_20d': 0.05
    }

    signal = system.generate_enhanced_signal(
        ticker='AAPL',
        data=data,
        portfolio=None,
        market_conditions=market_conditions
    )

    print(f"\nEnhanced Signal:")
    print(f"  Signal: {signal['signal']:+.3f}")
    print(f"  Confidence: {signal['confidence']:.2f}")
    print(f"  Asset Class: {signal['asset_class']}")
    print(f"  Timeframe Agreement: {signal['timeframe_agreement']:.2f}")
    print(f"  Regime Transition Prob: {signal['regime_transition_probability']:.2f}")
    print(f"  Drawdown Multiplier: {signal['drawdown_multiplier']:.2f}")

    # Test position sizing
    position_size = system.get_position_size(signal)
    print(f"\nPosition Size: {position_size:.1%}")

    # Get system status
    status = system.get_system_status()
    print(f"\nSystem Status:")
    print(f"  Initialized: {status['initialized']}")
    print(f"  Improvements Active: {status['improvements_active']}")

    print("\n" + "=" * 70)
    print("[SUCCESS] Phase 2 Improvement System Test Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
