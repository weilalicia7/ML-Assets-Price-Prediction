"""
Phase 4 Elite Enhancements

Based on: phase4 final fixing on C model.pdf

Takes Phase 4 from excellent to exceptional with 8 elite-level enhancements:
1. Adaptive Macro Feature Selection
2. Macro-Aware Ensemble Weighting
3. Dynamic Macro Sensitivity Adjustment
4. Macro Regime Transition Forecasting
5. Macro Feature Compression
6. Real-Time Macro Impact Monitoring
7. Macro-Aware Position Sizing
8. Cross-Timeframe Macro Analysis

Expected Additional Improvement: +3-5% profit rate (63-68% total)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 1. ADAPTIVE MACRO FEATURE SELECTION
# =============================================================================

class AdaptiveMacroFeatureSelector:
    """
    Select regime-optimized features instead of using all 50+ features.

    Different market regimes benefit from different macro features:
    - Risk-on: Focus on momentum and beta
    - Risk-off: Focus on safe haven and correlation features
    - High volatility: Focus on VIX and tail risk
    - Low volatility: Focus on trend and relative strength
    """

    def __init__(self):
        # Regime-specific feature priority sets
        self.regime_feature_sets = {
            'risk_on': [
                'spy_momentum_5d', 'spy_momentum_20d', 'vix_momentum_5d',
                'risk_regime_score', 'beta_spy_20d', 'rel_strength_spy_20d',
                'dxy_momentum_5d', 'risk_on', 'VIX_dist_ma_20d', 'SPY_dist_ma_20d'
            ],
            'risk_off': [
                'vix_regime', 'vix_elevated', 'vix_crisis', 'gld_momentum_20d',
                'tlt_momentum_20d', 'corr_network_avg', 'risk_off', 'vix_spike',
                'GLD_dist_ma_20d', 'TLT_dist_ma_20d', 'dxy_strong'
            ],
            'high_vol': [
                'vix_spike', 'vix_crisis', 'vix_elevated', 'corr_breakdown',
                'pca_pc1_explained', 'VIX', 'VIX_momentum_5d', 'risk_regime_score',
                'beta_spy_20d', 'gld_momentum_5d', 'tlt_momentum_5d'
            ],
            'low_vol': [
                'spy_dist_ma_20d', 'rel_strength_spy_20d', 'dxy_strong', 'dxy_weak',
                'spy_momentum_20d', 'spy_momentum_60d', 'risk_on', 'beta_spy_20d',
                'VIX_ma_20d', 'momentum_consistency'
            ],
            'neutral': [
                'risk_regime_score', 'beta_spy_20d', 'VIX', 'SPY_momentum_20d',
                'rel_strength_spy_20d', 'vix_normal', 'DXY_momentum_20d',
                'GLD_momentum_20d', 'TLT_momentum_20d', 'corr_spy_vix'
            ]
        }

        # Feature importance tracking
        self.feature_performance = {}

    def select_regime_optimized_features(
        self,
        current_regime: str,
        available_features: List[str],
        max_features: int = 15
    ) -> List[str]:
        """
        Select most relevant features for current market regime.

        Args:
            current_regime: Current market regime (risk_on, risk_off, high_vol, low_vol, neutral)
            available_features: List of available feature names
            max_features: Maximum number of features to select

        Returns:
            List of selected feature names optimized for the regime
        """
        # Get priority features for this regime
        regime_priority = self.regime_feature_sets.get(current_regime, [])

        # Select features that exist in available features
        selected = [f for f in regime_priority if f in available_features]

        # Fill remaining slots with other available macro features
        remaining_slots = max_features - len(selected)
        if remaining_slots > 0:
            other_features = [f for f in available_features if f not in selected]
            # Prioritize macro-related features
            macro_prefixes = ('vix_', 'spy_', 'gld_', 'tlt_', 'dxy_', 'risk_',
                            'corr_', 'beta_', 'rel_strength', 'VIX', 'SPY', 'GLD', 'TLT', 'DXY')
            macro_others = [f for f in other_features if any(f.startswith(p) or f == p for p in macro_prefixes)]
            selected.extend(macro_others[:remaining_slots])

        logger.info(f"Selected {len(selected)} features for {current_regime} regime")
        return selected[:max_features]

    def get_regime_from_macro_state(self, macro_state: Dict) -> str:
        """
        Determine current regime from macro state.

        Args:
            macro_state: Dict with macro indicators

        Returns:
            Regime name
        """
        vix_regime = macro_state.get('vix_regime', 'normal')
        risk_regime = macro_state.get('risk_regime', 'neutral')

        # High volatility takes precedence
        if vix_regime in ['crisis', 'elevated']:
            return 'high_vol'
        elif vix_regime == 'low_vol':
            return 'low_vol'
        elif risk_regime == 'risk_on':
            return 'risk_on'
        elif risk_regime == 'risk_off':
            return 'risk_off'
        else:
            return 'neutral'

    def update_feature_performance(
        self,
        feature_name: str,
        regime: str,
        prediction_accuracy: float
    ):
        """Track feature performance by regime for future optimization."""
        key = (feature_name, regime)
        if key not in self.feature_performance:
            self.feature_performance[key] = []
        self.feature_performance[key].append(prediction_accuracy)

        # Keep only recent history
        if len(self.feature_performance[key]) > 100:
            self.feature_performance[key] = self.feature_performance[key][-100:]


# =============================================================================
# 2. MACRO-AWARE ENSEMBLE WEIGHTING
# =============================================================================

class MacroAwareEnsembleWeighter:
    """
    Adjust ensemble model weights based on macro regime.

    Different strategies work better in different macro environments:
    - Risk-on: Momentum strategies perform well
    - Risk-off: Mean reversion and defensive strategies
    - High volatility: Volatility-based strategies
    - Crisis: Defensive, volatility-focused
    """

    def __init__(self):
        # Regime-specific ensemble weights
        self.regime_ensemble_weights = {
            'risk_on': {
                'momentum': 0.50,
                'mean_reversion': 0.30,
                'volatility': 0.20
            },
            'risk_off': {
                'momentum': 0.20,
                'mean_reversion': 0.50,
                'volatility': 0.30
            },
            'high_vol': {
                'momentum': 0.30,
                'mean_reversion': 0.40,
                'volatility': 0.30
            },
            'low_vol': {
                'momentum': 0.45,
                'mean_reversion': 0.35,
                'volatility': 0.20
            },
            'crisis': {
                'momentum': 0.10,
                'mean_reversion': 0.30,
                'volatility': 0.60
            },
            'neutral': {
                'momentum': 0.40,
                'mean_reversion': 0.35,
                'volatility': 0.25
            }
        }

        # Default weights
        self.default_weights = {'momentum': 0.40, 'mean_reversion': 0.35, 'volatility': 0.25}

    def get_regime_optimized_weights(self, macro_state: Dict) -> Dict[str, float]:
        """
        Adjust ensemble weights based on macro regime.

        Args:
            macro_state: Dict with macro indicators including risk_regime and vix_regime

        Returns:
            Dict mapping strategy type to weight
        """
        # Determine regime
        vix_regime = macro_state.get('vix_regime', 'normal')
        risk_regime = macro_state.get('risk_regime', 'neutral')

        # Crisis takes precedence
        if vix_regime == 'crisis':
            regime_key = 'crisis'
        elif vix_regime == 'elevated':
            regime_key = 'high_vol'
        elif vix_regime == 'low_vol':
            regime_key = 'low_vol'
        elif 'risk_on' in risk_regime:
            regime_key = 'risk_on'
        elif 'risk_off' in risk_regime:
            regime_key = 'risk_off'
        else:
            regime_key = 'neutral'

        # Get base weights for regime
        base_weights = self.regime_ensemble_weights.get(regime_key, self.default_weights).copy()

        # Additional VIX adjustment
        if vix_regime == 'crisis':
            # Reduce momentum, increase volatility strategy weight
            for key in base_weights:
                if key != 'volatility':
                    base_weights[key] *= 0.7
                else:
                    base_weights[key] *= 1.3

        # Normalize to sum to 1
        total = sum(base_weights.values())
        normalized = {k: v / total for k, v in base_weights.items()}

        logger.debug(f"Regime {regime_key}: weights = {normalized}")
        return normalized

    def combine_ensemble_predictions(
        self,
        predictions: Dict[str, float],
        macro_state: Dict
    ) -> float:
        """
        Combine ensemble predictions using macro-aware weights.

        Args:
            predictions: Dict mapping strategy name to prediction
            macro_state: Current macro state

        Returns:
            Weighted combined prediction
        """
        weights = self.get_regime_optimized_weights(macro_state)

        combined = 0.0
        total_weight = 0.0

        for strategy, prediction in predictions.items():
            weight = weights.get(strategy, 0.0)
            combined += prediction * weight
            total_weight += weight

        if total_weight > 0:
            return combined / total_weight
        return combined


# =============================================================================
# 3. DYNAMIC MACRO SENSITIVITY ADJUSTMENT
# =============================================================================

class DynamicMacroSensitivity:
    """
    Dynamically adjust sensitivity to macro signals based on their
    recent predictive power.

    If macro signals have been accurate recently, increase sensitivity.
    If they've been poor predictors, reduce sensitivity.
    """

    def __init__(self, learning_rate: float = 0.1):
        self.sensitivity_history = {}
        self.learning_rate = learning_rate
        self.min_sensitivity = 0.5
        self.max_sensitivity = 1.5
        self.default_sensitivity = 1.0

    def calculate_optimal_sensitivity(
        self,
        macro_features: pd.DataFrame,
        actual_returns: pd.Series,
        lookback: int = 63
    ) -> Tuple[float, Dict[str, float]]:
        """
        Dynamically adjust how sensitive the system should be to macro signals.

        Args:
            macro_features: DataFrame with macro features
            actual_returns: Series of actual returns
            lookback: Lookback period for correlation calculation

        Returns:
            Tuple of (optimal_sensitivity, feature_correlations)
        """
        correlations = {}

        # Calculate correlation of each macro feature with returns
        macro_prefixes = ('vix_', 'risk_', 'corr_', 'spy_', 'gld_', 'tlt_', 'dxy_', 'beta_')

        for col in macro_features.columns:
            if any(col.lower().startswith(prefix) for prefix in macro_prefixes):
                try:
                    # Rolling correlation
                    valid_mask = ~(macro_features[col].isna() | actual_returns.isna())
                    if valid_mask.sum() > lookback:
                        corr = macro_features[col][valid_mask].rolling(lookback).corr(
                            actual_returns[valid_mask]
                        )
                        avg_corr = corr.abs().mean()
                        if not np.isnan(avg_corr):
                            correlations[col] = avg_corr
                except Exception as e:
                    logger.debug(f"Error calculating correlation for {col}: {e}")

        # Higher average correlation = higher sensitivity warranted
        if correlations:
            avg_correlation = np.mean(list(correlations.values()))
            # Scale: 0 correlation -> 0.8 sensitivity, 0.3 correlation -> 1.4 sensitivity
            optimal_sensitivity = min(
                self.max_sensitivity,
                max(self.min_sensitivity, 0.8 + avg_correlation * 2)
            )
        else:
            optimal_sensitivity = self.default_sensitivity

        return optimal_sensitivity, correlations

    def get_adjusted_macro_multiplier(
        self,
        base_multiplier: float,
        sensitivity: float
    ) -> float:
        """
        Adjust macro multiplier based on current sensitivity.

        Args:
            base_multiplier: Base macro multiplier (0-1.5)
            sensitivity: Current sensitivity level

        Returns:
            Adjusted multiplier
        """
        # If multiplier < 1 (risk reduction), sensitivity increases the reduction
        # If multiplier > 1 (risk increase), sensitivity increases the boost
        if base_multiplier < 1.0:
            reduction = 1.0 - base_multiplier
            adjusted_reduction = reduction * sensitivity
            return max(0.0, 1.0 - adjusted_reduction)
        else:
            increase = base_multiplier - 1.0
            adjusted_increase = increase * sensitivity
            return min(1.5, 1.0 + adjusted_increase)


# =============================================================================
# 4. MACRO REGIME TRANSITION FORECASTING
# =============================================================================

class MacroRegimeForecaster:
    """
    Predict regime changes before they happen.

    Uses multiple signals to forecast regime transitions:
    - VIX compression (low volatility of volatility)
    - Correlation divergence
    - Momentum exhaustion
    - Volume anomalies
    """

    def __init__(self, forecast_horizon: int = 10):
        self.regime_patterns = {}
        self.forecast_horizon = forecast_horizon

    def forecast_regime_transitions(
        self,
        macro_features: pd.DataFrame,
        current_regime: str
    ) -> Dict:
        """
        Predict regime changes before they happen.

        Args:
            macro_features: DataFrame with macro features
            current_regime: Current regime name

        Returns:
            Dict with transition forecast
        """
        # Analyze regime duration patterns
        regime_durations = self.analyze_regime_duration_patterns(macro_features)

        # Technical indicators of impending change
        change_signals = {
            'vix_compression': self.detect_vix_compression(macro_features),
            'correlation_divergence': self.detect_correlation_divergence(macro_features),
            'momentum_exhaustion': self.detect_momentum_exhaustion(macro_features),
            'volume_anomalies': self.detect_volume_anomalies(macro_features)
        }

        # Composite transition probability
        valid_signals = [v for v in change_signals.values() if v is not None]
        transition_prob = np.mean(valid_signals) if valid_signals else 0.3

        # Check if regime is "overdue" to change
        current_duration = regime_durations.get(current_regime, {}).get('current_duration', 0)
        avg_duration = regime_durations.get(current_regime, {}).get('avg_duration', 21)

        if avg_duration > 0:
            duration_ratio = current_duration / avg_duration
            if duration_ratio > 1.2:
                transition_prob = min(0.9, transition_prob * 1.5)

        # Confidence based on signal agreement
        confidence = 1 - np.std(valid_signals) if len(valid_signals) > 1 else 0.5

        return {
            'transition_probability': float(transition_prob),
            'expected_days_to_transition': max(1, int(avg_duration - current_duration)),
            'confidence': float(confidence),
            'trigger_signals': change_signals,
            'current_duration': current_duration,
            'avg_duration': avg_duration
        }

    def analyze_regime_duration_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze how long each regime typically lasts."""
        if 'risk_regime' not in df.columns:
            return {}

        durations = {}

        # Calculate regime changes
        regime_changes = df['risk_regime'].ne(df['risk_regime'].shift())
        regime_periods = regime_changes.cumsum()

        # Get duration for each regime type
        for regime in df['risk_regime'].unique():
            regime_mask = df['risk_regime'] == regime
            if regime_mask.sum() > 0:
                # Count consecutive periods
                regime_durations = []
                current_count = 0
                prev_was_regime = False

                for is_regime in regime_mask:
                    if is_regime:
                        current_count += 1
                        prev_was_regime = True
                    elif prev_was_regime:
                        regime_durations.append(current_count)
                        current_count = 0
                        prev_was_regime = False

                # Add final period if still in regime
                if prev_was_regime and current_count > 0:
                    regime_durations.append(current_count)

                if regime_durations:
                    durations[regime] = {
                        'avg_duration': np.mean(regime_durations),
                        'std_duration': np.std(regime_durations) if len(regime_durations) > 1 else 5,
                        'current_duration': regime_durations[-1] if regime_mask.iloc[-1] else 0,
                        'count': len(regime_durations)
                    }

        return durations

    def detect_vix_compression(self, df: pd.DataFrame) -> Optional[float]:
        """Detect VIX compression (often precedes volatility spike)."""
        if 'VIX' not in df.columns or len(df) < 20:
            return None

        vix = df['VIX'].dropna()
        if len(vix) < 20:
            return None

        # Calculate volatility of VIX
        vix_vol = vix.rolling(20).std()
        vix_vol_ma = vix_vol.rolling(60).mean()

        if vix_vol_ma.iloc[-1] > 0:
            compression_ratio = vix_vol.iloc[-1] / vix_vol_ma.iloc[-1]
            # Low compression ratio = high probability of regime change
            return float(max(0, min(1, 1 - compression_ratio)))
        return None

    def detect_correlation_divergence(self, df: pd.DataFrame) -> Optional[float]:
        """Detect divergence in cross-asset correlations."""
        required_cols = ['SPY', 'VIX', 'GLD']
        if not all(col in df.columns for col in required_cols) or len(df) < 60:
            return None

        try:
            # Calculate rolling correlations
            spy_vix_corr = df['SPY'].pct_change().rolling(20).corr(df['VIX'].pct_change())
            spy_gld_corr = df['SPY'].pct_change().rolling(20).corr(df['GLD'].pct_change())

            # Check for divergence from historical norms
            # SPY-VIX is typically negative, SPY-GLD varies
            current_spy_vix = spy_vix_corr.iloc[-1]
            historical_spy_vix = spy_vix_corr.rolling(60).mean().iloc[-1]

            if not np.isnan(current_spy_vix) and not np.isnan(historical_spy_vix):
                divergence = abs(current_spy_vix - historical_spy_vix)
                return float(min(1, divergence * 2))
        except Exception:
            pass
        return None

    def detect_momentum_exhaustion(self, df: pd.DataFrame) -> Optional[float]:
        """Detect momentum exhaustion signals."""
        if 'SPY' not in df.columns or len(df) < 60:
            return None

        try:
            spy = df['SPY'].dropna()
            if len(spy) < 60:
                return None

            # Calculate momentum
            momentum_5d = spy.pct_change(5)
            momentum_20d = spy.pct_change(20)

            # Exhaustion: short-term momentum diverging from longer-term
            if momentum_20d.iloc[-1] != 0:
                divergence = abs(momentum_5d.iloc[-1] - momentum_20d.iloc[-1])
                return float(min(1, divergence * 10))
        except Exception:
            pass
        return None

    def detect_volume_anomalies(self, df: pd.DataFrame) -> Optional[float]:
        """Detect volume anomalies (placeholder - needs volume data)."""
        # Would need volume data for proper implementation
        return 0.3  # Default neutral value


# =============================================================================
# 5. MACRO FEATURE COMPRESSION
# =============================================================================

class MacroFeatureCompressor:
    """
    Reduce macro feature dimensionality while preserving predictive power.

    Uses:
    - Correlation-based feature removal (>95% correlated)
    - PCA for remaining features
    """

    def __init__(self, n_components: int = 15, correlation_threshold: float = 0.95):
        self.n_components = n_components
        self.correlation_threshold = correlation_threshold
        self.pca = None
        self.feature_names = []
        self.dropped_features = []
        self.is_fitted = False

    def fit(self, macro_features: pd.DataFrame) -> 'MacroFeatureCompressor':
        """
        Fit the compressor on macro features.

        Args:
            macro_features: DataFrame with macro features

        Returns:
            Self
        """
        # Remove highly correlated features
        reduced_features, dropped = self._remove_correlated_features(macro_features)
        self.dropped_features = dropped
        self.feature_names = list(reduced_features.columns)

        # Fit PCA if needed
        if len(reduced_features.columns) > self.n_components:
            # Fill NaN for PCA
            filled = reduced_features.fillna(reduced_features.mean())
            self.pca = PCA(n_components=self.n_components)
            self.pca.fit(filled)
        else:
            self.pca = None

        self.is_fitted = True
        logger.info(f"Fitted compressor: {len(macro_features.columns)} -> {self.n_components} features")
        return self

    def transform(self, macro_features: pd.DataFrame) -> pd.DataFrame:
        """
        Transform macro features using fitted compressor.

        Args:
            macro_features: DataFrame with macro features

        Returns:
            Compressed features DataFrame
        """
        if not self.is_fitted:
            return self.fit_transform(macro_features)

        # Remove dropped features
        available = [f for f in self.feature_names if f in macro_features.columns]
        reduced = macro_features[available].copy()

        # Apply PCA if fitted
        if self.pca is not None:
            filled = reduced.fillna(reduced.mean())
            # Ensure same features as training
            if len(filled.columns) == len(self.feature_names):
                compressed = self.pca.transform(filled)
                return pd.DataFrame(
                    compressed,
                    columns=[f'macro_pc_{i}' for i in range(compressed.shape[1])],
                    index=macro_features.index
                )

        return reduced

    def fit_transform(self, macro_features: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(macro_features)
        return self.transform(macro_features)

    def _remove_correlated_features(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with correlation > threshold."""
        # Calculate correlation matrix
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return df, []

        corr_matrix = numeric_df.corr().abs()

        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        )

        to_drop = []
        for column in upper_triangle.columns:
            if any(upper_triangle[column] > self.correlation_threshold):
                to_drop.append(column)

        reduced = numeric_df.drop(columns=to_drop, errors='ignore')
        logger.info(f"Removed {len(to_drop)} highly correlated features")

        return reduced, to_drop

    def get_feature_importance(self) -> Dict[str, float]:
        """Get importance of features in compressed space."""
        if self.pca is not None and hasattr(self.pca, 'components_'):
            # Sum absolute loadings across components
            importance = np.abs(self.pca.components_).sum(axis=0)
            return dict(zip(self.feature_names[:len(importance)], importance))
        return {}

    def compress_macro_features(self, macro_features: pd.DataFrame) -> pd.DataFrame:
        """
        Compress macro features (alias for fit_transform for PDF test compatibility).

        Args:
            macro_features: DataFrame with macro features

        Returns:
            Compressed features DataFrame
        """
        # Select only numeric columns
        numeric_features = macro_features.select_dtypes(include=[np.number])
        return self.fit_transform(numeric_features)


# =============================================================================
# 6. REAL-TIME MACRO IMPACT MONITORING
# =============================================================================

class MacroImpactMonitor:
    """
    Monitor how well macro features are predicting returns in real-time.

    Tracks:
    - Directional accuracy
    - Correlation with returns
    - Performance by regime
    """

    def __init__(self, alert_threshold: float = 0.1):
        self.performance_tracking = {}
        self.alert_threshold = alert_threshold
        self.alerts = []

    def track_macro_feature_performance(
        self,
        feature_predictions: Dict[str, np.ndarray],
        actual_returns: np.ndarray,
        macro_state: Dict
    ) -> Dict[str, Dict]:
        """
        Monitor how well macro features are predicting returns.

        Args:
            feature_predictions: Dict mapping feature name to predictions
            actual_returns: Array of actual returns
            macro_state: Current macro state

        Returns:
            Dict with performance metrics per feature
        """
        performance_metrics = {}
        regime = macro_state.get('risk_regime', 'neutral')

        for feature, predictions in feature_predictions.items():
            if len(predictions) != len(actual_returns):
                continue

            # Calculate prediction accuracy
            predictions = np.array(predictions)
            actual = np.array(actual_returns)

            # Directional accuracy
            directional_matches = (predictions * actual > 0)
            directional_accuracy = float(np.mean(directional_matches))

            # Correlation
            if len(predictions) > 5:
                try:
                    correlation = float(np.corrcoef(predictions, actual)[0, 1])
                except:
                    correlation = 0.0
            else:
                correlation = 0.0

            performance_metrics[feature] = {
                'directional_accuracy': directional_accuracy,
                'correlation': correlation if not np.isnan(correlation) else 0.0,
                'regime': regime,
                'sample_size': len(predictions)
            }

            # Check for performance degradation
            if feature in self.performance_tracking:
                prev_accuracy = self.performance_tracking[feature].get('directional_accuracy', 0.5)
                if directional_accuracy < prev_accuracy - self.alert_threshold:
                    self._trigger_performance_alert(feature, directional_accuracy, prev_accuracy)

        self.performance_tracking.update(performance_metrics)
        return performance_metrics

    def _trigger_performance_alert(
        self,
        feature: str,
        current_accuracy: float,
        previous_accuracy: float
    ):
        """Trigger alert for performance degradation."""
        alert = {
            'feature': feature,
            'current_accuracy': current_accuracy,
            'previous_accuracy': previous_accuracy,
            'degradation': previous_accuracy - current_accuracy
        }
        self.alerts.append(alert)
        logger.warning(f"Performance alert: {feature} degraded from {previous_accuracy:.2%} to {current_accuracy:.2%}")

    def get_top_performing_features(
        self,
        regime: Optional[str] = None,
        min_samples: int = 20,
        top_n: int = 10
    ) -> Dict[str, Dict]:
        """
        Get best-performing macro features by regime.

        Args:
            regime: Filter by specific regime (None for all)
            min_samples: Minimum sample size
            top_n: Number of top features to return

        Returns:
            Dict of top performing features
        """
        if regime:
            filtered = {
                k: v for k, v in self.performance_tracking.items()
                if v.get('regime') == regime and v.get('sample_size', 0) >= min_samples
            }
        else:
            filtered = {
                k: v for k, v in self.performance_tracking.items()
                if v.get('sample_size', 0) >= min_samples
            }

        # Sort by directional accuracy
        sorted_features = dict(
            sorted(
                filtered.items(),
                key=lambda x: x[1].get('directional_accuracy', 0),
                reverse=True
            )[:top_n]
        )

        return sorted_features

    def get_alerts(self, clear: bool = True) -> List[Dict]:
        """Get and optionally clear alerts."""
        alerts = self.alerts.copy()
        if clear:
            self.alerts = []
        return alerts


# =============================================================================
# 7. MACRO-AWARE POSITION SIZING
# =============================================================================

class MacroAwarePositionSizer:
    """
    Enhance position sizing with macro conviction.

    Considers:
    - Macro regime conviction
    - VIX regime adjustment
    - Correlation breakdown protection
    - Asset-class specific sensitivity
    """

    def __init__(self):
        # Macro conviction weights by regime
        self.macro_conviction_weights = {
            'strong_risk_on': 1.3,
            'risk_on': 1.1,
            'neutral': 1.0,
            'risk_off': 0.7,
            'strong_risk_off': 0.4
        }

        # VIX regime multipliers
        self.vix_multipliers = {
            'low_vol': 1.1,
            'normal': 1.0,
            'elevated': 0.8,
            'crisis': 0.5
        }

        # Asset class sensitivity to macro
        self.asset_sensitivity = {
            'equity': 1.0,
            'tech': 1.2,
            'crypto': 1.5,
            'bonds': 0.5,
            'gold': 0.6,
            'commodity': 1.1,
            'forex': 0.8,
            'international': 1.1
        }

    def get_macro_enhanced_position(
        self,
        base_position: float,
        macro_state: Dict,
        asset_class: str = 'equity'
    ) -> Dict:
        """
        Enhance position sizing with macro conviction.

        Args:
            base_position: Base position size (0-1)
            macro_state: Current macro state dict
            asset_class: Asset class for sensitivity adjustment

        Returns:
            Dict with position sizing details
        """
        # Macro conviction adjustment
        risk_regime = macro_state.get('risk_regime', 'neutral')
        regime_mult = self.macro_conviction_weights.get(risk_regime, 1.0)

        # VIX regime adjustment
        vix_regime = macro_state.get('vix_regime', 'normal')
        vix_mult = self.vix_multipliers.get(vix_regime, 1.0)

        # Correlation breakdown protection
        corr_breakdown = macro_state.get('correlation_breakdown', False)
        corr_mult = 0.7 if corr_breakdown else 1.0

        # Asset-class specific sensitivity
        sensitivity = self.asset_sensitivity.get(asset_class.lower(), 1.0)

        # For risk reduction scenarios, higher sensitivity = more reduction
        # For risk increase scenarios, higher sensitivity = more increase
        if regime_mult < 1.0:
            # Risk reduction - apply sensitivity to the reduction
            reduction = 1.0 - regime_mult
            adjusted_reduction = reduction * sensitivity
            regime_mult = 1.0 - adjusted_reduction
        else:
            # Risk increase - apply sensitivity to the boost
            increase = regime_mult - 1.0
            adjusted_increase = increase * sensitivity
            regime_mult = 1.0 + adjusted_increase

        # Combined macro enhancement
        macro_enhancement = regime_mult * vix_mult * corr_mult

        # Clamp enhancement to reasonable bounds
        macro_enhancement = max(0.2, min(1.5, macro_enhancement))

        # Final position
        final_position = base_position * macro_enhancement
        final_position = max(0.0, min(1.0, final_position))  # Clamp to 0-1

        return {
            'final_position': final_position,
            'base_position': base_position,
            'macro_enhancement': macro_enhancement,
            'components': {
                'regime_mult': regime_mult,
                'vix_mult': vix_mult,
                'corr_mult': corr_mult,
                'asset_sensitivity': sensitivity
            }
        }


# =============================================================================
# 8. CROSS-TIMEFRAME MACRO ANALYSIS
# =============================================================================

class MultiTimeframeMacroAnalyzer:
    """
    Analyze macro signals across multiple timeframes.

    Timeframes:
    - Daily (1d): Short-term signals
    - Weekly (1w): Medium-term trends
    - Monthly (1m): Long-term context
    """

    def __init__(self):
        self.timeframes = ['1d', '1w', '1m']
        self.timeframe_weights = {'1d': 0.40, '1w': 0.35, '1m': 0.25}

    def analyze_macro_across_timeframes(
        self,
        macro_data: pd.DataFrame
    ) -> Dict:
        """
        Analyze macro signals across multiple timeframes.

        Args:
            macro_data: DataFrame with daily macro data

        Returns:
            Dict with multi-timeframe analysis
        """
        if len(macro_data) < 30:
            return {
                'timeframe_analysis': {},
                'combined_signals': {},
                'timeframe_agreement': 1.0,
                'primary_timeframe': '1d'
            }

        timeframe_analysis = {}

        for tf in self.timeframes:
            # Resample data for timeframe
            if tf == '1d':
                tf_data = macro_data.copy()
            elif tf == '1w':
                tf_data = macro_data.resample('W').last()
            elif tf == '1m':
                tf_data = macro_data.resample('M').last()
            else:
                continue

            # Calculate macro features for this timeframe
            tf_features = self._calculate_timeframe_features(tf_data, tf)
            timeframe_analysis[tf] = tf_features

        # Combine across timeframes
        combined_analysis = self._combine_timeframe_analysis(timeframe_analysis)

        return {
            'timeframe_analysis': timeframe_analysis,
            'combined_signals': combined_analysis,
            'timeframe_agreement': self._calculate_timeframe_agreement(timeframe_analysis),
            'primary_timeframe': self._identify_primary_timeframe(timeframe_analysis)
        }

    def _calculate_timeframe_features(
        self,
        tf_data: pd.DataFrame,
        timeframe: str
    ) -> Dict:
        """Calculate macro features for a specific timeframe."""
        features = {'timeframe': timeframe}

        if 'VIX' in tf_data.columns and len(tf_data) > 0:
            vix = tf_data['VIX'].dropna()
            if len(vix) > 0:
                features['vix_level'] = float(vix.iloc[-1])
                features['vix_trend'] = 'up' if len(vix) > 1 and vix.iloc[-1] > vix.iloc[-2] else 'down'

        if 'SPY' in tf_data.columns and len(tf_data) > 1:
            spy = tf_data['SPY'].dropna()
            if len(spy) > 1:
                features['spy_momentum'] = float((spy.iloc[-1] / spy.iloc[-2] - 1))
                features['spy_trend'] = 'up' if spy.iloc[-1] > spy.iloc[-2] else 'down'

        # Calculate risk regime score for this timeframe
        risk_score = 0
        if features.get('vix_trend') == 'down':
            risk_score += 0.5
        if features.get('spy_trend') == 'up':
            risk_score += 0.5
        features['risk_regime_score'] = risk_score - 0.5  # Center around 0

        return features

    def _combine_timeframe_analysis(
        self,
        timeframe_analysis: Dict
    ) -> Dict:
        """Combine analysis from multiple timeframes."""
        combined = {}

        # Weighted average of risk scores
        total_weight = 0
        weighted_risk_score = 0

        for tf, analysis in timeframe_analysis.items():
            weight = self.timeframe_weights.get(tf, 0.33)
            if 'risk_regime_score' in analysis:
                weighted_risk_score += analysis['risk_regime_score'] * weight
                total_weight += weight

        if total_weight > 0:
            combined['risk_regime_score'] = weighted_risk_score / total_weight
        else:
            combined['risk_regime_score'] = 0

        # Determine combined risk regime
        if combined['risk_regime_score'] > 0.3:
            combined['risk_regime'] = 'risk_on'
        elif combined['risk_regime_score'] < -0.3:
            combined['risk_regime'] = 'risk_off'
        else:
            combined['risk_regime'] = 'neutral'

        return combined

    def _calculate_timeframe_agreement(
        self,
        timeframe_analysis: Dict
    ) -> float:
        """Measure how much different timeframes agree on macro outlook."""
        signals = []

        for tf, analysis in timeframe_analysis.items():
            if 'risk_regime_score' in analysis:
                signals.append(analysis['risk_regime_score'])

        if len(signals) > 1:
            # Agreement = 1 - normalized std
            std = np.std(signals)
            return float(max(0, 1 - std))
        return 1.0

    def _identify_primary_timeframe(
        self,
        timeframe_analysis: Dict
    ) -> str:
        """Identify which timeframe has the strongest signal."""
        strongest_signal = 0
        primary_tf = '1d'

        for tf, analysis in timeframe_analysis.items():
            signal_strength = abs(analysis.get('risk_regime_score', 0))
            if signal_strength > strongest_signal:
                strongest_signal = signal_strength
                primary_tf = tf

        return primary_tf


# =============================================================================
# UNIFIED ELITE SYSTEM
# =============================================================================

class Phase4EliteSystem:
    """
    Unified system combining all elite enhancements.

    Expected improvement: +3-5% profit rate on top of base Phase 4
    """

    def __init__(self):
        self.feature_selector = AdaptiveMacroFeatureSelector()
        self.ensemble_weighter = MacroAwareEnsembleWeighter()
        self.sensitivity_adjuster = DynamicMacroSensitivity()
        self.regime_forecaster = MacroRegimeForecaster()
        self.feature_compressor = MacroFeatureCompressor()
        self.impact_monitor = MacroImpactMonitor()
        self.position_sizer = MacroAwarePositionSizer()
        self.mtf_analyzer = MultiTimeframeMacroAnalyzer()

        logger.info("Phase 4 Elite System initialized with all 8 enhancements")

    def process_macro_data(
        self,
        macro_features: pd.DataFrame,
        macro_state: Dict,
        asset_class: str = 'equity'
    ) -> Dict:
        """
        Process macro data through all elite enhancements.

        Args:
            macro_features: DataFrame with macro features
            macro_state: Current macro state dict
            asset_class: Asset class for position sizing

        Returns:
            Dict with all enhancement outputs
        """
        results = {}

        # 1. Adaptive Feature Selection
        regime = self.feature_selector.get_regime_from_macro_state(macro_state)
        selected_features = self.feature_selector.select_regime_optimized_features(
            regime, list(macro_features.columns)
        )
        results['selected_features'] = selected_features
        results['regime'] = regime

        # 2. Ensemble Weights
        ensemble_weights = self.ensemble_weighter.get_regime_optimized_weights(macro_state)
        results['ensemble_weights'] = ensemble_weights

        # 4. Regime Forecasting
        forecast = self.regime_forecaster.forecast_regime_transitions(
            macro_features, regime
        )
        results['regime_forecast'] = forecast

        # 8. Multi-Timeframe Analysis
        mtf_analysis = self.mtf_analyzer.analyze_macro_across_timeframes(macro_features)
        results['mtf_analysis'] = mtf_analysis

        return results

    def get_enhanced_position(
        self,
        base_position: float,
        macro_state: Dict,
        asset_class: str = 'equity'
    ) -> Dict:
        """
        Get position size with all macro enhancements.

        Args:
            base_position: Base position size
            macro_state: Current macro state
            asset_class: Asset class

        Returns:
            Dict with enhanced position details
        """
        return self.position_sizer.get_macro_enhanced_position(
            base_position, macro_state, asset_class
        )


# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    """Test all elite enhancements."""
    print("=" * 70)
    print("PHASE 4 ELITE ENHANCEMENTS TEST")
    print("=" * 70)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-06-01', freq='D')

    macro_features = pd.DataFrame({
        'VIX': np.random.normal(18, 5, len(dates)).clip(10, 40),
        'SPY': 100 + np.cumsum(np.random.normal(0.001, 0.015, len(dates))),
        'GLD': 180 + np.cumsum(np.random.normal(0.0005, 0.01, len(dates))),
        'TLT': 90 + np.cumsum(np.random.normal(0.0002, 0.008, len(dates))),
        'DXY': 100 + np.cumsum(np.random.normal(0.0001, 0.005, len(dates))),
        'spy_momentum_5d': np.random.randn(len(dates)) * 0.02,
        'vix_momentum_5d': np.random.randn(len(dates)) * 0.05,
        'risk_regime_score': np.random.uniform(-0.5, 0.5, len(dates)),
        'beta_spy_20d': np.random.uniform(0.5, 1.5, len(dates)),
        'risk_regime': np.random.choice(['risk_on', 'neutral', 'risk_off'], len(dates))
    }, index=dates)

    macro_state = {
        'vix_regime': 'normal',
        'risk_regime': 'risk_on',
        'risk_score': 0.3,
        'correlation_breakdown': False,
        'vix_level': 18.5
    }

    results = {}

    # Test 1: Adaptive Feature Selection
    print("\n[1/8] Testing AdaptiveMacroFeatureSelector...")
    try:
        selector = AdaptiveMacroFeatureSelector()
        regime = selector.get_regime_from_macro_state(macro_state)
        features = selector.select_regime_optimized_features(
            regime, list(macro_features.columns)
        )
        print(f"  Regime: {regime}")
        print(f"  Selected features: {len(features)}")
        results['feature_selector'] = 'PASS'
        print("  [OK] AdaptiveMacroFeatureSelector works")
    except Exception as e:
        results['feature_selector'] = f'FAIL: {e}'
        print(f"  [FAIL] {e}")

    # Test 2: Macro-Aware Ensemble Weighting
    print("\n[2/8] Testing MacroAwareEnsembleWeighter...")
    try:
        weighter = MacroAwareEnsembleWeighter()
        weights = weighter.get_regime_optimized_weights(macro_state)
        print(f"  Weights: {weights}")
        assert abs(sum(weights.values()) - 1.0) < 0.01, "Weights should sum to 1"
        results['ensemble_weighter'] = 'PASS'
        print("  [OK] MacroAwareEnsembleWeighter works")
    except Exception as e:
        results['ensemble_weighter'] = f'FAIL: {e}'
        print(f"  [FAIL] {e}")

    # Test 3: Dynamic Macro Sensitivity
    print("\n[3/8] Testing DynamicMacroSensitivity...")
    try:
        sensitivity = DynamicMacroSensitivity()
        returns = pd.Series(np.random.randn(len(dates)) * 0.02, index=dates)
        opt_sens, correlations = sensitivity.calculate_optimal_sensitivity(
            macro_features, returns
        )
        print(f"  Optimal sensitivity: {opt_sens:.2f}")
        print(f"  Features analyzed: {len(correlations)}")
        results['sensitivity'] = 'PASS'
        print("  [OK] DynamicMacroSensitivity works")
    except Exception as e:
        results['sensitivity'] = f'FAIL: {e}'
        print(f"  [FAIL] {e}")

    # Test 4: Macro Regime Forecaster
    print("\n[4/8] Testing MacroRegimeForecaster...")
    try:
        forecaster = MacroRegimeForecaster()
        forecast = forecaster.forecast_regime_transitions(macro_features, 'risk_on')
        print(f"  Transition prob: {forecast['transition_probability']:.2f}")
        print(f"  Days to transition: {forecast['expected_days_to_transition']}")
        results['forecaster'] = 'PASS'
        print("  [OK] MacroRegimeForecaster works")
    except Exception as e:
        results['forecaster'] = f'FAIL: {e}'
        print(f"  [FAIL] {e}")

    # Test 5: Macro Feature Compressor
    print("\n[5/8] Testing MacroFeatureCompressor...")
    try:
        compressor = MacroFeatureCompressor(n_components=5)
        numeric_cols = macro_features.select_dtypes(include=[np.number]).columns
        compressed = compressor.fit_transform(macro_features[numeric_cols])
        print(f"  Original features: {len(numeric_cols)}")
        print(f"  Compressed features: {len(compressed.columns)}")
        results['compressor'] = 'PASS'
        print("  [OK] MacroFeatureCompressor works")
    except Exception as e:
        results['compressor'] = f'FAIL: {e}'
        print(f"  [FAIL] {e}")

    # Test 6: Macro Impact Monitor
    print("\n[6/8] Testing MacroImpactMonitor...")
    try:
        monitor = MacroImpactMonitor()
        predictions = {
            'vix_signal': np.random.randn(50),
            'spy_signal': np.random.randn(50)
        }
        actual = np.random.randn(50)
        metrics = monitor.track_macro_feature_performance(predictions, actual, macro_state)
        print(f"  Features tracked: {len(metrics)}")
        results['monitor'] = 'PASS'
        print("  [OK] MacroImpactMonitor works")
    except Exception as e:
        results['monitor'] = f'FAIL: {e}'
        print(f"  [FAIL] {e}")

    # Test 7: Macro-Aware Position Sizer
    print("\n[7/8] Testing MacroAwarePositionSizer...")
    try:
        sizer = MacroAwarePositionSizer()
        position = sizer.get_macro_enhanced_position(0.5, macro_state, 'equity')
        print(f"  Base position: {position['base_position']:.2f}")
        print(f"  Final position: {position['final_position']:.2f}")
        print(f"  Enhancement: {position['macro_enhancement']:.2f}")
        results['position_sizer'] = 'PASS'
        print("  [OK] MacroAwarePositionSizer works")
    except Exception as e:
        results['position_sizer'] = f'FAIL: {e}'
        print(f"  [FAIL] {e}")

    # Test 8: Multi-Timeframe Analyzer
    print("\n[8/8] Testing MultiTimeframeMacroAnalyzer...")
    try:
        mtf = MultiTimeframeMacroAnalyzer()
        analysis = mtf.analyze_macro_across_timeframes(macro_features)
        print(f"  Timeframes analyzed: {len(analysis['timeframe_analysis'])}")
        print(f"  Agreement: {analysis['timeframe_agreement']:.2f}")
        print(f"  Primary TF: {analysis['primary_timeframe']}")
        results['mtf_analyzer'] = 'PASS'
        print("  [OK] MultiTimeframeMacroAnalyzer works")
    except Exception as e:
        results['mtf_analyzer'] = f'FAIL: {e}'
        print(f"  [FAIL] {e}")

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 4 ELITE ENHANCEMENTS TEST RESULTS")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v == 'PASS')
    total = len(results)

    for name, result in results.items():
        status = "[PASS]" if result == 'PASS' else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\nTotal: {passed}/{total} components passed")

    if passed == total:
        print("\n[SUCCESS] All Phase 4 Elite Enhancements are working!")
        print("Expected additional improvement: +3-5% profit rate")
        print("Target profit rate: 63-68%")
    else:
        print(f"\n[WARNING] {total - passed} component(s) need attention")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
