"""
WEEK 7 IMPROVEMENT: Ensemble Model (CatBoost + LSTM)

Combines two complementary approaches:
- CatBoost (70% weight default): Feature-based, gradient boosting
- LSTM (30% weight default): Sequential pattern recognition

WEEK 8 UPDATE (Fix 21): Adaptive ensemble weights based on market regime
- Strong trends: More LSTM weight (captures momentum better)
- Mean-reverting: More CatBoost weight (feature engineering excels)
- High volatility: Balanced approach

Expected improvements:
- +5-10pp pass rate improvement (Week 5 target: 30%+ from 16.7%)
- Better capture of both short-term patterns and long-term trends
- Reduced overfitting through model diversity
- Better regime adaptation with adaptive weights

# ============================================================================
# PROTECTED CORE MODEL - DO NOT MODIFY WITHOUT USER PERMISSION
# This file contains the US/Intl ensemble predictor model.
# Any changes to model architecture, weights, or logic require explicit user approval.
# ============================================================================
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# FIX 21: ADAPTIVE ENSEMBLE WEIGHTS (December 2025)
# =============================================================================
ADAPTIVE_ENSEMBLE_WEIGHTS = {
    # In strong trends: LSTM better at capturing momentum
    'strong_downtrend': {'catboost': 0.50, 'lstm': 0.50},
    'downtrend': {'catboost': 0.45, 'lstm': 0.55},
    'neutral': {'catboost': 0.70, 'lstm': 0.30},     # Default
    'uptrend': {'catboost': 0.45, 'lstm': 0.55},
    'strong_uptrend': {'catboost': 0.40, 'lstm': 0.60},

    # Special regimes
    'mean_reverting': {'catboost': 0.80, 'lstm': 0.20},  # CatBoost excels
    'high_volatility': {'catboost': 0.60, 'lstm': 0.40}, # More balanced
}


def classify_trend(prices: pd.Series, lookback: int = 20) -> str:
    """
    Classify market trend based on price slope and SMA relationship.

    Returns: 'strong_downtrend', 'downtrend', 'neutral', 'uptrend', 'strong_uptrend'
    """
    if prices is None or len(prices) < lookback:
        return 'neutral'

    recent_prices = prices.tail(lookback)
    sma = recent_prices.mean()
    current_price = recent_prices.iloc[-1]

    # Calculate slope via linear regression
    x = np.arange(len(recent_prices))
    slope = np.polyfit(x, recent_prices.values, 1)[0]
    normalized_slope = slope / sma  # Normalize by price level

    # Classification thresholds
    if normalized_slope < -0.02 and current_price < sma * 0.97:
        return 'strong_downtrend'
    elif normalized_slope < -0.005 or current_price < sma * 0.99:
        return 'downtrend'
    elif normalized_slope > 0.02 and current_price > sma * 1.03:
        return 'strong_uptrend'
    elif normalized_slope > 0.005 or current_price > sma * 1.01:
        return 'uptrend'
    else:
        return 'neutral'


def get_adaptive_ensemble_weights(prices: pd.Series = None,
                                   trend: str = None,
                                   volatility: float = None) -> dict:
    """
    Get adaptive ensemble weights based on market conditions.

    Args:
        prices: Price series for trend detection
        trend: Pre-computed trend (optional)
        volatility: Daily volatility (optional)

    Returns:
        dict with 'catboost' and 'lstm' weights
    """
    if trend is None:
        trend = classify_trend(prices) if prices is not None else 'neutral'

    # Check for mean reversion (oscillating around mean)
    if prices is not None and len(prices) >= 20:
        recent = prices.tail(20)
        crossings = ((recent > recent.mean()).diff().abs().sum())
        if crossings >= 6:  # Many mean crossings = mean reverting
            trend = 'mean_reverting'

    # Check for high volatility
    if volatility is not None and volatility > 0.03:  # >3% daily vol
        trend = 'high_volatility'

    return ADAPTIVE_ENSEMBLE_WEIGHTS.get(trend, ADAPTIVE_ENSEMBLE_WEIGHTS['neutral'])

from .market_specific_predictor import MarketSpecificPredictor

# Import feature engineering - handle both relative and absolute imports
try:
    from ..features.feature_engineering import create_features
except ImportError:
    from features.feature_engineering import create_features

try:
    from .lstm_predictor import LSTMPredictor
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("WARNING: LSTM predictor not available. Install TensorFlow to enable ensemble mode.")


class EnsemblePredictor:
    """
    Ensemble predictor combining CatBoost and LSTM with adaptive weights.

    Architecture:
    - CatBoost: Market-specific gradient boosting (feature-based)
    - LSTM: Sequential pattern recognition (time-series)
    - Weighted voting: Adaptive based on market regime (Fix 21)

    Default weights: 70% CatBoost, 30% LSTM
    Adaptive weights vary by regime:
    - Strong trends: 40-50% CatBoost, 50-60% LSTM
    - Mean-reverting: 80% CatBoost, 20% LSTM
    - High volatility: 60% CatBoost, 40% LSTM

    This combines the strengths of both:
    - CatBoost excels at feature relationships and non-linear patterns
    - LSTM excels at sequential dependencies and momentum
    """

    def __init__(
        self,
        market_type,
        catboost_weight=0.7,
        lstm_weight=0.3,
        confidence_threshold=0.50,
        lstm_lookback=20,
        verbose=False,
        use_adaptive_weights=True  # Fix 21: Enable adaptive weights
    ):
        """
        Initialize ensemble predictor.

        Args:
            market_type: 'HK', 'SS', or 'SZ'
            catboost_weight: Weight for CatBoost predictions (default: 0.7)
            lstm_weight: Weight for LSTM predictions (default: 0.3)
            confidence_threshold: Minimum probability for 'up' prediction (default: 0.50)
            lstm_lookback: Lookback period for LSTM sequences (default: 20)
            verbose: Print training progress (default: False)
            use_adaptive_weights: Use Fix 21 adaptive weights (default: True)
        """
        if not np.isclose(catboost_weight + lstm_weight, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {catboost_weight} + {lstm_weight} = {catboost_weight + lstm_weight}")

        self.market_type = market_type
        self.catboost_weight = catboost_weight
        self.lstm_weight = lstm_weight
        self.confidence_threshold = confidence_threshold
        self.lstm_lookback = lstm_lookback
        self.verbose = verbose
        self.use_adaptive_weights = use_adaptive_weights  # Fix 21

        # Store last used weights for reporting
        self._last_weights = {'catboost': catboost_weight, 'lstm': lstm_weight}
        self._last_regime = 'neutral'

        # Initialize component models
        self.catboost_model = MarketSpecificPredictor(
            market_type=market_type,
            confidence_threshold=confidence_threshold,
            verbose=verbose
        )

        if not LSTM_AVAILABLE:
            raise ImportError("TensorFlow is required for ensemble mode. Install with: pip install tensorflow")

        self.lstm_model = LSTMPredictor(
            lookback_period=lstm_lookback,
            verbose=verbose
        )

        self.feature_names = None

    def train(self, ohlcv_data, y=None):
        """
        Train both CatBoost and LSTM models.

        Args:
            ohlcv_data: DataFrame with OHLCV columns (Open, High, Low, Close, Volume)
            y: Not used (both models generate their own targets from price data)
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("ENSEMBLE MODEL TRAINING")
            print("=" * 80)
            print(f"  Market: {self.market_type}")
            print(f"  Weights: CatBoost {self.catboost_weight:.0%}, LSTM {self.lstm_weight:.0%}")
            print(f"  Training samples: {len(ohlcv_data)}")
            print(f"  LSTM lookback: {self.lstm_lookback} days")

        # Train CatBoost (expects raw OHLCV data)
        if self.verbose:
            print("\n" + "-" * 80)
            print("TRAINING CATBOOST MODEL (70% weight)")
            print("-" * 80)

        self.catboost_model.fit(ohlcv_data)

        # Train LSTM (needs features generated from OHLCV data)
        if self.verbose:
            print("\n" + "-" * 80)
            print("TRAINING LSTM MODEL (30% weight)")
            print("-" * 80)

        # Generate features for LSTM from raw OHLCV data
        features_df = create_features(ohlcv_data)
        X = features_df.drop('target', axis=1)
        y = features_df['target']

        if self.verbose:
            print(f"  Features generated: {X.shape[1]} features, {len(X)} samples")

        self.lstm_model.train(X, y)

        if self.verbose:
            print("\n" + "=" * 80)
            print("ENSEMBLE TRAINING COMPLETE")
            print("=" * 80)

    def predict_proba(self, X, prices: pd.Series = None, volatility: float = None):
        """
        Predict probabilities using ensemble with adaptive weights (Fix 21).

        Args:
            X: Features (n_samples, n_features)
            prices: Price series for adaptive weight calculation (optional)
            volatility: Daily volatility for regime detection (optional)

        Returns:
            proba: Weighted ensemble probabilities (n_samples, 2) - [prob_down, prob_up]
        """
        # Get predictions from both models
        catboost_proba = self.catboost_model.predict_proba(X)
        lstm_proba = self.lstm_model.predict_proba(X)

        # Fix 21: Get adaptive weights based on market regime
        if self.use_adaptive_weights and prices is not None:
            weights = get_adaptive_ensemble_weights(prices=prices, volatility=volatility)
            catboost_w = weights['catboost']
            lstm_w = weights['lstm']

            # Store for reporting
            self._last_weights = weights
            self._last_regime = classify_trend(prices) if prices is not None else 'neutral'

            if self.verbose:
                print(f"  [Fix 21] Adaptive weights: CatBoost={catboost_w:.0%}, LSTM={lstm_w:.0%} (regime: {self._last_regime})")
        else:
            # Use default fixed weights
            catboost_w = self.catboost_weight
            lstm_w = self.lstm_weight

        # Weighted average
        ensemble_proba = (
            catboost_w * catboost_proba +
            lstm_w * lstm_proba
        )

        return ensemble_proba

    def predict_proba_adaptive(self, X, prices: pd.Series, volatility: float = None):
        """
        Predict probabilities using adaptive ensemble weights (Fix 21).

        Convenience method that always uses adaptive weights.

        Args:
            X: Features (n_samples, n_features)
            prices: Price series for trend detection (required)
            volatility: Daily volatility (optional)

        Returns:
            proba: Weighted ensemble probabilities (n_samples, 2)
        """
        return self.predict_proba(X, prices=prices, volatility=volatility)

    def predict(self, X, threshold=None):
        """
        Predict class labels using ensemble.

        Args:
            X: Features (n_samples, n_features)
            threshold: Probability threshold for 'up' prediction
                      (default: uses self.confidence_threshold)

        Returns:
            predictions: Class labels (n_samples,) - 0 (down) or 1 (up)
        """
        if threshold is None:
            threshold = self.confidence_threshold

        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)

    def get_component_predictions(self, X):
        """
        Get predictions from individual component models.
        Useful for debugging and understanding ensemble behavior.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            dict with:
                - 'catboost_proba': CatBoost probabilities
                - 'lstm_proba': LSTM probabilities
                - 'ensemble_proba': Weighted ensemble probabilities
        """
        catboost_proba = self.catboost_model.predict_proba(X)
        lstm_proba = self.lstm_model.predict_proba(X)
        ensemble_proba = self.predict_proba(X)

        return {
            'catboost_proba': catboost_proba,
            'lstm_proba': lstm_proba,
            'ensemble_proba': ensemble_proba
        }

    def get_feature_importance(self):
        """
        Get feature importance from CatBoost model.

        Note: LSTM doesn't provide traditional feature importance.
        This returns CatBoost importance only.

        Returns:
            dict with feature names and importance scores
        """
        return self.catboost_model.get_feature_importance()

    def get_last_weights(self) -> dict:
        """
        Get the last used ensemble weights (Fix 21).

        Returns:
            dict with 'catboost', 'lstm' weights and 'regime'
        """
        return {
            'catboost': self._last_weights.get('catboost', self.catboost_weight),
            'lstm': self._last_weights.get('lstm', self.lstm_weight),
            'regime': self._last_regime,
            'adaptive': self.use_adaptive_weights
        }

    def set_adaptive_weights(self, enabled: bool = True):
        """
        Enable or disable adaptive weights (Fix 21).

        Args:
            enabled: True to use adaptive weights, False for fixed weights
        """
        self.use_adaptive_weights = enabled
        if self.verbose:
            status = "ENABLED" if enabled else "DISABLED"
            print(f"  [Fix 21] Adaptive ensemble weights: {status}")
