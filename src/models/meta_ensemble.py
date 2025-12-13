"""
Meta-Ensemble Combiner Module

Combines predictions from all specialized asset-class ensembles:
- Intelligent weighting based on recent performance
- Regime-aware signal adjustment
- Dynamic re-weighting
- Integration with Phase 1 features

This is the top-level prediction system that orchestrates all models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import ensemble factory
try:
    from src.models.asset_class_ensembles import AssetClassEnsembleFactory
    ENSEMBLES_AVAILABLE = True
except ImportError:
    ENSEMBLES_AVAILABLE = False
    # Silent - will use fallback

# Import Phase 1 features
try:
    from src.trading.production_advanced import (
        MetaLearner, RegimeSignalWeighter, VolatilityScaler,
        StressHardenedTradingSystem
    )
    PHASE1_AVAILABLE = True
except ImportError:
    PHASE1_AVAILABLE = False


class PerformanceTracker:
    """Track historical performance for dynamic weighting."""

    def __init__(self, max_history: int = 100):
        """
        Initialize performance tracker.

        Args:
            max_history: Maximum predictions to track
        """
        self.max_history = max_history
        self.predictions = {}  # ticker -> deque of (prediction, actual)
        self.ensemble_performance = {}  # ensemble -> performance metrics

    def record_prediction(self, ticker: str, prediction: float, actual: float = None):
        """Record a prediction and optionally its outcome."""
        if ticker not in self.predictions:
            self.predictions[ticker] = deque(maxlen=self.max_history)

        self.predictions[ticker].append({
            'prediction': prediction,
            'actual': actual,
            'timestamp': datetime.now()
        })

    def update_actual(self, ticker: str, actual: float):
        """Update the most recent prediction with actual outcome."""
        if ticker in self.predictions and len(self.predictions[ticker]) > 0:
            self.predictions[ticker][-1]['actual'] = actual

    def get_ticker_accuracy(self, ticker: str) -> float:
        """Get accuracy for a specific ticker."""
        if ticker not in self.predictions:
            return 0.5  # Default

        predictions = list(self.predictions[ticker])
        valid = [(p['prediction'], p['actual']) for p in predictions
                 if p['actual'] is not None]

        if len(valid) < 5:
            return 0.5  # Not enough data

        correct = sum(1 for pred, actual in valid
                     if (pred > 0 and actual > 0) or (pred < 0 and actual < 0))
        return correct / len(valid)

    def get_ensemble_accuracy(self, ensemble_name: str) -> float:
        """Get accuracy for an ensemble type."""
        return self.ensemble_performance.get(ensemble_name, {}).get('accuracy', 0.5)

    def update_ensemble_performance(self, ensemble_name: str, accuracy: float, sharpe: float = None):
        """Update performance metrics for an ensemble."""
        self.ensemble_performance[ensemble_name] = {
            'accuracy': accuracy,
            'sharpe': sharpe,
            'last_updated': datetime.now()
        }


class MetaEnsembleCombiner:
    """
    Meta-ensemble that combines predictions from all specialized ensembles.

    Features:
    - Routes tickers to appropriate ensembles
    - Combines multiple ensemble predictions
    - Dynamic weighting based on performance
    - Integration with Phase 1 stress protection
    """

    # Default ensemble weights (based on historical category performance)
    DEFAULT_WEIGHTS = {
        'equity': 0.20,
        'forex': 0.15,
        'crypto': 0.10,
        'commodity': 0.15,
        'international': 0.15,
        'bond': 0.10,
        'etf': 0.15
    }

    def __init__(self, config: Dict = None):
        """
        Initialize meta-ensemble.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize ensemble factory
        if ENSEMBLES_AVAILABLE:
            self.ensemble_factory = AssetClassEnsembleFactory()
        else:
            self.ensemble_factory = None
            print("[WARN] Asset class ensembles not available")

        # Initialize Phase 1 components
        if PHASE1_AVAILABLE:
            self.meta_learner = MetaLearner()
            self.regime_weighter = RegimeSignalWeighter()
            self.volatility_scaler = VolatilityScaler()
            self.stress_system = StressHardenedTradingSystem()
        else:
            self.meta_learner = None
            self.regime_weighter = None
            self.volatility_scaler = None
            self.stress_system = None
            print("[WARN] Phase 1 features not available")

        # Performance tracking
        self.performance_tracker = PerformanceTracker()

        # Dynamic weights (initialized from defaults)
        self.ensemble_weights = self.DEFAULT_WEIGHTS.copy()

    def classify_asset(self, ticker: str) -> str:
        """Classify an asset into its asset class."""
        if self.ensemble_factory:
            return self.ensemble_factory.classify_asset(ticker)
        return 'equity'  # Default

    def get_ensemble_prediction(self, data: pd.DataFrame, ticker: str) -> Dict:
        """
        Get prediction from the appropriate specialized ensemble.

        Args:
            data: OHLCV data
            ticker: Asset ticker

        Returns:
            Prediction dictionary
        """
        if not self.ensemble_factory:
            return self._fallback_prediction(data, ticker)

        return self.ensemble_factory.predict(data, ticker)

    def _fallback_prediction(self, data: pd.DataFrame, ticker: str) -> Dict:
        """Fallback prediction when ensembles aren't available."""
        if len(data) < 20:
            return {'ticker': ticker, 'signal': 0, 'confidence': 0}

        close = data['Close']
        returns = close.pct_change()

        # Simple momentum signal
        momentum_20d = close.pct_change(20).iloc[-1]
        signal = np.tanh(momentum_20d * 5)

        return {
            'ticker': ticker,
            'signal': signal,
            'confidence': 0.5,
            'asset_class': 'unknown'
        }

    def apply_phase1_adjustments(self, prediction: Dict, data: pd.DataFrame,
                                  market_conditions: Dict = None) -> Dict:
        """
        Apply Phase 1 advanced feature adjustments.

        Args:
            prediction: Raw prediction from ensemble
            data: OHLCV data
            market_conditions: Current market conditions

        Returns:
            Adjusted prediction
        """
        adjusted = prediction.copy()
        signal = prediction.get('signal', 0)

        market_conditions = market_conditions or {}

        # Apply regime weighting
        if self.regime_weighter:
            regime = market_conditions.get('regime', 'normal')
            signal = self.regime_weighter.regime_aware_signal_weighting(signal, regime)
            adjusted['regime_adjusted'] = True

        # Apply volatility scaling
        if self.volatility_scaler and len(data) >= 20:
            returns = data['Close'].pct_change().dropna()
            vol_adjustment = self.volatility_scaler.get_volatility_adjustment(returns)
            signal = signal * vol_adjustment
            adjusted['vol_scaled'] = True

        # Apply stress system checks
        if self.stress_system:
            vix = market_conditions.get('vix', 20)
            market_return = market_conditions.get('market_return', 0)
            self.stress_system.update_market_conditions(vix, market_return)

            trading_allowed, reason = self.stress_system.check_trade_allowed()
            if not trading_allowed:
                signal = 0
                adjusted['stress_blocked'] = True
                adjusted['block_reason'] = reason
            else:
                position_mult = self.stress_system.get_adjusted_position_size(1.0)
                signal = signal * position_mult
                adjusted['stress_multiplier'] = position_mult

        adjusted['signal'] = float(np.clip(signal, -1, 1))
        return adjusted

    def combine_predictions(self, predictions: List[Dict]) -> Dict:
        """
        Combine multiple predictions (if applicable).

        Args:
            predictions: List of predictions from different sources

        Returns:
            Combined prediction
        """
        if not predictions:
            return {'signal': 0, 'confidence': 0}

        if len(predictions) == 1:
            return predictions[0]

        # Weighted combination
        total_weight = 0
        weighted_signal = 0
        weighted_confidence = 0

        for pred in predictions:
            asset_class = pred.get('asset_class', 'equity')
            weight = self.ensemble_weights.get(asset_class, 0.1)

            weighted_signal += pred.get('signal', 0) * weight * pred.get('confidence', 0.5)
            weighted_confidence += pred.get('confidence', 0.5) * weight
            total_weight += weight

        if total_weight > 0:
            combined_signal = weighted_signal / total_weight
            combined_confidence = weighted_confidence / total_weight
        else:
            combined_signal = 0
            combined_confidence = 0

        return {
            'signal': float(np.clip(combined_signal, -1, 1)),
            'confidence': float(np.clip(combined_confidence, 0, 1)),
            'predictions_combined': len(predictions)
        }

    def predict(self, data: pd.DataFrame, ticker: str,
                market_conditions: Dict = None) -> Dict:
        """
        Generate final prediction for an asset.

        Args:
            data: OHLCV data
            ticker: Asset ticker
            market_conditions: Current market conditions (vix, regime, etc.)

        Returns:
            Complete prediction with all adjustments
        """
        # Get base prediction from specialized ensemble
        base_prediction = self.get_ensemble_prediction(data, ticker)

        # Apply Phase 1 adjustments
        adjusted_prediction = self.apply_phase1_adjustments(
            base_prediction, data, market_conditions
        )

        # Calculate final recommendation
        signal = adjusted_prediction.get('signal', 0)
        confidence = adjusted_prediction.get('confidence', 0.5)

        # Generate action recommendation
        if abs(signal) < 0.1 or confidence < 0.3:
            action = 'HOLD'
        elif signal > 0.3 and confidence > 0.5:
            action = 'STRONG_BUY'
        elif signal > 0:
            action = 'BUY'
        elif signal < -0.3 and confidence > 0.5:
            action = 'STRONG_SELL'
        else:
            action = 'SELL'

        # Record prediction for tracking
        self.performance_tracker.record_prediction(ticker, signal)

        return {
            **adjusted_prediction,
            'action': action,
            'ticker': ticker,
            'asset_class': self.classify_asset(ticker),
            'timestamp': datetime.now().isoformat()
        }

    def predict_batch(self, data_dict: Dict[str, pd.DataFrame],
                      market_conditions: Dict = None) -> Dict[str, Dict]:
        """
        Generate predictions for multiple assets.

        Args:
            data_dict: Dict of ticker -> OHLCV data
            market_conditions: Current market conditions

        Returns:
            Dict of ticker -> prediction
        """
        results = {}
        for ticker, data in data_dict.items():
            try:
                results[ticker] = self.predict(data, ticker, market_conditions)
            except Exception as e:
                results[ticker] = {
                    'ticker': ticker,
                    'signal': 0,
                    'confidence': 0,
                    'error': str(e)
                }
        return results

    def update_weights(self, performance_data: Dict[str, float]):
        """
        Update ensemble weights based on recent performance.

        Args:
            performance_data: Dict of ensemble_name -> accuracy
        """
        if not performance_data:
            return

        # Calculate performance-weighted adjustment
        total_perf = sum(performance_data.values())
        if total_perf <= 0:
            return

        for ensemble_name, accuracy in performance_data.items():
            if ensemble_name in self.ensemble_weights:
                # Adjust weight based on relative performance
                # Better performance -> higher weight
                base_weight = self.DEFAULT_WEIGHTS.get(ensemble_name, 0.1)
                perf_factor = accuracy / (total_perf / len(performance_data))
                new_weight = base_weight * perf_factor

                # Smooth adjustment (don't change too fast)
                self.ensemble_weights[ensemble_name] = (
                    0.7 * self.ensemble_weights[ensemble_name] +
                    0.3 * new_weight
                )

        # Normalize weights to sum to 1
        total = sum(self.ensemble_weights.values())
        if total > 0:
            self.ensemble_weights = {k: v/total for k, v in self.ensemble_weights.items()}

    def get_model_recommendation(self, ticker: str, market_conditions: Dict = None) -> str:
        """
        Get recommended model type for a ticker.

        Uses Phase 1 MetaLearner if available.

        Args:
            ticker: Asset ticker
            market_conditions: Current conditions

        Returns:
            Recommended model name
        """
        if self.meta_learner and market_conditions:
            regime = market_conditions.get('regime', 'normal')
            return self.meta_learner.select_best_model(regime, market_conditions)
        return 'HybridEnsemble'

    def get_status(self) -> Dict:
        """Get current status of the meta-ensemble system."""
        return {
            'ensembles_available': ENSEMBLES_AVAILABLE,
            'phase1_available': PHASE1_AVAILABLE,
            'current_weights': self.ensemble_weights,
            'tracked_tickers': len(self.performance_tracker.predictions),
            'ensemble_types': list(self.DEFAULT_WEIGHTS.keys())
        }


class EnhancedTradingSystem:
    """
    Complete enhanced trading system combining Phase 1 + Phase 2 + Phase 3.

    This is the final integrated system with all features:
    - Phase 1: 20 Advanced Features (stress protection, etc.)
    - Phase 2: Asset-class specific ensembles
    - Phase 3: Structural features (regime detection, etc.)
    """

    def __init__(self, config: Dict = None):
        """
        Initialize enhanced trading system.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Meta-ensemble (combines Phase 2 ensembles)
        self.meta_ensemble = MetaEnsembleCombiner(config)

        # Phase 1 stress system
        if PHASE1_AVAILABLE:
            self.stress_system = StressHardenedTradingSystem()
        else:
            self.stress_system = None

        # System state
        self.is_active = True
        self.last_update = None

    def update_market_conditions(self, vix: float = None, market_return: float = None) -> Dict:
        """
        Update system with current market conditions.

        Args:
            vix: Current VIX level
            market_return: Recent market return

        Returns:
            System status after update
        """
        if self.stress_system:
            return self.stress_system.update_market_conditions(
                vix or 20, market_return or 0
            )
        return {'status': 'ok'}

    def generate_signals(self, data_dict: Dict[str, pd.DataFrame],
                         market_conditions: Dict = None) -> Dict[str, Dict]:
        """
        Generate trading signals for all assets.

        Args:
            data_dict: Dict of ticker -> OHLCV data
            market_conditions: Current market conditions

        Returns:
            Dict of ticker -> trading signal
        """
        # Check if trading is allowed
        if self.stress_system:
            allowed, reason = self.stress_system.check_trade_allowed()
            if not allowed:
                return {
                    ticker: {
                        'ticker': ticker,
                        'signal': 0,
                        'action': 'HALT',
                        'reason': reason
                    }
                    for ticker in data_dict
                }

        # Generate predictions through meta-ensemble
        predictions = self.meta_ensemble.predict_batch(data_dict, market_conditions)

        # Apply position sizing from stress system
        if self.stress_system:
            position_mult = self.stress_system.get_adjusted_position_size(1.0)
            for ticker in predictions:
                predictions[ticker]['position_multiplier'] = position_mult

        self.last_update = datetime.now()
        return predictions

    def get_top_picks(self, predictions: Dict[str, Dict], n: int = 10,
                      direction: str = 'both') -> List[Dict]:
        """
        Get top trading picks from predictions.

        Args:
            predictions: Dict of predictions
            n: Number of picks to return
            direction: 'long', 'short', or 'both'

        Returns:
            List of top picks sorted by signal strength
        """
        picks = []
        for ticker, pred in predictions.items():
            signal = pred.get('signal', 0)
            confidence = pred.get('confidence', 0)

            # Filter by direction
            if direction == 'long' and signal <= 0:
                continue
            if direction == 'short' and signal >= 0:
                continue

            picks.append({
                **pred,
                'score': abs(signal) * confidence
            })

        # Sort by score
        picks.sort(key=lambda x: x['score'], reverse=True)

        return picks[:n]

    def get_system_status(self) -> Dict:
        """Get complete system status."""
        status = {
            'is_active': self.is_active,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'meta_ensemble': self.meta_ensemble.get_status()
        }

        if self.stress_system:
            status['stress_system'] = self.stress_system.get_status_report()

        return status


# Singleton instances
_meta_ensemble = None
_enhanced_system = None


def get_meta_ensemble() -> MetaEnsembleCombiner:
    """Get singleton MetaEnsembleCombiner instance."""
    global _meta_ensemble
    if _meta_ensemble is None:
        _meta_ensemble = MetaEnsembleCombiner()
    return _meta_ensemble


def get_enhanced_system() -> EnhancedTradingSystem:
    """Get singleton EnhancedTradingSystem instance."""
    global _enhanced_system
    if _enhanced_system is None:
        _enhanced_system = EnhancedTradingSystem()
    return _enhanced_system


def main():
    """Test meta-ensemble combiner."""
    import yfinance as yf

    print("=" * 70)
    print("META-ENSEMBLE COMBINER TEST")
    print("=" * 70)

    # Initialize system
    system = get_enhanced_system()
    print(f"\nSystem Status:")
    status = system.get_system_status()
    print(f"  Active: {status['is_active']}")
    print(f"  Ensembles Available: {status['meta_ensemble']['ensembles_available']}")
    print(f"  Phase 1 Available: {status['meta_ensemble']['phase1_available']}")

    # Test predictions
    test_tickers = ['AAPL', 'COIN', 'GLD', 'BABA', 'SPY']
    data_dict = {}

    print(f"\nFetching data for {len(test_tickers)} tickers...")
    for ticker in test_tickers:
        data = yf.download(ticker, period='1y', progress=False)
        if len(data) > 0:
            data_dict[ticker] = data

    print(f"Loaded data for {len(data_dict)} tickers")

    # Generate signals
    market_conditions = {
        'vix': 18,
        'market_return': 0.005,
        'regime': 'normal'
    }

    print(f"\nGenerating signals...")
    predictions = system.generate_signals(data_dict, market_conditions)

    print("\nPredictions:")
    print("-" * 70)
    for ticker, pred in predictions.items():
        print(f"{ticker:8} | Signal: {pred.get('signal', 0):+.3f} | "
              f"Action: {pred.get('action', 'N/A'):12} | "
              f"Class: {pred.get('asset_class', 'N/A'):12} | "
              f"Conf: {pred.get('confidence', 0):.2f}")

    # Get top picks
    print("\nTop 3 Picks (Long):")
    top_picks = system.get_top_picks(predictions, n=3, direction='long')
    for i, pick in enumerate(top_picks, 1):
        print(f"  {i}. {pick['ticker']} - Score: {pick['score']:.3f}")

    print("\n[SUCCESS] Meta-ensemble test complete!")


if __name__ == "__main__":
    main()
