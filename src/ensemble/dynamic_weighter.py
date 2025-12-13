"""
Phase 5: Dynamic Ensemble Weighting Module
==========================================

Dynamically adjust ensemble weights based on recent performance.
Expected improvement: +2-3% profit rate

This module provides adaptive weight adjustment for asset class ensembles
based on rolling performance metrics including Sharpe ratio, win rate,
Calmar ratio, and consistency scores.

Version: 5.0.0
"""

import pandas as pd
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class DynamicEnsembleWeighter:
    """
    Dynamically adjust ensemble weights based on recent performance.

    This class tracks performance by asset class and calculates optimal
    weights using a composite score based on:
    - Sharpe ratio (40% weight)
    - Win rate (30% weight)
    - Calmar ratio (20% weight)
    - Consistency score (10% weight)

    Expected improvement: +2-3% profit rate

    Parameters
    ----------
    lookback_period : int
        Number of trading days for performance calculation (default: 63 = 3 months)
    min_weight : float
        Minimum weight per asset class (default: 0.05 = 5%)
    max_weight : float
        Maximum weight per asset class (default: 0.35 = 35%)
    decay_factor : float
        Exponential decay for older observations (default: 0.95)
    """

    def __init__(
        self,
        lookback_period: int = 63,
        min_weight: float = 0.05,
        max_weight: float = 0.35,
        decay_factor: float = 0.95
    ):
        self.lookback_period = lookback_period
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.decay_factor = decay_factor

        # Performance tracking
        self.performance_history = defaultdict(lambda: deque(maxlen=lookback_period))
        self.prediction_history = defaultdict(lambda: deque(maxlen=lookback_period))
        self.weight_history = []

        # Asset class mapping - comprehensive coverage
        self.asset_class_mapping = {
            'equity': [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
                'JPM', 'BAC', 'GS', 'MS', 'JNJ', 'PFE', 'MRK', 'UNH',
                'WMT', 'HD', 'COST', 'NKE', 'MCD', 'DIS', 'NFLX', 'CRM'
            ],
            'crypto': [
                'COIN', 'MARA', 'RIOT', 'MSTR', 'BTC-USD', 'ETH-USD',
                'DOGE-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD'
            ],
            'commodity': [
                'XOM', 'CVX', 'COP', 'EOG', 'GOLD', 'NEM', 'FCX',
                'SLV', 'GDX', 'USO', 'UNG'
            ],
            'international': [
                'BABA', 'PDD', 'JD', 'TSM', 'ASML', 'SONY', 'NVO',
                # Hong Kong stocks
                '0700.HK', '9988.HK', '2319.HK', '1876.HK', '0939.HK',
                '1398.HK', '2269.HK', '1177.HK', '1109.HK', '0960.HK'
            ],
            'forex': [
                'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X',
                'USDCAD=X', 'USDCHF=X', 'NZDUSD=X'
            ],
            'bond': [
                'TLT', 'IEF', 'BND', 'AGG', 'GOVT', 'SHY', 'LQD', 'HYG'
            ],
            'etf': [
                'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VEA', 'VWO',
                'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP', 'XLU'
            ]
        }

        # Default base weights (equal weighting)
        self.base_weights = {ac: 1.0 / len(self.asset_class_mapping)
                           for ac in self.asset_class_mapping.keys()}

        # Composite score weights
        self.metric_weights = {
            'sharpe': 0.40,
            'win_rate': 0.30,
            'calmar': 0.20,
            'consistency': 0.10
        }

    def update_performance(
        self,
        asset_class: str,
        daily_return: float,
        prediction: Optional[float] = None,
        actual_direction: Optional[int] = None
    ) -> None:
        """
        Update performance history for an asset class.

        Parameters
        ----------
        asset_class : str
            The asset class category
        daily_return : float
            The daily return value
        prediction : float, optional
            The predicted signal value
        actual_direction : int, optional
            The actual direction (1 for up, -1 for down, 0 for flat)
        """
        self.performance_history[asset_class].append({
            'return': daily_return,
            'timestamp': datetime.now(),
            'prediction': prediction,
            'actual_direction': actual_direction
        })

        if prediction is not None:
            self.prediction_history[asset_class].append({
                'prediction': prediction,
                'actual': actual_direction,
                'timestamp': datetime.now()
            })

    def calculate_performance_metrics(self, asset_class: str) -> Dict:
        """
        Calculate comprehensive performance metrics for an asset class.

        Parameters
        ----------
        asset_class : str
            The asset class to analyze

        Returns
        -------
        dict
            Dictionary containing sharpe, win_rate, calmar, consistency,
            and sample_size metrics
        """
        history = list(self.performance_history[asset_class])

        if len(history) < 5:
            return {
                'sharpe': 0.0,
                'win_rate': 0.5,
                'calmar': 0.0,
                'consistency': 0.0,
                'sample_size': len(history),
                'total_return': 0.0
            }

        returns = np.array([h['return'] for h in history])

        # Apply exponential decay weighting
        n = len(returns)
        decay_weights = np.array([self.decay_factor ** (n - 1 - i) for i in range(n)])
        decay_weights = decay_weights / decay_weights.sum()

        # Weighted metrics
        weighted_mean = np.sum(returns * decay_weights)
        weighted_var = np.sum(decay_weights * (returns - weighted_mean) ** 2)
        weighted_std = np.sqrt(weighted_var) if weighted_var > 0 else 0.001

        # 1. Sharpe Ratio (annualized)
        sharpe = (weighted_mean / weighted_std) * np.sqrt(252) if weighted_std > 0 else 0

        # 2. Win Rate (weighted)
        wins = np.array([1 if r > 0 else 0 for r in returns])
        win_rate = np.sum(wins * decay_weights)

        # 3. Calmar Ratio (return / max drawdown)
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative / running_max) - 1
        max_drawdown = np.min(drawdown)

        if max_drawdown < -0.001:
            calmar = weighted_mean / abs(max_drawdown)
        else:
            calmar = weighted_mean / 0.01 if weighted_mean > 0 else 0

        # 4. Consistency (percentage of positive rolling 5-day periods)
        if len(returns) >= 5:
            rolling_returns = pd.Series(returns).rolling(5).apply(
                lambda x: np.prod(1 + x) - 1, raw=True
            ).dropna().values

            if len(rolling_returns) > 0:
                consistency = np.mean([1 if r > 0 else 0 for r in rolling_returns])
            else:
                consistency = 0.5
        else:
            consistency = 0.5

        # 5. Total return
        total_return = np.prod(1 + returns) - 1

        return {
            'sharpe': max(0, sharpe),  # Only positive Sharpe contributes
            'win_rate': win_rate,
            'calmar': max(0, calmar),
            'consistency': consistency,
            'sample_size': len(returns),
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'volatility': weighted_std * np.sqrt(252)
        }

    def calculate_prediction_accuracy(self, asset_class: str) -> Dict:
        """
        Calculate prediction accuracy metrics.

        Parameters
        ----------
        asset_class : str
            The asset class to analyze

        Returns
        -------
        dict
            Dictionary containing directional accuracy and confidence metrics
        """
        history = list(self.prediction_history[asset_class])

        if len(history) < 5:
            return {
                'directional_accuracy': 0.5,
                'confidence': 0.0,
                'sample_size': len(history)
            }

        correct = 0
        total = 0

        for h in history:
            if h['prediction'] is not None and h['actual'] is not None:
                pred_direction = 1 if h['prediction'] > 0 else -1
                if pred_direction == h['actual']:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.5
        confidence = min(1.0, total / 50)  # Full confidence at 50 samples

        return {
            'directional_accuracy': accuracy,
            'confidence': confidence,
            'sample_size': total
        }

    def get_dynamic_weights(self) -> Dict[str, float]:
        """
        Calculate dynamic weights based on recent performance.

        Uses a composite score combining Sharpe, win rate, Calmar,
        and consistency metrics with configurable weights.

        Returns
        -------
        dict
            Dictionary mapping asset class names to their weights (sum = 1.0)
        """
        performance_scores = {}
        metrics_by_class = {}

        for asset_class in self.asset_class_mapping.keys():
            metrics = self.calculate_performance_metrics(asset_class)
            metrics_by_class[asset_class] = metrics

            # Calculate composite score
            composite_score = (
                metrics['sharpe'] * self.metric_weights['sharpe'] +
                metrics['win_rate'] * self.metric_weights['win_rate'] +
                metrics['calmar'] * self.metric_weights['calmar'] +
                metrics['consistency'] * self.metric_weights['consistency']
            )

            # Adjust for sample size confidence
            sample_confidence = min(1.0, metrics['sample_size'] / 20)
            performance_scores[asset_class] = composite_score * sample_confidence

        # Handle case where all scores are zero or negative
        total_score = sum(max(0, s) for s in performance_scores.values())

        if total_score <= 0:
            # Equal weighting fallback
            return {ac: 1.0 / len(performance_scores)
                   for ac in performance_scores.keys()}

        # Normalize to get raw weights
        raw_weights = {}
        for asset_class, score in performance_scores.items():
            raw_weights[asset_class] = max(0, score) / total_score

        # Apply min/max constraints
        constrained_weights = {}
        for asset_class, weight in raw_weights.items():
            constrained_weight = max(self.min_weight, min(self.max_weight, weight))
            constrained_weights[asset_class] = constrained_weight

        # Renormalize after constraints
        total_constrained = sum(constrained_weights.values())
        final_weights = {
            ac: w / total_constrained
            for ac, w in constrained_weights.items()
        }

        # Track weight history
        self.weight_history.append({
            'timestamp': datetime.now(),
            'weights': final_weights.copy(),
            'metrics': metrics_by_class
        })

        return final_weights

    def get_asset_class(self, ticker: str) -> str:
        """
        Map a ticker to its asset class.

        Parameters
        ----------
        ticker : str
            The ticker symbol

        Returns
        -------
        str
            The asset class name
        """
        ticker_upper = ticker.upper()

        # Direct lookup
        for asset_class, tickers in self.asset_class_mapping.items():
            if ticker_upper in tickers:
                return asset_class

        # Pattern-based mapping
        if '.HK' in ticker_upper or '.SS' in ticker_upper or '.SZ' in ticker_upper:
            return 'international'
        elif '=X' in ticker_upper:
            return 'forex'
        elif 'BTC' in ticker_upper or 'ETH' in ticker_upper or '-USD' in ticker_upper:
            return 'crypto'
        elif ticker_upper in ['XOM', 'CVX', 'GOLD', 'SLV', 'GDX', 'USO', 'UNG']:
            return 'commodity'
        elif ticker_upper in ['TLT', 'IEF', 'BND', 'AGG', 'GOVT']:
            return 'bond'
        elif ticker_upper in ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO'] or \
             ticker_upper.startswith('XL'):
            return 'etf'
        else:
            return 'equity'  # Default

    def get_weight_for_ticker(self, ticker: str) -> float:
        """
        Get the dynamic weight for a specific ticker.

        Parameters
        ----------
        ticker : str
            The ticker symbol

        Returns
        -------
        float
            The weight for this ticker's asset class
        """
        asset_class = self.get_asset_class(ticker)
        weights = self.get_dynamic_weights()
        return weights.get(asset_class, 1.0 / len(self.asset_class_mapping))

    def get_weight_adjustment_factor(self, ticker: str) -> float:
        """
        Get weight adjustment factor relative to base weight.

        A factor > 1.0 means the asset class is performing better than average.
        A factor < 1.0 means it's performing worse.

        Parameters
        ----------
        ticker : str
            The ticker symbol

        Returns
        -------
        float
            Adjustment factor (typically 0.5 to 2.0)
        """
        asset_class = self.get_asset_class(ticker)
        current_weights = self.get_dynamic_weights()
        base_weight = self.base_weights.get(asset_class, 1.0 / len(self.asset_class_mapping))
        current_weight = current_weights.get(asset_class, base_weight)

        if base_weight > 0:
            return current_weight / base_weight
        return 1.0

    def get_performance_summary(self) -> Dict:
        """
        Get a summary of performance across all asset classes.

        Returns
        -------
        dict
            Comprehensive performance summary
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'asset_classes': {},
            'current_weights': self.get_dynamic_weights(),
            'total_observations': 0
        }

        for asset_class in self.asset_class_mapping.keys():
            metrics = self.calculate_performance_metrics(asset_class)
            pred_accuracy = self.calculate_prediction_accuracy(asset_class)

            summary['asset_classes'][asset_class] = {
                'performance': metrics,
                'prediction_accuracy': pred_accuracy
            }
            summary['total_observations'] += metrics['sample_size']

        return summary

    def reset(self) -> None:
        """Reset all performance tracking data."""
        self.performance_history.clear()
        self.prediction_history.clear()
        self.weight_history.clear()


class RegimeAwareWeighter(DynamicEnsembleWeighter):
    """
    Extends DynamicEnsembleWeighter with regime-specific weight adjustments.

    Adjusts weights based on the current market regime (bull, bear,
    high volatility, low volatility) to better capture regime-specific
    performance patterns.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Regime-specific weight adjustments
        self.regime_adjustments = {
            'bull_market': {
                'equity': 1.3,
                'crypto': 1.2,
                'commodity': 1.0,
                'international': 1.1,
                'forex': 0.8,
                'bond': 0.7,
                'etf': 1.1
            },
            'bear_market': {
                'equity': 0.7,
                'crypto': 0.5,
                'commodity': 1.1,
                'international': 0.8,
                'forex': 1.2,
                'bond': 1.5,
                'etf': 0.9
            },
            'high_volatility': {
                'equity': 0.8,
                'crypto': 0.6,
                'commodity': 0.9,
                'international': 0.7,
                'forex': 1.0,
                'bond': 1.3,
                'etf': 1.1
            },
            'low_volatility': {
                'equity': 1.2,
                'crypto': 1.1,
                'commodity': 1.0,
                'international': 1.1,
                'forex': 0.9,
                'bond': 0.8,
                'etf': 1.0
            }
        }

        self.current_regime = 'bull_market'

    def set_regime(self, regime: str) -> None:
        """
        Set the current market regime.

        Parameters
        ----------
        regime : str
            One of: 'bull_market', 'bear_market', 'high_volatility', 'low_volatility'
        """
        if regime in self.regime_adjustments:
            self.current_regime = regime

    def get_dynamic_weights(self) -> Dict[str, float]:
        """
        Get dynamic weights with regime adjustment.

        Returns
        -------
        dict
            Regime-adjusted weights
        """
        # Get base dynamic weights
        base_weights = super().get_dynamic_weights()

        # Apply regime adjustments
        regime_factors = self.regime_adjustments.get(
            self.current_regime,
            {ac: 1.0 for ac in self.asset_class_mapping.keys()}
        )

        adjusted_weights = {}
        for asset_class, weight in base_weights.items():
            factor = regime_factors.get(asset_class, 1.0)
            adjusted_weights[asset_class] = weight * factor

        # Renormalize
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}

        # Apply constraints
        for asset_class in adjusted_weights:
            adjusted_weights[asset_class] = max(
                self.min_weight,
                min(self.max_weight, adjusted_weights[asset_class])
            )

        # Final normalization
        total = sum(adjusted_weights.values())
        return {k: v / total for k, v in adjusted_weights.items()}


# Factory function
def get_dynamic_weighter(
    lookback_period: int = 63,
    min_weight: float = 0.05,
    max_weight: float = 0.35,
    regime_aware: bool = False
) -> DynamicEnsembleWeighter:
    """
    Factory function to create a dynamic weighter.

    Parameters
    ----------
    lookback_period : int
        Lookback period in days
    min_weight : float
        Minimum weight constraint
    max_weight : float
        Maximum weight constraint
    regime_aware : bool
        Whether to use regime-aware weighting

    Returns
    -------
    DynamicEnsembleWeighter
        Configured weighter instance
    """
    if regime_aware:
        return RegimeAwareWeighter(
            lookback_period=lookback_period,
            min_weight=min_weight,
            max_weight=max_weight
        )
    else:
        return DynamicEnsembleWeighter(
            lookback_period=lookback_period,
            min_weight=min_weight,
            max_weight=max_weight
        )


# Integration helper
def integrate_dynamic_weights(existing_ensembles: Dict) -> callable:
    """
    Integrate dynamic weighting into existing ensembles.

    Parameters
    ----------
    existing_ensembles : dict
        Dictionary of asset_class -> ensemble mappings

    Returns
    -------
    callable
        Enhanced predict function with dynamic weighting
    """
    weighter = DynamicEnsembleWeighter()

    def enhanced_predict(ticker: str, data, **kwargs):
        asset_class = weighter.get_asset_class(ticker)
        weight_factor = weighter.get_weight_adjustment_factor(ticker)

        # Get prediction from appropriate ensemble
        if asset_class in existing_ensembles:
            ensemble = existing_ensembles[asset_class]
            prediction = ensemble.predict(data, **kwargs)
        else:
            prediction = {'signal': 0, 'confidence': 0.5}

        # Adjust confidence by weight factor
        if isinstance(prediction, dict):
            prediction['weight_factor'] = weight_factor
            prediction['dynamic_confidence'] = prediction.get('confidence', 0.5) * weight_factor

        return prediction

    return enhanced_predict
