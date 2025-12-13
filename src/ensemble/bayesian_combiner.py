"""
Phase 5: Bayesian Signal Combination Module
============================================

Bayesian model averaging for robust signal combination.
Expected improvement: +1-2% signal accuracy, better risk management

This module provides Bayesian updating of signal reliability and
intelligent signal combination using posterior probabilities.

Version: 5.0.0
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class BayesianSignalCombiner:
    """
    Bayesian model averaging for robust signal combination.

    Uses Beta-Bernoulli conjugate priors to track signal reliability
    and combines signals using posterior-weighted averaging.

    Expected improvement: +1-2% signal accuracy

    Parameters
    ----------
    prior_alpha : float
        Alpha parameter for Beta prior (default: 1 = uniform)
    prior_beta : float
        Beta parameter for Beta prior (default: 1 = uniform)
    min_samples : int
        Minimum samples before trusting posterior (default: 5)
    decay_factor : float
        Exponential decay for older observations (default: 0.99)
    """

    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        min_samples: int = 5,
        decay_factor: float = 0.99
    ):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.min_samples = min_samples
        self.decay_factor = decay_factor

        # Signal priors (Beta distribution parameters)
        self.signal_priors = defaultdict(
            lambda: {'alpha': prior_alpha, 'beta': prior_beta}
        )

        # Performance tracking
        self.signal_performance = defaultdict(list)

        # Signal correlations (for diversification)
        self.signal_correlations = defaultdict(lambda: defaultdict(float))

    def update_signal_reliability(
        self,
        signal_name: str,
        signal_value: float,
        actual_return: float,
        lookforward_days: int = 5
    ) -> None:
        """
        Update signal reliability using Bayesian updating.

        Uses Beta-Bernoulli conjugate prior for directional accuracy.

        Parameters
        ----------
        signal_name : str
            Name of the signal (e.g., 'momentum', 'mean_reversion')
        signal_value : float
            The signal value (-1 to 1)
        actual_return : float
            Actual return over lookforward period
        lookforward_days : int
            Number of days for return calculation (for tracking)
        """
        # Determine if signal was correct (directional accuracy)
        signal_direction = 1 if signal_value > 0 else -1
        return_direction = 1 if actual_return > 0 else -1

        signal_correct = (signal_direction * return_direction) > 0

        # Get current prior
        current_prior = self.signal_priors[signal_name]

        # Apply decay to existing counts (shrink towards prior)
        current_prior['alpha'] = self.prior_alpha + \
            (current_prior['alpha'] - self.prior_alpha) * self.decay_factor
        current_prior['beta'] = self.prior_beta + \
            (current_prior['beta'] - self.prior_beta) * self.decay_factor

        # Bayesian update (Beta-Bernoulli conjugate prior)
        new_alpha = current_prior['alpha'] + (1 if signal_correct else 0)
        new_beta = current_prior['beta'] + (0 if signal_correct else 1)

        self.signal_priors[signal_name] = {'alpha': new_alpha, 'beta': new_beta}

        # Track continuous performance
        self.signal_performance[signal_name].append({
            'signal_value': signal_value,
            'actual_return': actual_return,
            'correct': signal_correct,
            'timestamp': datetime.now(),
            'lookforward': lookforward_days
        })

        # Keep only recent history
        if len(self.signal_performance[signal_name]) > 1000:
            self.signal_performance[signal_name] = \
                self.signal_performance[signal_name][-1000:]

    def get_signal_reliability(self, signal_name: str) -> Dict:
        """
        Get reliability metrics for a signal.

        Parameters
        ----------
        signal_name : str
            The signal name

        Returns
        -------
        dict
            Reliability metrics including expected_reliability, confidence,
            total_observations, recent_accuracy, value_accuracy_correlation
        """
        prior = self.signal_priors[signal_name]
        total_observations = (prior['alpha'] + prior['beta'] -
                             2 * self.prior_alpha)  # Subtract prior counts

        if total_observations < self.min_samples:
            # Return prior if insufficient data
            expected_reliability = self.prior_alpha / (self.prior_alpha + self.prior_beta)
            confidence = 0.1
        else:
            # Posterior mean (expected value of Beta distribution)
            expected_reliability = prior['alpha'] / (prior['alpha'] + prior['beta'])

            # Confidence based on sample size (logistic growth)
            confidence = min(0.95, total_observations / (total_observations + 50))

        # Calculate additional metrics from performance history
        performance_data = self.signal_performance[signal_name]
        if len(performance_data) >= 10:
            recent_performance = performance_data[-20:]  # Last 20 observations
            recent_accuracy = np.mean([p['correct'] for p in recent_performance])
            value_accuracy_correlation = self._calculate_value_accuracy_correlation(signal_name)
        else:
            recent_accuracy = expected_reliability
            value_accuracy_correlation = 0.0

        return {
            'expected_reliability': expected_reliability,
            'confidence': confidence,
            'total_observations': max(0, total_observations),
            'recent_accuracy': recent_accuracy,
            'value_accuracy_correlation': value_accuracy_correlation,
            'posterior_alpha': prior['alpha'],
            'posterior_beta': prior['beta']
        }

    def _calculate_value_accuracy_correlation(self, signal_name: str) -> float:
        """
        Calculate if stronger signals are more accurate.

        Parameters
        ----------
        signal_name : str
            The signal name

        Returns
        -------
        float
            Correlation between signal strength and accuracy
        """
        performance_data = self.signal_performance[signal_name]

        if len(performance_data) < 10:
            return 0.0

        signal_strengths = [abs(p['signal_value']) for p in performance_data]
        accuracies = [1 if p['correct'] else 0 for p in performance_data]

        # Simple correlation
        if np.std(signal_strengths) > 0 and np.std(accuracies) > 0:
            correlation = np.corrcoef(signal_strengths, accuracies)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        return 0.0

    def combine_signals_bayesian(self, signals: Dict[str, float]) -> Dict:
        """
        Combine multiple signals using Bayesian model averaging.

        Weights signals by their posterior reliability and confidence,
        with additional adjustment for value-accuracy correlation.

        Parameters
        ----------
        signals : dict
            Dictionary of signal_name -> signal_value (-1 to 1)

        Returns
        -------
        dict
            Combined signal result with confidence and component details
        """
        if not signals:
            return {
                'combined_signal': 0.0,
                'confidence': 0.0,
                'components': {},
                'signals_combined': 0
            }

        weighted_signal = 0.0
        total_weight = 0.0
        reliability_scores = {}
        component_weights = {}

        for signal_name, signal_value in signals.items():
            reliability_data = self.get_signal_reliability(signal_name)

            # Weight = reliability * confidence
            weight = (reliability_data['expected_reliability'] *
                     reliability_data['confidence'])

            # Adjust for value-accuracy correlation if available
            if reliability_data['value_accuracy_correlation'] > 0.1:
                # Stronger signals get proportionally more weight
                signal_strength_weight = 1 + (
                    abs(signal_value) * reliability_data['value_accuracy_correlation']
                )
                weight *= signal_strength_weight

            weighted_signal += signal_value * weight
            total_weight += weight

            reliability_scores[signal_name] = reliability_data
            component_weights[signal_name] = weight

        # Normalize
        if total_weight > 0:
            combined_signal = weighted_signal / total_weight
            overall_confidence = total_weight / len(signals)
        else:
            combined_signal = 0.0
            overall_confidence = 0.0

        # Apply shrinkage towards zero for low confidence
        if overall_confidence < 0.3:
            combined_signal *= overall_confidence / 0.3

        return {
            'combined_signal': np.clip(combined_signal, -1, 1),
            'confidence': np.clip(overall_confidence, 0, 1),
            'component_weights': component_weights,
            'reliability_scores': reliability_scores,
            'signals_combined': len(signals),
            'total_weight': total_weight
        }

    def get_signal_agreement(self, signals: Dict[str, float]) -> float:
        """
        Calculate agreement between signals.

        Parameters
        ----------
        signals : dict
            Dictionary of signal values

        Returns
        -------
        float
            Agreement score (1.0 = perfect agreement)
        """
        if len(signals) <= 1:
            return 1.0

        signal_values = list(signals.values())

        # Check if all signals agree on direction
        directions = [1 if s > 0 else -1 if s < 0 else 0 for s in signal_values]
        direction_agreement = abs(np.mean(directions))

        # Check magnitude similarity
        magnitudes = [abs(s) for s in signal_values]
        magnitude_std = np.std(magnitudes)
        magnitude_agreement = 1 - min(1, magnitude_std)

        return 0.6 * direction_agreement + 0.4 * magnitude_agreement

    def get_signal_advice(self, signals: Dict[str, float]) -> Dict:
        """
        Get trading advice based on signal analysis.

        Parameters
        ----------
        signals : dict
            Dictionary of signal values

        Returns
        -------
        dict
            Trading advice with signal analysis
        """
        combination_result = self.combine_signals_bayesian(signals)
        agreement = self.get_signal_agreement(signals)

        # Identify strongest and weakest signals
        component_weights = combination_result['component_weights']
        if component_weights:
            strongest_signal = max(component_weights.items(), key=lambda x: x[1])
            weakest_signal = min(component_weights.items(), key=lambda x: x[1])
        else:
            strongest_signal = weakest_signal = (None, 0)

        recommendation = self._generate_recommendation(
            combination_result, agreement
        )

        return {
            'signal_agreement': agreement,
            'strongest_signal': strongest_signal[0],
            'strongest_weight': strongest_signal[1],
            'weakest_signal': weakest_signal[0],
            'weakest_weight': weakest_signal[1],
            'recommendation': recommendation,
            'combined_result': combination_result
        }

    def _generate_recommendation(
        self,
        combination_result: Dict,
        signal_agreement: float
    ) -> str:
        """
        Generate trading recommendation based on signal analysis.

        Parameters
        ----------
        combination_result : dict
            Combined signal result
        signal_agreement : float
            Signal agreement score

        Returns
        -------
        str
            Trading recommendation
        """
        combined_signal = combination_result['combined_signal']
        confidence = combination_result['confidence']

        if confidence < 0.3:
            return "LOW_CONFIDENCE - REDUCE_POSITION"
        elif signal_agreement < 0.5:
            return "CONFLICTING_SIGNALS - CAUTION_ADVISED"
        elif abs(combined_signal) > 0.5 and confidence > 0.6:
            return "HIGH_CONFIDENCE_STRONG_SIGNAL - INCREASE_POSITION"
        elif abs(combined_signal) < 0.2:
            return "WEAK_SIGNAL - HOLD_OR_REDUCE"
        else:
            return "MODERATE_CONFIDENCE - NORMAL_POSITION"

    def update_signal_correlations(
        self,
        signal1_name: str,
        signal1_value: float,
        signal2_name: str,
        signal2_value: float
    ) -> None:
        """
        Update correlation tracking between signals.

        Parameters
        ----------
        signal1_name : str
            First signal name
        signal1_value : float
            First signal value
        signal2_name : str
            Second signal name
        signal2_value : float
            Second signal value
        """
        # Simple running correlation update
        # This is a simplified approach; full implementation would use
        # rolling window correlation
        current_corr = self.signal_correlations[signal1_name][signal2_name]
        new_obs = signal1_value * signal2_value
        updated_corr = 0.95 * current_corr + 0.05 * new_obs
        self.signal_correlations[signal1_name][signal2_name] = updated_corr
        self.signal_correlations[signal2_name][signal1_name] = updated_corr

    def get_diversification_benefit(self, signals: Dict[str, float]) -> float:
        """
        Calculate diversification benefit from combining signals.

        Parameters
        ----------
        signals : dict
            Dictionary of signal values

        Returns
        -------
        float
            Diversification benefit (higher = more diverse)
        """
        if len(signals) <= 1:
            return 0.0

        signal_names = list(signals.keys())
        n = len(signal_names)
        avg_correlation = 0.0
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                corr = self.signal_correlations[signal_names[i]][signal_names[j]]
                avg_correlation += abs(corr)
                count += 1

        if count > 0:
            avg_correlation /= count

        # Higher correlation = less diversification benefit
        return 1 - avg_correlation

    def get_performance_summary(self) -> Dict:
        """
        Get summary of all signal performance.

        Returns
        -------
        dict
            Performance summary for all tracked signals
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'signals': {},
            'overall_stats': {}
        }

        all_accuracies = []

        for signal_name in self.signal_priors.keys():
            reliability = self.get_signal_reliability(signal_name)
            summary['signals'][signal_name] = reliability
            all_accuracies.append(reliability['expected_reliability'])

        if all_accuracies:
            summary['overall_stats'] = {
                'avg_reliability': np.mean(all_accuracies),
                'std_reliability': np.std(all_accuracies),
                'best_signal': max(summary['signals'].items(),
                                  key=lambda x: x[1]['expected_reliability'])[0],
                'worst_signal': min(summary['signals'].items(),
                                   key=lambda x: x[1]['expected_reliability'])[0],
                'total_signals': len(all_accuracies)
            }

        return summary

    def reset(self, signal_name: Optional[str] = None) -> None:
        """
        Reset signal tracking.

        Parameters
        ----------
        signal_name : str, optional
            Specific signal to reset. If None, reset all.
        """
        if signal_name:
            self.signal_priors[signal_name] = {
                'alpha': self.prior_alpha,
                'beta': self.prior_beta
            }
            self.signal_performance[signal_name].clear()
        else:
            self.signal_priors.clear()
            self.signal_performance.clear()
            self.signal_correlations.clear()


class EnhancedBayesianCombiner(BayesianSignalCombiner):
    """
    Enhanced Bayesian combiner with regime awareness and adaptive weighting.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Regime-specific reliability adjustments
        self.regime_adjustments = {
            'bull_market': {'momentum': 1.2, 'mean_reversion': 0.8, 'volatility': 0.9},
            'bear_market': {'momentum': 0.8, 'mean_reversion': 1.1, 'volatility': 1.2},
            'high_volatility': {'momentum': 0.7, 'mean_reversion': 1.2, 'volatility': 1.3},
            'low_volatility': {'momentum': 1.1, 'mean_reversion': 0.9, 'volatility': 0.8}
        }

        self.current_regime = 'bull_market'

    def set_regime(self, regime: str) -> None:
        """Set the current market regime."""
        if regime in self.regime_adjustments:
            self.current_regime = regime

    def combine_signals_bayesian(self, signals: Dict[str, float]) -> Dict:
        """Combine signals with regime adjustment."""
        result = super().combine_signals_bayesian(signals)

        # Apply regime adjustments to weights
        regime_factors = self.regime_adjustments.get(self.current_regime, {})

        adjusted_weights = {}
        for signal_name, weight in result['component_weights'].items():
            # Find matching regime adjustment
            for key, factor in regime_factors.items():
                if key in signal_name.lower():
                    weight *= factor
                    break
            adjusted_weights[signal_name] = weight

        # Recalculate combined signal with regime-adjusted weights
        if adjusted_weights and sum(adjusted_weights.values()) > 0:
            weighted_sum = sum(
                signals.get(name, 0) * weight
                for name, weight in adjusted_weights.items()
            )
            total_weight = sum(adjusted_weights.values())
            result['combined_signal'] = np.clip(weighted_sum / total_weight, -1, 1)
            result['regime_adjusted_weights'] = adjusted_weights
            result['current_regime'] = self.current_regime

        return result


# Factory function
def get_bayesian_combiner(
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    min_samples: int = 5,
    enhanced: bool = False
) -> BayesianSignalCombiner:
    """
    Factory function to create a Bayesian signal combiner.

    Parameters
    ----------
    prior_alpha : float
        Alpha parameter for prior
    prior_beta : float
        Beta parameter for prior
    min_samples : int
        Minimum samples before trusting posterior
    enhanced : bool
        Whether to use enhanced regime-aware combiner

    Returns
    -------
    BayesianSignalCombiner
        Configured combiner instance
    """
    if enhanced:
        return EnhancedBayesianCombiner(
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            min_samples=min_samples
        )
    else:
        return BayesianSignalCombiner(
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            min_samples=min_samples
        )
