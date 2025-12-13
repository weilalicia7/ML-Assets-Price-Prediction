"""
Unified Regime Detector - Phase 2 Conflict Resolution

Combines Phase 1 GMM-based regime detection with Phase 2 transition detection.

Phase 1 contribution:
- VolatilityRegimeDetector (GMM/HMM)
- 4 volatility regimes: Low, Medium, High, Crisis

Phase 2 contribution:
- RegimeTransitionDetector
- Early warning for regime changes
- Smooth transition probabilities

Based on: phase2 fixing on C model_conflict resolutions.pdf
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from sklearn.mixture import GaussianMixture
from collections import deque
import logging

logger = logging.getLogger(__name__)


class UnifiedRegimeDetector:
    """
    Unified regime detection combining GMM clustering with transition detection.

    Features:
    - GMM-based regime classification (Phase 1)
    - Transition probability tracking (Phase 2)
    - Early warning signals
    - Regime stability scoring
    - Smooth regime transitions
    """

    def __init__(
        self,
        n_regimes: int = 4,
        transition_window: int = 10,
        stability_window: int = 20,
        transition_threshold: float = 0.3,
        random_state: int = 42
    ):
        """
        Initialize unified regime detector.

        Args:
            n_regimes: Number of volatility regimes (4: Low, Medium, High, Crisis)
            transition_window: Window for transition probability calculation
            stability_window: Window for stability scoring
            transition_threshold: Threshold for transition warning
            random_state: Random seed
        """
        self.n_regimes = n_regimes
        self.transition_window = transition_window
        self.stability_window = stability_window
        self.transition_threshold = transition_threshold
        self.random_state = random_state

        # GMM model (Phase 1)
        self.gmm_model: Optional[GaussianMixture] = None
        self.regime_stats: Dict = {}
        self.regime_mapping: Dict = {}

        # Transition tracking (Phase 2)
        self.regime_history: deque = deque(maxlen=stability_window)
        self.volatility_history: deque = deque(maxlen=stability_window)

        # Regime names
        self.regime_names = {
            0: 'Low Volatility',
            1: 'Medium Volatility',
            2: 'High Volatility',
            3: 'Crisis'
        }

        # Trading recommendations per regime
        self.regime_recommendations = {
            0: {'trade': True, 'position_mult': 1.2, 'strategy': 'mean_reversion'},
            1: {'trade': True, 'position_mult': 1.0, 'strategy': 'momentum'},
            2: {'trade': True, 'position_mult': 0.5, 'strategy': 'defensive'},
            3: {'trade': False, 'position_mult': 0.0, 'strategy': 'cash'}
        }

        logger.info(f"Initialized UnifiedRegimeDetector with {n_regimes} regimes")

    def fit(self, data: pd.DataFrame, vol_col: str = 'volatility') -> 'UnifiedRegimeDetector':
        """
        Fit GMM model on historical volatility data.

        Args:
            data: DataFrame with volatility data
            vol_col: Name of volatility column

        Returns:
            self
        """
        # Calculate volatility if not present
        if vol_col not in data.columns:
            if 'High' in data.columns and 'Low' in data.columns:
                data = data.copy()
                data['volatility'] = (data['High'] - data['Low']) / data['Close']
            else:
                data = data.copy()
                data['volatility'] = data['Close'].pct_change().rolling(20).std()

        vol_data = data[vol_col].dropna().values.reshape(-1, 1)

        # Fit GMM (Phase 1 approach)
        self.gmm_model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=self.random_state,
            max_iter=200
        )
        self.gmm_model.fit(vol_data)

        # Get initial regime predictions
        regimes = self.gmm_model.predict(vol_data)

        # Calculate regime statistics
        raw_stats = {}
        for i in range(self.n_regimes):
            regime_vols = vol_data[regimes == i]
            if len(regime_vols) > 0:
                raw_stats[i] = {
                    'mean_vol': float(regime_vols.mean()),
                    'std_vol': float(regime_vols.std()),
                    'count': len(regime_vols),
                    'pct': len(regime_vols) / len(vol_data) * 100
                }

        # Sort regimes by volatility (0=lowest, 3=highest)
        sorted_regimes = sorted(raw_stats.items(), key=lambda x: x[1]['mean_vol'])
        self.regime_mapping = {old: new for new, (old, _) in enumerate(sorted_regimes)}

        # Recalculate stats with sorted labels
        new_regimes = np.array([self.regime_mapping[r] for r in regimes])
        self.regime_stats = {}
        for i in range(self.n_regimes):
            regime_vols = vol_data[new_regimes == i]
            if len(regime_vols) > 0:
                self.regime_stats[i] = {
                    'mean_vol': float(regime_vols.mean()),
                    'std_vol': float(regime_vols.std()),
                    'min_vol': float(regime_vols.min()),
                    'max_vol': float(regime_vols.max()),
                    'count': len(regime_vols),
                    'pct': len(regime_vols) / len(vol_data) * 100
                }

        logger.info("Regime detector fitted:")
        for i in range(self.n_regimes):
            if i in self.regime_stats:
                stats = self.regime_stats[i]
                logger.info(f"  {self.regime_names[i]}: {stats['pct']:.1f}%, mean_vol={stats['mean_vol']:.4f}")

        return self

    def predict_regime(self, volatility: float) -> int:
        """
        Predict regime for a single volatility value.

        Args:
            volatility: Current volatility

        Returns:
            Regime label (0-3)
        """
        if self.gmm_model is None:
            logger.warning("Model not fitted, returning default regime 1")
            return 1

        raw_regime = self.gmm_model.predict([[volatility]])[0]
        return self.regime_mapping.get(raw_regime, 1)

    def update(self, volatility: float) -> Dict:
        """
        Update regime state with new volatility observation.

        Combines Phase 1 GMM prediction with Phase 2 transition detection.

        Args:
            volatility: Current volatility

        Returns:
            Dict with regime state and transition info
        """
        # Store in history
        self.volatility_history.append(volatility)

        # Predict current regime (Phase 1)
        current_regime = self.predict_regime(volatility)
        self.regime_history.append(current_regime)

        # Calculate transition probability (Phase 2)
        transition_prob = self._calculate_transition_probability()

        # Calculate stability score
        stability = self._calculate_stability()

        # Get regime probabilities from GMM
        regime_probs = self._get_regime_probabilities(volatility)

        # Determine if transition warning should be issued
        transition_warning = transition_prob > self.transition_threshold

        # Get trading recommendation
        recommendation = self.regime_recommendations.get(
            current_regime,
            {'trade': True, 'position_mult': 1.0, 'strategy': 'default'}
        )

        return {
            'regime': current_regime,
            'regime_name': self.regime_names.get(current_regime, 'Unknown'),
            'regime_probabilities': regime_probs,
            'transition_probability': transition_prob,
            'transition_warning': transition_warning,
            'stability_score': stability,
            'recommendation': recommendation,
            'should_trade': recommendation['trade'],
            'position_multiplier': recommendation['position_mult'],
            'suggested_strategy': recommendation['strategy']
        }

    def _calculate_transition_probability(self) -> float:
        """
        Calculate probability of regime transition (Phase 2).

        Uses recent regime changes to estimate transition likelihood.
        """
        if len(self.regime_history) < 3:
            return 0.0

        history = list(self.regime_history)

        # Count regime changes in recent history
        changes = sum(1 for i in range(1, len(history)) if history[i] != history[i-1])
        transition_prob = changes / (len(history) - 1)

        return transition_prob

    def _calculate_stability(self) -> float:
        """
        Calculate regime stability score.

        Higher score = more stable regime.
        """
        if len(self.regime_history) < 3:
            return 1.0

        history = list(self.regime_history)

        # Calculate duration in current regime
        current = history[-1]
        duration = 0
        for r in reversed(history):
            if r == current:
                duration += 1
            else:
                break

        # Stability = duration / window (capped at 1.0)
        stability = min(1.0, duration / self.stability_window)

        return stability

    def _get_regime_probabilities(self, volatility: float) -> Dict[int, float]:
        """
        Get probability for each regime from GMM.

        Args:
            volatility: Current volatility

        Returns:
            Dict mapping regime to probability
        """
        if self.gmm_model is None:
            return {i: 0.25 for i in range(self.n_regimes)}

        probs = self.gmm_model.predict_proba([[volatility]])[0]

        # Map to sorted regime labels
        mapped_probs = {}
        for old_regime, prob in enumerate(probs):
            new_regime = self.regime_mapping.get(old_regime, old_regime)
            mapped_probs[new_regime] = float(prob)

        return mapped_probs

    def get_trading_multiplier(self, volatility: float) -> float:
        """
        Get position size multiplier based on current regime.

        Args:
            volatility: Current volatility

        Returns:
            Position multiplier (0.0 to 1.2)
        """
        state = self.update(volatility)
        return state['position_multiplier']

    def get_regime_features(self, data: pd.DataFrame, vol_col: str = 'volatility') -> pd.DataFrame:
        """
        Generate regime features for the entire dataset.

        Args:
            data: DataFrame with volatility data
            vol_col: Volatility column name

        Returns:
            DataFrame with regime features added
        """
        data = data.copy()

        # Calculate volatility if needed
        if vol_col not in data.columns:
            data['volatility'] = data['Close'].pct_change().rolling(20).std()

        # Predict regimes for all data
        vol_data = data[vol_col].fillna(method='ffill').values.reshape(-1, 1)
        raw_regimes = self.gmm_model.predict(vol_data)
        regimes = np.array([self.regime_mapping.get(r, 1) for r in raw_regimes])

        data['regime'] = regimes
        data['regime_name'] = [self.regime_names.get(r, 'Unknown') for r in regimes]

        # Calculate rolling features
        # Regime duration
        durations = []
        current_regime = regimes[0]
        duration = 0
        for r in regimes:
            if r == current_regime:
                duration += 1
            else:
                current_regime = r
                duration = 1
            durations.append(duration)
        data['regime_duration'] = durations

        # Regime change indicator
        data['regime_changed'] = (np.diff(regimes, prepend=regimes[0]) != 0).astype(int)

        # Rolling transition probability
        data['transition_prob'] = data['regime_changed'].rolling(
            window=self.transition_window, min_periods=1
        ).mean()

        # Stability score (inverse of transition prob)
        data['regime_stability'] = 1 - data['transition_prob']

        # One-hot encoding of regimes
        for i in range(self.n_regimes):
            data[f'regime_{i}'] = (regimes == i).astype(int)

        logger.info(f"Added {5 + self.n_regimes} regime features")

        return data

    def reset(self):
        """Reset tracking history."""
        self.regime_history.clear()
        self.volatility_history.clear()
        logger.info("Regime detector history reset")


# Convenience function
def get_unified_regime_detector(**kwargs) -> UnifiedRegimeDetector:
    """Get a configured UnifiedRegimeDetector instance."""
    return UnifiedRegimeDetector(**kwargs)
