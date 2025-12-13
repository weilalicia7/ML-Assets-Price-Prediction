"""
Volatility Regime Detection and Model Switching
Detects market regimes (low, medium, high volatility) and switches models accordingly.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from sklearn.mixture import GaussianMixture
import joblib


class RegimeDetector:
    """
    Detects volatility regimes using multiple methods.

    Methods:
    1. Percentile-based: Simple thresholds
    2. Gaussian Mixture Model: Statistical clustering
    3. Adaptive: Rolling window regime detection
    """

    def __init__(self, n_regimes: int = 3, method: str = 'gmm'):
        """
        Initialize regime detector.

        Args:
            n_regimes: Number of regimes (default: 3 for low/medium/high)
            method: Detection method ('percentile', 'gmm', 'adaptive')
        """
        self.n_regimes = n_regimes
        self.method = method
        self.gmm_model = None
        self.thresholds = None

    def detect_regime_percentile(
        self,
        volatility: np.ndarray,
        low_threshold: float = 0.33,
        high_threshold: float = 0.67
    ) -> np.ndarray:
        """
        Detect regimes using percentile thresholds.

        Args:
            volatility: Array of volatility values
            low_threshold: Percentile for low regime boundary
            high_threshold: Percentile for high regime boundary

        Returns:
            Array of regime labels (0=low, 1=medium, 2=high)
        """
        percentile_rank = pd.Series(volatility).rank(pct=True)

        regimes = np.zeros(len(volatility), dtype=int)
        regimes[percentile_rank <= low_threshold] = 0  # Low
        regimes[(percentile_rank > low_threshold) & (percentile_rank <= high_threshold)] = 1  # Medium
        regimes[percentile_rank > high_threshold] = 2  # High

        # Store thresholds
        self.thresholds = {
            'low': np.percentile(volatility, low_threshold * 100),
            'high': np.percentile(volatility, high_threshold * 100)
        }

        return regimes

    def detect_regime_gmm(self, volatility: np.ndarray) -> np.ndarray:
        """
        Detect regimes using Gaussian Mixture Model.

        Args:
            volatility: Array of volatility values

        Returns:
            Array of regime labels (0=low, 1=medium, 2=high)
        """
        # Reshape for GMM
        X = volatility.reshape(-1, 1)

        # Fit Gaussian Mixture Model
        self.gmm_model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=42,
            max_iter=200
        )

        regimes = self.gmm_model.fit_predict(X)

        # Sort regimes by mean volatility (0=low, 1=medium, 2=high)
        means = []
        for i in range(self.n_regimes):
            means.append(volatility[regimes == i].mean())

        # Create mapping from cluster ID to regime
        sorted_indices = np.argsort(means)
        regime_mapping = {sorted_indices[i]: i for i in range(self.n_regimes)}

        # Remap regimes
        remapped_regimes = np.array([regime_mapping[r] for r in regimes])

        return remapped_regimes

    def detect_regime_adaptive(
        self,
        volatility: np.ndarray,
        window: int = 60
    ) -> np.ndarray:
        """
        Detect regimes using adaptive rolling window.

        Args:
            volatility: Array of volatility values
            window: Rolling window size

        Returns:
            Array of regime labels
        """
        regimes = np.zeros(len(volatility), dtype=int)

        for i in range(window, len(volatility)):
            window_vol = volatility[i-window:i]

            # Calculate percentiles for this window
            low_thresh = np.percentile(window_vol, 33)
            high_thresh = np.percentile(window_vol, 67)

            current_vol = volatility[i]

            if current_vol <= low_thresh:
                regimes[i] = 0  # Low
            elif current_vol <= high_thresh:
                regimes[i] = 1  # Medium
            else:
                regimes[i] = 2  # High

        # Fill initial window with median regime
        regimes[:window] = 1

        return regimes

    def detect_regime(
        self,
        volatility: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Detect regimes using configured method.

        Args:
            volatility: Array of volatility values
            dates: Optional datetime index

        Returns:
            Tuple of (regimes, regime_info)
        """
        if self.method == 'percentile':
            regimes = self.detect_regime_percentile(volatility)
        elif self.method == 'gmm':
            regimes = self.detect_regime_gmm(volatility)
        elif self.method == 'adaptive':
            regimes = self.detect_regime_adaptive(volatility)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Calculate regime statistics
        regime_info = self._calculate_regime_stats(volatility, regimes, dates)

        return regimes, regime_info

    def _calculate_regime_stats(
        self,
        volatility: np.ndarray,
        regimes: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> Dict:
        """Calculate statistics for each regime."""
        stats = {}
        regime_names = ['Low', 'Medium', 'High']

        for i in range(self.n_regimes):
            mask = regimes == i
            regime_vol = volatility[mask]

            stats[regime_names[i]] = {
                'count': mask.sum(),
                'percentage': mask.sum() / len(regimes) * 100,
                'mean_volatility': regime_vol.mean() if len(regime_vol) > 0 else 0,
                'std_volatility': regime_vol.std() if len(regime_vol) > 0 else 0,
                'min_volatility': regime_vol.min() if len(regime_vol) > 0 else 0,
                'max_volatility': regime_vol.max() if len(regime_vol) > 0 else 0
            }

            if dates is not None and len(regime_vol) > 0:
                regime_dates = dates[mask]
                stats[regime_names[i]]['first_occurrence'] = regime_dates.min()
                stats[regime_names[i]]['last_occurrence'] = regime_dates.max()

        return stats

    def predict_regime(self, volatility: float) -> int:
        """
        Predict regime for a single volatility value.

        Args:
            volatility: Single volatility value

        Returns:
            Regime label (0, 1, or 2)
        """
        if self.method == 'percentile' and self.thresholds is not None:
            if volatility <= self.thresholds['low']:
                return 0
            elif volatility <= self.thresholds['high']:
                return 1
            else:
                return 2

        elif self.method == 'gmm' and self.gmm_model is not None:
            regime = self.gmm_model.predict([[volatility]])[0]

            # Apply same mapping as in training
            return regime

        else:
            # Default to medium regime if not trained
            return 1

    def get_regime_transitions(self, regimes: np.ndarray) -> Dict:
        """
        Analyze regime transitions.

        Args:
            regimes: Array of regime labels

        Returns:
            Dictionary with transition statistics
        """
        transitions = {
            'total_transitions': 0,
            'transition_matrix': np.zeros((self.n_regimes, self.n_regimes)),
            'average_regime_duration': {}
        }

        # Count transitions
        for i in range(len(regimes) - 1):
            if regimes[i] != regimes[i+1]:
                transitions['total_transitions'] += 1
                transitions['transition_matrix'][regimes[i], regimes[i+1]] += 1

        # Calculate average duration for each regime
        regime_names = ['Low', 'Medium', 'High']
        for regime_id in range(self.n_regimes):
            durations = []
            current_duration = 0

            for r in regimes:
                if r == regime_id:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                        current_duration = 0

            if current_duration > 0:
                durations.append(current_duration)

            transitions['average_regime_duration'][regime_names[regime_id]] = \
                np.mean(durations) if durations else 0

        return transitions

    def save(self, path: str):
        """Save regime detector to disk."""
        detector_data = {
            'n_regimes': self.n_regimes,
            'method': self.method,
            'gmm_model': self.gmm_model,
            'thresholds': self.thresholds
        }
        joblib.dump(detector_data, path)
        print(f"[OK] Regime detector saved to {path}")

    def load(self, path: str):
        """Load regime detector from disk."""
        detector_data = joblib.load(path)
        self.n_regimes = detector_data['n_regimes']
        self.method = detector_data['method']
        self.gmm_model = detector_data['gmm_model']
        self.thresholds = detector_data['thresholds']
        print(f"[OK] Regime detector loaded from {path}")


class RegimeSwitchingModel:
    """
    Model that switches between different models based on detected regime.

    Trains separate models for each volatility regime.
    """

    def __init__(self, detector: RegimeDetector):
        """
        Initialize regime-switching model.

        Args:
            detector: Trained RegimeDetector instance
        """
        self.detector = detector
        self.regime_models = {}  # Dictionary of models per regime
        self.regime_names = ['Low', 'Medium', 'High']

    def train_regime_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        volatility_train: np.ndarray,
        volatility_val: np.ndarray,
        model_type: str = 'lightgbm'
    ):
        """
        Train separate models for each regime.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            volatility_train: Historical volatility for regime detection (train)
            volatility_val: Historical volatility for regime detection (val)
            model_type: Type of model to train
        """
        from .base_models import VolatilityPredictor

        print("\n" + "="*60)
        print("TRAINING REGIME-SPECIFIC MODELS")
        print("="*60)

        # Detect regimes
        regimes_train, _ = self.detector.detect_regime(volatility_train)
        regimes_val, regime_info = self.detector.detect_regime(volatility_val)

        print(f"\nRegime Distribution (Training):")
        for regime_id, regime_name in enumerate(self.regime_names):
            count = (regimes_train == regime_id).sum()
            pct = count / len(regimes_train) * 100
            print(f"  {regime_name}: {count} samples ({pct:.1f}%)")

        # Train model for each regime
        for regime_id, regime_name in enumerate(self.regime_names):
            print(f"\n[INFO] Training model for {regime_name} volatility regime...")

            # Filter data for this regime
            train_mask = regimes_train == regime_id
            val_mask = regimes_val == regime_id

            if train_mask.sum() < 50:
                print(f"[WARNING] Not enough training data for {regime_name} regime ({train_mask.sum()} samples)")
                print(f"           Using global model instead")
                continue

            X_train_regime = X_train[train_mask]
            y_train_regime = y_train[train_mask]

            if val_mask.sum() > 0:
                X_val_regime = X_val[val_mask]
                y_val_regime = y_val[val_mask]
            else:
                # Use small portion of training data for validation
                split_idx = int(len(X_train_regime) * 0.8)
                X_val_regime = X_train_regime[split_idx:]
                y_val_regime = y_train_regime[split_idx:]
                X_train_regime = X_train_regime[:split_idx]
                y_train_regime = y_train_regime[:split_idx]

            # Train model
            model = VolatilityPredictor(model_type=model_type, random_state=42)

            if model_type == 'lightgbm':
                model.train_lightgbm(X_train_regime, y_train_regime, X_val_regime, y_val_regime)
            elif model_type == 'xgboost':
                model.train_xgboost(X_train_regime, y_train_regime, X_val_regime, y_val_regime)

            # Evaluate
            val_metrics = model.evaluate(X_val_regime, y_val_regime)
            print(f"[OK] {regime_name} regime model performance:")
            print(f"     MAE:  {val_metrics['mae']:.6f}")
            print(f"     RMSE: {val_metrics['rmse']:.6f}")
            print(f"     R²:   {val_metrics['r2']:.4f}")

            self.regime_models[regime_id] = model

        print("\n" + "="*60)
        print(f"REGIME-SPECIFIC MODELS TRAINED: {len(self.regime_models)}/{len(self.regime_names)}")
        print("="*60)

    def predict(
        self,
        X: pd.DataFrame,
        current_volatility: np.ndarray,
        fallback_model = None
    ) -> np.ndarray:
        """
        Make predictions using regime-appropriate models.

        Args:
            X: Features
            current_volatility: Current volatility values for regime detection
            fallback_model: Model to use if regime-specific model not available

        Returns:
            Array of predictions
        """
        predictions = np.zeros(len(X))

        # Detect current regimes
        regimes = np.array([self.detector.predict_regime(vol) for vol in current_volatility])

        # Predict for each regime
        for regime_id in range(self.detector.n_regimes):
            mask = regimes == regime_id

            if mask.sum() == 0:
                continue

            X_regime = X[mask]

            # Use regime-specific model if available
            if regime_id in self.regime_models:
                predictions[mask] = self.regime_models[regime_id].predict(X_regime)
            elif fallback_model is not None:
                predictions[mask] = fallback_model.predict(X_regime)
            else:
                raise ValueError(f"No model available for regime {regime_id}")

        return predictions

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        current_volatility: np.ndarray,
        fallback_model = None
    ) -> Dict:
        """
        Evaluate regime-switching model performance.

        Args:
            X: Features
            y: True targets
            current_volatility: Current volatility for regime detection
            fallback_model: Fallback model

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        predictions = self.predict(X, current_volatility, fallback_model)

        metrics = {
            'overall': {
                'mae': mean_absolute_error(y, predictions),
                'rmse': np.sqrt(mean_squared_error(y, predictions)),
                'r2': r2_score(y, predictions)
            },
            'by_regime': {}
        }

        # Calculate metrics per regime
        regimes = np.array([self.detector.predict_regime(vol) for vol in current_volatility])

        for regime_id, regime_name in enumerate(self.regime_names):
            mask = regimes == regime_id

            if mask.sum() == 0:
                continue

            metrics['by_regime'][regime_name] = {
                'count': mask.sum(),
                'mae': mean_absolute_error(y[mask], predictions[mask]),
                'rmse': np.sqrt(mean_squared_error(y[mask], predictions[mask])),
                'r2': r2_score(y[mask], predictions[mask])
            }

        return metrics


def main():
    """Example usage of regime detection."""
    import sys
    sys.path.insert(0, '.')

    from src.data.fetch_data import DataFetcher
    from src.features.technical_features import TechnicalFeatureEngineer
    from src.features.volatility_features import VolatilityFeatureEngineer
    from src.models.base_models import VolatilityPredictor

    print("="*60)
    print("REGIME DETECTION - EXAMPLE")
    print("="*60)

    # Get data
    print("\n[INFO] Fetching data...")
    fetcher = DataFetcher(['AAPL', 'BTC-USD'], start_date='2022-01-01')
    data = fetcher.fetch_all()

    # Process each ticker
    all_processed = []
    for ticker in data['Ticker'].unique():
        ticker_data = data[data['Ticker'] == ticker].copy()

        tech_eng = TechnicalFeatureEngineer()
        ticker_data = tech_eng.add_all_features(ticker_data)

        vol_eng = VolatilityFeatureEngineer()
        ticker_data = vol_eng.add_all_features(ticker_data)

        all_processed.append(ticker_data)

    processed = pd.concat(all_processed)

    # Create target
    predictor = VolatilityPredictor()
    processed = predictor.create_target(processed)

    # Test regime detection
    print("\n[INFO] Testing regime detection methods...")

    volatility = processed['hist_vol_20'].values

    for method in ['percentile', 'gmm', 'adaptive']:
        print(f"\n{'='*60}")
        print(f"Method: {method.upper()}")
        print(f"{'='*60}")

        detector = RegimeDetector(n_regimes=3, method=method)
        regimes, regime_info = detector.detect_regime(volatility, processed.index)

        print(f"\nRegime Statistics:")
        for regime_name, stats in regime_info.items():
            print(f"\n{regime_name} Volatility:")
            print(f"  Count:       {stats['count']}")
            print(f"  Percentage:  {stats['percentage']:.2f}%")
            print(f"  Mean Vol:    {stats['mean_volatility']:.6f}")
            print(f"  Std Vol:     {stats['std_volatility']:.6f}")

        # Analyze transitions
        transitions = detector.get_regime_transitions(regimes)
        print(f"\nRegime Transitions:")
        print(f"  Total transitions: {transitions['total_transitions']}")
        print(f"  Average regime duration:")
        for regime_name, duration in transitions['average_regime_duration'].items():
            print(f"    {regime_name}: {duration:.1f} days")

    print("\n[SUCCESS] Regime detection complete!")


if __name__ == "__main__":
    main()
