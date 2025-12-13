"""
Probabilistic Forecasting Models
Implements quantile regression and conformal prediction for uncertainty quantification

Based on professional quant firm standards:
- Quantile regression for distributional forecasts
- Conformal prediction for distribution-free uncertainty
- Calibrated prediction intervals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from sklearn.ensemble import GradientBoostingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class QuantileRegressor:
    """
    Quantile Regression for Probabilistic Forecasts

    Predicts multiple quantiles to estimate full distribution:
    - Lower quantile (e.g., 10th percentile) for downside risk
    - Median (50th percentile) for central forecast
    - Upper quantile (e.g., 90th percentile) for upside potential

    Advantages:
    - Distribution-free (no normality assumption)
    - Asymmetric uncertainty quantification
    - Robust to outliers
    """

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        model_type: str = 'lightgbm',
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        random_state: int = 42
    ):
        """
        Args:
            quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9])
            model_type: 'lightgbm' or 'sklearn'
            n_estimators: Number of boosting iterations
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            random_state: Random seed
        """
        self.quantiles = sorted(quantiles)
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        self.models = {}  # One model per quantile

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train quantile regression models.

        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        print(f"\n[QuantileRegressor] Training models for quantiles: {self.quantiles}")

        for quantile in self.quantiles:
            print(f"\n  Training quantile {quantile:.2f}...")

            if self.model_type == 'lightgbm':
                if not LIGHTGBM_AVAILABLE:
                    raise ImportError("LightGBM not available")

                params = {
                    'objective': 'quantile',
                    'alpha': quantile,
                    'metric': 'quantile',
                    'num_leaves': 2 ** self.max_depth - 1,
                    'learning_rate': self.learning_rate,
                    'verbose': -1,
                    'random_state': self.random_state
                }

                train_data = lgb.Dataset(X, label=y)

                if X_val is not None and y_val is not None:
                    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                    model = lgb.train(
                        params,
                        train_data,
                        num_boost_round=self.n_estimators,
                        valid_sets=[val_data],
                        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
                    )
                else:
                    model = lgb.train(
                        params,
                        train_data,
                        num_boost_round=self.n_estimators
                    )

                self.models[quantile] = model

            elif self.model_type == 'sklearn':
                if not SKLEARN_AVAILABLE:
                    raise ImportError("scikit-learn not available")

                model = GradientBoostingRegressor(
                    loss='quantile',
                    alpha=quantile,
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    random_state=self.random_state
                )

                model.fit(X, y)
                self.models[quantile] = model

            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

            print(f"    [OK] Quantile {quantile:.2f} trained")

        print(f"\n[OK] All {len(self.quantiles)} quantile models trained")

    def predict(self, X: np.ndarray, quantile: Optional[float] = None) -> np.ndarray:
        """
        Predict specific quantile or median.

        Args:
            X: Features
            quantile: Quantile to predict (if None, uses median)

        Returns:
            Predictions for specified quantile
        """
        if quantile is None:
            # Use median
            quantile = 0.5 if 0.5 in self.quantiles else self.quantiles[len(self.quantiles)//2]

        if quantile not in self.models:
            raise ValueError(f"Quantile {quantile} not trained. Available: {list(self.models.keys())}")

        model = self.models[quantile]

        if self.model_type == 'lightgbm':
            return model.predict(X)
        else:
            return model.predict(X)

    def predict_interval(self, X: np.ndarray, lower_q: float = 0.1, upper_q: float = 0.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict prediction interval.

        Args:
            X: Features
            lower_q: Lower quantile for interval
            upper_q: Upper quantile for interval

        Returns:
            (lower_bound, median, upper_bound)
        """
        if lower_q not in self.models:
            raise ValueError(f"Lower quantile {lower_q} not trained")
        if upper_q not in self.models:
            raise ValueError(f"Upper quantile {upper_q} not trained")

        lower = self.predict(X, quantile=lower_q)
        upper = self.predict(X, quantile=upper_q)
        median = self.predict(X, quantile=0.5)

        return lower, median, upper

    def predict_all_quantiles(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """
        Predict all quantiles.

        Args:
            X: Features

        Returns:
            Dictionary mapping quantile to predictions
        """
        predictions = {}
        for quantile in self.quantiles:
            predictions[quantile] = self.predict(X, quantile=quantile)

        return predictions

    def coverage_score(self, X: np.ndarray, y_true: np.ndarray, lower_q: float = 0.1, upper_q: float = 0.9) -> float:
        """
        Calculate empirical coverage of prediction interval.

        Args:
            X: Features
            y_true: True values
            lower_q: Lower quantile
            upper_q: Upper quantile

        Returns:
            Coverage percentage (should be close to upper_q - lower_q)
        """
        lower, _, upper = self.predict_interval(X, lower_q, upper_q)

        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        return coverage

    def interval_width(self, X: np.ndarray, lower_q: float = 0.1, upper_q: float = 0.9) -> np.ndarray:
        """
        Calculate prediction interval width (uncertainty measure).

        Args:
            X: Features
            lower_q: Lower quantile
            upper_q: Upper quantile

        Returns:
            Interval widths
        """
        lower, _, upper = self.predict_interval(X, lower_q, upper_q)
        return upper - lower


class ConformalPredictor:
    """
    Conformal Prediction for Distribution-Free Uncertainty Quantification

    Key advantages:
    - Valid prediction intervals with guaranteed coverage
    - No distributional assumptions
    - Works with any base model
    - Adaptive to heteroscedastic noise

    Algorithm:
    1. Train base model on training data
    2. Compute nonconformity scores on calibration set
    3. Use scores to construct prediction intervals with desired coverage
    """

    def __init__(
        self,
        base_model,
        confidence_level: float = 0.9,
        method: str = 'absolute'
    ):
        """
        Args:
            base_model: Any model with fit() and predict() methods
            confidence_level: Desired coverage probability (e.g., 0.9 for 90%)
            method: 'absolute' or 'normalized' nonconformity scores
        """
        self.base_model = base_model
        self.confidence_level = confidence_level
        self.method = method

        self.nonconformity_scores = None
        self.quantile_score = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Train conformal predictor.

        Args:
            X_train: Training features
            y_train: Training targets
            X_cal: Calibration features
            y_cal: Calibration targets
        """
        print(f"\n[ConformalPredictor] Training with {self.confidence_level:.0%} confidence")

        # Train base model
        print("  Training base model...")
        self.base_model.fit(X_train, y_train)

        # Compute nonconformity scores on calibration set
        print("  Computing nonconformity scores...")
        cal_predictions = self.base_model.predict(X_cal)

        if self.method == 'absolute':
            # Absolute residuals
            self.nonconformity_scores = np.abs(y_cal - cal_predictions)
        elif self.method == 'normalized':
            # Normalized by prediction magnitude
            self.nonconformity_scores = np.abs(y_cal - cal_predictions) / (np.abs(cal_predictions) + 1e-6)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Compute quantile for desired coverage
        # Add 1 to numerator for finite-sample guarantee
        n_cal = len(self.nonconformity_scores)
        quantile_level = np.ceil((n_cal + 1) * self.confidence_level) / n_cal
        quantile_level = min(quantile_level, 1.0)  # Clip to [0, 1]

        self.quantile_score = np.quantile(self.nonconformity_scores, quantile_level)

        print(f"  Nonconformity quantile ({quantile_level:.3f}): {self.quantile_score:.6f}")
        print(f"[OK] Conformal predictor trained")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point prediction (base model)."""
        return self.base_model.predict(X)

    def predict_interval(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with conformal intervals.

        Returns:
            (lower_bound, prediction, upper_bound)
        """
        if self.quantile_score is None:
            raise ValueError("Model not calibrated. Call fit() first.")

        predictions = self.base_model.predict(X)

        if self.method == 'absolute':
            lower = predictions - self.quantile_score
            upper = predictions + self.quantile_score
        elif self.method == 'normalized':
            interval_width = self.quantile_score * (np.abs(predictions) + 1e-6)
            lower = predictions - interval_width
            upper = predictions + interval_width
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return lower, predictions, upper

    def coverage_score(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate empirical coverage.

        Should be approximately equal to confidence_level.
        """
        lower, _, upper = self.predict_interval(X)
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        return coverage

    def interval_width(self, X: np.ndarray) -> np.ndarray:
        """Calculate prediction interval width."""
        lower, _, upper = self.predict_interval(X)
        return upper - lower


def compare_uncertainty_methods(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Compare different uncertainty quantification methods.

    Returns:
        DataFrame with comparison metrics
    """
    results = []

    # 1. Quantile Regression
    print("\n" + "="*60)
    print("QUANTILE REGRESSION")
    print("="*60)

    qr = QuantileRegressor(quantiles=[0.1, 0.5, 0.9])
    qr.fit(X_train, y_train, X_cal, y_cal)

    qr_coverage = qr.coverage_score(X_test, y_test, 0.1, 0.9)
    qr_width = np.mean(qr.interval_width(X_test, 0.1, 0.9))

    results.append({
        'method': 'Quantile Regression',
        'coverage': qr_coverage,
        'avg_interval_width': qr_width,
        'target_coverage': 0.8
    })

    print(f"  Coverage: {qr_coverage:.1%} (target: 80%)")
    print(f"  Avg Interval Width: {qr_width:.6f}")

    # 2. Conformal Prediction (Absolute)
    print("\n" + "="*60)
    print("CONFORMAL PREDICTION (Absolute)")
    print("="*60)

    try:
        import lightgbm as lgb
        base_model_abs = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    except:
        from sklearn.ensemble import RandomForestRegressor
        base_model_abs = RandomForestRegressor(n_estimators=100, random_state=42)

    cp_abs = ConformalPredictor(base_model_abs, confidence_level=0.8, method='absolute')
    cp_abs.fit(X_train, y_train, X_cal, y_cal)

    cp_abs_coverage = cp_abs.coverage_score(X_test, y_test)
    cp_abs_width = np.mean(cp_abs.interval_width(X_test))

    results.append({
        'method': 'Conformal (Absolute)',
        'coverage': cp_abs_coverage,
        'avg_interval_width': cp_abs_width,
        'target_coverage': 0.8
    })

    print(f"  Coverage: {cp_abs_coverage:.1%} (target: 80%)")
    print(f"  Avg Interval Width: {cp_abs_width:.6f}")

    # 3. Conformal Prediction (Normalized)
    print("\n" + "="*60)
    print("CONFORMAL PREDICTION (Normalized)")
    print("="*60)

    try:
        import lightgbm as lgb
        base_model_norm = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    except:
        from sklearn.ensemble import RandomForestRegressor
        base_model_norm = RandomForestRegressor(n_estimators=100, random_state=42)

    cp_norm = ConformalPredictor(base_model_norm, confidence_level=0.8, method='normalized')
    cp_norm.fit(X_train, y_train, X_cal, y_cal)

    cp_norm_coverage = cp_norm.coverage_score(X_test, y_test)
    cp_norm_width = np.mean(cp_norm.interval_width(X_test))

    results.append({
        'method': 'Conformal (Normalized)',
        'coverage': cp_norm_coverage,
        'avg_interval_width': cp_norm_width,
        'target_coverage': 0.8
    })

    print(f"  Coverage: {cp_norm_coverage:.1%} (target: 80%)")
    print(f"  Avg Interval Width: {cp_norm_width:.6f}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Testing Probabilistic Models...")

    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    # Heteroscedastic noise (variance increases with X[:, 0])
    noise_std = 0.1 + 0.3 * np.abs(X[:, 0])
    y = X[:, 0] + 0.5 * X[:, 1] + noise_std * np.random.randn(n_samples)

    # Split
    train_size = 600
    cal_size = 200

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_cal = X[train_size:train_size+cal_size]
    y_cal = y[train_size:train_size+cal_size]
    X_test = X[train_size+cal_size:]
    y_test = y[train_size+cal_size:]

    # Compare methods
    results_df = compare_uncertainty_methods(
        X_train, y_train,
        X_cal, y_cal,
        X_test, y_test
    )

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))

    print("\n[SUCCESS] All probabilistic models working!")
