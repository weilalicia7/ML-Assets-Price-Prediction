"""
Ensemble Model for Volatility Prediction
Combines multiple models (LightGBM, XGBoost) with adaptive weighting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib
from .base_models import VolatilityPredictor


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple models.

    Features:
    - Combines LightGBM and XGBoost
    - Adaptive weighting based on recent performance
    - Uncertainty quantification
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize ensemble predictor.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.weights = {}
        self.performance_history = []

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        models_to_train: List[str] = ['lightgbm', 'xgboost']
    ):
        """
        Train all models in the ensemble.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            models_to_train: List of models to train
        """
        print("="*60)
        print("TRAINING ENSEMBLE MODELS")
        print("="*60)

        for model_type in models_to_train:
            print(f"\n[INFO] Training {model_type.upper()}...")

            predictor = VolatilityPredictor(model_type=model_type, random_state=self.random_state)

            if model_type == 'lightgbm':
                predictor.train_lightgbm(X_train, y_train, X_val, y_val)
            elif model_type == 'xgboost':
                predictor.train_xgboost(X_train, y_train, X_val, y_val)

            # Evaluate on validation set (if available)
            if X_val is not None and y_val is not None:
                val_metrics = predictor.evaluate(X_val, y_val)

                print(f"[OK] {model_type.upper()} Validation Metrics:")
                print(f"     MAE:  {val_metrics['mae']:.6f}")
                print(f"     RMSE: {val_metrics['rmse']:.6f}")
                print(f"     R²:   {val_metrics['r2']:.4f}")
                print(f"     MAPE: {val_metrics['mape']:.2f}%")
            else:
                print(f"[OK] {model_type.upper()} trained (no validation set)")
                val_metrics = {'mae': 0, 'rmse': 0, 'r2': 0, 'mape': 0}

            self.models[model_type] = predictor

            # Store performance
            self.performance_history.append({
                'model': model_type,
                'mae': val_metrics['mae'],
                'rmse': val_metrics['rmse'],
                'r2': val_metrics['r2']
            })

        # Calculate initial weights based on validation performance
        self._update_weights_from_performance()

        print("\n" + "="*60)
        print("ENSEMBLE TRAINING COMPLETE")
        print("="*60)
        print("\nModel Weights:")
        for model_name, weight in self.weights.items():
            print(f"  {model_name}: {weight:.3f}")

    def _update_weights_from_performance(self):
        """
        Update model weights based on performance.

        Uses inverse MAE - better models get higher weight.
        """
        if not self.performance_history:
            # Equal weights if no history
            n_models = len(self.models)
            self.weights = {name: 1.0/n_models for name in self.models.keys()}
            return

        # Calculate inverse MAE (lower MAE = better = higher weight)
        mae_dict = {}
        for record in self.performance_history:
            model_name = record['model']
            mae = record['mae']
            mae_dict[model_name] = mae

        # Check if all MAEs are zero (no validation set)
        if all(mae == 0 for mae in mae_dict.values()):
            # Use equal weights when no validation metrics available
            n_models = len(self.models)
            self.weights = {name: 1.0/n_models for name in self.models.keys()}
            return

        # Inverse and normalize (handle zero MAE by using small epsilon)
        epsilon = 1e-10
        inverse_mae = {name: 1.0/(mae + epsilon) for name, mae in mae_dict.items()}
        total = sum(inverse_mae.values())
        self.weights = {name: inv/total for name, inv in inverse_mae.items()}

    def predict(self, X: pd.DataFrame, return_individual: bool = False) -> np.ndarray:
        """
        Make ensemble predictions.

        Args:
            X: Features
            return_individual: If True, also return individual model predictions

        Returns:
            Ensemble predictions (and optionally individual predictions)
        """
        if not self.models:
            raise ValueError("No models trained yet!")

        # Get predictions from each model
        predictions = {}
        for model_name, predictor in self.models.items():
            predictions[model_name] = predictor.predict(X)

        # Weighted average
        ensemble_pred = np.zeros(len(X))
        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 1.0/len(self.models))
            ensemble_pred += weight * pred

        if return_individual:
            return ensemble_pred, predictions
        else:
            return ensemble_pred

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
        confidence_level: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty bounds.

        Uses ensemble disagreement as uncertainty measure.

        Args:
            X: Features
            confidence_level: Confidence level (0-1)

        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        # Get individual predictions
        ensemble_pred, individual_preds = self.predict(X, return_individual=True)

        # Calculate standard deviation across models
        pred_array = np.array([pred for pred in individual_preds.values()])
        pred_std = np.std(pred_array, axis=0)

        # Confidence interval (using normal approximation)
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        lower_bound = ensemble_pred - z_score * pred_std
        upper_bound = ensemble_pred + z_score * pred_std

        # Ensure bounds are positive (volatility can't be negative)
        lower_bound = np.maximum(lower_bound, 0)

        return ensemble_pred, lower_bound, upper_bound

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluate ensemble performance.

        Args:
            X: Features
            y: True target values

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        predictions = self.predict(X)

        metrics = {
            'mae': mean_absolute_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'r2': r2_score(y, predictions),
            'mape': np.mean(np.abs((y - predictions) / y)) * 100
        }

        # Also get individual model metrics
        individual_metrics = {}
        for model_name, predictor in self.models.items():
            individual_metrics[model_name] = predictor.evaluate(X, y)

        metrics['individual'] = individual_metrics

        return metrics

    def get_feature_importance(self, top_n: int = 20, method: str = 'average') -> pd.DataFrame:
        """
        Get aggregated feature importance across models.

        Args:
            top_n: Number of top features
            method: 'average' or 'weighted' (by model weights)

        Returns:
            DataFrame with feature importance
        """
        all_importance = []

        for model_name, predictor in self.models.items():
            importance_df = predictor.get_feature_importance(top_n=len(predictor.feature_names))
            importance_df['model'] = model_name

            if method == 'weighted':
                weight = self.weights.get(model_name, 1.0)
                importance_df['importance'] = importance_df['importance'] * weight

            all_importance.append(importance_df)

        # Combine
        combined = pd.concat(all_importance)

        # Aggregate by feature
        aggregated = combined.groupby('feature')['importance'].sum().reset_index()
        aggregated = aggregated.sort_values('importance', ascending=False)

        return aggregated.head(top_n)

    def update_weights_from_recent_performance(
        self,
        X_recent: pd.DataFrame,
        y_recent: pd.Series,
        window_size: int = 20
    ):
        """
        Update model weights based on recent performance.

        This allows the ensemble to adapt over time.

        Args:
            X_recent: Recent features
            y_recent: Recent true values
            window_size: Number of recent predictions to consider
        """
        # Evaluate each model on recent data
        recent_performance = {}

        for model_name, predictor in self.models.items():
            metrics = predictor.evaluate(X_recent.tail(window_size), y_recent.tail(window_size))
            recent_performance[model_name] = metrics['mae']

        # Update weights (inverse MAE)
        inverse_mae = {name: 1.0/mae for name, mae in recent_performance.items()}
        total = sum(inverse_mae.values())
        self.weights = {name: inv/total for name, inv in inverse_mae.items()}

        print("[INFO] Updated ensemble weights based on recent performance:")
        for model_name, weight in self.weights.items():
            print(f"  {model_name}: {weight:.3f}")

    def save_ensemble(self, path: str):
        """Save entire ensemble to disk."""
        ensemble_data = {
            'models': self.models,
            'weights': self.weights,
            'performance_history': self.performance_history,
            'random_state': self.random_state
        }

        joblib.dump(ensemble_data, path)
        print(f"[OK] Ensemble saved to {path}")

    def load_ensemble(self, path: str):
        """Load ensemble from disk."""
        ensemble_data = joblib.load(path)

        self.models = ensemble_data['models']
        self.weights = ensemble_data['weights']
        self.performance_history = ensemble_data['performance_history']
        self.random_state = ensemble_data['random_state']

        print(f"[OK] Ensemble loaded from {path}")
        print(f"     {len(self.models)} models loaded")


def main():
    """
    Example usage of EnsemblePredictor.
    """
    import sys
    sys.path.insert(0, '.')

    from src.data.fetch_data import DataFetcher
    from src.features.technical_features import TechnicalFeatureEngineer
    from src.features.volatility_features import VolatilityFeatureEngineer
    from src.models.base_models import VolatilityPredictor

    print("="*60)
    print("ENSEMBLE MODEL - EXAMPLE")
    print("="*60)

    # Step 1: Get data
    print("\nStep 1: Fetching data...")
    fetcher = DataFetcher(['AAPL'], start_date='2022-01-01')
    data = fetcher.fetch_all()
    aapl = data[data['Ticker'] == 'AAPL'].copy()
    print(f"[OK] {len(aapl)} rows")

    # Step 2: Features
    print("\nStep 2: Engineering features...")
    tech_eng = TechnicalFeatureEngineer()
    aapl = tech_eng.add_all_features(aapl)

    vol_eng = VolatilityFeatureEngineer()
    aapl = vol_eng.add_all_features(aapl)
    print(f"[OK] {len(aapl.columns)} columns")

    # Step 3: Target
    print("\nStep 3: Creating target...")
    temp_predictor = VolatilityPredictor()
    aapl = temp_predictor.create_target(aapl, target_type='next_day_volatility')

    # Step 4: Split
    print("\nStep 4: Splitting data...")
    train_df, val_df, test_df = temp_predictor.prepare_data(aapl)

    exclude_cols = ['Ticker', 'AssetType', 'target_volatility', 'volatility_regime',
                    'Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    X_train = train_df[feature_cols]
    y_train = train_df['target_volatility']
    X_val = val_df[feature_cols]
    y_val = val_df['target_volatility']
    X_test = test_df[feature_cols]
    y_test = test_df['target_volatility']

    print(f"[OK] {len(feature_cols)} features")

    # Step 5: Train Ensemble
    print("\nStep 5: Training ensemble...")
    ensemble = EnsemblePredictor(random_state=42)
    ensemble.train_all_models(X_train, y_train, X_val, y_val,
                              models_to_train=['lightgbm', 'xgboost'])

    # Step 6: Evaluate
    print("\nStep 6: Evaluating on test set...")
    test_metrics = ensemble.evaluate(X_test, y_test)

    print("\n" + "="*60)
    print("ENSEMBLE RESULTS")
    print("="*60)
    print(f"\nEnsemble Test Metrics:")
    print(f"  MAE:  {test_metrics['mae']:.6f}")
    print(f"  RMSE: {test_metrics['rmse']:.6f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")

    print(f"\nIndividual Model Performance:")
    for model_name, metrics in test_metrics['individual'].items():
        print(f"\n{model_name.upper()}:")
        print(f"  MAE:  {metrics['mae']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  R²:   {metrics['r2']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")

    # Step 7: Predictions with uncertainty
    print("\nStep 7: Making predictions with uncertainty...")
    pred, lower, upper = ensemble.predict_with_uncertainty(X_test.head(5), confidence_level=0.8)

    print("\nSample Predictions (first 5):")
    results_df = pd.DataFrame({
        'True': y_test.head(5).values,
        'Predicted': pred,
        'Lower (80%)': lower,
        'Upper (80%)': upper
    })
    print(results_df.to_string(index=False))

    # Step 8: Feature importance
    print("\nTop 10 Important Features (Ensemble):")
    importance = ensemble.get_feature_importance(top_n=10, method='weighted')
    print(importance.to_string(index=False))

    print("\n[SUCCESS] Ensemble model complete!")
    print(f"\nEnsemble Improvement over best individual model:")
    best_individual_mae = min(m['mae'] for m in test_metrics['individual'].values())
    improvement = (best_individual_mae - test_metrics['mae']) / best_individual_mae * 100
    print(f"  {improvement:.2f}% better MAE")


if __name__ == "__main__":
    main()
