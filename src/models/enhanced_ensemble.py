"""
Enhanced Ensemble with Prediction Market Integration
Combines existing EnsemblePredictor with PredictionMarketEnsemble

This provides the best of both worlds:
1. Your existing volatility prediction framework
2. Prediction market information weighting
3. Kelly Criterion backtesting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib

from .ensemble_model import EnsemblePredictor
from .prediction_market_ensemble import PredictionMarketEnsemble, create_prediction_market_ensemble
from .base_models import VolatilityPredictor

# Try to import neural models
try:
    from .neural_models import NeuralPredictor, TORCH_AVAILABLE
    NEURAL_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    NEURAL_AVAILABLE = False
    print("[INFO] Neural models not available")


class EnhancedEnsemblePredictor(EnsemblePredictor):
    """
    Enhanced ensemble combining volatility prediction with prediction market logic.

    Extends EnsemblePredictor to use:
    - Information-based weighting (from prediction markets)
    - Automatic weight adaptation based on recent performance
    - Calibrated probability outputs
    - Edge detection capabilities
    """

    def __init__(self, random_state: int = 42, use_prediction_market: bool = True):
        """
        Initialize enhanced ensemble.

        Args:
            random_state: Random seed
            use_prediction_market: If True, use prediction market weighting
        """
        super().__init__(random_state=random_state)
        self.use_prediction_market = use_prediction_market
        self.pm_ensemble = None
        self.directional_accuracy_history = []

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        models_to_train: List[str] = ['lightgbm', 'xgboost'],
        neural_models: Optional[List[str]] = None
    ):
        """
        Train all models with prediction market integration.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            models_to_train: Traditional models to train (lightgbm, xgboost)
            neural_models: Neural models to train (tcn, lstm, transformer)
        """
        # First, train traditional models using parent class method
        super().train_all_models(X_train, y_train, X_val, y_val, models_to_train)

        # Train neural models if specified
        if neural_models and NEURAL_AVAILABLE:
            self._train_neural_models(X_train, y_train, X_val, y_val, neural_models)

        # If using prediction market, create PM ensemble (requires validation data)
        if self.use_prediction_market and X_val is not None and y_val is not None:
            print("\n" + "="*60)
            print("INITIALIZING PREDICTION MARKET ENSEMBLE")
            print("="*60)

            # Extract models for prediction market
            # For traditional models (VolatilityPredictor), extract the underlying model
            # For neural models (NeuralPredictor), pass the predictor itself (it has .predict())
            pm_models = {}
            for name, predictor in self.models.items():
                if hasattr(predictor, 'model') and not hasattr(predictor, 'lookback'):
                    # VolatilityPredictor - extract underlying model
                    pm_models[name] = predictor.model
                else:
                    # NeuralPredictor or other - pass the predictor itself
                    pm_models[name] = predictor

            # Create prediction market ensemble
            # For regression, we need to convert to classification format
            # Use: Will next value be > current value?
            y_train_binary = (y_train > y_train.shift(1)).astype(int).iloc[1:]
            y_val_binary = (y_val > y_val.shift(1)).astype(int).iloc[1:]
            X_train_shifted = X_train.iloc[1:]
            X_val_shifted = X_val.iloc[1:]

            try:
                # Keep X as DataFrames to preserve feature names for XGBoost
                self.pm_ensemble = create_prediction_market_ensemble(
                    models=pm_models,
                    X_train=X_train_shifted,  # Keep as DataFrame
                    y_train=y_train_binary.values if hasattr(y_train_binary, 'values') else y_train_binary,
                    X_val=X_val_shifted,  # Keep as DataFrame
                    y_val=y_val_binary.values if hasattr(y_val_binary, 'values') else y_val_binary
                )

                # Update weights from prediction market
                self._sync_weights_from_pm_ensemble()

                print("\n[OK] Prediction Market ensemble initialized")
                print("     Weights synchronized with information scores")

            except Exception as e:
                print(f"\n[WARN] Could not initialize PM ensemble: {e}")
                print("       Falling back to standard weighting")
                self.use_prediction_market = False

    def _train_neural_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        neural_models: List[str]
    ):
        """
        Train neural network models.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            neural_models: List of neural models to train (tcn, lstm, transformer)
        """
        print("\n" + "="*60)
        print("TRAINING NEURAL MODELS")
        print("="*60)

        # Convert to numpy arrays
        # BUGFIX: Explicitly convert to float32 for PyTorch compatibility with mixed dtype columns
        if hasattr(X_train, 'values'):
            X_train_np = X_train.values.astype(np.float32)
        else:
            X_train_np = np.asarray(X_train, dtype=np.float32)

        if hasattr(y_train, 'values'):
            y_train_np = y_train.values.astype(np.float32)
        else:
            y_train_np = np.asarray(y_train, dtype=np.float32)

        if hasattr(X_val, 'values'):
            X_val_np = X_val.values.astype(np.float32)
        else:
            X_val_np = np.asarray(X_val, dtype=np.float32)

        if hasattr(y_val, 'values'):
            y_val_np = y_val.values.astype(np.float32)
        else:
            y_val_np = np.asarray(y_val, dtype=np.float32)

        for model_name in neural_models:
            print(f"\n[INFO] Training {model_name.upper()}...")

            try:
                # Create neural predictor
                neural_pred = NeuralPredictor(
                    model_type=model_name,
                    lookback=20,
                    hidden_size=64,
                    num_layers=2,
                    dropout=0.2,
                    learning_rate=0.001,
                    batch_size=32,
                    epochs=50,
                    device='cpu',
                    random_state=self.random_state
                )

                # Train
                neural_pred.fit(X_train_np, y_train_np, X_val_np, y_val_np)

                # Store in models dict
                self.models[model_name] = neural_pred

                # Initialize weight
                self.weights[model_name] = 1.0 / (len(self.models))

                print(f"[OK] {model_name.upper()} trained successfully")

            except Exception as e:
                print(f"[FAILED] Could not train {model_name}: {e}")
                import traceback
                traceback.print_exc()

        # Re normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {name: w/total_weight for name, w in self.weights.items()}

    def _sync_weights_from_pm_ensemble(self):
        """
        Sync weights from prediction market ensemble to main ensemble.

        Uses information scores from PM ensemble to set weights.
        """
        if not self.pm_ensemble:
            return

        # Get information scores
        info_scores = self.pm_ensemble.model_info_scores

        # Normalize to sum to 1
        total_info = sum(abs(score) for score in info_scores.values())

        if total_info > 0:
            self.weights = {
                name: abs(score) / total_info
                for name, score in info_scores.items()
            }

            print("\n[INFO] Weights updated from Prediction Market:")
            for name, weight in self.weights.items():
                info_score = info_scores.get(name, 0)
                print(f"  {name:15s}: {weight:.3f} (info={info_score:.3f})")

    def predict(self, X: pd.DataFrame, return_individual: bool = False) -> np.ndarray:
        """
        Make predictions with optional PM weighting.
        Handles both traditional and neural models.

        Args:
            X: Features
            return_individual: Return individual predictions too

        Returns:
            Ensemble prediction
        """
        if not self.models:
            raise ValueError("No models trained yet!")

        # Convert to numpy if needed for neural models
        # BUGFIX: Explicitly convert to float32 for PyTorch compatibility
        if hasattr(X, 'values'):
            X_np = X.values.astype(np.float32)
        else:
            X_np = np.asarray(X, dtype=np.float32)

        # Get predictions from each model
        predictions = {}
        max_lookback = 0

        for model_name, predictor in self.models.items():
            # Check if neural model
            if hasattr(predictor, 'lookback'):
                # Neural model
                pred = predictor.predict(X_np)
                predictions[model_name] = pred
                max_lookback = max(max_lookback, predictor.lookback)
            else:
                # Traditional model (VolatilityPredictor)
                pred = predictor.predict(X)
                predictions[model_name] = pred

        # Align predictions (neural models have shorter output due to lookback)
        # Calculate minimum length across all predictions
        min_length = min(len(p) for p in predictions.values())

        if max_lookback > 0:
            # Trim all predictions to match minimum length
            for model_name in predictions:
                if len(predictions[model_name]) > min_length:
                    # Trim from the end to align with neural models
                    predictions[model_name] = predictions[model_name][-min_length:]

        # Weighted average
        ensemble_pred = np.zeros(min_length)

        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 1.0/len(self.models))
            ensemble_pred += weight * pred

        if return_individual:
            return ensemble_pred, predictions
        else:
            return ensemble_pred

    def predict_with_pm_consensus(
        self,
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict]:
        """
        Get predictions with prediction market consensus metrics.

        Args:
            X: Features

        Returns:
            (predictions, consensus_metrics)
        """
        # Get standard prediction
        predictions = self.predict(X)

        # Get PM consensus if available
        consensus = {}
        if self.pm_ensemble:
            try:
                consensus = self.pm_ensemble.get_market_consensus(
                    X.values if hasattr(X, 'values') else X
                )
            except Exception as e:
                print(f"[WARN] Could not get PM consensus: {e}")

        return predictions, consensus

    def update_weights_from_recent_performance(
        self,
        X_recent: pd.DataFrame,
        y_recent: pd.Series,
        window_size: int = 20
    ):
        """
        Update weights using prediction market logic.

        Args:
            X_recent: Recent features
            y_recent: Recent targets
            window_size: Window for evaluation
        """
        if not self.use_prediction_market or not self.pm_ensemble:
            # Fall back to parent method
            super().update_weights_from_recent_performance(X_recent, y_recent, window_size)
            return

        # Update PM ensemble performance
        # Convert to binary (directional)
        y_binary = (y_recent > y_recent.shift(1)).astype(int).iloc[1:]
        X_shifted = X_recent.iloc[1:]

        for name, predictor in self.models.items():
            try:
                # Get predictions
                y_pred = predictor.predict(X_shifted)

                # Convert to probabilities (above/below median)
                median_pred = np.median(y_pred)
                y_pred_proba = (y_pred > median_pred).astype(float)

                # Update PM ensemble
                self.pm_ensemble.update_model_performance(
                    model_name=name,
                    y_true=y_binary.values if hasattr(y_binary, 'values') else y_binary,
                    y_pred_proba=y_pred_proba,
                    regime=None
                )
            except Exception as e:
                print(f"[WARN] Could not update {name}: {e}")

        # Sync weights
        self._sync_weights_from_pm_ensemble()

    def get_model_rankings(self) -> pd.DataFrame:
        """
        Get model rankings with PM information scores.

        Returns:
            DataFrame with comprehensive model metrics
        """
        if not self.pm_ensemble:
            # Return basic rankings
            rankings = []
            for name in self.models.keys():
                rankings.append({
                    'model': name,
                    'weight': self.weights.get(name, 0),
                    'n_predictions': len(self.performance_history)
                })
            return pd.DataFrame(rankings)

        # Get PM rankings
        pm_rankings = self.pm_ensemble.get_model_rankings()

        # Add ensemble weights
        pm_rankings['ensemble_weight'] = pm_rankings['model'].map(self.weights)

        return pm_rankings

    def save_ensemble(self, path: str):
        """Save enhanced ensemble including PM components."""
        ensemble_data = {
            'models': self.models,
            'weights': self.weights,
            'performance_history': self.performance_history,
            'random_state': self.random_state,
            'use_prediction_market': self.use_prediction_market,
            'pm_ensemble': self.pm_ensemble,
            'directional_accuracy_history': self.directional_accuracy_history
        }

        joblib.dump(ensemble_data, path)
        print(f"[OK] Enhanced ensemble saved to {path}")

    def load_ensemble(self, path: str):
        """Load enhanced ensemble from disk."""
        ensemble_data = joblib.load(path)

        self.models = ensemble_data['models']
        self.weights = ensemble_data['weights']
        self.performance_history = ensemble_data['performance_history']
        self.random_state = ensemble_data['random_state']
        self.use_prediction_market = ensemble_data.get('use_prediction_market', False)
        self.pm_ensemble = ensemble_data.get('pm_ensemble', None)
        self.directional_accuracy_history = ensemble_data.get('directional_accuracy_history', [])

        print(f"[OK] Enhanced ensemble loaded from {path}")
        print(f"     {len(self.models)} models loaded")
        if self.pm_ensemble:
            print(f"     Prediction Market ensemble: Enabled")


def main():
    """
    Example usage of EnhancedEnsemblePredictor.
    """
    import sys
    sys.path.insert(0, '.')

    from src.data.fetch_data import DataFetcher
    from src.features.technical_features import TechnicalFeatureEngineer
    from src.features.volatility_features import VolatilityFeatureEngineer
    from src.models.base_models import VolatilityPredictor

    print("="*60)
    print("ENHANCED ENSEMBLE - EXAMPLE")
    print("With Prediction Market Integration")
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

    # Step 5: Train Enhanced Ensemble
    print("\nStep 5: Training enhanced ensemble...")
    ensemble = EnhancedEnsemblePredictor(random_state=42, use_prediction_market=True)
    ensemble.train_all_models(X_train, y_train, X_val, y_val,
                              models_to_train=['lightgbm', 'xgboost'])

    # Step 6: Evaluate
    print("\nStep 6: Evaluating on test set...")
    test_metrics = ensemble.evaluate(X_test, y_test)

    print("\n" + "="*60)
    print("ENHANCED ENSEMBLE RESULTS")
    print("="*60)
    print(f"\nEnsemble Test Metrics:")
    print(f"  MAE:  {test_metrics['mae']:.6f}")
    print(f"  RMSE: {test_metrics['rmse']:.6f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")

    # Step 7: Model Rankings
    print("\nModel Rankings (with PM information scores):")
    rankings = ensemble.get_model_rankings()
    print(rankings.to_string(index=False))

    # Step 8: Predictions with PM consensus
    print("\nPredictions with PM Consensus:")
    pred, consensus = ensemble.predict_with_pm_consensus(X_test.head(5))

    print("\nPM Consensus Metrics:")
    for key, value in consensus.items():
        print(f"  {key:25s}: {value}")

    # Step 9: Update weights based on recent performance
    print("\nUpdating weights from recent performance...")
    ensemble.update_weights_from_recent_performance(X_test.tail(50), y_test.tail(50))

    print("\n[SUCCESS] Enhanced ensemble complete!")


if __name__ == "__main__":
    main()
