"""
Base ML Models for Volatility Prediction
Implements LightGBM, XGBoost, and other models.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple, Optional
import joblib
from datetime import datetime


class VolatilityPredictor:
    """
    Base class for volatility prediction models.
    """

    def __init__(self, model_type: str = 'lightgbm', random_state: int = 42):
        """
        Initialize predictor.

        Args:
            model_type: 'lightgbm', 'xgboost', or 'ensemble'
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.scaler = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'target_volatility',
        test_size: float = 0.15,
        val_size: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for training (time-series split).

        Args:
            df: DataFrame with features
            target_col: Target column name
            test_size: Test set proportion
            val_size: Validation set proportion

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print(f"[INFO] Preparing data for {self.model_type}...")

        # Sort by date (critical for time series!)
        df = df.sort_index()

        # Calculate split points
        n = len(df)
        test_start = int(n * (1 - test_size))
        val_start = int(n * (1 - test_size - val_size))

        # Split
        train_df = df.iloc[:val_start].copy()
        val_df = df.iloc[val_start:test_start].copy()
        test_df = df.iloc[test_start:].copy()

        print(f"[OK] Train: {len(train_df)} rows")
        print(f"[OK] Val:   {len(val_df)} rows")
        print(f"[OK] Test:  {len(test_df)} rows")

        return train_df, val_df, test_df

    def create_target(self, df: pd.DataFrame, target_type: str = 'next_day_volatility') -> pd.DataFrame:
        """
        Create target variable for prediction.

        Args:
            df: DataFrame with OHLC data
            target_type: Type of target
                - 'next_day_volatility': Predict next day's volatility
                - 'next_day_range': Predict next day's High-Low range
                - 'next_day_return': Predict next day's return

        Returns:
            DataFrame with target column added
        """
        print(f"[INFO] Creating target: {target_type}")

        df = df.copy()

        if target_type == 'next_day_volatility':
            # Use next day's intraday range as proxy for volatility
            df['target_volatility'] = ((df['High'] - df['Low']) / df['Close']).shift(-1)

        elif target_type == 'next_day_range':
            # Predict absolute range
            df['target_volatility'] = (df['High'] - df['Low']).shift(-1)

        elif target_type == 'next_day_return':
            # Predict return
            df['target_volatility'] = df['Close'].pct_change().shift(-1)

        # Remove last row (no target)
        df = df.dropna(subset=['target_volatility'])

        print(f"[OK] Target created: {len(df)} rows remaining")
        print(f"[OK] Target stats - Mean: {df['target_volatility'].mean():.4f}, Std: {df['target_volatility'].std():.4f}")

        return df

    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: Optional[Dict] = None
    ):
        """
        Train LightGBM model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            params: Custom parameters (optional)
        """
        print("[INFO] Training LightGBM model...")

        # Default parameters optimized for volatility prediction
        # PHASE 1 IMPROVEMENT: Stronger regularization to prevent overfitting
        default_params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'learning_rate': 0.02,  # Reduced from 0.03 for slower, more stable learning
            'num_leaves': 31,  # Reduced from 63 to prevent overfitting
            'max_depth': 6,  # Reduced from 8 for simpler trees
            'min_data_in_leaf': 100,  # Increased from 50 to require more samples per leaf
            'feature_fraction': 0.6,  # Reduced from 0.8 to add more randomness
            'bagging_fraction': 0.7,  # Reduced from 0.8 for more bootstrap diversity
            'bagging_freq': 5,
            'lambda_l1': 0.5,  # Increased from 0.1 for stronger L1 regularization
            'lambda_l2': 0.5,  # Increased from 0.1 for stronger L2 regularization
            'min_gain_to_split': 0.01,  # Added: minimum gain required to make a split
            'verbose': -1,
            'random_state': self.random_state
        }

        if params:
            default_params.update(params)

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)

        # Handle validation data (may be None)
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets = [train_data, val_data]
            valid_names = ['train', 'val']
            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        else:
            valid_sets = [train_data]
            valid_names = ['train']
            callbacks = [lgb.log_evaluation(period=100)]

        # Train
        self.model = lgb.train(
            default_params,
            train_data,
            num_boost_round=1000,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        self.feature_names = X_train.columns.tolist()
        print(f"[OK] LightGBM trained ({self.model.best_iteration} iterations)")

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: Optional[Dict] = None
    ):
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            params: Custom parameters (optional)
        """
        print("[INFO] Training XGBoost model...")

        # Default parameters
        # PHASE 1 IMPROVEMENT: Stronger regularization to prevent overfitting
        default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'learning_rate': 0.02,  # Reduced from 0.03 for slower learning
            'max_depth': 5,  # Reduced from 7 for simpler trees
            'min_child_weight': 10,  # Increased from 5 for more conservative splits
            'subsample': 0.7,  # Reduced from 0.8 for more diversity
            'colsample_bytree': 0.6,  # Reduced from 0.8 for feature randomness
            'gamma': 0.2,  # Increased from 0.1 for higher pruning threshold
            'reg_alpha': 0.5,  # Increased from 0.1 for stronger L1 regularization
            'reg_lambda': 2.0,  # Increased from 1.0 for stronger L2 regularization
            'tree_method': 'hist',
            'random_state': self.random_state
        }

        if params:
            default_params.update(params)

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)

        # Handle validation data (may be None)
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, 'train'), (dval, 'val')]
            early_stopping_rounds = 50
        else:
            evals = [(dtrain, 'train')]
            early_stopping_rounds = None

        # Train
        self.model = xgb.train(
            default_params,
            dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100
        )

        self.feature_names = X_train.columns.tolist()

        # best_iteration only available with early stopping
        if early_stopping_rounds is not None:
            print(f"[OK] XGBoost trained ({self.model.best_iteration} iterations)")
        else:
            print(f"[OK] XGBoost trained (1000 iterations, no validation set)")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        if self.model_type == 'lightgbm':
            return self.model.predict(X)
        elif self.model_type == 'xgboost':
            dmatrix = xgb.DMatrix(X)
            return self.model.predict(dmatrix)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluate model performance.

        Args:
            X: Features
            y: True target values

        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X)

        metrics = {
            'mae': mean_absolute_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'r2': r2_score(y, predictions),
            'mape': np.mean(np.abs((y - predictions) / y)) * 100
        }

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        if self.model_type == 'lightgbm':
            importance = self.model.feature_importance(importance_type='gain')
            feature_names = self.feature_names
        elif self.model_type == 'xgboost':
            importance_dict = self.model.get_score(importance_type='gain')
            feature_names = list(importance_dict.keys())
            importance = list(importance_dict.values())

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })

        importance_df = importance_df.sort_values('importance', ascending=False)
        return importance_df.head(top_n)

    def save_model(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save!")

        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'random_state': self.random_state
        }

        joblib.dump(model_data, path)
        print(f"[OK] Model saved to {path}")

    def load_model(self, path: str):
        """Load model from disk."""
        model_data = joblib.load(path)

        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.random_state = model_data['random_state']

        print(f"[OK] Model loaded from {path}")


def main():
    """
    Example usage of VolatilityPredictor.
    """
    from src.data.fetch_data import DataFetcher
    from src.features.technical_features import TechnicalFeatureEngineer
    from src.features.volatility_features import VolatilityFeatureEngineer

    print("="*60)
    print("VOLATILITY PREDICTION - EXAMPLE")
    print("="*60)

    # Step 1: Fetch data
    print("\nStep 1: Fetching data...")
    fetcher = DataFetcher(['AAPL'], start_date='2022-01-01')
    data = fetcher.fetch_all()
    aapl = data[data['Ticker'] == 'AAPL'].copy()

    # Step 2: Engineer features
    print("\nStep 2: Engineering features...")
    tech_eng = TechnicalFeatureEngineer()
    aapl = tech_eng.add_all_features(aapl)

    vol_eng = VolatilityFeatureEngineer()
    aapl = vol_eng.add_all_features(aapl)

    print(f"[OK] Total features: {len(aapl.columns)}")

    # Step 3: Create target
    print("\nStep 3: Creating target...")
    predictor = VolatilityPredictor(model_type='lightgbm')
    aapl = predictor.create_target(aapl, target_type='next_day_volatility')

    # Step 4: Prepare data
    print("\nStep 4: Preparing train/val/test split...")
    train_df, val_df, test_df = predictor.prepare_data(aapl)

    # Select features (exclude non-feature columns)
    exclude_cols = ['Ticker', 'AssetType', 'target_volatility', 'volatility_regime',
                    'Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    X_train = train_df[feature_cols]
    y_train = train_df['target_volatility']
    X_val = val_df[feature_cols]
    y_val = val_df['target_volatility']
    X_test = test_df[feature_cols]
    y_test = test_df['target_volatility']

    print(f"[OK] Using {len(feature_cols)} features")

    # Step 5: Train model
    print("\nStep 5: Training LightGBM model...")
    predictor.train_lightgbm(X_train, y_train, X_val, y_val)

    # Step 6: Evaluate
    print("\nStep 6: Evaluating model...")
    train_metrics = predictor.evaluate(X_train, y_train)
    val_metrics = predictor.evaluate(X_val, y_val)
    test_metrics = predictor.evaluate(X_test, y_test)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nTrain Metrics:")
    print(f"  MAE:  {train_metrics['mae']:.6f}")
    print(f"  RMSE: {train_metrics['rmse']:.6f}")
    print(f"  R²:   {train_metrics['r2']:.4f}")
    print(f"  MAPE: {train_metrics['mape']:.2f}%")

    print(f"\nValidation Metrics:")
    print(f"  MAE:  {val_metrics['mae']:.6f}")
    print(f"  RMSE: {val_metrics['rmse']:.6f}")
    print(f"  R²:   {val_metrics['r2']:.4f}")
    print(f"  MAPE: {val_metrics['mape']:.2f}%")

    print(f"\nTest Metrics:")
    print(f"  MAE:  {test_metrics['mae']:.6f}")
    print(f"  RMSE: {test_metrics['rmse']:.6f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")

    # Step 7: Feature importance
    print("\nTop 10 Most Important Features:")
    importance = predictor.get_feature_importance(top_n=10)
    print(importance.to_string(index=False))

    print("\n[SUCCESS] Model training complete!")


if __name__ == "__main__":
    main()
