"""
China-Specific Prediction Pipeline

Complete prediction pipeline optimized for Chinese markets (HK, Shanghai, Shenzhen).

Key Differences from US Model:
1. China-specific macro features (CSI300, CNY, HSI instead of VIX, SPY, DXY)
2. Regime detection with dynamic weighting
3. Validation-based ensemble weights (not equal 50/50)
4. Category-specific feature filtering

Usage:
    from src.models.china_predictor import ChinaMarketPredictor

    model = ChinaMarketPredictor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
"""

import pandas as pd
import numpy as np
from typing import Optional

# Import base models
from src.models.hybrid_ensemble import HybridEnsemblePredictor

# Import China-specific features
from src.features.china_macro_features import ChinaMacroFeatureEngineer
from src.features.technical_features import TechnicalFeatureEngineer
from src.features.volatility_features import VolatilityFeatureEngineer
from src.features.regime_detection import VolatilityRegimeDetector, TrendRegimeDetector
from src.features.liquidity_features import LiquidityFeatureEngineer, TransactionCostModel


class ChinaMarketPredictor:
    """
    Prediction model optimized for Chinese markets.

    Automatically applies:
    - China-specific macro features
    - Regime detection
    - Validation-based model weighting
    - PDF Improvements: Reduced features, reweighted ensemble, regularized neural models
    """

    # PDF RECOMMENDATION 1: Core features (30-35 features instead of 114)
    # PHASE 1: 30 features → 50.26% accuracy, -0.21% profit (FAILED)
    # PHASE 2A: ULTRA-MINIMAL - 10 MOST IMPORTANT features only
    # Hypothesis: Fewer features = less noise = better generalization on limited Chinese data
    CORE_FEATURES = [
        # PHASE 2A: TOP 10 FEATURES ONLY (selected based on PDF + domain knowledge)
        # Price/Returns (2 features) - Most fundamental
        'returns_1d', 'returns_5d',  # FIX: was 'returns', actual name is 'returns_1d'

        # Momentum (2 features) - Proven effective in Chinese markets
        'rsi_14', 'macd',

        # Volatility (2 features) - Critical for regime detection
        'parkinson_vol_20', 'volatility',

        # China-Specific Macro (2 features) - Market drivers (NOTE: SSEC often fails to download)
        'hsi_return', 'cny_return',  # Removed csi300_return for now, may not exist

        # Volume (1 feature) - Liquidity proxy
        'volume_ratio',  # Added to replace missing macro feature

        # Regime (1 feature) - Market state awareness
        'regime',

        # PHASE 1 (30 features) - COMMENTED OUT (Phase 1 results: 50.26% acc, -0.21% profit)
        # 'high_low_ratio', 'volume_ratio', 'price_momentum_10',
        # 'macd_signal', 'roc_10', 'williams_r',
        # 'atr_14', 'bollinger_width', 'volatility_rank',
        # 'gold_return', 'spy_return',
        # 'regime_duration', 'regime_stability', 'vol_spike', 'trend_score',
        # 'ema_12', 'ema_26', 'sma_50', 'distance_from_sma', 'mean_reversion_score',
    ]

    def __init__(self, use_regime_detection=True, use_validation_weighting=True,
                 reduced_features=True, tree_weight=0.7):
        """
        Initialize China market predictor with PDF improvements.

        Args:
            use_regime_detection: Enable regime-based feature engineering
            use_validation_weighting: Use validation performance to weight models
            reduced_features: PDF REC 1 - Use only 20-30 core features (default: True)
            tree_weight: PDF REC 3 - Weight for tree models vs neural (default: 0.7 for 70/30)
        """
        self.use_regime_detection = use_regime_detection
        self.use_validation_weighting = use_validation_weighting
        self.reduced_features = reduced_features  # PDF REC 1
        self.tree_weight = tree_weight  # PDF REC 3 (70% tree, 30% neural)

        # Initialize base hybrid ensemble with custom tree weight
        self.base_model = HybridEnsemblePredictor()

        # PDF REC 3: Override default 50/50 weights to 70/30
        if hasattr(self.base_model, 'old_model_weight'):
            self.base_model.old_model_weight = tree_weight  # 70% for tree models
        if hasattr(self.base_model, 'new_model_weight'):
            self.base_model.new_model_weight = 1.0 - tree_weight  # 30% for neural models

        # Initialize regime detectors
        self.vol_regime_detector = VolatilityRegimeDetector(n_regimes=4, method='gmm') if use_regime_detection else None
        self.trend_regime_detector = TrendRegimeDetector(lookback=60) if use_regime_detection else None

        # Store validation performance
        self.validation_performance = {
            'old_model_accuracy': None,
            'new_model_accuracy': None,
            'old_model_weight': tree_weight,  # PDF REC 3: Start with 70/30
            'new_model_weight': 1.0 - tree_weight
        }

        print(f"[PDF IMPROVEMENTS] China Model initialized with:")
        print(f"  - Reduced features: {reduced_features} (30 core features vs 114)")
        print(f"  - Tree/Neural weights: {tree_weight*100:.0f}/{(1-tree_weight)*100:.0f}")
        print(f"  - Regime detection: {use_regime_detection}")

    def add_features(self, df: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
        """
        Add all features for Chinese markets.

        Args:
            df: Raw OHLC DataFrame
            ticker: Optional ticker symbol for category-specific features

        Returns:
            DataFrame with features added
        """
        print(f"[INFO] Adding features for Chinese markets...")

        # Phase 1: Technical features
        tech_eng = TechnicalFeatureEngineer()
        df = tech_eng.add_all_features(df)

        # Phase 2: Volatility features
        vol_eng = VolatilityFeatureEngineer()
        df = vol_eng.add_all_features(df)

        # PHASE 1 (DISABLED): Liquidity features - did not improve profitability
        # Phase 1 test results: 50.26% accuracy, -0.21% profit (identical to baseline)
        # Keeping code for reference but commented out
        # try:
        #     liq_eng = LiquidityFeatureEngineer()
        #     df = liq_eng.add_all_features(df)
        # except Exception as e:
        #     print(f"  [WARNING] Failed to add liquidity features: {e}")
        #     print(f"  [INFO] Continuing without liquidity features...")

        # Phase 4 (China-specific): Macro features
        try:
            china_macro_eng = ChinaMacroFeatureEngineer()
            df = china_macro_eng.add_all_features(df)
        except Exception as e:
            print(f"  [WARNING] Failed to add China macro features: {e}")
            print(f"  [INFO] Continuing without macro features...")

        # Regime detection features (if enabled)
        if self.use_regime_detection and self.vol_regime_detector is not None:
            try:
                if 'Close' in df.columns:
                    # Fit volatility regime detector (has fit method)
                    self.vol_regime_detector.fit(df)
                    df = self.vol_regime_detector.get_regime_features(df)

                    # Add trend features (no fit method needed)
                    df = self.trend_regime_detector.get_trend_features(df)

                    print(f"  [OK] Added regime detection features")
                else:
                    print(f"  [WARNING] Cannot add regime features: 'Close' column not found")
            except Exception as e:
                print(f"  [WARNING] Regime detection failed: {e}")

        # Clean data
        df = df.replace([np.inf, -np.inf], 0).fillna(0)

        print(f"[OK] Features added: {len(df.columns)} total features")

        return df

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train the China market model with PDF improvements.

        Args:
            X: Training features
            y: Training target
            X_val: Optional validation features
            y_val: Optional validation target
        """
        print(f"[INFO] Training China Market Predictor...")
        print(f"  Training samples: {len(X)}")

        # PDF REC 1: Filter to core features if reduced_features=True
        if self.reduced_features:
            available_core_features = [f for f in self.CORE_FEATURES if f in X.columns]
            if len(available_core_features) >= 5:  # Need at least 5 core features (PHASE 2A: lowered from 15 for 10-feature test)
                print(f"  [PDF REC 1] Reducing features: {len(X.columns)} -> {len(available_core_features)} core features")
                X_filtered = X[available_core_features]
                X_val_filtered = X_val[available_core_features] if X_val is not None else None
            else:
                print(f"  [WARNING] Only {len(available_core_features)} core features found, using all {len(X.columns)} features")
                X_filtered = X
                X_val_filtered = X_val
        else:
            X_filtered = X
            X_val_filtered = X_val

        # Fit regime detectors if enabled
        if self.use_regime_detection and hasattr(X_filtered, 'index'):
            try:
                # Reconstruct price data from features if possible
                if 'volatility' in X_filtered.columns:
                    temp_df = pd.DataFrame(index=X_filtered.index)
                    temp_df['volatility'] = X_filtered['volatility']
                    temp_df['Close'] = 1.0  # Dummy value
                    self.vol_regime_detector.fit(temp_df, vol_col='volatility')
            except Exception as e:
                print(f"  [WARNING] Could not fit regime detector: {e}")

        # PDF REC 2: Regularize neural models before training
        self._apply_neural_regularization()

        # Train base model with filtered features
        if X_val_filtered is not None and y_val is not None:
            self.base_model.fit(X_filtered, y, X_val_filtered, y_val)
        else:
            self.base_model.fit(X_filtered, y)

        # Calculate validation weights if validation data provided
        if self.use_validation_weighting and X_val_filtered is not None and y_val is not None:
            self._calculate_validation_weights(X_val_filtered, y_val)

        print(f"[OK] China Market Predictor trained successfully")

    def _apply_neural_regularization(self):
        """
        PDF REC 2: Apply regularization to neural models.

        - Reduce epochs: 100 → 20-30
        - Increase dropout: 0.2 → 0.4
        - Add L2 regularization
        """
        try:
            # Access the new model (Hybrid LSTM/CNN)
            if hasattr(self.base_model, 'new_model') and self.base_model.new_model is not None:
                new_model = self.base_model.new_model

                # Reduce epochs
                if hasattr(new_model, 'epochs'):
                    original_epochs = new_model.epochs
                    new_model.epochs = 30  # PDF recommendation: 20-30 epochs
                    print(f"  [PDF REC 2] Neural epochs: {original_epochs} -> {new_model.epochs}")

                # Increase dropout (if model supports it)
                if hasattr(new_model, 'dropout'):
                    original_dropout = new_model.dropout
                    new_model.dropout = 0.4  # PDF recommendation: 0.4
                    print(f"  [PDF REC 2] Neural dropout: {original_dropout} -> {new_model.dropout}")

                print(f"  [PDF REC 2] Neural model regularization applied")
        except Exception as e:
            print(f"  [WARNING] Could not apply neural regularization: {e}")

    def _calculate_validation_weights(self, X_val, y_val):
        """
        Calculate model weights based on validation performance.

        Args:
            X_val: Validation features
            y_val: Validation target
        """
        print(f"[INFO] Calculating validation-based model weights...")

        try:
            # Get predictions from old and hybrid models separately
            old_model = self.base_model.old_model
            hybrid_model = self.base_model.hybrid_model

            # Predict with each model
            old_pred = old_model.predict(X_val)
            hybrid_pred = hybrid_model.predict(X_val)

            # Convert to numpy arrays to avoid DataFrame issues
            y_val_array = np.array(y_val).flatten()
            old_pred_array = np.array(old_pred).flatten()
            hybrid_pred_array = np.array(hybrid_pred).flatten()

            # Calculate directional accuracy
            actual_direction = (y_val_array > np.roll(y_val_array, 1))[1:]
            old_pred_direction = (old_pred_array > np.roll(old_pred_array, 1))[1:]
            hybrid_pred_direction = (hybrid_pred_array > np.roll(hybrid_pred_array, 1))[1:]

            old_accuracy = np.mean(actual_direction == old_pred_direction)
            hybrid_accuracy = np.mean(actual_direction == hybrid_pred_direction)

            # Store accuracies
            self.validation_performance['old_model_accuracy'] = old_accuracy
            self.validation_performance['new_model_accuracy'] = hybrid_accuracy

            # Calculate weights based on relative performance
            # Better model gets more weight
            total_accuracy = old_accuracy + hybrid_accuracy
            if total_accuracy > 0:
                self.validation_performance['old_model_weight'] = old_accuracy / total_accuracy
                self.validation_performance['new_model_weight'] = hybrid_accuracy / total_accuracy
            else:
                # Equal weights if both fail
                self.validation_performance['old_model_weight'] = 0.5
                self.validation_performance['new_model_weight'] = 0.5

            print(f"  Old Model Accuracy: {old_accuracy:.1%} -> Weight: {self.validation_performance['old_model_weight']:.2f}")
            print(f"  Hybrid Model Accuracy: {hybrid_accuracy:.1%} -> Weight: {self.validation_performance['new_model_weight']:.2f}")

        except Exception as e:
            print(f"  [WARNING] Could not calculate validation weights: {e}")
            print(f"  [INFO] Using equal weights (50/50)")

    def predict(self, X):
        """
        Make predictions using validation-weighted ensemble with PDF improvements.

        Args:
            X: Features

        Returns:
            Predictions
        """
        # PDF REC 1: Filter to core features if reduced_features=True
        if self.reduced_features:
            available_core_features = [f for f in self.CORE_FEATURES if f in X.columns]
            # BUG FIX: Changed threshold from 15 to 5 to match fit() method
            if len(available_core_features) >= 5:
                X_filtered = X[available_core_features]
                print(f"  [PDF REC 1 PREDICT] Using {len(available_core_features)} core features (from {len(X.columns)} total)")
            else:
                X_filtered = X
                print(f"  [WARNING] Only {len(available_core_features)} core features available, using all {len(X.columns)} features")
        else:
            X_filtered = X

        if self.use_validation_weighting and self.validation_performance['old_model_accuracy'] is not None:
            # Use custom weights
            old_weight = self.validation_performance['old_model_weight']
            new_weight = self.validation_performance['new_model_weight']

            # Get predictions from each model
            old_pred = self.base_model.old_model.predict(X_filtered)
            new_pred = self.base_model.new_model.predict(X_filtered)

            # Weighted combination
            predictions = old_pred * old_weight + new_pred * new_weight

        else:
            # Use default base model prediction (equal weights)
            predictions = self.base_model.predict(X_filtered)

        return predictions

    def get_model_info(self) -> dict:
        """
        Get information about model configuration.

        Returns:
            Dict with model details
        """
        return {
            'model_type': 'China Market Predictor',
            'base_model': 'Hybrid Ensemble (LightGBM + XGBoost + LSTM + Hybrid LSTM/CNN)',
            'features': {
                'phase_1': 'Technical indicators (60 features)',
                'phase_2': 'Volatility features (37 features)',
                'phase_4_china': 'China macro features (10 features)',
                'regime': 'Regime detection features' if self.use_regime_detection else 'Disabled'
            },
            'macro_indicators': ['CSI300', 'SSEC', 'HSI', 'CNY', 'GLD'],
            'total_features': '~107 features (Phase 1+2+4 China)',
            'validation_weighting': {
                'enabled': self.use_validation_weighting,
                'old_model_weight': self.validation_performance['old_model_weight'],
                'new_model_weight': self.validation_performance['new_model_weight'],
                'old_model_accuracy': self.validation_performance['old_model_accuracy'],
                'new_model_accuracy': self.validation_performance['new_model_accuracy']
            },
            'optimization_for': 'Chinese markets (HK, Shanghai, Shenzhen)',
            'expected_performance': {
                'hong_kong': '~50% profitability',
                'a_shares': 'Poor performance - NOT RECOMMENDED'
            }
        }


if __name__ == "__main__":
    # Test the China market predictor
    import yfinance as yf

    print("="*60)
    print("CHINA MARKET PREDICTOR TEST")
    print("="*60)

    # Download sample data
    ticker = '0700.HK'  # Tencent
    print(f"\nTesting with {ticker}...")

    df = yf.download(ticker, start='2023-01-01', end='2025-11-22', progress=False)

    print(f"Downloaded {len(df)} days of data")

    # Initialize predictor
    predictor = ChinaMarketPredictor(
        use_regime_detection=True,
        use_validation_weighting=True
    )

    # Add features
    df = predictor.add_features(df, ticker=ticker)

    print(f"\nFeatures shape: {df.shape}")
    print(f"Feature columns: {len(df.columns)}")

    # Prepare train/test split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    # Create target
    target_col = (train_df['High'] - train_df['Low']) / train_df['Close']
    target_col = target_col.shift(-1).dropna()

    # Features
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    X_train = train_df[feature_cols].iloc[:-1]
    y_train = target_col

    X_test = test_df[feature_cols].iloc[:-1]
    y_test = ((test_df['High'] - test_df['Low']) / test_df['Close']).shift(-1).dropna()

    # Align lengths
    min_len = min(len(X_train), len(y_train))
    X_train = X_train.iloc[:min_len]
    y_train = y_train.iloc[:min_len]

    min_len = min(len(X_test), len(y_test))
    X_test = X_test.iloc[:min_len]
    y_test = y_test.iloc[:min_len]

    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    # Train model
    print("\nTraining China Market Predictor...")
    predictor.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    # Make predictions
    print("\nMaking predictions...")
    predictions = predictor.predict(X_test)

    # Evaluate
    actual_direction = (y_test > y_test.shift(1)).iloc[1:]
    pred_direction = (pd.Series(predictions, index=y_test.index) > pd.Series(predictions, index=y_test.index).shift(1)).iloc[1:]

    accuracy = (actual_direction == pred_direction).mean() * 100

    print(f"\nResults:")
    print(f"  Directional Accuracy: {accuracy:.1f}%")
    print(f"  MAE: {np.mean(np.abs(y_test.iloc[1:] - predictions[1:])):.6f}")

    # Show model info
    print("\nModel Info:")
    info = predictor.get_model_info()
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
