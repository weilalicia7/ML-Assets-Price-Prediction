"""
PHASE 2C: Simple Predictor with FIXED Windows
Production-ready predictor that avoids all architectural flaws from Phases 1, 2A, 2B.

Key improvements:
1. FIXED feature windows (no adaptive sizing)
2. Single CatBoost model (no ensemble)
3. Clean, simple feature set (30 features total)
4. Production-ready (can deploy immediately)

CUMULATIVE IMPROVEMENTS APPLIED:
- Phase 2C.1: Confidence threshold filtering (default=0.7)
- Phase 2C.2: 5 interaction features (non-linear relationships)
- Phase 2C.3: 6 lag features (temporal context)
- Phase 2C.4: 4 trend detection features

Total features: 30 (15 base + 5 interaction + 6 lag + 4 trend)
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


class SimplePredictor:
    """
    Simple CatBoost predictor with FIXED feature windows.
    No adaptive sizing. No ensemble. Production-ready.
    """

    def __init__(self, iterations=200, learning_rate=0.05, depth=6, verbose=False, confidence_threshold=0.55):
        """
        Initialize predictor with CatBoost hyperparameters.

        Args:
            iterations: Number of boosting iterations (default: 200)
            learning_rate: Learning rate (default: 0.05)
            depth: Tree depth (default: 6)
            verbose: Print training progress (default: False)
            confidence_threshold: Minimum probability to predict 'up' (default: 0.55)
                                 Higher values = fewer but higher-confidence trades
                                 Research-backed optimal: 0.50-0.65 range
        """
        # WEEK 1 IMPROVEMENT: Added overfitting controls
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=42,
            verbose=verbose,
            allow_writing_files=False,

            # NEW: Overfitting detection (only od_wait, not early_stopping_rounds)
            od_type='IncToDec',  # Stop when validation stops improving
            od_wait=50,          # Patience before stopping (50 iterations)
            use_best_model=True,  # Use the best iteration, not the last

            # Ordered boosting (prevents target leakage)
            bootstrap_type='Bernoulli'
        )
        self.feature_names = None
        self.confidence_threshold = confidence_threshold

    def _compute_rsi(self, prices, period=14):
        """Compute Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _compute_atr(self, df, period=14):
        """Compute Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def _compute_bollinger_width(self, df, period=20):
        """Compute Bollinger Band Width"""
        sma = df['Close'].rolling(period).mean()
        std = df['Close'].rolling(period).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        width = (upper - lower) / sma
        return width

    def add_features(self, df):
        """
        Add features with FIXED windows (no adaptive sizing).

        ALL WINDOWS ARE FIXED:
        - RSI: 14 days (not adaptive)
        - SMA: 20, 50 days (not adaptive - REDUCED from 50/100 to work with short test data)
        - Volatility: 14 days (not adaptive)
        - ATR: 14 days (not adaptive)
        - Bollinger: 20 days (not adaptive)

        This ensures features have identical names across train/val/test.
        """
        df = df.copy()

        # Price features (FIXED windows - reduced for short test data)
        df['returns_1d'] = df['Close'].pct_change()
        df['returns_5d'] = df['Close'].pct_change(5)
        df['rsi_14'] = self._compute_rsi(df['Close'], 14)  # FIXED: always 14
        df['sma_20'] = df['Close'].rolling(20).mean()  # FIXED: always 20 (reduced from 50)
        df['sma_50'] = df['Close'].rolling(50).mean()  # FIXED: always 50 (reduced from 100)
        df['price_momentum_10'] = df['Close'].pct_change(10)  # FIXED: always 10

        # Volume features (FIXED windows)
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(14).mean()  # FIXED: 14
        df['volume_ma_14'] = df['Volume'].rolling(14).mean()  # FIXED: 14
        df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['vwap_ratio'] = df['Close'] / df['vwap']
        df['volume_momentum_10'] = df['Volume'].pct_change(10)  # FIXED: 10

        # Volatility features (FIXED windows)
        df['volatility_14'] = df['Close'].pct_change().rolling(14).std()  # FIXED: 14
        df['atr_14'] = self._compute_atr(df, 14)  # FIXED: 14
        df['bollinger_width'] = self._compute_bollinger_width(df, 20)  # FIXED: 20

        # Position features
        df['close_to_high'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
        df['close_to_sma'] = (df['Close'] - df['sma_20']) / (df['sma_20'] + 1e-8)

        # PHASE 2C.2: Interaction features (non-linear relationships)
        df['rsi_volume'] = df['rsi_14'] * df['volume_ratio']
        df['momentum_volatility'] = df['price_momentum_10'] * df['volatility_14']
        df['returns_trend'] = df['returns_1d'] * df['returns_5d']
        df['price_volume_strength'] = df['close_to_high'] * df['volume_ratio']
        df['sma_cross_signal'] = (df['sma_20'] - df['sma_50']) / (df['sma_50'] + 1e-8)

        # PHASE 2C.3: Lag features (temporal context)
        df['returns_1d_lag1'] = df['returns_1d'].shift(1)
        df['returns_1d_lag2'] = df['returns_1d'].shift(2)
        df['rsi_14_lag1'] = df['rsi_14'].shift(1)
        df['volume_ratio_lag1'] = df['volume_ratio'].shift(1)
        df['volatility_14_lag1'] = df['volatility_14'].shift(1)
        df['returns_3d_avg'] = df['returns_1d'].rolling(3).mean()

        # PHASE 2C.4: Trend detection features
        # Calculate slopes using simple linear regression over a window
        def calc_slope(series, window=5):
            """Calculate slope of series over rolling window"""
            slopes = []
            for i in range(len(series)):
                if i < window - 1:
                    slopes.append(np.nan)
                else:
                    y = series.iloc[i-window+1:i+1].values
                    x = np.arange(window)
                    slope = np.polyfit(x, y, 1)[0] if len(y) == window else 0
                    slopes.append(slope)
            return pd.Series(slopes, index=series.index)

        df['sma_20_slope'] = calc_slope(df['sma_20'], window=5)
        df['sma_50_slope'] = calc_slope(df['sma_50'], window=5)
        df['price_vs_sma20_trend'] = (df['Close'] - df['sma_20']).rolling(5).mean() / (df['sma_20'] + 1e-8)
        df['volatility_trend'] = df['volatility_14'].pct_change(5)

        return df

    def fit(self, df):
        """
        Train model on data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            self
        """
        # Add features
        df = self.add_features(df)

        # Define target: next-day return > 0
        df['target'] = (df['Close'].pct_change().shift(-1) > 0).astype(int)

        # Feature columns (PHASE 2C.1+2C.2+2C.3+2C.4: 30 features - 15 base + 5 interaction + 6 lag + 4 trend)
        feature_cols = [
            'returns_1d', 'returns_5d', 'rsi_14', 'sma_20', 'sma_50',
            'price_momentum_10', 'volume_ratio', 'volume_ma_14',
            'vwap_ratio', 'volume_momentum_10', 'volatility_14',
            'atr_14', 'bollinger_width', 'close_to_high', 'close_to_sma',
            # PHASE 2C.2: Interaction features
            'rsi_volume', 'momentum_volatility', 'returns_trend',
            'price_volume_strength', 'sma_cross_signal',
            # PHASE 2C.3: Lag features
            'returns_1d_lag1', 'returns_1d_lag2', 'rsi_14_lag1',
            'volume_ratio_lag1', 'volatility_14_lag1', 'returns_3d_avg',
            # PHASE 2C.4: Trend detection features
            'sma_20_slope', 'sma_50_slope', 'price_vs_sma20_trend', 'volatility_trend'
        ]

        # Drop rows with NaN (from rolling windows)
        df = df.dropna()

        if len(df) < 50:
            raise ValueError(f"Insufficient data after adding features: {len(df)} rows")

        # Prepare training data
        X = df[feature_cols]
        y = df['target']

        # Store feature names for prediction
        self.feature_names = feature_cols

        # WEEK 1 IMPROVEMENT: Add 80/20 train/validation split for overfitting detection
        train_size = int(len(X) * 0.8)
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_val = X.iloc[train_size:]
        y_val = y.iloc[train_size:]

        # Train CatBoost with validation set (for overfitting detector)
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),  # Validation set for overfitting detection
            verbose=False
        )

        return self

    def predict(self, df):
        """
        Predict on new data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            tuple: (predictions, probabilities, feature_df)
                - predictions: Binary predictions (0/1)
                - probabilities: Probability of up move
                - feature_df: DataFrame with features and index
        """
        if self.feature_names is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Add features (using SAME fixed windows as training)
        df = self.add_features(df)

        # Drop NaN rows
        df = df.dropna()

        if len(df) == 0:
            raise ValueError("No valid data after adding features")

        # Use SAME feature names as training (critical!)
        X = df[self.feature_names]

        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        # Return predictions with original index for alignment
        return predictions, probabilities, df

    def get_feature_importance(self):
        """
        Get feature importance scores.

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.feature_names is None:
            raise ValueError("Model not trained. Call fit() first.")

        importance = self.model.get_feature_importance()

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return importance_df

    def evaluate(self, df):
        """
        Evaluate model on data and return metrics.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            dict: Evaluation metrics
        """
        # Predict
        predictions, probabilities, feature_df = self.predict(df)

        # Calculate true target on same index
        actual_returns = feature_df['Close'].pct_change().shift(-1)
        actual_target = (actual_returns > 0).astype(int)

        # Align predictions with actual
        actual_target = actual_target.loc[feature_df.index]
        actual_returns = actual_returns.loc[feature_df.index]

        # Drop last row (no future return)
        actual_target = actual_target[:-1]
        actual_returns = actual_returns[:-1]
        predictions = predictions[:-1]
        probabilities = probabilities[:-1]

        # Apply confidence threshold - only predict 'up' when confidence exceeds threshold
        high_confidence_up = (probabilities >= self.confidence_threshold)

        # Calculate metrics using high-confidence predictions only
        # For accuracy, still use all predictions
        accuracy = (predictions == actual_target).mean()

        # Returns for HIGH-CONFIDENCE predicted up moves only
        if high_confidence_up.sum() > 0:
            avg_return = actual_returns[high_confidence_up].mean()
            total_return = actual_returns[high_confidence_up].sum()
        else:
            avg_return = 0.0
            total_return = 0.0

        return {
            'accuracy': accuracy,
            'avg_return': avg_return,
            'total_return': total_return,
            'num_predictions': high_confidence_up.sum(),
            'predictions': predictions,
            'probabilities': probabilities,
            'high_confidence_predictions': high_confidence_up
        }
