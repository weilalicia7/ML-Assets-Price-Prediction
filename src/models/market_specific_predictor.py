"""
WEEK 2 IMPROVEMENT: Market-Specific Predictor

Create separate models for each Chinese market:
- Hong Kong (HKEX): International, high liquidity, stable
- Shanghai (SSE): Mainland A-shares, blue-chip focus
- Shenzhen (SZSE): Growth/tech, high volatility

Key improvements:
1. Market-specific CatBoost hyperparameters
2. Market-specific feature importance
3. Market-specific confidence thresholds
4. Better handling of different market dynamics

Based on Week 1 results:
- HK: 2/10 passing (20%)
- SS: 2/10 passing (20%)
- SZ: 0/10 passing (0%) - needs special handling
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


class MarketSpecificPredictor:
    """
    Market-specific CatBoost predictor for Chinese stock markets.

    Tailored configurations for:
    - Hong Kong (HK): More stable, international investors
    - Shanghai (SS): Blue-chip, policy-sensitive
    - Shenzhen (SZ): High volatility, tech/growth focus
    """

    def __init__(self, market_type, confidence_threshold=0.55, verbose=False):
        """
        Initialize market-specific predictor.

        Args:
            market_type: 'HK', 'SS', or 'SZ'
            confidence_threshold: Minimum probability to predict 'up' (default: 0.55)
            verbose: Print training progress (default: False)
        """
        if market_type not in ['HK', 'SS', 'SZ']:
            raise ValueError("market_type must be 'HK', 'SS', or 'SZ'")

        self.market_type = market_type
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        self.model = self._create_market_model()
        self.feature_names = None

    def _create_market_model(self):
        """Create CatBoost with market-specific hyperparameters"""

        if self.market_type == 'HK':
            # Hong Kong: More stable, can use higher depth
            # International investors, longer-term trends
            return CatBoostClassifier(
                iterations=200,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=self.verbose,
                allow_writing_files=False,

                # Overfitting controls
                od_type='IncToDec',
                od_wait=50,
                use_best_model=True,
                bootstrap_type='Bernoulli'
            )

        elif self.market_type == 'SS':
            # Shanghai: Blue-chip, medium complexity
            # Policy-sensitive, medium volatility
            return CatBoostClassifier(
                iterations=200,
                learning_rate=0.06,
                depth=5,
                random_seed=42,
                verbose=self.verbose,
                allow_writing_files=False,

                # Overfitting controls
                od_type='IncToDec',
                od_wait=50,
                use_best_model=True,
                bootstrap_type='Bernoulli'
            )

        else:  # SZ
            # Shenzhen: High volatility, need MORE regularization
            # Growth/tech stocks, retail-driven, extreme moves
            return CatBoostClassifier(
                iterations=150,  # Fewer iterations (prevent overfitting)
                learning_rate=0.08,  # Higher LR (faster convergence)
                depth=4,  # Shallow trees (prevent overfitting)
                l2_leaf_reg=5,  # MORE regularization (critical for high volatility)
                random_seed=42,
                verbose=self.verbose,
                allow_writing_files=False,

                # Overfitting controls (more aggressive)
                od_type='IncToDec',
                od_wait=30,  # Earlier stopping (vs 50 for HK/SS)
                use_best_model=True,
                bootstrap_type='Bernoulli'
            )

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
        Add features with FIXED windows.
        Using 15 BASE features only (no interaction/lag/trend features).

        Week 1 results showed:
        - 15 features: 0.25% avg return (BASELINE)
        - 30 features: 0.21% avg return (WORSE - overfitting)

        Solution: Revert to 15 base features.
        """
        df = df.copy()

        # Price features (FIXED windows)
        df['returns_1d'] = df['Close'].pct_change()
        df['returns_5d'] = df['Close'].pct_change(5)
        df['rsi_14'] = self._compute_rsi(df['Close'], 14)
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['price_momentum_10'] = df['Close'].pct_change(10)

        # Volume features (FIXED windows)
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(14).mean()
        df['volume_ma_14'] = df['Volume'].rolling(14).mean()
        df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['vwap_ratio'] = df['Close'] / df['vwap']
        df['volume_momentum_10'] = df['Volume'].pct_change(10)

        # Volatility features (FIXED windows)
        df['volatility_14'] = df['Close'].pct_change().rolling(14).std()
        df['atr_14'] = self._compute_atr(df, 14)
        df['bollinger_width'] = self._compute_bollinger_width(df, 20)

        # Position features
        df['close_to_high'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
        df['close_to_sma'] = (df['Close'] - df['sma_20']) / (df['sma_20'] + 1e-8)

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

        # Feature columns (15 base features only)
        feature_cols = [
            'returns_1d', 'returns_5d', 'rsi_14', 'sma_20', 'sma_50',
            'price_momentum_10', 'volume_ratio', 'volume_ma_14',
            'vwap_ratio', 'volume_momentum_10', 'volatility_14',
            'atr_14', 'bollinger_width', 'close_to_high', 'close_to_sma'
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

        # 80/20 train/validation split for overfitting detection
        train_size = int(len(X) * 0.8)
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_val = X.iloc[train_size:]
        y_val = y.iloc[train_size:]

        # Train CatBoost with validation set
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
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

        # Use SAME feature names as training
        X = df[self.feature_names]

        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        return predictions, probabilities, df

    def predict_proba(self, X):
        """
        Predict probabilities on feature array (for ensemble use).

        Args:
            X: Feature array or DataFrame (n_samples, n_features)

        Returns:
            proba: Probabilities (n_samples, 2) - [prob_down, prob_up]
        """
        if self.feature_names is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Predict probabilities using CatBoost
        proba = self.model.predict_proba(X)

        return proba

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

        # Apply confidence threshold
        high_confidence_up = (probabilities >= self.confidence_threshold)

        # Calculate metrics
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
