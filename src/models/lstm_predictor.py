"""
WEEK 7 IMPROVEMENT: LSTM Sequential Pattern Recognition

LSTM component for ensemble model to capture:
- Sequential patterns and trends
- Temporal dependencies
- Market momentum shifts

This will be combined with CatBoost (70%) for an ensemble predictor (30% LSTM).

# ============================================================================
# PROTECTED CORE MODEL - DO NOT MODIFY WITHOUT USER PERMISSION
# This file contains the LSTM predictor component.
# Any changes to model architecture or logic require explicit user approval.
# ============================================================================
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("WARNING: TensorFlow not installed. LSTM predictor will not be available.")


class LSTMPredictor:
    """
    LSTM-based predictor for sequential pattern recognition in stock prices.

    Designed to capture:
    - Sequential dependencies (last 20 days)
    - Momentum trends
    - Price pattern recognition
    """

    def __init__(self, lookback_period=20, verbose=False):
        """
        Initialize LSTM predictor.

        Args:
            lookback_period: Number of days to look back for sequence (default: 20)
            verbose: Print training progress (default: False)
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM predictor. Install with: pip install tensorflow")

        self.lookback_period = lookback_period
        self.verbose = verbose
        self.model = None
        self.feature_names = None

    def _create_sequences(self, X, y=None):
        """
        Create sequences for LSTM input.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,) - optional

        Returns:
            X_seq: Sequences (n_samples - lookback, lookback, n_features)
            y_seq: Labels (n_samples - lookback,) if y provided
        """
        X_seq = []
        y_seq = [] if y is not None else None

        for i in range(self.lookback_period, len(X)):
            X_seq.append(X[i - self.lookback_period:i])
            if y is not None:
                y_seq.append(y[i])

        X_seq = np.array(X_seq)
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        return X_seq

    def _create_model(self, n_features):
        """
        Create LSTM model architecture.

        Architecture:
        - LSTM layer (64 units) with dropout
        - LSTM layer (32 units) with dropout
        - Dense layer (16 units) with relu
        - Output layer (sigmoid for binary classification)
        """
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True,
                       input_shape=(self.lookback_period, n_features)),
            layers.Dropout(0.2),

            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),

            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),

            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, X, y):
        """
        Train LSTM model.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,) - 0 (down) or 1 (up)
        """
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            X = np.array(X)

        y = np.array(y)

        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y)

        if len(X_seq) < 50:
            raise ValueError(f"Insufficient data for LSTM training. Need at least {self.lookback_period + 50} samples, got {len(X)}")

        # Create model
        n_features = X_seq.shape[2]
        self.model = self._create_model(n_features)

        # Train with early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Split for validation
        val_split = 0.2
        val_size = int(len(X_seq) * val_split)
        X_train = X_seq[:-val_size]
        y_train = y_seq[:-val_size]
        X_val = X_seq[-val_size:]
        y_val = y_seq[-val_size:]

        if self.verbose:
            print(f"\nLSTM Training:")
            print(f"  Training samples: {len(X_train)}")
            print(f"  Validation samples: {len(X_val)}")
            print(f"  Sequence length: {self.lookback_period}")
            print(f"  Features per timestep: {n_features}")

        # Train
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1 if self.verbose else 0
        )

    def predict_proba(self, X):
        """
        Predict probabilities.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            proba: Probabilities (n_samples, 2) - [prob_down, prob_up]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            X = np.array(X)

        # Create sequences
        X_seq = self._create_sequences(X)

        # Predict
        prob_up = self.model.predict(X_seq, verbose=0).flatten()
        prob_down = 1 - prob_up

        # Stack to match CatBoost format (n_samples, 2)
        proba = np.column_stack([prob_down, prob_up])

        # Pad beginning with neutral probabilities (0.5, 0.5)
        # for samples before lookback period
        if len(proba) < len(X):
            pad_size = len(X) - len(proba)
            pad = np.full((pad_size, 2), 0.5)
            proba = np.vstack([pad, proba])

        return proba

    def predict(self, X, threshold=0.5):
        """
        Predict class labels.

        Args:
            X: Features (n_samples, n_features)
            threshold: Probability threshold for 'up' prediction (default: 0.5)

        Returns:
            predictions: Class labels (n_samples,) - 0 (down) or 1 (up)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
