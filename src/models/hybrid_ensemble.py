"""
Hybrid Ensemble: Combines Old Model + New Hybrid LSTM/CNN (ENHANCEMENT #6)

This ensemble combines the strengths of both:
- Old Model (EnhancedEnsemblePredictor): LGBM, XGB, TCN, LSTM, Transformer
- New Model (HybridLSTMCNNPredictor): Hybrid LSTM/CNN with profit-maximizing loss

Strategy: Weighted average based on validation performance

INTEGRATED FIXES (from us model fixing1.pdf and us model fixing 2.pdf):
- Fixes 1-7: SELL thresholds, position sizing, stop-losses, blacklists
- Fixes 8-15: Extended blocklists, Kelly criterion, win rate sizing, profit-taking

# ============================================================================
# PROTECTED CORE MODEL - DO NOT MODIFY WITHOUT USER PERMISSION
# This file contains the main HybridEnsemblePredictor used by the US/Intl model.
# Any changes to model architecture, weights, or logic require explicit user approval.
# ============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from .enhanced_ensemble import EnhancedEnsemblePredictor
    from .hybrid_lstm_cnn import HybridLSTMCNNPredictor
    from .us_intl_optimizer import USIntlModelOptimizer, create_optimizer, SignalOptimization
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from enhanced_ensemble import EnhancedEnsemblePredictor
    from hybrid_lstm_cnn import HybridLSTMCNNPredictor
    from us_intl_optimizer import USIntlModelOptimizer, create_optimizer, SignalOptimization


class HybridEnsemblePredictor:
    """
    ENHANCEMENT #6: Ensemble combining old and new models

    Combines:
    1. Old EnhancedEnsemblePredictor (LGBM, XGB, TCN, LSTM, Transformer)
    2. New HybridLSTMCNNPredictor (with profit-maximizing loss)

    Weighting strategy:
    - Validation performance determines weights
    - Better performing model gets higher weight

    INTEGRATED SIGNAL OPTIMIZER (Fixes 1-15):
    - Automatic signal validation and optimization
    - Asset-class specific thresholds
    - Position sizing and stop-loss management
    """

    def __init__(
        self,
        old_model_weight: Optional[float] = None,
        hybrid_epochs: int = 100,
        hybrid_lookback: int = 20,
        hybrid_cnn_channels: list = None,
        hybrid_kernel_sizes: list = None,
        hybrid_lstm_hidden_size: int = 64,
        hybrid_lstm_num_layers: int = 2,
        hybrid_dropout: float = 0.3,
        hybrid_learning_rate: float = 0.001,
        hybrid_batch_size: int = 32,
        use_profit_loss: bool = True,
        random_state: int = 42,
        enable_signal_optimizer: bool = True,
        enable_kelly_criterion: bool = True,
        enable_dynamic_sizing: bool = True,
    ):
        """
        Initialize hybrid ensemble.

        Args:
            old_model_weight: Weight for old model (0-1). If None, auto-determined from validation
            hybrid_epochs: Number of epochs for hybrid model training
            hybrid_lookback: Lookback window for hybrid LSTM/CNN model
            hybrid_cnn_channels: CNN channel sizes
            hybrid_kernel_sizes: Kernel sizes for CNN branches
            hybrid_lstm_hidden_size: LSTM hidden layer size
            hybrid_lstm_num_layers: Number of LSTM layers
            hybrid_dropout: Dropout rate
            hybrid_learning_rate: Learning rate
            hybrid_batch_size: Batch size for training
            use_profit_loss: Use profit-maximizing loss for hybrid model
            random_state: Random seed
            enable_signal_optimizer: Enable US/Intl signal optimizer (Fixes 1-15)
            enable_kelly_criterion: Enable Kelly Criterion position sizing (Fix 11)
            enable_dynamic_sizing: Enable win-rate based position sizing (Fix 9)
        """
        self.old_model_weight = old_model_weight
        self.hybrid_epochs = hybrid_epochs
        self.hybrid_lookback = hybrid_lookback
        self.hybrid_cnn_channels = hybrid_cnn_channels or [32, 64, 32]
        self.hybrid_kernel_sizes = hybrid_kernel_sizes or [3, 5, 7]
        self.hybrid_lstm_hidden_size = hybrid_lstm_hidden_size
        self.hybrid_lstm_num_layers = hybrid_lstm_num_layers
        self.hybrid_dropout = hybrid_dropout
        self.hybrid_learning_rate = hybrid_learning_rate
        self.hybrid_batch_size = hybrid_batch_size
        self.use_profit_loss = use_profit_loss
        self.random_state = random_state

        # Models
        self.old_model = None
        self.hybrid_model = None

        # Auto-determined weights
        self.final_old_weight = 0.5
        self.final_hybrid_weight = 0.5

        # Signal optimizer (Fixes 1-15)
        self.enable_signal_optimizer = enable_signal_optimizer
        if enable_signal_optimizer:
            self.signal_optimizer = create_optimizer(
                enable_kelly=enable_kelly_criterion,
                enable_dynamic_sizing=enable_dynamic_sizing,
            )
        else:
            self.signal_optimizer = None

        # Track historical win rates for dynamic sizing
        self.historical_win_rates: Dict[str, float] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ):
        """
        Train both models and determine ensemble weights.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        print("\n" + "="*60)
        print("HYBRID ENSEMBLE TRAINING")
        print("="*60)

        # Convert to numpy for hybrid model
        # BUGFIX: Explicitly convert to float32 to handle mixed dtype columns (int32, int64, bool, float64)
        # This prevents "can't convert np.ndarray of type numpy.object_" error in PyTorch
        if isinstance(X_train, pd.DataFrame):
            X_train_np = X_train.values.astype(np.float32)
        else:
            X_train_np = np.asarray(X_train, dtype=np.float32)

        if isinstance(y_train, pd.Series):
            y_train_np = y_train.values.astype(np.float32)
        else:
            y_train_np = np.asarray(y_train, dtype=np.float32)

        if X_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val_np = X_val.values.astype(np.float32)
            else:
                X_val_np = np.asarray(X_val, dtype=np.float32)

            if isinstance(y_val, pd.Series):
                y_val_np = y_val.values.astype(np.float32)
            else:
                y_val_np = np.asarray(y_val, dtype=np.float32)
        else:
            X_val_np = None
            y_val_np = None

        # Train old model (Enhanced Ensemble)
        print("\n[1/2] Training Old Model (Enhanced Ensemble)...")
        self.old_model = EnhancedEnsemblePredictor(use_prediction_market=True)

        self.old_model.train_all_models(
            X_train, y_train, X_val, y_val,
            models_to_train=['lightgbm', 'xgboost'],
            neural_models=['lstm']  # Lighter neural models for speed
        )

        # Train hybrid model (Hybrid LSTM/CNN)
        print("\n[2/2] Training Hybrid LSTM/CNN Model...")
        self.hybrid_model = HybridLSTMCNNPredictor(
            lookback=self.hybrid_lookback,
            cnn_channels=self.hybrid_cnn_channels,
            kernel_sizes=self.hybrid_kernel_sizes,
            lstm_hidden_size=self.hybrid_lstm_hidden_size,
            lstm_num_layers=self.hybrid_lstm_num_layers,
            dropout=self.hybrid_dropout,
            learning_rate=self.hybrid_learning_rate,
            batch_size=self.hybrid_batch_size,
            epochs=self.hybrid_epochs,
            device='cpu',
            use_profit_loss=self.use_profit_loss,
            random_state=self.random_state
        )

        self.hybrid_model.fit(X_train_np, y_train_np, X_val_np, y_val_np)

        # Determine weights from validation performance
        if X_val is not None and self.old_model_weight is None:
            print("\n[INFO] Determining ensemble weights from validation performance...")
            self._determine_weights(X_val, y_val, X_val_np, y_val_np)
        elif self.old_model_weight is not None:
            self.final_old_weight = self.old_model_weight
            self.final_hybrid_weight = 1.0 - self.old_model_weight
            print(f"\n[INFO] Using fixed weights:")
            print(f"       Old Model: {self.final_old_weight:.2f}")
            print(f"       Hybrid Model: {self.final_hybrid_weight:.2f}")
        else:
            # No validation data, use 50/50
            self.final_old_weight = 0.5
            self.final_hybrid_weight = 0.5
            print(f"\n[INFO] No validation data. Using equal weights (50/50)")

        print(f"\n[OK] Hybrid Ensemble Training Complete!")

    def _determine_weights(self, X_val_df, y_val_df, X_val_np, y_val_np):
        """Determine ensemble weights based on validation performance."""

        # Get predictions from both models
        old_pred = self.old_model.predict(X_val_df)
        hybrid_pred = self.hybrid_model.predict(X_val_np)

        # Both models already account for lookback
        # Align y_val with predictions
        lookback = self.hybrid_model.lookback
        y_val_aligned = y_val_np[lookback:]

        min_len = min(len(hybrid_pred), len(old_pred), len(y_val_aligned))
        hybrid_pred = hybrid_pred[:min_len]
        old_pred = old_pred[:min_len]
        y_val_aligned = y_val_aligned[:min_len]

        # Calculate directional accuracy for each model
        old_direction = np.sign(old_pred)
        hybrid_direction = np.sign(hybrid_pred)
        actual_direction = np.sign(y_val_aligned)

        old_accuracy = (old_direction == actual_direction).mean()
        hybrid_accuracy = (hybrid_direction == actual_direction).mean()

        # Calculate MSE for each model
        old_mse = np.mean((old_pred - y_val_aligned) ** 2)
        hybrid_mse = np.mean((hybrid_pred - y_val_aligned) ** 2)

        # Weight based on directional accuracy (more important for trading)
        # Inverse weighting: better accuracy = higher weight
        if old_accuracy + hybrid_accuracy > 0:
            self.final_old_weight = old_accuracy / (old_accuracy + hybrid_accuracy)
            self.final_hybrid_weight = hybrid_accuracy / (old_accuracy + hybrid_accuracy)
        else:
            self.final_old_weight = 0.5
            self.final_hybrid_weight = 0.5

        print(f"\n  Validation Performance:")
        print(f"    Old Model - Accuracy: {old_accuracy*100:.1f}%, MSE: {old_mse:.6f}")
        print(f"    Hybrid Model - Accuracy: {hybrid_accuracy*100:.1f}%, MSE: {hybrid_mse:.6f}")
        print(f"\n  Final Ensemble Weights:")
        print(f"    Old Model: {self.final_old_weight:.2f}")
        print(f"    Hybrid Model: {self.final_hybrid_weight:.2f}")

    def predict(self, X) -> np.ndarray:
        """
        Make predictions using weighted ensemble.

        Args:
            X: Features (DataFrame or numpy array)

        Returns:
            Ensemble predictions
        """
        if self.old_model is None or self.hybrid_model is None:
            raise ValueError("Models not trained yet! Call fit() first.")

        # Convert if needed
        # BUGFIX: Explicitly convert to float32 for PyTorch compatibility
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if isinstance(X, pd.DataFrame):
            X_np = X.values.astype(np.float32)
        else:
            X_np = np.asarray(X, dtype=np.float32)

        # Get predictions from both models
        old_pred = self.old_model.predict(X_df)
        hybrid_pred = self.hybrid_model.predict(X_np)

        # Both models already account for lookback internally
        # (EnhancedEnsemble uses LSTM which reduces output length)
        # Just take the minimum length
        min_len = min(len(hybrid_pred), len(old_pred))

        hybrid_pred = hybrid_pred[:min_len]
        old_pred_aligned = old_pred[:min_len]

        # Weighted ensemble
        ensemble_pred = (
            self.final_old_weight * old_pred_aligned +
            self.final_hybrid_weight * hybrid_pred
        )

        return ensemble_pred

    # ========== SIGNAL OPTIMIZATION METHODS (Fixes 1-15) ==========

    def optimize_signal(
        self,
        ticker: str,
        signal_type: str,
        confidence: float,
        volatility: float = 0.20,
        momentum: float = 0.0,
        win_rate: Optional[float] = None,
    ) -> Optional[SignalOptimization]:
        """
        Optimize a trading signal using the integrated US/Intl optimizer.

        Applies all 15 fixes from 'us model fixing1.pdf' and 'us model fixing 2.pdf'.

        Args:
            ticker: Stock/asset ticker
            signal_type: 'BUY' or 'SELL'
            confidence: Model confidence (0-1)
            volatility: Asset volatility (annualized)
            momentum: Recent momentum (e.g., 20-day return)
            win_rate: Optional historical win rate for this ticker

        Returns:
            SignalOptimization with position sizing, stop-loss, etc.
            Returns None if optimizer is disabled.
        """
        if not self.enable_signal_optimizer or self.signal_optimizer is None:
            return None

        # Use tracked win rate if not provided
        if win_rate is None:
            win_rate = self.historical_win_rates.get(ticker, 0.50)

        return self.signal_optimizer.optimize_signal(
            ticker=ticker,
            signal_type=signal_type,
            confidence=confidence,
            volatility=volatility,
            momentum=momentum,
            win_rate=win_rate,
        )

    def optimize_signals_batch(
        self,
        signals: List[Dict[str, Any]],
    ) -> Tuple[List[SignalOptimization], Dict[str, Any]]:
        """
        Optimize a batch of signals.

        Args:
            signals: List of dicts with keys: ticker, signal_type, confidence, etc.

        Returns:
            (optimized_signals, summary_stats)
        """
        if not self.enable_signal_optimizer or self.signal_optimizer is None:
            return [], {'optimizer_disabled': True}

        return self.signal_optimizer.process_signals_batch(signals)

    def update_win_rate(self, ticker: str, won: bool):
        """
        Update historical win rate for a ticker after trade completion.

        Args:
            ticker: Stock ticker
            won: Whether the trade was profitable
        """
        current_rate = self.historical_win_rates.get(ticker, 0.50)
        # Exponential moving average update (alpha=0.1)
        alpha = 0.1
        new_rate = alpha * (1.0 if won else 0.0) + (1 - alpha) * current_rate
        self.historical_win_rates[ticker] = new_rate

        # Update optimizer's win rates too
        if self.signal_optimizer:
            self.signal_optimizer.historical_win_rates[ticker] = new_rate

    def get_optimizer_config(self) -> Optional[Dict[str, Any]]:
        """Get the signal optimizer configuration summary."""
        if not self.enable_signal_optimizer or self.signal_optimizer is None:
            return None
        return self.signal_optimizer.get_configuration_summary()

    def is_signal_blocked(
        self,
        ticker: str,
        signal_type: str,
        confidence: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Quick check if a signal would be blocked by the optimizer.

        Args:
            ticker: Stock ticker
            signal_type: 'BUY' or 'SELL'
            confidence: Model confidence

        Returns:
            (is_blocked, block_reason)
        """
        if not self.enable_signal_optimizer:
            return False, None

        result = self.optimize_signal(ticker, signal_type, confidence)
        if result and result.blocked:
            return True, result.block_reason
        return False, None


def test_hybrid_ensemble():
    """Test hybrid ensemble."""
    print("="*60)
    print("HYBRID ENSEMBLE TEST")
    print("="*60)

    # Generate dummy data
    np.random.seed(42)
    n_samples = 500
    n_features = 50

    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples) * 0.02

    # Split
    split = int(n_samples * 0.7)
    val_split = int(n_samples * 0.85)

    X_train, X_val, X_test = X[:split], X[split:val_split], X[val_split:]
    y_train, y_val, y_test = y[:split], y[split:val_split], y[val_split:]

    # Convert to DataFrame
    X_train_df = pd.DataFrame(X_train)
    X_val_df = pd.DataFrame(X_val)
    X_test_df = pd.DataFrame(X_test)

    y_train_series = pd.Series(y_train)
    y_val_series = pd.Series(y_val)

    # Create and train ensemble
    print("\n[INFO] Creating Hybrid Ensemble...")
    ensemble = HybridEnsemblePredictor(
        hybrid_epochs=20,  # Reduced for testing
        use_profit_loss=True
    )

    print("\n[INFO] Training ensemble...")
    ensemble.fit(X_train_df, y_train_series, X_val_df, y_val_series)

    # Make predictions
    print("\n[INFO] Making predictions...")
    predictions = ensemble.predict(X_test_df)

    print(f"\n[OK] Generated {len(predictions)} predictions")
    print(f"     Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")

    # Calculate accuracy
    lookback = ensemble.hybrid_model.lookback
    y_test_aligned = y_test[lookback:]
    min_len = min(len(predictions), len(y_test_aligned))
    predictions = predictions[:min_len]
    y_test_aligned = y_test_aligned[:min_len]

    accuracy = (np.sign(predictions) == np.sign(y_test_aligned)).mean() * 100
    print(f"     Directional Accuracy: {accuracy:.1f}%")

    print(f"\n[SUCCESS] Hybrid Ensemble test complete!")


if __name__ == "__main__":
    test_hybrid_ensemble()
