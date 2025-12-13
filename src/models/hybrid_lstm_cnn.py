"""
Hybrid LSTM/CNN Model with Profit-Maximizing Loss
Based on 2025 research showing 96% directional accuracy (PDF research)

Key improvements over standard LSTM:
1. CNN layers capture spatial patterns from price charts
2. LSTM layers capture temporal dependencies
3. Custom profit-maximizing loss (Sharpe ratio optimization)
4. Designed for trading profitability, not just accuracy

Reference: PDF "search the most high accuracy and profit margin machine learning model"
- Hybrid LSTM/CNN: Up to 96% directional accuracy
- Transformer with custom loss: 48-51% annual returns
- Used by: Two Sigma, WorldQuant, Man AHL
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not available. Hybrid LSTM/CNN will not work.")
    print("      Install with: pip install torch")


class ProfitMaximizingLoss(nn.Module):
    """
    Custom loss function that maximizes trading profit instead of minimizing MSE.

    Based on transformer research showing 48-51% annual returns when using
    custom loss functions for profit maximization (PDF page 1).

    This loss combines:
    1. Sharpe Ratio maximization (risk-adjusted returns)
    2. Directional accuracy penalty
    3. Return magnitude weighting
    """

    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        """
        Args:
            alpha: Weight for Sharpe ratio component (default: 0.5)
            beta: Weight for directional accuracy (default: 0.3)
            gamma: Weight for return magnitude (default: 0.2)
        """
        super().__init__()
        self.alpha = alpha  # Sharpe ratio weight
        self.beta = beta    # Directional accuracy weight
        self.gamma = gamma  # Return magnitude weight

    def forward(self, predictions, actual_returns):
        """
        Calculate profit-maximizing loss.

        Args:
            predictions: Predicted returns (batch_size,)
            actual_returns: Actual returns (batch_size,)

        Returns:
            Combined loss (lower is better)
        """
        # 1. Sharpe Ratio Loss (negative because we minimize loss)
        # Trading returns = predicted direction * actual returns
        trading_returns = torch.sign(predictions) * actual_returns
        sharpe_loss = -self._sharpe_ratio(trading_returns)

        # 2. Directional Accuracy Loss
        # Penalize wrong direction predictions
        correct_direction = (torch.sign(predictions) == torch.sign(actual_returns)).float()
        direction_loss = 1.0 - correct_direction.mean()

        # 3. Return Magnitude Loss (MSE for calibration)
        # Still want accurate magnitude predictions
        magnitude_loss = nn.MSELoss()(predictions, actual_returns)

        # Combined loss
        total_loss = (
            self.alpha * sharpe_loss +
            self.beta * direction_loss +
            self.gamma * magnitude_loss
        )

        return total_loss

    def _sharpe_ratio(self, returns):
        """Calculate Sharpe ratio (annualized, assuming daily returns)"""
        mean_return = returns.mean()
        std_return = returns.std() + 1e-8  # Avoid division by zero

        # Annualize (252 trading days)
        sharpe = (mean_return / std_return) * np.sqrt(252)
        return sharpe


class HybridLSTMCNN(nn.Module):
    """
    Hybrid LSTM/CNN Model

    Architecture:
    1. CNN layers: Extract spatial patterns from price sequences
       - Multiple conv1d layers with different kernel sizes
       - Captures short-term, medium-term patterns

    2. LSTM layers: Capture long-term temporal dependencies
       - Bidirectional for better context
       - Dropout for regularization

    3. Attention mechanism: Focus on important features

    4. Fully connected layers: Final prediction

    Based on research achieving 96% directional accuracy (PDF page 1)
    """

    def __init__(
        self,
        input_size: int,
        cnn_channels: List[int] = [64, 128, 64],
        kernel_sizes: List[int] = [3, 5, 7],
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        """
        Args:
            input_size: Number of input features
            cnn_channels: Number of channels in CNN layers
            kernel_sizes: Kernel sizes for parallel CNN branches
            lstm_hidden_size: Hidden size for LSTM
            lstm_num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()

        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.bidirectional = bidirectional

        # === CNN Branch: Spatial Pattern Recognition ===
        # Multiple parallel CNN branches with different receptive fields
        self.cnn_branches = nn.ModuleList()

        for kernel_size in kernel_sizes:
            branch = nn.Sequential(
                # Conv1d expects (batch, channels, seq_len)
                nn.Conv1d(input_size, cnn_channels[0], kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(cnn_channels[0]),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(cnn_channels[1]),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(cnn_channels[2]),
                nn.ReLU(),
            )
            self.cnn_branches.append(branch)

        # Total CNN output channels
        total_cnn_channels = cnn_channels[-1] * len(kernel_sizes)

        # === LSTM Branch: Temporal Dependencies ===
        lstm_input_size = total_cnn_channels

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # LSTM output size
        lstm_output_size = lstm_hidden_size * (2 if bidirectional else 1)

        # === Attention Mechanism ===
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_size // 2, 1)
        )

        # === Fully Connected Layers ===
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input (batch, seq_len, features)

        Returns:
            Predictions (batch, 1)
        """
        batch_size, seq_len, features = x.shape

        # === CNN Processing ===
        # Conv1d expects (batch, channels, seq_len)
        x_cnn = x.transpose(1, 2)  # (batch, features, seq_len)

        # Process through parallel CNN branches
        cnn_outputs = []
        for branch in self.cnn_branches:
            out = branch(x_cnn)  # (batch, cnn_channels, seq_len)
            cnn_outputs.append(out)

        # Concatenate CNN outputs
        x_cnn = torch.cat(cnn_outputs, dim=1)  # (batch, total_cnn_channels, seq_len)

        # Transpose back for LSTM: (batch, seq_len, channels)
        x_cnn = x_cnn.transpose(1, 2)

        # === LSTM Processing ===
        lstm_out, (h_n, c_n) = self.lstm(x_cnn)
        # lstm_out: (batch, seq_len, lstm_output_size)

        # === Attention Mechanism ===
        # Calculate attention weights
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Apply attention
        context = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, lstm_output_size)

        # === Final Prediction ===
        output = self.fc(context)  # (batch, 1)

        return output.squeeze(-1)  # (batch,)


class HybridLSTMCNNPredictor:
    """
    Wrapper class for training and using Hybrid LSTM/CNN model.
    Compatible with the existing EnhancedEnsemblePredictor interface.
    """

    def __init__(
        self,
        lookback: int = 20,
        cnn_channels: List[int] = [64, 128, 64],
        kernel_sizes: List[int] = [3, 5, 7],
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        device: str = 'cpu',
        use_profit_loss: bool = True,
        random_state: int = 42
    ):
        """
        Args:
            lookback: Number of timesteps to look back
            cnn_channels: CNN channel sizes
            kernel_sizes: CNN kernel sizes
            lstm_hidden_size: LSTM hidden size
            lstm_num_layers: Number of LSTM layers
            dropout: Dropout probability
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Number of training epochs
            device: Device ('cpu' or 'cuda')
            use_profit_loss: Use profit-maximizing loss (True) or MSE (False)
            random_state: Random seed
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        self.lookback = lookback
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.use_profit_loss = use_profit_loss
        self.random_state = random_state
        self.learning_rate = learning_rate

        # Model parameters
        self.cnn_channels = cnn_channels
        self.kernel_sizes = kernel_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.dropout = dropout

        self.model = None
        self.input_size = None

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Train the hybrid model.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training targets (n_samples,)
            X_val: Validation features
            y_val: Validation targets
        """
        self.input_size = X_train.shape[1]

        # Create model
        self.model = HybridLSTMCNN(
            input_size=self.input_size,
            cnn_channels=self.cnn_channels,
            kernel_sizes=self.kernel_sizes,
            lstm_hidden_size=self.lstm_hidden_size,
            lstm_num_layers=self.lstm_num_layers,
            dropout=self.dropout,
            bidirectional=True
        )

        # Handle meta tensors properly for newer PyTorch versions
        try:
            self.model = self.model.to(self.device)
        except NotImplementedError:
            self.model = self.model.to_empty(device=self.device)
            self.model = self.model.to(self.device)

        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train, self.lookback)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            val_dataset = TimeSeriesDataset(X_val, y_val, self.lookback)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None

        # Loss and optimizer
        if self.use_profit_loss:
            criterion = ProfitMaximizingLoss(alpha=0.5, beta=0.3, gamma=0.2)
            print("[INFO] Using Profit-Maximizing Loss (Sharpe + Direction + Magnitude)")
        else:
            criterion = nn.MSELoss()
            print("[INFO] Using Standard MSE Loss")

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        # Training loop
        print(f"[INFO] Training Hybrid LSTM/CNN for {self.epochs} epochs...")
        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.squeeze().to(self.device)

                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()

                # IMPROVEMENT #2: Gradient clipping to prevent NaN losses
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.squeeze().to(self.device)

                        predictions = self.model(X_batch)
                        loss = criterion(predictions, y_batch)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.6f}")

        print("[OK] Hybrid LSTM/CNN training complete")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predictions (n_samples - lookback,)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        self.model.eval()

        # Create dataset
        # Use dummy y (not used for prediction)
        dummy_y = np.zeros(len(X))
        dataset = TimeSeriesDataset(X, dummy_y, self.lookback)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        predictions = []

        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                pred = self.model(X_batch)
                predictions.append(pred.cpu().numpy())

        if len(predictions) == 0:
            raise ValueError(f"No predictions generated. Input size ({len(X)}) may be too small for lookback window ({self.lookback})")

        return np.concatenate(predictions)


class TimeSeriesDataset(Dataset):
    """Dataset for time series with lookback windows."""

    def __init__(self, X: np.ndarray, y: np.ndarray, lookback: int = 20):
        """
        Args:
            X: Features (n_samples, n_features)
            y: Targets (n_samples,)
            lookback: Number of timesteps to look back
        """
        self.X = X
        self.y = y
        self.lookback = lookback

    def __len__(self):
        return len(self.X) - self.lookback

    def __getitem__(self, idx):
        # Get sequence of lookback timesteps
        x_seq = self.X[idx:idx+self.lookback]
        y_target = self.y[idx+self.lookback]

        return torch.FloatTensor(x_seq), torch.FloatTensor([y_target])


def main():
    """Example usage"""
    print("="*60)
    print("HYBRID LSTM/CNN WITH PROFIT-MAXIMIZING LOSS")
    print("Based on 2025 research (96% accuracy potential)")
    print("="*60)

    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    n_features = 90

    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples) * 0.02  # Simulated returns

    # Split
    split = int(n_samples * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Create model
    model = HybridLSTMCNNPredictor(
        lookback=20,
        epochs=50,
        use_profit_loss=True
    )

    # Train
    print("\n[INFO] Training with Profit-Maximizing Loss...")
    model.fit(X_train, y_train, X_val, y_val)

    # Predict
    predictions = model.predict(X_val)

    print(f"\n[OK] Generated {len(predictions)} predictions")
    print(f"     Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")

    # Calculate directional accuracy
    actual_direction = np.sign(y_val[20:])  # Skip lookback
    pred_direction = np.sign(predictions)
    accuracy = (actual_direction == pred_direction).mean() * 100

    print(f"     Directional Accuracy: {accuracy:.1f}%")

    print("\n[SUCCESS] Hybrid LSTM/CNN demonstration complete!")


if __name__ == "__main__":
    main()
