"""
Neural Network Models for Stock Prediction
Implements TCN, LSTM, and Transformer models for time series forecasting

Based on professional quant firm standards:
- Temporal Convolutional Networks (TCN) for sequence modeling
- LSTM/GRU for capturing long-term dependencies
- Transformers for attention-based predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
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
    print("[WARN] PyTorch not available. Neural models will not work.")
    print("      Install with: pip install torch")


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


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN)

    Key advantages:
    - Parallelizable (unlike RNN/LSTM)
    - Long effective history through dilated convolutions
    - Stable gradients
    - Fast training and inference

    Architecture:
    - Residual blocks with dilated causal convolutions
    - Exponentially increasing dilation factors
    - Dropout for regularization
    """

    def __init__(
        self,
        input_size: int,
        num_channels: List[int] = [32, 32, 32],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        """
        Args:
            input_size: Number of input features
            num_channels: Channels in each residual block
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
        """
        super().__init__()

        self.input_size = input_size
        self.num_channels = num_channels

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # TCN expects: (batch, features, seq_len)
        x = x.transpose(1, 2)

        # Pass through TCN
        y = self.network(x)

        # Take last timestep
        y = y[:, :, -1]

        # Linear projection
        return self.linear(y)


class TemporalBlock(nn.Module):
    """Residual block for TCN with dilated causal convolutions."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Removes padding from the end to ensure causality."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class LSTMModel(nn.Module):
    """
    LSTM Model for time series prediction

    Key advantages:
    - Captures long-term dependencies
    - Handles variable-length sequences
    - Well-understood and stable

    Architecture:
    - Multi-layer LSTM
    - Dropout for regularization
    - Linear projection layer
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Args:
            input_size: Number of input features
            hidden_size: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take last timestep
        last_output = lstm_out[:, -1, :]

        # Dropout and linear projection
        out = self.dropout(last_output)
        return self.linear(out)


class TransformerModel(nn.Module):
    """
    Transformer Model for time series prediction

    Key advantages:
    - Attention mechanism for important features
    - Parallelizable training
    - Captures complex dependencies

    Architecture:
    - Positional encoding for sequence order
    - Multi-head self-attention
    - Feed-forward layers
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2
    ):
        """
        Args:
            input_size: Number of input features
            d_model: Dimension of model (must be divisible by nhead)
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
        """
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model

        # Project input to d_model dimension
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)

        # Project to d_model
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer
        transformer_out = self.transformer(x)

        # Take last timestep
        last_output = transformer_out[:, -1, :]

        # Dropout and linear projection
        out = self.dropout(last_output)
        return self.linear(out)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, 0, :].unsqueeze(0)
        return self.dropout(x)


class NeuralPredictor:
    """
    Wrapper class for neural network models.
    Provides scikit-learn-like interface for training and prediction.
    """

    def __init__(
        self,
        model_type: str = 'tcn',
        lookback: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        device: str = 'cpu',
        random_state: int = 42
    ):
        """
        Args:
            model_type: 'tcn', 'lstm', or 'transformer'
            lookback: Number of timesteps to look back
            hidden_size: Hidden dimension size
            num_layers: Number of layers
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            device: 'cpu' or 'cuda'
            random_state: Random seed
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for neural models. Install with: pip install torch")

        self.model_type = model_type
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.random_state = random_state

        self.model = None
        self.input_size = None
        self.scaler_X = None
        self.scaler_y = None

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def _create_model(self, input_size: int):
        """Create the neural network model."""
        if self.model_type == 'tcn':
            model = TemporalConvNet(
                input_size=input_size,
                num_channels=[self.hidden_size] * self.num_layers,
                dropout=self.dropout
            )
        elif self.model_type == 'lstm':
            model = LSTMModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
        elif self.model_type == 'transformer':
            model = TransformerModel(
                input_size=input_size,
                d_model=self.hidden_size,
                nhead=4,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Handle meta tensor issue - ensure model is properly initialized before moving to device
        try:
            return model.to(self.device)
        except NotImplementedError as e:
            if "meta tensor" in str(e):
                # Model has meta tensors - need to use to_empty() or reinitialize
                # For meta tensors, we need to create the model directly on the target device
                print(f"[WARN] Meta tensor detected, recreating model on {self.device}")

                # Recreate model directly on device
                if self.model_type == 'tcn':
                    model = TemporalConvNet(
                        num_inputs=input_size,
                        num_channels=[self.hidden_size] * self.num_layers,
                        kernel_size=3,
                        dropout=self.dropout
                    )
                elif self.model_type == 'lstm':
                    model = LSTMModel(
                        input_size=input_size,
                        hidden_size=self.hidden_size,
                        num_layers=self.num_layers,
                        dropout=self.dropout
                    )
                elif self.model_type == 'transformer':
                    model = TransformerModel(
                        input_size=input_size,
                        d_model=self.hidden_size,
                        nhead=4,
                        num_layers=self.num_layers,
                        dropout=self.dropout
                    )

                # Move model to device first (handle meta tensors properly)
                try:
                    model = model.to(self.device)
                except NotImplementedError:
                    # Handle meta tensors in newer PyTorch versions
                    model = model.to_empty(device=self.device)
                    model = model.to(self.device)

                # Initialize weights manually
                for m in model.modules():
                    if isinstance(m, (nn.Conv1d, nn.Linear)):
                        if m.weight.device.type == 'meta':
                            continue
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.LSTM):
                        for name, param in m.named_parameters():
                            if param.device.type == 'meta':
                                continue
                            if 'weight' in name:
                                nn.init.xavier_uniform_(param)
                            elif 'bias' in name:
                                nn.init.zeros_(param)

                return model
            else:
                raise

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train the neural network.

        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        # Store input size
        self.input_size = X.shape[1]

        # Create model
        self.model = self._create_model(self.input_size)

        # Create datasets
        train_dataset = TimeSeriesDataset(X, y, self.lookback)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            val_dataset = TimeSeriesDataset(X_val, y_val, self.lookback)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            train_loss = np.mean(train_losses)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_losses = []

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)

                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_losses.append(loss.item())

                val_loss = np.mean(val_losses)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.6f}")

        print(f"[OK] {self.model_type.upper()} trained ({epoch+1} epochs)")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predictions (n_samples - lookback,) or (n_samples,) for single samples
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        self.model.eval()

        # Handle single sample prediction
        if len(X) <= self.lookback:
            # For single sample or insufficient samples, use the last lookback samples from training
            # or repeat the sample to create a sequence
            if len(X) == 1:
                # Repeat single sample to create a sequence
                X_seq = np.repeat(X, self.lookback, axis=0)
            else:
                # Pad with first sample to reach lookback length
                padding_needed = self.lookback - len(X) + 1
                X_seq = np.vstack([np.repeat(X[0:1], padding_needed, axis=0), X])

            # Create tensor and predict
            X_tensor = torch.FloatTensor(X_seq[-self.lookback:]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(X_tensor)
                return output.cpu().numpy().flatten()

        # Normal prediction for multiple samples
        dataset = TimeSeriesDataset(X, np.zeros(len(X)), self.lookback)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        predictions = []

        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.extend(outputs.cpu().numpy().flatten())

        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities (for compatibility with ensemble).
        Converts regression output to probability using sigmoid.
        """
        predictions = self.predict(X)
        # Convert to probabilities using sigmoid
        probabilities = 1 / (1 + np.exp(-predictions))
        return probabilities


if __name__ == "__main__":
    print("Testing Neural Models...")

    if not TORCH_AVAILABLE:
        print("PyTorch not available. Install with: pip install torch")
    else:
        # Create synthetic data
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1] + 0.2 * np.random.randn(n_samples)).astype(np.float32)

        # Split
        train_size = 800
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Test TCN
        print("\n" + "="*60)
        print("Testing TCN")
        print("="*60)
        tcn = NeuralPredictor(model_type='tcn', epochs=20, lookback=10)
        tcn.fit(X_train, y_train, X_test[:50], y_test[:50])
        pred_tcn = tcn.predict(X_test)
        print(f"TCN Predictions: {pred_tcn[:5]}")

        # Test LSTM
        print("\n" + "="*60)
        print("Testing LSTM")
        print("="*60)
        lstm = NeuralPredictor(model_type='lstm', epochs=20, lookback=10)
        lstm.fit(X_train, y_train, X_test[:50], y_test[:50])
        pred_lstm = lstm.predict(X_test)
        print(f"LSTM Predictions: {pred_lstm[:5]}")

        # Test Transformer
        print("\n" + "="*60)
        print("Testing Transformer")
        print("="*60)
        transformer = NeuralPredictor(model_type='transformer', epochs=20, lookback=10)
        transformer.fit(X_train, y_train, X_test[:50], y_test[:50])
        pred_transformer = transformer.predict(X_test)
        print(f"Transformer Predictions: {pred_transformer[:5]}")

        print("\n[SUCCESS] All neural models working!")
