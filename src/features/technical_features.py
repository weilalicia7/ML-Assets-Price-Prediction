"""
Technical Features Engineering
Implements 50+ technical indicators for stock/crypto prediction.

Features include:
- Momentum indicators (RSI, MACD, Stochastic)
- Trend indicators (Moving Averages, ADX)
- Volatility indicators (ATR, Bollinger Bands)
- Volume indicators (OBV, Volume MA)
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict


class TechnicalFeatureEngineer:
    """
    Engineers technical features from OHLCV data.

    Uses the 'ta' library for efficient calculation.
    """

    def __init__(self):
        """Initialize feature engineer."""
        self.feature_names = []
        self.adaptive_windows = None

    def _get_adaptive_windows(self, data_length: int) -> Dict[str, list]:
        """
        Get adaptive window sizes based on available data.

        For recently listed stocks with limited data, use proportionally smaller windows.

        Args:
            data_length: Number of available trading days

        Returns:
            Dict with window sizes for different feature types
        """
        if data_length >= 252:
            # Standard windows for stocks with 1+ year of data
            return {
                'sma': [5, 10, 20, 50, 200],
                'ema': [12, 26, 50],
                'volume': [20, 60],
                'volatility': [20, 60],
                'rsi': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'atr': 14,
                'bb': 20
            }
        elif data_length >= 100:
            # Reduced windows for 100-252 days of data
            return {
                'sma': [5, 10, 20, min(50, data_length // 2), min(100, data_length - 10)],
                'ema': [12, 26, min(50, data_length // 2)],
                'volume': [20, min(60, data_length // 2)],
                'volatility': [20, min(60, data_length // 2)],
                'rsi': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'atr': 14,
                'bb': 20
            }
        elif data_length >= 50:
            # Minimal windows for 50-100 days of data
            return {
                'sma': [5, 10, min(20, data_length // 3), min(30, data_length // 2), min(40, data_length - 10)],
                'ema': [12, min(26, data_length // 3), min(30, data_length // 2)],
                'volume': [min(20, data_length // 3), min(30, data_length // 2)],
                'volatility': [min(20, data_length // 3), min(30, data_length // 2)],
                'rsi': min(14, data_length // 4),
                'macd_fast': min(12, data_length // 5),
                'macd_slow': min(26, data_length // 3),
                'macd_signal': 9,
                'atr': min(14, data_length // 4),
                'bb': min(20, data_length // 3)
            }
        else:
            # Ultra-minimal windows for <50 days
            return {
                'sma': [3, 5, min(10, data_length // 3), min(15, data_length // 2), max(20, data_length - 10)],
                'ema': [min(10, data_length // 4), min(15, data_length // 3), min(20, data_length // 2)],
                'volume': [min(10, data_length // 3), min(15, data_length // 2)],
                'volatility': [min(10, data_length // 3), min(15, data_length // 2)],
                'rsi': min(10, data_length // 4),
                'macd_fast': min(8, data_length // 5),
                'macd_slow': min(15, data_length // 3),
                'macd_signal': min(7, data_length // 5),
                'atr': min(10, data_length // 4),
                'bb': min(15, data_length // 3)
            }

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical features to dataframe with adaptive windows.

        Args:
            df: DataFrame with OHLCV columns (Open, High, Low, Close, Volume)

        Returns:
            DataFrame with added technical features
        """
        print(f"[INFO] Engineering technical features for {len(df)} days of data...")

        # Make copy to avoid modifying original
        df = df.copy()

        # Get adaptive windows based on data length
        self.adaptive_windows = self._get_adaptive_windows(len(df))
        print(f"[INFO] Using adaptive windows: SMA={self.adaptive_windows['sma']}, EMA={self.adaptive_windows['ema']}")

        # Add each feature category
        df = self.add_momentum_features(df)
        df = self.add_trend_features(df)
        df = self.add_volatility_features(df)
        df = self.add_volume_features(df)
        df = self.add_price_features(df)

        # Remove rows only where critical features are NaN
        # Keep rows with some NaN in long-window features (they may not exist for short-history stocks)
        initial_rows = len(df)

        # Identify critical features (short-window features that should always exist)
        critical_features = []
        for col in self.feature_names:
            if col in df.columns:
                # Include RSI, MACD, ATR, Bollinger Bands
                if any(x in col for x in ['rsi', 'macd', 'atr', 'bb_']):
                    critical_features.append(col)
                # Include short SMAs and EMAs (extract window size safely)
                elif col.startswith('sma_') or col.startswith('ema_'):
                    try:
                        window = int(col.split('_')[1])
                        if window <= 26:  # Short windows only
                            critical_features.append(col)
                    except:
                        pass

        # Only drop rows where critical features are NaN
        if critical_features:
            df = df.dropna(subset=critical_features)
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                print(f"[INFO] Dropped {dropped_rows} rows with NaN in critical features")

        # Fill remaining NaN values with forward fill, then backward fill
        for col in self.feature_names:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].ffill().bfill()

        print(f"[OK] Added {len(self.feature_names)} technical features")

        return df

    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicators with adaptive windows.

        Features:
        - RSI (adaptive window)
        - MACD (adaptive windows)
        - Stochastic Oscillator
        - ROC (Rate of Change)
        - Williams %R
        """
        print("  - Adding momentum features...")

        # Use adaptive windows
        rsi_window = self.adaptive_windows['rsi']
        macd_fast = self.adaptive_windows['macd_fast']
        macd_slow = self.adaptive_windows['macd_slow']
        macd_signal = self.adaptive_windows['macd_signal']

        # RSI - Relative Strength Index
        df[f'rsi_{rsi_window}'] = ta.momentum.RSIIndicator(df['Close'], window=rsi_window).rsi()
        df[f'rsi_{min(21, rsi_window + 7)}'] = ta.momentum.RSIIndicator(df['Close'], window=min(21, rsi_window + 7)).rsi()

        # MACD - Moving Average Convergence Divergence
        macd = ta.trend.MACD(df['Close'], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # ROC - Rate of Change
        df['roc_10'] = ta.momentum.ROCIndicator(df['Close'], window=10).roc()
        df['roc_20'] = ta.momentum.ROCIndicator(df['Close'], window=min(20, len(df) // 3)).roc()

        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(
            df['High'], df['Low'], df['Close'], lbp=rsi_window
        ).williams_r()

        # Ultimate Oscillator
        df['ultimate_osc'] = ta.momentum.UltimateOscillator(
            df['High'], df['Low'], df['Close']
        ).ultimate_oscillator()

        features = [f'rsi_{rsi_window}', f'rsi_{min(21, rsi_window + 7)}', 'macd', 'macd_signal', 'macd_diff',
                   'stoch_k', 'stoch_d', 'roc_10', 'roc_20', 'williams_r',
                   'ultimate_osc']
        self.feature_names.extend(features)

        return df

    def add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend indicators with adaptive windows.

        Features:
        - SMA (adaptive windows based on data length)
        - EMA (adaptive windows based on data length)
        - ADX (Average Directional Index)
        - Aroon Indicator
        """
        print("  - Adding trend features...")

        # Simple Moving Averages - Use adaptive windows
        for window in self.adaptive_windows['sma']:
            df[f'sma_{window}'] = ta.trend.SMAIndicator(df['Close'], window=window).sma_indicator()
            self.feature_names.append(f'sma_{window}')

        # Exponential Moving Averages - Use adaptive windows
        for window in self.adaptive_windows['ema']:
            df[f'ema_{window}'] = ta.trend.EMAIndicator(df['Close'], window=window).ema_indicator()
            self.feature_names.append(f'ema_{window}')

        # ADX - Average Directional Index (trend strength)
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()

        # Aroon Indicator
        aroon = ta.trend.AroonIndicator(df['High'], df['Low'])
        df['aroon_up'] = aroon.aroon_up()
        df['aroon_down'] = aroon.aroon_down()
        df['aroon_indicator'] = aroon.aroon_indicator()

        # Moving Average Crossovers - Use adaptive windows
        sma_windows = self.adaptive_windows['sma']
        if len(sma_windows) >= 3:
            df[f'sma_cross_{sma_windows[0]}_{sma_windows[2]}'] = (df[f'sma_{sma_windows[0]}'] > df[f'sma_{sma_windows[2]}']).astype(int)
        if len(sma_windows) >= 5:
            df[f'sma_cross_{sma_windows[3]}_{sma_windows[4]}'] = (df[f'sma_{sma_windows[3]}'] > df[f'sma_{sma_windows[4]}']).astype(int)  # Golden/Death cross

        cross_features = []
        if len(sma_windows) >= 3:
            cross_features.append(f'sma_cross_{sma_windows[0]}_{sma_windows[2]}')
        if len(sma_windows) >= 5:
            cross_features.append(f'sma_cross_{sma_windows[3]}_{sma_windows[4]}')

        features = ['adx', 'adx_pos', 'adx_neg', 'aroon_up', 'aroon_down',
                   'aroon_indicator'] + cross_features
        self.feature_names.extend(features)

        return df

    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility indicators with adaptive windows.

        Features:
        - ATR (Average True Range)
        - Bollinger Bands
        - Keltner Channel
        - Donchian Channel
        - Historical volatility
        """
        print("  - Adding volatility features...")

        # Use adaptive windows
        atr_window = self.adaptive_windows['atr']
        bb_window = self.adaptive_windows['bb']
        vol_windows = self.adaptive_windows['volatility']

        # ATR - Average True Range
        df[f'atr_{atr_window}'] = ta.volatility.AverageTrueRange(
            df['High'], df['Low'], df['Close'], window=atr_window
        ).average_true_range()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'], window=bb_window)
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_width'] = bollinger.bollinger_wband()
        df['bb_pband'] = bollinger.bollinger_pband()

        # Keltner Channel
        keltner = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])
        df['keltner_high'] = keltner.keltner_channel_hband()
        df['keltner_low'] = keltner.keltner_channel_lband()
        df['keltner_mid'] = keltner.keltner_channel_mband()

        # Donchian Channel
        donchian = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'])
        df['donchian_high'] = donchian.donchian_channel_hband()
        df['donchian_low'] = donchian.donchian_channel_lband()
        df['donchian_mid'] = donchian.donchian_channel_mband()

        # Historical Volatility (rolling std of returns) - Use adaptive windows
        df[f'hist_vol_{vol_windows[0]}'] = df['Close'].pct_change().rolling(window=vol_windows[0]).std()
        df[f'hist_vol_{vol_windows[1]}'] = df['Close'].pct_change().rolling(window=vol_windows[1]).std()

        features = [f'atr_{atr_window}', 'bb_high', 'bb_low', 'bb_mid', 'bb_width', 'bb_pband',
                   'keltner_high', 'keltner_low', 'keltner_mid',
                   'donchian_high', 'donchian_low', 'donchian_mid',
                   f'hist_vol_{vol_windows[0]}', f'hist_vol_{vol_windows[1]}']
        self.feature_names.extend(features)

        return df

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume indicators with adaptive windows.

        Features:
        - OBV (On Balance Volume)
        - Volume MA
        - Volume Rate of Change
        - MFI (Money Flow Index)
        - A/D (Accumulation/Distribution)
        """
        print("  - Adding volume features...")

        # Use adaptive windows
        vol_windows = self.adaptive_windows['volume']

        # OBV - On Balance Volume
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

        # Volume Moving Averages - Use adaptive windows
        df[f'volume_ma_{vol_windows[0]}'] = df['Volume'].rolling(window=vol_windows[0]).mean()
        df[f'volume_ma_{vol_windows[1]}'] = df['Volume'].rolling(window=vol_windows[1]).mean()

        # Volume Rate of Change
        df['volume_roc'] = df['Volume'].pct_change(periods=10)

        # MFI - Money Flow Index
        df['mfi'] = ta.volume.MFIIndicator(
            df['High'], df['Low'], df['Close'], df['Volume']
        ).money_flow_index()

        # A/D - Accumulation/Distribution
        df['ad'] = ta.volume.AccDistIndexIndicator(
            df['High'], df['Low'], df['Close'], df['Volume']
        ).acc_dist_index()

        # Chaikin Money Flow
        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
            df['High'], df['Low'], df['Close'], df['Volume']
        ).chaikin_money_flow()

        # Volume ratio (current vs MA) - Use first adaptive volume window
        df['volume_ratio'] = df['Volume'] / df[f'volume_ma_{vol_windows[0]}']

        features = ['obv', f'volume_ma_{vol_windows[0]}', f'volume_ma_{vol_windows[1]}', 'volume_roc',
                   'mfi', 'ad', 'cmf', 'volume_ratio']
        self.feature_names.extend(features)

        return df

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features.

        Features:
        - Returns (daily, weekly)
        - Intraday range
        - Gap (open vs previous close)
        - Price momentum
        """
        print("  - Adding price features...")

        # Returns
        df['returns_1d'] = df['Close'].pct_change(1)
        df['returns_5d'] = df['Close'].pct_change(5)
        df['returns_20d'] = df['Close'].pct_change(20)

        # Intraday range
        df['intraday_range'] = (df['High'] - df['Low']) / df['Close']

        # Gap (overnight gap)
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

        # Price momentum (ROC on ROC)
        df['price_momentum'] = df['returns_1d'] - df['returns_1d'].shift(1)

        # High/Low ratios
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_high_ratio'] = df['Close'] / df['High']
        df['close_low_ratio'] = df['Close'] / df['Low']

        # Price position relative to MA - Use adaptive windows
        sma_windows = self.adaptive_windows['sma']
        if len(sma_windows) >= 3:
            df[f'price_vs_sma{sma_windows[2]}'] = (df['Close'] - df[f'sma_{sma_windows[2]}']) / df[f'sma_{sma_windows[2]}']
        if len(sma_windows) >= 4:
            df[f'price_vs_sma{sma_windows[3]}'] = (df['Close'] - df[f'sma_{sma_windows[3]}']) / df[f'sma_{sma_windows[3]}']

        price_sma_features = []
        if len(sma_windows) >= 3:
            price_sma_features.append(f'price_vs_sma{sma_windows[2]}')
        if len(sma_windows) >= 4:
            price_sma_features.append(f'price_vs_sma{sma_windows[3]}')

        features = ['returns_1d', 'returns_5d', 'returns_20d', 'intraday_range',
                   'gap', 'price_momentum', 'high_low_ratio', 'close_high_ratio',
                   'close_low_ratio'] + price_sma_features
        self.feature_names.extend(features)

        return df

    def get_feature_names(self) -> list:
        """Get list of all feature names."""
        return self.feature_names


def main():
    """
    Example usage of TechnicalFeatureEngineer.
    """
    from src.data.fetch_data import DataFetcher

    print("="*60)
    print("TECHNICAL FEATURE ENGINEERING - EXAMPLE")
    print("="*60)

    # Fetch data
    print("\nStep 1: Fetching data...")
    fetcher = DataFetcher(['AAPL'], start_date='2022-01-01')
    data = fetcher.fetch_all()

    # Filter to single ticker
    aapl_data = data[data['Ticker'] == 'AAPL'].copy()

    print(f"[OK] Fetched {len(aapl_data)} rows for AAPL")
    print(f"     Date range: {aapl_data.index.min()} to {aapl_data.index.max()}")

    # Engineer features
    print("\nStep 2: Engineering features...")
    engineer = TechnicalFeatureEngineer()
    aapl_data = engineer.add_all_features(aapl_data)

    print(f"\n[OK] Final dataset: {len(aapl_data)} rows × {len(aapl_data.columns)} columns")
    print(f"     Features added: {len(engineer.get_feature_names())}")

    # Show sample
    print("\nSample features (latest 3 rows):")
    feature_cols = ['Close', 'rsi_14', 'macd', 'atr_14', 'bb_width', 'obv']
    print(aapl_data[feature_cols].tail(3))

    # Feature statistics
    print("\nFeature statistics:")
    print(aapl_data[feature_cols].describe())


if __name__ == "__main__":
    main()
