"""
Advanced Volatility Features
Critical for volatility and price range prediction.

Implements:
- Parkinson volatility (High-Low based)
- Garman-Klass volatility (OHLC based)
- Rogers-Satchell volatility
- Yang-Zhang volatility
- Volatility regimes
- Volatility momentum
"""

import pandas as pd
import numpy as np
from typing import Dict


class VolatilityFeatureEngineer:
    """
    Engineers advanced volatility features.

    These are CRITICAL for predicting price ranges and volatility.
    """

    def __init__(self):
        """Initialize volatility feature engineer."""
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
                'short': [5, 10, 20, 60],
                'percentile': 252,
                'regime': 60
            }
        elif data_length >= 100:
            # Reduced windows for 100-252 days of data
            return {
                'short': [5, 10, 20, min(60, data_length // 2)],
                'percentile': data_length // 2,
                'regime': min(60, data_length // 2)
            }
        elif data_length >= 50:
            # Minimal windows for 50-100 days of data
            return {
                'short': [5, 10, min(20, data_length // 3), min(30, data_length // 2)],
                'percentile': data_length // 2,
                'regime': min(30, data_length // 2)
            }
        else:
            # Ultra-minimal windows for <50 days
            return {
                'short': [3, 5, min(10, data_length // 3), min(15, data_length // 2)],
                'percentile': max(20, data_length // 2),
                'regime': max(15, data_length // 3)
            }

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all volatility features with adaptive windows.

        Args:
            df: DataFrame with OHLC columns

        Returns:
            DataFrame with added volatility features
        """
        print(f"[INFO] Engineering volatility features for {len(df)} days of data...")

        df = df.copy()

        # Get adaptive windows based on data length
        self.adaptive_windows = self._get_adaptive_windows(len(df))
        print(f"[INFO] Using adaptive windows: {self.adaptive_windows}")

        # Add different volatility estimators
        df = self.add_parkinson_volatility(df)
        df = self.add_garman_klass_volatility(df)
        df = self.add_rogers_satchell_volatility(df)
        df = self.add_yang_zhang_volatility(df)
        df = self.add_volatility_ratios(df)
        df = self.add_volatility_regimes(df)
        df = self.add_volatility_momentum(df)

        print(f"[OK] Added {len(self.feature_names)} volatility features")

        return df

    def add_parkinson_volatility(self, df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
        """
        Parkinson volatility estimator (High-Low based).

        More efficient than close-to-close, uses intraday range.

        Formula: sqrt((1/(4*ln(2))) * ln(H/L)^2)
        """
        print("  - Adding Parkinson volatility...")

        # Use adaptive windows if available
        if windows is None:
            windows = self.adaptive_windows['short'] if self.adaptive_windows else [5, 10, 20, 60]

        for window in windows:
            # Calculate log(High/Low)^2
            hl_ratio = np.log(df['High'] / df['Low']) ** 2

            # Rolling mean and scale
            df[f'parkinson_vol_{window}'] = np.sqrt(
                (1 / (4 * np.log(2))) * hl_ratio.rolling(window=window).mean()
            )

            self.feature_names.append(f'parkinson_vol_{window}')

        return df

    def add_garman_klass_volatility(self, df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
        """
        Garman-Klass volatility estimator (OHLC based).

        More accurate than Parkinson, uses all OHLC data.

        Formula: sqrt(0.5 * ln(H/L)^2 - (2*ln(2)-1) * ln(C/O)^2)
        """
        print("  - Adding Garman-Klass volatility...")

        # Use adaptive windows if available
        if windows is None:
            windows = self.adaptive_windows['short'] if self.adaptive_windows else [5, 10, 20, 60]

        for window in windows:
            # High-Low component
            hl_component = 0.5 * (np.log(df['High'] / df['Low']) ** 2)

            # Close-Open component
            co_component = (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']) ** 2)

            # Combined
            gk = hl_component - co_component

            df[f'gk_vol_{window}'] = np.sqrt(gk.rolling(window=window).mean())

            self.feature_names.append(f'gk_vol_{window}')

        return df

    def add_rogers_satchell_volatility(self, df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
        """
        Rogers-Satchell volatility (handles drift).

        Good for trending markets.
        """
        print("  - Adding Rogers-Satchell volatility...")

        # Use adaptive windows if available
        if windows is None:
            windows = self.adaptive_windows['short'] if self.adaptive_windows else [5, 10, 20, 60]

        for window in windows:
            # RS = sqrt(mean(ln(H/C) * ln(H/O) + ln(L/C) * ln(L/O)))
            rs = (
                np.log(df['High'] / df['Close']) * np.log(df['High'] / df['Open']) +
                np.log(df['Low'] / df['Close']) * np.log(df['Low'] / df['Open'])
            )

            df[f'rs_vol_{window}'] = np.sqrt(rs.rolling(window=window).mean())

            self.feature_names.append(f'rs_vol_{window}')

        return df

    def add_yang_zhang_volatility(self, df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
        """
        Yang-Zhang volatility (most complete estimator).

        Combines overnight and intraday volatility.
        Most accurate but complex.
        """
        print("  - Adding Yang-Zhang volatility...")

        # Use adaptive windows if available
        if windows is None:
            windows = self.adaptive_windows['short'] if self.adaptive_windows else [5, 10, 20, 60]

        for window in windows:
            # Overnight volatility
            overnight_vol = (np.log(df['Open'] / df['Close'].shift(1)) ** 2).rolling(window=window).mean()

            # Open-to-close volatility
            oc_vol = (np.log(df['Close'] / df['Open']) ** 2).rolling(window=window).mean()

            # Rogers-Satchell component
            rs = (
                np.log(df['High'] / df['Close']) * np.log(df['High'] / df['Open']) +
                np.log(df['Low'] / df['Close']) * np.log(df['Low'] / df['Open'])
            ).rolling(window=window).mean()

            # Combine (simplified Yang-Zhang)
            k = 0.34 / (1.34 + (window + 1) / (window - 1))
            df[f'yz_vol_{window}'] = np.sqrt(overnight_vol + k * oc_vol + (1 - k) * rs)

            self.feature_names.append(f'yz_vol_{window}')

        return df

    def add_volatility_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility ratios (relative volatility measures).

        These help detect volatility regime changes.
        """
        print("  - Adding volatility ratios...")

        # Get adaptive windows
        windows = self.adaptive_windows['short'] if self.adaptive_windows else [5, 10, 20, 60]
        percentile_window = self.adaptive_windows['percentile'] if self.adaptive_windows else 252

        # Track dynamically created features
        features = []

        # Short-term vs long-term volatility (use first 2 windows)
        if len(windows) >= 3:
            col_name = f'vol_ratio_{windows[0]}_{windows[2]}'
            df[col_name] = df[f'parkinson_vol_{windows[0]}'] / df[f'parkinson_vol_{windows[2]}']
            features.append(col_name)

        if len(windows) >= 4:
            col_name = f'vol_ratio_{windows[1]}_{windows[3]}'
            df[col_name] = df[f'parkinson_vol_{windows[1]}'] / df[f'parkinson_vol_{windows[3]}']
            features.append(col_name)

        # Current vs historical average
        if len(windows) >= 3:
            col_name = f'vol_vs_avg_{windows[2]}'
            df[col_name] = df[f'parkinson_vol_{windows[0]}'] / df[f'parkinson_vol_{windows[2]}']
            features.append(col_name)

        # Volatility rank (percentile over adaptive window)
        col_name = f'vol_rank_{percentile_window}'
        df[col_name] = df[f'parkinson_vol_{windows[2] if len(windows) >= 3 else windows[-1]}'].rolling(window=percentile_window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == percentile_window else np.nan
        )
        features.append(col_name)

        self.feature_names.extend(features)

        return df

    def add_volatility_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect volatility regimes (Low/Medium/High).

        Critical for regime-switching models.
        """
        print("  - Adding volatility regimes...")

        # Get adaptive windows
        windows = self.adaptive_windows['short'] if self.adaptive_windows else [5, 10, 20, 60]
        regime_window = self.adaptive_windows['regime'] if self.adaptive_windows else 60

        # Use appropriate Parkinson volatility as base (index 2 if available, else last)
        vol_col = f'parkinson_vol_{windows[2] if len(windows) >= 3 else windows[-1]}'
        vol = df[vol_col]

        # Calculate rolling percentiles with adaptive window
        df[f'vol_percentile_{regime_window}'] = vol.rolling(window=regime_window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == regime_window else np.nan
        )

        # Define regimes
        df['volatility_regime'] = pd.cut(
            df[f'vol_percentile_{regime_window}'],
            bins=[0, 0.33, 0.67, 1.0],
            labels=['low', 'medium', 'high'],
            include_lowest=True
        )

        # One-hot encode
        df['regime_low'] = (df['volatility_regime'] == 'low').astype(int)
        df['regime_medium'] = (df['volatility_regime'] == 'medium').astype(int)
        df['regime_high'] = (df['volatility_regime'] == 'high').astype(int)

        # Drop the categorical column after encoding (XGBoost doesn't support categorical without enable_categorical=True)
        df = df.drop('volatility_regime', axis=1)

        # Volatility spike detection (>2x recent average)
        short_vol = f'parkinson_vol_{windows[0]}'
        medium_vol = f'parkinson_vol_{windows[2] if len(windows) >= 3 else windows[-1]}'
        df['vol_spike'] = (df[short_vol] > 2 * df[medium_vol]).astype(int)

        features = [f'vol_percentile_{regime_window}', 'regime_low', 'regime_medium',
                   'regime_high', 'vol_spike']
        self.feature_names.extend(features)

        return df

    def add_volatility_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility momentum (is volatility increasing?).

        Important for predicting volatility trends.
        """
        print("  - Adding volatility momentum...")

        # Get adaptive windows
        windows = self.adaptive_windows['short'] if self.adaptive_windows else [5, 10, 20, 60]

        # Volatility change using adaptive windows
        short_vol = f'parkinson_vol_{windows[0]}'
        medium_vol = f'parkinson_vol_{windows[2] if len(windows) >= 3 else windows[-1]}'

        # Track dynamically created features
        features = []

        # Short-term volatility change - use actual window size
        short_window = windows[0]
        short_change_col = f'vol_change_{short_window}d'
        df[short_change_col] = df[short_vol].pct_change(short_window)
        features.append(short_change_col)

        # Medium-term volatility change
        medium_window = windows[2] if len(windows) >= 3 else windows[-1]
        medium_change_col = f'vol_change_{medium_window}d'
        df[medium_change_col] = df[medium_vol].pct_change(medium_window)
        features.append(medium_change_col)

        # Volatility direction (increasing/decreasing) - use dynamic column name
        df['vol_increasing'] = (df[short_change_col] > 0).astype(int)
        features.append('vol_increasing')

        # Volatility acceleration (change of change) - use dynamic column name
        df['vol_acceleration'] = df[short_change_col] - df[short_change_col].shift(short_window)
        features.append('vol_acceleration')

        # Days since volatility spike
        spike_mask = df['vol_spike'] == 1
        df['days_since_spike'] = 0
        days_counter = 0
        for i in range(len(df)):
            if spike_mask.iloc[i]:
                days_counter = 0
            else:
                days_counter += 1
            df['days_since_spike'].iloc[i] = days_counter
        features.append('days_since_spike')

        self.feature_names.extend(features)

        return df

    def get_feature_names(self) -> list:
        """Get list of all feature names."""
        return self.feature_names


def main():
    """
    Example usage of VolatilityFeatureEngineer.
    """
    from src.data.fetch_data import DataFetcher

    print("="*60)
    print("VOLATILITY FEATURE ENGINEERING - EXAMPLE")
    print("="*60)

    # Fetch data
    print("\nStep 1: Fetching data...")
    fetcher = DataFetcher(['AAPL'], start_date='2022-01-01')
    data = fetcher.fetch_all()

    aapl_data = data[data['Ticker'] == 'AAPL'].copy()
    print(f"[OK] Fetched {len(aapl_data)} rows")

    # Engineer volatility features
    print("\nStep 2: Engineering volatility features...")
    engineer = VolatilityFeatureEngineer()
    aapl_data = engineer.add_all_features(aapl_data)

    print(f"\n[OK] Added {len(engineer.get_feature_names())} volatility features")

    # Show different volatility measures
    print("\nVolatility measures comparison (last 5 days):")
    vol_cols = ['Close', 'parkinson_vol_20', 'gk_vol_20', 'rs_vol_20', 'yz_vol_20']
    print(aapl_data[vol_cols].tail(5))

    # Show regime
    print("\nVolatility regime (last 5 days):")
    regime_cols = ['Close', 'volatility_regime', 'vol_spike', 'vol_percentile_60']
    print(aapl_data[regime_cols].tail(5))

    # Statistics
    print("\nVolatility feature statistics:")
    stats_cols = ['parkinson_vol_20', 'vol_ratio_5_20', 'vol_change_5d']
    print(aapl_data[stats_cols].describe())


if __name__ == "__main__":
    main()
