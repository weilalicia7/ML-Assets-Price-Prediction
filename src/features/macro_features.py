"""
Phase 4: Macro Feature Engineering

Engineers features from macro-economic indicators:
- VIX: Market volatility/fear gauge
- DXY: US Dollar strength
- SPY: S&P 500 market benchmark
- TLT: Treasury bonds (interest rates)
- GLD: Gold (safe haven)

These features help detect:
- Risk-on vs risk-off regimes
- Market volatility environments
- Currency impacts on international stocks
- Interest rate trends
- Safe haven flows
"""

import pandas as pd
import numpy as np
import os

class MacroFeatureEngineer:
    """
    Engineers macro-economic features for stock prediction.

    Uses pre-downloaded macro indicators from data/macro_indicators.csv
    """

    def __init__(self, macro_data_path='data/macro_indicators.csv'):
        """
        Initialize macro feature engineer.

        Args:
            macro_data_path: Path to macro indicators CSV file
        """
        self.macro_data_path = macro_data_path
        self.macro_data = None
        self.feature_names = []

        # Load macro data
        self._load_macro_data()

    def _load_macro_data(self):
        """Load pre-downloaded macro indicators."""
        if not os.path.exists(self.macro_data_path):
            print(f"[WARNING] Macro data not found at {self.macro_data_path}")
            print(f"          Run download_macro_data.py first!")
            self.macro_data = None
            return

        try:
            self.macro_data = pd.read_csv(self.macro_data_path, index_col=0, parse_dates=True)
            print(f"[INFO] Loaded macro data: {len(self.macro_data)} days, {list(self.macro_data.columns)}")
        except Exception as e:
            print(f"[ERROR] Failed to load macro data: {e}")
            self.macro_data = None

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all macro features to DataFrame.

        Args:
            df: DataFrame with OHLCV data (must have DatetimeIndex)

        Returns:
            DataFrame with macro features added
        """
        if self.macro_data is None:
            print("[WARNING] No macro data available, skipping macro features")
            return df

        print(f"[INFO] Engineering macro features...")

        df = df.copy()
        initial_cols = len(df.columns)

        # Merge macro data with stock data (align by date)
        df = self._merge_macro_data(df)

        # Add macro momentum features
        df = self._add_macro_momentum(df)

        # Add macro regime features
        df = self._add_macro_regimes(df)

        # Add relative strength features
        df = self._add_relative_strength(df)

        added_cols = len(df.columns) - initial_cols
        print(f"[OK] Added {added_cols} macro features")

        return df

    def _merge_macro_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge macro indicators with stock data."""
        # Ensure both indices are timezone-naive to avoid comparison warnings
        df_index = df.index.copy()
        macro_index = self.macro_data.index.copy()

        # Convert timezone-aware indices to timezone-naive if needed
        if df_index.tz is not None:
            df_index = df_index.tz_localize(None)
            df = df.copy()
            df.index = df_index
        if macro_index.tz is not None:
            macro_index = macro_index.tz_localize(None)
            self.macro_data = self.macro_data.copy()
            self.macro_data.index = macro_index

        # Align macro data to stock data dates (forward fill for weekends/holidays)
        macro_aligned = self.macro_data.reindex(df_index, method='ffill')

        # Add raw macro values
        for col in self.macro_data.columns:
            df[col] = macro_aligned[col]
            self.feature_names.append(col)

        return df

    def _add_macro_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add macro momentum features (rate of change)."""
        windows = [5, 20, 60]

        for col in ['VIX', 'DXY', 'SPY', 'TLT', 'GLD']:
            if col not in df.columns:
                continue

            for window in windows:
                # Momentum (percent change)
                momentum_col = f'{col}_momentum_{window}d'
                df[momentum_col] = df[col].pct_change(window)
                self.feature_names.append(momentum_col)

                # Moving average
                ma_col = f'{col}_ma_{window}d'
                df[ma_col] = df[col].rolling(window).mean()
                self.feature_names.append(ma_col)

                # Distance from MA (normalized)
                dist_col = f'{col}_dist_ma_{window}d'
                df[dist_col] = (df[col] - df[ma_col]) / df[ma_col]
                self.feature_names.append(dist_col)

        return df

    def _add_macro_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add macro regime classification features."""

        # VIX regimes (volatility)
        if 'VIX' in df.columns:
            df['vix_regime'] = pd.cut(
                df['VIX'],
                bins=[0, 15, 20, 30, 100],
                labels=['low_vol', 'normal', 'elevated', 'crisis']
            ).astype(str)

            # One-hot encode
            df['vix_low'] = (df['vix_regime'] == 'low_vol').astype(int)
            df['vix_normal'] = (df['vix_regime'] == 'normal').astype(int)
            df['vix_elevated'] = (df['vix_regime'] == 'elevated').astype(int)
            df['vix_crisis'] = (df['vix_regime'] == 'crisis').astype(int)

            df = df.drop('vix_regime', axis=1)

            self.feature_names.extend(['vix_low', 'vix_normal', 'vix_elevated', 'vix_crisis'])

            # VIX spike detection
            df['vix_spike'] = (df['VIX'] > df['VIX'].rolling(60).mean() +
                              2 * df['VIX'].rolling(60).std()).astype(int)
            self.feature_names.append('vix_spike')

        # Market regime (SPY + VIX + GLD)
        if all(col in df.columns for col in ['SPY', 'VIX', 'GLD']):
            # Risk-on: SPY up, VIX down, GLD flat/down
            spy_up = df['SPY'].pct_change(20) > 0
            vix_down = df['VIX'].pct_change(20) < 0
            gld_down = df['GLD'].pct_change(20) < 0

            df['risk_on'] = (spy_up & vix_down).astype(int)
            df['risk_off'] = ((~spy_up) & (~vix_down) & (~gld_down)).astype(int)

            self.feature_names.extend(['risk_on', 'risk_off'])

        # DXY regimes (dollar strength)
        if 'DXY' in df.columns:
            dxy_ma = df['DXY'].rolling(60).mean()
            df['dxy_strong'] = (df['DXY'] > dxy_ma).astype(int)
            df['dxy_weak'] = (df['DXY'] < dxy_ma).astype(int)

            self.feature_names.extend(['dxy_strong', 'dxy_weak'])

        return df

    def _add_relative_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add relative strength vs market (SPY)."""
        if 'SPY' not in df.columns or 'Close' not in df.columns:
            return df

        windows = [20, 60]

        for window in windows:
            # Stock return
            stock_return = df['Close'].pct_change(window)

            # Market return
            spy_return = df['SPY'].pct_change(window)

            # Relative strength (stock outperformance)
            rs_col = f'rel_strength_spy_{window}d'
            df[rs_col] = stock_return - spy_return
            self.feature_names.append(rs_col)

            # Beta to SPY (rolling correlation * volatility ratio)
            stock_vol = df['Close'].pct_change().rolling(window).std()
            spy_vol = df['SPY'].pct_change().rolling(window).std()
            corr = df['Close'].pct_change().rolling(window).corr(df['SPY'].pct_change())

            beta_col = f'beta_spy_{window}d'
            df[beta_col] = corr * (stock_vol / spy_vol)
            self.feature_names.append(beta_col)

        return df

    def get_feature_names(self) -> list:
        """Get list of all macro feature names."""
        return self.feature_names


def main():
    """Test macro feature engineering."""
    import yfinance as yf

    print("="*60)
    print("MACRO FEATURE ENGINEERING TEST")
    print("="*60)

    # Download sample stock data
    print("\n[INFO] Downloading AAPL data...")
    df = yf.download('AAPL', start='2024-01-01', end='2025-11-22',
                    progress=False, auto_adjust=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    print(f"[OK] Downloaded {len(df)} days")
    print(f"[OK] Initial columns: {len(df.columns)}")

    # Add macro features
    print("\n" + "="*60)
    print("ADDING MACRO FEATURES")
    print("="*60)

    engineer = MacroFeatureEngineer()
    df_enhanced = engineer.add_all_features(df)

    print(f"\n[OK] Final columns: {len(df_enhanced.columns)}")
    print(f"[OK] Added {len(df_enhanced.columns) - len(df.columns)} macro features")

    # Show sample
    print("\n" + "="*60)
    print("SAMPLE MACRO FEATURES (Last 5 days)")
    print("="*60)

    macro_cols = ['VIX', 'DXY', 'SPY', 'vix_low', 'vix_normal', 'vix_elevated',
                  'risk_on', 'risk_off', 'rel_strength_spy_20d', 'beta_spy_20d']
    existing_cols = [c for c in macro_cols if c in df_enhanced.columns]

    if len(existing_cols) > 0:
        print(df_enhanced[existing_cols].tail())

    print(f"\n[SUCCESS] Macro feature engineering test complete!")
    print(f"           Total macro features: {len(engineer.get_feature_names())}")


if __name__ == "__main__":
    main()
