"""
Selective Macro Feature Engineering

Only generates the TOP 10 Phase 4 features based on feature importance analysis:
1. GLD_dist_ma_5d
2. VIX_dist_ma_5d
3. SPY_dist_ma_5d
4. DXY
5. VIX_dist_ma_20d
6. VIX_momentum_5d
7. beta_spy_20d
8. VIX_momentum_20d
9. DXY_momentum_5d
10. VIX

These were the highest importance Phase 4 features.
All other Phase 4 features are excluded to reduce noise.
"""

import pandas as pd
import numpy as np
import os

class SelectiveMacroFeatureEngineer:
    """
    Engineers only the top 10 macro features for stock prediction.

    Uses pre-downloaded macro indicators from data/macro_indicators.csv
    """

    def __init__(self, macro_data_path='data/macro_indicators.csv'):
        """
        Initialize selective macro feature engineer.

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
            print(f"[INFO] Loaded macro data: {len(self.macro_data)} days")
        except Exception as e:
            print(f"[ERROR] Failed to load macro data: {e}")
            self.macro_data = None

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ONLY top 10 macro features to DataFrame.

        Args:
            df: DataFrame with OHLCV data (must have DatetimeIndex)

        Returns:
            DataFrame with selective macro features added
        """
        if self.macro_data is None:
            print("[WARNING] No macro data available, skipping macro features")
            return df

        print(f"[INFO] Engineering selective macro features (top 10 only)...")

        df = df.copy()
        initial_cols = len(df.columns)

        # Ensure both indices are timezone-naive to avoid comparison warnings
        df_index = df.index.copy()
        macro_data = self.macro_data.copy()

        # Convert timezone-aware indices to timezone-naive if needed
        if df_index.tz is not None:
            df_index = df_index.tz_localize(None)
            df.index = df_index
        if macro_data.index.tz is not None:
            macro_data.index = macro_data.index.tz_localize(None)

        # Align macro data to stock data dates (forward fill for weekends/holidays)
        macro_aligned = macro_data.reindex(df.index, method='ffill')

        # Feature 1-3: Distance from MA for GLD, VIX, SPY (5-day)
        for col in ['GLD', 'VIX', 'SPY']:
            if col in macro_aligned.columns:
                ma_col = macro_aligned[col].rolling(5).mean()
                dist_col = f'{col}_dist_ma_5d'
                df[dist_col] = (macro_aligned[col] - ma_col) / ma_col
                self.feature_names.append(dist_col)

        # Feature 4: DXY raw value
        if 'DXY' in macro_aligned.columns:
            df['DXY'] = macro_aligned['DXY']
            self.feature_names.append('DXY')

        # Feature 5: VIX distance from MA (20-day)
        if 'VIX' in macro_aligned.columns:
            ma_20 = macro_aligned['VIX'].rolling(20).mean()
            dist_col = 'VIX_dist_ma_20d'
            df[dist_col] = (macro_aligned['VIX'] - ma_20) / ma_20
            self.feature_names.append(dist_col)

        # Feature 6, 8: VIX momentum (5-day and 20-day)
        if 'VIX' in macro_aligned.columns:
            df['VIX_momentum_5d'] = macro_aligned['VIX'].pct_change(5)
            df['VIX_momentum_20d'] = macro_aligned['VIX'].pct_change(20)
            self.feature_names.extend(['VIX_momentum_5d', 'VIX_momentum_20d'])

        # Feature 7: Beta to SPY (20-day)
        if 'SPY' in macro_aligned.columns and 'Close' in df.columns:
            stock_vol = df['Close'].pct_change().rolling(20).std()
            spy_vol = macro_aligned['SPY'].pct_change().rolling(20).std()
            corr = df['Close'].pct_change().rolling(20).corr(macro_aligned['SPY'].pct_change())

            df['beta_spy_20d'] = corr * (stock_vol / spy_vol)
            self.feature_names.append('beta_spy_20d')

        # Feature 9: DXY momentum (5-day)
        if 'DXY' in macro_aligned.columns:
            df['DXY_momentum_5d'] = macro_aligned['DXY'].pct_change(5)
            self.feature_names.append('DXY_momentum_5d')

        # Feature 10: VIX raw value
        if 'VIX' in macro_aligned.columns:
            df['VIX'] = macro_aligned['VIX']
            if 'VIX' not in self.feature_names:  # Only add if not already added
                self.feature_names.append('VIX')

        added_cols = len(df.columns) - initial_cols
        print(f"[OK] Added {added_cols} selective macro features (Phase 4 top performers)")

        return df

    def get_feature_names(self) -> list:
        """Get list of all selective macro feature names."""
        return self.feature_names


def main():
    """Test selective macro feature engineering."""
    import yfinance as yf

    print("="*60)
    print("SELECTIVE MACRO FEATURE ENGINEERING TEST")
    print("="*60)
    print("\nGenerating ONLY top 10 Phase 4 features:")
    print("1. GLD_dist_ma_5d")
    print("2. VIX_dist_ma_5d")
    print("3. SPY_dist_ma_5d")
    print("4. DXY")
    print("5. VIX_dist_ma_20d")
    print("6. VIX_momentum_5d")
    print("7. beta_spy_20d")
    print("8. VIX_momentum_20d")
    print("9. DXY_momentum_5d")
    print("10. VIX")

    # Download sample stock data
    print("\n[INFO] Downloading AAPL data...")
    df = yf.download('AAPL', start='2024-01-01', end='2025-11-22',
                    progress=False, auto_adjust=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    print(f"[OK] Downloaded {len(df)} days")
    print(f"[OK] Initial columns: {len(df.columns)}")

    # Add selective macro features
    print("\n" + "="*60)
    print("ADDING SELECTIVE MACRO FEATURES")
    print("="*60)

    engineer = SelectiveMacroFeatureEngineer()
    df_enhanced = engineer.add_all_features(df)

    print(f"\n[OK] Final columns: {len(df_enhanced.columns)}")
    print(f"[OK] Added {len(df_enhanced.columns) - len(df.columns)} features")

    # Show sample
    print("\n" + "="*60)
    print("SAMPLE FEATURES (Last 5 days)")
    print("="*60)

    feature_cols = engineer.get_feature_names()
    existing_cols = [c for c in feature_cols if c in df_enhanced.columns]

    if len(existing_cols) > 0:
        print(df_enhanced[existing_cols].tail())

    print(f"\n[SUCCESS] Selective macro feature engineering complete!")
    print(f"           Total features: {len(engineer.get_feature_names())}")


if __name__ == "__main__":
    main()
