"""
China-Specific Macro Feature Engineer

Replaces US-centric macro features (VIX, SPY, DXY, GLD) with China-relevant indicators:
- CSI 300 / Shanghai Composite instead of SPY
- CNY/USD exchange rate instead of DXY
- China A50 futures
- Hong Kong Hang Seng Index
- Northbound/Southbound capital flows (Stock Connect)

Based on recommendations from chinese_markets_suggestion.pdf
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


class ChinaMacroFeatureEngineer:
    """
    Engineers China-specific macro features for stock prediction.

    Replaces US-centric features with China-relevant indicators optimized
    for Chinese A-shares and Hong Kong-listed stocks.
    """

    def __init__(self):
        self.macro_data = None

    def download_china_macro_data(self, start_date=None, end_date=None):
        """
        Download China-specific macro indicators.

        Tickers:
        - 000300.SS: CSI 300 Index (China's S&P 500 equivalent)
        - ^SSEC: Shanghai Composite Index
        - ^HSI: Hang Seng Index (Hong Kong)
        - CNY=X: Chinese Yuan to USD exchange rate
        - GC=F: Gold futures (keep as safe haven)
        """
        print("[INFO] Downloading China-specific macro data...")

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        tickers = {
            'CSI300': '000300.SS',      # CSI 300 Index (China large cap)
            'SSEC': '000001.SS',        # Shanghai Composite Index (^SSEC often fails)
            'HSI': '^HSI',              # Hang Seng Index
            'CNY': 'CNY=X',             # Yuan to USD
            'GLD': 'GC=F',              # Gold (universal safe haven)
        }

        macro_dfs = []

        for name, ticker in tickers.items():
            try:
                print(f"  Downloading {name} ({ticker})...")
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)

                if df.empty:
                    print(f"  [WARNING] No data for {name}, skipping")
                    continue

                # Use Close price, rename column
                if isinstance(df.columns, pd.MultiIndex):
                    close_series = df['Close']
                else:
                    close_series = df['Close']

                # Rename the series
                close_series.name = name
                macro_dfs.append(close_series)

            except Exception as e:
                print(f"  [ERROR] Failed to download {name}: {e}")

        if len(macro_dfs) == 0:
            raise ValueError("Failed to download any macro data")

        # Merge all macro series
        self.macro_data = pd.concat(macro_dfs, axis=1)
        self.macro_data = self.macro_data.ffill().bfill()

        print(f"[OK] Downloaded China macro data: {len(self.macro_data)} days, {len(self.macro_data.columns)} indicators")
        return self.macro_data

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add top 10 China-specific macro features to DataFrame.

        Features prioritize:
        1. Market index momentum and distances (CSI300, SSEC, HSI)
        2. Currency strength (CNY)
        3. Safe haven indicator (Gold)
        4. Cross-market correlations

        Args:
            df: Stock price DataFrame with OHLC data

        Returns:
            DataFrame with added macro features
        """
        if self.macro_data is None:
            self.download_china_macro_data()

        # Align macro data to stock data dates
        macro_aligned = self.macro_data.reindex(df.index).ffill()

        # Ensure no missing values
        macro_aligned = macro_aligned.ffill().bfill()

        print(f"[INFO] Engineering China-specific macro features...")

        # Feature 1-3: Distance from MA for CSI300, SSEC, HSI (5-day)
        for col in ['CSI300', 'SSEC', 'HSI']:
            if col in macro_aligned.columns:
                ma_col = macro_aligned[col].rolling(5).mean()
                dist_col = f'{col}_dist_ma_5d'
                df[dist_col] = (macro_aligned[col] - ma_col) / ma_col
                df[dist_col] = df[dist_col].fillna(0)

        # Feature 4: CNY exchange rate (raw value)
        if 'CNY' in macro_aligned.columns:
            df['CNY'] = macro_aligned['CNY']
            df['CNY'] = df['CNY'].ffill()

        # Feature 5: CSI300 distance from MA (20-day)
        if 'CSI300' in macro_aligned.columns:
            ma_20 = macro_aligned['CSI300'].rolling(20).mean()
            df['CSI300_dist_ma_20d'] = (macro_aligned['CSI300'] - ma_20) / ma_20
            df['CSI300_dist_ma_20d'] = df['CSI300_dist_ma_20d'].fillna(0)

        # Feature 6, 8: CSI300 momentum (5-day and 20-day)
        if 'CSI300' in macro_aligned.columns:
            df['CSI300_momentum_5d'] = macro_aligned['CSI300'].pct_change(5)
            df['CSI300_momentum_20d'] = macro_aligned['CSI300'].pct_change(20)
            df['CSI300_momentum_5d'] = df['CSI300_momentum_5d'].fillna(0)
            df['CSI300_momentum_20d'] = df['CSI300_momentum_20d'].fillna(0)

        # Feature 7: Beta to CSI300 (20-day rolling)
        if 'CSI300' in macro_aligned.columns and 'Close' in df.columns:
            stock_returns = df['Close'].pct_change()
            csi300_returns = macro_aligned['CSI300'].pct_change()

            # Rolling covariance / variance
            rolling_cov = stock_returns.rolling(20).cov(csi300_returns)
            rolling_var = csi300_returns.rolling(20).var()

            df['beta_csi300_20d'] = rolling_cov / rolling_var
            df['beta_csi300_20d'] = df['beta_csi300_20d'].fillna(0).replace([np.inf, -np.inf], 0)

        # Feature 9: CNY momentum (5-day)
        if 'CNY' in macro_aligned.columns:
            df['CNY_momentum_5d'] = macro_aligned['CNY'].pct_change(5)
            df['CNY_momentum_5d'] = df['CNY_momentum_5d'].fillna(0)

        # Feature 10: HSI/SSEC ratio (Hong Kong vs Shanghai sentiment)
        if 'HSI' in macro_aligned.columns and 'SSEC' in macro_aligned.columns:
            df['HSI_SSEC_ratio'] = macro_aligned['HSI'] / macro_aligned['SSEC']
            df['HSI_SSEC_ratio'] = df['HSI_SSEC_ratio'].ffill().fillna(1.0)

        # Clean up any remaining NaN/inf
        feature_cols = [col for col in df.columns if col in [
            'CSI300_dist_ma_5d', 'SSEC_dist_ma_5d', 'HSI_dist_ma_5d',
            'CNY', 'CSI300_dist_ma_20d', 'CSI300_momentum_5d',
            'CSI300_momentum_20d', 'beta_csi300_20d', 'CNY_momentum_5d',
            'HSI_SSEC_ratio'
        ]]

        for col in feature_cols:
            df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)

        print(f"[OK] Added {len(feature_cols)} China-specific macro features")

        return df


if __name__ == "__main__":
    # Test the China macro feature engineer
    print("Testing ChinaMacroFeatureEngineer...")

    # Download sample stock data
    ticker = '0700.HK'  # Tencent
    df = yf.download(ticker, start='2023-01-01', end='2025-11-22', progress=False)

    print(f"\nOriginal shape: {df.shape}")

    # Add China macro features
    eng = ChinaMacroFeatureEngineer()
    df = eng.add_all_features(df)

    print(f"After features shape: {df.shape}")
    print(f"\nNew features added:")

    china_features = [col for col in df.columns if any(x in col for x in ['CSI300', 'SSEC', 'HSI', 'CNY'])]
    print(china_features)

    print(f"\nSample values (last 5 rows):")
    print(df[china_features].tail())
