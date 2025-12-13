"""
China-Specific Macro Feature Engineer

Replaces US-centric macro features (VIX, SPY, DXY, GLD) with China-relevant indicators:
- CSI 300 / Shanghai Composite instead of SPY
- CNY/USD exchange rate instead of DXY
- China A50 futures
- Hong Kong Hang Seng Index
- Northbound/Southbound capital flows (Stock Connect)

Based on recommendations from chinese_markets_suggestion.pdf

# ============================================================================
# PROTECTED CORE MODEL - DO NOT MODIFY WITHOUT USER PERMISSION
# This file contains China macro feature engineering.
# Any changes to features or calculations require explicit user approval.
# ============================================================================
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import threading

# ============================================================================
# YAHOO FINANCE RATE LIMITER FOR CHINA MODEL
# ============================================================================
_CHINA_YF_LOCK = threading.Lock()
_CHINA_YF_LAST_REQUEST = 0
_CHINA_YF_MIN_INTERVAL = 0.5  # Minimum 0.5 seconds between requests
_CHINA_MACRO_CACHE = {}  # Cache for macro data to reduce API calls
_CHINA_MACRO_CACHE_TTL = 300  # 5 minutes cache TTL


def _rate_limited_download(ticker, start=None, end=None, period=None, max_retries=3):
    """
    Rate-limited Yahoo Finance download with caching.
    Prevents 401 errors during concurrent model training.
    """
    global _CHINA_YF_LAST_REQUEST

    # Create cache key
    cache_key = f"{ticker}_{start}_{end}_{period}"

    # Check cache first
    if cache_key in _CHINA_MACRO_CACHE:
        cached_data, cached_time = _CHINA_MACRO_CACHE[cache_key]
        if time.time() - cached_time < _CHINA_MACRO_CACHE_TTL:
            return cached_data

    # Rate limiting
    with _CHINA_YF_LOCK:
        now = time.time()
        elapsed = now - _CHINA_YF_LAST_REQUEST
        if elapsed < _CHINA_YF_MIN_INTERVAL:
            time.sleep(_CHINA_YF_MIN_INTERVAL - elapsed)
        _CHINA_YF_LAST_REQUEST = time.time()

    # Try to fetch with retries
    for attempt in range(max_retries):
        try:
            if period:
                df = yf.download(ticker, period=period, progress=False)
            else:
                df = yf.download(ticker, start=start, end=end, progress=False)

            if df is not None and not df.empty:
                # Cache the result
                _CHINA_MACRO_CACHE[cache_key] = (df, time.time())
                return df
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"  [RATE LIMIT] Retry {attempt + 1}/{max_retries} for {ticker} in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                print(f"  [RATE LIMIT] All retries failed for {ticker}: {e}")

    return pd.DataFrame()


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
            'SSEC': '000001.SS',        # Shanghai Composite (SSE Composite Index)
            'HSI': '^HSI',              # Hang Seng Index
            'CNY': 'CNY=X',             # Yuan to USD
            'GLD': 'GC=F',              # Gold (universal safe haven)
        }

        macro_dfs = []

        for name, ticker in tickers.items():
            try:
                print(f"  Downloading {name} ({ticker})...")
                # Use rate-limited download to prevent 401 errors
                df = _rate_limited_download(ticker, start=start_date, end=end_date)

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

        UPDATED: Added fallback mechanisms when macro data is unavailable (Issue #7 fix)

        Args:
            df: Stock price DataFrame with OHLC data

        Returns:
            DataFrame with added macro features
        """
        if self.macro_data is None:
            try:
                self.download_china_macro_data()
            except Exception as e:
                print(f"[WARNING] Failed to download macro data: {e}")
                print("[WARNING] Using fallback values for macro features")
                return self._add_fallback_features(df)

        # Check if we have any valid macro data
        if self.macro_data is None or len(self.macro_data) == 0:
            print("[WARNING] No macro data available, using fallbacks")
            return self._add_fallback_features(df)

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

        # Feature 7: Beta to CSI300 (60-day rolling with regularization)
        # UPDATED: Changed from 20-day to 60-day for stability (Issue #5 fix)
        # Also added regularization to prevent extreme beta values
        if 'CSI300' in macro_aligned.columns and 'Close' in df.columns:
            stock_returns = df['Close'].pct_change()
            csi300_returns = macro_aligned['CSI300'].pct_change()

            # Use 60-day rolling window for more stable beta estimation
            # min_periods=30 ensures we have enough data before calculating
            rolling_cov = stock_returns.rolling(60, min_periods=30).cov(csi300_returns)
            rolling_var = csi300_returns.rolling(60, min_periods=30).var()

            # Calculate raw beta
            raw_beta = rolling_cov / rolling_var

            # Regularize extreme betas: clamp to [-1, 3] range
            # Most stocks have beta between 0.5 and 1.5, extreme values are likely noise
            df['beta_csi300_60d'] = raw_beta.clip(lower=-1.0, upper=3.0)
            df['beta_csi300_60d'] = df['beta_csi300_60d'].fillna(1.0).replace([np.inf, -np.inf], 1.0)

            # Keep old column name for backward compatibility
            df['beta_csi300_20d'] = df['beta_csi300_60d']

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
            'CSI300_momentum_20d', 'beta_csi300_20d', 'beta_csi300_60d',
            'CNY_momentum_5d', 'HSI_SSEC_ratio'
        ]]

        for col in feature_cols:
            df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)

        print(f"[OK] Added {len(feature_cols)} China-specific macro features")

        return df

    def _add_fallback_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add fallback macro features when real macro data is unavailable.

        Uses stock's own price data to generate proxy features (Issue #7 fix).
        This ensures the model can still run even if CSI300/HSI data is missing.

        Args:
            df: Stock price DataFrame with OHLC data

        Returns:
            DataFrame with fallback macro features (neutral/derived values)
        """
        print("[INFO] Generating fallback macro features from stock data...")

        # Use stock's own momentum as proxy for market momentum
        if 'Close' in df.columns:
            # CSI300 proxy: stock's own 5-day momentum (assume market correlation)
            df['CSI300_dist_ma_5d'] = 0  # Neutral - no market info
            df['SSEC_dist_ma_5d'] = 0
            df['HSI_dist_ma_5d'] = 0

            # Use stock's own 20-day MA distance as proxy
            if len(df) >= 20:
                ma_20 = df['Close'].rolling(20).mean()
                df['CSI300_dist_ma_20d'] = (df['Close'] - ma_20) / ma_20
                df['CSI300_dist_ma_20d'] = df['CSI300_dist_ma_20d'].fillna(0)
            else:
                df['CSI300_dist_ma_20d'] = 0

            # Momentum proxies from stock's own returns
            if len(df) >= 5:
                df['CSI300_momentum_5d'] = df['Close'].pct_change(5).fillna(0)
            else:
                df['CSI300_momentum_5d'] = 0

            if len(df) >= 20:
                df['CSI300_momentum_20d'] = df['Close'].pct_change(20).fillna(0)
            else:
                df['CSI300_momentum_20d'] = 0

            # Beta: assume market beta = 1.0 (neutral)
            df['beta_csi300_20d'] = 1.0
            df['beta_csi300_60d'] = 1.0

            # CNY: use neutral value (approximate CNY/USD rate)
            df['CNY'] = 7.2  # Approximate CNY/USD rate
            df['CNY_momentum_5d'] = 0  # No currency movement data

            # HSI/SSEC ratio: neutral
            df['HSI_SSEC_ratio'] = 1.0

        print(f"[OK] Added fallback macro features (10 neutral/proxy values)")

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
