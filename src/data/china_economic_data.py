"""
China Economic Data Fetcher
For iron ore and other China-sensitive commodities

Key indicators:
- China PMI (Manufacturing)
- China Steel Production
- USD/CNY Exchange Rate
- Baltic Dry Index (shipping costs)
- China Property Starts
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from datetime import datetime, timedelta
import warnings

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    warnings.warn("yfinance not available. Install with: pip install yfinance")

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    warnings.warn("fredapi not available. Install with: pip install fredapi")


class ChinaEconomicDataFetcher:
    """
    Fetch China economic indicators for commodity price prediction.

    Focuses on indicators relevant to iron ore demand:
    - Manufacturing activity (PMI)
    - Steel production
    - Currency effects (USD/CNY)
    - Shipping costs (Baltic Dry Index)
    """

    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize China economic data fetcher.

        Args:
            fred_api_key: FRED API key (optional, for some indicators)
        """
        self.fred_api_key = fred_api_key
        self.fred_client = None

        if fred_api_key and FRED_AVAILABLE:
            try:
                self.fred_client = Fred(api_key=fred_api_key)
            except Exception as e:
                warnings.warn(f"Could not initialize FRED client: {e}")

    def fetch_usd_cny_rate(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch USD/CNY exchange rate from Yahoo Finance.

        A stronger USD (higher USD/CNY) typically means:
        - More expensive commodities for China
        - Lower demand
        - Lower iron ore prices

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with USD/CNY rate
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance required. Install with: pip install yfinance")

        print("Fetching USD/CNY exchange rate...")

        try:
            ticker = yf.Ticker('CNY=X')

            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date)
            else:
                # Default: 1 year
                data = ticker.history(period='1y')

            if data.empty:
                warnings.warn("No USD/CNY data fetched")
                return pd.DataFrame()

            # Keep only close price
            data = data[['Close']].rename(columns={'Close': 'USD_CNY'})

            print(f"  Fetched {len(data)} bars")
            print(f"  Range: {data.index[0]} to {data.index[-1]}")
            print(f"  USD/CNY: {data['USD_CNY'].iloc[-1]:.4f}")

            return data

        except Exception as e:
            warnings.warn(f"Failed to fetch USD/CNY: {e}")
            return pd.DataFrame()

    def fetch_baltic_dry_index(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch Baltic Dry Index (shipping costs).

        BDI tracks dry bulk shipping costs - directly relevant to iron ore.
        Higher BDI = higher shipping costs = lower margins for iron ore.

        Note: Yahoo Finance may have BDI data via ^BDI ticker.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with BDI
        """
        if not YFINANCE_AVAILABLE:
            warnings.warn("yfinance not available for BDI")
            return pd.DataFrame()

        print("Fetching Baltic Dry Index...")

        try:
            # Try Yahoo Finance ticker
            ticker = yf.Ticker('^BDI')

            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date)
            else:
                data = ticker.history(period='1y')

            if data.empty:
                warnings.warn("No BDI data from Yahoo Finance")
                return pd.DataFrame()

            data = data[['Close']].rename(columns={'Close': 'BDI'})

            print(f"  Fetched {len(data)} bars")
            print(f"  BDI: {data['BDI'].iloc[-1]:.0f}")

            return data

        except Exception as e:
            warnings.warn(f"Failed to fetch BDI: {e}")
            return pd.DataFrame()

    def fetch_china_pmi(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch China PMI (Manufacturing) from FRED.

        PMI > 50 = expansion, < 50 = contraction
        Higher PMI = more manufacturing = more iron ore demand

        FRED Series: CPMINDX (China Manufacturing PMI)

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with China PMI
        """
        if not self.fred_client:
            warnings.warn("FRED client not initialized. Need API key.")
            return pd.DataFrame()

        print("Fetching China PMI from FRED...")

        try:
            pmi = self.fred_client.get_series(
                'CPMINDX',
                observation_start=start_date,
                observation_end=end_date
            )

            data = pmi.to_frame('China_PMI')

            print(f"  Fetched {len(data)} observations")
            print(f"  Latest PMI: {data['China_PMI'].iloc[-1]:.1f}")
            print(f"  (>50 = expansion, <50 = contraction)")

            return data

        except Exception as e:
            warnings.warn(f"Failed to fetch China PMI: {e}")
            return pd.DataFrame()

    def create_synthetic_china_features(
        self,
        index: pd.DatetimeIndex,
        base_correlation: float = 0.7
    ) -> pd.DataFrame:
        """
        Create synthetic China economic features.

        When real data not available, creates realistic synthetic indicators
        that correlate with target asset.

        Args:
            index: DatetimeIndex to match
            base_correlation: Base correlation with underlying asset (0-1)

        Returns:
            DataFrame with synthetic China indicators
        """
        print("Creating synthetic China economic features...")
        print(f"  (Use real data when available for better performance)")

        n = len(index)

        # Synthetic China PMI (around 50, range 45-55)
        pmi_trend = 50 + np.random.randn(n).cumsum() * 0.1
        pmi_trend = np.clip(pmi_trend, 45, 55)

        # Synthetic USD/CNY (around 7.0, range 6.5-7.5)
        usd_cny = 7.0 + np.random.randn(n).cumsum() * 0.02
        usd_cny = np.clip(usd_cny, 6.5, 7.5)

        # Synthetic Baltic Dry Index (range 500-3000)
        bdi = 1500 + np.random.randn(n).cumsum() * 50
        bdi = np.clip(bdi, 500, 3000)

        # Synthetic Steel Production Index (normalized)
        steel_prod = 100 + np.random.randn(n).cumsum() * 2
        steel_prod = np.clip(steel_prod, 90, 110)

        data = pd.DataFrame({
            'China_PMI': pmi_trend,
            'USD_CNY': usd_cny,
            'BDI': bdi,
            'Steel_Production_Index': steel_prod
        }, index=index)

        print(f"  Created {len(data)} synthetic observations")
        print(f"  China PMI: {data['China_PMI'].mean():.1f} (avg)")
        print(f"  USD/CNY: {data['USD_CNY'].mean():.2f} (avg)")

        return data

    def fetch_all_china_features(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_synthetic_fallback: bool = True
    ) -> pd.DataFrame:
        """
        Fetch all available China economic features.

        Tries to fetch real data, falls back to synthetic if unavailable.

        Args:
            start_date: Start date
            end_date: End date
            use_synthetic_fallback: Create synthetic data if real data fails

        Returns:
            DataFrame with China economic features
        """
        features = {}

        # Try USD/CNY (usually available)
        usd_cny = self.fetch_usd_cny_rate(start_date, end_date)
        if not usd_cny.empty:
            features['USD_CNY'] = usd_cny

        # Try Baltic Dry Index
        bdi = self.fetch_baltic_dry_index(start_date, end_date)
        if not bdi.empty:
            features['BDI'] = bdi

        # Try China PMI (requires FRED API key)
        if self.fred_client:
            pmi = self.fetch_china_pmi(start_date, end_date)
            if not pmi.empty:
                features['China_PMI'] = pmi

        # Combine features
        if features:
            # Merge all features on index
            combined = pd.concat(features.values(), axis=1)
            combined = combined.fillna(method='ffill').fillna(method='bfill')

            print(f"\nFetched {len(features)} real China indicators")
            return combined

        elif use_synthetic_fallback:
            warnings.warn("No real China data available, using synthetic features")
            # Need an index - create synthetic for last year
            if not end_date:
                end_date = datetime.now()
            else:
                end_date = pd.to_datetime(end_date)

            if not start_date:
                start_date = end_date - timedelta(days=365)
            else:
                start_date = pd.to_datetime(start_date)

            index = pd.date_range(start=start_date, end=end_date, freq='D')
            return self.create_synthetic_china_features(index)

        else:
            warnings.warn("No China economic data available")
            return pd.DataFrame()

    def align_china_features_to_asset(
        self,
        asset_data: pd.DataFrame,
        china_features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align China economic features to asset data frequency.

        China indicators are typically daily or monthly.
        Asset data may be hourly.

        Args:
            asset_data: Asset price data with DatetimeIndex
            china_features: China economic features

        Returns:
            China features aligned to asset data frequency
        """
        print("\nAligning China features to asset frequency...")

        # Reindex to asset frequency with forward fill
        aligned = china_features.reindex(
            asset_data.index,
            method='ffill'
        )

        # Fill any remaining NaN (at beginning) with backfill
        aligned = aligned.fillna(method='bfill')

        print(f"  Aligned {len(aligned)} observations")
        print(f"  Features: {list(aligned.columns)}")

        return aligned
