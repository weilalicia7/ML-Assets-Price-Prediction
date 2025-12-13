"""
Commodity Macro Features Fetcher
Add macro economic indicators for commodity prediction

Commodity-specific features:
- Gold: VIX, Real Rates, USD Index
- Oil: USD Index, EIA Inventory, OPEC Production
- Copper: China PMI, Industrial Production
- Silver: Similar to Gold
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

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False


class CommodityMacroFeaturesFetcher:
    """
    Fetch macro features for commodity prediction.
    """

    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize commodity macro features fetcher.

        Args:
            fred_api_key: FRED API key
        """
        self.fred_api_key = fred_api_key
        self.fred_client = None

        if fred_api_key and FRED_AVAILABLE:
            try:
                self.fred_client = Fred(api_key=fred_api_key)
            except Exception as e:
                warnings.warn(f"Could not initialize FRED client: {e}")

    def fetch_vix(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch VIX - higher VIX = flight to gold."""
        if not YFINANCE_AVAILABLE:
            return pd.DataFrame()

        print("Fetching VIX...")
        try:
            ticker = yf.Ticker('^VIX')
            data = ticker.history(start=start_date, end=end_date) if start_date else ticker.history(period='1y')
            if data.empty:
                return pd.DataFrame()
            data = data[['Close']].rename(columns={'Close': 'VIX'})
            print(f"  Fetched {len(data)} bars, VIX: {data['VIX'].iloc[-1]:.2f}")
            return data
        except Exception as e:
            warnings.warn(f"Failed to fetch VIX: {e}")
            return pd.DataFrame()

    def fetch_dxy(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch USD Index - stronger USD = lower commodity prices."""
        if not YFINANCE_AVAILABLE:
            return pd.DataFrame()

        print("Fetching DXY (USD Index)...")
        try:
            ticker = yf.Ticker('DX-Y.NYB')
            data = ticker.history(start=start_date, end=end_date) if start_date else ticker.history(period='1y')
            if data.empty:
                return pd.DataFrame()
            data = data[['Close']].rename(columns={'Close': 'DXY'})
            print(f"  Fetched {len(data)} bars, DXY: {data['DXY'].iloc[-1]:.2f}")
            return data
        except Exception as e:
            warnings.warn(f"Failed to fetch DXY: {e}")
            return pd.DataFrame()

    def fetch_real_rates(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch real interest rates (nominal - inflation) - negative rates support gold."""
        if not self.fred_client:
            return pd.DataFrame()

        print("Fetching Real Rates...")
        try:
            # 10Y Treasury Yield
            nominal = self.fred_client.get_series('DGS10', observation_start=start_date, observation_end=end_date)
            # 10Y Breakeven Inflation
            breakeven = self.fred_client.get_series('T10YIE', observation_start=start_date, observation_end=end_date)

            df = pd.DataFrame({'Nominal': nominal, 'Breakeven': breakeven})
            df['Real_Rate'] = df['Nominal'] - df['Breakeven']

            print(f"  Fetched {len(df)} observations, Real Rate: {df['Real_Rate'].iloc[-1]:.2f}%")
            return df[['Real_Rate']]
        except Exception as e:
            warnings.warn(f"Failed to fetch real rates: {e}")
            return pd.DataFrame()

    def fetch_industrial_production(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch US Industrial Production - indicates copper demand."""
        if not self.fred_client:
            return pd.DataFrame()

        print("Fetching Industrial Production...")
        try:
            ip = self.fred_client.get_series('INDPRO', observation_start=start_date, observation_end=end_date)
            data = ip.to_frame('Industrial_Production')
            print(f"  Fetched {len(data)} observations")
            return data
        except Exception as e:
            warnings.warn(f"Failed to fetch industrial production: {e}")
            return pd.DataFrame()

    def create_synthetic_commodity_features(self, index: pd.DatetimeIndex, commodity: str = 'gold') -> pd.DataFrame:
        """Create synthetic commodity features."""
        print(f"Creating synthetic {commodity} macro features...")

        n = len(index)

        # VIX
        vix = 20 + np.random.randn(n).cumsum() * 0.5
        vix = np.clip(vix, 12, 35)

        # DXY
        dxy = 102 + np.random.randn(n).cumsum() * 0.3
        dxy = np.clip(dxy, 95, 110)

        # Real Rates
        real_rate = 1.5 + np.random.randn(n).cumsum() * 0.1
        real_rate = np.clip(real_rate, -1, 3)

        data = pd.DataFrame({
            'VIX': vix,
            'DXY': dxy,
            'Real_Rate': real_rate
        }, index=index)

        print(f"  Created {len(data)} synthetic observations")
        return data

    def fetch_gold_features(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_synthetic_fallback: bool = True
    ) -> pd.DataFrame:
        """Fetch all features for gold prediction."""
        features = {}

        # VIX (fear gauge)
        vix = self.fetch_vix(start_date, end_date)
        if not vix.empty:
            features['VIX'] = vix

        # DXY (USD strength)
        dxy = self.fetch_dxy(start_date, end_date)
        if not dxy.empty:
            features['DXY'] = dxy

        # Real rates
        if self.fred_client:
            real_rates = self.fetch_real_rates(start_date, end_date)
            if not real_rates.empty:
                features['Real_Rate'] = real_rates

        if features:
            combined = pd.concat(features.values(), axis=1)
            combined = combined.fillna(method='ffill').fillna(method='bfill')
            print(f"\nFetched {len(features)} real gold features")
            return combined
        elif use_synthetic_fallback:
            warnings.warn("Using synthetic gold features")
            end_date = pd.to_datetime(end_date) if end_date else datetime.now()
            start_date = pd.to_datetime(start_date) if start_date else end_date - timedelta(days=365)
            index = pd.date_range(start=start_date, end=end_date, freq='D')
            return self.create_synthetic_commodity_features(index, 'gold')
        else:
            return pd.DataFrame()

    def fetch_oil_features(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_synthetic_fallback: bool = True
    ) -> pd.DataFrame:
        """Fetch all features for oil prediction."""
        features = {}

        # DXY
        dxy = self.fetch_dxy(start_date, end_date)
        if not dxy.empty:
            features['DXY'] = dxy

        # VIX (risk sentiment)
        vix = self.fetch_vix(start_date, end_date)
        if not vix.empty:
            features['VIX'] = vix

        if features:
            combined = pd.concat(features.values(), axis=1)
            combined = combined.fillna(method='ffill').fillna(method='bfill')
            print(f"\nFetched {len(features)} real oil features")
            return combined
        elif use_synthetic_fallback:
            warnings.warn("Using synthetic oil features")
            end_date = pd.to_datetime(end_date) if end_date else datetime.now()
            start_date = pd.to_datetime(start_date) if start_date else end_date - timedelta(days=365)
            index = pd.date_range(start=start_date, end=end_date, freq='D')
            return self.create_synthetic_commodity_features(index, 'oil')
        else:
            return pd.DataFrame()

    def fetch_copper_features(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_synthetic_fallback: bool = True
    ) -> pd.DataFrame:
        """Fetch all features for copper prediction."""
        features = {}

        # DXY
        dxy = self.fetch_dxy(start_date, end_date)
        if not dxy.empty:
            features['DXY'] = dxy

        # Industrial Production
        if self.fred_client:
            ip = self.fetch_industrial_production(start_date, end_date)
            if not ip.empty:
                features['Industrial_Production'] = ip

        if features:
            combined = pd.concat(features.values(), axis=1)
            combined = combined.fillna(method='ffill').fillna(method='bfill')
            print(f"\nFetched {len(features)} real copper features")
            return combined
        elif use_synthetic_fallback:
            warnings.warn("Using synthetic copper features")
            end_date = pd.to_datetime(end_date) if end_date else datetime.now()
            start_date = pd.to_datetime(start_date) if start_date else end_date - timedelta(days=365)
            index = pd.date_range(start=start_date, end=end_date, freq='D')
            return self.create_synthetic_commodity_features(index, 'copper')
        else:
            return pd.DataFrame()

    def align_features_to_asset(
        self,
        asset_data: pd.DataFrame,
        macro_features: pd.DataFrame
    ) -> pd.DataFrame:
        """Align macro features to asset data frequency."""
        print("\nAligning macro features to asset frequency...")

        aligned = macro_features.reindex(asset_data.index, method='ffill')
        aligned = aligned.fillna(method='bfill')

        print(f"  Aligned {len(aligned)} observations")
        print(f"  Features: {list(aligned.columns)}")

        return aligned
