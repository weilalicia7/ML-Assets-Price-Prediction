"""
Forex Features Fetcher
Add interest rate differentials and carry trade signals for forex pairs

Key features:
- Interest rate differentials
- Central bank policy divergence
- Carry trade signals
- Risk sentiment indicators
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
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


class ForexFeaturesFetcher:
    """
    Fetch forex-specific features for currency pair prediction.

    Focuses on fundamental drivers:
    - Interest rate differentials
    - Risk sentiment (VIX)
    - Dollar strength (DXY)
    """

    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize forex features fetcher.

        Args:
            fred_api_key: FRED API key (optional, for interest rates)
        """
        self.fred_api_key = fred_api_key
        self.fred_client = None

        if fred_api_key and FRED_AVAILABLE:
            try:
                self.fred_client = Fred(api_key=fred_api_key)
            except Exception as e:
                warnings.warn(f"Could not initialize FRED client: {e}")

    def fetch_vix(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch VIX (market fear gauge).

        Higher VIX = higher risk aversion = flight to safe haven currencies (USD, JPY, CHF).

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with VIX
        """
        if not YFINANCE_AVAILABLE:
            warnings.warn("yfinance not available for VIX")
            return pd.DataFrame()

        print("Fetching VIX (Risk Sentiment)...")

        try:
            ticker = yf.Ticker('^VIX')

            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date)
            else:
                data = ticker.history(period='1y')

            if data.empty:
                warnings.warn("No VIX data fetched")
                return pd.DataFrame()

            data = data[['Close']].rename(columns={'Close': 'VIX'})

            print(f"  Fetched {len(data)} bars")
            print(f"  VIX: {data['VIX'].iloc[-1]:.2f}")

            return data

        except Exception as e:
            warnings.warn(f"Failed to fetch VIX: {e}")
            return pd.DataFrame()

    def fetch_dxy(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch DXY (US Dollar Index).

        Measures USD strength against basket of currencies.
        Higher DXY = stronger USD.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with DXY
        """
        if not YFINANCE_AVAILABLE:
            warnings.warn("yfinance not available for DXY")
            return pd.DataFrame()

        print("Fetching DXY (US Dollar Index)...")

        try:
            ticker = yf.Ticker('DX-Y.NYB')

            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date)
            else:
                data = ticker.history(period='1y')

            if data.empty:
                warnings.warn("No DXY data from Yahoo Finance")
                return pd.DataFrame()

            data = data[['Close']].rename(columns={'Close': 'DXY'})

            print(f"  Fetched {len(data)} bars")
            print(f"  DXY: {data['DXY'].iloc[-1]:.2f}")

            return data

        except Exception as e:
            warnings.warn(f"Failed to fetch DXY: {e}")
            return pd.DataFrame()

    def fetch_interest_rate_differential(
        self,
        base_currency: str,
        quote_currency: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch interest rate differential between two currencies.

        FRED Series IDs:
        - USD: DFF (Fed Funds Rate)
        - EUR: ECBDFR (ECB Deposit Facility Rate)
        - GBP: GBPONTD (Bank of England Official Rate)
        - JPY: IRSTCI01JPM156N (Japan Policy Rate)
        - CHF: IRSTCI01CHM156N (Swiss Policy Rate)

        Args:
            base_currency: Base currency (e.g., 'EUR')
            quote_currency: Quote currency (e.g., 'USD')
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with interest rate differential
        """
        if not self.fred_client:
            warnings.warn("FRED client not initialized for interest rates")
            return pd.DataFrame()

        # FRED series mapping
        rate_series = {
            'USD': 'DFF',
            'EUR': 'ECBDFR',
            'GBP': 'GBPONTD',
            'JPY': 'IRSTCI01JPM156N',
            'CHF': 'IRSTCI01CHM156N'
        }

        if base_currency not in rate_series or quote_currency not in rate_series:
            warnings.warn(f"Interest rate series not available for {base_currency}/{quote_currency}")
            return pd.DataFrame()

        print(f"Fetching interest rate differential ({base_currency} - {quote_currency})...")

        try:
            base_rate = self.fred_client.get_series(
                rate_series[base_currency],
                observation_start=start_date,
                observation_end=end_date
            )

            quote_rate = self.fred_client.get_series(
                rate_series[quote_currency],
                observation_start=start_date,
                observation_end=end_date
            )

            # Align and calculate differential
            df = pd.DataFrame({
                f'{base_currency}_Rate': base_rate,
                f'{quote_currency}_Rate': quote_rate
            })

            df['Interest_Rate_Diff'] = df[f'{base_currency}_Rate'] - df[f'{quote_currency}_Rate']

            print(f"  Fetched {len(df)} observations")
            print(f"  Latest differential: {df['Interest_Rate_Diff'].iloc[-1]:.2f}%")

            return df[['Interest_Rate_Diff']]

        except Exception as e:
            warnings.warn(f"Failed to fetch interest rate differential: {e}")
            return pd.DataFrame()

    def create_synthetic_forex_features(
        self,
        index: pd.DatetimeIndex,
        pair: str = 'EURUSD'
    ) -> pd.DataFrame:
        """
        Create synthetic forex features when real data unavailable.

        Args:
            index: DatetimeIndex to match
            pair: Currency pair (e.g., 'EURUSD')

        Returns:
            DataFrame with synthetic forex features
        """
        print("Creating synthetic forex features...")
        print(f"  (Use real data when available for better performance)")

        n = len(index)

        # Synthetic VIX (range 12-30)
        vix = 20 + np.random.randn(n).cumsum() * 0.5
        vix = np.clip(vix, 12, 35)

        # Synthetic DXY (range 95-110)
        dxy = 102 + np.random.randn(n).cumsum() * 0.3
        dxy = np.clip(dxy, 95, 110)

        # Synthetic interest rate differential
        # EUR/USD: typically -1.5% to +1.5%
        rate_diff = np.random.randn(n).cumsum() * 0.05
        rate_diff = np.clip(rate_diff, -2, 2)

        data = pd.DataFrame({
            'VIX': vix,
            'DXY': dxy,
            'Interest_Rate_Diff': rate_diff
        }, index=index)

        print(f"  Created {len(data)} synthetic observations")
        print(f"  VIX: {data['VIX'].mean():.1f} (avg)")
        print(f"  DXY: {data['DXY'].mean():.2f} (avg)")

        return data

    def fetch_all_forex_features(
        self,
        pair: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_synthetic_fallback: bool = True
    ) -> pd.DataFrame:
        """
        Fetch all available forex features for a currency pair.

        Args:
            pair: Currency pair (e.g., 'EURUSD', 'GBPUSD')
            start_date: Start date
            end_date: End date
            use_synthetic_fallback: Create synthetic data if real data fails

        Returns:
            DataFrame with forex features
        """
        features = {}

        # Extract currencies from pair
        # e.g., 'EURUSD' -> base='EUR', quote='USD'
        if len(pair) >= 6:
            base_currency = pair[:3]
            quote_currency = pair[3:6]
        else:
            base_currency = None
            quote_currency = None

        # Fetch VIX (always useful)
        vix = self.fetch_vix(start_date, end_date)
        if not vix.empty:
            features['VIX'] = vix

        # Fetch DXY (for USD pairs)
        if quote_currency == 'USD' or base_currency == 'USD':
            dxy = self.fetch_dxy(start_date, end_date)
            if not dxy.empty:
                features['DXY'] = dxy

        # Fetch interest rate differential (if FRED available)
        if self.fred_client and base_currency and quote_currency:
            rate_diff = self.fetch_interest_rate_differential(
                base_currency, quote_currency, start_date, end_date
            )
            if not rate_diff.empty:
                features['Interest_Rate_Diff'] = rate_diff

        # Combine features
        if features:
            combined = pd.concat(features.values(), axis=1)
            combined = combined.fillna(method='ffill').fillna(method='bfill')

            print(f"\nFetched {len(features)} real forex indicators")
            return combined

        elif use_synthetic_fallback:
            warnings.warn("No real forex data available, using synthetic features")
            if not end_date:
                end_date = datetime.now()
            else:
                end_date = pd.to_datetime(end_date)

            if not start_date:
                start_date = end_date - timedelta(days=365)
            else:
                start_date = pd.to_datetime(start_date)

            index = pd.date_range(start=start_date, end=end_date, freq='D')
            return self.create_synthetic_forex_features(index, pair)

        else:
            warnings.warn("No forex features available")
            return pd.DataFrame()

    def align_forex_features_to_asset(
        self,
        asset_data: pd.DataFrame,
        forex_features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align forex features to asset data frequency.

        Args:
            asset_data: Asset price data with DatetimeIndex
            forex_features: Forex features

        Returns:
            Forex features aligned to asset data frequency
        """
        print("\nAligning forex features to asset frequency...")

        # Reindex to asset frequency with forward fill
        aligned = forex_features.reindex(
            asset_data.index,
            method='ffill'
        )

        # Fill any remaining NaN with backfill
        aligned = aligned.fillna(method='bfill')

        print(f"  Aligned {len(aligned)} observations")
        print(f"  Features: {list(aligned.columns)}")

        return aligned
