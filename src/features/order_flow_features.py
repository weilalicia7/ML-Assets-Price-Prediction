"""
Volume-Based Order Flow Features for Financial Time Series

Detects buying vs selling pressure, institutional flows, and accumulation/distribution.
Used to identify smart money activity and market momentum.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class OrderFlowAnalyzer:
    """
    Analyze order flow and volume dynamics.

    Features:
    - On-Balance Volume (OBV) - Cumulative buying/selling pressure
    - Accumulation/Distribution (A/D) - Price-volume relationship
    - Money Flow Index (MFI) - Volume-weighted RSI
    - Volume-weighted momentum
    - Large order detection
    """

    def __init__(self, lookback_periods=[5, 10, 20, 60]):
        """
        Initialize order flow analyzer.

        Args:
            lookback_periods: Windows for rolling calculations
        """
        self.lookback_periods = lookback_periods

    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).

        OBV accumulates volume on up days and subtracts on down days.
        Rising OBV = buying pressure, falling OBV = selling pressure.

        Args:
            df: DataFrame with Close and Volume columns

        Returns:
            Series of OBV values
        """
        close_change = df['Close'].diff()
        obv = pd.Series(0, index=df.index, dtype=float)

        # Up days: add volume, down days: subtract volume
        obv[close_change > 0] = df['Volume'][close_change > 0]
        obv[close_change < 0] = -df['Volume'][close_change < 0]
        obv[close_change == 0] = 0

        # Cumulative sum
        obv = obv.cumsum()

        return obv

    def calculate_ad_line(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Accumulation/Distribution Line.

        A/D line uses the close location value (CLV) to weight volume:
        CLV = ((Close - Low) - (High - Close)) / (High - Low)
        A/D = Cumulative sum of CLV * Volume

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Series of A/D values
        """
        # Money Flow Multiplier (CLV)
        high_low_diff = df['High'] - df['Low']

        # Avoid division by zero
        clv = pd.Series(0.0, index=df.index)
        mask = high_low_diff != 0

        clv[mask] = (
            ((df['Close'][mask] - df['Low'][mask]) -
             (df['High'][mask] - df['Close'][mask])) /
            high_low_diff[mask]
        )

        # Money Flow Volume
        mfv = clv * df['Volume']

        # Accumulation/Distribution Line
        ad_line = mfv.cumsum()

        return ad_line

    def calculate_mfi(self, df: pd.DataFrame, period=14) -> pd.Series:
        """
        Calculate Money Flow Index (MFI).

        MFI is like RSI but volume-weighted. Ranges 0-100.
        MFI > 80: Overbought (selling pressure likely)
        MFI < 20: Oversold (buying pressure likely)

        Args:
            df: DataFrame with OHLCV data
            period: Lookback period (default 14)

        Returns:
            Series of MFI values
        """
        # Typical Price
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3

        # Raw Money Flow
        raw_money_flow = typical_price * df['Volume']

        # Positive and Negative Money Flow
        price_change = typical_price.diff()

        positive_flow = pd.Series(0.0, index=df.index)
        negative_flow = pd.Series(0.0, index=df.index)

        positive_flow[price_change > 0] = raw_money_flow[price_change > 0]
        negative_flow[price_change < 0] = raw_money_flow[price_change < 0]

        # Sum over period
        positive_mf = positive_flow.rolling(window=period, min_periods=1).sum()
        negative_mf = negative_flow.rolling(window=period, min_periods=1).sum()

        # Money Flow Ratio
        mfr = positive_mf / (negative_mf + 1e-10)  # Avoid division by zero

        # Money Flow Index
        mfi = 100 - (100 / (1 + mfr))

        return mfi

    def calculate_vwap(self, df: pd.DataFrame, period=20) -> pd.Series:
        """
        Calculate Volume-Weighted Average Price (VWAP).

        VWAP = Sum(Price * Volume) / Sum(Volume)
        Price above VWAP = bullish, below = bearish

        Args:
            df: DataFrame with OHLCV data
            period: Rolling window (default 20)

        Returns:
            Series of VWAP values
        """
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3

        # Rolling VWAP
        vwap = (
            (typical_price * df['Volume']).rolling(window=period, min_periods=1).sum() /
            df['Volume'].rolling(window=period, min_periods=1).sum()
        )

        return vwap

    def detect_volume_spikes(self, df: pd.DataFrame, threshold=2.0, window=20) -> pd.Series:
        """
        Detect abnormally high volume (large orders).

        Volume spike = Volume > threshold * average volume
        Indicates institutional activity or significant events.

        Args:
            df: DataFrame with Volume column
            threshold: Multiplier for spike detection (default 2.0)
            window: Lookback for average volume

        Returns:
            Series of spike indicators (1 = spike, 0 = normal)
        """
        avg_volume = df['Volume'].rolling(window=window, min_periods=1).mean()
        volume_ratio = df['Volume'] / (avg_volume + 1e-10)

        spikes = (volume_ratio > threshold).astype(int)

        return spikes

    def calculate_volume_momentum(self, df: pd.DataFrame, period=10) -> pd.Series:
        """
        Calculate volume momentum (rate of change in volume).

        Rising volume momentum = increasing participation
        Falling volume momentum = decreasing interest

        Args:
            df: DataFrame with Volume column
            period: Lookback period

        Returns:
            Series of volume momentum values
        """
        volume_roc = df['Volume'].pct_change(periods=period)

        return volume_roc

    def calculate_force_index(self, df: pd.DataFrame, period=13) -> pd.Series:
        """
        Calculate Force Index (price change * volume).

        Force Index = (Close - Close[1]) * Volume
        Positive = buying force, negative = selling force

        Args:
            df: DataFrame with Close and Volume
            period: EMA smoothing period

        Returns:
            Series of force index values
        """
        price_change = df['Close'].diff()
        force = price_change * df['Volume']

        # Smooth with EMA
        force_ema = force.ewm(span=period, adjust=False).mean()

        return force_ema

    def get_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all order flow features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with order flow features added
        """
        print("[INFO] Calculating order flow features...")
        df = df.copy()

        # 1. On-Balance Volume (OBV)
        df['obv'] = self.calculate_obv(df)

        # OBV momentum at different windows
        for period in self.lookback_periods:
            df[f'obv_momentum_{period}d'] = df['obv'].pct_change(periods=period)

        # OBV trend (positive = accumulation, negative = distribution)
        df['obv_trend'] = np.sign(df['obv_momentum_20d'])

        # 2. Accumulation/Distribution Line
        df['ad_line'] = self.calculate_ad_line(df)

        # A/D momentum
        for period in self.lookback_periods:
            df[f'ad_momentum_{period}d'] = df['ad_line'].pct_change(periods=period)

        # 3. Money Flow Index
        df['mfi'] = self.calculate_mfi(df, period=14)
        df['mfi_overbought'] = (df['mfi'] > 80).astype(int)
        df['mfi_oversold'] = (df['mfi'] < 20).astype(int)

        # MFI divergence (price up, MFI down = bearish)
        price_change_20d = df['Close'].pct_change(20)
        mfi_change_20d = df['mfi'].diff(20)
        df['mfi_divergence'] = np.sign(price_change_20d) != np.sign(mfi_change_20d)
        df['mfi_divergence'] = df['mfi_divergence'].astype(int)

        # 4. VWAP
        for period in [10, 20, 60]:
            df[f'vwap_{period}d'] = self.calculate_vwap(df, period)
            df[f'price_vs_vwap_{period}d'] = (df['Close'] - df[f'vwap_{period}d']) / df['Close']

        # 5. Volume Spikes
        df['volume_spike'] = self.detect_volume_spikes(df, threshold=2.0, window=20)
        df['volume_spike_3std'] = self.detect_volume_spikes(df, threshold=3.0, window=20)

        # Volume spike frequency (how often in last N days)
        df['volume_spike_freq_20d'] = df['volume_spike'].rolling(window=20, min_periods=1).mean()

        # 6. Volume Momentum
        for period in self.lookback_periods:
            df[f'volume_momentum_{period}d'] = self.calculate_volume_momentum(df, period)

        # 7. Force Index
        df['force_index'] = self.calculate_force_index(df, period=13)
        df['force_index_positive'] = (df['force_index'] > 0).astype(int)

        # Force index strength (normalized)
        force_std = df['force_index'].rolling(window=60, min_periods=1).std()
        df['force_index_zscore'] = df['force_index'] / (force_std + 1e-10)

        # 8. Volume Relative to Price Movement
        returns = df['Close'].pct_change().abs()
        volume_norm = df['Volume'] / df['Volume'].rolling(window=20, min_periods=1).mean()

        # High volume on small moves = distribution/accumulation
        # High volume on large moves = momentum/breakout
        df['volume_price_ratio'] = volume_norm / (returns + 1e-10)

        # 9. Buying vs Selling Pressure Score
        # Composite score: OBV trend + A/D trend + MFI signal + Force Index
        df['buying_pressure'] = (
            (df['obv_trend'] > 0).astype(int) +
            (df['ad_momentum_20d'] > 0).astype(int) +
            (df['mfi'] > 50).astype(int) +
            (df['force_index'] > 0).astype(int)
        ) / 4.0  # Normalize to 0-1

        df['selling_pressure'] = 1 - df['buying_pressure']

        # 10. Institutional Activity Score
        # Large volume spikes with strong price movement = likely institutional
        large_price_moves = (df['Close'].pct_change().abs() > df['Close'].pct_change().abs().rolling(20).mean())
        df['institutional_activity'] = (df['volume_spike'] & large_price_moves).astype(int)
        df['institutional_activity_freq'] = df['institutional_activity'].rolling(window=20, min_periods=1).mean()

        # Count features
        order_flow_cols = [col for col in df.columns if col not in
                          ['Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'AssetType']]

        # Exclude columns that existed before
        new_cols = len([col for col in order_flow_cols if
                       'obv' in col or 'ad_' in col or 'mfi' in col or
                       'vwap' in col or 'volume_' in col or 'force' in col or
                       'buying' in col or 'selling' in col or 'institutional' in col])

        print(f"[OK] Added {new_cols} order flow features")

        return df


class SmartMoneyDetector:
    """
    Detect smart money (institutional) activity patterns.

    Patterns:
    - Quiet accumulation: Price stable/down, volume up, OBV up
    - Quiet distribution: Price stable/up, volume up, OBV down
    - Breakout confirmation: Price up, volume spike, OBV up
    - False breakout: Price up, volume spike, OBV down
    """

    def __init__(self):
        """Initialize smart money detector."""
        pass

    def detect_accumulation(self, df: pd.DataFrame, window=10) -> pd.Series:
        """
        Detect accumulation pattern.

        Signs:
        - Price flat or slightly down
        - Volume above average
        - OBV rising
        - A/D line rising

        Args:
            df: DataFrame with order flow features
            window: Detection window

        Returns:
            Series of accumulation signals (1 = accumulation, 0 = none)
        """
        # Price change (small or negative)
        price_change = df['Close'].pct_change(window)
        price_stable = (price_change.abs() < 0.05) | (price_change < 0)

        # Volume above average
        avg_volume = df['Volume'].rolling(window=20, min_periods=1).mean()
        volume_high = df['Volume'] > avg_volume

        # OBV rising
        obv_rising = df['obv_momentum_20d'] > 0

        # A/D rising
        ad_rising = df['ad_momentum_20d'] > 0

        # Accumulation signal
        accumulation = (price_stable & volume_high & obv_rising & ad_rising).astype(int)

        return accumulation

    def detect_distribution(self, df: pd.DataFrame, window=10) -> pd.Series:
        """
        Detect distribution pattern.

        Signs:
        - Price flat or slightly up
        - Volume above average
        - OBV falling
        - A/D line falling

        Args:
            df: DataFrame with order flow features
            window: Detection window

        Returns:
            Series of distribution signals (1 = distribution, 0 = none)
        """
        # Price change (small or positive)
        price_change = df['Close'].pct_change(window)
        price_stable = (price_change.abs() < 0.05) | (price_change > 0)

        # Volume above average
        avg_volume = df['Volume'].rolling(window=20, min_periods=1).mean()
        volume_high = df['Volume'] > avg_volume

        # OBV falling
        obv_falling = df['obv_momentum_20d'] < 0

        # A/D falling
        ad_falling = df['ad_momentum_20d'] < 0

        # Distribution signal
        distribution = (price_stable & volume_high & obv_falling & ad_falling).astype(int)

        return distribution

    def detect_breakout_confirmation(self, df: pd.DataFrame, threshold=0.02) -> pd.Series:
        """
        Detect confirmed breakouts (smart money buying).

        Signs:
        - Price breaks above resistance (>2% move)
        - Volume spike
        - OBV confirms (rising)

        Args:
            df: DataFrame with order flow features
            threshold: Price move threshold for breakout

        Returns:
            Series of breakout confirmation signals
        """
        # Strong upward price move
        price_up = df['Close'].pct_change() > threshold

        # Volume spike
        volume_spike = df['volume_spike'] == 1

        # OBV confirming
        obv_up = df['obv_momentum_5d'] > 0

        # Confirmed breakout
        breakout = (price_up & volume_spike & obv_up).astype(int)

        return breakout

    def detect_false_breakout(self, df: pd.DataFrame, threshold=0.02) -> pd.Series:
        """
        Detect false breakouts (trap for retail traders).

        Signs:
        - Price breaks above resistance
        - Volume spike
        - OBV NOT confirming (falling) = smart money selling

        Args:
            df: DataFrame with order flow features
            threshold: Price move threshold

        Returns:
            Series of false breakout signals
        """
        # Strong upward price move
        price_up = df['Close'].pct_change() > threshold

        # Volume spike
        volume_spike = df['volume_spike'] == 1

        # OBV NOT confirming
        obv_down = df['obv_momentum_5d'] < 0

        # False breakout
        false_breakout = (price_up & volume_spike & obv_down).astype(int)

        return false_breakout

    def get_smart_money_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all smart money detection features.

        Args:
            df: DataFrame with order flow features (must have OBV, A/D, etc.)

        Returns:
            DataFrame with smart money features added
        """
        print("[INFO] Detecting smart money patterns...")
        df = df.copy()

        # Accumulation/Distribution patterns
        df['smart_accumulation'] = self.detect_accumulation(df, window=10)
        df['smart_distribution'] = self.detect_distribution(df, window=10)

        # Accumulation/distribution strength (frequency in last 20 days)
        df['accumulation_strength'] = df['smart_accumulation'].rolling(window=20, min_periods=1).mean()
        df['distribution_strength'] = df['smart_distribution'].rolling(window=20, min_periods=1).mean()

        # Breakout patterns
        df['confirmed_breakout'] = self.detect_breakout_confirmation(df, threshold=0.02)
        df['false_breakout'] = self.detect_false_breakout(df, threshold=0.02)

        # Breakout frequency
        df['confirmed_breakout_freq'] = df['confirmed_breakout'].rolling(window=20, min_periods=1).mean()
        df['false_breakout_freq'] = df['false_breakout'].rolling(window=20, min_periods=1).mean()

        # Net smart money pressure (-1 to +1)
        df['smart_money_pressure'] = (
            df['accumulation_strength'] - df['distribution_strength']
        ).clip(-1, 1)

        print(f"[OK] Added 9 smart money detection features")

        return df


def main():
    """Test order flow features on sample data."""
    from src.data.fetch_data import DataFetcher

    print("="*60)
    print("ORDER FLOW ANALYSIS TEST")
    print("="*60)

    # Fetch sample data
    print("\n[INFO] Fetching AAPL data...")
    fetcher = DataFetcher(['AAPL'], start_date='2022-01-01')
    data = fetcher.fetch_all()
    aapl = data[data['Ticker'] == 'AAPL'].copy()

    print(f"[OK] Loaded {len(aapl)} days of data")

    # Test order flow analyzer
    print("\n[1/2] Testing Order Flow Analyzer...")
    analyzer = OrderFlowAnalyzer(lookback_periods=[5, 10, 20, 60])
    aapl = analyzer.get_all_features(aapl)

    print(f"\nSample order flow features:")
    flow_cols = ['obv_momentum_20d', 'ad_momentum_20d', 'mfi', 'volume_spike',
                 'buying_pressure', 'institutional_activity_freq']
    print(aapl[flow_cols].tail(10))

    print(f"\nBuying vs Selling Pressure (last 10 days):")
    print(f"Average Buying Pressure: {aapl['buying_pressure'].tail(10).mean():.2f}")
    print(f"Average Selling Pressure: {aapl['selling_pressure'].tail(10).mean():.2f}")

    # Test smart money detector
    print("\n[2/2] Testing Smart Money Detector...")
    detector = SmartMoneyDetector()
    aapl = detector.get_smart_money_features(aapl)

    print(f"\nSmart money patterns detected:")
    print(f"Accumulation days: {aapl['smart_accumulation'].sum()}")
    print(f"Distribution days: {aapl['smart_distribution'].sum()}")
    print(f"Confirmed breakouts: {aapl['confirmed_breakout'].sum()}")
    print(f"False breakouts: {aapl['false_breakout'].sum()}")

    print(f"\nCurrent smart money pressure: {aapl['smart_money_pressure'].iloc[-1]:.2f}")
    print("  (+1 = Strong accumulation, -1 = Strong distribution)")

    print(f"\nSample smart money features:")
    smart_cols = ['smart_accumulation', 'smart_distribution', 'confirmed_breakout',
                  'smart_money_pressure']
    print(aapl[smart_cols].tail(10))

    print("\n[SUCCESS] Order flow analysis test complete!")

    # Count total features added
    total_features = len([col for col in aapl.columns if
                         'obv' in col or 'ad_' in col or 'mfi' in col or
                         'vwap' in col or 'volume_' in col or 'force' in col or
                         'buying' in col or 'selling' in col or 'institutional' in col or
                         'smart' in col or 'breakout' in col])
    print(f"Total order flow features: {total_features}")


if __name__ == "__main__":
    main()
