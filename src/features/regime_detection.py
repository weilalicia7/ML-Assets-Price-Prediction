"""
Regime Detection Module for Financial Time Series

Detects volatility regimes and market states using:
- Hidden Markov Models (HMM)
- Gaussian Mixture Models (GMM)
- Volatility clustering
- Trend detection

Used to determine when models are likely to perform well vs when to stay out.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')


class VolatilityRegimeDetector:
    """
    Detect volatility regimes in financial time series.

    Regimes:
    - Low Volatility (0): Calm markets, mean-reverting
    - Medium Volatility (1): Normal trading conditions
    - High Volatility (2): Turbulent markets, trending
    - Crisis (3): Extreme volatility, unpredictable
    """

    def __init__(self, n_regimes=4, method='gmm', random_state=42):
        """
        Initialize regime detector.

        Args:
            n_regimes: Number of volatility regimes (default: 4)
            method: 'gmm' (Gaussian Mixture) or 'hmm' (Hidden Markov Model)
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.method = method
        self.random_state = random_state
        self.model = None
        self.regime_stats = None

    def fit(self, df: pd.DataFrame, vol_col='volatility'):
        """
        Fit regime detector on historical volatility data.

        Args:
            df: DataFrame with datetime index and volatility column
            vol_col: Name of volatility column
        """
        print(f"[INFO] Fitting {self.method.upper()} regime detector with {self.n_regimes} regimes...")

        # Calculate volatility if not provided
        if vol_col not in df.columns:
            # Use Parkinson volatility (high-low range based)
            df['volatility'] = (df['High'] - df['Low']) / df['Close']

        vol_data = df[vol_col].values.reshape(-1, 1)

        # Remove NaN/inf values
        vol_data = vol_data[~np.isnan(vol_data).any(axis=1)]
        vol_data = vol_data[~np.isinf(vol_data).any(axis=1)]

        if self.method == 'gmm':
            # Gaussian Mixture Model
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                random_state=self.random_state,
                max_iter=200
            )
            self.model.fit(vol_data)

        elif self.method == 'hmm':
            # Hidden Markov Model
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type='full',
                random_state=self.random_state,
                n_iter=200
            )
            self.model.fit(vol_data)

        # Calculate regime statistics
        regimes = self.model.predict(vol_data)
        self.regime_stats = {}
        for i in range(self.n_regimes):
            regime_vols = vol_data[regimes == i]
            if len(regime_vols) > 0:
                self.regime_stats[i] = {
                    'mean_vol': regime_vols.mean(),
                    'std_vol': regime_vols.std(),
                    'min_vol': regime_vols.min(),
                    'max_vol': regime_vols.max(),
                    'count': len(regime_vols)
                }

        # Sort regimes by volatility (0=low, 1=medium, 2=high, 3=crisis)
        sorted_regimes = sorted(self.regime_stats.items(), key=lambda x: x[1]['mean_vol'])
        regime_mapping = {old: new for new, (old, _) in enumerate(sorted_regimes)}

        # Update regime labels
        new_regimes = np.array([regime_mapping[r] for r in regimes])

        # Recalculate stats with sorted regimes
        self.regime_stats = {}
        for i in range(self.n_regimes):
            regime_vols = vol_data[new_regimes == i]
            if len(regime_vols) > 0:
                self.regime_stats[i] = {
                    'mean_vol': regime_vols.mean(),
                    'std_vol': regime_vols.std(),
                    'min_vol': regime_vols.min(),
                    'max_vol': regime_vols.max(),
                    'count': len(regime_vols),
                    'pct': len(regime_vols) / len(vol_data) * 100
                }

        print(f"[OK] Regime detector fitted")
        print(f"     Regime distribution:")
        for i in range(self.n_regimes):
            stats = self.regime_stats[i]
            regime_name = ['Low Vol', 'Medium Vol', 'High Vol', 'Crisis'][i]
            print(f"       {i} ({regime_name}): {stats['pct']:.1f}% of data, "
                  f"mean vol={stats['mean_vol']:.4f}")

        return self

    def predict(self, df: pd.DataFrame, vol_col='volatility'):
        """
        Predict current regime for each time period.

        Args:
            df: DataFrame with volatility data
            vol_col: Name of volatility column

        Returns:
            Array of regime labels (0=low, 1=medium, 2=high, 3=crisis)
        """
        if self.model is None:
            raise ValueError("Model not fitted yet! Call fit() first.")

        # Calculate volatility if not provided
        if vol_col not in df.columns:
            df['volatility'] = (df['High'] - df['Low']) / df['Close']

        vol_data = df[vol_col].values.reshape(-1, 1)

        # Predict regimes
        regimes = self.model.predict(vol_data)

        # Apply same sorting as in fit()
        sorted_regimes = sorted(self.regime_stats.items(), key=lambda x: x[1]['mean_vol'])
        regime_mapping = {old: new for new, (old, _) in enumerate(sorted_regimes)}

        return np.array([regime_mapping.get(r, 1) for r in regimes])  # Default to medium vol

    def get_regime_features(self, df: pd.DataFrame, vol_col='volatility'):
        """
        Generate regime-based features for modeling.

        Args:
            df: DataFrame with volatility data
            vol_col: Name of volatility column

        Returns:
            DataFrame with regime features added
        """
        df = df.copy()

        # Predict regimes
        regimes = self.predict(df, vol_col)

        # Add regime label
        df['regime'] = regimes

        # Add regime duration (how many periods in current regime)
        df['regime_duration'] = 0
        current_regime = regimes[0]
        duration = 0
        durations = []

        for r in regimes:
            if r == current_regime:
                duration += 1
            else:
                current_regime = r
                duration = 1
            durations.append(duration)

        df['regime_duration'] = durations

        # Add regime transition probability (volatility of regime changes)
        regime_changes = (np.diff(regimes, prepend=regimes[0]) != 0).astype(int)
        transition_prob = pd.Series(regime_changes).rolling(window=20, min_periods=1).mean()
        df['regime_transition_prob'] = transition_prob.values

        # Add regime-specific volatility metrics
        for i in range(self.n_regimes):
            regime_mask = (regimes == i)
            df[f'regime_{i}_indicator'] = regime_mask.astype(int)

        # Add regime stability score (inverse of transition probability)
        df['regime_stability'] = 1 - df['regime_transition_prob']

        # Add distance to regime boundaries
        vol_data = df[vol_col].values
        df['vol_zscore_in_regime'] = 0.0

        for i in range(self.n_regimes):
            regime_mask = (regimes == i)
            if regime_mask.sum() > 0 and i in self.regime_stats:
                mean_vol = self.regime_stats[i]['mean_vol']
                std_vol = self.regime_stats[i]['std_vol']
                if std_vol > 0:
                    df.loc[regime_mask, 'vol_zscore_in_regime'] = \
                        (vol_data[regime_mask] - mean_vol) / std_vol

        print(f"[OK] Added {self.n_regimes + 5} regime features")

        return df


class TrendRegimeDetector:
    """
    Detect trend regimes: Trending vs Mean-Reverting markets.

    Uses:
    - Hurst exponent (H < 0.5 = mean-reverting, H > 0.5 = trending)
    - Autocorrelation
    - Momentum strength
    """

    def __init__(self, lookback=60):
        """
        Initialize trend regime detector.

        Args:
            lookback: Window size for trend calculations
        """
        self.lookback = lookback

    def calculate_hurst_exponent(self, ts):
        """
        Calculate Hurst exponent using R/S method.

        Returns:
            H: Hurst exponent
                H < 0.5: Mean-reverting
                H = 0.5: Random walk
                H > 0.5: Trending
        """
        if len(ts) < 100:
            return 0.5  # Not enough data, assume random walk

        lags = range(2, min(100, len(ts) // 2))
        tau = []

        for lag in lags:
            # Calculate standard deviation of differenced series
            std = np.std(np.subtract(ts[lag:], ts[:-lag]))
            tau.append(std)

        # Avoid log(0)
        tau = np.array(tau)
        lags = np.array(list(lags))

        # Filter out zeros
        mask = tau > 0
        tau = tau[mask]
        lags = lags[mask]

        if len(tau) < 2:
            return 0.5

        # Linear regression of log(tau) vs log(lags)
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0]

        return hurst

    def get_trend_features(self, df: pd.DataFrame, price_col='Close'):
        """
        Generate trend regime features.

        Args:
            df: DataFrame with price data
            price_col: Name of price column

        Returns:
            DataFrame with trend features added
        """
        df = df.copy()
        prices = df[price_col].values

        # Calculate Hurst exponent (rolling window)
        hurst_values = []
        for i in range(len(df)):
            start = max(0, i - self.lookback)
            if i - start >= 100:  # Need minimum data for Hurst
                hurst = self.calculate_hurst_exponent(prices[start:i+1])
            else:
                hurst = 0.5
            hurst_values.append(hurst)

        df['hurst_exponent'] = hurst_values

        # Trend vs Mean-Reversion score (-1 to +1)
        # H < 0.4: Strong mean-reversion (-1)
        # H = 0.5: Neutral (0)
        # H > 0.6: Strong trend (+1)
        df['trend_score'] = (df['hurst_exponent'] - 0.5) * 2
        df['trend_score'] = df['trend_score'].clip(-1, 1)

        # Calculate autocorrelation at multiple lags
        returns = df[price_col].pct_change()

        for lag in [1, 5, 10, 20]:
            df[f'autocorr_lag{lag}'] = returns.rolling(
                window=self.lookback, min_periods=20
            ).apply(lambda x: x.autocorr(lag=lag) if len(x) > lag else 0, raw=False)

        # Mean-reversion strength (negative autocorrelation)
        df['mean_reversion_strength'] = -df['autocorr_lag1'].clip(-1, 0)

        # Momentum strength (positive autocorrelation + trend)
        df['momentum_strength'] = df['autocorr_lag1'].clip(0, 1) * (df['hurst_exponent'] - 0.5)

        # Regime classification
        df['trend_regime'] = 'neutral'
        df.loc[df['trend_score'] < -0.3, 'trend_regime'] = 'mean_reverting'
        df.loc[df['trend_score'] > 0.3, 'trend_regime'] = 'trending'

        print(f"[OK] Added 7 + {len([1,5,10,20])} trend/mean-reversion features")

        return df


def main():
    """Test regime detection on sample data."""
    from src.data.fetch_data import DataFetcher

    print("="*60)
    print("REGIME DETECTION TEST")
    print("="*60)

    # Fetch sample data
    print("\n[INFO] Fetching AAPL data...")
    fetcher = DataFetcher(['AAPL'], start_date='2022-01-01')
    data = fetcher.fetch_all()
    aapl = data[data['Ticker'] == 'AAPL'].copy()

    print(f"[OK] Loaded {len(aapl)} days of data")

    # Test volatility regime detector
    print("\n[1/2] Testing Volatility Regime Detector...")
    vol_detector = VolatilityRegimeDetector(n_regimes=4, method='gmm')
    vol_detector.fit(aapl)

    # Add regime features
    aapl = vol_detector.get_regime_features(aapl)

    print(f"\nRegime distribution:")
    print(aapl['regime'].value_counts().sort_index())

    print(f"\nSample regime features:")
    regime_cols = [col for col in aapl.columns if 'regime' in col]
    print(aapl[regime_cols].tail(10))

    # Test trend regime detector
    print("\n[2/2] Testing Trend Regime Detector...")
    trend_detector = TrendRegimeDetector(lookback=60)
    aapl = trend_detector.get_trend_features(aapl)

    print(f"\nTrend regime distribution:")
    print(aapl['trend_regime'].value_counts())

    print(f"\nSample trend features:")
    trend_cols = ['hurst_exponent', 'trend_score', 'mean_reversion_strength',
                  'momentum_strength', 'trend_regime']
    print(aapl[trend_cols].tail(10))

    print("\n[SUCCESS] Regime detection test complete!")
    print(f"Total features added: {len(regime_cols) + len(trend_cols)}")


if __name__ == "__main__":
    main()
