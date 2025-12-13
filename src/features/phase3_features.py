"""
Phase 3 Feature Engineering - Structural Features Integration

Combines all Phase 3 feature modules:
1. Regime Detection (volatility & trend regimes)
2. Order Flow Analysis (volume dynamics & smart money)
3. Intermarket Correlations (market relationships)

This module provides a single interface to add all Phase 3 features to any asset.
"""

import numpy as np
import pandas as pd
from src.features.regime_detection import VolatilityRegimeDetector, TrendRegimeDetector
from src.features.order_flow_features import OrderFlowAnalyzer, SmartMoneyDetector
from src.features.intermarket_features import IntermarketAnalyzer, SectorAnalyzer
import warnings
warnings.filterwarnings('ignore')


class Phase3FeatureEngineer:
    """
    Unified interface for all Phase 3 structural features.

    This class orchestrates:
    - Volatility regime detection
    - Trend/mean-reversion classification
    - Order flow and smart money detection
    - Intermarket correlations and market regime
    """

    def __init__(
        self,
        n_regimes=4,
        regime_method='gmm',
        enable_intermarket=True,
        enable_sector=False,
        sector_ticker=None,
        random_state=42
    ):
        """
        Initialize Phase 3 feature engineer.

        Args:
            n_regimes: Number of volatility regimes (default 4)
            regime_method: 'gmm' or 'hmm' for regime detection
            enable_intermarket: Whether to fetch and add intermarket features
            enable_sector: Whether to add sector-specific features
            sector_ticker: Sector ETF ticker (e.g., 'XLK' for tech)
            random_state: Random seed
        """
        self.n_regimes = n_regimes
        self.regime_method = regime_method
        self.enable_intermarket = enable_intermarket
        self.enable_sector = enable_sector
        self.sector_ticker = sector_ticker
        self.random_state = random_state

        # Initialize all detectors/analyzers
        self.vol_regime_detector = VolatilityRegimeDetector(
            n_regimes=n_regimes,
            method=regime_method,
            random_state=random_state
        )
        self.trend_detector = TrendRegimeDetector(lookback=60)
        self.order_flow_analyzer = OrderFlowAnalyzer(lookback_periods=[5, 10, 20, 60])
        self.smart_money_detector = SmartMoneyDetector()

        if enable_intermarket:
            self.intermarket_analyzer = IntermarketAnalyzer(lookback_periods=[20, 60, 120])
        else:
            self.intermarket_analyzer = None

        if enable_sector and sector_ticker:
            self.sector_analyzer = SectorAnalyzer()
        else:
            self.sector_analyzer = None

        self.is_fitted = False

    def fit(self, df: pd.DataFrame, vol_col='volatility'):
        """
        Fit regime detectors on historical data.

        Args:
            df: Training DataFrame with OHLCV data
            vol_col: Volatility column name (will be calculated if missing)

        Returns:
            self
        """
        print("="*60)
        print("PHASE 3 FEATURE ENGINEERING - FITTING")
        print("="*60)

        # Fit volatility regime detector
        print("\n[1/2] Fitting volatility regime detector...")
        self.vol_regime_detector.fit(df, vol_col=vol_col)

        # Trend detector doesn't need fitting (stateless)
        print("\n[2/2] Trend detector ready (no fitting needed)")

        self.is_fitted = True
        print("\n[OK] Phase 3 feature engineer fitted!")

        return self

    def transform(self, df: pd.DataFrame, fit_first=False) -> pd.DataFrame:
        """
        Add all Phase 3 features to DataFrame.

        Args:
            df: DataFrame with OHLCV data
            fit_first: If True, fit regime detector before transform

        Returns:
            DataFrame with all Phase 3 features added
        """
        print("="*60)
        print("PHASE 3 FEATURE ENGINEERING - TRANSFORM")
        print("="*60)

        df = df.copy()
        initial_cols = len(df.columns)

        # Fit if requested
        if fit_first:
            self.fit(df)
        elif not self.is_fitted:
            raise ValueError("Regime detector not fitted! Call fit() first or use fit_first=True")

        # 1. Volatility Regime Features
        print("\n[1/5] Adding volatility regime features...")
        df = self.vol_regime_detector.get_regime_features(df)

        # 2. Trend/Mean-Reversion Features
        print("\n[2/5] Adding trend/mean-reversion features...")
        df = self.trend_detector.get_trend_features(df)

        # 3. Order Flow Features
        print("\n[3/5] Adding order flow features...")
        df = self.order_flow_analyzer.get_all_features(df)

        # 4. Smart Money Detection
        print("\n[4/5] Adding smart money detection features...")
        df = self.smart_money_detector.get_smart_money_features(df)

        # 5. Intermarket Correlations (optional)
        if self.enable_intermarket and self.intermarket_analyzer is not None:
            print("\n[5/5] Adding intermarket correlation features...")
            df = self.intermarket_analyzer.get_all_features(df, fetch_market_data=True)
        else:
            print("\n[5/5] Skipping intermarket features (disabled)")

        # 6. Sector Features (optional)
        if self.enable_sector and self.sector_analyzer is not None and self.sector_ticker:
            print("\n[6/5] Adding sector features...")
            df = self.sector_analyzer.get_sector_features(df, self.sector_ticker)
        else:
            print("\n[6/5] Skipping sector features (disabled)")

        final_cols = len(df.columns)
        added_cols = final_cols - initial_cols

        print("\n" + "="*60)
        print(f"PHASE 3 COMPLETE: Added {added_cols} new features")
        print("="*60)

        return df

    def fit_transform(self, df: pd.DataFrame, vol_col='volatility') -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            df: DataFrame with OHLCV data
            vol_col: Volatility column name

        Returns:
            DataFrame with all Phase 3 features
        """
        self.fit(df, vol_col=vol_col)
        return self.transform(df)

    def get_feature_names(self) -> list:
        """
        Get list of Phase 3 feature names.

        Returns:
            List of feature column names
        """
        features = []

        # Regime features
        features.extend([
            'regime', 'regime_duration', 'regime_transition_prob',
            'regime_stability', 'vol_zscore_in_regime'
        ])
        for i in range(self.n_regimes):
            features.append(f'regime_{i}_indicator')

        # Trend features
        features.extend([
            'hurst_exponent', 'trend_score', 'mean_reversion_strength',
            'momentum_strength', 'trend_regime'
        ])
        for lag in [1, 5, 10, 20]:
            features.append(f'autocorr_lag{lag}')

        # Order flow features (dynamic based on lookback periods)
        features.extend([
            'obv', 'obv_trend', 'ad_line', 'mfi', 'mfi_overbought',
            'mfi_oversold', 'mfi_divergence', 'volume_spike',
            'volume_spike_3std', 'volume_spike_freq_20d', 'force_index',
            'force_index_positive', 'force_index_zscore',
            'volume_price_ratio', 'buying_pressure', 'selling_pressure',
            'institutional_activity', 'institutional_activity_freq'
        ])

        # Smart money features
        features.extend([
            'smart_accumulation', 'smart_distribution',
            'accumulation_strength', 'distribution_strength',
            'confirmed_breakout', 'false_breakout',
            'confirmed_breakout_freq', 'false_breakout_freq',
            'smart_money_pressure'
        ])

        # Intermarket features (if enabled)
        if self.enable_intermarket:
            indices = ['SPY', '^VIX', 'DX-Y.NYB', 'TLT', 'GLD']
            for idx in indices:
                for window in [20, 60, 120]:
                    features.append(f'corr_{idx}_{window}d')
            features.extend([
                'beta_SPY_20d', 'beta_SPY_60d', 'beta_SPY_120d',
                'rel_strength_SPY_20d', 'rel_strength_SPY_60d',
                'rel_strength_SPY_120d', 'risk_regime_score', 'market_regime'
            ])

        # Sector features (if enabled)
        if self.enable_sector and self.sector_ticker:
            features.extend([
                f'sector_{self.sector_ticker}',
                'rel_strength_sector_20d', 'rel_strength_sector_60d',
                'corr_sector_20d', 'corr_sector_60d'
            ])

        return features

    def get_feature_importance_groups(self) -> dict:
        """
        Group features by category for importance analysis.

        Returns:
            Dictionary mapping category names to feature lists
        """
        groups = {
            'Volatility Regime': [
                'regime', 'regime_duration', 'regime_transition_prob',
                'regime_stability', 'vol_zscore_in_regime'
            ] + [f'regime_{i}_indicator' for i in range(self.n_regimes)],

            'Trend/Mean-Reversion': [
                'hurst_exponent', 'trend_score', 'mean_reversion_strength',
                'momentum_strength', 'trend_regime'
            ] + [f'autocorr_lag{lag}' for lag in [1, 5, 10, 20]],

            'Order Flow': [
                'obv', 'obv_trend', 'ad_line', 'mfi', 'volume_spike',
                'volume_spike_freq_20d', 'force_index', 'volume_price_ratio',
                'buying_pressure', 'selling_pressure',
                'institutional_activity_freq'
            ],

            'Smart Money': [
                'smart_accumulation', 'smart_distribution',
                'accumulation_strength', 'distribution_strength',
                'confirmed_breakout', 'false_breakout',
                'smart_money_pressure'
            ]
        }

        if self.enable_intermarket:
            groups['Market Correlations'] = [
                f'corr_SPY_{w}d' for w in [20, 60, 120]
            ] + [f'beta_SPY_{w}d' for w in [20, 60, 120]]

            groups['Market Regime'] = [
                'risk_regime_score', 'market_regime',
                'SPY_momentum_20d', 'VIX_level', 'VIX_spike',
                'DXY_strong', 'GLD_strong'
            ]

        if self.enable_sector and self.sector_ticker:
            groups['Sector Relative'] = [
                'rel_strength_sector_20d', 'rel_strength_sector_60d',
                'corr_sector_20d', 'corr_sector_60d'
            ]

        return groups


def add_phase3_features_to_pipeline(
    df: pd.DataFrame,
    asset_type: str = 'stock',
    sector_ticker: str = None,
    fit_regimes: bool = True
) -> pd.DataFrame:
    """
    Convenience function to add Phase 3 features with smart defaults.

    Args:
        df: DataFrame with OHLCV data
        asset_type: 'stock', 'forex', 'crypto', 'commodity'
        sector_ticker: Sector ETF ticker (auto-detected if None)
        fit_regimes: Whether to fit regime detector

    Returns:
        DataFrame with Phase 3 features added
    """
    # Smart defaults based on asset type
    config = {
        'stock': {
            'enable_intermarket': True,
            'enable_sector': True,
            'n_regimes': 4
        },
        'forex': {
            'enable_intermarket': True,
            'enable_sector': False,
            'n_regimes': 3  # Forex has less extreme volatility regimes
        },
        'crypto': {
            'enable_intermarket': True,
            'enable_sector': False,
            'n_regimes': 4  # Crypto has extreme regimes
        },
        'commodity': {
            'enable_intermarket': True,
            'enable_sector': False,
            'n_regimes': 4
        }
    }

    settings = config.get(asset_type, config['stock'])

    # Initialize engineer
    engineer = Phase3FeatureEngineer(
        n_regimes=settings['n_regimes'],
        regime_method='gmm',
        enable_intermarket=settings['enable_intermarket'],
        enable_sector=settings['enable_sector'],
        sector_ticker=sector_ticker,
        random_state=42
    )

    # Transform
    if fit_regimes:
        df = engineer.fit_transform(df)
    else:
        df = engineer.transform(df, fit_first=True)

    return df


def main():
    """Test Phase 3 feature integration."""
    from src.data.fetch_data import DataFetcher

    print("="*60)
    print("PHASE 3 FEATURE INTEGRATION TEST")
    print("="*60)

    # Fetch sample data
    print("\n[INFO] Fetching AAPL data...")
    fetcher = DataFetcher(['AAPL'], start_date='2023-01-01')
    data = fetcher.fetch_all()
    aapl = data[data['Ticker'] == 'AAPL'].copy()

    print(f"[OK] Loaded {len(aapl)} days of data")
    print(f"[OK] Initial features: {len(aapl.columns)}")

    # Add Phase 3 features
    print("\n" + "="*60)
    print("ADDING ALL PHASE 3 FEATURES")
    print("="*60)

    # Option 1: Using convenience function
    aapl_enhanced = add_phase3_features_to_pipeline(
        aapl,
        asset_type='stock',
        sector_ticker='XLK',  # Technology sector
        fit_regimes=True
    )

    print(f"\n[OK] Final features: {len(aapl_enhanced.columns)}")
    print(f"[OK] Added {len(aapl_enhanced.columns) - len(aapl.columns)} Phase 3 features")

    # Show sample of new features
    print("\n" + "="*60)
    print("SAMPLE OF PHASE 3 FEATURES (Last 5 days)")
    print("="*60)

    sample_features = [
        'regime', 'trend_regime', 'buying_pressure',
        'smart_money_pressure', 'market_regime', 'beta_SPY_60d'
    ]

    existing_features = [f for f in sample_features if f in aapl_enhanced.columns]
    if len(existing_features) > 0:
        print(aapl_enhanced[existing_features].tail())

    # Show regime distribution
    print("\n" + "="*60)
    print("REGIME ANALYSIS")
    print("="*60)

    if 'regime' in aapl_enhanced.columns:
        print("\nVolatility Regime Distribution:")
        regime_dist = aapl_enhanced['regime'].value_counts().sort_index()
        regime_names = ['Low Vol', 'Medium Vol', 'High Vol', 'Crisis']
        for regime, count in regime_dist.items():
            pct = count / len(aapl_enhanced) * 100
            print(f"  {regime} ({regime_names[regime]}): {count} days ({pct:.1f}%)")

    if 'trend_regime' in aapl_enhanced.columns:
        print("\nTrend Regime Distribution:")
        trend_dist = aapl_enhanced['trend_regime'].value_counts()
        for regime, count in trend_dist.items():
            pct = count / len(aapl_enhanced) * 100
            print(f"  {regime}: {count} days ({pct:.1f}%)")

    if 'market_regime' in aapl_enhanced.columns:
        print("\nMarket Regime Distribution:")
        market_dist = aapl_enhanced['market_regime'].value_counts()
        for regime, count in market_dist.items():
            pct = count / len(aapl_enhanced) * 100
            print(f"  {regime}: {count} days ({pct:.1f}%)")

    print("\n[SUCCESS] Phase 3 feature integration test complete!")


if __name__ == "__main__":
    main()
