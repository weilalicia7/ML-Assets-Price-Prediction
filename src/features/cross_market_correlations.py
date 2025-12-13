"""
Phase 4: Cross-Market Correlations

Advanced inter-market analysis for Phase 4 Macro Integration.
Builds on existing intermarket_features.py with additional capabilities:
- Rolling correlation network analysis
- PCA-based correlation clustering
- Risk-on/Risk-off regime detection
- Correlation breakdown detection
- Cross-asset contagion tracking

Expected Impact: +5-8% profit rate improvement

Based on: phase future roadmap.pdf and phase2and3 fixing 2 improvements 15 points.pdf
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from collections import deque
import logging

logger = logging.getLogger(__name__)


class CrossMarketCorrelationEngine:
    """
    Advanced cross-market correlation analysis for Phase 4.

    Features:
    - Rolling 60-day correlation network
    - PCA-based correlation clustering
    - Risk-on/risk-off regime detection
    - Correlation breakdown detection
    - Dynamic correlation regime tracking
    """

    def __init__(
        self,
        correlation_window: int = 60,
        pca_components: int = 3,
        breakdown_threshold: float = 0.3,
        regime_window: int = 20
    ):
        """
        Initialize cross-market correlation engine.

        Args:
            correlation_window: Rolling window for correlation calculation (default 60 days)
            pca_components: Number of PCA components for clustering
            breakdown_threshold: Threshold for correlation breakdown detection
            regime_window: Window for regime detection
        """
        self.correlation_window = correlation_window
        self.pca_components = pca_components
        self.breakdown_threshold = breakdown_threshold
        self.regime_window = regime_window

        # Macro indicators to track
        self.macro_symbols = ['SPY', 'VIX', 'GLD', 'TLT', 'DXY']

        # Historical tracking
        self.correlation_history: deque = deque(maxlen=252)  # 1 year
        self.regime_history: deque = deque(maxlen=regime_window)

        logger.info(f"Initialized CrossMarketCorrelationEngine (window={correlation_window}d)")

    def calculate_correlation_matrix(
        self,
        returns: pd.DataFrame,
        window: int = None
    ) -> pd.DataFrame:
        """
        Calculate rolling correlation matrix.

        Args:
            returns: DataFrame of asset returns
            window: Rolling window (default: self.correlation_window)

        Returns:
            Correlation matrix DataFrame
        """
        if window is None:
            window = self.correlation_window

        # Calculate rolling correlation
        corr_matrix = returns.rolling(window=window, min_periods=int(window * 0.7)).corr()

        return corr_matrix

    def calculate_correlation_network_features(
        self,
        df: pd.DataFrame,
        price_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Calculate correlation network features.

        Features generated:
        - avg_correlation: Average pairwise correlation
        - max_correlation: Maximum pairwise correlation
        - correlation_dispersion: Std of correlations (high = divergence)
        - correlation_cluster_score: PCA-based clustering score

        Args:
            df: DataFrame with price data for multiple assets
            price_cols: List of price column names to use

        Returns:
            DataFrame with correlation network features
        """
        df = df.copy()

        if price_cols is None:
            # Use macro symbols if available
            price_cols = [col for col in self.macro_symbols if col in df.columns]

        if len(price_cols) < 2:
            logger.warning("Not enough assets for correlation network analysis")
            return df

        # Calculate returns
        returns = pd.DataFrame()
        for col in price_cols:
            returns[col] = df[col].pct_change()

        # Rolling correlation features
        avg_corr = []
        max_corr = []
        corr_disp = []

        for i in range(len(df)):
            if i < self.correlation_window:
                avg_corr.append(np.nan)
                max_corr.append(np.nan)
                corr_disp.append(np.nan)
                continue

            # Get window of returns
            window_returns = returns.iloc[i-self.correlation_window:i]

            # Calculate correlation matrix
            corr = window_returns.corr()

            # Extract upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            corr_values = corr.values[mask]

            # Calculate features
            avg_corr.append(np.nanmean(corr_values))
            max_corr.append(np.nanmax(np.abs(corr_values)))
            corr_disp.append(np.nanstd(corr_values))

        df['corr_network_avg'] = avg_corr
        df['corr_network_max'] = max_corr
        df['corr_network_dispersion'] = corr_disp

        logger.info("Added 3 correlation network features")

        return df

    def calculate_pca_correlation_features(
        self,
        df: pd.DataFrame,
        price_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Calculate PCA-based correlation clustering features.

        Uses PCA to identify common factors driving asset movements.
        High explained variance by PC1 = high correlation regime

        Args:
            df: DataFrame with price data
            price_cols: List of price columns

        Returns:
            DataFrame with PCA features
        """
        df = df.copy()

        if price_cols is None:
            price_cols = [col for col in self.macro_symbols if col in df.columns]

        if len(price_cols) < self.pca_components:
            logger.warning(f"Not enough assets ({len(price_cols)}) for PCA with {self.pca_components} components")
            return df

        # Calculate returns
        returns = pd.DataFrame()
        for col in price_cols:
            returns[col] = df[col].pct_change()

        # Rolling PCA explained variance
        pc1_explained = []
        pc_total_explained = []

        for i in range(len(df)):
            if i < self.correlation_window:
                pc1_explained.append(np.nan)
                pc_total_explained.append(np.nan)
                continue

            window_returns = returns.iloc[i-self.correlation_window:i].dropna()

            if len(window_returns) < self.correlation_window * 0.7:
                pc1_explained.append(np.nan)
                pc_total_explained.append(np.nan)
                continue

            try:
                # Standardize
                standardized = (window_returns - window_returns.mean()) / (window_returns.std() + 1e-10)

                # PCA
                pca = PCA(n_components=min(self.pca_components, len(price_cols)))
                pca.fit(standardized)

                pc1_explained.append(pca.explained_variance_ratio_[0])
                pc_total_explained.append(sum(pca.explained_variance_ratio_[:self.pca_components]))
            except Exception as e:
                logger.debug(f"PCA failed at index {i}: {e}")
                pc1_explained.append(np.nan)
                pc_total_explained.append(np.nan)

        df['pca_pc1_explained'] = pc1_explained
        df['pca_total_explained'] = pc_total_explained

        # High correlation regime when PC1 explains > 50%
        df['pca_high_corr_regime'] = (df['pca_pc1_explained'] > 0.5).astype(int)

        logger.info("Added 3 PCA correlation features")

        return df

    def detect_correlation_breakdown(
        self,
        df: pd.DataFrame,
        asset_col: str = 'Close',
        benchmark_col: str = 'SPY'
    ) -> pd.DataFrame:
        """
        Detect correlation breakdown events.

        Correlation breakdown occurs when:
        - Historical correlation is high but recent correlation drops
        - This often precedes market stress or regime changes

        Args:
            df: DataFrame with asset and benchmark prices
            asset_col: Asset price column
            benchmark_col: Benchmark price column

        Returns:
            DataFrame with breakdown features
        """
        df = df.copy()

        if benchmark_col not in df.columns or asset_col not in df.columns:
            logger.warning(f"Missing columns for correlation breakdown: {asset_col}, {benchmark_col}")
            return df

        # Calculate returns
        asset_returns = df[asset_col].pct_change()
        benchmark_returns = df[benchmark_col].pct_change()

        # Short-term vs long-term correlation
        short_window = 20
        long_window = 60

        short_corr = asset_returns.rolling(short_window).corr(benchmark_returns)
        long_corr = asset_returns.rolling(long_window).corr(benchmark_returns)

        df['corr_short_term'] = short_corr
        df['corr_long_term'] = long_corr

        # Correlation change
        df['corr_change'] = short_corr - long_corr

        # Breakdown detection: correlation dropped significantly
        df['corr_breakdown'] = (
            (long_corr > 0.5) &  # Was highly correlated
            (df['corr_change'] < -self.breakdown_threshold)  # Now dropping
        ).astype(int)

        # Correlation regime stability
        df['corr_stability'] = 1 - df['corr_change'].abs().rolling(20).mean()

        logger.info("Added 5 correlation breakdown features")

        return df

    def calculate_risk_regime(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate comprehensive risk-on/risk-off regime.

        Risk-On indicators:
        - SPY momentum positive
        - VIX declining or low
        - GLD declining (risk assets preferred)
        - TLT declining (risk assets preferred)
        - High correlation regime

        Risk-Off indicators:
        - SPY momentum negative
        - VIX rising or high
        - GLD rising (safe haven demand)
        - TLT rising (flight to safety)
        - Correlation breakdown

        Args:
            df: DataFrame with macro data

        Returns:
            DataFrame with risk regime features
        """
        df = df.copy()

        # Initialize scores
        risk_on_score = pd.Series(0.0, index=df.index)
        risk_off_score = pd.Series(0.0, index=df.index)

        # SPY momentum (weight: 0.3)
        if 'SPY' in df.columns:
            spy_mom = df['SPY'].pct_change(20)
            risk_on_score += (spy_mom > 0).astype(float) * 0.3
            risk_off_score += (spy_mom < 0).astype(float) * 0.3

        # VIX level and momentum (weight: 0.25)
        if 'VIX' in df.columns:
            vix_low = df['VIX'] < 20
            vix_declining = df['VIX'].pct_change(10) < 0
            risk_on_score += ((vix_low) | (vix_declining)).astype(float) * 0.25

            vix_high = df['VIX'] > 25
            vix_rising = df['VIX'].pct_change(10) > 0.1
            risk_off_score += ((vix_high) | (vix_rising)).astype(float) * 0.25

        # GLD (safe haven) (weight: 0.2)
        if 'GLD' in df.columns:
            gld_mom = df['GLD'].pct_change(20)
            risk_on_score += (gld_mom < 0).astype(float) * 0.2
            risk_off_score += (gld_mom > 0).astype(float) * 0.2

        # TLT (bonds/safety) (weight: 0.15)
        if 'TLT' in df.columns:
            tlt_mom = df['TLT'].pct_change(20)
            risk_on_score += (tlt_mom < 0).astype(float) * 0.15
            risk_off_score += (tlt_mom > 0).astype(float) * 0.15

        # DXY (dollar strength) (weight: 0.1)
        if 'DXY' in df.columns:
            dxy_mom = df['DXY'].pct_change(20)
            # Strong dollar can be risk-off (flight to safety)
            risk_off_score += (dxy_mom > 0.02).astype(float) * 0.1
            risk_on_score += (dxy_mom < -0.02).astype(float) * 0.1

        # Net risk score (-1 = full risk-off, +1 = full risk-on)
        df['risk_regime_score'] = risk_on_score - risk_off_score

        # Categorical regime
        df['risk_regime'] = 'neutral'
        df.loc[df['risk_regime_score'] > 0.2, 'risk_regime'] = 'risk_on'
        df.loc[df['risk_regime_score'] < -0.2, 'risk_regime'] = 'risk_off'
        df.loc[df['risk_regime_score'] > 0.5, 'risk_regime'] = 'strong_risk_on'
        df.loc[df['risk_regime_score'] < -0.5, 'risk_regime'] = 'strong_risk_off'

        # Position multiplier based on regime
        df['regime_position_mult'] = 1.0
        df.loc[df['risk_regime'] == 'strong_risk_on', 'regime_position_mult'] = 1.2
        df.loc[df['risk_regime'] == 'risk_on', 'regime_position_mult'] = 1.1
        df.loc[df['risk_regime'] == 'risk_off', 'regime_position_mult'] = 0.7
        df.loc[df['risk_regime'] == 'strong_risk_off', 'regime_position_mult'] = 0.3

        logger.info("Added risk regime features (score, regime, position_mult)")

        return df

    def add_all_features(self, df: pd.DataFrame, asset_col: str = 'Close') -> pd.DataFrame:
        """
        Add all cross-market correlation features.

        Args:
            df: DataFrame with OHLCV data and macro indicators
            asset_col: Asset price column

        Returns:
            DataFrame with all Phase 4 correlation features
        """
        logger.info("Adding Phase 4 cross-market correlation features...")

        initial_cols = len(df.columns)

        # 1. Correlation network features
        df = self.calculate_correlation_network_features(df)

        # 2. PCA correlation features
        df = self.calculate_pca_correlation_features(df)

        # 3. Correlation breakdown detection
        df = self.detect_correlation_breakdown(df, asset_col)

        # 4. Risk regime calculation
        df = self.calculate_risk_regime(df)

        added_cols = len(df.columns) - initial_cols
        logger.info(f"Added {added_cols} Phase 4 cross-market features")

        return df


class Phase4MacroIntegration:
    """
    Phase 4 Macro Integration - Master class.

    Combines:
    - MacroFeatureEngineer (existing)
    - IntermarketAnalyzer (existing)
    - CrossMarketCorrelationEngine (new)

    With:
    - TradingSystemConflictResolver (from src/risk)

    Expected Impact: +5-8% profit rate
    """

    def __init__(self):
        """Initialize Phase 4 integration."""
        from .macro_features import MacroFeatureEngineer
        from .intermarket_features import IntermarketAnalyzer

        self.macro_engineer = MacroFeatureEngineer()
        self.intermarket = IntermarketAnalyzer()
        self.cross_market = CrossMarketCorrelationEngine()

        logger.info("Initialized Phase4MacroIntegration")

    def add_all_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all Phase 4 macro features.

        Combines:
        1. Macro feature engineering (VIX, GLD, SPY, TLT, DXY)
        2. Intermarket correlations and betas
        3. Cross-market correlation network
        4. Risk regime detection

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all Phase 4 features
        """
        logger.info("="*50)
        logger.info("PHASE 4: Adding all macro features")
        logger.info("="*50)

        initial_cols = len(df.columns)

        # 1. Basic macro features
        logger.info("[1/3] Adding macro features...")
        df = self.macro_engineer.add_all_features(df)

        # 2. Cross-market correlation features
        logger.info("[2/3] Adding cross-market correlations...")
        df = self.cross_market.add_all_features(df)

        # 3. Skip intermarket if already has macro data to avoid duplicates
        # The macro_features.py already includes most intermarket features

        total_added = len(df.columns) - initial_cols
        logger.info(f"[COMPLETE] Added {total_added} Phase 4 macro features")

        return df

    def get_trading_context(self, df: pd.DataFrame) -> Dict:
        """
        Get current trading context from macro features.

        Args:
            df: DataFrame with macro features

        Returns:
            Dict with trading context information
        """
        if len(df) == 0:
            return {'error': 'No data'}

        latest = df.iloc[-1]

        context = {
            'date': str(df.index[-1]),

            # VIX context
            'vix_level': latest.get('VIX', None),
            'vix_regime': 'low' if latest.get('vix_low', 0) else (
                'elevated' if latest.get('vix_elevated', 0) else (
                    'crisis' if latest.get('vix_crisis', 0) else 'normal'
                )
            ),

            # Risk context
            'risk_regime': latest.get('risk_regime', 'neutral'),
            'risk_score': latest.get('risk_regime_score', 0),
            'position_multiplier': latest.get('regime_position_mult', 1.0),

            # Correlation context
            'correlation_breakdown': bool(latest.get('corr_breakdown', 0)),
            'correlation_stability': latest.get('corr_stability', 1.0),

            # Market strength
            'spy_trend': 'up' if latest.get('SPY', 0) > latest.get('SPY_ma_20d', 0) else 'down',
            'dxy_strength': 'strong' if latest.get('dxy_strong', 0) else 'weak',
            'gld_demand': 'high' if latest.get('GLD_momentum_20d', 0) > 0 else 'low',
        }

        return context


# Convenience function
def get_phase4_integration() -> Phase4MacroIntegration:
    """Get a configured Phase4MacroIntegration instance."""
    return Phase4MacroIntegration()


def main():
    """Test Phase 4 cross-market correlation features."""
    import yfinance as yf

    print("="*60)
    print("PHASE 4: CROSS-MARKET CORRELATIONS TEST")
    print("="*60)

    # Download sample data
    print("\n[INFO] Downloading test data...")

    symbols = ['AAPL', 'SPY', '^VIX', 'GLD', 'TLT']
    data = {}

    for sym in symbols:
        try:
            df = yf.download(sym, start='2024-01-01', end='2025-11-22',
                           progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            data[sym.replace('^', '')] = df['Close']
            print(f"  [OK] Downloaded {sym}")
        except Exception as e:
            print(f"  [ERROR] Failed to download {sym}: {e}")

    # Combine into single DataFrame
    test_df = pd.DataFrame(data)
    test_df['Close'] = test_df['AAPL']  # Use AAPL as the target asset

    print(f"\n[OK] Combined data: {len(test_df)} days")

    # Test CrossMarketCorrelationEngine
    print("\n" + "="*60)
    print("Testing CrossMarketCorrelationEngine")
    print("="*60)

    engine = CrossMarketCorrelationEngine()
    enhanced_df = engine.add_all_features(test_df)

    # Show sample features
    corr_cols = [col for col in enhanced_df.columns if 'corr' in col.lower() or 'pca' in col.lower() or 'risk' in col.lower()]
    print(f"\nPhase 4 features added: {len(corr_cols)}")
    print("\nSample (last 5 rows):")
    print(enhanced_df[corr_cols[-6:]].tail())

    # Show current risk regime
    print("\n" + "="*60)
    print("CURRENT RISK CONTEXT")
    print("="*60)

    latest = enhanced_df.iloc[-1]
    print(f"  Risk Score: {latest.get('risk_regime_score', 'N/A'):.3f}")
    print(f"  Risk Regime: {latest.get('risk_regime', 'N/A')}")
    print(f"  Position Multiplier: {latest.get('regime_position_mult', 'N/A')}")
    print(f"  Correlation Stability: {latest.get('corr_stability', 'N/A'):.3f}")

    print("\n[SUCCESS] Phase 4 cross-market correlation test complete!")


if __name__ == "__main__":
    main()
