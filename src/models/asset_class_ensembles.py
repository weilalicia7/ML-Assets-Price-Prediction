"""
Asset Class Specialized Ensembles Module

Phase 2 Implementation: 7 specialized ensembles for different asset classes:
1. EquitySpecificEnsemble - US stocks
2. ForexSpecificEnsemble - Currency pairs
3. CryptoSpecificEnsemble - Crypto-related stocks
4. CommoditySpecificEnsemble - Commodities and commodity stocks
5. InternationalEnsemble - ADRs and international stocks
6. BondSpecificEnsemble - Bonds and rate-sensitive assets
7. ETFSpecificEnsemble - ETFs and indices

Each ensemble uses asset-class specific features and model configurations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Import feature modules
try:
    from src.features.international_features import InternationalFeatures
    from src.features.crypto_features import CryptoFeatures
    from src.features.commodity_features import CommodityFeatures
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False


class BaseEnsemble(ABC):
    """Base class for all asset-class specific ensembles."""

    def __init__(self, config: Dict = None):
        """
        Initialize base ensemble.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.models = {}
        self.feature_importance = {}
        self.performance_history = []

    @abstractmethod
    def get_asset_class_features(self) -> List[str]:
        """Return list of features specific to this asset class."""
        pass

    @abstractmethod
    def prepare_features(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Prepare asset-class specific features."""
        pass

    def predict(self, data: pd.DataFrame, ticker: str) -> Dict:
        """
        Generate prediction for an asset.

        Args:
            data: OHLCV data
            ticker: Asset ticker

        Returns:
            Dict with prediction results
        """
        # Prepare features
        featured_data = self.prepare_features(data, ticker)

        # Generate signals
        signals = self._generate_signals(featured_data)

        # Combine signals
        combined_signal = self._combine_signals(signals)

        # Calculate confidence
        confidence = self._calculate_confidence(signals, featured_data)

        return {
            'ticker': ticker,
            'signal': combined_signal,
            'confidence': confidence,
            'signals': signals,
            'features_used': len(self.get_asset_class_features())
        }

    def _generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """Generate individual signals from features."""
        signals = {}

        # Momentum signal
        if 'Close' in data.columns:
            returns_20d = data['Close'].pct_change(20).iloc[-1]
            signals['momentum'] = np.tanh(returns_20d * 10)  # Scale and bound

        # Volatility signal (reduce in high vol)
        if 'Close' in data.columns:
            vol = data['Close'].pct_change().rolling(20).std().iloc[-1]
            signals['volatility'] = 1 - min(vol * 10, 1)  # Lower is better

        # Mean-reversion signal
        if 'Close' in data.columns:
            ma_20 = data['Close'].rolling(20).mean().iloc[-1]
            current = data['Close'].iloc[-1]
            deviation = (current - ma_20) / ma_20
            signals['mean_reversion'] = -np.tanh(deviation * 5)

        return signals

    def _combine_signals(self, signals: Dict[str, float]) -> float:
        """Combine multiple signals into final prediction."""
        if not signals:
            return 0.0

        # Default equal weighting
        weights = {k: 1.0 / len(signals) for k in signals}

        combined = sum(signals[k] * weights[k] for k in signals)
        return float(np.clip(combined, -1, 1))

    def _calculate_confidence(self, signals: Dict[str, float], data: pd.DataFrame) -> float:
        """Calculate confidence in the prediction."""
        if not signals:
            return 0.0

        # Base confidence on signal agreement
        signal_values = list(signals.values())
        if len(signal_values) < 2:
            return 0.5

        # Agreement: all signals same direction = high confidence
        signs = [np.sign(s) for s in signal_values if s != 0]
        if len(signs) == 0:
            return 0.3

        agreement = abs(sum(signs)) / len(signs)
        return float(np.clip(0.3 + 0.5 * agreement, 0, 1))


class EquitySpecificEnsemble(BaseEnsemble):
    """Ensemble specialized for US equities."""

    EQUITY_TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'CRM', 'ORCL',
        'JPM', 'BAC', 'GS', 'MS', 'C', 'WFC',
        'AMZN', 'TSLA', 'DIS', 'NKE', 'SBUX', 'MCD'
    ]

    def get_asset_class_features(self) -> List[str]:
        return [
            'sector_momentum', 'earnings_surprise', 'analyst_revisions',
            'short_interest', 'institutional_flows', 'market_beta',
            'size_factor', 'value_factor', 'quality_factor'
        ]

    def prepare_features(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Prepare equity-specific features."""
        df = data.copy()

        if len(df) < 60:
            return df

        close = df['Close']
        volume = df['Volume']
        returns = close.pct_change()

        # Market beta proxy (correlation with market momentum)
        df['market_beta_proxy'] = returns.rolling(60).std() / returns.rolling(60).mean().abs()
        df['market_beta_proxy'] = df['market_beta_proxy'].clip(-5, 5)

        # Size factor proxy (volume-based)
        df['size_proxy'] = np.log(volume.rolling(20).mean() + 1)

        # Value factor proxy (price level relative to range)
        high_52w = close.rolling(252).max()
        low_52w = close.rolling(252).min()
        df['value_proxy'] = (close - low_52w) / (high_52w - low_52w + 0.01)

        # Quality factor proxy (volatility-adjusted return)
        df['quality_proxy'] = returns.rolling(60).mean() / (returns.rolling(60).std() + 0.001)

        # Momentum factor
        df['momentum_factor'] = close.pct_change(60)

        # Volume trend
        vol_ma = volume.rolling(20).mean()
        df['volume_trend'] = (volume - vol_ma) / vol_ma

        return df

    def _generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        signals = super()._generate_signals(data)

        # Add equity-specific signals
        if 'momentum_factor' in data.columns:
            signals['momentum_factor'] = np.tanh(data['momentum_factor'].iloc[-1] * 5)

        if 'quality_proxy' in data.columns:
            signals['quality'] = np.tanh(data['quality_proxy'].iloc[-1])

        if 'value_proxy' in data.columns:
            # Contrarian: buy when value proxy is low (near 52w low)
            signals['value'] = -np.tanh((data['value_proxy'].iloc[-1] - 0.5) * 2)

        return signals


class ForexSpecificEnsemble(BaseEnsemble):
    """Ensemble specialized for Forex/currency pairs."""

    FOREX_TICKERS = [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X',
        'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X',
        'FXE', 'FXY', 'FXB', 'FXA', 'FXC'
    ]

    def get_asset_class_features(self) -> List[str]:
        return [
            'carry_trade', 'rate_differential', 'purchasing_power_parity',
            'current_account', 'central_bank_policy', 'risk_sentiment',
            'correlation_matrix', 'volatility_regime'
        ]

    def prepare_features(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Prepare forex-specific features."""
        df = data.copy()

        if len(df) < 60:
            return df

        close = df['Close']
        high = df['High']
        low = df['Low']
        returns = close.pct_change()

        # FX volatility (critical for forex)
        df['fx_volatility'] = returns.rolling(20).std() * np.sqrt(252)

        # Range as percentage of price
        df['daily_range_pct'] = (high - low) / close

        # Trend strength (ADX proxy)
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean() / close

        # Mean reversion tendency (FX often mean-reverts)
        ma_50 = close.rolling(50).mean()
        ma_200 = close.rolling(200).mean()
        df['ma_deviation'] = (close - ma_50) / ma_50
        df['trend_strength'] = (ma_50 - ma_200) / ma_200

        # Risk sentiment proxy (volatility regime)
        vol_percentile = returns.rolling(252).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 0.0001) if len(x) > 1 else 0.5,
            raw=False
        )
        df['risk_regime'] = vol_percentile

        return df

    def _generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        signals = super()._generate_signals(data)

        # FX-specific: strong mean-reversion signal
        if 'ma_deviation' in data.columns:
            deviation = data['ma_deviation'].iloc[-1]
            signals['mean_reversion_fx'] = -np.tanh(deviation * 10)

        # Trend following for trending pairs
        if 'trend_strength' in data.columns:
            trend = data['trend_strength'].iloc[-1]
            signals['trend'] = np.tanh(trend * 5)

        # Risk-off signal
        if 'risk_regime' in data.columns:
            risk = data['risk_regime'].iloc[-1]
            signals['risk_sentiment'] = 1 - risk  # Lower risk = more bullish

        return signals


class CryptoSpecificEnsemble(BaseEnsemble):
    """Ensemble specialized for crypto-related stocks."""

    CRYPTO_TICKERS = ['COIN', 'MSTR', 'MARA', 'RIOT', 'CLSK', 'HUT', 'BITF']

    def __init__(self, config: Dict = None):
        super().__init__(config)
        if FEATURES_AVAILABLE:
            self.crypto_features = CryptoFeatures()
        else:
            self.crypto_features = None

    def get_asset_class_features(self) -> List[str]:
        return [
            'btc_correlation', 'btc_beta', 'crypto_sentiment',
            'btc_momentum', 'eth_correlation', 'fear_greed_proxy',
            'crypto_volatility', 'hodl_proxy'
        ]

    def prepare_features(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Prepare crypto-specific features."""
        if self.crypto_features:
            return self.crypto_features.create_all_features(data, ticker)

        # Fallback: basic features
        df = data.copy()
        close = df['Close']
        returns = close.pct_change()

        df['crypto_volatility'] = returns.rolling(20).std() * np.sqrt(252)
        df['momentum_20d'] = close.pct_change(20)

        return df

    def _generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        signals = super()._generate_signals(data)

        # BTC correlation signal
        if 'btc_correlation' in data.columns:
            btc_corr = data['btc_correlation'].iloc[-1]
            signals['btc_following'] = btc_corr

        # Crypto sentiment
        if 'crypto_fear_greed_proxy' in data.columns:
            sentiment = data['crypto_fear_greed_proxy'].iloc[-1]
            # Contrarian at extremes
            if sentiment > 75:
                signals['sentiment'] = -0.5  # Extreme greed = bearish
            elif sentiment < 25:
                signals['sentiment'] = 0.5   # Extreme fear = bullish
            else:
                signals['sentiment'] = (sentiment - 50) / 100

        # BTC momentum
        if 'btc_momentum' in data.columns:
            btc_mom = data['btc_momentum'].iloc[-1]
            signals['btc_momentum'] = np.tanh(btc_mom * 3)

        return signals


class CommoditySpecificEnsemble(BaseEnsemble):
    """Ensemble specialized for commodities."""

    COMMODITY_TICKERS = [
        'GC=F', 'SI=F', 'CL=F', 'NG=F', 'HG=F',
        'GLD', 'SLV', 'USO', 'UNG',
        'XOM', 'CVX', 'NEM', 'GOLD'
    ]

    def __init__(self, config: Dict = None):
        super().__init__(config)
        if FEATURES_AVAILABLE:
            self.commodity_features = CommodityFeatures()
        else:
            self.commodity_features = None

    def get_asset_class_features(self) -> List[str]:
        return [
            'oil_correlation', 'gold_correlation', 'dxy_inverse',
            'seasonal_factor', 'supply_demand_proxy', 'inventory_proxy',
            'contango_backwardation', 'inflation_hedge'
        ]

    def prepare_features(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Prepare commodity-specific features."""
        if self.commodity_features:
            return self.commodity_features.create_all_features(data, ticker)

        # Fallback: basic features
        df = data.copy()
        close = df['Close']

        df['seasonal_month'] = pd.to_datetime(df.index).month
        df['momentum_20d'] = close.pct_change(20)

        return df

    def _generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        signals = super()._generate_signals(data)

        # Dollar inverse signal (commodities negatively correlated with USD)
        if 'dxy_correlation' in data.columns:
            dxy_corr = data['dxy_correlation'].iloc[-1]
            signals['dollar_inverse'] = -dxy_corr

        # Seasonal signal
        if 'oil_seasonal' in data.columns:
            signals['seasonal_oil'] = data['oil_seasonal'].iloc[-1] - 0.5

        if 'gold_seasonal' in data.columns:
            signals['seasonal_gold'] = data['gold_seasonal'].iloc[-1] - 0.5

        # Supply/demand signal
        if 'supply_demand_signal' in data.columns:
            signals['supply_demand'] = np.tanh(data['supply_demand_signal'].iloc[-1] * 2)

        return signals


class InternationalEnsemble(BaseEnsemble):
    """Ensemble specialized for international/ADR stocks."""

    INTERNATIONAL_TICKERS = [
        'BABA', 'JD', 'PDD', 'NIO', 'BIDU',
        'SONY', 'TM', 'HMC',
        'SAP', 'ASML', 'AZN', 'BP', 'SHEL',
        'TSM'
    ]

    def __init__(self, config: Dict = None):
        super().__init__(config)
        if FEATURES_AVAILABLE:
            self.intl_features = InternationalFeatures()
        else:
            self.intl_features = None

    def get_asset_class_features(self) -> List[str]:
        return [
            'fx_exposure', 'home_market_correlation', 'adr_premium',
            'geopolitical_risk', 'regional_momentum', 'currency_hedge',
            'time_zone_effect', 'local_vs_us_performance'
        ]

    def prepare_features(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Prepare international-specific features."""
        if self.intl_features:
            return self.intl_features.create_all_features(data, ticker)

        # Fallback: basic features
        df = data.copy()
        close = df['Close']

        df['overnight_gap'] = (df['Open'] - close.shift(1)) / close.shift(1)
        df['momentum_20d'] = close.pct_change(20)

        return df

    def _generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        signals = super()._generate_signals(data)

        # FX exposure signal
        if 'fx_momentum' in data.columns:
            fx_mom = data['fx_momentum'].iloc[-1]
            signals['fx_momentum'] = np.tanh(fx_mom * 5)

        # Home market signal
        if 'home_index_return_5d' in data.columns:
            home_ret = data['home_index_return_5d'].iloc[-1]
            signals['home_market'] = np.tanh(home_ret * 10)

        # Relative strength vs home
        if 'relative_strength' in data.columns:
            rel_str = data['relative_strength'].iloc[-1]
            signals['relative_strength'] = np.tanh(rel_str * 5)

        return signals


class BondSpecificEnsemble(BaseEnsemble):
    """Ensemble specialized for bonds and rate-sensitive assets."""

    BOND_TICKERS = [
        'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'AGG',
        '^TNX', '^TYX', '^FVX'
    ]

    def get_asset_class_features(self) -> List[str]:
        return [
            'yield_curve', 'duration', 'credit_spread',
            'fed_policy', 'inflation_expectation', 'flight_to_quality',
            'term_premium', 'real_yield'
        ]

    def prepare_features(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Prepare bond-specific features."""
        df = data.copy()

        if len(df) < 60:
            return df

        close = df['Close']
        returns = close.pct_change()

        # Bond volatility (usually low)
        df['bond_volatility'] = returns.rolling(20).std() * np.sqrt(252)

        # Trend (bonds can trend for long periods)
        ma_20 = close.rolling(20).mean()
        ma_60 = close.rolling(60).mean()
        df['bond_trend'] = (ma_20 - ma_60) / ma_60

        # Yield proxy (inverse of price for bonds)
        df['yield_proxy'] = -returns.rolling(20).sum()

        # Flight to quality proxy (vol spike = bonds up)
        vol_change = df['bond_volatility'].pct_change(5)
        df['flight_to_quality'] = vol_change

        return df

    def _generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        signals = super()._generate_signals(data)

        # Bond trend signal (bonds can trend strongly)
        if 'bond_trend' in data.columns:
            trend = data['bond_trend'].iloc[-1]
            signals['bond_trend'] = np.tanh(trend * 10)

        # Flight to quality
        if 'flight_to_quality' in data.columns:
            ftq = data['flight_to_quality'].iloc[-1]
            signals['flight_to_quality'] = np.tanh(ftq * 5)

        return signals


class ETFSpecificEnsemble(BaseEnsemble):
    """Ensemble specialized for ETFs and indices."""

    ETF_TICKERS = [
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO',
        'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP'
    ]

    def get_asset_class_features(self) -> List[str]:
        return [
            'market_breadth', 'sector_rotation', 'risk_on_off',
            'correlation_regime', 'volatility_regime', 'flow_momentum',
            'relative_sector_strength', 'macro_factor'
        ]

    def prepare_features(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Prepare ETF-specific features."""
        df = data.copy()

        if len(df) < 60:
            return df

        close = df['Close']
        volume = df['Volume']
        returns = close.pct_change()

        # Market breadth proxy (volume trend)
        vol_ma = volume.rolling(20).mean()
        df['breadth_proxy'] = (volume - vol_ma) / vol_ma

        # Risk-on/off regime
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std()
        df['risk_regime'] = vol_20 / vol_60

        # Trend strength
        ma_50 = close.rolling(50).mean()
        ma_200 = close.rolling(200).mean()
        df['trend_strength'] = (ma_50 - ma_200) / ma_200

        # Momentum rank
        df['momentum_20'] = close.pct_change(20)
        df['momentum_60'] = close.pct_change(60)

        return df

    def _generate_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        signals = super()._generate_signals(data)

        # Trend following for ETFs
        if 'trend_strength' in data.columns:
            trend = data['trend_strength'].iloc[-1]
            signals['trend'] = np.tanh(trend * 5)

        # Risk regime
        if 'risk_regime' in data.columns:
            risk = data['risk_regime'].iloc[-1]
            # High risk regime = reduce exposure
            signals['risk_adjustment'] = 1 - min(risk, 2) / 2

        # Breadth
        if 'breadth_proxy' in data.columns:
            breadth = data['breadth_proxy'].iloc[-1]
            signals['breadth'] = np.tanh(breadth)

        return signals


class AssetClassEnsembleFactory:
    """
    Factory for creating asset-class specific ensembles.

    Routes tickers to appropriate specialized ensemble.
    """

    def __init__(self):
        """Initialize factory with all ensemble types."""
        self.ensembles = {
            'equity': EquitySpecificEnsemble(),
            'forex': ForexSpecificEnsemble(),
            'crypto': CryptoSpecificEnsemble(),
            'commodity': CommoditySpecificEnsemble(),
            'international': InternationalEnsemble(),
            'bond': BondSpecificEnsemble(),
            'etf': ETFSpecificEnsemble()
        }

        # Build ticker mapping
        self.ticker_map = {}
        self._build_ticker_map()

    def _build_ticker_map(self):
        """Build mapping from tickers to ensemble types."""
        for ticker in EquitySpecificEnsemble.EQUITY_TICKERS:
            self.ticker_map[ticker] = 'equity'

        for ticker in ForexSpecificEnsemble.FOREX_TICKERS:
            self.ticker_map[ticker] = 'forex'

        for ticker in CryptoSpecificEnsemble.CRYPTO_TICKERS:
            self.ticker_map[ticker] = 'crypto'

        for ticker in CommoditySpecificEnsemble.COMMODITY_TICKERS:
            self.ticker_map[ticker] = 'commodity'

        for ticker in InternationalEnsemble.INTERNATIONAL_TICKERS:
            self.ticker_map[ticker] = 'international'

        for ticker in BondSpecificEnsemble.BOND_TICKERS:
            self.ticker_map[ticker] = 'bond'

        for ticker in ETFSpecificEnsemble.ETF_TICKERS:
            self.ticker_map[ticker] = 'etf'

    def classify_asset(self, ticker: str) -> str:
        """
        Classify an asset into its asset class.

        Args:
            ticker: Asset ticker

        Returns:
            Asset class name
        """
        return self.ticker_map.get(ticker, 'equity')  # Default to equity

    def get_ensemble(self, asset_class: str) -> BaseEnsemble:
        """
        Get ensemble for an asset class.

        Args:
            asset_class: Asset class name

        Returns:
            Specialized ensemble
        """
        return self.ensembles.get(asset_class, self.ensembles['equity'])

    def create_ensemble(self, ticker: str) -> BaseEnsemble:
        """
        Create ensemble for a specific ticker.

        Args:
            ticker: Asset ticker

        Returns:
            Specialized ensemble for the ticker's asset class
        """
        asset_class = self.classify_asset(ticker)
        return self.get_ensemble(asset_class)

    def predict(self, data: pd.DataFrame, ticker: str) -> Dict:
        """
        Generate prediction using appropriate ensemble.

        Args:
            data: OHLCV data
            ticker: Asset ticker

        Returns:
            Prediction results
        """
        ensemble = self.create_ensemble(ticker)
        result = ensemble.predict(data, ticker)
        result['asset_class'] = self.classify_asset(ticker)
        return result

    def get_all_asset_classes(self) -> List[str]:
        """Get list of all supported asset classes."""
        return list(self.ensembles.keys())

    def get_ensemble_info(self) -> Dict:
        """Get information about all ensembles."""
        return {
            name: {
                'features': ensemble.get_asset_class_features(),
                'ticker_count': len([t for t, c in self.ticker_map.items() if c == name])
            }
            for name, ensemble in self.ensembles.items()
        }


def main():
    """Test asset class ensembles."""
    import yfinance as yf

    print("=" * 70)
    print("ASSET CLASS ENSEMBLE TEST")
    print("=" * 70)

    factory = AssetClassEnsembleFactory()

    # Test different asset classes
    test_tickers = {
        'equity': 'AAPL',
        'crypto': 'COIN',
        'commodity': 'GLD',
        'international': 'BABA',
        'etf': 'SPY'
    }

    for asset_class, ticker in test_tickers.items():
        print(f"\n[{asset_class.upper()}] Testing {ticker}...")

        # Fetch data
        data = yf.download(ticker, period='1y', progress=False)
        if len(data) == 0:
            print(f"  SKIP - No data available")
            continue

        # Get prediction
        result = factory.predict(data, ticker)

        print(f"  Signal: {result['signal']:.3f}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Asset Class: {result['asset_class']}")
        print(f"  Features Used: {result['features_used']}")

    # Show ensemble info
    print("\n" + "=" * 70)
    print("ENSEMBLE SUMMARY")
    print("=" * 70)

    info = factory.get_ensemble_info()
    for name, details in info.items():
        print(f"\n{name.upper()}:")
        print(f"  Tickers: {details['ticker_count']}")
        print(f"  Features: {len(details['features'])}")

    print("\n[SUCCESS] Asset class ensemble test complete!")


if __name__ == "__main__":
    main()
