"""
Pytest Configuration and Fixtures
==================================
Shared fixtures for profit maximization tests.
"""

import pytest
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.us_intl_optimizer import (
    USIntlModelOptimizer,
    create_optimizer,
    SignalOptimization,
    AssetClass
)


# ============================================================================
# OPTIMIZER FIXTURES
# ============================================================================

@pytest.fixture
def basic_optimizer():
    """Basic optimizer with default settings."""
    return create_optimizer()


@pytest.fixture
def optimizer_with_win_rates():
    """Optimizer with pre-configured win rates."""
    win_rates = {
        # US Stocks - high performers
        'AAPL': 0.75,
        'MSFT': 0.72,
        'GOOGL': 0.70,
        'TSLA': 0.60,
        'NVDA': 0.68,

        # ETFs
        'SPY': 0.70,
        'QQQ': 0.68,

        # Commodities - lower performers
        'GC=F': 0.45,
        'CL=F': 0.40,
        'NG=F': 0.33,

        # Crypto - mixed
        'BTC-USD': 0.45,
        'ETH-USD': 0.42,

        # China stocks - worst performers
        '0700.HK': 0.30,
        '9988.HK': 0.25,
        '2319.HK': 0.28,
    }
    return create_optimizer(
        historical_win_rates=win_rates,
        enable_kelly=True,
        enable_dynamic_sizing=True
    )


@pytest.fixture
def optimizer_no_dynamic_sizing():
    """Optimizer without dynamic sizing (for testing base multipliers)."""
    return create_optimizer(
        enable_kelly=False,
        enable_dynamic_sizing=False
    )


@pytest.fixture
def optimizer_kelly_only():
    """Optimizer with only Kelly Criterion enabled."""
    return create_optimizer(
        enable_kelly=True,
        enable_dynamic_sizing=False
    )


# ============================================================================
# SAMPLE DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_buy_signals():
    """Sample BUY signals for batch testing."""
    return [
        {'ticker': 'AAPL', 'signal_type': 'BUY', 'confidence': 0.85, 'volatility': 0.25},
        {'ticker': 'MSFT', 'signal_type': 'BUY', 'confidence': 0.80, 'volatility': 0.22},
        {'ticker': 'GOOGL', 'signal_type': 'BUY', 'confidence': 0.78, 'volatility': 0.28},
        {'ticker': 'TSLA', 'signal_type': 'BUY', 'confidence': 0.72, 'volatility': 0.45},
        {'ticker': 'NVDA', 'signal_type': 'BUY', 'confidence': 0.88, 'volatility': 0.35},
    ]


@pytest.fixture
def sample_sell_signals():
    """Sample SELL signals for batch testing."""
    return [
        {'ticker': 'AAPL', 'signal_type': 'SELL', 'confidence': 0.85, 'volatility': 0.25},
        {'ticker': 'MSFT', 'signal_type': 'SELL', 'confidence': 0.82, 'volatility': 0.22},
        {'ticker': 'GC=F', 'signal_type': 'SELL', 'confidence': 0.90, 'volatility': 0.18},
        {'ticker': 'NG=F', 'signal_type': 'SELL', 'confidence': 0.95, 'volatility': 0.50},  # Should be blocked
        {'ticker': 'BTC-USD', 'signal_type': 'SELL', 'confidence': 0.88, 'volatility': 0.60},
    ]


@pytest.fixture
def sample_mixed_signals():
    """Mixed BUY and SELL signals."""
    return [
        {'ticker': 'AAPL', 'signal_type': 'BUY', 'confidence': 0.85},
        {'ticker': 'MSFT', 'signal_type': 'SELL', 'confidence': 0.82},
        {'ticker': 'LUMN', 'signal_type': 'BUY', 'confidence': 0.94},   # Should be blocked
        {'ticker': 'NG=F', 'signal_type': 'SELL', 'confidence': 0.90},  # Should be blocked
        {'ticker': 'BTC-USD', 'signal_type': 'BUY', 'confidence': 0.75},
        {'ticker': 'USDT-USD', 'signal_type': 'SELL', 'confidence': 0.88},  # Should be blocked
        {'ticker': '0700.HK', 'signal_type': 'BUY', 'confidence': 0.70},
        {'ticker': 'SPY', 'signal_type': 'BUY', 'confidence': 0.78},
    ]


@pytest.fixture
def blocklist_test_signals():
    """Signals specifically for testing blocklists."""
    return [
        # Extended blocklist (Fix 8)
        {'ticker': 'LUMN', 'signal_type': 'BUY', 'confidence': 0.94},
        {'ticker': '3800.HK', 'signal_type': 'BUY', 'confidence': 0.85},

        # SELL blacklist (Fix 6)
        {'ticker': 'NG=F', 'signal_type': 'SELL', 'confidence': 0.95},

        # Commodity SELL blacklist (Fix 13)
        {'ticker': 'ZC=F', 'signal_type': 'SELL', 'confidence': 0.90},
        {'ticker': 'ZW=F', 'signal_type': 'SELL', 'confidence': 0.88},

        # Stablecoin SELL blacklist (Fix 13)
        {'ticker': 'USDT-USD', 'signal_type': 'SELL', 'confidence': 0.92},
        {'ticker': 'USDC-USD', 'signal_type': 'SELL', 'confidence': 0.90},
    ]


@pytest.fixture
def asset_class_test_signals():
    """Signals for testing asset class classification."""
    return [
        # US Stocks
        {'ticker': 'AAPL', 'expected_class': 'stock'},
        {'ticker': 'TSLA', 'expected_class': 'stock'},

        # ETFs
        {'ticker': 'SPY', 'expected_class': 'etf'},
        {'ticker': 'QQQ', 'expected_class': 'etf'},

        # Commodities
        {'ticker': 'GC=F', 'expected_class': 'commodity'},
        {'ticker': 'CL=F', 'expected_class': 'commodity'},

        # Crypto
        {'ticker': 'BTC-USD', 'expected_class': 'cryptocurrency'},
        {'ticker': 'ETH-USD', 'expected_class': 'cryptocurrency'},

        # Forex
        {'ticker': 'EURUSD=X', 'expected_class': 'forex'},

        # China stocks
        {'ticker': '0700.HK', 'expected_class': 'china_stock'},
        {'ticker': '9988.HK', 'expected_class': 'china_stock'},
    ]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@pytest.fixture
def assert_blocked():
    """Helper to assert signal is blocked."""
    def _assert_blocked(result: SignalOptimization, reason_contains: str = None):
        assert result.blocked is True, f"Expected signal to be blocked, got blocked={result.blocked}"
        if reason_contains:
            assert reason_contains.upper() in (result.block_reason or '').upper(), \
                f"Expected '{reason_contains}' in block reason, got: {result.block_reason}"
    return _assert_blocked


@pytest.fixture
def assert_not_blocked():
    """Helper to assert signal is not blocked."""
    def _assert_not_blocked(result: SignalOptimization):
        assert result.blocked is False, f"Expected signal not blocked, got blocked={result.blocked}, reason={result.block_reason}"
    return _assert_not_blocked


@pytest.fixture
def assert_fix_applied():
    """Helper to assert a specific fix was applied."""
    def _assert_fix_applied(result: SignalOptimization, fix_number: int):
        fix_str = f"Fix {fix_number}"
        assert fix_str in str(result.fixes_applied), \
            f"Expected {fix_str} in fixes, got: {result.fixes_applied}"
    return _assert_fix_applied


# ============================================================================
# EXPECTED VALUES
# ============================================================================

@pytest.fixture
def expected_stop_losses():
    """Expected stop-losses by asset class."""
    return {
        'stock': 0.08,
        'etf': 0.08,
        'commodity': 0.10,
        'cryptocurrency': 0.12,
        'forex': 0.06,
        'china_stock': 0.08,
    }


@pytest.fixture
def expected_buy_boosts():
    """Expected BUY position boosts by asset class."""
    return {
        'stock': 1.30,
        'etf': 1.20,
        'forex': 1.10,
        'china_stock': 1.00,
        'cryptocurrency': 0.90,
        'commodity': 0.80,
    }


@pytest.fixture
def expected_sell_multipliers():
    """Expected SELL position multipliers by asset class."""
    return {
        'commodity': 0.30,
        'cryptocurrency': 0.50,
        'stock': 0.50,
        'forex': 0.60,
        'etf': 0.50,
        'china_stock': 0.50,
    }


@pytest.fixture
def win_rate_multipliers():
    """Expected win rate multipliers."""
    return {
        (0.80, 1.00): 2.0,
        (0.70, 0.80): 1.5,
        (0.60, 0.70): 1.2,
        (0.50, 0.60): 1.0,
        (0.40, 0.50): 0.7,
        (0.30, 0.40): 0.5,
        (0.20, 0.30): 0.3,
        (0.00, 0.20): 0.1,
    }


# ============================================================================
# TEST DATA FILE LOADING
# ============================================================================

@pytest.fixture
def sample_signals_from_file():
    """Load sample signals from JSON file if it exists."""
    test_data_path = os.path.join(
        os.path.dirname(__file__),
        'test_data',
        'sample_signals.json'
    )

    if os.path.exists(test_data_path):
        with open(test_data_path, 'r') as f:
            return json.load(f)
    else:
        # Return default if file doesn't exist
        return {
            'buy_signals': [
                {'ticker': 'AAPL', 'signal_type': 'BUY', 'confidence': 0.85},
            ],
            'sell_signals': [
                {'ticker': 'AAPL', 'signal_type': 'SELL', 'confidence': 0.85},
            ],
        }


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add skip marker for slow tests if not explicitly requested
    if not config.getoption("--runslow", default=False):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


# ============================================================================
# FIX 16-19: MARKET REGIME AND ADAPTIVE BLOCKING FIXTURES
# ============================================================================

# Import risk management modules (handle import errors gracefully)
try:
    from src.risk_management.market_regime_detector import (
        MarketRegimeDetector,
        MarketRegime,
        RegimeIndicators,
        create_indicators_from_data,
    )
    from src.risk_management.adaptive_blocker import (
        AdaptiveBlocker,
        BlockingLevel,
    )
    from src.risk_management.position_adjuster import (
        PositionAdjuster,
    )
    RISK_MANAGEMENT_AVAILABLE = True
except ImportError:
    RISK_MANAGEMENT_AVAILABLE = False


@pytest.fixture
def regime_detector():
    """Create a MarketRegimeDetector instance."""
    if not RISK_MANAGEMENT_AVAILABLE:
        pytest.skip("Risk management modules not available")
    return MarketRegimeDetector()


@pytest.fixture
def adaptive_blocker(regime_detector):
    """Create an AdaptiveBlocker instance."""
    if not RISK_MANAGEMENT_AVAILABLE:
        pytest.skip("Risk management modules not available")
    return AdaptiveBlocker(regime_detector=regime_detector)


@pytest.fixture
def position_adjuster(regime_detector):
    """Create a PositionAdjuster instance with historical win rates."""
    if not RISK_MANAGEMENT_AVAILABLE:
        pytest.skip("Risk management modules not available")
    return PositionAdjuster(
        regime_detector=regime_detector,
        historical_win_rates={
            'AAPL': 0.72,
            'MSFT': 0.68,
            'USDJPY=X': 0.35,
            'EURJPY=X': 0.38,
            'CL=F': 0.42,
            'BTC-USD': 0.55,
        },
    )


# Market Condition Fixtures

@pytest.fixture
def bull_market_indicators():
    """Create indicators for a bull market regime."""
    if not RISK_MANAGEMENT_AVAILABLE:
        pytest.skip("Risk management modules not available")
    return create_indicators_from_data(
        vix=14.0,
        spy_return_20d=0.08,
        spy_return_60d=0.15,
        spy_above_200ma=True,
        gold_return_20d=-0.02,
    )


@pytest.fixture
def bear_market_indicators():
    """Create indicators for a bear market regime."""
    if not RISK_MANAGEMENT_AVAILABLE:
        pytest.skip("Risk management modules not available")
    return create_indicators_from_data(
        vix=30.0,
        spy_return_20d=-0.10,
        spy_return_60d=-0.15,
        spy_above_200ma=False,
        gold_return_20d=0.05,
    )


@pytest.fixture
def crisis_indicators():
    """Create indicators for a crisis regime."""
    if not RISK_MANAGEMENT_AVAILABLE:
        pytest.skip("Risk management modules not available")
    return create_indicators_from_data(
        vix=45.0,
        spy_return_20d=-0.15,
        spy_return_60d=-0.20,
        spy_above_200ma=False,
        high_yield_spread=9.0,
        gold_return_20d=0.08,
    )


@pytest.fixture
def jpy_tickers():
    """List of JPY currency pair tickers."""
    return ['USDJPY=X', 'EURJPY=X', 'GBPJPY=X', 'AUDJPY=X']


@pytest.fixture
def crude_tickers():
    """List of crude oil tickers."""
    return ['CL=F', 'BZ=F']


@pytest.fixture
def china_tickers():
    """List of China A-share tickers."""
    return ['600519.SS', '000858.SZ', '601318.SS', '000002.SZ']


# Helper functions for asset classification

def is_china_a_share(ticker: str) -> bool:
    """Check if ticker is a China A-share."""
    ticker_upper = ticker.upper()
    return ticker_upper.endswith('.SS') or ticker_upper.endswith('.SZ')


def is_jpy_pair(ticker: str) -> bool:
    """Check if ticker is a JPY currency pair."""
    return 'JPY' in ticker.upper()


def is_crude_oil(ticker: str) -> bool:
    """Check if ticker is crude oil."""
    return ticker.upper() in ['CL=F', 'BZ=F']
