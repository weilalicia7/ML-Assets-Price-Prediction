# tests/test_performance_module.py
"""
Test suite for the Performance Tracker Module.
Tests the comprehensive performance tracking system for recording trade history and
calculating metrics.
"""
import pytest
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from performance.performance_tracker import (
    PerformanceTracker,
    TradeRecord,
    TradeResult
)

# Try to import integration components (may fail if dependencies not available)
try:
    from performance.integration import (
        EnhancedPositionAdjuster,
        RiskManagerWithPerformance,
        create_risk_manager_with_performance
    )
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False


# ======================================================================
# FIXTURES
# ======================================================================

@pytest.fixture
def sample_trade_record():
    """Create a sample TradeRecord for testing."""
    return TradeRecord(
        trade_id="TEST_001",
        timestamp=datetime.now() - timedelta(days=5),  # Recent date within history window
        asset="AAPL",
        direction="BUY",
        entry_price=150.0,
        exit_price=155.0,
        pnl=None,  # Will be calculated
        pnl_percent=None,
        regime="BULL",
        exit_regime="BULL",
        result=None,
        position_size=1.0,
        holding_period=None,
        tags=["test", "equity"]
    )


@pytest.fixture
def performance_tracker():
    """Create a fresh PerformanceTracker instance."""
    return PerformanceTracker()


@pytest.fixture
def tracker_with_history():
    """Create PerformanceTracker with historical trades."""
    tracker = PerformanceTracker()

    # Add winning trades for AAPL in BULL regime
    for i in range(5):
        trade = TradeRecord(
            trade_id=f"AAPL_WIN_{i}",
            timestamp=datetime.now() - timedelta(days=10 - i),
            asset="AAPL",
            direction="BUY",
            entry_price=150.0 + i,
            exit_price=155.0 + i,
            regime="BULL",
            tags=["winning", "tech"]
        )
        tracker.update_performance(trade, calculate_metrics=False)

    # Add losing trades for TSLA in VOLATILE regime
    for i in range(3):
        trade = TradeRecord(
            trade_id=f"TSLA_LOSS_{i}",
            timestamp=datetime.now() - timedelta(days=5 - i),
            asset="TSLA",
            direction="BUY",
            entry_price=200.0 + i * 10,
            exit_price=195.0 + i * 10,
            regime="VOLATILE",
            tags=["losing", "ev"]
        )
        tracker.update_performance(trade, calculate_metrics=False)

    # Add mixed trades for JPY in different regimes
    regimes = ["BULL", "BEAR", "RISK_ON", "RISK_OFF"]
    for i, regime in enumerate(regimes):
        trade = TradeRecord(
            trade_id=f"JPY_{regime}_{i}",
            timestamp=datetime.now() - timedelta(days=8 - i),
            asset="USDJPY",
            direction="SELL",
            entry_price=150.0,
            exit_price=148.0 if regime in ["BEAR", "RISK_OFF"] else 152.0,
            regime=regime,
            tags=["forex", "jpy", regime.lower()]
        )
        tracker.update_performance(trade, calculate_metrics=False)

    tracker._update_performance_history()
    return tracker


@pytest.fixture
def tracker_with_asset_types():
    """Create tracker with trades across different asset types."""
    tracker = PerformanceTracker()

    # Stocks
    stocks = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    for i, stock in enumerate(stocks):
        trade = TradeRecord(
            trade_id=f"STOCK_{i}",
            timestamp=datetime.now() - timedelta(days=i),
            asset=stock,
            direction="BUY",
            entry_price=100 + i * 10,
            exit_price=105 + i * 10,
            regime="BULL",
            tags=["equity", "stock"]
        )
        tracker.update_performance(trade, calculate_metrics=False)

    # Forex
    forex_pairs = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD"]
    for i, pair in enumerate(forex_pairs):
        trade = TradeRecord(
            trade_id=f"FOREX_{i}",
            timestamp=datetime.now() - timedelta(days=i),
            asset=pair,
            direction="SELL" if i % 2 == 0 else "BUY",
            entry_price=1.0 + i * 0.1,
            exit_price=1.05 + i * 0.1 if i % 2 == 0 else 0.95 + i * 0.1,
            regime="RISK_ON" if i % 2 == 0 else "RISK_OFF",
            tags=["forex", "currency"]
        )
        tracker.update_performance(trade, calculate_metrics=False)

    # Commodities
    commodities = ["CL=F", "GC=F", "SI=F", "NG=F"]
    for i, commodity in enumerate(commodities):
        trade = TradeRecord(
            trade_id=f"COMM_{i}",
            timestamp=datetime.now() - timedelta(days=i),
            asset=commodity,
            direction="BUY",
            entry_price=50 + i * 5,
            exit_price=55 + i * 5 if i % 3 != 0 else 48 + i * 5,
            regime="INFLATION" if i % 2 == 0 else "DEFLATION",
            tags=["commodity", "futures"]
        )
        tracker.update_performance(trade, calculate_metrics=False)

    tracker._update_performance_history()
    return tracker


@pytest.fixture
def temp_storage_path():
    """Create a temporary directory for storage tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "performance_data.json"


# ======================================================================
# TESTS FOR TradeResult ENUM
# ======================================================================

class TestTradeResultEnum:
    """Tests for TradeResult enumeration."""

    def test_enum_values(self):
        """Test that TradeResult has all expected values."""
        assert TradeResult.WIN.value == "win"
        assert TradeResult.LOSS.value == "loss"
        assert TradeResult.BREAKEVEN.value == "breakeven"
        assert TradeResult.PENDING.value == "pending"

    def test_enum_iteration(self):
        """Test iteration over TradeResult enum."""
        values = {member.value for member in TradeResult}
        expected = {"win", "loss", "breakeven", "pending"}
        assert values == expected

    def test_enum_from_value(self):
        """Test creating enum from string value."""
        assert TradeResult("win") == TradeResult.WIN
        assert TradeResult("loss") == TradeResult.LOSS
        assert TradeResult("breakeven") == TradeResult.BREAKEVEN
        assert TradeResult("pending") == TradeResult.PENDING


# ======================================================================
# TESTS FOR TradeRecord CLASS
# ======================================================================

class TestTradeRecordClass:
    """Tests for TradeRecord data class."""

    def test_trade_record_creation(self, sample_trade_record):
        """Test basic TradeRecord creation."""
        assert sample_trade_record.trade_id == "TEST_001"
        assert sample_trade_record.asset == "AAPL"
        assert sample_trade_record.direction == "BUY"
        assert sample_trade_record.entry_price == 150.0
        assert sample_trade_record.regime == "BULL"
        assert sample_trade_record.tags == ["test", "equity"]

    def test_pnl_calculation_buy_win(self):
        """Test P&L calculation for winning BUY trades."""
        trade = TradeRecord(
            trade_id="TEST_BUY_WIN",
            timestamp=datetime.now(),
            asset="GOOGL",
            direction="BUY",
            entry_price=100.0,
            exit_price=110.0,
            regime="BULL"
        )

        # P&L should be calculated automatically
        assert trade.pnl == 10.0
        assert trade.pnl_percent == 10.0
        assert trade.result == TradeResult.WIN

    def test_pnl_calculation_buy_loss(self):
        """Test P&L calculation for losing BUY trades."""
        trade = TradeRecord(
            trade_id="TEST_BUY_LOSS",
            timestamp=datetime.now(),
            asset="MSFT",
            direction="BUY",
            entry_price=100.0,
            exit_price=95.0,
            regime="BEAR"
        )

        assert trade.pnl == -5.0
        assert trade.pnl_percent == -5.0
        assert trade.result == TradeResult.LOSS

    def test_pnl_calculation_sell_win(self):
        """Test P&L calculation for winning SELL trades."""
        trade = TradeRecord(
            trade_id="TEST_SELL_WIN",
            timestamp=datetime.now(),
            asset="AMZN",
            direction="SELL",
            entry_price=100.0,
            exit_price=90.0,
            regime="BEAR"
        )

        assert trade.pnl == 10.0
        assert trade.pnl_percent == 10.0
        assert trade.result == TradeResult.WIN

    def test_pnl_calculation_sell_loss(self):
        """Test P&L calculation for losing SELL trades."""
        trade = TradeRecord(
            trade_id="TEST_SELL_LOSS",
            timestamp=datetime.now(),
            asset="NVDA",
            direction="SELL",
            entry_price=100.0,
            exit_price=105.0,
            regime="BULL"
        )

        assert trade.pnl == -5.0
        assert trade.pnl_percent == -5.0
        assert trade.result == TradeResult.LOSS

    def test_breakeven_trade(self):
        """Test P&L calculation for breakeven trades."""
        trade = TradeRecord(
            trade_id="TEST_BREAKEVEN",
            timestamp=datetime.now(),
            asset="BTCUSD",
            direction="BUY",
            entry_price=50000.0,
            exit_price=50000.0,
            regime="SIDEWAYS"
        )

        assert abs(trade.pnl) < 0.0001  # Essentially zero
        assert trade.result == TradeResult.BREAKEVEN

    def test_to_dict_serialization(self, sample_trade_record):
        """Test conversion to dictionary for serialization."""
        data = sample_trade_record.to_dict()

        assert isinstance(data, dict)
        assert data["trade_id"] == "TEST_001"
        assert data["asset"] == "AAPL"
        assert data["direction"] == "BUY"
        assert data["result"] == "win"  # Should be calculated as win

        # Check timestamp serialization
        assert isinstance(data["timestamp"], str)

    def test_from_dict_deserialization(self):
        """Test creating TradeRecord from dictionary."""
        data = {
            "trade_id": "TEST_FROM_DICT",
            "timestamp": "2024-01-15T10:30:00",
            "asset": "TSLA",
            "direction": "SELL",
            "entry_price": 250.0,
            "exit_price": 245.0,
            "pnl": 5.0,
            "pnl_percent": 2.0,
            "regime": "BEAR",
            "exit_regime": "BEAR",
            "result": "win",
            "position_size": 1.0,
            "holding_period": 86400.0,  # 1 day in seconds
            "tags": ["test", "deserialization"],
            "asset_type": "stock",
            "signal_confidence": None,
            "regime_confidence": None,
            "adjustments_applied": [],
            "blocked": False,
            "blocking_reason": None
        }

        trade = TradeRecord.from_dict(data)

        assert trade.trade_id == "TEST_FROM_DICT"
        assert trade.asset == "TSLA"
        assert trade.direction == "SELL"
        assert trade.entry_price == 250.0
        assert trade.exit_price == 245.0
        assert trade.pnl == 5.0
        assert trade.result == TradeResult.WIN
        assert trade.holding_period == timedelta(seconds=86400)
        assert "test" in trade.tags

    def test_pending_trade(self):
        """Test trade with no exit price (pending)."""
        trade = TradeRecord(
            trade_id="TEST_PENDING",
            timestamp=datetime.now(),
            asset="AAPL",
            direction="BUY",
            entry_price=150.0,
            exit_price=None,  # No exit yet
            regime="BULL"
        )

        assert trade.exit_price is None
        assert trade.pnl is None
        assert trade.result is None
        assert trade.holding_period is None

    def test_holding_period_calculation(self):
        """Test holding period calculation."""
        entry_time = datetime(2024, 1, 1, 10, 0, 0)
        exit_time = datetime(2024, 1, 2, 15, 30, 0)

        trade = TradeRecord(
            trade_id="TEST_HOLDING",
            timestamp=entry_time,
            asset="AAPL",
            direction="BUY",
            entry_price=150.0,
            exit_price=155.0,
            regime="BULL"
        )

        # Manually set holding period
        trade.holding_period = exit_time - entry_time

        assert trade.holding_period.days == 1
        assert trade.holding_period.seconds == 19800  # 5.5 hours


# ======================================================================
# TESTS FOR PerformanceTracker CLASS
# ======================================================================

class TestPerformanceTrackerBasic:
    """Basic tests for PerformanceTracker class."""

    def test_initialization(self):
        """Test PerformanceTracker initialization."""
        tracker = PerformanceTracker()

        assert tracker.trade_history == []
        assert tracker.performance_history == []
        assert tracker.max_history_days == 365
        assert tracker.rolling_window == 50
        assert tracker._next_trade_id == 1

    def test_initialization_with_params(self):
        """Test PerformanceTracker initialization with custom parameters."""
        tracker = PerformanceTracker(
            max_history_days=180,
            rolling_window=30,
            storage_path="test_data.json"
        )

        assert tracker.max_history_days == 180
        assert tracker.rolling_window == 30
        assert tracker.storage_path is not None

    def test_update_performance_basic(self, performance_tracker, sample_trade_record):
        """Test basic update_performance functionality."""
        performance_tracker.update_performance(sample_trade_record)

        assert len(performance_tracker.trade_history) == 1
        assert performance_tracker.trade_history[0].trade_id == "TEST_001"
        # Note: _next_trade_id only increments when auto-generating IDs (when trade_id is empty)
        # When trade_id is provided (like "TEST_001"), it doesn't increment
        assert performance_tracker._next_trade_id >= 1

    def test_update_performance_auto_id(self, performance_tracker):
        """Test update_performance with auto-generated trade ID."""
        trade = TradeRecord(
            trade_id="",  # Empty ID
            timestamp=datetime.now(),
            asset="AAPL",
            direction="BUY",
            entry_price=150.0,
            exit_price=155.0,
            regime="BULL"
        )

        performance_tracker.update_performance(trade)

        assert len(performance_tracker.trade_history) == 1
        # Should have auto-generated ID
        assert performance_tracker.trade_history[0].trade_id.startswith("TRADE_")

    def test_record_trade_entry(self, performance_tracker):
        """Test recording a trade entry."""
        trade_id = performance_tracker.record_trade_entry(
            asset="EURUSD",
            direction="BUY",
            entry_price=1.0850,
            regime="RISK_ON",
            position_size=1.5,
            tags=["forex", "test"]
        )

        assert trade_id.startswith("TRADE_")
        assert len(performance_tracker.trade_history) == 1

        trade = performance_tracker.trade_history[0]
        assert trade.asset == "EURUSD"
        assert trade.direction == "BUY"
        assert trade.entry_price == 1.0850
        assert trade.regime == "RISK_ON"
        assert trade.position_size == 1.5
        assert trade.exit_price is None  # Not exited yet
        assert "forex" in trade.tags

    def test_update_trade_exit_success(self, performance_tracker):
        """Test successfully updating a trade exit."""
        # Record entry
        trade_id = performance_tracker.record_trade_entry(
            asset="GBPUSD",
            direction="SELL",
            entry_price=1.2700,
            regime="RISK_OFF"
        )

        # Update exit
        success = performance_tracker.update_trade_exit(
            trade_id=trade_id,
            exit_price=1.2650,
            exit_regime="RISK_OFF"
        )

        assert success is True
        trade = performance_tracker.trade_history[0]
        assert trade.exit_price == 1.2650
        assert trade.exit_regime == "RISK_OFF"
        assert trade.result == TradeResult.WIN  # SELL at 1.27, exit at 1.265 = profit

    def test_update_trade_exit_failure(self, performance_tracker):
        """Test updating trade exit with invalid trade ID."""
        success = performance_tracker.update_trade_exit(
            trade_id="NONEXISTENT_ID",
            exit_price=100.0,
            exit_regime="BULL"
        )

        assert success is False
        assert len(performance_tracker.trade_history) == 0

    def test_update_trade_exit_already_exited(self, performance_tracker):
        """Test updating a trade that already has an exit price."""
        # Record and exit a trade
        trade_id = performance_tracker.record_trade_entry(
            asset="AAPL",
            direction="BUY",
            entry_price=150.0,
            regime="BULL"
        )

        performance_tracker.update_trade_exit(trade_id, 155.0, "BULL")

        # Try to update again
        success = performance_tracker.update_trade_exit(trade_id, 160.0, "BULL")

        assert success is False  # Should fail because already exited


class TestPerformanceMetrics:
    """Tests for performance metrics calculation."""

    def test_get_performance_metrics_empty(self, performance_tracker):
        """Test getting metrics with no trades."""
        metrics = performance_tracker.get_performance_metrics()

        assert metrics["total_trades"] == 0
        assert metrics["win_rate"] == 0
        assert metrics["total_pnl"] == 0
        assert metrics["profit_factor"] == 0

    def test_get_performance_metrics_basic(self, tracker_with_history):
        """Test basic metrics calculation."""
        metrics = tracker_with_history.get_performance_metrics()

        # Should have aggregated metrics
        assert metrics["total_trades"] > 0
        assert 0 <= metrics["win_rate"] <= 1
        assert "profit_factor" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "kelly_fraction" in metrics

    def test_get_performance_metrics_filter_asset(self, tracker_with_history):
        """Test metrics filtered by asset."""
        aapl_metrics = tracker_with_history.get_performance_metrics(asset="AAPL")
        tsla_metrics = tracker_with_history.get_performance_metrics(asset="TSLA")

        # AAPL should have all wins
        assert aapl_metrics["total_trades"] == 5
        assert aapl_metrics["win_rate"] == 1.0

        # TSLA should have all losses
        assert tsla_metrics["total_trades"] == 3
        assert tsla_metrics["win_rate"] == 0.0

    def test_get_performance_metrics_filter_regime(self, tracker_with_history):
        """Test metrics filtered by regime."""
        bull_metrics = tracker_with_history.get_performance_metrics(regime="BULL")
        bear_metrics = tracker_with_history.get_performance_metrics(regime="BEAR")

        # Should have different performance in different regimes
        assert bull_metrics["total_trades"] > 0
        assert bear_metrics["total_trades"] > 0

    def test_get_performance_metrics_filter_direction(self, tracker_with_history):
        """Test metrics filtered by direction."""
        buy_metrics = tracker_with_history.get_performance_metrics(direction="BUY")
        sell_metrics = tracker_with_history.get_performance_metrics(direction="SELL")

        assert buy_metrics["total_trades"] > 0
        assert sell_metrics["total_trades"] > 0

    def test_get_performance_metrics_lookback(self, tracker_with_history):
        """Test metrics with lookback period."""
        # Get metrics for last 5 days
        recent_metrics = tracker_with_history.get_performance_metrics(lookback_days=5)

        # Get all metrics
        all_metrics = tracker_with_history.get_performance_metrics(lookback_days=None)

        # Recent trades should be subset of all trades
        assert recent_metrics["total_trades"] <= all_metrics["total_trades"]

    def test_get_regime_performance(self, tracker_with_history):
        """Test getting performance by regime."""
        regime_perf = tracker_with_history.get_regime_performance()

        assert isinstance(regime_perf, dict)
        assert "BULL" in regime_perf
        assert "BEAR" in regime_perf
        assert "VOLATILE" in regime_perf

        # Check each regime has metrics
        for regime, metrics in regime_perf.items():
            assert "total_trades" in metrics
            assert "win_rate" in metrics

    def test_get_asset_performance(self, tracker_with_history):
        """Test getting performance by asset."""
        asset_perf = tracker_with_history.get_asset_performance()

        assert isinstance(asset_perf, dict)
        assert "AAPL" in asset_perf
        assert "TSLA" in asset_perf
        assert "USDJPY" in asset_perf

        # Check each asset has metrics
        for asset, metrics in asset_perf.items():
            assert "total_trades" in metrics
            assert "total_pnl" in metrics

    def test_get_direction_performance(self, tracker_with_history):
        """Test getting BUY vs SELL performance."""
        # Get metrics for BUY trades
        buy_trades = [t for t in tracker_with_history.trade_history if t.direction == "BUY"]
        buy_metrics = tracker_with_history.get_performance_metrics(direction="BUY")

        # Get metrics for SELL trades
        sell_trades = [t for t in tracker_with_history.trade_history if t.direction == "SELL"]
        sell_metrics = tracker_with_history.get_performance_metrics(direction="SELL")

        assert buy_metrics["total_trades"] == len([t for t in buy_trades if t.exit_price is not None])
        assert sell_metrics["total_trades"] == len([t for t in sell_trades if t.exit_price is not None])


class TestAdaptiveMultiplier:
    """Tests for adaptive multiplier calculation."""

    def test_get_adaptive_multiplier_good_performance(self):
        """Test adaptive multiplier with good historical performance."""
        tracker = PerformanceTracker()

        # Add winning trades
        for i in range(10):
            trade = TradeRecord(
                trade_id=f"WIN_{i}",
                timestamp=datetime.now() - timedelta(days=20 - i),
                asset="AAPL",
                direction="BUY",
                entry_price=150.0,
                exit_price=160.0,
                regime="BULL",
                tags=["good_perf"]
            )
            tracker.update_performance(trade, calculate_metrics=False)

        multiplier = tracker.get_adaptive_multiplier(
            asset="AAPL",
            regime="BULL",
            direction="BUY",
            base_multiplier=1.0
        )

        # Good performance should increase multiplier
        assert multiplier > 1.0
        assert multiplier <= 2.0  # Max clamp

    def test_get_adaptive_multiplier_poor_performance(self):
        """Test adaptive multiplier with poor historical performance."""
        tracker = PerformanceTracker()

        # Add losing trades
        for i in range(10):
            trade = TradeRecord(
                trade_id=f"LOSS_{i}",
                timestamp=datetime.now() - timedelta(days=20 - i),
                asset="TSLA",
                direction="BUY",
                entry_price=200.0,
                exit_price=190.0,
                regime="VOLATILE",
                tags=["poor_perf"]
            )
            tracker.update_performance(trade, calculate_metrics=False)

        multiplier = tracker.get_adaptive_multiplier(
            asset="TSLA",
            regime="VOLATILE",
            direction="BUY",
            base_multiplier=1.0
        )

        # Poor performance should decrease multiplier
        assert multiplier < 1.0
        assert multiplier >= 0.1  # Min clamp

    def test_get_adaptive_multiplier_no_history(self, performance_tracker):
        """Test adaptive multiplier with no historical data."""
        multiplier = performance_tracker.get_adaptive_multiplier(
            asset="NEW_ASSET",
            regime="BULL",
            direction="BUY",
            base_multiplier=1.0
        )

        # With no history, should return base multiplier with regime adjustment
        assert 0.1 <= multiplier <= 2.0

    def test_get_adaptive_multiplier_regime_adjustments(self):
        """Test adaptive multiplier with regime-specific adjustments."""
        tracker = PerformanceTracker()

        # Test different regimes - should all be in valid range
        test_cases = [
            ("VOLATILE", "BUY"),
            ("CRISIS", "BUY"),
            ("BULL", "BUY"),
            ("RISK_ON", "BUY"),
            ("BULL", "SELL"),
        ]

        for regime, direction in test_cases:
            multiplier = tracker.get_adaptive_multiplier(
                asset="TEST",
                regime=regime,
                direction=direction,
                base_multiplier=1.0,
                lookback_days=7
            )

            # Check it's in reasonable range
            assert 0.1 <= multiplier <= 2.0

    def test_get_adaptive_multiplier_jpy_scenario(self):
        """Test adaptive multiplier for JPY SELL scenario from Fix 17."""
        tracker = PerformanceTracker()

        # Simulate JPY SELL performing poorly in BULL regime
        for i in range(8):
            trade = TradeRecord(
                trade_id=f"JPY_BULL_LOSS_{i}",
                timestamp=datetime.now() - timedelta(days=30 - i),
                asset="USDJPY",
                direction="SELL",
                entry_price=150.0,
                exit_price=152.0,  # Loss
                regime="BULL",
                tags=["jpy", "sell", "bull"]
            )
            tracker.update_performance(trade, calculate_metrics=False)

        # JPY SELL performing well in BEAR regime
        for i in range(8):
            trade = TradeRecord(
                trade_id=f"JPY_BEAR_WIN_{i}",
                timestamp=datetime.now() - timedelta(days=20 - i),
                asset="USDJPY",
                direction="SELL",
                entry_price=150.0,
                exit_price=148.0,  # Win
                regime="BEAR",
                tags=["jpy", "sell", "bear"]
            )
            tracker.update_performance(trade, calculate_metrics=False)

        # Get multipliers
        bull_multiplier = tracker.get_adaptive_multiplier(
            asset="USDJPY",
            regime="BULL",
            direction="SELL",
            base_multiplier=1.0,
            lookback_days=30
        )

        bear_multiplier = tracker.get_adaptive_multiplier(
            asset="USDJPY",
            regime="BEAR",
            direction="SELL",
            base_multiplier=1.0,
            lookback_days=30
        )

        # JPY SELL should have lower multiplier in BULL (poor performance)
        # and higher multiplier in BEAR (good performance)
        assert bull_multiplier < bear_multiplier

    def test_get_adaptive_multiplier_crude_oil_scenario(self):
        """Test adaptive multiplier for Crude Oil scenario from Fix 18."""
        tracker = PerformanceTracker()

        # Crude BUY performs well in INFLATION
        for i in range(6):
            trade = TradeRecord(
                trade_id=f"CL_INFLATION_WIN_{i}",
                timestamp=datetime.now() - timedelta(days=25 - i),
                asset="CL=F",
                direction="BUY",
                entry_price=75.0,
                exit_price=80.0,  # Win
                regime="INFLATION",
                tags=["crude", "buy", "inflation"]
            )
            tracker.update_performance(trade, calculate_metrics=False)

        # Crude BUY performs poorly in DEFLATION
        for i in range(6):
            trade = TradeRecord(
                trade_id=f"CL_DEFLATION_LOSS_{i}",
                timestamp=datetime.now() - timedelta(days=15 - i),
                asset="CL=F",
                direction="BUY",
                entry_price=75.0,
                exit_price=72.0,  # Loss
                regime="DEFLATION",
                tags=["crude", "buy", "deflation"]
            )
            tracker.update_performance(trade, calculate_metrics=False)

        # Get multipliers
        inflation_multiplier = tracker.get_adaptive_multiplier(
            asset="CL=F",
            regime="INFLATION",
            direction="BUY",
            base_multiplier=1.0
        )

        deflation_multiplier = tracker.get_adaptive_multiplier(
            asset="CL=F",
            regime="DEFLATION",
            direction="BUY",
            base_multiplier=1.0
        )

        # Crude BUY should have higher multiplier in INFLATION
        # and lower multiplier in DEFLATION
        assert inflation_multiplier > deflation_multiplier


class TestDataPersistence:
    """Tests for data persistence (save/load)."""

    def test_save_performance_data(self, performance_tracker, temp_storage_path):
        """Test saving performance data to disk."""
        # Add some trades
        for i in range(3):
            trade = TradeRecord(
                trade_id=f"SAVE_TEST_{i}",
                timestamp=datetime.now() - timedelta(days=i),
                asset=f"ASSET_{i}",
                direction="BUY",
                entry_price=100 + i * 10,
                exit_price=105 + i * 10,
                regime="BULL"
            )
            performance_tracker.update_performance(trade, calculate_metrics=False)

        # Set storage path and save
        performance_tracker.storage_path = temp_storage_path
        success = performance_tracker.save_performance_data()

        assert success is True
        assert temp_storage_path.with_suffix('.json').exists()
        assert temp_storage_path.with_suffix('.pkl').exists()

    def test_load_performance_data(self, performance_tracker, temp_storage_path):
        """Test loading performance data from disk."""
        # First save some data
        for i in range(2):
            trade = TradeRecord(
                trade_id=f"LOAD_TEST_{i}",
                timestamp=datetime.now() - timedelta(days=i),
                asset="TEST",
                direction="BUY",
                entry_price=100,
                exit_price=110,
                regime="BULL"
            )
            performance_tracker.update_performance(trade, calculate_metrics=False)

        performance_tracker.storage_path = temp_storage_path
        performance_tracker.save_performance_data()

        # Create new tracker and load
        new_tracker = PerformanceTracker(storage_path=str(temp_storage_path))

        # Should load the data
        assert len(new_tracker.trade_history) == 2
        assert new_tracker.trade_history[0].trade_id == "LOAD_TEST_0"
        assert new_tracker.trade_history[1].trade_id == "LOAD_TEST_1"

    def test_save_without_storage_path(self, performance_tracker):
        """Test saving without storage path."""
        success = performance_tracker.save_performance_data()
        assert success is False  # Should fail without path

    def test_load_without_storage_path(self):
        """Test loading without storage path."""
        tracker = PerformanceTracker()
        success = tracker.load_performance_data()
        assert success is False  # Should fail without path

    def test_load_nonexistent_file(self, temp_storage_path):
        """Test loading from non-existent file."""
        tracker = PerformanceTracker(storage_path=str(temp_storage_path))
        success = tracker.load_performance_data()
        assert success is False  # Should fail if file doesn't exist


class TestReporting:
    """Tests for report generation."""

    def test_generate_report_basic(self, tracker_with_history):
        """Test basic report generation."""
        report = tracker_with_history.generate_report()

        assert "PERFORMANCE REPORT" in report
        assert "Total Trades:" in report
        assert "Win Rate:" in report

    def test_generate_report_with_output_file(self, tracker_with_history, temp_storage_path):
        """Test report generation with output file."""
        output_file = temp_storage_path.with_suffix('.txt')
        report = tracker_with_history.generate_report(output_file=str(output_file))

        assert output_file.exists()
        with open(output_file, 'r') as f:
            file_content = f.read()

        assert "PERFORMANCE REPORT" in file_content

    def test_generate_report_empty(self, performance_tracker):
        """Test report generation with no trades."""
        report = performance_tracker.generate_report()

        assert "PERFORMANCE REPORT" in report
        assert "Total Trades: 0" in report

    def test_export_to_dataframe(self, tracker_with_history):
        """Test exporting to pandas DataFrame."""
        df = tracker_with_history.export_to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(tracker_with_history.trade_history)

        # Check columns
        expected_columns = ['trade_id', 'asset', 'direction', 'entry_price', 'exit_price', 'pnl', 'regime']
        for col in expected_columns:
            assert col in df.columns


class TestCleanupAndMaintenance:
    """Tests for cleanup and maintenance functions."""

    def test_clean_old_records(self):
        """Test cleaning old trade records."""
        tracker = PerformanceTracker(max_history_days=7)  # Keep only 7 days

        # Add old trade (30 days ago)
        old_trade = TradeRecord(
            trade_id="OLD_TRADE",
            timestamp=datetime.now() - timedelta(days=30),
            asset="OLD",
            direction="BUY",
            entry_price=100,
            exit_price=110,
            regime="BULL"
        )
        tracker.update_performance(old_trade, calculate_metrics=False)

        # Add recent trade
        recent_trade = TradeRecord(
            trade_id="RECENT_TRADE",
            timestamp=datetime.now() - timedelta(days=3),
            asset="RECENT",
            direction="BUY",
            entry_price=100,
            exit_price=110,
            regime="BULL"
        )
        tracker.update_performance(recent_trade, calculate_metrics=False)

        # Add pending trade (no exit)
        pending_trade = TradeRecord(
            trade_id="PENDING_TRADE",
            timestamp=datetime.now() - timedelta(days=30),  # Old but pending
            asset="PENDING",
            direction="BUY",
            entry_price=100,
            exit_price=None,  # Pending
            regime="BULL"
        )
        tracker.update_performance(pending_trade, calculate_metrics=False)

        # Manually trigger cleanup
        tracker._clean_old_records()

        # Old completed trade should be removed
        # Recent completed trade should remain
        # Pending trade should remain even if old
        trade_ids = {t.trade_id for t in tracker.trade_history}

        assert "OLD_TRADE" not in trade_ids
        assert "RECENT_TRADE" in trade_ids
        assert "PENDING_TRADE" in trade_ids  # Should keep pending trades

    def test_update_performance_history(self, performance_tracker):
        """Test updating aggregated performance history."""
        # Add trades on different days - use recent dates within max_history_days
        base_date = datetime.now()
        dates = [
            base_date - timedelta(days=3),
            base_date - timedelta(days=3),  # Same day
            base_date - timedelta(days=2),
            base_date - timedelta(days=1),
        ]

        for i, date in enumerate(dates):
            trade = TradeRecord(
                trade_id=f"DAY_TEST_{i}",
                timestamp=date,
                asset="TEST",
                direction="BUY",
                entry_price=100,
                exit_price=105,
                regime="BULL"
            )
            performance_tracker.update_performance(trade, calculate_metrics=False)

        # Update performance history
        performance_tracker._update_performance_history()

        # Should have aggregated by day
        assert len(performance_tracker.performance_history) == 3  # 3 unique days

        # Check aggregation - find the day with 2 trades
        day_with_two_trades = [h for h in performance_tracker.performance_history if h['trades'] == 2]
        assert len(day_with_two_trades) == 1


# ======================================================================
# INTEGRATION TESTS (Conditional on availability)
# ======================================================================

@pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration components not available")
class TestIntegrationComponents:
    """Tests for integration components."""

    def test_enhanced_position_adjuster_creation(self):
        """Test EnhancedPositionAdjuster creation."""
        tracker = PerformanceTracker()
        adjuster = EnhancedPositionAdjuster(performance_tracker=tracker)

        assert adjuster.performance_tracker == tracker
        # EnhancedPositionAdjuster uses adjust_position_with_performance method
        assert hasattr(adjuster, 'adjust_position_with_performance')


# ======================================================================
# EDGE CASE AND ERROR HANDLING TESTS
# ======================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_trade_with_missing_fields(self):
        """Test creating trade with missing optional fields."""
        # Should work with minimal fields
        trade = TradeRecord(
            trade_id="MINIMAL",
            timestamp=datetime.now(),
            asset="TEST",
            direction="BUY",
            entry_price=100.0,
            exit_price=105.0,
            regime="BULL"
        )

        assert trade.trade_id == "MINIMAL"
        assert trade.tags == []  # Default empty list

    def test_metrics_with_single_trade(self, performance_tracker):
        """Test metrics calculation with only one trade."""
        trade = TradeRecord(
            trade_id="SINGLE",
            timestamp=datetime.now(),
            asset="TEST",
            direction="BUY",
            entry_price=100.0,
            exit_price=110.0,
            regime="BULL"
        )

        performance_tracker.update_performance(trade)
        metrics = performance_tracker.get_performance_metrics()

        assert metrics["total_trades"] == 1
        assert metrics["win_rate"] == 1.0
        assert metrics["profit_factor"] == float('inf')  # No losses

    def test_metrics_with_all_losses(self, performance_tracker):
        """Test metrics calculation with all losing trades."""
        for i in range(5):
            trade = TradeRecord(
                trade_id=f"LOSS_{i}",
                timestamp=datetime.now() - timedelta(days=i),
                asset="TEST",
                direction="BUY",
                entry_price=100.0,
                exit_price=95.0,
                regime="BEAR"
            )
            performance_tracker.update_performance(trade, calculate_metrics=False)

        metrics = performance_tracker.get_performance_metrics()

        assert metrics["total_trades"] == 5
        assert metrics["win_rate"] == 0.0
        assert metrics["profit_factor"] == 0.0  # No wins

    def test_metrics_with_mixed_results(self, performance_tracker):
        """Test metrics calculation with mixed wins and losses."""
        trades = [
            (100, 110),  # Win
            (100, 90),   # Loss (for SELL)
            (100, 100),  # Breakeven
            (100, 105),  # Win
            (100, 98),   # Loss
        ]

        for i, (entry, exit) in enumerate(trades):
            direction = "SELL" if i == 1 else "BUY"  # Second trade is SELL
            trade = TradeRecord(
                trade_id=f"MIXED_{i}",
                timestamp=datetime.now() - timedelta(days=i),
                asset="TEST",
                direction=direction,
                entry_price=entry,
                exit_price=exit,
                regime="MIXED"
            )
            performance_tracker.update_performance(trade, calculate_metrics=False)

        metrics = performance_tracker.get_performance_metrics()

        assert metrics["total_trades"] == 5
        assert 0 < metrics["win_rate"] < 1
        assert metrics["profit_factor"] > 0

    def test_division_by_zero_handling(self, performance_tracker):
        """Test handling of division by zero in metrics."""
        # Add trades that could cause division by zero
        trade = TradeRecord(
            trade_id="ZERO_TEST",
            timestamp=datetime.now(),
            asset="TEST",
            direction="BUY",
            entry_price=100.0,
            exit_price=100.0,  # Breakeven
            regime="SIDEWAYS"
        )

        performance_tracker.update_performance(trade)
        metrics = performance_tracker.get_performance_metrics()

        # Should handle breakeven trades without errors
        assert metrics["total_trades"] == 1
        assert metrics["win_rate"] == 0.0  # Breakeven counts as not win

    def test_negative_prices(self):
        """Test handling of negative prices (shouldn't happen in reality)."""
        trade = TradeRecord(
            trade_id="NEGATIVE",
            timestamp=datetime.now(),
            asset="TEST",
            direction="BUY",
            entry_price=-10.0,  # Negative!
            exit_price=-5.0,   # Negative!
            regime="CRISIS"
        )

        # Should still calculate P&L
        assert trade.pnl == 5.0  # -5 - (-10) = 5
        # Implementation sets pnl_percent to 0.0 when entry_price <= 0 for safety
        assert trade.pnl_percent == 0.0  # Division protection for negative entry price


# ======================================================================
# COMPREHENSIVE SCENARIO TESTS
# ======================================================================

class TestComprehensiveScenarios:
    """Comprehensive scenario tests."""

    def test_full_trading_scenario(self):
        """Test a full trading scenario from entry to exit to analysis."""
        tracker = PerformanceTracker()

        # Phase 1: Record several trades
        assets = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        regimes = ["BULL", "BEAR", "RISK_ON", "RISK_OFF", "VOLATILE"]

        trade_ids = []
        for i, (asset, regime) in enumerate(zip(assets, regimes)):
            trade_id = tracker.record_trade_entry(
                asset=asset,
                direction="BUY",
                entry_price=100 + i * 20,
                regime=regime,
                position_size=1.0 + i * 0.2,
                tags=["scenario", "phase1"]
            )
            trade_ids.append(trade_id)

        # Phase 2: Update exits with different outcomes
        outcomes = [0.05, -0.03, 0.08, -0.02, 0.10]  # P&L percentages

        for trade_id, outcome in zip(trade_ids, outcomes):
            # Get the trade to calculate exit price
            trade = next(t for t in tracker.trade_history if t.trade_id == trade_id)
            exit_price = trade.entry_price * (1 + outcome)

            tracker.update_trade_exit(
                trade_id=trade_id,
                exit_price=exit_price,
                exit_regime=trade.regime  # Same regime for simplicity
            )

        # Phase 3: Analyze performance
        metrics = tracker.get_performance_metrics()
        regime_perf = tracker.get_regime_performance()
        asset_perf = tracker.get_asset_performance()

        # Phase 4: Test adaptive multipliers for new signals
        new_assets = ["NVDA", "META", "NFLX"]
        new_regimes = ["BULL", "RISK_ON", "VOLATILE"]

        multipliers = []
        for asset, regime in zip(new_assets, new_regimes):
            multiplier = tracker.get_adaptive_multiplier(
                asset=asset if asset in assets else "AAPL",  # Use AAPL as proxy for new assets
                regime=regime,
                direction="BUY",
                base_multiplier=1.0,
                lookback_days=30
            )
            multipliers.append(multiplier)

        # Phase 5: Generate report
        report = tracker.generate_report()

        # Assertions
        assert len(tracker.trade_history) == 5
        assert all(t.exit_price is not None for t in tracker.trade_history)
        assert metrics["total_trades"] == 5
        assert len(regime_perf) == 5  # One for each regime
        assert len(asset_perf) == 5  # One for each asset
        assert len(multipliers) == 3
        assert "PERFORMANCE REPORT" in report

    def test_china_stock_scenario_fix16(self):
        """Test scenario with China stocks (Fix 16)."""
        tracker = PerformanceTracker()

        # China A-shares
        china_tickers = ["600000.SS", "000001.SZ", "688001.SS", "300001.SZ"]

        for ticker in china_tickers:
            trade = TradeRecord(
                trade_id=f"CHINA_{ticker}",
                timestamp=datetime.now(),
                asset=ticker,
                direction="BUY",
                entry_price=10.0,
                exit_price=10.5,  # Small gain
                regime="BULL",
                tags=["china", "a-share", "t+1"]
            )
            tracker.update_performance(trade, calculate_metrics=False)

        # Get metrics
        metrics = tracker.get_performance_metrics()

        assert metrics["total_trades"] == 4
        assert metrics["win_rate"] == 1.0  # All small gains

    def test_jpy_adaptive_scenario_fix17(self):
        """Test JPY adaptive blocking scenario (Fix 17)."""
        tracker = PerformanceTracker()

        # JPY pairs
        jpy_pairs = ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY"]

        # Simulate historical performance
        # JPY SELL performs poorly in BULL/RISK_ON
        for pair in jpy_pairs:
            for j in range(4):  # Multiple trades per pair
                trade = TradeRecord(
                    trade_id=f"JPY_BULL_{pair}_{j}",
                    timestamp=datetime.now() - timedelta(days=30 - j),
                    asset=pair,
                    direction="SELL",
                    entry_price=150.0 if "USD" in pair else 160.0,
                    exit_price=152.0 if "USD" in pair else 162.0,  # Loss
                    regime="BULL",
                    tags=["jpy", "sell", "bull", "adaptive"]
                )
                tracker.update_performance(trade, calculate_metrics=False)

        # Get adaptive multiplier for JPY SELL in BULL
        multiplier = tracker.get_adaptive_multiplier(
            asset="USDJPY",
            regime="BULL",
            direction="SELL",
            base_multiplier=1.0
        )

        # Should be reduced due to poor performance
        assert multiplier < 1.0

    def test_crude_oil_scenario_fix18(self):
        """Test Crude Oil adaptive scenario (Fix 18)."""
        tracker = PerformanceTracker()

        # Crude Oil tickers
        crude_tickers = ["CL=F", "BZ=F"]

        # Simulate regime-based performance
        regimes_favorable_buy = ["INFLATION", "BULL", "RISK_ON"]
        regimes_favorable_sell = ["DEFLATION", "BEAR"]
        regimes_risky = ["VOLATILE", "CRISIS", "SIDEWAYS"]

        # Add trades for each regime
        for ticker in crude_tickers:
            # Favorable BUY regimes
            for regime in regimes_favorable_buy:
                trade = TradeRecord(
                    trade_id=f"CRUDE_{ticker}_{regime}_BUY",
                    timestamp=datetime.now(),
                    asset=ticker,
                    direction="BUY",
                    entry_price=75.0,
                    exit_price=80.0,  # Win
                    regime=regime,
                    tags=["crude", "buy", regime.lower()]
                )
                tracker.update_performance(trade, calculate_metrics=False)

            # Favorable SELL regimes
            for regime in regimes_favorable_sell:
                trade = TradeRecord(
                    trade_id=f"CRUDE_{ticker}_{regime}_SELL",
                    timestamp=datetime.now(),
                    asset=ticker,
                    direction="SELL",
                    entry_price=75.0,
                    exit_price=70.0,  # Win
                    regime=regime,
                    tags=["crude", "sell", regime.lower()]
                )
                tracker.update_performance(trade, calculate_metrics=False)

            # Risky regimes (both directions reduced)
            for regime in regimes_risky:
                trade = TradeRecord(
                    trade_id=f"CRUDE_{ticker}_{regime}_MIXED",
                    timestamp=datetime.now(),
                    asset=ticker,
                    direction="BUY",
                    entry_price=75.0,
                    exit_price=74.0,  # Small loss
                    regime=regime,
                    tags=["crude", "risky", regime.lower()]
                )
                tracker.update_performance(trade, calculate_metrics=False)

        # Test adaptive multipliers
        inflation_buy_mult = tracker.get_adaptive_multiplier(
            asset="CL=F",
            regime="INFLATION",
            direction="BUY",
            base_multiplier=1.0
        )

        volatile_buy_mult = tracker.get_adaptive_multiplier(
            asset="CL=F",
            regime="VOLATILE",
            direction="BUY",
            base_multiplier=1.0
        )

        # BUY in INFLATION should have higher multiplier than in VOLATILE
        assert inflation_buy_mult > volatile_buy_mult


# ======================================================================
# PERFORMANCE AND SCALABILITY TESTS
# ======================================================================

class TestPerformanceScalability:
    """Tests for performance and scalability."""

    def test_large_number_of_trades(self):
        """Test with a large number of trades."""
        # Use large max_history_days to avoid cleanup removing trades
        tracker = PerformanceTracker(max_history_days=2000)

        # Add 1000 trades within the history window
        n_trades = 1000
        for i in range(n_trades):
            # Spread trades over last 365 days to keep within reasonable timeframe
            trade = TradeRecord(
                trade_id=f"BULK_{i:04d}",
                timestamp=datetime.now() - timedelta(days=i % 365),
                asset=f"ASSET_{i % 50}",  # 50 different assets
                direction="BUY" if i % 3 != 0 else "SELL",
                entry_price=100.0 + (i % 20) * 5,
                exit_price=105.0 + (i % 20) * 5 if i % 4 != 0 else 95.0 + (i % 20) * 5,
                regime=["BULL", "BEAR", "RISK_ON", "RISK_OFF"][i % 4],
                tags=["bulk", f"asset_{i % 50}"]
            )
            tracker.update_performance(trade, calculate_metrics=False)

        # Update metrics once
        tracker._update_performance_history()

        # Get metrics - should handle efficiently
        import time
        start_time = time.time()
        metrics = tracker.get_performance_metrics()
        end_time = time.time()

        assert len(tracker.trade_history) == n_trades
        assert metrics["total_trades"] == n_trades

        # Should process within reasonable time
        processing_time = end_time - start_time
        assert processing_time < 1.0, f"Metrics calculation took {processing_time:.2f} seconds"

    def test_memory_cleanup(self):
        """Test memory cleanup with old records."""
        tracker = PerformanceTracker(max_history_days=1)  # Keep only 1 day

        # Add old trades (2 days ago)
        for i in range(100):
            trade = TradeRecord(
                trade_id=f"OLD_{i}",
                timestamp=datetime.now() - timedelta(days=2),
                asset="OLD",
                direction="BUY",
                entry_price=100,
                exit_price=105,
                regime="BULL"
            )
            tracker.update_performance(trade, calculate_metrics=False)

        # Add recent trades
        for i in range(50):
            trade = TradeRecord(
                trade_id=f"RECENT_{i}",
                timestamp=datetime.now(),
                asset="RECENT",
                direction="BUY",
                entry_price=100,
                exit_price=105,
                regime="BULL"
            )
            tracker.update_performance(trade, calculate_metrics=False)

        # Trigger cleanup
        tracker._clean_old_records()

        # Should only keep recent trades
        assert len(tracker.trade_history) == 50


# ======================================================================
# MAIN TEST RUNNER
# ======================================================================

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
