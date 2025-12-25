"""
Dynamic Market Trend Adaptation System
======================================
Adjusts BUY/SELL strategy based on current market trend.

Based on findings from Nov-Dec 2025:
- BUY signals: 17% win rate in downtrend
- SELL signals: 100% win rate in downtrend

This module detects market regime and adapts signals accordingly.

Author: Claude Code
Date: 2025-12-16
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


class MarketTrend(Enum):
    """Market trend classification."""
    STRONG_UPTREND = "strong_uptrend"      # SPX up >5% in 20 days
    MODERATE_UPTREND = "moderate_uptrend"  # SPX up 2-5% in 20 days
    SIDEWAYS_BULLISH = "sideways_bullish"  # SPX -2% to +2%, bullish bias
    SIDEWAYS_BEARISH = "sideways_bearish"  # SPX -2% to +2%, bearish bias
    MODERATE_DOWNTREND = "moderate_downtrend"  # SPX down 2-5% in 20 days
    STRONG_DOWNTREND = "strong_downtrend"  # SPX down >5% in 20 days
    BEAR_MARKET = "bear_market"            # SPX down >20% from highs
    CRASH = "crash"                        # SPX down >10% rapidly


@dataclass
class TrendAdjustedSignal:
    """Result of trend-adjusted signal processing."""
    ticker: str
    signal_type: str
    original_confidence: float
    adjusted_confidence: float
    position_multiplier: float
    stop_loss_pct: float
    should_trade: bool
    block_reason: Optional[str]
    market_trend: str
    trend_description: str
    trend_win_rate: float
    trend_multiplier: float
    confidence_boost: float
    position_adjustment: float
    stop_loss_adjustment: float
    risk_rules: Dict


class DynamicTrendAdapter:
    """
    Dynamically adjusts BUY/SELL strategy based on market trend.

    Based on actual results: SELL excels in downtrends, BUY fails.
    """

    def __init__(self):
        self.current_trend = None
        self.trend_history = []
        self.spx_data = []
        self.last_trend_check = None
        self.trend_cache_minutes = 15  # Cache trend for 15 minutes

        # Trend-based strategy adjustments
        self.TREND_STRATEGIES = {
            MarketTrend.STRONG_UPTREND: {
                'description': 'Strong uptrend - Favor BUY',
                'buy_win_rate': 0.70,
                'sell_win_rate': 0.30,
                'buy_multiplier': 1.4,
                'sell_multiplier': 0.6,
                'position_adjustment': {'buy': 1.3, 'sell': 0.7},
                'confidence_boost': {'buy': 0.10, 'sell': -0.10},
                'stop_loss_adjustment': {'buy': 1.2, 'sell': 0.8},
            },
            MarketTrend.MODERATE_UPTREND: {
                'description': 'Moderate uptrend - Balanced',
                'buy_win_rate': 0.60,
                'sell_win_rate': 0.40,
                'buy_multiplier': 1.2,
                'sell_multiplier': 0.8,
                'position_adjustment': {'buy': 1.1, 'sell': 0.9},
                'confidence_boost': {'buy': 0.05, 'sell': -0.05},
                'stop_loss_adjustment': {'buy': 1.1, 'sell': 0.9},
            },
            MarketTrend.SIDEWAYS_BULLISH: {
                'description': 'Sideways with bullish bias',
                'buy_win_rate': 0.55,
                'sell_win_rate': 0.45,
                'buy_multiplier': 1.1,
                'sell_multiplier': 0.9,
                'position_adjustment': {'buy': 1.0, 'sell': 1.0},
                'confidence_boost': {'buy': 0.03, 'sell': -0.03},
                'stop_loss_adjustment': {'buy': 1.0, 'sell': 1.0},
            },
            MarketTrend.SIDEWAYS_BEARISH: {
                'description': 'Sideways with bearish bias',
                'buy_win_rate': 0.45,
                'sell_win_rate': 0.55,
                'buy_multiplier': 0.9,
                'sell_multiplier': 1.1,
                'position_adjustment': {'buy': 0.9, 'sell': 1.1},
                'confidence_boost': {'buy': -0.03, 'sell': 0.03},
                'stop_loss_adjustment': {'buy': 0.9, 'sell': 1.1},
            },
            MarketTrend.MODERATE_DOWNTREND: {
                'description': 'Moderate downtrend - Favor SELL',
                'buy_win_rate': 0.40,
                'sell_win_rate': 0.60,
                'buy_multiplier': 0.8,
                'sell_multiplier': 1.2,
                'position_adjustment': {'buy': 0.8, 'sell': 1.2},
                'confidence_boost': {'buy': -0.05, 'sell': 0.05},
                'stop_loss_adjustment': {'buy': 0.9, 'sell': 1.1},
            },
            MarketTrend.STRONG_DOWNTREND: {
                'description': 'Strong downtrend - Heavily favor SELL',
                'buy_win_rate': 0.25,
                'sell_win_rate': 0.75,
                'buy_multiplier': 0.6,
                'sell_multiplier': 1.4,
                'position_adjustment': {'buy': 0.6, 'sell': 1.4},
                'confidence_boost': {'buy': -0.15, 'sell': 0.15},
                'stop_loss_adjustment': {'buy': 0.8, 'sell': 1.2},
                'special_rules': {
                    'block_buy_on_bounces': True,
                    'aggressive_shorting': True,
                    'reduce_buy_size': 0.5,
                }
            },
            MarketTrend.BEAR_MARKET: {
                'description': 'Bear market - Mostly SELL',
                'buy_win_rate': 0.20,
                'sell_win_rate': 0.80,
                'buy_multiplier': 0.4,
                'sell_multiplier': 1.6,
                'position_adjustment': {'buy': 0.4, 'sell': 1.6},
                'confidence_boost': {'buy': -0.20, 'sell': 0.20},
                'stop_loss_adjustment': {'buy': 0.7, 'sell': 1.3},
            },
            MarketTrend.CRASH: {
                'description': 'Market crash - Extreme SELL',
                'buy_win_rate': 0.10,
                'sell_win_rate': 0.90,
                'buy_multiplier': 0.2,
                'sell_multiplier': 1.8,
                'position_adjustment': {'buy': 0.2, 'sell': 1.8},
                'confidence_boost': {'buy': -0.30, 'sell': 0.30},
                'stop_loss_adjustment': {'buy': 0.6, 'sell': 1.4},
            },
        }

        # Statistics tracking
        self.stats = {
            'signals_processed': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'blocked_signals': 0,
            'trend_detections': 0,
        }

    def detect_market_trend(self, spx_prices: List[float] = None) -> MarketTrend:
        """
        Detect current market trend from SPX data.

        Args:
            spx_prices: List of SPX closing prices (most recent last)

        Returns:
            MarketTrend classification
        """
        # Check cache
        now = datetime.now()
        if (self.current_trend is not None and
            self.last_trend_check is not None and
            (now - self.last_trend_check).total_seconds() < self.trend_cache_minutes * 60):
            return self.current_trend

        if spx_prices is None:
            spx_prices = self.fetch_spx_data()

        if len(spx_prices) < 20:
            self.current_trend = MarketTrend.SIDEWAYS_BEARISH
            self.last_trend_check = now
            return self.current_trend

        # Calculate returns
        current_price = spx_prices[-1]
        price_20d_ago = spx_prices[-20] if len(spx_prices) >= 20 else spx_prices[0]
        price_5d_ago = spx_prices[-5] if len(spx_prices) >= 5 else spx_prices[0]

        return_20d = (current_price - price_20d_ago) / price_20d_ago
        return_5d = (current_price - price_5d_ago) / price_5d_ago

        # Calculate volatility
        returns = np.diff(spx_prices) / np.array(spx_prices[:-1])
        volatility = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0.15

        # Trend classification
        if return_5d < -0.10:
            trend = MarketTrend.CRASH
        elif return_20d < -0.20:
            trend = MarketTrend.BEAR_MARKET
        elif return_20d < -0.05 and return_5d < -0.01:
            trend = MarketTrend.STRONG_DOWNTREND
        elif return_20d < -0.02:
            trend = MarketTrend.MODERATE_DOWNTREND
        elif return_20d > 0.05 and return_5d > 0.01:
            trend = MarketTrend.STRONG_UPTREND
        elif return_20d > 0.02:
            trend = MarketTrend.MODERATE_UPTREND
        elif return_20d > -0.02 and return_5d > 0:
            trend = MarketTrend.SIDEWAYS_BULLISH
        else:
            trend = MarketTrend.SIDEWAYS_BEARISH

        # Update cache
        self.current_trend = trend
        self.last_trend_check = now
        self.trend_history.append((now, trend))
        self.stats['trend_detections'] += 1

        return trend

    def fetch_spx_data(self, days: int = 30) -> List[float]:
        """Fetch SPX data for trend analysis."""
        try:
            import yfinance as yf

            spx = yf.download('^GSPC', period=f'{days}d', progress=False)
            if not spx.empty:
                # Handle multi-index columns
                if hasattr(spx.columns, 'levels'):
                    prices = spx['Close'].iloc[:, 0].tolist() if isinstance(spx['Close'], type(spx)) else spx['Close'].tolist()
                else:
                    prices = spx['Close'].tolist()
                self.spx_data = prices
                return prices
        except Exception as e:
            print(f"[TREND] Error fetching SPX data: {e}")

        # Fallback: return empty list to trigger default trend
        return []

    def adjust_signal_for_trend(
        self,
        ticker: str,
        signal_type: str,
        confidence: float,
        position_multiplier: float = 1.0,
        stop_loss_pct: float = 0.08,
        market_trend: MarketTrend = None
    ) -> TrendAdjustedSignal:
        """
        Adjust signal based on market trend.

        Args:
            ticker: Stock ticker
            signal_type: 'BUY' or 'SELL'
            confidence: Original confidence score
            position_multiplier: Original position size multiplier
            stop_loss_pct: Original stop loss percentage
            market_trend: Force specific trend (auto-detect if None)

        Returns:
            TrendAdjustedSignal with all adjustments
        """
        if market_trend is None:
            market_trend = self.detect_market_trend()

        self.stats['signals_processed'] += 1

        strategy = self.TREND_STRATEGIES.get(
            market_trend,
            self.TREND_STRATEGIES[MarketTrend.SIDEWAYS_BEARISH]
        )

        # Get trend multipliers based on signal type
        if signal_type.upper() == 'BUY':
            self.stats['buy_signals'] += 1
            multiplier = strategy['buy_multiplier']
            conf_boost = strategy['confidence_boost']['buy']
            pos_adj = strategy['position_adjustment']['buy']
            sl_adj = strategy['stop_loss_adjustment']['buy']
            win_rate = strategy['buy_win_rate']
        else:  # SELL
            self.stats['sell_signals'] += 1
            multiplier = strategy['sell_multiplier']
            conf_boost = strategy['confidence_boost']['sell']
            pos_adj = strategy['position_adjustment']['sell']
            sl_adj = strategy['stop_loss_adjustment']['sell']
            win_rate = strategy['sell_win_rate']

        # Adjust confidence
        adjusted_confidence = confidence * (1 + conf_boost)
        adjusted_confidence = max(0.05, min(0.95, adjusted_confidence))

        # Adjust position size
        adjusted_position = position_multiplier * pos_adj

        # Adjust stop-loss
        adjusted_stop = stop_loss_pct * sl_adj

        # Check for blocking conditions
        should_trade = True
        block_reason = None

        # Apply special rules for strong downtrend
        special_rules = strategy.get('special_rules', {})

        if market_trend == MarketTrend.STRONG_DOWNTREND:
            if signal_type.upper() == 'BUY':
                if special_rules.get('block_buy_on_bounces', False):
                    # Check if this is an oversold bounce
                    is_oversold_bounce = self._is_oversold_bounce(ticker)
                    if not is_oversold_bounce:
                        should_trade = False
                        block_reason = 'No buying in strong downtrend except oversold bounces'

                if special_rules.get('reduce_buy_size', 1.0) < 1.0:
                    adjusted_position *= special_rules['reduce_buy_size']

            elif signal_type.upper() == 'SELL':
                if special_rules.get('aggressive_shorting', False):
                    adjusted_position *= 1.2  # Extra 20% for shorts

        # Block low confidence signals in extreme conditions
        if market_trend in [MarketTrend.BEAR_MARKET, MarketTrend.CRASH]:
            if signal_type.upper() == 'BUY' and adjusted_confidence < 0.80:
                should_trade = False
                block_reason = f'No BUY in {market_trend.value} except high conviction'

        # Build risk rules based on trend
        risk_rules = self._get_trend_risk_rules(market_trend, signal_type)

        if not should_trade:
            self.stats['blocked_signals'] += 1

        return TrendAdjustedSignal(
            ticker=ticker,
            signal_type=signal_type.upper(),
            original_confidence=confidence,
            adjusted_confidence=adjusted_confidence,
            position_multiplier=adjusted_position,
            stop_loss_pct=adjusted_stop,
            should_trade=should_trade,
            block_reason=block_reason,
            market_trend=market_trend.value,
            trend_description=strategy['description'],
            trend_win_rate=win_rate,
            trend_multiplier=multiplier,
            confidence_boost=conf_boost,
            position_adjustment=pos_adj,
            stop_loss_adjustment=sl_adj,
            risk_rules=risk_rules,
        )

    def _is_oversold_bounce(self, ticker: str) -> bool:
        """Check if stock is in oversold bounce condition."""
        try:
            import yfinance as yf

            data = yf.download(ticker, period='30d', progress=False)
            if data.empty or len(data) < 14:
                return False

            # Handle multi-index columns
            if hasattr(data.columns, 'levels'):
                close = data['Close'].iloc[:, 0]
            else:
                close = data['Close']

            # Calculate RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)

            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            current_rsi = rsi.iloc[-1]

            # Oversold bounce: RSI was below 30 recently and now recovering
            recent_min_rsi = rsi.iloc[-5:].min()

            return recent_min_rsi < 30 and current_rsi > recent_min_rsi

        except Exception:
            return False

    def _get_trend_risk_rules(self, trend: MarketTrend, signal_type: str) -> Dict:
        """Get trend-specific risk management rules."""

        trend_rules = {
            MarketTrend.STRONG_DOWNTREND: {
                'BUY': {
                    'max_position_pct': 0.05,
                    'max_portfolio_buy_pct': 0.20,
                    'require_tight_stops': True,
                    'exit_on_rally': 0.08,
                },
                'SELL': {
                    'max_position_pct': 0.15,
                    'max_portfolio_sell_pct': 0.50,
                    'trailing_stop': 0.15,
                    'add_on_weakness': True,
                }
            },
            MarketTrend.STRONG_UPTREND: {
                'BUY': {
                    'max_position_pct': 0.15,
                    'max_portfolio_buy_pct': 0.70,
                    'trailing_stop': 0.10,
                    'add_on_strength': True,
                },
                'SELL': {
                    'max_position_pct': 0.05,
                    'max_portfolio_sell_pct': 0.15,
                    'require_tight_stops': True,
                    'exit_quickly': True,
                }
            },
        }

        default_rules = {
            'BUY': {'max_position_pct': 0.10, 'max_portfolio_buy_pct': 0.40},
            'SELL': {'max_position_pct': 0.10, 'max_portfolio_sell_pct': 0.30},
        }

        return trend_rules.get(trend, {}).get(signal_type.upper(),
                                               default_rules.get(signal_type.upper(), {}))

    def get_current_trend_info(self) -> Dict:
        """Get information about current market trend."""
        trend = self.detect_market_trend()
        strategy = self.TREND_STRATEGIES.get(trend)

        return {
            'trend': trend.value,
            'description': strategy['description'],
            'buy_win_rate': strategy['buy_win_rate'],
            'sell_win_rate': strategy['sell_win_rate'],
            'buy_multiplier': strategy['buy_multiplier'],
            'sell_multiplier': strategy['sell_multiplier'],
            'recommendation': 'Favor SELL' if strategy['sell_multiplier'] > 1.0 else 'Favor BUY',
        }

    def get_statistics(self) -> Dict:
        """Get adapter statistics."""
        stats = self.stats.copy()

        if stats['signals_processed'] > 0:
            stats['blocked_pct'] = stats['blocked_signals'] / stats['signals_processed'] * 100
            stats['buy_pct'] = stats['buy_signals'] / stats['signals_processed'] * 100
            stats['sell_pct'] = stats['sell_signals'] / stats['signals_processed'] * 100
        else:
            stats['blocked_pct'] = 0
            stats['buy_pct'] = 0
            stats['sell_pct'] = 0

        if self.current_trend:
            stats['current_trend'] = self.current_trend.value
        else:
            stats['current_trend'] = 'unknown'

        return stats


class TrendBasedSignalGenerator:
    """
    Generates BUY/SELL signals dynamically based on market trend.
    Adjusts signal generation probability based on current regime.
    """

    def __init__(self):
        self.trend_adapter = DynamicTrendAdapter()
        self.current_trend = None

        # Signal generation probabilities by trend
        self.SIGNAL_GENERATION_PROBS = {
            MarketTrend.STRONG_UPTREND: {'BUY': 0.80, 'SELL': 0.20},
            MarketTrend.MODERATE_UPTREND: {'BUY': 0.70, 'SELL': 0.30},
            MarketTrend.SIDEWAYS_BULLISH: {'BUY': 0.60, 'SELL': 0.40},
            MarketTrend.SIDEWAYS_BEARISH: {'BUY': 0.40, 'SELL': 0.60},
            MarketTrend.MODERATE_DOWNTREND: {'BUY': 0.30, 'SELL': 0.70},
            MarketTrend.STRONG_DOWNTREND: {'BUY': 0.20, 'SELL': 0.80},
            MarketTrend.BEAR_MARKET: {'BUY': 0.10, 'SELL': 0.90},
            MarketTrend.CRASH: {'BUY': 0.05, 'SELL': 0.95},
        }

        # Defensive stocks that can be bought even in downtrends
        self.defensive_tickers = {
            'MRK', 'PFE', 'JNJ', 'PG', 'KO', 'WMT', 'VZ', 'T',
            'ED', 'DUK', 'SO', 'NEE',  # Utilities
            'GIS', 'K', 'KHC', 'CPB',  # Consumer staples
        }

        # Aggressive short candidates in downtrends
        self.aggressive_shorts = {
            'TSLA', 'NVDA', 'ARKK', 'UBER', 'AFRM', 'COIN',
            'RIVN', 'LCID', 'PLTR', 'SNAP', 'PINS',
        }

    def should_generate_signal(
        self,
        ticker: str,
        signal_type: str,
        market_trend: MarketTrend = None
    ) -> Tuple[bool, float]:
        """
        Determine if a signal should be generated based on trend.

        Returns:
            Tuple of (should_generate, probability)
        """
        import random

        if market_trend is None:
            market_trend = self.trend_adapter.detect_market_trend()

        self.current_trend = market_trend

        probs = self.SIGNAL_GENERATION_PROBS.get(market_trend, {'BUY': 0.5, 'SELL': 0.5})
        signal_prob = probs.get(signal_type.upper(), 0.5)

        # Adjust for specific ticker characteristics
        ticker_upper = ticker.upper()

        if market_trend in [MarketTrend.STRONG_DOWNTREND, MarketTrend.BEAR_MARKET, MarketTrend.CRASH]:
            if ticker_upper in self.defensive_tickers and signal_type.upper() == 'BUY':
                signal_prob = min(signal_prob * 1.5, 0.6)  # Boost defensive BUYs
            elif ticker_upper in self.aggressive_shorts and signal_type.upper() == 'SELL':
                signal_prob = min(signal_prob * 1.3, 0.95)  # Boost aggressive shorts

        elif market_trend in [MarketTrend.STRONG_UPTREND, MarketTrend.MODERATE_UPTREND]:
            if ticker_upper in self.aggressive_shorts and signal_type.upper() == 'BUY':
                signal_prob = min(signal_prob * 1.3, 0.90)  # Growth stocks do well in uptrends

        should_generate = random.random() < signal_prob

        return should_generate, signal_prob

    def get_recommended_signal_type(self, ticker: str) -> str:
        """Get recommended signal type based on current trend."""
        trend = self.trend_adapter.detect_market_trend()

        probs = self.SIGNAL_GENERATION_PROBS.get(trend, {'BUY': 0.5, 'SELL': 0.5})

        if probs['SELL'] > probs['BUY']:
            return 'SELL'
        else:
            return 'BUY'


def create_trend_adapter() -> DynamicTrendAdapter:
    """Factory function to create trend adapter."""
    return DynamicTrendAdapter()


def create_signal_generator() -> TrendBasedSignalGenerator:
    """Factory function to create signal generator."""
    return TrendBasedSignalGenerator()


# Test function
if __name__ == "__main__":
    print("=" * 70)
    print("MARKET TREND ADAPTER TEST")
    print("=" * 70)

    adapter = create_trend_adapter()

    # Detect current trend
    trend_info = adapter.get_current_trend_info()
    print(f"\nCurrent Market Trend: {trend_info['trend']}")
    print(f"Description: {trend_info['description']}")
    print(f"BUY win rate: {trend_info['buy_win_rate']:.0%}")
    print(f"SELL win rate: {trend_info['sell_win_rate']:.0%}")
    print(f"Recommendation: {trend_info['recommendation']}")

    # Test signal adjustment
    test_signals = [
        ('AAPL', 'BUY', 0.75),
        ('TSLA', 'SELL', 0.80),
        ('XOM', 'BUY', 0.65),
        ('NVDA', 'SELL', 0.85),
    ]

    print(f"\n{'='*70}")
    print("SIGNAL ADJUSTMENTS")
    print(f"{'Ticker':<10} {'Type':<6} {'Orig':<8} {'Adj':<8} {'Position':<10} {'Trade':<6}")
    print("-" * 70)

    for ticker, signal_type, confidence in test_signals:
        result = adapter.adjust_signal_for_trend(ticker, signal_type, confidence)
        print(f"{ticker:<10} {signal_type:<6} {confidence:<8.1%} "
              f"{result.adjusted_confidence:<8.1%} "
              f"{result.position_multiplier:<10.2f}x "
              f"{'Yes' if result.should_trade else 'No':<6}")
        if result.block_reason:
            print(f"  Blocked: {result.block_reason}")

    print(f"\n{'='*70}")
    print("ADAPTER STATISTICS")
    stats = adapter.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
