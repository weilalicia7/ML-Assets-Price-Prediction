"""
US/International Model Optimizer
================================
Implements all 15 fixes from 'us model fixing1.pdf' and 'us model fixing 2.pdf'

FIXES 1-7 (from us model fixing1.pdf):
    Fix 1: Higher SELL confidence threshold (75% base)
    Fix 2: Asset-class specific SELL thresholds
    Fix 3: 50% SELL position size reduction
    Fix 4: Asset-specific stop-losses (6-12%)
    Fix 5: 30% boost for US stock BUY allocations
    Fix 6: NG=F added to SELL blacklist
    Fix 7: 70% reduction for commodity SELL positions

FIXES 8-15 (from us model fixing 2.pdf):
    Fix 8: Extended Profit Blocklist
    Fix 9: Dynamic Position Sizing by Win Rate
    Fix 10: Asset Class Base Allocations
    Fix 11: Kelly Criterion Position Sizing
    Fix 12: BUY Signal Rules (min conf, multiplier, stop-loss)
    Fix 13: SELL Signal Rules (min conf, multiplier, stop-loss, blacklists)
    Fix 14: High-Profit Pattern Detection
    Fix 15: Dynamic Profit-Taking Rules

NEW FIXES (from us model fixing 2.pdf - adaptive blocking):
    Fix 16: COMPLETE China stock blocking (use China model with DeepSeek instead)
    Fix 17: Adaptive blocking for JPY SELL signals (0% win rate)
    Fix 18: Adaptive blocking for Crude Oil CL=F (consistent loser)
    Fix 19: Dynamic position adjustment based on market regime

FIX 20 (from us model only fixing1.pdf - Dynamic SELL Thresholds):
    Fix 20: Dynamic SELL confidence thresholds based on trend/volatility
            - In downtrends: Lower threshold to allow more SELL signals (0.55 floor)
            - In uptrends: Raise threshold to filter weak SELL signals (0.85 ceiling)
            - Volatility-adjusted trend classification
            - Replaces static 80% SELL_MIN_CONFIDENCE that blocked almost ALL SELL signals
            - ONLY applies to US/Intl model (China model unchanged)

FIXES 24-26 (from us model fixing3.pdf - Profit Maximization):
    Fix 24: Dynamic Kelly Fraction - Adaptive position sizing based on regime/account/momentum
    Fix 25: Position Concentration - Concentrate capital in top signals (top 3 get 70%)
    Fix 26: Dynamic Profit Targets - ATR-based profit targets adjusted for volatility/trend

FIXES 27-33 (from us model fixing4.pdf - US Market Specific Optimizations):
    Fix 27: US Market Regime Classifier - Bull/bear momentum, FOMC, earnings, sector rotation
    Fix 28: Sector Momentum Integration - Sector ETF momentum, relative strength, rotation detection
    Fix 29: Earnings Season Optimizer - Pre/post-earnings adjustments, earnings surprise momentum
    Fix 30: FOMC & Economic Calendar - Fed meeting adjustments, rate-sensitive sector handling
    Fix 31: Options Expiration Optimizer - Gamma hedging adjustments, volatility crush, OpEx effects
    Fix 32: Market Internals Integration - Advance-Decline, New Highs/Lows, TRIN, McClellan Oscillator
    Fix 33: US-Specific Risk Models - Sector concentration risk, factor exposure optimization

FIXES 34-49 (from us model fixing5.pdf, fixing6.pdf - Advanced Profit Strategies):
    Fix 34: Intraday Momentum Timing - Time-of-day patterns, opening/closing sessions
    Fix 35: Market Cap Tier Optimizer - Mega/large/mid/small cap specific rules
    Fix 36: Quarter-End Window Dressing - Institutional rebalancing exploitation
    Fix 37: Earnings Gap Trading - Pre/post earnings gap capture
    Fix 38: Sector Rotation Momentum - Cross-sector flow detection
    Fix 39: VIX Term Structure Arbitrage - Contango/backwardation signals
    Fix 40: Economic Data Reactions - CPI, NFP, GDP reaction patterns
    Fix 41: Put/Call Ratio Reversals - Sentiment extreme detection
    Fix 42: Unified US Profit Maximizer - Combines all US-specific optimizations
    Fix 43: Enhanced Sector Rotation Detector - Leading indicator integration
    Fix 44: US Catalyst Detector - Event-driven signal enhancement
    Fix 45: Enhanced Intraday with Volume Profile - VWAP/volume analysis
    Fix 46: Momentum Acceleration Detector - 2nd derivative momentum
    Fix 47: US-Specific Profit Rules - Stock type specific rules
    Fix 48: Smart Profit Taker - Multi-factor profit optimization
    Fix 49: Backtest Profit Maximizer - Theoretical maximum strategies

FIXES 50-53 (from us model fixing7.pdf - Market Structure & Institutional):
    Fix 50: US Market Structure Arbitrage - ETF premium/discount, MOC imbalances, gamma exposure, dark pools
    Fix 51: Smart Beta Overlay - Factor timing (momentum, value, low vol, quality, size)
    Fix 52: Volatility Regime Switching - VIX-based strategy adaptation (trend vs mean reversion)
    Fix 53: Institutional Flow Mirroring - 13F analysis, ETF flows, block trades, insider patterns

FIXES 54-61 (from us model fixing8.pdf - US-Specific Aggressive Alpha Strategies):
    Fix 54: Mega-Cap Tech Momentum Exploitation - FAANG+M relative strength, spillover effects, golden cross
    Fix 55: Semiconductor Super-Cycle Detection - Book-to-bill, inventory days, capex cycle phases
    Fix 56: AI Thematic Concentration - AI pure-plays, sentiment tracking, hyperscaler capex signals
    Fix 57: Fed Liquidity Regime Optimization - Balance sheet, RRP, rate cycle, sector sensitivity
    Fix 58: Retail Options Flow Analysis - Call/put ratio, unusual activity, gamma exposure, OPEX effects
    Fix 59: Meme Stock Pattern Detection - Social momentum, short squeeze potential, meme phases
    Fix 60: Earnings-Driven Sector Rotation - Beat rate trends, revision momentum, sector leadership
    Fix 61: Real-Time US News Analysis - News categorization, sentiment scoring, impact decay modeling

Author: Claude Code
Last Updated: 2025-12-09
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque

# Import risk management components (Fixes 16-19)
try:
    from ..risk_management import (
        MarketRegimeDetector,
        MarketRegime,
        AdaptiveBlocker,
        BlockingLevel,
        BlockingResult,
        PositionAdjuster,
        PositionAdjustment,
    )
    RISK_MANAGEMENT_AVAILABLE = True
except ImportError:
    # Fallback if risk_management module not available
    RISK_MANAGEMENT_AVAILABLE = False
    MarketRegimeDetector = None
    MarketRegime = None
    AdaptiveBlocker = None

# Import Yahoo screener for dynamic ticker discovery (Fixes 54-61)
try:
    import yfinance as yf
    from yfinance import EquityQuery
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None
    EquityQuery = None


# ============================================================================
# DYNAMIC US TICKER DISCOVERY (Fixes 54-61)
# ============================================================================
# Uses Yahoo Finance real-time screener to dynamically discover tickers
# instead of hardcoded lists. Falls back to static defaults if unavailable.
# NOTE: This is US/INTL model ONLY - China stocks use DeepSeek model.
# ============================================================================

class DynamicUSTickerDiscovery:
    """
    Dynamic ticker discovery for US/International model using Yahoo Finance.

    This class provides real-time ticker lists for Fixes 54-61 strategies:
    - Mega-cap tech stocks (Fix 54)
    - Semiconductor stocks (Fix 55)
    - AI-related stocks (Fix 56)
    - High retail options interest stocks (Fix 58)
    - Meme stocks (Fix 59)

    Uses Yahoo Finance EquityQuery for dynamic discovery with static fallbacks.

    NOTE: This is for US/INTL model ONLY. China stocks (.HK, .SS, .SZ) are
    handled by the separate China/DeepSeek model and should NOT appear here.
    """

    # Cache settings
    CACHE_TTL_SECONDS = 3600  # 1 hour cache for ticker lists

    # Static fallback lists (used when Yahoo Finance unavailable)
    # These are periodically updated but serve as safety net
    FALLBACK_MEGA_CAP_TECH = [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA',
        'NFLX', 'AMD', 'CRM', 'ADBE', 'ORCL', 'INTC', 'AVGO',
    ]

    FALLBACK_SEMICONDUCTORS = [
        'NVDA', 'AMD', 'AVGO', 'QCOM', 'MRVL', 'NXPI', 'ON', 'INTC',
        'TXN', 'MU', 'ADI', 'MCHP', 'ASML', 'AMAT', 'LRCX', 'KLAC',
        'TSM', 'WDC', 'STX', 'ARM',
    ]

    FALLBACK_AI_STOCKS = [
        'NVDA', 'AMD', 'AVGO', 'MRVL', 'ARM', 'MSFT', 'GOOGL', 'GOOG',
        'META', 'AMZN', 'CRM', 'NOW', 'PLTR', 'SNOW', 'MDB', 'SMCI',
        'DELL', 'HPE', 'VRT', 'ANET', 'MU', 'WDC', 'ASML', 'AMAT',
        'LRCX', 'KLAC', 'TSM',
    ]

    FALLBACK_HIGH_RETAIL_OPTIONS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD',
        'NFLX', 'SPY', 'QQQ', 'IWM', 'GME', 'AMC', 'PLTR', 'NIO',
        'RIVN', 'LCID', 'F', 'COIN', 'HOOD', 'SOFI', 'SQ', 'PYPL',
    ]

    FALLBACK_MEME_STOCKS = [
        'GME', 'AMC', 'BB', 'NOK', 'PLTR', 'SOFI', 'HOOD', 'WISH',
        'CLOV', 'WKHS', 'RIVN', 'LCID', 'NIO', 'COIN', 'MSTR',
    ]

    def __init__(self):
        self._cache = {}
        self._cache_timestamps = {}
        self._discovery_available = YFINANCE_AVAILABLE

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        import time
        if cache_key not in self._cache:
            return False
        timestamp = self._cache_timestamps.get(cache_key, 0)
        return (time.time() - timestamp) < self.CACHE_TTL_SECONDS

    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache discovery result."""
        import time
        self._cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()

    def _filter_us_only(self, tickers: List[str]) -> List[str]:
        """
        Filter to US stocks only - exclude China stocks.
        China stocks (.HK, .SS, .SZ) should go to DeepSeek model.
        """
        china_suffixes = ('.HK', '.SS', '.SZ')
        return [t for t in tickers if not t.upper().endswith(china_suffixes)]

    def discover_mega_cap_tech(self, min_market_cap: float = 100e9) -> Dict[str, Dict]:
        """
        Dynamically discover mega-cap tech stocks using Yahoo Finance.

        Args:
            min_market_cap: Minimum market cap in USD (default 100B)

        Returns:
            Dict mapping ticker to metadata (name, weight, category, market_cap)
        """
        cache_key = f"mega_cap_tech_{min_market_cap}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        result = {}

        if self._discovery_available:
            try:
                # Query for large tech stocks
                tech_query = EquityQuery('AND', [
                    EquityQuery('EQ', ['sector', 'Technology']),
                    EquityQuery('GT', ['intradaymarketcap', min_market_cap]),
                ])

                screen_result = yf.screen(tech_query, size=50, sortField='intradaymarketcap', sortAsc=False)

                if screen_result and 'quotes' in screen_result:
                    for quote in screen_result['quotes']:
                        symbol = quote.get('symbol', '')
                        if symbol and not symbol.endswith(('.HK', '.SS', '.SZ')):
                            market_cap = quote.get('marketCap', 0) or 0
                            # Calculate weight based on market cap tier
                            if market_cap > 2e12:  # >2T
                                weight = 1.4
                            elif market_cap > 1e12:  # >1T
                                weight = 1.3
                            elif market_cap > 500e9:  # >500B
                                weight = 1.2
                            elif market_cap > 200e9:  # >200B
                                weight = 1.1
                            else:
                                weight = 1.0

                            result[symbol] = {
                                'name': quote.get('shortName', symbol),
                                'weight': weight,
                                'category': self._classify_tech_category(symbol),
                                'market_cap': market_cap,
                            }

                if result:
                    self._cache_result(cache_key, result)
                    return result

            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Dynamic mega-cap discovery failed: {e}")

        # Fallback to static list
        for ticker in self.FALLBACK_MEGA_CAP_TECH:
            result[ticker] = {
                'name': ticker,
                'weight': 1.2,
                'category': self._classify_tech_category(ticker),
                'market_cap': 0,
            }

        self._cache_result(cache_key, result)
        return result

    def _classify_tech_category(self, ticker: str) -> str:
        """Classify tech stock into category."""
        ticker = ticker.upper()
        gpu_chips = {'NVDA', 'AMD', 'INTC'}
        software = {'MSFT', 'CRM', 'ADBE', 'ORCL', 'NOW', 'PLTR'}
        social = {'META'}
        ecommerce = {'AMZN'}
        advertising = {'GOOGL', 'GOOG'}
        streaming = {'NFLX'}
        ev = {'TSLA'}
        semiconductors = {'AVGO', 'QCOM', 'MRVL', 'TXN', 'MU', 'ARM'}

        if ticker in gpu_chips:
            return 'gpu_compute'
        elif ticker in software:
            return 'software'
        elif ticker in social:
            return 'social'
        elif ticker in ecommerce:
            return 'ecommerce'
        elif ticker in advertising:
            return 'advertising'
        elif ticker in streaming:
            return 'streaming'
        elif ticker in ev:
            return 'ev'
        elif ticker in semiconductors:
            return 'semiconductors'
        else:
            return 'technology'

    def discover_semiconductors(self) -> Dict[str, Dict]:
        """
        Dynamically discover semiconductor stocks using Yahoo Finance.

        Returns:
            Dict mapping ticker to metadata (type, segment, cycle_beta)
        """
        cache_key = "semiconductors"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        result = {}

        if self._discovery_available:
            try:
                # Query for semiconductor stocks
                semi_query = EquityQuery('AND', [
                    EquityQuery('EQ', ['industry', 'Semiconductors']),
                    EquityQuery('GT', ['intradaymarketcap', 10e9]),  # >10B market cap
                ])

                screen_result = yf.screen(semi_query, size=50, sortField='intradaymarketcap', sortAsc=False)

                if screen_result and 'quotes' in screen_result:
                    for quote in screen_result['quotes']:
                        symbol = quote.get('symbol', '')
                        if symbol and not symbol.endswith(('.HK', '.SS', '.SZ')):
                            market_cap = quote.get('marketCap', 0) or 0
                            segment = self._classify_semi_segment(symbol)
                            semi_type = self._classify_semi_type(symbol)

                            # Cycle beta based on segment
                            cycle_beta = self._get_semi_cycle_beta(segment)

                            result[symbol] = {
                                'type': semi_type,
                                'segment': segment,
                                'cycle_beta': cycle_beta,
                                'market_cap': market_cap,
                            }

                # Also query equipment makers
                equip_query = EquityQuery('AND', [
                    EquityQuery('EQ', ['industry', 'Semiconductor Equipment & Materials']),
                    EquityQuery('GT', ['intradaymarketcap', 10e9]),
                ])

                equip_result = yf.screen(equip_query, size=20, sortField='intradaymarketcap', sortAsc=False)

                if equip_result and 'quotes' in equip_result:
                    for quote in equip_result['quotes']:
                        symbol = quote.get('symbol', '')
                        if symbol and symbol not in result and not symbol.endswith(('.HK', '.SS', '.SZ')):
                            result[symbol] = {
                                'type': 'equipment',
                                'segment': 'equipment',
                                'cycle_beta': 1.5,
                                'market_cap': quote.get('marketCap', 0) or 0,
                            }

                if result:
                    self._cache_result(cache_key, result)
                    return result

            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Dynamic semiconductor discovery failed: {e}")

        # Fallback to static list
        for ticker in self.FALLBACK_SEMICONDUCTORS:
            result[ticker] = {
                'type': self._classify_semi_type(ticker),
                'segment': self._classify_semi_segment(ticker),
                'cycle_beta': self._get_semi_cycle_beta(self._classify_semi_segment(ticker)),
                'market_cap': 0,
            }

        self._cache_result(cache_key, result)
        return result

    def _classify_semi_type(self, ticker: str) -> str:
        """Classify semiconductor company type."""
        ticker = ticker.upper()
        fabless = {'NVDA', 'AMD', 'AVGO', 'QCOM', 'MRVL', 'NXPI', 'ON', 'ARM'}
        idm = {'INTC', 'TXN', 'MU', 'ADI', 'MCHP'}
        foundry = {'TSM'}
        equipment = {'ASML', 'AMAT', 'LRCX', 'KLAC'}
        memory = {'WDC', 'STX'}

        if ticker in fabless:
            return 'fabless'
        elif ticker in idm:
            return 'idm'
        elif ticker in foundry:
            return 'foundry'
        elif ticker in equipment:
            return 'equipment'
        elif ticker in memory:
            return 'memory'
        else:
            return 'fabless'

    def _classify_semi_segment(self, ticker: str) -> str:
        """Classify semiconductor segment."""
        ticker = ticker.upper()
        segments = {
            'NVDA': 'gpu', 'AMD': 'cpu_gpu', 'INTC': 'cpu', 'AVGO': 'diversified',
            'QCOM': 'mobile', 'MRVL': 'infrastructure', 'NXPI': 'auto',
            'ON': 'auto_industrial', 'TXN': 'analog', 'MU': 'memory',
            'ADI': 'analog', 'MCHP': 'mcu', 'TSM': 'foundry',
            'ASML': 'litho', 'AMAT': 'equipment', 'LRCX': 'etch',
            'KLAC': 'inspection', 'WDC': 'nand', 'STX': 'hdd', 'ARM': 'ip',
        }
        return segments.get(ticker, 'diversified')

    def _get_semi_cycle_beta(self, segment: str) -> float:
        """Get cycle beta for semiconductor segment."""
        betas = {
            'memory': 2.0, 'nand': 1.8, 'gpu': 1.8, 'etch': 1.7,
            'equipment': 1.6, 'cpu_gpu': 1.6, 'litho': 1.5, 'inspection': 1.5,
            'foundry': 1.4, 'infrastructure': 1.4, 'hdd': 1.4, 'mobile': 1.3,
            'auto': 1.3, 'auto_industrial': 1.2, 'diversified': 1.2, 'mcu': 1.1,
            'cpu': 1.0, 'analog': 0.9, 'ip': 1.3,
        }
        return betas.get(segment, 1.2)

    def discover_ai_stocks(self) -> Dict[str, Dict]:
        """
        Dynamically discover AI-related stocks.

        Returns:
            Dict mapping ticker to metadata (exposure, category, ai_score)
        """
        cache_key = "ai_stocks"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        result = {}

        # AI stocks are harder to screen dynamically - use enhanced static list
        # but validate with real-time data
        if self._discovery_available:
            try:
                # Get mega-cap tech (many are AI plays)
                mega_caps = self.discover_mega_cap_tech()

                # Get semiconductors (AI infrastructure)
                semis = self.discover_semiconductors()

                # Combine and classify AI exposure
                ai_pure_plays = {'NVDA', 'SMCI', 'ARM'}
                ai_high = {'AMD', 'AVGO', 'MRVL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'PLTR', 'TSM'}
                ai_medium = {'AMZN', 'CRM', 'NOW', 'SNOW', 'MDB', 'DELL', 'HPE', 'VRT', 'ANET', 'MU', 'WDC'}

                for ticker in set(mega_caps.keys()) | set(semis.keys()) | ai_pure_plays | ai_high | ai_medium:
                    if ticker.endswith(('.HK', '.SS', '.SZ')):
                        continue

                    if ticker in ai_pure_plays:
                        exposure = 'pure_play'
                        ai_score = 10
                    elif ticker in ai_high:
                        exposure = 'high'
                        ai_score = 8
                    elif ticker in ai_medium:
                        exposure = 'medium'
                        ai_score = 6
                    else:
                        exposure = 'low'
                        ai_score = 4

                    result[ticker] = {
                        'exposure': exposure,
                        'category': self._classify_ai_category(ticker),
                        'ai_score': ai_score,
                    }

                if result:
                    self._cache_result(cache_key, result)
                    return result

            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Dynamic AI stock discovery failed: {e}")

        # Fallback
        for ticker in self.FALLBACK_AI_STOCKS:
            ai_pure = {'NVDA', 'SMCI'}
            ai_high = {'AMD', 'AVGO', 'MRVL', 'ARM', 'MSFT', 'GOOGL', 'META', 'PLTR', 'TSM'}

            if ticker in ai_pure:
                exposure, ai_score = 'pure_play', 10
            elif ticker in ai_high:
                exposure, ai_score = 'high', 8
            else:
                exposure, ai_score = 'medium', 6

            result[ticker] = {
                'exposure': exposure,
                'category': self._classify_ai_category(ticker),
                'ai_score': ai_score,
            }

        self._cache_result(cache_key, result)
        return result

    def _classify_ai_category(self, ticker: str) -> str:
        """Classify AI stock category."""
        ticker = ticker.upper()
        categories = {
            'NVDA': 'gpu_compute', 'AMD': 'gpu_compute', 'AVGO': 'ai_networking',
            'MRVL': 'ai_networking', 'ARM': 'ai_chips', 'MSFT': 'ai_platform',
            'GOOGL': 'ai_platform', 'GOOG': 'ai_platform', 'META': 'ai_platform',
            'AMZN': 'ai_cloud', 'CRM': 'ai_enterprise', 'NOW': 'ai_enterprise',
            'PLTR': 'ai_analytics', 'SNOW': 'ai_data', 'MDB': 'ai_data',
            'SMCI': 'ai_servers', 'DELL': 'ai_servers', 'HPE': 'ai_servers',
            'VRT': 'ai_power', 'ANET': 'ai_networking', 'MU': 'ai_memory',
            'WDC': 'ai_storage', 'ASML': 'ai_equipment', 'AMAT': 'ai_equipment',
            'LRCX': 'ai_equipment', 'KLAC': 'ai_equipment', 'TSM': 'ai_foundry',
        }
        return categories.get(ticker, 'ai_related')

    def discover_high_retail_options(self) -> set:
        """
        Discover stocks with high retail options interest.

        Returns:
            Set of tickers with high retail options activity
        """
        cache_key = "high_retail_options"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        result = set()

        if self._discovery_available:
            try:
                # Get most active stocks (proxy for options activity)
                active_result = yf.screen('most_actives', size=50)

                if active_result and 'quotes' in active_result:
                    for quote in active_result['quotes']:
                        symbol = quote.get('symbol', '')
                        if symbol and not symbol.endswith(('.HK', '.SS', '.SZ')):
                            result.add(symbol)

                # Add known high-options ETFs
                result.update({'SPY', 'QQQ', 'IWM', 'TQQQ', 'SQQQ'})

                if result:
                    self._cache_result(cache_key, result)
                    return result

            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Dynamic retail options discovery failed: {e}")

        # Fallback
        result = set(self.FALLBACK_HIGH_RETAIL_OPTIONS)
        self._cache_result(cache_key, result)
        return result

    def discover_meme_stocks(self) -> Dict[str, Dict]:
        """
        Discover meme stocks using social/volume indicators.

        Returns:
            Dict mapping ticker to metadata (tier, category, base_volatility)
        """
        cache_key = "meme_stocks"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        result = {}

        if self._discovery_available:
            try:
                # Get day gainers (meme stocks often show up here)
                gainers = yf.screen('day_gainers', size=30)

                # Get small cap gainers
                small_caps = yf.screen('small_cap_gainers', size=30)

                potential_memes = set()

                if gainers and 'quotes' in gainers:
                    for quote in gainers['quotes']:
                        symbol = quote.get('symbol', '')
                        pct_change = quote.get('regularMarketChangePercent', 0) or 0
                        # High daily move is meme indicator
                        if symbol and pct_change > 10 and not symbol.endswith(('.HK', '.SS', '.SZ')):
                            potential_memes.add(symbol)

                if small_caps and 'quotes' in small_caps:
                    for quote in small_caps['quotes']:
                        symbol = quote.get('symbol', '')
                        pct_change = quote.get('regularMarketChangePercent', 0) or 0
                        if symbol and pct_change > 15 and not symbol.endswith(('.HK', '.SS', '.SZ')):
                            potential_memes.add(symbol)

                # Add known meme stocks
                known_memes = {'GME', 'AMC', 'BB', 'BBBY', 'NOK', 'PLTR', 'SOFI', 'HOOD',
                               'WISH', 'CLOV', 'WKHS', 'RIVN', 'LCID', 'NIO', 'COIN', 'MSTR'}
                potential_memes.update(known_memes)

                for ticker in potential_memes:
                    tier = 1 if ticker in {'GME', 'AMC'} else 2 if ticker in known_memes else 3
                    result[ticker] = {
                        'tier': tier,
                        'category': self._classify_meme_category(ticker),
                        'base_volatility': 3.0 if tier == 1 else 2.5 if tier == 2 else 2.0,
                    }

                if result:
                    self._cache_result(cache_key, result)
                    return result

            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Dynamic meme stock discovery failed: {e}")

        # Fallback
        for ticker in self.FALLBACK_MEME_STOCKS:
            tier = 1 if ticker in {'GME', 'AMC'} else 2
            result[ticker] = {
                'tier': tier,
                'category': self._classify_meme_category(ticker),
                'base_volatility': 2.5 if tier == 1 else 2.0,
            }

        self._cache_result(cache_key, result)
        return result

    def _classify_meme_category(self, ticker: str) -> str:
        """Classify meme stock category."""
        ticker = ticker.upper()
        categories = {
            'GME': 'original', 'AMC': 'original', 'BB': 'second_wave',
            'BBBY': 'second_wave', 'NOK': 'second_wave', 'PLTR': 'tech_meme',
            'SOFI': 'fintech_meme', 'HOOD': 'fintech_meme', 'WISH': 'retail_meme',
            'CLOV': 'healthcare_meme', 'WKHS': 'ev_meme', 'RIVN': 'ev_meme',
            'LCID': 'ev_meme', 'NIO': 'ev_meme', 'COIN': 'crypto_meme',
            'MSTR': 'crypto_meme',
        }
        return categories.get(ticker, 'meme')

    def refresh_all_caches(self):
        """Force refresh all cached ticker lists."""
        self._cache.clear()
        self._cache_timestamps.clear()

        # Re-discover all categories
        self.discover_mega_cap_tech()
        self.discover_semiconductors()
        self.discover_ai_stocks()
        self.discover_high_retail_options()
        self.discover_meme_stocks()

    def get_discovery_status(self) -> Dict[str, Any]:
        """Get status of dynamic discovery."""
        import time
        status = {
            'yfinance_available': self._discovery_available,
            'cached_categories': list(self._cache.keys()),
            'cache_ages': {},
        }

        for key, timestamp in self._cache_timestamps.items():
            age_seconds = time.time() - timestamp
            status['cache_ages'][key] = f"{age_seconds:.0f}s ago"

        return status


# Module-level singleton for dynamic discovery
_dynamic_discovery_instance = None


def get_dynamic_us_ticker_discovery() -> DynamicUSTickerDiscovery:
    """Get singleton instance of dynamic ticker discovery."""
    global _dynamic_discovery_instance
    if _dynamic_discovery_instance is None:
        _dynamic_discovery_instance = DynamicUSTickerDiscovery()
    return _dynamic_discovery_instance


class AssetClass(Enum):
    """Asset class enumeration for US/International model ONLY.

    NOTE: China stocks (.HK, .SS, .SZ) are NOT handled here.
    They should be routed to the China model with DeepSeek API.
    See market_classifier.py for proper routing.
    """
    STOCK = 'stock'              # US/International stocks (includes .L, .T, .DE, etc.)
    CRYPTOCURRENCY = 'cryptocurrency'
    COMMODITY = 'commodity'
    FOREX = 'forex'
    ETF = 'etf'
    JPY_PAIR = 'jpy_pair'        # JPY currency pairs (Fix 17: adaptive blocking)
    CRUDE_OIL = 'crude_oil'      # Crude Oil CL=F (Fix 18: adaptive blocking)


@dataclass
class SignalOptimization:
    """Result of signal optimization."""
    ticker: str
    signal_type: str  # 'BUY' or 'SELL'
    original_confidence: float
    adjusted_confidence: float
    position_multiplier: float
    stop_loss_pct: float
    take_profit_levels: Dict[float, float]  # {profit_threshold: close_pct}
    blocked: bool
    block_reason: Optional[str]
    fixes_applied: List[str]
    asset_class: str
    kelly_fraction: Optional[float]
    # Fix 17-19: Regime-based fields
    market_regime: Optional[str] = None
    regime_confidence: Optional[float] = None
    regime_position_adjustment: Optional[float] = None
    adaptive_blocking_applied: bool = False
    # Fix 22: Signal quality score
    signal_quality_score: Optional[float] = None
    quality_components: Optional[Dict[str, float]] = None


# ========== FIX 22: Signal Quality Scorer (US/INTL MODEL ONLY) ==========
class SignalQualityScorer:
    """
    Score signal quality based on multiple factors beyond just confidence.

    FIX 22: Signal Quality vs Position Sizing Match (US/INTL MODEL ONLY)

    The problem: Current model only uses confidence number for position sizing,
    ignoring other signal quality indicators.

    Solution: Score signals on multiple dimensions:
    1. Model confidence (base score)
    2. Trend alignment (signal direction matches trend)
    3. Volume confirmation (high volume supports the signal)
    4. Volatility regime (moderate volatility is ideal)
    5. Win rate history (ticker-specific historical performance)

    Position sizing should scale with quality score, not just confidence.
    """

    # Quality score weights
    QUALITY_WEIGHTS = {
        'confidence': 0.35,      # Model confidence
        'trend_alignment': 0.25, # Signal aligns with trend
        'volume_confirm': 0.15,  # Volume supports signal
        'volatility_regime': 0.15, # Volatility in good range
        'win_rate': 0.10,        # Historical win rate
    }

    # Ideal volatility range (annualized)
    IDEAL_VOLATILITY_RANGE = (0.15, 0.35)

    def __init__(self):
        self.historical_scores = []

    def score_signal(
        self,
        signal_type: str,
        confidence: float,
        momentum: float,
        volatility: float,
        volume_ratio: float = 1.0,
        win_rate: float = 0.50,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate comprehensive signal quality score.

        Args:
            signal_type: 'BUY' or 'SELL'
            confidence: Model confidence (0-1)
            momentum: Recent price momentum (e.g., 20-day return)
            volatility: Asset volatility (annualized)
            volume_ratio: Current volume / average volume
            win_rate: Historical win rate for this signal type

        Returns:
            (quality_score, component_scores)
            quality_score: 0-1, higher is better
            component_scores: Dict with individual component scores
        """
        components = {}

        # 1. Confidence score (0-1)
        components['confidence'] = min(1.0, confidence)

        # 2. Trend alignment score (0-1)
        # BUY signals should align with uptrend, SELL with downtrend
        if signal_type.upper() == 'BUY':
            # Positive momentum is good for BUY
            trend_score = 0.5 + min(0.5, max(-0.5, momentum * 2.5))
        else:
            # Negative momentum is good for SELL
            trend_score = 0.5 + min(0.5, max(-0.5, -momentum * 2.5))
        components['trend_alignment'] = trend_score

        # 3. Volume confirmation (0-1)
        # High volume (>1.2x average) confirms signal
        if volume_ratio > 1.5:
            vol_score = 1.0
        elif volume_ratio > 1.2:
            vol_score = 0.8
        elif volume_ratio > 0.8:
            vol_score = 0.6
        else:
            vol_score = 0.4  # Low volume = less reliable signal
        components['volume_confirm'] = vol_score

        # 4. Volatility regime score (0-1)
        # Moderate volatility is ideal for trading
        min_vol, max_vol = self.IDEAL_VOLATILITY_RANGE
        if min_vol <= volatility <= max_vol:
            vol_regime_score = 1.0
        elif volatility < min_vol:
            # Low volatility - less opportunity
            vol_regime_score = 0.5 + (volatility / min_vol) * 0.5
        else:
            # High volatility - more risk
            vol_regime_score = max(0.3, 1.0 - (volatility - max_vol) * 2)
        components['volatility_regime'] = vol_regime_score

        # 5. Win rate score (0-1)
        # Scale win rate to 0-1 (50% = 0.5, 70% = 0.9)
        components['win_rate'] = min(1.0, win_rate * 1.4)

        # Calculate weighted quality score
        quality_score = sum(
            components[k] * self.QUALITY_WEIGHTS[k]
            for k in self.QUALITY_WEIGHTS
        )

        return quality_score, components

    def get_position_multiplier(self, quality_score: float) -> float:
        """
        Convert quality score to position size multiplier.

        High quality signals get larger positions.
        Low quality signals get smaller positions.

        Args:
            quality_score: 0-1 quality score

        Returns:
            Position multiplier (0.5 to 1.5)
        """
        # Linear mapping: 0.3 quality -> 0.5x, 0.7 quality -> 1.0x, 0.9 quality -> 1.5x
        if quality_score < 0.3:
            return 0.5
        elif quality_score > 0.9:
            return 1.5
        else:
            # Linear interpolation between 0.3-0.9 -> 0.5-1.5
            return 0.5 + (quality_score - 0.3) / 0.6 * 1.0


# ========== FIX 23: Sentiment Gate (US/INTL MODEL ONLY) ==========
class SentimentGate:
    """
    Gate trading signals based on sentiment alignment.

    FIX 23: Sentiment Gating (US/INTL MODEL ONLY)

    The problem: Model ignores sentiment data when making trading decisions.
    Even with price-derived sentiment proxy, it's not being used for gating.

    Solution: Block or boost signals based on sentiment alignment:
    - BUY + Positive sentiment = BOOST (increase position)
    - BUY + Negative sentiment = GATE (reduce or block)
    - SELL + Negative sentiment = BOOST
    - SELL + Positive sentiment = GATE

    NOTE: With mock/proxy sentiment, use soft gating (reduce position)
    rather than hard blocking. Only hard block with real sentiment data.
    """

    # Sentiment thresholds
    STRONG_POSITIVE = 0.3
    WEAK_POSITIVE = 0.1
    WEAK_NEGATIVE = -0.1
    STRONG_NEGATIVE = -0.3

    # Position adjustments based on sentiment alignment
    ALIGNMENT_MULTIPLIERS = {
        'strong_aligned': 1.3,    # Strong sentiment supports signal
        'aligned': 1.1,           # Weak sentiment supports signal
        'neutral': 1.0,           # No clear sentiment
        'misaligned': 0.7,        # Weak sentiment opposes signal
        'strong_misaligned': 0.4, # Strong sentiment opposes signal (soft block)
    }

    def __init__(self, use_hard_blocking: bool = False):
        """
        Initialize SentimentGate.

        Args:
            use_hard_blocking: If True, strongly misaligned signals are blocked.
                              If False, they get reduced position size (safer with proxy data).
        """
        self.use_hard_blocking = use_hard_blocking

    def evaluate_sentiment_alignment(
        self,
        signal_type: str,
        sentiment: float,
    ) -> Tuple[str, float, bool]:
        """
        Evaluate how sentiment aligns with signal direction.

        Args:
            signal_type: 'BUY' or 'SELL'
            sentiment: Combined sentiment score (-1 to 1)

        Returns:
            (alignment_type, position_multiplier, should_block)
        """
        is_buy = signal_type.upper() == 'BUY'

        # Determine alignment
        if is_buy:
            # BUY signals want positive sentiment
            if sentiment >= self.STRONG_POSITIVE:
                alignment = 'strong_aligned'
            elif sentiment >= self.WEAK_POSITIVE:
                alignment = 'aligned'
            elif sentiment <= self.STRONG_NEGATIVE:
                alignment = 'strong_misaligned'
            elif sentiment <= self.WEAK_NEGATIVE:
                alignment = 'misaligned'
            else:
                alignment = 'neutral'
        else:
            # SELL signals want negative sentiment
            if sentiment <= self.STRONG_NEGATIVE:
                alignment = 'strong_aligned'
            elif sentiment <= self.WEAK_NEGATIVE:
                alignment = 'aligned'
            elif sentiment >= self.STRONG_POSITIVE:
                alignment = 'strong_misaligned'
            elif sentiment >= self.WEAK_POSITIVE:
                alignment = 'misaligned'
            else:
                alignment = 'neutral'

        multiplier = self.ALIGNMENT_MULTIPLIERS[alignment]
        should_block = self.use_hard_blocking and alignment == 'strong_misaligned'

        return alignment, multiplier, should_block

    def gate_signal(
        self,
        signal_type: str,
        confidence: float,
        sentiment: float,
        is_proxy_sentiment: bool = True,
    ) -> Dict[str, Any]:
        """
        Apply sentiment gating to a trading signal.

        Args:
            signal_type: 'BUY' or 'SELL'
            confidence: Model confidence
            sentiment: Combined sentiment score
            is_proxy_sentiment: True if using price-derived proxy (softer gating)

        Returns:
            Dict with gating results
        """
        alignment, multiplier, should_block = self.evaluate_sentiment_alignment(
            signal_type, sentiment
        )

        # With proxy sentiment, never hard block - just reduce position
        if is_proxy_sentiment and should_block:
            should_block = False
            multiplier = 0.5  # Soft block instead

        return {
            'alignment': alignment,
            'multiplier': multiplier,
            'blocked': should_block,
            'sentiment': sentiment,
            'is_proxy': is_proxy_sentiment,
            'reason': f"Sentiment {alignment}: {sentiment:.2f} -> {multiplier:.1f}x"
        }


# ========== FIX 24: Adaptive Kelly Fraction (US/INTL MODEL ONLY) ==========
class AdaptiveKellyOptimizer:
    """
    Adaptive Kelly Criterion position sizing.

    FIX 24: Dynamic Kelly Fraction Optimization (US/INTL MODEL ONLY)

    The problem: Current Kelly uses fixed quarter-Kelly (25% cap) regardless
    of market conditions, account size, or recent performance.

    Solution: Adapt Kelly fraction based on:
    1. Market volatility regime (reduce in high vol)
    2. Account size (more aggressive with smaller accounts for growth)
    3. Recent performance momentum (reduce after losing streak)
    4. Correlation with existing positions (reduce for correlated)

    This maximizes compound growth while managing risk dynamically.
    """

    # Regime multipliers for Kelly fraction
    REGIME_MULTIPLIERS = {
        'high_volatility': 0.50,    # Half-Kelly in high vol
        'crisis': 0.25,             # Quarter-Kelly in crisis
        'normal_volatility': 0.75,  # Three-quarter Kelly normally
        'low_volatility': 1.00,     # Full Kelly in calm markets
        'strong_uptrend': 0.90,     # Slightly reduced in strong trends (momentum risk)
        'strong_downtrend': 0.60,   # More conservative in downtrends
    }

    # Account size multipliers (smaller accounts can be more aggressive)
    ACCOUNT_SIZE_THRESHOLDS = [
        (10000, 1.50),   # < $10k: 1.5x (aggressive growth)
        (50000, 1.20),   # $10k-$50k: 1.2x (moderate growth)
        (100000, 1.00),  # $50k-$100k: 1.0x (standard)
        (500000, 0.80),  # $100k-$500k: 0.8x (conservative)
        (float('inf'), 0.60),  # > $500k: 0.6x (very conservative)
    ]

    # Performance momentum adjustments
    MOMENTUM_THRESHOLDS = {
        'hot_streak': (0.70, 1.20),   # >70% recent wins: 1.2x
        'winning': (0.55, 1.10),       # 55-70% recent wins: 1.1x
        'neutral': (0.45, 1.00),       # 45-55% recent wins: 1.0x
        'losing': (0.35, 0.80),        # 35-45% recent wins: 0.8x
        'cold_streak': (0.00, 0.60),   # <35% recent wins: 0.6x
    }

    def __init__(
        self,
        base_kelly_cap: float = 0.25,
        min_kelly: float = 0.05,
        max_kelly: float = 0.50,
    ):
        """
        Initialize Adaptive Kelly Optimizer.

        Args:
            base_kelly_cap: Default Kelly cap (0.25 = quarter-Kelly)
            min_kelly: Minimum Kelly fraction (floor)
            max_kelly: Maximum Kelly fraction (ceiling)
        """
        self.base_kelly_cap = base_kelly_cap
        self.min_kelly = min_kelly
        self.max_kelly = max_kelly
        self.recent_trades = deque(maxlen=20)  # Last 20 trades

    def calculate_base_kelly(
        self,
        win_rate: float,
        avg_win: float = 0.05,
        avg_loss: float = 0.03,
    ) -> float:
        """Calculate base Kelly fraction from win rate and payoff ratio."""
        if win_rate <= 0 or avg_loss <= 0:
            return 0.0

        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss if avg_loss > 0 else 1.0

        kelly = (p * b - q) / b if b > 0 else 0.0
        return max(0.0, kelly)

    def get_regime_multiplier(self, market_regime: str, volatility: float) -> float:
        """Get Kelly multiplier based on market regime."""
        # Check for crisis/high vol conditions
        if volatility > 0.40:  # >40% annualized vol
            return self.REGIME_MULTIPLIERS['crisis']
        elif volatility > 0.30:
            return self.REGIME_MULTIPLIERS['high_volatility']
        elif volatility < 0.15:
            return self.REGIME_MULTIPLIERS['low_volatility']

        # Use regime if provided
        return self.REGIME_MULTIPLIERS.get(market_regime, 0.75)

    def get_account_size_multiplier(self, account_size: float) -> float:
        """Get Kelly multiplier based on account size."""
        for threshold, multiplier in self.ACCOUNT_SIZE_THRESHOLDS:
            if account_size < threshold:
                return multiplier
        return 0.60  # Default conservative for large accounts

    def get_momentum_multiplier(self, recent_win_rate: float = None) -> float:
        """Get Kelly multiplier based on recent performance."""
        if recent_win_rate is None:
            # Calculate from recent trades
            if len(self.recent_trades) < 5:
                return 1.0  # Not enough data
            recent_win_rate = sum(self.recent_trades) / len(self.recent_trades)

        for label, (threshold, multiplier) in self.MOMENTUM_THRESHOLDS.items():
            if recent_win_rate >= threshold:
                return multiplier
        return 0.60  # Cold streak

    def record_trade_result(self, won: bool):
        """Record a trade result for momentum tracking."""
        self.recent_trades.append(1.0 if won else 0.0)

    def get_recent_win_rate(self) -> float:
        """Get recent win rate from tracked trades."""
        if len(self.recent_trades) < 3:
            return 0.50  # Default to 50% if not enough data
        return sum(self.recent_trades) / len(self.recent_trades)

    def calculate_adaptive_kelly(
        self,
        win_rate: float,
        avg_win: float = 0.05,
        avg_loss: float = 0.03,
        market_regime: str = 'neutral',
        volatility: float = 0.20,
        account_size: float = 50000,
        recent_win_rate: float = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate adaptive Kelly fraction with all adjustments.

        Args:
            win_rate: Historical win rate for this signal/ticker
            avg_win: Average winning return
            avg_loss: Average losing return (absolute)
            market_regime: Current market regime
            volatility: Current volatility (annualized)
            account_size: Account size in dollars
            recent_win_rate: Recent win rate (last 20 trades)

        Returns:
            (adaptive_kelly, components)
            adaptive_kelly: Final position fraction (0-1)
            components: Dict with multiplier breakdown
        """
        # Step 1: Calculate base Kelly
        base_kelly = self.calculate_base_kelly(win_rate, avg_win, avg_loss)

        # Step 2: Get regime multiplier
        regime_mult = self.get_regime_multiplier(market_regime, volatility)

        # Step 3: Get account size multiplier
        size_mult = self.get_account_size_multiplier(account_size)

        # Step 4: Get momentum multiplier
        momentum_mult = self.get_momentum_multiplier(recent_win_rate)

        # Step 5: Calculate final Kelly
        adaptive_kelly = base_kelly * regime_mult * size_mult * momentum_mult

        # Step 6: Apply floor and ceiling
        adaptive_kelly = np.clip(adaptive_kelly, self.min_kelly, self.max_kelly)

        components = {
            'base_kelly': base_kelly,
            'regime_multiplier': regime_mult,
            'size_multiplier': size_mult,
            'momentum_multiplier': momentum_mult,
            'final_kelly': adaptive_kelly,
        }

        return adaptive_kelly, components


# ========== FIX 25: Position Concentration Optimizer (US/INTL MODEL ONLY) ==========
class PositionConcentrationOptimizer:
    """
    Optimize position concentration for maximum profit.

    FIX 25: Position Concentration Optimization (US/INTL MODEL ONLY)

    The problem: Equal distribution across signals dilutes returns.
    Best opportunities get same weight as mediocre ones.

    Solution: Concentrate capital in highest-conviction trades:
    1. Rank signals by composite score (confidence * quality * win_rate)
    2. Allocate using exponential weighting (top-heavy)
    3. Ensure top 3 positions get ~70% of allocated capital

    This focuses capital where it has the highest expected value.
    """

    def __init__(
        self,
        max_positions: int = 5,
        top_concentration_pct: float = 0.70,
        min_position_pct: float = 0.05,
    ):
        """
        Initialize Position Concentration Optimizer.

        Args:
            max_positions: Maximum number of positions to hold
            top_concentration_pct: Percentage of capital for top 3 positions
            min_position_pct: Minimum position size (to avoid tiny positions)
        """
        self.max_positions = max_positions
        self.top_concentration_pct = top_concentration_pct
        self.min_position_pct = min_position_pct

    def calculate_composite_score(
        self,
        confidence: float,
        quality_score: float = 0.50,
        win_rate: float = 0.50,
        trend_alignment: float = 0.50,
    ) -> float:
        """
        Calculate composite signal score for ranking.

        Weights:
        - Confidence: 40% (model output)
        - Quality Score: 25% (signal quality from Fix 22)
        - Win Rate: 20% (historical performance)
        - Trend Alignment: 15% (signal aligns with trend)
        """
        composite = (
            confidence * 0.40 +
            quality_score * 0.25 +
            win_rate * 0.20 +
            trend_alignment * 0.15
        )
        return composite

    def calculate_exponential_weights(self, n_positions: int) -> List[float]:
        """
        Calculate exponential allocation weights.

        Uses 2^(-i) weighting so top positions get much more weight.
        Example with 5 positions: [51.6%, 25.8%, 12.9%, 6.5%, 3.2%]
        """
        if n_positions == 0:
            return []

        raw_weights = [2 ** (-i) for i in range(n_positions)]
        total = sum(raw_weights)
        return [w / total for w in raw_weights]

    def optimize_allocations(
        self,
        signals: List[Dict[str, Any]],
        total_capital: float = 1.0,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Optimize capital allocation across signals.

        Args:
            signals: List of signal dicts with keys:
                - ticker: str
                - confidence: float
                - quality_score: float (optional)
                - win_rate: float (optional)
                - trend_alignment: float (optional)
            total_capital: Total capital to allocate (1.0 = 100%)

        Returns:
            (allocations, metadata)
            allocations: Dict of ticker -> allocation percentage
            metadata: Statistics about the allocation
        """
        if not signals:
            return {}, {'error': 'No signals provided'}

        # Step 1: Calculate composite scores
        scored_signals = []
        for sig in signals:
            score = self.calculate_composite_score(
                confidence=sig.get('confidence', 0.5),
                quality_score=sig.get('quality_score', 0.5),
                win_rate=sig.get('win_rate', 0.5),
                trend_alignment=sig.get('trend_alignment', 0.5),
            )
            scored_signals.append({
                **sig,
                'composite_score': score,
            })

        # Step 2: Sort by composite score (descending)
        scored_signals.sort(key=lambda x: x['composite_score'], reverse=True)

        # Step 3: Take top N positions
        selected = scored_signals[:self.max_positions]
        n_selected = len(selected)

        # Step 4: Calculate exponential weights
        weights = self.calculate_exponential_weights(n_selected)

        # Step 5: Apply minimum position size filter
        allocations = {}
        total_allocated = 0.0

        for i, (sig, weight) in enumerate(zip(selected, weights)):
            allocation = weight * total_capital

            # Ensure minimum position size
            if allocation < self.min_position_pct * total_capital:
                allocation = self.min_position_pct * total_capital

            allocations[sig['ticker']] = allocation
            total_allocated += allocation

        # Step 6: Normalize if over-allocated
        if total_allocated > total_capital:
            factor = total_capital / total_allocated
            allocations = {k: v * factor for k, v in allocations.items()}

        # Calculate metadata
        top_3_pct = sum(list(allocations.values())[:3]) / total_capital if allocations else 0
        metadata = {
            'n_signals_received': len(signals),
            'n_positions_selected': n_selected,
            'top_3_concentration': top_3_pct,
            'weights_used': weights,
            'rankings': [(s['ticker'], s['composite_score']) for s in selected],
        }

        return allocations, metadata


# ========== FIX 26: Dynamic Profit Targets (US/INTL MODEL ONLY) ==========
class DynamicProfitTargets:
    """
    Calculate dynamic profit-taking targets based on market conditions.

    FIX 26: Dynamic Profit Targets (US/INTL MODEL ONLY)

    The problem: Fixed profit-taking levels (15%/25%/40%) ignore:
    - Current volatility (should be wider in high vol)
    - Trend strength (should let winners run in strong trends)
    - Asset class characteristics (crypto more volatile than stocks)

    Solution: Dynamic ATR-based targets that adapt to conditions.
    """

    # Base targets by asset class (at 1.5% daily volatility baseline)
    BASE_TARGETS_BY_ASSET = {
        'stock': [0.08, 0.15, 0.25],      # 8%, 15%, 25%
        'etf': [0.06, 0.12, 0.20],        # 6%, 12%, 20%
        'cryptocurrency': [0.12, 0.25, 0.50],  # 12%, 25%, 50%
        'commodity': [0.10, 0.20, 0.35],  # 10%, 20%, 35%
        'forex': [0.04, 0.08, 0.15],      # 4%, 8%, 15%
        'jpy_pair': [0.03, 0.06, 0.12],   # 3%, 6%, 12%
        'crude_oil': [0.08, 0.15, 0.30],  # 8%, 15%, 30%
    }

    # Default targets if asset class not found
    DEFAULT_TARGETS = [0.08, 0.15, 0.25]

    # Baseline daily volatility for normalization
    BASELINE_VOLATILITY = 0.015  # 1.5% daily

    # Trend strength thresholds
    STRONG_TREND_THRESHOLD = 0.70
    WEAK_TREND_THRESHOLD = 0.30

    def __init__(
        self,
        partial_close_pcts: List[float] = None,
        enable_trailing_stop: bool = True,
    ):
        """
        Initialize Dynamic Profit Targets.

        Args:
            partial_close_pcts: Percentage to close at each target [50%, 75%, 100%]
            enable_trailing_stop: Whether to use trailing stop after first target
        """
        self.partial_close_pcts = partial_close_pcts or [0.50, 0.75, 1.00]
        self.enable_trailing_stop = enable_trailing_stop

    def calculate_targets(
        self,
        asset_class: str,
        volatility: float,
        trend_strength: float = 0.50,
        momentum: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Calculate dynamic profit targets.

        Args:
            asset_class: Asset class string
            volatility: Current daily volatility (decimal, e.g., 0.02 = 2%)
            trend_strength: Trend strength score (0-1, higher = stronger trend)
            momentum: Current momentum (positive = uptrend, negative = downtrend)

        Returns:
            Dict with targets and trailing stop settings
        """
        # Step 1: Get base targets for asset class
        base_targets = self.BASE_TARGETS_BY_ASSET.get(
            asset_class.lower(),
            self.DEFAULT_TARGETS
        ).copy()

        # Step 2: Adjust for volatility (ATR-based scaling)
        # Higher volatility = wider targets
        vol_multiplier = volatility / self.BASELINE_VOLATILITY
        vol_multiplier = np.clip(vol_multiplier, 0.5, 3.0)  # Cap between 0.5x and 3x

        adjusted_targets = [t * vol_multiplier for t in base_targets]

        # Step 3: Adjust for trend strength
        # Strong trends: Let winners run (wider targets)
        # Weak trends: Take profits earlier (tighter targets)
        if trend_strength > self.STRONG_TREND_THRESHOLD:
            trend_mult = 1.5  # 50% wider targets
        elif trend_strength < self.WEAK_TREND_THRESHOLD:
            trend_mult = 0.75  # 25% tighter targets
        else:
            trend_mult = 1.0  # Normal targets

        adjusted_targets = [t * trend_mult for t in adjusted_targets]

        # Step 4: Calculate trailing stop (half of first target)
        trailing_stop_pct = adjusted_targets[0] * 0.5 if self.enable_trailing_stop else None

        # Step 5: Build profit-taking levels dict
        take_profit_levels = {}
        for target, close_pct in zip(adjusted_targets, self.partial_close_pcts):
            take_profit_levels[target] = close_pct

        return {
            'targets': adjusted_targets,
            'take_profit_levels': take_profit_levels,
            'trailing_stop_pct': trailing_stop_pct,
            'vol_multiplier': vol_multiplier,
            'trend_multiplier': trend_mult,
            'asset_class': asset_class,
            'partial_take_1': adjusted_targets[0],
            'partial_take_2': adjusted_targets[1],
            'full_exit': adjusted_targets[2],
        }

    def get_recommended_action(
        self,
        current_profit_pct: float,
        targets: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Get recommended action based on current profit level.

        Args:
            current_profit_pct: Current profit percentage (decimal)
            targets: Output from calculate_targets()

        Returns:
            Dict with recommended action
        """
        profit_levels = targets['take_profit_levels']

        for target_pct, close_pct in sorted(profit_levels.items(), reverse=True):
            if current_profit_pct >= target_pct:
                return {
                    'action': 'CLOSE',
                    'close_percentage': close_pct,
                    'target_hit': target_pct,
                    'reason': f"Hit {target_pct*100:.0f}% target -> close {close_pct*100:.0f}%"
                }

        # Check trailing stop
        if targets.get('trailing_stop_pct') and current_profit_pct > 0:
            return {
                'action': 'HOLD_WITH_TRAILING',
                'trailing_stop': targets['trailing_stop_pct'],
                'reason': f"Hold with {targets['trailing_stop_pct']*100:.1f}% trailing stop"
            }

        return {
            'action': 'HOLD',
            'next_target': targets['targets'][0],
            'reason': f"Hold until {targets['targets'][0]*100:.0f}% target"
        }


# ========== FIX 27: US Market Regime Classifier (US/INTL MODEL ONLY) ==========
class USMarketRegimeClassifier:
    """
    Classify US-specific market regimes for optimized ensemble weights.

    FIX 27: US Market Regime Classifier (US/INTL MODEL ONLY)

    The problem: Generic trend classification ignores US-specific market patterns
    like FOMC weeks, earnings seasons, sector rotations, and options expiration.

    Solution: Classify into US-specific regimes and adjust ensemble weights accordingly.

    Regimes:
    1. bull_momentum - Strong uptrend with momentum
    2. bull_consolidation - Uptrend but ranging/consolidating
    3. bear_momentum - Strong downtrend
    4. bear_rally - Downtrend with counter-trend rally
    5. fomc_week - FOMC meeting week (high uncertainty)
    6. earnings_season - Major earnings season
    7. sector_rotation - Sector rotation in progress
    8. opex_week - Options expiration week
    """

    # US-specific ensemble weights by regime
    US_ADAPTIVE_ENSEMBLE_WEIGHTS = {
        # Bull market regimes
        'bull_momentum': {'catboost': 0.35, 'lstm': 0.65},      # LSTM excels in trends
        'bull_consolidation': {'catboost': 0.60, 'lstm': 0.40}, # CatBoost better in ranges

        # Bear market regimes
        'bear_momentum': {'catboost': 0.30, 'lstm': 0.70},      # LSTM for momentum shorts
        'bear_rally': {'catboost': 0.75, 'lstm': 0.25},         # CatBoost for mean reversion

        # Special US regimes
        'fomc_week': {'catboost': 0.90, 'lstm': 0.10},          # CatBoost with news features
        'earnings_season': {'catboost': 0.80, 'lstm': 0.20},    # Earnings patterns
        'sector_rotation': {'catboost': 0.70, 'lstm': 0.30},    # Sector analysis
        'opex_week': {'catboost': 0.65, 'lstm': 0.35},          # Options effects

        # Default
        'neutral': {'catboost': 0.70, 'lstm': 0.30},
    }

    # VIX thresholds for regime detection
    VIX_THRESHOLDS = {
        'low': 15,      # Below 15 = low volatility
        'normal': 20,   # 15-20 = normal
        'elevated': 25, # 20-25 = elevated
        'high': 30,     # 25-30 = high
        'extreme': 40,  # Above 30 = extreme
    }

    def __init__(self):
        self.current_regime = 'neutral'
        self.regime_history = deque(maxlen=20)

    def classify_regime(
        self,
        spy_returns_20d: float = 0.0,
        spy_returns_5d: float = 0.0,
        vix_level: float = 20.0,
        is_fomc_week: bool = False,
        is_earnings_season: bool = False,
        is_opex_week: bool = False,
        sector_dispersion: float = 0.0,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Classify current US market regime.

        Args:
            spy_returns_20d: 20-day SPY returns (decimal)
            spy_returns_5d: 5-day SPY returns (decimal)
            vix_level: Current VIX level
            is_fomc_week: True if FOMC meeting this week
            is_earnings_season: True if in major earnings season
            is_opex_week: True if options expiration week
            sector_dispersion: Cross-sector return dispersion

        Returns:
            (regime, weights) tuple
        """
        # Priority 1: Event-driven regimes (override trend-based)
        if is_fomc_week:
            regime = 'fomc_week'
        elif is_opex_week:
            regime = 'opex_week'
        elif is_earnings_season:
            regime = 'earnings_season'
        elif sector_dispersion > 0.03:  # >3% sector dispersion
            regime = 'sector_rotation'
        # Priority 2: Trend-based regimes
        elif spy_returns_20d > 0.05:  # Bull market (>5% in 20 days)
            if spy_returns_5d > 0.02:  # Strong recent momentum
                regime = 'bull_momentum'
            else:
                regime = 'bull_consolidation'
        elif spy_returns_20d < -0.05:  # Bear market (<-5% in 20 days)
            if spy_returns_5d < -0.02:  # Strong downward momentum
                regime = 'bear_momentum'
            else:
                regime = 'bear_rally'
        else:
            regime = 'neutral'

        # VIX adjustment: In extreme VIX, shift more to CatBoost (less momentum-following)
        weights = self.US_ADAPTIVE_ENSEMBLE_WEIGHTS.get(regime, self.US_ADAPTIVE_ENSEMBLE_WEIGHTS['neutral']).copy()

        if vix_level > self.VIX_THRESHOLDS['extreme']:
            # Extreme VIX: More CatBoost (conservative)
            weights['catboost'] = min(weights['catboost'] + 0.15, 0.95)
            weights['lstm'] = max(weights['lstm'] - 0.15, 0.05)
        elif vix_level > self.VIX_THRESHOLDS['high']:
            weights['catboost'] = min(weights['catboost'] + 0.10, 0.90)
            weights['lstm'] = max(weights['lstm'] - 0.10, 0.10)

        self.current_regime = regime
        self.regime_history.append(regime)

        return regime, weights

    def get_regime_position_multiplier(self, regime: str, signal_type: str) -> float:
        """
        Get position size multiplier based on regime.

        Args:
            regime: Current market regime
            signal_type: 'BUY' or 'SELL'

        Returns:
            Position multiplier (0.5 to 1.5)
        """
        multipliers = {
            'bull_momentum': {'BUY': 1.3, 'SELL': 0.6},
            'bull_consolidation': {'BUY': 1.0, 'SELL': 0.8},
            'bear_momentum': {'BUY': 0.6, 'SELL': 1.2},
            'bear_rally': {'BUY': 0.8, 'SELL': 0.7},
            'fomc_week': {'BUY': 0.5, 'SELL': 0.5},      # Reduce all during FOMC
            'earnings_season': {'BUY': 0.8, 'SELL': 0.6},
            'sector_rotation': {'BUY': 1.0, 'SELL': 0.9},
            'opex_week': {'BUY': 0.7, 'SELL': 0.6},      # Reduce during OpEx
            'neutral': {'BUY': 1.0, 'SELL': 1.0},
        }
        return multipliers.get(regime, multipliers['neutral']).get(signal_type, 1.0)


# ========== FIX 28: Sector Momentum Integration (US/INTL MODEL ONLY) ==========
class SectorMomentumAnalyzer:
    """
    US sector momentum analysis for enhanced signal quality.

    FIX 28: Sector Momentum Integration (US/INTL MODEL ONLY)

    The problem: US markets are heavily sector-driven. Ignoring sector
    momentum misses key alpha opportunities.

    Solution: Add sector ETF momentum, relative strength, and rotation detection.
    """

    # SPDR Sector ETFs
    SECTOR_ETFS = {
        'XLK': 'Technology',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLE': 'Energy',
        'XLF': 'Financials',
        'XLV': 'Healthcare',
        'XLI': 'Industrials',
        'XLRE': 'Real Estate',
        'XLU': 'Utilities',
        'XLC': 'Communications',
        'XLB': 'Materials',
    }

    # Stock to sector mapping (simplified - major stocks)
    STOCK_SECTOR_MAP = {
        # Technology
        'AAPL': 'XLK', 'MSFT': 'XLK', 'NVDA': 'XLK', 'GOOGL': 'XLK', 'GOOG': 'XLK',
        'META': 'XLC', 'AMZN': 'XLY', 'TSLA': 'XLY', 'AMD': 'XLK', 'INTC': 'XLK',
        # Financials
        'JPM': 'XLF', 'BAC': 'XLF', 'WFC': 'XLF', 'GS': 'XLF', 'MS': 'XLF',
        # Healthcare
        'JNJ': 'XLV', 'UNH': 'XLV', 'PFE': 'XLV', 'MRK': 'XLV', 'ABBV': 'XLV',
        # Energy
        'XOM': 'XLE', 'CVX': 'XLE', 'COP': 'XLE', 'SLB': 'XLE', 'EOG': 'XLE',
        # Consumer
        'WMT': 'XLP', 'PG': 'XLP', 'KO': 'XLP', 'PEP': 'XLP', 'COST': 'XLP',
        'HD': 'XLY', 'MCD': 'XLY', 'NKE': 'XLY', 'SBUX': 'XLY', 'TGT': 'XLY',
        # Industrials
        'CAT': 'XLI', 'BA': 'XLI', 'HON': 'XLI', 'UPS': 'XLI', 'GE': 'XLI',
        # Utilities
        'NEE': 'XLU', 'DUK': 'XLU', 'SO': 'XLU', 'D': 'XLU', 'AEP': 'XLU',
    }

    # Sector leader boosts
    SECTOR_LEADER_BOOSTS = {
        'XLK': 1.3,  # Technology - historically strong
        'XLV': 1.2,  # Healthcare - defensive with growth
        'XLI': 1.1,  # Industrials - economic indicator
        'XLY': 1.1,  # Consumer Discretionary - economic indicator
        'XLF': 1.0,  # Financials - rate sensitive
        'XLE': 0.9,  # Energy - volatile
        'XLU': 0.9,  # Utilities - defensive, low growth
        'XLRE': 0.9, # Real Estate - rate sensitive
    }

    def __init__(self):
        self.sector_momentum = {}  # sector -> momentum score
        self.sector_ranking = []   # ordered list of sectors by momentum

    def get_stock_sector(self, ticker: str) -> Optional[str]:
        """Get sector ETF for a given stock."""
        return self.STOCK_SECTOR_MAP.get(ticker.upper())

    def calculate_relative_strength(
        self,
        ticker_returns: float,
        sector_returns: float,
    ) -> float:
        """
        Calculate relative strength ratio.

        RS Ratio = ticker_returns / sector_returns
        Ratio > 1 = outperforming sector
        Ratio < 1 = underperforming sector
        """
        if sector_returns == 0:
            return 1.0
        return ticker_returns / sector_returns if sector_returns != 0 else 1.0

    def get_sector_momentum_score(
        self,
        ticker: str,
        ticker_returns_20d: float,
        sector_returns_20d: float,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Get sector-adjusted momentum score for a ticker.

        Args:
            ticker: Stock ticker
            ticker_returns_20d: 20-day ticker returns
            sector_returns_20d: 20-day sector ETF returns

        Returns:
            (momentum_score, metadata)
        """
        sector = self.get_stock_sector(ticker)

        if sector is None:
            # Not in our mapping - use neutral score
            return 0.5, {'sector': None, 'rs_ratio': 1.0, 'sector_boost': 1.0}

        rs_ratio = self.calculate_relative_strength(ticker_returns_20d, sector_returns_20d)
        sector_boost = self.SECTOR_LEADER_BOOSTS.get(sector, 1.0)

        # Momentum score: combination of absolute and relative performance
        abs_score = np.tanh(ticker_returns_20d * 10)  # -1 to 1
        rel_score = np.tanh((rs_ratio - 1) * 5)       # -1 to 1

        momentum_score = 0.6 * abs_score + 0.4 * rel_score

        return momentum_score, {
            'sector': sector,
            'sector_name': self.SECTOR_ETFS.get(sector, 'Unknown'),
            'rs_ratio': rs_ratio,
            'sector_boost': sector_boost,
            'abs_momentum': abs_score,
            'rel_momentum': rel_score,
        }

    def get_position_adjustment(
        self,
        ticker: str,
        signal_type: str,
        ticker_returns_20d: float = 0.0,
        sector_returns_20d: float = 0.0,
    ) -> Tuple[float, str]:
        """
        Get position size adjustment based on sector momentum.

        Returns:
            (multiplier, reason)
        """
        sector = self.get_stock_sector(ticker)
        if sector is None:
            return 1.0, "No sector mapping"

        rs_ratio = self.calculate_relative_strength(ticker_returns_20d, sector_returns_20d)
        sector_boost = self.SECTOR_LEADER_BOOSTS.get(sector, 1.0)

        # For BUY: Boost outperformers in strong sectors
        if signal_type == 'BUY':
            if rs_ratio > 1.2:  # Strong outperformer
                multiplier = 1.2 * sector_boost
                reason = f"Strong sector outperformer (RS={rs_ratio:.2f})"
            elif rs_ratio > 1.0:
                multiplier = 1.1 * sector_boost
                reason = f"Sector outperformer (RS={rs_ratio:.2f})"
            elif rs_ratio < 0.8:
                multiplier = 0.8
                reason = f"Sector underperformer (RS={rs_ratio:.2f})"
            else:
                multiplier = 1.0 * sector_boost
                reason = f"Neutral sector (RS={rs_ratio:.2f})"
        else:  # SELL
            if rs_ratio < 0.8:  # Weak stock in sector
                multiplier = 1.15
                reason = f"Weak stock for shorting (RS={rs_ratio:.2f})"
            elif rs_ratio > 1.2:  # Strong stock - risky to short
                multiplier = 0.7
                reason = f"Strong stock - risky short (RS={rs_ratio:.2f})"
            else:
                multiplier = 1.0
                reason = f"Neutral for shorting (RS={rs_ratio:.2f})"

        return multiplier, reason


# ========== FIX 29: Earnings Season Optimizer (US/INTL MODEL ONLY) ==========
class EarningsSeasonOptimizer:
    """
    Optimize signals around US earnings season.

    FIX 29: Earnings Season Optimizer (US/INTL MODEL ONLY)

    The problem: Earnings announcements cause significant volatility
    and predictable patterns (pre-earnings drift, PEAD).

    Solution: Adjust signals based on proximity to earnings dates.
    """

    # Earnings season months (Jan, Apr, Jul, Oct are peak)
    EARNINGS_SEASON_MONTHS = {1, 4, 7, 10}

    # Major earnings companies (simplified list)
    MAJOR_EARNINGS_TICKERS = {
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'JPM',
        'BAC', 'JNJ', 'UNH', 'XOM', 'WMT', 'PG', 'HD', 'MA', 'V', 'DIS',
    }

    def __init__(self):
        self.earnings_calendar = {}  # ticker -> next earnings date

    def is_earnings_season(self, date: datetime = None) -> bool:
        """Check if we're in a major earnings season."""
        if date is None:
            date = datetime.now()
        # First 3 weeks of Jan, Apr, Jul, Oct
        return date.month in self.EARNINGS_SEASON_MONTHS and date.day <= 21

    def get_days_to_earnings(self, ticker: str, earnings_date: datetime = None) -> Optional[int]:
        """
        Get days until next earnings for a ticker.

        Returns None if unknown.
        """
        if earnings_date is None:
            return None
        return (earnings_date - datetime.now()).days

    def optimize_for_earnings(
        self,
        ticker: str,
        signal_type: str,
        confidence: float,
        days_to_earnings: Optional[int] = None,
        is_earnings_season: bool = False,
    ) -> Dict[str, Any]:
        """
        Adjust signal for earnings proximity.

        Args:
            ticker: Stock ticker
            signal_type: 'BUY' or 'SELL'
            confidence: Original confidence
            days_to_earnings: Days until earnings (None if unknown)
            is_earnings_season: Whether we're in earnings season

        Returns:
            Dict with adjustments
        """
        adjustments = {
            'position_multiplier': 1.0,
            'stop_loss_adjustment': 1.0,
            'confidence_adjustment': 0.0,
            'reason': 'Normal conditions',
            'blocked': False,
        }

        # If we know the earnings date
        if days_to_earnings is not None:
            if -2 <= days_to_earnings <= 2:  # Earnings week
                if signal_type == 'BUY':
                    # Reduce position size around earnings (high volatility)
                    adjustments['position_multiplier'] = 0.7
                    adjustments['stop_loss_adjustment'] = 0.8  # Tighter stops
                    adjustments['reason'] = 'Earnings week - reduced BUY size'
                else:  # SELL
                    # Avoid shorts around earnings (unpredictable moves)
                    adjustments['position_multiplier'] = 0.0
                    adjustments['blocked'] = True
                    adjustments['reason'] = 'Earnings week - SELL blocked'

            elif 3 <= days_to_earnings <= 10:  # Pre-earnings period
                # Historical pattern: positive drift before earnings
                if signal_type == 'BUY':
                    adjustments['position_multiplier'] = 1.2
                    adjustments['confidence_adjustment'] = 0.05
                    adjustments['reason'] = 'Pre-earnings drift - boosted BUY'
                else:
                    adjustments['position_multiplier'] = 0.7
                    adjustments['reason'] = 'Pre-earnings drift - reduced SELL'

            elif -10 <= days_to_earnings < -2:  # Post-earnings period
                # Post-earnings announcement drift (PEAD)
                adjustments['reason'] = 'Post-earnings - follow the move'
                # PEAD suggests continuing in direction of surprise
                # (would need surprise data to fully implement)

        # General earnings season adjustments
        elif is_earnings_season and ticker.upper() in self.MAJOR_EARNINGS_TICKERS:
            adjustments['position_multiplier'] = 0.85
            adjustments['stop_loss_adjustment'] = 0.9
            adjustments['reason'] = 'Earnings season - major company'

        return adjustments


# ========== FIX 30: FOMC & Economic Calendar (US/INTL MODEL ONLY) ==========
class FOMCOptimizer:
    """
    Optimize signals around FOMC meetings and economic events.

    FIX 30: FOMC & Economic Calendar (US/INTL MODEL ONLY)

    The problem: FOMC meetings and major economic reports cause
    significant market volatility and predictable patterns.

    Solution: Adjust signals based on proximity to FOMC and economic events.
    """

    # Rate-sensitive sectors (affected more by Fed policy)
    RATE_SENSITIVE_SECTORS = {
        'XLF': 'Financials',    # Banks benefit/suffer from rate changes
        'XLU': 'Utilities',     # Dividend stocks affected by rates
        'XLRE': 'Real Estate',  # REITs highly rate sensitive
        'TLT': 'Long Bonds',    # Treasury bonds
    }

    # Growth-sensitive sectors (affected by rate outlook)
    GROWTH_SENSITIVE_SECTORS = {
        'XLK': 'Technology',
        'XLY': 'Consumer Discretionary',
        'XLC': 'Communications',
    }

    # FOMC meeting months (typically 8 meetings per year)
    FOMC_MONTHS = {1, 3, 5, 6, 7, 9, 11, 12}

    def __init__(self):
        self.next_fomc_date = None
        self.rate_expectation = 'hold'  # 'hike', 'cut', 'hold'

    def get_days_to_fomc(self, fomc_date: datetime = None) -> Optional[int]:
        """Get days until next FOMC meeting."""
        if fomc_date is None:
            return None
        return (fomc_date - datetime.now()).days

    def is_rate_sensitive(self, ticker: str) -> bool:
        """Check if ticker is in a rate-sensitive sector."""
        # Check if it's a sector ETF
        if ticker.upper() in self.RATE_SENSITIVE_SECTORS:
            return True
        # Could expand with stock-to-sector mapping
        return False

    def is_growth_sensitive(self, ticker: str) -> bool:
        """Check if ticker is in a growth-sensitive sector."""
        if ticker.upper() in self.GROWTH_SENSITIVE_SECTORS:
            return True
        return False

    def adjust_for_fomc(
        self,
        ticker: str,
        signal_type: str,
        days_to_fomc: Optional[int] = None,
        rate_expectation: str = 'hold',
    ) -> Dict[str, Any]:
        """
        Adjust signal for FOMC proximity.

        Args:
            ticker: Stock ticker
            signal_type: 'BUY' or 'SELL'
            days_to_fomc: Days until FOMC (None if unknown)
            rate_expectation: 'hike', 'cut', or 'hold'

        Returns:
            Dict with adjustments
        """
        adjustments = {
            'position_multiplier': 1.0,
            'stop_loss_adjustment': 1.0,
            'reason': 'Normal conditions',
        }

        if days_to_fomc is None:
            return adjustments

        is_rate_sens = self.is_rate_sensitive(ticker)
        is_growth_sens = self.is_growth_sensitive(ticker)

        if -1 <= days_to_fomc <= 2:  # FOMC week
            # Reduce all positions during FOMC week (high volatility)
            adjustments['position_multiplier'] = 0.5
            adjustments['reason'] = 'FOMC week - reduced exposure'

            # Extra conservative for rate-sensitive sectors
            if is_rate_sens:
                adjustments['position_multiplier'] = 0.3
                adjustments['stop_loss_adjustment'] = 0.6
                adjustments['reason'] = 'FOMC week - rate sensitive sector'

        elif 3 <= days_to_fomc <= 7:  # Week before FOMC
            # Historical: positive drift before FOMC
            if signal_type == 'BUY':
                adjustments['position_multiplier'] = 1.15
                adjustments['reason'] = 'Pre-FOMC drift - boosted BUY'
            else:
                adjustments['position_multiplier'] = 0.8
                adjustments['reason'] = 'Pre-FOMC drift - reduced SELL'

        # Rate expectation adjustments
        if rate_expectation == 'hike' and is_growth_sens:
            # Rate hike hurts growth stocks
            if signal_type == 'BUY':
                adjustments['position_multiplier'] *= 0.8
                adjustments['reason'] += ' | Rate hike expected - growth sensitive'
            else:
                adjustments['position_multiplier'] *= 1.1

        elif rate_expectation == 'cut' and is_growth_sens:
            # Rate cut helps growth stocks
            if signal_type == 'BUY':
                adjustments['position_multiplier'] *= 1.15
                adjustments['reason'] += ' | Rate cut expected - growth boost'

        return adjustments


# ========== FIX 31: Options Expiration Optimizer (US/INTL MODEL ONLY) ==========
class OpExOptimizer:
    """
    Handle US options expiration effects.

    FIX 31: Options Expiration Optimizer (US/INTL MODEL ONLY)

    The problem: Monthly options expiration (3rd Friday) causes
    significant gamma hedging flows and volatility patterns.

    Solution: Adjust signals around OpEx dates.

    Effects:
    - Gamma pinning: Prices tend to gravitate toward max pain
    - Volatility crush: IV drops after expiration
    - Gamma hedging: Large flows from market makers
    """

    def __init__(self):
        pass

    def get_days_to_opex(self, date: datetime = None) -> int:
        """
        Calculate days to next monthly options expiration (3rd Friday).

        Args:
            date: Current date (default: now)

        Returns:
            Days until OpEx
        """
        if date is None:
            date = datetime.now()

        # Find 3rd Friday of current month
        year, month = date.year, date.month

        # First day of month
        first_day = datetime(year, month, 1)

        # Days until first Friday
        days_to_friday = (4 - first_day.weekday()) % 7

        # Third Friday = first Friday + 14 days
        third_friday = first_day + timedelta(days=days_to_friday + 14)

        # If already past this month's OpEx, get next month's
        if date > third_friday:
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
            first_day = datetime(year, month, 1)
            days_to_friday = (4 - first_day.weekday()) % 7
            third_friday = first_day + timedelta(days=days_to_friday + 14)

        return (third_friday - date).days

    def is_opex_week(self, date: datetime = None) -> bool:
        """Check if we're in OpEx week (Monday-Friday of expiration week)."""
        days_to_opex = self.get_days_to_opex(date)
        return 0 <= days_to_opex <= 4

    def adjust_for_opex(
        self,
        ticker: str,
        signal_type: str,
        days_to_opex: int,
        is_high_gamma_stock: bool = False,
    ) -> Dict[str, Any]:
        """
        Adjust signal for options expiration effects.

        Args:
            ticker: Stock ticker
            signal_type: 'BUY' or 'SELL'
            days_to_opex: Days until options expiration
            is_high_gamma_stock: True for mega-caps with heavy options activity

        Returns:
            Dict with adjustments
        """
        adjustments = {
            'position_multiplier': 1.0,
            'stop_loss_adjustment': 1.0,
            'reason': 'Normal conditions',
            'avoid_entry': False,
        }

        if -2 <= days_to_opex <= 2:  # OpEx window
            # Reduce position sizes due to gamma hedging flows
            adjustments['position_multiplier'] = 0.6
            adjustments['reason'] = 'OpEx week - gamma hedging'

            # Wider stops (gamma-induced volatility)
            adjustments['stop_loss_adjustment'] = 1.3

            # High gamma stocks (AAPL, TSLA, SPY, etc.) need extra caution
            if is_high_gamma_stock:
                adjustments['position_multiplier'] = 0.4
                adjustments['reason'] = 'OpEx week - high gamma stock'

            # Avoid new positions on expiration Friday
            if days_to_opex == 0:
                adjustments['avoid_entry'] = True
                adjustments['reason'] = 'OpEx Friday - avoid new entries'

        elif 3 <= days_to_opex <= 5:  # Days before OpEx
            # Slightly reduce exposure as we approach OpEx
            adjustments['position_multiplier'] = 0.85
            adjustments['reason'] = 'Approaching OpEx'

        return adjustments

    def get_high_gamma_stocks(self) -> set:
        """Return set of stocks with heavy options activity."""
        return {
            'SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'AMZN',
            'META', 'GOOGL', 'MSFT', 'NFLX', 'COIN', 'GME', 'AMC',
        }


# ========== FIX 32: Market Internals Integration (US/INTL MODEL ONLY) ==========
class USMarketInternals:
    """
    US market internal indicators for market health assessment.

    FIX 32: Market Internals Integration (US/INTL MODEL ONLY)

    The problem: Ignoring market breadth misses important signals
    about overall market health.

    Solution: Integrate advance-decline, new highs/lows, TRIN, McClellan.
    """

    def __init__(self):
        self.ad_ratio_history = deque(maxlen=20)
        self.nhnl_history = deque(maxlen=20)
        self.trin_history = deque(maxlen=10)

    def calculate_ad_ratio(
        self,
        advances: int,
        declines: int,
    ) -> float:
        """
        Calculate Advance-Decline ratio.

        AD Ratio = Advances / Declines
        > 1.0 = More stocks advancing (bullish)
        < 1.0 = More stocks declining (bearish)
        """
        if declines == 0:
            return 2.0 if advances > 0 else 1.0
        ratio = advances / declines
        self.ad_ratio_history.append(ratio)
        return ratio

    def calculate_nhnl_ratio(
        self,
        new_highs: int,
        new_lows: int,
    ) -> float:
        """
        Calculate New Highs / New Lows ratio.

        > 1.0 = Healthy market (more new highs)
        < 1.0 = Weak market (more new lows)
        < 0.5 = Very weak market
        """
        if new_lows == 0:
            return 10.0 if new_highs > 0 else 1.0
        ratio = new_highs / new_lows
        self.nhnl_history.append(ratio)
        return ratio

    def calculate_trin(
        self,
        advances: int,
        declines: int,
        advancing_volume: float,
        declining_volume: float,
    ) -> float:
        """
        Calculate TRIN (Arms Index).

        TRIN = (Advances/Declines) / (Advancing Volume/Declining Volume)

        < 1.0 = Bullish (volume confirming advances)
        > 1.0 = Bearish (volume confirming declines)
        > 2.0 = Extreme bearish (potential reversal)
        < 0.5 = Extreme bullish (potential reversal)
        """
        if declines == 0 or declining_volume == 0:
            return 1.0

        ad_ratio = advances / declines
        vol_ratio = advancing_volume / declining_volume if declining_volume > 0 else 1.0

        trin = ad_ratio / vol_ratio if vol_ratio > 0 else 1.0
        self.trin_history.append(trin)
        return trin

    def calculate_mcclellan_oscillator(
        self,
        ad_diff_19ema: float,
        ad_diff_39ema: float,
    ) -> float:
        """
        Calculate McClellan Oscillator.

        McClellan = 19-day EMA of (Advances - Declines) - 39-day EMA of (Advances - Declines)

        > 0 = Bullish momentum
        < 0 = Bearish momentum
        > 100 = Overbought
        < -100 = Oversold
        """
        return ad_diff_19ema - ad_diff_39ema

    def get_market_health_score(
        self,
        ad_ratio: float = 1.0,
        nhnl_ratio: float = 1.0,
        trin: float = 1.0,
        mcclellan: float = 0.0,
    ) -> Tuple[float, str]:
        """
        Calculate overall market health score.

        Returns:
            (score, interpretation)
            score: -1 (very bearish) to +1 (very bullish)
        """
        scores = []

        # AD Ratio score
        ad_score = np.clip((ad_ratio - 1.0) / 0.5, -1, 1)
        scores.append(ad_score * 0.25)

        # NHNL Ratio score
        nhnl_score = np.clip((nhnl_ratio - 1.0) / 2.0, -1, 1)
        scores.append(nhnl_score * 0.25)

        # TRIN score (inverted - lower is better)
        trin_score = np.clip((1.0 - trin) / 0.5, -1, 1)
        scores.append(trin_score * 0.25)

        # McClellan score
        mcclellan_score = np.clip(mcclellan / 100, -1, 1)
        scores.append(mcclellan_score * 0.25)

        total_score = sum(scores)

        if total_score > 0.5:
            interpretation = 'Strong bullish breadth'
        elif total_score > 0.2:
            interpretation = 'Moderately bullish breadth'
        elif total_score > -0.2:
            interpretation = 'Neutral breadth'
        elif total_score > -0.5:
            interpretation = 'Moderately bearish breadth'
        else:
            interpretation = 'Strong bearish breadth'

        return total_score, interpretation

    def get_position_adjustment(
        self,
        signal_type: str,
        market_health_score: float,
    ) -> Tuple[float, str]:
        """
        Get position adjustment based on market health.

        Returns:
            (multiplier, reason)
        """
        if signal_type == 'BUY':
            if market_health_score > 0.5:
                return 1.2, 'Strong bullish breadth - boost BUY'
            elif market_health_score > 0.2:
                return 1.1, 'Moderately bullish breadth'
            elif market_health_score < -0.5:
                return 0.7, 'Weak breadth - reduce BUY'
            elif market_health_score < -0.2:
                return 0.85, 'Moderately weak breadth'
        else:  # SELL
            if market_health_score < -0.5:
                return 1.15, 'Weak breadth - boost SELL'
            elif market_health_score < -0.2:
                return 1.05, 'Moderately bearish breadth'
            elif market_health_score > 0.5:
                return 0.7, 'Strong breadth - reduce SELL'
            elif market_health_score > 0.2:
                return 0.85, 'Healthy breadth - caution on SELL'

        return 1.0, 'Neutral breadth'


# ========== FIX 33: US-Specific Risk Models (US/INTL MODEL ONLY) ==========
class USRiskModel:
    """
    US-specific risk models for portfolio optimization.

    FIX 33: US-Specific Risk Models (US/INTL MODEL ONLY)

    The problem: Generic risk models ignore US-specific factors
    like sector concentration, factor exposure, and style drift.

    Solution: Implement US-specific risk constraints and optimization.
    """

    # Maximum sector concentration
    MAX_SECTOR_CONCENTRATION = 0.35  # 35% max in any sector

    # Factor exposure limits
    FACTOR_LIMITS = {
        'momentum': (-0.5, 1.5),    # Allow positive momentum tilt
        'value': (-0.5, 0.5),       # Neutral value
        'size': (-0.5, 0.5),        # Neutral size
        'volatility': (-0.5, 0.3),  # Slight low-vol tilt
        'quality': (0.0, 1.0),      # Positive quality bias
    }

    def __init__(self):
        self.sector_allocations = {}  # sector -> allocation
        self.factor_exposures = {}    # factor -> exposure

    def check_sector_concentration(
        self,
        ticker: str,
        proposed_allocation: float,
        current_allocations: Dict[str, float],
        sector_mapper: SectorMomentumAnalyzer = None,
    ) -> Tuple[bool, float, str]:
        """
        Check if adding a position would exceed sector concentration limits.

        Args:
            ticker: Stock ticker
            proposed_allocation: Proposed allocation (as decimal)
            current_allocations: Current portfolio allocations by ticker
            sector_mapper: SectorMomentumAnalyzer for sector lookup

        Returns:
            (allowed, adjusted_allocation, reason)
        """
        if sector_mapper is None:
            return True, proposed_allocation, 'No sector mapper'

        sector = sector_mapper.get_stock_sector(ticker)
        if sector is None:
            return True, proposed_allocation, 'Unknown sector'

        # Calculate current sector allocation
        current_sector_alloc = 0.0
        for t, alloc in current_allocations.items():
            t_sector = sector_mapper.get_stock_sector(t)
            if t_sector == sector:
                current_sector_alloc += alloc

        new_sector_alloc = current_sector_alloc + proposed_allocation

        if new_sector_alloc > self.MAX_SECTOR_CONCENTRATION:
            # Reduce to fit within limit
            max_allowed = self.MAX_SECTOR_CONCENTRATION - current_sector_alloc
            if max_allowed <= 0:
                return False, 0.0, f'Sector {sector} at max concentration'
            return True, max_allowed, f'Reduced to fit sector limit ({sector})'

        return True, proposed_allocation, f'Within sector limits ({sector})'

    def calculate_portfolio_risk_score(
        self,
        allocations: Dict[str, float],
        volatilities: Dict[str, float] = None,
        correlations: Dict[Tuple[str, str], float] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate portfolio risk score.

        Args:
            allocations: ticker -> allocation
            volatilities: ticker -> annualized volatility
            correlations: (ticker1, ticker2) -> correlation

        Returns:
            (risk_score, breakdown)
        """
        if volatilities is None:
            volatilities = {t: 0.25 for t in allocations}  # Default 25% vol

        # Calculate weighted volatility
        weighted_vol = sum(
            alloc * volatilities.get(ticker, 0.25)
            for ticker, alloc in allocations.items()
        )

        # Concentration penalty (Herfindahl index)
        herfindahl = sum(alloc ** 2 for alloc in allocations.values())
        concentration_penalty = herfindahl * 0.5  # Scale factor

        # Risk score
        risk_score = weighted_vol + concentration_penalty

        return risk_score, {
            'weighted_volatility': weighted_vol,
            'herfindahl_index': herfindahl,
            'concentration_penalty': concentration_penalty,
            'num_positions': len(allocations),
        }

    def get_risk_adjusted_allocation(
        self,
        ticker: str,
        base_allocation: float,
        volatility: float = 0.25,
        market_health: float = 0.0,
    ) -> Tuple[float, str]:
        """
        Get risk-adjusted allocation.

        Args:
            ticker: Stock ticker
            base_allocation: Base allocation from optimizer
            volatility: Stock annualized volatility
            market_health: Market health score (-1 to 1)

        Returns:
            (adjusted_allocation, reason)
        """
        # Volatility scaling (reduce allocation for high-vol stocks)
        vol_factor = 0.20 / max(volatility, 0.10)  # Normalize to 20% baseline
        vol_factor = np.clip(vol_factor, 0.5, 1.5)

        # Market health adjustment
        if market_health < -0.3:
            health_factor = 0.8  # Reduce in weak markets
        elif market_health > 0.3:
            health_factor = 1.1  # Slightly increase in strong markets
        else:
            health_factor = 1.0

        adjusted = base_allocation * vol_factor * health_factor
        adjusted = np.clip(adjusted, 0.01, 0.25)  # 1% min, 25% max per position

        reason = f"Vol factor: {vol_factor:.2f}, Health factor: {health_factor:.2f}"
        return adjusted, reason


# ========== FIX 34: Intraday Momentum Timing (US/INTL MODEL ONLY) ==========
class IntradayMomentumOptimizer:
    """
    US markets show distinct intraday patterns.

    FIX 34: Intraday Momentum Timing

    Key patterns:
    - Opening hour (9:30-10:30): High volatility, trend establishment
    - Midday lull (12:00-13:30): Lower volume, mean reversion
    - Power hour (15:00-16:00): Institutional rebalancing, momentum continuation

    Position entry timing can boost returns by 15-25%.
    """

    # Intraday timing windows (Eastern Time)
    INTRADAY_ENTRY_TIMING = {
        'opening_range_breakout': {
            'start_hour': 10, 'start_min': 15,
            'end_hour': 10, 'end_min': 45,
            'strategy': 'Wait for 45-min opening range, enter on breakout',
            'buy_boost': 1.15,
            'sell_boost': 1.10,
        },
        'midday_dip': {
            'start_hour': 12, 'start_min': 30,
            'end_hour': 13, 'end_min': 30,
            'strategy': 'Buy weakness during lunchtime lull',
            'buy_boost': 1.10,
            'sell_boost': 0.85,  # Avoid selling in low volume
        },
        'power_hour_momentum': {
            'start_hour': 15, 'start_min': 0,
            'end_hour': 15, 'end_min': 30,
            'strategy': 'Ride institutional end-of-day flows',
            'buy_boost': 1.20,
            'sell_boost': 1.15,
        },
        'close_avoidance': {
            'start_hour': 15, 'start_min': 45,
            'end_hour': 16, 'end_min': 0,
            'strategy': 'Avoid new entries near close',
            'buy_boost': 0.70,
            'sell_boost': 0.70,
        },
    }

    # Opening volatility window - avoid immediate open
    OPENING_VOLATILITY_WINDOW = {
        'start_hour': 9, 'start_min': 30,
        'end_hour': 9, 'end_min': 45,
        'position_mult': 0.60,  # Reduce position in first 15 mins
    }

    def __init__(self):
        pass

    def get_current_window(self, current_time: datetime = None) -> Optional[str]:
        """
        Determine which intraday window we're in.

        Args:
            current_time: Current datetime (defaults to now)

        Returns:
            Window name or None if outside special windows
        """
        if current_time is None:
            current_time = datetime.now()

        hour = current_time.hour
        minute = current_time.minute
        time_mins = hour * 60 + minute

        # Check opening volatility first
        open_start = self.OPENING_VOLATILITY_WINDOW['start_hour'] * 60 + self.OPENING_VOLATILITY_WINDOW['start_min']
        open_end = self.OPENING_VOLATILITY_WINDOW['end_hour'] * 60 + self.OPENING_VOLATILITY_WINDOW['end_min']
        if open_start <= time_mins <= open_end:
            return 'opening_volatility'

        # Check other windows
        for window_name, window in self.INTRADAY_ENTRY_TIMING.items():
            start_mins = window['start_hour'] * 60 + window['start_min']
            end_mins = window['end_hour'] * 60 + window['end_min']
            if start_mins <= time_mins <= end_mins:
                return window_name

        return None

    def get_timing_boost(
        self,
        signal_type: str,
        current_time: datetime = None,
    ) -> Tuple[float, str]:
        """
        Get position multiplier based on time of day.

        Args:
            signal_type: 'BUY' or 'SELL'
            current_time: Current datetime

        Returns:
            (multiplier, reason)
        """
        window = self.get_current_window(current_time)

        if window is None:
            return 1.0, "Normal trading hours"

        if window == 'opening_volatility':
            return self.OPENING_VOLATILITY_WINDOW['position_mult'], "Opening volatility - reduced position"

        timing = self.INTRADAY_ENTRY_TIMING[window]
        if signal_type.upper() == 'BUY':
            return timing['buy_boost'], f"{window}: {timing['strategy']}"
        else:
            return timing['sell_boost'], f"{window}: {timing['strategy']}"

    def is_optimal_entry_time(
        self,
        signal_type: str,
        current_time: datetime = None,
    ) -> Tuple[bool, str]:
        """
        Check if current time is optimal for entry.

        Returns:
            (is_optimal, reason)
        """
        window = self.get_current_window(current_time)

        if window == 'opening_volatility':
            return False, "Wait for opening range to establish (after 9:45)"

        if window == 'close_avoidance':
            return False, "Avoid new entries in last 15 minutes"

        if window in ['opening_range_breakout', 'power_hour_momentum']:
            return True, f"Optimal entry window: {window}"

        if window == 'midday_dip' and signal_type.upper() == 'BUY':
            return True, "Midday dip buying opportunity"

        return True, "Acceptable entry time"


# ========== FIX 35: Market Cap Tier Optimizer (US/INTL MODEL ONLY) ==========
class MarketCapTierOptimizer:
    """
    Different strategies for different market cap tiers.

    FIX 35: Mega-Cap vs Small-Cap Differentiation

    MEGA-CAP (>$200B): Lower volatility, trend-following works better
    - Position size: 15-25% of portfolio per stock
    - Hold period: 5-20 days
    - Profit target: 8-15%

    SMALL-CAP (<$2B): Higher volatility, mean-reversion better
    - Position size: 3-8% of portfolio
    - Hold period: 2-7 days
    - Profit target: 12-30%
    """

    MARKET_CAP_THRESHOLDS = {
        'mega': 200_000_000_000,    # >$200B
        'large': 10_000_000_000,    # $10B-$200B
        'mid': 2_000_000_000,       # $2B-$10B
        'small': 300_000_000,       # $300M-$2B
        'micro': 50_000_000,        # $50M-$300M
    }

    CAP_TIER_STRATEGIES = {
        'mega': {
            'model_weight': {'catboost': 0.60, 'lstm': 0.40},
            'position_cap': 0.25,
            'position_floor': 0.10,
            'profit_targets': [0.08, 0.15, 0.25],
            'stop_loss': 0.06,
            'hold_period_days': (5, 20),
            'strategy': 'trend_following',
            'buy_mult': 1.20,
            'sell_mult': 0.90,
        },
        'large': {
            'model_weight': {'catboost': 0.65, 'lstm': 0.35},
            'position_cap': 0.20,
            'position_floor': 0.08,
            'profit_targets': [0.10, 0.18, 0.30],
            'stop_loss': 0.07,
            'hold_period_days': (4, 15),
            'strategy': 'trend_following',
            'buy_mult': 1.15,
            'sell_mult': 0.95,
        },
        'mid': {
            'model_weight': {'catboost': 0.70, 'lstm': 0.30},
            'position_cap': 0.12,
            'position_floor': 0.05,
            'profit_targets': [0.12, 0.22, 0.35],
            'stop_loss': 0.08,
            'hold_period_days': (3, 12),
            'strategy': 'balanced',
            'buy_mult': 1.10,
            'sell_mult': 1.00,
        },
        'small': {
            'model_weight': {'catboost': 0.75, 'lstm': 0.25},
            'position_cap': 0.08,
            'position_floor': 0.03,
            'profit_targets': [0.12, 0.25, 0.40],
            'stop_loss': 0.10,
            'hold_period_days': (2, 7),
            'strategy': 'mean_reversion',
            'buy_mult': 1.05,
            'sell_mult': 1.10,
        },
        'micro': {
            'model_weight': {'catboost': 0.80, 'lstm': 0.20},
            'position_cap': 0.05,
            'position_floor': 0.01,
            'profit_targets': [0.15, 0.30, 0.50],
            'stop_loss': 0.12,
            'hold_period_days': (1, 5),
            'strategy': 'mean_reversion',
            'buy_mult': 0.90,  # Reduce for higher risk
            'sell_mult': 1.15,
        },
    }

    # Known mega-cap tickers for quick lookup
    KNOWN_MEGA_CAPS = {
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA',
        'BRK-A', 'BRK-B', 'JPM', 'V', 'UNH', 'JNJ', 'XOM', 'MA', 'PG',
        'HD', 'CVX', 'MRK', 'ABBV', 'LLY', 'PFE', 'BAC', 'COST', 'AVGO',
    }

    KNOWN_LARGE_CAPS = {
        'CRM', 'NFLX', 'AMD', 'INTC', 'CSCO', 'VZ', 'T', 'DIS', 'NKE',
        'MCD', 'WMT', 'PEP', 'KO', 'ADBE', 'PYPL', 'QCOM', 'TXN', 'BMY',
    }

    def __init__(self):
        pass

    def classify_market_cap(self, market_cap: float = None, ticker: str = None) -> str:
        """
        Classify stock by market cap tier.

        Args:
            market_cap: Market cap in dollars (optional)
            ticker: Stock ticker for quick lookup

        Returns:
            Tier name: 'mega', 'large', 'mid', 'small', 'micro'
        """
        # Quick lookup for known tickers
        if ticker:
            ticker_upper = ticker.upper()
            if ticker_upper in self.KNOWN_MEGA_CAPS:
                return 'mega'
            if ticker_upper in self.KNOWN_LARGE_CAPS:
                return 'large'

        # Use market cap if provided
        if market_cap is not None:
            if market_cap >= self.MARKET_CAP_THRESHOLDS['mega']:
                return 'mega'
            elif market_cap >= self.MARKET_CAP_THRESHOLDS['large']:
                return 'large'
            elif market_cap >= self.MARKET_CAP_THRESHOLDS['mid']:
                return 'mid'
            elif market_cap >= self.MARKET_CAP_THRESHOLDS['small']:
                return 'small'
            else:
                return 'micro'

        # Default to mid-cap if unknown
        return 'mid'

    def get_tier_strategy(self, tier: str) -> Dict:
        """Get strategy parameters for a market cap tier."""
        return self.CAP_TIER_STRATEGIES.get(tier, self.CAP_TIER_STRATEGIES['mid'])

    def get_position_adjustment(
        self,
        signal_type: str,
        market_cap: float = None,
        ticker: str = None,
    ) -> Tuple[float, Dict, str]:
        """
        Get position adjustment based on market cap tier.

        Returns:
            (multiplier, strategy_params, reason)
        """
        tier = self.classify_market_cap(market_cap, ticker)
        strategy = self.get_tier_strategy(tier)

        if signal_type.upper() == 'BUY':
            mult = strategy['buy_mult']
        else:
            mult = strategy['sell_mult']

        reason = f"{tier.upper()}-cap: {strategy['strategy']} strategy"
        return mult, strategy, reason


# ========== FIX 36: Quarter-End Window Dressing (US/INTL MODEL ONLY) ==========
class QuarterEndOptimizer:
    """
    Exploit institutional window dressing patterns.

    FIX 36: Quarter-End Window Dressing Exploitation

    Institutional window dressing creates predictable patterns:
    - Last 3-5 trading days of quarter: Add winners, dump losers
    - First 3 days of new quarter: Rebalance
    """

    # Quarter end months
    QUARTER_END_MONTHS = {3, 6, 9, 12}  # March, June, September, December

    # Performance thresholds for classification
    TOP_PERFORMER_THRESHOLD = 0.15    # +15% QTD
    BOTTOM_PERFORMER_THRESHOLD = -0.10  # -10% QTD

    def __init__(self):
        pass

    def get_quarter_end_date(self, date: datetime) -> datetime:
        """Get the last day of the current quarter."""
        month = date.month
        year = date.year

        if month <= 3:
            end_month, end_day = 3, 31
        elif month <= 6:
            end_month, end_day = 6, 30
        elif month <= 9:
            end_month, end_day = 9, 30
        else:
            end_month, end_day = 12, 31

        return datetime(year, end_month, end_day)

    def get_days_to_quarter_end(self, date: datetime = None) -> int:
        """
        Calculate trading days until quarter end.

        Returns:
            Positive = days until quarter end
            Negative = days into new quarter
        """
        if date is None:
            date = datetime.now()

        quarter_end = self.get_quarter_end_date(date)

        # Simple day difference (not exact trading days, but close enough)
        delta = (quarter_end - date).days

        # Adjust for weekends roughly
        weeks = abs(delta) // 7
        trading_days = abs(delta) - (weeks * 2)

        return trading_days if delta >= 0 else -trading_days

    def is_quarter_end_window(self, date: datetime = None) -> Tuple[bool, str]:
        """
        Check if we're in a quarter-end window.

        Returns:
            (is_in_window, window_type)
            window_type: 'last_week', 'first_week', or None
        """
        days_to_end = self.get_days_to_quarter_end(date)

        if 0 <= days_to_end <= 5:
            return True, 'last_week'
        elif -3 <= days_to_end < 0:
            return True, 'first_week'

        return False, None

    def get_quarter_end_adjustment(
        self,
        ticker: str,
        signal_type: str,
        qtd_return: float = 0.0,
        date: datetime = None,
    ) -> Dict:
        """
        Get quarter-end position adjustment.

        Args:
            ticker: Stock ticker
            signal_type: 'BUY' or 'SELL'
            qtd_return: Quarter-to-date return
            date: Current date

        Returns:
            Dict with adjustment details
        """
        is_window, window_type = self.is_quarter_end_window(date)

        if not is_window:
            return {
                'position_multiplier': 1.0,
                'is_quarter_end_window': False,
                'window_type': None,
                'reason': 'Outside quarter-end window',
            }

        # Classify performance
        is_top_performer = qtd_return >= self.TOP_PERFORMER_THRESHOLD
        is_bottom_performer = qtd_return <= self.BOTTOM_PERFORMER_THRESHOLD

        if window_type == 'last_week':
            # Last week of quarter: institutions add winners, dump losers
            if is_top_performer:
                buy_mult, sell_mult = 1.25, 0.50
                reason = "Quarter-end: Window dressing for top performer"
            elif is_bottom_performer:
                buy_mult, sell_mult = 0.60, 1.30
                reason = "Quarter-end: Institutions dumping loser"
            else:
                buy_mult, sell_mult = 1.05, 0.90
                reason = "Quarter-end: Neutral performer"
        else:  # first_week
            # First week of new quarter: rebalancing
            if is_top_performer:
                buy_mult, sell_mult = 0.80, 1.10
                reason = "New quarter: Profit taking on winners"
            elif is_bottom_performer:
                buy_mult, sell_mult = 1.15, 0.70
                reason = "New quarter: Oversold bounce opportunity"
            else:
                buy_mult, sell_mult = 1.0, 1.0
                reason = "New quarter: Neutral rebalancing"

        mult = buy_mult if signal_type.upper() == 'BUY' else sell_mult

        return {
            'position_multiplier': mult,
            'is_quarter_end_window': True,
            'window_type': window_type,
            'is_top_performer': is_top_performer,
            'is_bottom_performer': is_bottom_performer,
            'qtd_return': qtd_return,
            'reason': reason,
        }


# ========== FIX 37: Earnings Gap Trading (US/INTL MODEL ONLY) ==========
class EarningsGapTrader:
    """
    Specialized strategy for post-earnings gap plays.

    FIX 37: Earnings Gap Trading Strategy

    GAP-UP > 5% with high volume:
    - 65% probability of continued momentum next day
    - BUY with tight 2% stop, target 3-8% follow-through

    GAP-DOWN > 5% with high volume:
    - 40% probability of dead-cat bounce
    - SELL with 3% stop, target 4-10% further decline

    This strategy alone can add 8-12% annual alpha.
    """

    # Gap thresholds
    SIGNIFICANT_GAP_UP = 0.05    # 5%
    SIGNIFICANT_GAP_DOWN = -0.05  # -5%
    MODERATE_GAP = 0.03          # 3%

    # Volume requirements
    HIGH_VOLUME_RATIO = 2.0      # 2x average volume
    ELEVATED_VOLUME_RATIO = 1.5  # 1.5x average

    # Gap strategy parameters
    GAP_STRATEGIES = {
        'strong_gap_up': {
            'gap_range': (0.05, 1.0),
            'volume_min': 2.0,
            'action': 'BUY_GAP',
            'entry_window': '10:15-10:30',
            'stop_loss': 0.02,
            'targets': [0.03, 0.06, 0.10],
            'probability': 0.65,
            'position_mult': 1.25,
        },
        'moderate_gap_up': {
            'gap_range': (0.03, 0.05),
            'volume_min': 1.5,
            'action': 'BUY_GAP',
            'entry_window': '10:00-10:30',
            'stop_loss': 0.025,
            'targets': [0.02, 0.04, 0.07],
            'probability': 0.55,
            'position_mult': 1.10,
        },
        'strong_gap_down': {
            'gap_range': (-1.0, -0.05),
            'volume_min': 2.0,
            'action': 'SELL_GAP',
            'entry_window': '9:45-10:00',
            'stop_loss': 0.03,
            'targets': [0.04, 0.08, 0.15],
            'probability': 0.40,
            'position_mult': 1.15,
        },
        'moderate_gap_down': {
            'gap_range': (-0.05, -0.03),
            'volume_min': 1.5,
            'action': 'FADE_GAP',  # Fade = buy the dip
            'entry_window': '10:00-10:30',
            'stop_loss': 0.035,
            'targets': [0.02, 0.04, 0.06],
            'probability': 0.50,
            'position_mult': 0.90,
        },
    }

    def __init__(self):
        pass

    def analyze_gap(
        self,
        gap_percent: float,
        volume_ratio: float,
        is_earnings_day: bool = True,
    ) -> Optional[Dict]:
        """
        Analyze a gap and return trading strategy.

        Args:
            gap_percent: Gap size as decimal (0.05 = 5%)
            volume_ratio: Current volume / average volume
            is_earnings_day: Whether this is post-earnings

        Returns:
            Strategy dict or None if no gap strategy applies
        """
        if not is_earnings_day:
            # Reduce confidence for non-earnings gaps
            volume_ratio *= 0.8

        for strategy_name, strategy in self.GAP_STRATEGIES.items():
            gap_min, gap_max = strategy['gap_range']

            # Check if gap is in range
            if gap_min <= gap_percent <= gap_max:
                # Check volume requirement
                if volume_ratio >= strategy['volume_min']:
                    return {
                        'strategy': strategy_name,
                        'action': strategy['action'],
                        'entry_window': strategy['entry_window'],
                        'stop_loss': strategy['stop_loss'],
                        'targets': strategy['targets'],
                        'probability': strategy['probability'],
                        'position_multiplier': strategy['position_mult'],
                        'gap_percent': gap_percent,
                        'volume_ratio': volume_ratio,
                        'is_earnings': is_earnings_day,
                    }

        return None

    def get_gap_adjustment(
        self,
        signal_type: str,
        gap_percent: float,
        volume_ratio: float = 1.0,
        is_earnings_day: bool = False,
    ) -> Tuple[float, str]:
        """
        Get position adjustment for gap situations.

        Returns:
            (multiplier, reason)
        """
        gap_analysis = self.analyze_gap(gap_percent, volume_ratio, is_earnings_day)

        if gap_analysis is None:
            return 1.0, "No significant gap detected"

        action = gap_analysis['action']

        # Align signal with gap action
        if signal_type.upper() == 'BUY':
            if action in ['BUY_GAP', 'FADE_GAP']:
                return gap_analysis['position_multiplier'], f"Gap strategy: {action}"
            else:
                return 0.70, "Signal conflicts with gap direction"
        else:  # SELL
            if action == 'SELL_GAP':
                return gap_analysis['position_multiplier'], f"Gap strategy: {action}"
            else:
                return 0.70, "Signal conflicts with gap direction"


# ========== FIX 38: Sector Rotation Momentum (US/INTL MODEL ONLY) ==========
class SectorRotationMomentum:
    """
    Track which sectors are rotating IN/OUT of favor.

    FIX 38: Sector Rotation Momentum

    LEADING SECTORS (rotating IN):
    - Increase position size by 1.25-1.50x
    - Extend profit targets by 1.3x
    - Reduce stop-loss to 0.75x

    LAGGING SECTORS (rotating OUT):
    - Reduce position size to 0.5-0.7x
    - Tighten profit targets to 0.7x
    - Increase stop-loss to 1.3x
    """

    # Sector rotation status adjustments
    ROTATION_ADJUSTMENTS = {
        'ROTATING_IN': {
            'position_mult': 1.35,
            'profit_target_mult': 1.30,
            'stop_loss_mult': 0.75,
            'confidence_boost': 0.05,
        },
        'STRENGTHENING': {
            'position_mult': 1.20,
            'profit_target_mult': 1.15,
            'stop_loss_mult': 0.85,
            'confidence_boost': 0.03,
        },
        'NEUTRAL': {
            'position_mult': 1.00,
            'profit_target_mult': 1.00,
            'stop_loss_mult': 1.00,
            'confidence_boost': 0.00,
        },
        'WEAKENING': {
            'position_mult': 0.75,
            'profit_target_mult': 0.85,
            'stop_loss_mult': 1.15,
            'confidence_boost': -0.03,
        },
        'ROTATING_OUT': {
            'position_mult': 0.55,
            'profit_target_mult': 0.70,
            'stop_loss_mult': 1.30,
            'confidence_boost': -0.05,
        },
    }

    # Thresholds for rotation detection
    RS_ROTATING_IN = 1.15      # >15% relative strength
    RS_STRENGTHENING = 1.05    # >5% RS
    RS_WEAKENING = 0.95        # <-5% RS
    RS_ROTATING_OUT = 0.85     # <-15% RS

    FUND_FLOW_STRONG = 0.02    # >2% inflows
    FUND_FLOW_WEAK = -0.02     # >2% outflows

    def __init__(self):
        pass

    def detect_rotation_status(
        self,
        relative_strength: float,
        fund_flow: float = 0.0,
        momentum_5d: float = 0.0,
    ) -> str:
        """
        Determine sector rotation status.

        Args:
            relative_strength: Sector RS vs market (1.0 = equal)
            fund_flow: Net fund flow as decimal
            momentum_5d: 5-day price momentum

        Returns:
            Rotation status string
        """
        # Score based on multiple factors
        rs_score = 0
        if relative_strength >= self.RS_ROTATING_IN:
            rs_score = 2
        elif relative_strength >= self.RS_STRENGTHENING:
            rs_score = 1
        elif relative_strength <= self.RS_ROTATING_OUT:
            rs_score = -2
        elif relative_strength <= self.RS_WEAKENING:
            rs_score = -1

        flow_score = 0
        if fund_flow >= self.FUND_FLOW_STRONG:
            flow_score = 1
        elif fund_flow <= self.FUND_FLOW_WEAK:
            flow_score = -1

        momentum_score = 0
        if momentum_5d > 0.02:
            momentum_score = 1
        elif momentum_5d < -0.02:
            momentum_score = -1

        total_score = rs_score + flow_score + momentum_score

        if total_score >= 3:
            return 'ROTATING_IN'
        elif total_score >= 1:
            return 'STRENGTHENING'
        elif total_score <= -3:
            return 'ROTATING_OUT'
        elif total_score <= -1:
            return 'WEAKENING'
        else:
            return 'NEUTRAL'

    def get_rotation_adjustment(
        self,
        signal_type: str,
        relative_strength: float,
        fund_flow: float = 0.0,
        momentum_5d: float = 0.0,
    ) -> Tuple[float, Dict, str]:
        """
        Get position adjustment based on sector rotation.

        Returns:
            (position_mult, adjustments_dict, reason)
        """
        status = self.detect_rotation_status(relative_strength, fund_flow, momentum_5d)
        adjustments = self.ROTATION_ADJUSTMENTS[status]

        # BUY favors rotating-in, SELL favors rotating-out
        if signal_type.upper() == 'BUY':
            if status in ['ROTATING_IN', 'STRENGTHENING']:
                mult = adjustments['position_mult']
            elif status in ['ROTATING_OUT', 'WEAKENING']:
                mult = 0.80  # Reduce BUY in weak sectors
            else:
                mult = 1.0
        else:  # SELL
            if status in ['ROTATING_OUT', 'WEAKENING']:
                mult = adjustments['position_mult'] * 1.2  # Boost SELL in weak sectors
            elif status in ['ROTATING_IN', 'STRENGTHENING']:
                mult = 0.70  # Reduce SELL in strong sectors
            else:
                mult = 1.0

        reason = f"Sector {status}: RS={relative_strength:.2f}, Flow={fund_flow:.2%}"
        return mult, adjustments, reason


# ========== FIX 39: VIX Term Structure Arbitrage (US/INTL MODEL ONLY) ==========
class VIXTermStructureAnalyzer:
    """
    Exploit VIX futures term structure for market timing.

    FIX 39: VIX Term Structure Arbitrage

    VIX FUTURES IN CONTANGO (normal):
    - Market complacent, favorable for BUY signals
    - Position multiplier: 1.10x
    - Confidence boost: +5%

    VIX FUTURES IN BACKWARDATION (inverted):
    - Market fearful, favorable for SELL signals
    - Position multiplier: 1.15x for SELL
    - Confidence boost: +8% for SELL signals
    """

    # Contango/backwardation thresholds
    STRONG_CONTANGO = 0.05     # >5% futures premium
    MODERATE_CONTANGO = 0.02   # 2-5%
    MODERATE_BACKWARDATION = -0.02  # -2% to 0%
    STRONG_BACKWARDATION = -0.05    # <-5%

    # Term structure regimes
    TERM_STRUCTURE_REGIMES = {
        'strong_contango': {
            'buy_mult': 1.15,
            'sell_mult': 0.80,
            'buy_conf_boost': 0.05,
            'sell_conf_boost': -0.03,
            'market_regime': 'COMPLACENT',
        },
        'moderate_contango': {
            'buy_mult': 1.08,
            'sell_mult': 0.90,
            'buy_conf_boost': 0.03,
            'sell_conf_boost': 0.0,
            'market_regime': 'CALM',
        },
        'flat': {
            'buy_mult': 1.0,
            'sell_mult': 1.0,
            'buy_conf_boost': 0.0,
            'sell_conf_boost': 0.0,
            'market_regime': 'NEUTRAL',
        },
        'moderate_backwardation': {
            'buy_mult': 0.90,
            'sell_mult': 1.10,
            'buy_conf_boost': -0.02,
            'sell_conf_boost': 0.05,
            'market_regime': 'CAUTIOUS',
        },
        'strong_backwardation': {
            'buy_mult': 0.80,
            'sell_mult': 1.20,
            'buy_conf_boost': -0.05,
            'sell_conf_boost': 0.08,
            'market_regime': 'FEARFUL',
        },
    }

    def __init__(self):
        pass

    def analyze_term_structure(
        self,
        vix_spot: float,
        vix_futures_1m: float,
        vix_futures_2m: float = None,
    ) -> Dict:
        """
        Analyze VIX term structure.

        Args:
            vix_spot: Spot VIX level
            vix_futures_1m: 1-month VIX futures
            vix_futures_2m: 2-month VIX futures (optional)

        Returns:
            Term structure analysis dict
        """
        if vix_spot <= 0:
            return {'regime': 'flat', 'contango_1m': 0.0}

        # Calculate contango (positive = futures > spot)
        contango_1m = (vix_futures_1m / vix_spot) - 1

        contango_2m = None
        if vix_futures_2m is not None and vix_futures_1m > 0:
            contango_2m = (vix_futures_2m / vix_futures_1m) - 1

        # Classify regime
        if contango_1m >= self.STRONG_CONTANGO:
            regime = 'strong_contango'
        elif contango_1m >= self.MODERATE_CONTANGO:
            regime = 'moderate_contango'
        elif contango_1m <= self.STRONG_BACKWARDATION:
            regime = 'strong_backwardation'
        elif contango_1m <= self.MODERATE_BACKWARDATION:
            regime = 'moderate_backwardation'
        else:
            regime = 'flat'

        return {
            'regime': regime,
            'contango_1m': contango_1m,
            'contango_2m': contango_2m,
            'vix_spot': vix_spot,
            'vix_futures_1m': vix_futures_1m,
            'market_regime': self.TERM_STRUCTURE_REGIMES[regime]['market_regime'],
        }

    def get_term_structure_adjustment(
        self,
        signal_type: str,
        vix_spot: float,
        vix_futures_1m: float,
        vix_futures_2m: float = None,
    ) -> Tuple[float, float, str]:
        """
        Get adjustment based on VIX term structure.

        Returns:
            (position_mult, confidence_boost, reason)
        """
        analysis = self.analyze_term_structure(vix_spot, vix_futures_1m, vix_futures_2m)
        regime = analysis['regime']
        params = self.TERM_STRUCTURE_REGIMES[regime]

        if signal_type.upper() == 'BUY':
            mult = params['buy_mult']
            conf_boost = params['buy_conf_boost']
        else:
            mult = params['sell_mult']
            conf_boost = params['sell_conf_boost']

        reason = f"VIX {params['market_regime']}: contango={analysis['contango_1m']:.1%}"
        return mult, conf_boost, reason


# ========== FIX 40: Economic Data Reactions (US/INTL MODEL ONLY) ==========
class EconomicDataReactor:
    """
    Handle immediate market reactions to economic data releases.

    FIX 40: High-Frequency Economic Data Reaction

    POSITIVE SURPRISE (beat expectations):
    - Initial spike: 60% continue higher next 2 hours
    - Strategy: Fade if >2% spike, else ride momentum

    NEGATIVE SURPRISE (miss expectations):
    - Initial drop: 55% continue lower next 2 hours
    - Strategy: Buy dip if oversold (RSI<25), else short

    Key data points: CPI, Jobs Report, Fed Decision, GDP
    """

    # Economic event volatility profiles
    ECONOMIC_EVENTS = {
        'cpi': {
            'name': 'Consumer Price Index',
            'volatility_mult': 1.8,
            'duration_hours': 4,
            'typical_time': '08:30',
            'importance': 'HIGH',
        },
        'jobs_report': {
            'name': 'Non-Farm Payrolls',
            'volatility_mult': 2.2,
            'duration_hours': 6,
            'typical_time': '08:30',
            'importance': 'HIGH',
        },
        'fed_decision': {
            'name': 'FOMC Decision',
            'volatility_mult': 2.5,
            'duration_hours': 8,
            'typical_time': '14:00',
            'importance': 'CRITICAL',
        },
        'gdp': {
            'name': 'GDP Report',
            'volatility_mult': 1.5,
            'duration_hours': 3,
            'typical_time': '08:30',
            'importance': 'MEDIUM',
        },
        'ppi': {
            'name': 'Producer Price Index',
            'volatility_mult': 1.4,
            'duration_hours': 2,
            'typical_time': '08:30',
            'importance': 'MEDIUM',
        },
        'retail_sales': {
            'name': 'Retail Sales',
            'volatility_mult': 1.3,
            'duration_hours': 2,
            'typical_time': '08:30',
            'importance': 'MEDIUM',
        },
    }

    # Surprise reaction thresholds
    MAJOR_SPIKE_THRESHOLD = 0.02   # 2% move
    MINOR_SPIKE_THRESHOLD = 0.01  # 1% move

    def __init__(self):
        pass

    def get_event_profile(self, event_type: str) -> Optional[Dict]:
        """Get profile for an economic event."""
        return self.ECONOMIC_EVENTS.get(event_type.lower())

    def analyze_data_reaction(
        self,
        event_type: str,
        actual_vs_expected: float,  # Positive = beat, Negative = miss
        immediate_move: float,      # Market reaction (price change)
        rsi: float = 50.0,
        hours_since_release: float = 0.0,
    ) -> Dict:
        """
        Analyze market reaction to economic data.

        Args:
            event_type: Type of economic event
            actual_vs_expected: Surprise (positive = beat)
            immediate_move: Market price reaction
            rsi: Current RSI
            hours_since_release: Hours since data release

        Returns:
            Reaction analysis dict
        """
        profile = self.get_event_profile(event_type)
        if profile is None:
            return {'action': 'IGNORE', 'reason': 'Unknown event type'}

        # Check if still in reaction window
        if hours_since_release > profile['duration_hours']:
            return {
                'action': 'NORMAL',
                'reason': 'Outside reaction window',
                'position_mult': 1.0,
            }

        # Analyze surprise direction
        is_positive_surprise = actual_vs_expected > 0
        is_major_move = abs(immediate_move) >= self.MAJOR_SPIKE_THRESHOLD
        is_minor_move = abs(immediate_move) >= self.MINOR_SPIKE_THRESHOLD

        # Determine action
        if is_positive_surprise:
            if is_major_move:
                # Major spike up - fade it
                action = 'FADE_SPIKE'
                buy_mult, sell_mult = 0.80, 1.15
            elif is_minor_move:
                # Minor spike - ride momentum
                action = 'RIDE_MOMENTUM'
                buy_mult, sell_mult = 1.15, 0.85
            else:
                action = 'NORMAL'
                buy_mult, sell_mult = 1.05, 0.95
        else:  # Negative surprise
            if is_major_move and rsi < 25:
                # Oversold after drop - buy dip
                action = 'BUY_DIP'
                buy_mult, sell_mult = 1.25, 0.70
            elif is_major_move:
                # Major drop - short
                action = 'RIDE_MOMENTUM_DOWN'
                buy_mult, sell_mult = 0.75, 1.20
            elif is_minor_move:
                action = 'CAUTION'
                buy_mult, sell_mult = 0.90, 1.05
            else:
                action = 'NORMAL'
                buy_mult, sell_mult = 0.95, 1.05

        # Adjust for volatility
        vol_factor = profile['volatility_mult']

        return {
            'action': action,
            'event_type': event_type,
            'event_name': profile['name'],
            'importance': profile['importance'],
            'buy_mult': buy_mult,
            'sell_mult': sell_mult,
            'volatility_mult': vol_factor,
            'surprise_direction': 'POSITIVE' if is_positive_surprise else 'NEGATIVE',
            'move_magnitude': abs(immediate_move),
            'hours_remaining': profile['duration_hours'] - hours_since_release,
            'reason': f"{profile['name']}: {action}",
        }

    def get_economic_adjustment(
        self,
        signal_type: str,
        event_type: str = None,
        actual_vs_expected: float = 0.0,
        immediate_move: float = 0.0,
        rsi: float = 50.0,
        hours_since_release: float = 0.0,
    ) -> Tuple[float, str]:
        """
        Get position adjustment for economic data reaction.

        Returns:
            (multiplier, reason)
        """
        if event_type is None:
            return 1.0, "No economic event"

        analysis = self.analyze_data_reaction(
            event_type, actual_vs_expected, immediate_move, rsi, hours_since_release
        )

        if analysis['action'] == 'IGNORE':
            return 1.0, analysis['reason']

        if signal_type.upper() == 'BUY':
            return analysis.get('buy_mult', 1.0), analysis['reason']
        else:
            return analysis.get('sell_mult', 1.0), analysis['reason']


# ========== FIX 41: Put/Call Ratio Reversals (US/INTL MODEL ONLY) ==========
class PutCallRatioAnalyzer:
    """
    Use extreme put/call ratios to signal market reversals.

    FIX 41: Put/Call Ratio Extreme Reversals

    PUT/CALL RATIO > 1.0 (excessive bearishness):
    - 70% probability of bullish reversal in 1-3 days
    - BUY signal strength: 1.25-1.40x

    PUT/CALL RATIO < 0.6 (excessive bullishness):
    - 65% probability of bearish reversal in 1-3 days
    - SELL signal strength: 1.20-1.35x
    """

    # Put/Call ratio thresholds
    PUT_CALL_THRESHOLDS = {
        'extreme_bearish': 1.2,    # >1.2 = extreme fear
        'bearish': 1.0,            # >1.0 = elevated fear
        'slightly_bearish': 0.85,  # 0.85-1.0 = mild concern
        'neutral_high': 0.75,
        'neutral_low': 0.65,
        'slightly_bullish': 0.60,  # 0.60-0.65 = mild complacency
        'bullish': 0.55,           # <0.55 = elevated greed
        'extreme_bullish': 0.45,   # <0.45 = extreme greed
    }

    # Contrarian adjustments
    CONTRARIAN_ADJUSTMENTS = {
        'extreme_bearish': {
            'buy_mult': 1.40,
            'sell_mult': 0.60,
            'reversal_prob': 0.75,
            'signal': 'STRONG_CONTRARIAN_BUY',
        },
        'bearish': {
            'buy_mult': 1.25,
            'sell_mult': 0.75,
            'reversal_prob': 0.70,
            'signal': 'CONTRARIAN_BUY',
        },
        'slightly_bearish': {
            'buy_mult': 1.10,
            'sell_mult': 0.90,
            'reversal_prob': 0.55,
            'signal': 'MILD_CONTRARIAN_BUY',
        },
        'neutral': {
            'buy_mult': 1.00,
            'sell_mult': 1.00,
            'reversal_prob': 0.50,
            'signal': 'NEUTRAL',
        },
        'slightly_bullish': {
            'buy_mult': 0.90,
            'sell_mult': 1.10,
            'reversal_prob': 0.55,
            'signal': 'MILD_CONTRARIAN_SELL',
        },
        'bullish': {
            'buy_mult': 0.75,
            'sell_mult': 1.25,
            'reversal_prob': 0.65,
            'signal': 'CONTRARIAN_SELL',
        },
        'extreme_bullish': {
            'buy_mult': 0.60,
            'sell_mult': 1.40,
            'reversal_prob': 0.70,
            'signal': 'STRONG_CONTRARIAN_SELL',
        },
    }

    def __init__(self):
        self.historical_ratios = deque(maxlen=20)  # Track recent ratios

    def classify_put_call_ratio(self, ratio: float) -> str:
        """
        Classify put/call ratio into sentiment category.

        Args:
            ratio: Put/call ratio

        Returns:
            Category name
        """
        if ratio >= self.PUT_CALL_THRESHOLDS['extreme_bearish']:
            return 'extreme_bearish'
        elif ratio >= self.PUT_CALL_THRESHOLDS['bearish']:
            return 'bearish'
        elif ratio >= self.PUT_CALL_THRESHOLDS['slightly_bearish']:
            return 'slightly_bearish'
        elif ratio <= self.PUT_CALL_THRESHOLDS['extreme_bullish']:
            return 'extreme_bullish'
        elif ratio <= self.PUT_CALL_THRESHOLDS['bullish']:
            return 'bullish'
        elif ratio <= self.PUT_CALL_THRESHOLDS['slightly_bullish']:
            return 'slightly_bullish'
        else:
            return 'neutral'

    def get_contrarian_signal(
        self,
        put_call_ratio: float,
        put_call_ratio_5d_avg: float = None,
    ) -> Dict:
        """
        Generate contrarian signal from put/call ratio.

        Args:
            put_call_ratio: Current put/call ratio
            put_call_ratio_5d_avg: 5-day average (optional)

        Returns:
            Contrarian signal analysis
        """
        category = self.classify_put_call_ratio(put_call_ratio)
        adjustments = self.CONTRARIAN_ADJUSTMENTS[category]

        # Check for extreme divergence from average
        is_extreme_divergence = False
        if put_call_ratio_5d_avg is not None:
            divergence = put_call_ratio - put_call_ratio_5d_avg
            if abs(divergence) > 0.15:
                is_extreme_divergence = True
                # Boost contrarian signal on extreme divergence
                if divergence > 0:  # Ratio spiked up = more bearish
                    adjustments = dict(adjustments)
                    adjustments['buy_mult'] *= 1.10
                    adjustments['sell_mult'] *= 0.90
                else:  # Ratio dropped = more bullish
                    adjustments = dict(adjustments)
                    adjustments['buy_mult'] *= 0.90
                    adjustments['sell_mult'] *= 1.10

        return {
            'category': category,
            'signal': adjustments['signal'],
            'buy_mult': adjustments['buy_mult'],
            'sell_mult': adjustments['sell_mult'],
            'reversal_probability': adjustments['reversal_prob'],
            'put_call_ratio': put_call_ratio,
            'is_extreme_divergence': is_extreme_divergence,
        }

    def get_put_call_adjustment(
        self,
        signal_type: str,
        put_call_ratio: float,
        put_call_ratio_5d_avg: float = None,
    ) -> Tuple[float, str]:
        """
        Get position adjustment based on put/call ratio.

        Returns:
            (multiplier, reason)
        """
        analysis = self.get_contrarian_signal(put_call_ratio, put_call_ratio_5d_avg)

        if signal_type.upper() == 'BUY':
            mult = analysis['buy_mult']
        else:
            mult = analysis['sell_mult']

        reason = f"P/C={put_call_ratio:.2f} -> {analysis['signal']} (prob={analysis['reversal_probability']:.0%})"
        return mult, reason


# ========== FIX 42: Unified US Profit Maximizer (US/INTL MODEL ONLY) ==========
class UnifiedUSProfitMaximizer:
    """
    Master optimizer combining ALL fixes (27-41) for maximum US profits.

    FIX 42: Unified Profit Optimizer

    Combines all individual optimizers into a single multiplicative pipeline:
    - Market regime (Fix 27) + VIX term structure (Fix 39)
    - Sector momentum + rotation (Fix 28 + 38)
    - Earnings + economic calendar (Fix 29 + 40)
    - Intraday timing (Fix 34) + market cap tier (Fix 35)
    - Put/Call sentiment (Fix 41) + internals (Fix 32)
    - Quarter-end patterns (Fix 36) + gap trading (Fix 37)
    - Risk constraints (Fix 33) as final check
    """

    # Multiplier caps to prevent extreme positions
    MAX_COMBINED_MULTIPLIER = 3.0
    MIN_COMBINED_MULTIPLIER = 0.20

    def __init__(
        self,
        regime_classifier: 'USMarketRegimeClassifier' = None,
        sector_momentum: 'SectorMomentumAnalyzer' = None,
        earnings_optimizer: 'EarningsSeasonOptimizer' = None,
        fomc_optimizer: 'FOMCOptimizer' = None,
        opex_optimizer: 'OpExOptimizer' = None,
        market_internals: 'USMarketInternals' = None,
        us_risk_model: 'USRiskModel' = None,
        intraday_optimizer: 'IntradayMomentumOptimizer' = None,
        market_cap_optimizer: 'MarketCapTierOptimizer' = None,
        quarter_end_optimizer: 'QuarterEndOptimizer' = None,
        earnings_gap_trader: 'EarningsGapTrader' = None,
        sector_rotation: 'SectorRotationMomentum' = None,
        vix_term_structure: 'VIXTermStructureAnalyzer' = None,
        economic_reactor: 'EconomicDataReactor' = None,
        put_call_analyzer: 'PutCallRatioAnalyzer' = None,
    ):
        self.regime_classifier = regime_classifier
        self.sector_momentum = sector_momentum
        self.earnings_optimizer = earnings_optimizer
        self.fomc_optimizer = fomc_optimizer
        self.opex_optimizer = opex_optimizer
        self.market_internals = market_internals
        self.us_risk_model = us_risk_model
        self.intraday_optimizer = intraday_optimizer
        self.market_cap_optimizer = market_cap_optimizer
        self.quarter_end_optimizer = quarter_end_optimizer
        self.earnings_gap_trader = earnings_gap_trader
        self.sector_rotation = sector_rotation
        self.vix_term_structure = vix_term_structure
        self.economic_reactor = economic_reactor
        self.put_call_analyzer = put_call_analyzer

    def optimize_us_signal(
        self,
        ticker: str,
        signal_type: str,
        base_confidence: float,
        market_data: Dict = None,
    ) -> Dict:
        """
        Apply ALL optimizations to a US signal.

        Args:
            ticker: Stock ticker
            signal_type: 'BUY' or 'SELL'
            base_confidence: Initial confidence (0-1)
            market_data: Dict with optional market data:
                - vix_level, vix_futures_1m, vix_futures_2m
                - spy_returns_20d, spy_returns_5d
                - sector_returns_20d, ticker_returns_20d
                - is_fomc_week, is_earnings_season, is_opex_week
                - days_to_earnings, days_to_fomc, days_to_opex
                - put_call_ratio, ad_ratio, nhnl_ratio, trin, mcclellan
                - gap_percent, volume_ratio, qtd_return
                - market_cap, relative_strength, fund_flow

        Returns:
            Dict with optimized signal details
        """
        market_data = market_data or {}
        multipliers = {}
        reasons = []

        # 1. Market Regime (Fix 27) + VIX Term Structure (Fix 39)
        regime_mult = 1.0
        if self.regime_classifier:
            regime, weights = self.regime_classifier.classify_regime(
                spy_returns_20d=market_data.get('spy_returns_20d', 0.0),
                spy_returns_5d=market_data.get('spy_returns_5d', 0.0),
                vix_level=market_data.get('vix_level', 20.0),
                is_fomc_week=market_data.get('is_fomc_week', False),
                is_earnings_season=market_data.get('is_earnings_season', False),
                is_opex_week=market_data.get('is_opex_week', False),
            )
            pos_mult = self.regime_classifier.POSITION_MULTIPLIERS.get(regime, {})
            regime_mult = pos_mult.get(signal_type.upper(), 1.0)
            multipliers['regime'] = regime_mult
            reasons.append(f"Regime({regime}): {regime_mult:.2f}x")

        vix_mult = 1.0
        if self.vix_term_structure and market_data.get('vix_futures_1m'):
            vix_mult, _, vix_reason = self.vix_term_structure.get_term_structure_adjustment(
                signal_type,
                vix_spot=market_data.get('vix_level', 20.0),
                vix_futures_1m=market_data.get('vix_futures_1m', 21.0),
            )
            multipliers['vix_structure'] = vix_mult
            reasons.append(f"VIX: {vix_mult:.2f}x")

        # 2. Sector Momentum (Fix 28) + Rotation (Fix 38)
        sector_mult = 1.0
        if self.sector_momentum and market_data.get('ticker_returns_20d'):
            _, sector_reason = self.sector_momentum.get_position_adjustment(
                ticker=ticker,
                signal_type=signal_type,
                ticker_returns_20d=market_data.get('ticker_returns_20d', 0.0),
                sector_returns_20d=market_data.get('sector_returns_20d', 0.0),
            )
            # Extract multiplier from adjustment
            adj = self.sector_momentum.get_position_adjustment(
                ticker, signal_type,
                market_data.get('ticker_returns_20d', 0.0),
                market_data.get('sector_returns_20d', 0.0),
            )
            sector_mult = adj[0]
            multipliers['sector_momentum'] = sector_mult
            reasons.append(f"Sector: {sector_mult:.2f}x")

        rotation_mult = 1.0
        if self.sector_rotation and market_data.get('relative_strength'):
            rotation_mult, _, _ = self.sector_rotation.get_rotation_adjustment(
                signal_type,
                relative_strength=market_data.get('relative_strength', 1.0),
                fund_flow=market_data.get('fund_flow', 0.0),
            )
            multipliers['rotation'] = rotation_mult
            reasons.append(f"Rotation: {rotation_mult:.2f}x")

        # 3. Earnings (Fix 29) + Economic (Fix 40)
        earnings_mult = 1.0
        if self.earnings_optimizer and market_data.get('days_to_earnings') is not None:
            result = self.earnings_optimizer.optimize_for_earnings(
                ticker=ticker,
                signal_type=signal_type,
                confidence=base_confidence,
                days_to_earnings=market_data.get('days_to_earnings', 30),
                is_earnings_season=market_data.get('is_earnings_season', False),
            )
            earnings_mult = result.get('position_multiplier', 1.0)
            multipliers['earnings'] = earnings_mult
            reasons.append(f"Earnings: {earnings_mult:.2f}x")

        econ_mult = 1.0
        if self.economic_reactor and market_data.get('economic_event_type'):
            econ_mult, _ = self.economic_reactor.get_economic_adjustment(
                signal_type,
                event_type=market_data.get('economic_event_type'),
                actual_vs_expected=market_data.get('economic_surprise', 0.0),
            )
            multipliers['economic'] = econ_mult
            reasons.append(f"Economic: {econ_mult:.2f}x")

        # 4. Intraday Timing (Fix 34) + Market Cap (Fix 35)
        intraday_mult = 1.0
        if self.intraday_optimizer:
            intraday_mult, _ = self.intraday_optimizer.get_timing_boost(signal_type)
            multipliers['intraday'] = intraday_mult
            reasons.append(f"Intraday: {intraday_mult:.2f}x")

        cap_mult = 1.0
        if self.market_cap_optimizer:
            cap_mult, _, _ = self.market_cap_optimizer.get_position_adjustment(
                signal_type,
                market_cap=market_data.get('market_cap'),
                ticker=ticker,
            )
            multipliers['market_cap'] = cap_mult
            reasons.append(f"MarketCap: {cap_mult:.2f}x")

        # 5. Put/Call (Fix 41) + Internals (Fix 32)
        pcr_mult = 1.0
        if self.put_call_analyzer and market_data.get('put_call_ratio'):
            pcr_mult, _ = self.put_call_analyzer.get_put_call_adjustment(
                signal_type,
                put_call_ratio=market_data.get('put_call_ratio', 0.80),
            )
            multipliers['put_call'] = pcr_mult
            reasons.append(f"P/C: {pcr_mult:.2f}x")

        internals_mult = 1.0
        if self.market_internals and market_data.get('ad_ratio'):
            health, _ = self.market_internals.get_market_health_score(
                ad_ratio=market_data.get('ad_ratio', 1.0),
                nhnl_ratio=market_data.get('nhnl_ratio', 1.0),
                trin=market_data.get('trin', 1.0),
                mcclellan=market_data.get('mcclellan', 0.0),
            )
            internals_mult, _ = self.market_internals.get_position_adjustment(signal_type, health)
            multipliers['internals'] = internals_mult
            reasons.append(f"Internals: {internals_mult:.2f}x")

        # 6. Quarter-End (Fix 36) + Gap Trading (Fix 37)
        quarter_mult = 1.0
        if self.quarter_end_optimizer:
            result = self.quarter_end_optimizer.get_quarter_end_adjustment(
                ticker=ticker,
                signal_type=signal_type,
                qtd_return=market_data.get('qtd_return', 0.0),
            )
            quarter_mult = result.get('position_multiplier', 1.0)
            multipliers['quarter_end'] = quarter_mult
            reasons.append(f"QuarterEnd: {quarter_mult:.2f}x")

        gap_mult = 1.0
        if self.earnings_gap_trader and market_data.get('gap_percent'):
            gap_mult, _ = self.earnings_gap_trader.get_gap_adjustment(
                signal_type,
                gap_percent=market_data.get('gap_percent', 0.0),
                volume_ratio=market_data.get('volume_ratio', 1.0),
                is_earnings_day=market_data.get('is_earnings_day', False),
            )
            multipliers['gap'] = gap_mult
            reasons.append(f"Gap: {gap_mult:.2f}x")

        # 7. FOMC (Fix 30) + OpEx (Fix 31)
        fomc_mult = 1.0
        if self.fomc_optimizer and market_data.get('days_to_fomc') is not None:
            result = self.fomc_optimizer.adjust_for_fomc(
                ticker=ticker,
                signal_type=signal_type,
                days_to_fomc=market_data.get('days_to_fomc', 30),
            )
            fomc_mult = result.get('position_multiplier', 1.0)
            multipliers['fomc'] = fomc_mult
            reasons.append(f"FOMC: {fomc_mult:.2f}x")

        opex_mult = 1.0
        if self.opex_optimizer and market_data.get('days_to_opex') is not None:
            result = self.opex_optimizer.adjust_for_opex(
                ticker=ticker,
                signal_type=signal_type,
                days_to_opex=market_data.get('days_to_opex', 30),
            )
            opex_mult = result.get('position_multiplier', 1.0)
            multipliers['opex'] = opex_mult
            reasons.append(f"OpEx: {opex_mult:.2f}x")

        # 8. Combine ALL multipliers (MULTIPLICATIVE)
        combined_mult = 1.0
        for key, mult in multipliers.items():
            combined_mult *= mult

        # Apply caps
        combined_mult = max(self.MIN_COMBINED_MULTIPLIER,
                          min(self.MAX_COMBINED_MULTIPLIER, combined_mult))

        # Calculate adjusted confidence
        adjusted_confidence = min(0.99, base_confidence * (1 + (combined_mult - 1) * 0.1))

        return {
            'ticker': ticker,
            'signal_type': signal_type,
            'base_confidence': base_confidence,
            'adjusted_confidence': adjusted_confidence,
            'combined_multiplier': combined_mult,
            'individual_multipliers': multipliers,
            'reasons': reasons,
            'is_capped': combined_mult in [self.MIN_COMBINED_MULTIPLIER, self.MAX_COMBINED_MULTIPLIER],
        }


# ========== FIX 43: Enhanced Sector Rotation Detector (US/INTL MODEL ONLY) ==========
class EnhancedSectorRotationDetector:
    """
    PREDICTIVE sector rotation detection using leading indicators.

    FIX 43: Enhanced Sector Rotation Detector

    Uses multiple leading indicators to PREDICT rotation:
    - Institutional flow momentum (30% weight)
    - Relative strength momentum (25% weight)
    - Earnings estimate revisions (20% weight)
    - Technical breakout confirmation (25% weight)
    """

    LEADING_INDICATORS = {
        'institutional_flows': {'weight': 0.30, 'description': '3-day ETF flow momentum'},
        'relative_strength': {'weight': 0.25, 'description': 'RS vs SPY > 1.15 for 5+ days'},
        'earnings_revisions': {'weight': 0.20, 'description': 'Upward EPS revisions trend'},
        'breakout_confirmation': {'weight': 0.25, 'description': 'Price > 50-day high with volume'},
    }

    # Rotation prediction thresholds
    ROTATING_IN_SOON_THRESHOLD = 0.70
    STRENGTHENING_THRESHOLD = 0.50
    WEAKENING_THRESHOLD = 0.30
    ROTATING_OUT_SOON_THRESHOLD = 0.20

    # Sector ETF tickers
    SECTOR_ETFS = ['XLK', 'XLV', 'XLF', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB', 'XLC']

    def __init__(self):
        pass

    def calculate_flow_momentum(
        self,
        etf_flow_3d: float,
        etf_flow_20d: float,
    ) -> float:
        """
        Calculate institutional flow momentum score.

        Args:
            etf_flow_3d: 3-day fund flow (as % of AUM)
            etf_flow_20d: 20-day fund flow (as % of AUM)

        Returns:
            Score 0-1 (higher = more inflows)
        """
        # Recent flow momentum
        if etf_flow_3d > 0.02:  # >2% inflows in 3 days
            recent_score = 1.0
        elif etf_flow_3d > 0.01:
            recent_score = 0.75
        elif etf_flow_3d > 0:
            recent_score = 0.5
        elif etf_flow_3d > -0.01:
            recent_score = 0.25
        else:
            recent_score = 0.0

        # Trend confirmation
        if etf_flow_20d > 0.03:  # >3% over 20 days
            trend_score = 1.0
        elif etf_flow_20d > 0.01:
            trend_score = 0.75
        elif etf_flow_20d > 0:
            trend_score = 0.5
        else:
            trend_score = 0.25

        return recent_score * 0.7 + trend_score * 0.3

    def calculate_rs_momentum(
        self,
        rs_current: float,
        rs_5d_ago: float,
        rs_20d_ago: float,
    ) -> float:
        """
        Calculate relative strength momentum score.

        Args:
            rs_current: Current RS vs SPY
            rs_5d_ago: RS 5 days ago
            rs_20d_ago: RS 20 days ago

        Returns:
            Score 0-1 (higher = strengthening RS)
        """
        # RS level score
        if rs_current > 1.15:
            level_score = 1.0
        elif rs_current > 1.05:
            level_score = 0.75
        elif rs_current > 0.95:
            level_score = 0.50
        elif rs_current > 0.85:
            level_score = 0.25
        else:
            level_score = 0.0

        # RS momentum (is it accelerating?)
        rs_change_5d = rs_current - rs_5d_ago
        rs_change_20d = rs_current - rs_20d_ago

        if rs_change_5d > 0.05 and rs_change_20d > 0.10:
            momentum_score = 1.0
        elif rs_change_5d > 0.02 and rs_change_20d > 0.05:
            momentum_score = 0.75
        elif rs_change_5d > 0:
            momentum_score = 0.50
        else:
            momentum_score = 0.25

        return level_score * 0.5 + momentum_score * 0.5

    def analyze_earnings_revisions(
        self,
        revision_trend: float,
        upgrade_downgrade_ratio: float,
    ) -> float:
        """
        Analyze earnings estimate revisions.

        Args:
            revision_trend: % change in consensus EPS (e.g., 0.05 = +5%)
            upgrade_downgrade_ratio: Ratio of upgrades to downgrades

        Returns:
            Score 0-1 (higher = more positive revisions)
        """
        # Revision trend score
        if revision_trend > 0.05:
            revision_score = 1.0
        elif revision_trend > 0.02:
            revision_score = 0.75
        elif revision_trend > 0:
            revision_score = 0.50
        elif revision_trend > -0.02:
            revision_score = 0.25
        else:
            revision_score = 0.0

        # Upgrade/downgrade ratio score
        if upgrade_downgrade_ratio > 2.0:
            ratio_score = 1.0
        elif upgrade_downgrade_ratio > 1.5:
            ratio_score = 0.75
        elif upgrade_downgrade_ratio > 1.0:
            ratio_score = 0.50
        elif upgrade_downgrade_ratio > 0.5:
            ratio_score = 0.25
        else:
            ratio_score = 0.0

        return revision_score * 0.6 + ratio_score * 0.4

    def detect_breakout(
        self,
        price_vs_50d_high: float,
        volume_vs_avg: float,
    ) -> float:
        """
        Detect technical breakout with volume confirmation.

        Args:
            price_vs_50d_high: Current price as % of 50-day high (1.0 = at high)
            volume_vs_avg: Volume ratio vs 20-day average

        Returns:
            Score 0-1 (higher = stronger breakout)
        """
        # Price breakout score
        if price_vs_50d_high >= 1.0:  # At or above 50d high
            price_score = 1.0
        elif price_vs_50d_high >= 0.98:
            price_score = 0.75
        elif price_vs_50d_high >= 0.95:
            price_score = 0.50
        elif price_vs_50d_high >= 0.90:
            price_score = 0.25
        else:
            price_score = 0.0

        # Volume confirmation
        if volume_vs_avg >= 2.0:
            volume_score = 1.0
        elif volume_vs_avg >= 1.5:
            volume_score = 0.75
        elif volume_vs_avg >= 1.2:
            volume_score = 0.50
        else:
            volume_score = 0.25

        # Breakout only strong with volume confirmation
        return price_score * 0.6 + volume_score * 0.4

    def predict_rotation(
        self,
        etf_flow_3d: float = 0.0,
        etf_flow_20d: float = 0.0,
        rs_current: float = 1.0,
        rs_5d_ago: float = 1.0,
        rs_20d_ago: float = 1.0,
        revision_trend: float = 0.0,
        upgrade_downgrade_ratio: float = 1.0,
        price_vs_50d_high: float = 0.95,
        volume_vs_avg: float = 1.0,
    ) -> Dict:
        """
        Predict sector rotation using all leading indicators.

        Returns:
            Dict with rotation prediction and confidence
        """
        # Calculate individual scores
        flow_score = self.calculate_flow_momentum(etf_flow_3d, etf_flow_20d)
        rs_score = self.calculate_rs_momentum(rs_current, rs_5d_ago, rs_20d_ago)
        revision_score = self.analyze_earnings_revisions(revision_trend, upgrade_downgrade_ratio)
        breakout_score = self.detect_breakout(price_vs_50d_high, volume_vs_avg)

        # Weighted composite score
        composite_score = (
            flow_score * 0.30 +
            rs_score * 0.25 +
            revision_score * 0.20 +
            breakout_score * 0.25
        )

        # Determine rotation status
        if composite_score >= self.ROTATING_IN_SOON_THRESHOLD:
            status = 'ROTATING_IN_SOON'
            action = 'BUY_ANTICIPATE'
            multiplier = 1.25
        elif composite_score >= self.STRENGTHENING_THRESHOLD:
            status = 'STRENGTHENING'
            action = 'BUY_NORMAL'
            multiplier = 1.10
        elif composite_score >= self.WEAKENING_THRESHOLD:
            status = 'NEUTRAL'
            action = 'HOLD'
            multiplier = 1.00
        elif composite_score >= self.ROTATING_OUT_SOON_THRESHOLD:
            status = 'WEAKENING'
            action = 'REDUCE'
            multiplier = 0.85
        else:
            status = 'ROTATING_OUT_SOON'
            action = 'SELL_ANTICIPATE'
            multiplier = 0.65

        return {
            'status': status,
            'composite_score': composite_score,
            'suggested_action': action,
            'position_multiplier': multiplier,
            'confidence': min(composite_score * 1.2, 1.0),
            'component_scores': {
                'flow_momentum': flow_score,
                'rs_momentum': rs_score,
                'earnings_revisions': revision_score,
                'breakout': breakout_score,
            },
        }


# ========== FIX 44: US Catalyst Detector (US/INTL MODEL ONLY) ==========
class USCatalystDetector:
    """
    Detect and rank profit catalysts for US stocks.

    FIX 44: US Catalyst Detector

    Scans for major profit catalysts and adjusts positions accordingly.
    """

    CATALYST_TYPES = {
        'FDA_APPROVAL': {
            'multiplier': 1.50,
            'keywords': ['fda approval', 'fda approves', 'fda clears', 'drug approved'],
            'sectors': ['XLV'],  # Healthcare
            'duration_days': 5,
        },
        'MERGERS_ACQUISITIONS': {
            'multiplier': 1.40,
            'keywords': ['merger', 'acquisition', 'acquires', 'buyout', 'takeover', 'm&a'],
            'sectors': None,  # All sectors
            'duration_days': 10,
        },
        'PRODUCT_LAUNCH': {
            'multiplier': 1.30,
            'keywords': ['launches', 'unveils', 'announces new', 'introduces', 'new product'],
            'sectors': ['XLK', 'XLY'],  # Tech, Consumer
            'duration_days': 7,
        },
        'ANALYST_UPGRADE': {
            'multiplier': 1.20,
            'keywords': ['upgrade', 'raises target', 'raises price', 'initiates coverage buy'],
            'sectors': None,
            'duration_days': 3,
        },
        'SHARE_BUYBACK': {
            'multiplier': 1.15,
            'keywords': ['buyback', 'repurchase', 'share repurchase'],
            'sectors': None,
            'duration_days': 14,
        },
        'GUIDANCE_RAISE': {
            'multiplier': 1.25,
            'keywords': ['raises guidance', 'raises outlook', 'increases guidance', 'beats and raises'],
            'sectors': None,
            'duration_days': 5,
        },
        'CONTRACT_WIN': {
            'multiplier': 1.20,
            'keywords': ['wins contract', 'awarded contract', 'secures deal', 'partnership'],
            'sectors': ['XLI', 'XLK'],  # Industrials, Tech
            'duration_days': 7,
        },
        'SHORT_SQUEEZE': {
            'multiplier': 1.60,
            'keywords': ['short squeeze', 'short interest', 'shorts covering'],
            'sectors': None,
            'duration_days': 3,
        },
        'DIVIDEND_INCREASE': {
            'multiplier': 1.15,
            'keywords': ['dividend increase', 'raises dividend', 'special dividend'],
            'sectors': None,
            'duration_days': 7,
        },
        'INSIDER_BUYING': {
            'multiplier': 1.20,
            'keywords': ['insider buying', 'ceo buys', 'director purchases'],
            'sectors': None,
            'duration_days': 14,
        },
    }

    # High-priority tickers for catalyst monitoring
    TOP_50_US_STOCKS = {
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ',
        'V', 'UNH', 'XOM', 'MA', 'PG', 'HD', 'CVX', 'MRK', 'ABBV', 'PFE',
        'KO', 'COST', 'PEP', 'WMT', 'MCD', 'CSCO', 'AVGO', 'TMO', 'ABT', 'CRM',
        'NKE', 'ORCL', 'DHR', 'LLY', 'TXN', 'AMD', 'NFLX', 'INTC', 'QCOM', 'ADBE',
        'BA', 'GE', 'CAT', 'UPS', 'HON', 'DE', 'RTX', 'LMT', 'GS', 'MS',
    }

    def __init__(self):
        self.detected_catalysts = {}  # Cache of detected catalysts

    def _contains_catalyst_keywords(self, text: str, catalyst_type: str) -> bool:
        """Check if text contains catalyst keywords."""
        text_lower = text.lower()
        keywords = self.CATALYST_TYPES[catalyst_type]['keywords']
        return any(kw in text_lower for kw in keywords)

    def analyze_headline(self, ticker: str, headline: str, headline_date: datetime = None) -> Optional[Dict]:
        """
        Analyze a single headline for catalysts.

        Args:
            ticker: Stock ticker
            headline: News headline text
            headline_date: When headline was published

        Returns:
            Catalyst dict if detected, None otherwise
        """
        if headline_date is None:
            headline_date = datetime.now()

        for cat_type, config in self.CATALYST_TYPES.items():
            if self._contains_catalyst_keywords(headline, cat_type):
                return {
                    'type': cat_type,
                    'ticker': ticker,
                    'multiplier': config['multiplier'],
                    'headline': headline,
                    'detected_at': headline_date,
                    'expires_at': headline_date + timedelta(days=config['duration_days']),
                    'is_active': True,
                }

        return None

    def detect_active_catalysts(
        self,
        ticker: str,
        headlines: List[str] = None,
        current_date: datetime = None,
    ) -> Dict:
        """
        Detect active catalysts for a ticker.

        Args:
            ticker: Stock ticker
            headlines: List of recent headlines (if available)
            current_date: Current date for expiration check

        Returns:
            Dict with catalyst detection results
        """
        if current_date is None:
            current_date = datetime.now()

        catalysts = []

        # Analyze provided headlines
        if headlines:
            for headline in headlines:
                result = self.analyze_headline(ticker, headline, current_date)
                if result:
                    catalysts.append(result)

        # Check cached catalysts that haven't expired
        if ticker in self.detected_catalysts:
            for cached_cat in self.detected_catalysts[ticker]:
                if cached_cat['expires_at'] > current_date:
                    catalysts.append(cached_cat)

        # Update cache
        if catalysts:
            self.detected_catalysts[ticker] = catalysts

        # Return best catalyst (highest multiplier)
        if catalysts:
            best_catalyst = max(catalysts, key=lambda x: x['multiplier'])
            return {
                'has_catalyst': True,
                'catalysts': catalysts,
                'best_catalyst': best_catalyst,
                'position_multiplier': best_catalyst['multiplier'],
                'confidence_boost': 0.10,
                'total_catalysts': len(catalysts),
            }

        return {
            'has_catalyst': False,
            'catalysts': [],
            'position_multiplier': 1.0,
            'confidence_boost': 0.0,
            'total_catalysts': 0,
        }

    def get_catalyst_adjustment(
        self,
        ticker: str,
        signal_type: str,
        headlines: List[str] = None,
    ) -> Tuple[float, str]:
        """
        Get position adjustment based on detected catalysts.

        Returns:
            (multiplier, reason)
        """
        detection = self.detect_active_catalysts(ticker, headlines)

        if not detection['has_catalyst']:
            return 1.0, "No active catalysts"

        best = detection['best_catalyst']
        multiplier = best['multiplier']

        # SELL signals should be cautious during positive catalysts
        if signal_type.upper() == 'SELL':
            # Reduce SELL confidence during positive catalyst
            multiplier = max(0.5, 1.0 / multiplier)
            return multiplier, f"Catalyst {best['type']}: Reduced SELL to {multiplier:.2f}x"

        return multiplier, f"Catalyst {best['type']}: {multiplier:.2f}x boost"


# ========== FIX 45: Enhanced Intraday with Volume Profile (US/INTL MODEL ONLY) ==========
class EnhancedIntradayOptimizer:
    """
    Volume Profile + Order Flow enhanced intraday timing.

    FIX 45: Enhanced Intraday with Volume Profile

    Adds Volume Profile analysis to intraday timing:
    - Point of Control (POC) - highest volume price
    - Value Area (70% of volume)
    - Low Volume Nodes - areas of easy price movement
    """

    # Volume Profile strategies
    VP_STRATEGIES = {
        'OPENING_RANGE_BREAKOUT': {
            'time_range': ('10:15', '10:45'),
            'description': 'Enter on break of first 30-min range',
            'buy_mult': 1.20,
            'sell_mult': 1.15,
            'stop_loss': 'Below opening range low',
        },
        'VALUE_AREA_FADE': {
            'time_range': ('11:00', '14:00'),
            'description': 'Fade to Point of Control from value area edge',
            'buy_mult': 1.15,
            'sell_mult': 1.10,
            'stop_loss': 'Beyond value area',
        },
        'LOW_VOLUME_NODE_BREAK': {
            'time_range': ('10:00', '15:00'),
            'description': 'Momentum entry through low volume node',
            'buy_mult': 1.25,
            'sell_mult': 1.20,
            'stop_loss': 'Tight (half-ATR)',
        },
        'POWER_HOUR_MOMENTUM': {
            'time_range': ('15:00', '15:30'),
            'description': 'Ride institutional end-of-day flows',
            'buy_mult': 1.30,
            'sell_mult': 1.25,
            'stop_loss': 'Trailing (2x ATR)',
        },
        'POC_REVERSION': {
            'time_range': ('12:00', '14:30'),
            'description': 'Mean reversion to Point of Control',
            'buy_mult': 1.15,
            'sell_mult': 1.10,
            'stop_loss': 'Beyond 2x ATR from POC',
        },
    }

    def __init__(self):
        pass

    def calculate_volume_profile(
        self,
        prices: List[float],
        volumes: List[float],
        num_bins: int = 20,
    ) -> Dict:
        """
        Calculate basic volume profile from price/volume data.

        Args:
            prices: List of prices (high granularity preferred)
            volumes: Corresponding volumes
            num_bins: Number of price bins

        Returns:
            Volume profile dict
        """
        if not prices or not volumes or len(prices) != len(volumes):
            return {
                'point_of_control': sum(prices) / len(prices) if prices else 0,
                'value_area_high': max(prices) if prices else 0,
                'value_area_low': min(prices) if prices else 0,
                'low_volume_nodes': [],
            }

        import numpy as np

        prices = np.array(prices)
        volumes = np.array(volumes)

        # Create price bins
        price_min, price_max = prices.min(), prices.max()
        bins = np.linspace(price_min, price_max, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Calculate volume at each price level
        volume_at_price = np.zeros(num_bins)
        for i, (p, v) in enumerate(zip(prices, volumes)):
            bin_idx = min(num_bins - 1, max(0, int((p - price_min) / (price_max - price_min) * num_bins)))
            volume_at_price[bin_idx] += v

        # Point of Control (highest volume)
        poc_idx = np.argmax(volume_at_price)
        poc = bin_centers[poc_idx]

        # Value Area (70% of total volume)
        total_vol = volume_at_price.sum()
        target_vol = total_vol * 0.70

        # Expand from POC until we capture 70%
        va_vol = volume_at_price[poc_idx]
        low_idx, high_idx = poc_idx, poc_idx

        while va_vol < target_vol:
            # Check which side to expand
            low_vol = volume_at_price[low_idx - 1] if low_idx > 0 else 0
            high_vol = volume_at_price[high_idx + 1] if high_idx < num_bins - 1 else 0

            if low_vol >= high_vol and low_idx > 0:
                low_idx -= 1
                va_vol += low_vol
            elif high_idx < num_bins - 1:
                high_idx += 1
                va_vol += high_vol
            else:
                break

        value_area_high = bin_centers[high_idx]
        value_area_low = bin_centers[low_idx]

        # Low Volume Nodes (bins with <10% of POC volume)
        poc_vol = volume_at_price[poc_idx]
        low_volume_threshold = poc_vol * 0.10
        low_volume_nodes = [
            bin_centers[i] for i in range(num_bins)
            if volume_at_price[i] < low_volume_threshold and i != poc_idx
        ]

        return {
            'point_of_control': poc,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'low_volume_nodes': low_volume_nodes,
            'volume_distribution': volume_at_price.tolist(),
            'bin_centers': bin_centers.tolist(),
        }

    def get_volume_profile_strategy(
        self,
        current_price: float,
        volume_profile: Dict,
        current_time: datetime = None,
    ) -> Dict:
        """
        Determine best strategy based on volume profile.

        Returns:
            Strategy recommendation
        """
        if current_time is None:
            current_time = datetime.now()

        time_str = current_time.strftime('%H:%M')
        poc = volume_profile.get('point_of_control', current_price)
        va_high = volume_profile.get('value_area_high', current_price * 1.01)
        va_low = volume_profile.get('value_area_low', current_price * 0.99)
        low_vol_nodes = volume_profile.get('low_volume_nodes', [])

        # Determine price position
        if current_price >= va_high:
            price_position = 'ABOVE_VALUE'
        elif current_price <= va_low:
            price_position = 'BELOW_VALUE'
        elif abs(current_price - poc) < (va_high - va_low) * 0.1:
            price_position = 'AT_POC'
        else:
            price_position = 'IN_VALUE_AREA'

        # Check if near low volume node
        near_lvn = any(abs(current_price - lvn) < (va_high - va_low) * 0.05 for lvn in low_vol_nodes)

        # Determine strategy based on time and position
        if '10:15' <= time_str <= '10:45':
            strategy = 'OPENING_RANGE_BREAKOUT'
        elif '15:00' <= time_str <= '15:30':
            strategy = 'POWER_HOUR_MOMENTUM'
        elif near_lvn:
            strategy = 'LOW_VOLUME_NODE_BREAK'
        elif price_position in ['ABOVE_VALUE', 'BELOW_VALUE']:
            strategy = 'VALUE_AREA_FADE'
        else:
            strategy = 'POC_REVERSION'

        config = self.VP_STRATEGIES[strategy]

        return {
            'strategy': strategy,
            'description': config['description'],
            'buy_multiplier': config['buy_mult'],
            'sell_multiplier': config['sell_mult'],
            'stop_loss_type': config['stop_loss'],
            'price_position': price_position,
            'near_lvn': near_lvn,
            'point_of_control': poc,
            'value_area': (va_low, va_high),
        }

    def get_enhanced_timing_boost(
        self,
        signal_type: str,
        current_price: float = None,
        prices: List[float] = None,
        volumes: List[float] = None,
        current_time: datetime = None,
    ) -> Tuple[float, Dict]:
        """
        Get enhanced timing boost using volume profile.

        Returns:
            (multiplier, strategy_details)
        """
        if current_time is None:
            current_time = datetime.now()

        # Calculate volume profile if data provided
        if prices and volumes:
            vp = self.calculate_volume_profile(prices, volumes)
        else:
            # Use default values
            vp = {
                'point_of_control': current_price or 100,
                'value_area_high': (current_price or 100) * 1.02,
                'value_area_low': (current_price or 100) * 0.98,
                'low_volume_nodes': [],
            }

        strategy = self.get_volume_profile_strategy(
            current_price or vp['point_of_control'],
            vp,
            current_time,
        )

        if signal_type.upper() == 'BUY':
            mult = strategy['buy_multiplier']
        else:
            mult = strategy['sell_multiplier']

        return mult, strategy


# ========== FIX 46: Momentum Acceleration Detector (US/INTL MODEL ONLY) ==========
class MomentumAccelerationDetector:
    """
    Detect momentum acceleration for maximum profit capture.

    FIX 46: Momentum Acceleration Detector

    Uses 2nd derivative of price to detect when momentum is:
    - ACCELERATING: Add to position (up to 1.65x)
    - STEADY: Hold position (1.0x)
    - DECELERATING: Reduce position (down to 0.5x)
    """

    # Acceleration thresholds
    STRONG_ACCELERATION = 0.50
    MODERATE_ACCELERATION = 0.30
    MODERATE_DECELERATION = -0.30
    STRONG_DECELERATION = -0.50

    # Position adjustments
    ADJUSTMENTS = {
        'STRONG_ACCELERATING_UP': {'multiplier': 1.65, 'action': 'ADD_AGGRESSIVELY'},
        'ACCELERATING_UP': {'multiplier': 1.35, 'action': 'ADD_TO_POSITION'},
        'STEADY_UP': {'multiplier': 1.10, 'action': 'HOLD_POSITION'},
        'STEADY': {'multiplier': 1.00, 'action': 'HOLD'},
        'STEADY_DOWN': {'multiplier': 0.90, 'action': 'HOLD_POSITION'},
        'DECELERATING': {'multiplier': 0.70, 'action': 'REDUCE_POSITION'},
        'STRONG_DECELERATING': {'multiplier': 0.50, 'action': 'REDUCE_AGGRESSIVELY'},
    }

    def __init__(self):
        pass

    def calculate_acceleration(
        self,
        prices: List[float],
        lookback: int = 10,
    ) -> Dict:
        """
        Calculate momentum acceleration from price data.

        Args:
            prices: List of closing prices (most recent last)
            lookback: Number of periods for calculation

        Returns:
            Acceleration analysis dict
        """
        import numpy as np

        if len(prices) < lookback + 5:
            return {
                'status': 'INSUFFICIENT_DATA',
                'acceleration_score': 0.0,
                'momentum': 0.0,
                'acceleration': 0.0,
            }

        prices = np.array(prices[-lookback - 5:])

        # Calculate returns
        returns = np.diff(prices) / prices[:-1]

        # First derivative: Momentum (rolling 5-period average of returns)
        momentum = np.convolve(returns, np.ones(5) / 5, mode='valid')

        # Second derivative: Acceleration (change in momentum)
        acceleration = np.diff(momentum)

        # Smooth acceleration
        if len(acceleration) >= 3:
            smoothed_accel = np.mean(acceleration[-3:])
        else:
            smoothed_accel = acceleration[-1] if len(acceleration) > 0 else 0

        # Convert to score (-1 to +1) using tanh
        accel_score = np.tanh(smoothed_accel * 100)

        # Get current momentum direction
        current_momentum = momentum[-1] if len(momentum) > 0 else 0

        return {
            'acceleration_score': float(accel_score),
            'momentum': float(current_momentum),
            'acceleration': float(smoothed_accel),
            'momentum_direction': 'UP' if current_momentum > 0 else 'DOWN',
        }

    def get_acceleration_status(
        self,
        prices: List[float],
        lookback: int = 10,
    ) -> Dict:
        """
        Determine acceleration status and recommended action.

        Returns:
            Status dict with multiplier and action
        """
        analysis = self.calculate_acceleration(prices, lookback)

        if analysis.get('status') == 'INSUFFICIENT_DATA':
            return {
                'status': 'INSUFFICIENT_DATA',
                'score': 0.0,
                'action': 'HOLD',
                'multiplier': 1.0,
                'reason': 'Insufficient price data',
            }

        accel_score = analysis['acceleration_score']
        momentum_dir = analysis['momentum_direction']

        # Determine status
        if accel_score >= self.STRONG_ACCELERATION:
            if momentum_dir == 'UP':
                status = 'STRONG_ACCELERATING_UP'
            else:
                status = 'DECELERATING'  # Slowing down decline
        elif accel_score >= self.MODERATE_ACCELERATION:
            if momentum_dir == 'UP':
                status = 'ACCELERATING_UP'
            else:
                status = 'STEADY_DOWN'
        elif accel_score <= self.STRONG_DECELERATION:
            if momentum_dir == 'DOWN':
                status = 'STRONG_DECELERATING'
            else:
                status = 'ACCELERATING_UP'  # Slowing ascent could flip
        elif accel_score <= self.MODERATE_DECELERATION:
            if momentum_dir == 'DOWN':
                status = 'DECELERATING'
            else:
                status = 'STEADY_UP'
        else:
            status = 'STEADY'

        config = self.ADJUSTMENTS.get(status, self.ADJUSTMENTS['STEADY'])

        return {
            'status': status,
            'score': accel_score,
            'action': config['action'],
            'multiplier': config['multiplier'],
            'momentum': analysis['momentum'],
            'acceleration': analysis['acceleration'],
            'reason': f"Momentum {status.lower().replace('_', ' ')}: score={accel_score:.2f}",
        }

    def get_acceleration_adjustment(
        self,
        signal_type: str,
        prices: List[float],
    ) -> Tuple[float, str]:
        """
        Get position adjustment based on momentum acceleration.

        Returns:
            (multiplier, reason)
        """
        status = self.get_acceleration_status(prices)

        multiplier = status['multiplier']

        # For SELL signals, invert the logic
        if signal_type.upper() == 'SELL':
            # Strong acceleration UP = bad for SELL
            if status['status'] in ['STRONG_ACCELERATING_UP', 'ACCELERATING_UP']:
                multiplier = 1.0 / multiplier  # Reduce SELL
            # Strong deceleration = good for SELL
            elif status['status'] in ['STRONG_DECELERATING', 'DECELERATING']:
                multiplier = multiplier  # Boost SELL

        return multiplier, status['reason']


# ========== FIX 47: US-Specific Profit Rules (US/INTL MODEL ONLY) ==========
class USProfitRules:
    """
    US-specific profit maximization rules by stock type.

    FIX 47: US-Specific Profit Rules

    Different strategies for different stock types:
    - MEGA_CAP_TECH: Trend following, let winners run
    - HIGH_MOMENTUM: Quick profit taking
    - DIVIDEND_VALUE: Conservative, earnings focused
    - SMALL_CAP_MOMENTUM: Aggressive, volume confirmed
    """

    STOCK_PROFILES = {
        'MEGA_CAP_TECH': {
            'tickers': {'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META'},
            'optimal_holding_days': (5, 20),
            'profit_targets': [0.08, 0.15, 0.25],
            'let_winners_run': True,
            'trailing_stop_atr_mult': 2.5,
            'catalyst_aware': True,
            'sector_rotation_sensitive': True,
            'buy_multiplier': 1.25,
            'sell_multiplier': 0.85,
        },
        'HIGH_MOMENTUM': {
            'tickers': {'TSLA', 'AMD', 'NFLX', 'COIN', 'SQ', 'SHOP'},
            'optimal_holding_days': (3, 10),
            'profit_targets': [0.12, 0.25, 0.40],
            'let_winners_run': False,
            'trailing_stop_atr_mult': 3.0,
            'volatility_adjusted': True,
            'momentum_acceleration': True,
            'buy_multiplier': 1.30,
            'sell_multiplier': 1.10,
        },
        'DIVIDEND_VALUE': {
            'tickers': {'JNJ', 'PG', 'KO', 'PEP', 'MCD', 'WMT', 'VZ', 'T'},
            'optimal_holding_days': (20, 60),
            'profit_targets': [0.06, 0.12, 0.20],
            'let_winners_run': True,
            'trailing_stop_atr_mult': 2.0,
            'earnings_season_focus': True,
            'sector_rotation_sensitive': False,
            'buy_multiplier': 1.10,
            'sell_multiplier': 0.75,
        },
        'SMALL_CAP_MOMENTUM': {
            'tickers': set(),  # Dynamically identified
            'optimal_holding_days': (2, 7),
            'profit_targets': [0.15, 0.30, 0.50],
            'let_winners_run': False,
            'trailing_stop_atr_mult': 4.0,
            'volume_confirmation': True,
            'min_volume_ratio': 2.0,
            'gap_trading_enabled': True,
            'buy_multiplier': 1.15,
            'sell_multiplier': 1.20,
        },
        'FINANCIALS': {
            'tickers': {'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW'},
            'optimal_holding_days': (5, 30),
            'profit_targets': [0.08, 0.15, 0.25],
            'let_winners_run': True,
            'trailing_stop_atr_mult': 2.0,
            'rate_sensitive': True,
            'fomc_aware': True,
            'buy_multiplier': 1.15,
            'sell_multiplier': 0.90,
        },
        'ENERGY': {
            'tickers': {'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY'},
            'optimal_holding_days': (5, 20),
            'profit_targets': [0.10, 0.20, 0.35],
            'let_winners_run': False,
            'trailing_stop_atr_mult': 3.0,
            'commodity_correlated': True,
            'buy_multiplier': 1.10,
            'sell_multiplier': 1.05,
        },
    }

    def __init__(self):
        pass

    def classify_stock(self, ticker: str, market_cap: float = None) -> str:
        """
        Classify a stock into its profile type.

        Args:
            ticker: Stock ticker
            market_cap: Market cap in dollars (optional)

        Returns:
            Profile type string
        """
        ticker = ticker.upper()

        # Check explicit ticker lists first
        for profile_type, config in self.STOCK_PROFILES.items():
            if ticker in config.get('tickers', set()):
                return profile_type

        # Small cap detection by market cap
        if market_cap and market_cap < 2_000_000_000:
            return 'SMALL_CAP_MOMENTUM'

        # Default to generic
        return 'MEGA_CAP_TECH'  # Default conservative profile

    def get_profit_rules(self, ticker: str, market_cap: float = None) -> Dict:
        """
        Get profit rules for a ticker.

        Returns:
            Dict with all profit rules for the stock type
        """
        profile_type = self.classify_stock(ticker, market_cap)
        profile = self.STOCK_PROFILES.get(profile_type, self.STOCK_PROFILES['MEGA_CAP_TECH'])

        return {
            'profile_type': profile_type,
            **profile,
        }

    def get_position_adjustment(
        self,
        ticker: str,
        signal_type: str,
        market_cap: float = None,
    ) -> Tuple[float, Dict]:
        """
        Get position adjustment based on stock profile.

        Returns:
            (multiplier, rules_dict)
        """
        rules = self.get_profit_rules(ticker, market_cap)

        if signal_type.upper() == 'BUY':
            mult = rules.get('buy_multiplier', 1.0)
        else:
            mult = rules.get('sell_multiplier', 1.0)

        return mult, rules


# ========== FIX 48: Smart Profit Taker (US/INTL MODEL ONLY) ==========
class SmartProfitTaker:
    """
    Intelligent profit-taking based on 10+ factors.

    FIX 48: Smart Profit Taker

    Decision matrix considering:
    - Current profit percentage
    - Days held
    - Momentum acceleration
    - Sector rotation status
    - Market regime
    - VIX term structure
    - Intraday timing
    - Earnings proximity
    - Volume trend
    - Institutional flow
    """

    # Factor weights
    FACTOR_WEIGHTS = {
        'profit_level': 0.20,
        'days_held': 0.10,
        'momentum_accel': 0.15,
        'sector_rotation': 0.10,
        'market_regime': 0.10,
        'vix_structure': 0.08,
        'intraday_timing': 0.07,
        'earnings_proximity': 0.08,
        'volume_trend': 0.06,
        'institutional_flow': 0.06,
    }

    # Action thresholds
    FULL_PROFIT_THRESHOLD = 0.80
    PARTIAL_75_THRESHOLD = 0.60
    PARTIAL_50_THRESHOLD = 0.40
    PARTIAL_25_THRESHOLD = 0.25

    def __init__(self):
        pass

    def _score_profit_level(self, profit_pct: float, targets: List[float]) -> float:
        """Score based on profit percentage vs targets."""
        if profit_pct >= targets[2]:  # Hit final target
            return 1.0
        elif profit_pct >= targets[1]:
            return 0.8
        elif profit_pct >= targets[0]:
            return 0.6
        elif profit_pct > 0:
            return 0.3
        else:
            return 0.0

    def _score_days_held(self, days: int, optimal_range: Tuple[int, int]) -> float:
        """Score based on holding period."""
        min_days, max_days = optimal_range

        if days < min_days:
            return 0.2  # Too early
        elif days <= max_days:
            return 0.5  # In optimal window
        elif days <= max_days * 1.5:
            return 0.7  # Starting to overstay
        else:
            return 0.9  # Time to exit

    def _score_momentum_acceleration(self, accel_status: Dict) -> float:
        """Score based on momentum acceleration."""
        status = accel_status.get('status', 'STEADY')

        if status in ['STRONG_DECELERATING', 'DECELERATING']:
            return 0.9  # Exit signal
        elif status == 'STEADY_DOWN':
            return 0.7
        elif status == 'STEADY':
            return 0.5
        elif status == 'STEADY_UP':
            return 0.3
        else:
            return 0.1  # Strong momentum, hold

    def _score_sector_rotation(self, rotation_status: str) -> float:
        """Score based on sector rotation."""
        scores = {
            'ROTATING_OUT_SOON': 0.9,
            'WEAKENING': 0.7,
            'NEUTRAL': 0.5,
            'STRENGTHENING': 0.3,
            'ROTATING_IN_SOON': 0.1,
        }
        return scores.get(rotation_status, 0.5)

    def _score_market_regime(self, regime: str) -> float:
        """Score based on market regime."""
        # Higher score = more reason to take profit
        scores = {
            'bear_momentum': 0.9,
            'fomc_week': 0.8,
            'opex_week': 0.7,
            'high_volatility': 0.7,
            'bear_rally': 0.6,
            'neutral': 0.5,
            'bull_consolidation': 0.4,
            'sector_rotation': 0.4,
            'earnings_season': 0.3,
            'bull_momentum': 0.2,
        }
        return scores.get(regime, 0.5)

    def _score_vix_structure(self, vix_regime: str) -> float:
        """Score based on VIX term structure."""
        scores = {
            'strong_backwardation': 0.9,
            'moderate_backwardation': 0.7,
            'flat': 0.5,
            'moderate_contango': 0.3,
            'strong_contango': 0.2,
        }
        return scores.get(vix_regime, 0.5)

    def _score_intraday_timing(self, time_str: str) -> float:
        """Score based on intraday timing."""
        # Better to exit during power hour or avoid close
        if '15:00' <= time_str <= '15:45':
            return 0.8  # Good exit time
        elif '15:45' <= time_str <= '16:00':
            return 0.6  # Near close
        elif '09:30' <= time_str <= '10:00':
            return 0.3  # Opening volatility
        else:
            return 0.5

    def _score_earnings_proximity(self, days_to_earnings: int) -> float:
        """Score based on earnings proximity."""
        if days_to_earnings <= 2:
            return 0.9  # Exit before earnings
        elif days_to_earnings <= 5:
            return 0.7
        elif days_to_earnings <= 10:
            return 0.5
        else:
            return 0.3

    def _score_volume_trend(self, volume_ratio: float, is_declining: bool) -> float:
        """Score based on volume trend."""
        if is_declining and volume_ratio < 0.7:
            return 0.8  # Decreasing interest
        elif is_declining:
            return 0.6
        elif volume_ratio > 1.5:
            return 0.3  # Strong interest
        else:
            return 0.5

    def _score_institutional_flow(self, flow_pct: float) -> float:
        """Score based on institutional flow."""
        if flow_pct < -0.02:  # Outflows
            return 0.9
        elif flow_pct < 0:
            return 0.7
        elif flow_pct > 0.02:
            return 0.2  # Strong inflows
        else:
            return 0.5

    def should_take_profit(
        self,
        profit_pct: float,
        days_held: int,
        profit_targets: List[float] = None,
        optimal_holding: Tuple[int, int] = None,
        momentum_status: Dict = None,
        rotation_status: str = None,
        market_regime: str = None,
        vix_regime: str = None,
        current_time: str = None,
        days_to_earnings: int = None,
        volume_ratio: float = None,
        volume_declining: bool = None,
        institutional_flow: float = None,
    ) -> Dict:
        """
        Multi-factor profit-taking decision.

        Returns:
            Dict with action and confidence
        """
        # Defaults
        profit_targets = profit_targets or [0.08, 0.15, 0.25]
        optimal_holding = optimal_holding or (5, 20)
        current_time = current_time or datetime.now().strftime('%H:%M')

        # Calculate scores
        scores = {}

        scores['profit_level'] = self._score_profit_level(profit_pct, profit_targets)
        scores['days_held'] = self._score_days_held(days_held, optimal_holding)

        if momentum_status:
            scores['momentum_accel'] = self._score_momentum_acceleration(momentum_status)
        else:
            scores['momentum_accel'] = 0.5

        if rotation_status:
            scores['sector_rotation'] = self._score_sector_rotation(rotation_status)
        else:
            scores['sector_rotation'] = 0.5

        if market_regime:
            scores['market_regime'] = self._score_market_regime(market_regime)
        else:
            scores['market_regime'] = 0.5

        if vix_regime:
            scores['vix_structure'] = self._score_vix_structure(vix_regime)
        else:
            scores['vix_structure'] = 0.5

        scores['intraday_timing'] = self._score_intraday_timing(current_time)

        if days_to_earnings is not None:
            scores['earnings_proximity'] = self._score_earnings_proximity(days_to_earnings)
        else:
            scores['earnings_proximity'] = 0.5

        if volume_ratio is not None:
            scores['volume_trend'] = self._score_volume_trend(
                volume_ratio, volume_declining or False
            )
        else:
            scores['volume_trend'] = 0.5

        if institutional_flow is not None:
            scores['institutional_flow'] = self._score_institutional_flow(institutional_flow)
        else:
            scores['institutional_flow'] = 0.5

        # Calculate weighted composite score
        composite_score = sum(
            scores[factor] * weight
            for factor, weight in self.FACTOR_WEIGHTS.items()
        )

        # Determine action
        if composite_score >= self.FULL_PROFIT_THRESHOLD:
            action = 'TAKE_FULL_PROFIT'
            close_pct = 1.0
        elif composite_score >= self.PARTIAL_75_THRESHOLD:
            action = 'TAKE_PARTIAL_75'
            close_pct = 0.75
        elif composite_score >= self.PARTIAL_50_THRESHOLD:
            action = 'TAKE_PARTIAL_50'
            close_pct = 0.50
        elif composite_score >= self.PARTIAL_25_THRESHOLD:
            action = 'TAKE_PARTIAL_25'
            close_pct = 0.25
        else:
            action = 'HOLD'
            close_pct = 0.0

        return {
            'action': action,
            'close_percentage': close_pct,
            'composite_score': composite_score,
            'factor_scores': scores,
            'confidence': min(composite_score * 1.2, 1.0),
            'reason': f"Score={composite_score:.2f}, top factor={max(scores, key=scores.get)}",
        }


# ========== FIX 49: Backtest Profit Maximizer (US/INTL MODEL ONLY) ==========
class BacktestProfitMaximizer:
    """
    Aggressive profit-maximizing strategies for backtesting ONLY.

    FIX 49: Backtest Profit Maximizer

    These strategies are too risky for live trading but show
    theoretical maximum performance potential.
    """

    BACKTEST_STRATEGIES = {
        'PERFECT_ENTRY_TIMING': {
            'description': 'Assume perfect entry at optimal intraday time',
            'implementation': 'Use 1-minute data for exact entry',
            'expected_boost': 0.20,  # +20% returns
            'risk_level': 'MEDIUM',
        },
        'CATALYST_FORECAST': {
            'description': 'Position before known earnings/events',
            'implementation': 'Enter 3 days before catalyst',
            'expected_boost': 0.30,
            'risk_level': 'HIGH',
        },
        'SECTOR_ROTATION_PREDICTION': {
            'description': 'Perfect sector rotation timing',
            'implementation': 'Enter rotating sectors 5 days early',
            'expected_boost': 0.30,
            'risk_level': 'HIGH',
        },
        'MAXIMUM_CONCENTRATION': {
            'description': '100% in top signal',
            'implementation': 'Single position with full capital',
            'expected_boost': 0.40,
            'risk_level': 'EXTREME',
        },
        'PERFECT_EXIT': {
            'description': 'Exit at exact local maximum',
            'implementation': 'Use hindsight for optimal exit',
            'expected_boost': 0.35,
            'risk_level': 'HIGH',
        },
        'MOMENTUM_STACKING': {
            'description': 'Double down on accelerating winners',
            'implementation': 'Add 50% on momentum confirmation',
            'expected_boost': 0.25,
            'risk_level': 'HIGH',
        },
    }

    def __init__(self):
        self.strategy_results = {}

    def apply_strategy(
        self,
        strategy_name: str,
        base_return: float,
    ) -> Dict:
        """
        Apply a backtest-only strategy to base return.

        Args:
            strategy_name: Name of strategy to apply
            base_return: Base return without strategy

        Returns:
            Dict with enhanced return details
        """
        if strategy_name not in self.BACKTEST_STRATEGIES:
            return {
                'enhanced_return': base_return,
                'boost_applied': 0.0,
                'error': f"Unknown strategy: {strategy_name}",
            }

        config = self.BACKTEST_STRATEGIES[strategy_name]
        boost = config['expected_boost']

        # Apply boost multiplicatively
        enhanced_return = base_return * (1 + boost)

        return {
            'strategy': strategy_name,
            'base_return': base_return,
            'enhanced_return': enhanced_return,
            'boost_applied': boost,
            'risk_level': config['risk_level'],
            'description': config['description'],
        }

    def run_max_profit_backtest(
        self,
        base_returns: List[float],
        strategies_to_apply: List[str] = None,
    ) -> Dict:
        """
        Run backtest with multiple profit-maximizing strategies.

        Args:
            base_returns: List of base trade returns
            strategies_to_apply: Which strategies to use (default: all)

        Returns:
            Comprehensive backtest results
        """
        if strategies_to_apply is None:
            strategies_to_apply = list(self.BACKTEST_STRATEGIES.keys())

        results = {
            'base_performance': {
                'total_return': sum(base_returns),
                'avg_return': sum(base_returns) / len(base_returns) if base_returns else 0,
                'num_trades': len(base_returns),
            },
            'strategy_results': {},
        }

        import numpy as np

        for strategy in strategies_to_apply:
            enhanced_returns = [
                self.apply_strategy(strategy, r)['enhanced_return']
                for r in base_returns
            ]

            results['strategy_results'][strategy] = {
                'total_return': sum(enhanced_returns),
                'avg_return': sum(enhanced_returns) / len(enhanced_returns) if enhanced_returns else 0,
                'improvement_vs_base': sum(enhanced_returns) - sum(base_returns),
                'improvement_pct': (sum(enhanced_returns) / sum(base_returns) - 1) * 100 if sum(base_returns) != 0 else 0,
                'sharpe_estimate': np.mean(enhanced_returns) / np.std(enhanced_returns) if len(enhanced_returns) > 1 and np.std(enhanced_returns) > 0 else 0,
            }

        # Combined theoretical maximum
        max_boost = sum(
            self.BACKTEST_STRATEGIES[s]['expected_boost']
            for s in strategies_to_apply
        )
        max_returns = [r * (1 + max_boost) for r in base_returns]

        results['theoretical_maximum'] = {
            'total_return': sum(max_returns),
            'total_boost': max_boost,
            'improvement_vs_base': sum(max_returns) - sum(base_returns),
            'warning': 'THEORETICAL ONLY - Not achievable in live trading',
        }

        return results

    def get_strategy_recommendations(self, risk_tolerance: str = 'MEDIUM') -> List[str]:
        """
        Get recommended strategies based on risk tolerance.

        Args:
            risk_tolerance: 'LOW', 'MEDIUM', 'HIGH', or 'EXTREME'

        Returns:
            List of recommended strategy names
        """
        risk_levels = {
            'LOW': ['PERFECT_ENTRY_TIMING'],
            'MEDIUM': ['PERFECT_ENTRY_TIMING', 'MOMENTUM_STACKING'],
            'HIGH': ['PERFECT_ENTRY_TIMING', 'MOMENTUM_STACKING', 'SECTOR_ROTATION_PREDICTION', 'CATALYST_FORECAST'],
            'EXTREME': list(self.BACKTEST_STRATEGIES.keys()),
        }

        return risk_levels.get(risk_tolerance, risk_levels['MEDIUM'])


# ========== FIX 50: US Market Structure Arbitrage (US/INTL MODEL ONLY) ==========
class MarketStructureArbitrage:
    """
    Exploit US market structure inefficiencies for alpha generation.

    FIX 50: US Market Structure Arbitrage

    The problem: Model doesn't exploit known market structure inefficiencies
    unique to US markets (ETF creation/redemption, closing auctions, gamma).

    Solution: Detect and exploit:
    1. ETF premium/discount to NAV (creation/redemption arbitrage)
    2. Closing auction imbalances (MOC order flow)
    3. Dark pool print analysis (block trade detection)
    4. Options gamma exposure imbalances (dealer hedging flows)

    Expected Alpha: +5-10% with low risk
    """

    # ETF arbitrage thresholds
    ETF_PREMIUM_THRESHOLD = 0.005  # 0.5% premium triggers sell
    ETF_DISCOUNT_THRESHOLD = -0.005  # 0.5% discount triggers buy
    ETF_EXTREME_THRESHOLD = 0.015  # 1.5% = strong signal

    # Closing auction imbalance thresholds
    MOC_IMBALANCE_THRESHOLD = 0.02  # 2% of avg volume
    MOC_EXTREME_THRESHOLD = 0.05  # 5% = strong signal

    # Gamma exposure thresholds (normalized)
    GAMMA_POSITIVE_THRESHOLD = 0.3  # Dealers long gamma = suppressed vol
    GAMMA_NEGATIVE_THRESHOLD = -0.3  # Dealers short gamma = amplified vol

    # Signal multipliers
    STRUCTURE_MULTIPLIERS = {
        'etf_discount_extreme': 1.8,
        'etf_discount': 1.4,
        'etf_premium_extreme': 1.8,  # For SELL signals
        'etf_premium': 1.4,
        'moc_buy_imbalance_extreme': 1.6,
        'moc_buy_imbalance': 1.3,
        'moc_sell_imbalance_extreme': 1.6,
        'moc_sell_imbalance': 1.3,
        'negative_gamma_buy': 1.5,  # Momentum amplified
        'negative_gamma_sell': 1.5,
        'positive_gamma': 0.8,  # Mean reversion expected
        'dark_pool_accumulation': 1.4,
        'dark_pool_distribution': 1.4,
    }

    # Major ETFs for arbitrage detection
    MAJOR_ETFS = {
        'SPY': {'index': '^GSPC', 'type': 'broad'},
        'QQQ': {'index': '^NDX', 'type': 'tech'},
        'IWM': {'index': '^RUT', 'type': 'small_cap'},
        'DIA': {'index': '^DJI', 'type': 'blue_chip'},
        'XLF': {'index': None, 'type': 'sector_financials'},
        'XLE': {'index': None, 'type': 'sector_energy'},
        'XLK': {'index': None, 'type': 'sector_tech'},
        'XLV': {'index': None, 'type': 'sector_healthcare'},
        'GLD': {'index': 'GC=F', 'type': 'commodity'},
        'SLV': {'index': 'SI=F', 'type': 'commodity'},
    }

    def __init__(self):
        self.etf_premium_history = {}
        self.moc_imbalance_history = {}
        self.gamma_exposure_history = {}
        self.dark_pool_prints = {}

    def calculate_etf_premium(
        self,
        etf_price: float,
        nav_estimate: float,
    ) -> Tuple[float, str]:
        """
        Calculate ETF premium/discount to NAV.

        Args:
            etf_price: Current ETF market price
            nav_estimate: Estimated Net Asset Value

        Returns:
            (premium_pct, signal_type)
        """
        if nav_estimate <= 0:
            return 0.0, 'neutral'

        premium = (etf_price - nav_estimate) / nav_estimate

        if premium >= self.ETF_EXTREME_THRESHOLD:
            return premium, 'etf_premium_extreme'
        elif premium >= self.ETF_PREMIUM_THRESHOLD:
            return premium, 'etf_premium'
        elif premium <= -self.ETF_EXTREME_THRESHOLD:
            return premium, 'etf_discount_extreme'
        elif premium <= self.ETF_DISCOUNT_THRESHOLD:
            return premium, 'etf_discount'
        else:
            return premium, 'neutral'

    def analyze_moc_imbalance(
        self,
        buy_imbalance: float,
        sell_imbalance: float,
        avg_volume: float,
    ) -> Tuple[float, str]:
        """
        Analyze Market-on-Close order imbalances.

        Args:
            buy_imbalance: Total MOC buy orders
            sell_imbalance: Total MOC sell orders
            avg_volume: Average daily volume

        Returns:
            (imbalance_ratio, signal_type)
        """
        if avg_volume <= 0:
            return 0.0, 'neutral'

        net_imbalance = buy_imbalance - sell_imbalance
        imbalance_ratio = net_imbalance / avg_volume

        if imbalance_ratio >= self.MOC_EXTREME_THRESHOLD:
            return imbalance_ratio, 'moc_buy_imbalance_extreme'
        elif imbalance_ratio >= self.MOC_IMBALANCE_THRESHOLD:
            return imbalance_ratio, 'moc_buy_imbalance'
        elif imbalance_ratio <= -self.MOC_EXTREME_THRESHOLD:
            return imbalance_ratio, 'moc_sell_imbalance_extreme'
        elif imbalance_ratio <= -self.MOC_IMBALANCE_THRESHOLD:
            return imbalance_ratio, 'moc_sell_imbalance'
        else:
            return imbalance_ratio, 'neutral'

    def analyze_gamma_exposure(
        self,
        dealer_gamma: float,
        spot_price: float,
    ) -> Tuple[float, str]:
        """
        Analyze options dealer gamma exposure.

        Negative gamma = dealers hedge by buying high, selling low (momentum)
        Positive gamma = dealers hedge by selling high, buying low (mean reversion)

        Args:
            dealer_gamma: Estimated aggregate dealer gamma
            spot_price: Current spot price

        Returns:
            (normalized_gamma, regime_type)
        """
        # Normalize gamma by spot price
        normalized_gamma = dealer_gamma / spot_price if spot_price > 0 else 0

        if normalized_gamma >= self.GAMMA_POSITIVE_THRESHOLD:
            return normalized_gamma, 'positive_gamma'
        elif normalized_gamma <= self.GAMMA_NEGATIVE_THRESHOLD:
            return normalized_gamma, 'negative_gamma'
        else:
            return normalized_gamma, 'neutral_gamma'

    def detect_dark_pool_activity(
        self,
        dark_pool_volume: float,
        total_volume: float,
        price_change: float,
    ) -> Tuple[float, str]:
        """
        Detect institutional dark pool activity patterns.

        Accumulation: High dark pool volume + price stable/rising
        Distribution: High dark pool volume + price stable/falling

        Args:
            dark_pool_volume: Volume traded in dark pools
            total_volume: Total daily volume
            price_change: Price change during period

        Returns:
            (dark_pool_ratio, activity_type)
        """
        if total_volume <= 0:
            return 0.0, 'neutral'

        dp_ratio = dark_pool_volume / total_volume

        # Dark pool typically ~40% of volume; >50% is notable
        if dp_ratio > 0.50:
            if price_change >= 0:
                return dp_ratio, 'dark_pool_accumulation'
            else:
                return dp_ratio, 'dark_pool_distribution'

        return dp_ratio, 'neutral'

    def get_structure_signal(
        self,
        ticker: str,
        signal_type: str,
        etf_premium: float = None,
        moc_imbalance: float = None,
        gamma_exposure: float = None,
        dark_pool_ratio: float = None,
        price_data: Dict = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive market structure signal.

        Args:
            ticker: Asset ticker
            signal_type: 'BUY' or 'SELL'
            etf_premium: ETF premium/discount (optional)
            moc_imbalance: MOC order imbalance ratio (optional)
            gamma_exposure: Dealer gamma exposure (optional)
            dark_pool_ratio: Dark pool volume ratio (optional)
            price_data: Dict with price info (optional)

        Returns:
            Dict with structure analysis and multipliers
        """
        is_buy = signal_type.upper() == 'BUY'
        multipliers = []
        signals = []
        explanations = []

        # ETF Premium/Discount
        if etf_premium is not None:
            _, etf_signal = self.calculate_etf_premium(1.0, 1.0 - etf_premium)
            if etf_signal != 'neutral':
                # Discount is good for BUY, premium is good for SELL
                if (is_buy and 'discount' in etf_signal) or \
                   (not is_buy and 'premium' in etf_signal):
                    mult = self.STRUCTURE_MULTIPLIERS.get(etf_signal, 1.0)
                    multipliers.append(mult)
                    signals.append(etf_signal)
                    explanations.append(f"ETF {etf_signal}: {etf_premium:.2%}")

        # MOC Imbalance
        if moc_imbalance is not None:
            if moc_imbalance > self.MOC_IMBALANCE_THRESHOLD:
                signal = 'moc_buy_imbalance_extreme' if moc_imbalance > self.MOC_EXTREME_THRESHOLD else 'moc_buy_imbalance'
                if is_buy:
                    mult = self.STRUCTURE_MULTIPLIERS.get(signal, 1.0)
                    multipliers.append(mult)
                    signals.append(signal)
                    explanations.append(f"MOC buy imbalance: {moc_imbalance:.2%}")
            elif moc_imbalance < -self.MOC_IMBALANCE_THRESHOLD:
                signal = 'moc_sell_imbalance_extreme' if moc_imbalance < -self.MOC_EXTREME_THRESHOLD else 'moc_sell_imbalance'
                if not is_buy:
                    mult = self.STRUCTURE_MULTIPLIERS.get(signal, 1.0)
                    multipliers.append(mult)
                    signals.append(signal)
                    explanations.append(f"MOC sell imbalance: {moc_imbalance:.2%}")

        # Gamma Exposure
        if gamma_exposure is not None:
            if gamma_exposure < self.GAMMA_NEGATIVE_THRESHOLD:
                # Negative gamma amplifies momentum
                signal = 'negative_gamma_buy' if is_buy else 'negative_gamma_sell'
                mult = self.STRUCTURE_MULTIPLIERS.get(signal, 1.0)
                multipliers.append(mult)
                signals.append(signal)
                explanations.append(f"Negative gamma regime: {gamma_exposure:.2f}")
            elif gamma_exposure > self.GAMMA_POSITIVE_THRESHOLD:
                # Positive gamma suppresses moves
                mult = self.STRUCTURE_MULTIPLIERS.get('positive_gamma', 1.0)
                multipliers.append(mult)
                signals.append('positive_gamma')
                explanations.append(f"Positive gamma (mean reversion): {gamma_exposure:.2f}")

        # Dark Pool Activity
        if dark_pool_ratio is not None and dark_pool_ratio > 0.50:
            price_change = price_data.get('price_change', 0) if price_data else 0
            if price_change >= 0 and is_buy:
                mult = self.STRUCTURE_MULTIPLIERS.get('dark_pool_accumulation', 1.0)
                multipliers.append(mult)
                signals.append('dark_pool_accumulation')
                explanations.append(f"Dark pool accumulation: {dark_pool_ratio:.1%}")
            elif price_change < 0 and not is_buy:
                mult = self.STRUCTURE_MULTIPLIERS.get('dark_pool_distribution', 1.0)
                multipliers.append(mult)
                signals.append('dark_pool_distribution')
                explanations.append(f"Dark pool distribution: {dark_pool_ratio:.1%}")

        # Combine multipliers (geometric mean for multiple signals)
        if multipliers:
            combined_multiplier = np.prod(multipliers) ** (1 / len(multipliers))
        else:
            combined_multiplier = 1.0

        return {
            'ticker': ticker,
            'signal_type': signal_type,
            'structure_signals': signals,
            'combined_multiplier': combined_multiplier,
            'individual_multipliers': multipliers,
            'explanations': explanations,
            'has_structure_edge': len(signals) > 0,
        }


# ========== FIX 51: Smart Beta Overlay (US/INTL MODEL ONLY) ==========
class SmartBetaOverlay:
    """
    Apply factor timing based on market regime for enhanced returns.

    FIX 51: Smart Beta Overlay

    The problem: Model doesn't adjust for factor exposures that perform
    differently across market regimes.

    Solution: Time factor exposures based on regime:
    - Momentum factor in trending markets (1.3x)
    - Value factor in recovery phases (1.2x)
    - Low volatility factor in uncertain times (0.8x)
    - Quality factor in late cycle (1.1x)
    - Size factor based on risk appetite

    Expected Alpha: +8-12% with diversified risk
    """

    # Factor definitions and their regime preferences
    FACTOR_REGIMES = {
        'momentum': {
            'description': 'Price momentum winners continue winning',
            'ideal_regimes': ['strong_uptrend', 'uptrend'],
            'avoid_regimes': ['crisis', 'high_volatility', 'reversal'],
            'multiplier_ideal': 1.40,
            'multiplier_neutral': 1.00,
            'multiplier_avoid': 0.60,
        },
        'value': {
            'description': 'Cheap stocks outperform expensive',
            'ideal_regimes': ['recovery', 'early_cycle', 'mean_reverting'],
            'avoid_regimes': ['strong_uptrend', 'bubble'],
            'multiplier_ideal': 1.30,
            'multiplier_neutral': 1.00,
            'multiplier_avoid': 0.70,
        },
        'low_volatility': {
            'description': 'Low vol stocks outperform high vol',
            'ideal_regimes': ['high_volatility', 'crisis', 'uncertain'],
            'avoid_regimes': ['strong_uptrend', 'recovery'],
            'multiplier_ideal': 1.20,
            'multiplier_neutral': 1.00,
            'multiplier_avoid': 0.80,
        },
        'quality': {
            'description': 'High quality companies outperform',
            'ideal_regimes': ['late_cycle', 'slowing', 'defensive'],
            'avoid_regimes': ['early_cycle', 'recovery'],
            'multiplier_ideal': 1.25,
            'multiplier_neutral': 1.05,  # Quality has positive baseline
            'multiplier_avoid': 0.90,
        },
        'size': {
            'description': 'Small caps vs large caps',
            'ideal_regimes': ['risk_on', 'early_cycle', 'recovery'],
            'avoid_regimes': ['crisis', 'risk_off', 'flight_to_safety'],
            'multiplier_ideal': 1.35,  # Small caps in risk-on
            'multiplier_neutral': 1.00,
            'multiplier_avoid': 0.65,  # Large caps in risk-off
        },
    }

    # Factor exposure by stock characteristics
    FACTOR_CHARACTERISTICS = {
        'momentum': {
            'momentum_20d': (0.10, float('inf')),  # >10% 20d return
            'momentum_60d': (0.15, float('inf')),  # >15% 60d return
            'rs_rank': (0.70, 1.00),  # Top 30% relative strength
        },
        'value': {
            'pe_ratio': (0, 15),  # Low P/E
            'pb_ratio': (0, 2),  # Low P/B
            'dividend_yield': (0.03, float('inf')),  # >3% yield
        },
        'low_volatility': {
            'volatility_20d': (0, 0.20),  # <20% annualized vol
            'beta': (0, 0.80),  # Low beta
        },
        'quality': {
            'roe': (0.15, float('inf')),  # >15% ROE
            'debt_to_equity': (0, 0.50),  # Low debt
            'profit_margin': (0.10, float('inf')),  # >10% margin
        },
        'size': {
            'market_cap': (0, 10e9),  # <$10B = small cap
        },
    }

    # Sector factor tilts
    SECTOR_FACTOR_TILTS = {
        'XLK': {'momentum': 1.2, 'quality': 1.1, 'value': 0.8},  # Tech
        'XLF': {'value': 1.3, 'quality': 0.9, 'momentum': 0.9},  # Financials
        'XLE': {'value': 1.2, 'momentum': 1.1, 'quality': 0.9},  # Energy
        'XLV': {'quality': 1.3, 'low_volatility': 1.1, 'momentum': 0.9},  # Healthcare
        'XLU': {'low_volatility': 1.4, 'value': 1.1, 'momentum': 0.7},  # Utilities
        'XLP': {'low_volatility': 1.3, 'quality': 1.2, 'momentum': 0.8},  # Staples
        'XLY': {'momentum': 1.2, 'size': 1.1, 'value': 0.9},  # Discretionary
        'XLI': {'value': 1.1, 'quality': 1.0, 'momentum': 1.0},  # Industrials
        'XLB': {'momentum': 1.1, 'value': 1.1, 'size': 1.0},  # Materials
        'XLRE': {'value': 1.2, 'low_volatility': 1.1, 'quality': 1.0},  # Real Estate
    }

    def __init__(self):
        self.current_regime = 'neutral'
        self.factor_scores = {}

    def set_market_regime(self, regime: str):
        """Set current market regime for factor timing."""
        self.current_regime = regime

    def calculate_factor_exposure(
        self,
        ticker: str,
        stock_characteristics: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate factor exposures for a stock.

        Args:
            ticker: Stock ticker
            stock_characteristics: Dict of stock metrics

        Returns:
            Dict of factor -> exposure score (0-1)
        """
        exposures = {}

        for factor, criteria in self.FACTOR_CHARACTERISTICS.items():
            score = 0.0
            matches = 0
            total_criteria = len(criteria)

            for metric, (low, high) in criteria.items():
                if metric in stock_characteristics:
                    value = stock_characteristics[metric]
                    if low <= value <= high:
                        matches += 1

            if total_criteria > 0:
                score = matches / total_criteria
            exposures[factor] = score

        return exposures

    def get_regime_factor_multiplier(
        self,
        factor: str,
        regime: str = None,
    ) -> float:
        """
        Get factor multiplier based on market regime.

        Args:
            factor: Factor name (momentum, value, etc.)
            regime: Market regime (or use current)

        Returns:
            Multiplier for this factor in current regime
        """
        regime = regime or self.current_regime

        if factor not in self.FACTOR_REGIMES:
            return 1.0

        config = self.FACTOR_REGIMES[factor]

        if regime in config['ideal_regimes']:
            return config['multiplier_ideal']
        elif regime in config['avoid_regimes']:
            return config['multiplier_avoid']
        else:
            return config['multiplier_neutral']

    def apply_factor_overlay(
        self,
        ticker: str,
        signal_type: str,
        base_multiplier: float,
        stock_characteristics: Dict[str, float] = None,
        sector_etf: str = None,
        regime: str = None,
    ) -> Dict[str, Any]:
        """
        Apply smart beta factor overlay to signal.

        Args:
            ticker: Stock ticker
            signal_type: 'BUY' or 'SELL'
            base_multiplier: Base position multiplier
            stock_characteristics: Dict of stock metrics
            sector_etf: Sector ETF ticker
            regime: Market regime override

        Returns:
            Dict with factor analysis and adjusted multiplier
        """
        regime = regime or self.current_regime
        is_buy = signal_type.upper() == 'BUY'

        # Calculate factor exposures
        if stock_characteristics:
            exposures = self.calculate_factor_exposure(ticker, stock_characteristics)
        else:
            exposures = {}

        # Get sector factor tilts
        sector_tilts = self.SECTOR_FACTOR_TILTS.get(sector_etf, {})

        # Calculate factor-adjusted multiplier
        factor_adjustments = {}
        total_weight = 0.0
        weighted_multiplier = 0.0

        for factor in self.FACTOR_REGIMES:
            # Base exposure from stock characteristics
            base_exposure = exposures.get(factor, 0.5)

            # Sector tilt adjustment
            sector_tilt = sector_tilts.get(factor, 1.0)

            # Regime multiplier for this factor
            regime_mult = self.get_regime_factor_multiplier(factor, regime)

            # Combined factor score
            factor_score = base_exposure * sector_tilt * regime_mult

            # Weight by exposure
            if base_exposure > 0.3:  # Only count significant exposures
                weight = base_exposure
                total_weight += weight
                weighted_multiplier += factor_score * weight

            factor_adjustments[factor] = {
                'exposure': base_exposure,
                'sector_tilt': sector_tilt,
                'regime_multiplier': regime_mult,
                'combined_score': factor_score,
            }

        # Calculate final multiplier
        if total_weight > 0:
            factor_multiplier = weighted_multiplier / total_weight
        else:
            factor_multiplier = 1.0

        # Apply to base multiplier
        adjusted_multiplier = base_multiplier * factor_multiplier

        # Determine dominant factor
        dominant_factor = max(factor_adjustments.items(),
                             key=lambda x: x[1]['combined_score'])[0] if factor_adjustments else None

        return {
            'ticker': ticker,
            'signal_type': signal_type,
            'regime': regime,
            'base_multiplier': base_multiplier,
            'factor_multiplier': factor_multiplier,
            'adjusted_multiplier': adjusted_multiplier,
            'factor_adjustments': factor_adjustments,
            'dominant_factor': dominant_factor,
            'explanation': f"Factor overlay ({regime}): {dominant_factor} dominant, {factor_multiplier:.2f}x adjustment",
        }


# ========== FIX 52: Volatility Regime Switching (US/INTL MODEL ONLY) ==========
class VolatilityRegimeSwitcher:
    """
    Switch strategies based on VIX regime for optimal risk-adjusted returns.

    FIX 52: Volatility Regime Switching

    The problem: Model uses same strategy regardless of volatility environment,
    leading to suboptimal risk-adjusted returns.

    Solution: Adapt strategy based on VIX regime:
    - VIX < 15: Trend following (1.4x position) - calm markets trend
    - VIX 15-25: Mean reversion (1.0x) - normal volatility
    - VIX 25-40: Volatility selling (0.6x) + tail hedge - elevated risk
    - VIX > 40: Crisis mode (0.3x) + extreme caution

    Expected Alpha: +10-15% with improved risk management
    """

    # VIX regime definitions
    VIX_REGIMES = {
        'ultra_low': {
            'range': (0, 12),
            'description': 'Complacency - trend following with caution',
            'strategy': 'trend_following_cautious',
            'position_multiplier': 1.20,
            'stop_loss_adjustment': 0.80,  # Tighter stops
            'profit_target_adjustment': 0.90,  # Lower targets
        },
        'low': {
            'range': (12, 15),
            'description': 'Calm markets - aggressive trend following',
            'strategy': 'trend_following',
            'position_multiplier': 1.40,
            'stop_loss_adjustment': 0.90,
            'profit_target_adjustment': 1.00,
        },
        'normal': {
            'range': (15, 20),
            'description': 'Normal volatility - balanced approach',
            'strategy': 'balanced',
            'position_multiplier': 1.00,
            'stop_loss_adjustment': 1.00,
            'profit_target_adjustment': 1.00,
        },
        'elevated': {
            'range': (20, 25),
            'description': 'Elevated volatility - mean reversion favored',
            'strategy': 'mean_reversion',
            'position_multiplier': 0.85,
            'stop_loss_adjustment': 1.20,  # Wider stops
            'profit_target_adjustment': 1.20,  # Higher targets
        },
        'high': {
            'range': (25, 35),
            'description': 'High volatility - reduced exposure',
            'strategy': 'defensive',
            'position_multiplier': 0.60,
            'stop_loss_adjustment': 1.50,
            'profit_target_adjustment': 1.50,
        },
        'extreme': {
            'range': (35, 50),
            'description': 'Extreme volatility - capital preservation',
            'strategy': 'capital_preservation',
            'position_multiplier': 0.35,
            'stop_loss_adjustment': 2.00,
            'profit_target_adjustment': 2.00,
        },
        'crisis': {
            'range': (50, float('inf')),
            'description': 'Crisis - minimal exposure only',
            'strategy': 'crisis_mode',
            'position_multiplier': 0.20,
            'stop_loss_adjustment': 2.50,
            'profit_target_adjustment': 3.00,
        },
    }

    # VIX term structure signals
    TERM_STRUCTURE_SIGNALS = {
        'contango_steep': {
            'threshold': 0.10,  # >10% contango
            'signal': 'risk_on',
            'multiplier': 1.15,
        },
        'contango_normal': {
            'threshold': 0.03,
            'signal': 'neutral',
            'multiplier': 1.00,
        },
        'flat': {
            'threshold': -0.03,
            'signal': 'cautious',
            'multiplier': 0.90,
        },
        'backwardation': {
            'threshold': -0.10,
            'signal': 'risk_off',
            'multiplier': 0.70,
        },
        'backwardation_extreme': {
            'threshold': -0.20,
            'signal': 'crisis_warning',
            'multiplier': 0.50,
        },
    }

    # Strategy-specific signal adjustments
    STRATEGY_SIGNAL_ADJUSTMENTS = {
        'trend_following': {
            'BUY_uptrend': 1.30,
            'BUY_downtrend': 0.60,
            'SELL_uptrend': 0.50,
            'SELL_downtrend': 1.30,
        },
        'trend_following_cautious': {
            'BUY_uptrend': 1.15,
            'BUY_downtrend': 0.70,
            'SELL_uptrend': 0.60,
            'SELL_downtrend': 1.15,
        },
        'mean_reversion': {
            'BUY_uptrend': 0.80,
            'BUY_downtrend': 1.20,
            'SELL_uptrend': 1.20,
            'SELL_downtrend': 0.80,
        },
        'balanced': {
            'BUY_uptrend': 1.00,
            'BUY_downtrend': 0.90,
            'SELL_uptrend': 0.90,
            'SELL_downtrend': 1.00,
        },
        'defensive': {
            'BUY_uptrend': 0.90,
            'BUY_downtrend': 0.50,
            'SELL_uptrend': 0.70,
            'SELL_downtrend': 1.10,
        },
        'capital_preservation': {
            'BUY_uptrend': 0.70,
            'BUY_downtrend': 0.30,
            'SELL_uptrend': 0.50,
            'SELL_downtrend': 0.80,
        },
        'crisis_mode': {
            'BUY_uptrend': 0.40,
            'BUY_downtrend': 0.20,
            'SELL_uptrend': 0.30,
            'SELL_downtrend': 0.50,
        },
    }

    def __init__(self):
        self.current_vix = 20.0
        self.current_regime = 'normal'
        self.vix_history = deque(maxlen=252)  # 1 year of daily VIX

    def update_vix(self, vix_value: float):
        """Update current VIX value."""
        self.current_vix = vix_value
        self.vix_history.append(vix_value)
        self.current_regime = self.get_vix_regime(vix_value)

    def get_vix_regime(self, vix: float = None) -> str:
        """
        Get VIX regime classification.

        Args:
            vix: VIX value (or use current)

        Returns:
            Regime name
        """
        vix = vix if vix is not None else self.current_vix

        for regime_name, config in self.VIX_REGIMES.items():
            low, high = config['range']
            if low <= vix < high:
                return regime_name

        return 'crisis'  # Default for extreme values

    def get_vix_percentile(self) -> float:
        """Get current VIX percentile based on history."""
        if len(self.vix_history) < 20:
            return 0.50  # Default to median

        sorted_vix = sorted(self.vix_history)
        percentile = sum(1 for v in sorted_vix if v <= self.current_vix) / len(sorted_vix)
        return percentile

    def calculate_term_structure_signal(
        self,
        vix_spot: float,
        vix_futures_1m: float,
    ) -> Dict[str, Any]:
        """
        Calculate VIX term structure signal.

        Args:
            vix_spot: Spot VIX
            vix_futures_1m: 1-month VIX futures

        Returns:
            Dict with term structure analysis
        """
        if vix_spot <= 0:
            return {'signal': 'neutral', 'multiplier': 1.0, 'spread': 0}

        spread = (vix_futures_1m - vix_spot) / vix_spot

        for signal_name, config in self.TERM_STRUCTURE_SIGNALS.items():
            if spread >= config['threshold']:
                return {
                    'signal': config['signal'],
                    'multiplier': config['multiplier'],
                    'spread': spread,
                    'structure': signal_name,
                }

        return {
            'signal': 'backwardation_extreme',
            'multiplier': 0.50,
            'spread': spread,
            'structure': 'backwardation_extreme',
        }

    def get_strategy_adjustment(
        self,
        signal_type: str,
        price_trend: str,
        regime: str = None,
    ) -> float:
        """
        Get strategy-specific signal adjustment.

        Args:
            signal_type: 'BUY' or 'SELL'
            price_trend: 'uptrend' or 'downtrend'
            regime: VIX regime override

        Returns:
            Signal multiplier
        """
        regime = regime or self.current_regime
        strategy = self.VIX_REGIMES.get(regime, {}).get('strategy', 'balanced')

        key = f"{signal_type}_{price_trend}"
        adjustments = self.STRATEGY_SIGNAL_ADJUSTMENTS.get(strategy, {})

        return adjustments.get(key, 1.0)

    def apply_volatility_regime(
        self,
        signal_type: str,
        base_position: float,
        base_stop_loss: float,
        base_profit_target: float,
        price_trend: str = 'neutral',
        vix: float = None,
        vix_futures: float = None,
    ) -> Dict[str, Any]:
        """
        Apply volatility regime adjustments to signal.

        Args:
            signal_type: 'BUY' or 'SELL'
            base_position: Base position size
            base_stop_loss: Base stop loss percentage
            base_profit_target: Base profit target percentage
            price_trend: Current price trend
            vix: VIX value override
            vix_futures: VIX futures for term structure

        Returns:
            Dict with regime-adjusted parameters
        """
        if vix is not None:
            self.update_vix(vix)

        regime = self.current_regime
        regime_config = self.VIX_REGIMES.get(regime, self.VIX_REGIMES['normal'])

        # Base regime adjustments
        position_mult = regime_config['position_multiplier']
        stop_mult = regime_config['stop_loss_adjustment']
        profit_mult = regime_config['profit_target_adjustment']

        # Strategy-specific adjustment
        strategy_mult = self.get_strategy_adjustment(signal_type, price_trend, regime)

        # Term structure adjustment
        term_mult = 1.0
        term_structure = None
        if vix_futures is not None and self.current_vix > 0:
            ts_result = self.calculate_term_structure_signal(self.current_vix, vix_futures)
            term_mult = ts_result['multiplier']
            term_structure = ts_result['structure']

        # Calculate final values
        final_position = base_position * position_mult * strategy_mult * term_mult
        final_stop_loss = base_stop_loss * stop_mult
        final_profit_target = base_profit_target * profit_mult

        # Cap position at reasonable levels
        final_position = min(final_position, base_position * 2.0)
        final_position = max(final_position, base_position * 0.10)

        return {
            'signal_type': signal_type,
            'vix': self.current_vix,
            'regime': regime,
            'strategy': regime_config['strategy'],
            'base_position': base_position,
            'adjusted_position': final_position,
            'position_multiplier': position_mult,
            'strategy_multiplier': strategy_mult,
            'term_structure_multiplier': term_mult,
            'term_structure': term_structure,
            'base_stop_loss': base_stop_loss,
            'adjusted_stop_loss': final_stop_loss,
            'base_profit_target': base_profit_target,
            'adjusted_profit_target': final_profit_target,
            'vix_percentile': self.get_vix_percentile(),
            'explanation': f"VIX {self.current_vix:.1f} ({regime}): {regime_config['description']}",
        }


# ========== FIX 53: Institutional Flow Mirroring (US/INTL MODEL ONLY) ==========
class InstitutionalFlowMirror:
    """
    Track and mirror institutional flows for alpha generation.

    FIX 53: Institutional Flow Mirroring

    The problem: Model doesn't incorporate institutional positioning data
    which often leads price moves.

    Solution: Track and follow institutional flows:
    1. 13F filing analysis (quarterly positioning)
    2. ETF flow momentum (daily/weekly)
    3. Block trade detection (large prints)
    4. Hedge fund positioning via 13F-HR
    5. Insider trading patterns (Form 4)

    Expected Alpha: +7-12% with medium risk
    """

    # Institutional signal thresholds
    FLOW_THRESHOLDS = {
        'etf_inflow_strong': 0.03,  # >3% AUM inflow
        'etf_inflow': 0.01,  # >1% AUM inflow
        'etf_outflow': -0.01,  # <-1% AUM outflow
        'etf_outflow_strong': -0.03,  # <-3% AUM outflow
    }

    # 13F position change thresholds
    POSITION_CHANGE_THRESHOLDS = {
        'major_increase': 0.25,  # >25% increase
        'increase': 0.10,  # >10% increase
        'decrease': -0.10,  # <-10% decrease
        'major_decrease': -0.25,  # <-25% decrease
        'new_position': 1.0,  # New position
        'liquidation': -1.0,  # Complete exit
    }

    # Signal multipliers
    FLOW_MULTIPLIERS = {
        'etf_inflow_strong': 1.50,
        'etf_inflow': 1.25,
        'etf_outflow': 0.80,
        'etf_outflow_strong': 0.60,
        '13f_major_increase': 1.40,
        '13f_increase': 1.20,
        '13f_decrease': 0.85,
        '13f_major_decrease': 0.65,
        '13f_new_position': 1.35,
        '13f_liquidation': 0.50,
        'block_buy_large': 1.45,
        'block_buy': 1.20,
        'block_sell_large': 0.75,
        'block_sell': 0.85,
        'insider_buy_cluster': 1.50,
        'insider_buy': 1.20,
        'insider_sell_cluster': 0.70,
        'insider_sell': 0.90,
        'hedge_fund_bullish': 1.30,
        'hedge_fund_bearish': 0.75,
    }

    # Notable institutional investors (for 13F tracking)
    NOTABLE_INVESTORS = {
        'berkshire_hathaway': {'weight': 1.5, 'style': 'value'},
        'bridgewater': {'weight': 1.3, 'style': 'macro'},
        'renaissance': {'weight': 1.4, 'style': 'quant'},
        'tiger_global': {'weight': 1.2, 'style': 'growth'},
        'citadel': {'weight': 1.3, 'style': 'multi'},
        'millennium': {'weight': 1.2, 'style': 'multi'},
        'two_sigma': {'weight': 1.3, 'style': 'quant'},
        'de_shaw': {'weight': 1.2, 'style': 'quant'},
        'point72': {'weight': 1.2, 'style': 'multi'},
        'pershing_square': {'weight': 1.3, 'style': 'activist'},
    }

    # Sector ETF flows to track
    SECTOR_ETFS = {
        'XLK': 'technology',
        'XLF': 'financials',
        'XLE': 'energy',
        'XLV': 'healthcare',
        'XLI': 'industrials',
        'XLP': 'consumer_staples',
        'XLY': 'consumer_discretionary',
        'XLU': 'utilities',
        'XLB': 'materials',
        'XLRE': 'real_estate',
        'XLC': 'communications',
    }

    def __init__(self):
        self.etf_flows = {}
        self.institutional_positions = {}
        self.block_trades = []
        self.insider_trades = {}
        self.hedge_fund_positions = {}

    def analyze_etf_flows(
        self,
        etf_ticker: str,
        flow_amount: float,
        aum: float,
        lookback_days: int = 5,
    ) -> Dict[str, Any]:
        """
        Analyze ETF fund flows for institutional sentiment.

        Args:
            etf_ticker: ETF ticker
            flow_amount: Net flow amount
            aum: Assets under management
            lookback_days: Days to consider

        Returns:
            Dict with flow analysis
        """
        if aum <= 0:
            return {'signal': 'neutral', 'multiplier': 1.0, 'flow_pct': 0}

        flow_pct = flow_amount / aum

        # Determine signal
        if flow_pct >= self.FLOW_THRESHOLDS['etf_inflow_strong']:
            signal = 'etf_inflow_strong'
        elif flow_pct >= self.FLOW_THRESHOLDS['etf_inflow']:
            signal = 'etf_inflow'
        elif flow_pct <= self.FLOW_THRESHOLDS['etf_outflow_strong']:
            signal = 'etf_outflow_strong'
        elif flow_pct <= self.FLOW_THRESHOLDS['etf_outflow']:
            signal = 'etf_outflow'
        else:
            signal = 'neutral'

        multiplier = self.FLOW_MULTIPLIERS.get(signal, 1.0)

        return {
            'etf': etf_ticker,
            'sector': self.SECTOR_ETFS.get(etf_ticker, 'unknown'),
            'signal': signal,
            'multiplier': multiplier,
            'flow_pct': flow_pct,
            'flow_amount': flow_amount,
            'aum': aum,
        }

    def analyze_13f_changes(
        self,
        ticker: str,
        current_shares: int,
        previous_shares: int,
        investor_name: str = None,
    ) -> Dict[str, Any]:
        """
        Analyze 13F filing position changes.

        Args:
            ticker: Stock ticker
            current_shares: Current quarter shares
            previous_shares: Previous quarter shares
            investor_name: Name of institutional investor

        Returns:
            Dict with 13F analysis
        """
        # Calculate change
        if previous_shares == 0:
            if current_shares > 0:
                change_type = 'new_position'
                change_pct = 1.0
            else:
                change_type = 'neutral'
                change_pct = 0.0
        elif current_shares == 0:
            change_type = 'liquidation'
            change_pct = -1.0
        else:
            change_pct = (current_shares - previous_shares) / previous_shares

            if change_pct >= self.POSITION_CHANGE_THRESHOLDS['major_increase']:
                change_type = 'major_increase'
            elif change_pct >= self.POSITION_CHANGE_THRESHOLDS['increase']:
                change_type = 'increase'
            elif change_pct <= self.POSITION_CHANGE_THRESHOLDS['major_decrease']:
                change_type = 'major_decrease'
            elif change_pct <= self.POSITION_CHANGE_THRESHOLDS['decrease']:
                change_type = 'decrease'
            else:
                change_type = 'neutral'

        # Get multiplier
        signal_key = f"13f_{change_type}"
        base_multiplier = self.FLOW_MULTIPLIERS.get(signal_key, 1.0)

        # Apply investor weight if notable
        investor_weight = 1.0
        if investor_name:
            investor_key = investor_name.lower().replace(' ', '_')
            if investor_key in self.NOTABLE_INVESTORS:
                investor_weight = self.NOTABLE_INVESTORS[investor_key]['weight']

        final_multiplier = base_multiplier * investor_weight

        return {
            'ticker': ticker,
            'investor': investor_name,
            'change_type': change_type,
            'change_pct': change_pct,
            'base_multiplier': base_multiplier,
            'investor_weight': investor_weight,
            'final_multiplier': final_multiplier,
            'current_shares': current_shares,
            'previous_shares': previous_shares,
        }

    def detect_block_trade(
        self,
        ticker: str,
        trade_size: int,
        avg_volume: int,
        trade_direction: str,  # 'buy' or 'sell'
        price_impact: float,
    ) -> Dict[str, Any]:
        """
        Detect and analyze block trades.

        Args:
            ticker: Stock ticker
            trade_size: Number of shares
            avg_volume: Average daily volume
            trade_direction: 'buy' or 'sell'
            price_impact: Price impact of trade

        Returns:
            Dict with block trade analysis
        """
        if avg_volume <= 0:
            return {'signal': 'neutral', 'multiplier': 1.0}

        size_ratio = trade_size / avg_volume
        is_large = size_ratio > 0.05  # >5% of daily volume

        if trade_direction == 'buy':
            if is_large:
                signal = 'block_buy_large'
            else:
                signal = 'block_buy'
        else:
            if is_large:
                signal = 'block_sell_large'
            else:
                signal = 'block_sell'

        multiplier = self.FLOW_MULTIPLIERS.get(signal, 1.0)

        return {
            'ticker': ticker,
            'signal': signal,
            'multiplier': multiplier,
            'trade_size': trade_size,
            'size_ratio': size_ratio,
            'direction': trade_direction,
            'price_impact': price_impact,
            'is_large': is_large,
        }

    def analyze_insider_trades(
        self,
        ticker: str,
        insider_buys: int,
        insider_sells: int,
        net_shares: int,
        lookback_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Analyze insider trading patterns.

        Args:
            ticker: Stock ticker
            insider_buys: Number of insider buy transactions
            insider_sells: Number of insider sell transactions
            net_shares: Net shares bought/sold
            lookback_days: Period to analyze

        Returns:
            Dict with insider trading analysis
        """
        # Determine signal based on activity
        if insider_buys >= 3 and net_shares > 0:
            signal = 'insider_buy_cluster'
        elif insider_buys >= 1 and net_shares > 0:
            signal = 'insider_buy'
        elif insider_sells >= 3 and net_shares < 0:
            signal = 'insider_sell_cluster'
        elif insider_sells >= 1 and net_shares < 0:
            signal = 'insider_sell'
        else:
            signal = 'neutral'

        multiplier = self.FLOW_MULTIPLIERS.get(signal, 1.0)

        return {
            'ticker': ticker,
            'signal': signal,
            'multiplier': multiplier,
            'insider_buys': insider_buys,
            'insider_sells': insider_sells,
            'net_shares': net_shares,
            'lookback_days': lookback_days,
        }

    def get_institutional_signal(
        self,
        ticker: str,
        signal_type: str,
        etf_flow_data: Dict = None,
        filing_data: Dict = None,
        block_trade_data: Dict = None,
        insider_data: Dict = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive institutional flow signal.

        Args:
            ticker: Stock ticker
            signal_type: 'BUY' or 'SELL'
            etf_flow_data: ETF flow information
            filing_data: 13F filing data
            block_trade_data: Block trade data
            insider_data: Insider trading data

        Returns:
            Dict with institutional analysis and combined multiplier
        """
        is_buy = signal_type.upper() == 'BUY'
        multipliers = []
        signals = []
        explanations = []

        # ETF Flows
        if etf_flow_data:
            flow_result = self.analyze_etf_flows(
                etf_flow_data.get('etf', 'SPY'),
                etf_flow_data.get('flow_amount', 0),
                etf_flow_data.get('aum', 1),
            )
            if flow_result['signal'] != 'neutral':
                # Inflows support BUY, outflows support SELL
                flow_supports = ('inflow' in flow_result['signal'] and is_buy) or \
                               ('outflow' in flow_result['signal'] and not is_buy)
                if flow_supports:
                    multipliers.append(flow_result['multiplier'])
                    signals.append(flow_result['signal'])
                    explanations.append(f"ETF flow: {flow_result['flow_pct']:.2%}")

        # 13F Changes
        if filing_data:
            filing_result = self.analyze_13f_changes(
                ticker,
                filing_data.get('current_shares', 0),
                filing_data.get('previous_shares', 0),
                filing_data.get('investor', None),
            )
            if filing_result['change_type'] != 'neutral':
                # Increases support BUY, decreases support SELL
                filing_supports = (filing_result['change_pct'] > 0 and is_buy) or \
                                 (filing_result['change_pct'] < 0 and not is_buy)
                if filing_supports:
                    multipliers.append(filing_result['final_multiplier'])
                    signals.append(f"13f_{filing_result['change_type']}")
                    explanations.append(f"13F {filing_result['change_type']}: {filing_result['change_pct']:.1%}")

        # Block Trades
        if block_trade_data:
            block_result = self.detect_block_trade(
                ticker,
                block_trade_data.get('size', 0),
                block_trade_data.get('avg_volume', 1),
                block_trade_data.get('direction', 'buy'),
                block_trade_data.get('price_impact', 0),
            )
            if block_result['signal'] != 'neutral':
                # Block direction should match signal
                block_supports = (block_result['direction'] == 'buy' and is_buy) or \
                                (block_result['direction'] == 'sell' and not is_buy)
                if block_supports:
                    multipliers.append(block_result['multiplier'])
                    signals.append(block_result['signal'])
                    explanations.append(f"Block trade: {block_result['size_ratio']:.1%} of volume")

        # Insider Trading
        if insider_data:
            insider_result = self.analyze_insider_trades(
                ticker,
                insider_data.get('buys', 0),
                insider_data.get('sells', 0),
                insider_data.get('net_shares', 0),
            )
            if insider_result['signal'] != 'neutral':
                # Insider buys support BUY, sells support SELL
                insider_supports = ('buy' in insider_result['signal'] and is_buy) or \
                                  ('sell' in insider_result['signal'] and not is_buy)
                if insider_supports:
                    multipliers.append(insider_result['multiplier'])
                    signals.append(insider_result['signal'])
                    explanations.append(f"Insider: {insider_result['net_shares']:+} shares")

        # Combine multipliers
        if multipliers:
            # Use geometric mean for multiple signals
            combined_multiplier = np.prod(multipliers) ** (1 / len(multipliers))
        else:
            combined_multiplier = 1.0

        return {
            'ticker': ticker,
            'signal_type': signal_type,
            'institutional_signals': signals,
            'combined_multiplier': combined_multiplier,
            'individual_multipliers': multipliers,
            'explanations': explanations,
            'has_institutional_support': len(signals) > 0,
            'signal_count': len(signals),
        }


class MegaCapMomentumExploiter:
    """
    Exploit mega-cap tech momentum patterns for enhanced returns.

    FIX 54: Mega-Cap Tech Momentum Exploitation

    The problem: Model treats all stocks equally, missing the unique
    momentum characteristics of mega-cap tech stocks that dominate
    US market returns.

    Solution: Specialized momentum tracking for FAANG+M stocks:
    1. Relative strength vs QQQ/SPY
    2. Earnings momentum clustering
    3. Options gamma ramp detection
    4. Institutional accumulation patterns
    5. Cross-stock momentum spillover effects

    Expected Alpha: +15-25% in trending tech markets

    NOTE: Uses DynamicUSTickerDiscovery for real-time stock universe updates.
    Falls back to static list if Yahoo Finance unavailable.
    This is US/INTL model ONLY - China stocks use DeepSeek model.
    """

    # Static fallback for FAANG+M and mega-cap tech tickers
    # Dynamic discovery will override this when available
    _STATIC_MEGA_CAP_TECH = {
        'AAPL': {'name': 'Apple', 'weight': 1.3, 'category': 'hardware'},
        'MSFT': {'name': 'Microsoft', 'weight': 1.3, 'category': 'software'},
        'GOOGL': {'name': 'Alphabet', 'weight': 1.2, 'category': 'advertising'},
        'GOOG': {'name': 'Alphabet C', 'weight': 1.2, 'category': 'advertising'},
        'AMZN': {'name': 'Amazon', 'weight': 1.2, 'category': 'ecommerce'},
        'META': {'name': 'Meta', 'weight': 1.1, 'category': 'social'},
        'NVDA': {'name': 'NVIDIA', 'weight': 1.4, 'category': 'semiconductors'},
        'TSLA': {'name': 'Tesla', 'weight': 1.0, 'category': 'ev'},
        'NFLX': {'name': 'Netflix', 'weight': 0.9, 'category': 'streaming'},
        'AMD': {'name': 'AMD', 'weight': 1.1, 'category': 'semiconductors'},
        'CRM': {'name': 'Salesforce', 'weight': 0.9, 'category': 'software'},
        'ADBE': {'name': 'Adobe', 'weight': 0.9, 'category': 'software'},
        'ORCL': {'name': 'Oracle', 'weight': 0.8, 'category': 'software'},
        'INTC': {'name': 'Intel', 'weight': 0.7, 'category': 'semiconductors'},
        'AVGO': {'name': 'Broadcom', 'weight': 1.0, 'category': 'semiconductors'},
    }

    # Momentum thresholds
    MOMENTUM_THRESHOLDS = {
        'strong_uptrend': 0.15,     # >15% above 50DMA
        'uptrend': 0.05,            # >5% above 50DMA
        'neutral_high': 0.02,       # 2-5% above 50DMA
        'neutral_low': -0.02,       # 2% below to 2% above 50DMA
        'downtrend': -0.05,         # >5% below 50DMA
        'strong_downtrend': -0.15,  # >15% below 50DMA
    }

    # Signal multipliers based on momentum
    MOMENTUM_MULTIPLIERS = {
        'strong_uptrend': 1.60,
        'uptrend': 1.35,
        'neutral_high': 1.15,
        'neutral_low': 1.00,
        'downtrend': 0.75,
        'strong_downtrend': 0.50,
    }

    # Relative strength thresholds vs index
    RS_THRESHOLDS = {
        'strong_outperform': 0.10,   # >10% vs index
        'outperform': 0.03,          # >3% vs index
        'inline': -0.03,             # -3% to 3%
        'underperform': -0.10,       # <-10% vs index
    }

    RS_MULTIPLIERS = {
        'strong_outperform': 1.50,
        'outperform': 1.25,
        'inline': 1.00,
        'underperform': 0.70,
        'strong_underperform': 0.50,
    }

    def __init__(self, use_dynamic_discovery: bool = True):
        """
        Initialize with optional dynamic ticker discovery.

        Args:
            use_dynamic_discovery: If True, use Yahoo Finance to dynamically
                                   discover mega-cap tech stocks. Falls back
                                   to static list if unavailable.
        """
        self.momentum_data = {}
        self.relative_strength = {}
        self.spillover_effects = {}
        self._use_dynamic = use_dynamic_discovery
        self._dynamic_cache = None
        self._dynamic_cache_time = 0

    @property
    def MEGA_CAP_TECH(self) -> Dict[str, Dict]:
        """
        Get mega-cap tech stock universe - dynamically or static fallback.
        """
        import time

        # Use dynamic discovery if enabled
        if self._use_dynamic:
            # Refresh cache every hour
            if self._dynamic_cache is None or (time.time() - self._dynamic_cache_time) > 3600:
                try:
                    discovery = get_dynamic_us_ticker_discovery()
                    self._dynamic_cache = discovery.discover_mega_cap_tech()
                    self._dynamic_cache_time = time.time()
                except Exception:
                    pass  # Fall through to static

            if self._dynamic_cache:
                return self._dynamic_cache

        # Fallback to static list
        return self._STATIC_MEGA_CAP_TECH

    def analyze_momentum(
        self,
        ticker: str,
        current_price: float,
        ma_50: float,
        ma_200: float,
        volume_ratio: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Analyze momentum for mega-cap tech stock.

        Args:
            ticker: Stock ticker
            current_price: Current price
            ma_50: 50-day moving average
            ma_200: 200-day moving average
            volume_ratio: Current volume / average volume

        Returns:
            Dict with momentum analysis
        """
        if ticker not in self.MEGA_CAP_TECH:
            return {'is_mega_cap': False, 'multiplier': 1.0}

        stock_info = self.MEGA_CAP_TECH[ticker]

        # Calculate momentum vs 50DMA
        if ma_50 > 0:
            momentum_50 = (current_price - ma_50) / ma_50
        else:
            momentum_50 = 0

        # Calculate momentum vs 200DMA (trend)
        if ma_200 > 0:
            momentum_200 = (current_price - ma_200) / ma_200
            golden_cross = ma_50 > ma_200
        else:
            momentum_200 = 0
            golden_cross = False

        # Determine momentum state
        if momentum_50 >= self.MOMENTUM_THRESHOLDS['strong_uptrend']:
            momentum_state = 'strong_uptrend'
        elif momentum_50 >= self.MOMENTUM_THRESHOLDS['uptrend']:
            momentum_state = 'uptrend'
        elif momentum_50 >= self.MOMENTUM_THRESHOLDS['neutral_high']:
            momentum_state = 'neutral_high'
        elif momentum_50 >= self.MOMENTUM_THRESHOLDS['neutral_low']:
            momentum_state = 'neutral_low'
        elif momentum_50 >= self.MOMENTUM_THRESHOLDS['downtrend']:
            momentum_state = 'downtrend'
        else:
            momentum_state = 'strong_downtrend'

        base_multiplier = self.MOMENTUM_MULTIPLIERS[momentum_state]

        # Apply stock weight
        weight_adjusted = base_multiplier * stock_info['weight']

        # Volume confirmation bonus
        if volume_ratio > 1.5 and momentum_50 > 0:
            volume_bonus = 1.10
        elif volume_ratio > 2.0 and momentum_50 > 0:
            volume_bonus = 1.20
        else:
            volume_bonus = 1.0

        # Golden cross bonus
        golden_bonus = 1.15 if golden_cross else 1.0

        final_multiplier = weight_adjusted * volume_bonus * golden_bonus

        return {
            'ticker': ticker,
            'is_mega_cap': True,
            'name': stock_info['name'],
            'category': stock_info['category'],
            'momentum_50': momentum_50,
            'momentum_200': momentum_200,
            'momentum_state': momentum_state,
            'golden_cross': golden_cross,
            'base_multiplier': base_multiplier,
            'stock_weight': stock_info['weight'],
            'volume_ratio': volume_ratio,
            'volume_bonus': volume_bonus,
            'golden_bonus': golden_bonus,
            'final_multiplier': final_multiplier,
        }

    def calculate_relative_strength(
        self,
        ticker: str,
        stock_return: float,
        index_return: float,
        period_days: int = 20,
    ) -> Dict[str, Any]:
        """
        Calculate relative strength vs index.

        Args:
            ticker: Stock ticker
            stock_return: Stock return over period
            index_return: Index return over period
            period_days: Lookback period

        Returns:
            Dict with relative strength analysis
        """
        if ticker not in self.MEGA_CAP_TECH:
            return {'is_mega_cap': False, 'rs_multiplier': 1.0}

        # Calculate relative strength
        rs = stock_return - index_return

        # Determine RS state
        if rs >= self.RS_THRESHOLDS['strong_outperform']:
            rs_state = 'strong_outperform'
        elif rs >= self.RS_THRESHOLDS['outperform']:
            rs_state = 'outperform'
        elif rs >= self.RS_THRESHOLDS['inline']:
            rs_state = 'inline'
        elif rs >= self.RS_THRESHOLDS['underperform']:
            rs_state = 'underperform'
        else:
            rs_state = 'strong_underperform'

        rs_multiplier = self.RS_MULTIPLIERS[rs_state]

        return {
            'ticker': ticker,
            'is_mega_cap': True,
            'stock_return': stock_return,
            'index_return': index_return,
            'relative_strength': rs,
            'rs_state': rs_state,
            'rs_multiplier': rs_multiplier,
            'period_days': period_days,
        }

    def detect_momentum_spillover(
        self,
        ticker: str,
        peer_momentum: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Detect momentum spillover from peer stocks.

        Args:
            ticker: Stock ticker
            peer_momentum: Dict of peer tickers to momentum values

        Returns:
            Dict with spillover analysis
        """
        if ticker not in self.MEGA_CAP_TECH:
            return {'spillover_multiplier': 1.0}

        stock_info = self.MEGA_CAP_TECH[ticker]
        category = stock_info['category']

        # Find peers in same category
        peer_count = 0
        peer_positive = 0
        peer_avg_momentum = 0

        for peer, momentum in peer_momentum.items():
            if peer != ticker and peer in self.MEGA_CAP_TECH:
                peer_info = self.MEGA_CAP_TECH[peer]
                if peer_info['category'] == category:
                    peer_count += 1
                    peer_avg_momentum += momentum
                    if momentum > 0.05:  # Strong positive momentum
                        peer_positive += 1

        if peer_count > 0:
            peer_avg_momentum /= peer_count
            peer_ratio = peer_positive / peer_count
        else:
            peer_avg_momentum = 0
            peer_ratio = 0

        # Calculate spillover multiplier
        if peer_ratio >= 0.7 and peer_avg_momentum > 0.10:
            spillover_multiplier = 1.30
            spillover_signal = 'strong_positive'
        elif peer_ratio >= 0.5 and peer_avg_momentum > 0.05:
            spillover_multiplier = 1.15
            spillover_signal = 'positive'
        elif peer_ratio <= 0.3 and peer_avg_momentum < -0.05:
            spillover_multiplier = 0.80
            spillover_signal = 'negative'
        else:
            spillover_multiplier = 1.0
            spillover_signal = 'neutral'

        return {
            'ticker': ticker,
            'category': category,
            'peer_count': peer_count,
            'peer_positive': peer_positive,
            'peer_avg_momentum': peer_avg_momentum,
            'spillover_signal': spillover_signal,
            'spillover_multiplier': spillover_multiplier,
        }

    def get_mega_cap_signal(
        self,
        ticker: str,
        signal_type: str,
        momentum_data: Dict = None,
        rs_data: Dict = None,
        spillover_data: Dict = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive mega-cap tech signal.

        Args:
            ticker: Stock ticker
            signal_type: 'BUY' or 'SELL'
            momentum_data: Momentum analysis data
            rs_data: Relative strength data
            spillover_data: Spillover analysis data

        Returns:
            Dict with combined signal
        """
        if ticker not in self.MEGA_CAP_TECH:
            return {'is_mega_cap': False, 'combined_multiplier': 1.0}

        is_buy = signal_type.upper() == 'BUY'
        multipliers = []
        signals = []

        # Momentum component
        if momentum_data:
            mom_mult = momentum_data.get('final_multiplier', 1.0)
            if is_buy and mom_mult > 1.0:
                multipliers.append(mom_mult)
                signals.append(f"momentum_{momentum_data.get('momentum_state', 'neutral')}")
            elif not is_buy and mom_mult < 1.0:
                multipliers.append(2.0 - mom_mult)  # Invert for SELL
                signals.append(f"momentum_{momentum_data.get('momentum_state', 'neutral')}")

        # Relative strength component
        if rs_data:
            rs_mult = rs_data.get('rs_multiplier', 1.0)
            if is_buy and rs_mult > 1.0:
                multipliers.append(rs_mult)
                signals.append(f"rs_{rs_data.get('rs_state', 'neutral')}")
            elif not is_buy and rs_mult < 1.0:
                multipliers.append(2.0 - rs_mult)
                signals.append(f"rs_{rs_data.get('rs_state', 'neutral')}")

        # Spillover component
        if spillover_data:
            spill_mult = spillover_data.get('spillover_multiplier', 1.0)
            if is_buy and spill_mult > 1.0:
                multipliers.append(spill_mult)
                signals.append(spillover_data.get('spillover_signal', 'neutral'))
            elif not is_buy and spill_mult < 1.0:
                multipliers.append(2.0 - spill_mult)
                signals.append(spillover_data.get('spillover_signal', 'neutral'))

        # Combine multipliers
        if multipliers:
            combined_multiplier = np.prod(multipliers) ** (1 / len(multipliers))
        else:
            combined_multiplier = 1.0

        return {
            'ticker': ticker,
            'is_mega_cap': True,
            'signal_type': signal_type,
            'signals': signals,
            'individual_multipliers': multipliers,
            'combined_multiplier': combined_multiplier,
            'stock_info': self.MEGA_CAP_TECH[ticker],
        }


class SemiconductorCycleDetector:
    """
    Detect and exploit semiconductor industry super-cycles.

    FIX 55: Semiconductor Super-Cycle Detection

    The problem: Model misses the cyclical nature of semiconductor stocks
    which follow predictable boom-bust patterns driven by capex cycles,
    inventory cycles, and technology transitions.

    Solution: Track semiconductor cycle indicators:
    1. Book-to-bill ratio trends
    2. Inventory channel checks
    3. Capex cycle phase detection
    4. Technology node transitions
    5. End-market demand signals

    Expected Alpha: +30-50% during cycle transitions

    NOTE: Uses DynamicUSTickerDiscovery for real-time stock universe updates.
    Falls back to static list if Yahoo Finance unavailable.
    This is US/INTL model ONLY - China stocks use DeepSeek model.
    """

    # Static fallback for semiconductor stock universe
    _STATIC_SEMICONDUCTOR_STOCKS = {
        # Fabless
        'NVDA': {'type': 'fabless', 'segment': 'gpu', 'cycle_beta': 1.8},
        'AMD': {'type': 'fabless', 'segment': 'cpu_gpu', 'cycle_beta': 1.6},
        'AVGO': {'type': 'fabless', 'segment': 'diversified', 'cycle_beta': 1.2},
        'QCOM': {'type': 'fabless', 'segment': 'mobile', 'cycle_beta': 1.3},
        'MRVL': {'type': 'fabless', 'segment': 'infrastructure', 'cycle_beta': 1.4},
        'NXPI': {'type': 'fabless', 'segment': 'auto', 'cycle_beta': 1.3},
        'ON': {'type': 'fabless', 'segment': 'auto_industrial', 'cycle_beta': 1.2},

        # IDM (Integrated Device Manufacturers)
        'INTC': {'type': 'idm', 'segment': 'cpu', 'cycle_beta': 1.0},
        'TXN': {'type': 'idm', 'segment': 'analog', 'cycle_beta': 0.9},
        'MU': {'type': 'idm', 'segment': 'memory', 'cycle_beta': 2.0},
        'ADI': {'type': 'idm', 'segment': 'analog', 'cycle_beta': 1.0},
        'MCHP': {'type': 'idm', 'segment': 'mcu', 'cycle_beta': 1.1},

        # Foundry
        'TSM': {'type': 'foundry', 'segment': 'foundry', 'cycle_beta': 1.4},

        # Equipment
        'ASML': {'type': 'equipment', 'segment': 'litho', 'cycle_beta': 1.5},
        'AMAT': {'type': 'equipment', 'segment': 'equipment', 'cycle_beta': 1.6},
        'LRCX': {'type': 'equipment', 'segment': 'etch', 'cycle_beta': 1.7},
        'KLAC': {'type': 'equipment', 'segment': 'inspection', 'cycle_beta': 1.5},

        # Memory
        'WDC': {'type': 'memory', 'segment': 'nand', 'cycle_beta': 1.8},
        'STX': {'type': 'memory', 'segment': 'hdd', 'cycle_beta': 1.4},
    }

    # Cycle phase definitions
    CYCLE_PHASES = {
        'early_recovery': {'position': 'aggressive_long', 'multiplier': 1.80},
        'mid_expansion': {'position': 'long', 'multiplier': 1.50},
        'late_expansion': {'position': 'reduce', 'multiplier': 0.90},
        'peak': {'position': 'exit', 'multiplier': 0.60},
        'early_downturn': {'position': 'short', 'multiplier': 0.40},
        'mid_contraction': {'position': 'wait', 'multiplier': 0.50},
        'late_contraction': {'position': 'accumulate', 'multiplier': 1.20},
        'trough': {'position': 'aggressive_accumulate', 'multiplier': 1.60},
    }

    # Book-to-bill thresholds
    BOOK_TO_BILL_THRESHOLDS = {
        'very_strong': 1.20,    # >1.2 very bullish
        'strong': 1.10,         # >1.1 bullish
        'neutral_high': 1.02,   # 1.02-1.10 neutral to positive
        'neutral': 0.98,        # 0.98-1.02 balanced
        'weak': 0.90,           # <0.98 bearish
        'very_weak': 0.80,      # <0.90 very bearish
    }

    # Inventory days thresholds
    INVENTORY_THRESHOLDS = {
        'critical_high': 120,   # >120 days - major concern
        'elevated': 100,        # 100-120 days - elevated
        'normal': 80,           # 60-100 days - normal
        'lean': 60,             # 40-60 days - lean
        'very_lean': 40,        # <40 days - very lean (bullish)
    }

    def __init__(self, use_dynamic_discovery: bool = True):
        """
        Initialize with optional dynamic ticker discovery.

        Args:
            use_dynamic_discovery: If True, use Yahoo Finance to dynamically
                                   discover semiconductor stocks.
        """
        self.current_phase = 'mid_expansion'
        self.book_to_bill_history = []
        self.inventory_data = {}
        self.capex_signals = {}
        self._use_dynamic = use_dynamic_discovery
        self._dynamic_cache = None
        self._dynamic_cache_time = 0

    @property
    def SEMICONDUCTOR_STOCKS(self) -> Dict[str, Dict]:
        """Get semiconductor stock universe - dynamically or static fallback."""
        import time

        if self._use_dynamic:
            if self._dynamic_cache is None or (time.time() - self._dynamic_cache_time) > 3600:
                try:
                    discovery = get_dynamic_us_ticker_discovery()
                    self._dynamic_cache = discovery.discover_semiconductors()
                    self._dynamic_cache_time = time.time()
                except Exception:
                    pass

            if self._dynamic_cache:
                return self._dynamic_cache

        return self._STATIC_SEMICONDUCTOR_STOCKS

    def detect_cycle_phase(
        self,
        book_to_bill: float,
        inventory_days: float,
        yoy_revenue_growth: float,
        capex_growth: float,
        utilization_rate: float = 0.80,
    ) -> Dict[str, Any]:
        """
        Detect current semiconductor cycle phase.

        Args:
            book_to_bill: Current book-to-bill ratio
            inventory_days: Days of inventory on hand
            yoy_revenue_growth: Year-over-year revenue growth
            capex_growth: Year-over-year capex growth
            utilization_rate: Fab utilization rate

        Returns:
            Dict with cycle phase analysis
        """
        signals = []
        phase_scores = {}

        # Book-to-bill signal
        if book_to_bill >= self.BOOK_TO_BILL_THRESHOLDS['very_strong']:
            btb_signal = 'very_strong'
            signals.append('btb_very_bullish')
        elif book_to_bill >= self.BOOK_TO_BILL_THRESHOLDS['strong']:
            btb_signal = 'strong'
            signals.append('btb_bullish')
        elif book_to_bill >= self.BOOK_TO_BILL_THRESHOLDS['neutral_high']:
            btb_signal = 'neutral_high'
            signals.append('btb_neutral_positive')
        elif book_to_bill >= self.BOOK_TO_BILL_THRESHOLDS['neutral']:
            btb_signal = 'neutral'
            signals.append('btb_neutral')
        elif book_to_bill >= self.BOOK_TO_BILL_THRESHOLDS['weak']:
            btb_signal = 'weak'
            signals.append('btb_bearish')
        else:
            btb_signal = 'very_weak'
            signals.append('btb_very_bearish')

        # Inventory signal
        if inventory_days >= self.INVENTORY_THRESHOLDS['critical_high']:
            inv_signal = 'critical_high'
            signals.append('inventory_crisis')
        elif inventory_days >= self.INVENTORY_THRESHOLDS['elevated']:
            inv_signal = 'elevated'
            signals.append('inventory_elevated')
        elif inventory_days >= self.INVENTORY_THRESHOLDS['normal']:
            inv_signal = 'normal'
            signals.append('inventory_normal')
        elif inventory_days >= self.INVENTORY_THRESHOLDS['lean']:
            inv_signal = 'lean'
            signals.append('inventory_lean')
        else:
            inv_signal = 'very_lean'
            signals.append('inventory_very_lean')

        # Determine cycle phase based on multiple signals
        if btb_signal in ['very_strong', 'strong'] and inv_signal in ['lean', 'very_lean']:
            if yoy_revenue_growth > 0.20:
                phase = 'mid_expansion'
            else:
                phase = 'early_recovery'
        elif btb_signal in ['very_strong', 'strong'] and inv_signal in ['normal', 'elevated']:
            phase = 'late_expansion'
        elif btb_signal in ['neutral', 'neutral_high'] and inv_signal in ['elevated', 'critical_high']:
            phase = 'peak'
        elif btb_signal in ['weak', 'very_weak'] and inv_signal in ['critical_high', 'elevated']:
            phase = 'early_downturn'
        elif btb_signal in ['weak', 'very_weak'] and inv_signal in ['normal']:
            phase = 'mid_contraction'
        elif btb_signal in ['very_weak'] and inv_signal in ['lean', 'normal']:
            phase = 'late_contraction'
        elif btb_signal in ['neutral', 'neutral_high'] and inv_signal in ['very_lean', 'lean']:
            phase = 'trough'
        else:
            phase = 'mid_expansion'  # Default

        phase_info = self.CYCLE_PHASES[phase]

        return {
            'cycle_phase': phase,
            'position_strategy': phase_info['position'],
            'base_multiplier': phase_info['multiplier'],
            'book_to_bill': book_to_bill,
            'btb_signal': btb_signal,
            'inventory_days': inventory_days,
            'inv_signal': inv_signal,
            'yoy_revenue_growth': yoy_revenue_growth,
            'capex_growth': capex_growth,
            'utilization_rate': utilization_rate,
            'signals': signals,
        }

    def analyze_stock_cycle_exposure(
        self,
        ticker: str,
        cycle_phase: str,
    ) -> Dict[str, Any]:
        """
        Analyze a stock's exposure to the current cycle phase.

        Args:
            ticker: Stock ticker
            cycle_phase: Current cycle phase

        Returns:
            Dict with stock cycle analysis
        """
        if ticker not in self.SEMICONDUCTOR_STOCKS:
            return {'is_semi': False, 'multiplier': 1.0}

        stock_info = self.SEMICONDUCTOR_STOCKS[ticker]
        phase_info = self.CYCLE_PHASES.get(cycle_phase, self.CYCLE_PHASES['mid_expansion'])

        # Apply cycle beta
        cycle_beta = stock_info['cycle_beta']
        base_mult = phase_info['multiplier']

        # Higher beta stocks move more in both directions
        if base_mult > 1.0:
            adjusted_mult = 1.0 + (base_mult - 1.0) * cycle_beta
        else:
            adjusted_mult = 1.0 - (1.0 - base_mult) * cycle_beta

        # Segment-specific adjustments
        segment = stock_info['segment']
        segment_adj = 1.0

        if segment == 'memory':
            # Memory most cyclical
            segment_adj = 1.20 if base_mult > 1.0 else 0.80
        elif segment == 'equipment':
            # Equipment leads cycle
            if cycle_phase in ['trough', 'early_recovery']:
                segment_adj = 1.25
            elif cycle_phase in ['peak', 'early_downturn']:
                segment_adj = 0.75
        elif segment == 'analog':
            # Analog less cyclical
            segment_adj = 0.90 if abs(base_mult - 1.0) > 0.3 else 1.0

        final_multiplier = adjusted_mult * segment_adj

        return {
            'ticker': ticker,
            'is_semi': True,
            'type': stock_info['type'],
            'segment': segment,
            'cycle_beta': cycle_beta,
            'cycle_phase': cycle_phase,
            'position_strategy': phase_info['position'],
            'base_multiplier': base_mult,
            'beta_adjusted': adjusted_mult,
            'segment_adjustment': segment_adj,
            'final_multiplier': final_multiplier,
        }

    def get_semiconductor_signal(
        self,
        ticker: str,
        signal_type: str,
        cycle_data: Dict = None,
    ) -> Dict[str, Any]:
        """
        Get semiconductor-adjusted signal.

        Args:
            ticker: Stock ticker
            signal_type: 'BUY' or 'SELL'
            cycle_data: Cycle phase data

        Returns:
            Dict with semiconductor signal
        """
        if ticker not in self.SEMICONDUCTOR_STOCKS:
            return {'is_semi': False, 'multiplier': 1.0}

        is_buy = signal_type.upper() == 'BUY'
        cycle_phase = cycle_data.get('cycle_phase', 'mid_expansion') if cycle_data else 'mid_expansion'

        stock_analysis = self.analyze_stock_cycle_exposure(ticker, cycle_phase)
        phase_info = self.CYCLE_PHASES[cycle_phase]

        # Check if signal aligns with cycle
        position = phase_info['position']

        if is_buy:
            if position in ['aggressive_long', 'long', 'aggressive_accumulate', 'accumulate']:
                signal_alignment = 'aligned'
                alignment_bonus = 1.20
            elif position in ['reduce', 'exit']:
                signal_alignment = 'contrary'
                alignment_bonus = 0.60
            else:
                signal_alignment = 'neutral'
                alignment_bonus = 1.0
        else:  # SELL
            if position in ['short', 'exit', 'reduce']:
                signal_alignment = 'aligned'
                alignment_bonus = 1.20
            elif position in ['aggressive_long', 'long', 'accumulate']:
                signal_alignment = 'contrary'
                alignment_bonus = 0.60
            else:
                signal_alignment = 'neutral'
                alignment_bonus = 1.0

        final_multiplier = stock_analysis['final_multiplier'] * alignment_bonus

        return {
            'ticker': ticker,
            'is_semi': True,
            'signal_type': signal_type,
            'cycle_phase': cycle_phase,
            'position_strategy': position,
            'signal_alignment': signal_alignment,
            'alignment_bonus': alignment_bonus,
            'stock_multiplier': stock_analysis['final_multiplier'],
            'final_multiplier': final_multiplier,
            'stock_info': self.SEMICONDUCTOR_STOCKS[ticker],
        }


class AIThematicConcentrator:
    """
    Concentrate positions in AI-related stocks during AI boom periods.

    FIX 56: AI Thematic Concentration

    The problem: Model doesn't recognize or capitalize on AI theme momentum
    which has driven massive outperformance in related stocks.

    Solution: Track and exploit AI theme:
    1. AI pure-play identification
    2. AI beneficiary mapping
    3. AI sentiment tracking
    4. AI capex cycle signals
    5. AI adoption curve position

    Expected Alpha: +40-60% during AI momentum phases

    NOTE: Uses DynamicUSTickerDiscovery for real-time stock universe updates.
    This is US/INTL model ONLY - China stocks use DeepSeek model.
    """

    # Static fallback for AI stock universe with exposure levels
    _STATIC_AI_STOCKS = {
        # Pure-play AI infrastructure
        'NVDA': {'exposure': 'pure_play', 'category': 'gpu_compute', 'ai_score': 10},
        'AMD': {'exposure': 'high', 'category': 'gpu_compute', 'ai_score': 8},
        'AVGO': {'exposure': 'high', 'category': 'ai_networking', 'ai_score': 7},
        'MRVL': {'exposure': 'high', 'category': 'ai_networking', 'ai_score': 7},
        'ARM': {'exposure': 'high', 'category': 'ai_chips', 'ai_score': 8},

        # AI software/platforms
        'MSFT': {'exposure': 'high', 'category': 'ai_platform', 'ai_score': 9},
        'GOOGL': {'exposure': 'high', 'category': 'ai_platform', 'ai_score': 9},
        'GOOG': {'exposure': 'high', 'category': 'ai_platform', 'ai_score': 9},
        'META': {'exposure': 'high', 'category': 'ai_platform', 'ai_score': 8},
        'AMZN': {'exposure': 'medium', 'category': 'ai_cloud', 'ai_score': 7},
        'CRM': {'exposure': 'medium', 'category': 'ai_enterprise', 'ai_score': 6},
        'NOW': {'exposure': 'medium', 'category': 'ai_enterprise', 'ai_score': 6},
        'PLTR': {'exposure': 'high', 'category': 'ai_analytics', 'ai_score': 8},
        'SNOW': {'exposure': 'medium', 'category': 'ai_data', 'ai_score': 6},
        'MDB': {'exposure': 'medium', 'category': 'ai_data', 'ai_score': 5},

        # AI infrastructure enablers
        'SMCI': {'exposure': 'pure_play', 'category': 'ai_servers', 'ai_score': 9},
        'DELL': {'exposure': 'medium', 'category': 'ai_servers', 'ai_score': 5},
        'HPE': {'exposure': 'low', 'category': 'ai_servers', 'ai_score': 4},
        'VRT': {'exposure': 'high', 'category': 'ai_power', 'ai_score': 7},
        'ANET': {'exposure': 'high', 'category': 'ai_networking', 'ai_score': 7},

        # Memory for AI
        'MU': {'exposure': 'high', 'category': 'ai_memory', 'ai_score': 7},
        'WDC': {'exposure': 'medium', 'category': 'ai_storage', 'ai_score': 5},

        # Equipment for AI chips
        'ASML': {'exposure': 'high', 'category': 'ai_equipment', 'ai_score': 7},
        'AMAT': {'exposure': 'medium', 'category': 'ai_equipment', 'ai_score': 6},
        'LRCX': {'exposure': 'medium', 'category': 'ai_equipment', 'ai_score': 6},
        'KLAC': {'exposure': 'medium', 'category': 'ai_equipment', 'ai_score': 5},

        # Foundry
        'TSM': {'exposure': 'high', 'category': 'ai_foundry', 'ai_score': 8},
    }

    # Exposure multipliers
    EXPOSURE_MULTIPLIERS = {
        'pure_play': 2.00,
        'high': 1.60,
        'medium': 1.30,
        'low': 1.10,
    }

    # AI sentiment thresholds
    AI_SENTIMENT_THRESHOLDS = {
        'euphoria': 0.90,      # Market in AI euphoria
        'bullish': 0.70,       # Strong AI bullishness
        'positive': 0.55,      # Generally positive
        'neutral': 0.45,       # Mixed signals
        'cautious': 0.30,      # Caution emerging
        'bearish': 0.15,       # AI skepticism
    }

    SENTIMENT_MULTIPLIERS = {
        'euphoria': 1.50,
        'bullish': 1.35,
        'positive': 1.20,
        'neutral': 1.00,
        'cautious': 0.80,
        'bearish': 0.60,
    }

    # AI capex cycle phases
    AI_CAPEX_PHASES = {
        'acceleration': {'multiplier': 1.60, 'signal': 'aggressive_long'},
        'steady_growth': {'multiplier': 1.30, 'signal': 'long'},
        'plateau': {'multiplier': 0.90, 'signal': 'reduce'},
        'deceleration': {'multiplier': 0.60, 'signal': 'exit'},
        'contraction': {'multiplier': 0.40, 'signal': 'avoid'},
    }

    def __init__(self, use_dynamic_discovery: bool = True):
        """Initialize with optional dynamic ticker discovery."""
        self.ai_sentiment = 0.70  # Default bullish
        self.capex_phase = 'steady_growth'
        self.theme_momentum = {}
        self._use_dynamic = use_dynamic_discovery
        self._dynamic_cache = None
        self._dynamic_cache_time = 0

    @property
    def AI_STOCKS(self) -> Dict[str, Dict]:
        """Get AI stock universe - dynamically or static fallback."""
        import time

        if self._use_dynamic:
            if self._dynamic_cache is None or (time.time() - self._dynamic_cache_time) > 3600:
                try:
                    discovery = get_dynamic_us_ticker_discovery()
                    self._dynamic_cache = discovery.discover_ai_stocks()
                    self._dynamic_cache_time = time.time()
                except Exception:
                    pass

            if self._dynamic_cache:
                return self._dynamic_cache

        return self._STATIC_AI_STOCKS

    def calculate_ai_exposure(
        self,
        ticker: str,
        revenue_ai_pct: float = None,
    ) -> Dict[str, Any]:
        """
        Calculate stock's AI exposure level.

        Args:
            ticker: Stock ticker
            revenue_ai_pct: Percentage of revenue from AI (if known)

        Returns:
            Dict with AI exposure analysis
        """
        if ticker not in self.AI_STOCKS:
            return {'is_ai_stock': False, 'ai_multiplier': 1.0}

        stock_info = self.AI_STOCKS[ticker]
        base_exposure = stock_info['exposure']
        ai_score = stock_info['ai_score']

        base_multiplier = self.EXPOSURE_MULTIPLIERS[base_exposure]

        # Adjust for AI score (1-10 scale)
        score_adjustment = 0.8 + (ai_score / 10) * 0.4  # 0.8 to 1.2

        # Adjust for revenue exposure if known
        if revenue_ai_pct is not None:
            if revenue_ai_pct > 0.50:
                revenue_adj = 1.30
            elif revenue_ai_pct > 0.30:
                revenue_adj = 1.15
            elif revenue_ai_pct > 0.15:
                revenue_adj = 1.05
            else:
                revenue_adj = 0.95
        else:
            revenue_adj = 1.0

        final_multiplier = base_multiplier * score_adjustment * revenue_adj

        return {
            'ticker': ticker,
            'is_ai_stock': True,
            'exposure_level': base_exposure,
            'category': stock_info['category'],
            'ai_score': ai_score,
            'base_multiplier': base_multiplier,
            'score_adjustment': score_adjustment,
            'revenue_adjustment': revenue_adj,
            'final_multiplier': final_multiplier,
        }

    def assess_ai_sentiment(
        self,
        news_sentiment: float,
        social_sentiment: float,
        analyst_sentiment: float,
        nvda_momentum: float,  # NVDA as AI bellwether
    ) -> Dict[str, Any]:
        """
        Assess overall AI market sentiment.

        Args:
            news_sentiment: News sentiment score (0-1)
            social_sentiment: Social media sentiment (0-1)
            analyst_sentiment: Analyst sentiment (0-1)
            nvda_momentum: NVDA price momentum (% above/below MA)

        Returns:
            Dict with AI sentiment analysis
        """
        # Weighted average sentiment
        weights = {'news': 0.25, 'social': 0.20, 'analyst': 0.25, 'nvda': 0.30}

        # Convert NVDA momentum to sentiment score (0-1)
        nvda_sentiment = min(max((nvda_momentum + 0.20) / 0.40, 0), 1)

        combined_sentiment = (
            news_sentiment * weights['news'] +
            social_sentiment * weights['social'] +
            analyst_sentiment * weights['analyst'] +
            nvda_sentiment * weights['nvda']
        )

        # Determine sentiment state
        if combined_sentiment >= self.AI_SENTIMENT_THRESHOLDS['euphoria']:
            state = 'euphoria'
        elif combined_sentiment >= self.AI_SENTIMENT_THRESHOLDS['bullish']:
            state = 'bullish'
        elif combined_sentiment >= self.AI_SENTIMENT_THRESHOLDS['positive']:
            state = 'positive'
        elif combined_sentiment >= self.AI_SENTIMENT_THRESHOLDS['neutral']:
            state = 'neutral'
        elif combined_sentiment >= self.AI_SENTIMENT_THRESHOLDS['cautious']:
            state = 'cautious'
        else:
            state = 'bearish'

        sentiment_multiplier = self.SENTIMENT_MULTIPLIERS[state]

        return {
            'combined_sentiment': combined_sentiment,
            'sentiment_state': state,
            'sentiment_multiplier': sentiment_multiplier,
            'components': {
                'news': news_sentiment,
                'social': social_sentiment,
                'analyst': analyst_sentiment,
                'nvda_momentum': nvda_momentum,
                'nvda_sentiment': nvda_sentiment,
            },
        }

    def detect_capex_phase(
        self,
        hyperscaler_capex_growth: float,
        cloud_revenue_growth: float,
        gpu_demand_signal: float,
    ) -> Dict[str, Any]:
        """
        Detect AI capex cycle phase.

        Args:
            hyperscaler_capex_growth: Y/Y capex growth of MSFT/GOOGL/AMZN/META
            cloud_revenue_growth: Y/Y cloud revenue growth
            gpu_demand_signal: GPU demand signal (-1 to 1)

        Returns:
            Dict with capex phase analysis
        """
        # Determine phase based on signals
        if hyperscaler_capex_growth > 0.40 and gpu_demand_signal > 0.5:
            phase = 'acceleration'
        elif hyperscaler_capex_growth > 0.20 and gpu_demand_signal > 0.2:
            phase = 'steady_growth'
        elif hyperscaler_capex_growth > 0 and gpu_demand_signal > -0.2:
            phase = 'plateau'
        elif hyperscaler_capex_growth > -0.10:
            phase = 'deceleration'
        else:
            phase = 'contraction'

        phase_info = self.AI_CAPEX_PHASES[phase]

        return {
            'capex_phase': phase,
            'phase_multiplier': phase_info['multiplier'],
            'signal': phase_info['signal'],
            'hyperscaler_capex_growth': hyperscaler_capex_growth,
            'cloud_revenue_growth': cloud_revenue_growth,
            'gpu_demand_signal': gpu_demand_signal,
        }

    def get_ai_thematic_signal(
        self,
        ticker: str,
        signal_type: str,
        sentiment_data: Dict = None,
        capex_data: Dict = None,
    ) -> Dict[str, Any]:
        """
        Get AI thematic signal adjustment.

        Args:
            ticker: Stock ticker
            signal_type: 'BUY' or 'SELL'
            sentiment_data: AI sentiment data
            capex_data: AI capex phase data

        Returns:
            Dict with AI thematic signal
        """
        if ticker not in self.AI_STOCKS:
            return {'is_ai_stock': False, 'multiplier': 1.0}

        is_buy = signal_type.upper() == 'BUY'

        # Get base AI exposure
        exposure = self.calculate_ai_exposure(ticker)
        exposure_mult = exposure['final_multiplier']

        # Apply sentiment if available
        if sentiment_data:
            sent_mult = sentiment_data.get('sentiment_multiplier', 1.0)
        else:
            sent_mult = 1.0

        # Apply capex phase if available
        if capex_data:
            capex_mult = capex_data.get('phase_multiplier', 1.0)
            capex_signal = capex_data.get('signal', 'long')
        else:
            capex_mult = 1.0
            capex_signal = 'long'

        # Check signal alignment with capex phase
        if is_buy:
            if capex_signal in ['aggressive_long', 'long']:
                alignment = 'aligned'
                alignment_mult = 1.15
            elif capex_signal in ['avoid', 'exit']:
                alignment = 'contrary'
                alignment_mult = 0.50
            else:
                alignment = 'neutral'
                alignment_mult = 1.0
        else:
            if capex_signal in ['avoid', 'exit', 'reduce']:
                alignment = 'aligned'
                alignment_mult = 1.15
            elif capex_signal in ['aggressive_long', 'long']:
                alignment = 'contrary'
                alignment_mult = 0.50
            else:
                alignment = 'neutral'
                alignment_mult = 1.0

        # Combine multipliers
        if is_buy:
            combined = exposure_mult * sent_mult * capex_mult * alignment_mult
        else:
            # For SELL, invert the bullish multipliers
            inv_exposure = 2.0 - exposure_mult if exposure_mult > 1.0 else exposure_mult
            inv_sent = 2.0 - sent_mult if sent_mult > 1.0 else sent_mult
            combined = inv_exposure * inv_sent * (2.0 - capex_mult) * alignment_mult

        return {
            'ticker': ticker,
            'is_ai_stock': True,
            'signal_type': signal_type,
            'exposure_multiplier': exposure_mult,
            'sentiment_multiplier': sent_mult,
            'capex_multiplier': capex_mult,
            'capex_signal': capex_signal,
            'alignment': alignment,
            'alignment_multiplier': alignment_mult,
            'combined_multiplier': combined,
            'stock_info': self.AI_STOCKS[ticker],
        }


class FedLiquidityRegimeOptimizer:
    """
    Optimize positions based on Federal Reserve liquidity regime.

    FIX 57: Fed Liquidity Regime Optimization

    The problem: Model doesn't account for Fed policy regime which
    dramatically affects equity market returns (QE bullish, QT bearish).

    Solution: Track Fed liquidity signals:
    1. Fed balance sheet trends
    2. RRP facility usage
    3. TGA balance changes
    4. Bank reserves levels
    5. Rate hike/cut cycle position

    Expected Alpha: +20-35% by aligning with Fed policy
    """

    # Liquidity regime definitions
    LIQUIDITY_REGIMES = {
        'qe_full': {
            'description': 'Active QE program',
            'equity_bias': 'very_bullish',
            'multiplier': 1.60,
            'risk_appetite': 'high',
        },
        'qe_tapering': {
            'description': 'QE tapering phase',
            'equity_bias': 'moderately_bullish',
            'multiplier': 1.30,
            'risk_appetite': 'medium_high',
        },
        'neutral': {
            'description': 'Fed on hold',
            'equity_bias': 'neutral',
            'multiplier': 1.00,
            'risk_appetite': 'medium',
        },
        'qt_gradual': {
            'description': 'Gradual QT',
            'equity_bias': 'moderately_bearish',
            'multiplier': 0.80,
            'risk_appetite': 'low_medium',
        },
        'qt_aggressive': {
            'description': 'Aggressive QT',
            'equity_bias': 'bearish',
            'multiplier': 0.60,
            'risk_appetite': 'low',
        },
        'crisis_response': {
            'description': 'Emergency liquidity',
            'equity_bias': 'volatile_bullish',
            'multiplier': 1.40,
            'risk_appetite': 'variable',
        },
    }

    # Rate cycle phases
    RATE_CYCLE_PHASES = {
        'cutting_aggressive': {'multiplier': 1.50, 'bias': 'bullish'},
        'cutting_gradual': {'multiplier': 1.30, 'bias': 'moderately_bullish'},
        'paused_low': {'multiplier': 1.20, 'bias': 'slightly_bullish'},
        'paused_neutral': {'multiplier': 1.00, 'bias': 'neutral'},
        'paused_high': {'multiplier': 0.90, 'bias': 'slightly_bearish'},
        'hiking_gradual': {'multiplier': 0.80, 'bias': 'moderately_bearish'},
        'hiking_aggressive': {'multiplier': 0.60, 'bias': 'bearish'},
    }

    # Balance sheet thresholds (in trillions)
    BALANCE_SHEET_THRESHOLDS = {
        'expanding_fast': 0.05,    # >5% monthly growth
        'expanding_slow': 0.01,   # 1-5% monthly growth
        'stable': -0.01,           # -1% to 1%
        'contracting_slow': -0.03, # -1% to -3%
        'contracting_fast': -0.05, # <-3%
    }

    # Sector sensitivity to Fed policy
    SECTOR_FED_SENSITIVITY = {
        'technology': 1.50,        # High duration, very sensitive
        'consumer_discretionary': 1.30,
        'financials': 1.20,        # Benefits from steeper curve
        'real_estate': 1.40,       # Rate sensitive
        'utilities': 1.30,         # Rate sensitive
        'healthcare': 0.80,        # Defensive, less sensitive
        'consumer_staples': 0.70,  # Defensive
        'energy': 0.90,            # Inflation hedge
        'materials': 1.00,
        'industrials': 1.10,
        'communications': 1.20,
    }

    def __init__(self):
        self.current_regime = 'neutral'
        self.rate_phase = 'paused_neutral'
        self.balance_sheet_trend = 0
        self.liquidity_indicators = {}

    def detect_liquidity_regime(
        self,
        balance_sheet_change: float,  # Monthly % change
        rrp_usage: float,             # Reverse repo in trillions
        bank_reserves: float,         # Bank reserves in trillions
        fed_funds_rate: float,
        rate_change_12m: float,       # 12-month rate change
    ) -> Dict[str, Any]:
        """
        Detect current Fed liquidity regime.

        Args:
            balance_sheet_change: Monthly balance sheet % change
            rrp_usage: Current RRP facility usage
            bank_reserves: Total bank reserves
            fed_funds_rate: Current fed funds rate
            rate_change_12m: Rate change over past 12 months

        Returns:
            Dict with regime analysis
        """
        signals = []

        # Balance sheet signal
        if balance_sheet_change >= self.BALANCE_SHEET_THRESHOLDS['expanding_fast']:
            bs_signal = 'expanding_fast'
            signals.append('balance_sheet_growing_fast')
        elif balance_sheet_change >= self.BALANCE_SHEET_THRESHOLDS['expanding_slow']:
            bs_signal = 'expanding_slow'
            signals.append('balance_sheet_growing')
        elif balance_sheet_change >= self.BALANCE_SHEET_THRESHOLDS['stable']:
            bs_signal = 'stable'
            signals.append('balance_sheet_stable')
        elif balance_sheet_change >= self.BALANCE_SHEET_THRESHOLDS['contracting_slow']:
            bs_signal = 'contracting_slow'
            signals.append('balance_sheet_shrinking')
        else:
            bs_signal = 'contracting_fast'
            signals.append('balance_sheet_shrinking_fast')

        # RRP signal (high RRP = excess liquidity)
        if rrp_usage > 2.0:
            rrp_signal = 'very_high'
            signals.append('rrp_very_high')
        elif rrp_usage > 1.0:
            rrp_signal = 'high'
            signals.append('rrp_high')
        elif rrp_usage > 0.3:
            rrp_signal = 'moderate'
        else:
            rrp_signal = 'low'
            signals.append('rrp_low')

        # Determine regime
        if bs_signal == 'expanding_fast':
            regime = 'qe_full'
        elif bs_signal == 'expanding_slow':
            regime = 'qe_tapering'
        elif bs_signal == 'contracting_fast':
            regime = 'qt_aggressive'
        elif bs_signal == 'contracting_slow':
            regime = 'qt_gradual'
        elif rrp_signal == 'very_high' and bs_signal == 'stable':
            regime = 'neutral'  # Excess liquidity but not expanding
        else:
            regime = 'neutral'

        # Rate cycle phase
        if rate_change_12m <= -1.5:
            rate_phase = 'cutting_aggressive'
        elif rate_change_12m <= -0.5:
            rate_phase = 'cutting_gradual'
        elif rate_change_12m <= 0.25:
            if fed_funds_rate < 2.0:
                rate_phase = 'paused_low'
            elif fed_funds_rate < 4.0:
                rate_phase = 'paused_neutral'
            else:
                rate_phase = 'paused_high'
        elif rate_change_12m <= 1.5:
            rate_phase = 'hiking_gradual'
        else:
            rate_phase = 'hiking_aggressive'

        regime_info = self.LIQUIDITY_REGIMES[regime]
        rate_info = self.RATE_CYCLE_PHASES[rate_phase]

        # Combined multiplier
        combined_mult = regime_info['multiplier'] * rate_info['multiplier']
        # Normalize to prevent extreme values
        combined_mult = min(max(combined_mult, 0.40), 2.00)

        return {
            'liquidity_regime': regime,
            'regime_multiplier': regime_info['multiplier'],
            'equity_bias': regime_info['equity_bias'],
            'risk_appetite': regime_info['risk_appetite'],
            'rate_phase': rate_phase,
            'rate_multiplier': rate_info['multiplier'],
            'rate_bias': rate_info['bias'],
            'combined_multiplier': combined_mult,
            'balance_sheet_change': balance_sheet_change,
            'bs_signal': bs_signal,
            'rrp_usage': rrp_usage,
            'rrp_signal': rrp_signal,
            'bank_reserves': bank_reserves,
            'fed_funds_rate': fed_funds_rate,
            'rate_change_12m': rate_change_12m,
            'signals': signals,
        }

    def apply_sector_sensitivity(
        self,
        sector: str,
        regime_multiplier: float,
    ) -> Dict[str, Any]:
        """
        Apply sector sensitivity to Fed regime multiplier.

        Args:
            sector: Stock sector
            regime_multiplier: Base regime multiplier

        Returns:
            Dict with sector-adjusted multiplier
        """
        sensitivity = self.SECTOR_FED_SENSITIVITY.get(sector, 1.0)

        # Apply sensitivity - higher sensitivity = more impact
        if regime_multiplier >= 1.0:
            adjustment = 1.0 + (regime_multiplier - 1.0) * sensitivity
        else:
            adjustment = 1.0 - (1.0 - regime_multiplier) * sensitivity

        return {
            'sector': sector,
            'sensitivity': sensitivity,
            'base_multiplier': regime_multiplier,
            'adjusted_multiplier': adjustment,
        }

    def get_fed_adjusted_signal(
        self,
        ticker: str,
        signal_type: str,
        sector: str,
        regime_data: Dict = None,
    ) -> Dict[str, Any]:
        """
        Get Fed-adjusted trading signal.

        Args:
            ticker: Stock ticker
            signal_type: 'BUY' or 'SELL'
            sector: Stock sector
            regime_data: Liquidity regime data

        Returns:
            Dict with Fed-adjusted signal
        """
        is_buy = signal_type.upper() == 'BUY'

        if regime_data:
            regime = regime_data.get('liquidity_regime', 'neutral')
            regime_mult = regime_data.get('combined_multiplier', 1.0)
            equity_bias = regime_data.get('equity_bias', 'neutral')
        else:
            regime = 'neutral'
            regime_mult = 1.0
            equity_bias = 'neutral'

        # Apply sector sensitivity
        sector_adj = self.apply_sector_sensitivity(sector, regime_mult)
        adjusted_mult = sector_adj['adjusted_multiplier']

        # Check signal alignment with regime
        bullish_biases = ['very_bullish', 'moderately_bullish', 'slightly_bullish', 'volatile_bullish']
        bearish_biases = ['bearish', 'moderately_bearish', 'slightly_bearish']

        if is_buy:
            if equity_bias in bullish_biases:
                alignment = 'aligned'
                final_mult = adjusted_mult
            elif equity_bias in bearish_biases:
                alignment = 'contrary'
                final_mult = adjusted_mult * 0.70
            else:
                alignment = 'neutral'
                final_mult = adjusted_mult
        else:
            if equity_bias in bearish_biases:
                alignment = 'aligned'
                final_mult = 2.0 - adjusted_mult
            elif equity_bias in bullish_biases:
                alignment = 'contrary'
                final_mult = (2.0 - adjusted_mult) * 0.70
            else:
                alignment = 'neutral'
                final_mult = 1.0

        return {
            'ticker': ticker,
            'signal_type': signal_type,
            'sector': sector,
            'liquidity_regime': regime,
            'equity_bias': equity_bias,
            'regime_multiplier': regime_mult,
            'sector_sensitivity': sector_adj['sensitivity'],
            'sector_adjusted': adjusted_mult,
            'alignment': alignment,
            'final_multiplier': final_mult,
        }


class RetailOptionsFlowAnalyzer:
    """
    Analyze retail options flow for signal enhancement.

    FIX 58: Retail Options Flow Analysis

    The problem: Model ignores retail options activity which can
    drive significant price momentum, especially in popular stocks.

    Solution: Track retail options flow:
    1. Call/put volume ratios
    2. Unusual options activity detection
    3. Gamma exposure mapping
    4. OPEX effects
    5. 0DTE flow impact

    Expected Alpha: +15-25% in high-options-activity stocks

    NOTE: Uses DynamicUSTickerDiscovery for real-time stock universe updates.
    This is US/INTL model ONLY - China stocks use DeepSeek model.
    """

    # Static fallback for high retail options interest stocks
    _STATIC_HIGH_RETAIL_OPTIONS_STOCKS = {
        # Mega caps with heavy options
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA',
        'AMD', 'NFLX', 'SPY', 'QQQ', 'IWM',

        # Meme stocks
        'GME', 'AMC', 'BBBY', 'BB', 'PLTR',

        # Popular retail stocks
        'NIO', 'RIVN', 'LCID', 'F', 'COIN', 'HOOD', 'SOFI',
        'UPST', 'AFRM', 'SQ', 'PYPL', 'RBLX', 'SNAP',
    }

    # Call/Put ratio thresholds
    CALL_PUT_THRESHOLDS = {
        'extreme_bullish': 3.0,    # >3.0 very bullish
        'bullish': 1.5,            # 1.5-3.0 bullish
        'slightly_bullish': 1.1,   # 1.1-1.5 slightly bullish
        'neutral': 0.9,            # 0.9-1.1 neutral
        'slightly_bearish': 0.7,   # 0.7-0.9 slightly bearish
        'bearish': 0.5,            # 0.5-0.7 bearish
        'extreme_bearish': 0.3,    # <0.5 very bearish
    }

    CALL_PUT_MULTIPLIERS = {
        'extreme_bullish': 1.50,
        'bullish': 1.30,
        'slightly_bullish': 1.15,
        'neutral': 1.00,
        'slightly_bearish': 0.85,
        'bearish': 0.70,
        'extreme_bearish': 0.50,
    }

    # Unusual activity thresholds
    UNUSUAL_ACTIVITY_THRESHOLDS = {
        'extreme': 5.0,     # >5x normal volume
        'very_high': 3.0,   # 3-5x normal
        'high': 2.0,        # 2-3x normal
        'elevated': 1.5,    # 1.5-2x normal
        'normal': 1.0,      # baseline
    }

    # Gamma exposure impact
    GAMMA_MULTIPLIERS = {
        'major_squeeze': 1.70,    # Massive gamma ramp
        'squeeze': 1.40,          # Gamma squeeze developing
        'positive': 1.20,         # Positive gamma (supportive)
        'neutral': 1.00,
        'negative': 0.85,         # Negative gamma (amplifies moves)
        'major_negative': 0.70,   # Major negative gamma
    }

    # OPEX week effects
    OPEX_MULTIPLIERS = {
        'opex_week': 1.15,        # Increased activity/volatility
        'monthly_opex': 1.25,     # Monthly options expiration
        'quarterly_opex': 1.35,   # Quarterly expiration
        'normal': 1.00,
    }

    def __init__(self, use_dynamic_discovery: bool = True):
        """Initialize with optional dynamic ticker discovery."""
        self.options_data = {}
        self.gamma_exposure = {}
        self.unusual_activity = {}
        self._use_dynamic = use_dynamic_discovery
        self._dynamic_cache = None
        self._dynamic_cache_time = 0

    @property
    def HIGH_RETAIL_OPTIONS_STOCKS(self) -> set:
        """Get high retail options stocks - dynamically or static fallback."""
        import time

        if self._use_dynamic:
            if self._dynamic_cache is None or (time.time() - self._dynamic_cache_time) > 3600:
                try:
                    discovery = get_dynamic_us_ticker_discovery()
                    self._dynamic_cache = discovery.discover_high_retail_options()
                    self._dynamic_cache_time = time.time()
                except Exception:
                    pass

            if self._dynamic_cache:
                return self._dynamic_cache

        return self._STATIC_HIGH_RETAIL_OPTIONS_STOCKS

    def analyze_call_put_ratio(
        self,
        ticker: str,
        call_volume: int,
        put_volume: int,
        avg_call_volume: int = None,
        avg_put_volume: int = None,
    ) -> Dict[str, Any]:
        """
        Analyze call/put volume ratio.

        Args:
            ticker: Stock ticker
            call_volume: Current call volume
            put_volume: Current put volume
            avg_call_volume: Average call volume
            avg_put_volume: Average put volume

        Returns:
            Dict with call/put analysis
        """
        if put_volume <= 0:
            cp_ratio = 10.0  # Cap at extreme
        else:
            cp_ratio = call_volume / put_volume

        # Determine signal
        if cp_ratio >= self.CALL_PUT_THRESHOLDS['extreme_bullish']:
            cp_signal = 'extreme_bullish'
        elif cp_ratio >= self.CALL_PUT_THRESHOLDS['bullish']:
            cp_signal = 'bullish'
        elif cp_ratio >= self.CALL_PUT_THRESHOLDS['slightly_bullish']:
            cp_signal = 'slightly_bullish'
        elif cp_ratio >= self.CALL_PUT_THRESHOLDS['neutral']:
            cp_signal = 'neutral'
        elif cp_ratio >= self.CALL_PUT_THRESHOLDS['slightly_bearish']:
            cp_signal = 'slightly_bearish'
        elif cp_ratio >= self.CALL_PUT_THRESHOLDS['bearish']:
            cp_signal = 'bearish'
        else:
            cp_signal = 'extreme_bearish'

        multiplier = self.CALL_PUT_MULTIPLIERS[cp_signal]

        # Check if unusual
        is_unusual = False
        if avg_call_volume and avg_put_volume:
            total_avg = avg_call_volume + avg_put_volume
            total_current = call_volume + put_volume
            if total_avg > 0:
                volume_ratio = total_current / total_avg
                is_unusual = volume_ratio >= self.UNUSUAL_ACTIVITY_THRESHOLDS['elevated']
            else:
                volume_ratio = 1.0
        else:
            volume_ratio = 1.0

        return {
            'ticker': ticker,
            'call_volume': call_volume,
            'put_volume': put_volume,
            'cp_ratio': cp_ratio,
            'cp_signal': cp_signal,
            'multiplier': multiplier,
            'is_unusual': is_unusual,
            'volume_ratio': volume_ratio,
            'is_high_retail': ticker in self.HIGH_RETAIL_OPTIONS_STOCKS,
        }

    def detect_unusual_activity(
        self,
        ticker: str,
        current_volume: int,
        avg_volume: int,
        premium_spent: float,
        avg_premium: float,
    ) -> Dict[str, Any]:
        """
        Detect unusual options activity.

        Args:
            ticker: Stock ticker
            current_volume: Current options volume
            avg_volume: Average options volume
            premium_spent: Total premium spent today
            avg_premium: Average daily premium

        Returns:
            Dict with unusual activity detection
        """
        if avg_volume <= 0:
            volume_ratio = 1.0
        else:
            volume_ratio = current_volume / avg_volume

        if avg_premium <= 0:
            premium_ratio = 1.0
        else:
            premium_ratio = premium_spent / avg_premium

        # Combined unusual score
        unusual_score = (volume_ratio + premium_ratio) / 2

        if unusual_score >= self.UNUSUAL_ACTIVITY_THRESHOLDS['extreme']:
            activity_level = 'extreme'
            multiplier = 1.50
        elif unusual_score >= self.UNUSUAL_ACTIVITY_THRESHOLDS['very_high']:
            activity_level = 'very_high'
            multiplier = 1.35
        elif unusual_score >= self.UNUSUAL_ACTIVITY_THRESHOLDS['high']:
            activity_level = 'high'
            multiplier = 1.20
        elif unusual_score >= self.UNUSUAL_ACTIVITY_THRESHOLDS['elevated']:
            activity_level = 'elevated'
            multiplier = 1.10
        else:
            activity_level = 'normal'
            multiplier = 1.00

        return {
            'ticker': ticker,
            'current_volume': current_volume,
            'avg_volume': avg_volume,
            'volume_ratio': volume_ratio,
            'premium_spent': premium_spent,
            'avg_premium': avg_premium,
            'premium_ratio': premium_ratio,
            'unusual_score': unusual_score,
            'activity_level': activity_level,
            'multiplier': multiplier,
        }

    def analyze_gamma_exposure(
        self,
        ticker: str,
        net_gamma: float,
        spot_price: float,
        gamma_flip_level: float = None,
    ) -> Dict[str, Any]:
        """
        Analyze gamma exposure and potential squeeze.

        Args:
            ticker: Stock ticker
            net_gamma: Net gamma exposure (positive = dealers long gamma)
            spot_price: Current stock price
            gamma_flip_level: Price where gamma flips

        Returns:
            Dict with gamma analysis
        """
        # Normalize gamma relative to price
        gamma_normalized = net_gamma / spot_price if spot_price > 0 else 0

        # Determine gamma state
        if gamma_normalized > 0.05:
            gamma_state = 'major_squeeze'
        elif gamma_normalized > 0.02:
            gamma_state = 'squeeze'
        elif gamma_normalized > 0:
            gamma_state = 'positive'
        elif gamma_normalized > -0.02:
            gamma_state = 'neutral'
        elif gamma_normalized > -0.05:
            gamma_state = 'negative'
        else:
            gamma_state = 'major_negative'

        multiplier = self.GAMMA_MULTIPLIERS[gamma_state]

        # Check proximity to gamma flip
        near_flip = False
        if gamma_flip_level:
            distance_to_flip = abs(spot_price - gamma_flip_level) / spot_price
            near_flip = distance_to_flip < 0.03  # Within 3%

        return {
            'ticker': ticker,
            'net_gamma': net_gamma,
            'gamma_normalized': gamma_normalized,
            'gamma_state': gamma_state,
            'multiplier': multiplier,
            'spot_price': spot_price,
            'gamma_flip_level': gamma_flip_level,
            'near_flip': near_flip,
        }

    def check_opex_effect(
        self,
        days_to_opex: int,
        is_monthly: bool = False,
        is_quarterly: bool = False,
    ) -> Dict[str, Any]:
        """
        Check for OPEX-related effects.

        Args:
            days_to_opex: Days until options expiration
            is_monthly: Is monthly expiration
            is_quarterly: Is quarterly expiration

        Returns:
            Dict with OPEX effect
        """
        in_opex_week = days_to_opex <= 5

        if in_opex_week:
            if is_quarterly:
                opex_type = 'quarterly_opex'
            elif is_monthly:
                opex_type = 'monthly_opex'
            else:
                opex_type = 'opex_week'
        else:
            opex_type = 'normal'

        multiplier = self.OPEX_MULTIPLIERS[opex_type]

        return {
            'days_to_opex': days_to_opex,
            'in_opex_week': in_opex_week,
            'is_monthly': is_monthly,
            'is_quarterly': is_quarterly,
            'opex_type': opex_type,
            'multiplier': multiplier,
        }

    def get_options_flow_signal(
        self,
        ticker: str,
        signal_type: str,
        cp_data: Dict = None,
        unusual_data: Dict = None,
        gamma_data: Dict = None,
        opex_data: Dict = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive options flow signal.

        Args:
            ticker: Stock ticker
            signal_type: 'BUY' or 'SELL'
            cp_data: Call/put ratio data
            unusual_data: Unusual activity data
            gamma_data: Gamma exposure data
            opex_data: OPEX effect data

        Returns:
            Dict with options flow signal
        """
        is_buy = signal_type.upper() == 'BUY'
        multipliers = []
        signals = []

        is_high_retail = ticker in self.HIGH_RETAIL_OPTIONS_STOCKS

        # Call/Put ratio component
        if cp_data:
            cp_mult = cp_data.get('multiplier', 1.0)
            cp_signal = cp_data.get('cp_signal', 'neutral')

            # Check alignment
            bullish_signals = ['extreme_bullish', 'bullish', 'slightly_bullish']
            bearish_signals = ['extreme_bearish', 'bearish', 'slightly_bearish']

            if is_buy and cp_signal in bullish_signals:
                multipliers.append(cp_mult)
                signals.append(f"cp_{cp_signal}")
            elif not is_buy and cp_signal in bearish_signals:
                multipliers.append(2.0 - cp_mult)
                signals.append(f"cp_{cp_signal}")

        # Unusual activity component
        if unusual_data:
            if unusual_data.get('activity_level', 'normal') != 'normal':
                unusual_mult = unusual_data.get('multiplier', 1.0)
                multipliers.append(unusual_mult)
                signals.append(f"unusual_{unusual_data['activity_level']}")

        # Gamma component
        if gamma_data:
            gamma_mult = gamma_data.get('multiplier', 1.0)
            gamma_state = gamma_data.get('gamma_state', 'neutral')

            if is_buy and gamma_mult > 1.0:
                multipliers.append(gamma_mult)
                signals.append(f"gamma_{gamma_state}")
            elif not is_buy and gamma_mult < 1.0:
                multipliers.append(2.0 - gamma_mult)
                signals.append(f"gamma_{gamma_state}")

        # OPEX component
        if opex_data:
            opex_mult = opex_data.get('multiplier', 1.0)
            if opex_mult != 1.0:
                multipliers.append(opex_mult)
                signals.append(opex_data.get('opex_type', 'normal'))

        # Apply high retail boost if applicable
        if is_high_retail and multipliers:
            retail_boost = 1.15
        else:
            retail_boost = 1.0

        # Combine multipliers
        if multipliers:
            combined = np.prod(multipliers) ** (1 / len(multipliers)) * retail_boost
        else:
            combined = 1.0

        return {
            'ticker': ticker,
            'signal_type': signal_type,
            'is_high_retail': is_high_retail,
            'signals': signals,
            'individual_multipliers': multipliers,
            'retail_boost': retail_boost,
            'combined_multiplier': combined,
        }


class MemeStockPatternDetector:
    """
    Detect and exploit meme stock patterns.

    FIX 59: Meme Stock Pattern Detection

    The problem: Model treats meme stocks like regular stocks, missing
    their unique social-driven dynamics and extreme momentum potential.

    Solution: Track meme stock indicators:
    1. Social media sentiment spikes
    2. Short interest squeeze potential
    3. Retail volume surges
    4. Pattern recognition from GME/AMC playbook
    5. Gamma squeeze setups

    Expected Alpha: +50-100% during meme events (high risk)

    NOTE: Uses DynamicUSTickerDiscovery for real-time stock universe updates.
    This is US/INTL model ONLY - China stocks use DeepSeek model.
    """

    # Static fallback for known meme stocks
    _STATIC_MEME_STOCKS = {
        # Original meme stocks
        'GME': {'tier': 1, 'category': 'original', 'base_volatility': 3.0},
        'AMC': {'tier': 1, 'category': 'original', 'base_volatility': 2.5},

        # Second wave meme stocks
        'BBBY': {'tier': 2, 'category': 'second_wave', 'base_volatility': 2.5},
        'BB': {'tier': 2, 'category': 'second_wave', 'base_volatility': 2.0},
        'NOK': {'tier': 2, 'category': 'second_wave', 'base_volatility': 1.5},
        'KOSS': {'tier': 2, 'category': 'second_wave', 'base_volatility': 2.5},
        'EXPR': {'tier': 2, 'category': 'second_wave', 'base_volatility': 2.0},

        # Newer meme stocks
        'PLTR': {'tier': 2, 'category': 'tech_meme', 'base_volatility': 1.8},
        'SOFI': {'tier': 2, 'category': 'fintech_meme', 'base_volatility': 2.0},
        'HOOD': {'tier': 2, 'category': 'fintech_meme', 'base_volatility': 2.2},
        'WISH': {'tier': 3, 'category': 'retail_meme', 'base_volatility': 2.5},
        'CLOV': {'tier': 3, 'category': 'healthcare_meme', 'base_volatility': 2.5},
        'WKHS': {'tier': 3, 'category': 'ev_meme', 'base_volatility': 2.5},

        # EV meme stocks
        'RIVN': {'tier': 2, 'category': 'ev_meme', 'base_volatility': 2.0},
        'LCID': {'tier': 2, 'category': 'ev_meme', 'base_volatility': 2.0},
        'NIO': {'tier': 2, 'category': 'ev_meme', 'base_volatility': 2.2},

        # Crypto-adjacent
        'COIN': {'tier': 2, 'category': 'crypto_meme', 'base_volatility': 2.3},
        'MSTR': {'tier': 2, 'category': 'crypto_meme', 'base_volatility': 2.5},
    }

    # Social sentiment thresholds
    SOCIAL_THRESHOLDS = {
        'viral': 10.0,     # 10x normal mentions
        'hot': 5.0,        # 5x normal
        'trending': 2.5,   # 2.5x normal
        'elevated': 1.5,   # 1.5x normal
        'normal': 1.0,
    }

    SOCIAL_MULTIPLIERS = {
        'viral': 2.00,
        'hot': 1.60,
        'trending': 1.30,
        'elevated': 1.15,
        'normal': 1.00,
    }

    # Short squeeze potential
    SHORT_INTEREST_THRESHOLDS = {
        'extreme': 0.40,    # >40% short interest
        'very_high': 0.30,  # 30-40%
        'high': 0.20,       # 20-30%
        'elevated': 0.15,   # 15-20%
        'moderate': 0.10,   # 10-15%
        'low': 0.05,        # <10%
    }

    SHORT_SQUEEZE_MULTIPLIERS = {
        'extreme': 2.50,
        'very_high': 2.00,
        'high': 1.60,
        'elevated': 1.30,
        'moderate': 1.15,
        'low': 1.00,
    }

    # Meme event phases
    MEME_PHASES = {
        'accumulation': {'multiplier': 1.40, 'risk': 'medium'},
        'breakout': {'multiplier': 2.00, 'risk': 'high'},
        'squeeze': {'multiplier': 2.50, 'risk': 'very_high'},
        'peak': {'multiplier': 0.50, 'risk': 'extreme'},
        'crash': {'multiplier': 0.30, 'risk': 'extreme'},
        'consolidation': {'multiplier': 1.00, 'risk': 'medium'},
    }

    def __init__(self, use_dynamic_discovery: bool = True):
        """Initialize with optional dynamic ticker discovery."""
        self.social_data = {}
        self.short_interest = {}
        self.meme_phase = {}
        self._use_dynamic = use_dynamic_discovery
        self._dynamic_cache = None
        self._dynamic_cache_time = 0

    @property
    def MEME_STOCKS(self) -> Dict[str, Dict]:
        """Get meme stock universe - dynamically or static fallback."""
        import time

        if self._use_dynamic:
            if self._dynamic_cache is None or (time.time() - self._dynamic_cache_time) > 3600:
                try:
                    discovery = get_dynamic_us_ticker_discovery()
                    self._dynamic_cache = discovery.discover_meme_stocks()
                    self._dynamic_cache_time = time.time()
                except Exception:
                    pass

            if self._dynamic_cache:
                return self._dynamic_cache

        return self._STATIC_MEME_STOCKS

    def detect_social_momentum(
        self,
        ticker: str,
        reddit_mentions: int,
        avg_reddit_mentions: int,
        twitter_mentions: int,
        avg_twitter_mentions: int,
        sentiment_score: float,  # -1 to 1
    ) -> Dict[str, Any]:
        """
        Detect social media momentum for meme stock.

        Args:
            ticker: Stock ticker
            reddit_mentions: Current Reddit mentions
            avg_reddit_mentions: Average Reddit mentions
            twitter_mentions: Current Twitter mentions
            avg_twitter_mentions: Average Twitter mentions
            sentiment_score: Combined sentiment score

        Returns:
            Dict with social momentum analysis
        """
        # Calculate mention ratios
        if avg_reddit_mentions > 0:
            reddit_ratio = reddit_mentions / avg_reddit_mentions
        else:
            reddit_ratio = 1.0

        if avg_twitter_mentions > 0:
            twitter_ratio = twitter_mentions / avg_twitter_mentions
        else:
            twitter_ratio = 1.0

        # Combined social ratio (weight Reddit higher for meme stocks)
        social_ratio = reddit_ratio * 0.6 + twitter_ratio * 0.4

        # Determine social state
        if social_ratio >= self.SOCIAL_THRESHOLDS['viral']:
            social_state = 'viral'
        elif social_ratio >= self.SOCIAL_THRESHOLDS['hot']:
            social_state = 'hot'
        elif social_ratio >= self.SOCIAL_THRESHOLDS['trending']:
            social_state = 'trending'
        elif social_ratio >= self.SOCIAL_THRESHOLDS['elevated']:
            social_state = 'elevated'
        else:
            social_state = 'normal'

        base_multiplier = self.SOCIAL_MULTIPLIERS[social_state]

        # Adjust for sentiment
        if sentiment_score > 0.5:
            sentiment_adj = 1.20
        elif sentiment_score > 0.2:
            sentiment_adj = 1.10
        elif sentiment_score < -0.2:
            sentiment_adj = 0.80
        elif sentiment_score < -0.5:
            sentiment_adj = 0.60
        else:
            sentiment_adj = 1.0

        final_multiplier = base_multiplier * sentiment_adj

        return {
            'ticker': ticker,
            'is_meme': ticker in self.MEME_STOCKS,
            'reddit_mentions': reddit_mentions,
            'reddit_ratio': reddit_ratio,
            'twitter_mentions': twitter_mentions,
            'twitter_ratio': twitter_ratio,
            'social_ratio': social_ratio,
            'social_state': social_state,
            'sentiment_score': sentiment_score,
            'sentiment_adjustment': sentiment_adj,
            'base_multiplier': base_multiplier,
            'final_multiplier': final_multiplier,
        }

    def analyze_squeeze_potential(
        self,
        ticker: str,
        short_interest_pct: float,
        days_to_cover: float,
        cost_to_borrow: float,
        recent_price_change: float,
    ) -> Dict[str, Any]:
        """
        Analyze short squeeze potential.

        Args:
            ticker: Stock ticker
            short_interest_pct: Short interest as % of float
            days_to_cover: Days to cover short position
            cost_to_borrow: Annual cost to borrow %
            recent_price_change: Recent price change %

        Returns:
            Dict with squeeze potential analysis
        """
        # Determine short interest level
        if short_interest_pct >= self.SHORT_INTEREST_THRESHOLDS['extreme']:
            si_level = 'extreme'
        elif short_interest_pct >= self.SHORT_INTEREST_THRESHOLDS['very_high']:
            si_level = 'very_high'
        elif short_interest_pct >= self.SHORT_INTEREST_THRESHOLDS['high']:
            si_level = 'high'
        elif short_interest_pct >= self.SHORT_INTEREST_THRESHOLDS['elevated']:
            si_level = 'elevated'
        elif short_interest_pct >= self.SHORT_INTEREST_THRESHOLDS['moderate']:
            si_level = 'moderate'
        else:
            si_level = 'low'

        base_multiplier = self.SHORT_SQUEEZE_MULTIPLIERS[si_level]

        # Days to cover adjustment
        if days_to_cover > 10:
            dtc_adj = 1.30
        elif days_to_cover > 5:
            dtc_adj = 1.15
        else:
            dtc_adj = 1.00

        # Cost to borrow adjustment (high CTB = harder to short)
        if cost_to_borrow > 50:
            ctb_adj = 1.40
        elif cost_to_borrow > 20:
            ctb_adj = 1.20
        elif cost_to_borrow > 10:
            ctb_adj = 1.10
        else:
            ctb_adj = 1.00

        # Price momentum adjustment
        if recent_price_change > 0.20:
            momentum_adj = 1.30  # Squeeze potentially starting
        elif recent_price_change > 0.10:
            momentum_adj = 1.15
        elif recent_price_change < -0.20:
            momentum_adj = 0.70  # Squeeze unlikely
        else:
            momentum_adj = 1.00

        # Calculate squeeze score
        squeeze_score = (
            short_interest_pct * 2 +
            min(days_to_cover / 10, 1) +
            min(cost_to_borrow / 50, 1)
        )

        final_multiplier = base_multiplier * dtc_adj * ctb_adj * momentum_adj

        return {
            'ticker': ticker,
            'short_interest_pct': short_interest_pct,
            'si_level': si_level,
            'days_to_cover': days_to_cover,
            'cost_to_borrow': cost_to_borrow,
            'recent_price_change': recent_price_change,
            'squeeze_score': squeeze_score,
            'base_multiplier': base_multiplier,
            'dtc_adjustment': dtc_adj,
            'ctb_adjustment': ctb_adj,
            'momentum_adjustment': momentum_adj,
            'final_multiplier': final_multiplier,
            'squeeze_potential': 'high' if squeeze_score > 1.5 else 'moderate' if squeeze_score > 0.8 else 'low',
        }

    def detect_meme_phase(
        self,
        ticker: str,
        price_vs_20ma: float,
        volume_ratio: float,
        social_ratio: float,
        days_since_spike: int,
    ) -> Dict[str, Any]:
        """
        Detect current meme stock phase.

        Args:
            ticker: Stock ticker
            price_vs_20ma: Price % above/below 20-day MA
            volume_ratio: Current volume / average volume
            social_ratio: Social mentions ratio
            days_since_spike: Days since last major spike

        Returns:
            Dict with meme phase detection
        """
        # Determine phase based on multiple signals
        if social_ratio > 5 and volume_ratio > 3 and price_vs_20ma > 0.30:
            phase = 'squeeze'
        elif social_ratio > 3 and volume_ratio > 2 and price_vs_20ma > 0.15:
            phase = 'breakout'
        elif social_ratio > 2 and price_vs_20ma > 0 and days_since_spike > 10:
            phase = 'accumulation'
        elif price_vs_20ma > 0.50 and volume_ratio < 2:
            phase = 'peak'
        elif price_vs_20ma < -0.30 and days_since_spike < 10:
            phase = 'crash'
        else:
            phase = 'consolidation'

        phase_info = self.MEME_PHASES[phase]

        return {
            'ticker': ticker,
            'phase': phase,
            'multiplier': phase_info['multiplier'],
            'risk_level': phase_info['risk'],
            'price_vs_20ma': price_vs_20ma,
            'volume_ratio': volume_ratio,
            'social_ratio': social_ratio,
            'days_since_spike': days_since_spike,
        }

    def get_meme_signal(
        self,
        ticker: str,
        signal_type: str,
        social_data: Dict = None,
        squeeze_data: Dict = None,
        phase_data: Dict = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive meme stock signal.

        Args:
            ticker: Stock ticker
            signal_type: 'BUY' or 'SELL'
            social_data: Social momentum data
            squeeze_data: Squeeze potential data
            phase_data: Meme phase data

        Returns:
            Dict with meme stock signal
        """
        if ticker not in self.MEME_STOCKS:
            return {'is_meme': False, 'multiplier': 1.0}

        is_buy = signal_type.upper() == 'BUY'
        stock_info = self.MEME_STOCKS[ticker]

        multipliers = []
        signals = []
        warnings = []

        # Social momentum component
        if social_data:
            social_mult = social_data.get('final_multiplier', 1.0)
            if social_mult > 1.0:
                multipliers.append(social_mult)
                signals.append(social_data.get('social_state', 'normal'))

        # Squeeze potential (primarily for BUY)
        if squeeze_data and is_buy:
            squeeze_mult = squeeze_data.get('final_multiplier', 1.0)
            squeeze_potential = squeeze_data.get('squeeze_potential', 'low')
            if squeeze_mult > 1.0:
                multipliers.append(squeeze_mult)
                signals.append(f"squeeze_{squeeze_potential}")

        # Phase component
        if phase_data:
            phase = phase_data.get('phase', 'consolidation')
            phase_mult = phase_data.get('multiplier', 1.0)
            risk = phase_data.get('risk_level', 'medium')

            if is_buy:
                if phase in ['accumulation', 'breakout', 'squeeze']:
                    multipliers.append(phase_mult)
                    signals.append(f"phase_{phase}")
                elif phase in ['peak', 'crash']:
                    multipliers.append(phase_mult)  # This will reduce
                    warnings.append(f"Warning: {phase} phase - high risk")
            else:  # SELL
                if phase in ['peak', 'crash']:
                    multipliers.append(2.0 - phase_mult)
                    signals.append(f"phase_{phase}")

            if risk in ['very_high', 'extreme']:
                warnings.append(f"Risk level: {risk}")

        # Apply meme tier adjustment
        tier = stock_info['tier']
        if tier == 1:
            tier_mult = 1.30
        elif tier == 2:
            tier_mult = 1.15
        else:
            tier_mult = 1.00

        # Combine multipliers
        if multipliers:
            base_combined = np.prod(multipliers) ** (1 / len(multipliers))
            final_multiplier = base_combined * tier_mult
        else:
            final_multiplier = tier_mult

        # Cap extreme multipliers
        final_multiplier = min(max(final_multiplier, 0.20), 3.00)

        return {
            'ticker': ticker,
            'is_meme': True,
            'signal_type': signal_type,
            'tier': tier,
            'category': stock_info['category'],
            'base_volatility': stock_info['base_volatility'],
            'signals': signals,
            'warnings': warnings,
            'individual_multipliers': multipliers,
            'tier_multiplier': tier_mult,
            'final_multiplier': final_multiplier,
        }


class EarningsDrivenSectorRotation:
    """
    Rotate sector exposure based on earnings momentum.

    FIX 60: Earnings-Driven Sector Rotation

    The problem: Model doesn't systematically exploit earnings momentum
    across sectors, missing predictable sector rotation opportunities.

    Solution: Track sector earnings signals:
    1. Earnings beat/miss trends by sector
    2. Guidance revision patterns
    3. Estimate revision momentum
    4. Sector leadership rotation
    5. Earnings surprise clustering

    Expected Alpha: +20-30% through sector timing
    """

    # Sector definitions
    SECTORS = {
        'technology': {
            'etf': 'XLK',
            'earnings_sensitivity': 1.50,
            'beat_multiplier': 1.40,
            'miss_multiplier': 0.60,
        },
        'financials': {
            'etf': 'XLF',
            'earnings_sensitivity': 1.30,
            'beat_multiplier': 1.30,
            'miss_multiplier': 0.70,
        },
        'healthcare': {
            'etf': 'XLV',
            'earnings_sensitivity': 1.10,
            'beat_multiplier': 1.20,
            'miss_multiplier': 0.80,
        },
        'consumer_discretionary': {
            'etf': 'XLY',
            'earnings_sensitivity': 1.40,
            'beat_multiplier': 1.35,
            'miss_multiplier': 0.65,
        },
        'consumer_staples': {
            'etf': 'XLP',
            'earnings_sensitivity': 0.80,
            'beat_multiplier': 1.15,
            'miss_multiplier': 0.90,
        },
        'industrials': {
            'etf': 'XLI',
            'earnings_sensitivity': 1.20,
            'beat_multiplier': 1.25,
            'miss_multiplier': 0.75,
        },
        'energy': {
            'etf': 'XLE',
            'earnings_sensitivity': 1.30,
            'beat_multiplier': 1.30,
            'miss_multiplier': 0.70,
        },
        'utilities': {
            'etf': 'XLU',
            'earnings_sensitivity': 0.70,
            'beat_multiplier': 1.10,
            'miss_multiplier': 0.90,
        },
        'materials': {
            'etf': 'XLB',
            'earnings_sensitivity': 1.10,
            'beat_multiplier': 1.20,
            'miss_multiplier': 0.80,
        },
        'real_estate': {
            'etf': 'XLRE',
            'earnings_sensitivity': 0.90,
            'beat_multiplier': 1.15,
            'miss_multiplier': 0.85,
        },
        'communications': {
            'etf': 'XLC',
            'earnings_sensitivity': 1.30,
            'beat_multiplier': 1.30,
            'miss_multiplier': 0.70,
        },
    }

    # Earnings momentum thresholds
    EARNINGS_MOMENTUM_THRESHOLDS = {
        'strong_acceleration': 0.15,   # >15% beat rate improvement
        'acceleration': 0.08,          # 8-15% improvement
        'stable_positive': 0.02,       # 2-8% improvement
        'neutral': -0.02,              # -2% to 2%
        'stable_negative': -0.08,      # -2% to -8%
        'deceleration': -0.15,         # -8% to -15%
        'strong_deceleration': -0.30,  # <-15%
    }

    MOMENTUM_MULTIPLIERS = {
        'strong_acceleration': 1.50,
        'acceleration': 1.30,
        'stable_positive': 1.15,
        'neutral': 1.00,
        'stable_negative': 0.85,
        'deceleration': 0.70,
        'strong_deceleration': 0.50,
    }

    # Revision direction thresholds
    REVISION_THRESHOLDS = {
        'strong_up': 0.05,       # >5% upward revision
        'up': 0.02,              # 2-5% up
        'stable': -0.02,         # -2% to 2%
        'down': -0.05,           # -2% to -5%
        'strong_down': -0.10,    # <-5%
    }

    REVISION_MULTIPLIERS = {
        'strong_up': 1.40,
        'up': 1.20,
        'stable': 1.00,
        'down': 0.80,
        'strong_down': 0.60,
    }

    def __init__(self):
        self.sector_earnings = {}
        self.revision_data = {}
        self.leadership_rankings = []

    def analyze_sector_earnings(
        self,
        sector: str,
        beat_rate: float,
        prev_beat_rate: float,
        avg_surprise: float,
        guidance_trend: float,  # -1 to 1
    ) -> Dict[str, Any]:
        """
        Analyze sector earnings momentum.

        Args:
            sector: Sector name
            beat_rate: Current quarter beat rate
            prev_beat_rate: Previous quarter beat rate
            avg_surprise: Average earnings surprise %
            guidance_trend: Guidance trend score

        Returns:
            Dict with sector earnings analysis
        """
        if sector not in self.SECTORS:
            return {'valid': False, 'multiplier': 1.0}

        sector_info = self.SECTORS[sector]

        # Calculate momentum
        beat_momentum = beat_rate - prev_beat_rate

        # Determine momentum state
        if beat_momentum >= self.EARNINGS_MOMENTUM_THRESHOLDS['strong_acceleration']:
            momentum_state = 'strong_acceleration'
        elif beat_momentum >= self.EARNINGS_MOMENTUM_THRESHOLDS['acceleration']:
            momentum_state = 'acceleration'
        elif beat_momentum >= self.EARNINGS_MOMENTUM_THRESHOLDS['stable_positive']:
            momentum_state = 'stable_positive'
        elif beat_momentum >= self.EARNINGS_MOMENTUM_THRESHOLDS['neutral']:
            momentum_state = 'neutral'
        elif beat_momentum >= self.EARNINGS_MOMENTUM_THRESHOLDS['stable_negative']:
            momentum_state = 'stable_negative'
        elif beat_momentum >= self.EARNINGS_MOMENTUM_THRESHOLDS['deceleration']:
            momentum_state = 'deceleration'
        else:
            momentum_state = 'strong_deceleration'

        base_multiplier = self.MOMENTUM_MULTIPLIERS[momentum_state]

        # Apply sector sensitivity
        sensitivity = sector_info['earnings_sensitivity']
        if base_multiplier >= 1.0:
            sensitivity_adjusted = 1.0 + (base_multiplier - 1.0) * sensitivity
        else:
            sensitivity_adjusted = 1.0 - (1.0 - base_multiplier) * sensitivity

        # Guidance adjustment
        if guidance_trend > 0.5:
            guidance_adj = 1.15
        elif guidance_trend > 0.2:
            guidance_adj = 1.08
        elif guidance_trend < -0.5:
            guidance_adj = 0.85
        elif guidance_trend < -0.2:
            guidance_adj = 0.92
        else:
            guidance_adj = 1.0

        # Surprise magnitude adjustment
        if avg_surprise > 0.10:
            surprise_adj = 1.20
        elif avg_surprise > 0.05:
            surprise_adj = 1.10
        elif avg_surprise < -0.05:
            surprise_adj = 0.85
        elif avg_surprise < -0.10:
            surprise_adj = 0.75
        else:
            surprise_adj = 1.0

        final_multiplier = sensitivity_adjusted * guidance_adj * surprise_adj

        return {
            'sector': sector,
            'valid': True,
            'beat_rate': beat_rate,
            'prev_beat_rate': prev_beat_rate,
            'beat_momentum': beat_momentum,
            'momentum_state': momentum_state,
            'avg_surprise': avg_surprise,
            'guidance_trend': guidance_trend,
            'base_multiplier': base_multiplier,
            'sensitivity': sensitivity,
            'sensitivity_adjusted': sensitivity_adjusted,
            'guidance_adjustment': guidance_adj,
            'surprise_adjustment': surprise_adj,
            'final_multiplier': final_multiplier,
            'sector_etf': sector_info['etf'],
        }

    def analyze_revision_momentum(
        self,
        sector: str,
        current_estimate: float,
        estimate_30d_ago: float,
        estimate_90d_ago: float,
    ) -> Dict[str, Any]:
        """
        Analyze estimate revision momentum.

        Args:
            sector: Sector name
            current_estimate: Current consensus estimate
            estimate_30d_ago: Estimate 30 days ago
            estimate_90d_ago: Estimate 90 days ago

        Returns:
            Dict with revision momentum analysis
        """
        if sector not in self.SECTORS:
            return {'valid': False, 'multiplier': 1.0}

        # Calculate revision rates
        if estimate_30d_ago > 0:
            revision_30d = (current_estimate - estimate_30d_ago) / estimate_30d_ago
        else:
            revision_30d = 0

        if estimate_90d_ago > 0:
            revision_90d = (current_estimate - estimate_90d_ago) / estimate_90d_ago
        else:
            revision_90d = 0

        # Use 30-day revision as primary signal
        if revision_30d >= self.REVISION_THRESHOLDS['strong_up']:
            revision_state = 'strong_up'
        elif revision_30d >= self.REVISION_THRESHOLDS['up']:
            revision_state = 'up'
        elif revision_30d >= self.REVISION_THRESHOLDS['stable']:
            revision_state = 'stable'
        elif revision_30d >= self.REVISION_THRESHOLDS['down']:
            revision_state = 'down'
        else:
            revision_state = 'strong_down'

        base_multiplier = self.REVISION_MULTIPLIERS[revision_state]

        # Acceleration check (comparing 30d to 90d momentum)
        avg_90d_monthly = revision_90d / 3 if revision_90d != 0 else 0
        if revision_30d > avg_90d_monthly * 1.5:
            acceleration_adj = 1.15
        elif revision_30d < avg_90d_monthly * 0.5:
            acceleration_adj = 0.85
        else:
            acceleration_adj = 1.0

        final_multiplier = base_multiplier * acceleration_adj

        return {
            'sector': sector,
            'valid': True,
            'current_estimate': current_estimate,
            'revision_30d': revision_30d,
            'revision_90d': revision_90d,
            'revision_state': revision_state,
            'base_multiplier': base_multiplier,
            'acceleration_adjustment': acceleration_adj,
            'final_multiplier': final_multiplier,
        }

    def rank_sector_leadership(
        self,
        sector_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Rank sectors by earnings leadership.

        Args:
            sector_scores: Dict of sector to composite score

        Returns:
            Dict with sector rankings
        """
        # Sort sectors by score
        ranked = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)

        # Assign tier multipliers
        tier_multipliers = {
            1: 1.40,  # Top 2 sectors
            2: 1.20,  # Next 3 sectors
            3: 1.00,  # Middle sectors
            4: 0.80,  # Lower sectors
            5: 0.60,  # Bottom sectors
        }

        rankings = {}
        for i, (sector, score) in enumerate(ranked):
            if i < 2:
                tier = 1
            elif i < 5:
                tier = 2
            elif i < 8:
                tier = 3
            elif i < 10:
                tier = 4
            else:
                tier = 5

            rankings[sector] = {
                'rank': i + 1,
                'score': score,
                'tier': tier,
                'multiplier': tier_multipliers[tier],
            }

        return {
            'rankings': rankings,
            'leader': ranked[0][0] if ranked else None,
            'laggard': ranked[-1][0] if ranked else None,
        }

    def get_earnings_rotation_signal(
        self,
        ticker: str,
        sector: str,
        signal_type: str,
        earnings_data: Dict = None,
        revision_data: Dict = None,
        leadership_data: Dict = None,
    ) -> Dict[str, Any]:
        """
        Get earnings-driven sector rotation signal.

        Args:
            ticker: Stock ticker
            sector: Stock sector
            signal_type: 'BUY' or 'SELL'
            earnings_data: Sector earnings data
            revision_data: Revision momentum data
            leadership_data: Sector leadership rankings

        Returns:
            Dict with earnings rotation signal
        """
        if sector not in self.SECTORS:
            return {'valid': False, 'multiplier': 1.0}

        is_buy = signal_type.upper() == 'BUY'
        multipliers = []
        signals = []

        sector_info = self.SECTORS[sector]

        # Earnings momentum component
        if earnings_data and earnings_data.get('valid'):
            earnings_mult = earnings_data.get('final_multiplier', 1.0)
            momentum_state = earnings_data.get('momentum_state', 'neutral')

            if is_buy:
                multipliers.append(earnings_mult)
                signals.append(f"earnings_{momentum_state}")
            else:
                multipliers.append(2.0 - earnings_mult if earnings_mult > 1.0 else earnings_mult)
                signals.append(f"earnings_{momentum_state}")

        # Revision momentum component
        if revision_data and revision_data.get('valid'):
            revision_mult = revision_data.get('final_multiplier', 1.0)
            revision_state = revision_data.get('revision_state', 'stable')

            if is_buy:
                multipliers.append(revision_mult)
                signals.append(f"revision_{revision_state}")
            else:
                multipliers.append(2.0 - revision_mult if revision_mult > 1.0 else revision_mult)

        # Leadership component
        if leadership_data:
            rankings = leadership_data.get('rankings', {})
            if sector in rankings:
                leadership_mult = rankings[sector].get('multiplier', 1.0)
                tier = rankings[sector].get('tier', 3)

                if is_buy:
                    multipliers.append(leadership_mult)
                    signals.append(f"tier_{tier}_leadership")
                else:
                    multipliers.append(2.0 - leadership_mult if leadership_mult > 1.0 else leadership_mult)

        # Combine multipliers
        if multipliers:
            combined = np.prod(multipliers) ** (1 / len(multipliers))
        else:
            combined = 1.0

        return {
            'ticker': ticker,
            'sector': sector,
            'signal_type': signal_type,
            'valid': True,
            'signals': signals,
            'individual_multipliers': multipliers,
            'combined_multiplier': combined,
            'sector_etf': sector_info['etf'],
            'beat_multiplier': sector_info['beat_multiplier'],
            'miss_multiplier': sector_info['miss_multiplier'],
        }


class RealTimeUSNewsAnalyzer:
    """
    Analyze real-time news for US market signal enhancement.

    FIX 61: Real-Time US News Analysis

    The problem: Model doesn't incorporate real-time news flow which
    drives significant short-term price movements.

    Solution: Analyze news for trading signals:
    1. Breaking news detection
    2. Sentiment scoring
    3. Event categorization
    4. Impact estimation
    5. Decay modeling

    Expected Alpha: +25-40% through news-driven signals
    """

    # News categories and their typical impacts
    NEWS_CATEGORIES = {
        'earnings_beat': {'base_impact': 1.40, 'decay_hours': 24},
        'earnings_miss': {'base_impact': 0.60, 'decay_hours': 24},
        'guidance_raise': {'base_impact': 1.30, 'decay_hours': 48},
        'guidance_cut': {'base_impact': 0.65, 'decay_hours': 48},
        'analyst_upgrade': {'base_impact': 1.20, 'decay_hours': 72},
        'analyst_downgrade': {'base_impact': 0.80, 'decay_hours': 72},
        'ceo_change': {'base_impact': 0.90, 'decay_hours': 168},
        'acquisition_target': {'base_impact': 1.50, 'decay_hours': 168},
        'acquisition_buyer': {'base_impact': 0.95, 'decay_hours': 168},
        'product_launch': {'base_impact': 1.15, 'decay_hours': 48},
        'product_failure': {'base_impact': 0.75, 'decay_hours': 48},
        'lawsuit_filed': {'base_impact': 0.85, 'decay_hours': 72},
        'lawsuit_settled': {'base_impact': 1.10, 'decay_hours': 24},
        'fda_approval': {'base_impact': 1.60, 'decay_hours': 48},
        'fda_rejection': {'base_impact': 0.50, 'decay_hours': 48},
        'contract_win': {'base_impact': 1.25, 'decay_hours': 48},
        'contract_loss': {'base_impact': 0.80, 'decay_hours': 48},
        'insider_buying': {'base_impact': 1.15, 'decay_hours': 168},
        'insider_selling': {'base_impact': 0.90, 'decay_hours': 168},
        'stock_buyback': {'base_impact': 1.10, 'decay_hours': 168},
        'dividend_increase': {'base_impact': 1.10, 'decay_hours': 72},
        'dividend_cut': {'base_impact': 0.75, 'decay_hours': 72},
        'regulatory_approval': {'base_impact': 1.30, 'decay_hours': 48},
        'regulatory_issue': {'base_impact': 0.70, 'decay_hours': 72},
        'partnership_announced': {'base_impact': 1.20, 'decay_hours': 48},
        'layoffs_announced': {'base_impact': 0.90, 'decay_hours': 48},
        'cybersecurity_breach': {'base_impact': 0.75, 'decay_hours': 72},
    }

    # Sentiment thresholds
    SENTIMENT_THRESHOLDS = {
        'very_positive': 0.80,
        'positive': 0.50,
        'slightly_positive': 0.20,
        'neutral': -0.20,
        'slightly_negative': -0.50,
        'negative': -0.80,
        'very_negative': -1.0,
    }

    SENTIMENT_MULTIPLIERS = {
        'very_positive': 1.50,
        'positive': 1.30,
        'slightly_positive': 1.15,
        'neutral': 1.00,
        'slightly_negative': 0.85,
        'negative': 0.70,
        'very_negative': 0.50,
    }

    # News source credibility
    SOURCE_CREDIBILITY = {
        'sec_filing': 1.50,
        'company_pr': 1.30,
        'major_wire': 1.20,     # Reuters, Bloomberg, AP
        'financial_news': 1.10,  # CNBC, WSJ, FT
        'analyst_report': 1.15,
        'social_media': 0.80,
        'blog': 0.70,
        'unknown': 0.90,
    }

    def __init__(self):
        self.news_cache = {}
        self.sentiment_history = {}
        self.active_events = {}

    def categorize_news(
        self,
        headline: str,
        content: str = None,
        source: str = 'unknown',
    ) -> Dict[str, Any]:
        """
        Categorize news event.

        Args:
            headline: News headline
            content: Full news content (optional)
            source: News source type

        Returns:
            Dict with categorized news
        """
        text = (headline + ' ' + (content or '')).lower()

        # Keyword-based categorization
        categories_detected = []

        if any(w in text for w in ['beats', 'exceeded', 'tops estimates', 'earnings beat']):
            categories_detected.append('earnings_beat')
        if any(w in text for w in ['misses', 'fell short', 'earnings miss', 'disappoints']):
            categories_detected.append('earnings_miss')
        if any(w in text for w in ['raises guidance', 'outlook raised', 'increases forecast']):
            categories_detected.append('guidance_raise')
        if any(w in text for w in ['cuts guidance', 'lowers outlook', 'reduces forecast']):
            categories_detected.append('guidance_cut')
        if any(w in text for w in ['upgrade', 'raised to buy', 'raised price target']):
            categories_detected.append('analyst_upgrade')
        if any(w in text for w in ['downgrade', 'cut to sell', 'lowered price target']):
            categories_detected.append('analyst_downgrade')
        if any(w in text for w in ['acquire', 'acquisition', 'buyout', 'merger']):
            if any(w in text for w in ['will acquire', 'to buy', 'acquiring']):
                categories_detected.append('acquisition_buyer')
            else:
                categories_detected.append('acquisition_target')
        if any(w in text for w in ['fda approv', 'regulatory approval']):
            categories_detected.append('fda_approval')
        if any(w in text for w in ['fda reject', 'regulatory rejection', 'not approved']):
            categories_detected.append('fda_rejection')
        if any(w in text for w in ['contract win', 'awarded contract', 'wins deal']):
            categories_detected.append('contract_win')
        if any(w in text for w in ['layoff', 'workforce reduction', 'job cuts']):
            categories_detected.append('layoffs_announced')
        if any(w in text for w in ['buyback', 'repurchase']):
            categories_detected.append('stock_buyback')
        if any(w in text for w in ['dividend increase', 'raises dividend']):
            categories_detected.append('dividend_increase')
        if any(w in text for w in ['dividend cut', 'suspends dividend']):
            categories_detected.append('dividend_cut')

        # Use primary category if multiple detected
        primary_category = categories_detected[0] if categories_detected else 'general'

        return {
            'headline': headline,
            'categories': categories_detected,
            'primary_category': primary_category,
            'source': source,
            'source_credibility': self.SOURCE_CREDIBILITY.get(source, 0.90),
        }

    def score_sentiment(
        self,
        text: str,
        category: str = None,
    ) -> Dict[str, Any]:
        """
        Score news sentiment.

        Args:
            text: News text
            category: Pre-determined category

        Returns:
            Dict with sentiment score
        """
        # Simple keyword-based sentiment
        positive_words = [
            'beat', 'exceed', 'strong', 'growth', 'profit', 'gain',
            'bullish', 'upgrade', 'approve', 'win', 'success', 'record',
            'innovation', 'breakthrough', 'optimistic', 'outperform',
        ]

        negative_words = [
            'miss', 'decline', 'loss', 'weak', 'cut', 'bearish',
            'downgrade', 'reject', 'fail', 'lawsuit', 'investigation',
            'layoff', 'concern', 'disappointing', 'underperform', 'warning',
        ]

        text_lower = text.lower()

        positive_count = sum(1 for w in positive_words if w in text_lower)
        negative_count = sum(1 for w in negative_words if w in text_lower)

        total = positive_count + negative_count
        if total > 0:
            sentiment = (positive_count - negative_count) / total
        else:
            sentiment = 0.0

        # Determine sentiment state
        if sentiment >= 0.80:
            state = 'very_positive'
        elif sentiment >= 0.50:
            state = 'positive'
        elif sentiment >= 0.20:
            state = 'slightly_positive'
        elif sentiment >= -0.20:
            state = 'neutral'
        elif sentiment >= -0.50:
            state = 'slightly_negative'
        elif sentiment >= -0.80:
            state = 'negative'
        else:
            state = 'very_negative'

        multiplier = self.SENTIMENT_MULTIPLIERS[state]

        return {
            'sentiment_score': sentiment,
            'sentiment_state': state,
            'multiplier': multiplier,
            'positive_signals': positive_count,
            'negative_signals': negative_count,
        }

    def calculate_impact(
        self,
        category: str,
        sentiment_score: float,
        hours_since_news: float,
        source_credibility: float,
    ) -> Dict[str, Any]:
        """
        Calculate news impact with decay.

        Args:
            category: News category
            sentiment_score: Sentiment score
            hours_since_news: Hours since news was published
            source_credibility: Source credibility multiplier

        Returns:
            Dict with impact calculation
        """
        if category not in self.NEWS_CATEGORIES:
            return {'impact_multiplier': 1.0, 'is_active': False}

        category_info = self.NEWS_CATEGORIES[category]
        base_impact = category_info['base_impact']
        decay_hours = category_info['decay_hours']

        # Apply time decay
        if hours_since_news <= 0:
            decay_factor = 1.0
        else:
            # Exponential decay
            decay_factor = np.exp(-hours_since_news / decay_hours)

        # Combine factors
        raw_impact = base_impact * source_credibility

        # Adjust based on sentiment alignment
        if base_impact > 1.0 and sentiment_score > 0:
            sentiment_boost = 1.0 + sentiment_score * 0.20
        elif base_impact < 1.0 and sentiment_score < 0:
            sentiment_boost = 1.0 + abs(sentiment_score) * 0.20
        else:
            sentiment_boost = 1.0

        adjusted_impact = raw_impact * sentiment_boost

        # Apply decay to move impact toward neutral
        if adjusted_impact > 1.0:
            decayed_impact = 1.0 + (adjusted_impact - 1.0) * decay_factor
        else:
            decayed_impact = 1.0 - (1.0 - adjusted_impact) * decay_factor

        is_active = decay_factor > 0.1  # Consider news "active" if >10% impact remaining

        return {
            'category': category,
            'base_impact': base_impact,
            'source_credibility': source_credibility,
            'sentiment_boost': sentiment_boost,
            'raw_impact': raw_impact,
            'adjusted_impact': adjusted_impact,
            'decay_factor': decay_factor,
            'hours_since_news': hours_since_news,
            'decay_hours': decay_hours,
            'impact_multiplier': decayed_impact,
            'is_active': is_active,
        }

    def get_news_signal(
        self,
        ticker: str,
        signal_type: str,
        news_events: List[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive news-driven signal.

        Args:
            ticker: Stock ticker
            signal_type: 'BUY' or 'SELL'
            news_events: List of news event dicts with category, sentiment, hours, source

        Returns:
            Dict with news signal
        """
        is_buy = signal_type.upper() == 'BUY'

        if not news_events:
            return {'has_news': False, 'multiplier': 1.0}

        active_impacts = []
        news_signals = []

        for event in news_events:
            category = event.get('category', 'general')
            sentiment = event.get('sentiment', 0)
            hours = event.get('hours_since', 0)
            source = event.get('source', 'unknown')
            source_cred = self.SOURCE_CREDIBILITY.get(source, 0.90)

            impact = self.calculate_impact(category, sentiment, hours, source_cred)

            if impact['is_active']:
                impact_mult = impact['impact_multiplier']

                # Check if impact aligns with signal direction
                if is_buy:
                    if impact_mult > 1.0:
                        active_impacts.append(impact_mult)
                        news_signals.append(category)
                else:
                    if impact_mult < 1.0:
                        active_impacts.append(2.0 - impact_mult)
                        news_signals.append(category)

        # Combine impacts
        if active_impacts:
            # Use geometric mean for multiple news events
            combined = np.prod(active_impacts) ** (1 / len(active_impacts))
        else:
            combined = 1.0

        # Cap extreme values
        combined = min(max(combined, 0.30), 2.50)

        return {
            'ticker': ticker,
            'signal_type': signal_type,
            'has_news': len(active_impacts) > 0,
            'active_news_count': len(active_impacts),
            'news_categories': news_signals,
            'individual_impacts': active_impacts,
            'combined_multiplier': combined,
        }


class USIntlModelOptimizer:
    """
    Comprehensive optimizer for US/International model signals.
    Implements all 15 fixes from the fixing documents.
    """

    # ========== FIX 1 & 2: SELL Confidence Thresholds ==========
    SELL_CONFIDENCE_THRESHOLDS = {
        AssetClass.COMMODITY: 0.80,
        AssetClass.CRYPTOCURRENCY: 0.78,
        AssetClass.STOCK: 0.75,
        AssetClass.FOREX: 0.72,
        AssetClass.ETF: 0.75,
        AssetClass.JPY_PAIR: 0.85,      # Higher threshold for JPY pairs
        AssetClass.CRUDE_OIL: 0.85,     # Higher threshold for crude oil
    }

    # ========== FIX 3 & 7: SELL Position Multipliers ==========
    SELL_POSITION_MULTIPLIERS = {
        AssetClass.COMMODITY: 0.30,      # 70% reduction (Fix 7)
        AssetClass.CRYPTOCURRENCY: 0.50,  # 50% reduction
        AssetClass.STOCK: 0.50,           # 50% reduction
        AssetClass.FOREX: 0.60,           # 40% reduction
        AssetClass.ETF: 0.50,
        AssetClass.JPY_PAIR: 0.20,       # 80% reduction for JPY pairs (Fix 17)
        AssetClass.CRUDE_OIL: 0.20,      # 80% reduction for crude oil (Fix 18)
    }

    # ========== FIX 4: Asset-Specific Stop-Losses ==========
    STOP_LOSS_BY_ASSET = {
        AssetClass.COMMODITY: 0.10,       # 10%
        AssetClass.CRYPTOCURRENCY: 0.12,  # 12%
        AssetClass.STOCK: 0.08,           # 8%
        AssetClass.FOREX: 0.06,           # 6%
        AssetClass.ETF: 0.08,
        AssetClass.JPY_PAIR: 0.04,        # Tighter stop for JPY pairs
        AssetClass.CRUDE_OIL: 0.06,       # Tighter stop for crude oil
    }

    # ========== FIX 5: BUY Position Boosts ==========
    BUY_POSITION_BOOSTS = {
        AssetClass.STOCK: 1.30,           # 30% boost for US stocks
        AssetClass.COMMODITY: 0.80,       # 20% reduction
        AssetClass.CRYPTOCURRENCY: 0.90,  # 10% reduction
        AssetClass.FOREX: 1.10,           # 10% boost
        AssetClass.ETF: 1.20,             # 20% boost
        AssetClass.JPY_PAIR: 0.70,        # 30% reduction for JPY pairs
        AssetClass.CRUDE_OIL: 0.60,       # 40% reduction for crude oil
    }

    # ========== FIX 6 & 8: Blocklists ==========
    SELL_BLACKLIST = {
        'NG=F',      # Fix 6: Natural Gas - too volatile for shorts
    }

    EXTENDED_BLOCKLIST = {
        'LUMN',      # Fix 8: Poor performer
        'USDT-USD',  # Fix 8: Stablecoin - no point shorting
        'ZC=F',      # Fix 8: Corn futures - choppy
        'ZW=F',      # Fix 8: Wheat futures - choppy
        # Note: 3800.HK removed - China stocks now routed to China model (Fix 16)
    }

    # Commodity/Crypto SELL blacklist (Fix 13)
    COMMODITY_SELL_BLACKLIST = {
        'NG=F', 'ZC=F', 'ZW=F', 'HG=F', 'LE=F', 'HE=F'
    }

    CRYPTO_SELL_BLACKLIST = {
        'USDT-USD', 'USDC-USD', 'DAI-USD', 'BUSD-USD'  # Stablecoins
    }

    # ========== FIX 9: Win Rate Position Multipliers ==========
    WIN_RATE_MULTIPLIERS = {
        (0.80, 1.00): 2.0,   # 80%+ win rate = 2x position
        (0.70, 0.80): 1.5,   # 70-80% = 1.5x
        (0.60, 0.70): 1.2,   # 60-70% = 1.2x
        (0.50, 0.60): 1.0,   # 50-60% = 1.0x (baseline)
        (0.40, 0.50): 0.7,   # 40-50% = 0.7x
        (0.30, 0.40): 0.5,   # 30-40% = 0.5x
        (0.20, 0.30): 0.3,   # 20-30% = 0.3x
        (0.00, 0.20): 0.1,   # <20% = 0.1x (near-skip)
    }

    # ========== FIX 10: Base Allocations ==========
    BASE_ALLOCATIONS = {
        AssetClass.STOCK: 0.10,           # 10% for US stocks
        AssetClass.ETF: 0.08,             # 8% for ETFs
        AssetClass.CRYPTOCURRENCY: 0.05,  # 5% for crypto
        AssetClass.COMMODITY: 0.05,       # 5% for commodities
        AssetClass.FOREX: 0.04,           # 4% for forex
        AssetClass.JPY_PAIR: 0.02,        # 2% for JPY pairs (reduced due to low win rate)
        AssetClass.CRUDE_OIL: 0.02,       # 2% for crude oil (reduced due to losses)
    }

    # ========== FIX 12: BUY Signal Rules ==========
    BUY_MIN_CONFIDENCE = 0.60
    BUY_POSITION_MULTIPLIER = 1.30
    BUY_STOP_LOSS = 0.08

    # ========== FIX 13: SELL Signal Rules ==========
    # NOTE: SELL_MIN_CONFIDENCE is now DYNAMIC - see get_dynamic_sell_threshold()
    # Base value kept for reference, actual threshold adjusted by trend/volatility
    SELL_MIN_CONFIDENCE_BASE = 0.80  # Base threshold (reduced in downtrends)
    SELL_MIN_CONFIDENCE_FLOOR = 0.55  # Minimum threshold (strong downtrend)
    SELL_MIN_CONFIDENCE_CEILING = 0.85  # Maximum threshold (strong uptrend)
    SELL_POSITION_MULTIPLIER = 0.40
    SELL_STOP_LOSS = 0.04

    # Dynamic threshold adjustment factors
    SELL_TREND_ADJUSTMENT = {
        'strong_downtrend': 0.70,   # -30% from base (e.g., 0.80 -> 0.56)
        'downtrend': 0.85,          # -15% from base (e.g., 0.80 -> 0.68)
        'neutral': 1.00,            # No adjustment
        'uptrend': 1.05,            # +5% from base (e.g., 0.80 -> 0.84)
        'strong_uptrend': 1.10,     # +10% from base (e.g., 0.80 -> 0.88, capped at ceiling)
    }

    # ========== FIX 21: Adaptive Ensemble Weights (US/INTL MODEL ONLY) ==========
    # CatBoost vs LSTM weights adjusted by market regime
    # - LSTM excels at capturing trends (sequential patterns)
    # - CatBoost excels at feature relationships (mean reversion, fundamental factors)
    ADAPTIVE_ENSEMBLE_WEIGHTS = {
        'strong_downtrend': {'catboost': 0.50, 'lstm': 0.50},  # Balanced - rapid change
        'downtrend': {'catboost': 0.45, 'lstm': 0.55},         # LSTM for trend continuation
        'neutral': {'catboost': 0.70, 'lstm': 0.30},           # Default - feature-based
        'uptrend': {'catboost': 0.45, 'lstm': 0.55},           # LSTM for trend continuation
        'strong_uptrend': {'catboost': 0.40, 'lstm': 0.60},    # LSTM dominates in strong trends
        'mean_reverting': {'catboost': 0.80, 'lstm': 0.20},    # CatBoost for mean reversion
        'high_volatility': {'catboost': 0.60, 'lstm': 0.40},   # More feature-based in vol
    }
    DEFAULT_ENSEMBLE_WEIGHTS = {'catboost': 0.70, 'lstm': 0.30}

    # ========== FIX 15: Profit-Taking Levels ==========
    PROFIT_TAKING_LEVELS = {
        0.15: 0.50,  # At 15% profit, close 50%
        0.25: 0.75,  # At 25% profit, close 75%
        0.40: 1.00,  # At 40% profit, close 100%
    }

    def __init__(
        self,
        historical_win_rates: Optional[Dict[str, float]] = None,
        high_profit_patterns: Optional[Dict[str, Dict]] = None,
        enable_kelly: bool = True,
        enable_dynamic_sizing: bool = True,
        enable_regime_adjustment: bool = True,  # Fix 19
        enable_adaptive_blocking: bool = True,  # Fixes 17 & 18
        enable_adaptive_kelly: bool = True,  # Fix 24
        enable_position_concentration: bool = True,  # Fix 25
        enable_dynamic_profit_targets: bool = True,  # Fix 26
        account_size: float = 50000,  # Default account size for Fix 24
        # Fixes 27-33: US-specific optimizations
        enable_us_regime_classifier: bool = True,  # Fix 27
        enable_sector_momentum: bool = True,  # Fix 28
        enable_earnings_optimizer: bool = True,  # Fix 29
        enable_fomc_optimizer: bool = True,  # Fix 30
        enable_opex_optimizer: bool = True,  # Fix 31
        enable_market_internals: bool = True,  # Fix 32
        enable_us_risk_model: bool = True,  # Fix 33
        # Fixes 34-41: Advanced profit-maximizing strategies
        enable_intraday_timing: bool = True,  # Fix 34
        enable_market_cap_tiers: bool = True,  # Fix 35
        enable_quarter_end_optimizer: bool = True,  # Fix 36
        enable_earnings_gap_trading: bool = True,  # Fix 37
        enable_sector_rotation: bool = True,  # Fix 38
        enable_vix_term_structure: bool = True,  # Fix 39
        enable_economic_data_reactions: bool = True,  # Fix 40
        enable_put_call_ratio: bool = True,  # Fix 41
        # Fixes 42-49: Advanced profit-maximizing strategies II
        enable_unified_optimizer: bool = True,  # Fix 42
        enable_enhanced_sector_rotation: bool = True,  # Fix 43
        enable_catalyst_detector: bool = True,  # Fix 44
        enable_enhanced_intraday: bool = True,  # Fix 45
        enable_momentum_acceleration: bool = True,  # Fix 46
        enable_us_profit_rules: bool = True,  # Fix 47
        enable_smart_profit_taker: bool = True,  # Fix 48
        enable_backtest_maximizer: bool = True,  # Fix 49
        # Fixes 50-53: Advanced profit-maximizing strategies III
        enable_market_structure_arbitrage: bool = True,  # Fix 50
        enable_smart_beta_overlay: bool = True,  # Fix 51
        enable_volatility_regime_switching: bool = True,  # Fix 52
        enable_institutional_flow_mirroring: bool = True,  # Fix 53
        # Fixes 54-61: US-Specific Aggressive Alpha Strategies
        enable_mega_cap_momentum: bool = True,  # Fix 54
        enable_semiconductor_cycle: bool = True,  # Fix 55
        enable_ai_thematic: bool = True,  # Fix 56
        enable_fed_liquidity_regime: bool = True,  # Fix 57
        enable_retail_options_flow: bool = True,  # Fix 58
        enable_meme_stock_patterns: bool = True,  # Fix 59
        enable_earnings_sector_rotation: bool = True,  # Fix 60
        enable_realtime_news: bool = True,  # Fix 61
    ):
        """
        Initialize the optimizer.

        Args:
            historical_win_rates: Dict mapping ticker -> win_rate (0-1)
            high_profit_patterns: Dict of detected high-profit patterns
            enable_kelly: Whether to use Kelly Criterion sizing
            enable_dynamic_sizing: Whether to use dynamic win-rate sizing
            enable_regime_adjustment: Whether to use market regime position adjustment (Fix 19)
            enable_adaptive_blocking: Whether to use adaptive blocking for JPY/Crude (Fixes 17 & 18)
            enable_adaptive_kelly: Whether to use adaptive Kelly (Fix 24)
            enable_position_concentration: Whether to use position concentration (Fix 25)
            enable_dynamic_profit_targets: Whether to use dynamic profit targets (Fix 26)
            account_size: Account size in dollars for adaptive Kelly calculation
            enable_us_regime_classifier: Whether to use US regime classifier (Fix 27)
            enable_sector_momentum: Whether to use sector momentum analysis (Fix 28)
            enable_earnings_optimizer: Whether to use earnings season optimizer (Fix 29)
            enable_fomc_optimizer: Whether to use FOMC optimizer (Fix 30)
            enable_opex_optimizer: Whether to use options expiration optimizer (Fix 31)
            enable_market_internals: Whether to use market internals (Fix 32)
            enable_us_risk_model: Whether to use US-specific risk model (Fix 33)
            enable_intraday_timing: Whether to use intraday momentum timing (Fix 34)
            enable_market_cap_tiers: Whether to use market cap tier optimizer (Fix 35)
            enable_quarter_end_optimizer: Whether to use quarter-end window dressing (Fix 36)
            enable_earnings_gap_trading: Whether to use earnings gap trading (Fix 37)
            enable_sector_rotation: Whether to use sector rotation momentum (Fix 38)
            enable_vix_term_structure: Whether to use VIX term structure arbitrage (Fix 39)
            enable_economic_data_reactions: Whether to use economic data reactions (Fix 40)
            enable_put_call_ratio: Whether to use put/call ratio reversals (Fix 41)
            enable_unified_optimizer: Whether to use unified profit maximizer (Fix 42)
            enable_enhanced_sector_rotation: Whether to use enhanced sector rotation (Fix 43)
            enable_catalyst_detector: Whether to use catalyst detector (Fix 44)
            enable_enhanced_intraday: Whether to use enhanced intraday with volume profile (Fix 45)
            enable_momentum_acceleration: Whether to use momentum acceleration detector (Fix 46)
            enable_us_profit_rules: Whether to use US-specific profit rules (Fix 47)
            enable_smart_profit_taker: Whether to use smart profit taker (Fix 48)
            enable_backtest_maximizer: Whether to use backtest profit maximizer (Fix 49)
            enable_market_structure_arbitrage: Whether to use market structure arbitrage (Fix 50)
            enable_smart_beta_overlay: Whether to use smart beta factor overlay (Fix 51)
            enable_volatility_regime_switching: Whether to use VIX regime switching (Fix 52)
            enable_institutional_flow_mirroring: Whether to use institutional flow mirroring (Fix 53)
            enable_mega_cap_momentum: Whether to use mega-cap tech momentum (Fix 54)
            enable_semiconductor_cycle: Whether to use semiconductor cycle detector (Fix 55)
            enable_ai_thematic: Whether to use AI thematic concentration (Fix 56)
            enable_fed_liquidity_regime: Whether to use Fed liquidity regime (Fix 57)
            enable_retail_options_flow: Whether to use retail options flow (Fix 58)
            enable_meme_stock_patterns: Whether to use meme stock patterns (Fix 59)
            enable_earnings_sector_rotation: Whether to use earnings sector rotation (Fix 60)
            enable_realtime_news: Whether to use real-time news analysis (Fix 61)
        """
        self.historical_win_rates = historical_win_rates or {}
        self.high_profit_patterns = high_profit_patterns or {}
        self.enable_kelly = enable_kelly
        self.enable_dynamic_sizing = enable_dynamic_sizing
        self.enable_regime_adjustment = enable_regime_adjustment
        self.enable_adaptive_blocking = enable_adaptive_blocking
        self.enable_adaptive_kelly = enable_adaptive_kelly
        self.enable_position_concentration = enable_position_concentration
        self.enable_dynamic_profit_targets = enable_dynamic_profit_targets
        self.account_size = account_size

        # Fixes 27-33 enable flags
        self.enable_us_regime_classifier = enable_us_regime_classifier
        self.enable_sector_momentum = enable_sector_momentum
        self.enable_earnings_optimizer = enable_earnings_optimizer
        self.enable_fomc_optimizer = enable_fomc_optimizer
        self.enable_opex_optimizer = enable_opex_optimizer
        self.enable_market_internals = enable_market_internals
        self.enable_us_risk_model = enable_us_risk_model

        # Initialize risk management components (Fixes 16-19)
        self.regime_detector = None
        self.adaptive_blocker = None
        self.position_adjuster = None
        self._current_regime_detection = None

        if RISK_MANAGEMENT_AVAILABLE:
            self.regime_detector = MarketRegimeDetector()
            self.adaptive_blocker = AdaptiveBlocker(self.regime_detector)
            self.position_adjuster = PositionAdjuster(
                regime_detector=self.regime_detector,
                historical_win_rates=self.historical_win_rates,
            )

        # Initialize profit maximization components (Fixes 24-26)
        self.adaptive_kelly_optimizer = AdaptiveKellyOptimizer() if enable_adaptive_kelly else None
        self.position_concentrator = PositionConcentrationOptimizer() if enable_position_concentration else None
        self.dynamic_profit_targets = DynamicProfitTargets() if enable_dynamic_profit_targets else None

        # Initialize US-specific optimization components (Fixes 27-33)
        self.us_regime_classifier = USMarketRegimeClassifier() if enable_us_regime_classifier else None
        self.sector_momentum_analyzer = SectorMomentumAnalyzer() if enable_sector_momentum else None
        self.earnings_optimizer = EarningsSeasonOptimizer() if enable_earnings_optimizer else None
        self.fomc_optimizer = FOMCOptimizer() if enable_fomc_optimizer else None
        self.opex_optimizer = OpExOptimizer() if enable_opex_optimizer else None
        self.market_internals = USMarketInternals() if enable_market_internals else None
        self.us_risk_model = USRiskModel() if enable_us_risk_model else None

        # Initialize advanced profit-maximizing components (Fixes 34-41)
        self.enable_intraday_timing = enable_intraday_timing
        self.enable_market_cap_tiers = enable_market_cap_tiers
        self.enable_quarter_end_optimizer = enable_quarter_end_optimizer
        self.enable_earnings_gap_trading = enable_earnings_gap_trading
        self.enable_sector_rotation = enable_sector_rotation
        self.enable_vix_term_structure = enable_vix_term_structure
        self.enable_economic_data_reactions = enable_economic_data_reactions
        self.enable_put_call_ratio = enable_put_call_ratio

        self.intraday_optimizer = IntradayMomentumOptimizer() if enable_intraday_timing else None
        self.market_cap_optimizer = MarketCapTierOptimizer() if enable_market_cap_tiers else None
        self.quarter_end_optimizer = QuarterEndOptimizer() if enable_quarter_end_optimizer else None
        self.earnings_gap_trader = EarningsGapTrader() if enable_earnings_gap_trading else None
        self.sector_rotation_momentum = SectorRotationMomentum() if enable_sector_rotation else None
        self.vix_term_structure = VIXTermStructureAnalyzer() if enable_vix_term_structure else None
        self.economic_data_reactor = EconomicDataReactor() if enable_economic_data_reactions else None
        self.put_call_analyzer = PutCallRatioAnalyzer() if enable_put_call_ratio else None

        # Initialize advanced profit-maximizing components II (Fixes 42-49)
        self.enable_unified_optimizer = enable_unified_optimizer
        self.enable_enhanced_sector_rotation = enable_enhanced_sector_rotation
        self.enable_catalyst_detector = enable_catalyst_detector
        self.enable_enhanced_intraday = enable_enhanced_intraday
        self.enable_momentum_acceleration = enable_momentum_acceleration
        self.enable_us_profit_rules = enable_us_profit_rules
        self.enable_smart_profit_taker = enable_smart_profit_taker
        self.enable_backtest_maximizer = enable_backtest_maximizer

        # Fix 43: Enhanced sector rotation with leading indicators
        self.enhanced_sector_rotation = EnhancedSectorRotationDetector() if enable_enhanced_sector_rotation else None

        # Fix 44: Catalyst detector for news-based events
        self.catalyst_detector = USCatalystDetector() if enable_catalyst_detector else None

        # Fix 45: Enhanced intraday with volume profile
        self.enhanced_intraday = EnhancedIntradayOptimizer() if enable_enhanced_intraday else None

        # Fix 46: Momentum acceleration detector
        self.momentum_acceleration = MomentumAccelerationDetector() if enable_momentum_acceleration else None

        # Fix 47: US-specific profit rules by stock type
        self.us_profit_rules = USProfitRules() if enable_us_profit_rules else None

        # Fix 48: Smart profit taker with multi-factor scoring
        self.smart_profit_taker = SmartProfitTaker() if enable_smart_profit_taker else None

        # Fix 49: Backtest profit maximizer (theoretical maximum)
        self.backtest_maximizer = BacktestProfitMaximizer() if enable_backtest_maximizer else None

        # Fixes 50-53: Advanced profit-maximizing strategies III
        self.enable_market_structure_arbitrage = enable_market_structure_arbitrage
        self.enable_smart_beta_overlay = enable_smart_beta_overlay
        self.enable_volatility_regime_switching = enable_volatility_regime_switching
        self.enable_institutional_flow_mirroring = enable_institutional_flow_mirroring

        # Fix 50: Market Structure Arbitrage (ETF premium/discount, MOC imbalance, gamma, dark pools)
        self.market_structure_arbitrage = MarketStructureArbitrage() if enable_market_structure_arbitrage else None

        # Fix 51: Smart Beta Overlay (factor timing based on regime)
        self.smart_beta_overlay = SmartBetaOverlay() if enable_smart_beta_overlay else None

        # Fix 52: Volatility Regime Switching (VIX-based strategy adaptation)
        self.volatility_regime_switcher = VolatilityRegimeSwitcher() if enable_volatility_regime_switching else None

        # Fix 53: Institutional Flow Mirroring (13F, ETF flows, block trades, insider)
        self.institutional_flow_mirror = InstitutionalFlowMirror() if enable_institutional_flow_mirroring else None

        # Fixes 54-61: US-Specific Aggressive Alpha Strategies
        self.enable_mega_cap_momentum = enable_mega_cap_momentum
        self.enable_semiconductor_cycle = enable_semiconductor_cycle
        self.enable_ai_thematic = enable_ai_thematic
        self.enable_fed_liquidity_regime = enable_fed_liquidity_regime
        self.enable_retail_options_flow = enable_retail_options_flow
        self.enable_meme_stock_patterns = enable_meme_stock_patterns
        self.enable_earnings_sector_rotation = enable_earnings_sector_rotation
        self.enable_realtime_news = enable_realtime_news

        # Fix 54: Mega-Cap Tech Momentum Exploitation (FAANG+M momentum, relative strength)
        self.mega_cap_momentum = MegaCapMomentumExploiter() if enable_mega_cap_momentum else None

        # Fix 55: Semiconductor Super-Cycle Detection (book-to-bill, inventory, capex)
        self.semiconductor_cycle = SemiconductorCycleDetector() if enable_semiconductor_cycle else None

        # Fix 56: AI Thematic Concentration (AI pure-plays, sentiment, capex cycle)
        self.ai_thematic = AIThematicConcentrator() if enable_ai_thematic else None

        # Fix 57: Fed Liquidity Regime Optimization (balance sheet, RRP, rate cycle)
        self.fed_liquidity_regime = FedLiquidityRegimeOptimizer() if enable_fed_liquidity_regime else None

        # Fix 58: Retail Options Flow Analysis (call/put ratio, gamma, OPEX)
        self.retail_options_flow = RetailOptionsFlowAnalyzer() if enable_retail_options_flow else None

        # Fix 59: Meme Stock Pattern Detection (social momentum, squeeze potential)
        self.meme_stock_detector = MemeStockPatternDetector() if enable_meme_stock_patterns else None

        # Fix 60: Earnings-Driven Sector Rotation (beat rates, revisions, leadership)
        self.earnings_sector_rotation = EarningsDrivenSectorRotation() if enable_earnings_sector_rotation else None

        # Fix 61: Real-Time US News Analysis (categorization, sentiment, impact decay)
        self.realtime_news_analyzer = RealTimeUSNewsAnalyzer() if enable_realtime_news else None

        # Fix 42: Unified profit maximizer - combines ALL fix components
        # Must be initialized LAST as it takes other optimizers as dependencies
        if enable_unified_optimizer:
            self.unified_profit_maximizer = UnifiedUSProfitMaximizer(
                regime_classifier=self.us_regime_classifier,
                sector_momentum=self.sector_momentum_analyzer,
                earnings_optimizer=self.earnings_optimizer,
                fomc_optimizer=self.fomc_optimizer,
                opex_optimizer=self.opex_optimizer,
                market_internals=self.market_internals,
                us_risk_model=self.us_risk_model,
                intraday_optimizer=self.intraday_optimizer,
                market_cap_optimizer=self.market_cap_optimizer,
                quarter_end_optimizer=self.quarter_end_optimizer,
                earnings_gap_trader=self.earnings_gap_trader,
                sector_rotation=self.sector_rotation_momentum,
                vix_term_structure=self.vix_term_structure,
                economic_reactor=self.economic_data_reactor,
                put_call_analyzer=self.put_call_analyzer,
            )
        else:
            self.unified_profit_maximizer = None

    def classify_asset(self, ticker: str) -> AssetClass:
        """
        Classify a ticker into its asset class using pattern-based detection.

        This method automatically classifies assets based on ticker patterns
        without requiring hardcoded lists of specific tickers.

        Classification Rules:
        1. JPY Pairs: Forex pairs containing JPY (Fix 17)
        2. Crude Oil: CL=F ticker (Fix 18)
        3. Cryptocurrency: Ends with -USD, -USDT, -EUR, -GBP, -BTC, -ETH (crypto pairs)
        4. Other Commodity futures: Ends with =F (Yahoo Finance futures format)
        5. Other Forex: Ends with =X (Yahoo Finance forex format)
        6. ETFs: Known ETF patterns
        7. US/International stocks: Everything else (.L, .T, .DE, .PA, no suffix) -> STOCK

        IMPORTANT - Fix 16: China stocks (.HK, .SS, .SZ) should NEVER reach this model.
        They must be routed to China model via market_classifier.py.
        This function simply classifies - China stock check is done separately via is_china_stock().
        """
        ticker_upper = ticker.upper()

        # Fix 16: China stocks should be checked via is_china_stock() before calling this
        # If they somehow reach here, classify as STOCK but caller should filter them out
        # This ensures no ValueError that could affect other code paths

        # Fix 17: JPY Pairs Detection (before general forex)
        # JPY pairs have 0% win rate on SELL, need special handling
        jpy_patterns = ['JPY=X', 'USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'CADJPY', 'CHFJPY']
        if any(pattern in ticker_upper for pattern in jpy_patterns) or ticker_upper.endswith('JPY=X'):
            return AssetClass.JPY_PAIR

        # Fix 18: Crude Oil Detection (before general commodity)
        # CL=F is a consistent loser, needs special handling
        if ticker_upper == 'CL=F':
            return AssetClass.CRUDE_OIL

        # 1. Cryptocurrency Detection
        # Pattern: Any ticker ending with common crypto quote currencies
        crypto_quote_suffixes = ['-USD', '-USDT', '-EUR', '-GBP', '-BTC', '-ETH', '-BUSD']
        if any(ticker_upper.endswith(suffix) for suffix in crypto_quote_suffixes):
            return AssetClass.CRYPTOCURRENCY

        # 2. Commodity Futures Detection
        # Pattern: Ends with =F (Yahoo Finance futures format)
        if ticker_upper.endswith('=F'):
            return AssetClass.COMMODITY

        # 3. Forex Detection
        # Pattern: Ends with =X (Yahoo Finance forex format)
        if ticker_upper.endswith('=X'):
            return AssetClass.FOREX

        # 5. ETF Detection
        # Common ETF patterns:
        # - Well-known ETF tickers
        # - Tickers with common ETF naming patterns
        well_known_etfs = {
            # Major index ETFs
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'IVV', 'VTV', 'VUG',
            # Sector ETFs
            'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP', 'XLU', 'XLB', 'XLY', 'XLRE',
            # Commodity ETFs
            'GLD', 'SLV', 'USO', 'UNG', 'DBC', 'GSG',
            # Bond ETFs
            'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'BND', 'AGG',
            # International ETFs
            'EFA', 'EEM', 'VWO', 'IEFA', 'IEMG', 'VEA', 'VGK',
            # Leveraged/Inverse ETFs
            'TQQQ', 'SQQQ', 'SPXU', 'UPRO', 'TNA', 'TZA',
            # Thematic ETFs
            'ARKK', 'ARKG', 'ARKW', 'ARKF', 'ARKQ',
        }
        if ticker_upper in well_known_etfs:
            return AssetClass.ETF

        # Additional ETF pattern detection:
        # Many ETFs follow naming patterns like starting with V (Vanguard),
        # I (iShares), X (SPDR sectors), or ending with common suffixes
        etf_prefixes = ['VT', 'VO', 'VB', 'VG', 'IW', 'IJ', 'IY', 'EW']
        if any(ticker_upper.startswith(prefix) and len(ticker_upper) <= 5 for prefix in etf_prefixes):
            # Short tickers starting with ETF provider prefixes likely ETFs
            # But need to be careful not to misclassify stocks
            pass  # Let it fall through to stock for safety

        # 6. Default: US/International Stock
        # This includes: US stocks (AAPL), UK (.L), Japan (.T), Germany (.DE), etc.
        return AssetClass.STOCK

    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float = 0.05,
        avg_loss: float = 0.03,
    ) -> float:
        """
        Calculate Kelly Criterion fraction for optimal position sizing.

        Fix 11: Kelly Criterion Position Sizing
        Formula: f* = (p * b - q) / b
        Where:
            p = probability of winning
            q = probability of losing (1 - p)
            b = win/loss ratio (avg_win / avg_loss)

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning return
            avg_loss: Average losing return (absolute value)

        Returns:
            Kelly fraction (capped at 0.25 for safety)
        """
        if win_rate <= 0 or avg_loss <= 0:
            return 0.0

        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss if avg_loss > 0 else 1.0

        kelly = (p * b - q) / b if b > 0 else 0.0

        # Cap at 25% (quarter-Kelly is common in practice)
        kelly = max(0.0, min(kelly, 0.25))

        return kelly

    def get_win_rate_multiplier(self, win_rate: float) -> float:
        """
        Get position multiplier based on historical win rate.

        Fix 9: Dynamic Position Sizing by Win Rate
        """
        for (low, high), multiplier in self.WIN_RATE_MULTIPLIERS.items():
            if low <= win_rate < high:
                return multiplier
        return 1.0

    def classify_trend(self, momentum: float, volatility: float = 0.20) -> str:
        """
        Classify the current market trend based on momentum.

        Dynamic SELL Threshold Fix: Trend Classification

        Momentum thresholds are volatility-adjusted:
        - Strong downtrend: momentum < -2 * volatility
        - Downtrend: -2 * volatility <= momentum < -0.5 * volatility
        - Neutral: -0.5 * volatility <= momentum <= 0.5 * volatility
        - Uptrend: 0.5 * volatility < momentum <= 2 * volatility
        - Strong uptrend: momentum > 2 * volatility

        Args:
            momentum: Recent price momentum (e.g., 20-day return)
            volatility: Asset volatility (annualized, default 0.20)

        Returns:
            Trend classification string
        """
        # Volatility-adjusted thresholds (more sensitive in low-vol, less in high-vol)
        vol_factor = max(0.10, min(volatility, 0.40))  # Clamp between 10% and 40%

        strong_threshold = 2.0 * vol_factor
        weak_threshold = 0.5 * vol_factor

        if momentum < -strong_threshold:
            return 'strong_downtrend'
        elif momentum < -weak_threshold:
            return 'downtrend'
        elif momentum > strong_threshold:
            return 'strong_uptrend'
        elif momentum > weak_threshold:
            return 'uptrend'
        else:
            return 'neutral'

    def get_dynamic_sell_threshold(
        self,
        asset_class: AssetClass,
        momentum: float = 0.0,
        volatility: float = 0.20,
    ) -> Tuple[float, str]:
        """
        Calculate dynamic SELL confidence threshold based on market conditions.

        Dynamic SELL Threshold Fix (US/INTL MODEL ONLY):
        - In DOWNTRENDS: Lower threshold to allow more SELL signals
        - In UPTRENDS: Raise threshold to filter weak SELL signals
        - Volatility-adjusted trend classification

        This addresses the issue where static 80% threshold blocks almost ALL SELL signals.
        In downtrends, we WANT to execute SELL signals more aggressively.

        Args:
            asset_class: Asset classification
            momentum: Recent momentum (e.g., 20-day return)
            volatility: Asset volatility (annualized)

        Returns:
            (dynamic_threshold, trend_classification)

        Example adjustments:
            - Strong downtrend (mom=-0.15): 0.80 * 0.70 = 0.56 (55% floor)
            - Downtrend (mom=-0.05): 0.80 * 0.85 = 0.68
            - Neutral: 0.80 * 1.00 = 0.80
            - Uptrend (mom=+0.05): 0.80 * 1.05 = 0.84
            - Strong uptrend (mom=+0.15): 0.80 * 1.10 = 0.88 (capped at ceiling)
        """
        # Get base threshold for this asset class
        base_threshold = self.SELL_CONFIDENCE_THRESHOLDS.get(asset_class, 0.75)

        # Classify trend
        trend = self.classify_trend(momentum, volatility)

        # Get adjustment factor
        adjustment = self.SELL_TREND_ADJUSTMENT.get(trend, 1.0)

        # Calculate dynamic threshold
        dynamic_threshold = base_threshold * adjustment

        # Apply floor and ceiling
        dynamic_threshold = max(
            self.SELL_MIN_CONFIDENCE_FLOOR,
            min(dynamic_threshold, self.SELL_MIN_CONFIDENCE_CEILING)
        )

        return dynamic_threshold, trend

    def get_adaptive_ensemble_weights(
        self,
        momentum: float = 0.0,
        volatility: float = 0.20,
        volatility_percentile: float = 50.0,
    ) -> Tuple[Dict[str, float], str]:
        """
        Calculate adaptive ensemble weights for CatBoost/LSTM based on market conditions.

        FIX 21: Adaptive Ensemble Weights (US/INTL MODEL ONLY)

        Theory:
        - LSTM excels at sequential patterns (trend continuation, momentum)
        - CatBoost excels at feature relationships (mean reversion, fundamentals)

        In trending markets, increase LSTM weight.
        In mean-reverting/high-vol markets, increase CatBoost weight.

        Args:
            momentum: Recent price momentum (e.g., 20-day return)
            volatility: Asset volatility (annualized)
            volatility_percentile: Current volatility vs historical (0-100)

        Returns:
            (weights_dict, regime_classification)
            weights_dict: {'catboost': float, 'lstm': float}
        """
        # Classify trend using existing method
        trend = self.classify_trend(momentum, volatility)

        # Check for high volatility regime
        if volatility_percentile > 80:
            regime = 'high_volatility'
        # Check for mean reversion (low momentum, low volatility)
        elif abs(momentum) < 0.02 and volatility_percentile < 30:
            regime = 'mean_reverting'
        else:
            regime = trend

        # Get weights for this regime
        weights = self.ADAPTIVE_ENSEMBLE_WEIGHTS.get(regime, self.DEFAULT_ENSEMBLE_WEIGHTS).copy()

        return weights, regime

    def detect_high_profit_pattern(
        self,
        ticker: str,
        confidence: float,
        volatility: float,
        momentum: float,
    ) -> Tuple[bool, float]:
        """
        Detect high-profit trading patterns.

        Fix 14: High-Profit Pattern Detection

        Returns:
            (is_high_profit, boost_multiplier)
        """
        # Check pre-defined patterns
        if ticker in self.high_profit_patterns:
            pattern = self.high_profit_patterns[ticker]
            if pattern.get('is_high_profit'):
                return True, pattern.get('boost', 1.5)

        # Dynamic pattern detection
        # High confidence + moderate volatility + positive momentum = high profit potential
        if confidence > 0.80 and 0.15 < volatility < 0.40 and momentum > 0.02:
            return True, 1.3

        # Very high confidence alone
        if confidence > 0.90:
            return True, 1.2

        return False, 1.0

    def optimize_buy_signal(
        self,
        ticker: str,
        confidence: float,
        volatility: float = 0.20,
        momentum: float = 0.0,
        win_rate: Optional[float] = None,
    ) -> SignalOptimization:
        """
        Optimize a BUY signal using all applicable fixes.

        Applies:
            Fix 5: BUY position boosts
            Fix 8: Extended blocklist check
            Fix 9: Win rate sizing
            Fix 10: Base allocations
            Fix 11: Kelly criterion
            Fix 12: BUY signal rules
            Fix 14: High-profit pattern detection
            Fix 15: Profit-taking levels
        """
        fixes_applied = []
        asset_class = self.classify_asset(ticker)

        # Initialize result
        blocked = False
        block_reason = None
        position_mult = 1.0
        adjusted_confidence = confidence
        kelly_fraction = None

        # Fix 8: Check extended blocklist
        if ticker in self.EXTENDED_BLOCKLIST:
            blocked = True
            block_reason = f"BLOCKLISTED (Fix 8): {ticker} in extended blocklist"
            fixes_applied.append("Fix 8: Extended blocklist - BLOCKED")

        # Fix 12: BUY minimum confidence check
        if not blocked and confidence < self.BUY_MIN_CONFIDENCE:
            blocked = True
            block_reason = f"LOW_CONF (Fix 12): {confidence*100:.0f}% < {self.BUY_MIN_CONFIDENCE*100:.0f}%"
            fixes_applied.append("Fix 12: Min confidence - BLOCKED")

        if not blocked:
            # Fix 5: BUY position boost by asset class
            asset_boost = self.BUY_POSITION_BOOSTS.get(asset_class, 1.0)
            position_mult *= asset_boost
            fixes_applied.append(f"Fix 5: Asset boost {asset_boost:.2f}x")

            # Fix 10: Base allocation
            base_alloc = self.BASE_ALLOCATIONS.get(asset_class, 0.05)
            fixes_applied.append(f"Fix 10: Base allocation {base_alloc*100:.0f}%")

            # Fix 9: Win rate multiplier
            ticker_win_rate = win_rate or self.historical_win_rates.get(ticker, 0.50)
            if self.enable_dynamic_sizing:
                wr_mult = self.get_win_rate_multiplier(ticker_win_rate)
                position_mult *= wr_mult
                fixes_applied.append(f"Fix 9: Win rate {ticker_win_rate*100:.0f}% -> {wr_mult:.1f}x")

            # Fix 11: Kelly Criterion
            if self.enable_kelly and ticker_win_rate > 0:
                kelly_fraction = self.calculate_kelly_fraction(ticker_win_rate)
                fixes_applied.append(f"Fix 11: Kelly fraction {kelly_fraction*100:.1f}%")

            # Fix 12: BUY multiplier
            position_mult *= self.BUY_POSITION_MULTIPLIER
            fixes_applied.append(f"Fix 12: BUY multiplier {self.BUY_POSITION_MULTIPLIER:.2f}x")

            # Fix 14: High-profit pattern detection
            is_high_profit, pattern_boost = self.detect_high_profit_pattern(
                ticker, confidence, volatility, momentum
            )
            if is_high_profit:
                position_mult *= pattern_boost
                fixes_applied.append(f"Fix 14: High-profit pattern {pattern_boost:.2f}x")

        # Fix 4: Stop-loss
        stop_loss = self.STOP_LOSS_BY_ASSET.get(asset_class, 0.08)
        fixes_applied.append(f"Fix 4: Stop-loss {stop_loss*100:.0f}%")

        # Fix 15: Profit-taking levels
        take_profit_levels = self.PROFIT_TAKING_LEVELS.copy()
        fixes_applied.append("Fix 15: Profit-taking levels")

        return SignalOptimization(
            ticker=ticker,
            signal_type='BUY',
            original_confidence=confidence,
            adjusted_confidence=adjusted_confidence,
            position_multiplier=position_mult,
            stop_loss_pct=stop_loss,
            take_profit_levels=take_profit_levels,
            blocked=blocked,
            block_reason=block_reason,
            fixes_applied=fixes_applied,
            asset_class=asset_class.value,
            kelly_fraction=kelly_fraction,
        )

    def optimize_sell_signal(
        self,
        ticker: str,
        confidence: float,
        volatility: float = 0.20,
        momentum: float = 0.0,
        win_rate: Optional[float] = None,
    ) -> SignalOptimization:
        """
        Optimize a SELL (short) signal using all applicable fixes.

        Applies:
            Fix 1: Higher SELL confidence threshold
            Fix 2: Asset-class specific thresholds
            Fix 3: 50% SELL position reduction
            Fix 4: Asset-specific stop-losses
            Fix 6: NG=F blacklist
            Fix 7: 70% commodity reduction
            Fix 8: Extended blocklist
            Fix 9: Win rate sizing
            Fix 10: Base allocations
            Fix 13: SELL signal rules
            NEW: Dynamic SELL thresholds based on trend/volatility
        """
        fixes_applied = []
        asset_class = self.classify_asset(ticker)

        # Initialize result
        blocked = False
        block_reason = None
        position_mult = 1.0
        adjusted_confidence = confidence
        kelly_fraction = None
        trend_classification = 'neutral'

        # Fix 6: SELL blacklist check
        if ticker in self.SELL_BLACKLIST:
            blocked = True
            block_reason = f"BLACKLISTED (Fix 6): {ticker} in SELL blacklist"
            fixes_applied.append("Fix 6: SELL blacklist - BLOCKED")

        # Fix 13: Commodity/Crypto SELL blacklist
        if not blocked and asset_class == AssetClass.COMMODITY:
            if ticker in self.COMMODITY_SELL_BLACKLIST:
                blocked = True
                block_reason = f"COMMODITY_BLACKLIST (Fix 13): {ticker}"
                fixes_applied.append("Fix 13: Commodity SELL blacklist - BLOCKED")

        if not blocked and asset_class == AssetClass.CRYPTOCURRENCY:
            if ticker in self.CRYPTO_SELL_BLACKLIST:
                blocked = True
                block_reason = f"CRYPTO_BLACKLIST (Fix 13): {ticker} is stablecoin"
                fixes_applied.append("Fix 13: Crypto SELL blacklist - BLOCKED")

        # NEW: Dynamic SELL threshold based on trend/volatility
        # This replaces the static Fix 1/2 and Fix 13 confidence thresholds
        if not blocked:
            # Calculate dynamic threshold (lowered in downtrends, raised in uptrends)
            dynamic_threshold, trend_classification = self.get_dynamic_sell_threshold(
                asset_class=asset_class,
                momentum=momentum,
                volatility=volatility,
            )

            # Log the dynamic adjustment
            base_threshold = self.SELL_CONFIDENCE_THRESHOLDS.get(asset_class, 0.75)
            fixes_applied.append(
                f"Dynamic SELL: {trend_classification} -> threshold {base_threshold*100:.0f}% -> {dynamic_threshold*100:.0f}%"
            )

            if confidence < dynamic_threshold:
                blocked = True
                block_reason = (
                    f"LOW_CONF (Dynamic): {confidence*100:.0f}% < {dynamic_threshold*100:.0f}% "
                    f"(trend: {trend_classification}, base: {base_threshold*100:.0f}%)"
                )
                fixes_applied.append(f"Dynamic SELL threshold - BLOCKED")

        if not blocked:
            # Fix 3 & 7: SELL position multiplier
            sell_mult = self.SELL_POSITION_MULTIPLIERS.get(asset_class, 0.50)
            position_mult *= sell_mult
            if asset_class == AssetClass.COMMODITY:
                fixes_applied.append(f"Fix 7: Commodity 70% reduction -> {sell_mult:.2f}x")
            else:
                fixes_applied.append(f"Fix 3: SELL {int((1-sell_mult)*100)}% reduction -> {sell_mult:.2f}x")

            # Fix 10: Base allocation
            base_alloc = self.BASE_ALLOCATIONS.get(asset_class, 0.05)
            fixes_applied.append(f"Fix 10: Base allocation {base_alloc*100:.0f}%")

            # Fix 9: Win rate multiplier
            if self.enable_dynamic_sizing:
                ticker_win_rate = win_rate or self.historical_win_rates.get(ticker, 0.30)
                wr_mult = self.get_win_rate_multiplier(ticker_win_rate)
                position_mult *= wr_mult
                fixes_applied.append(f"Fix 9: Win rate {ticker_win_rate*100:.0f}% -> {wr_mult:.1f}x")

            # Fix 13: SELL position multiplier
            position_mult *= self.SELL_POSITION_MULTIPLIER
            fixes_applied.append(f"Fix 13: SELL multiplier {self.SELL_POSITION_MULTIPLIER:.2f}x")

        # Fix 4 & 13: Stop-loss (tighter for SELL)
        stop_loss = min(
            self.STOP_LOSS_BY_ASSET.get(asset_class, 0.06),
            self.SELL_STOP_LOSS
        )
        fixes_applied.append(f"Fix 4/13: Stop-loss {stop_loss*100:.0f}%")

        return SignalOptimization(
            ticker=ticker,
            signal_type='SELL',
            original_confidence=confidence,
            adjusted_confidence=adjusted_confidence,
            position_multiplier=position_mult,
            stop_loss_pct=stop_loss,
            take_profit_levels=self.PROFIT_TAKING_LEVELS.copy(),
            blocked=blocked,
            block_reason=block_reason,
            fixes_applied=fixes_applied,
            asset_class=asset_class.value,
            kelly_fraction=kelly_fraction,
        )

    def optimize_signal(
        self,
        ticker: str,
        signal_type: str,
        confidence: float,
        volatility: float = 0.20,
        momentum: float = 0.0,
        win_rate: Optional[float] = None,
    ) -> SignalOptimization:
        """
        Optimize any signal (BUY or SELL).

        Args:
            ticker: Stock/asset ticker
            signal_type: 'BUY' or 'SELL'
            confidence: Model confidence (0-1)
            volatility: Asset volatility (annualized)
            momentum: Recent momentum (e.g., 20-day return)
            win_rate: Historical win rate for this ticker

        Returns:
            SignalOptimization with all adjustments
        """
        if signal_type.upper() == 'BUY':
            return self.optimize_buy_signal(ticker, confidence, volatility, momentum, win_rate)
        else:
            return self.optimize_sell_signal(ticker, confidence, volatility, momentum, win_rate)

    def process_signals_batch(
        self,
        signals: List[Dict[str, Any]],
    ) -> Tuple[List[SignalOptimization], Dict[str, Any]]:
        """
        Process a batch of signals with optimization.

        Args:
            signals: List of dicts with keys: ticker, signal_type, confidence, etc.

        Returns:
            (optimized_signals, summary_stats)
        """
        optimized = []
        blocked_count = 0
        buy_count = 0
        sell_count = 0

        for sig in signals:
            result = self.optimize_signal(
                ticker=sig.get('ticker', ''),
                signal_type=sig.get('signal_type', 'BUY'),
                confidence=sig.get('confidence', 0.5),
                volatility=sig.get('volatility', 0.20),
                momentum=sig.get('momentum', 0.0),
                win_rate=sig.get('win_rate'),
            )
            optimized.append(result)

            if result.blocked:
                blocked_count += 1
            elif result.signal_type == 'BUY':
                buy_count += 1
            else:
                sell_count += 1

        summary = {
            'total_signals': len(signals),
            'blocked_count': blocked_count,
            'active_buy_count': buy_count,
            'active_sell_count': sell_count,
            'block_rate': blocked_count / len(signals) if signals else 0,
        }

        return optimized, summary

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of all fix configurations."""
        return {
            'fixes_1_2': {
                'description': 'SELL confidence thresholds by asset class',
                'thresholds': {k.value: v for k, v in self.SELL_CONFIDENCE_THRESHOLDS.items()},
            },
            'fix_3_7': {
                'description': 'SELL position multipliers',
                'multipliers': {k.value: v for k, v in self.SELL_POSITION_MULTIPLIERS.items()},
            },
            'fix_4': {
                'description': 'Asset-specific stop-losses',
                'stop_losses': {k.value: v for k, v in self.STOP_LOSS_BY_ASSET.items()},
            },
            'fix_5': {
                'description': 'BUY position boosts',
                'boosts': {k.value: v for k, v in self.BUY_POSITION_BOOSTS.items()},
            },
            'fix_6': {
                'description': 'SELL blacklist',
                'tickers': list(self.SELL_BLACKLIST),
            },
            'fix_8': {
                'description': 'Extended blocklist',
                'tickers': list(self.EXTENDED_BLOCKLIST),
            },
            'fix_9': {
                'description': 'Win rate multipliers',
                'ranges': {f"{low*100:.0f}-{high*100:.0f}%": mult
                          for (low, high), mult in self.WIN_RATE_MULTIPLIERS.items()},
            },
            'fix_10': {
                'description': 'Base allocations',
                'allocations': {k.value: v for k, v in self.BASE_ALLOCATIONS.items()},
            },
            'fix_11': {
                'description': 'Kelly Criterion',
                'enabled': self.enable_kelly,
            },
            'fix_12': {
                'description': 'BUY signal rules',
                'min_confidence': self.BUY_MIN_CONFIDENCE,
                'position_multiplier': self.BUY_POSITION_MULTIPLIER,
                'stop_loss': self.BUY_STOP_LOSS,
            },
            'fix_13': {
                'description': 'SELL signal rules (now using dynamic thresholds - see fix_20)',
                'base_confidence': self.SELL_MIN_CONFIDENCE_BASE,
                'confidence_floor': self.SELL_MIN_CONFIDENCE_FLOOR,
                'confidence_ceiling': self.SELL_MIN_CONFIDENCE_CEILING,
                'position_multiplier': self.SELL_POSITION_MULTIPLIER,
                'stop_loss': self.SELL_STOP_LOSS,
                'commodity_blacklist': list(self.COMMODITY_SELL_BLACKLIST),
                'crypto_blacklist': list(self.CRYPTO_SELL_BLACKLIST),
            },
            'fix_14': {
                'description': 'High-profit pattern detection',
                'enabled': True,
            },
            'fix_15': {
                'description': 'Profit-taking levels',
                'levels': {f"{k*100:.0f}%": f"close {v*100:.0f}%"
                          for k, v in self.PROFIT_TAKING_LEVELS.items()},
            },
            'fix_16': {
                'description': 'China stock blocking',
                'enabled': True,
                'note': 'China stocks (.HK, .SS, .SZ) raise ValueError - route to China model',
            },
            'fix_17': {
                'description': 'JPY SELL adaptive blocking',
                'enabled': self.enable_adaptive_blocking and RISK_MANAGEMENT_AVAILABLE,
                'rules': 'Block in BULL/RISK_ON, Allow in BEAR/RISK_OFF',
            },
            'fix_18': {
                'description': 'Crude Oil adaptive blocking',
                'enabled': self.enable_adaptive_blocking and RISK_MANAGEMENT_AVAILABLE,
                'rules': 'BUY in INFLATION/BULL, SELL in DEFLATION/BEAR',
            },
            'fix_19': {
                'description': 'Market regime position adjustment',
                'enabled': self.enable_regime_adjustment and RISK_MANAGEMENT_AVAILABLE,
                'current_regime': self._current_regime_detection.primary_regime.value if self._current_regime_detection else None,
            },
            'fix_20': {
                'description': 'Dynamic SELL thresholds based on trend/volatility (US/INTL ONLY)',
                'enabled': True,
                'base_threshold': self.SELL_MIN_CONFIDENCE_BASE,
                'floor': self.SELL_MIN_CONFIDENCE_FLOOR,
                'ceiling': self.SELL_MIN_CONFIDENCE_CEILING,
                'trend_adjustments': self.SELL_TREND_ADJUSTMENT,
                'note': 'Lowers threshold in downtrends (more SELL signals), raises in uptrends',
            },
            'fix_24': {
                'description': 'Adaptive Kelly Fraction (US/INTL ONLY)',
                'enabled': self.enable_adaptive_kelly,
                'account_size': self.account_size,
                'kelly_range': '5%-50%',
                'adjustments': ['regime', 'account_size', 'momentum'],
                'note': 'Adapts position sizing based on market regime, account size, and recent performance',
            },
            'fix_25': {
                'description': 'Position Concentration Optimizer (US/INTL ONLY)',
                'enabled': self.enable_position_concentration,
                'max_positions': 5,
                'top_3_concentration': '~70%',
                'note': 'Concentrates capital in highest-conviction trades using exponential weighting',
            },
            'fix_26': {
                'description': 'Dynamic Profit Targets (US/INTL ONLY)',
                'enabled': self.enable_dynamic_profit_targets,
                'adjustments': ['volatility (ATR)', 'trend_strength', 'asset_class'],
                'note': 'ATR-based profit targets that widen in high volatility and strong trends',
            },
            # ========== FIXES 27-33: US-SPECIFIC OPTIMIZATIONS ==========
            'fix_27': {
                'description': 'US Market Regime Classifier (US/INTL ONLY)',
                'enabled': self.enable_us_regime_classifier,
                'regimes': ['bull_momentum', 'bear_momentum', 'fomc_week', 'earnings_season', 'opex_week', 'sector_rotation'],
                'vix_thresholds': USMarketRegimeClassifier.VIX_THRESHOLDS if self.us_regime_classifier else None,
                'note': 'US-specific regime detection with VIX-based adjustments and special event handling',
            },
            'fix_28': {
                'description': 'Sector Momentum Integration (US/INTL ONLY)',
                'enabled': self.enable_sector_momentum,
                'sector_etfs': list(SectorMomentumAnalyzer.SECTOR_ETFS.keys()) if self.sector_momentum_analyzer else None,
                'leader_boosts': SectorMomentumAnalyzer.SECTOR_LEADER_BOOSTS if self.sector_momentum_analyzer else None,
                'note': 'Relative strength analysis vs sector ETFs for position adjustment',
            },
            'fix_29': {
                'description': 'Earnings Season Optimizer (US/INTL ONLY)',
                'enabled': self.enable_earnings_optimizer,
                'earnings_months': [1, 4, 7, 10],
                'major_tickers': list(EarningsSeasonOptimizer.MAJOR_EARNINGS_TICKERS)[:10] if self.earnings_optimizer else None,
                'note': 'Pre-earnings drift exploitation and SELL blocking during earnings week',
            },
            'fix_30': {
                'description': 'FOMC & Economic Calendar (US/INTL ONLY)',
                'enabled': self.enable_fomc_optimizer,
                'rate_sensitive_sectors': list(FOMCOptimizer.RATE_SENSITIVE_SECTORS) if self.fomc_optimizer else None,
                'growth_sensitive_sectors': list(FOMCOptimizer.GROWTH_SENSITIVE_SECTORS) if self.fomc_optimizer else None,
                'note': 'Position reduction during FOMC weeks, especially for rate-sensitive sectors',
            },
            'fix_31': {
                'description': 'Options Expiration Optimizer (US/INTL ONLY)',
                'enabled': self.enable_opex_optimizer,
                'high_gamma_stocks': list(self.opex_optimizer.get_high_gamma_stocks())[:10] if self.opex_optimizer else None,
                'note': 'Gamma hedging impact handling on OpEx week (3rd Friday)',
            },
            'fix_32': {
                'description': 'Market Internals Integration (US/INTL ONLY)',
                'enabled': self.enable_market_internals,
                'indicators': ['AD_ratio', 'NHNL_ratio', 'TRIN', 'McClellan_Oscillator'],
                'note': 'Market breadth indicators for position adjustment',
            },
            'fix_33': {
                'description': 'US-Specific Risk Models (US/INTL ONLY)',
                'enabled': self.enable_us_risk_model,
                'max_sector_concentration': USRiskModel.MAX_SECTOR_CONCENTRATION if self.us_risk_model else 0.35,
                'factor_limits': USRiskModel.FACTOR_LIMITS if self.us_risk_model else None,
                'note': 'Sector concentration limits and factor exposure management',
            },
            # ========== FIXES 34-41: ADVANCED PROFIT-MAXIMIZING STRATEGIES ==========
            'fix_34': {
                'description': 'Intraday Momentum Timing (US/INTL ONLY)',
                'enabled': self.enable_intraday_timing,
                'entry_windows': list(IntradayMomentumOptimizer.INTRADAY_ENTRY_TIMING.keys()) if self.intraday_optimizer else None,
                'note': 'Opening range breakout, midday dip, power hour momentum timing',
            },
            'fix_35': {
                'description': 'Market Cap Tier Optimizer (US/INTL ONLY)',
                'enabled': self.enable_market_cap_tiers,
                'tiers': list(MarketCapTierOptimizer.MARKET_CAP_THRESHOLDS.keys()) if self.market_cap_optimizer else None,
                'mega_cap_list': list(MarketCapTierOptimizer.KNOWN_MEGA_CAPS)[:10] if self.market_cap_optimizer else None,
                'note': 'Different strategies for mega-cap vs small-cap stocks',
            },
            'fix_36': {
                'description': 'Quarter-End Window Dressing (US/INTL ONLY)',
                'enabled': self.enable_quarter_end_optimizer,
                'quarter_end_months': [3, 6, 9, 12],
                'window_days': 5,  # Last 3-5 trading days of quarter
                'note': 'Exploit institutional window dressing at quarter-end',
            },
            'fix_37': {
                'description': 'Earnings Gap Trading (US/INTL ONLY)',
                'enabled': self.enable_earnings_gap_trading,
                'gap_strategies': list(EarningsGapTrader.GAP_STRATEGIES.keys()) if self.earnings_gap_trader else None,
                'note': 'Gap-up momentum and gap-down fade strategies post-earnings',
            },
            'fix_38': {
                'description': 'Sector Rotation Momentum (US/INTL ONLY)',
                'enabled': self.enable_sector_rotation,
                'rotation_statuses': list(SectorRotationMomentum.ROTATION_ADJUSTMENTS.keys()) if self.sector_rotation_momentum else None,
                'note': 'Boost positions in rotating-in sectors, reduce in rotating-out',
            },
            'fix_39': {
                'description': 'VIX Term Structure Arbitrage (US/INTL ONLY)',
                'enabled': self.enable_vix_term_structure,
                'regimes': list(VIXTermStructureAnalyzer.TERM_STRUCTURE_REGIMES.keys()) if self.vix_term_structure else None,
                'note': 'Contango favors BUY, backwardation favors SELL signals',
            },
            'fix_40': {
                'description': 'Economic Data Reactions (US/INTL ONLY)',
                'enabled': self.enable_economic_data_reactions,
                'economic_events': list(EconomicDataReactor.ECONOMIC_EVENTS.keys()) if self.economic_data_reactor else None,
                'note': 'High-frequency reaction strategies for CPI, jobs report, Fed decisions',
            },
            'fix_41': {
                'description': 'Put/Call Ratio Reversals (US/INTL ONLY)',
                'enabled': self.enable_put_call_ratio,
                'thresholds': PutCallRatioAnalyzer.PUT_CALL_THRESHOLDS if self.put_call_analyzer else None,
                'note': 'Contrarian signals from extreme put/call ratio readings',
            },
            # ========== FIXES 42-49: ADVANCED PROFIT-MAXIMIZING STRATEGIES II ==========
            'fix_42': {
                'description': 'Unified US Profit Maximizer (US/INTL ONLY)',
                'enabled': self.enable_unified_optimizer,
                'max_multiplier': UnifiedUSProfitMaximizer.MAX_COMBINED_MULTIPLIER if self.unified_profit_maximizer else 3.0,
                'min_multiplier': UnifiedUSProfitMaximizer.MIN_COMBINED_MULTIPLIER if self.unified_profit_maximizer else 0.20,
                'note': 'Master optimizer combining ALL fixes (27-41) multiplicatively',
            },
            'fix_43': {
                'description': 'Enhanced Sector Rotation Detector (US/INTL ONLY)',
                'enabled': self.enable_enhanced_sector_rotation,
                'leading_indicators': list(EnhancedSectorRotationDetector.LEADING_INDICATORS.keys()) if self.enhanced_sector_rotation else None,
                'note': 'Predictive rotation using institutional flows, RS momentum, earnings revisions',
            },
            'fix_44': {
                'description': 'US Catalyst Detector (US/INTL ONLY)',
                'enabled': self.enable_catalyst_detector,
                'catalyst_types': list(USCatalystDetector.CATALYST_TYPES.keys()) if self.catalyst_detector else None,
                'note': 'News-based catalyst detection (FDA approval, M&A, short squeeze, etc.)',
            },
            'fix_45': {
                'description': 'Enhanced Intraday with Volume Profile (US/INTL ONLY)',
                'enabled': self.enable_enhanced_intraday,
                'strategies': list(EnhancedIntradayOptimizer.VP_STRATEGIES.keys()) if self.enhanced_intraday else None,
                'note': 'POC, Value Area, Low Volume Nodes for optimal entry timing',
            },
            'fix_46': {
                'description': 'Momentum Acceleration Detector (US/INTL ONLY)',
                'enabled': self.enable_momentum_acceleration,
                'adjustments': list(MomentumAccelerationDetector.ADJUSTMENTS.keys()) if self.momentum_acceleration else None,
                'note': '2nd derivative of price for early trend detection',
            },
            'fix_47': {
                'description': 'US-Specific Profit Rules (US/INTL ONLY)',
                'enabled': self.enable_us_profit_rules,
                'stock_profiles': list(USProfitRules.STOCK_PROFILES.keys()) if self.us_profit_rules else None,
                'note': 'Different strategies per stock type (mega-cap, momentum, value, etc.)',
            },
            'fix_48': {
                'description': 'Smart Profit Taker (US/INTL ONLY)',
                'enabled': self.enable_smart_profit_taker,
                'factor_count': len(SmartProfitTaker.FACTOR_WEIGHTS) if self.smart_profit_taker else 10,
                'note': '10+ factor profit-taking decision matrix',
            },
            'fix_49': {
                'description': 'Backtest Profit Maximizer (US/INTL ONLY)',
                'enabled': self.enable_backtest_maximizer,
                'strategies': list(BacktestProfitMaximizer.BACKTEST_STRATEGIES.keys()) if self.backtest_maximizer else None,
                'note': 'Aggressive backtest-only strategies for theoretical maximum performance',
            },
        }

    # ========== FIX 16-19: Risk Management Methods ==========

    def update_market_regime(self, indicators: Dict[str, float]) -> Optional[Dict]:
        """
        Update the current market regime detection.

        Fix 19: Market regime drives position sizing and adaptive blocking.

        Args:
            indicators: Dict with keys matching RegimeIndicators:
                - vix: VIX index level
                - spy_return_20d: SPY 20-day return
                - spy_return_60d: SPY 60-day return
                - spy_above_200ma: bool
                - treasury_10y, treasury_2y: yields
                - usd_index_return: USD index return
                - tip_ief_spread: inflation indicator
                - gold_return_20d: gold return
                - high_yield_spread: optional credit spread

        Returns:
            Dict with regime detection results, or None if not available
        """
        if not RISK_MANAGEMENT_AVAILABLE or not self.regime_detector:
            return None

        from ..risk_management.market_regime_detector import create_indicators_from_data

        regime_indicators = create_indicators_from_data(
            vix=indicators.get('vix', 20.0),
            spy_return_20d=indicators.get('spy_return_20d', 0.0),
            spy_return_60d=indicators.get('spy_return_60d', 0.0),
            spy_above_200ma=indicators.get('spy_above_200ma', True),
            treasury_10y=indicators.get('treasury_10y', 4.0),
            treasury_2y=indicators.get('treasury_2y', 4.5),
            usd_index_return=indicators.get('usd_index_return', 0.0),
            tip_ief_spread=indicators.get('tip_ief_spread', 0.0),
            gold_return_20d=indicators.get('gold_return_20d', 0.0),
            high_yield_spread=indicators.get('high_yield_spread'),
        )

        self._current_regime_detection = self.regime_detector.detect_regime(regime_indicators)

        # Update adaptive blocker and position adjuster
        if self.adaptive_blocker:
            self.adaptive_blocker.update_regime(self._current_regime_detection)
        if self.position_adjuster:
            self.position_adjuster.update_regime(self._current_regime_detection)

        return {
            'primary_regime': self._current_regime_detection.primary_regime.value,
            'secondary_regime': self._current_regime_detection.secondary_regime.value if self._current_regime_detection.secondary_regime else None,
            'confidence': self._current_regime_detection.confidence,
            'regime_scores': {k.value: v for k, v in self._current_regime_detection.regime_scores.items()},
        }

    def get_current_regime(self) -> Optional[Dict]:
        """Get the current market regime detection."""
        if not self._current_regime_detection:
            return None

        return {
            'primary_regime': self._current_regime_detection.primary_regime.value,
            'secondary_regime': self._current_regime_detection.secondary_regime.value if self._current_regime_detection.secondary_regime else None,
            'confidence': self._current_regime_detection.confidence,
        }

    def check_adaptive_blocking(
        self,
        ticker: str,
        signal_type: str,
        asset_class: AssetClass,
    ) -> Tuple[bool, float, str]:
        """
        Check if a signal should be adaptively blocked.

        Fixes 17 & 18: Check JPY SELL and Crude Oil signals against regime.

        Args:
            ticker: Asset ticker
            signal_type: 'BUY' or 'SELL'
            asset_class: Asset classification

        Returns:
            (blocked, position_reduction, reason)
        """
        if not self.enable_adaptive_blocking or not RISK_MANAGEMENT_AVAILABLE:
            return False, 1.0, "Adaptive blocking disabled"

        if not self.adaptive_blocker or not self._current_regime_detection:
            return False, 1.0, "No regime detection available"

        # Only apply to JPY pairs and Crude Oil
        if asset_class not in [AssetClass.JPY_PAIR, AssetClass.CRUDE_OIL]:
            return False, 1.0, f"Adaptive blocking not applicable to {asset_class.value}"

        asset_type = asset_class.value
        result = self.adaptive_blocker.evaluate_signal(
            ticker=ticker,
            signal_type=signal_type,
            asset_type=asset_type,
            detection=self._current_regime_detection,
        )

        return result.blocked, result.position_reduction, result.reason

    def get_regime_position_adjustment(
        self,
        ticker: str,
        signal_type: str,
        asset_class: AssetClass,
        base_position: float,
        win_rate: Optional[float] = None,
    ) -> Tuple[float, str]:
        """
        Get position size adjustment based on market regime.

        Fix 19: Dynamic position sizing.

        Args:
            ticker: Asset ticker
            signal_type: 'BUY' or 'SELL'
            asset_class: Asset classification
            base_position: Base position size (0-1)
            win_rate: Optional historical win rate

        Returns:
            (adjusted_position, explanation)
        """
        if not self.enable_regime_adjustment or not RISK_MANAGEMENT_AVAILABLE:
            return base_position, "Regime adjustment disabled"

        if not self.position_adjuster or not self._current_regime_detection:
            return base_position, "No regime detection available"

        adjustment = self.position_adjuster.adjust_position(
            ticker=ticker,
            signal_type=signal_type,
            asset_type=asset_class.value,
            base_position=base_position,
            win_rate=win_rate,
            detection=self._current_regime_detection,
        )

        explanation = f"Regime {adjustment.regime.value}: {adjustment.total_multiplier:.2f}x multiplier"
        return adjustment.adjusted_position, explanation

    def is_china_stock(self, ticker: str) -> bool:
        """
        Check if a ticker is a China stock.

        Fix 16: China stocks should be routed to China model.

        Args:
            ticker: Stock ticker

        Returns:
            True if China stock (should not be processed by this model)
        """
        ticker_upper = ticker.upper()
        china_suffixes = ['.HK', '.SS', '.SZ']
        return any(ticker_upper.endswith(suffix) for suffix in china_suffixes)


# Convenience function for external use
def create_optimizer(
    historical_win_rates: Optional[Dict[str, float]] = None,
    enable_kelly: bool = True,
    enable_dynamic_sizing: bool = True,
    enable_regime_adjustment: bool = True,
    enable_adaptive_blocking: bool = True,
) -> USIntlModelOptimizer:
    """
    Create a configured US/Intl model optimizer with all fixes (1-19).

    Args:
        historical_win_rates: Optional dict of ticker -> win_rate
        enable_kelly: Enable Kelly Criterion sizing (Fix 11)
        enable_dynamic_sizing: Enable win-rate based sizing (Fix 9)
        enable_regime_adjustment: Enable market regime position adjustment (Fix 19)
        enable_adaptive_blocking: Enable adaptive blocking for JPY/Crude (Fixes 17 & 18)

    Returns:
        Configured USIntlModelOptimizer instance
    """
    return USIntlModelOptimizer(
        historical_win_rates=historical_win_rates,
        enable_kelly=enable_kelly,
        enable_dynamic_sizing=enable_dynamic_sizing,
        enable_regime_adjustment=enable_regime_adjustment,
        enable_adaptive_blocking=enable_adaptive_blocking,
    )


if __name__ == '__main__':
    # Test the optimizer
    print("="*80)
    print("US/INTL MODEL OPTIMIZER - TEST")
    print("="*80)

    optimizer = create_optimizer()

    # Test signals
    test_signals = [
        {'ticker': 'LUMN', 'signal_type': 'BUY', 'confidence': 0.94},
        {'ticker': 'VERA', 'signal_type': 'BUY', 'confidence': 0.837},
        {'ticker': 'MDB', 'signal_type': 'BUY', 'confidence': 0.80},
        {'ticker': 'NG=F', 'signal_type': 'SELL', 'confidence': 0.68},
        {'ticker': 'ZC=F', 'signal_type': 'SELL', 'confidence': 0.63},
        {'ticker': 'USDT-USD', 'signal_type': 'SELL', 'confidence': 0.65},
        {'ticker': 'GC=F', 'signal_type': 'BUY', 'confidence': 0.632},
        {'ticker': 'BNB-USD', 'signal_type': 'BUY', 'confidence': 0.927},
    ]

    print("\nProcessing test signals...")
    results, summary = optimizer.process_signals_batch(test_signals)

    print(f"\nSummary: {summary}")
    print("\nDetailed Results:")
    print("-"*80)

    for r in results:
        status = "BLOCKED" if r.blocked else "ACTIVE"
        print(f"\n{r.ticker} ({r.signal_type}) [{status}]")
        print(f"  Confidence: {r.original_confidence*100:.0f}%")
        print(f"  Position Mult: {r.position_multiplier:.2f}x")
        print(f"  Stop-Loss: {r.stop_loss_pct*100:.0f}%")
        if r.blocked:
            print(f"  Block Reason: {r.block_reason}")
        print(f"  Fixes Applied: {', '.join(r.fixes_applied[:3])}...")

    print("\n" + "="*80)
    print("Configuration Summary:")
    print("="*80)
    config = optimizer.get_configuration_summary()
    for fix, details in config.items():
        print(f"\n{fix}: {details['description']}")
