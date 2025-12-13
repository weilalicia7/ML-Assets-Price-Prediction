"""
Yahoo Finance Screener Discovery Module - REAL-TIME ONLY

Real-time ticker discovery using Yahoo Finance built-in screeners and EquityQuery.
NO HARDCODED FALLBACKS - only live data from Yahoo Finance.

If Yahoo Finance is unavailable, we tell the user honestly.
"""

import yfinance as yf
from yfinance import EquityQuery
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# ============================================================================
# YAHOO SCREENER DISCOVERER - REAL-TIME ONLY
# ============================================================================

class YahooScreenerDiscoverer:
    """
    Discover tickers using Yahoo Finance built-in screeners and EquityQuery.
    NO HARDCODED FALLBACKS - real-time only.
    """

    def __init__(self):
        self.screeners = {
            # US Market Screeners (built-in Yahoo Finance screeners)
            'us_gainers': 'day_gainers',
            'us_losers': 'day_losers',
            'us_active': 'most_actives',
            'us_undervalued': 'undervalued_growth_stocks',
            'us_tech_growth': 'growth_technology_stocks',
            'us_small_cap': 'small_cap_gainers',
        }

        # Cache for screener results
        self._cache = {}
        self._cache_timestamps = {}

        # Cache TTL in seconds
        self._cache_ttl = {
            'us_gainers': 900,
            'us_losers': 900,
            'us_active': 1800,
            'us_undervalued': 3600,
            'us_tech_growth': 3600,
            'us_small_cap': 1800,
            'hk_active': 1800,
            'china_active': 1800,
            'crypto': 600,
            'commodity': 1800,
            'forex': 1800,
        }

    def discover_tickers(self, screener_type: str, count: int = 30) -> List[str]:
        """
        Discover tickers using Yahoo Finance screeners - REAL-TIME ONLY.
        Returns empty list if Yahoo Finance is unavailable (no hardcoded fallbacks).
        """
        # Check cache first
        cache_key = f"{screener_type}_{count}"
        if self._is_cache_valid(cache_key, screener_type):
            logger.info(f"[CACHE HIT] Using cached {screener_type} results")
            return self._cache[cache_key][:count]

        try:
            if screener_type == 'hk_active':
                tickers = self._hk_screener_realtime(count)
            elif screener_type == 'china_active':
                tickers = self._china_screener_realtime(count)
            elif screener_type == 'crypto':
                tickers = self._crypto_screener_realtime(count)
            elif screener_type == 'commodity':
                tickers = self._commodity_screener_realtime(count)
            elif screener_type == 'forex':
                tickers = self._forex_screener_realtime(count)
            else:
                # Use built-in Yahoo screeners for US stocks
                tickers = self._builtin_screener(screener_type, count)

            # Cache results only if we got real data
            if tickers:
                self._cache[cache_key] = tickers
                self._cache_timestamps[cache_key] = time.time()
                logger.info(f"[SCREENER] Found {len(tickers)} real-time tickers for {screener_type}")
            else:
                logger.warning(f"[SCREENER] No tickers found for {screener_type} - Yahoo Finance may be unavailable")

            return tickers[:count] if tickers else []

        except Exception as e:
            logger.error(f"[SCREENER ERROR] {screener_type}: {e}")
            return []  # Return empty, no hardcoded fallback

    def _is_cache_valid(self, cache_key: str, screener_type: str) -> bool:
        """Check if cached results are still valid"""
        if cache_key not in self._cache:
            return False

        timestamp = self._cache_timestamps.get(cache_key, 0)
        ttl = self._cache_ttl.get(screener_type, 1800)

        return (time.time() - timestamp) < ttl

    def _builtin_screener(self, screener_type: str, count: int) -> List[str]:
        """Use Yahoo Finance built-in screeners (day_gainers, most_actives, etc.)"""
        if screener_type not in self.screeners:
            logger.warning(f"Unknown screener type: {screener_type}")
            return []

        screener_name = self.screeners[screener_type]

        try:
            results = yf.screen(screener_name, size=count)

            if results and 'quotes' in results:
                symbols = [q.get('symbol', '') for q in results['quotes'] if q.get('symbol')]
                logger.info(f"[BUILTIN] {screener_name}: found {len(symbols)} tickers")
                return symbols[:count]

            return []

        except Exception as e:
            logger.error(f"[BUILTIN SCREENER ERROR] {screener_name}: {e}")
            return []

    def _hk_screener_realtime(self, count: int) -> List[str]:
        """Screen Hong Kong market using EquityQuery - REAL-TIME ONLY"""
        try:
            # Use market cap filter to get main board stocks, exclude small warrants
            hk_query = EquityQuery('AND', [
                EquityQuery('EQ', ['exchange', 'HKG']),
                EquityQuery('GT', ['intradaymarketcap', 1000000000])  # >1B HKD market cap
            ])

            results = yf.screen(hk_query, size=count * 3, sortField='dayvolume', sortAsc=False)

            if results and 'quotes' in results:
                symbols = []
                for q in results['quotes']:
                    symbol = q.get('symbol', '')
                    if symbol and symbol.endswith('.HK'):
                        ticker_num = symbol.replace('.HK', '').lstrip('0')
                        # Filter: only main board stocks (0001-4999), exclude warrants
                        if ticker_num.isdigit():
                            num = int(ticker_num)
                            if num < 5000:
                                symbols.append(symbol)

                if symbols:
                    logger.info(f"[HK SCREENER REALTIME] Found {len(symbols)} HK main board stocks")
                    return symbols[:count]

            logger.warning("[HK SCREENER] No real-time data available from Yahoo Finance")
            return []

        except Exception as e:
            logger.error(f"[HK SCREENER ERROR] {e}")
            return []

    def _shanghai_screener_realtime(self, count: int) -> List[str]:
        """Screen Shanghai A-shares using EquityQuery - REAL-TIME ONLY"""
        try:
            # Shanghai stocks have .SS suffix
            cn_query = EquityQuery('AND', [
                EquityQuery('EQ', ['exchange', 'SHH']),  # Shanghai
                EquityQuery('GT', ['dayvolume', 10000000])
            ])

            results = yf.screen(cn_query, size=count * 5, sortField='dayvolume', sortAsc=False)

            if results and 'quotes' in results:
                symbols = []
                for q in results['quotes']:
                    symbol = q.get('symbol', '')
                    if symbol.endswith('.SS'):
                        # Filter for actual A-shares only:
                        # 600xxx, 601xxx, 603xxx = Main board
                        # 688xxx = STAR Market (Science and Technology)
                        code = symbol.replace('.SS', '')
                        if code.startswith('60') or code.startswith('688'):
                            symbols.append(symbol)

                if symbols:
                    logger.info(f"[SHANGHAI SCREENER REALTIME] Found {len(symbols)} A-shares")
                    return symbols[:count]

            logger.warning("[SHANGHAI SCREENER] No real-time data from Yahoo Finance")
            return []

        except Exception as e:
            logger.error(f"[SHANGHAI SCREENER ERROR] {e}")
            return []

    def _shenzhen_screener_realtime(self, count: int) -> List[str]:
        """Screen Shenzhen A-shares using EquityQuery - REAL-TIME ONLY"""
        try:
            # Shenzhen stocks have .SZ suffix
            sz_query = EquityQuery('AND', [
                EquityQuery('EQ', ['exchange', 'SHZ']),  # Shenzhen
                EquityQuery('GT', ['dayvolume', 10000000])
            ])

            results = yf.screen(sz_query, size=count * 5, sortField='dayvolume', sortAsc=False)

            if results and 'quotes' in results:
                symbols = []
                for q in results['quotes']:
                    symbol = q.get('symbol', '')
                    if symbol.endswith('.SZ'):
                        # Filter for actual A-shares only:
                        # 000xxx, 001xxx, 002xxx = Main board
                        # 300xxx = ChiNext (Growth Enterprise Market)
                        code = symbol.replace('.SZ', '')
                        if code.startswith('00') or code.startswith('30'):
                            symbols.append(symbol)

                if symbols:
                    logger.info(f"[SHENZHEN SCREENER REALTIME] Found {len(symbols)} A-shares")
                    return symbols[:count]

            logger.warning("[SHENZHEN SCREENER] No real-time data from Yahoo Finance")
            return []

        except Exception as e:
            logger.error(f"[SHENZHEN SCREENER ERROR] {e}")
            return []

    def _china_screener_realtime(self, count: int) -> List[str]:
        """Screen all China markets (HK, Shanghai, Shenzhen) - REAL-TIME ONLY"""
        per_market = max(count // 3, 5)

        # Get real-time data from all 3 markets
        hk_tickers = self._hk_screener_realtime(per_market)
        ss_tickers = self._shanghai_screener_realtime(per_market)
        sz_tickers = self._shenzhen_screener_realtime(per_market)

        combined = list(dict.fromkeys(hk_tickers + ss_tickers + sz_tickers))

        logger.info(f"[CHINA SCREENER REALTIME] HK={len(hk_tickers)}, SS={len(ss_tickers)}, SZ={len(sz_tickers)}, total={len(combined)}")

        return combined[:count]

    def _crypto_screener_realtime(self, count: int) -> List[str]:
        """Screen cryptocurrencies using Yahoo Finance built-in screener - REAL-TIME ONLY"""
        try:
            # Use Yahoo's built-in crypto screener (all_cryptocurrencies_us)
            results = yf.screen('all_cryptocurrencies_us', count=count * 2)

            if results and 'quotes' in results:
                symbols = []
                for q in results['quotes']:
                    symbol = q.get('symbol', '')
                    if symbol and '-USD' in symbol:
                        symbols.append(symbol)

                if symbols:
                    logger.info(f"[CRYPTO SCREENER REALTIME] Found {len(symbols)} cryptocurrencies")
                    return symbols[:count]

            logger.warning("[CRYPTO SCREENER] No real-time data from Yahoo Finance")
            return []

        except Exception as e:
            logger.error(f"[CRYPTO SCREENER ERROR] {e}")
            return []

    def _commodity_screener_realtime(self, count: int) -> List[str]:
        """
        Screen commodities/futures - REAL-TIME ONLY

        Yahoo Finance does NOT have a built-in screener for futures like 'all_cryptocurrencies_us'.
        We use comprehensive symbol patterns and validate with real-time data, sorted by volume.
        """
        # Comprehensive list of ALL Yahoo Finance futures symbols
        # These are standard CME/NYMEX/COMEX/CBOT/ICE futures codes
        all_futures_symbols = [
            # Precious Metals
            'GC=F',   # Gold (COMEX)
            'SI=F',   # Silver (COMEX)
            'PL=F',   # Platinum (NYMEX)
            'PA=F',   # Palladium (NYMEX)
            'HG=F',   # Copper (COMEX)

            # Energy
            'CL=F',   # Crude Oil WTI (NYMEX)
            'BZ=F',   # Brent Crude (ICE)
            'NG=F',   # Natural Gas (NYMEX)
            'RB=F',   # RBOB Gasoline (NYMEX)
            'HO=F',   # Heating Oil (NYMEX)

            # Grains & Oilseeds
            'ZC=F',   # Corn (CBOT)
            'ZW=F',   # Wheat (CBOT)
            'ZS=F',   # Soybeans (CBOT)
            'ZM=F',   # Soybean Meal (CBOT)
            'ZL=F',   # Soybean Oil (CBOT)
            'ZO=F',   # Oats (CBOT)
            'ZR=F',   # Rough Rice (CBOT)
            'KE=F',   # KC Wheat (CBOT)

            # Softs
            'KC=F',   # Coffee (ICE)
            'SB=F',   # Sugar #11 (ICE)
            'CC=F',   # Cocoa (ICE)
            'CT=F',   # Cotton (ICE)
            'OJ=F',   # Orange Juice (ICE)
            'LBS=F',  # Lumber

            # Livestock
            'LE=F',   # Live Cattle (CME)
            'HE=F',   # Lean Hogs (CME)
            'GF=F',   # Feeder Cattle (CME)

            # Index Futures
            'ES=F',   # E-mini S&P 500
            'NQ=F',   # E-mini Nasdaq 100
            'YM=F',   # E-mini Dow
            'RTY=F',  # E-mini Russell 2000

            # Treasury Futures
            'ZB=F',   # 30-Year T-Bond
            'ZN=F',   # 10-Year T-Note
            'ZF=F',   # 5-Year T-Note
            'ZT=F',   # 2-Year T-Note

            # Currency Futures
            '6E=F',   # Euro FX
            '6J=F',   # Japanese Yen
            '6B=F',   # British Pound
            '6C=F',   # Canadian Dollar
            '6A=F',   # Australian Dollar
            '6S=F',   # Swiss Franc
            '6N=F',   # New Zealand Dollar
            '6M=F',   # Mexican Peso

            # Note: VX=F (VIX Futures) removed - not available on Yahoo Finance
        ]

        # Validate and get volume data in parallel for speed
        valid_with_volume = []

        def check_ticker(ticker):
            try:
                t = yf.Ticker(ticker)
                info = t.fast_info
                price = getattr(info, 'last_price', None)
                volume = getattr(info, 'last_volume', 0) or 0
                if price and price > 0:
                    return (ticker, volume)
            except Exception:
                pass
            return None

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(check_ticker, all_futures_symbols))

        valid_with_volume = [r for r in results if r is not None]

        # Sort by volume (most active first) for better trading opportunities
        valid_with_volume.sort(key=lambda x: x[1], reverse=True)
        valid_tickers = [t[0] for t in valid_with_volume]

        if valid_tickers:
            logger.info(f"[COMMODITY SCREENER REALTIME] Found {len(valid_tickers)} tradeable futures (sorted by volume)")
            return valid_tickers[:count]

        logger.warning("[COMMODITY SCREENER] No real-time commodity data available from Yahoo Finance")
        return []

    def _forex_screener_realtime(self, count: int) -> List[str]:
        """
        Screen forex pairs - REAL-TIME ONLY

        Yahoo Finance does NOT have a built-in screener for forex.
        We use comprehensive currency pair patterns and validate with real-time data, sorted by volume.
        """
        # Comprehensive list of ALL major, minor, and emerging market forex pairs on Yahoo Finance
        all_forex_symbols = [
            # G10 Majors (USD base)
            'EURUSD=X',   # Euro
            'GBPUSD=X',   # British Pound
            'USDJPY=X',   # Japanese Yen
            'USDCHF=X',   # Swiss Franc
            'AUDUSD=X',   # Australian Dollar
            'NZDUSD=X',   # New Zealand Dollar
            'USDCAD=X',   # Canadian Dollar

            # G10 Crosses
            'EURGBP=X',   # Euro/Pound
            'EURJPY=X',   # Euro/Yen
            'EURCHF=X',   # Euro/Franc
            'EURAUD=X',   # Euro/Aussie
            'EURNZD=X',   # Euro/Kiwi
            'EURCAD=X',   # Euro/CAD
            'GBPJPY=X',   # Pound/Yen
            'GBPCHF=X',   # Pound/Franc
            'GBPAUD=X',   # Pound/Aussie
            'GBPNZD=X',   # Pound/Kiwi
            'GBPCAD=X',   # Pound/CAD
            'AUDJPY=X',   # Aussie/Yen
            'AUDNZD=X',   # Aussie/Kiwi
            'AUDCAD=X',   # Aussie/CAD
            'AUDCHF=X',   # Aussie/Franc
            'NZDJPY=X',   # Kiwi/Yen
            'NZDCAD=X',   # Kiwi/CAD
            'NZDCHF=X',   # Kiwi/Franc
            'CADJPY=X',   # CAD/Yen
            'CADCHF=X',   # CAD/Franc
            'CHFJPY=X',   # Franc/Yen

            # Asia Pacific
            'USDCNH=X',   # USD/Chinese Yuan (offshore)
            'USDCNY=X',   # USD/Chinese Yuan (onshore)
            'USDHKD=X',   # USD/Hong Kong Dollar
            'USDSGD=X',   # USD/Singapore Dollar
            'USDKRW=X',   # USD/Korean Won
            'USDINR=X',   # USD/Indian Rupee
            'USDTHB=X',   # USD/Thai Baht
            'USDTWD=X',   # USD/Taiwan Dollar
            'USDPHP=X',   # USD/Philippine Peso
            'USDIDR=X',   # USD/Indonesian Rupiah
            'USDMYR=X',   # USD/Malaysian Ringgit
            'USDVND=X',   # USD/Vietnamese Dong

            # Americas (Emerging)
            'USDMXN=X',   # USD/Mexican Peso
            'USDBRL=X',   # USD/Brazilian Real
            'USDCLP=X',   # USD/Chilean Peso
            'USDCOP=X',   # USD/Colombian Peso
            'USDARS=X',   # USD/Argentine Peso
            'USDPEN=X',   # USD/Peruvian Sol

            # EMEA (Emerging)
            'USDTRY=X',   # USD/Turkish Lira
            'USDZAR=X',   # USD/South African Rand
            'USDRUB=X',   # USD/Russian Ruble
            'USDPLN=X',   # USD/Polish Zloty
            'USDCZK=X',   # USD/Czech Koruna
            'USDHUF=X',   # USD/Hungarian Forint
            'USDRON=X',   # USD/Romanian Leu
            'USDILS=X',   # USD/Israeli Shekel
            'USDAED=X',   # USD/UAE Dirham
            'USDSAR=X',   # USD/Saudi Riyal
            'USDEGP=X',   # USD/Egyptian Pound
            'USDNOK=X',   # USD/Norwegian Krone
            'USDSEK=X',   # USD/Swedish Krona
            'USDDKK=X',   # USD/Danish Krone

            # Note: BTCUSD=X/ETHUSD=X removed - crypto uses -USD format (BTC-USD), not =X
        ]

        # Validate and get volume data in parallel for speed
        valid_with_volume = []

        def check_pair(pair):
            try:
                t = yf.Ticker(pair)
                info = t.fast_info
                price = getattr(info, 'last_price', None)
                volume = getattr(info, 'last_volume', 0) or 0
                if price and price > 0:
                    return (pair, volume)
            except Exception:
                pass
            return None

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(check_pair, all_forex_symbols))

        valid_with_volume = [r for r in results if r is not None]

        # Sort by volume (most active first) for better trading opportunities
        valid_with_volume.sort(key=lambda x: x[1], reverse=True)
        valid_pairs = [t[0] for t in valid_with_volume]

        if valid_pairs:
            logger.info(f"[FOREX SCREENER REALTIME] Found {len(valid_pairs)} tradeable forex pairs (sorted by volume)")
            return valid_pairs[:count]

        logger.warning("[FOREX SCREENER] No real-time forex data available from Yahoo Finance")
        return []

    def clear_cache(self):
        """Clear all cached screener results"""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("[CACHE] Cleared all screener cache")


# ============================================================================
# REGIME SCREENER STRATEGY - REAL-TIME ONLY
# ============================================================================

class RegimeScreenerStrategy:
    """
    Select appropriate screeners based on market regime.
    NO HARDCODED FALLBACKS - real-time only.
    """

    def __init__(self):
        self.discoverer = YahooScreenerDiscoverer()

    def get_tickers_for_regime(self, regime: str, count: int = 30) -> Tuple[List[str], str]:
        """
        Get tickers using regime-appropriate screeners - REAL-TIME ONLY.
        Returns empty list if Yahoo Finance is unavailable.
        """
        regime_lower = regime.lower()

        if regime_lower == 'stock':
            return self._us_stock_strategy(count), 'us_screener'
        elif regime_lower == 'cryptocurrency':
            return self._crypto_strategy(count), 'crypto_screener'
        elif regime_lower == 'china':
            return self._china_strategy(count), 'china_screener'
        elif regime_lower == 'commodity':
            return self._commodity_strategy(count), 'commodity_screener'
        elif regime_lower == 'forex':
            return self._forex_strategy(count), 'forex_screener'
        elif regime_lower == 'all':
            return self._all_markets_strategy(count), 'mixed_screener'
        else:
            return self._default_strategy(count), 'default_screener'

    def _us_stock_strategy(self, count: int) -> List[str]:
        """US stocks: combine gainers + active + undervalued - REAL-TIME ONLY"""
        per_screener = max(count // 3, 10)

        gainers = self.discoverer.discover_tickers('us_gainers', per_screener)
        active = self.discoverer.discover_tickers('us_active', per_screener)
        undervalued = self.discoverer.discover_tickers('us_undervalued', per_screener)

        combined = list(dict.fromkeys(gainers + active + undervalued))
        logger.info(f"[US STRATEGY] Combined {len(combined)} unique tickers from real-time screeners")

        return combined[:count]

    def _crypto_strategy(self, count: int) -> List[str]:
        """Cryptocurrencies - REAL-TIME ONLY"""
        return self.discoverer.discover_tickers('crypto', count)

    def _china_strategy(self, count: int) -> List[str]:
        """China markets (HK + Shanghai + Shenzhen) - REAL-TIME ONLY"""
        return self.discoverer.discover_tickers('china_active', count)

    def _commodity_strategy(self, count: int) -> List[str]:
        """Commodities/Futures - REAL-TIME ONLY"""
        return self.discoverer.discover_tickers('commodity', count)

    def _forex_strategy(self, count: int) -> List[str]:
        """Forex pairs - REAL-TIME ONLY"""
        return self.discoverer.discover_tickers('forex', count)

    def _all_markets_strategy(self, count: int) -> List[str]:
        """Mixed markets - REAL-TIME ONLY"""
        per_type = max(count // 4, 5)

        stocks = self._us_stock_strategy(per_type)
        crypto = self._crypto_strategy(per_type)
        commodity = self._commodity_strategy(per_type)
        china = self._china_strategy(per_type)

        combined = list(dict.fromkeys(stocks + crypto + commodity + china))
        return combined[:count]

    def _default_strategy(self, count: int) -> List[str]:
        """Default: US stocks - REAL-TIME ONLY"""
        return self._us_stock_strategy(count)


# ============================================================================
# SMART SCREENER CACHE
# ============================================================================

class SmartScreenerCache:
    """
    Cache manager for screener results.
    """

    def __init__(self, default_ttl: int = 1800):
        self._cache = {}
        self._timestamps = {}
        self._default_ttl = default_ttl

    def get(self, key: str, ttl: int = None) -> Optional[List[str]]:
        """Get cached value if still valid"""
        if key not in self._cache:
            return None

        age = time.time() - self._timestamps.get(key, 0)
        max_age = ttl or self._default_ttl

        if age < max_age:
            return self._cache[key]

        return None

    def set(self, key: str, value: List[str]):
        """Store value in cache"""
        self._cache[key] = value
        self._timestamps[key] = time.time()

    def clear(self):
        """Clear all cache"""
        self._cache.clear()
        self._timestamps.clear()


# ============================================================================
# RELIABILITY MANAGER
# ============================================================================

class ReliabilityManager:
    """
    Track screener reliability and decide when to report unavailable.
    """

    def __init__(self, threshold: float = 0.7):
        self._success_counts = {}
        self._failure_counts = {}
        self._threshold = threshold

    def should_use_screeners(self, regime: str) -> bool:
        """Check if screeners are reliable enough for this regime"""
        success_rate = self.get_success_rate(regime)
        return success_rate >= self._threshold

    def track_performance(self, regime: str, success: bool):
        """Track screener performance"""
        if regime not in self._success_counts:
            self._success_counts[regime] = 0
            self._failure_counts[regime] = 0

        if success:
            self._success_counts[regime] += 1
        else:
            self._failure_counts[regime] += 1

    def get_success_rate(self, regime: str) -> float:
        """Get success rate for a regime"""
        successes = self._success_counts.get(regime, 0)
        failures = self._failure_counts.get(regime, 0)
        total = successes + failures

        if total == 0:
            return 1.0  # Default to trusting screeners

        return successes / total

    def get_stats(self) -> Dict:
        """Get all reliability stats"""
        stats = {}
        all_regimes = set(self._success_counts.keys()) | set(self._failure_counts.keys())

        for regime in all_regimes:
            stats[regime] = {
                'success_rate': self.get_success_rate(regime),
                'successes': self._success_counts.get(regime, 0),
                'failures': self._failure_counts.get(regime, 0)
            }

        return stats


# ============================================================================
# HELPER FUNCTIONS - NO HARDCODED DATA
# ============================================================================

def filter_quality_tickers(tickers: List[str], min_volume: float = 1000000) -> List[str]:
    """
    Filter tickers by quality metrics using real-time Yahoo data.
    """
    quality_tickers = []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).fast_info
            volume = getattr(info, 'last_volume', 0) or 0
            if volume >= min_volume:
                quality_tickers.append(ticker)
        except Exception:
            continue

    return quality_tickers


# ============================================================================
# MODULE SINGLETONS
# ============================================================================

_discoverer_instance = None
_strategy_instance = None
_cache_instance = None
_reliability_instance = None


def get_screener_discoverer() -> YahooScreenerDiscoverer:
    global _discoverer_instance
    if _discoverer_instance is None:
        _discoverer_instance = YahooScreenerDiscoverer()
    return _discoverer_instance


def get_regime_strategy() -> RegimeScreenerStrategy:
    global _strategy_instance
    if _strategy_instance is None:
        _strategy_instance = RegimeScreenerStrategy()
    return _strategy_instance


def get_screener_cache() -> SmartScreenerCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SmartScreenerCache()
    return _cache_instance


def get_reliability_manager() -> ReliabilityManager:
    global _reliability_instance
    if _reliability_instance is None:
        _reliability_instance = ReliabilityManager()
    return _reliability_instance
