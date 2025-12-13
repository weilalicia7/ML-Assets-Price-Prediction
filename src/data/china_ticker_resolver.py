"""
China Stock Ticker Resolution System

Resolves company names (English, Chinese, partial) to standardized ticker symbols.
Supports:
- Hong Kong (HK) stocks: 0700.HK, 9988.HK
- Shanghai A-shares: 600519.SS
- Shenzhen A-shares: 000001.SZ, 300274.SZ

Features:
- Fuzzy matching for company names
- Multiple input formats support
- Auto-complete suggestions
- Ticker validation
"""

import re
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CHINA STOCK DATABASE
# ============================================================================

# Comprehensive database of China stocks with multiple name variants
CHINA_STOCKS_DATABASE = {
    # ========== HONG KONG LARGE CAPS ==========
    '0700.HK': {
        'names': ['Tencent', 'Tencent Holdings', '腾讯', '腾讯控股'],
        'sector': 'Tech',
        'market_cap': 'Large'
    },
    '9988.HK': {
        'names': ['Alibaba', 'BABA', '阿里巴巴', '阿里'],
        'sector': 'Tech',
        'market_cap': 'Large'
    },
    '9999.HK': {
        'names': ['NetEase', '网易', 'NTES'],
        'sector': 'Tech',
        'market_cap': 'Large'
    },
    '9888.HK': {
        'names': ['Baidu', '百度', 'BIDU'],
        'sector': 'Tech',
        'market_cap': 'Large'
    },
    '3690.HK': {
        'names': ['Meituan', '美团', 'Meituan Dianping'],
        'sector': 'Tech',
        'market_cap': 'Large'
    },
    '9618.HK': {
        'names': ['JD.com', 'JD', 'Jingdong', '京东'],
        'sector': 'Tech',
        'market_cap': 'Large'
    },
    '9961.HK': {
        'names': ['Trip.com', 'Ctrip', '携程', 'TCOM'],
        'sector': 'Tech',
        'market_cap': 'Large'
    },
    '1810.HK': {
        'names': ['Xiaomi', '小米', 'MI'],
        'sector': 'Tech',
        'market_cap': 'Large'
    },
    '2015.HK': {
        'names': ['Li Auto', 'LI', '理想汽车', '理想'],
        'sector': 'Auto',
        'market_cap': 'Large'
    },
    '9866.HK': {
        'names': ['NIO', '蔚来', 'Nio Inc'],
        'sector': 'Auto',
        'market_cap': 'Large'
    },
    '9868.HK': {
        'names': ['XPeng', 'XPEV', '小鹏', '小鹏汽车'],
        'sector': 'Auto',
        'market_cap': 'Large'
    },
    '0175.HK': {
        'names': ['Geely', 'Geely Auto', '吉利', '吉利汽车'],
        'sector': 'Auto',
        'market_cap': 'Large'
    },
    '2333.HK': {
        'names': ['Great Wall Motor', 'GWM', '长城汽车', '长城'],
        'sector': 'Auto',
        'market_cap': 'Large'
    },
    '1211.HK': {
        'names': ['BYD', '比亚迪', 'Build Your Dreams'],
        'sector': 'Auto',
        'market_cap': 'Large'
    },

    # ========== HONG KONG FINANCIALS ==========
    '0939.HK': {
        'names': ['CCB', 'China Construction Bank', '建设银行', '建行'],
        'sector': 'Financials',
        'market_cap': 'Large'
    },
    '1398.HK': {
        'names': ['ICBC', 'Industrial and Commercial Bank', '工商银行', '工行'],
        'sector': 'Financials',
        'market_cap': 'Large'
    },
    '3988.HK': {
        'names': ['Bank of China', 'BOC', '中国银行', '中行'],
        'sector': 'Financials',
        'market_cap': 'Large'
    },
    '2628.HK': {
        'names': ['China Life', '中国人寿', '国寿'],
        'sector': 'Financials',
        'market_cap': 'Large'
    },
    '2318.HK': {
        'names': ['Ping An', 'Ping An Insurance', '平安', '中国平安'],
        'sector': 'Financials',
        'market_cap': 'Large'
    },
    '1299.HK': {
        'names': ['AIA', 'AIA Group', '友邦', '友邦保险'],
        'sector': 'Financials',
        'market_cap': 'Large'
    },

    # ========== HONG KONG HEALTHCARE ==========
    '6160.HK': {
        'names': ['BeiGene', '百济神州', 'Beigene Ltd'],
        'sector': 'Healthcare',
        'market_cap': 'Large'
    },
    '1801.HK': {
        'names': ['Innovent', 'Innovent Biologics', '信达生物', '信达'],
        'sector': 'Healthcare',
        'market_cap': 'Mid'
    },
    '1093.HK': {
        'names': ['CSPC Pharma', 'CSPC', '石药集团', '石药'],
        'sector': 'Healthcare',
        'market_cap': 'Large'
    },
    '2269.HK': {
        'names': ['WuXi Bio', 'WuXi Biologics', '药明生物', '药明'],
        'sector': 'Healthcare',
        'market_cap': 'Large'
    },
    '1530.HK': {
        'names': ['3SBio', '三生制药', '三生国健'],
        'sector': 'Healthcare',
        'market_cap': 'Small'
    },

    # ========== HONG KONG MATERIALS/MINING ==========
    '3993.HK': {
        'names': ['CMOC', 'China Moly', '洛阳钼业', '洛钼'],
        'sector': 'Materials',
        'market_cap': 'Large'
    },
    '2600.HK': {
        'names': ['Chalco', 'Aluminum Corp', '中国铝业', '中铝'],
        'sector': 'Materials',
        'market_cap': 'Large'
    },
    '0358.HK': {
        'names': ['Jiangxi Copper', 'JCC', '江西铜业', '江铜'],
        'sector': 'Materials',
        'market_cap': 'Mid'
    },

    # ========== HONG KONG ENERGY ==========
    '0857.HK': {
        'names': ['PetroChina', '中国石油', '中石油'],
        'sector': 'Energy',
        'market_cap': 'Large'
    },
    '0386.HK': {
        'names': ['Sinopec', 'China Petroleum', '中国石化', '中石化'],
        'sector': 'Energy',
        'market_cap': 'Large'
    },
    '0883.HK': {
        'names': ['CNOOC', 'China National Offshore Oil', '中海油', '中国海洋石油'],
        'sector': 'Energy',
        'market_cap': 'Large'
    },
    '1088.HK': {
        'names': ['China Shenhua', 'Shenhua', '中国神华', '神华'],
        'sector': 'Energy',
        'market_cap': 'Large'
    },

    # ========== HONG KONG CONSUMER ==========
    '2020.HK': {
        'names': ['ANTA Sports', 'ANTA', '安踏', '安踏体育'],
        'sector': 'Consumer',
        'market_cap': 'Large'
    },
    '2331.HK': {
        'names': ['Li Ning', '李宁'],
        'sector': 'Consumer',
        'market_cap': 'Large'
    },
    '0291.HK': {
        'names': ['China Resources Beer', 'CR Beer', '华润啤酒', '华润'],
        'sector': 'Consumer',
        'market_cap': 'Large'
    },
    '6862.HK': {
        'names': ['Haidilao', '海底捞'],
        'sector': 'Consumer',
        'market_cap': 'Mid'
    },

    # ========== HONG KONG REAL ESTATE ==========
    '0016.HK': {
        'names': ['Sun Hung Kai', 'SHKP', '新鸿基', '新鸿基地产'],
        'sector': 'Real Estate',
        'market_cap': 'Large'
    },
    '1109.HK': {
        'names': ['China Resources Land', 'CR Land', '华润置地'],
        'sector': 'Real Estate',
        'market_cap': 'Large'
    },
    '6869.HK': {
        'names': ['Yuexiu Property', '越秀地产', '越秀'],
        'sector': 'Real Estate',
        'market_cap': 'Small'
    },

    # ========== HONG KONG TELECOM ==========
    '0941.HK': {
        'names': ['China Mobile', '中国移动', '移动'],
        'sector': 'Telecom',
        'market_cap': 'Large'
    },
    '0728.HK': {
        'names': ['China Telecom', '中国电信', '电信'],
        'sector': 'Telecom',
        'market_cap': 'Large'
    },
    '0762.HK': {
        'names': ['China Unicom', '中国联通', '联通'],
        'sector': 'Telecom',
        'market_cap': 'Large'
    },

    # ========== HONG KONG SMALL/MID CAPS ==========
    '0981.HK': {
        'names': ['SMIC', 'Semiconductor Manufacturing', '中芯国际', '中芯'],
        'sector': 'Tech',
        'market_cap': 'Large'
    },
    '1357.HK': {
        'names': ['Meitu', '美图', 'Meitu Inc'],
        'sector': 'Tech',
        'market_cap': 'Small'
    },
    '0772.HK': {
        'names': ['China Literature', '阅文集团', '阅文'],
        'sector': 'Tech',
        'market_cap': 'Mid'
    },
    '1024.HK': {
        'names': ['Kuaishou', '快手', 'Kwai'],
        'sector': 'Tech',
        'market_cap': 'Large'
    },
    '2013.HK': {
        'names': ['Weimob', '微盟', 'Weimob Inc'],
        'sector': 'Tech',
        'market_cap': 'Small'
    },

    # ========== SHANGHAI A-SHARES ==========
    '600519.SS': {
        'names': ['Kweichow Moutai', 'Moutai', '贵州茅台', '茅台'],
        'sector': 'Consumer',
        'market_cap': 'Large'
    },
    '600036.SS': {
        'names': ['China Merchants Bank', 'CMB', '招商银行', '招行'],
        'sector': 'Financials',
        'market_cap': 'Large'
    },
    '601318.SS': {
        'names': ['Ping An Insurance A', '中国平安A', '平安A'],
        'sector': 'Financials',
        'market_cap': 'Large'
    },
    '600276.SS': {
        'names': ['Hengrui Medicine', 'Hengrui', '恒瑞医药', '恒瑞'],
        'sector': 'Healthcare',
        'market_cap': 'Large'
    },
    '600900.SS': {
        'names': ['Yangtze Power', 'CYPC', '长江电力', '长电'],
        'sector': 'Utilities',
        'market_cap': 'Large'
    },
    '601012.SS': {
        'names': ['LONGi Green Energy', 'LONGi', '隆基绿能', '隆基'],
        'sector': 'Energy',
        'market_cap': 'Large'
    },
    '600309.SS': {
        'names': ['Wanhua Chemical', '万华化学', '万华'],
        'sector': 'Materials',
        'market_cap': 'Large'
    },
    '601888.SS': {
        'names': ['China Tourism Group', 'CTG', '中国中免', '中免'],
        'sector': 'Consumer',
        'market_cap': 'Large'
    },

    # ========== SHENZHEN A-SHARES ==========
    '300274.SZ': {
        'names': ['Sungrow', 'Sungrow Power', '阳光电源', '阳光'],
        'sector': 'Industrials',
        'market_cap': 'Large'
    },
    '000858.SZ': {
        'names': ['Wuliangye', '五粮液'],
        'sector': 'Consumer',
        'market_cap': 'Large'
    },
    '000001.SZ': {
        'names': ['Ping An Bank', '平安银行'],
        'sector': 'Financials',
        'market_cap': 'Large'
    },
    '000333.SZ': {
        'names': ['Midea', 'Midea Group', '美的集团', '美的'],
        'sector': 'Consumer',
        'market_cap': 'Large'
    },
    '000651.SZ': {
        'names': ['Gree Electric', 'Gree', '格力电器', '格力'],
        'sector': 'Consumer',
        'market_cap': 'Large'
    },
    '002594.SZ': {
        'names': ['BYD A', 'BYD Company A', '比亚迪A'],
        'sector': 'Auto',
        'market_cap': 'Large'
    },
    '300750.SZ': {
        'names': ['CATL', 'Contemporary Amperex', '宁德时代', '宁德'],
        'sector': 'Auto',
        'market_cap': 'Large'
    },
    '002415.SZ': {
        'names': ['Hikvision', '海康威视', '海康'],
        'sector': 'Tech',
        'market_cap': 'Large'
    },
    '300059.SZ': {
        'names': ['East Money', '东方财富', '东财'],
        'sector': 'Financials',
        'market_cap': 'Large'
    },

    # ========== ETFs ==========
    '2828.HK': {
        'names': ['HS China ETF', 'Hang Seng China', '恒生中国企业'],
        'sector': 'ETF',
        'market_cap': 'ETF'
    },
    '2800.HK': {
        'names': ['Tracker Fund', 'TraHK', '盈富基金'],
        'sector': 'ETF',
        'market_cap': 'ETF'
    },
    '3188.HK': {
        'names': ['China A50 ETF', 'ChinaAMC A50', 'A50ETF'],
        'sector': 'ETF',
        'market_cap': 'ETF'
    },
}

# Build reverse lookup for fast name-to-ticker matching
_NAME_TO_TICKER = {}
_TICKER_ALIASES = {}

for ticker, data in CHINA_STOCKS_DATABASE.items():
    for name in data['names']:
        name_lower = name.lower()
        _NAME_TO_TICKER[name_lower] = ticker

        # Also index without spaces
        name_no_space = name_lower.replace(' ', '')
        _NAME_TO_TICKER[name_no_space] = ticker

    # Add numeric-only alias for A-shares
    if ticker.endswith('.SS') or ticker.endswith('.SZ'):
        numeric = ticker.split('.')[0]
        _TICKER_ALIASES[numeric] = ticker


# ============================================================================
# TICKER RESOLUTION
# ============================================================================

def _fuzzy_match(query: str, candidates: List[str], threshold: float = 0.6) -> List[Tuple[str, float]]:
    """
    Fuzzy match query against candidates.

    Args:
        query: Search query
        candidates: List of candidate strings
        threshold: Minimum similarity score (0-1)

    Returns:
        List of (candidate, score) tuples, sorted by score descending
    """
    matches = []
    query_lower = query.lower()

    for candidate in candidates:
        candidate_lower = candidate.lower()

        # Exact match
        if query_lower == candidate_lower:
            matches.append((candidate, 1.0))
            continue

        # Substring match (high score)
        if query_lower in candidate_lower or candidate_lower in query_lower:
            # Score based on length ratio
            score = min(len(query_lower), len(candidate_lower)) / max(len(query_lower), len(candidate_lower))
            score = 0.7 + (score * 0.3)  # Scale to 0.7-1.0
            matches.append((candidate, score))
            continue

        # Fuzzy match using SequenceMatcher
        score = SequenceMatcher(None, query_lower, candidate_lower).ratio()
        if score >= threshold:
            matches.append((candidate, score))

    # Sort by score descending
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def resolve_china_stock(query: str) -> Optional[str]:
    """
    Convert company name/search query to standardized ticker format.

    Supports:
    - "Tencent" -> "0700.HK"
    - "阿里巴巴" -> "9988.HK"
    - "600519" -> "600519.SS"
    - "贵州茅台" -> "600519.SS"
    - "9999.HK" -> "9999.HK" (already a ticker)
    - "BABA" -> "9988.HK"
    - "SH601127" -> "601127.SS" (Shanghai prefix format)
    - "SZ000001" -> "000001.SZ" (Shenzhen prefix format)
    - "sh600519" -> "600519.SS" (case-insensitive)

    Args:
        query: Company name, partial name, or ticker

    Returns:
        Standardized ticker symbol or None if not found
    """
    if not query:
        return None

    query = query.strip()
    query_upper = query.upper()

    # Check for SH/SZ prefix formats (e.g., SH601127, SZ000001)
    # This is a common Chinese stock format
    sh_match = re.match(r'^SH(\d{6})$', query_upper)
    if sh_match:
        numeric = sh_match.group(1)
        ticker = f"{numeric}.SS"
        logger.info(f"[RESOLVER] Converted SH prefix: {query} -> {ticker}")
        return ticker

    sz_match = re.match(r'^SZ(\d{6})$', query_upper)
    if sz_match:
        numeric = sz_match.group(1)
        ticker = f"{numeric}.SZ"
        logger.info(f"[RESOLVER] Converted SZ prefix: {query} -> {ticker}")
        return ticker

    # Check if already a valid ticker format
    ticker_patterns = [
        r'^\d{4,5}\.HK$',    # HK stocks: 0700.HK, 9988.HK
        r'^\d{6}\.SS$',      # Shanghai: 600519.SS
        r'^\d{6}\.SZ$',      # Shenzhen: 000001.SZ, 300274.SZ
    ]

    for pattern in ticker_patterns:
        if re.match(pattern, query_upper):
            ticker = query_upper
            # Verify it's in our database
            if ticker in CHINA_STOCKS_DATABASE:
                return ticker
            # Return anyway (might be valid but not in our DB)
            return ticker

    # Check for numeric-only input (A-share shorthand)
    if query.isdigit():
        if query in _TICKER_ALIASES:
            return _TICKER_ALIASES[query]

        # Handle 5-digit HK stock codes like 09927, 03750
        # These are HK stocks where the first digit is 0 (leading zero)
        # Convert 09927 -> 9927.HK, 03750 -> 3750.HK
        if len(query) == 5 and query.startswith('0'):
            # Strip leading zero and add .HK suffix
            stock_code = query.lstrip('0')
            # Ensure we have at least 4 digits (pad if needed)
            if len(stock_code) < 4:
                stock_code = stock_code.zfill(4)
            logger.info(f"[RESOLVER] Converted 5-digit HK code: {query} -> {stock_code}.HK")
            return f"{stock_code}.HK"

        # Infer exchange based on prefix (6-digit A-share codes)
        if len(query) == 6:
            if query.startswith('6'):
                return f"{query}.SS"  # Shanghai
            elif query.startswith('0') or query.startswith('3'):
                return f"{query}.SZ"  # Shenzhen

        # 4-digit codes are HK stocks
        if len(query) == 4:
            return f"{query}.HK"  # HK stocks

    # Check exact name match (case-insensitive)
    query_lower = query.lower()
    if query_lower in _NAME_TO_TICKER:
        return _NAME_TO_TICKER[query_lower]

    # CRITICAL FIX: Skip fuzzy matching for queries that look like standard US/International stock tickers
    # This prevents "META" from fuzzy matching to "Meituan" (3690.HK), "TM" to Chinese stocks, etc.
    # US tickers are typically 1-5 uppercase letters, optionally with suffixes like -USD, =X, =F
    us_ticker_patterns = [
        r'^[A-Z]{1,5}$',           # Standard US tickers: AAPL, META, NVDA, TM
        r'^[A-Z]{1,5}-[A-Z]+$',    # Crypto: BTC-USD, ETH-USD
        r'^[A-Z]{2,6}=X$',         # Forex: EURUSD=X, GBPUSD=X
        r'^[A-Z]{1,4}=F$',         # Futures/Commodities: GC=F, CL=F
    ]

    for pattern in us_ticker_patterns:
        if re.match(pattern, query_upper):
            # This looks like a US/International ticker - don't fuzzy match to China stocks
            logger.debug(f"[RESOLVER] Query '{query}' looks like US/Int'l ticker, skipping China fuzzy match")
            return None

    # Try fuzzy matching on all names (only for queries that could be China stock names)
    all_names = list(_NAME_TO_TICKER.keys())
    matches = _fuzzy_match(query, all_names, threshold=0.6)

    if matches:
        best_match = matches[0][0]
        return _NAME_TO_TICKER[best_match]

    # No match found
    logger.warning(f"[RESOLVER] Could not resolve: {query}")
    return None


def search_china_stocks(query: str, limit: int = 10) -> List[Dict]:
    """
    Search China stocks and return suggestions.

    Args:
        query: Search query
        limit: Maximum results to return

    Returns:
        List of matching stocks with details
    """
    if not query or len(query) < 1:
        return []

    results = []
    query_lower = query.lower()

    for ticker, data in CHINA_STOCKS_DATABASE.items():
        # Check if query matches ticker
        if query_lower in ticker.lower():
            results.append({
                'ticker': ticker,
                'name': data['names'][0],
                'aliases': data['names'][1:] if len(data['names']) > 1 else [],
                'sector': data['sector'],
                'market_cap': data['market_cap'],
                'match_type': 'ticker',
                'score': 1.0
            })
            continue

        # Check if query matches any name
        for name in data['names']:
            if query_lower in name.lower():
                results.append({
                    'ticker': ticker,
                    'name': data['names'][0],
                    'aliases': data['names'][1:] if len(data['names']) > 1 else [],
                    'sector': data['sector'],
                    'market_cap': data['market_cap'],
                    'match_type': 'name',
                    'matched_name': name,
                    'score': 0.9
                })
                break

    # If few results, try fuzzy matching
    if len(results) < limit:
        all_names = []
        name_to_ticker = {}
        for ticker, data in CHINA_STOCKS_DATABASE.items():
            for name in data['names']:
                all_names.append(name)
                name_to_ticker[name.lower()] = ticker

        fuzzy_matches = _fuzzy_match(query, all_names, threshold=0.5)

        for name, score in fuzzy_matches:
            ticker = name_to_ticker.get(name.lower())
            if ticker and not any(r['ticker'] == ticker for r in results):
                data = CHINA_STOCKS_DATABASE[ticker]
                results.append({
                    'ticker': ticker,
                    'name': data['names'][0],
                    'aliases': data['names'][1:] if len(data['names']) > 1 else [],
                    'sector': data['sector'],
                    'market_cap': data['market_cap'],
                    'match_type': 'fuzzy',
                    'matched_name': name,
                    'score': score
                })

    # Sort by score and limit
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:limit]


def get_stock_info(ticker: str) -> Optional[Dict]:
    """
    Get detailed info for a ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with stock details or None if not found
    """
    # Resolve ticker if needed
    if ticker not in CHINA_STOCKS_DATABASE:
        resolved = resolve_china_stock(ticker)
        if resolved:
            ticker = resolved
        else:
            return None

    if ticker in CHINA_STOCKS_DATABASE:
        data = CHINA_STOCKS_DATABASE[ticker]
        return {
            'ticker': ticker,
            'name': data['names'][0],
            'aliases': data['names'][1:],
            'sector': data['sector'],
            'market_cap': data['market_cap'],
            'exchange': 'HKEX' if ticker.endswith('.HK') else ('SSE' if ticker.endswith('.SS') else 'SZSE')
        }

    return None


def get_all_tickers_by_sector(sector: str = None) -> List[str]:
    """Get all tickers, optionally filtered by sector."""
    if sector:
        return [t for t, d in CHINA_STOCKS_DATABASE.items() if d['sector'] == sector]
    return list(CHINA_STOCKS_DATABASE.keys())


def get_sectors() -> List[str]:
    """Get list of all sectors."""
    return list(set(d['sector'] for d in CHINA_STOCKS_DATABASE.values()))


# ============================================================================
# AUTO-COMPLETE SUPPORT
# ============================================================================

def autocomplete(query: str, limit: int = 5) -> List[Dict]:
    """
    Get auto-complete suggestions for a query.

    Args:
        query: Partial search query
        limit: Maximum suggestions

    Returns:
        List of suggestion dictionaries with 'value' and 'label'
    """
    if not query or len(query) < 1:
        return []

    results = search_china_stocks(query, limit=limit)

    suggestions = []
    for r in results:
        label = f"{r['ticker']} - {r['name']}"
        if r.get('matched_name') and r['matched_name'] != r['name']:
            label += f" ({r['matched_name']})"

        suggestions.append({
            'value': r['ticker'],
            'label': label,
            'sector': r['sector']
        })

    return suggestions


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CHINA STOCK TICKER RESOLVER - TEST")
    print("=" * 60)

    # Test cases
    test_queries = [
        "Tencent",
        "腾讯",
        "BABA",
        "阿里巴巴",
        "600519",
        "贵州茅台",
        "Moutai",
        "9999.HK",
        "NetEase",
        "比亚迪",
        "BYD",
        "CATL",
        "宁德时代",
        "ping an",
        "平安",
        "sungrow",
        "300274",
        # SH/SZ prefix formats (new support)
        "SH601127",
        "SH600519",
        "sh600036",
        "SZ000001",
        "SZ300750",
        "sz002594",
        "Invalid Company XYZ"
    ]

    print("\n--- Ticker Resolution ---")
    for query in test_queries:
        result = resolve_china_stock(query)
        info = get_stock_info(result) if result else None
        name = info['name'] if info else "NOT FOUND"
        print(f"  '{query}' -> {result} ({name})")

    print("\n--- Search Results ---")
    search_results = search_china_stocks("bank", limit=5)
    print(f"  Search 'bank': {len(search_results)} results")
    for r in search_results:
        print(f"    {r['ticker']}: {r['name']} ({r['sector']})")

    print("\n--- Auto-complete ---")
    suggestions = autocomplete("ten", limit=5)
    print(f"  Autocomplete 'ten': {len(suggestions)} suggestions")
    for s in suggestions:
        print(f"    {s['label']}")

    print("\n--- Sectors ---")
    sectors = get_sectors()
    print(f"  Available sectors: {sectors}")

    print("\n--- Database Stats ---")
    print(f"  Total stocks: {len(CHINA_STOCKS_DATABASE)}")
    for sector in sectors:
        count = len(get_all_tickers_by_sector(sector))
        print(f"    {sector}: {count}")
