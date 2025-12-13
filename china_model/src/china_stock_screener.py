"""
China Stock Universe Screener

Scalable system to filter and analyze ALL China market instruments.

Architecture:
    Stage 1: Liquidity Filter (>$500M daily volume)
    Stage 2: Quick Robustness Pre-screen (1 seed, reduced features)
    Stage 3: Deep Analysis (5 seeds, full features - promising only)
    Stage 4: Portfolio Optimization (weight by edge strength)

Supported Markets:
    - Hong Kong (HK Mainboard)
    - Shanghai A-shares (SSE)
    - Shenzhen A-shares (SZSE)
    - China ETFs
    - China Futures (commodities)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CHINA MARKET UNIVERSE DEFINITIONS
# ============================================================================

# Hong Kong Main Board - Large Caps (proven alpha)
HK_LARGE_CAPS = [
    # Tech/Internet
    '0700.HK',  # Tencent
    '9988.HK',  # Alibaba
    '3690.HK',  # Meituan
    '9999.HK',  # NetEase
    '9618.HK',  # JD.com
    '1024.HK',  # Kuaishou
    '9888.HK',  # Baidu
    '0981.HK',  # SMIC
    '0285.HK',  # BYD Electronic
    '2382.HK',  # Sunny Optical

    # Financial
    '0939.HK',  # CCB
    '1398.HK',  # ICBC
    '3988.HK',  # Bank of China
    '2318.HK',  # Ping An
    '2628.HK',  # China Life
    '1299.HK',  # AIA
    '0388.HK',  # HKEX
    '2319.HK',  # Mengniu
    '3968.HK',  # China Merchants Bank
    '1658.HK',  # Postal Savings Bank

    # Real Estate
    '1109.HK',  # China Resources Land
    '0960.HK',  # Longfor
    '0688.HK',  # China Overseas Land
    '2007.HK',  # Country Garden
    '3333.HK',  # Evergrande
    '1113.HK',  # CK Asset

    # Consumer
    '0027.HK',  # Galaxy Entertainment
    '1928.HK',  # Sands China
    '0291.HK',  # China Resources Beer
    '2020.HK',  # Anta Sports
    '1876.HK',  # Budweiser APAC
    '6862.HK',  # Haidilao
    '9633.HK',  # Nongfu Spring

    # Healthcare/Pharma
    '2269.HK',  # WuXi Biologics
    '1177.HK',  # Sino Biopharm
    '1093.HK',  # CSPC Pharma
    '2359.HK',  # WuXi AppTec
    '6160.HK',  # BeiGene

    # Energy/Materials
    '0883.HK',  # CNOOC
    '0857.HK',  # PetroChina
    '0386.HK',  # Sinopec
    '3993.HK',  # CMOC Group
    '1088.HK',  # China Shenhua
    '2600.HK',  # Aluminum Corp

    # Industrials
    '1211.HK',  # BYD
    '2333.HK',  # Great Wall Motor
    '0175.HK',  # Geely Auto
    '0914.HK',  # Anhui Conch
    '0669.HK',  # Techtronic
    '2313.HK',  # Shenzhou International

    # Telecom/Utilities
    '0941.HK',  # China Mobile
    '0728.HK',  # China Telecom
    '0762.HK',  # China Unicom
    '0002.HK',  # CLP Holdings
    '0003.HK',  # HK & China Gas
]

# Hong Kong Mid-Caps - Expanding alpha opportunity
HK_MID_CAPS = [
    # Tech/Software
    '0268.HK',  # Kingdee International
    '0992.HK',  # Lenovo Group
    '1810.HK',  # Xiaomi
    '6618.HK',  # JD Health
    '9698.HK',  # GDS Holdings
    '0772.HK',  # China Literature
    '9626.HK',  # Bilibili
    '1347.HK',  # Hua Hong Semi
    '0522.HK',  # ASM Pacific

    # Consumer/Retail
    '0151.HK',  # Want Want China
    '0220.HK',  # Uni-President China
    '1044.HK',  # Hengan International
    '0168.HK',  # Tsingtao Brewery
    '0322.HK',  # Tingyi
    '2331.HK',  # Li Ning
    '6969.HK',  # Smoore International
    '1368.HK',  # Xtep International
    '2018.HK',  # AAC Technologies

    # Healthcare/Biotech
    '1801.HK',  # Innovent Biologics
    '2696.HK',  # Shandong Weigao
    '2186.HK',  # Luye Pharma
    '1548.HK',  # Genscript Biotech
    '6185.HK',  # CanSino Biologics
    '9969.HK',  # Imeik Technology

    # Industrials/Manufacturing
    '2689.HK',  # Nine Dragons Paper
    '1072.HK',  # Dongfang Electric
    '1766.HK',  # CRRC
    '0916.HK',  # China Longyuan Power
    '1816.HK',  # CGN Power
    '6823.HK',  # HK Electric
    '2688.HK',  # ENN Energy
    '0384.HK',  # China Gas Holdings
    '0135.HK',  # Kunlun Energy

    # Materials/Mining
    '1171.HK',  # Yanzhou Coal
    '3323.HK',  # CNBM
    '0358.HK',  # Jiangxi Copper
    '0489.HK',  # Dongfeng Motor
    '1800.HK',  # China Communications Construction
    '1186.HK',  # China Railway Construction

    # Financial Services
    '6060.HK',  # ZhongAn Online
    '1336.HK',  # New China Life
    '6066.HK',  # CSC Financial
    '6886.HK',  # HTSC
    '3908.HK',  # China International Capital Corp
    '6881.HK',  # China Galaxy Securities

    # Property/REITs
    '1997.HK',  # Wharf REIC
    '0823.HK',  # Link REIT
    '2778.HK',  # Champion REIT
    '0778.HK',  # Fortune REIT
    '0435.HK',  # Sunlight REIT
]

# Hong Kong Small-Caps - High alpha potential (market cap $1-5B)
HK_SMALL_CAPS = [
    # Tech/Software
    '1357.HK',  # Meitu
    '0302.HK',  # Wing Hang Bank
    '1833.HK',  # Ping An Healthcare
    '6690.HK',  # Haier Smart Home H
    '1378.HK',  # China Hongqiao

    # Consumer/Retail
    '1083.HK',  # Towngas China
    '0010.HK',  # Hang Lung Group
    '0017.HK',  # New World Development
    '0083.HK',  # Sino Land
    '0012.HK',  # Henderson Land
    '0101.HK',  # Hang Lung Properties
    '2688.HK',  # ENN Energy
    '2688.HK',  # Hengan International

    # Healthcare/Biotech Small Caps
    '1530.HK',  # 3SBio
    '1877.HK',  # Shanghai Junshi Biosciences
    '2126.HK',  # JW Therapeutics
    '9995.HK',  # RemeGen
    '6978.HK',  # Imeik Technology
    '2171.HK',  # Cansino Biologics H
    '6160.HK',  # BeiGene (already included but key)

    # Industrials/Manufacturing Small Caps
    '1458.HK',  # Zhou Hei Ya
    '0861.HK',  # Digital China
    '6869.HK',  # Yuexiu Property
    '1966.HK',  # China SCE Group
    '1638.HK',  # Kaisa Group
    '3380.HK',  # Logan Group
    '1813.HK',  # KWG Group
    '0123.HK',  # Guangzhou Investment

    # Materials/Mining Small Caps
    '1818.HK',  # Zhaojin Mining
    '1899.HK',  # Xingda International
    '0696.HK',  # Travelsky Technology
    '1888.HK',  # China Nonferrous Mining

    # Financial Services Small Caps
    '1359.HK',  # China Cinda
    '6030.HK',  # CITIC Securities H
    '3328.HK',  # Bank of Communications H
    '1916.HK',  # Jiangxi Bank
    '1963.HK',  # Bank of Chongqing

    # Energy/Utilities Small Caps
    '1071.HK',  # Huadian Power
    '2380.HK',  # China Power
    '0836.HK',  # China Resources Power
    '1193.HK',  # China Resources Gas
    '0270.HK',  # Guangdong Investment
]

# Combined HK Mainboard (large + mid + small caps)
HK_MAINBOARD = HK_LARGE_CAPS + HK_MID_CAPS + HK_SMALL_CAPS

# Shanghai A-shares (SSE) - Major stocks
SHANGHAI_A = [
    # Consumer/Liquor
    '600519.SS',  # Kweichow Moutai
    '600809.SS',  # Shanxi Fenjiu
    '603369.SS',  # Jinzai Food

    # Financial
    '601318.SS',  # Ping An Insurance
    '600036.SS',  # China Merchants Bank
    '601166.SS',  # Industrial Bank
    '600000.SS',  # Shanghai Pudong Bank
    '601328.SS',  # Bank of Communications
    '601398.SS',  # ICBC
    '601288.SS',  # Agricultural Bank
    '601939.SS',  # CCB
    '601988.SS',  # Bank of China
    '601628.SS',  # China Life
    '601601.SS',  # China Pacific Insurance

    # Energy
    '601857.SS',  # PetroChina
    '600028.SS',  # Sinopec
    '601088.SS',  # China Shenhua
    '600900.SS',  # Yangtze Power

    # Healthcare
    '600276.SS',  # Hengrui Medicine
    '603259.SS',  # WuXi AppTec
    '600196.SS',  # Fosun Pharma

    # Industrials
    '600031.SS',  # SANY Heavy
    '601012.SS',  # LONGi Green Energy
    '600690.SS',  # Haier Smart Home
    '600887.SS',  # Yili Industrial

    # Tech
    '688981.SS',  # SMIC (STAR)
    '688111.SS',  # Kingsoft Cloud (STAR)

    # Tourism/Consumer
    '601888.SS',  # China Tourism Group
    '600009.SS',  # Shanghai Airport

    # Materials
    '600309.SS',  # Wanhua Chemical
    '600585.SS',  # Anhui Conch

    # Auto
    '600104.SS',  # SAIC Motor
    '601238.SS',  # GAC Group
]

# Shenzhen A-shares (SZSE) - Major stocks
SHENZHEN_A = [
    # Consumer
    '000858.SZ',  # Wuliangye
    '000568.SZ',  # Luzhou Laojiao
    '002304.SZ',  # Yanghe Brewery
    '000651.SZ',  # Gree Electric
    '000333.SZ',  # Midea Group

    # Financial
    '000001.SZ',  # Ping An Bank
    '000002.SZ',  # Vanke
    '002142.SZ',  # Bank of Ningbo

    # Tech/Electronics
    '002415.SZ',  # Hikvision
    '000725.SZ',  # BOE Technology
    '002475.SZ',  # Luxshare Precision
    '300750.SZ',  # CATL
    '002594.SZ',  # BYD
    '300059.SZ',  # East Money

    # Healthcare
    '300760.SZ',  # Mindray Medical
    '300122.SZ',  # Zhifei Biological
    '300015.SZ',  # Aier Eye Hospital

    # New Energy
    '002129.SZ',  # TCL Technology
    '300274.SZ',  # Sungrow Power
    '002459.SZ',  # Tianhe Tech

    # Materials
    '002460.SZ',  # Ganfeng Lithium
    '002466.SZ',  # Tianqi Lithium
]

# Hong Kong ETFs
HK_ETFS = [
    '2800.HK',  # Tracker Fund of HK
    '2828.HK',  # Hang Seng China ETF
    '3188.HK',  # China A50 ETF
    '2822.HK',  # CSOP A50 ETF
    '3033.HK',  # CSOP Hang Seng Tech ETF
    '3067.HK',  # iShares Hang Seng Tech ETF
    '2801.HK',  # iShares Core CSI 300 ETF
    '3040.HK',  # Samsung CSI China Dragon ETF
]

# China Futures (via Yahoo - using related instruments)
CHINA_FUTURES_PROXIES = [
    'GC=F',   # Gold (China largest consumer)
    'SI=F',   # Silver
    'HG=F',   # Copper (China demand driver)
    'CL=F',   # Crude Oil
    'NG=F',   # Natural Gas
    'ZS=F',   # Soybeans (China importer)
    'ZC=F',   # Corn
    'ZW=F',   # Wheat
    'ALI=F',  # Aluminum
]

# Full Universe
FULL_UNIVERSE = {
    'HK_Mainboard': HK_MAINBOARD,
    'Shanghai_A': SHANGHAI_A,
    'Shenzhen_A': SHENZHEN_A,
    'HK_ETFs': HK_ETFS,
    'Futures_Proxies': CHINA_FUTURES_PROXIES,
}


# ============================================================================
# LIQUIDITY ANALYSIS
# ============================================================================

class LiquidityAnalyzer:
    """Analyze and filter instruments by liquidity"""

    def __init__(self, min_dollar_volume=500_000_000, lookback_days=60):
        """
        Args:
            min_dollar_volume: Minimum avg daily dollar volume (default $500M)
            lookback_days: Days to calculate average volume
        """
        self.min_dollar_volume = min_dollar_volume
        self.lookback_days = lookback_days
        self.cache = {}

    def get_liquidity_metrics(self, symbol):
        """Get liquidity metrics for a single symbol"""
        if symbol in self.cache:
            return self.cache[symbol]

        try:
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days + 30)

            df = ticker.history(start=start_date.strftime('%Y-%m-%d'),
                               end=end_date.strftime('%Y-%m-%d'))

            if len(df) < 20:
                return None

            # Calculate metrics
            df['dollar_volume'] = df['Close'] * df['Volume']

            metrics = {
                'symbol': symbol,
                'avg_dollar_volume': df['dollar_volume'].tail(self.lookback_days).mean(),
                'avg_volume': df['Volume'].tail(self.lookback_days).mean(),
                'avg_price': df['Close'].tail(self.lookback_days).mean(),
                'volume_stability': df['Volume'].tail(self.lookback_days).std() / df['Volume'].tail(self.lookback_days).mean(),
                'spread_proxy': ((df['High'] - df['Low']) / df['Close']).tail(self.lookback_days).mean(),
                'trading_days': len(df),
            }

            self.cache[symbol] = metrics
            return metrics

        except Exception as e:
            return None

    def filter_by_liquidity(self, symbols, verbose=True):
        """Filter symbols by minimum liquidity threshold"""
        liquid_symbols = []
        all_metrics = []

        if verbose:
            print(f"Screening {len(symbols)} symbols for liquidity (min ${self.min_dollar_volume/1e6:.0f}M)...")

        for i, symbol in enumerate(symbols):
            metrics = self.get_liquidity_metrics(symbol)

            if metrics and metrics['avg_dollar_volume'] >= self.min_dollar_volume:
                liquid_symbols.append(symbol)
                all_metrics.append(metrics)

            if verbose and (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(symbols)} - Found {len(liquid_symbols)} liquid")

        if verbose:
            print(f"  Final: {len(liquid_symbols)}/{len(symbols)} passed liquidity filter")

        return liquid_symbols, all_metrics

    def parallel_filter(self, symbols, max_workers=10, verbose=True):
        """Parallel liquidity filtering for speed"""
        liquid_symbols = []
        all_metrics = []

        if verbose:
            print(f"Parallel screening {len(symbols)} symbols (workers={max_workers})...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(self.get_liquidity_metrics, s): s
                               for s in symbols}

            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1

                try:
                    metrics = future.result()
                    if metrics and metrics['avg_dollar_volume'] >= self.min_dollar_volume:
                        liquid_symbols.append(symbol)
                        all_metrics.append(metrics)
                except Exception as e:
                    pass

                if verbose and completed % 50 == 0:
                    print(f"  Processed {completed}/{len(symbols)} - Found {len(liquid_symbols)} liquid")

        if verbose:
            print(f"  Final: {len(liquid_symbols)}/{len(symbols)} passed liquidity filter")

        # Sort by liquidity
        all_metrics.sort(key=lambda x: x['avg_dollar_volume'], reverse=True)
        liquid_symbols = [m['symbol'] for m in all_metrics]

        return liquid_symbols, all_metrics


# ============================================================================
# MARKET TYPE DETECTION
# ============================================================================

def detect_market_type(symbol):
    """Detect which market type a symbol belongs to"""
    if symbol.endswith('.HK'):
        if symbol in HK_ETFS:
            return 'HK_ETF'
        return 'HK_Stock'
    elif symbol.endswith('.SS'):
        return 'Shanghai_A'
    elif symbol.endswith('.SZ'):
        return 'Shenzhen_A'
    elif '=' in symbol:
        return 'Futures'
    else:
        return 'Unknown'


def get_market_config(market_type):
    """Get market-specific configuration"""
    configs = {
        'HK_Stock': {
            'profit_gate': 0.004,
            'confidence_threshold': 0.52,
            'min_trades': 10,
        },
        'HK_ETF': {
            'profit_gate': 0.002,
            'confidence_threshold': 0.51,
            'min_trades': 15,
        },
        'Shanghai_A': {
            'profit_gate': 0.005,
            'confidence_threshold': 0.53,
            'min_trades': 10,
        },
        'Shenzhen_A': {
            'profit_gate': 0.005,
            'confidence_threshold': 0.53,
            'min_trades': 10,
        },
        'Futures': {
            'profit_gate': 0.003,
            'confidence_threshold': 0.52,
            'min_trades': 20,
        },
        'Unknown': {
            'profit_gate': 0.004,
            'confidence_threshold': 0.52,
            'min_trades': 10,
        },
    }
    return configs.get(market_type, configs['Unknown'])


# ============================================================================
# UNIVERSE BUILDER
# ============================================================================

class ChinaUniverseBuilder:
    """Build and manage the China stock universe"""

    def __init__(self, min_liquidity=500_000_000):
        self.min_liquidity = min_liquidity
        self.liquidity_analyzer = LiquidityAnalyzer(min_dollar_volume=min_liquidity)
        self.universe = {}
        self.metrics = {}

    def build_full_universe(self, verbose=True):
        """Build the complete liquid universe"""
        if verbose:
            print("=" * 70)
            print("BUILDING CHINA MARKET UNIVERSE")
            print("=" * 70)
            print(f"Minimum liquidity threshold: ${self.min_liquidity/1e9:.1f}B daily volume")
            print()

        all_liquid = []
        all_metrics = []

        for market, symbols in FULL_UNIVERSE.items():
            if verbose:
                print(f"\n{market}:")
                print("-" * 40)

            liquid, metrics = self.liquidity_analyzer.filter_by_liquidity(symbols, verbose=verbose)

            self.universe[market] = liquid
            self.metrics[market] = metrics
            all_liquid.extend(liquid)
            all_metrics.extend(metrics)

        # Summary
        if verbose:
            print("\n" + "=" * 70)
            print("UNIVERSE SUMMARY")
            print("=" * 70)
            total_screened = sum(len(v) for v in FULL_UNIVERSE.values())
            print(f"Total screened: {total_screened}")
            print(f"Passed liquidity: {len(all_liquid)}")
            print(f"Pass rate: {len(all_liquid)/total_screened*100:.1f}%")
            print()
            print("By Market:")
            for market, liquid in self.universe.items():
                original = len(FULL_UNIVERSE[market])
                print(f"  {market}: {len(liquid)}/{original}")

        return all_liquid, all_metrics

    def get_top_by_liquidity(self, n=50):
        """Get top N most liquid instruments across all markets"""
        all_metrics = []
        for market_metrics in self.metrics.values():
            all_metrics.extend(market_metrics)

        all_metrics.sort(key=lambda x: x['avg_dollar_volume'], reverse=True)
        return all_metrics[:n]

    def save_universe(self, filepath):
        """Save universe to JSON file"""
        output = {
            'generated': datetime.now().isoformat(),
            'min_liquidity': self.min_liquidity,
            'universe': self.universe,
            'metrics': {
                market: [
                    {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                     for k, v in m.items()}
                    for m in metrics
                ]
                for market, metrics in self.metrics.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Universe saved to: {filepath}")

    def load_universe(self, filepath):
        """Load universe from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.universe = data['universe']
        self.metrics = data['metrics']
        self.min_liquidity = data['min_liquidity']

        print(f"Universe loaded from: {filepath}")
        return self.universe


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Build and save the China market universe"""
    print("=" * 70)
    print("CHINA MARKET UNIVERSE BUILDER")
    print("=" * 70)
    print()

    # Build universe with $500M minimum daily volume
    builder = ChinaUniverseBuilder(min_liquidity=500_000_000)
    liquid_symbols, metrics = builder.build_full_universe(verbose=True)

    # Show top 20 most liquid
    print("\n" + "=" * 70)
    print("TOP 20 MOST LIQUID INSTRUMENTS")
    print("=" * 70)
    print(f"{'Rank':<5} {'Symbol':<12} {'Daily Volume':>15} {'Spread':>10}")
    print("-" * 50)

    top_20 = builder.get_top_by_liquidity(20)
    for i, m in enumerate(top_20, 1):
        vol_str = f"${m['avg_dollar_volume']/1e9:.2f}B"
        spread_str = f"{m['spread_proxy']*100:.2f}%"
        print(f"{i:<5} {m['symbol']:<12} {vol_str:>15} {spread_str:>10}")

    # Save universe
    save_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'china_universe.json')
    builder.save_universe(save_path)

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Run tiered screening on liquid universe")
    print("2. Quick screen (1 seed) → identify promising candidates")
    print("3. Deep analysis (5 seeds) → confirm robust performers")
    print("4. Build portfolio from robust performers")
    print()

if __name__ == '__main__':
    main()
