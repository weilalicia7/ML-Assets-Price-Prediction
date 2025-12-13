# Data Sources Guide

This project supports multiple reliable public data sources for comprehensive financial data collection.

## Overview of Supported Data Sources

| Source | Data Types | API Key Required | Cost | Rate Limits |
|--------|-----------|------------------|------|-------------|
| **Yahoo Finance** | Stocks, indices, commodities, crypto | No | Free | Moderate |
| **CoinGecko** | Cryptocurrencies | No | Free | 50 calls/min |
| **Alpha Vantage** | Stocks, forex, crypto | Yes (Free) | Free tier available | 5 calls/min (free) |
| **FRED** | Economic indicators | Yes (Free) | Free | Generous |
| **Polygon.io** | Stocks, forex, crypto | Yes (Free tier) | Free tier available | Limited on free tier |

---

## 1. Yahoo Finance (yfinance)

### Overview
- **Primary data source** for stocks, indices, ETFs, and cryptocurrencies
- No API key required
- Reliable historical data
- Real-time delayed quotes

### What You Can Get
- **Stocks**: All major global stocks
- **Indices**: S&P 500, NASDAQ, Dow Jones, etc.
- **ETFs**: All major ETFs
- **Cryptocurrencies**: BTC-USD, ETH-USD, etc.
- **Commodities**: Gold (GC=F), Oil (CL=F), etc.
- **Historical OHLCV data**

### Usage
```python
from src.data.fetch_data import DataFetcher

fetcher = DataFetcher(['AAPL', 'MSFT'], '2020-01-01')
data = fetcher.fetch_all()
```

### Advantages
- ✅ No registration needed
- ✅ Comprehensive coverage
- ✅ Easy to use
- ✅ Good for backtesting

### Limitations
- ⚠️ Rate limits (unofficial API)
- ⚠️ Occasional data gaps
- ⚠️ May change without notice

---

## 2. CoinGecko

### Overview
- **Best free source for cryptocurrency data**
- No API key required
- Comprehensive crypto coverage (10,000+ coins)
- Market data, volume, market cap

### What You Can Get
- **OHLC data** for all major cryptocurrencies
- **Market cap** and **trading volume**
- **Price data** in multiple currencies
- **Historical data** (up to several years)
- **DeFi tokens**, **Layer 1/2 protocols**

### Getting Coin IDs
Common crypto symbols to CoinGecko IDs:
- BTC → `bitcoin`
- ETH → `ethereum`
- BNB → `binancecoin`
- SOL → `solana`
- ADA → `cardano`

### Usage
```python
from src.data.multi_source_fetcher import MultiSourceDataFetcher

fetcher = MultiSourceDataFetcher()
# OHLC data
data = fetcher.fetch_coingecko('bitcoin', days=365)

# Market data with volume and market cap
market_data = fetcher.fetch_coingecko_market_data('ethereum', days=90)
```

### Advantages
- ✅ Completely free
- ✅ No API key needed
- ✅ Excellent crypto coverage
- ✅ Reliable and well-documented

### Limitations
- ⚠️ 50 calls per minute limit
- ⚠️ Crypto only (no stocks)

### API Documentation
https://www.coingecko.com/en/api/documentation

---

## 3. Alpha Vantage

### Overview
- Professional-grade financial data API
- Free tier available (500 calls/day, 5 calls/minute)
- Supports stocks, forex, and cryptocurrencies

### Getting API Key
1. Visit: https://www.alphavantage.co/support/#api-key
2. Fill out simple form (email required)
3. Get free API key instantly
4. Set environment variable: `export ALPHA_VANTAGE_KEY=your_key`

### What You Can Get
- **Daily, weekly, monthly stock data** (20+ years)
- **Intraday data** (1min, 5min, 15min, 30min, 60min)
- **Cryptocurrency data** (daily digital currency prices)
- **Forex data**
- **Technical indicators** (SMA, EMA, RSI, MACD, etc.)
- **Fundamental data** (earnings, balance sheets, etc.)

### Usage
```python
fetcher = MultiSourceDataFetcher(alpha_vantage_key='YOUR_KEY')

# Stock data
stock_data = fetcher.fetch_alpha_vantage_daily('AAPL', outputsize='full')

# Crypto data
crypto_data = fetcher.fetch_alpha_vantage_crypto('BTC', market='USD')
```

### Advantages
- ✅ High-quality data
- ✅ Free tier generous for research
- ✅ Technical indicators included
- ✅ Fundamental data available

### Limitations
- ⚠️ 5 calls/minute on free tier
- ⚠️ 500 calls/day limit
- ⚠️ Requires API key

### API Documentation
https://www.alphavantage.co/documentation/

---

## 4. FRED (Federal Reserve Economic Data)

### Overview
- **Economic indicators** from US Federal Reserve
- 800,000+ economic time series
- Completely free with API key
- Excellent for macroeconomic analysis

### Getting API Key
1. Visit: https://fred.stlouisfed.org/docs/api/api_key.html
2. Create free account
3. Request API key (instant approval)
4. Set environment variable: `export FRED_API_KEY=your_key`

### What You Can Get
**Economic Indicators:**
- `DFF` - Federal Funds Rate
- `DGS10` - 10-Year Treasury Rate
- `UNRATE` - Unemployment Rate
- `CPIAUCSL` - Consumer Price Index (Inflation)
- `GDP` - Gross Domestic Product
- `FEDFUNDS` - Federal Funds Effective Rate
- `T10Y2Y` - 10-Year minus 2-Year Treasury Spread
- `VIXCLS` - VIX Volatility Index

### Usage
```python
fetcher = MultiSourceDataFetcher(fred_key='YOUR_KEY')

# Get unemployment rate
unemployment = fetcher.fetch_fred('UNRATE', '2020-01-01', '2024-01-01')

# Get inflation data
inflation = fetcher.fetch_fred('CPIAUCSL', '2020-01-01', '2024-01-01')
```

### Use Cases
- **Feature engineering**: Add macroeconomic features to stock predictions
- **Correlation analysis**: How stocks react to Fed rate changes
- **Market regime detection**: Identify bull/bear markets using economic indicators

### Advantages
- ✅ Authoritative source (Federal Reserve)
- ✅ Free with generous limits
- ✅ Extensive historical data
- ✅ Well-documented

### API Documentation
https://fred.stlouisfed.org/docs/api/fred/

---

## 5. Polygon.io

### Overview
- Real-time and historical market data
- Free tier available (limited)
- WebSocket support for real-time data

### Getting API Key
1. Visit: https://polygon.io/
2. Sign up for free tier
3. Get API key
4. Set environment variable: `export POLYGON_API_KEY=your_key`

### What You Can Get (Free Tier)
- **Historical stock data**
- **Aggregates** (bars): minute, hour, day, week, month
- **Previous close**
- **Daily open/close**

### Usage
```python
fetcher = MultiSourceDataFetcher(polygon_key='YOUR_KEY')

data = fetcher.fetch_polygon_stocks('AAPL', '2024-01-01', '2024-02-01')
```

### Advantages
- ✅ High-quality data
- ✅ Real-time support (paid tiers)
- ✅ Good documentation

### Limitations (Free Tier)
- ⚠️ 5 API calls per minute
- ⚠️ Delayed data (15 minutes)
- ⚠️ Limited features

### API Documentation
https://polygon.io/docs/stocks/getting-started

---

## Auto-Selection Feature

The `MultiSourceDataFetcher` includes an auto-selection feature that tries multiple sources automatically:

```python
fetcher = MultiSourceDataFetcher(
    alpha_vantage_key='YOUR_AV_KEY',
    fred_key='YOUR_FRED_KEY'
)

# Automatically tries Yahoo -> Alpha Vantage -> CoinGecko
data = fetcher.fetch_auto('AAPL', '2024-01-01', '2024-02-01')
print(f"Data fetched from: {data['Source'].iloc[0]}")
```

**Logic:**
- For **stocks**: Yahoo → Alpha Vantage → Polygon
- For **crypto**: CoinGecko → Yahoo → Alpha Vantage
- Falls back automatically if one source fails

---

## Recommended Setup

### For Basic Usage (No API Keys)
```python
# Uses Yahoo Finance + CoinGecko (both free, no keys)
from src.data.fetch_data import DataFetcher

fetcher = DataFetcher(['AAPL', 'BTC-USD'], '2020-01-01')
data = fetcher.fetch_all()
```

### For Advanced Usage (With API Keys)
```python
# Set environment variables first:
# export ALPHA_VANTAGE_KEY=your_av_key
# export FRED_API_KEY=your_fred_key

from src.data.multi_source_fetcher import MultiSourceDataFetcher

fetcher = MultiSourceDataFetcher(
    alpha_vantage_key=os.getenv('ALPHA_VANTAGE_KEY'),
    fred_key=os.getenv('FRED_API_KEY')
)

# Fetch stock data
stock_data = fetcher.fetch_auto('AAPL', '2020-01-01', '2024-01-01')

# Fetch crypto with CoinGecko
crypto_data = fetcher.fetch_coingecko('bitcoin', days=365)

# Fetch economic indicators
fed_rate = fetcher.fetch_fred('DFF', '2020-01-01', '2024-01-01')
```

---

## Setting Up API Keys

### Method 1: Environment Variables (Recommended)
```bash
# Add to your ~/.bashrc or ~/.zshrc
export ALPHA_VANTAGE_KEY="your_alpha_vantage_key"
export FRED_API_KEY="your_fred_key"
export POLYGON_API_KEY="your_polygon_key"
```

### Method 2: .env File
Create `.env` file in project root:
```
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
FRED_API_KEY=your_fred_key
POLYGON_API_KEY=your_polygon_key
```

Then load with python-dotenv:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Method 3: Pass Directly
```python
fetcher = MultiSourceDataFetcher(
    alpha_vantage_key='YOUR_KEY_HERE',
    fred_key='YOUR_KEY_HERE'
)
```

---

## Data Quality Comparison

| Metric | Yahoo | CoinGecko | Alpha Vantage | FRED | Polygon |
|--------|-------|-----------|---------------|------|---------|
| **Reliability** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Coverage** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Free Tier** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## Best Practices

1. **Start with free sources** (Yahoo, CoinGecko) for prototyping
2. **Get free API keys** (Alpha Vantage, FRED) for production
3. **Implement caching** to avoid hitting rate limits
4. **Use auto-selection** for robustness
5. **Combine sources** for comprehensive data:
   - Stock prices: Yahoo/Alpha Vantage
   - Crypto prices: CoinGecko
   - Economic indicators: FRED
6. **Respect rate limits** to avoid being blocked

---

## Next Steps

1. Get free API keys from Alpha Vantage and FRED
2. Set up environment variables
3. Test data fetching with `multi_source_fetcher.py`
4. Build your prediction models with comprehensive data!
