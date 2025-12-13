# Quick Start Guide

Get started with the Stock & Crypto Price Prediction system in minutes!

## 📋 Table of Contents
1. [Installation](#installation)
2. [Choose Your Domain](#choose-your-domain)
3. [Fetch Data](#fetch-data)
4. [Data Sources](#data-sources)
5. [Examples](#examples)

---

## 🚀 Installation

### Step 1: Install Dependencies
```bash
cd stock-prediction-model
pip install -r requirements.txt
```

### Step 2: (Optional) Set Up API Keys
For enhanced data access, get free API keys:

**Essential (Recommended):**
- **Alpha Vantage**: https://www.alphavantage.co/support/#api-key
- **FRED**: https://fred.stlouisfed.org/docs/api/api_key.html

**Optional:**
- **Polygon.io**: https://polygon.io/

Set as environment variables:
```bash
export ALPHA_VANTAGE_KEY="your_key_here"
export FRED_API_KEY="your_key_here"
```

Or copy the template:
```bash
cp config/api_keys.template config/api_keys.env
# Edit api_keys.env with your keys
```

**Note**: Yahoo Finance and CoinGecko work without API keys!

---

## 🎯 Choose Your Domain

The system supports multiple domains/sectors. Choose what you want to analyze:

### Available Domains

#### Stocks
- **Technology**: AAPL, MSFT, GOOGL, NVDA, META, TSLA
- **Semiconductors**: NVDA, AMD, INTC, TSM, ASML, QCOM
- **Oil & Energy**: XOM, CVX, COP, SLB, EOG
- **Renewable Energy**: NEE, ENPH, FSLR, PLUG
- **Real Estate**: AMT, PLD, EQIX, PSA, O
- **Banking & Finance**: JPM, BAC, WFC, GS, MS
- **Healthcare**: JNJ, UNH, PFE, ABBV, TMO
- **Consumer Goods**: AMZN, WMT, COST, HD, NKE
- **Automotive**: TSLA, F, GM, TM, RIVN
- **Aerospace & Defense**: BA, LMT, RTX, NOC
- **Entertainment & Media**: DIS, NFLX, CMCSA

#### Cryptocurrencies
- **Major Coins**: BTC-USD, ETH-USD, BNB-USD, SOL-USD
- **DeFi**: UNI-USD, LINK-USD, AAVE-USD
- **Layer 1**: ETH-USD, SOL-USD, ADA-USD, AVAX-USD

#### Presets (Quick Selection)
- **tech_focus**: Top 5 tech stocks
- **energy_focus**: Top 5 energy stocks
- **crypto_major**: Top 4 cryptocurrencies
- **diversified**: Mixed portfolio across sectors
- **semiconductor_focus**: Top chip stocks
- **real_estate_focus**: Top REITs

---

## 📊 Fetch Data

### Method 1: Interactive Selection
```python
from src.utils.asset_selector import AssetSelector

# Interactive mode
selector = AssetSelector()
assets = selector.select_interactive()
```

### Method 2: Use Presets
```python
from src.utils.asset_selector import AssetSelector

selector = AssetSelector()

# Tech focus
assets = selector.get_preset('tech_focus')
print(assets)  # ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META']
```

### Method 3: Select Specific Domains
```python
from src.utils.asset_selector import AssetSelector

selector = AssetSelector()

# Get technology stocks
tech_stocks = selector.get_assets('stocks', 'technology')

# Get oil/energy stocks
energy_stocks = selector.get_assets('stocks', 'oil_energy')

# Get cryptocurrencies
crypto = selector.get_assets('crypto', 'major_coins')

# Combine multiple domains
all_assets = list(set(tech_stocks + energy_stocks + crypto))
selector.print_summary(all_assets)
```

### Method 4: Config-Based Selection
```python
from src.utils.asset_selector import AssetSelector

selector = AssetSelector()

# Select multiple domains at once
config = {
    'category': 'stocks',
    'domains': ['technology', 'semiconductors', 'oil_energy']
}
assets = selector.select_from_config(config)
selector.print_summary(assets)
```

---

## 🌐 Data Sources

### Basic Usage (No API Keys Needed)
```python
from src.data.fetch_data import DataFetcher

# Fetch using Yahoo Finance (default, no API key needed)
fetcher = DataFetcher(
    tickers=['AAPL', 'MSFT', 'BTC-USD'],
    start_date='2020-01-01'
)

data = fetcher.fetch_all()
fetcher.save_data(data, 'data/raw/my_data.csv')
```

### Multi-Source Usage (With API Keys)
```python
from src.data.multi_source_fetcher import MultiSourceDataFetcher
import os

# Initialize with API keys
fetcher = MultiSourceDataFetcher(
    alpha_vantage_key=os.getenv('ALPHA_VANTAGE_KEY'),
    fred_key=os.getenv('FRED_API_KEY')
)

# Auto-select best source
stock_data = fetcher.fetch_auto('AAPL', '2020-01-01', '2024-01-01')
print(f"Fetched from: {stock_data['Source'].iloc[0]}")

# Crypto from CoinGecko (free, no key)
crypto_data = fetcher.fetch_coingecko('bitcoin', days=365)

# Economic data from FRED
fed_rate = fetcher.fetch_fred('DFF', '2020-01-01', '2024-01-01')
inflation = fetcher.fetch_fred('CPIAUCSL', '2020-01-01', '2024-01-01')
```

---

## 💡 Complete Examples

### Example 1: Tech Stocks Analysis
```python
from src.utils.asset_selector import AssetSelector
from src.data.fetch_data import DataFetcher

# Select tech stocks
selector = AssetSelector()
tech_stocks = selector.get_preset('tech_focus')
selector.print_summary(tech_stocks)

# Fetch data
fetcher = DataFetcher(tech_stocks, start_date='2020-01-01')
data = fetcher.fetch_all()

# Save
fetcher.save_data(data, 'data/raw/tech_stocks.csv')

print(f"Fetched {len(data)} rows for {len(tech_stocks)} tech stocks")
```

### Example 2: Oil vs Renewable Energy
```python
from src.utils.asset_selector import AssetSelector
from src.data.fetch_data import DataFetcher

selector = AssetSelector()

# Get both energy sectors
oil = selector.get_assets('stocks', 'oil_energy')
renewable = selector.get_assets('stocks', 'renewable_energy')

all_energy = list(set(oil + renewable))
selector.print_summary(all_energy)

# Fetch data
fetcher = DataFetcher(all_energy, start_date='2019-01-01')
data = fetcher.fetch_all()

# Analyze by sector
data_by_sector = data.groupby('Ticker').size()
print(data_by_sector)
```

### Example 3: Crypto Portfolio
```python
from src.utils.asset_selector import AssetSelector
from src.data.multi_source_fetcher import MultiSourceDataFetcher

selector = AssetSelector()
crypto = selector.get_preset('crypto_major')

# Use CoinGecko for better crypto data (no API key needed!)
fetcher = MultiSourceDataFetcher()

all_data = []
for symbol in crypto:
    coin_symbol = symbol.replace('-USD', '')
    coin_id = fetcher.get_coingecko_id(coin_symbol)

    # Get OHLC data
    ohlc = fetcher.fetch_coingecko(coin_id, days=365)
    ohlc['Symbol'] = symbol
    all_data.append(ohlc)

import pandas as pd
combined = pd.concat(all_data)
print(f"Fetched {len(combined)} rows for {len(crypto)} cryptocurrencies")
```

### Example 4: Semiconductor Sector Deep Dive
```python
from src.utils.asset_selector import AssetSelector
from src.data.fetch_data import DataFetcher

selector = AssetSelector()

# Get semiconductor stocks
chips = selector.get_assets('stocks', 'semiconductors')
print(f"Analyzing {len(chips)} semiconductor stocks: {chips}")

# Fetch data
fetcher = DataFetcher(chips, start_date='2020-01-01')
data = fetcher.fetch_all()

# Analysis by stock
print("\nData summary:")
print(data.groupby('Ticker').agg({
    'Close': ['mean', 'std', 'min', 'max'],
    'Volume': 'mean'
}))
```

### Example 5: Multi-Domain Diversified Portfolio
```python
from src.utils.asset_selector import AssetSelector
from src.data.fetch_data import DataFetcher

selector = AssetSelector()

# Build diversified portfolio
config = {
    'category': 'stocks',
    'domains': ['technology', 'oil_energy', 'healthcare', 'real_estate']
}

portfolio = selector.select_from_config(config)

# Add some crypto
crypto = selector.get_assets('crypto', 'major_coins')
portfolio.extend(crypto[:2])  # Add BTC and ETH

selector.print_summary(portfolio)

# Fetch all data
fetcher = DataFetcher(portfolio, start_date='2021-01-01')
data = fetcher.fetch_all()

print(f"\nFetched {len(data)} rows across {len(portfolio)} assets")
print(f"Date range: {data.index.min()} to {data.index.max()}")
```

### Example 6: Real Estate Investment Trusts (REITs)
```python
from src.utils.asset_selector import AssetSelector
from src.data.fetch_data import DataFetcher

selector = AssetSelector()

# Get real estate stocks
reits = selector.get_assets('stocks', 'real_estate')
print(f"Analyzing {len(reits)} REITs")

# Fetch data
fetcher = DataFetcher(reits, start_date='2020-01-01')
data = fetcher.fetch_all()

fetcher.save_data(data, 'data/raw/reits_data.csv')
```

---

## 🔍 Test Your Setup

Run the test files to ensure everything works:

```bash
# Test asset selector
cd stock-prediction-model
python tests/test_asset_selector.py

# Test data fetching
python tests/test_data_fetch.py
```

---

## 📚 Next Steps

1. **Choose your domain** from the list above
2. **Fetch historical data** using the examples
3. **Feature engineering** - Add technical indicators
4. **Build models** - Train ensemble models
5. **Predict volatility** - Make predictions!

See full documentation in:
- `docs/DATA_SOURCES.md` - Detailed data source guide
- `config/assets.yaml` - All available assets
- `PROJECT_OUTLINE.md` - Complete project roadmap

---

## 🆘 Troubleshooting

**Problem**: "No data retrieved for ticker"
- **Solution**: Check ticker symbol is correct, try different date range

**Problem**: "API key required"
- **Solution**: Get free API key from provider and set environment variable

**Problem**: "Rate limit exceeded"
- **Solution**: Wait a minute, or use different data source

**Problem**: Unicode errors on Windows
- **Solution**: Tests use ASCII-compatible output

---

## 💪 You're Ready!

You now have access to:
- ✅ **12+ stock sectors** covering all major industries
- ✅ **Multiple crypto categories** (major coins, DeFi, Layer 1)
- ✅ **5+ data sources** (Yahoo, CoinGecko, Alpha Vantage, FRED, Polygon)
- ✅ **Pre-configured presets** for quick start
- ✅ **Flexible domain selection** system

Start building your prediction models! 🚀
