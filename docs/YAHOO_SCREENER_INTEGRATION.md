# Yahoo Finance Screener Integration

## Overview

The Top-10 Picks feature now uses **real-time Yahoo Finance screeners** instead of a hardcoded database. This enables dynamic discovery of trending, active, and high-performing stocks.

## Problem Solved

**Before (Bug):**
- Top-10 picks only selected from ~100 pre-configured tickers in `COMPANY_DATABASE`
- Missed trending stocks, new IPOs, and market movers
- Static universe that didn't adapt to market conditions

**After (Fix):**
- Real-time ticker discovery using Yahoo Finance screeners
- Dynamic discovery of day gainers, most active, undervalued stocks
- Automatic fallback to database when screeners fail
- Faster cache refresh (30 min vs 3 hours)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TOP-10 PICKS FLOW (NEW)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Request: /api/top-picks?regime=Stock                           │
│                    │                                                │
│                    ▼                                                │
│  2. ReliabilityManager: Should use screeners? ────┐                │
│                    │                               │                │
│           ┌───────┴───────┐                       │                │
│           │               │                       │                │
│           ▼               ▼                       │                │
│    [SCREENERS]     [DATABASE]                     │                │
│         │               │                         │                │
│         ▼               ▼                         │                │
│  Yahoo Finance    COMPANY_DATABASE                │                │
│  - day_gainers                                    │                │
│  - most_actives                                   │                │
│  - HK/China custom                                │                │
│         │               │                         │                │
│         └───────┬───────┘                         │                │
│                 ▼                                 │                │
│  3. ML Prediction Pipeline (unchanged)            │                │
│                 │                                 │                │
│                 ▼                                 │                │
│  4. Hybrid Ranking (profit_score)                │                │
│                 │                                 │                │
│                 ▼                                 │                │
│  5. Return Top 10 Buys + Top 10 Sells            │                │
│     + ticker_source: 'screener' or 'database'    │                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Files Created/Modified

### New Files

| File | Description |
|------|-------------|
| `src/screeners/yahoo_screener_discovery.py` | Main screener discovery module |
| `src/screeners/__init__.py` | Module exports |

### Modified Files

| File | Changes |
|------|---------|
| `webapp.py` | Added screener imports, updated `/api/top-picks`, added `/api/screener-status` |

---

## Key Classes

### YahooScreenerDiscoverer

Discovers tickers using Yahoo Finance built-in screeners.

```python
from src.screeners import YahooScreenerDiscoverer

discoverer = YahooScreenerDiscoverer()

# Get US day gainers
gainers = discoverer.discover_tickers('us_gainers', count=20)
# Returns: ['TMC', 'EXK', 'AG', 'CLSK', ...]

# Get most active stocks
active = discoverer.discover_tickers('us_active', count=20)
# Returns: ['NVDA', 'HBI', 'INTC', 'BITF', ...]

# Get HK stocks
hk = discoverer.discover_tickers('hk_active', count=20)
# Returns: ['0700.HK', '9988.HK', '3690.HK', ...]

# Get crypto
crypto = discoverer.discover_tickers('crypto', count=20)
# Returns: ['BTC-USD', 'ETH-USD', 'SOL-USD', ...]
```

### RegimeScreenerStrategy

Selects appropriate screeners based on market regime.

```python
from src.screeners import RegimeScreenerStrategy

strategy = RegimeScreenerStrategy()

# Get tickers for specific regime
tickers, source = strategy.get_tickers_for_regime('Stock', count=25)
# Returns: (['TMC', 'NVDA', 'INTC', ...], 'us_screener')

tickers, source = strategy.get_tickers_for_regime('China', count=20)
# Returns: (['0700.HK', '9988.HK', ...], 'china_screener')
```

### ReliabilityManager

Tracks screener performance and decides when to use fallback.

```python
from src.screeners import ReliabilityManager

mgr = ReliabilityManager()

# Check if screeners are reliable for a regime
if mgr.should_use_screeners('Stock'):
    # Use screeners
    pass
else:
    # Use database fallback
    pass

# Track performance
mgr.track_performance('Stock', success=True)

# Get success rate
rate = mgr.get_success_rate('Stock')  # e.g., 0.95
```

---

## Supported Screeners

### Built-in Yahoo Finance Screeners

| Screener ID | Description | Use Case |
|-------------|-------------|----------|
| `us_gainers` | Day gainers (>3% up) | Trending stocks |
| `us_losers` | Day losers (>2.5% down) | Oversold stocks |
| `us_active` | Most active by volume | High liquidity |
| `us_undervalued` | Low P/E, high growth | Value stocks |
| `us_tech_growth` | High-growth tech | Growth stocks |

### Custom Screeners

| Screener ID | Description | Use Case |
|-------------|-------------|----------|
| `hk_active` | HK stocks by volume | China/HK exposure |
| `china_active` | HK + A-shares combined | Full China coverage |
| `crypto` | Top cryptocurrencies | Crypto exposure |

---

## Cache Configuration

Dynamic cache TTL based on regime volatility:

| Regime | Cache TTL | Rationale |
|--------|-----------|-----------|
| Cryptocurrency | 10 min | Very volatile |
| Stock | 30 min | Real-time screeners |
| China | 30 min | Real-time screeners |
| Forex | 30 min | Active market |
| Commodity | 1 hour | Less volatile |
| all | 30 min | Mixed |

---

## API Endpoints

### GET /api/top-picks

Returns top 10 BUY and SELL signals using real-time screeners.

**Parameters:**
- `regime`: Stock, Cryptocurrency, China, Commodity, Forex, all
- `force_refresh`: true/false (bypass cache)
- `use_screeners`: true/false (force database fallback)

**Response:**
```json
{
  "status": "success",
  "regime": "Stock",
  "top_buys": [...],
  "top_sells": [...],
  "total_analyzed": 25,
  "ticker_source": "us_screener",
  "screener_available": true,
  "cache_ttl_seconds": 1800,
  "from_cache": false
}
```

### GET /api/screener-status

Check screener health and reliability metrics.

**Response:**
```json
{
  "screener_available": true,
  "reliability_metrics": {
    "Stock": {"use_screeners": true, "success_rate": 0.95},
    "Cryptocurrency": {"use_screeners": true, "success_rate": 1.0},
    ...
  },
  "connectivity_test": {
    "status": "ok",
    "sample_tickers": ["NVDA", "HBI", "INTC"]
  }
}
```

---

## Fallback Behavior

The system automatically falls back to database when:

1. Screener discovery module not available
2. Screener returns insufficient results (<5 tickers)
3. Screener reliability drops below 70%
4. User explicitly sets `use_screeners=false`

---

## CPU Optimization

The implementation is optimized for CPU-only systems:

1. **Sequential processing** for screener calls (no GPU needed)
2. **Caching** reduces API calls
3. **Fallback lists** for when screeners are slow
4. **ThreadPoolExecutor** with limited workers (3-5)

---

## Testing

Run screener discovery tests:

```bash
cd stock-prediction-model
python -c "
from src.screeners import YahooScreenerDiscoverer

d = YahooScreenerDiscoverer()
print('US Gainers:', d.discover_tickers('us_gainers', 5))
print('US Active:', d.discover_tickers('us_active', 5))
print('HK Active:', d.discover_tickers('hk_active', 5))
print('Crypto:', d.discover_tickers('crypto', 5))
"
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-11-30 | Initial Yahoo Finance Screener Integration |

---

*This fix transforms the Top-10 system from static database selection to dynamic real-time market discovery.*
