# US_IPO Tab - Code Analysis & Fix

## Status: FIXED (2025-12-22)

The US_IPO tab now correctly returns only **actual recent IPOs** (stocks with < 60 days trading history).

## Problems Fixed
1. **Established stocks** (like BE, ONDS, BBAI) were being returned instead of actual IPOs
2. **Warrants and derivatives** (like PSNYW) were incorrectly included

## Solution Implemented

**File:** `src/screeners/yahoo_screener_discovery.py`

### 1. Added `_filter_us_ipos()` Method (lines 696-748)
```python
def _filter_us_ipos(self, tickers: List[str], max_days: int = 60) -> List[str]:
    """
    Filter US tickers to only include actual IPOs (< max_days trading).
    Excludes warrants, SPAC derivatives, and other non-stock securities.
    Uses parallel checking for speed.
    """
    def check_ticker(ticker):
        try:
            # Exclude warrants by ticker pattern (ends with W, WS, WT, +)
            if ticker.endswith('W') or ticker.endswith('WS') or ticker.endswith('WT') or '+' in ticker:
                return None

            t = yf.Ticker(ticker)

            # Check quote type - only accept EQUITY (common stock)
            try:
                info = t.info
                quote_type = info.get('quoteType', '').upper()
                if quote_type not in ['EQUITY', '']:
                    return None
                # Also check for warrant/unit indicators in name
                long_name = info.get('longName', '').lower()
                if any(x in long_name for x in ['warrant', 'unit', 'right', 'preferred']):
                    return None
            except:
                pass

            hist = t.history(period="6mo")
            if hist is not None and len(hist) > 0 and len(hist) < max_days:
                return (ticker, len(hist))
        except:
            pass
        return None

    # Check in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(check_ticker, tickers))

    ipos = [(t, d) for t, d in [r for r in results if r is not None]]
    ipos.sort(key=lambda x: x[1])  # Sort by days (newest first)
    return [t[0] for t in ipos]
```

### 2. Updated `_us_ipo_strategy()` (lines 727-752)
```python
def _us_ipo_strategy(self, count: int) -> List[str]:
    """
    US IPOs and new stocks (< 60 days trading) - REAL-TIME ONLY.

    Fetches candidates from multiple screeners, then FILTERS to only actual IPOs.
    No hardcoded tickers - 100% real-time data.
    """
    # Get candidates from multiple screeners (more sources = better chance of finding IPOs)
    gainers = self.discoverer.discover_tickers('us_gainers', count * 3)
    active = self.discoverer.discover_tickers('us_active', count * 2)
    small_cap = self.discoverer.discover_tickers('us_small_cap', count * 2)

    # Combine and deduplicate
    combined = list(dict.fromkeys(gainers + active + small_cap))

    logger.info(f"[US IPO STRATEGY] Checking {len(combined)} candidates for actual IPOs...")

    # FILTER: Only keep stocks with < 60 days trading history
    ipos = self._filter_us_ipos(combined, max_days=60)

    if ipos:
        logger.info(f"[US IPO STRATEGY] Found {len(ipos)} actual US IPOs")
        return ipos[:count]
    else:
        logger.warning("[US IPO STRATEGY] No US IPOs found with < 60 days trading")
        return []
```

## Test Results (2025-12-22)

```bash
curl http://localhost:5000/api/top-picks?regime=US_IPO
```

### Server Logs
```
[US IPO STRATEGY] Checking 163 candidates for actual IPOs...
[US IPO FILTER] Found 3 actual IPOs from 163 candidates:
  - BLLN: 31 trading days
  - EVMN: 31 trading days
  - NAVN: 36 trading days
[US IPO STRATEGY] Found 3 actual US IPOs
```

### IPOs Found
| Ticker | Company | Trading Days | Status |
|--------|---------|--------------|--------|
| BLLN | BillionToOne, Inc. | 31 days | Recent IPO |
| EVMN | Evommune, Inc. | 31 days | Recent IPO |
| NAVN | Navan, Inc. | 36 days | Recent IPO |

### Excluded (Warrants/Derivatives)
| Ticker | Reason |
|--------|--------|
| PSNYW | Warrant (ends with W) - Polestar SPAC warrant |

## webapp.py Integration

### Minimum Ticker Threshold (Lines 4295-4310)
```python
min_tickers = 1 if regime in ['US_IPO', 'China_IPO'] else 5
is_ipo_regime = regime in ['US_IPO', 'China_IPO']

if screener_tickers and len(screener_tickers) >= min_tickers:
    tickers = screener_tickers
elif is_ipo_regime:
    # Accept 0 results for IPO regimes (no current IPOs is valid)
    tickers = screener_tickers if screener_tickers else []
```

### Optimizer Skip (Line 4512)
```python
# US_IPO skips the US/INTL optimizer
if regime not in ['China', 'China_IPO', 'US_IPO']:
```

## How It Works

1. **Fetch Candidates**: Gets ~160+ stocks from `us_gainers`, `us_active`, `us_small_cap` screeners
2. **Exclude Derivatives**: Filters out warrants (W, WS, WT), units, rights, preferred stock
3. **Filter by Trading History**: Checks each stock's 6-month history in parallel
4. **Keep Only IPOs**: Returns only stocks with < 60 trading days
5. **Sort by Recency**: Newest IPOs (fewest trading days) appear first

## Exclusion Rules

Tickers are excluded if they match any of these patterns:
- **Ticker suffix**: Ends with W, WS, WT, or contains +
- **Quote type**: Not EQUITY (e.g., WARRANT, RIGHT, UNIT)
- **Name contains**: "warrant", "unit", "right", "preferred"

## Notes

- Uses parallel processing (10 workers) for fast filtering
- Returns empty list if no IPOs found (valid state)
- No hardcoded tickers - 100% real-time data from Yahoo Finance
- SPAC warrants like PSNYW are correctly excluded
