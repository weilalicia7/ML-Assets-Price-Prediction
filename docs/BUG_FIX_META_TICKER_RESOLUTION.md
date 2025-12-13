# Bug Fix: META/TM Ticker Resolution Issue

## Issue Summary

**Date Fixed:** 2025-11-28
**Severity:** High
**Component:** `src/data/china_ticker_resolver.py`

US stock tickers like META (Meta Platforms Inc.) and TM (Toyota Motor) were incorrectly being resolved to Chinese stocks during predictions and mock trading operations.

## Symptoms

| Ticker | Expected | Actual (Before Fix) | Price Discrepancy |
|--------|----------|---------------------|-------------------|
| META | Meta Platforms Inc. ($644) | Meituan 3690.HK (HK$130) | ~5x difference |
| TM | Toyota Motor ADR ($201) | East Money 300059.SZ | Incorrect company |

**User Report:** "The purchase price was right about META; but the sold price switched to Meituan"

This caused incorrect P&L calculations in the mock trading portfolio when closing positions.

## Root Cause Analysis

### Location
`src/data/china_ticker_resolver.py`, line 532-538 (fuzzy matching section)

### Problem
The `resolve_china_stock()` function used fuzzy string matching with a threshold of 0.6 on **all** incoming ticker queries. This caused:

1. "META" to fuzzy-match "Meituan" (both start with "ME", similarity > 0.6)
2. "TM" to potentially match Chinese company names

### Code Flow (Before Fix)
```
User requests META prediction
    ↓
predict() endpoint calls resolve_china_stock("META")
    ↓
Fuzzy matching compares "META" against all Chinese company names
    ↓
"META" matches "Meituan" with similarity > 0.6
    ↓
Returns "3690.HK" instead of None
    ↓
Prediction uses Meituan data instead of Meta Platforms
```

## Solution

Added pattern detection to recognize standard US/International ticker formats **before** the fuzzy matching section. If a query matches a US ticker pattern, the function returns `None` immediately, allowing the system to process it as-is.

### Fix Location
`src/data/china_ticker_resolver.py`, lines 532-547

### Code Added
```python
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
```

## Verification

### Test Results (After Fix)

```
=== META PREDICTION TEST ===
Status: success
Ticker: META
Current Price: $644.00
SUCCESS: META returned correct ticker (not Meituan 3690.HK)

=== TM (Toyota) PREDICTION TEST ===
Status: success
Ticker: TM
Current Price: $201.70
SUCCESS: TM returned correct ticker (not East Money 300059.SZ)

=== CHINA STOCK RESOLUTION STILL WORKS ===
1. "Tencent" → 0700.HK (HK$611.50) ✓
2. "Meituan" → 3690.HK (HK$102.50) ✓
```

### What Still Works
- Searching by Chinese company name (e.g., "Tencent" → 0700.HK)
- Direct HK/China ticker lookup (e.g., "0700.HK", "9988.HK")
- All other asset types (crypto, forex, commodities)

### What's Fixed
- US stock tickers no longer fuzzy-match to Chinese companies
- Mock trading positions close at correct prices
- Top 10 picks show correct company data

## Files Modified

| File | Change |
|------|--------|
| `src/data/china_ticker_resolver.py` | Added US ticker pattern detection before fuzzy matching |

## Testing Checklist

- [x] META returns Meta Platforms Inc. data
- [x] TM returns Toyota Motor ADR data
- [x] AAPL, NVDA, GOOGL work correctly
- [x] "Tencent" still resolves to 0700.HK
- [x] "Meituan" still resolves to 3690.HK
- [x] "Alibaba" still resolves to 9988.HK
- [x] BTC-USD, ETH-USD work correctly
- [x] GC=F, CL=F (commodities) work correctly
- [x] EURUSD=X, GBPUSD=X (forex) work correctly

## Prevention

To prevent similar issues in the future:

1. **Pattern-based filtering** should always precede fuzzy matching
2. **Price sanity checks** exist but only catch extreme differences (>10x)
3. Consider adding **exchange-based validation** (US tickers should have US exchange data)

## Related Issues

- Mock trading P&L calculation errors when closing META positions
- Top 10 picks showing incorrect company data for US stocks
