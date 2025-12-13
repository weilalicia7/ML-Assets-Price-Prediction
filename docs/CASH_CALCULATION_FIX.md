# Cash Calculation Fix - Review Document

## Current Bug Status

```
Cash:        $645,864.13
Total Value: $645,193.44
Difference:  $670.69
```

**Problem**: Cash > Total Value (mathematically impossible)

---

## Current Calculation Logic

### File: `static/js/auth.js`

### Function: `recalculateCashFromPositions()` (lines 386-420)

```javascript
function recalculateCashFromPositions(portfolio) {
    const STARTING_CASH = 100000;
    let cash = STARTING_CASH;

    // Step 1: Subtract cost of all open LONG positions
    for (const trade of (portfolio.trades || [])) {
        if (trade.action === 'BUY' || trade.action === 'LONG') {
            const cost = trade.entry_price * trade.quantity;
            cash -= cost;
        }
        // SHORT positions: Do NOT modify cash (proceeds are collateral)
    }

    // Step 2: Add/subtract P&L from all closed trades
    for (const closedTrade of (portfolio.closedTrades || [])) {
        if (closedTrade.action === 'BUY' || closedTrade.action === 'LONG') {
            // Closed LONG: Add sale proceeds
            const saleProceeds = closedTrade.closePrice * closedTrade.quantity;
            cash += saleProceeds;
        } else {
            // Closed SHORT: Add P&L only (NOT collateral)
            const pnl = (closedTrade.entry_price - closedTrade.closePrice) * closedTrade.quantity;
            cash += pnl;
        }
    }

    return cash;
}
```

---

## Expected Formulas

### Total Value Calculation

```
Total Value = Cash + Total Unrealized P&L
```

### Unrealized P&L (per position)

```
LONG P&L  = (currentPrice - entryPrice) * quantity
SHORT P&L = (entryPrice - currentPrice) * quantity
```

### Cash Changes

| Action | Cash Change |
|--------|-------------|
| Open LONG | `cash -= entryPrice * quantity` |
| Open SHORT | `cash += 0` (proceeds held as collateral) |
| Close LONG | `cash += closePrice * quantity` |
| Close SHORT | `cash += (entryPrice - closePrice) * quantity` (P&L only) |

---

## Mathematical Constraint

**Rule**: `Cash <= Total Value` (always)

**Why?**
- `Total Value = Cash + Unrealized P&L`
- `Cash > Total Value` only if `Unrealized P&L < 0` AND `|Unrealized P&L| > (Total Value - Cash)`
- But this is mathematically impossible since Total Value includes Cash

---

## Diagnostic Check Added (lines 991-1013)

```javascript
if (portfolio.cash > totalValue) {
    console.error('CRITICAL BUG: Cash > Total Value!');

    // Recalculate cash from scratch
    const correctedCash = recalculateCashFromPositions(portfolio);
    portfolio.cash = correctedCash;
    saveLocalPortfolio(portfolio);
}
```

---

## Questions to Investigate

1. **Are there open SHORT positions?**
   - If yes, how was their cash initially calculated?

2. **Are there closed SHORT positions in `closedTrades`?**
   - If yes, what is the P&L for each?

3. **What is the stored `portfolio.cash` value vs recalculated?**
   - Check console output for `Corrected Cash would be:`

4. **Is there a mismatch in position tracking?**
   - Any trades that didn't properly record entry/close?

---

## To Debug

Open browser console and look for:
```
=== CASH DIAGNOSTIC ===
Current Cash: XXXXX
Current Total Value: XXXXX
Unrealized P&L: XXXXX
```

If corruption detected, you should see:
```
CRITICAL BUG: Cash > Total Value!
Corrected Cash would be: XXXXX
```

---

*Document created: 2025-11-29*
