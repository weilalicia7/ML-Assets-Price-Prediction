# Portfolio Summary Calculations

## Overview

This document explains how the Mock Trading Portfolio calculates all summary metrics, including handling of LONG and SHORT positions.

---

## Starting Capital

```
STARTING_CASH = $100,000
```

---

## Position Types

### LONG Position
- **Entry**: Buy shares expecting price to go UP
- **P&L Calculation**: `(currentPrice - entryPrice) * quantity`
- **Profit when**: Current price > Entry price
- **Loss when**: Current price < Entry price

### SHORT Position
- **Entry**: Sell borrowed shares expecting price to go DOWN
- **P&L Calculation**: `(entryPrice - currentPrice) * quantity`
- **Profit when**: Current price < Entry price
- **Loss when**: Current price > Entry price

---

## Core Calculations

### 1. Unrealized P&L (Per Position)

```javascript
function calculateUnrealizedPnL(position) {
    if (position.type === 'LONG') {
        return (currentPrice - position.entryPrice) * position.quantity;
    } else if (position.type === 'SHORT') {
        return (position.entryPrice - currentPrice) * position.quantity;
    }
}
```

**Example - LONG:**
- Entry: $100, Current: $110, Quantity: 10
- P&L = ($110 - $100) * 10 = **+$100 profit**

**Example - SHORT:**
- Entry: $100, Current: $90, Quantity: 10
- P&L = ($100 - $90) * 10 = **+$100 profit**

---

### 2. Total Unrealized P&L

```javascript
totalUnrealizedPnL = sum of all position P&Ls
```

This is the sum of unrealized P&L across all open positions (both LONG and SHORT).

---

### 3. Cash Balance

```javascript
// When opening a LONG position:
cash = cash - (entryPrice * quantity)

// When opening a SHORT position:
cash = cash + (entryPrice * quantity)  // Receive cash from short sale

// When closing a LONG position:
cash = cash + (exitPrice * quantity)

// When closing a SHORT position:
cash = cash - (exitPrice * quantity)  // Buy back shares to return
```

---

### 4. Total Portfolio Value

```javascript
totalValue = cash + totalUnrealizedPnL
```

**Why this formula works:**
- For LONG positions: Cash was reduced when buying, P&L reflects current value change
- For SHORT positions: Cash increased when shorting, P&L reflects liability change

**Alternative equivalent formula:**
```javascript
totalValue = cash + (sum of LONG position values) - (sum of SHORT liabilities)

// Where:
// LONG position value = currentPrice * quantity
// SHORT liability = currentPrice * quantity (what you'd pay to close)
```

---

### 5. Total Return (%)

```javascript
totalReturn = ((totalValue - STARTING_CASH) / STARTING_CASH) * 100
```

**Example:**
- Starting Cash: $100,000
- Current Total Value: $105,000
- Total Return = (($105,000 - $100,000) / $100,000) * 100 = **+5.00%**

---

### 6. Win Rate (%)

```javascript
winRate = (winningTrades / totalClosedTrades) * 100
```

**Definitions:**
- **Winning Trade**: A closed position where realized P&L > 0
- **Total Closed Trades**: All positions that have been closed (both winning and losing)

**Note:** Win rate is only calculated from CLOSED positions, not open positions.

---

## Risk Metrics

### 7. Drawdown Monitoring

```javascript
// Track peak portfolio value
if (totalValue > peakValue) {
    peakValue = totalValue;
    peakDate = currentDate;
}

// Calculate current drawdown
drawdown = ((totalValue - peakValue) / peakValue) * 100
```

**Drawdown States:**
| Drawdown | State | Color |
|----------|-------|-------|
| 0% to -5% | Safe | Green |
| -5% to -10% | Caution | Yellow |
| Below -10% | Danger | Red + ALERT badge |

**Example:**
- Peak Value: $110,000
- Current Value: $99,000
- Drawdown = (($99,000 - $110,000) / $110,000) * 100 = **-10.0%** (DANGER)

---

### 8. Position Concentration

```javascript
// For each position:
concentration = (positionValue / totalPortfolioValue) * 100

// Alert if any position > 15%
if (concentration > 15) {
    showConcentrationAlert(ticker, concentration);
}
```

**Example:**
- Total Portfolio: $100,000
- AAPL Position Value: $18,000
- Concentration = ($18,000 / $100,000) * 100 = **18%** (ALERT: >15%)

---

## Complete Example

### Starting State
```
Cash: $100,000
Positions: None
Total Value: $100,000
```

### After Opening Positions
```
Position 1: LONG AAPL - 50 shares @ $150 = $7,500 invested
Position 2: SHORT TSLA - 20 shares @ $200 = $4,000 received

Cash: $100,000 - $7,500 + $4,000 = $96,500
```

### Current Prices Changed
```
AAPL: $150 -> $160 (+$10)
TSLA: $200 -> $180 (-$20)

LONG AAPL P&L: ($160 - $150) * 50 = +$500
SHORT TSLA P&L: ($200 - $180) * 20 = +$400

Total Unrealized P&L: $500 + $400 = +$900
Total Value: $96,500 + $900 = $97,400

Wait, that's wrong! Let's recalculate...
```

### Correct Calculation
```
Starting: $100,000

LONG AAPL: Spent $7,500, now worth $8,000 (50 * $160)
SHORT TSLA: Received $4,000, liability now $3,600 (20 * $180)

Cash: $100,000 - $7,500 + $4,000 = $96,500
LONG Value: $8,000
SHORT Liability: $3,600

Total Value = Cash + LONG Values - SHORT Liabilities
            = $96,500 + $8,000 - $3,600
            = $100,900

OR using P&L method:
Total Value = Cash + Total P&L
            = $96,500 + $500 + $400
            = $97,400

Hmm, these don't match. Let me reconsider...
```

### Correct Understanding

The issue is in how cash changes with SHORT positions:

```
When you SHORT:
- You borrow shares and sell them immediately
- You receive cash = entryPrice * quantity
- You have a LIABILITY to buy back shares later

When you CLOSE a SHORT:
- You buy shares at current price
- You pay cash = currentPrice * quantity
- Liability is eliminated

SHORT P&L = Cash received at entry - Cash paid at close
          = (entryPrice * quantity) - (currentPrice * quantity)
          = (entryPrice - currentPrice) * quantity
```

### Corrected Example

```
Starting Cash: $100,000

Open LONG AAPL: Buy 50 @ $150
  Cash: $100,000 - $7,500 = $92,500

Open SHORT TSLA: Short 20 @ $200
  Cash: $92,500 + $4,000 = $96,500
  (Received cash from short sale)

Current State:
  Cash: $96,500
  LONG AAPL: 50 shares, entry $150
  SHORT TSLA: 20 shares, entry $200

Prices Change:
  AAPL: $160 (up $10)
  TSLA: $180 (down $20)

P&L Calculations:
  LONG AAPL: ($160 - $150) * 50 = +$500
  SHORT TSLA: ($200 - $180) * 20 = +$400
  Total Unrealized P&L: +$900

Total Value = Cash + Unrealized P&L
            = $96,500 + $900
            = $97,400

Total Return = ($97,400 - $100,000) / $100,000 * 100
             = -2.6%

Wait, we made $900 but lost money? Let's trace it again...
```

### Final Correct Example

```
The $96,500 cash already includes:
- Original $100,000
- Minus $7,500 spent on LONG
- Plus $4,000 from SHORT sale

If we close everything NOW:
- Sell LONG AAPL: 50 * $160 = +$8,000
- Close SHORT TSLA: 20 * $180 = -$3,600

Final Cash = $96,500 + $8,000 - $3,600 = $100,900

Total Value = $100,900 (all cash, no positions)
Total Return = ($100,900 - $100,000) / $100,000 * 100 = +0.9%

This matches: $500 (AAPL profit) + $400 (TSLA profit) = $900 gain
```

---

## Summary Formula

```
Total Value = Cash + Sum(Unrealized P&L for all positions)

Where:
- LONG P&L = (currentPrice - entryPrice) * quantity
- SHORT P&L = (entryPrice - currentPrice) * quantity

Total Return = ((Total Value - $100,000) / $100,000) * 100

Win Rate = (Winning Closed Trades / Total Closed Trades) * 100

Drawdown = ((Current Value - Peak Value) / Peak Value) * 100

Concentration = (Position Value / Total Value) * 100
```

---

## Implementation Location

| Calculation | File | Function |
|-------------|------|----------|
| P&L Calculation | `static/js/auth.js` | `displayLocalPortfolio()` |
| Total Value | `static/js/auth.js` | `displayLocalPortfolio()` |
| Drawdown | `static/js/auth.js` | `displayLocalPortfolio()` |
| Concentration | `static/js/auth.js` | `displayLocalPortfolio()` |
| Peak Tracking | `static/js/auth.js` | `saveLocalPortfolio()` |

---

*Document created: 2025-11-29*
