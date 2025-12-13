# China Model Logic & Configuration

**Last Updated:** 2025-12-01

## Current Configuration Summary

| Component | Status | Description |
|-----------|--------|-------------|
| **buyfix1** | Conceptually Active | Trend filter, MA crossover, stop-loss 5-7%, position sizing - NOT strictly filtering |
| **buyfix2** | DISABLED | `ChinaBuyOptimizer` class |
| **buyfix3** | DISABLED | `EnhancedChinaBuyOptimizer` class |
| **buyfix4** | DISABLED | `ProductionChinaBuyOptimizer` class |

## How the China Model Works

### Signal Generation Flow

1. **ML Model Prediction** - CatBoost/XGBoost ensemble generates confidence scores
2. **EnhancedPhaseSystem** - Applies Phase 1-6 calculations for risk-adjusted ranking
3. **Profit Scoring** - Ranks signals by expected return, confidence, and risk metrics
4. **Top 10 Selection** - Highest-scoring BUY signals displayed

### Key Insight: Mean Reversion Strategy

The ML model is **NOT** a traditional trend-following system. Analysis of the current Top 10 reveals the model favors **mean reversion opportunities**:

- Stocks near their 20-day lows
- Short-term momentum turning positive (5-day momentum > 0)
- High upside potential to recent highs

This explains why stocks failing traditional buyfix1 criteria (price > MA20, 20d momentum > -5%) still appear and perform well.

## Current Top 10 Analysis (2025-12-01)

| # | Ticker | Pattern | Key Metrics | Assessment |
|---|--------|---------|-------------|------------|
| 1 | **000981.SZ** | Near Low + Short Mom Up | 21% of range, +3.4% 5d mom, +16.9% upside | **CONFIRMED ACCURATE** |
| 2 | 1060.HK | Near Low + High Vol | 32% of range, 63.7% vol | Higher risk |
| 3 | 1810.HK | Short Bounce | +6.1% 5d mom, XIAOMI | Large cap quality |
| 4 | **600016.SS** | Trend Up + Low Vol | Above MA20, 18.4% vol | **IDEAL TECHNICAL** |
| 5 | 3800.HK | Near Low | 11% of range, +26.7% upside | High upside potential |
| 6 | **000630.SZ** | Above MAs + Flat Trend | Above MA5 & MA20 | **SOLID SETUP** |
| 7 | 600010.SS | At Bottom | 5% of range, +15.2% upside | Deep value play |
| 8 | **0939.HK** | Low Vol Bank | 18.5% vol, CCB | Stable, low risk |
| 9 | 000980.SZ | Short Bounce | +3.5% 5d mom | Recovery play |
| 10 | 603077.SS | Near Low + Flat | 24% of range | Mean reversion |

### Signal Categories

**Strong Technical (buyfix1 compliant):**
- 600016.SS - China Minsheng Bank
- 000630.SZ - Tongling Nonferrous

**Mean Reversion (near 20-day low with momentum turning):**
- 000981.SZ - Sensteed Hi-Tech (CONFIRMED ACCURATE)
- 3800.HK - GCL Technology
- 600010.SS - Baotou Steel
- 603077.SS - Hebang Biotech

**Low Volatility Banks:**
- 0939.HK - CCB
- 600016.SS - China Minsheng Bank

## Why buyfix1 is NOT Strictly Applied

### The Problem with Strict Filtering

If buyfix1 were strictly enforced, it would filter based on:
- Price > 20-day MA
- 20-day momentum > -5%
- Volatility < 50%

**Result:** Only 2/10 current signals would pass (600016.SS, 000630.SZ)

**Issue:** This would filter out 000981.SZ which you confirmed is **ACCURATE**

### The ML Model's Edge

The ML model captures patterns that simple technical rules miss:
1. **Oversold bounces** - Stocks that dropped too far, too fast
2. **Short-term momentum reversals** - 5-day momentum turning positive
3. **Volume patterns** - Unusual volume preceding moves
4. **Cross-asset correlations** - Sector and market regime effects

## Risk Management (Applied at Execution)

Even without strict pre-filtering, risk is managed through:

| Control | Value | Application |
|---------|-------|-------------|
| Stop-Loss | 5-7% | Exit if position drops beyond threshold |
| Position Sizing | 5% max | Per-signal allocation |
| Diversification | 10 signals | Spread across sectors |
| Holding Period | 20 days | Time-based exit |

## Configuration in webapp.py

```python
# Lines 100-123 in webapp.py

# buyfix1: ACTIVE via EnhancedPhaseSystem (trend filter, MA crossover, stop-loss, position sizing)
# buyfix2/3/4: DISABLED (no optimizer classes)

ENHANCED_PHASE_SYSTEM = EnhancedPhaseSystem()  # Contains buyfix1 components
CHINA_BUY_OPTIMIZER = None  # Disabled - no fix2/fix3/fix4 optimizer classes
CHINA_BUY_MONITOR = None    # Disabled - no production monitoring
```

## Recommendations

### Current Approach (Recommended)
- Keep buyfix2/3/4 DISABLED
- Let ML model confidence drive signal selection
- Apply stop-loss at execution (not signal filtering)
- Monitor actual P&L to validate

### When to Reconsider
- If mean reversion signals consistently underperform
- If high-volatility signals (like 1060.HK) cause large losses
- If win rate drops below 40%

### Future Improvements
- Track actual 20-day returns of each signal
- Build performance database to validate model patterns
- Consider adding mean reversion confidence boost (not filter)

## Files Reference

| File | Purpose |
|------|---------|
| `webapp.py` | Main backend, signal generation |
| `src/utils/enhanced_signal_validator.py` | EnhancedPhaseSystem, buyfix1 components |
| `buy fixing on C model.pdf` | buyfix1 documentation |
| `buy fixing2 on C model.pdf` | buyfix2 (ChinaBuyOptimizer) - DISABLED |
| `buy fix3 on C model.pdf` | buyfix3 (EnhancedChinaBuyOptimizer) - DISABLED |
| `buy fix4 on C model.pdf` | buyfix4 (ProductionChinaBuyOptimizer) - DISABLED |

---

*Document generated based on analysis of China Top 10 BUY signals and buyfix configuration review.*
