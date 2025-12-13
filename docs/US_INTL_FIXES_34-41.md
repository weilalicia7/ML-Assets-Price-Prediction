# US/Intl Model Fixes 34-41 (December 2025)

## Overview

These fixes implement advanced profit-maximizing strategies for the US/International model:
- Fix 34: Intraday Momentum Timing - Optimal entry windows
- Fix 35: Market Cap Tier Optimizer - Size-based strategy differentiation
- Fix 36: Quarter-End Window Dressing - Institutional pattern exploitation
- Fix 37: Earnings Gap Trading - Post-earnings momentum/fade strategies
- Fix 38: Sector Rotation Momentum - Leading/lagging sector detection
- Fix 39: VIX Term Structure Arbitrage - Contango/backwardation signals
- Fix 40: Economic Data Reactions - High-frequency event trading
- Fix 41: Put/Call Ratio Reversals - Contrarian sentiment signals

**Important**: These fixes apply ONLY to US/Intl model. China/DeepSeek model is unchanged.

---

## Fix 34: Intraday Momentum Timing

**Problem**: Generic signal timing ignores well-documented intraday patterns in US markets.

**Solution**: Optimize entry timing based on proven intraday patterns.

### Entry Windows
| Window | Time (ET) | BUY Boost | SELL Boost | Strategy |
|--------|-----------|-----------|------------|----------|
| Opening Range Breakout | 10:15-10:45 | 1.15x | 1.10x | Trade breakout direction |
| Midday Dip | 12:30-13:30 | 1.10x | 0.85x | Buy dips, avoid shorts |
| Power Hour Momentum | 15:00-15:30 | 1.20x | 1.15x | Strong momentum follows |
| Close Avoidance | 15:45-16:00 | 0.70x | 0.70x | Avoid late entries |

### Usage Example
```python
from src.models.us_intl_optimizer import IntradayMomentumOptimizer

optimizer = IntradayMomentumOptimizer()

# Get timing boost for current time
boost, reason = optimizer.get_timing_boost('BUY')
print(f"Boost: {boost:.2f}x - {reason}")

# Check if optimal entry time
is_optimal, window = optimizer.is_optimal_entry_time('BUY')
print(f"Optimal: {is_optimal}, Window: {window}")
```

---

## Fix 35: Market Cap Tier Optimizer

**Problem**: One-size-fits-all approach ignores significant differences between mega-cap and small-cap dynamics.

**Solution**: Different strategies, position sizes, and profit targets by market cap tier.

### Market Cap Thresholds
| Tier | Market Cap | Strategy | Position Cap |
|------|------------|----------|--------------|
| Mega | >$200B | Trend Following | 25% |
| Large | $10B-$200B | Momentum | 15% |
| Mid | $2B-$10B | Hybrid | 10% |
| Small | $300M-$2B | Mean Reversion | 8% |
| Micro | <$300M | Speculative | 5% |

### Position Adjustments by Tier
| Tier | BUY Multiplier | SELL Multiplier | Profit Targets |
|------|----------------|-----------------|----------------|
| Mega | 1.20x | 0.90x | 8%, 15%, 25% |
| Large | 1.10x | 1.00x | 10%, 18%, 30% |
| Mid | 1.00x | 1.05x | 10%, 20%, 35% |
| Small | 0.90x | 1.10x | 12%, 25%, 40% |
| Micro | 0.75x | 1.15x | 15%, 30%, 50% |

### Known Mega-Cap Stocks
AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, BRK-B, JPM, JNJ, V, UNH, XOM, MA, PG, HD, CVX, MRK, ABBV, PFE, KO, COST, PEP, WMT, MCD, CSCO, AVGO, TMO, ABT, CRM

### Usage Example
```python
from src.models.us_intl_optimizer import MarketCapTierOptimizer

optimizer = MarketCapTierOptimizer()

# Classify by ticker (uses known lists)
tier = optimizer.classify_market_cap(ticker='AAPL')  # Returns 'mega'

# Or classify by market cap
tier = optimizer.classify_market_cap(market_cap=500_000_000_000)  # Returns 'mega'

# Get position adjustment
mult, details, reason = optimizer.get_position_adjustment('BUY', ticker='AAPL')
print(f"Multiplier: {mult:.2f}x")
print(f"Profit targets: {details['profit_targets']}")
```

---

## Fix 36: Quarter-End Window Dressing

**Problem**: Ignoring institutional window dressing creates missed opportunities.

**Solution**: Exploit predictable quarter-end institutional behavior.

### Quarter-End Months
March, June, September, December (Q1-Q4 ends)

### Window Dressing Patterns
**Last Week of Quarter (Days -5 to 0)**:
- Institutions ADD winners to portfolios
- Institutions DUMP losers before reporting

**First Week of New Quarter (Days +1 to +3)**:
- Profit-taking on recent winners
- Oversold bounce on dumped stocks

### Position Adjustments
| Window | Performance | BUY | SELL |
|--------|-------------|-----|------|
| Last Week | Top Performer (>+15% QTD) | 1.25x | 0.50x |
| Last Week | Bottom Performer (<-10% QTD) | 0.60x | 1.30x |
| Last Week | Neutral | 1.05x | 0.90x |
| First Week | Top Performer | 0.80x | 1.10x |
| First Week | Bottom Performer | 1.15x | 0.70x |
| First Week | Neutral | 1.00x | 1.00x |

### Usage Example
```python
from src.models.us_intl_optimizer import QuarterEndOptimizer

optimizer = QuarterEndOptimizer()

# Check if in quarter-end window
is_window, window_type = optimizer.is_quarter_end_window()

# Get adjustment for a stock
result = optimizer.get_quarter_end_adjustment(
    ticker='AAPL',
    signal_type='BUY',
    qtd_return=0.20  # +20% QTD (top performer)
)
print(f"Multiplier: {result['position_multiplier']:.2f}x")
print(f"Reason: {result['reason']}")
```

---

## Fix 37: Earnings Gap Trading

**Problem**: Earnings gaps create high-probability setups that are being ignored.

**Solution**: Implement gap-and-go and fade strategies based on gap size and volume.

### Gap Classification
| Strategy | Gap Range | Volume Req | Action | Probability |
|----------|-----------|------------|--------|-------------|
| Strong Gap Up | +5% to +∞ | 2x avg | BUY momentum | 65% |
| Moderate Gap Up | +3% to +5% | 1.5x avg | FADE (sell) | 55% |
| Strong Gap Down | -5% to -∞ | 2x avg | SELL momentum | 40% |
| Moderate Gap Down | -3% to -5% | 1.5x avg | FADE (buy dip) | 50% |

### Position Multipliers
| Strategy | Position Mult | Entry Window | Stop Loss |
|----------|---------------|--------------|-----------|
| Strong Gap Up | 1.25x | 9:45-10:15 | 3.5% |
| Moderate Gap Up | 0.85x | 10:00-10:30 | 2.5% |
| Strong Gap Down | 1.15x | 9:45-10:15 | 4.0% |
| Moderate Gap Down | 0.90x | 10:00-10:30 | 3.5% |

### Usage Example
```python
from src.models.us_intl_optimizer import EarningsGapTrader

trader = EarningsGapTrader()

# Analyze a gap
strategy = trader.analyze_gap(
    gap_percent=0.07,      # 7% gap up
    volume_ratio=2.5,      # 2.5x normal volume
    is_earnings_day=True
)
if strategy:
    print(f"Strategy: {strategy['strategy']}")
    print(f"Action: {strategy['action']}")
    print(f"Targets: {strategy['targets']}")

# Get position adjustment
mult, reason = trader.get_gap_adjustment('BUY', gap_percent=0.07, volume_ratio=2.5)
print(f"Multiplier: {mult:.2f}x - {reason}")
```

---

## Fix 38: Sector Rotation Momentum

**Problem**: Not tracking sector rotation leaves alpha on the table.

**Solution**: Boost positions in rotating-in sectors, reduce in rotating-out.

### Rotation Detection
Based on three factors:
1. **Relative Strength** vs SPY (>15% = rotating in, <-15% = rotating out)
2. **Fund Flows** (>2% inflows = strong, <-2% = weak)
3. **5-Day Momentum** (>2% = positive, <-2% = negative)

### Rotation Statuses
| Status | Score | Position Mult | Profit Target | Stop Loss |
|--------|-------|---------------|---------------|-----------|
| ROTATING_IN | ≥3 | 1.35x | 1.30x | 0.75x |
| STRENGTHENING | 1-2 | 1.20x | 1.15x | 0.85x |
| NEUTRAL | 0 | 1.00x | 1.00x | 1.00x |
| WEAKENING | -1 to -2 | 0.75x | 0.85x | 1.15x |
| ROTATING_OUT | ≤-3 | 0.55x | 0.70x | 1.30x |

### Signal Alignment
- BUY signals in ROTATING_IN sectors: Full boost (1.35x)
- BUY signals in ROTATING_OUT sectors: Reduced (0.80x)
- SELL signals in ROTATING_OUT sectors: Boosted (1.2x of base)
- SELL signals in ROTATING_IN sectors: Reduced (0.70x)

### Usage Example
```python
from src.models.us_intl_optimizer import SectorRotationMomentum

analyzer = SectorRotationMomentum()

# Detect rotation status
status = analyzer.detect_rotation_status(
    relative_strength=1.20,  # 20% outperforming
    fund_flow=0.03,          # 3% inflows
    momentum_5d=0.025        # 2.5% momentum
)
print(f"Status: {status}")  # ROTATING_IN

# Get position adjustment
mult, adjustments, reason = analyzer.get_rotation_adjustment(
    'BUY',
    relative_strength=1.20,
    fund_flow=0.03
)
print(f"Multiplier: {mult:.2f}x - {reason}")
```

---

## Fix 39: VIX Term Structure Arbitrage

**Problem**: VIX term structure signals market sentiment but isn't being used.

**Solution**: Use contango/backwardation for market timing.

### Term Structure Regimes
| Regime | Contango | Market Sentiment | BUY Mult | SELL Mult |
|--------|----------|------------------|----------|-----------|
| Strong Contango | >5% | COMPLACENT | 1.15x | 0.80x |
| Moderate Contango | 2-5% | CALM | 1.08x | 0.90x |
| Flat | -2% to 2% | NEUTRAL | 1.00x | 1.00x |
| Moderate Backwardation | -2% to -5% | CAUTIOUS | 0.90x | 1.10x |
| Strong Backwardation | <-5% | FEARFUL | 0.80x | 1.20x |

### Confidence Adjustments
| Regime | BUY Confidence | SELL Confidence |
|--------|----------------|-----------------|
| Strong Contango | +5% | -3% |
| Strong Backwardation | -3% | +8% |

### Formula
```
Contango = (VIX_Futures_1M - VIX_Spot) / VIX_Spot
```

### Usage Example
```python
from src.models.us_intl_optimizer import VIXTermStructureAnalyzer

analyzer = VIXTermStructureAnalyzer()

# Analyze term structure
analysis = analyzer.analyze_term_structure(
    vix_spot=15.0,
    vix_futures_1m=17.0,
    vix_futures_2m=18.5
)
print(f"Regime: {analysis['regime']}")
print(f"Contango: {analysis['contango_pct']:.1%}")

# Get position adjustment
mult, conf_adj, reason = analyzer.get_term_structure_adjustment(
    'BUY',
    vix_spot=15.0,
    vix_futures_1m=17.0
)
print(f"Multiplier: {mult:.2f}x - {reason}")
```

---

## Fix 40: Economic Data Reactions

**Problem**: High-impact economic releases create tradeable patterns.

**Solution**: React to economic data surprises with appropriate positioning.

### Economic Events
| Event | Volatility Mult | Duration | Importance |
|-------|-----------------|----------|------------|
| CPI | 1.8x | 4 hours | HIGH |
| Jobs Report (NFP) | 2.2x | 6 hours | HIGH |
| Fed Decision | 2.5x | 8 hours | CRITICAL |
| GDP | 1.5x | 4 hours | MEDIUM |
| Retail Sales | 1.3x | 3 hours | MEDIUM |

### Reaction Types
| Reaction | Trigger | BUY Adj | SELL Adj |
|----------|---------|---------|----------|
| STRONG_POSITIVE | >0.5% vs expected | 1.15x | 0.85x |
| POSITIVE | 0.2-0.5% vs expected | 1.08x | 0.95x |
| NEUTRAL | -0.2% to 0.2% | 1.00x | 1.00x |
| NEGATIVE | -0.5% to -0.2% | 0.95x | 1.08x |
| STRONG_NEGATIVE | <-0.5% vs expected | 0.85x | 1.15x |

### Oversold/Overbought Detection
- RSI < 30 + Negative data: Potential oversold bounce (BUY +20%)
- RSI > 70 + Positive data: Potential overbought reversal (SELL +20%)

### Usage Example
```python
from src.models.us_intl_optimizer import EconomicDataReactor

reactor = EconomicDataReactor()

# Analyze data reaction
analysis = reactor.analyze_data_reaction(
    event_type='cpi',
    actual_vs_expected=0.003,  # 0.3% hotter than expected
    immediate_move=-0.015,     # Market dropped 1.5%
    rsi=25.0                   # Oversold
)
print(f"Reaction: {analysis['reaction_type']}")
print(f"Oversold bounce: {analysis['oversold_bounce']}")

# Get adjustment
mult, reason = reactor.get_economic_adjustment(
    'BUY',
    event_type='cpi',
    actual_vs_expected=0.003
)
print(f"Multiplier: {mult:.2f}x - {reason}")
```

---

## Fix 41: Put/Call Ratio Reversals

**Problem**: Extreme put/call readings are contrarian signals being ignored.

**Solution**: Use put/call ratio for contrarian positioning.

### Put/Call Thresholds
| Level | P/C Ratio | Signal | Probability |
|-------|-----------|--------|-------------|
| Extreme Bearish | >1.20 | STRONG_CONTRARIAN_BUY | 75% |
| Bearish | 1.00-1.20 | CONTRARIAN_BUY | 60% |
| Neutral | 0.55-1.00 | NEUTRAL | 50% |
| Bullish | 0.45-0.55 | CONTRARIAN_SELL | 55% |
| Extreme Bullish | <0.45 | STRONG_CONTRARIAN_SELL | 70% |

### Position Adjustments
| Sentiment | BUY Mult | SELL Mult |
|-----------|----------|-----------|
| Extreme Bearish | 1.40x | 0.60x |
| Bearish | 1.20x | 0.85x |
| Neutral | 1.00x | 1.00x |
| Bullish | 0.85x | 1.20x |
| Extreme Bullish | 0.60x | 1.40x |

### Trend Confirmation
Compares current ratio to 5-day average:
- Rising P/C: Increasing fear (more contrarian bullish)
- Falling P/C: Increasing complacency (more contrarian bearish)

### Usage Example
```python
from src.models.us_intl_optimizer import PutCallRatioAnalyzer

analyzer = PutCallRatioAnalyzer()

# Classify ratio
sentiment = analyzer.classify_put_call_ratio(1.30)
print(f"Sentiment: {sentiment}")  # extreme_bearish

# Get adjustment
mult, reason = analyzer.get_put_call_adjustment(
    'BUY',
    put_call_ratio=1.30,
    put_call_ratio_5d_avg=1.10  # Rising from 1.10
)
print(f"Multiplier: {mult:.2f}x - {reason}")
```

---

## Integration with USIntlModelOptimizer

All eight fixes are integrated into the main optimizer:

```python
from src.models.us_intl_optimizer import USIntlModelOptimizer

optimizer = USIntlModelOptimizer(
    # Fix 34: Intraday Momentum Timing
    enable_intraday_timing=True,

    # Fix 35: Market Cap Tier Optimizer
    enable_market_cap_tiers=True,

    # Fix 36: Quarter-End Window Dressing
    enable_quarter_end_optimizer=True,

    # Fix 37: Earnings Gap Trading
    enable_earnings_gap_trading=True,

    # Fix 38: Sector Rotation Momentum
    enable_sector_rotation=True,

    # Fix 39: VIX Term Structure Arbitrage
    enable_vix_term_structure=True,

    # Fix 40: Economic Data Reactions
    enable_economic_data_reactions=True,

    # Fix 41: Put/Call Ratio Reversals
    enable_put_call_ratio=True,
)

# Access components directly
optimizer.intraday_optimizer.get_timing_boost(...)
optimizer.market_cap_optimizer.get_position_adjustment(...)
optimizer.quarter_end_optimizer.get_quarter_end_adjustment(...)
optimizer.earnings_gap_trader.get_gap_adjustment(...)
optimizer.sector_rotation_momentum.get_rotation_adjustment(...)
optimizer.vix_term_structure.get_term_structure_adjustment(...)
optimizer.economic_data_reactor.get_economic_adjustment(...)
optimizer.put_call_analyzer.get_put_call_adjustment(...)

# Check configuration
config = optimizer.get_configuration_summary()
for fix in ['fix_34', 'fix_35', 'fix_36', 'fix_37', 'fix_38', 'fix_39', 'fix_40', 'fix_41']:
    print(f"{fix}: {config[fix]['enabled']}")
```

---

## Summary

| Fix | Problem | Solution | Impact |
|-----|---------|----------|--------|
| Fix 34 | Generic entry timing | Intraday window optimization | Up to 1.20x in power hour |
| Fix 35 | One-size-fits-all sizing | Market cap tier strategies | Mega-cap: 1.20x BUY boost |
| Fix 36 | Missing institutional patterns | Quarter-end window dressing | 1.25x for winners in last week |
| Fix 37 | Ignoring earnings gaps | Gap momentum/fade strategies | 1.25x for strong gaps |
| Fix 38 | Not tracking rotation | Sector rotation momentum | 1.35x in rotating-in sectors |
| Fix 39 | VIX structure ignored | Contango/backwardation signals | 1.15x in complacent markets |
| Fix 40 | Missing economic events | High-frequency data reactions | Event-specific adjustments |
| Fix 41 | Ignoring sentiment | Put/call contrarian signals | 1.40x at extreme readings |

All fixes in `src/models/us_intl_optimizer.py`.
China/DeepSeek model remains unchanged.
