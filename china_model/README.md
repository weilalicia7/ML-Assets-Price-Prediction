# China Market Prediction Platform

## Overview

Production-ready scalable China market prediction system with:
- **219 instruments** screened across HK large/mid/small caps + A-shares
- **13 robust performers** identified through 3-tier validation
- **6-position diversified portfolio** with 10.1% target volatility
- **DeepSeek API integration** for sentiment-based threshold adjustment

## Latest Results (November 2025)

### Robust Performers

| Symbol | Name | Sector | Return | Pass Rate | Market Cap |
|--------|------|--------|--------|-----------|------------|
| 1801.HK | Innovent Biologics | Healthcare | 1.49% | 100% | Mid-cap |
| 0358.HK | Jiangxi Copper | Materials | 1.22% | 100% | Mid-cap |
| 300274.SZ | Sungrow Power | Industrials | 1.21% | 80% | A-share |
| 9999.HK | NetEase | Tech | 1.17% | 100% | Large-cap |
| 1530.HK | 3SBio | Healthcare | 1.08% | 60% | Small-cap |
| 1357.HK | Meitu | Tech | 1.07% | 100% | Small-cap |
| 6160.HK | BeiGene | Healthcare | 0.98% | 100% | Large-cap |
| 6869.HK | Yuexiu Property | Real Estate | 0.87% | 80% | Small-cap |
| 3993.HK | CMOC Group | Materials | 0.80% | 100% | Large-cap |
| 300274.SZ | Sungrow Power | Industrials | 0.74% | 80% | A-share |
| 2628.HK | China Life | Financials | 0.61% | 80% | Large-cap |
| 0175.HK | Geely Auto | Consumer | 0.52% | 80% | Large-cap |
| 2600.HK | Aluminum Corp | Materials | 0.51% | 60% | Large-cap |

### Current Portfolio

| Symbol | Sector | Weight | Signal |
|--------|--------|--------|--------|
| 9999.HK | Tech | 20.0% | BUY |
| 0358.HK | Materials | 9.8% | BUY |
| 300274.SZ | Industrials | 9.6% | HOLD |
| 0175.HK | Consumer | 8.5% | HOLD |
| 2600.HK | Materials | 5.3% | HOLD |
| 6869.HK | Real Estate | 3.1% | HOLD |
| CASH | - | 43.8% | - |

**Risk Metrics:** Vol 10.1% | VaR95 -1.51% | Max DD -20.31%

## Architecture

```
china_model/
├── src/                              # Core production code
│   ├── china_stock_screener.py       # Universe builder (219 instruments)
│   │   ├── HK_LARGE_CAPS (55)        # Proven alpha source
│   │   ├── HK_MID_CAPS (50)          # New alpha (Innovent, Jiangxi)
│   │   ├── HK_SMALL_CAPS (42)        # New alpha (Meitu, 3SBio)
│   │   ├── SHANGHAI_A (33)           # Policy-sensitive
│   │   └── SHENZHEN_A (22)           # Retail-dominated
│   ├── tiered_screener.py            # 3-tier screening pipeline
│   │   ├── Market-adjusted thresholds
│   │   ├── HK enhanced features
│   │   └── A-share specific features
│   ├── model_factory.py              # Automated model building
│   │   ├── Strategy detection
│   │   ├── Ensemble models (5 seeds)
│   │   └── 30+ features
│   └── portfolio_constructor.py      # Portfolio & risk management
│       ├── Inverse volatility weighting
│       ├── Signal strength weighting
│       └── Sector exposure limits
├── models/                           # Trained models
│   ├── production_models.pkl
│   └── production_models_metadata.json
├── results/                          # JSON outputs
│   ├── tiered_screening_results.json
│   └── current_portfolio.json
├── tests/                            # Test scripts
└── docs/                             # Documentation
```

## Quick Start

### 1. Full Pipeline (Screen -> Model -> Portfolio)
```bash
cd china_model/src
python tiered_screener.py      # Find robust performers
python model_factory.py        # Build production models
python portfolio_constructor.py # Construct portfolio
```

### 2. Daily Update
```python
from portfolio_constructor import PortfolioConstructor
constructor = PortfolioConstructor()
constructor.daily_update()
```

### 3. Get Current Signals
```python
from model_factory import ModelFactory
factory = ModelFactory()
factory.load_models()
for symbol in factory.models:
    print(factory.predict(symbol))
```

## Tiered Screening Pipeline

| Tier | Seeds | Features | HK Threshold | A-Share Threshold |
|------|-------|----------|--------------|-------------------|
| Quick | 1 | Basic (9) | 0% | 0% |
| Medium | 3 | Full (21-33) | 50% | 35-40% |
| Deep | 5 | Full (21-33) | 60% | 40-45% |

**Results:** 219 screened -> 101 liquid -> 55 promising -> 15 medium -> 13 robust

## Feature Sets

### HK Enhanced Features (33 total)
- Base features (21): returns, RSI, MA, volume, volatility, momentum
- Institutional flow proxy: volume_persistence, volume_price_diverge, ad_proxy
- Intraday patterns: open_strength, close_strength, range_position
- Sector rotation: price_position_60d, breakout_up/down

### A-Share Specific Features (33 total)
- Base features (21)
- Retail flow: turnover_spike, volume_accel, panic_indicator
- Policy sensitivity: overnight_gap, near_limit
- Mean reversion: rsi_extreme, consec_down/up, momentum_divergence

## Market-Adjusted Thresholds

| Market | Medium Pass | Deep Pass | Rationale |
|--------|-------------|-----------|-----------|
| HK Stock | 50% | 60% | Institutional, efficient |
| HK ETF | 50% | 55% | Diversified, stable |
| Shanghai A | 40% | 45% | Policy-driven, SOE heavy |
| Shenzhen A | 35% | 40% | Retail-dominated, tech/growth |

## Anti-Overfitting Measures

1. **5 Random Seeds** per instrument for robustness
2. **Strict Train/Test Split** (2023-2024 train, 2025 test)
3. **Market-Adjusted Thresholds** (stricter for efficient markets)
4. **Conservative Hyperparameters** (depth 4-6, iter 100-200)
5. **Liquidity Filter** ($500M+ daily volume)
6. **3-Tier Validation** (quick -> medium -> deep)

## Key Findings

### What Works
- **HK large/mid/small caps**: Clean institutional signals
- **Healthcare/Biotech**: 1801.HK (1.49%), 1530.HK (1.08%)
- **Materials/Mining**: 0358.HK (1.22%), 3993.HK (0.80%)
- **Export-focused A-shares**: 300274.SZ (Sungrow - solar inverters)

### What Doesn't Work
- **Shanghai A-shares**: 0% pass rate even with relaxed thresholds
- **Policy-sensitive stocks**: Too much noise from government intervention
- **Low liquidity instruments**: Insufficient data for robust signals

## DeepSeek Integration

API used for confidence threshold adjustment:
```bash
DEEPSEEK_API_KEY="your_key" python tests/test_week8_phase4_5_deepseek_china.py
```

Sentiment sources:
- Policy sentiment (PBOC/CSRC)
- Social sentiment (Weibo/Xueqiu)
- Retail sentiment (trading patterns)
- Sector policy alignment (Five-Year Plan)

## Performance Summary

| Metric | Value |
|--------|-------|
| Universe Screened | 219 |
| Liquid Instruments | 101 (46.1%) |
| Robust Performers | 13 (12.9%) |
| Compute Savings | 85.1% |
| Portfolio Volatility | 10.1% |
| VaR (95%, 1-day) | -1.51% |
| Max Drawdown | -20.31% |

## Next Phase: Live Implementation

### Phase 2: Real-Time Trading
- Real-time data integration
- Automated position sizing
- Portfolio rebalancing engine
- Performance attribution system

### Phase 3: Edge Enhancement
- Alternative data (news, sentiment)
- Multi-timeframe strategy stacking
- Cross-asset correlation signals
- Dynamic risk allocation

### Phase 4: Market Expansion
- Taiwan (TWSE)
- Korea (KOSPI)
- Singapore (SGX)
- ASEAN markets

---

*Last updated: November 27, 2025*
*Platform version: 2.0 (Production Ready)*
