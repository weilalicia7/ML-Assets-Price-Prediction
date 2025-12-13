# China Market Prediction Platform - Executive Summary

## Project Overview

A production-ready quantitative trading system for China markets, developed through rigorous machine learning and anti-overfitting validation.

**Development Period:** November 2025
**Platform Version:** 2.0 (Production Ready)

---

## Key Achievements

| Metric | Value |
|--------|-------|
| Total Universe | 219 instruments |
| Liquid Instruments | 101 (46.1%) |
| Robust Performers | 13 (12.9%) |
| Portfolio Positions | 6 |
| Portfolio Volatility | 10.1% |
| Compute Savings | 85.1% |

---

## Platform Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CHINA MODEL PLATFORM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  UNIVERSE   │───>│   TIERED    │───>│    MODEL    │         │
│  │   BUILDER   │    │  SCREENER   │    │   FACTORY   │         │
│  │  (219 stk)  │    │  (3-tier)   │    │  (ensemble) │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                  │                  │                 │
│         v                  v                  v                 │
│  ┌─────────────────────────────────────────────────┐           │
│  │              PORTFOLIO CONSTRUCTOR              │           │
│  │   • Inverse volatility weighting                │           │
│  │   • Signal strength allocation                  │           │
│  │   • Sector exposure limits                      │           │
│  │   • Risk analytics (VaR, Max DD)                │           │
│  └─────────────────────────────────────────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Universe Coverage

### Hong Kong (Main Alpha Source)

| Category | Count | Liquid | Robust | Top Performer |
|----------|-------|--------|--------|---------------|
| Large Caps | 55 | 33 | 5 | 9999.HK (NetEase) |
| Mid Caps | 50 | 8 | 4 | 1801.HK (Innovent) |
| Small Caps | 42 | 6 | 4 | 1357.HK (Meitu) |
| **Total HK** | **147** | **47** | **13** | |

### A-Shares (Selective Opportunities)

| Category | Count | Liquid | Robust | Top Performer |
|----------|-------|--------|--------|---------------|
| Shanghai A | 33 | 30 | 0 | - |
| Shenzhen A | 22 | 21 | 1 | 300274.SZ (Sungrow) |
| **Total A** | **55** | **51** | **1** | |

### Other

| Category | Count | Liquid | Robust |
|----------|-------|--------|--------|
| HK ETFs | 8 | 3 | 0 |
| Futures | 9 | 0 | 0 |

---

## Robust Performers (Final 13)

### Tier 1: Highest Alpha (>1.0% return)

| Rank | Symbol | Name | Sector | Return | Pass Rate |
|------|--------|------|--------|--------|-----------|
| 1 | 1801.HK | Innovent Biologics | Healthcare | 1.49% | 100% |
| 2 | 0358.HK | Jiangxi Copper | Materials | 1.22% | 100% |
| 3 | 300274.SZ | Sungrow Power | Industrials | 1.21% | 80% |
| 4 | 9999.HK | NetEase | Tech | 1.17% | 100% |
| 5 | 1530.HK | 3SBio | Healthcare | 1.08% | 60% |
| 6 | 1357.HK | Meitu | Tech | 1.07% | 100% |

### Tier 2: Strong Alpha (0.5-1.0% return)

| Rank | Symbol | Name | Sector | Return | Pass Rate |
|------|--------|------|--------|--------|-----------|
| 7 | 6160.HK | BeiGene | Healthcare | 0.98% | 100% |
| 8 | 6869.HK | Yuexiu Property | Real Estate | 0.87% | 80% |
| 9 | 3993.HK | CMOC Group | Materials | 0.80% | 100% |
| 10 | 2628.HK | China Life | Financials | 0.61% | 80% |
| 11 | 0175.HK | Geely Auto | Consumer | 0.52% | 80% |
| 12 | 2600.HK | Aluminum Corp | Materials | 0.51% | 60% |

---

## Current Portfolio

### Allocation

| Symbol | Name | Sector | Weight | Signal | Expected Return |
|--------|------|--------|--------|--------|-----------------|
| 9999.HK | NetEase | Tech | 20.0% | BUY | 1.17% |
| 0358.HK | Jiangxi Copper | Materials | 9.8% | BUY | 1.22% |
| 300274.SZ | Sungrow Power | Industrials | 9.6% | HOLD | 1.21% |
| 0175.HK | Geely Auto | Consumer | 8.5% | HOLD | 0.84% |
| 2600.HK | Aluminum Corp | Materials | 5.3% | HOLD | 0.61% |
| 6869.HK | Yuexiu Property | Real Estate | 3.1% | HOLD | 0.49% |
| **CASH** | - | - | **43.8%** | - | 0% |

### Sector Diversification

```
Tech          ████████████████████  20.0%
Materials     ███████████████       15.1%
Industrials   ██████████            9.6%
Consumer      █████████             8.5%
Real Estate   ███                   3.1%
Cash          ████████████████████████████████████████████  43.8%
```

### Risk Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| Portfolio Volatility | 10.1% | Conservative |
| VaR (95%, 1-day) | -1.51% | Well-controlled |
| VaR (99%, 1-day) | -2.30% | Acceptable |
| Max Drawdown (1Y) | -20.31% | Moderate |
| Positions | 6 | Diversified |
| Cash Buffer | 43.8% | Defensive |

---

## Screening Methodology

### 3-Tier Validation Pipeline

```
TIER 1: QUICK SCREEN          TIER 2: MEDIUM           TIER 3: DEEP
─────────────────────         ─────────────            ────────────
• 1 seed                      • 3 seeds                • 5 seeds
• Basic features (9)          • Full features (21-33)  • Full features
• Threshold: 0%               • Threshold: 35-50%      • Threshold: 40-60%
• ALL liquid stocks           • Promising only         • Best candidates

     101 stocks          →         55 stocks       →       13 ROBUST
        (54.5% pass)                (27.3% pass)           (86.7% pass)
```

### Market-Adjusted Thresholds

| Market | Medium Tier | Deep Tier | Rationale |
|--------|-------------|-----------|-----------|
| HK Stock | 50% | 60% | Institutional, efficient pricing |
| HK ETF | 50% | 55% | Diversified, lower noise |
| Shanghai A | 40% | 45% | Policy-driven, SOE influence |
| Shenzhen A | 35% | 40% | Retail-dominated, high noise |

---

## Feature Engineering

### Base Features (21)

| Category | Features |
|----------|----------|
| Returns | 1d, 5d, 10d, 20d |
| RSI | 14-period |
| Moving Averages | SMA 20, 50, cross signals |
| Volume | ratio, trend, price correlation |
| Volatility | 10d, 20d, ratio |
| Momentum | 5d, 10d, 20d |
| Bollinger | %B, width |
| Other | distance from mean, trend strength |

### HK Enhanced Features (+12)

| Category | Features | Purpose |
|----------|----------|---------|
| Institutional Flow | volume_persistence, volume_price_diverge, ad_proxy | Smart money detection |
| Intraday Patterns | open_strength, close_strength, range_position | Sentiment capture |
| Sector Rotation | price_position_60d, breakout_up/down, high_vol_regime | Trend identification |

### A-Share Specific Features (+12)

| Category | Features | Purpose |
|----------|----------|---------|
| Retail Flow | turnover_spike, volume_accel, panic_indicator | Retail behavior |
| Policy Sensitivity | overnight_gap, gap_magnitude, near_limit | Policy reaction |
| Mean Reversion | rsi_extreme, consec_down/up, momentum_divergence | Oversold/overbought |

---

## Anti-Overfitting Measures

| Measure | Implementation | Impact |
|---------|----------------|--------|
| Multi-Seed Validation | 5 random seeds per stock | Ensures reproducibility |
| Train/Test Split | 2023-2024 train, 2025 test | No look-ahead bias |
| Robustness Threshold | 60% pass rate required | Filters lucky results |
| Conservative Hyperparameters | depth 4-6, iter 100-200 | Prevents overfitting |
| Liquidity Filter | $500M+ daily volume | Quality data only |
| 3-Tier Validation | Quick → Medium → Deep | Progressive filtering |

---

## Key Findings

### What Works

1. **HK Markets** - Clean institutional signals, predictable patterns
2. **Healthcare/Biotech** - 1801.HK (1.49%), 1530.HK (1.08%), 6160.HK (0.98%)
3. **Materials/Mining** - 0358.HK (1.22%), 3993.HK (0.80%), 2600.HK (0.51%)
4. **Export-focused A-shares** - 300274.SZ (Sungrow - global solar leader)
5. **Mid/Small Caps** - Higher alpha potential than large caps

### What Doesn't Work

1. **Shanghai A-shares** - 0% pass rate, too policy-sensitive
2. **Shenzhen retail stocks** - High noise, low predictability
3. **Low liquidity instruments** - Insufficient data quality
4. **Futures proxies** - None passed liquidity filter

### Strategic Insights

| Insight | Evidence |
|---------|----------|
| HK > A-shares for ML | 12 HK robust vs 1 A-share |
| Mid-caps have edge | Innovent, Jiangxi Copper top performers |
| Sector matters | Healthcare, Materials outperform |
| Liquidity = quality | All robust performers >$500M/day |

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| ML Framework | CatBoost |
| Data Source | Yahoo Finance (yfinance) |
| AI Integration | DeepSeek API |
| Language | Python 3.12 |
| Key Libraries | pandas, numpy, scikit-learn |

---

## File Structure

```
china_model/
├── src/
│   ├── china_stock_screener.py    # Universe builder (219 instruments)
│   ├── tiered_screener.py         # 3-tier screening pipeline
│   ├── model_factory.py           # Automated model building
│   └── portfolio_constructor.py   # Portfolio & risk management
├── models/
│   ├── production_models.pkl      # Trained ensemble models
│   └── production_models_metadata.json
├── results/
│   ├── tiered_screening_results.json
│   └── current_portfolio.json
├── tests/
│   ├── test_week8_phase4_5_deepseek_china.py
│   ├── test_week8_phase4_6_multi_domain_china.py
│   └── test_week8_phase4_7_robust_performer_analysis.py
├── docs/
│   └── CHINA_MULTI_DOMAIN_ANALYSIS_REPORT.md
├── README.md
└── CHINA_MODEL_SUMMARY.md         # This file
```

---

## Usage

### Full Pipeline
```bash
cd china_model/src
python tiered_screener.py       # Screen universe
python model_factory.py         # Build models
python portfolio_constructor.py # Construct portfolio
```

### Daily Update
```python
from portfolio_constructor import PortfolioConstructor
constructor = PortfolioConstructor()
result = constructor.daily_update()
print(result['portfolio'])
```

### Get Signals
```python
from model_factory import ModelFactory
factory = ModelFactory()
factory.load_models()
for symbol in factory.models:
    signal = factory.predict(symbol)
    print(f"{symbol}: {signal['signal']} ({signal['probability']:.1%})")
```

---

## Next Steps

### Phase 2: Live Trading
- Real-time data integration
- Automated execution
- Performance monitoring

### Phase 3: Enhancement
- Alternative data (news, sentiment)
- Multi-timeframe strategies
- Dynamic risk allocation

### Phase 4: Expansion
- Taiwan (TWSE)
- Korea (KOSPI)
- Singapore (SGX)

---

## Conclusion

The China Model Platform successfully identifies **13 robust performers** from a universe of **219 instruments**, achieving:

- **Systematic alpha discovery** across market caps
- **Professional risk management** with 10.1% portfolio volatility
- **Scalable architecture** ready for live deployment
- **Rigorous validation** through 3-tier screening and 5-seed testing

The platform demonstrates that **HK markets offer cleaner signals** than A-shares, with **Healthcare and Materials sectors** providing the strongest alpha opportunities.

---

*Generated: November 27, 2025*
*Platform: China Market Prediction Platform v2.0*
