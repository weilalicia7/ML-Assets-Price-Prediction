# Stock Prediction Model Documentation

**Last Updated**: 2025-11-27
**Model Version**: Phase 2C Cumulative
**Status**: Under Development

---

## Table of Contents
1. [Model Overview](#model-overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Performance Results](#performance-results)
5. [Data Configuration](#data-configuration)
6. [Limitations and Issues](#limitations-and-issues)
7. [Next Steps](#next-steps)

---

## Model Overview

### Description
Simple CatBoost-based stock price prediction model designed to predict next-day price movements (up/down) for Chinese stocks across Hong Kong, Shanghai, and Shenzhen exchanges.

### Key Characteristics
- **Algorithm**: CatBoost Classifier (gradient boosting)
- **Prediction Type**: Binary classification (price goes up = 1, down = 0)
- **Trading Strategy**: Only trade when model confidence ≥ 70%
- **Objective**: Maximize average return per trade (target: >0.5%)

### File Location
- **Main Model**: `src/models/simple_predictor.py`
- **Test Script**: `test_phase2c_cumulative.py`
- **Results**: `results/china_phase2c_cumulative_2025.json`

---

## Architecture

### Model Type
```
CatBoostClassifier
├── Iterations: 200
├── Learning Rate: 0.05
├── Tree Depth: 6
├── Random Seed: 42
└── Confidence Threshold: 0.7 (70%)
```

### Training Pipeline
```
1. Data Ingestion (Yahoo Finance)
   └── OHLCV data (Open, High, Low, Close, Volume)

2. Feature Engineering
   ├── 15 Base Features
   ├── 5 Interaction Features (Phase 2C.2)
   ├── 6 Lag Features (Phase 2C.3)
   └── 4 Trend Detection Features (Phase 2C.4)

3. Model Training
   └── CatBoost on training period (2025-01-01 to 2025-08-31)

4. Prediction
   ├── Generate probabilities for each day
   ├── Apply confidence threshold (0.7)
   └── Only predict "up" when P(up) ≥ 0.7

5. Evaluation
   └── Calculate returns on test period (2025-09-01 to 2025-11-26)
```

---

## Features

### Total Features: 30

#### 1. Base Features (15)
**Price-based features:**
- `returns_1d` - 1-day price return
- `returns_5d` - 5-day price return
- `rsi_14` - 14-day Relative Strength Index
- `sma_20` - 20-day Simple Moving Average
- `sma_50` - 50-day Simple Moving Average
- `price_momentum_10` - 10-day price momentum

**Volume-based features:**
- `volume_ratio` - Volume / 14-day average volume
- `volume_ma_14` - 14-day volume moving average
- `vwap` - Volume Weighted Average Price
- `vwap_ratio` - Close / VWAP
- `volume_momentum_10` - 10-day volume momentum

**Volatility features:**
- `volatility_14` - 14-day return standard deviation
- `atr_14` - 14-day Average True Range
- `bollinger_width` - Bollinger Band width (20-day)

**Position features:**
- `close_to_high` - Where close is relative to day's range
- `close_to_sma` - Distance from 20-day SMA

#### 2. Interaction Features (5) - Phase 2C.2
Capture non-linear relationships:
- `rsi_volume` = `rsi_14 × volume_ratio`
- `momentum_volatility` = `price_momentum_10 × volatility_14`
- `returns_trend` = `returns_1d × returns_5d`
- `price_volume_strength` = `close_to_high × volume_ratio`
- `sma_cross_signal` = `(sma_20 - sma_50) / sma_50`

#### 3. Lag Features (6) - Phase 2C.3
Temporal context from previous days:
- `returns_1d_lag1` - Previous day's return
- `returns_1d_lag2` - 2 days ago return
- `rsi_14_lag1` - Previous day's RSI
- `volume_ratio_lag1` - Previous day's volume ratio
- `volatility_14_lag1` - Previous day's volatility
- `returns_3d_avg` - 3-day average return

#### 4. Trend Detection Features (4) - Phase 2C.4
Identify trend direction and strength:
- `sma_20_slope` - Slope of 20-day SMA (5-day window)
- `sma_50_slope` - Slope of 50-day SMA (5-day window)
- `price_vs_sma20_trend` - How price trends vs SMA20 (5-day)
- `volatility_trend` - Change in volatility over 5 days

---

## Performance Results

### Latest Test: Phase 2C Cumulative (2025 Data)

**Test Configuration:**
- **Training Period**: 2025-01-01 to 2025-08-31 (~8 months)
- **Test Period**: 2025-09-01 to 2025-11-26 (~3 months)
- **Stock Universe**: 30 stocks (10 HK + 10 Shanghai + 10 Shenzhen)
- **Data Source**: Yahoo Finance (real-time)

### Overall Performance

| Metric | Value | Status |
|--------|-------|--------|
| **Average Return per Trade** | 0.21% | ❌ FAILED (target: >0.5%) |
| **Stocks Passing (>0.5%)** | 6/30 (20%) | ❌ Low pass rate |
| **Directional Accuracy** | 55.67% | ⚠️ Slightly above random |
| **Total Predictions** | 45 trades | ⚠️ Very low volume |
| **Comparison to Baseline** | -17.9% worse | ❌ Degradation |

### Performance by Market

#### Hong Kong (.HK) - Best Performance ✓
| Stock | Trades | Avg Return | Status |
|-------|--------|------------|--------|
| 0700.HK (Tencent) | 5 | 0.06% | FAIL |
| 9988.HK (Alibaba) | 3 | **0.72%** | ✓ PASS |
| 2319.HK (Mengniu) | 0 | 0.00% | FAIL |
| 1876.HK (BuddiesBoss) | 5 | 0.06% | FAIL |
| 0939.HK (CCB) | 2 | 0.43% | FAIL |
| 1398.HK (ICBC) | 1 | **1.41%** | ✓ PASS |
| 2269.HK (Wuxi Bio) | 5 | 0.02% | FAIL |
| 1177.HK (Sino Bio) | 5 | -0.04% | FAIL |
| 1109.HK (CR Land) | 3 | **0.79%** | ✓ PASS |
| 0960.HK (Longfor) | 4 | **1.33%** | ✓ PASS |
| **HK Summary** | **33 trades** | **0.38%** | **4/10 passed** |

#### Shanghai (.SS) - Moderate Performance
| Stock | Trades | Avg Return | Status |
|-------|--------|------------|--------|
| 600519.SS (Moutai) | 1 | -1.00% | FAIL |
| 601398.SS (ICBC-A) | 0 | 0.00% | FAIL |
| 600036.SS (CMB) | 0 | 0.00% | FAIL |
| 600887.SS (Yili) | 1 | **0.86%** | ✓ PASS |
| 601318.SS (Ping An) | 2 | 0.48% | FAIL |
| 600276.SS (Hengrui) | 1 | **1.95%** | ✓ PASS (Best!) |
| 601166.SS (Ind Bank) | 0 | 0.00% | FAIL |
| 600031.SS (Sany) | 1 | -0.34% | FAIL |
| 600028.SS (Sinopec) | 0 | 0.00% | FAIL |
| 601988.SS (BoC) | 0 | 0.00% | FAIL |
| **SS Summary** | **6 trades** | **0.33%** | **2/10 passed** |

#### Shenzhen (.SZ) - Worst Performance ❌
| Stock | Trades | Avg Return | Status |
|-------|--------|------------|--------|
| 000858.SZ (Wuliangye) | 1 | -0.97% | FAIL |
| 000333.SZ (Midea) | 0 | 0.00% | FAIL |
| 002415.SZ (Hikvision) | 1 | 0.13% | FAIL |
| 000001.SZ (Ping An Bank) | 2 | 0.48% | FAIL |
| 002594.SZ (BYD) | 0 | 0.00% | FAIL |
| 000651.SZ (Gree) | 2 | -0.22% | FAIL |
| 300750.SZ (CATL) | 0 | 0.00% | FAIL |
| 000002.SZ (Vanke) | 0 | 0.00% | FAIL |
| 002475.SZ (Luxshare) | 0 | 0.00% | FAIL |
| 000568.SZ (Luzhou) | 0 | 0.00% | FAIL |
| **SZ Summary** | **6 trades** | **-0.10%** | **0/10 passed** |

### Best Performing Stocks
1. **600276.SS** (Jiangsu Hengrui): 1.95% avg return (1 trade)
2. **1398.HK** (ICBC): 1.41% avg return (1 trade)
3. **0960.HK** (Longfor): 1.33% avg return (4 trades)
4. **600887.SS** (Yili): 0.86% avg return (1 trade)
5. **1109.HK** (CR Land): 0.79% avg return (3 trades)

---

## Data Configuration

### Training Data
```python
TRAIN_START = '2025-01-01'
TRAIN_END = '2025-08-31'
# Duration: ~8 months (~161-165 trading days)
```

### Test Data
```python
TEST_START = '2025-09-01'
TEST_END = '2025-11-26'
# Duration: ~3 months (~56-59 trading days)
```

### Stock Universe (30 stocks)

**Hong Kong (10):**
- 0700.HK (Tencent), 9988.HK (Alibaba), 2319.HK (Mengniu)
- 1876.HK (BuddiesBoss), 0939.HK (CCB), 1398.HK (ICBC)
- 2269.HK (Wuxi Biologics), 1177.HK (Sino Biopharm)
- 1109.HK (CR Land), 0960.HK (Longfor)

**Shanghai A-shares (10):**
- 600519.SS (Moutai), 601398.SS (ICBC), 600036.SS (CMB)
- 600887.SS (Yili), 601318.SS (Ping An), 600276.SS (Hengrui)
- 601166.SS (Industrial Bank), 600031.SS (Sany)
- 600028.SS (Sinopec), 601988.SS (Bank of China)

**Shenzhen A-shares (10):**
- 000858.SZ (Wuliangye), 000333.SZ (Midea), 002415.SZ (Hikvision)
- 000001.SZ (Ping An Bank), 002594.SZ (BYD), 000651.SZ (Gree)
- 300750.SZ (CATL), 000002.SZ (Vanke)
- 002475.SZ (Luxshare), 000568.SZ (Luzhou Laojiao)

---

## Limitations and Issues

### 1. Low Trade Volume ❌ CRITICAL
- **Problem**: Only 45 total predictions across 30 stocks (1.5 trades per stock)
- **Root Cause**: Confidence threshold of 0.7 is too restrictive
- **Impact**: 14 stocks generated 0 trades (no opportunity to profit)
- **Evidence**:
  - Many stocks: 0 predictions (can't evaluate model)
  - Best performers: Only 1-4 trades each (small sample size)

### 2. Market-Specific Performance Issues
- **Shenzhen A-shares**: 0/10 stocks passed (worst performance)
- **Possible Causes**:
  - Different market microstructure
  - Data quality issues with .SZ tickers
  - Model not capturing A-share specific patterns
  - Regulatory differences (T+1 settlement, price limits)

### 3. Short Training Period
- **Issue**: Only 8 months of training data (2025-01-08)
- **Impact**: May not capture sufficient market regimes
- **Comparison**: Typical ML models use 2-5 years of training data

### 4. Confidence Threshold Too High
- **Current**: 0.7 (70% confidence required)
- **Result**: Model rarely triggers trades
- **Trade-off**: Higher quality but insufficient quantity

### 5. Feature Engineering Paradox
- **Observation**: More features degraded performance
  - Phase 2C.0 (15 features): 0.25% avg return
  - Phase 2C Cumulative (30 features): 0.21% avg return (-17.9%)
- **Implication**: Additional features may introduce noise or overfitting

### 6. Model Calibration Issues
- **Problem**: Model probabilities may not be well-calibrated
- **Evidence**: Phase 2C.1 (confidence filtering) degraded performance from 0.25% to -0.13%
- **Impact**: High-confidence predictions are not necessarily higher quality

---

## Next Steps

### Immediate Investigations (Priority Order)

#### 1. Confidence Threshold Analysis
**Objective**: Find optimal threshold balancing quality vs quantity

**Test Plan**:
- Run same model with thresholds: 0.5, 0.55, 0.6, 0.65, 0.7
- Track: avg return, num trades, stocks passing
- Expected outcome: Lower threshold = more trades, find sweet spot

**Hypothesis**: 0.6 may be optimal (higher quality than 0.5, more trades than 0.7)

#### 2. Training Period Extension
**Objective**: Use more historical data for robust patterns

**Test Plan**:
- Option A: Train on 2024 data, test on 2025 (full year of training)
- Option B: Train on 2024-2025 combined, test on holdout 2025 period
- Track: generalization, overfitting, performance stability

**Hypothesis**: More training data will improve model robustness

#### 3. Market-Specific Models
**Objective**: Build separate models for HK, Shanghai, Shenzhen

**Test Plan**:
- Train 3 separate models (one per exchange)
- Each model learns market-specific patterns
- Evaluate if Shenzhen model improves on .SZ stocks

**Hypothesis**: Market structure differences require specialized models

#### 4. Feature Reduction
**Objective**: Return to simpler model (baseline)

**Test Plan**:
- Revert to 15 base features only (Phase 2C.0 configuration)
- Test on expanded universe (30 stocks)
- Compare to 30-feature cumulative model

**Hypothesis**: Simpler model may generalize better

#### 5. Data Quality Audit
**Objective**: Verify Shenzhen data integrity

**Test Plan**:
- Check for: missing data, extreme outliers, data gaps
- Compare data quality: .HK vs .SS vs .SZ
- Investigate why .SZ generated so few trades

**Hypothesis**: Data quality issues may explain poor .SZ performance

---

## Model Evolution History

### Phase 2C.0 - Baseline
- **Features**: 15 base features
- **Confidence**: 0.5 (default)
- **Result**: 0.25% avg return, 2/10 HK stocks passed
- **Status**: BASELINE

### Phase 2C.1 - Confidence Filtering
- **Change**: Increased confidence threshold to 0.7
- **Result**: -0.13% avg return, 0/10 passed
- **Decision**: REVERTED (but kept in cumulative)
- **Learning**: Model probabilities poorly calibrated

### Phase 2C.2 - Interaction Features
- **Change**: Added 5 interaction features (20 total)
- **Result**: -0.20% avg return, 2/10 passed
- **Decision**: REVERTED (but kept in cumulative)
- **Learning**: Interaction features degraded performance

### Phase 2C.3 - Lag Features
- **Change**: Added 6 lag features (26 total)
- **Result**: [Included in cumulative]
- **Decision**: Part of cumulative test

### Phase 2C.4 - Trend Features
- **Change**: Added 4 trend features (30 total)
- **Result**: [Included in cumulative]
- **Decision**: Part of cumulative test

### Phase 2C Cumulative - All Improvements
- **Features**: 30 total (all phases combined)
- **Confidence**: 0.7
- **Universe**: 30 stocks (expanded to Shanghai + Shenzhen)
- **Result**: 0.21% avg return, 6/30 passed
- **Status**: FAILED (-17.9% vs baseline)
- **Decision**: INVESTIGATE - Do not revert yet

---

## Technical Details

### Model Hyperparameters
```python
CatBoostClassifier(
    iterations=200,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=False,
    allow_writing_files=False
)
```

### Evaluation Metrics
```python
# Primary metric (profit-focused)
avg_return = mean(actual_returns[high_confidence_predictions])

# Success gate
passes_gate = avg_return > 0.005  # 0.5%

# Secondary metrics
accuracy = (predictions == actual).mean()
total_trades = sum(high_confidence_predictions)
```

### Feature Importance (Top 10 Average Across All Stocks)
Based on CatBoost feature importance scores:

1. `returns_1d_lag2` [LAG] - 6.82
2. `returns_1d` [BASE] - 6.45
3. `volume_ratio_lag1` [LAG] - 5.23
4. `returns_trend` [INTERACTION] - 5.18
5. `price_volume_strength` [INTERACTION] - 4.95
6. `volatility_trend` [TREND] - 4.87
7. `returns_1d_lag1` [LAG] - 4.76
8. `volume_ratio` [BASE] - 4.52
9. `sma_cross_signal` [INTERACTION] - 4.31
10. `close_to_high` [BASE] - 4.29

**Insights**:
- Lag features dominate top importance
- Interaction features also rank highly
- Suggests temporal patterns are valuable
- But more features still degraded performance (overfitting?)

---

## Files and Code Structure

### Main Model File
```
src/models/simple_predictor.py
├── Class: SimplePredictor
├── Methods:
│   ├── __init__() - Initialize with hyperparameters
│   ├── add_features() - Feature engineering (30 features)
│   ├── fit() - Train CatBoost model
│   ├── predict() - Generate predictions with confidence threshold
│   ├── evaluate() - Calculate profit metrics
│   └── get_feature_importance() - Feature importance scores
```

### Test Script
```
test_phase2c_cumulative.py
├── CHINA_STOCKS - 30 stock tickers
├── Data periods (2025 train/test)
├── fetch_data() - Download from Yahoo Finance
├── test_stock() - Test single stock
└── main() - Run full test suite
```

### Results Files
```
results/
├── china_phase2c_cumulative_2025.json - Full test results
└── PHASE2C_CUMULATIVE_2025.log - Detailed execution log
```

---

## Conclusion

### Current Status: UNDER INVESTIGATION ⚠️

The Phase 2C cumulative model with 30 features shows:
- **Strengths**:
  - Some stocks perform well (4 HK stocks passed)
  - Best individual returns up to 1.95%
  - Model can identify profitable trades when confident

- **Weaknesses**:
  - Overall avg return (0.21%) below profit threshold (0.5%)
  - Very low trade volume (45 total trades)
  - Poor performance on Shenzhen A-shares (0/10 passed)
  - More features degraded performance vs baseline

### Recommendation: DO NOT DEPLOY YET

**Before deployment, must investigate**:
1. Why does confidence=0.7 generate so few trades?
2. Why do Shenzhen A-shares perform poorly?
3. Would simpler model (15 features) generalize better?
4. Is 8 months of training data sufficient?

**Next milestone**: Complete 5 priority investigations, then decide:
- DEPLOY if avg return >0.5% and robust across markets
- REVERT to baseline if complex model cannot improve
- PIVOT to different approach if current architecture is fundamentally flawed

---

**Document Version**: 1.0
**Created By**: Claude Code
**Date**: 2025-11-27
