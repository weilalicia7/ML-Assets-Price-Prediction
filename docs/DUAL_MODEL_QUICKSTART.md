# Dual Model System - Quick Start Guide

## Overview

The webapp now supports **two separate prediction models**:

1. **China Model** - For Chinese stocks (*.HK, *.SS, *.SZ)
   - Uses China-specific macro features: CSI300, CNY/USD, HSI
   - Expected profitability: **+71% annually** (based on backtest)
   - Trained on Chinese A-shares and Hong Kong stocks

2. **US/International Model** - For US and other stocks
   - Uses US macro features: VIX, SPY, DXY, GLD
   - Expected profitability: **+22% annually** (based on backtest)
   - Trained on US and international stocks

## How It Works

### Automatic Market Detection

The system automatically detects the market based on the ticker suffix:

```python
# Chinese stocks (use China Model)
0700.HK   # Tencent - Hong Kong
9988.HK   # Alibaba - Hong Kong
600519.SS # Moutai - Shanghai
000001.SZ # Ping An - Shenzhen

# US stocks (use US/Intl Model)
AAPL      # Apple
TSLA      # Tesla
MSFT      # Microsoft

# International stocks (use US/Intl Model)
BP.L      # BP - London
SAP       # SAP - Germany
TM        # Toyota - Japan
```

### Feature Engineering

The system uses **different macro features** for each market:

**China Model Features:**
- CSI300 Index (China's S&P 500 equivalent)
- CNY/USD Exchange Rate
- Hang Seng Index (HSI)

**US/Intl Model Features:**
- VIX (Volatility Index)
- SPY (S&P 500 ETF)
- DXY (US Dollar Index)
- GLD (Gold ETF)

Both models use the same technical and volatility features.

## Using the Webapp

### 1. Start the Webapp

```bash
cd stock-prediction-model
python webapp.py
```

The webapp will start on `http://localhost:5001`

### 2. Check Market Classification

Use the new `/api/market_info/<ticker>` endpoint to see which model will be used:

```bash
# Check Tencent (Chinese stock)
curl http://localhost:5001/api/market_info/0700.HK

# Response:
{
  "ticker": "0700.HK",
  "dual_model_system": true,
  "market": "chinese",
  "exchange": "HKG",
  "model_type": "China Model",
  "macro_features": "CSI300, CNY, HSI",
  "expected_profitability": "High (71% annual return in backtest)",
  "confidence": "Very High",
  "recommendation": "Recommended for trading"
}
```

```bash
# Check Apple (US stock)
curl http://localhost:5001/api/market_info/AAPL

# Response:
{
  "ticker": "AAPL",
  "dual_model_system": true,
  "market": "us_international",
  "exchange": "NMS",
  "model_type": "US/International Model",
  "macro_features": "VIX, SPY, DXY, GLD",
  "expected_profitability": "Medium (22% annual return in backtest)",
  "confidence": "High",
  "recommendation": "Suitable for trading"
}
```

### 3. Get Predictions

Use the standard `/api/predict/<ticker>` endpoint:

```bash
# Predict Tencent (will use China Model)
curl "http://localhost:5001/api/predict/0700.HK?account_size=100000"

# Predict Apple (will use US/Intl Model)
curl "http://localhost:5001/api/predict/AAPL?account_size=100000"
```

The prediction response includes:
- Signal (BUY/SELL/HOLD)
- Direction and confidence
- Predicted return
- Position size and shares
- Risk metrics

## Configuration

### Enable/Disable Dual Model System

Edit `webapp.py` line 238:

```python
# Enable dual model system (default)
USE_DUAL_MODEL_SYSTEM = True

# Disable dual model system (use only US/Intl model for all markets)
USE_DUAL_MODEL_SYSTEM = False
```

### Model Training

The first time you predict a Chinese stock, the system will:
1. Fetch CSI300, CNY/USD, and HSI data
2. Engineer China-specific features
3. Train the China model (takes 5-10 minutes)
4. Cache the model for future use

Subsequent predictions will use the cached model.

## Testing

### Run Automated Tests

```bash
cd stock-prediction-model
python test_webapp_dual_model.py
```

This will test:
- Health check
- Market classification for 11 tickers (Chinese, US, International)
- Predictions for Chinese and US stocks

### Manual Testing

1. **Test Market Classification**
   ```bash
   # Chinese stocks
   curl http://localhost:5001/api/market_info/0700.HK
   curl http://localhost:5001/api/market_info/9988.HK

   # US stocks
   curl http://localhost:5001/api/market_info/AAPL
   curl http://localhost:5001/api/market_info/TSLA
   ```

2. **Test Predictions**
   ```bash
   # Chinese stock (China Model)
   curl "http://localhost:5001/api/predict/0700.HK?account_size=100000"

   # US stock (US/Intl Model)
   curl "http://localhost:5001/api/predict/AAPL?account_size=100000"
   ```

3. **Check Logs**
   Look for these log messages:
   ```
   [FEATURES] Using China pipeline for 0700.HK (CSI300, CNY, HSI)
   [DUAL MODEL] Using China model for 0700.HK
   [DUAL MODEL] China model prediction: 0.0234
   ```

## Expected Performance

Based on backtests (see `results/china_model_backtest_20251124.json`):

| Model | Market | Annual Return | Sharpe Ratio | Win Rate |
|-------|--------|--------------|--------------|----------|
| China Model | Chinese stocks | +71% | 1.18 | 57% |
| US/Intl Model | US/International | +22% | 0.85 | 54% |

**Note:** Past performance does not guarantee future results. These are backtest results.

## Troubleshooting

### Issue: "Failed to add China macro features"

**Solution:** The system will fall back to technical features only. Check:
1. Internet connection (needs to fetch CSI300, CNY, HSI)
2. Yahoo Finance availability
3. Date range (need sufficient historical data)

### Issue: "China model failed, falling back to US/Intl"

**Solution:** The system will use the US/Intl model instead. Check:
1. Model file exists: `models/china_market_model.pkl`
2. Model training logs for errors
3. Feature compatibility (model expects specific features)

### Issue: Predictions are slow for Chinese stocks

**Explanation:** First prediction trains the model (5-10 minutes). Subsequent predictions are fast (1-2 seconds).

**Solution:** Pre-train the model:
```python
from src.models.china_predictor import ChinaMarketPredictor

predictor = ChinaMarketPredictor()
predictor.train(['0700.HK', '9988.HK', '600519.SS'])
```

## Architecture Notes

### Code Changes

The integration involved:

1. **Feature Engineering** (`webapp.py:58-129`)
   - New function: `engineer_market_specific_features(df, ticker)`
   - Routes to China or US/Intl feature pipeline

2. **Model Router** (`webapp.py:242-263`)
   - New function: `get_model_router()`
   - Initializes dual model system

3. **Prediction Routing** (`webapp.py:1562-1594`)
   - Modified prediction logic to route based on market
   - Falls back to US/Intl model on errors

4. **New Endpoint** (`webapp.py:2132-2145`)
   - `/api/market_info/<ticker>` - Market classification and performance info

### File Structure

```
stock-prediction-model/
├── src/
│   ├── models/
│   │   ├── china_predictor.py          # China model (new)
│   │   ├── market_classifier.py         # Market detection (new)
│   │   └── model_router.py              # Model routing (new)
│   └── features/
│       ├── china_macro_features.py      # CSI300, CNY, HSI (new)
│       └── selective_macro_features.py  # VIX, SPY, DXY, GLD (new)
├── webapp.py                             # Main webapp (modified)
├── test_webapp_dual_model.py            # Integration tests (new)
└── DUAL_MODEL_QUICKSTART.md             # This guide (new)
```

## Next Steps

1. **Monitor Performance**
   - Track predictions for Chinese vs US stocks
   - Compare actual vs predicted returns
   - Adjust confidence thresholds if needed

2. **Expand Coverage**
   - Add more Chinese stocks to training data
   - Consider regional models (Europe, Asia-Pacific)
   - Implement sector-specific models

3. **Optimize Features**
   - Test additional China macro indicators
   - Experiment with feature combinations
   - Implement feature selection

## Contact

For issues or questions, please check the main README or open an issue on GitHub.

---

**Last Updated:** November 24, 2025
**Version:** 1.0
**Status:** Production Ready
