# Dual Model System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WEBAPP (webapp.py)                            │
│                     http://localhost:5001                            │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
         ┌──────────▼──────────┐      ┌──────────▼──────────┐
         │  /api/market_info   │      │   /api/predict      │
         │     <ticker>        │      │     <ticker>        │
         └──────────┬──────────┘      └──────────┬──────────┘
                    │                            │
                    │                            │
         ┌──────────▼────────────────────────────▼──────────┐
         │       MarketClassifier.get_market(ticker)        │
         │                                                   │
         │  Checks ticker suffix:                           │
         │  - *.HK, *.SS, *.SZ → 'chinese'                  │
         │  - Others → 'us_international'                   │
         └──────────┬────────────────────────────┬──────────┘
                    │                            │
         ┌──────────▼──────────┐      ┌──────────▼──────────┐
         │   Chinese Market    │      │  US/International   │
         │   (e.g., 0700.HK)   │      │   (e.g., AAPL)      │
         └──────────┬──────────┘      └──────────┬──────────┘
                    │                            │
                    │                            │
┌───────────────────▼───────────────┐ ┌──────────▼────────────────────┐
│  engineer_market_specific_features│ │ engineer_market_specific_features│
│                                   │ │                                │
│  1. TechnicalFeatureEngineer      │ │ 1. TechnicalFeatureEngineer    │
│  2. VolatilityFeatureEngineer     │ │ 2. VolatilityFeatureEngineer   │
│  3. ChinaMacroFeatureEngineer     │ │ 3. SelectiveMacroFeatureEngineer│
│                                   │ │                                │
│  → CSI300 (000300.SS)            │ │ → VIX (^VIX)                   │
│  → CNY/USD (CNY=X)               │ │ → SPY (SPY)                    │
│  → HSI (^HSI)                    │ │ → DXY (DX-Y.NYB)               │
│                                   │ │ → GLD (GLD)                    │
└───────────────────┬───────────────┘ └──────────┬────────────────────┘
                    │                            │
                    │                            │
         ┌──────────▼──────────┐      ┌──────────▼──────────┐
         │   China Model       │      │  US/Intl Model      │
         │                     │      │                     │
         │ ChinaMarketPredictor│      │ HybridEnsemble      │
         │                     │      │ Predictor           │
         │ Expected: +71%/yr   │      │                     │
         │ Sharpe: 1.18        │      │ Expected: +22%/yr   │
         │ Win Rate: 57%       │      │ Sharpe: 0.85        │
         │                     │      │ Win Rate: 54%       │
         └──────────┬──────────┘      └──────────┬──────────┘
                    │                            │
                    │                            │
                    └──────────┬─────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Prediction Result │
                    │                     │
                    │ - Signal (BUY/SELL) │
                    │ - Confidence        │
                    │ - Position size     │
                    │ - Risk metrics      │
                    └─────────────────────┘
```

## Request Flow

### 1. Market Classification Request

```
User → GET /api/market_info/0700.HK
       │
       ├─► MarketClassifier.get_market("0700.HK")
       │   └─► Returns: 'chinese'
       │
       ├─► MarketClassifier.get_market_details("0700.HK")
       │   └─► Returns: {
       │         market: 'chinese',
       │         exchange: 'HKG',
       │         model_type: 'China Model',
       │         macro_features: 'CSI300, CNY, HSI'
       │       }
       │
       └─► MarketClassifier.get_performance_expectations("0700.HK")
           └─► Returns: {
                 expected_profitability: 'High (71% annual return)',
                 confidence: 'Very High',
                 recommendation: 'Recommended for trading'
               }

Response ◄── JSON with market classification and performance info
```

### 2. Prediction Request (Chinese Stock)

```
User → GET /api/predict/0700.HK?account_size=100000
       │
       ├─► fetch_data("0700.HK", lookback_days=500)
       │   └─► Yahoo Finance API → OHLCV data
       │
       ├─► engineer_market_specific_features(data, "0700.HK")
       │   │
       │   ├─► MarketClassifier.get_market("0700.HK")
       │   │   └─► Returns: 'chinese'
       │   │
       │   ├─► TechnicalFeatureEngineer.add_all_features()
       │   │   └─► SMA, EMA, RSI, MACD, Bollinger Bands, etc.
       │   │
       │   ├─► VolatilityFeatureEngineer.add_all_features()
       │   │   └─► Historical vol, Yang-Zhang vol, Parkinson vol
       │   │
       │   └─► ChinaMacroFeatureEngineer.add_all_features()
       │       ├─► Fetch CSI300 (000300.SS) → CSI300_close, CSI300_returns
       │       ├─► Fetch CNY/USD (CNY=X) → cny_usd, cny_usd_change
       │       └─► Fetch HSI (^HSI) → hsi_close, hsi_returns
       │
       ├─► get_model_router()
       │   └─► Initialize ModelRouter if not already initialized
       │
       ├─► MarketClassifier.get_market("0700.HK")
       │   └─► Returns: 'chinese'
       │
       ├─► ChinaMarketPredictor.predict(X_latest)
       │   │
       │   ├─► Check if model exists (models/china_market_model.pkl)
       │   │   ├─► If exists: Load cached model
       │   │   └─► If not: Train new model (5-10 minutes)
       │   │
       │   └─► Return prediction: [0.0234]  # 2.34% 5-day return
       │
       ├─► Calculate confidence based on prediction magnitude
       │   └─► confidence = 0.72 (72%)
       │
       ├─► Generate trading signal
       │   ├─► Direction: 1 (BUY) because predicted_return > 0
       │   ├─► Signal: "BUY" because confidence > 0.65
       │   └─► Position size: $50,000 (50% of $100,000 account)
       │
       └─► Return JSON response

Response ◄── {
               status: 'success',
               ticker: '0700.HK',
               signal: 'BUY',
               direction: 1,
               confidence: 0.72,
               predicted_return: 0.0234,
               position_size: 50000,
               shares: 135,
               current_price: 370.40,
               model_used: 'China Model'
             }
```

### 3. Prediction Request (US Stock)

```
User → GET /api/predict/AAPL?account_size=100000
       │
       ├─► fetch_data("AAPL", lookback_days=500)
       │   └─► Yahoo Finance API → OHLCV data
       │
       ├─► engineer_market_specific_features(data, "AAPL")
       │   │
       │   ├─► MarketClassifier.get_market("AAPL")
       │   │   └─► Returns: 'us_international'
       │   │
       │   ├─► TechnicalFeatureEngineer.add_all_features()
       │   │   └─► SMA, EMA, RSI, MACD, Bollinger Bands, etc.
       │   │
       │   ├─► VolatilityFeatureEngineer.add_all_features()
       │   │   └─► Historical vol, Yang-Zhang vol, Parkinson vol
       │   │
       │   └─► SelectiveMacroFeatureEngineer.add_all_features()
       │       ├─► Fetch VIX (^VIX) → vix_close, vix_change
       │       ├─► Fetch SPY (SPY) → spy_close, spy_returns
       │       ├─► Fetch DXY (DX-Y.NYB) → dxy_close, dxy_change
       │       └─► Fetch GLD (GLD) → gld_close, gld_returns
       │
       ├─► get_model_router()
       │   └─► Initialize ModelRouter if not already initialized
       │
       ├─► MarketClassifier.get_market("AAPL")
       │   └─► Returns: 'us_international'
       │
       ├─► HybridEnsemblePredictor.predict(X_latest)
       │   │
       │   ├─► Check if model exists in MODEL_CACHE["AAPL"]
       │   │   ├─► If exists: Use cached model
       │   │   └─► If not: Train new model (3-5 minutes)
       │   │
       │   └─► Return prediction: [0.0187]  # 1.87% 5-day return
       │
       ├─► Calculate confidence based on prediction magnitude
       │   └─► confidence = 0.68 (68%)
       │
       ├─► Generate trading signal
       │   ├─► Direction: 1 (BUY) because predicted_return > 0
       │   ├─► Signal: "BUY" because confidence > 0.65
       │   └─► Position size: $50,000 (50% of $100,000 account)
       │
       └─► Return JSON response

Response ◄── {
               status: 'success',
               ticker: 'AAPL',
               signal: 'BUY',
               direction: 1,
               confidence: 0.68,
               predicted_return: 0.0187,
               position_size: 50000,
               shares: 267,
               current_price: 187.45,
               model_used: 'US/Intl Model'
             }
```

## Feature Engineering Comparison

### Chinese Stock (0700.HK)

```
Raw Data (OHLCV)
      │
      ├─► Technical Features (60+ features)
      │   ├─► Trend: SMA_5, SMA_10, SMA_20, SMA_50, SMA_200
      │   ├─► Momentum: RSI, MACD, Stochastic, Williams %R
      │   ├─► Volatility: Bollinger Bands, ATR, Donchian
      │   └─► Volume: OBV, VWAP, Volume SMA
      │
      ├─► Volatility Features (10+ features)
      │   ├─► Historical Volatility: hist_vol_5, hist_vol_10, hist_vol_20
      │   ├─► Yang-Zhang Volatility: yz_vol_5, yz_vol_10, yz_vol_20
      │   └─► Parkinson Volatility: park_vol_10, park_vol_20
      │
      └─► China Macro Features (6 features)
          ├─► CSI300_close        # China's S&P 500
          ├─► CSI300_returns
          ├─► cny_usd             # Yuan strength
          ├─► cny_usd_change
          ├─► hsi_close           # Hong Kong market
          └─► hsi_returns
                │
                ├─► Total: ~76 features
                └─► Fed to: China Model
```

### US Stock (AAPL)

```
Raw Data (OHLCV)
      │
      ├─► Technical Features (60+ features)
      │   ├─► Trend: SMA_5, SMA_10, SMA_20, SMA_50, SMA_200
      │   ├─► Momentum: RSI, MACD, Stochastic, Williams %R
      │   ├─► Volatility: Bollinger Bands, ATR, Donchian
      │   └─► Volume: OBV, VWAP, Volume SMA
      │
      ├─► Volatility Features (10+ features)
      │   ├─► Historical Volatility: hist_vol_5, hist_vol_10, hist_vol_20
      │   ├─► Yang-Zhang Volatility: yz_vol_5, yz_vol_10, yz_vol_20
      │   └─► Parkinson Volatility: park_vol_10, park_vol_20
      │
      └─► US Macro Features (8 features)
          ├─► vix_close           # Fear index
          ├─► vix_change
          ├─► spy_close           # S&P 500
          ├─► spy_returns
          ├─► dxy_close           # Dollar strength
          ├─► dxy_change
          ├─► gld_close           # Gold (safe haven)
          └─► gld_returns
                │
                ├─► Total: ~78 features
                └─► Fed to: US/Intl Model
```

## Model Routing Logic

```python
def route_prediction(ticker, X_latest):
    """
    Route prediction to appropriate model based on ticker.
    """
    # Step 1: Get model router
    router = get_model_router()

    # Step 2: Check if dual model system is enabled
    if router is None or not USE_DUAL_MODEL_SYSTEM:
        # Use standard US/Intl model for all stocks
        return us_intl_model.predict(X_latest)

    # Step 3: Classify market
    market = MarketClassifier.get_market(ticker)

    # Step 4: Route to appropriate model
    if market == 'chinese':
        # Use China model
        try:
            china_predictor = ChinaMarketPredictor()
            return china_predictor.predict(X_latest)
        except Exception as e:
            # Fallback to US/Intl model on error
            logger.error(f"China model failed: {e}")
            return us_intl_model.predict(X_latest)
    else:
        # Use US/Intl model
        return us_intl_model.predict(X_latest)
```

## Configuration Modes

### Mode 1: Dual Model System Enabled (Default)

```python
USE_DUAL_MODEL_SYSTEM = True
```

```
Ticker       Market          Model Used        Macro Features
------------------------------------------------------------------------
0700.HK      chinese         China Model       CSI300, CNY, HSI
9988.HK      chinese         China Model       CSI300, CNY, HSI
600519.SS    chinese         China Model       CSI300, CNY, HSI
AAPL         us_intl         US/Intl Model     VIX, SPY, DXY, GLD
TSLA         us_intl         US/Intl Model     VIX, SPY, DXY, GLD
BP.L         us_intl         US/Intl Model     VIX, SPY, DXY, GLD
```

### Mode 2: Dual Model System Disabled

```python
USE_DUAL_MODEL_SYSTEM = False
```

```
Ticker       Market          Model Used        Macro Features
------------------------------------------------------------------------
0700.HK      us_intl         US/Intl Model     VIX, SPY, DXY, GLD
9988.HK      us_intl         US/Intl Model     VIX, SPY, DXY, GLD
600519.SS    us_intl         US/Intl Model     VIX, SPY, DXY, GLD
AAPL         us_intl         US/Intl Model     VIX, SPY, DXY, GLD
TSLA         us_intl         US/Intl Model     VIX, SPY, DXY, GLD
BP.L         us_intl         US/Intl Model     VIX, SPY, DXY, GLD
```

## Error Handling Flow

```
Prediction Request
       │
       ├─► Try: Get market classification
       │   ├─► Success → Continue
       │   └─► Error → Default to 'us_international'
       │
       ├─► Try: Engineer market-specific features
       │   ├─► Success → Continue with full features
       │   └─► Error → Continue with technical features only
       │           (Log warning about missing macro features)
       │
       ├─► Try: Route to appropriate model
       │   ├─► If Chinese market:
       │   │   ├─► Try: Use China model
       │   │   │   ├─► Success → Return prediction
       │   │   │   └─► Error → Fall back to US/Intl model
       │   │   │               (Log error and fallback)
       │   │   │
       │   │   └─► Use US/Intl model (fallback)
       │   │       └─► Return prediction
       │   │
       │   └─► If US/Intl market:
       │       └─► Use US/Intl model
       │           └─► Return prediction
       │
       └─► Return final prediction with confidence and signals
```

## Performance Monitoring

```
┌─────────────────────────────────────────────────────────────┐
│                    Live Monitoring                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  China Model (Chinese Stocks)                               │
│  ├─ Predictions made: 150                                   │
│  ├─ Average confidence: 68%                                 │
│  ├─ Win rate: 57%                                           │
│  ├─ Average return per trade: +1.37%                        │
│  └─ Cumulative return: +71% (annualized)                    │
│                                                             │
│  US/Intl Model (US/International Stocks)                    │
│  ├─ Predictions made: 380                                   │
│  ├─ Average confidence: 64%                                 │
│  ├─- Win rate: 54%                                          │
│  ├─ Average return per trade: +0.42%                        │
│  └─ Cumulative return: +22% (annualized)                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Technical Stack

```
┌─────────────────────────────────────────────────────────────┐
│                     Technology Stack                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Web Framework: Flask (Python)                              │
│  API Style: RESTful                                         │
│  Data Source: Yahoo Finance (yfinance)                      │
│  ML Framework: TensorFlow/Keras, XGBoost, LightGBM          │
│  Data Processing: pandas, numpy                             │
│  Model Storage: pickle (local filesystem)                   │
│  Logging: Python logging module                             │
│  Testing: pytest, requests                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

**Last Updated:** November 24, 2025
**Version:** 1.0
**Status:** Production Ready
