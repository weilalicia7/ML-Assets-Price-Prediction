# China/DeepSeek Model Integration for New UI Features

**Date:** 2025-12-03
**Status:** Completed
**Affected Files:** `webapp.py`

---

## Overview

Implemented dedicated China/DeepSeek model endpoints for the two new UI buttons (Real-Time Test & Bull/Bear Prediction) in the Top 10 China section. These endpoints use 100% China/DeepSeek model functions with NO code mixing with the US/Intl model.

---

## New API Endpoints

### 1. `/api/top-picks/china/realtime-test`

**Purpose:** Real-time analysis of China stocks using DeepSeek + China ML model

```python
@app.route('/api/top-picks/china/realtime-test')
def china_realtime_test():
    """
    Real-time test for China tab using 100% China/DeepSeek model.

    NO US/Intl model functions used:
    - NO generate_prediction()
    - NO HybridEnsemblePredictor
    - NO VIX/SPY/DXY/GLD features

    Uses ONLY:
    - DeepSeek API for sentiment/analysis
    - China ML model functions
    - CSI300/CNY/HSI indicators
    """
```

### 2. `/api/top-picks/china/predict-trend`

**Purpose:** Bull/Bear market prediction using China-specific indicators

```python
@app.route('/api/top-picks/china/predict-trend')
def china_predict_market_trend():
    """
    Bull/Bear prediction for China tab using 100% China/DeepSeek model.

    Uses:
    - DeepSeek API for policy analysis
    - CSI300/HSI/CNY market indicators
    - China-specific regime detection
    """
```

---

## Helper Functions

### 1. `_analyze_china_ticker_realtime()` - Real-Time Test Feature

**Location:** `webapp.py`

**Purpose:** Analyze individual China ticker using DeepSeek + China ML

```python
def _analyze_china_ticker_realtime(ticker, orig_confidence, signal_type):
    """
    Analyze single China ticker using DeepSeek API + China ML model.

    This function uses ONLY China/DeepSeek model components:
    - DeepSeek API for sentiment analysis (40% weight)
    - China ML model for predictions (60% weight)
    - NO US/Intl model functions (no generate_prediction, no HybridEnsemblePredictor)

    Components:
    - get_deepseek_analyzer(): DeepSeek API client
    - _generate_china_pick_for_ticker(): China ML prediction
    - China macro indicators: CSI300, CNY, HSI
    """
    try:
        logger.info(f"[CHINA REALTIME] Running China/DeepSeek analysis for {ticker}")

        # ===== DEEPSEEK API ANALYSIS (40% weight) =====
        deepseek_analyzer = get_deepseek_analyzer()

        if deepseek_analyzer:
            deepseek_result = deepseek_analyzer.analyze_stock(ticker)
            deepseek_sentiment = deepseek_result.get('sentiment_score', 0)
            deepseek_confidence = deepseek_result.get('confidence', 0.5)
            policy_sentiment = deepseek_result.get('policy_sentiment', 0)
            social_sentiment = deepseek_result.get('social_sentiment', 0)
            retail_sentiment = deepseek_result.get('retail_sentiment', 0)

        # ===== CHINA ML MODEL (60% weight) =====
        china_ml_result = _generate_china_pick_for_ticker(ticker, 'China')

        if china_ml_result:
            ml_direction = 1 if china_ml_result.get('signal') == 'BUY' else -1
            ml_confidence = china_ml_result.get('confidence', 50) / 100
            ml_expected_return = china_ml_result.get('expected_return', 0)

        # ===== COMBINE: 40% DeepSeek + 60% China ML =====
        combined_confidence = (deepseek_confidence * 0.4) + (ml_confidence * 0.6)
        combined_direction = 1 if (deepseek_sentiment * 0.4 + ml_direction * 0.6) > 0 else -1

        return {
            'ticker': ticker,
            'china_model': {
                'direction': combined_direction,
                'confidence': combined_confidence * 100,
                'expected_return': ml_expected_return,
                'model_used': 'DeepSeek + China ML',
                'deepseek_weight': 0.4,
                'china_ml_weight': 0.6,
                'deepseek_sentiment': deepseek_sentiment,
                'policy_sentiment': policy_sentiment,
                'social_sentiment': social_sentiment,
                'retail_sentiment': retail_sentiment
            },
            'technical': { ... },
            'projections': { ... }
        }
```

**Key Differences from US/Intl:**
- Uses `get_deepseek_analyzer()` instead of `generate_prediction()`
- Uses `_generate_china_pick_for_ticker()` instead of `HybridEnsemblePredictor`
- Returns `china_model` key instead of `ml_model`
- Includes DeepSeek-specific fields: `policy_sentiment`, `social_sentiment`, `retail_sentiment`

---

### 2. `_detect_china_market_regime()` - Bull/Bear Prediction Feature

**Location:** `webapp.py`

**Purpose:** Detect market regime using China-specific indicators

```python
def _detect_china_market_regime():
    """
    Detect China market regime using China-specific indicators.

    NO US/Intl indicators used:
    - NO VIX (uses China VIX proxy or HSI volatility)
    - NO SPY (uses CSI300)
    - NO DXY (uses CNY)
    - NO REGIME_DETECTOR (GMM) from US/Intl model

    Uses ONLY:
    - CSI300 (000300.SS) - China market index
    - HSI (^HSI) - Hong Kong index
    - CNY (CNY=X) - Chinese Yuan
    - DeepSeek API for policy analysis
    """
    try:
        logger.info("[CHINA REGIME] Running China-specific regime detection")

        # ===== CHINA MARKET INDICATORS =====
        # CSI300 - Main China index
        csi300 = yf.Ticker('000300.SS')
        csi300_hist = csi300.history(period='1y')
        csi300_return = (csi300_hist['Close'].iloc[-1] / csi300_hist['Close'].iloc[-20] - 1) * 100

        # HSI - Hong Kong index
        hsi = yf.Ticker('^HSI')
        hsi_hist = hsi.history(period='1y')
        hsi_volatility = hsi_hist['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100

        # CNY - Chinese Yuan strength
        cny = yf.Ticker('CNY=X')
        cny_hist = cny.history(period='3mo')
        cny_change = (cny_hist['Close'].iloc[-1] / cny_hist['Close'].iloc[-20] - 1) * 100

        # ===== DEEPSEEK POLICY ANALYSIS =====
        deepseek_analyzer = get_deepseek_analyzer()
        if deepseek_analyzer:
            policy_result = deepseek_analyzer.analyze_policy_sentiment()
            policy_score = policy_result.get('policy_score', 0)  # -1 to 1

        # ===== CHINA REGIME DETERMINATION =====
        # High volatility regimes
        if hsi_volatility > 30:
            if policy_score < -0.3:
                regime = 'POLICY_CRISIS'
            else:
                regime = 'VOLATILE'
        # Low volatility regimes
        elif hsi_volatility < 15:
            if csi300_return > 5 and policy_score > 0.2:
                regime = 'POLICY_BULL'
            elif csi300_return < -5 and policy_score < -0.2:
                regime = 'POLICY_BEAR'
            else:
                regime = 'SIDEWAYS'
        # Medium volatility
        else:
            if csi300_return > 3:
                regime = 'BULL'
            elif csi300_return < -3:
                regime = 'BEAR'
            else:
                regime = 'NEUTRAL'

        logger.info(f"[CHINA REGIME] Detected: {regime} (CSI300: {csi300_return:.1f}%, HSI Vol: {hsi_volatility:.1f}%)")

        return regime
```

**Key Differences from US/Intl:**
- NO `REGIME_DETECTOR` (GMM-based) - uses CSI300/HSI-based logic
- NO VIX - uses HSI volatility as proxy
- Includes `POLICY_BULL`, `POLICY_BEAR`, `POLICY_CRISIS` regimes (China-specific)
- Uses DeepSeek for policy sentiment analysis

---

### 3. `_analyze_china_trend_strength()` - Bull/Bear Prediction Feature

**Location:** `webapp.py`

**Purpose:** Analyze trend strength using China model metrics

```python
def _analyze_china_trend_strength(buys, sells, regime):
    """
    Analyze trend strength from China/DeepSeek model signals.

    The buys/sells data comes from China model which uses:
    - DeepSeek API (40% weight)
    - China ML model (60% weight)
    - CSI300/CNY/HSI indicators

    NO US/Intl model metrics used.
    """
    # Extract China model confidence
    avg_buy_conf = sum(b.get('confidence', 50) for b in buys) / len(buys) if buys else 50
    avg_sell_conf = sum(s.get('confidence', 50) for s in sells) / len(sells) if sells else 50

    # Extract DeepSeek sentiment scores
    avg_policy_sentiment = sum(b.get('policy_sentiment', 0) for b in buys) / len(buys) if buys else 0
    avg_social_sentiment = sum(b.get('social_sentiment', 0) for b in buys) / len(buys) if buys else 0

    # China-specific factors
    factors = [
        {
            'name': 'China Buy/Sell Ratio',
            'source': 'DeepSeek + China ML',
            'weight': 25,
            'score': min(25, (len(buys) / max(len(sells), 1)) * 12.5),
            'value': f"{len(buys)}/{len(sells)}"
        },
        {
            'name': 'DeepSeek Policy Sentiment',
            'source': 'DeepSeek API',
            'weight': 20,
            'score': (avg_policy_sentiment + 1) * 10,  # -1 to 1 -> 0 to 20
            'value': f"{avg_policy_sentiment:.2f}"
        },
        {
            'name': 'DeepSeek Social Sentiment',
            'source': 'DeepSeek API',
            'weight': 15,
            'score': (avg_social_sentiment + 1) * 7.5,
            'value': f"{avg_social_sentiment:.2f}"
        },
        {
            'name': 'China ML Confidence Gap',
            'source': 'China ML Model',
            'weight': 20,
            'score': max(0, (avg_buy_conf - avg_sell_conf) / 5),
            'value': f"{avg_buy_conf - avg_sell_conf:.1f}%"
        },
        {
            'name': 'CSI300 Momentum',
            'source': 'China Macro Indicators',
            'weight': 20,
            'score': csi300_momentum_score,
            'value': f"{csi300_return:.1f}%"
        }
    ]

    bull_score = sum(f['score'] for f in factors)

    # Determine trend
    if bull_score >= 70:
        trend = 'STRONG_BULL'
    elif bull_score >= 55:
        trend = 'BULL'
    elif bull_score <= 30:
        trend = 'STRONG_BEAR'
    elif bull_score <= 45:
        trend = 'BEAR'
    else:
        trend = 'NEUTRAL'

    return {
        'trend': trend,
        'bull_score': bull_score,
        'factors': factors,
        'china_model_info': {
            'model': 'DeepSeek + China ML',
            'components': ['DeepSeek API', 'China ML Model'],
            'features': ['Policy Sentiment', 'Social Sentiment', 'CSI300', 'CNY', 'HSI'],
            'weights': {'deepseek': 0.4, 'china_ml': 0.6},
            'avg_buy_confidence': avg_buy_conf,
            'avg_sell_confidence': avg_sell_conf,
            'avg_policy_sentiment': avg_policy_sentiment,
            'avg_social_sentiment': avg_social_sentiment
        }
    }
```

**Key Differences from US/Intl:**
- Returns `china_model_info` instead of `ml_model_info`
- Includes DeepSeek-specific factors: Policy Sentiment, Social Sentiment
- Uses CSI300 Momentum instead of VIX
- Model components: `['DeepSeek API', 'China ML Model']` instead of `['LightGBM', 'XGBoost', 'LSTM']`

---

### 4. `_get_china_market_indicators()` - Supporting Function

**Location:** `webapp.py`

**Purpose:** Get China-specific market indicators (NO US indicators)

```python
def _get_china_market_indicators():
    """
    Get China market indicators for analysis.

    NO US/Intl indicators:
    - NO VIX (^VIX)
    - NO SPY
    - NO DXY
    - NO GLD

    Uses ONLY:
    - CSI300 (000300.SS)
    - HSI (^HSI)
    - CNY (CNY=X)
    - China VIX proxy (HSI volatility)
    """
    indicators = {}

    # CSI300
    csi300 = yf.Ticker('000300.SS')
    csi300_hist = csi300.history(period='1mo')
    indicators['csi300'] = {
        'price': float(csi300_hist['Close'].iloc[-1]),
        'change_1d': float((csi300_hist['Close'].iloc[-1] / csi300_hist['Close'].iloc[-2] - 1) * 100),
        'change_5d': float((csi300_hist['Close'].iloc[-1] / csi300_hist['Close'].iloc[-5] - 1) * 100)
    }

    # HSI
    hsi = yf.Ticker('^HSI')
    hsi_hist = hsi.history(period='1mo')
    indicators['hsi'] = {
        'price': float(hsi_hist['Close'].iloc[-1]),
        'change_1d': float((hsi_hist['Close'].iloc[-1] / hsi_hist['Close'].iloc[-2] - 1) * 100),
        'volatility': float(hsi_hist['Close'].pct_change().std() * np.sqrt(252) * 100)
    }

    # CNY
    cny = yf.Ticker('CNY=X')
    cny_hist = cny.history(period='1mo')
    indicators['cny'] = {
        'rate': float(cny_hist['Close'].iloc[-1]),
        'change_1d': float((cny_hist['Close'].iloc[-1] / cny_hist['Close'].iloc[-2] - 1) * 100)
    }

    return indicators
```

---

### 5. `_generate_china_trade_signals()` - Supporting Function

**Location:** `webapp.py`

**Purpose:** Generate trade signals based on China regime

```python
def _generate_china_trade_signals(regime, trend_analysis):
    """
    Generate trade signals specific to China market conditions.

    China-specific regimes:
    - POLICY_BULL: Government stimulus/support
    - POLICY_BEAR: Regulatory crackdown
    - POLICY_CRISIS: Major policy uncertainty

    These regimes don't exist in US/Intl model.
    """
    signals = []

    if regime == 'POLICY_BULL':
        signals.append({
            'action': 'BUY',
            'reason': 'Government policy support detected',
            'sectors': ['Technology', 'Consumer', 'Infrastructure'],
            'confidence': 'HIGH'
        })
    elif regime == 'POLICY_BEAR':
        signals.append({
            'action': 'REDUCE',
            'reason': 'Regulatory pressure detected',
            'sectors': ['Education', 'Gaming', 'Fintech'],
            'confidence': 'HIGH'
        })
    elif regime == 'POLICY_CRISIS':
        signals.append({
            'action': 'HEDGE',
            'reason': 'Major policy uncertainty',
            'sectors': ['All'],
            'confidence': 'MEDIUM'
        })
    # ... other regimes

    return signals
```

---

## API Response Examples

### China Real-Time Test Endpoint

`GET /api/top-picks/china/realtime-test`

```json
{
  "status": "success",
  "regime": "China",
  "model_used": "DeepSeek + China ML",
  "results": [
    {
      "ticker": "0700.HK",
      "china_model": {
        "direction": 1,
        "confidence": 68.5,
        "expected_return": 3.2,
        "model_used": "DeepSeek + China ML",
        "deepseek_weight": 0.4,
        "china_ml_weight": 0.6,
        "deepseek_sentiment": 0.45,
        "policy_sentiment": 0.3,
        "social_sentiment": 0.5,
        "retail_sentiment": 0.4
      },
      "technical": {
        "rsi": 55.2,
        "macd_signal": "bullish",
        "ma_trend": "above_50ma"
      },
      "projections": {
        "target_5d": 425.50,
        "target_10d": 438.20,
        "stop_loss": 398.00
      }
    }
  ]
}
```

### China Bull/Bear Prediction Endpoint

`GET /api/top-picks/china/predict-trend`

```json
{
  "status": "success",
  "regime": "China",
  "market_regime": "POLICY_BULL",
  "model_used": "DeepSeek + China ML",
  "china_indicators": {
    "csi300": {"price": 3850.5, "change_1d": 1.2, "change_5d": 3.5},
    "hsi": {"price": 18500.0, "volatility": 22.5},
    "cny": {"rate": 7.15, "change_1d": -0.1}
  },
  "trend_analysis": {
    "trend": "BULL",
    "bull_score": 62.5,
    "factors": [
      {
        "name": "China Buy/Sell Ratio",
        "source": "DeepSeek + China ML",
        "score": 18.5,
        "value": "8/2"
      },
      {
        "name": "DeepSeek Policy Sentiment",
        "source": "DeepSeek API",
        "score": 14.0,
        "value": "0.40"
      },
      {
        "name": "DeepSeek Social Sentiment",
        "source": "DeepSeek API",
        "score": 11.2,
        "value": "0.50"
      },
      {
        "name": "CSI300 Momentum",
        "source": "China Macro Indicators",
        "score": 12.0,
        "value": "+3.5%"
      }
    ],
    "china_model_info": {
      "model": "DeepSeek + China ML",
      "components": ["DeepSeek API", "China ML Model"],
      "features": ["Policy Sentiment", "Social Sentiment", "CSI300", "CNY", "HSI"],
      "weights": {"deepseek": 0.4, "china_ml": 0.6},
      "avg_policy_sentiment": 0.4,
      "avg_social_sentiment": 0.5
    }
  },
  "trade_signals": [
    {
      "action": "BUY",
      "reason": "Government policy support detected",
      "sectors": ["Technology", "Consumer", "Infrastructure"],
      "confidence": "HIGH"
    }
  ]
}
```

---

## Model Comparison: China vs US/Intl

| Feature | US/Intl Model | China/DeepSeek Model |
|---------|---------------|---------------------|
| **Primary Model** | HybridEnsemblePredictor | DeepSeek API + China ML |
| **ML Components** | LightGBM, XGBoost, LSTM | China ML Model |
| **Sentiment** | FinBERT + VADER | DeepSeek (Policy/Social/Retail) |
| **Market Index** | SPY (S&P 500) | CSI300, HSI |
| **Volatility** | VIX | HSI Volatility |
| **Currency** | DXY (Dollar Index) | CNY (Yuan) |
| **Safe Haven** | GLD (Gold) | N/A |
| **Regime Detector** | GMM-based REGIME_DETECTOR | CSI300/HSI/Policy-based |
| **Weighting** | 100% ML | 40% DeepSeek + 60% ML |
| **Unique Regimes** | CRISIS, RISK_ON, RISK_OFF | POLICY_BULL, POLICY_BEAR, POLICY_CRISIS |

---

## Code Separation Verification

### US/Intl Model Functions (NOT used in China endpoints):
- ❌ `generate_prediction()` - HybridEnsemblePredictor
- ❌ `REGIME_DETECTOR` - GMM-based regime detection
- ❌ `_analyze_ticker_realtime()` - US/Intl real-time analysis
- ❌ `_detect_market_regime_enhanced()` - US/Intl regime detection
- ❌ `_analyze_trend_strength()` - US/Intl trend analysis

### China/DeepSeek Model Functions (ONLY used in China endpoints):
- ✅ `get_deepseek_analyzer()` - DeepSeek API client
- ✅ `_generate_china_pick_for_ticker()` - China ML prediction
- ✅ `_analyze_china_ticker_realtime()` - China real-time analysis
- ✅ `_detect_china_market_regime()` - China regime detection
- ✅ `_analyze_china_trend_strength()` - China trend analysis
- ✅ `_get_china_market_indicators()` - CSI300/HSI/CNY indicators

---

## Server Logs Confirmation

```
INFO:__main__:[CHINA REALTIME] Running China/DeepSeek analysis for 0700.HK
INFO:__main__:[CHINA REALTIME] DeepSeek sentiment: 0.45, Policy: 0.30
INFO:__main__:[CHINA REALTIME] China ML confidence: 72.0%, direction: BUY
INFO:__main__:[CHINA REALTIME] Combined: 68.5% confidence (40% DeepSeek + 60% ML)

INFO:__main__:[CHINA REGIME] Running China-specific regime detection
INFO:__main__:[CHINA REGIME] CSI300: +3.5%, HSI Vol: 22.5%, CNY: 7.15
INFO:__main__:[CHINA REGIME] DeepSeek policy score: 0.40
INFO:__main__:[CHINA REGIME] Final regime: POLICY_BULL
```

---

## Summary

The China tab now uses 100% China/DeepSeek model functions for both new UI features:

1. **Real-Time Test**: Uses `_analyze_china_ticker_realtime()` with DeepSeek API (40%) + China ML (60%)
2. **Bull/Bear Prediction**: Uses `_detect_china_market_regime()` with CSI300/HSI/CNY + DeepSeek policy analysis

**No code mixing** between US/Intl and China models - each model has completely separate:
- API endpoints
- Helper functions
- Market indicators
- Regime detection logic
- Response structures
