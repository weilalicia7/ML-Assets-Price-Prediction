# Phase 2 + Phase 3 Calculation Code Reference

This document details the core calculation logic for the Phase 2 (Asset-Class Ensembles) and Phase 3 (Specialized Features) implementation.

---

## Phase 3: Specialized Feature Calculations

### 1. International Features (`src/features/international_features.py`)

#### FX Exposure Features
```python
# FX returns at different horizons
df['fx_return_1d'] = fx_close.pct_change(1)
df['fx_return_5d'] = fx_close.pct_change(5)
df['fx_return_20d'] = fx_close.pct_change(20)

# FX volatility (20-day rolling standard deviation)
df['fx_volatility'] = fx_close.pct_change().rolling(20).std()

# FX momentum (trend direction using MA crossover)
fx_ma_20 = fx_close.rolling(20).mean()
fx_ma_60 = fx_close.rolling(60).mean()
df['fx_momentum'] = (fx_ma_20 - fx_ma_60) / fx_ma_60

# Stock-FX correlation (60-day rolling)
stock_returns = df['Close'].pct_change()
fx_returns = fx_close.pct_change()
df['fx_correlation'] = stock_returns.rolling(60).corr(fx_returns)
```

#### Home Market Features
```python
# Home index returns
df['home_index_return_1d'] = idx_close.pct_change(1)
df['home_index_return_5d'] = idx_close.pct_change(5)

# Correlation with home index (60-day rolling)
df['home_index_correlation'] = stock_returns.rolling(60).corr(idx_returns)

# Beta to home index
covariance = stock_returns.rolling(60).cov(idx_returns)
variance = idx_returns.rolling(60).var()
df['home_index_beta'] = (covariance / variance).clip(-3, 3)

# Relative strength vs home index (20-day cumulative)
stock_cum = (1 + stock_returns).rolling(20).apply(lambda x: x.prod()) - 1
idx_cum = (1 + idx_returns).rolling(20).apply(lambda x: x.prod()) - 1
df['relative_strength'] = stock_cum - idx_cum
```

#### ADR Premium Features
```python
# Overnight gap (reflects home market movement)
df['overnight_gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

# Gap vs return correlation (higher for ADRs)
df['gap_return_corr'] = df['overnight_gap'].rolling(60).corr(returns)

# Intraday range relative to gap
intraday_range = (df['High'] - df['Low']) / df['Open']
df['range_vs_gap'] = (intraday_range / (df['overnight_gap'].abs() + 0.001)).clip(0, 10)

# Volume ratio
df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
```

#### Geopolitical Risk Features
```python
# Volatility regime changes
vol_20 = returns.rolling(20).std()
vol_60 = returns.rolling(60).std()
df['vol_regime_ratio'] = vol_20 / vol_60

# Tail risk frequency (large moves > 2 std)
large_move_threshold = returns.rolling(252).std() * 2
df['tail_risk_freq'] = (returns.abs() > large_move_threshold).rolling(20).mean()

# Return skewness (asymmetric risk)
df['return_skewness'] = returns.rolling(60).skew()

# US correlation and change
df['us_correlation'] = returns.rolling(20).corr(spy_returns)
df['correlation_change'] = df['us_correlation'] - df['us_correlation'].rolling(60).mean()
```

---

### 2. Crypto Features (`src/features/crypto_features.py`)

#### Bitcoin Correlation Features
```python
# BTC returns at different horizons
df['btc_return_1d'] = btc_close.pct_change(1)
df['btc_return_5d'] = btc_close.pct_change(5)
df['btc_return_20d'] = btc_close.pct_change(20)

# BTC annualized volatility
btc_returns = btc_close.pct_change()
df['btc_volatility'] = btc_returns.rolling(20).std() * np.sqrt(365)

# Stock-BTC correlation (60-day rolling)
df['btc_correlation'] = stock_returns.rolling(60).corr(btc_returns)

# Beta to Bitcoin
covariance = stock_returns.rolling(60).cov(btc_returns)
variance = btc_returns.rolling(60).var()
df['btc_beta'] = (covariance / variance).clip(-5, 5)

# BTC momentum
btc_ma_20 = btc_close.rolling(20).mean()
btc_ma_50 = btc_close.rolling(50).mean()
df['btc_momentum'] = (btc_close - btc_ma_50) / btc_ma_50

# BTC MA crossover signal (binary)
df['btc_ma_cross'] = (btc_ma_20 > btc_ma_50).astype(int)
```

#### Crypto Sentiment Proxy Features
```python
# Fear & Greed proxy (volatility + momentum based, 0-100 scale)
btc_vol = btc_returns.rolling(14).std() * np.sqrt(365)
btc_mom = btc_close.pct_change(14)

vol_norm = 1 - (btc_vol / btc_vol.rolling(90).max()).clip(0, 1)
mom_norm = ((btc_mom / btc_mom.rolling(90).std()) + 2) / 4
mom_norm = mom_norm.clip(0, 1)

df['crypto_fear_greed_proxy'] = (vol_norm * 0.4 + mom_norm * 0.6) * 100

# Crypto momentum rank (position in 52-week range)
btc_52w_high = btc_close.rolling(252).max()
btc_52w_low = btc_close.rolling(252).min()
df['crypto_momentum_rank'] = (btc_close - btc_52w_low) / (btc_52w_high - btc_52w_low + 0.01)

# Bitcoin drawdown from ATH
btc_ath = btc_close.cummax()
df['btc_drawdown'] = (btc_close - btc_ath) / btc_ath

# Recovery rate from recent low
btc_recent_low = btc_close.rolling(30).min()
df['btc_recovery_rate'] = (btc_close - btc_recent_low) / btc_recent_low
```

#### On-Chain Proxy Features
```python
# HODL proxy (volume-weighted price deviation)
vwap_90 = (btc_close * btc_volume).rolling(90).sum() / btc_volume.rolling(90).sum()
df['hodl_proxy'] = (btc_close - vwap_90) / vwap_90

# Whale activity proxy (high volume days)
volume_ma = btc_volume.rolling(20).mean()
volume_std = btc_volume.rolling(20).std()
whale_threshold = volume_ma + 2 * volume_std
df['whale_activity_proxy'] = (btc_volume > whale_threshold).rolling(10).mean()

# Network growth proxy (near ATH frequency)
ath_distance = btc_close / btc_ath
df['network_growth_proxy'] = (ath_distance > 0.95).rolling(30).mean()
```

---

### 3. Commodity Features (`src/features/commodity_features.py`)

#### Oil Features
```python
# Oil returns at different horizons
df['oil_return_1d'] = oil_close.pct_change(1)
df['oil_return_5d'] = oil_close.pct_change(5)
df['oil_return_20d'] = oil_close.pct_change(20)

# Oil volatility (annualized)
df['oil_volatility'] = oil_close.pct_change().rolling(20).std() * np.sqrt(252)

# Stock-Oil correlation
df['oil_correlation'] = stock_returns.rolling(60).corr(oil_returns)

# Oil momentum
oil_ma_20 = oil_close.rolling(20).mean()
oil_ma_50 = oil_close.rolling(50).mean()
df['oil_momentum'] = (oil_close - oil_ma_50) / oil_ma_50

# Oil MA crossover
df['oil_ma_cross'] = (oil_ma_20 > oil_ma_50).astype(int)

# Oil percentile rank (where in 1-year range)
oil_52w_high = oil_close.rolling(252).max()
oil_52w_low = oil_close.rolling(252).min()
df['oil_percentile'] = (oil_close - oil_52w_low) / (oil_52w_high - oil_52w_low + 0.01)
```

#### Gold Features
```python
# Gold returns
df['gold_return_1d'] = gold_close.pct_change(1)
df['gold_return_5d'] = gold_close.pct_change(5)

# Stock-Gold correlation
df['gold_correlation'] = stock_returns.rolling(60).corr(gold_returns)

# Gold as safe haven indicator
df['gold_safe_haven'] = (gold_close.pct_change(5) > 0) & (spy_close.pct_change(5) < 0)
df['gold_safe_haven'] = df['gold_safe_haven'].rolling(20).mean()

# Gold/Oil ratio (inflation indicator)
df['gold_oil_ratio'] = gold_close / oil_close
df['gold_oil_ratio_change'] = df['gold_oil_ratio'].pct_change(20)
```

#### Dollar Features
```python
# Dollar index returns
df['dollar_return_1d'] = dxy_close.pct_change(1)
df['dollar_return_5d'] = dxy_close.pct_change(5)

# Dollar momentum
dxy_ma_20 = dxy_close.rolling(20).mean()
dxy_ma_50 = dxy_close.rolling(50).mean()
df['dollar_momentum'] = (dxy_close - dxy_ma_50) / dxy_ma_50

# Dollar-commodity inverse correlation
df['dollar_commodity_corr'] = dxy_returns.rolling(60).corr(oil_returns)
```

#### Seasonal Patterns
```python
# Month of year (1-12)
df['month'] = pd.to_datetime(df.index).month

# Seasonal strength (historical monthly performance)
monthly_returns = df.groupby('month')['Close'].transform(
    lambda x: x.pct_change().rolling(252).mean()
)
df['seasonal_strength'] = monthly_returns

# Quarter indicator
df['quarter'] = ((df['month'] - 1) // 3) + 1
```

---

## Phase 2: Asset-Class Ensemble Calculations

### Base Ensemble Prediction (`src/models/asset_class_ensembles.py`)

#### Momentum Signal Calculation
```python
def calculate_momentum_signal(self, data: pd.DataFrame) -> float:
    """Calculate momentum-based signal."""
    close = data['Close']

    # Multiple timeframe momentum
    mom_5 = close.pct_change(5).iloc[-1]
    mom_20 = close.pct_change(20).iloc[-1]
    mom_60 = close.pct_change(60).iloc[-1] if len(close) >= 60 else 0

    # Weighted combination
    momentum = mom_5 * 0.5 + mom_20 * 0.3 + mom_60 * 0.2

    # Normalize to [-1, 1]
    return float(np.tanh(momentum * 10))
```

#### Mean Reversion Signal
```python
def calculate_mean_reversion_signal(self, data: pd.DataFrame) -> float:
    """Calculate mean reversion signal."""
    close = data['Close']

    # Z-score from 20-day mean
    ma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    z_score = (close.iloc[-1] - ma_20.iloc[-1]) / (std_20.iloc[-1] + 0.0001)

    # Invert for mean reversion (oversold = buy, overbought = sell)
    return float(np.tanh(-z_score * 0.5))
```

#### Volatility Signal
```python
def calculate_volatility_signal(self, data: pd.DataFrame) -> float:
    """Calculate volatility-based signal."""
    returns = data['Close'].pct_change()

    # Current vs historical volatility
    vol_current = returns.rolling(10).std().iloc[-1]
    vol_historical = returns.rolling(60).std().iloc[-1]

    vol_ratio = vol_current / (vol_historical + 0.0001)

    # High vol ratio = reduce exposure
    return float(np.clip(1 - vol_ratio, -1, 1))
```

#### Combined Ensemble Prediction
```python
def predict(self, data: pd.DataFrame) -> Dict:
    """Generate ensemble prediction."""
    # Get individual signals
    momentum = self.calculate_momentum_signal(data)
    mean_rev = self.calculate_mean_reversion_signal(data)
    vol_signal = self.calculate_volatility_signal(data)

    # Asset-class specific weights
    weights = self.get_signal_weights()  # e.g., {'momentum': 0.5, 'mean_rev': 0.3, 'vol': 0.2}

    # Weighted combination
    combined = (
        momentum * weights['momentum'] +
        mean_rev * weights['mean_rev'] +
        vol_signal * weights['vol']
    )

    # Calculate confidence
    signal_agreement = 1 - np.std([momentum, mean_rev, vol_signal])
    confidence = float(np.clip(signal_agreement, 0.3, 0.9))

    # Convert signal to prediction (0-1 scale)
    prediction = (combined + 1) / 2

    return {
        'prediction': float(np.clip(prediction, 0, 1)),
        'signal': float(np.clip(combined, -1, 1)),
        'confidence': confidence,
        'direction': 'LONG' if combined > 0.1 else ('SHORT' if combined < -0.1 else 'HOLD'),
        'momentum_component': momentum,
        'mean_rev_component': mean_rev,
        'vol_component': vol_signal
    }
```

---

### Asset-Class Specific Weights

```python
# EquitySpecificEnsemble
EQUITY_WEIGHTS = {'momentum': 0.4, 'mean_rev': 0.35, 'vol': 0.25}

# ForexSpecificEnsemble
FOREX_WEIGHTS = {'momentum': 0.3, 'mean_rev': 0.5, 'vol': 0.2}

# CryptoSpecificEnsemble
CRYPTO_WEIGHTS = {'momentum': 0.6, 'mean_rev': 0.2, 'vol': 0.2}

# CommoditySpecificEnsemble
COMMODITY_WEIGHTS = {'momentum': 0.35, 'mean_rev': 0.35, 'vol': 0.3}

# InternationalEnsemble
INTERNATIONAL_WEIGHTS = {'momentum': 0.4, 'mean_rev': 0.3, 'vol': 0.3}

# BondSpecificEnsemble
BOND_WEIGHTS = {'momentum': 0.25, 'mean_rev': 0.5, 'vol': 0.25}

# ETFSpecificEnsemble
ETF_WEIGHTS = {'momentum': 0.45, 'mean_rev': 0.35, 'vol': 0.2}
```

---

## Meta-Ensemble Combiner (`src/models/meta_ensemble.py`)

### Ensemble Prediction Combination
```python
def combine_predictions(self, predictions: List[Dict]) -> Dict:
    """Combine multiple predictions with dynamic weighting."""

    total_weight = 0
    weighted_signal = 0
    weighted_confidence = 0

    for pred in predictions:
        asset_class = pred.get('asset_class', 'equity')
        weight = self.ensemble_weights.get(asset_class, 0.1)

        # Weight by both asset class importance and prediction confidence
        effective_weight = weight * pred.get('confidence', 0.5)

        weighted_signal += pred.get('signal', 0) * effective_weight
        weighted_confidence += pred.get('confidence', 0.5) * weight
        total_weight += weight

    if total_weight > 0:
        combined_signal = weighted_signal / total_weight
        combined_confidence = weighted_confidence / total_weight
    else:
        combined_signal = 0
        combined_confidence = 0

    return {
        'combined_signal': float(np.clip(combined_signal, -1, 1)),
        'confidence': float(np.clip(combined_confidence, 0, 1)),
        'predictions_combined': len(predictions)
    }
```

### Phase 1 Signal Adjustments
```python
def apply_phase1_adjustments(self, prediction: Dict, data: pd.DataFrame,
                              market_conditions: Dict = None) -> Dict:
    """Apply Phase 1 risk management adjustments."""
    signal = prediction.get('signal', 0)

    # 1. Regime-aware weighting
    if self.regime_weighter:
        regime = market_conditions.get('regime', 'normal')
        signal = self.regime_weighter.regime_aware_signal_weighting(signal, regime)

    # 2. Volatility scaling
    if self.volatility_scaler and len(data) >= 20:
        returns = data['Close'].pct_change().dropna()
        vol_adjustment = self.volatility_scaler.get_volatility_adjustment(returns)
        signal = signal * vol_adjustment

    # 3. Stress system checks
    if self.stress_system:
        vix = market_conditions.get('vix', 20)
        market_return = market_conditions.get('market_return', 0)
        self.stress_system.update_market_conditions(vix, market_return)

        trading_allowed, reason = self.stress_system.check_trade_allowed()
        if not trading_allowed:
            signal = 0  # Block trading
        else:
            position_mult = self.stress_system.get_adjusted_position_size(1.0)
            signal = signal * position_mult

    return float(np.clip(signal, -1, 1))
```

---

## Unified Trading System (`src/trading/phase2_phase3_integration.py`)

### Asset Class Detection
```python
def detect_asset_class(self, ticker: str) -> str:
    """Detect asset class based on ticker."""
    ticker_upper = ticker.upper()

    # Check specific mappings
    if ticker_upper in INTERNATIONAL_STOCKS:
        return 'international'
    elif ticker_upper in CRYPTO_STOCKS:
        return 'crypto'
    elif ticker_upper in COMMODITY_STOCKS:
        return 'commodity'
    elif '=X' in ticker_upper:
        return 'forex'
    elif ticker_upper in BOND_ETFS:
        return 'bond'
    elif ticker_upper in ETFS:
        return 'etf'
    elif '.HK' in ticker_upper or '.SS' in ticker_upper:
        return 'international'

    return 'equity'  # Default
```

### Final Signal Generation
```python
def generate_signal(self, ticker: str, market_data: Dict,
                    portfolio: Dict = None, price_data: pd.DataFrame = None) -> Dict:
    """Generate comprehensive trading signal."""

    # Step 1: Detect asset class
    asset_class = self.detect_asset_class(ticker)

    # Step 2: Generate specialized features
    if price_data is not None:
        enhanced_data = self.generate_specialized_features(ticker, price_data, asset_class)

    # Step 3: Get ensemble prediction
    ensemble_result = self.get_ensemble_prediction(ticker, enhanced_data, asset_class)

    # Step 4: Apply meta-ensemble (if available)
    if self.meta_ensemble and enhanced_data is not None:
        meta_result = self.meta_ensemble.predict(enhanced_data, ticker, market_data)
        meta_signal = (meta_result.get('signal', 0) + 1) / 2  # Convert to 0-1

        # Blend: 60% ensemble, 40% meta
        final_prediction = ensemble_result['prediction'] * 0.6 + meta_signal * 0.4
        final_confidence = ensemble_result['confidence'] * 0.6 + meta_result.get('confidence', 0.5) * 0.4
    else:
        final_prediction = ensemble_result['prediction']
        final_confidence = ensemble_result['confidence']

    # Step 5: Apply Phase 1 risk management
    if self.phase1_system:
        phase1_result = self.phase1_system.generate_enhanced_signal(
            ticker=ticker,
            market_data=market_data,
            portfolio=portfolio,
            base_signal=final_prediction
        )
        final_signal = phase1_result.get('weighted_signal', final_prediction)
        position_size = phase1_result.get('position_size', 0.1)

    # Step 6: Determine action
    if final_signal > 0.6:
        action = 'LONG'
    elif final_signal < 0.4:
        action = 'SHORT'
    else:
        action = 'HOLD'

    return {
        'ticker': ticker,
        'action': action,
        'final_signal': final_signal,
        'confidence': final_confidence,
        'position_size': position_size,
        'asset_class': asset_class
    }
```

---

## Summary of Feature Counts

| Module | Features | Description |
|--------|----------|-------------|
| InternationalFeatures | 20 | FX, home market, ADR, geopolitical |
| CryptoFeatures | 22 | BTC/ETH correlation, sentiment, on-chain proxies |
| CommodityFeatures | 28 | Oil, gold, dollar, seasonal patterns |
| **Total Phase 3** | **70** | Specialized features |
| Asset-Class Ensembles | 7 | Specialized prediction models |
| Meta-Ensemble | 1 | Combines all ensembles |
| Phase 1 Features | 20 | Risk management, stress protection |
| **Total System** | **91+ features** | Unified trading system |

---

*Document generated: 2025-11-29*
*Version: Phase 2+3 Implementation*
