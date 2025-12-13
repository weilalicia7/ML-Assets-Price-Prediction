# China Stock Model - Complete Calculation Pipeline

## Table of Contents
1. [Data Tier Classification](#data-tier-classification)
2. [Basic Technical Analysis](#basic-technical-analysis)
3. [Full ML Pipeline](#full-ml-pipeline)
4. [China-Specific Feature Engineering](#china-specific-feature-engineering)
5. [Confidence Interval Calculation](#confidence-interval-calculation)
6. [News Feed Generation](#news-feed-generation)
7. [DeepSeek API Integration](#deepseek-api-integration)

---

## Data Tier Classification

The system uses a tiered approach to handle stocks with varying amounts of historical data. The tier determines which analysis methods are available and the confidence level of predictions.

### Tier Definition Logic

```python
def get_data_tier(days_available):
    """
    Determine analysis tier based on available trading days.
    Returns tier info for tiered ML analysis of short-listed stocks.
    """
    if days_available < 30:
        return {
            'tier': 'insufficient',
            'level': 0,
            'analysis_type': 'basic_info_only',
            'message': 'Very new listing - insufficient data for any analysis',
            'confidence': 'none',
            'features_available': ['price', 'volume'],
            'ml_available': False
        }
    elif days_available < 100:
        return {
            'tier': 'minimal',
            'level': 1,
            'analysis_type': 'basic_technical',
            'message': f'Limited history ({days_available} days) - using basic technical indicators',
            'confidence': 'low',
            'features_available': ['price_trend', 'volume_analysis', 'basic_momentum'],
            'ml_available': False
        }
    elif days_available < 250:
        return {
            'tier': 'light',
            'level': 2,
            'analysis_type': 'light_ml',
            'message': f'Moderate history ({days_available} days) - light ML analysis with reduced features',
            'confidence': 'medium',
            'features_available': ['RSI', 'moving_averages', 'volatility', 'momentum'],
            'ml_available': True
        }
    else:
        return {
            'tier': 'full',
            'level': 3,
            'analysis_type': 'full_ml',
            'message': f'Sufficient history ({days_available} days) - full ML analysis available',
            'confidence': 'high',
            'features_available': ['all'],
            'ml_available': True
        }
```

### Tier Thresholds

| Tier | Days Required | Analysis Type | Confidence | Features | ML Available |
|------|--------------|---------------|-----------|----------|--------------|
| Insufficient | < 30 | Basic Info Only | None | Price, Volume | No |
| Minimal | 30-99 | Basic Technical | Low (0.3) | 3 Indicators | No |
| Light | 100-249 | Light ML | Medium | 4 Indicators | Yes |
| Full | 250+ | Full ML | High | All (30+) | Yes |

**Note:** In practice, we use **150 days** as the minimum for ML because feature engineering drops ~30-40% of rows due to lookback periods.

---

## Basic Technical Analysis

For stocks with fewer than 150 days of data, the system uses basic technical analysis since ML models cannot be reliably trained on limited data.

### Momentum Calculation

**5-Day Momentum:**
```
momentum_5d = (current_price / price_5_days_ago - 1) × 100
```

**Example:**
```
If current_price = 100 and price_5_days_ago = 98
momentum_5d = (100 / 98 - 1) × 100 = 2.04%
```

**Code Implementation:**
```python
def basic_technical_analysis(data, ticker):
    current_price = float(data['Close'].iloc[-1])

    # Simple momentum (5-day return)
    if len(data) >= 5:
        momentum_5d = (current_price / float(data['Close'].iloc[-5]) - 1) * 100
    else:
        momentum_5d = 0
```

### Volume Trend Analysis

**Comparison Method:**
```
Recent Volume (last 5 days) vs. Previous Volume (days 5-10)

If: recent_vol > older_vol × 1.1 → increasing
If: recent_vol < older_vol × 0.9 → decreasing
Else → stable
```

**Code:**
```python
if len(data) >= 10 and 'Volume' in data.columns:
    recent_vol = data['Volume'].iloc[-5:].mean()
    older_vol = data['Volume'].iloc[-10:-5].mean()
    vol_trend = 'increasing' if recent_vol > older_vol * 1.1 else (
        'decreasing' if recent_vol < older_vol * 0.9 else 'stable'
    )
```

### Trend Direction Detection

**20-Day Simple Moving Average:**
```
SMA_20 = Average of closing prices for last 20 days

If: current_price > SMA_20 → bullish
If: current_price < SMA_20 → bearish
If: insufficient data → unknown
```

**Code:**
```python
if len(data) >= 20:
    sma_20 = data['Close'].rolling(20).mean().iloc[-1]
    trend = 'bullish' if current_price > sma_20 else 'bearish'
```

### Signal Generation Logic

**Signal Classification:**
```
If momentum_5d > 3%     → BULLISH (direction = 1)
If momentum_5d < -3%    → BEARISH (direction = -1)
Else                    → NEUTRAL (direction = 0)
```

**Confidence Scoring for Basic Analysis:**
- All basic technical analysis carries confidence = **0.3** (low)
- This reflects the inherent uncertainty with limited historical data

**Code:**
```python
if momentum_5d > 3:
    signal = 'BULLISH'
    direction = 1
elif momentum_5d < -3:
    signal = 'BEARISH'
    direction = -1
else:
    signal = 'NEUTRAL'
    direction = 0

return {
    'signal': signal,
    'direction': direction,
    'momentum_5d': round(momentum_5d, 2),
    'volume_trend': vol_trend,
    'trend': trend,
    'confidence': 0.3,  # Low confidence for basic analysis
}
```

### Confidence Interval for Basic Analysis

The 80% confidence interval is centered around the expected return:

```python
'confidence_interval': {
    'lower': expected_return - volatility * 1.28,  # 80% CI centered on expected return
    'upper': expected_return + volatility * 1.28   # 1.28 = z-score for 80% CI
}
```

**Example for bullish signal:**
```
expected_return = 7.3% (from momentum_5d)
volatility = 4% (default for new stocks)

Lower = 7.3% - (4% × 1.28) = 2.18%
Upper = 7.3% + (4% × 1.28) = 12.42%
```

---

## Full ML Pipeline

For stocks with 150+ days of historical data, the system uses a comprehensive ML pipeline.

### Feature Engineering (90+ Features)

#### 1. Price/Returns Features
```python
for period in [1, 2, 3, 5, 10, 20]:
    df[f'returns_{period}d'] = df['Close'].pct_change(period)
```

**Calculation:**
```
returns_1d = (Price_today - Price_yesterday) / Price_yesterday
returns_5d = (Price_today - Price_5_days_ago) / Price_5_days_ago
```

#### 2. RSI (Relative Strength Index)
```python
for period in [14, 21]:
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
```

**Formula:**
```
Gain = Average of positive price changes over period
Loss = Average of negative price changes over period
RS = Gain / Loss
RSI = 100 - (100 / (1 + RS))

Range: 0-100 (30 = oversold, 70 = overbought)
```

#### 3. Moving Averages
```python
for period in [5, 10, 20, 50, 200]:
    df[f'sma_{period}'] = df['Close'].rolling(period).mean()
    df[f'price_vs_sma{period}'] = df['Close'] / df[f'sma_{period}']
```

**MA Crossovers:**
```python
df['sma_5_20_cross'] = (df['sma_5'] > df['sma_20']).astype(int)
df['sma_10_50_cross'] = (df['sma_10'] > df['sma_50']).astype(int)
```

#### 4. Volume Features
```python
if 'Volume' in df.columns and df['Volume'].sum() > 0:
    df['volume_ratio_5'] = df['Volume'] / df['Volume'].rolling(5).mean()
    df['volume_ratio_20'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['volume_trend'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()
    df['vol_price_corr'] = df['returns_1d'].rolling(20).corr(df['Volume'].pct_change())
```

#### 5. Volatility Features
```python
for period in [5, 10, 20]:
    df[f'volatility_{period}'] = df['returns_1d'].rolling(period).std()

df['volatility_ratio'] = df['volatility_5'] / (df['volatility_20'] + 1e-10)
```

**Calculation:**
```
volatility_5 = Standard deviation of 1-day returns over 5 days
volatility_20 = Standard deviation of 1-day returns over 20 days
```

#### 6. Momentum Features
```python
for period in [5, 10, 20]:
    df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
```

**Formula:**
```
momentum_10 = (Price_today / Price_10_days_ago) - 1
```

#### 7. MACD (Moving Average Convergence Divergence)
```python
ema_12 = df['Close'].ewm(span=12).mean()
ema_26 = df['Close'].ewm(span=26).mean()
df['macd'] = ema_12 - ema_26
df['macd_signal'] = df['macd'].ewm(span=9).mean()
df['macd_hist'] = df['macd'] - df['macd_signal']
```

**Formula:**
```
MACD = EMA_12 - EMA_26
Signal = EMA_9 of MACD
Histogram = MACD - Signal
```

#### 8. Bollinger Bands
```python
sma = df['Close'].rolling(20).mean()
std = df['Close'].rolling(20).std()
df['bollinger_upper'] = sma + 2 * std
df['bollinger_lower'] = sma - 2 * std
df['bollinger_pct'] = (df['Close'] - sma) / (2 * std + 1e-10)
df['bollinger_width'] = (4 * std) / (sma + 1e-10)
```

**Formula:**
```
BBand_upper = SMA_20 + (2 × σ)
BBand_lower = SMA_20 - (2 × σ)
BBand_pct = (Price - SMA_20) / (2 × σ)

Range: -1 to 1 (>0.5 = near upper, <-0.5 = near lower)
```

---

## China-Specific Feature Engineering

The model replaces US-centric indicators (VIX, SPY, DXY) with China-relevant ones.

### Macro Indicators Downloaded
```python
tickers = {
    'CSI300': '000300.SS',      # CSI 300 Index (China's S&P 500 equivalent)
    'SSEC': '000001.SS',        # Shanghai Composite (SSE Composite Index)
    'HSI': '^HSI',              # Hang Seng Index (Hong Kong)
    'CNY': 'CNY=X',             # Yuan to USD exchange rate
    'GLD': 'GC=F',              # Gold (universal safe haven)
}
```

### Feature Calculations (10 China-Specific Features)

```python
# Feature 1-3: Distance from MA for CSI300, SSEC, HSI (5-day)
for col in ['CSI300', 'SSEC', 'HSI']:
    ma_col = macro_aligned[col].rolling(5).mean()
    df[f'{col}_dist_ma_5d'] = (macro_aligned[col] - ma_col) / ma_col

# Feature 4: CNY exchange rate (raw value)
df['CNY'] = macro_aligned['CNY']

# Feature 5: CSI300 distance from MA (20-day)
ma_20 = macro_aligned['CSI300'].rolling(20).mean()
df['CSI300_dist_ma_20d'] = (macro_aligned['CSI300'] - ma_20) / ma_20

# Feature 6: CSI300 momentum (5-day)
df['CSI300_momentum_5d'] = macro_aligned['CSI300'].pct_change(5)

# Feature 7: Beta to CSI300 (20-day rolling)
stock_returns = df['Close'].pct_change()
csi300_returns = macro_aligned['CSI300'].pct_change()
rolling_cov = stock_returns.rolling(20).cov(csi300_returns)
rolling_var = csi300_returns.rolling(20).var()
df['beta_csi300_20d'] = rolling_cov / rolling_var

# Feature 8: CSI300 momentum (20-day)
df['CSI300_momentum_20d'] = macro_aligned['CSI300'].pct_change(20)

# Feature 9: CNY momentum (5-day)
df['CNY_momentum_5d'] = macro_aligned['CNY'].pct_change(5)

# Feature 10: HSI/SSEC ratio (Hong Kong vs Shanghai sentiment)
df['HSI_SSEC_ratio'] = macro_aligned['HSI'] / macro_aligned['SSEC']
```

**Key Formulas:**
```
CSI300_dist_ma_5d = (CSI300_price - MA_5) / MA_5

beta_csi300 = Cov(stock_returns, CSI300_returns) / Var(CSI300_returns)

CNY_momentum_5d = (CNY_today / CNY_5_days_ago) - 1

HSI_SSEC_ratio = HSI_index / SSEC_index
```

### Model Training (Hybrid Ensemble)

The full ML pipeline uses a **HybridEnsemblePredictor** combining:
- **LightGBM** - Fast gradient boosting
- **XGBoost** - Robust gradient boosting
- **CatBoost** - Handles categorical features well
- **LSTM** - Captures sequential patterns

```python
# Train/Test Split
# Training: 90 samples, Validation: 60 samples (after feature engineering)

# Model Parameters (asset-specific)
params = {
    'iterations': 150-200,
    'learning_rate': 0.02-0.03,
    'depth': 4-6,
    'l2_leaf_reg': 3-5,
    'subsample': 0.8,
}
```

---

## Confidence Interval Calculation

### Direction Confidence (Signal Strength)

The system calculates confidence using a Signal-to-Noise Ratio (SNR) with sigmoid transformation:

#### SNR Calculation
```
SNR = |predicted_return| / historical_volatility

Where:
- predicted_return = return predicted by the model
- historical_volatility = recent realized volatility (last 20 days)
```

#### Asset-Specific SNR Thresholds
```python
SNR_THRESHOLDS = {
    'stock': 0.5,           # Standard threshold
    'china_stock': 0.5,     # Same as regular stocks
    'crypto': 0.8,          # Higher (crypto is inherently volatile)
    'forex': 0.3,           # Lower (forex moves smoothly)
    'commodity': 0.5,       # Standard threshold
}
```

#### Sigmoid Confidence Transformation
```python
import math

# Formula: sigmoid-like transformation of SNR
direction_confidence = 0.3 + 0.65 * (1 / (1 + math.exp(-1.5 * (snr - snr_threshold))))
```

**Examples for Stock (SNR threshold = 0.5):**
```
SNR = 0.3  → confidence = 0.35
SNR = 0.5  → confidence = 0.625 (threshold point)
SNR = 0.8  → confidence = 0.76
SNR = 1.2  → confidence = 0.87
```

### Volatility Confidence Interval

#### 80% Confidence Interval Bounds
```python
lower_bound = predicted_vol * 0.7
upper_bound = predicted_vol * 1.3
```

**Interpretation:**
```
If predicted_vol = 0.02 (2% daily volatility):
- Lower bound: 0.02 × 0.7 = 0.014 (1.4%)
- Upper bound: 0.02 × 1.3 = 0.026 (2.6%)

The model is 80% confident the actual volatility will be between 1.4% and 2.6% daily
```

### Regime Detection
```python
# Detect market regime from historical volatilities
# 0 = low volatility, 1 = medium, 2 = high

regime_names = ['low', 'medium', 'high']
# Or for signals: ['bullish', 'neutral', 'bearish']
```

---

## News Feed Generation

The system generates news and social media feeds based on predictions and company info.

### News Template Structure
```python
news_item = {
    'source': 'Bloomberg',      # News source
    'icon': '📰',              # Emoji icon
    'time': '45 min ago',      # Time reference
    'headline': 'Company shares rise...',  # Headline
    'url': 'https://...',       # Source URL
    'sentiment': 'positive',    # positive/negative/neutral
    'relevance': 0.85,         # 0-1 relevance score
}
```

### Stock-Specific News Generation

#### For Bullish Signals (direction > 0):
```python
news_items.append({
    'source': 'Bloomberg',
    'headline': f'{company_name} shares rise on strong earnings outlook',
    'sentiment': 'positive',
    'relevance': 0.85
})

news_items.append({
    'source': 'Twitter',
    'headline': f'Analysts upgrade {ticker} to Buy with ${current_price * 1.15:.0f} price target',
    'sentiment': 'positive',
    'relevance': 0.75
})
```

#### For Bearish Signals (direction < 0):
```python
news_items.append({
    'source': 'Reuters',
    'headline': f'{company_name} faces headwinds amid market concerns',
    'sentiment': 'negative',
    'relevance': 0.80
})
```

### Exchange-Specific News
```python
# For Hong Kong stocks
if exchange == 'HKEX':
    news_items.append({
        'source': 'CNBC',
        'headline': f'Market update: HKEX trading {"higher" if direction > 0 else "lower"} in afternoon session',
        'sentiment': 'positive' if direction > 0 else 'negative',
        'relevance': 0.70
    })
```

### Volatility-Based News
```python
if volatility > 0.03:  # 3% daily volatility threshold
    news_items.append({
        'source': 'MarketWatch',
        'headline': f'{ticker} volatility spikes - Options traders brace for big moves',
        'sentiment': 'neutral',
        'relevance': 0.75
    })
```

---

## DeepSeek API Integration

DeepSeek is integrated for sentiment-based confidence adjustment and China-specific analysis in the China Top Picks.

### DeepSeek Analyzer Class
```python
class DeepSeekAnalyzer:
    """Analyzes stocks using DeepSeek AI for sentiment-driven signals"""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('DEEPSEEK_API_KEY')
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
```

### API Request Format
```python
def get_comprehensive_analysis(self, ticker, stock_name, sector):
    """Get DeepSeek analysis for a stock"""

    prompt = f"""Analyze {stock_name} ({ticker}) in the {sector} sector for trading signal.

    Provide your analysis in this exact format:
    Direction: (1 for buy, 0 for neutral, -1 for sell)
    Confidence: (0.3 to 0.95)
    Reason: (brief explanation)

    Consider:
    - Policy sentiment (PBOC/CSRC)
    - Social sentiment (Weibo/Xueqiu)
    - Retail sentiment (trading patterns)
    - Sector policy alignment
    """

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 150
    }
```

### Integration with China Model

```python
# Get DeepSeek analysis
deepseek_analysis = analyzer.get_comprehensive_analysis(ticker, stock_name, sector)

# Combine with ML prediction (weighted average)
if deepseek_analysis['deepseek_used']:
    final_direction = (ml_direction * 0.6 + deepseek_analysis['direction'] * 0.4)
    final_confidence = (ml_confidence * 0.6 + deepseek_analysis['confidence'] * 0.4)
```

---

## Complete Calculation Workflow

### Step 1: Data Preparation
```
Fetch historical data (1500 days max)
├─ Check data availability
├─ Classify into tier (insufficient/minimal/light/full)
└─ Handle missing values (forward fill, backward fill)
```

### Step 2: Feature Engineering
```
Add features based on tier:
├─ Basic (5 features): momentum, volume trend, SMA trend
├─ Light (20+ features): + RSI, MACD, Bollinger, volatility
├─ Full (90+ features): + all technical + macro features
└─ China-specific (10 features): CSI300, HSI, CNY correlations
```

### Step 3: Model Training/Prediction
```
If ML available (150+ days):
├─ Train HybridEnsemble (LightGBM + XGBoost + CatBoost + LSTM)
├─ Generate volatility and direction predictions
├─ Calculate confidence using SNR transformation
└─ Return full prediction with all metrics

If ML unavailable (<150 days):
├─ Use basic technical analysis
├─ Calculate momentum, volume, trend
├─ Set confidence = 0.3 (low)
└─ Return basic prediction
```

### Step 4: Signal Generation
```
Generate trading signal:
├─ Direction: BUY (1), SELL (-1), HOLD (0)
├─ Confidence: 0.3 to 0.95
├─ Entry price, stop loss, take profit
├─ Position sizing based on risk
└─ Risk/reward ratio
```

### Step 5: News & Sentiment
```
Generate news feed:
├─ Create asset-specific news templates
├─ Add exchange-specific market updates
├─ Include volatility alerts if applicable
└─ Apply NLP sentiment analysis (FinBERT/VADER)
```

---

## Summary of Key Metrics

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|-----------------|
| **Momentum** | (Price_t / Price_t-n) - 1 | -∞ to +∞ | Positive = bullish |
| **RSI** | 100 - (100 / (1 + RS)) | 0-100 | >70 overbought, <30 oversold |
| **Volatility** | σ(returns) × √252 | 0-∞ | Higher = more risk |
| **Direction Confidence** | 0.3 + 0.65 / (1 + e^(-1.5×(SNR-threshold))) | 0.3-0.95 | Higher = more certain |
| **Beta** | Cov(stock, CSI300) / Var(CSI300) | -∞ to +∞ | 1.0 = moves with market |
| **CSI300_dist_ma** | (CSI300 - MA) / MA | -1 to +1 | Distance from moving average |

---

## File References

- **Tier Classification**: `webapp.py:get_data_tier()` (lines 1591-1635)
- **Basic Analysis**: `webapp.py:basic_technical_analysis()` (lines 1638-1699)
- **Predict Endpoint**: `webapp.py:predict()` (lines 2568-2780)
- **China Macro Features**: `china_model/src/china_macro_features.py`
- **News Generation**: `webapp.py:generate_news_feed()` (lines 1394-1590)
- **DeepSeek Integration**: `china_model/src/deepseek_analyzer.py`

---

*Last Updated: November 29, 2025*
*China Stock Model v2.0 - Production Ready*
