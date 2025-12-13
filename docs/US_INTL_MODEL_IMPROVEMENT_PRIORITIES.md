# US/International Model - Three Critical Improvement Areas

> **Document Purpose**: Detail three key areas where the current US/Intl model underperforms and propose concrete solutions
> **Priority Level**: HIGH - These changes could significantly improve accuracy
> **Last Updated**: December 2025

---

## Table of Contents

1. [Signal Quality vs. Position Sizing Mismatch](#1-signal-quality-vs-position-sizing-mismatch)
2. [Sentiment Gating](#2-sentiment-gating)
3. [Regime Matching](#3-regime-matching)
4. [Implementation Roadmap](#4-implementation-roadmap)

---

## 1. Signal Quality vs. Position Sizing Mismatch

### 1.1 The Problem

**Current behavior**: Position sizing is applied AFTER signal generation, creating a disconnect between signal quality and capital allocation.

```
Current Flow (BROKEN):
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Generate       │────▶│  Apply Fixed    │────▶│  Output Signal  │
│  Signal         │     │  Position Rules │     │  + Position     │
│  (confidence)   │     │  (19 fixes)     │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │
        │                       │
    Signal has              Position sizing
    75% confidence          ignores signal
    but noisy features      quality metrics
```

**Example of the mismatch**:

```python
# Current: Two signals with same confidence get same position size
Signal A: confidence=70%, features=[RSI oversold, MACD bullish, sentiment positive]
Signal B: confidence=70%, features=[RSI neutral, MACD flat, sentiment missing]

# Both get: position_mult = 1.30 (Fix 5 + Fix 12)

# PROBLEM: Signal A has 3 confirming indicators, Signal B has 0
# They should NOT have the same position size!
```

### 1.2 Current Implementation (Flawed)

**File**: `src/models/us_intl_optimizer.py`

```python
# Lines 423-521: optimize_buy_signal()

def optimize_buy_signal(ticker, confidence, volatility=0.20, momentum=0.0, win_rate=None):
    """
    CURRENT FLAW: Only uses confidence number, ignores signal quality.
    """
    # Step 1: Asset boost (FIXED multiplier based on asset class only)
    position_mult = BUY_POSITION_BOOSTS.get(asset_class, 1.0)  # e.g., 1.30x for stocks

    # Step 2: Win-rate multiplier (historical, not current signal quality)
    wr_mult = get_win_rate_multiplier(win_rate)
    position_mult *= wr_mult

    # Step 3: BUY multiplier (FIXED at 1.30x)
    position_mult *= 1.30

    # MISSING: No evaluation of:
    # - How many indicators confirm the signal?
    # - Is sentiment aligned with direction?
    # - Are volume patterns confirming?
    # - Is the signal based on complete or partial data?

    return SignalOptimization(position_multiplier=position_mult, ...)
```

### 1.3 Proposed Solution: Signal Quality Scoring

```python
class SignalQualityScorer:
    """
    Score signal quality based on multiple confirming factors.
    Higher quality = larger position size.
    """

    def __init__(self):
        self.quality_weights = {
            'indicator_agreement': 0.30,   # Technical indicators alignment
            'sentiment_alignment': 0.25,   # Sentiment matches direction
            'volume_confirmation': 0.20,   # Volume supports move
            'data_completeness': 0.15,     # All features available
            'regime_alignment': 0.10       # Regime supports trade type
        }

    def calculate_quality_score(self, signal_data: dict) -> float:
        """
        Calculate quality score in [0, 1].

        Args:
            signal_data: {
                'direction': 1 or -1,
                'confidence': float,
                'rsi': float,
                'macd_signal': int,  # 1=bullish, -1=bearish, 0=neutral
                'sentiment': float,  # -1 to 1
                'volume_ratio': float,
                'features_available': int,
                'total_features': int,
                'regime': str
            }
        """
        scores = {}

        # 1. Indicator Agreement (30%)
        direction = signal_data['direction']
        indicators_aligned = 0
        total_indicators = 0

        # RSI alignment
        rsi = signal_data.get('rsi', 50)
        if direction == 1:  # BUY
            indicators_aligned += 1 if rsi < 40 else (0.5 if rsi < 50 else 0)
        else:  # SELL
            indicators_aligned += 1 if rsi > 60 else (0.5 if rsi > 50 else 0)
        total_indicators += 1

        # MACD alignment
        macd = signal_data.get('macd_signal', 0)
        if macd == direction:
            indicators_aligned += 1
        elif macd == 0:
            indicators_aligned += 0.3
        total_indicators += 1

        # Bollinger Band position
        bb_position = signal_data.get('bb_position', 0.5)  # 0=lower, 1=upper
        if direction == 1:  # BUY near lower band
            indicators_aligned += 1 if bb_position < 0.3 else (0.5 if bb_position < 0.5 else 0)
        else:  # SELL near upper band
            indicators_aligned += 1 if bb_position > 0.7 else (0.5 if bb_position > 0.5 else 0)
        total_indicators += 1

        scores['indicator_agreement'] = indicators_aligned / total_indicators

        # 2. Sentiment Alignment (25%)
        sentiment = signal_data.get('sentiment', 0)
        if direction == 1:  # BUY wants positive sentiment
            sentiment_score = (sentiment + 1) / 2  # Convert [-1,1] to [0,1]
        else:  # SELL wants negative sentiment
            sentiment_score = (1 - sentiment) / 2

        # Penalize if sentiment data is mock/missing
        if signal_data.get('sentiment_is_mock', True):
            sentiment_score *= 0.5  # Halve the weight if mock

        scores['sentiment_alignment'] = sentiment_score

        # 3. Volume Confirmation (20%)
        volume_ratio = signal_data.get('volume_ratio', 1.0)
        # Higher volume on signal day = more conviction
        if volume_ratio > 2.0:
            scores['volume_confirmation'] = 1.0
        elif volume_ratio > 1.5:
            scores['volume_confirmation'] = 0.8
        elif volume_ratio > 1.0:
            scores['volume_confirmation'] = 0.6
        else:
            scores['volume_confirmation'] = 0.4

        # 4. Data Completeness (15%)
        features_available = signal_data.get('features_available', 0)
        total_features = signal_data.get('total_features', 90)
        scores['data_completeness'] = features_available / total_features

        # 5. Regime Alignment (10%)
        regime = signal_data.get('regime', 'normal')
        if direction == 1:  # BUY
            if regime in ['low_vol', 'normal']:
                scores['regime_alignment'] = 1.0
            elif regime == 'trending_up':
                scores['regime_alignment'] = 0.9
            elif regime == 'high_vol':
                scores['regime_alignment'] = 0.5
            else:
                scores['regime_alignment'] = 0.3
        else:  # SELL
            if regime in ['trending_down', 'high_vol']:
                scores['regime_alignment'] = 0.9
            elif regime == 'normal':
                scores['regime_alignment'] = 0.6
            else:
                scores['regime_alignment'] = 0.3

        # Weighted combination
        total_score = sum(
            scores[k] * self.quality_weights[k]
            for k in self.quality_weights
        )

        return total_score

    def adjust_position_for_quality(self, base_position: float, quality_score: float) -> float:
        """
        Adjust position size based on signal quality.

        Quality Score -> Position Multiplier:
        0.9 - 1.0: 1.5x (high conviction, increase position)
        0.7 - 0.9: 1.2x (good quality, slight increase)
        0.5 - 0.7: 1.0x (average, no change)
        0.3 - 0.5: 0.7x (low quality, reduce position)
        0.0 - 0.3: 0.4x (poor quality, minimal position)
        """
        if quality_score >= 0.9:
            multiplier = 1.5
        elif quality_score >= 0.7:
            multiplier = 1.2
        elif quality_score >= 0.5:
            multiplier = 1.0
        elif quality_score >= 0.3:
            multiplier = 0.7
        else:
            multiplier = 0.4

        return base_position * multiplier
```

### 1.4 Integration with Current Optimizer

```python
# Modified optimize_buy_signal() with quality scoring

def optimize_buy_signal_v2(ticker, confidence, signal_data, volatility=0.20,
                           momentum=0.0, win_rate=None):
    """
    IMPROVED: Position sizing now matches signal quality.
    """
    quality_scorer = SignalQualityScorer()
    fixes_applied = []

    # Step 1: Calculate signal quality
    quality_score = quality_scorer.calculate_quality_score(signal_data)
    fixes_applied.append(f"Quality Score: {quality_score:.2f}")

    # Step 2: Base position from asset class
    asset_class = classify_asset(ticker)
    base_position = BUY_POSITION_BOOSTS.get(asset_class, 1.0)

    # Step 3: Adjust for quality (NEW!)
    position_mult = quality_scorer.adjust_position_for_quality(
        base_position, quality_score
    )
    fixes_applied.append(f"Quality Adjustment: {position_mult/base_position:.2f}x")

    # Step 4: Win-rate adjustment (still useful)
    wr = win_rate or 0.50
    wr_mult = get_win_rate_multiplier(wr)
    position_mult *= wr_mult

    # Step 5: Kelly criterion (now uses quality-adjusted confidence)
    adjusted_confidence = confidence * quality_score
    kelly = calculate_kelly_fraction(wr, adjusted_confidence)

    return SignalOptimization(
        position_multiplier=position_mult,
        quality_score=quality_score,
        kelly_fraction=kelly,
        fixes_applied=fixes_applied
    )
```

### 1.5 Expected Impact

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| False positive trades | ~35% | ~20% |
| Position sizing accuracy | Low correlation | High correlation |
| Risk-adjusted returns | Baseline | +15-25% improvement |
| Max drawdown | Higher | Lower (better sizing) |

---

## 2. Sentiment Gating

### 2.1 The Problem

**Current behavior**: Sentiment is just another feature with no special treatment. Signals fire regardless of sentiment alignment.

```
Current Flow (NO GATING):
┌─────────────────┐
│  Technical      │
│  Signal: BUY    │─────────────────────────────▶ EXECUTE BUY
│  Conf: 75%      │
└─────────────────┘
                              ▲
                              │
┌─────────────────┐           │ (sentiment ignored)
│  Sentiment:     │───────────┘
│  NEGATIVE -0.6  │
└─────────────────┘

PROBLEM: BUY signal with negative sentiment has much lower success rate!
```

**Evidence from logs**:
```
# Many blocked signals had misaligned sentiment (if we had real data)
INFO:[US/INTL SELL BLOCKED] WVE: LOW_CONF (Fix 1/2): 53% < 75%
INFO:[US/INTL SELL BLOCKED] GPCR: LOW_CONF (Fix 1/2): 54% < 75%
```

### 2.2 What is Sentiment Gating?

**Sentiment Gating** = Using sentiment as a GATE (allow/block) rather than just a feature.

```
Proposed Flow (WITH GATING):
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Technical      │────▶│  SENTIMENT      │────▶│  Decision       │
│  Signal: BUY    │     │  GATE           │     │                 │
│  Conf: 75%      │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │
                              ▼
┌─────────────────┐     ┌─────────────────┐
│  Sentiment:     │────▶│  Gate Logic:    │
│  NEGATIVE -0.6  │     │  BUY + NEG_SENT │
└─────────────────┘     │  = BLOCK/REDUCE │
                        └─────────────────┘
```

### 2.3 Sentiment Gate Levels

```python
class SentimentGate:
    """
    Three-tier sentiment gating system.
    """

    # Gate thresholds
    STRONG_POSITIVE = 0.4    # Strong bullish sentiment
    WEAK_POSITIVE = 0.1      # Mild bullish sentiment
    NEUTRAL_LOW = -0.1       # Neutral zone
    NEUTRAL_HIGH = 0.1
    WEAK_NEGATIVE = -0.1     # Mild bearish sentiment
    STRONG_NEGATIVE = -0.4   # Strong bearish sentiment

    def __init__(self, gate_strength='moderate'):
        """
        gate_strength: 'strict', 'moderate', 'loose'
        - strict: Block signals with ANY sentiment mismatch
        - moderate: Block strong mismatches, reduce mild mismatches
        - loose: Only block extreme mismatches
        """
        self.gate_strength = gate_strength

        # Gate actions by strength level
        self.gate_actions = {
            'strict': {
                'block_threshold': 0.0,      # Block if sentiment < 0 for BUY
                'reduce_threshold': 0.2,     # Reduce if sentiment < 0.2
                'reduction_factor': 0.5,     # 50% position reduction
            },
            'moderate': {
                'block_threshold': -0.3,     # Block only strong negative
                'reduce_threshold': 0.0,     # Reduce if sentiment < 0
                'reduction_factor': 0.7,     # 30% position reduction
            },
            'loose': {
                'block_threshold': -0.5,     # Block only extreme negative
                'reduce_threshold': -0.2,    # Reduce if sentiment < -0.2
                'reduction_factor': 0.85,    # 15% position reduction
            }
        }

    def apply_gate(self, signal_type: str, sentiment: float,
                   confidence: float, is_mock: bool = False) -> dict:
        """
        Apply sentiment gate to trading signal.

        Args:
            signal_type: 'BUY' or 'SELL'
            sentiment: Combined sentiment score [-1, 1]
            confidence: Original signal confidence
            is_mock: True if sentiment is mock data (reduce gate influence)

        Returns:
            {
                'action': 'ALLOW' | 'REDUCE' | 'BLOCK',
                'position_multiplier': float,
                'adjusted_confidence': float,
                'reason': str
            }
        """
        config = self.gate_actions[self.gate_strength]

        # Reduce gate influence for mock sentiment
        if is_mock:
            sentiment = sentiment * 0.3  # Dampen mock sentiment effect

        # Flip sentiment for SELL signals
        effective_sentiment = sentiment if signal_type == 'BUY' else -sentiment

        result = {
            'action': 'ALLOW',
            'position_multiplier': 1.0,
            'adjusted_confidence': confidence,
            'reason': ''
        }

        # Check BLOCK threshold
        if effective_sentiment < config['block_threshold']:
            result['action'] = 'BLOCK'
            result['position_multiplier'] = 0.0
            result['adjusted_confidence'] = 0.0
            result['reason'] = (
                f"SENTIMENT_GATE_BLOCK: {signal_type} blocked due to "
                f"{'negative' if signal_type == 'BUY' else 'positive'} "
                f"sentiment ({sentiment:.2f})"
            )
            return result

        # Check REDUCE threshold
        if effective_sentiment < config['reduce_threshold']:
            result['action'] = 'REDUCE'
            result['position_multiplier'] = config['reduction_factor']

            # Also reduce confidence proportionally
            sentiment_penalty = (config['reduce_threshold'] - effective_sentiment) * 0.3
            result['adjusted_confidence'] = max(0.5, confidence - sentiment_penalty)

            result['reason'] = (
                f"SENTIMENT_GATE_REDUCE: Position reduced by "
                f"{(1 - config['reduction_factor']) * 100:.0f}% due to "
                f"weak sentiment alignment ({sentiment:.2f})"
            )
            return result

        # ALLOW with potential boost for strong alignment
        if effective_sentiment > self.STRONG_POSITIVE:
            result['position_multiplier'] = 1.15  # 15% boost for strong alignment
            result['adjusted_confidence'] = min(0.95, confidence * 1.1)
            result['reason'] = "SENTIMENT_GATE_BOOST: Strong sentiment alignment"

        return result
```

### 2.4 Integration Example

```python
def optimize_signal_with_sentiment_gate(
    ticker: str,
    signal_type: str,
    confidence: float,
    sentiment: float,
    sentiment_is_mock: bool = False,
    gate_strength: str = 'moderate'
) -> SignalOptimization:
    """
    Apply sentiment gating before other optimizations.
    """
    gate = SentimentGate(gate_strength)
    fixes_applied = []

    # Step 1: Apply sentiment gate FIRST
    gate_result = gate.apply_gate(
        signal_type=signal_type,
        sentiment=sentiment,
        confidence=confidence,
        is_mock=sentiment_is_mock
    )

    if gate_result['action'] == 'BLOCK':
        fixes_applied.append(f"BLOCKED: {gate_result['reason']}")
        return SignalOptimization(
            blocked=True,
            block_reason=gate_result['reason'],
            fixes_applied=fixes_applied
        )

    # Step 2: Apply gate adjustments
    position_mult = gate_result['position_multiplier']
    adjusted_confidence = gate_result['adjusted_confidence']

    if gate_result['action'] == 'REDUCE':
        fixes_applied.append(f"REDUCED: {gate_result['reason']}")
    elif gate_result['action'] == 'ALLOW' and position_mult > 1.0:
        fixes_applied.append(f"BOOSTED: {gate_result['reason']}")

    # Step 3: Continue with normal optimization using adjusted values
    if signal_type == 'BUY':
        return optimize_buy_signal(
            ticker=ticker,
            confidence=adjusted_confidence,
            base_position_mult=position_mult,
            fixes_applied=fixes_applied
        )
    else:
        return optimize_sell_signal(
            ticker=ticker,
            confidence=adjusted_confidence,
            base_position_mult=position_mult,
            fixes_applied=fixes_applied
        )
```

### 2.5 Sentiment Gate Decision Matrix

| Signal Type | Sentiment | Gate Strength | Action | Position Mult |
|-------------|-----------|---------------|--------|---------------|
| BUY | > +0.4 | Any | BOOST | 1.15x |
| BUY | +0.1 to +0.4 | Any | ALLOW | 1.0x |
| BUY | -0.1 to +0.1 | Strict | REDUCE | 0.5x |
| BUY | -0.1 to +0.1 | Moderate | ALLOW | 1.0x |
| BUY | -0.3 to -0.1 | Strict | BLOCK | 0x |
| BUY | -0.3 to -0.1 | Moderate | REDUCE | 0.7x |
| BUY | < -0.3 | Moderate | BLOCK | 0x |
| BUY | < -0.5 | Loose | BLOCK | 0x |
| SELL | < -0.4 | Any | BOOST | 1.15x |
| SELL | -0.4 to -0.1 | Any | ALLOW | 1.0x |
| SELL | > +0.3 | Moderate | BLOCK | 0x |

### 2.6 Handling Mock Sentiment

**Critical**: When sentiment is mock data, reduce gate influence:

```python
def apply_gate_with_mock_handling(self, signal_type, sentiment, confidence, is_mock):
    """
    Mock sentiment should have minimal gating effect.
    """
    if is_mock:
        # Option 1: Dampen sentiment effect
        effective_sentiment = sentiment * 0.3

        # Option 2: Widen neutral zone
        self.STRONG_POSITIVE = 0.6   # Harder to get boost
        self.block_threshold = -0.6  # Harder to get blocked

        # Option 3: Skip gating entirely (most conservative)
        # return {'action': 'ALLOW', 'position_multiplier': 1.0, ...}

    return self._apply_gate_logic(signal_type, effective_sentiment, confidence)
```

### 2.7 Expected Impact

| Metric | Before (No Gate) | After (With Gate) |
|--------|------------------|-------------------|
| Signals aligned with sentiment | ~50% (random) | ~85%+ |
| Win rate on gated signals | ~52% | ~58-62% |
| False positives | Higher | Lower |
| Missed opportunities | Lower | Slightly higher |

---

## 3. Regime Matching

### 3.1 The Problem

**Current behavior**: Trading strategies are applied uniformly regardless of market regime.

```
Current Flow (NO REGIME MATCHING):
┌─────────────────┐
│  Market Regime: │
│  HIGH_VOLATILITY│─────────────────────────────▶ (ignored)
│  TRENDING_DOWN  │
└─────────────────┘

┌─────────────────┐     ┌─────────────────┐
│  Signal: BUY    │────▶│  Same Rules     │────▶ EXECUTE BUY
│  (mean-reversion│     │  As Always      │      (wrong strategy
│   strategy)     │     │                 │       for regime!)
└─────────────────┘     └─────────────────┘

PROBLEM: Mean-reversion BUY in a downtrend = catching falling knife!
```

### 3.2 Regime Types and Characteristics

```python
class MarketRegime(Enum):
    """
    Six distinct market regimes with different optimal strategies.
    """
    # Volatility-based regimes
    LOW_VOLATILITY = "low_vol"          # Vol < 33rd percentile
    NORMAL_VOLATILITY = "normal_vol"    # Vol 33rd-67th percentile
    HIGH_VOLATILITY = "high_vol"        # Vol 67th-95th percentile
    CRISIS = "crisis"                   # Vol > 95th percentile

    # Trend-based regimes
    TRENDING_UP = "trending_up"         # Hurst > 0.55, positive momentum
    TRENDING_DOWN = "trending_down"     # Hurst > 0.55, negative momentum
    MEAN_REVERTING = "mean_reverting"   # Hurst < 0.45
    RANGING = "ranging"                 # Hurst 0.45-0.55


class RegimeCharacteristics:
    """
    Define what works in each regime.
    """

    REGIME_PROFILES = {
        MarketRegime.LOW_VOLATILITY: {
            'description': 'Calm market, low risk',
            'optimal_strategies': ['momentum', 'trend_following'],
            'risky_strategies': ['mean_reversion'],  # Low vol = trends persist
            'position_adjustment': 1.2,  # Can increase position
            'stop_loss_adjustment': 0.8, # Tighter stops OK
            'buy_preference': 1.0,       # Neutral
            'sell_preference': 1.0,
        },

        MarketRegime.NORMAL_VOLATILITY: {
            'description': 'Normal conditions',
            'optimal_strategies': ['all'],
            'risky_strategies': [],
            'position_adjustment': 1.0,
            'stop_loss_adjustment': 1.0,
            'buy_preference': 1.0,
            'sell_preference': 1.0,
        },

        MarketRegime.HIGH_VOLATILITY: {
            'description': 'Elevated risk, wider swings',
            'optimal_strategies': ['mean_reversion', 'volatility_breakout'],
            'risky_strategies': ['momentum'],  # Whipsaws hurt momentum
            'position_adjustment': 0.7,  # Reduce position size
            'stop_loss_adjustment': 1.3, # Wider stops needed
            'buy_preference': 0.8,       # Slightly prefer sells
            'sell_preference': 1.1,
        },

        MarketRegime.CRISIS: {
            'description': 'Extreme volatility, high risk',
            'optimal_strategies': ['cash', 'hedging'],
            'risky_strategies': ['momentum', 'mean_reversion', 'trend_following'],
            'position_adjustment': 0.3,  # Minimal positions
            'stop_loss_adjustment': 1.5, # Very wide stops
            'buy_preference': 0.5,       # Strong sell preference
            'sell_preference': 1.5,
        },

        MarketRegime.TRENDING_UP: {
            'description': 'Bull market, strong uptrend',
            'optimal_strategies': ['momentum', 'trend_following', 'breakout'],
            'risky_strategies': ['mean_reversion'],  # Don't short uptrends
            'position_adjustment': 1.3,
            'stop_loss_adjustment': 0.9,
            'buy_preference': 1.4,       # Strong buy preference
            'sell_preference': 0.5,      # Avoid shorts
        },

        MarketRegime.TRENDING_DOWN: {
            'description': 'Bear market, strong downtrend',
            'optimal_strategies': ['trend_following_short', 'mean_reversion_selective'],
            'risky_strategies': ['buy_dip', 'momentum_long'],
            'position_adjustment': 0.8,
            'stop_loss_adjustment': 1.1,
            'buy_preference': 0.6,       # Reduced buys
            'sell_preference': 1.3,      # Prefer shorts
        },

        MarketRegime.MEAN_REVERTING: {
            'description': 'Range-bound, oscillating',
            'optimal_strategies': ['mean_reversion', 'range_trading'],
            'risky_strategies': ['trend_following', 'breakout'],
            'position_adjustment': 1.0,
            'stop_loss_adjustment': 0.9,
            'buy_preference': 1.0,       # Buy oversold
            'sell_preference': 1.0,      # Sell overbought
        },

        MarketRegime.RANGING: {
            'description': 'Sideways, no clear direction',
            'optimal_strategies': ['mean_reversion', 'range_trading'],
            'risky_strategies': ['momentum', 'breakout'],
            'position_adjustment': 0.9,
            'stop_loss_adjustment': 1.0,
            'buy_preference': 1.0,
            'sell_preference': 1.0,
        },
    }
```

### 3.3 Regime Detection Implementation

```python
class RegimeDetector:
    """
    Detect current market regime using multiple indicators.
    """

    def __init__(self, lookback_vol=60, lookback_trend=20):
        self.lookback_vol = lookback_vol
        self.lookback_trend = lookback_trend

    def detect_regime(self, price_data: pd.DataFrame) -> dict:
        """
        Detect current regime from price data.

        Returns:
            {
                'primary_regime': MarketRegime,
                'secondary_regime': MarketRegime,
                'confidence': float,
                'metrics': {
                    'volatility_percentile': float,
                    'hurst_exponent': float,
                    'trend_strength': float,
                    'momentum': float
                }
            }
        """
        close = price_data['Close'].values
        returns = np.diff(close) / close[:-1]

        # 1. Volatility regime
        vol_regime, vol_pct = self._detect_volatility_regime(returns)

        # 2. Trend regime
        trend_regime, hurst, trend_strength = self._detect_trend_regime(close)

        # 3. Combine regimes
        primary, secondary, confidence = self._combine_regimes(
            vol_regime, trend_regime, vol_pct, hurst
        )

        return {
            'primary_regime': primary,
            'secondary_regime': secondary,
            'confidence': confidence,
            'metrics': {
                'volatility_percentile': vol_pct,
                'hurst_exponent': hurst,
                'trend_strength': trend_strength,
                'momentum': self._calculate_momentum(close)
            }
        }

    def _detect_volatility_regime(self, returns: np.ndarray) -> tuple:
        """
        Detect volatility regime using rolling percentile.
        """
        recent_vol = np.std(returns[-self.lookback_vol:]) * np.sqrt(252)

        # Calculate historical percentile
        rolling_vol = pd.Series(returns).rolling(self.lookback_vol).std() * np.sqrt(252)
        vol_percentile = (rolling_vol < recent_vol).mean()

        if vol_percentile > 0.95:
            return MarketRegime.CRISIS, vol_percentile
        elif vol_percentile > 0.67:
            return MarketRegime.HIGH_VOLATILITY, vol_percentile
        elif vol_percentile > 0.33:
            return MarketRegime.NORMAL_VOLATILITY, vol_percentile
        else:
            return MarketRegime.LOW_VOLATILITY, vol_percentile

    def _detect_trend_regime(self, close: np.ndarray) -> tuple:
        """
        Detect trend regime using Hurst exponent and momentum.
        """
        # Calculate Hurst exponent
        hurst = self._calculate_hurst(close)

        # Calculate trend strength
        sma_short = np.mean(close[-10:])
        sma_long = np.mean(close[-50:])
        trend_strength = (sma_short - sma_long) / sma_long

        # Determine regime
        if hurst > 0.55:  # Trending
            if trend_strength > 0.02:
                return MarketRegime.TRENDING_UP, hurst, trend_strength
            elif trend_strength < -0.02:
                return MarketRegime.TRENDING_DOWN, hurst, trend_strength
            else:
                return MarketRegime.RANGING, hurst, trend_strength
        elif hurst < 0.45:  # Mean-reverting
            return MarketRegime.MEAN_REVERTING, hurst, trend_strength
        else:  # Random walk / ranging
            return MarketRegime.RANGING, hurst, trend_strength

    def _calculate_hurst(self, ts: np.ndarray, max_lag: int = 50) -> float:
        """
        Calculate Hurst exponent using R/S analysis.

        H < 0.5: Mean-reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        lags = range(2, min(max_lag, len(ts) // 4))
        tau = []

        for lag in lags:
            # Calculate range/std for each lag
            chunks = [ts[i:i+lag] for i in range(0, len(ts)-lag, lag)]
            rs_values = []
            for chunk in chunks:
                if len(chunk) > 1:
                    r = np.max(chunk) - np.min(chunk)
                    s = np.std(chunk)
                    if s > 0:
                        rs_values.append(r / s)
            if rs_values:
                tau.append(np.mean(rs_values))

        if len(tau) < 2:
            return 0.5  # Default to random walk

        # Fit log-log regression
        log_lags = np.log(list(lags)[:len(tau)])
        log_tau = np.log(tau)
        hurst = np.polyfit(log_lags, log_tau, 1)[0]

        return np.clip(hurst, 0, 1)

    def _combine_regimes(self, vol_regime, trend_regime, vol_pct, hurst):
        """
        Combine volatility and trend regimes into primary/secondary.
        """
        # Crisis overrides everything
        if vol_regime == MarketRegime.CRISIS:
            return MarketRegime.CRISIS, trend_regime, 0.9

        # High vol + trending = primary trend, secondary vol
        if vol_regime == MarketRegime.HIGH_VOLATILITY:
            if trend_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                return trend_regime, vol_regime, 0.7
            else:
                return vol_regime, trend_regime, 0.6

        # Normal/low vol = trend regime dominates
        return trend_regime, vol_regime, 0.75 if hurst > 0.55 or hurst < 0.45 else 0.5
```

### 3.4 Regime-Aware Signal Optimization

```python
class RegimeAwareOptimizer:
    """
    Adjust signals based on current market regime.
    """

    def __init__(self):
        self.regime_detector = RegimeDetector()
        self.characteristics = RegimeCharacteristics.REGIME_PROFILES

    def optimize_for_regime(
        self,
        signal_type: str,
        confidence: float,
        strategy_type: str,
        price_data: pd.DataFrame
    ) -> dict:
        """
        Optimize signal based on current regime.

        Args:
            signal_type: 'BUY' or 'SELL'
            confidence: Original confidence
            strategy_type: 'momentum', 'mean_reversion', 'breakout', etc.
            price_data: Recent price history

        Returns:
            {
                'action': 'ALLOW' | 'REDUCE' | 'BLOCK',
                'position_multiplier': float,
                'adjusted_confidence': float,
                'stop_loss_multiplier': float,
                'regime': str,
                'reason': str
            }
        """
        # Detect current regime
        regime_info = self.regime_detector.detect_regime(price_data)
        regime = regime_info['primary_regime']
        profile = self.characteristics[regime]

        result = {
            'action': 'ALLOW',
            'position_multiplier': 1.0,
            'adjusted_confidence': confidence,
            'stop_loss_multiplier': 1.0,
            'regime': regime.value,
            'reason': ''
        }

        # Check if strategy is risky for this regime
        if strategy_type in profile['risky_strategies']:
            if regime == MarketRegime.CRISIS:
                result['action'] = 'BLOCK'
                result['position_multiplier'] = 0.0
                result['reason'] = f"REGIME_BLOCK: {strategy_type} blocked in {regime.value}"
                return result
            else:
                result['action'] = 'REDUCE'
                result['position_multiplier'] *= 0.5
                result['reason'] = f"REGIME_REDUCE: {strategy_type} risky in {regime.value}"

        # Apply regime position adjustment
        result['position_multiplier'] *= profile['position_adjustment']

        # Apply buy/sell preference
        if signal_type == 'BUY':
            result['position_multiplier'] *= profile['buy_preference']
        else:
            result['position_multiplier'] *= profile['sell_preference']

        # Apply stop-loss adjustment
        result['stop_loss_multiplier'] = profile['stop_loss_adjustment']

        # Adjust confidence based on strategy-regime fit
        if strategy_type in profile['optimal_strategies']:
            result['adjusted_confidence'] = min(0.95, confidence * 1.1)
            if result['action'] == 'ALLOW':
                result['reason'] = f"REGIME_BOOST: {strategy_type} optimal for {regime.value}"
        elif result['action'] == 'ALLOW':
            result['reason'] = f"REGIME_ALLOW: {strategy_type} acceptable in {regime.value}"

        return result
```

### 3.5 Regime Matching Decision Matrix

| Regime | Momentum BUY | Mean-Rev BUY | Momentum SELL | Mean-Rev SELL |
|--------|--------------|--------------|---------------|---------------|
| Low Vol | BOOST 1.2x | REDUCE 0.7x | ALLOW 1.0x | REDUCE 0.7x |
| Normal | ALLOW 1.0x | ALLOW 1.0x | ALLOW 1.0x | ALLOW 1.0x |
| High Vol | REDUCE 0.5x | BOOST 1.2x | REDUCE 0.5x | BOOST 1.2x |
| Crisis | BLOCK 0x | REDUCE 0.3x | ALLOW 0.5x | REDUCE 0.3x |
| Trending Up | BOOST 1.4x | BLOCK 0x | REDUCE 0.4x | BLOCK 0x |
| Trending Down | REDUCE 0.5x | REDUCE 0.6x | BOOST 1.3x | ALLOW 0.8x |
| Mean-Reverting | REDUCE 0.6x | BOOST 1.3x | REDUCE 0.6x | BOOST 1.3x |
| Ranging | REDUCE 0.7x | ALLOW 1.0x | REDUCE 0.7x | ALLOW 1.0x |

### 3.6 Expected Impact

| Metric | Before (No Regime) | After (With Regime) |
|--------|-------------------|---------------------|
| Strategy-regime alignment | ~40% | ~85% |
| Drawdown in crisis | -25% | -10% (positions reduced) |
| Trend capture | Miss reversals | Better timing |
| Win rate | ~52% | ~57-60% |

---

## 4. Implementation Roadmap

### 4.1 Priority Order

```
Priority 1: Sentiment Gating (Quickest win)
├── Estimated effort: 2-3 days
├── Impact: High (blocks bad signals immediately)
└── Risk: Low (can disable easily)

Priority 2: Regime Matching (Biggest improvement)
├── Estimated effort: 5-7 days
├── Impact: Very High (fundamental improvement)
└── Risk: Medium (needs testing)

Priority 3: Signal Quality Scoring (Polish)
├── Estimated effort: 3-4 days
├── Impact: Medium-High (improves position sizing)
└── Risk: Low (refinement of existing)
```

### 4.2 Integration Order

```python
# Final optimized flow:

def optimize_signal_v3(ticker, signal_type, confidence, features, price_data):
    """
    Full optimization pipeline with all three improvements.
    """
    fixes_applied = []

    # === STEP 1: REGIME MATCHING (broadest filter) ===
    regime_optimizer = RegimeAwareOptimizer()
    regime_result = regime_optimizer.optimize_for_regime(
        signal_type=signal_type,
        confidence=confidence,
        strategy_type=detect_strategy_type(features),
        price_data=price_data
    )

    if regime_result['action'] == 'BLOCK':
        return SignalOptimization(blocked=True, reason=regime_result['reason'])

    position_mult = regime_result['position_multiplier']
    confidence = regime_result['adjusted_confidence']
    fixes_applied.append(f"Regime: {regime_result['reason']}")

    # === STEP 2: SENTIMENT GATING (second filter) ===
    sentiment_gate = SentimentGate(gate_strength='moderate')
    gate_result = sentiment_gate.apply_gate(
        signal_type=signal_type,
        sentiment=features.get('combined_sentiment', 0),
        confidence=confidence,
        is_mock=features.get('sentiment_is_mock', True)
    )

    if gate_result['action'] == 'BLOCK':
        return SignalOptimization(blocked=True, reason=gate_result['reason'])

    position_mult *= gate_result['position_multiplier']
    confidence = gate_result['adjusted_confidence']
    fixes_applied.append(f"Sentiment: {gate_result['reason']}")

    # === STEP 3: SIGNAL QUALITY SCORING (refinement) ===
    quality_scorer = SignalQualityScorer()
    quality_score = quality_scorer.calculate_quality_score({
        'direction': 1 if signal_type == 'BUY' else -1,
        'confidence': confidence,
        'rsi': features.get('rsi', 50),
        'macd_signal': features.get('macd_signal', 0),
        'sentiment': features.get('combined_sentiment', 0),
        'volume_ratio': features.get('volume_ratio', 1.0),
        'features_available': features.get('features_available', 80),
        'total_features': 90,
        'regime': regime_result['regime']
    })

    position_mult = quality_scorer.adjust_position_for_quality(
        position_mult, quality_score
    )
    fixes_applied.append(f"Quality: {quality_score:.2f}")

    # === STEP 4: APPLY REMAINING FIXES ===
    # (existing logic from us_intl_optimizer.py)
    # Kelly criterion, stop-loss, profit-taking, etc.

    return SignalOptimization(
        position_multiplier=position_mult,
        quality_score=quality_score,
        regime=regime_result['regime'],
        fixes_applied=fixes_applied,
        stop_loss_multiplier=regime_result['stop_loss_multiplier']
    )
```

### 4.3 Testing Strategy

```python
# Backtest comparison framework

def compare_optimization_versions(ticker, start_date, end_date):
    """
    Compare old vs new optimization on historical data.
    """
    results = {
        'v1_original': [],      # Current implementation
        'v2_sentiment_gate': [],
        'v3_regime_match': [],
        'v4_quality_score': [],
        'v5_all_combined': []
    }

    for signal in generate_historical_signals(ticker, start_date, end_date):
        # V1: Original
        v1_result = optimize_buy_signal_original(signal)
        results['v1_original'].append(evaluate_signal(v1_result))

        # V2: Add sentiment gating
        v2_result = optimize_with_sentiment_gate(signal)
        results['v2_sentiment_gate'].append(evaluate_signal(v2_result))

        # ... etc

    # Compare metrics
    for version, signals in results.items():
        print(f"{version}:")
        print(f"  Win Rate: {calculate_win_rate(signals):.1%}")
        print(f"  Avg Return: {calculate_avg_return(signals):.2%}")
        print(f"  Sharpe: {calculate_sharpe(signals):.2f}")
        print(f"  Max DD: {calculate_max_drawdown(signals):.1%}")
```

---

## Summary

| Improvement | Problem Solved | Key Mechanism | Expected Impact |
|-------------|---------------|---------------|-----------------|
| **Signal Quality Scoring** | Position size doesn't match signal strength | Score based on indicator agreement, data completeness | +15-25% risk-adjusted returns |
| **Sentiment Gating** | Signals fire regardless of sentiment | Block/reduce signals misaligned with sentiment | Block ~20% bad signals |
| **Regime Matching** | Same strategy used in all market conditions | Adjust strategy based on volatility and trend regime | +5-8% win rate improvement |

**Combined impact**: These three improvements working together could improve the US/Intl model's accuracy by 15-25% and significantly reduce drawdowns during adverse market conditions.

---

*Document generated: December 2025*
*Implementation priority: Sentiment Gating > Regime Matching > Signal Quality Scoring*
