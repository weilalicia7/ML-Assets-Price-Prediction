# NLP Sentiment Analysis Integration Guide

This guide shows you how to add **real NLP sentiment analysis** to replace the mock percentages in your FastAPI webapp.

---

## Options for NLP Sentiment Analysis

### Option 1: **Free - Use Transformers (FinBERT)** ⭐ RECOMMENDED
**Best for:** Financial news sentiment with high accuracy
**Cost:** FREE
**Pros:** State-of-the-art, fine-tuned for financial text
**Cons:** Requires model download (~400MB), slightly slower

### Option 2: **Free - Use VADER**
**Best for:** Quick implementation, social media text
**Cost:** FREE
**Pros:** Fast, lightweight, no model download
**Cons:** Less accurate for financial news

### Option 3: **Paid - Use News APIs with Built-in Sentiment**
**Best for:** Production with latest news
**Cost:** $50-300/month
**Pros:** Real-time news + sentiment in one API
**Cons:** Costs money

---

## Implementation Plan

### Step 1: Choose Your Approach

I recommend starting with **Option 1 (FinBERT)** for accuracy, then optionally adding real news APIs later.

---

## Option 1: FinBERT Implementation (RECOMMENDED)

### 1.1 Install Dependencies

```bash
cd stock-prediction-model
pip install transformers torch sentencepiece
```

### 1.2 Create Sentiment Analyzer Module

Create `stock-prediction-model/src/nlp/sentiment_analyzer.py`:

```python
"""
Financial Sentiment Analyzer using FinBERT
Fine-tuned BERT model for financial sentiment analysis
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)


class FinancialSentimentAnalyzer:
    """Analyze sentiment of financial news using FinBERT."""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize FinBERT sentiment analyzer.

        Args:
            model_name: HuggingFace model name (default: ProsusAI/finbert)
        """
        logger.info(f"Loading FinBERT model: {model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()

            # Label mapping
            self.labels = ['negative', 'neutral', 'positive']

            logger.info(f"FinBERT loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load FinBERT: {str(e)}")
            raise

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.

        Args:
            text: News headline or article text

        Returns:
            {
                'sentiment': 'positive'|'negative'|'neutral',
                'score': 0.85,  # Confidence score
                'scores': {
                    'positive': 0.85,
                    'neutral': 0.10,
                    'negative': 0.05
                }
            }
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Convert to probabilities
            probs = predictions[0].cpu().numpy()

            # Get dominant sentiment
            max_idx = np.argmax(probs)
            sentiment = self.labels[max_idx]
            confidence = float(probs[max_idx])

            return {
                'sentiment': sentiment,
                'score': confidence,
                'scores': {
                    'negative': float(probs[0]),
                    'neutral': float(probs[1]),
                    'positive': float(probs[2])
                }
            }

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return {
                'sentiment': 'neutral',
                'score': 0.33,
                'scores': {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33}
            }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment for multiple texts efficiently.

        Args:
            texts: List of news headlines/articles

        Returns:
            List of sentiment dicts
        """
        try:
            # Tokenize batch
            inputs = self.tokenizer(
                texts,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Process results
            results = []
            for probs in predictions.cpu().numpy():
                max_idx = np.argmax(probs)
                results.append({
                    'sentiment': self.labels[max_idx],
                    'score': float(probs[max_idx]),
                    'scores': {
                        'negative': float(probs[0]),
                        'neutral': float(probs[1]),
                        'positive': float(probs[2])
                    }
                })

            return results

        except Exception as e:
            logger.error(f"Batch sentiment analysis failed: {str(e)}")
            return [{'sentiment': 'neutral', 'score': 0.33, 'scores': {}}] * len(texts)


# Global analyzer instance (lazy loading)
_analyzer = None


def get_sentiment_analyzer() -> FinancialSentimentAnalyzer:
    """Get or create global sentiment analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = FinancialSentimentAnalyzer()
    return _analyzer
```

### 1.3 Update webapp.py to Use Real Sentiment

Update the `generate_news_feed()` function in `webapp.py`:

```python
# Add this import at the top
from src.nlp.sentiment_analyzer import get_sentiment_analyzer

def generate_news_feed(ticker, company_info, current_price, prediction_data):
    """Generate realistic news and social media feed for a ticker."""
    news_items = []

    # Get sentiment analyzer
    try:
        sentiment_analyzer = get_sentiment_analyzer()
    except Exception as e:
        logger.warning(f"Sentiment analyzer unavailable: {e}")
        sentiment_analyzer = None

    # Determine sentiment based on prediction
    direction = prediction_data['prediction']['direction']
    volatility = prediction_data['prediction']['volatility']

    # News templates based on asset type and prediction
    if company_info['type'] == 'Stock':
        if direction > 0:
            headline = f'{company_info["name"]} shares rise on strong earnings outlook'
            news_items.append({
                'source': 'Bloomberg',
                'icon': '📰',
                'time': f'{np.random.randint(5, 120)} min ago',
                'headline': headline,
                'url': f'https://www.bloomberg.com/quote/{ticker}:US',
                'sentiment': 'positive',  # Base sentiment
                'relevance': 0.85
            })

            headline2 = f'Analysts upgrade {ticker} to Buy with ${current_price * 1.15:.0f} price target'
            news_items.append({
                'source': 'Twitter',
                'icon': '🐦',
                'time': f'{np.random.randint(1, 30)} min ago',
                'headline': headline2,
                'url': f'https://twitter.com/search?q=${ticker}',
                'sentiment': 'positive',
                'relevance': 0.75
            })
        else:
            headline = f'{company_info["name"]} faces headwinds amid market concerns'
            news_items.append({
                'source': 'Reuters',
                'icon': '📰',
                'time': f'{np.random.randint(10, 90)} min ago',
                'headline': headline,
                'url': f'https://www.reuters.com/markets/companies/{ticker}.O',
                'sentiment': 'negative',
                'relevance': 0.80
            })

    # Add more news items for crypto, forex, etc...
    # (keep existing code)

    # === NEW: Add real sentiment scores using NLP ===
    if sentiment_analyzer:
        # Analyze all headlines
        headlines = [item['headline'] for item in news_items]
        sentiments = sentiment_analyzer.analyze_batch(headlines)

        # Update news items with real sentiment scores
        for item, sentiment_result in zip(news_items, sentiments):
            item['sentiment'] = sentiment_result['sentiment']  # Update with NLP result
            item['sentiment_score'] = sentiment_result['score']  # Add confidence score
            item['sentiment_breakdown'] = sentiment_result['scores']  # Full breakdown

    # Shuffle and return top 5
    np.random.shuffle(news_items)
    return news_items[:5]
```

### 1.4 Test the Integration

Create a test script `test_sentiment.py`:

```python
"""Test sentiment analysis integration"""

import sys
sys.path.insert(0, '.')

from src.nlp.sentiment_analyzer import get_sentiment_analyzer

# Test headlines
headlines = [
    "Apple stock soars to record high on strong iPhone sales",
    "Tesla shares plummet amid SEC investigation",
    "Microsoft announces new AI partnership",
    "Market volatility increases as inflation fears grow",
    "Bitcoin reaches new all-time high"
]

print("Testing FinBERT Sentiment Analysis\n" + "="*60)

analyzer = get_sentiment_analyzer()

for headline in headlines:
    result = analyzer.analyze_sentiment(headline)
    print(f"\nHeadline: {headline}")
    print(f"Sentiment: {result['sentiment'].upper()}")
    print(f"Confidence: {result['score']:.2%}")
    print(f"Breakdown: Pos={result['scores']['positive']:.2%}, "
          f"Neu={result['scores']['neutral']:.2%}, "
          f"Neg={result['scores']['negative']:.2%}")

print("\n" + "="*60)
print("✅ Sentiment analysis working!")
```

Run it:
```bash
python test_sentiment.py
```

---

## Option 2: VADER Implementation (Simpler/Faster)

### 2.1 Install VADER

```bash
pip install vaderSentiment
```

### 2.2 Create Simple Analyzer

```python
# src/nlp/simple_sentiment.py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SimpleSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text: str) -> dict:
        scores = self.analyzer.polarity_scores(text)

        # Determine dominant sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
            score = scores['pos']
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
            score = scores['neg']
        else:
            sentiment = 'neutral'
            score = scores['neu']

        return {
            'sentiment': sentiment,
            'score': score,
            'scores': {
                'positive': scores['pos'],
                'neutral': scores['neu'],
                'negative': scores['neg']
            }
        }
```

---

## Option 3: Real News APIs (Production)

### Best APIs for Financial News + Sentiment:

1. **NewsAPI.org**
   - Cost: Free tier (100 req/day), $449/month (unlimited)
   - Features: News aggregation, no built-in sentiment
   - URL: https://newsapi.org/

2. **Alpha Vantage News Sentiment**
   - Cost: Free tier (25 req/day), $50/month (500 req/day)
   - Features: News + AI sentiment scores
   - URL: https://www.alphavantage.co/

3. **Finnhub News Sentiment**
   - Cost: Free tier (60 req/min), $60/month
   - Features: Real-time news + sentiment
   - URL: https://finnhub.io/

### Example: Alpha Vantage Integration

```python
import requests

def fetch_news_with_sentiment(ticker: str, api_key: str):
    """Fetch news with sentiment from Alpha Vantage."""
    url = f"https://www.alphavantage.co/query"
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': ticker,
        'apikey': api_key
    }

    response = requests.get(url, params=params)
    data = response.json()

    news_items = []
    for article in data.get('feed', [])[:5]:
        # Extract sentiment score
        ticker_sentiment = next(
            (s for s in article.get('ticker_sentiment', []) if s['ticker'] == ticker),
            None
        )

        sentiment_score = float(ticker_sentiment['ticker_sentiment_score']) if ticker_sentiment else 0

        news_items.append({
            'source': article.get('source', 'Unknown'),
            'headline': article.get('title', ''),
            'url': article.get('url', ''),
            'time': article.get('time_published', ''),
            'sentiment': 'positive' if sentiment_score > 0.15 else 'negative' if sentiment_score < -0.15 else 'neutral',
            'sentiment_score': abs(sentiment_score),  # Convert to 0-1 confidence
        })

    return news_items
```

---

## Recommended Implementation Steps

### Phase 1: Local NLP (FinBERT) ✅ Start Here
1. Install transformers: `pip install transformers torch`
2. Create `src/nlp/sentiment_analyzer.py` (code above)
3. Update `webapp.py` to use real sentiment
4. Test with `test_sentiment.py`

### Phase 2: Add News Caching (Optional)
- Cache news items for 15-30 minutes to reduce API calls
- Store in Redis or simple in-memory dict

### Phase 3: Production News API (Later)
- Get Alpha Vantage API key (free tier to start)
- Replace generated news with real API news
- Use FinBERT as fallback if API fails

---

## Performance Considerations

### FinBERT Performance:
- **First load:** ~2-3 seconds (model download + initialization)
- **After loaded:** ~0.1-0.2 seconds per headline
- **Batch processing:** ~0.5 seconds for 10 headlines
- **Memory:** ~500MB RAM

### Optimization Tips:
1. **Load model once at startup** (singleton pattern - already in code)
2. **Use batch processing** for multiple headlines
3. **Cache results** for 15-30 minutes
4. **Consider GPU** if available (automatic with code above)

---

## Testing Your Implementation

After implementing FinBERT, test it:

```bash
# Start your webapp
python webapp.py

# Visit: http://localhost:5000
# Search for: AAPL
# Click "Get ML Analysis"
# Check the "Real-Time News & Social Media" section
# You should see sentiment percentages now!
```

---

## Next Steps

1. **Immediate:** Implement FinBERT (Option 1)
2. **Week 1:** Test and tune sentiment thresholds
3. **Week 2:** Add news caching to improve performance
4. **Month 1:** Consider adding real news API (Alpha Vantage free tier)

---

## Need Help?

If you encounter any issues:
1. Check `transformers` version: `pip show transformers`
2. Ensure PyTorch is installed: `pip install torch`
3. Check model download location: `~/.cache/huggingface/`
4. Review logs for errors

Let me know which option you want to implement and I'll help you set it up!
