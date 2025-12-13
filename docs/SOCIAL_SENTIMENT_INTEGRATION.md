# Social Media & Sentiment Integration Guide

## 🎯 Why Social Sentiment Matters

### Modern Market Reality
Traditional analysis is NO LONGER ENOUGH. Social media drives:
- **GameStop (GME)** squeeze (2021): Reddit r/WallStreetBets drove 1,700% rally
- **Dogecoin** surge: Elon Musk tweets caused 10,000%+ gains
- **AMC Entertainment**: Retail sentiment pushed from $2 to $72
- **Tesla**: Twitter sentiment correlates with price movements
- **Robinhood** effect: Retail traders now move markets

### Key Insight
**You need to track what retail investors are saying/doing, not just fundamentals!**

---

## 📊 Data Sources for Social Sentiment

### 1. REDDIT (Crucial - Free!)

#### Key Subreddits to Monitor
```python
HIGH PRIORITY (Market Moving):
- r/wallstreetbets (14M+ members) - Options, meme stocks
- r/stocks (5M+ members) - General stock discussion
- r/investing (2M+ members) - Long-term investing
- r/CryptoCurrency (7M+ members) - Crypto discussion
- r/Superstonk (900K+ members) - GME community
- r/pennystocks (400K+ members) - Small cap stocks

SECONDARY:
- r/options - Options trading
- r/algotrading - Algo traders
- r/thetagang - Options strategies
- r/dividends - Dividend investing
```

#### What to Track
```python
For each stock/crypto:
1. Mention Count
   - Number of times ticker mentioned (hourly/daily)
   - Spike in mentions = increased interest

2. Sentiment Scores
   - Positive mentions vs negative
   - Bullish vs bearish comments
   - Award count (Reddit gold = strong sentiment)

3. Engagement Metrics
   - Upvotes on posts mentioning ticker
   - Comments count
   - Post velocity (posts per hour)

4. Key Phrases
   - "DD" (Due Diligence) posts
   - "YOLO" (risky plays)
   - "Diamond hands" vs "Paper hands"
   - "To the moon" 🚀 (bullish)
   - "Puts" vs "Calls" sentiment
```

#### Reddit API Access
```python
# FREE - Use PRAW (Python Reddit API Wrapper)
import praw

reddit = praw.Reddit(
    client_id='YOUR_CLIENT_ID',
    client_secret='YOUR_SECRET',
    user_agent='stock_predictor'
)

# Get posts from r/wallstreetbets
subreddit = reddit.subreddit('wallstreetbets')
hot_posts = subreddit.hot(limit=100)

for post in hot_posts:
    # Extract tickers mentioned
    # Analyze sentiment
    # Track engagement
```

**Get API Key**: https://www.reddit.com/prefs/apps (FREE)

---

### 2. TWITTER/X (Real-Time Market Moving)

#### Key Accounts to Monitor

**Market Influencers (Tweet = Price Movement)**
```python
TIER 1 (Huge Impact):
- @elonmusk - Tesla, crypto markets
- @michael_saylor - Bitcoin advocate
- @chamath - SPACs, tech stocks
- @CathieDWood / @ARKInvest - Disruptive tech

TIER 2 (Significant Impact):
- @jimcramer - CNBC, retail sentiment
- @DeItaone - Real-time news
- @zerohedge - Market news/sentiment
- @realDonaldTrump - Policy/market impact
- @federalreserve - Fed communications

TIER 3 (Analyst/Trader Sentiment):
- @markets - Bloomberg
- @CNBC
- @WSJ
- @FT
- Top FinTwit traders
```

**What to Track**
```python
1. Ticker Mentions
   - Volume of mentions for each ticker
   - Spike detection (mentions > 3x average)

2. Sentiment Analysis
   - Positive/negative tweet ratio
   - Emoji analysis (🚀📈 = bullish, 📉💩 = bearish)
   - Retweet/like velocity

3. Influencer Activity
   - When Elon tweets about crypto → immediate price impact
   - Cathie Wood buys → ARK followers buy
   - Cramer mention → inverse Cramer effect

4. Breaking News Detection
   - Real-time event detection
   - Trending hashtags
   - News velocity
```

**API Access**
```python
# Twitter API v2 (Free tier: 500K tweets/month)
# Get at: https://developer.twitter.com/

Options:
1. Official Twitter API v2 (Limited free tier)
2. Tweepy library (Python wrapper)
3. snscrape (No API key needed, scraping)

# Example with Tweepy
import tweepy

api = tweepy.API(auth)
tweets = api.search_tweets(q="$AAPL", count=100)

for tweet in tweets:
    # Sentiment analysis
    # Influencer check
    # Volume tracking
```

---

### 3. STOCKTWITS (Stock-Specific Sentiment - Free!)

#### What is StockTwits?
- Twitter for traders
- Stock-specific message boards
- **Built-in sentiment** (bullish/bearish labels)
- Real-time trader sentiment

**Key Features**
```python
Advantages:
✅ Users self-label sentiment (bullish/bearish)
✅ Stock-specific ($AAPL, $TSLA tickers)
✅ Trader-focused (quality over noise)
✅ Free API access
✅ Real-time stream

Data Available:
- Message volume per ticker
- Bullish % vs Bearish %
- Trending tickers
- Influencer activity
```

**API Access**
```python
# StockTwits API (FREE)
# https://api.stocktwits.com/developers/docs/api

import requests

# Get messages for ticker
url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
response = requests.get(url)
data = response.json()

# Get sentiment
for message in data['messages']:
    sentiment = message.get('entities', {}).get('sentiment', {}).get('basic')
    # 'Bullish' or 'Bearish'
```

---

### 4. ROBINHOOD DATA (Retail Investor Activity)

#### Robinhood Popularity Tracking
```python
What to Track:
1. Number of users holding each stock
   - Increasing holders = retail buying
   - Decreasing holders = retail selling

2. Top 100 most popular stocks
   - Retail sentiment indicator
   - Meme stock detection

3. Crypto holdings on Robinhood
   - Retail crypto interest

Sources:
- Robintrack.net (historical data)
- RobinhoodAPI (unofficial)
- Alternative: Track app store rankings, mentions
```

**Why This Matters**
- Robinhood users drove GME, AMC squeezes
- When retail piles in → volatility increases
- Retail sentiment = contrarian indicator sometimes

---

### 5. GOOGLE TRENDS (Search Interest)

#### Search Volume = Interest Level
```python
What to Track:
- Search volume for ticker symbols
- Trending financial topics
- Regional interest (where is interest growing?)
- Related queries (what else are they searching?)

Example Insights:
- "Bitcoin" searches spike before price moves
- "How to buy Tesla stock" → retail interest
- Regional trends predict which markets heat up
```

**API Access**
```python
# PyTrends (FREE)
from pytrends.request import TrendReq

pytrends = TrendReq()

# Get interest over time
pytrends.build_payload(['Bitcoin', 'Ethereum'])
interest = pytrends.interest_over_time()

# Trending searches
trending = pytrends.trending_searches(pn='united_states')
```

---

### 6. DISCORD & TELEGRAM (Crypto Communities)

#### Crypto-Focused Communities
```python
Key Channels:
- Discord crypto servers (token-specific)
- Telegram pump groups
- Whale alert channels

What to Track:
- Member count growth
- Message velocity
- Whale movements
- Pump signals

Warning Signs:
- Coordinated pump mentions
- Shill campaigns
- Bot activity
```

---

### 7. NEWS SENTIMENT APIs

#### Financial News Aggregators

**Option 1: Finnhub (FREE tier)**
```python
# Company news sentiment
# https://finnhub.io/

import finnhub

finnhub_client = finnhub.Client(api_key="YOUR_KEY")

# Get news sentiment
news = finnhub_client.news_sentiment(ticker)
# Returns: buzz, sentiment scores
```

**Option 2: NewsAPI (FREE tier)**
```python
# General news about companies
# https://newsapi.org/

from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key='YOUR_KEY')

# Get articles
articles = newsapi.get_everything(
    q='Tesla',
    language='en',
    sort_by='publishedAt'
)

# Analyze sentiment
```

**Option 3: Benzinga (Paid but comprehensive)**
- Real-time news
- Analyst ratings
- Social sentiment

---

## 🧠 Sentiment Analysis Techniques

### 1. Rule-Based Sentiment (Simple, Fast)

```python
# Using VADER (Valence Aware Dictionary)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

text = "TSLA to the moon! 🚀 Great earnings!"
sentiment = analyzer.polarity_scores(text)

# Output: {'neg': 0.0, 'neu': 0.5, 'pos': 0.5, 'compound': 0.8}
# compound > 0.05 = positive
# compound < -0.05 = negative
```

**Pros**: Fast, works out of box
**Cons**: Misses sarcasm, context

---

### 2. ML-Based Sentiment (More Accurate)

```python
# Using FinBERT (Finance-specific BERT)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

text = "Apple stock crashes on poor earnings"
inputs = tokenizer(text, return_tensors="pt", padding=True)
outputs = model(**inputs)

# Get sentiment (positive/negative/neutral)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
# [neutral_prob, positive_prob, negative_prob]
```

**Pros**: More accurate, understands context
**Cons**: Slower, needs GPU for scale

---

### 3. Financial Lexicon-Based

```python
# Loughran-McDonald Financial Sentiment Dictionary
# Specifically designed for financial text

from pysentiment2 import LM

lm = LM()
tokens = lm.tokenize(text)
score = lm.get_score(tokens)

# Output:
# {'Positive': 5, 'Negative': 2, 'Polarity': 0.6, ...}
```

**Pros**: Finance-specific, better for earnings calls, reports
**Cons**: Slower than VADER

---

## 🎯 Key Social Sentiment Features to Engineer

### Feature Set 1: Volume-Based (15 features)
```python
1. Reddit mention count (1h, 6h, 24h windows)
2. Twitter mention volume (1h, 6h, 24h)
3. StockTwits message volume
4. Google search volume (daily)
5. Mention velocity (rate of change)
6. Mention spike detection (vol > 3x avg)
7. Robinhood holders count
8. Robinhood holders change (daily)
9. Discord member growth
10. News article count (24h)
11. Reddit upvote count
12. Twitter retweet count
13. StockTwits trending rank
14. Influencer mention (binary)
15. Viral post detection (>10K upvotes/retweets)
```

### Feature Set 2: Sentiment-Based (15 features)
```python
16. Reddit sentiment score (avg of all mentions)
17. Twitter sentiment score
18. StockTwits bullish % (built-in)
19. News sentiment score
20. Sentiment momentum (trend)
21. Sentiment divergence (social vs price)
22. Bull/bear ratio
23. Emoji sentiment (🚀📈 vs 📉💩)
24. Options sentiment (calls vs puts mentions)
25. "DD" post quality score
26. Sarcasm-adjusted sentiment
27. Fear & Greed indicator (from sentiment)
28. Hype score (mentions + sentiment + velocity)
29. Contrarian indicator (when sentiment extreme)
30. Influencer sentiment (weighted by followers)
```

### Feature Set 3: Event-Based (10 features)
```python
31. Breaking news detected (binary)
32. Major influencer tweet (Elon, Cathie, etc.)
33. Viral meme detected
34. Coordinated pump detected
35. Retail squeeze potential (Robinhood + WSB activity)
36. Short squeeze mentions
37. Earnings hype (pre-earnings mentions)
38. FOMO indicator (rapid mention growth)
39. Panic selling mentions
40. "Buy the dip" mentions
```

---

## 🚨 Special Event Detection

### 1. WallStreetBets Squeeze Detection
```python
# GameStop-style squeeze predictor

Signals:
✅ WSB mentions spike (>10x normal)
✅ High short interest (>30%)
✅ Robinhood popularity surge
✅ "Short squeeze" mentions increase
✅ Options volume spike (calls)
✅ Sentiment extremely bullish (>0.8)

Action:
- Predict HIGH volatility
- Widen prediction ranges
- Reduce confidence in normal models
- Use social-sentiment-heavy model
```

### 2. Elon Musk Tweet Impact
```python
# Crypto/Tesla prediction after Elon tweets

Detection:
- Monitor @elonmusk Twitter feed
- Extract ticker mentions (Bitcoin, Doge, TSLA)
- Immediate sentiment analysis

Impact Timeline:
- 0-15 min: Immediate spike (5-20%)
- 1-24 hours: Sustained movement or reversal
- 2-7 days: Return to fundamentals

Features:
- Minutes since Elon tweet
- Tweet sentiment
- Retweet velocity
- Historical Elon-tweet impact
```

### 3. Wealth Manager / Analyst Upgrade
```python
# Track influential analysts

Sources:
- Cathie Wood (ARK Invest) buys/sells
- Goldman Sachs upgrades/downgrades
- Jim Cramer recommendations (inverse often!)
- Warren Buffett portfolio changes

Features:
- Analyst rating change
- Price target change
- Institutional buying/selling
- Media coverage of the change

Impact:
- Usually 2-5% price movement
- Sustained if fundamentals support
```

---

## 💡 Integration Strategy

### Tier 1: Must Have (Free & High Impact)
```python
Priority 1:
✅ Reddit (r/wallstreetbets) - mention count, sentiment
✅ StockTwits - bullish/bearish %
✅ Google Trends - search volume
✅ Twitter - ticker mentions (free tier)

Implementation: 1-2 days
Impact: High (captures retail sentiment)
```

### Tier 2: Should Have (Enhanced Accuracy)
```python
Priority 2:
✅ Robinhood popularity tracking
✅ News sentiment (Finnhub free tier)
✅ Influencer monitoring (Elon, Cathie, etc.)
✅ FinBERT sentiment analysis

Implementation: 2-3 days
Impact: Medium-High
```

### Tier 3: Nice to Have (Advanced)
```python
Priority 3:
✅ Discord/Telegram crypto sentiment
✅ Options flow data
✅ Dark pool data
✅ Insider trading tracking
✅ Hedge fund positioning

Implementation: 3-5 days
Impact: Medium
```

---

## 🔧 Practical Implementation

### Step 1: Data Collection Pipeline
```python
# Daily social sentiment collection

import praw  # Reddit
import tweepy  # Twitter
import requests  # StockTwits, news
from pytrends.request import TrendReq  # Google Trends

class SocialSentimentCollector:
    def __init__(self):
        self.reddit = praw.Reddit(...)
        self.twitter = tweepy.API(...)
        self.stocktwits_base = "https://api.stocktwits.com/api/2"

    def collect_reddit_sentiment(self, ticker):
        """Collect from WSB, r/stocks"""
        mentions = 0
        total_sentiment = 0

        subreddit = self.reddit.subreddit('wallstreetbets')
        for post in subreddit.hot(limit=100):
            if ticker in post.title or ticker in post.selftext:
                mentions += 1
                sentiment = self.analyze_sentiment(post.title + post.selftext)
                total_sentiment += sentiment

        avg_sentiment = total_sentiment / mentions if mentions > 0 else 0
        return {
            'reddit_mentions': mentions,
            'reddit_sentiment': avg_sentiment,
            'reddit_upvotes': sum([p.score for p in posts])
        }

    def collect_twitter_sentiment(self, ticker):
        """Collect Twitter mentions"""
        tweets = self.twitter.search_tweets(q=f"${ticker}", count=100)

        mentions = len(tweets)
        sentiments = [self.analyze_sentiment(t.text) for t in tweets]

        return {
            'twitter_mentions': mentions,
            'twitter_sentiment': np.mean(sentiments),
            'twitter_engagement': sum([t.retweet_count + t.favorite_count for t in tweets])
        }

    def collect_stocktwits_sentiment(self, ticker):
        """Collect StockTwits data"""
        url = f"{self.stocktwits_base}/streams/symbol/{ticker}.json"
        data = requests.get(url).json()

        messages = data.get('messages', [])
        bullish = sum([1 for m in messages if m.get('entities', {}).get('sentiment', {}).get('basic') == 'Bullish'])
        bearish = sum([1 for m in messages if m.get('entities', {}).get('sentiment', {}).get('basic') == 'Bearish'])

        total = bullish + bearish
        bullish_pct = bullish / total if total > 0 else 0.5

        return {
            'stocktwits_messages': len(messages),
            'stocktwits_bullish_pct': bullish_pct
        }

    def collect_all(self, ticker):
        """Collect from all sources"""
        return {
            **self.collect_reddit_sentiment(ticker),
            **self.collect_twitter_sentiment(ticker),
            **self.collect_stocktwits_sentiment(ticker)
        }
```

### Step 2: Sentiment Analysis Function
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    """
    Analyze text sentiment.
    Returns: float between -1 (very negative) and 1 (very positive)
    """
    analyzer = SentimentIntensityAnalyzer()

    # Add financial slang
    financial_slang = {
        '🚀': 4.0,  # Very bullish
        '📈': 3.0,  # Bullish
        '📉': -3.0,  # Bearish
        '💎🙌': 4.0,  # Diamond hands - bullish
        'moon': 3.0,
        'bull': 2.0,
        'bear': -2.0,
        'crash': -3.0,
        'pump': 2.0,
        'dump': -2.0,
        'YOLO': 2.0,
        'DD': 1.0,
        'bagholding': -2.0
    }

    analyzer.lexicon.update(financial_slang)

    scores = analyzer.polarity_scores(text)
    return scores['compound']  # -1 to 1
```

### Step 3: Feature Engineering
```python
def engineer_social_features(ticker, lookback_days=7):
    """
    Create social sentiment features for a ticker
    """
    collector = SocialSentimentCollector()

    # Collect historical data
    social_data = []
    for day in range(lookback_days):
        date = datetime.now() - timedelta(days=day)
        daily_data = collector.collect_all(ticker)
        daily_data['date'] = date
        social_data.append(daily_data)

    df = pd.DataFrame(social_data)

    # Engineer features
    features = {}

    # Volume features
    features['reddit_mentions_1d'] = df['reddit_mentions'].iloc[0]
    features['reddit_mentions_7d_avg'] = df['reddit_mentions'].mean()
    features['reddit_mentions_spike'] = features['reddit_mentions_1d'] / features['reddit_mentions_7d_avg']

    # Sentiment features
    features['reddit_sentiment'] = df['reddit_sentiment'].iloc[0]
    features['twitter_sentiment'] = df['twitter_sentiment'].iloc[0]
    features['stocktwits_bullish_pct'] = df['stocktwits_bullish_pct'].iloc[0]

    # Momentum features
    features['sentiment_momentum'] = df['reddit_sentiment'].iloc[0] - df['reddit_sentiment'].iloc[-1]
    features['mention_momentum'] = df['reddit_mentions'].iloc[0] - df['reddit_mentions'].iloc[-1]

    # Hype score (combined metric)
    features['hype_score'] = (
        features['reddit_mentions_spike'] * 0.3 +
        features['reddit_sentiment'] * 0.3 +
        features['stocktwits_bullish_pct'] * 0.4
    )

    return features
```

---

## 📊 Model Integration

### Add Social Features to Your Model
```python
# Your existing features + social features

all_features = {
    # Technical features (existing)
    **technical_features,

    # Volatility features (existing)
    **volatility_features,

    # Macro features (existing)
    **macro_features,

    # NEW: Social sentiment features
    **social_sentiment_features
}

# Train model
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

# Feature importance
importance = model.feature_importances_
# Check if social features are important!
```

### Expected Impact
```python
Normal Stocks:
- Social features: 10-15% of model importance
- Helps predict retail-driven volatility

Meme Stocks (GME, AMC, DOGE):
- Social features: 40-60% of model importance!
- CRITICAL for prediction

Crypto:
- Social features: 30-50% of model importance
- Twitter/Reddit drive crypto markets
```

---

## ⚠️ Important Considerations

### 1. API Rate Limits
```python
Reddit (PRAW): 60 requests/minute
Twitter Free: 500K tweets/month
StockTwits: No official limit (be reasonable)
Google Trends: Use sparingly, no official API
```

**Solution**: Cache data, batch requests, use delays

### 2. Data Quality
```python
Challenges:
- Bots and spam
- Sarcasm detection
- Coordinated manipulation
- Pump and dump schemes

Solutions:
- Filter by account age/karma
- Use ML sentiment (FinBERT)
- Detect coordinated activity
- Cross-reference multiple sources
```

### 3. Timing
```python
Social sentiment is LEADING indicator:
- Mentions spike → Price follows (hours to days)
- Use 1-6 hour lag features
- Don't use same-day sentiment for same-day price

Example:
- Reddit mentions spike at 10 AM
- Price movement typically 2-24 hours later
```

---

## 🎯 Quick Start (1 Week Implementation)

### Day 1-2: Reddit Integration
```python
- [ ] Get Reddit API credentials
- [ ] Implement Reddit scraper (r/wallstreetbets)
- [ ] Basic mention counting
- [ ] Sentiment analysis with VADER
```

### Day 3-4: Twitter & StockTwits
```python
- [ ] Twitter API setup
- [ ] StockTwits API integration
- [ ] Combine data sources
- [ ] Feature engineering
```

### Day 5: Google Trends
```python
- [ ] PyTrends integration
- [ ] Search volume tracking
- [ ] Combine with other social data
```

### Day 6-7: Model Integration & Testing
```python
- [ ] Add social features to model
- [ ] Test on meme stocks (GME, AMC)
- [ ] Test on crypto (DOGE, SHIB)
- [ ] Measure improvement
```

---

## 📈 Expected Improvements

### For Regular Stocks
- MAPE improvement: 5-10%
- Better volatility prediction: 10-15%
- Retail activity detection: 80%+

### For Meme Stocks / Crypto
- MAPE improvement: 20-40%!
- Essential for prediction
- Detects pumps before they happen

---

## 🚀 Summary

**Must Track:**
1. ✅ Reddit (r/wallstreetbets) - Free, high impact
2. ✅ StockTwits - Free, ticker-specific sentiment
3. ✅ Twitter - Influencers, breaking news
4. ✅ Google Trends - Search interest

**Key Features:**
- Mention volume (spikes = volatility incoming)
- Sentiment scores (bullish vs bearish)
- Influencer activity (Elon, Cathie, etc.)
- Retail activity (Robinhood popularity)
- Hype detection (pump warnings)

**Impact:**
- Regular stocks: 10-15% model improvement
- Meme stocks: 30-50% improvement
- Crypto: 40-60% improvement
- Early warning for retail squeezes

**Time to Implement**: 1-2 weeks for basic system

See implementation code in next update!
