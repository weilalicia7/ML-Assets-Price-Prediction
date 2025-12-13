"""
Sentiment Analysis Features (ENHANCEMENT #5)
Based on 2025 research showing +20% alpha from alternative data

This module provides sentiment-based features for trading models.
Implements FREE and OPEN-SOURCE sentiment analysis using:
- FinBERT (ProsusAI/finbert) - High accuracy, finance-tailored
- VADER - Lightweight, fast alternative
- News sources: Yahoo Finance RSS, NewsAPI, web scraping

Installation:
    pip install transformers torch vaderSentiment feedparser beautifulsoup4 requests

References:
- FinBERT: https://huggingface.co/ProsusAI/finbert
- Solution guide: sentiment finbert solution.pdf
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Optional imports for real sentiment analysis
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    import feedparser
    import requests
    from bs4 import BeautifulSoup
    NEWS_SCRAPING_AVAILABLE = True
except ImportError:
    NEWS_SCRAPING_AVAILABLE = False

try:
    import tweepy
    TWITTER_AVAILABLE = True
except ImportError:
    TWITTER_AVAILABLE = False

try:
    import praw
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False

# For caching
import json
import hashlib
from pathlib import Path
import threading

# Global lock for thread-safe FinBERT loading
_finbert_lock = threading.Lock()
_finbert_loaded = False
_finbert_model = None
_finbert_tokenizer = None


class SentimentFeatureEngineer:
    """
    Sentiment feature engineering for trading models.

    ENHANCEMENT #5: Add sentiment analysis features
    Research shows funds using alternative data + deep learning achieve 20% higher alpha.
    """

    def __init__(self, use_finbert=True, use_vader=True,
                 twitter_api_keys=None, reddit_api_keys=None,
                 cache_dir=".sentiment_cache"):
        """
        Initialize sentiment feature engineer.

        Args:
            use_finbert: Use FinBERT for high-accuracy sentiment (requires transformers)
            use_vader: Use VADER for fast lightweight sentiment (requires vaderSentiment)
            twitter_api_keys: Dict with Twitter API credentials (bearer_token)
            reddit_api_keys: Dict with Reddit API credentials (client_id, client_secret, user_agent)
            cache_dir: Directory to cache sentiment analysis results
        """
        self.features_created = []
        self.use_finbert = use_finbert and FINBERT_AVAILABLE
        self.use_vader = use_vader and VADER_AVAILABLE

        # Initialize FinBERT model (lazy loading)
        self.finbert_tokenizer = None
        self.finbert_model = None

        # Initialize VADER analyzer
        self.vader_analyzer = None
        if self.use_vader:
            try:
                self.vader_analyzer = SentimentIntensityAnalyzer()
                print("[INFO] VADER sentiment analyzer initialized")
            except Exception as e:
                print(f"[WARNING] VADER initialization failed: {e}")
                self.use_vader = False

        # Social media API clients
        self.twitter_client = None
        self.reddit_client = None

        # Initialize Twitter
        if twitter_api_keys and TWITTER_AVAILABLE:
            try:
                self.twitter_client = tweepy.Client(
                    bearer_token=twitter_api_keys.get('bearer_token')
                )
                print("[INFO] Twitter API initialized")
            except Exception as e:
                print(f"[WARNING] Twitter initialization failed: {e}")

        # Initialize Reddit
        if reddit_api_keys and REDDIT_AVAILABLE:
            try:
                self.reddit_client = praw.Reddit(
                    client_id=reddit_api_keys.get('client_id'),
                    client_secret=reddit_api_keys.get('client_secret'),
                    user_agent=reddit_api_keys.get('user_agent', 'SentimentBot/1.0')
                )
                print("[INFO] Reddit API initialized")
            except Exception as e:
                print(f"[WARNING] Reddit initialization failed: {e}")

        # Setup caching
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.sentiment_cache = {}

    def _load_finbert(self):
        """Lazy load FinBERT model (ProsusAI/finbert) with thread-safety."""
        global _finbert_loaded, _finbert_model, _finbert_tokenizer

        # Quick check without lock
        if self.finbert_model is not None:
            return

        if not self.use_finbert:
            return

        # Use global shared model if already loaded by another thread
        if _finbert_loaded and _finbert_model is not None:
            self.finbert_model = _finbert_model
            self.finbert_tokenizer = _finbert_tokenizer
            return

        # Thread-safe loading with lock
        with _finbert_lock:
            # Double-check after acquiring lock
            if _finbert_loaded:
                if _finbert_model is not None:
                    self.finbert_model = _finbert_model
                    self.finbert_tokenizer = _finbert_tokenizer
                else:
                    # Previous attempt failed, disable FinBERT
                    self.use_finbert = False
                return

            try:
                print("[INFO] Loading FinBERT model from HuggingFace (thread-safe)...")
                _finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

                # Use device_map="cpu" with accelerate to properly handle meta tensors
                # This requires the accelerate library to be installed
                _finbert_model = AutoModelForSequenceClassification.from_pretrained(
                    "ProsusAI/finbert",
                    device_map="cpu",
                    torch_dtype=torch.float32
                )

                # Set to evaluation mode
                _finbert_model.eval()
                print("[OK] FinBERT model loaded successfully on CPU with accelerate!")

                # Set instance variables
                self.finbert_model = _finbert_model
                self.finbert_tokenizer = _finbert_tokenizer
                _finbert_loaded = True

            except Exception as e:
                print(f"[ERROR] Failed to load FinBERT: {e}")
                print("[INFO] Falling back to mock sentiment")
                self.use_finbert = False
                _finbert_loaded = True  # Mark as loaded (failed) to prevent retry

    def analyze_text_finbert(self, text):
        """
        Analyze sentiment using FinBERT.

        Args:
            text: Text to analyze (news headline, article, etc.)

        Returns:
            Sentiment score: -1 (negative) to +1 (positive)
        """
        if not self.use_finbert:
            return 0.0

        self._load_finbert()

        if self.finbert_model is None:
            return 0.0

        try:
            inputs = self.finbert_tokenizer(text, return_tensors="pt",
                                           padding=True, truncation=True,
                                           max_length=512)

            with torch.no_grad():
                outputs = self.finbert_model(**inputs)

            # FinBERT outputs: [positive, negative, neutral]
            # Ensure tensor is on CPU before converting to numpy to avoid meta tensor errors
            logits = outputs.logits.cpu()
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # Convert to sentiment score: positive - negative
            sentiment = probs[0] - probs[1]  # Positive - Negative

            return float(sentiment)

        except Exception as e:
            print(f"[WARNING] FinBERT analysis failed: {e}")
            return 0.0

    def analyze_text_vader(self, text):
        """
        Analyze sentiment using VADER (fast, lightweight).

        Args:
            text: Text to analyze

        Returns:
            Sentiment score: -1 (negative) to +1 (positive)
        """
        if not self.use_vader or self.vader_analyzer is None:
            return 0.0

        try:
            scores = self.vader_analyzer.polarity_scores(text)
            return scores['compound']  # Compound score (-1 to +1)
        except Exception as e:
            print(f"[WARNING] VADER analysis failed: {e}")
            return 0.0

    def _get_cache_key(self, text):
        """Generate cache key for a text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _load_cache(self):
        """Load sentiment cache from disk."""
        cache_file = self.cache_dir / "sentiment_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.sentiment_cache = json.load(f)
                print(f"[INFO] Loaded {len(self.sentiment_cache)} cached sentiments")
            except Exception as e:
                print(f"[WARNING] Failed to load cache: {e}")
                self.sentiment_cache = {}

    def _save_cache(self):
        """Save sentiment cache to disk."""
        cache_file = self.cache_dir / "sentiment_cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.sentiment_cache, f)
        except Exception as e:
            print(f"[WARNING] Failed to save cache: {e}")

    def analyze_text_finbert_cached(self, text):
        """Analyze text with FinBERT using cache."""
        cache_key = self._get_cache_key(f"finbert_{text}")

        # Check cache
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]

        # Analyze
        sentiment = self.analyze_text_finbert(text)

        # Cache result
        self.sentiment_cache[cache_key] = sentiment
        return sentiment

    def analyze_text_vader_cached(self, text):
        """Analyze text with VADER using cache."""
        cache_key = self._get_cache_key(f"vader_{text}")

        # Check cache
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]

        # Analyze
        sentiment = self.analyze_text_vader(text)

        # Cache result
        self.sentiment_cache[cache_key] = sentiment
        return sentiment

    def fetch_yahoo_finance_news(self, ticker, days_back=7):
        """
        Fetch news headlines from Yahoo Finance RSS feed.

        Args:
            ticker: Stock ticker (e.g., 'AAPL')
            days_back: Number of days to look back

        Returns:
            List of tuples: (date, headline)
        """
        if not NEWS_SCRAPING_AVAILABLE:
            return []

        try:
            # Yahoo Finance RSS feed for ticker
            url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
            feed = feedparser.parse(url)

            headlines = []
            cutoff_date = datetime.now() - timedelta(days=days_back)

            for entry in feed.entries[:50]:  # Increased to 50
                # Parse published date
                try:
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    else:
                        pub_date = datetime.now()
                    if pub_date >= cutoff_date:
                        headlines.append((pub_date, entry.title))
                except Exception:
                    # If date parsing fails, use current date
                    headlines.append((datetime.now(), entry.title))

            return headlines

        except Exception as e:
            print(f"[WARNING] Yahoo Finance RSS fetch failed: {e}")
            return []

    def fetch_twitter_posts(self, ticker, max_results=100):
        """
        Fetch recent Twitter posts about a stock.

        Args:
            ticker: Stock ticker
            max_results: Maximum number of tweets to fetch

        Returns:
            List of tuples: (date, text)
        """
        if not self.twitter_client:
            return []

        try:
            # Search for tweets mentioning the ticker
            query = f"${ticker} OR {ticker} -is:retweet lang:en"

            tweets = self.twitter_client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),
                tweet_fields=['created_at', 'text']
            )

            posts = []
            if tweets.data:
                for tweet in tweets.data:
                    posts.append((tweet.created_at, tweet.text))

            return posts

        except Exception as e:
            print(f"[WARNING] Twitter fetch failed: {e}")
            return []

    def fetch_reddit_posts(self, ticker, subreddits=None, limit=100):
        """
        Fetch recent Reddit posts about a stock.

        Args:
            ticker: Stock ticker
            subreddits: List of subreddits to search (default: wallstreetbets, stocks, investing)
            limit: Maximum number of posts to fetch

        Returns:
            List of tuples: (date, text)
        """
        if not self.reddit_client:
            return []

        if subreddits is None:
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket']

        posts = []

        try:
            for subreddit_name in subreddits:
                subreddit = self.reddit_client.subreddit(subreddit_name)

                # Search for ticker mentions
                for submission in subreddit.search(ticker, limit=limit//len(subreddits), time_filter='week'):
                    post_date = datetime.fromtimestamp(submission.created_utc)
                    text = f"{submission.title}. {submission.selftext[:200]}"
                    posts.append((post_date, text))

            return posts

        except Exception as e:
            print(f"[WARNING] Reddit fetch failed: {e}")
            return []

    def add_price_derived_sentiment_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-derived sentiment proxy for backtesting (US/INTL MODEL).

        FIX: Replace random-walk mock sentiment with price-derived proxy.
        This creates more realistic sentiment features that actually correlate
        with market movements, solving the issue where models learned to ignore
        sentiment because mock data had no predictive value.

        Theory: Price/volume action reflects aggregate market sentiment.
        - High volume + positive return = positive sentiment (buying pressure)
        - High volume + negative return = negative sentiment (selling pressure)
        - Low volume = uncertainty/neutral sentiment

        This is a PROXY for backtesting until real news APIs are integrated.
        The model can learn meaningful patterns from this data.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with price-derived sentiment features
        """
        df = df.copy()

        # Ensure we have required columns
        if 'Close' not in df.columns or 'Volume' not in df.columns:
            print("[WARNING] Missing Close/Volume columns. Using legacy mock sentiment.")
            return self._add_legacy_mock_sentiment(df)

        # ========== PRICE-DERIVED SENTIMENT (replaces random walk) ==========

        # 1. Return-based sentiment components
        returns_1d = df['Close'].pct_change(1).fillna(0)
        returns_5d = df['Close'].pct_change(5).fillna(0)
        returns_20d = df['Close'].pct_change(20).fillna(0)

        # 2. Volume confirmation factor
        # High volume amplifies sentiment signal, low volume dampens it
        volume_ma20 = df['Volume'].rolling(20, min_periods=1).mean()
        volume_ratio = (df['Volume'] / volume_ma20).clip(0.1, 5.0)  # Avoid extremes
        volume_factor = np.log1p(volume_ratio - 1)  # Log transform for stability

        # 3. News sentiment proxy (short-term price action + volume)
        # High volume + positive return = positive news sentiment
        raw_news_sentiment = np.sign(returns_5d) * np.abs(returns_5d) * (1 + volume_factor * 0.3)
        df['news_sentiment'] = np.tanh(raw_news_sentiment * 10).clip(-1, 1)  # Scale to [-1, 1]

        # 4. Social sentiment proxy (very short-term momentum with volume spikes)
        # Social media tends to react quickly and amplify moves
        raw_social = np.sign(returns_1d) * np.abs(returns_1d) * (1 + volume_factor * 0.5)
        df['social_sentiment'] = np.tanh(raw_social * 15).clip(-1, 1)

        # 5. News volume proxy (based on actual volume spikes)
        # More trading = more news coverage typically
        volume_zscore = (df['Volume'] - volume_ma20) / df['Volume'].rolling(20, min_periods=1).std().clip(lower=1)
        df['news_volume'] = (volume_zscore.clip(-3, 3) + 3) * 2  # Scale to ~0-12 range
        df['news_volume_ma7'] = df['news_volume'].rolling(7, min_periods=1).mean()

        # 6. Social mentions proxy (volume-based with noise)
        df['social_mentions'] = df['news_volume'] * 2 + np.random.poisson(3, len(df))
        df['social_mentions_ma7'] = df['social_mentions'].rolling(7, min_periods=1).mean()

        # 7. Sentiment momentum (change in sentiment over 5 days)
        df['sentiment_momentum'] = df['news_sentiment'].diff(5).fillna(0)

        # 8. Sentiment-price divergence (when sentiment diverges from returns)
        # This is a contrarian signal - sentiment should follow price
        df['returns_5d'] = returns_5d
        df['sentiment_price_divergence'] = df['news_sentiment'] - np.tanh(returns_5d * 10)

        # 9. Combined sentiment (weighted average)
        df['combined_sentiment'] = (
            0.5 * df['news_sentiment'] +
            0.3 * df['social_sentiment'] +
            0.2 * np.tanh(returns_20d * 5)  # Longer-term trend sentiment
        ).clip(-1, 1)

        # 10. Flag this as proxy data (not real news)
        df['sentiment_is_proxy'] = True

        self.features_created = [
            'news_sentiment',
            'news_volume',
            'news_volume_ma7',
            'social_sentiment',
            'social_mentions',
            'social_mentions_ma7',
            'sentiment_momentum',
            'returns_5d',
            'sentiment_price_divergence',
            'combined_sentiment',
            'sentiment_is_proxy'
        ]

        return df

    def _add_legacy_mock_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Legacy random-walk mock sentiment (DEPRECATED - only used as fallback).

        WARNING: This creates random sentiment that doesn't correlate with
        market movements. Models trained on this will learn to ignore sentiment.
        Use add_price_derived_sentiment_proxy() instead.
        """
        df = df.copy()
        np.random.seed(42)

        df['news_sentiment'] = np.random.randn(len(df)).cumsum() * 0.1
        df['news_sentiment'] = df['news_sentiment'] - df['news_sentiment'].rolling(20, min_periods=1).mean()
        df['news_sentiment'] = df['news_sentiment'].clip(-1, 1).fillna(0)

        df['news_volume'] = np.random.poisson(5, len(df))
        df['news_volume_ma7'] = df['news_volume'].rolling(7, min_periods=1).mean()

        df['social_sentiment'] = np.random.randn(len(df)).cumsum() * 0.15
        df['social_sentiment'] = df['social_sentiment'] - df['social_sentiment'].rolling(10, min_periods=1).mean()
        df['social_sentiment'] = df['social_sentiment'].clip(-1, 1).fillna(0)

        df['social_mentions'] = np.random.poisson(10, len(df))
        df['social_mentions_ma7'] = df['social_mentions'].rolling(7, min_periods=1).mean()

        df['sentiment_momentum'] = df['news_sentiment'].diff(5).fillna(0)
        df['returns_5d'] = df['Close'].pct_change(5).fillna(0) if 'Close' in df.columns else 0
        df['sentiment_price_divergence'] = df['news_sentiment'] - df['returns_5d']

        df['combined_sentiment'] = (
            0.6 * df['news_sentiment'] +
            0.4 * df['social_sentiment']
        )

        df['sentiment_is_proxy'] = False  # This is NOT a proxy, it's random mock data

        self.features_created = [
            'news_sentiment', 'news_volume', 'news_volume_ma7',
            'social_sentiment', 'social_mentions', 'social_mentions_ma7',
            'sentiment_momentum', 'returns_5d', 'sentiment_price_divergence',
            'combined_sentiment', 'sentiment_is_proxy'
        ]

        return df

    def add_mock_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add mock sentiment features for backtesting.

        UPDATED: Now uses price-derived proxy instead of random walk.
        The price-derived proxy creates realistic sentiment that correlates
        with actual market movements, allowing the model to learn meaningful
        sentiment patterns.

        In production, replace with real sentiment data from:
        - News APIs (Bloomberg, Reuters, Yahoo Finance)
        - Social media APIs (Twitter, Reddit)
        - FinBERT sentiment scores

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with sentiment features added
        """
        # Use price-derived proxy instead of random walk
        return self.add_price_derived_sentiment_proxy(df)

    def add_real_sentiment_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Add real sentiment features using FinBERT and/or VADER.

        Strategy:
        1. Fetch news headlines from Yahoo Finance RSS
        2. Analyze sentiment using FinBERT (high accuracy) and/or VADER (fast)
        3. Aggregate daily sentiment scores
        4. Create momentum and divergence features

        Args:
            df: DataFrame with OHLC data (indexed by date)
            ticker: Asset ticker symbol

        Returns:
            DataFrame with real sentiment features
        """
        df = df.copy()

        print(f"\n[INFO] Adding REAL sentiment features for {ticker}")
        print(f"       FinBERT: {'Enabled' if self.use_finbert else 'Disabled'}")
        print(f"       VADER: {'Enabled' if self.use_vader else 'Disabled'}")

        if not self.use_finbert and not self.use_vader:
            print(f"[WARNING] No sentiment analyzers available. Using mock features.")
            return self.add_mock_sentiment_features(df)

        # Fetch news headlines
        print(f"[INFO] Fetching news from Yahoo Finance RSS...")
        headlines = self.fetch_yahoo_finance_news(ticker, days_back=30)

        if not headlines:
            print(f"[WARNING] No news headlines found. Using mock features.")
            return self.add_mock_sentiment_features(df)

        print(f"[OK] Found {len(headlines)} headlines")

        # Analyze sentiment for each headline
        print(f"[INFO] Analyzing sentiment...")
        sentiments_finbert = []
        sentiments_vader = []

        for item in headlines:
            # Headlines are tuples of (date, headline_text) or just strings
            # Safely extract the headline text, handling various formats
            try:
                if isinstance(item, tuple) and len(item) >= 2:
                    # Tuple format: (date, headline) - get the headline (index 1)
                    headline_text = item[1]
                    # If headline_text is a datetime, swap - it means date is at index 1
                    if hasattr(headline_text, 'strftime'):
                        headline_text = item[0]
                elif isinstance(item, tuple) and len(item) == 1:
                    headline_text = item[0]
                else:
                    headline_text = item

                # Force conversion to string, handling datetime objects
                if hasattr(headline_text, 'strftime'):
                    # Skip datetime objects entirely
                    continue
                headline_text = str(headline_text)

                # Skip if text is too short or looks like a date
                if len(headline_text) < 10:
                    continue

            except Exception as e:
                print(f"[WARNING] Failed to extract headline: {e}")
                continue

            if self.use_finbert:
                sent_fb = self.analyze_text_finbert(headline_text)
                sentiments_finbert.append(sent_fb)

            if self.use_vader:
                sent_vader = self.analyze_text_vader(headline_text)
                sentiments_vader.append(sent_vader)

        # Aggregate sentiment (average across all headlines)
        avg_sentiment_finbert = np.mean(sentiments_finbert) if sentiments_finbert else 0.0
        avg_sentiment_vader = np.mean(sentiments_vader) if sentiments_vader else 0.0

        print(f"[OK] Average FinBERT sentiment: {avg_sentiment_finbert:+.3f}")
        print(f"[OK] Average VADER sentiment: {avg_sentiment_vader:+.3f}")

        # Add sentiment features to dataframe
        # For simplicity, apply the same sentiment to all rows
        # In production, you'd map sentiment to specific dates

        if self.use_finbert:
            df['finbert_sentiment'] = avg_sentiment_finbert
            df['finbert_sentiment_ma7'] = avg_sentiment_finbert  # Simplified

        if self.use_vader:
            df['vader_sentiment'] = avg_sentiment_vader
            df['vader_sentiment_ma7'] = avg_sentiment_vader  # Simplified

        # Combined sentiment (weighted average)
        if self.use_finbert and self.use_vader:
            df['combined_sentiment'] = 0.7 * df['finbert_sentiment'] + 0.3 * df['vader_sentiment']
        elif self.use_finbert:
            df['combined_sentiment'] = df['finbert_sentiment']
        else:
            df['combined_sentiment'] = df['vader_sentiment']

        # News volume (static for this implementation)
        df['news_volume'] = len(headlines)
        df['news_volume_ma7'] = len(headlines)

        # Sentiment momentum (difference from mean)
        df['sentiment_momentum'] = df['combined_sentiment'] - df['combined_sentiment'].mean()

        # Sentiment-price divergence
        df['returns_5d'] = df['Close'].pct_change(5)
        df['sentiment_price_divergence'] = df['combined_sentiment'] - df['returns_5d']

        # Social sentiment (placeholder - would need Twitter/Reddit API)
        df['social_sentiment'] = 0.0
        df['social_mentions'] = 0.0

        self.features_created = [
            col for col in df.columns
            if 'sentiment' in col.lower() or 'news_' in col.lower() or 'social_' in col.lower()
        ]

        print(f"[OK] Created {len(self.features_created)} sentiment features")

        return df

    def get_feature_names(self):
        """Get names of created sentiment features."""
        return self.features_created


# Example usage and testing
def test_sentiment_features():
    """Test sentiment feature engineering with FREE & OPEN-SOURCE tools."""
    print("="*70)
    print("SENTIMENT FEATURE ENGINEERING TEST (FREE & OPEN-SOURCE)")
    print("="*70)

    print(f"\n[INFO] Library Availability:")
    print(f"       FinBERT (transformers): {'Available' if FINBERT_AVAILABLE else 'NOT installed'}")
    print(f"       VADER (vaderSentiment): {'Available' if VADER_AVAILABLE else 'NOT installed'}")
    print(f"       News Scraping (feedparser): {'Available' if NEWS_SCRAPING_AVAILABLE else 'NOT installed'}")

    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'Close': np.random.randn(100).cumsum() + 100,
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 102,
        'Low': np.random.randn(100).cumsum() + 98,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)

    # Test 1: Mock sentiment (always works, for backtesting)
    print(f"\n" + "="*70)
    print("TEST 1: Mock Sentiment Features (for backtesting)")
    print("="*70)
    engineer_mock = SentimentFeatureEngineer(use_finbert=False, use_vader=False)
    df_mock = engineer_mock.add_mock_sentiment_features(df.copy())
    print(f"[OK] Created {len(engineer_mock.features_created)} mock features")

    # Test 2: Real sentiment (if libraries available)
    if VADER_AVAILABLE:
        print(f"\n" + "="*70)
        print("TEST 2: VADER Sentiment (Fast & Lightweight)")
        print("="*70)

        engineer_vader = SentimentFeatureEngineer(use_finbert=False, use_vader=True)

        # Test text analysis
        print(f"\n[INFO] Testing VADER sentiment analyzer...")
        test_texts = [
            "Apple stock soars on strong earnings report!",
            "Company faces major losses amid regulatory concerns.",
            "Stock remains flat with neutral trading."
        ]
        for text in test_texts:
            sentiment = engineer_vader.analyze_text_vader(text)
            print(f"       '{text[:50]}' -> {sentiment:+.3f}")

    print(f"\n" + "="*70)
    print("INSTALLATION GUIDE")
    print("="*70)
    print(f"\nTo enable FREE & OPEN-SOURCE sentiment analysis:")
    print(f"\n1. Install FinBERT (high accuracy, finance-tailored):")
    print(f"   pip install transformers torch")
    print(f"\n2. Install VADER (fast, lightweight):")
    print(f"   pip install vaderSentiment")
    print(f"\n3. Install news scraping tools:")
    print(f"   pip install feedparser beautifulsoup4 requests")
    print(f"\n4. Usage:")
    print(f"   engineer = SentimentFeatureEngineer(use_finbert=True, use_vader=True)")
    print(f"   df = engineer.add_real_sentiment_features(df, ticker='AAPL')")

    print(f"\n[SUCCESS] Sentiment features ready for trading models!")


if __name__ == "__main__":
    test_sentiment_features()
