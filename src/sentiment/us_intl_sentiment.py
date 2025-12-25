"""
US/International Stock Sentiment Collector

Integrates FREE sentiment analysis tools for US/Intl model:
- FinBERT: High-accuracy financial sentiment (existing)
- Reddit: Real-time social sentiment (free public API)
- Twitter: Placeholder for future integration ($100/mo API)

NOTE: This file is ONLY for US/Intl model. China model uses DeepSeek API separately.

Cost: $0 (all free tools)
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class USIntlSentimentResult:
    """Sentiment analysis result for US/Intl stocks."""
    ticker: str
    timestamp: datetime

    # Sentiment scores
    finbert_sentiment: float        # -1 to +1 from FinBERT
    reddit_sentiment: float         # -1 to +1 from Reddit
    twitter_sentiment: float        # -1 to +1 from Twitter (placeholder)
    combined_sentiment: float       # Weighted combination

    # Volume/activity metrics
    reddit_mention_count: int
    reddit_engagement: int          # Total upvotes + comments
    twitter_mention_count: int      # 0 if placeholder

    # Anomaly detection
    spike_detected: bool
    spike_magnitude: float          # Percentage increase

    # Risk assessment
    risk_level: str                 # LOW, MEDIUM, HIGH
    warnings: List[str]
    confidence_multiplier: float    # Applied to ML prediction

    # Source tracking
    sources_used: List[str]         # Which sources had data
    twitter_placeholder: bool       # True if Twitter data is placeholder


class FinBERTAnalyzer:
    """
    FinBERT sentiment analyzer wrapper.

    Uses existing FinancialSentimentAnalyzer from src/nlp/sentiment_analyzer.py
    """

    def __init__(self):
        self.analyzer = None
        self._loaded = False

    def _load(self):
        """Lazy load FinBERT analyzer."""
        if self._loaded:
            return

        try:
            from src.nlp.sentiment_analyzer import get_sentiment_analyzer
            self.analyzer = get_sentiment_analyzer()
            self._loaded = True
            logger.info("[OK] FinBERT analyzer loaded")
        except Exception as e:
            logger.warning(f"[WARNING] FinBERT not available: {e}")
            self._loaded = True  # Mark as attempted

    def analyze(self, texts: List[str]) -> Dict:
        """
        Analyze sentiment of texts using FinBERT.

        Returns:
            Dict with average_sentiment (-1 to 1), individual scores, etc.
        """
        if not texts:
            return {'average_sentiment': 0, 'scores': [], 'available': False}

        self._load()

        if self.analyzer is None:
            return {'average_sentiment': 0, 'scores': [], 'available': False}

        try:
            results = self.analyzer.analyze_batch(texts[:50])  # Limit for performance

            # Convert FinBERT output to simple scores
            scores = []
            for r in results:
                # FinBERT returns: positive - negative
                score = r['scores'].get('positive', 0) - r['scores'].get('negative', 0)
                scores.append(score)

            return {
                'average_sentiment': np.mean(scores) if scores else 0,
                'scores': scores,
                'positive_count': sum(1 for s in scores if s > 0.3),
                'negative_count': sum(1 for s in scores if s < -0.3),
                'neutral_count': sum(1 for s in scores if -0.3 <= s <= 0.3),
                'available': True
            }
        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return {'average_sentiment': 0, 'scores': [], 'available': False}


class RedditCollector:
    """
    Reddit sentiment collector using public API (no auth required).

    Collects from: wallstreetbets, stocks, investing, StockMarket, options
    """

    def __init__(self):
        self.subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket', 'options']
        self.headers = {'User-Agent': 'Mozilla/5.0 (Stock Sentiment Bot)'}

        # Sentiment keywords
        self.positive_keywords = {
            'moon', 'bull', 'bullish', 'rocket', 'buy', 'calls', 'gains',
            'up', 'green', 'tendies', 'diamond', 'hold', 'long', 'breakout'
        }
        self.negative_keywords = {
            'bear', 'bearish', 'sell', 'puts', 'crash', 'down', 'red',
            'rip', 'loss', 'short', 'dump', 'plunge', 'tank', 'bag'
        }

    def get_mentions(self, ticker: str, hours: int = 24) -> Dict:
        """
        Get Reddit mentions for a ticker.

        Uses Reddit's public JSON API (no authentication needed).
        Rate limited but free.
        """
        import requests

        all_posts = []

        for subreddit in self.subreddits:
            try:
                url = f'https://www.reddit.com/r/{subreddit}/search.json'
                params = {
                    'q': f'${ticker} OR {ticker}',
                    'restrict_sr': 'on',
                    'sort': 'new',
                    'limit': 50,
                    't': 'day'
                }

                response = requests.get(url, params=params, headers=self.headers, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    posts = data.get('data', {}).get('children', [])

                    for post in posts:
                        post_data = post.get('data', {})
                        all_posts.append({
                            'title': post_data.get('title', ''),
                            'text': post_data.get('selftext', '')[:300],
                            'score': post_data.get('score', 0),
                            'comments': post_data.get('num_comments', 0),
                            'subreddit': subreddit,
                            'url': f"https://reddit.com{post_data.get('permalink', '')}"
                        })
                elif response.status_code == 429:
                    logger.warning(f"Reddit rate limit hit for {subreddit}")
                    break

            except requests.exceptions.Timeout:
                logger.warning(f"Reddit timeout for {subreddit}")
                continue
            except Exception as e:
                logger.warning(f"Reddit error for {subreddit}: {e}")
                continue

        if not all_posts:
            return {
                'posts': [],
                'count': 0,
                'engagement': 0,
                'sentiment': 0,
                'wsb_ratio': 0,
                'hot_posts': []
            }

        # Calculate sentiment from keywords
        total_sentiment = 0
        total_weight = 0
        wsb_count = 0
        hot_posts = []

        for post in all_posts:
            title_lower = post['title'].lower()
            text_lower = post.get('text', '').lower()
            combined = title_lower + ' ' + text_lower

            pos_count = sum(1 for kw in self.positive_keywords if kw in combined)
            neg_count = sum(1 for kw in self.negative_keywords if kw in combined)

            if pos_count > neg_count:
                post_sentiment = 1
            elif neg_count > pos_count:
                post_sentiment = -1
            else:
                post_sentiment = 0

            # Weight by engagement (upvotes + comments)
            weight = 1 + (post['score'] + post['comments']) / 100
            total_sentiment += post_sentiment * weight
            total_weight += weight

            if post['subreddit'] == 'wallstreetbets':
                wsb_count += 1

            # Track hot posts (high engagement)
            if post['score'] > 100:
                hot_posts.append({
                    'title': post['title'][:100],
                    'score': post['score'],
                    'subreddit': post['subreddit'],
                    'sentiment': 'bullish' if post_sentiment > 0 else 'bearish' if post_sentiment < 0 else 'neutral'
                })

        avg_sentiment = total_sentiment / total_weight if total_weight > 0 else 0
        total_engagement = sum(p['score'] + p['comments'] for p in all_posts)

        return {
            'posts': all_posts,
            'count': len(all_posts),
            'engagement': total_engagement,
            'sentiment': np.clip(avg_sentiment, -1, 1),
            'wsb_ratio': wsb_count / len(all_posts) if all_posts else 0,
            'hot_posts': sorted(hot_posts, key=lambda x: x['score'], reverse=True)[:5]
        }


class TwitterPlaceholder:
    """
    Twitter API placeholder for future integration.

    Twitter API v2 Basic tier costs $100/month.
    When ready to enable:
    1. Get API key from developer.twitter.com
    2. Set TWITTER_BEARER_TOKEN environment variable
    3. This class will automatically detect and use real data

    For now, returns empty/placeholder data.
    """

    def __init__(self):
        self.bearer_token = os.environ.get('TWITTER_BEARER_TOKEN')
        self.enabled = self.bearer_token is not None

        if self.enabled:
            logger.info("[OK] Twitter API enabled")
        else:
            logger.info("[INFO] Twitter API not configured (placeholder mode)")
            logger.info("[INFO] To enable: export TWITTER_BEARER_TOKEN=your_token")

    def get_mentions(self, ticker: str, hours: int = 24) -> Dict:
        """
        Get Twitter mentions for a ticker.

        Returns placeholder data if API not configured.
        """
        if not self.enabled:
            return {
                'tweets': [],
                'count': 0,
                'engagement': 0,
                'sentiment': 0,
                'placeholder': True,
                'message': 'Twitter API not configured. Using Reddit + FinBERT only.'
            }

        # Real Twitter API implementation
        try:
            import tweepy

            client = tweepy.Client(bearer_token=self.bearer_token)

            # Search for cashtag and ticker
            query = f"${ticker} OR #{ticker} -is:retweet lang:en"
            start_time = datetime.utcnow() - timedelta(hours=hours)

            tweets = client.search_recent_tweets(
                query=query,
                max_results=100,
                tweet_fields=['created_at', 'public_metrics', 'text'],
                start_time=start_time
            )

            if not tweets.data:
                return {
                    'tweets': [],
                    'count': 0,
                    'engagement': 0,
                    'sentiment': 0,
                    'placeholder': False
                }

            tweet_texts = [t.text for t in tweets.data]
            total_engagement = sum(
                t.public_metrics.get('like_count', 0) +
                t.public_metrics.get('retweet_count', 0)
                for t in tweets.data
            )

            return {
                'tweets': tweet_texts,
                'count': len(tweet_texts),
                'engagement': total_engagement,
                'sentiment': 0,  # Will be analyzed by FinBERT
                'placeholder': False
            }

        except ImportError:
            logger.warning("tweepy not installed. Run: pip install tweepy")
            return {'tweets': [], 'count': 0, 'engagement': 0, 'sentiment': 0, 'placeholder': True}
        except Exception as e:
            logger.error(f"Twitter API error: {e}")
            return {'tweets': [], 'count': 0, 'engagement': 0, 'sentiment': 0, 'placeholder': True}


class USIntlSentimentCollector:
    """
    Complete sentiment collector for US/International stocks.

    Combines:
    - FinBERT: High-accuracy NLP sentiment (FREE)
    - Reddit: Real-time social sentiment (FREE public API)
    - Twitter: Placeholder for future ($100/mo when enabled)

    This is ONLY for US/Intl model. China model uses DeepSeek separately.
    """

    def __init__(self):
        self.finbert = FinBERTAnalyzer()
        self.reddit = RedditCollector()
        self.twitter = TwitterPlaceholder()

        # Historical data for spike detection
        self.historical_counts: Dict[str, List[int]] = defaultdict(list)
        self.baseline_window = 30

        logger.info("=" * 50)
        logger.info("US/INTL SENTIMENT COLLECTOR INITIALIZED")
        logger.info("=" * 50)
        logger.info("Components:")
        logger.info("  - FinBERT: FREE (local NLP)")
        logger.info("  - Reddit:  FREE (public API)")
        logger.info(f"  - Twitter: {'ENABLED' if self.twitter.enabled else 'PLACEHOLDER ($100/mo to enable)'}")
        logger.info("=" * 50)

    def _detect_spike(self, ticker: str, count: int) -> Dict:
        """Detect unusual activity spikes using Z-score."""
        history = self.historical_counts[ticker]
        history.append(count)

        if len(history) > self.baseline_window:
            history.pop(0)

        if len(history) < 7:
            return {'spike_detected': False, 'z_score': 0, 'magnitude': 0}

        mean = np.mean(history[:-1])
        std = np.std(history[:-1])

        z_score = (count - mean) / std if std > 0 else 0
        magnitude = ((count - mean) / mean * 100) if mean > 0 else 0

        return {
            'spike_detected': z_score > 2,
            'z_score': z_score,
            'magnitude': magnitude
        }

    def get_sentiment(self, ticker: str, hours: int = 24) -> USIntlSentimentResult:
        """
        Get comprehensive sentiment for a US/Intl stock.

        Combines FinBERT analysis with Reddit (and Twitter if enabled).
        """
        sources_used = []
        warnings = []

        # 1. Get Reddit data
        reddit_data = self.reddit.get_mentions(ticker, hours)
        reddit_sentiment = reddit_data.get('sentiment', 0)
        reddit_count = reddit_data.get('count', 0)
        reddit_engagement = reddit_data.get('engagement', 0)

        if reddit_count > 0:
            sources_used.append('reddit')

        # 2. Get Twitter data (placeholder or real)
        twitter_data = self.twitter.get_mentions(ticker, hours)
        twitter_count = twitter_data.get('count', 0)
        twitter_placeholder = twitter_data.get('placeholder', True)

        if twitter_count > 0 and not twitter_placeholder:
            sources_used.append('twitter')

        # 3. Combine all texts for FinBERT analysis
        all_texts = []

        # Reddit titles and text
        for post in reddit_data.get('posts', []):
            all_texts.append(post['title'])
            if post.get('text'):
                all_texts.append(post['text'])

        # Twitter texts (if available)
        all_texts.extend(twitter_data.get('tweets', []))

        # 4. Analyze with FinBERT
        finbert_result = self.finbert.analyze(all_texts)
        finbert_sentiment = finbert_result.get('average_sentiment', 0)

        if finbert_result.get('available'):
            sources_used.append('finbert')

        # 5. Calculate combined sentiment
        # Weighting: FinBERT 50%, Reddit 40%, Twitter 10% (when available)
        if twitter_placeholder:
            # No Twitter data - reweight
            combined = 0.55 * finbert_sentiment + 0.45 * reddit_sentiment
        else:
            twitter_sentiment = twitter_data.get('sentiment', 0)
            combined = 0.50 * finbert_sentiment + 0.35 * reddit_sentiment + 0.15 * twitter_sentiment

        # 6. Detect activity spike
        total_mentions = reddit_count + twitter_count
        spike_info = self._detect_spike(ticker, total_mentions)

        # 7. Generate warnings

        # WSB concentration warning (potential meme stock)
        wsb_ratio = reddit_data.get('wsb_ratio', 0)
        if wsb_ratio > 0.7:
            warnings.append("WARNING: High WSB concentration - potential meme stock behavior")

        # Activity spike warning
        if spike_info['spike_detected']:
            warnings.append(f"SPIKE: Activity spike: +{spike_info['magnitude']:.0f}% mentions")

        # Sentiment warning
        if combined < -0.3:
            warnings.append("ALERT: Strong negative sentiment detected")

        # Pump/dump pattern (high WSB + extreme spike)
        if wsb_ratio > 0.5 and spike_info.get('magnitude', 0) > 300:
            warnings.append("RISK: Potential pump/dump pattern detected")

        # Twitter placeholder notice
        if twitter_placeholder:
            warnings.append("INFO: Twitter data unavailable (placeholder mode)")

        # 8. Calculate confidence multiplier
        confidence_mult = 1.0

        if spike_info['spike_detected']:
            confidence_mult *= 0.85

        if combined < -0.3:
            confidence_mult *= 0.90

        if wsb_ratio > 0.7 and spike_info.get('magnitude', 0) > 300:
            confidence_mult *= 0.70  # Pump/dump risk

        # 9. Determine risk level
        if len(warnings) >= 3 or any('pump' in w.lower() for w in warnings):
            risk_level = 'HIGH'
        elif len(warnings) >= 1:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'

        return USIntlSentimentResult(
            ticker=ticker,
            timestamp=datetime.utcnow(),
            finbert_sentiment=finbert_sentiment,
            reddit_sentiment=reddit_sentiment,
            twitter_sentiment=twitter_data.get('sentiment', 0),
            combined_sentiment=np.clip(combined, -1, 1),
            reddit_mention_count=reddit_count,
            reddit_engagement=reddit_engagement,
            twitter_mention_count=twitter_count,
            spike_detected=spike_info['spike_detected'],
            spike_magnitude=spike_info.get('magnitude', 0),
            risk_level=risk_level,
            warnings=warnings,
            confidence_multiplier=confidence_mult,
            sources_used=sources_used,
            twitter_placeholder=twitter_placeholder
        )

    def get_batch_sentiment(self, tickers: List[str], hours: int = 24) -> Dict[str, USIntlSentimentResult]:
        """Get sentiment for multiple tickers."""
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.get_sentiment(ticker, hours)
            except Exception as e:
                logger.error(f"Failed to get sentiment for {ticker}: {e}")
        return results

    def adjust_prediction(
        self,
        ticker: str,
        ml_prediction: float,
        ml_confidence: float,
        hours: int = 24
    ) -> Dict:
        """
        Adjust ML prediction with sentiment analysis.

        Weighting: 60% ML, 40% Sentiment

        Returns dict with original and adjusted values.
        """
        sentiment = self.get_sentiment(ticker, hours)

        ML_WEIGHT = 0.6
        SENTIMENT_WEIGHT = 0.4

        # Direction from ML
        ml_direction = 1 if ml_prediction > 0 else (-1 if ml_prediction < 0 else 0)

        # Direction from sentiment
        sent_direction = 1 if sentiment.combined_sentiment > 0.2 else (-1 if sentiment.combined_sentiment < -0.2 else 0)

        # Combined direction
        combined_dir = ML_WEIGHT * ml_direction + SENTIMENT_WEIGHT * sent_direction
        final_direction = 1 if combined_dir > 0.3 else (-1 if combined_dir < -0.3 else 0)

        # Adjusted confidence
        adjusted_ml_conf = ml_confidence * sentiment.confidence_multiplier

        return {
            'ticker': ticker,
            'timestamp': sentiment.timestamp.isoformat(),

            # Original ML prediction
            'ml_prediction': ml_prediction,
            'ml_confidence': ml_confidence,
            'ml_direction': ml_direction,

            # Sentiment analysis
            'finbert_sentiment': sentiment.finbert_sentiment,
            'reddit_sentiment': sentiment.reddit_sentiment,
            'twitter_sentiment': sentiment.twitter_sentiment,
            'combined_sentiment': sentiment.combined_sentiment,

            # Activity metrics
            'reddit_mentions': sentiment.reddit_mention_count,
            'twitter_mentions': sentiment.twitter_mention_count,
            'spike_detected': sentiment.spike_detected,

            # Combined output
            'final_direction': final_direction,
            'adjusted_confidence': adjusted_ml_conf,
            'confidence_multiplier': sentiment.confidence_multiplier,

            # Risk info
            'risk_level': sentiment.risk_level,
            'warnings': sentiment.warnings,

            # Source tracking
            'sources_used': sentiment.sources_used,
            'twitter_placeholder': sentiment.twitter_placeholder
        }


# Singleton instance
_us_intl_sentiment_collector = None


def get_us_intl_sentiment_collector() -> USIntlSentimentCollector:
    """Get or create global US/Intl sentiment collector."""
    global _us_intl_sentiment_collector
    if _us_intl_sentiment_collector is None:
        _us_intl_sentiment_collector = USIntlSentimentCollector()
    return _us_intl_sentiment_collector


# =============================================================================
# TESTING
# =============================================================================

def test_us_intl_sentiment():
    """Test the US/Intl sentiment collector."""
    print("=" * 60)
    print("US/INTL SENTIMENT COLLECTOR TEST")
    print("=" * 60)

    collector = USIntlSentimentCollector()

    # Test with a popular stock
    test_tickers = ['AAPL', 'TSLA', 'GME']

    for ticker in test_tickers:
        print(f"\n--- Testing {ticker} ---")
        try:
            result = collector.get_sentiment(ticker, hours=24)

            print(f"FinBERT Sentiment: {result.finbert_sentiment:+.3f}")
            print(f"Reddit Sentiment:  {result.reddit_sentiment:+.3f}")
            print(f"Combined:          {result.combined_sentiment:+.3f}")
            print(f"Reddit Mentions:   {result.reddit_mention_count}")
            print(f"Risk Level:        {result.risk_level}")
            print(f"Confidence Mult:   {result.confidence_multiplier:.2f}")
            print(f"Sources Used:      {result.sources_used}")
            print(f"Warnings:          {result.warnings}")
            print(f"Twitter Placeholder: {result.twitter_placeholder}")

        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_us_intl_sentiment()
