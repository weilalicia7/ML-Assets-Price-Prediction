"""
Social Sentiment Data Collector
Tracks Reddit, Twitter, StockTwits, and Google Trends for stock/crypto sentiment.

Critical for:
- Meme stocks (GME, AMC)
- Crypto (influenced heavily by social media)
- Retail investor activity detection
- Early warning for volatility spikes
"""

import praw
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SocialSentimentCollector:
    """
    Collects social media sentiment from multiple sources.

    Sources:
    - Reddit (r/wallstreetbets, r/stocks, r/CryptoCurrency)
    - StockTwits (ticker-specific sentiment)
    - Twitter (optional - requires API key)
    - Google Trends (search volume)
    """

    def __init__(
        self,
        reddit_client_id: Optional[str] = None,
        reddit_client_secret: Optional[str] = None,
        twitter_bearer_token: Optional[str] = None
    ):
        """
        Initialize social sentiment collector.

        Args:
            reddit_client_id: Reddit API client ID (get from https://www.reddit.com/prefs/apps)
            reddit_client_secret: Reddit API secret
            twitter_bearer_token: Twitter API v2 bearer token (optional)
        """
        # Reddit setup
        self.reddit_client_id = reddit_client_id or os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = reddit_client_secret or os.getenv('REDDIT_CLIENT_SECRET')

        if self.reddit_client_id and self.reddit_client_secret:
            self.reddit = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent='stock_predictor/1.0'
            )
            self.reddit_enabled = True
            print("[OK] Reddit API initialized")
        else:
            self.reddit_enabled = False
            print("[WARN] Reddit API not configured. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET")

        # Twitter setup (optional)
        self.twitter_bearer_token = twitter_bearer_token or os.getenv('TWITTER_BEARER_TOKEN')
        self.twitter_enabled = bool(self.twitter_bearer_token)

        if self.twitter_enabled:
            print("[OK] Twitter API initialized")
        else:
            print("[INFO] Twitter API not configured (optional)")

        # StockTwits (no auth needed)
        self.stocktwits_base = "https://api.stocktwits.com/api/2"

        # Sentiment analyzer
        self.sentiment_analyzer = self._setup_sentiment_analyzer()

        # Subreddits to monitor
        self.stock_subreddits = [
            'wallstreetbets',
            'stocks',
            'investing',
            'pennystocks'
        ]

        self.crypto_subreddits = [
            'CryptoCurrency',
            'Bitcoin',
            'ethereum',
            'CryptoMarkets'
        ]

    def _setup_sentiment_analyzer(self) -> SentimentIntensityAnalyzer:
        """
        Setup VADER sentiment analyzer with financial slang.
        """
        analyzer = SentimentIntensityAnalyzer()

        # Add financial/WSB slang
        financial_lexicon = {
            # Emojis (highly bullish/bearish)
            '🚀': 4.0,
            '📈': 3.0,
            '📉': -3.0,
            '💎': 3.0,
            '🙌': 2.0,
            '💩': -3.0,
            '🐻': -2.0,
            '🐂': 2.0,

            # WSB slang
            'moon': 3.0,
            'mooning': 3.0,
            'diamond hands': 3.0,
            'paper hands': -2.0,
            'YOLO': 2.0,
            'DD': 1.0,
            'tendies': 2.0,
            'bagholding': -2.0,
            'bagholder': -2.0,
            'apes': 2.0,
            'stonks': 2.0,
            'HODL': 2.0,
            'lambo': 3.0,

            # Market terms
            'bull': 2.0,
            'bullish': 2.0,
            'bear': -2.0,
            'bearish': -2.0,
            'pump': 2.0,
            'dump': -2.0,
            'crash': -3.0,
            'squeeze': 3.0,
            'short squeeze': 3.0,
            'rally': 2.0,
            'dip': -1.0,
            'buy the dip': 1.5,
            'rekt': -3.0,
            'winning': 2.0,
            'losing': -2.0
        }

        analyzer.lexicon.update(financial_lexicon)
        return analyzer

    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score from -1 (very negative) to 1 (very positive)
        """
        scores = self.sentiment_analyzer.polarity_scores(text)
        return scores['compound']

    def collect_reddit_sentiment(
        self,
        ticker: str,
        is_crypto: bool = False,
        limit: int = 100
    ) -> Dict:
        """
        Collect Reddit sentiment for a ticker.

        Args:
            ticker: Stock/crypto ticker (e.g., 'AAPL', 'BTC')
            is_crypto: Whether this is a cryptocurrency
            limit: Number of posts to analyze per subreddit

        Returns:
            Dictionary with Reddit metrics
        """
        if not self.reddit_enabled:
            return self._empty_reddit_data()

        subreddits = self.crypto_subreddits if is_crypto else self.stock_subreddits

        all_mentions = []
        total_upvotes = 0
        total_comments = 0

        # Search pattern
        search_terms = [f"${ticker}", ticker.upper()]

        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)

                # Get hot posts
                for post in subreddit.hot(limit=limit):
                    # Check if ticker mentioned
                    text = f"{post.title} {post.selftext}"
                    if any(term in text.upper() for term in search_terms):
                        sentiment = self.analyze_sentiment(text)

                        all_mentions.append({
                            'text': text,
                            'sentiment': sentiment,
                            'upvotes': post.score,
                            'comments': post.num_comments,
                            'subreddit': subreddit_name,
                            'created_utc': datetime.fromtimestamp(post.created_utc)
                        })

                        total_upvotes += post.score
                        total_comments += post.num_comments

            except Exception as e:
                print(f"[WARN] Error fetching from r/{subreddit_name}: {str(e)}")
                continue

        # Calculate metrics
        if not all_mentions:
            return self._empty_reddit_data()

        sentiments = [m['sentiment'] for m in all_mentions]

        return {
            'reddit_mentions': len(all_mentions),
            'reddit_sentiment_avg': np.mean(sentiments),
            'reddit_sentiment_std': np.std(sentiments),
            'reddit_bullish_pct': len([s for s in sentiments if s > 0.05]) / len(sentiments),
            'reddit_bearish_pct': len([s for s in sentiments if s < -0.05]) / len(sentiments),
            'reddit_total_upvotes': total_upvotes,
            'reddit_total_comments': total_comments,
            'reddit_avg_upvotes': total_upvotes / len(all_mentions),
            'reddit_engagement': total_upvotes + total_comments,
            'reddit_mentions_wsb': len([m for m in all_mentions if m['subreddit'] == 'wallstreetbets'])
        }

    def collect_stocktwits_sentiment(self, ticker: str) -> Dict:
        """
        Collect StockTwits sentiment for a ticker.

        Args:
            ticker: Stock ticker (e.g., 'AAPL')

        Returns:
            Dictionary with StockTwits metrics
        """
        try:
            url = f"{self.stocktwits_base}/streams/symbol/{ticker}.json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            messages = data.get('messages', [])

            if not messages:
                return self._empty_stocktwits_data()

            # Count sentiment
            bullish = 0
            bearish = 0
            neutral = 0

            for msg in messages:
                sentiment = msg.get('entities', {}).get('sentiment', {})
                if sentiment:
                    basic = sentiment.get('basic')
                    if basic == 'Bullish':
                        bullish += 1
                    elif basic == 'Bearish':
                        bearish += 1
                    else:
                        neutral += 1

            total_labeled = bullish + bearish

            return {
                'stocktwits_messages': len(messages),
                'stocktwits_bullish': bullish,
                'stocktwits_bearish': bearish,
                'stocktwits_neutral': neutral,
                'stocktwits_bullish_pct': bullish / total_labeled if total_labeled > 0 else 0.5,
                'stocktwits_sentiment_score': (bullish - bearish) / total_labeled if total_labeled > 0 else 0.0
            }

        except Exception as e:
            print(f"[WARN] Error fetching StockTwits for {ticker}: {str(e)}")
            return self._empty_stocktwits_data()

    def collect_twitter_sentiment(self, ticker: str, max_results: int = 100) -> Dict:
        """
        Collect Twitter sentiment for a ticker (requires API key).

        Args:
            ticker: Stock/crypto ticker
            max_results: Maximum tweets to fetch (10-100)

        Returns:
            Dictionary with Twitter metrics
        """
        if not self.twitter_enabled:
            return self._empty_twitter_data()

        try:
            # Twitter API v2 endpoint
            url = "https://api.twitter.com/2/tweets/search/recent"

            headers = {
                'Authorization': f'Bearer {self.twitter_bearer_token}'
            }

            params = {
                'query': f'${ticker} -is:retweet lang:en',
                'max_results': min(max_results, 100),
                'tweet.fields': 'public_metrics,created_at'
            }

            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            tweets = data.get('data', [])

            if not tweets:
                return self._empty_twitter_data()

            sentiments = []
            total_engagement = 0

            for tweet in tweets:
                text = tweet.get('text', '')
                sentiment = self.analyze_sentiment(text)
                sentiments.append(sentiment)

                metrics = tweet.get('public_metrics', {})
                engagement = (
                    metrics.get('retweet_count', 0) +
                    metrics.get('reply_count', 0) +
                    metrics.get('like_count', 0)
                )
                total_engagement += engagement

            return {
                'twitter_mentions': len(tweets),
                'twitter_sentiment_avg': np.mean(sentiments),
                'twitter_sentiment_std': np.std(sentiments),
                'twitter_bullish_pct': len([s for s in sentiments if s > 0.05]) / len(sentiments),
                'twitter_engagement': total_engagement,
                'twitter_avg_engagement': total_engagement / len(tweets)
            }

        except Exception as e:
            print(f"[WARN] Error fetching Twitter for {ticker}: {str(e)}")
            return self._empty_twitter_data()

    def collect_all(self, ticker: str, is_crypto: bool = False) -> Dict:
        """
        Collect sentiment from all available sources.

        Args:
            ticker: Stock/crypto ticker
            is_crypto: Whether this is a cryptocurrency

        Returns:
            Combined dictionary with all metrics
        """
        print(f"\n[INFO] Collecting social sentiment for {ticker}...")

        # Collect from all sources
        reddit_data = self.collect_reddit_sentiment(ticker, is_crypto)
        stocktwits_data = self.collect_stocktwits_sentiment(ticker)
        twitter_data = self.collect_twitter_sentiment(ticker)

        # Combine
        combined = {
            **reddit_data,
            **stocktwits_data,
            **twitter_data,
            'ticker': ticker,
            'timestamp': datetime.now(),
            'is_crypto': is_crypto
        }

        # Calculate aggregate metrics
        combined['total_mentions'] = (
            combined['reddit_mentions'] +
            combined['stocktwits_messages'] +
            combined['twitter_mentions']
        )

        # Weighted sentiment (Reddit has more signal for stocks)
        weights = {
            'reddit': 0.5,
            'stocktwits': 0.3,
            'twitter': 0.2
        }

        combined['sentiment_weighted'] = (
            combined['reddit_sentiment_avg'] * weights['reddit'] +
            combined['stocktwits_sentiment_score'] * weights['stocktwits'] +
            combined['twitter_sentiment_avg'] * weights['twitter']
        )

        # Hype score (mentions + sentiment)
        combined['hype_score'] = (
            combined['total_mentions'] / 100 *  # Normalize mentions
            (1 + combined['sentiment_weighted'])  # Weight by sentiment
        )

        print(f"[OK] Collected {combined['total_mentions']} total mentions")
        print(f"     Sentiment: {combined['sentiment_weighted']:.2f}")
        print(f"     Hype Score: {combined['hype_score']:.2f}")

        return combined

    def _empty_reddit_data(self) -> Dict:
        """Return empty Reddit data structure."""
        return {
            'reddit_mentions': 0,
            'reddit_sentiment_avg': 0.0,
            'reddit_sentiment_std': 0.0,
            'reddit_bullish_pct': 0.5,
            'reddit_bearish_pct': 0.5,
            'reddit_total_upvotes': 0,
            'reddit_total_comments': 0,
            'reddit_avg_upvotes': 0,
            'reddit_engagement': 0,
            'reddit_mentions_wsb': 0
        }

    def _empty_stocktwits_data(self) -> Dict:
        """Return empty StockTwits data structure."""
        return {
            'stocktwits_messages': 0,
            'stocktwits_bullish': 0,
            'stocktwits_bearish': 0,
            'stocktwits_neutral': 0,
            'stocktwits_bullish_pct': 0.5,
            'stocktwits_sentiment_score': 0.0
        }

    def _empty_twitter_data(self) -> Dict:
        """Return empty Twitter data structure."""
        return {
            'twitter_mentions': 0,
            'twitter_sentiment_avg': 0.0,
            'twitter_sentiment_std': 0.0,
            'twitter_bullish_pct': 0.5,
            'twitter_engagement': 0,
            'twitter_avg_engagement': 0
        }

    def detect_viral_event(self, ticker: str, historical_avg: Dict) -> bool:
        """
        Detect if ticker is going viral (potential squeeze/pump).

        Args:
            ticker: Stock/crypto ticker
            historical_avg: Historical average metrics

        Returns:
            True if viral event detected
        """
        current = self.collect_all(ticker)

        # Check for spikes
        mention_spike = current['total_mentions'] > historical_avg.get('total_mentions', 10) * 3
        sentiment_extreme = abs(current['sentiment_weighted']) > 0.7
        wsb_activity = current['reddit_mentions_wsb'] > 10

        viral = mention_spike and (sentiment_extreme or wsb_activity)

        if viral:
            print(f"\n[ALERT] Viral event detected for {ticker}!")
            print(f"        Mentions: {current['total_mentions']} (avg: {historical_avg.get('total_mentions', 0)})")
            print(f"        Sentiment: {current['sentiment_weighted']:.2f}")
            print(f"        WSB mentions: {current['reddit_mentions_wsb']}")

        return viral


def main():
    """
    Example usage of SocialSentimentCollector.
    """
    print("="*60)
    print("SOCIAL SENTIMENT COLLECTOR - EXAMPLE")
    print("="*60)

    # Initialize (uses environment variables if available)
    collector = SocialSentimentCollector()

    # Example 1: Collect for a stock
    print("\n\nExample 1: GameStop (Meme Stock)")
    print("-"*60)
    gme_sentiment = collector.collect_all('GME', is_crypto=False)
    print(f"\nGME Total Mentions: {gme_sentiment['total_mentions']}")
    print(f"GME Sentiment: {gme_sentiment['sentiment_weighted']:.2f}")
    print(f"GME Hype Score: {gme_sentiment['hype_score']:.2f}")

    # Example 2: Collect for crypto
    print("\n\nExample 2: Bitcoin (Crypto)")
    print("-"*60)
    btc_sentiment = collector.collect_all('BTC', is_crypto=True)
    print(f"\nBTC Total Mentions: {btc_sentiment['total_mentions']}")
    print(f"BTC Sentiment: {btc_sentiment['sentiment_weighted']:.2f}")

    # Example 3: Regular stock
    print("\n\nExample 3: Apple (Regular Stock)")
    print("-"*60)
    aapl_sentiment = collector.collect_all('AAPL', is_crypto=False)
    print(f"\nAAPL Total Mentions: {aapl_sentiment['total_mentions']}")
    print(f"AAPL Sentiment: {aapl_sentiment['sentiment_weighted']:.2f}")

    # Example 4: Viral detection
    print("\n\nExample 4: Viral Event Detection")
    print("-"*60)
    historical_avg = {'total_mentions': 50}  # Example baseline
    is_viral = collector.detect_viral_event('GME', historical_avg)
    print(f"GME Viral Event: {is_viral}")


if __name__ == "__main__":
    main()
