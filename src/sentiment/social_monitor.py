"""
Social Sentiment Monitoring System
Tracks sentiment from Reddit, Twitter/X, news, and influencer mentions
Sends push notifications when significant changes occur
"""

import requests
import time
from datetime import datetime, timedelta
from collections import defaultdict
import re
from typing import Dict, List, Optional

class SocialSentimentMonitor:
    """Monitor social media sentiment for stock tickers"""

    def __init__(self):
        self.sentiment_cache = {}
        self.influencers = {
            'twitter': [
                '@elonmusk',
                '@chamath',
                '@cathiedwood',
                '@jimcramer',
                '@carl_c_icahn',
                '@WarrenBuffett',
                '@BillAckman',
                '@RayDalio',
            ],
            'reddit': [
                'wallstreetbets',
                'stocks',
                'investing',
                'StockMarket',
                'options',
                'SPACs'
            ]
        }

    def get_reddit_sentiment(self, ticker: str, hours: int = 24) -> Dict:
        """
        Get Reddit sentiment for a ticker from multiple subreddits
        Uses pushshift.io API (free, no auth required)
        """
        sentiment_data = {
            'score': 0.0,  # -1 to 1
            'volume': 0,
            'trending': False,
            'hot_posts': [],
            'sentiment_shift': 0.0
        }

        try:
            # Search multiple subreddits
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket']
            all_posts = []

            for subreddit in subreddits:
                # Use Reddit API (no auth needed for public posts)
                url = f'https://www.reddit.com/r/{subreddit}/search.json'
                params = {
                    'q': f'${ticker} OR {ticker}',
                    'restrict_sr': 'on',
                    'sort': 'new',
                    'limit': 100,
                    't': 'day'
                }
                headers = {'User-Agent': 'Mozilla/5.0'}

                try:
                    response = requests.get(url, params=params, headers=headers, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        posts = data.get('data', {}).get('children', [])
                        all_posts.extend(posts)
                except Exception as e:
                    print(f"Reddit API error for {subreddit}: {e}")
                    continue

            if not all_posts:
                return sentiment_data

            sentiment_data['volume'] = len(all_posts)

            # Analyze sentiment from titles and scores
            positive_keywords = ['moon', 'bull', 'bullish', 'rocket', 'buy', 'calls', 'gains', 'up', 'green']
            negative_keywords = ['bear', 'bearish', 'sell', 'puts', 'crash', 'down', 'red', 'rip', 'loss']

            total_score = 0
            weighted_sentiment = 0
            hot_threshold = 500  # upvotes

            for post in all_posts:
                post_data = post.get('data', {})
                title = post_data.get('title', '').lower()
                score = post_data.get('score', 0)

                # Calculate sentiment from keywords
                pos_count = sum(1 for kw in positive_keywords if kw in title)
                neg_count = sum(1 for kw in negative_keywords if kw in title)

                if pos_count > neg_count:
                    post_sentiment = 1
                elif neg_count > pos_count:
                    post_sentiment = -1
                else:
                    post_sentiment = 0

                # Weight by upvotes
                weighted_sentiment += post_sentiment * (1 + score / 100)
                total_score += (1 + score / 100)

                # Track hot posts
                if score > hot_threshold:
                    sentiment_data['hot_posts'].append({
                        'title': post_data.get('title'),
                        'score': score,
                        'url': f"https://reddit.com{post_data.get('permalink', '')}",
                        'subreddit': post_data.get('subreddit'),
                        'sentiment': 'bullish' if post_sentiment > 0 else 'bearish' if post_sentiment < 0 else 'neutral'
                    })

            # Calculate overall sentiment score
            if total_score > 0:
                sentiment_data['score'] = weighted_sentiment / total_score

            # Detect trending (high volume in last 24h)
            sentiment_data['trending'] = len(all_posts) > 50

            return sentiment_data

        except Exception as e:
            print(f"Error getting Reddit sentiment for {ticker}: {e}")
            return sentiment_data

    def get_twitter_sentiment(self, ticker: str) -> Dict:
        """
        Get Twitter/X sentiment and influencer mentions
        Note: Requires Twitter API credentials (optional)
        Falls back to web scraping approach
        """
        sentiment_data = {
            'score': 0.0,
            'volume': 0,
            'influencer_mentions': [],
            'viral_tweets': []
        }

        # Placeholder for Twitter API integration
        # In production, use Twitter API v2 with bearer token
        # For now, return placeholder data

        return sentiment_data

    def get_news_sentiment(self, ticker: str) -> Dict:
        """
        Get news sentiment from financial news sources
        Uses NewsAPI (free tier available)
        """
        sentiment_data = {
            'score': 0.0,
            'articles': [],
            'headline_sentiment': 'neutral'
        }

        try:
            # Use NewsAPI (free tier: 100 requests/day)
            # Alternative: Use RSS feeds from Yahoo Finance, MarketWatch

            # Yahoo Finance RSS approach (no auth needed)
            url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}'
            headers = {'User-Agent': 'Mozilla/5.0'}

            try:
                response = requests.get(url, headers=headers, timeout=5)
                if response.status_code == 200:
                    # Parse RSS feed (simplified)
                    # In production, use feedparser library
                    content = response.text

                    # Extract headlines (basic regex parsing)
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(content)

                    articles = []
                    for item in root.findall('.//item')[:10]:
                        title = item.find('title')
                        link = item.find('link')
                        pubDate = item.find('pubDate')

                        if title is not None:
                            articles.append({
                                'title': title.text,
                                'url': link.text if link is not None else '',
                                'published': pubDate.text if pubDate is not None else ''
                            })

                    sentiment_data['articles'] = articles

                    # Simple sentiment analysis on headlines
                    positive_words = ['surge', 'gain', 'rally', 'beat', 'upgrade', 'growth', 'profit']
                    negative_words = ['fall', 'drop', 'loss', 'downgrade', 'miss', 'decline', 'warning']

                    pos_count = sum(1 for article in articles
                                  for word in positive_words
                                  if word in article['title'].lower())
                    neg_count = sum(1 for article in articles
                                  for word in negative_words
                                  if word in article['title'].lower())

                    if pos_count > neg_count:
                        sentiment_data['headline_sentiment'] = 'positive'
                        sentiment_data['score'] = 0.5
                    elif neg_count > pos_count:
                        sentiment_data['headline_sentiment'] = 'negative'
                        sentiment_data['score'] = -0.5

            except Exception as e:
                print(f"Yahoo Finance RSS error: {e}")

        except Exception as e:
            print(f"Error getting news sentiment for {ticker}: {e}")

        return sentiment_data

    def analyze_ticker(self, ticker: str) -> Dict:
        """
        Comprehensive sentiment analysis combining all sources
        """
        analysis = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'reddit': self.get_reddit_sentiment(ticker),
            'twitter': self.get_twitter_sentiment(ticker),
            'news': self.get_news_sentiment(ticker),
            'overall_sentiment': 'neutral',
            'sentiment_score': 0.0,
            'alert_level': 'none'  # none, low, medium, high, critical
        }

        # Calculate overall sentiment
        reddit_score = analysis['reddit']['score']
        news_score = analysis['news']['score']

        # Weighted average (Reddit has more weight due to retail sentiment)
        overall_score = (reddit_score * 0.6) + (news_score * 0.4)
        analysis['sentiment_score'] = overall_score

        if overall_score > 0.3:
            analysis['overall_sentiment'] = 'bullish'
        elif overall_score < -0.3:
            analysis['overall_sentiment'] = 'bearish'

        # Determine alert level
        reddit_trending = analysis['reddit']['trending']
        has_hot_posts = len(analysis['reddit']['hot_posts']) > 0
        high_volume = analysis['reddit']['volume'] > 100

        if reddit_trending and has_hot_posts and abs(overall_score) > 0.5:
            analysis['alert_level'] = 'critical'
        elif (reddit_trending or high_volume) and abs(overall_score) > 0.3:
            analysis['alert_level'] = 'high'
        elif has_hot_posts or abs(overall_score) > 0.2:
            analysis['alert_level'] = 'medium'
        elif abs(overall_score) > 0.1:
            analysis['alert_level'] = 'low'

        return analysis

    def should_send_alert(self, current_analysis: Dict, previous_analysis: Optional[Dict] = None) -> bool:
        """
        Determine if alert should be sent based on sentiment changes
        """
        if not previous_analysis:
            # First time - only alert on critical level
            return current_analysis['alert_level'] == 'critical'

        # Alert on:
        # 1. Sentiment shift from positive to negative or vice versa
        prev_sentiment = previous_analysis.get('overall_sentiment', 'neutral')
        curr_sentiment = current_analysis['overall_sentiment']

        sentiment_flip = (
            (prev_sentiment == 'bullish' and curr_sentiment == 'bearish') or
            (prev_sentiment == 'bearish' and curr_sentiment == 'bullish')
        )

        # 2. Alert level increase
        alert_levels = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        prev_level = alert_levels[previous_analysis.get('alert_level', 'none')]
        curr_level = alert_levels[current_analysis['alert_level']]

        level_increase = curr_level > prev_level and curr_level >= 2  # At least medium

        # 3. New viral content
        new_hot_posts = len(current_analysis['reddit']['hot_posts']) > len(previous_analysis.get('reddit', {}).get('hot_posts', []))

        return sentiment_flip or level_increase or new_hot_posts

    def generate_alert_message(self, analysis: Dict) -> str:
        """Generate human-readable alert message"""
        ticker = analysis['ticker']
        sentiment = analysis['overall_sentiment']
        level = analysis['alert_level']
        reddit_volume = analysis['reddit']['volume']

        emoji_map = {
            'bullish': '🚀',
            'bearish': '📉',
            'neutral': '➡️'
        }

        level_map = {
            'critical': '🔴 CRITICAL',
            'high': '🟠 HIGH',
            'medium': '🟡 MEDIUM',
            'low': '🟢 LOW'
        }

        message = f"{emoji_map[sentiment]} {ticker} - {level_map.get(level, 'INFO')}\n"
        message += f"Sentiment: {sentiment.upper()}\n"
        message += f"Social Volume: {reddit_volume} Reddit posts\n"

        # Add hot posts
        if analysis['reddit']['hot_posts']:
            message += "\n📊 Trending Posts:\n"
            for post in analysis['reddit']['hot_posts'][:3]:
                message += f"• {post['title']} ({post['score']} ⬆️)\n"

        # Add news headlines
        if analysis['news']['articles']:
            message += "\n📰 Latest News:\n"
            for article in analysis['news']['articles'][:2]:
                message += f"• {article['title']}\n"

        return message
