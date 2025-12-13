"""
Alternative Data Sources - Beyond Price Data
Professional quant firms use multi-modal data for alpha generation

Sources:
1. Google Trends - Search volume & interest
2. News Sentiment - Financial news analysis
3. Economic Calendar - Macro events
4. Corporate Events - Earnings, splits, dividends

This is what differentiates professional quant research from academic projects.
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import os
import time
from pytrends.request import TrendReq


class AlternativeDataCollector:
    """
    Collects alternative (non-price) data for enhanced predictions.

    Critical for:
    - Early signal detection
    - Sentiment analysis
    - Macro regime awareness
    - Event-driven strategies
    """

    def __init__(
        self,
        news_api_key: Optional[str] = None,
        fred_api_key: Optional[str] = None
    ):
        """
        Initialize alternative data collector.

        Args:
            news_api_key: NewsAPI.org API key (free at https://newsapi.org/)
            fred_api_key: FRED API key (free at https://fred.stlouisfed.org/)
        """
        self.news_api_key = news_api_key or os.getenv('NEWS_API_KEY')
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')

        # API endpoints
        self.news_api_url = "https://newsapi.org/v2"
        self.fred_url = "https://api.stlouisfed.org/fred/series/observations"

        # Google Trends setup (no API key needed)
        self.pytrends = TrendReq(hl='en-US', tz=360)

        print("[AlternativeData] Initialized")
        if self.news_api_key:
            print("[OK] NewsAPI configured")
        else:
            print("[INFO] NewsAPI not configured (optional)")

        if self.fred_api_key:
            print("[OK] FRED API configured")
        else:
            print("[INFO] FRED API not configured (optional)")

    # ===========================
    # GOOGLE TRENDS
    # ===========================

    def fetch_google_trends(
        self,
        keywords: List[str],
        timeframe: str = 'today 3-m',
        geo: str = 'US'
    ) -> pd.DataFrame:
        """
        Fetch Google Trends data for keywords.

        Critical for:
        - Retail investor interest
        - Meme stock detection
        - Crypto hype cycles
        - Brand awareness

        Args:
            keywords: List of search terms (max 5)
            timeframe: 'today 1-m', 'today 3-m', 'today 12-m', 'today 5-y', 'all'
            geo: Geographic location ('US', 'GB', 'DE', '')

        Returns:
            DataFrame with search interest over time (0-100 scale)
        """
        print(f"[Google Trends] Fetching data for: {keywords}")

        if len(keywords) > 5:
            raise ValueError("Google Trends allows max 5 keywords per request")

        try:
            # Build payload
            self.pytrends.build_payload(
                keywords,
                cat=0,
                timeframe=timeframe,
                geo=geo,
                gprop=''
            )

            # Get interest over time
            trends_data = self.pytrends.interest_over_time()

            if trends_data.empty:
                print(f"[WARN] No Google Trends data found for {keywords}")
                return pd.DataFrame()

            # Remove 'isPartial' column if present
            if 'isPartial' in trends_data.columns:
                trends_data = trends_data.drop('isPartial', axis=1)

            print(f"[OK] Retrieved {len(trends_data)} data points from Google Trends")
            print(f"     Date range: {trends_data.index.min()} to {trends_data.index.max()}")

            return trends_data

        except Exception as e:
            print(f"[ERROR] Google Trends error: {str(e)}")
            return pd.DataFrame()

    def fetch_related_queries(self, keyword: str) -> Dict:
        """
        Fetch related rising and top queries for a keyword.

        Useful for:
        - Detecting emerging trends
        - Understanding market narratives

        Args:
            keyword: Search term

        Returns:
            Dictionary with 'rising' and 'top' related queries
        """
        print(f"[Google Trends] Fetching related queries for: {keyword}")

        try:
            self.pytrends.build_payload([keyword], timeframe='today 3-m')
            related_queries = self.pytrends.related_queries()

            result = {
                'rising': related_queries[keyword]['rising'],
                'top': related_queries[keyword]['top']
            }

            print(f"[OK] Retrieved related queries")
            return result

        except Exception as e:
            print(f"[ERROR] Related queries error: {str(e)}")
            return {'rising': pd.DataFrame(), 'top': pd.DataFrame()}

    def calculate_trend_momentum(
        self,
        trends_df: pd.DataFrame,
        window: int = 7
    ) -> pd.DataFrame:
        """
        Calculate momentum and acceleration in search trends.

        Args:
            trends_df: DataFrame from fetch_google_trends
            window: Rolling window for calculations

        Returns:
            DataFrame with added momentum metrics
        """
        df = trends_df.copy()

        for col in df.columns:
            # Momentum (rate of change)
            df[f'{col}_momentum'] = df[col].pct_change(window)

            # Moving average
            df[f'{col}_ma'] = df[col].rolling(window).mean()

            # Trend strength (current vs MA)
            df[f'{col}_strength'] = (df[col] - df[f'{col}_ma']) / df[f'{col}_ma']

            # Acceleration (2nd derivative)
            df[f'{col}_acceleration'] = df[f'{col}_momentum'].diff()

        return df

    # ===========================
    # NEWS SENTIMENT
    # ===========================

    def fetch_news_sentiment(
        self,
        query: str,
        days: int = 7,
        language: str = 'en'
    ) -> pd.DataFrame:
        """
        Fetch news articles and sentiment for a query.

        Args:
            query: Search query (e.g., 'Apple OR AAPL', 'Bitcoin')
            days: Number of days to look back
            language: Language code ('en', 'de', 'fr', etc.)

        Returns:
            DataFrame with news articles and metadata
        """
        if not self.news_api_key:
            print("[WARN] NewsAPI key not configured. Returning empty data.")
            return pd.DataFrame()

        print(f"[NewsAPI] Fetching news for: {query}")

        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)

            url = f"{self.news_api_url}/everything"
            params = {
                'q': query,
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'language': language,
                'sortBy': 'publishedAt',
                'apiKey': self.news_api_key,
                'pageSize': 100
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data['status'] != 'ok':
                raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")

            articles = data.get('articles', [])

            if not articles:
                print(f"[WARN] No news articles found for {query}")
                return pd.DataFrame()

            # Parse articles
            news_data = []
            for article in articles:
                news_data.append({
                    'published_at': pd.to_datetime(article['publishedAt']),
                    'source': article['source']['name'],
                    'title': article['title'],
                    'description': article['description'],
                    'url': article['url'],
                    'author': article.get('author', 'Unknown')
                })

            df = pd.DataFrame(news_data)
            df = df.set_index('published_at').sort_index()

            print(f"[OK] Retrieved {len(df)} news articles")
            print(f"     Date range: {df.index.min()} to {df.index.max()}")

            return df

        except Exception as e:
            print(f"[ERROR] NewsAPI error: {str(e)}")
            return pd.DataFrame()

    def calculate_news_sentiment_features(
        self,
        news_df: pd.DataFrame,
        sentiment_analyzer=None
    ) -> pd.DataFrame:
        """
        Calculate sentiment features from news articles.

        Args:
            news_df: DataFrame from fetch_news_sentiment
            sentiment_analyzer: Optional sentiment analyzer (VADER or similar)

        Returns:
            Daily aggregated news sentiment features
        """
        if news_df.empty:
            return pd.DataFrame()

        print(f"[News Sentiment] Analyzing {len(news_df)} articles...")

        # Import VADER if not provided
        if sentiment_analyzer is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                sentiment_analyzer = SentimentIntensityAnalyzer()
            except ImportError:
                print("[WARN] vaderSentiment not installed. Install with: pip install vaderSentiment")
                return pd.DataFrame()

        # Analyze sentiment
        sentiments = []
        for _, row in news_df.iterrows():
            text = f"{row['title']} {row['description']}"
            if pd.notna(text):
                score = sentiment_analyzer.polarity_scores(text)
                sentiments.append(score['compound'])
            else:
                sentiments.append(0.0)

        news_df['sentiment'] = sentiments

        # Aggregate to daily
        daily_news = pd.DataFrame()
        daily_news['news_count'] = news_df.groupby(news_df.index.date).size()
        daily_news['news_sentiment_avg'] = news_df.groupby(news_df.index.date)['sentiment'].mean()
        daily_news['news_sentiment_std'] = news_df.groupby(news_df.index.date)['sentiment'].std()
        daily_news['news_positive_pct'] = news_df.groupby(news_df.index.date)['sentiment'].apply(
            lambda x: (x > 0.05).sum() / len(x) if len(x) > 0 else 0.5
        )
        daily_news['news_negative_pct'] = news_df.groupby(news_df.index.date)['sentiment'].apply(
            lambda x: (x < -0.05).sum() / len(x) if len(x) > 0 else 0.5
        )

        daily_news.index = pd.to_datetime(daily_news.index)
        daily_news.index.name = 'Date'

        print(f"[OK] Aggregated to {len(daily_news)} daily news features")

        return daily_news

    # ===========================
    # ECONOMIC CALENDAR
    # ===========================

    def fetch_economic_indicators(
        self,
        series_ids: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch economic indicators from FRED.

        Common indicators:
        - 'DFF': Federal Funds Rate
        - 'DGS10': 10-Year Treasury Yield
        - 'VIXCLS': VIX (Market Volatility)
        - 'DEXUSEU': USD/EUR Exchange Rate
        - 'UNRATE': Unemployment Rate
        - 'CPIAUCSL': Consumer Price Index
        - 'GDP': Gross Domestic Product

        Args:
            series_ids: List of FRED series IDs
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with economic indicators
        """
        if not self.fred_api_key:
            print("[WARN] FRED API key not configured. Returning empty data.")
            return pd.DataFrame()

        print(f"[FRED] Fetching economic indicators: {series_ids}")

        all_data = []

        for series_id in series_ids:
            try:
                params = {
                    'series_id': series_id,
                    'api_key': self.fred_api_key,
                    'file_type': 'json'
                }

                if start_date:
                    params['observation_start'] = start_date
                if end_date:
                    params['observation_end'] = end_date

                response = requests.get(self.fred_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if 'observations' not in data:
                    print(f"[WARN] No data for {series_id}")
                    continue

                # Parse observations
                obs = data['observations']
                df = pd.DataFrame(obs)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df[['value']].dropna()
                df.columns = [series_id]

                all_data.append(df)

                print(f"[OK] {series_id}: {len(df)} observations")
                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                print(f"[ERROR] Error fetching {series_id}: {str(e)}")
                continue

        if not all_data:
            return pd.DataFrame()

        # Combine all series
        combined = pd.concat(all_data, axis=1)
        combined.index.name = 'Date'

        print(f"[OK] Combined {len(series_ids)} economic indicators")

        return combined

    # ===========================
    # COMBINED DATA PIPELINE
    # ===========================

    def collect_all_alternative_data(
        self,
        ticker: str,
        keywords: Optional[List[str]] = None,
        days: int = 90
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect all alternative data sources for a ticker.

        Args:
            ticker: Stock/crypto ticker
            keywords: Google Trends keywords (default: [ticker])
            days: Lookback period

        Returns:
            Dictionary with all alternative data DataFrames
        """
        print(f"\n{'='*60}")
        print(f"COLLECTING ALTERNATIVE DATA FOR {ticker}")
        print(f"{'='*60}")

        result = {}

        # 1. Google Trends
        if keywords is None:
            keywords = [ticker.replace('-USD', '').replace('-USDT', '')]

        try:
            trends = self.fetch_google_trends(keywords, timeframe=f'today {days//30}-m')
            if not trends.empty:
                trends_with_features = self.calculate_trend_momentum(trends)
                result['google_trends'] = trends_with_features
        except Exception as e:
            print(f"[WARN] Google Trends collection failed: {str(e)}")

        # 2. News Sentiment
        try:
            news = self.fetch_news_sentiment(ticker, days=min(days, 30))
            if not news.empty:
                news_features = self.calculate_news_sentiment_features(news)
                result['news_sentiment'] = news_features
        except Exception as e:
            print(f"[WARN] News sentiment collection failed: {str(e)}")

        # 3. Economic Indicators (for context)
        try:
            economic = self.fetch_economic_indicators(
                ['VIXCLS', 'DFF', 'DGS10'],
                start_date=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            )
            if not economic.empty:
                result['economic_indicators'] = economic
        except Exception as e:
            print(f"[WARN] Economic indicators collection failed: {str(e)}")

        print(f"\n[OK] Collected {len(result)} alternative data sources")
        return result


def main():
    """
    Example usage of AlternativeDataCollector.
    """
    print("="*80)
    print("ALTERNATIVE DATA COLLECTOR - EXAMPLES")
    print("="*80)

    collector = AlternativeDataCollector()

    # Example 1: Google Trends
    print("\n\nExample 1: Google Trends - Tesla")
    print("-" * 60)
    try:
        trends = collector.fetch_google_trends(['Tesla', 'TSLA'], timeframe='today 3-m')
        print(trends.tail())

        # Calculate momentum
        trends_momentum = collector.calculate_trend_momentum(trends)
        print("\nWith momentum features:")
        print(trends_momentum[['Tesla', 'Tesla_momentum', 'Tesla_strength']].tail())
    except Exception as e:
        print(f"Error: {e}")

    # Example 2: News Sentiment
    print("\n\nExample 2: News Sentiment - Apple")
    print("-" * 60)
    try:
        news = collector.fetch_news_sentiment('Apple OR AAPL', days=7)
        if not news.empty:
            print(f"Retrieved {len(news)} articles")
            print(news[['source', 'title']].head())

            # Calculate sentiment features
            news_features = collector.calculate_news_sentiment_features(news)
            print("\nDaily sentiment features:")
            print(news_features.tail())
    except Exception as e:
        print(f"Error: {e}")

    # Example 3: Economic Indicators
    print("\n\nExample 3: Economic Indicators")
    print("-" * 60)
    try:
        economic = collector.fetch_economic_indicators(
            ['VIXCLS', 'DFF', 'DGS10'],
            start_date='2024-01-01'
        )
        print(economic.tail())
    except Exception as e:
        print(f"Error: {e}")

    # Example 4: Collect all alternative data
    print("\n\nExample 4: Collect All Alternative Data - Bitcoin")
    print("-" * 60)
    try:
        btc_alt_data = collector.collect_all_alternative_data(
            'BTC-USD',
            keywords=['Bitcoin', 'BTC'],
            days=90
        )

        print("\nAvailable data sources:")
        for source, df in btc_alt_data.items():
            print(f"  - {source}: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
