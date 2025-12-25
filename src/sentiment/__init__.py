"""
Social sentiment monitoring and analysis

Components:
- SocialSentimentMonitor: Original social monitoring (Reddit, Twitter, News)
- USIntlSentimentCollector: NEW - Real-time sentiment for US/Intl stocks
  - FinBERT: High-accuracy NLP sentiment (FREE)
  - Reddit: Real-time social sentiment (FREE public API)
  - Twitter: Placeholder for future ($100/mo API)

NOTE: China model uses DeepSeek API separately (not in this module).
"""
from .social_monitor import SocialSentimentMonitor

# US/Intl sentiment collector (FinBERT + Reddit + Twitter placeholder)
try:
    from .us_intl_sentiment import (
        USIntlSentimentCollector,
        USIntlSentimentResult,
        get_us_intl_sentiment_collector,
    )
    US_INTL_SENTIMENT_AVAILABLE = True
except ImportError:
    US_INTL_SENTIMENT_AVAILABLE = False

__all__ = [
    'SocialSentimentMonitor',
    'USIntlSentimentCollector',
    'USIntlSentimentResult',
    'get_us_intl_sentiment_collector',
    'US_INTL_SENTIMENT_AVAILABLE',
]
