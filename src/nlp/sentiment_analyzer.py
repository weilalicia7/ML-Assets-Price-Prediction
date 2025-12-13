"""
Financial Sentiment Analyzer using FinBERT
Fine-tuned BERT model for financial sentiment analysis
"""

import warnings
import os

# Suppress PyTorch 2.6 weights_only warning (CVE-2025-32434)
# This is safe for trusted HuggingFace models like FinBERT
warnings.filterwarnings('ignore', message='.*weights_only.*')
os.environ['TORCH_WARN_WEIGHTS_ONLY'] = '0'

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
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Load model directly to device (fixes meta tensor issue)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False  # Disable meta device
            )

            # Handle meta tensor issue when moving to device
            try:
                self.model.to(self.device)
            except NotImplementedError as e:
                if "meta tensor" in str(e):
                    # Try loading directly to device
                    logger.warning(f"Meta tensor detected, reloading model directly on {self.device}")
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=False,
                        device_map=self.device
                    )
                else:
                    raise

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
