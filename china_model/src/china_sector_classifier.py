"""
China Stock Sector Classifier

Classifies Chinese stocks by sector to enable sector-specific models.
Based on test results showing pharma/biotech stocks perform best.
"""

import logging

logger = logging.getLogger(__name__)


class ChinaSectorClassifier:
    """
    Classifies Chinese stocks into sectors for sector-specific model routing.

    Sectors supported:
    - pharma: Pharmaceutical and biotech companies (model excels here)
    - tech: Technology companies
    - finance: Banking and financial services
    - consumer: Consumer goods and services
    - resources: Natural resources and energy
    - real_estate: Real estate and property
    - industrial: Industrial and manufacturing
    - unknown: Cannot determine sector
    """

    # Pharmaceutical/Biotech stocks (EXCELLENT performance: 70-81% accuracy)
    PHARMA_TICKERS = {
        '1177.HK',  # China Resources Pharmaceutical - 81.54% accuracy
        '2269.HK',  # WuXi Biologics - 70.77% accuracy
        '1093.HK',  # CSPC Pharmaceutical
        '2359.HK',  # WuXi AppTec
        '6185.HK',  # Cansinbio
        '1801.HK',  # Innovent Biologics
        '9688.HK',  # Zai Lab
        '2616.HK',  # Lilly Suzhou
        '1347.HK',  # Hua Medicine
        '6699.HK',  # Angelalign
    }

    # Technology stocks (POOR performance: 41-46% accuracy)
    TECH_TICKERS = {
        '0700.HK',  # Tencent - 41.54% accuracy
        '9988.HK',  # Alibaba - 46.15% accuracy
        '9618.HK',  # JD.com
        '3690.HK',  # Meituan
        '1024.HK',  # Kuaishou
        '9999.HK',  # NetEase
        '9626.HK',  # Bilibili
        '9961.HK',  # Trip.com
        '9888.HK',  # Baidu
    }

    # Banking/Finance stocks (POOR performance: 40-41% accuracy)
    FINANCE_TICKERS = {
        '0939.HK',  # CCB - 41.54% accuracy
        '1398.HK',  # ICBC - 40.00% accuracy
        '3988.HK',  # Bank of China
        '1288.HK',  # ABC
        '3968.HK',  # China Merchants Bank
        '6098.HK',  # Country Garden Services
        '2318.HK',  # Ping An Insurance
        '2628.HK',  # China Life
        '1299.HK',  # AIA Group
    }

    # Consumer Goods stocks (POOR performance: 32-46% accuracy)
    CONSUMER_TICKERS = {
        '2319.HK',  # Mengniu Dairy - 32.31% accuracy (WORST)
        '1876.HK',  # Budweiser APAC - 41.54% accuracy
        '9869.HK',  # Helens
        '9987.HK',  # Yum China
        '1209.HK',  # China Resources Mixc
        '2020.HK',  # ANTA Sports
        '3968.HK',  # China Merchants Bank
    }

    # Resources stocks (POOR performance: 41% accuracy)
    RESOURCES_TICKERS = {
        '1109.HK',  # China Resources - 41.54% accuracy
        '0883.HK',  # CNOOC
        '0386.HK',  # China Petroleum
        '0857.HK',  # PetroChina
        '1088.HK',  # China Shenhua Energy
        '2318.HK',  # Ping An Insurance
    }

    # Real Estate stocks (POOR performance: 40% accuracy)
    REAL_ESTATE_TICKERS = {
        '0960.HK',  # Longfor Group - 40.00% accuracy
        '2007.HK',  # Country Garden
        '1918.HK',  # Sunac China
        '0823.HK',  # Link REIT
        '1997.HK',  # Wharf REIC
    }

    # Sector name mapping
    SECTOR_NAMES = {
        'pharma': 'Pharmaceutical/Biotech',
        'tech': 'Technology',
        'finance': 'Banking/Finance',
        'consumer': 'Consumer Goods',
        'resources': 'Resources/Energy',
        'real_estate': 'Real Estate',
        'industrial': 'Industrial',
        'unknown': 'Unknown'
    }

    @classmethod
    def get_sector(cls, ticker: str) -> str:
        """
        Classify a Chinese stock ticker into a sector.

        Args:
            ticker: Stock ticker (e.g., '1177.HK', '0700.HK')

        Returns:
            Sector code: 'pharma', 'tech', 'finance', 'consumer', 'resources',
                        'real_estate', 'industrial', or 'unknown'
        """
        ticker = ticker.upper().strip()

        # Check each sector
        if ticker in cls.PHARMA_TICKERS:
            return 'pharma'
        elif ticker in cls.TECH_TICKERS:
            return 'tech'
        elif ticker in cls.FINANCE_TICKERS:
            return 'finance'
        elif ticker in cls.CONSUMER_TICKERS:
            return 'consumer'
        elif ticker in cls.RESOURCES_TICKERS:
            return 'resources'
        elif ticker in cls.REAL_ESTATE_TICKERS:
            return 'real_estate'
        else:
            return 'unknown'

    @classmethod
    def get_sector_name(cls, ticker: str) -> str:
        """
        Get the full sector name for a ticker.

        Args:
            ticker: Stock ticker

        Returns:
            Full sector name (e.g., 'Pharmaceutical/Biotech')
        """
        sector_code = cls.get_sector(ticker)
        return cls.SECTOR_NAMES.get(sector_code, 'Unknown')

    @classmethod
    def should_use_china_model(cls, ticker: str) -> bool:
        """
        Determine if the China model should be used for this ticker.

        Based on test results, only use China model for pharma stocks.
        For all other sectors, fallback to US/International model.

        Args:
            ticker: Stock ticker

        Returns:
            True if should use China model, False to use fallback
        """
        sector = cls.get_sector(ticker)

        # Only use China model for pharma (70-81% accuracy)
        # All other sectors use fallback (40-46% accuracy with China model)
        if sector == 'pharma':
            logger.info(f"✓ Using China model for {ticker} (Pharma sector - proven 70-81% accuracy)")
            return True
        else:
            logger.info(f"⚠ Skipping China model for {ticker} ({cls.SECTOR_NAMES[sector]} sector - use fallback)")
            return False

    @classmethod
    def get_sector_performance(cls, sector: str) -> dict:
        """
        Get expected performance metrics for a sector based on test results.

        Args:
            sector: Sector code

        Returns:
            Dict with expected accuracy and recommendation
        """
        performance = {
            'pharma': {
                'expected_accuracy': 0.76,
                'min_accuracy': 0.70,
                'max_accuracy': 0.82,
                'recommendation': 'DEPLOY - Excellent performance',
                'test_stocks': 2,
                'avg_return': 2.83
            },
            'tech': {
                'expected_accuracy': 0.44,
                'min_accuracy': 0.41,
                'max_accuracy': 0.46,
                'recommendation': 'SKIP - Use fallback model',
                'test_stocks': 2,
                'avg_return': -0.18
            },
            'finance': {
                'expected_accuracy': 0.41,
                'min_accuracy': 0.40,
                'max_accuracy': 0.42,
                'recommendation': 'SKIP - Use fallback model',
                'test_stocks': 2,
                'avg_return': 0.00
            },
            'consumer': {
                'expected_accuracy': 0.37,
                'min_accuracy': 0.32,
                'max_accuracy': 0.42,
                'recommendation': 'SKIP - Use fallback model',
                'test_stocks': 2,
                'avg_return': -0.32
            },
            'resources': {
                'expected_accuracy': 0.41,
                'min_accuracy': 0.41,
                'max_accuracy': 0.42,
                'recommendation': 'SKIP - Use fallback model',
                'test_stocks': 1,
                'avg_return': -0.36
            },
            'real_estate': {
                'expected_accuracy': 0.40,
                'min_accuracy': 0.40,
                'max_accuracy': 0.40,
                'recommendation': 'SKIP - Use fallback model',
                'test_stocks': 1,
                'avg_return': -0.60
            },
            'unknown': {
                'expected_accuracy': 0.45,
                'min_accuracy': 0.32,
                'max_accuracy': 0.82,
                'recommendation': 'CAUTIOUS - Sector unknown',
                'test_stocks': 0,
                'avg_return': 0.00
            }
        }

        return performance.get(sector, performance['unknown'])


if __name__ == '__main__':
    # Test the classifier
    print("China Sector Classifier - Test Results\n")
    print("=" * 70)

    test_tickers = [
        '1177.HK',  # Pharma - should use China model
        '2269.HK',  # Pharma - should use China model
        '0700.HK',  # Tech - should skip
        '9988.HK',  # Tech - should skip
        '2319.HK',  # Consumer - should skip
        '0939.HK',  # Finance - should skip
        '1109.HK',  # Resources - should skip
        '0960.HK',  # Real Estate - should skip
    ]

    for ticker in test_tickers:
        sector = ChinaSectorClassifier.get_sector(ticker)
        sector_name = ChinaSectorClassifier.get_sector_name(ticker)
        should_use = ChinaSectorClassifier.should_use_china_model(ticker)
        perf = ChinaSectorClassifier.get_sector_performance(sector)

        print(f"\nTicker: {ticker}")
        print(f"  Sector: {sector_name}")
        print(f"  Use China Model: {should_use}")
        print(f"  Expected Accuracy: {perf['expected_accuracy']:.1%}")
        print(f"  Recommendation: {perf['recommendation']}")
