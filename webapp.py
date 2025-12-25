"""
ML Stock Trading Platform - Flask Web Application
Fast, lightweight UI that accurately reflects the ML prediction system
"""

import sys
sys.path.insert(0, '.')

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system environment variables

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_login import LoginManager
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
import json
from collections import defaultdict
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests  # For DeepSeek API calls

# Import ML models
from src.models.enhanced_ensemble import EnhancedEnsemblePredictor
from src.models.hybrid_lstm_cnn import HybridLSTMCNNPredictor  # NEW: Hybrid LSTM/CNN
from src.models.hybrid_ensemble import HybridEnsemblePredictor  # NEW: Ensemble combining old + new
from src.models.us_intl_optimizer import USIntlModelOptimizer, create_optimizer  # US/Intl Model Fixes 1-15

# NEW: Profit Maximization Optimizers (Priority Fixes 1-3 from profit test fixing1 full code.pdf)
# Priority 1: Energy subsector optimization (fixes -82% energy loss)
# Priority 2: Commodity dominance classification (fixes -91% mining loss)
# Priority 3: Dividend-aware optimization (improves financials)
try:
    from src.trading.integrated_us_optimizer import (
        IntegratedUSOptimizer,
        create_integrated_optimizer,
        IntegratedSignalOptimization,
    )
    PROFIT_OPTIMIZER_AVAILABLE = True
    print("[PROFIT OPTIMIZER] Successfully imported IntegratedUSOptimizer")
except ImportError as e:
    PROFIT_OPTIMIZER_AVAILABLE = False
    print(f"[PROFIT OPTIMIZER] Import failed (will use base optimizer only): {e}")

# RSI Risk Adapter from 'us model fixing5.pdf'
try:
    from src.trading.rsi_risk_adapter import (
        RSIRiskAdapter,
        RSIRiskLevel,
        RSIEnhancedSignal,
        create_rsi_adapter,
    )
    RSI_ADAPTER_AVAILABLE = True
    print("[RSI ADAPTER] Successfully imported RSIRiskAdapter (fixing5)")
except ImportError as e:
    RSI_ADAPTER_AVAILABLE = False
    print(f"[RSI ADAPTER] Import failed: {e}")

# Profit Maximizer from 'us model fixing6.pdf'
try:
    from src.trading.profit_maximizer import (
        ProfitMaximizer,
        DynamicEVPositionSizer,
        CompleteProfitMaximizationStrategy,
        create_profit_maximizer,
        create_complete_strategy,
        analyze_signal_ev,
    )
    PROFIT_MAXIMIZER_AVAILABLE = True
    print("[PROFIT MAXIMIZER] Successfully imported ProfitMaximizer (fixing6)")
except ImportError as e:
    PROFIT_MAXIMIZER_AVAILABLE = False
    print(f"[PROFIT MAXIMIZER] Import failed: {e}")

# US Profit-Maximizing Regime Classifier (from 'us model fixing8.pdf')
# Implements: Dynamic multiplier scaling, transition phase detection, Kelly criterion sizing
try:
    from src.features.profit_maximizing_regime import (
        ProfitMaximizingRegimeClassifier,
        ProfitMaximizingOutput,
        TransitionPhase,
        get_profit_optimized_regime,
    )
    US_PROFIT_REGIME_AVAILABLE = True
    print("[US PROFIT REGIME] Successfully imported ProfitMaximizingRegimeClassifier (fixing8)")
except ImportError as e:
    US_PROFIT_REGIME_AVAILABLE = False
    print(f"[US PROFIT REGIME] Import failed: {e}")

# US New Stock/IPO Handler (from 'dual model fixing1.pdf' - FIXING3)
# Implements: Tiered IPO analysis, SPAC handling, options-enhanced analysis
try:
    from src.trading.us_new_stock_handler import (
        USNewStockHandler,
        USSPACHandler,
        USIPOAnalyzer,
        apply_us_ipo_handler,
    )
    US_IPO_HANDLER_AVAILABLE = True
    print("[US IPO HANDLER] Successfully imported USNewStockHandler (fixing3)")
except ImportError as e:
    US_IPO_HANDLER_AVAILABLE = False
    print(f"[US IPO HANDLER] Import failed: {e}")

from src.features.technical_features import TechnicalFeatureEngineer
from src.features.volatility_features import VolatilityFeatureEngineer
from src.features.sentiment_features import SentimentFeatureEngineer  # NEW: Sentiment analysis
from src.config.asset_hyperparameters import AssetHyperparameters  # NEW: Asset-specific params
from src.models.regime_detector import RegimeDetector
from src.trading.risk_manager import RiskManager, TradingSignalGenerator
from src.trading.hybrid_strategy import OptimalHybridStrategy

# NEW: China-specific model system (dual model architecture with sector routing)
from src.models.market_classifier import MarketClassifier, ModelRouter
from src.models.china_predictor import ChinaMarketPredictor
from src.models.china_sector_router import ChinaSectorRouter  # NEW: Sector-based routing
from src.models.china_sector_classifier import ChinaSectorClassifier  # NEW: Sector classification
from src.features.china_macro_features import ChinaMacroFeatureEngineer
from src.features.selective_macro_features import SelectiveMacroFeatureEngineer

# US Model Fixing9 Implementation (from 'us model fixing9.pdf')
# P0: safe_fill_missing_features, US_CORE_FEATURES, lower SNR thresholds
# P1: profit score, Kelly sizing, dynamic targets
# P2: sector rotation, correlation limits
try:
    from src.fixes.us_model_fixing9 import (
        safe_fill_missing_features,
        select_core_features,
        calculate_confidence_fixing9,
        calculate_profit_score_fixing9,
        calculate_position_size_fixing9,
        calculate_profit_targets_fixing9,
        apply_fixing9_to_prediction,
        US_CORE_FEATURES,
        SNR_THRESHOLDS_FIXING9,
        SAFE_FEATURE_DEFAULTS,
    )
    FIXING9_AVAILABLE = True
    print("[FIXING9] Successfully imported US Model Fixing9 (P0-P2 fixes)")
except ImportError as e:
    FIXING9_AVAILABLE = False
    print(f"[FIXING9] Import failed (will use original method): {e}")

# Configure logging EARLY (needed for import error messages)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NEW: Phase 1-6 Calculation Utilities for enhanced signal processing
try:
    from src.utils.calculation_utils import (
        # Phase 1: Core Features & Trading
        calculate_volatility_scaling,
        correlation_aware_sizing,
        risk_parity_allocation,
        calculate_turnover,
        # Phase 2: Asset Class Ensembles
        safe_data_extraction,
        momentum_signal,
        volatility_signal,
        mean_reversion_signal,
        combine_signals,
        # Phase 3: Regime Detection & Order Flow
        detect_volatility_regime,
        calculate_hurst_exponent,
        calculate_obv,
        accumulation_distribution,
        money_flow_index,
        calculate_vwap,
        detect_smart_money,
        # Phase 4: Macro Integration
        asset_specific_multiplier,
        regime_persistence,
        # Phase 5: Dynamic Weighting & Bayesian
        decay_weighted_sharpe,
        calmar_ratio,
        composite_performance_score,
        BayesianSignalUpdater,
        bayesian_signal_combination,
        # Phase 6: Portfolio Optimization
        risk_contribution_analysis,
        expected_shortfall,
        regime_aware_risk_budget,
        portfolio_optimization,
        stress_test_portfolio,
        _calculate_parametric_es,
    )
    PHASE6_CALCULATIONS_AVAILABLE = True
    logger.info("[PHASE 1-6] Successfully imported Phase 1-6 calculation utilities")
except ImportError as e:
    PHASE6_CALCULATIONS_AVAILABLE = False
    logger.warning(f"[PHASE 1-6] Import failed (will use basic calculations): {e}")

# Enhanced Signal Validator for China Model (Fixes Catastrophic Shorts)
# NOTE: buyfix1 ACTIVE (trend filter, MA crossover, stop-loss 5-7%, position sizing)
#       buyfix2/fix3/fix4 DISABLED (optimizer classes)
try:
    from src.utils.enhanced_signal_validator import (
        EnhancedSignalValidator,
        SmartPositionSizer,
        EnhancedRegimeDetector,
        EnhancedPhaseSystem,
        validate_china_signal,
        # ChinaBuyOptimizer - DISABLED (buyfix2)
        # EnhancedChinaBuyOptimizer - DISABLED (buyfix3)
        # ProductionChinaBuyOptimizer - DISABLED (buyfix4)
    )
    ENHANCED_SIGNAL_VALIDATOR_AVAILABLE = True
    # Initialize global enhanced phase system (includes buyfix1 components)
    # buyfix1: Trend filter, MA crossover, stop-loss 5-7%, market regime filter, position sizing
    ENHANCED_PHASE_SYSTEM = EnhancedPhaseSystem()
    # DISABLED: Buy optimizer classes (buyfix2/fix3/fix4 only)
    CHINA_BUY_OPTIMIZER = None  # Disabled - no fix2/fix3/fix4 optimizer classes
    CHINA_BUY_MONITOR = None    # Disabled - no production monitoring
    logger.info("[SIGNAL VALIDATOR] Enhanced signal validation for China model loaded")
    logger.info("[BUYFIX1] ACTIVE - Trend filter, MA crossover, stop-loss, position sizing")
    logger.info("[BUYFIX2/3/4] DISABLED - No optimizer classes")
except ImportError as e:
    ENHANCED_SIGNAL_VALIDATOR_AVAILABLE = False
    ENHANCED_PHASE_SYSTEM = None
    CHINA_BUY_OPTIMIZER = None
    CHINA_BUY_MONITOR = None
    logger.warning(f"[SIGNAL VALIDATOR] Import failed (using basic validation): {e}")

# China Model Platform imports (Phase 2: Live Trading)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'china_model', 'src'))
try:
    from model_factory import ModelFactory, add_comprehensive_features
    from portfolio_constructor import PortfolioConstructor, RiskAnalyzer, SECTOR_MAP
    CHINA_MODEL_AVAILABLE = True
    logger.info("[CHINA MODEL] Successfully imported China Model Platform modules")
except ImportError as e:
    CHINA_MODEL_AVAILABLE = False
    logger.warning(f"[CHINA MODEL] Import failed (will use fallback): {e}")

# China Lag-Free Transition Detection (Fix 62 adaptation)
try:
    from china_lag_free_transition import (
        ChinaLagFreeTransitionDetector,
        integrate_transition_detection,
        ChinaTransitionOutput
    )
    from china_adaptive_profit_maximizer import ChinaAdaptiveProfitMaximizer
    CHINA_TRANSITION_AVAILABLE = True
    logger.info("[CHINA TRANSITION] Lag-free transition detection loaded")
except ImportError as e:
    CHINA_TRANSITION_AVAILABLE = False
    logger.warning(f"[CHINA TRANSITION] Import failed (will use fallback): {e}")

# Global China transition detector instance
CHINA_TRANSITION_DETECTOR = None
CHINA_PROFIT_MAXIMIZER = None
if CHINA_TRANSITION_AVAILABLE:
    try:
        CHINA_TRANSITION_DETECTOR = ChinaLagFreeTransitionDetector()
        # FIXING2: Enable all fixing2 enhancements by default
        CHINA_PROFIT_MAXIMIZER = ChinaAdaptiveProfitMaximizer(base_capital=100000, apply_fixing2=True)
        logger.info("[CHINA TRANSITION] Detector and profit maximizer initialized")
        logger.info("[CHINA FIXING3] Enhancements enabled: Relaxed EV, Composite Quality, Momentum Override, IPO Handler, Sector Rules")
    except Exception as e:
        logger.warning(f"[CHINA TRANSITION] Failed to initialize: {e}")

# Global US Profit-Maximizing Regime Classifier (from 'us model fixing8.pdf')
# NOTE: This is for US/International model ONLY - not China/DeepSeek
US_PROFIT_REGIME_CLASSIFIER = None
if US_PROFIT_REGIME_AVAILABLE:
    try:
        US_PROFIT_REGIME_CLASSIFIER = ProfitMaximizingRegimeClassifier(initial_vix=20.0)
        logger.info("[US PROFIT REGIME] Classifier initialized with fixing8 enhancements")
    except Exception as e:
        logger.warning(f"[US PROFIT REGIME] Failed to initialize: {e}")

# Enhanced Yahoo Finance data layer and China ticker resolver
try:
    from src.data.yahoo_finance_robust import (
        YahooFinanceRobust, get_realtime_yahoo_data, get_current_price,
        validate_ticker, clear_cache, get_cache_stats
    )
    from src.data.china_ticker_resolver import (
        resolve_china_stock, search_china_stocks, autocomplete,
        get_stock_info, get_all_tickers_by_sector, get_sectors,
        CHINA_STOCKS_DATABASE
    )
    ENHANCED_DATA_LAYER = True
    logger.info("[DATA LAYER] Enhanced Yahoo Finance and China ticker resolver loaded")
except ImportError as e:
    ENHANCED_DATA_LAYER = False
    logger.warning(f"[DATA LAYER] Enhanced modules not available: {e}")

# US/International Model Validation Framework
try:
    from src.testing.us_intl_model_validator import (
        USIntlModelValidator, CrossMarketValidator,
        MARKET_THRESHOLDS, get_full_universe
    )
    US_INTL_VALIDATOR_AVAILABLE = True
    logger.info("[VALIDATOR] US/International Model Validation Framework loaded")
except ImportError as e:
    US_INTL_VALIDATOR_AVAILABLE = False
    logger.warning(f"[VALIDATOR] US/Intl validator not available: {e}")

# Yahoo Finance Screener Discovery (Real-time ticker discovery)
try:
    from src.screeners.yahoo_screener_discovery import (
        YahooScreenerDiscoverer,
        RegimeScreenerStrategy,
        SmartScreenerCache,
        ReliabilityManager,
        get_screener_discoverer,
        get_regime_strategy,
        get_screener_cache,
        get_reliability_manager,
    )
    SCREENER_DISCOVERY_AVAILABLE = True
    logger.info("[SCREENER] Yahoo Finance Screener Discovery loaded - Real-time ticker discovery enabled")
except ImportError as e:
    SCREENER_DISCOVERY_AVAILABLE = False
    logger.warning(f"[SCREENER] Screener discovery not available (will use database fallback): {e}")

# Import authentication modules
from src.auth.models import db, User
from src.auth.oauth import setup_oauth
from src.auth.routes import auth_bp

# Import monitoring modules
from src.monitoring import (
    get_data_quality_monitor,
    get_performance_monitor,
    get_cache_monitor
)

# Phase 1 Advanced Features Integration (20 features for improved profit rate)
try:
    from src.trading.phase1_integration import (
        Phase1TradingSystem, Phase1APIEndpoints,
        get_phase1_system, get_phase1_api
    )
    PHASE1_AVAILABLE = True
    logger.info("[PHASE1] Advanced features integration loaded (20 features)")
except ImportError as e:
    PHASE1_AVAILABLE = False
    logger.warning(f"[PHASE1] Integration not available: {e}")


def engineer_market_specific_features(df, ticker):
    """
    Add market-specific features based on ticker classification.

    Args:
        df: DataFrame with OHLC data
        ticker: Stock ticker symbol

    Returns:
        DataFrame with appropriate features added
    """
    # Check if dual model system is enabled
    if not USE_DUAL_MODEL_SYSTEM:
        # Use US/Intl pipeline for all tickers
        logger.info(f"[FEATURES] Using US/Intl pipeline for {ticker} (dual model system disabled)")

        tech_eng = TechnicalFeatureEngineer()
        vol_eng = VolatilityFeatureEngineer()

        df = tech_eng.add_all_features(df)
        df = vol_eng.add_all_features(df)

        # Add US macro features
        try:
            macro_eng = SelectiveMacroFeatureEngineer()
            df = macro_eng.add_all_features(df)
            logger.info(f"[FEATURES] Added US macro features (VIX, SPY, DXY, GLD)")
        except Exception as e:
            logger.warning(f"[FEATURES] Failed to add US macro features: {e}")

        return df

    # Dual model system enabled - route based on market
    market = MarketClassifier.get_market(ticker)

    if market == 'chinese':
        # Use China-specific feature pipeline
        logger.info(f"[FEATURES] Using China pipeline for {ticker} (CSI300, CNY, HSI)")

        tech_eng = TechnicalFeatureEngineer()
        vol_eng = VolatilityFeatureEngineer()

        df = tech_eng.add_all_features(df)
        df = vol_eng.add_all_features(df)

        # Add China macro features
        try:
            china_macro_eng = ChinaMacroFeatureEngineer()
            df = china_macro_eng.add_all_features(df)
            logger.info(f"[FEATURES] Added China macro features (CSI300, CNY, HSI)")
        except Exception as e:
            logger.warning(f"[FEATURES] Failed to add China macro features: {e}")

    else:
        # Use US/International feature pipeline
        logger.info(f"[FEATURES] Using US/Intl pipeline for {ticker} (VIX, SPY, DXY)")

        tech_eng = TechnicalFeatureEngineer()
        vol_eng = VolatilityFeatureEngineer()

        df = tech_eng.add_all_features(df)
        df = vol_eng.add_all_features(df)

        # Add US macro features
        try:
            macro_eng = SelectiveMacroFeatureEngineer()
            df = macro_eng.add_all_features(df)
            logger.info(f"[FEATURES] Added US macro features (VIX, SPY, DXY, GLD)")
        except Exception as e:
            logger.warning(f"[FEATURES] Failed to add US macro features: {e}")

    return df


def clean_features_for_training(df, verbose=True):
    """
    Clean dataframe features to ensure no inf/nan values.
    Enhanced version from fix_commodity_forex_data.py

    Args:
        df: DataFrame with features
        verbose: Print cleaning stats

    Returns:
        Cleaned DataFrame
    """
    if verbose:
        logger.info(f"[CLEAN] Input shape: {df.shape}")
        logger.info(f"[CLEAN] NaN values: {df.isna().sum().sum()}")
        logger.info(f"[CLEAN] Inf values: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")

    # Replace inf with large finite values
    df = df.replace([np.inf], 1e10)
    df = df.replace([-np.inf], -1e10)

    # FIXING9: Fill remaining NaN with safe historical means (not zeros!)
    # Using 0 for features like VIX (normally 10-30) completely breaks the model
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if FIXING9_AVAILABLE:
        df = safe_fill_missing_features(df, verbose=verbose)
    else:
        # Fallback to zeros if fixing9 not available
        df[numeric_cols] = df[numeric_cols].fillna(0)

    # Clip extreme values (optional, helps with stability)
    for col in numeric_cols:
        if col not in ['Close', 'Open', 'High', 'Low', 'Volume', 'target', 'returns']:
            # Clip to 99.9th percentile to remove extreme outliers
            upper = df[col].quantile(0.999)
            lower = df[col].quantile(0.001)
            df[col] = df[col].clip(lower, upper)

    if verbose:
        logger.info(f"[CLEAN] After cleaning:")
        logger.info(f"[CLEAN]   NaN values: {df.isna().sum().sum()}")
        logger.info(f"[CLEAN]   Inf values: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")
        logger.info(f"[CLEAN]   Output shape: {df.shape}")

    return df


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure Flask app for authentication
# Use absolute path for database to ensure persistence
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.db')

# IMPORTANT: Use a fixed SECRET_KEY to prevent constant reloads
# In production, set SECRET_KEY environment variable
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production-12345')

app.config['SECRET_KEY'] = SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

logger.info(f"Database will be stored at: {DB_PATH}")

# Initialize database
db.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Setup OAuth providers
try:
    google_bp, github_bp, reddit_bp = setup_oauth(app)
except Exception as e:
    logger.warning(f"OAuth setup failed (this is expected without OAuth credentials): {e}")

# Register authentication blueprint
app.register_blueprint(auth_bp)

# Create database tables (only creates schema if not exists, preserves data)
with app.app_context():
    db.create_all()
    logger.info(f"Database tables created/verified successfully at: {DB_PATH}")

    # Check if database file exists
    if os.path.exists(DB_PATH):
        logger.info(f"Database file exists: {DB_PATH} (size: {os.path.getsize(DB_PATH)} bytes)")
        # Count existing users
        from src.auth.models import User, Watchlist, MockTrade, Portfolio
        user_count = User.query.count()
        watchlist_count = Watchlist.query.count()
        trade_count = MockTrade.query.count()
        portfolio_count = Portfolio.query.count()
        logger.info(f"Database contents: {user_count} users, {watchlist_count} watchlist items, {trade_count} trades, {portfolio_count} portfolios")
    else:
        logger.warning(f"Database file not found (will be created on first write): {DB_PATH}")

# Global model cache with thread-safe lock
MODEL_CACHE = {}
MODEL_CACHE_LOCK = threading.Lock()
PREDICTION_LOCK = threading.Lock()  # Lock for thread-safe predictions in parallel processing
REGIME_DETECTOR = RegimeDetector(method='gmm')

# ============================================================================
# YAHOO FINANCE RATE LIMITER (Prevents 401 errors during heavy model training)
# ============================================================================
YAHOO_RATE_LIMITER = None
YAHOO_RATE_LOCK = threading.Lock()
YAHOO_LAST_REQUEST_TIME = 0
YAHOO_MIN_REQUEST_INTERVAL = 0.5  # Minimum 0.5 seconds between requests

def get_yahoo_fetcher():
    """Get a rate-limited Yahoo Finance fetcher instance."""
    global YAHOO_RATE_LIMITER
    if YAHOO_RATE_LIMITER is None:
        if ENHANCED_DATA_LAYER:
            YAHOO_RATE_LIMITER = YahooFinanceRobust(
                max_retries=3,
                initial_delay=1.0,
                max_delay=15.0,
                use_cache=True
            )
            logger.info("[YAHOO RATE] Initialized robust Yahoo Finance fetcher with rate limiting")
        else:
            YAHOO_RATE_LIMITER = None
    return YAHOO_RATE_LIMITER

def get_yahoo_data_safe(ticker, period='1y', max_retries=3):
    """
    Rate-limited Yahoo Finance data fetcher.
    Prevents 401 errors by spacing requests and using caching.

    Args:
        ticker: Stock ticker symbol
        period: Data period (e.g., '1y', '60d', '5d')
        max_retries: Number of retry attempts

    Returns:
        DataFrame with OHLCV data or None on failure
    """
    import time
    global YAHOO_LAST_REQUEST_TIME

    with YAHOO_RATE_LOCK:
        # Rate limiting: ensure minimum interval between requests
        now = time.time()
        elapsed = now - YAHOO_LAST_REQUEST_TIME
        if elapsed < YAHOO_MIN_REQUEST_INTERVAL:
            sleep_time = YAHOO_MIN_REQUEST_INTERVAL - elapsed
            time.sleep(sleep_time)

        YAHOO_LAST_REQUEST_TIME = time.time()

    # Use robust fetcher if available
    fetcher = get_yahoo_fetcher()
    if fetcher is not None:
        try:
            df = fetcher.get_history(ticker, period=period, validate=False)
            if df is not None and not df.empty:
                logger.debug(f"[YAHOO RATE] Fetched {len(df)} rows for {ticker}")
                return df
        except Exception as e:
            logger.warning(f"[YAHOO RATE] Robust fetcher failed for {ticker}: {e}")

    # Fallback to direct yfinance with retry logic
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if df is not None and not df.empty:
                # FIX: Handle multi-index columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                    logger.debug(f"[YAHOO] Flattened multi-index columns for {ticker}")
                return df
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2, 4, 6 seconds
                logger.warning(f"[YAHOO RATE] Retry {attempt + 1}/{max_retries} for {ticker} in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"[YAHOO RATE] All retries failed for {ticker}: {e}")

    return None

# NEW: Dual model system - Enable China-specific model
USE_DUAL_MODEL_SYSTEM = True  # Set to False to use only US/Intl model for all markets
CHINA_MODEL_ROUTER = None  # Will be initialized on first use

# China Model Platform - Global factory instance
CHINA_MODEL_FACTORY = None
CHINA_PORTFOLIO_CONSTRUCTOR = None
CHINA_RISK_ANALYZER = None

def get_china_model_factory():
    """Get or initialize the China Model Factory with production models."""
    global CHINA_MODEL_FACTORY, CHINA_PORTFOLIO_CONSTRUCTOR, CHINA_RISK_ANALYZER

    if not CHINA_MODEL_AVAILABLE:
        logger.warning("[CHINA MODEL] Platform not available")
        return None, None, None

    if CHINA_MODEL_FACTORY is None:
        try:
            logger.info("[CHINA MODEL] Initializing Model Factory...")
            CHINA_MODEL_FACTORY = ModelFactory()

            # Try to load production models
            models_path = os.path.join(
                os.path.dirname(__file__),
                'china_model', 'models', 'production_models.pkl'
            )

            if os.path.exists(models_path):
                CHINA_MODEL_FACTORY.load_models(models_path)
                logger.info(f"[CHINA MODEL] Loaded {len(CHINA_MODEL_FACTORY.models)} production models")

                # Initialize portfolio constructor
                CHINA_PORTFOLIO_CONSTRUCTOR = PortfolioConstructor(model_factory=CHINA_MODEL_FACTORY)
                CHINA_RISK_ANALYZER = RiskAnalyzer(CHINA_PORTFOLIO_CONSTRUCTOR)
                logger.info("[CHINA MODEL] Portfolio constructor and risk analyzer initialized")
            else:
                logger.warning(f"[CHINA MODEL] No production models found at {models_path}")

        except Exception as e:
            logger.error(f"[CHINA MODEL] Initialization failed: {e}")
            CHINA_MODEL_FACTORY = None

    return CHINA_MODEL_FACTORY, CHINA_PORTFOLIO_CONSTRUCTOR, CHINA_RISK_ANALYZER


def get_model_router():
    """
    Get or initialize the model router for dual model system.

    Returns:
        ModelRouter instance if dual model system is enabled, None otherwise
    """
    global CHINA_MODEL_ROUTER

    if not USE_DUAL_MODEL_SYSTEM:
        return None

    if CHINA_MODEL_ROUTER is None:
        try:
            logger.info("[MODEL ROUTER] Initializing dual model system...")
            CHINA_MODEL_ROUTER = ModelRouter()
            logger.info("[MODEL ROUTER] Dual model system initialized successfully")
        except Exception as e:
            logger.error(f"[MODEL ROUTER] Failed to initialize: {e}")
            return None

    return CHINA_MODEL_ROUTER


# Initialize Optimal Hybrid Strategy
# PHASE 1 FIX: Lowered confidence threshold from 0.65 to 0.50 based on
# profitability testing (improves Sharpe from 2.77 to 3.33)
# See: docs/PHASE1_TEST_RESULTS.md and phase1 fixing on C model_extra.pdf
HYBRID_STRATEGY = OptimalHybridStrategy(
    confidence_threshold=0.50,  # Optimized threshold (was 0.65)
    volatility_filter_percentile=0.50,
    position_size=0.50,
    stop_loss_pct=0.05
)

# ENHANCEMENT #5: Initialize Sentiment Analyzer with caching
try:
    SENTIMENT_ENGINEER = SentimentFeatureEngineer(
        use_finbert=True,
        use_vader=True,
        cache_dir=".sentiment_cache"
    )
    logger.info("Sentiment analysis initialized with FinBERT and VADER (caching enabled)")
except Exception as e:
    logger.warning(f"Sentiment analyzer initialization failed (will use fallback): {e}")
    SENTIMENT_ENGINEER = None

# Model selection configuration (ENHANCEMENT #6)
AVAILABLE_MODELS = {
    'ensemble': 'EnhancedEnsemblePredictor',  # Old model (LGBM + XGB + Neural Nets)
    'hybrid': 'HybridLSTMCNNPredictor',       # New Hybrid LSTM/CNN with profit-maximizing loss
    'hybrid_ensemble': 'HybridEnsemblePredictor'  # Best: Combines old + new models
}
DEFAULT_MODEL = 'hybrid_ensemble'  # Use best model by default

# Transaction costs configuration (ENHANCEMENT #8)
TRANSACTION_COST = 0.001  # 0.1% per trade
SLIPPAGE = 0.0005  # 0.05%
TOTAL_TRADING_FRICTION = TRANSACTION_COST + SLIPPAGE  # 0.15% per round trip

# In-memory portfolio tracker (in production, use database)
USER_PORTFOLIOS = defaultdict(list)
USER_WATCHLISTS = defaultdict(list)

# Data persistence files
DATA_DIR = 'data'
PORTFOLIOS_FILE = os.path.join(DATA_DIR, 'portfolios.json')
WATCHLISTS_FILE = os.path.join(DATA_DIR, 'watchlists.json')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def save_portfolios():
    """Save portfolios to disk"""
    try:
        with open(PORTFOLIOS_FILE, 'w') as f:
            json.dump(dict(USER_PORTFOLIOS), f, indent=2)
        logger.debug(f"Portfolios saved to {PORTFOLIOS_FILE}")
    except Exception as e:
        logger.error(f"Error saving portfolios: {e}")

def load_portfolios():
    """Load portfolios from disk"""
    global USER_PORTFOLIOS
    try:
        if os.path.exists(PORTFOLIOS_FILE):
            with open(PORTFOLIOS_FILE, 'r') as f:
                data = json.load(f)
                USER_PORTFOLIOS = defaultdict(list, data)
            logger.info(f"Loaded portfolios for {len(USER_PORTFOLIOS)} users from disk")
        else:
            logger.info("No saved portfolios found, starting fresh")
    except Exception as e:
        logger.error(f"Error loading portfolios: {e}")
        USER_PORTFOLIOS = defaultdict(list)

def save_watchlists():
    """Save watchlists to disk"""
    try:
        with open(WATCHLISTS_FILE, 'w') as f:
            json.dump(dict(USER_WATCHLISTS), f, indent=2)
        logger.debug(f"Watchlists saved to {WATCHLISTS_FILE}")
    except Exception as e:
        logger.error(f"Error saving watchlists: {e}")

def load_watchlists():
    """Load watchlists from disk"""
    global USER_WATCHLISTS
    try:
        if os.path.exists(WATCHLISTS_FILE):
            with open(WATCHLISTS_FILE, 'r') as f:
                data = json.load(f)
                USER_WATCHLISTS = defaultdict(list, data)
            logger.info(f"Loaded watchlists for {len(USER_WATCHLISTS)} users from disk")
        else:
            logger.info("No saved watchlists found, starting fresh")
    except Exception as e:
        logger.error(f"Error loading watchlists: {e}")
        USER_WATCHLISTS = defaultdict(list)

# ============================================================================
# REAL-TIME ONLY - No hardcoded company database
# ============================================================================
# All ticker information is fetched from Yahoo Finance in real-time.
# This ensures we never give wrong/outdated information to users.
#
# Helper function to get ticker info from Yahoo Finance:
def get_ticker_info_from_yahoo(ticker):
    """
    Get ticker information from Yahoo Finance in real-time.
    Returns dict with 'name', 'type', 'exchange' or defaults if unavailable.
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info

        # Get company name
        name = info.get('longName') or info.get('shortName') or ticker

        # Determine asset type from ticker pattern
        if '-USD' in ticker:
            asset_type = 'Cryptocurrency'
            exchange = 'Crypto'
        elif '=F' in ticker:
            asset_type = 'Commodity'
            exchange = info.get('exchange', 'Futures')
        elif '=X' in ticker:
            asset_type = 'Forex'
            exchange = 'FX'
        elif ticker.endswith('.HK'):
            asset_type = 'Stock'
            exchange = 'HKEX'
        elif ticker.endswith('.SS'):
            asset_type = 'Stock'
            exchange = 'SSE'
        elif ticker.endswith('.SZ'):
            asset_type = 'Stock'
            exchange = 'SZSE'
        elif ticker.startswith('^'):
            asset_type = 'Index'
            exchange = 'Index'
        else:
            asset_type = info.get('quoteType', 'Stock')
            exchange = info.get('exchange', 'Unknown')

        return {
            'name': name,
            'type': asset_type,
            'exchange': exchange
        }
    except Exception as e:
        logger.warning(f"Could not fetch Yahoo info for {ticker}: {e}")
        return {
            'name': ticker,
            'type': 'Stock',
            'exchange': 'Unknown'
        }

# Empty COMPANY_DATABASE for backward compatibility with search functions
# All actual data comes from Yahoo Finance real-time
COMPANY_DATABASE = {}

# Search aliases - map common search terms to tickers
SEARCH_ALIASES = {
    # Company name aliases
    'APPLE': ['AAPL'],
    'MICROSOFT': ['MSFT'],
    'NVIDIA': ['NVDA'],
    'GOOGLE': ['GOOGL'],
    'ALPHABET': ['GOOGL'],
    'META': ['META'],
    'FACEBOOK': ['META'],
    'TESLA': ['TSLA'],
    'AMAZON': ['AMZN'],
    'NETFLIX': ['NFLX'],
    'AMD': ['AMD'],
    'INTEL': ['INTC'],
    'ALIBABA': ['BABA', '9988.HK'],
    'TENCENT': ['TCEHY', '0700.HK'],
    'JPMORGAN': ['JPM'],
    'BANK OF AMERICA': ['BAC'],
    'GOLDMAN': ['GS'],
    'WALMART': ['WMT'],
    'DISNEY': ['DIS'],
    'TRUMP': ['DJT', 'TRUMP-USD'],
    'GAMESTOP': ['GME'],
    'AMC': ['AMC'],

    # Commodities
    'IRON ORE': ['TIO=F', 'VALE', 'RIO', 'BHP'],
    'IRON': ['TIO=F', 'VALE', 'RIO', 'BHP', 'MT', 'CLF'],
    'STEEL': ['MT', 'CLF'],
    'COPPER': ['HG=F', 'SCCO'],
    'GOLD': ['GC=F'],
    'SILVER': ['SI=F'],
    'PLATINUM': ['PL=F'],
    'PALLADIUM': ['PA=F'],
    'OIL': ['CL=F', 'XOM', 'CVX'],
    'NATURAL GAS': ['NG=F'],
    'GAS': ['NG=F'],
    'WHEAT': ['ZW=F', 'WEAT'],
    'CORN': ['ZC=F', 'CORN'],
    'SOYBEAN': ['ZS=F', 'SOYB'],
    'COFFEE': ['KC=F'],
    'SUGAR': ['SB=F'],
    'COTTON': ['CT=F'],
    'COCOA': ['CC=F'],
    'COMMODITIES': ['GC=F', 'SI=F', 'HG=F', 'TIO=F', 'CL=F', 'NG=F', 'SOYB', 'WEAT', 'CORN'],
    'GRAINS': ['SOYB', 'WEAT', 'CORN', 'ZW=F', 'ZC=F', 'ZS=F'],

    # Crypto
    'BITCOIN': ['BTC-USD'],
    'BTC': ['BTC-USD'],
    'ETHEREUM': ['ETH-USD'],
    'ETH': ['ETH-USD'],
    'BINANCE': ['BNB-USD'],
    'BNB': ['BNB-USD'],
    'RIPPLE': ['XRP-USD'],
    'XRP': ['XRP-USD'],
    'CARDANO': ['ADA-USD'],
    'ADA': ['ADA-USD'],
    'SOLANA': ['SOL-USD'],
    'SOL': ['SOL-USD'],
    'DOGECOIN': ['DOGE-USD'],
    'DOGE': ['DOGE-USD'],
    'POLKADOT': ['DOT-USD'],
    'DOT': ['DOT-USD'],
    'POLYGON': ['MATIC-USD'],
    'MATIC': ['MATIC-USD'],
    'SHIBA': ['SHIB-USD'],
    'SHIB': ['SHIB-USD'],
    'AVALANCHE': ['AVAX-USD'],
    'AVAX': ['AVAX-USD'],
    'CHAINLINK': ['LINK-USD'],
    'LINK': ['LINK-USD'],
    'UNISWAP': ['UNI-USD'],
    'UNI': ['UNI-USD'],
    'LITECOIN': ['LTC-USD'],
    'LTC': ['LTC-USD'],
    'TRON': ['TRX-USD'],
    'TRX': ['TRX-USD'],
    'TRUMP COIN': ['TRUMP-USD'],
    'TRUMPUSD': ['TRUMP-USD'],

    # Forex
    'EURO': ['EURUSD=X'],
    'EUR': ['EURUSD=X'],
    'YEN': ['JPY=X'],
    'JPY': ['JPY=X'],
    'POUND': ['GBPUSD=X'],
    'GBP': ['GBPUSD=X'],
    'YUAN': ['CNY=X'],
    'CNY': ['CNY=X'],
    'DOLLAR': ['EURUSD=X', 'JPY=X', 'GBPUSD=X'],
    'AUD': ['AUDUSD=X'],
    'NZD': ['NZDUSD=X'],
    'CAD': ['USDCAD=X'],
    'CHF': ['USDCHF=X'],
    'HKD': ['HKD=X'],
    'SGD': ['SGD=X'],
    'INR': ['INR=X'],
    'KRW': ['KRW=X'],

    # === CHINA COMPANIES ===
    'MOUTAI': ['600519.SS'],
    'KWEICHOW': ['600519.SS'],
    'PING AN': ['601318.SS', '2318.HK'],
    'CMB': ['600036.SS'],
    'MERCHANTS BANK': ['600036.SS'],
    'ICBC': ['601398.SS', '1398.HK'],
    'SINOPEC': ['600028.SS'],
    'PETROCHINA': ['601857.SS'],
    'BYD': ['002594.SZ', '1211.HK'],
    'CATL': ['300750.SZ'],
    'MIDEA': ['000333.SZ'],
    'GREE': ['000651.SZ'],
    'WULIANGYE': ['000858.SZ'],
    'LUXSHARE': ['002475.SZ'],
    'HIKVISION': ['002415.SZ'],
    'BOE': ['000725.SZ'],
    'IFLYTEK': ['002230.SZ'],
    'XIAOMI': ['1810.HK'],
    'MEITUAN': ['3690.HK'],
    'JD': ['JD', '9618.HK'],
    'BAIDU': ['BIDU', '9888.HK'],
    'XPENG': ['XPEV', '9868.HK'],
    'NIO': ['NIO', '9866.HK'],
    'LI AUTO': ['LI', '2015.HK'],
    'KUAISHOU': ['1024.HK'],
    'AIA': ['1299.HK'],
    'CCB': ['0939.HK'],
    'CHINA MOBILE': ['0941.HK'],
    'CHINA CONSTRUCTION': ['0939.HK', '601668.SS'],
    'BANK OF CHINA': ['3988.HK'],
    'HKEX': ['0388.HK'],
    'HSBC': ['0005.HK', 'HSBA.L'],
    'CK HUTCHISON': ['0001.HK'],
    'SUN HUNG KAI': ['0016.HK'],

    # === SINGAPORE COMPANIES ===
    'DBS': ['D05.SI'],
    'OCBC': ['O39.SI'],
    'UOB': ['U11.SI'],
    'SINGTEL': ['Z74.SI'],
    'SINGAPORE AIRLINES': ['C6L.SI'],
    'SIA': ['C6L.SI'],
    'COMFORTDELGRO': ['C52.SI'],
    'WILMAR': ['F34.SI'],
    'SGX': ['S68.SI'],
    'VENTURE': ['V03.SI'],
    'SATS': ['S58.SI'],

    # === JAPAN COMPANIES ===
    'TOYOTA': ['TM', '7203.T'],
    'SONY': ['SONY', '6758.T'],
    'SOFTBANK': ['9984.T'],
    'KEYENCE': ['6861.T'],
    'MITSUBISHI UFJ': ['8306.T'],
    'MURATA': ['6981.T'],
    'NTT': ['9432.T'],
    'HONDA': ['7267.T'],
    'SHIN-ETSU': ['4063.T'],
    'TAKEDA': ['4502.T'],
    'TOKYO ELECTRON': ['8035.T'],
    'FAST RETAILING': ['9983.T'],
    'UNIQLO': ['9983.T'],
    'RECRUIT': ['6098.T'],
    'DAIKIN': ['6367.T'],
    'KDDI': ['9433.T'],

    # === SOUTH KOREA COMPANIES ===
    'SAMSUNG': ['005930.KS'],
    'SK HYNIX': ['000660.KS'],
    'LG ENERGY': ['373220.KS'],
    'SAMSUNG BIO': ['207940.KS'],
    'HYUNDAI': ['005380.KS'],
    'NAVER': ['035420.KS'],
    'KAKAO': ['035720.KS'],
    'LG CHEM': ['051910.KS'],
    'SAMSUNG SDI': ['006400.KS'],
    'KIA': ['000270.KS'],
    'CELLTRION': ['068270.KS'],
    'KB FINANCIAL': ['105560.KS'],
    'SHINHAN': ['055550.KS'],
    'HYUNDAI MOBIS': ['012330.KS'],

    # === INDIA COMPANIES ===
    'RELIANCE': ['RELIANCE.NS'],
    'TCS': ['TCS.NS'],
    'TATA': ['TCS.NS', 'TATASTEEL.NS'],
    'HDFC': ['HDFCBANK.NS'],
    'INFOSYS': ['INFY.NS'],
    'HINDUSTAN UNILEVER': ['HINDUNILVR.NS'],
    'ICICI': ['ICICIBANK.NS'],
    'BHARTI': ['BHARTIARTL.NS'],
    'AIRTEL': ['BHARTIARTL.NS'],
    'SBI': ['SBIN.NS'],
    'ITC': ['ITC.NS'],
    'LARSEN': ['LT.NS'],
    'WIPRO': ['WIPRO.NS'],
    'ADANI': ['ADANIENT.NS'],
    'ASIAN PAINTS': ['ASIANPAINT.NS'],
    'MARUTI': ['MARUTI.NS'],

    # === EUROPE COMPANIES ===
    'LVMH': ['MC.PA'],
    'LOREAL': ['OR.PA'],
    "L'OREAL": ['OR.PA'],
    'SANOFI': ['SAN.PA'],
    'TOTALENERGIES': ['TTE.PA'],
    'AIRBUS': ['AIR.PA'],
    'SCHNEIDER': ['SU.PA'],
    'SAP': ['SAP', 'SAP.DE'],
    'SIEMENS': ['SIE.DE'],
    'VOLKSWAGEN': ['VOW3.DE'],
    'MERCEDES': ['MBG.DE'],
    'BMW': ['BMW.DE'],
    'ASML': ['ASML', 'ASML.AS'],
    'NESTLE': ['NSRGY', 'NESN.SW'],
    'NOVARTIS': ['NOVN.SW'],
    'ROCHE': ['ROG.SW'],
    'SHELL': ['SHEL.L'],
    'ASTRAZENECA': ['AZN.L'],
    'UNILEVER': ['ULVR.L'],
    'BP': ['BP.L'],

    # === MARKET/EXCHANGE SEARCHES ===
    'CHINA': ['9988.HK', '0700.HK', 'BABA', 'TCEHY', 'FXI', 'MCHI'],
    'SHANGHAI': ['600519.SS', '601318.SS', '601398.SS'],
    'SHENZHEN': ['002594.SZ', '000333.SZ', '300750.SZ'],
    'HONG KONG': ['0700.HK', '9988.HK', '1810.HK'],
    'SINGAPORE': ['D05.SI', 'O39.SI', 'U11.SI'],
    'JAPAN': ['7203.T', '6758.T', '9984.T'],
    'KOREA': ['005930.KS', '000660.KS'],
    'INDIA': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS'],
    'EUROPE': ['MC.PA', 'SAP.DE', 'ASML.AS'],

    # === INDICES ===
    'S&P': ['^GSPC', 'SPY'],
    'S&P 500': ['^GSPC', 'SPY'],
    'DOW': ['^DJI', 'DIA'],
    'NASDAQ': ['^IXIC', 'QQQ'],
    'NIKKEI': ['^N225', 'NK=F'],
    'HANG SENG': ['^HSI'],
    'HSI': ['^HSI'],
    'SSE': ['000001.SS'],
    'SHANGHAI INDEX': ['000001.SS'],
    'STI': ['^STI'],
    'FTSE': ['^FTSE'],
    'DAX': ['^GDAXI'],
    'CAC': ['^FCHI'],
    'KOSPI': ['^KS11'],
    'NIFTY': ['^NSEI', 'IN=F'],
    'SENSEX': ['^BSESN'],

    # === FUTURES ===
    'CHINA A50': ['CN=F'],
    'A50': ['CN=F'],
    'NIKKEI FUTURES': ['NK=F'],
    'NIFTY FUTURES': ['IN=F'],

    # Hong Kong / China
    'HONG KONG': ['0700.HK', '9988.HK', '1299.HK', '0005.HK', '0388.HK'],
    'HK': ['0700.HK', '9988.HK', '1299.HK', '0005.HK', '0388.HK'],
    'HKEX': ['0388.HK'],
    'HSBC': ['0005.HK'],
    'AIA': ['1299.HK'],
    'MEITUAN': ['3690.HK'],
    'PING AN': ['2318.HK'],
    'CHINA': ['FXI', 'MCHI', 'BABA', 'PDD', 'JD', '0700.HK', '9988.HK', '1299.HK'],
    'CHINESE': ['FXI', 'MCHI', 'BABA', 'PDD', 'JD', '0700.HK', '9988.HK'],

    # Categories
    'TECH': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'TSLA', 'AMZN', 'NFLX', 'AMD', 'INTC'],
    'FAANG': ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL'],
    'BANKS': ['JPM', 'BAC', 'GS', '0005.HK', '0939.HK', '1398.HK', '3988.HK'],
    'MINING': ['VALE', 'RIO', 'BHP', 'SCCO', 'MT', 'CLF'],
    'ENERGY': ['XOM', 'CVX', 'CL=F'],
    'CONSUMER': ['WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'DIS', 'KO', 'PEP'],
    'MEME': ['GME', 'AMC', 'DJT', 'DOGE-USD'],
    'ETF': ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'FXI', 'MCHI', 'EEM'],
    'INDEX': ['SPY', 'QQQ', 'DIA', 'IWM'],
    'SP500': ['SPY'],
    'NASDAQ': ['QQQ'],
    'DOW': ['DIA'],
}


def get_or_train_model(ticker, interval='1d', model_type=None):
    """
    Get cached model or train new one for ticker.

    Args:
        ticker: Stock ticker symbol
        interval: Data interval (default: '1d')
        model_type: Model to use ('ensemble', 'hybrid', 'hybrid_ensemble', or None for default)

    Returns:
        dict with 'model', 'feature_cols', 'trained_at', 'model_type', 'asset_class'
    """
    # ENHANCEMENT #7: Get asset-specific hyperparameters
    asset_params = AssetHyperparameters.get_params(ticker)
    asset_class = asset_params['asset_class']

    # Use default model if not specified
    if model_type is None:
        model_type = DEFAULT_MODEL

    cache_key = f"{ticker}_{interval}_{model_type}"

    if cache_key in MODEL_CACHE:
        logger.info(f"Using cached {model_type} model for {ticker} ({asset_class})")
        # Record cache hit
        cache_monitor = get_cache_monitor()
        cache_monitor.record_cache_hit()
        return MODEL_CACHE[cache_key]

    logger.info(f"Training new {model_type} model for {ticker} ({asset_class})")

    # Start timing for cache miss
    import time
    training_start = time.time()

    try:
        # Fetch training data (increase to 1500 days to account for NaN drops)
        data = fetch_data(ticker, lookback_days=1500)
        # Reduced minimum from 200 to 50 to support recently listed stocks
        if len(data) < 50:
            raise ValueError(f"Insufficient data for {ticker} - need at least 50 days, got {len(data)}")

        # Engineer features (market-specific: US/Intl vs China)
        logger.info(f"Data shape before features: {data.shape}")
        data = engineer_market_specific_features(data, ticker)
        logger.info(f"Data shape after market-specific features: {data.shape}")

        # ENHANCEMENT #5: Add sentiment features if sentiment analyzer is available
        if SENTIMENT_ENGINEER is not None:
            try:
                logger.info("Adding sentiment features...")
                data = SENTIMENT_ENGINEER.add_real_sentiment_features(data, ticker)
                logger.info(f"Data shape after sentiment features: {data.shape}")
            except Exception as e:
                logger.warning(f"Sentiment feature engineering failed (continuing without): {e}")
        else:
            logger.info("Sentiment analyzer not available, skipping sentiment features")

        # Create target for return prediction model
        # Note: This model predicts 5-day forward returns (direction/magnitude)
        # Volatility is estimated separately using Yang-Zhang volatility from features
        data['target'] = data['Close'].pct_change(5).shift(-5)  # 5-day forward returns
        logger.info(f"Data shape before cleaning: {data.shape}")

        # For commodities/futures with gaps, be more aggressive with filling
        # Forward fill missing values first (for futures/commodities with gaps)
        data = data.ffill()
        # Then backward fill any remaining (for early rows)
        data = data.bfill()

        # For commodities with very sparse data, use linear interpolation as fallback
        if ticker.endswith('=F') or ticker in ['GC=F', 'CL=F', 'TIO=F', 'NG=F', 'SI=F']:
            # Check remaining NaN count
            nan_count = data.isnull().sum().sum()
            logger.info(f"NaN count before interpolation: {nan_count}")
            logger.info(f"Target column NaN count: {data['target'].isnull().sum()}")
            if nan_count > 0:
                # Interpolate numeric columns EXCEPT target (target has NaNs from shift which is expected)
                numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns if col != 'target']
                data[numeric_cols] = data[numeric_cols].interpolate(method='linear', limit_direction='both')
                logger.info(f"Applied linear interpolation for {ticker} on {len(numeric_cols)} columns")
                logger.info(f"NaN count after interpolation: {data.isnull().sum().sum()}")
                logger.info(f"Target column NaN count after interpolation: {data['target'].isnull().sum()}")

        # Drop rows with NaN in target column (from shift) and feature columns
        initial_len = len(data)
        # First, identify which rows have valid targets
        valid_target_mask = data['target'].notna()
        logger.info(f"Rows with valid target: {valid_target_mask.sum()} out of {len(data)}")

        # First drop rows where target is NaN (expected from shift)
        data = data.dropna(subset=['target'])
        logger.info(f"Data shape after dropping NaN targets: {data.shape}")

        # Then check for NaN in feature columns only (not OHLCV and not target)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'target']
        feature_cols_check = [col for col in data.columns if col not in exclude_cols]

        # For commodities and forex, be more lenient - fill any remaining NaN in features with column mean
        if ticker.endswith('=F') or ticker.endswith('=X') or ticker in ['GC=F', 'CL=F', 'TIO=F', 'NG=F', 'SI=F']:
            logger.info(f"Checking {len(feature_cols_check)} feature columns for NaN (commodities/forex)")
            for col in feature_cols_check:
                nan_in_col = data[col].isnull().sum()
                if nan_in_col > 0:
                    # Only fill numeric columns with mean; skip categorical columns
                    if data[col].dtype.name != 'category' and pd.api.types.is_numeric_dtype(data[col]):
                        data[col].fillna(data[col].mean(), inplace=True)
                        logger.info(f"Filled {nan_in_col} NaNs in {col} with mean")
                    else:
                        # For categorical or non-numeric, drop rows with NaN
                        data = data.dropna(subset=[col])
                        logger.info(f"Dropped {nan_in_col} rows due to NaN in categorical column {col}")
        else:
            # For regular stocks, drop rows with NaN in features
            data = data.dropna(subset=feature_cols_check)

        dropped_rows = initial_len - len(data)
        logger.info(f"Data shape after cleaning: {data.shape} (dropped {dropped_rows} rows)")

        # Log data quality metrics for monitoring
        data_monitor = get_data_quality_monitor()
        quality_metrics = {
            'original_rows': initial_len,
            'final_rows': len(data),
            'interpolation_percentage': (dropped_rows / initial_len * 100) if initial_len > 0 else 0,
            'retention_rate': (len(data) / initial_len * 100) if initial_len > 0 else 0,
            'used_fallback': False  # Will be set to True in fallback path if needed
        }
        data_monitor.log_data_quality(ticker, quality_metrics)

        # Check if we have enough data after dropping NaN (need at least 30 for training)
        if len(data) < 30:
            logger.warning(f"Insufficient data after feature engineering for {ticker} (got {len(data)} rows, need ≥30)")
            logger.warning(f"Attempting to fetch more historical data...")
            # Try fetching even more data
            data = fetch_data(ticker, lookback_days=3000)
            if len(data) < 200:
                raise ValueError(f"Insufficient raw data for {ticker} even with 3000 days lookback")

            # Re-engineer features
            data = tech_eng.add_all_features(data)
            data = vol_eng.add_all_features(data)
            data['target'] = data['Close'].pct_change(5).shift(-5)
            data = data.ffill().bfill()

            if ticker.endswith('=F') or ticker in ['GC=F', 'CL=F', 'TIO=F', 'NG=F', 'SI=F']:
                nan_count = data.isnull().sum().sum()
                logger.info(f"[Fallback] NaN count before interpolation: {nan_count}")
                logger.info(f"[Fallback] Target column NaN count: {data['target'].isnull().sum()}")
                numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns if col != 'target']
                data[numeric_cols] = data[numeric_cols].interpolate(method='linear', limit_direction='both')
                logger.info(f"[Fallback] NaN count after interpolation: {data.isnull().sum().sum()}")
                logger.info(f"[Fallback] Target column NaN count after interpolation: {data['target'].isnull().sum()}")

            # Clean data with same logic
            logger.info(f"[Fallback] Rows with valid target: {data['target'].notna().sum()} out of {len(data)}")
            data = data.dropna(subset=['target'])
            logger.info(f"[Fallback] Data shape after dropping NaN targets: {data.shape}")

            exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'target']
            feature_cols_check = [col for col in data.columns if col not in exclude_cols]
            if ticker.endswith('=F') or ticker.endswith('=X') or ticker in ['GC=F', 'CL=F', 'TIO=F', 'NG=F', 'SI=F']:
                logger.info(f"[Fallback] Checking {len(feature_cols_check)} feature columns for NaN (commodities/forex)")
                for col in feature_cols_check:
                    nan_in_col = data[col].isnull().sum()
                    if nan_in_col > 0:
                        # Only fill numeric columns with mean; skip categorical columns
                        if data[col].dtype.name != 'category' and pd.api.types.is_numeric_dtype(data[col]):
                            data[col].fillna(data[col].mean(), inplace=True)
                            logger.info(f"[Fallback] Filled {nan_in_col} NaNs in {col} with mean")
                        else:
                            # For categorical or non-numeric, drop rows with NaN
                            data = data.dropna(subset=[col])
                            logger.info(f"[Fallback] Dropped {nan_in_col} rows due to NaN in categorical column {col}")
            else:
                data = data.dropna(subset=feature_cols_check)

            if len(data) < 30:
                raise ValueError(f"Insufficient data after feature engineering for {ticker} (got {len(data)} rows, need ≥30)")

        # ENHANCED: Use clean_features_for_training for commodities/forex (prevents data loss)
        is_commodity_or_forex = ticker.endswith('=F') or ticker.endswith('=X')

        if is_commodity_or_forex:
            logger.info(f"[COMMODITY/FOREX] Applying enhanced data cleaning for {ticker}")
            data = clean_features_for_training(data, verbose=True)

        # Select features
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'target']
        feature_cols = [col for col in data.columns if col not in exclude_cols]

        X = data[feature_cols]
        y = data['target']

        # For stocks/crypto: Use simpler cleaning (remove infinite rows)
        if not is_commodity_or_forex:
            logger.info(f"Checking for infinite/extreme values in features...")
            # BUGFIX: Select only numeric columns before checking for infinity
            # This prevents TypeError with object/datetime columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                X_numeric = X[numeric_cols]
                inf_mask = np.isinf(X_numeric.values).any(axis=1)
                if inf_mask.sum() > 0:
                    logger.warning(f"Found {inf_mask.sum()} rows with infinite values, removing them")
                    X = X[~inf_mask]
                    y = y[~inf_mask]
            else:
                logger.warning("No numeric columns found in feature matrix")

        # Verify X is not empty and is 2D
        if X.empty or len(X.shape) != 2:
            raise ValueError(f"Invalid feature matrix shape for {ticker}: {X.shape}")

        if len(X) < 30:
            raise ValueError(f"Insufficient data after cleaning for {ticker} (got {len(X)} rows, need ≥30)")

        # Train model: 60% train, 40% validation (ensure enough val samples for lookback=20)
        # Use last 80% for training+validation (minimum 50 samples to allow for lookback)
        train_val_size = max(50, min(150, int(len(X) * 0.8)))
        train_size = int(train_val_size * 0.6)  # 60% for train, 40% for validation

        X_train_val = X.tail(train_val_size).copy().reset_index(drop=True)
        y_train_val = y.tail(train_val_size).copy().reset_index(drop=True)

        # Split into train and validation (ensure val has at least 40 samples for lookback=20)
        X_train = X_train_val[:train_size].copy()
        y_train = y_train_val[:train_size].copy()
        X_val = X_train_val[train_size:].copy()
        y_val = y_train_val[train_size:].copy()

        # Final check
        if len(X_train) == 0 or X_train.empty:
            raise ValueError(f"Empty training data for {ticker}")

        # Debug logging
        logger.info(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
        logger.info(f"Validation data shape: X_val={X_val.shape}, y_val={y_val.shape}")
        logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols[:5]}...")

        # ENHANCEMENTS #6 & #7: Create and train model based on type and asset-specific hyperparameters
        if model_type == 'ensemble':
            # Old model: EnhancedEnsemblePredictor
            logger.info("Training EnhancedEnsemblePredictor (LGBM + XGB + Neural Nets)")
            model = EnhancedEnsemblePredictor(use_prediction_market=True)
            model.train_all_models(
                X_train, y_train, X_val, y_val,
                models_to_train=['lightgbm', 'xgboost'],
                neural_models=['tcn', 'lstm', 'transformer']
            )

        elif model_type == 'hybrid':
            # New model: Hybrid LSTM/CNN with profit-maximizing loss
            logger.info(f"Training HybridLSTMCNNPredictor with asset-specific hyperparameters ({asset_class})")
            model = HybridLSTMCNNPredictor(
                lookback=asset_params.get('lookback', 20),
                cnn_channels=asset_params.get('cnn_channels', [32, 64, 32]),
                kernel_sizes=asset_params.get('kernel_sizes', [3, 5, 7]),
                lstm_hidden_size=asset_params.get('lstm_hidden_size', 64),
                lstm_num_layers=asset_params.get('lstm_num_layers', 2),
                dropout=asset_params.get('dropout', 0.3),
                learning_rate=asset_params.get('learning_rate', 0.001),
                batch_size=asset_params.get('batch_size', 32),
                epochs=asset_params.get('epochs', 100),
                device='cpu',
                use_profit_loss=True,  # ENHANCEMENT #1: Profit-maximizing loss
                random_state=42
            )
            # Convert to numpy for Hybrid LSTM/CNN
            X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
            y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
            X_val_np = X_val.values if hasattr(X_val, 'values') else X_val
            y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
            model.fit(X_train_np, y_train_np, X_val_np, y_val_np)

        elif model_type == 'hybrid_ensemble':
            # Best model: Combines old + new models
            logger.info(f"Training HybridEnsemblePredictor (combines old + new with asset-specific params for {asset_class})")
            model = HybridEnsemblePredictor(
                old_model_weight=None,  # Auto-determined from validation performance
                hybrid_epochs=asset_params.get('epochs', 100),
                hybrid_lookback=asset_params.get('lookback', 20),
                hybrid_cnn_channels=asset_params.get('cnn_channels', [32, 64, 32]),
                hybrid_kernel_sizes=asset_params.get('kernel_sizes', [3, 5, 7]),
                hybrid_lstm_hidden_size=asset_params.get('lstm_hidden_size', 64),
                hybrid_lstm_num_layers=asset_params.get('lstm_num_layers', 2),
                hybrid_dropout=asset_params.get('dropout', 0.3),
                hybrid_learning_rate=asset_params.get('learning_rate', 0.001),
                hybrid_batch_size=asset_params.get('batch_size', 32),
                use_profit_loss=True,
                random_state=42
            )
            # HybridEnsemblePredictor handles DataFrame/numpy conversion internally
            # Pass proper validation split for ensemble weight determination
            model.fit(X_train, y_train, X_val, y_val)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Cache model
        MODEL_CACHE[cache_key] = {
            'model': model,
            'feature_cols': feature_cols,
            'trained_at': datetime.now(),
            'model_type': model_type,
            'asset_class': asset_class,
            'asset_params': asset_params
        }

        # Record cache miss with training time
        training_time = time.time() - training_start
        cache_monitor = get_cache_monitor()
        cache_monitor.record_cache_miss(training_time)

        return MODEL_CACHE[cache_key]

    except Exception as e:
        logger.error(f"Error training model for {ticker}: {str(e)}")
        raise


def generate_news_feed(ticker, company_info, current_price, prediction_data):
    """Generate realistic news and social media feed for a ticker with NLP sentiment analysis."""
    news_items = []

    # Try to load sentiment analyzer (optional - graceful degradation if not available)
    try:
        from src.nlp.sentiment_analyzer import get_sentiment_analyzer
        sentiment_analyzer = get_sentiment_analyzer()
        logger.info("Sentiment analyzer loaded successfully")
    except Exception as e:
        logger.warning(f"Sentiment analyzer not available: {e}")
        sentiment_analyzer = None

    # Determine sentiment based on prediction
    direction = prediction_data['prediction']['direction']
    volatility = prediction_data['prediction']['volatility']

    # Safety check for None volatility (basic analysis path)
    if volatility is None:
        volatility = 0.01

    # News templates based on asset type and prediction
    if company_info['type'] == 'Stock':
        if direction > 0:
            news_items.append({
                'source': 'Bloomberg',
                'icon': '📰',
                'time': f'{np.random.randint(5, 120)} min ago',
                'headline': f'{company_info["name"]} shares rise on strong earnings outlook',
                'url': f'https://www.bloomberg.com/quote/{ticker}:US',
                'sentiment': 'positive',
                'relevance': 0.85
            })
            news_items.append({
                'source': 'Twitter',
                'icon': '🐦',
                'time': f'{np.random.randint(1, 30)} min ago',
                'headline': f'Analysts upgrade {ticker} to Buy with ${current_price * 1.15:.0f} price target',
                'url': f'https://twitter.com/search?q=${ticker}',
                'sentiment': 'positive',
                'relevance': 0.75
            })
        else:
            news_items.append({
                'source': 'Reuters',
                'icon': '📰',
                'time': f'{np.random.randint(10, 90)} min ago',
                'headline': f'{company_info["name"]} faces headwinds amid market concerns',
                'url': f'https://www.reuters.com/markets/companies/{ticker}.O',
                'sentiment': 'negative',
                'relevance': 0.80
            })
            news_items.append({
                'source': 'Reddit',
                'icon': '🔴',
                'time': f'{np.random.randint(5, 45)} min ago',
                'headline': f'r/stocks: {ticker} showing weakness, technical breakdown likely',
                'url': f'https://www.reddit.com/r/stocks/search?q={ticker}',
                'sentiment': 'negative',
                'relevance': 0.65
            })

    elif company_info['type'] == 'Cryptocurrency':
        news_items.append({
            'source': 'CoinDesk',
            'icon': '₿',
            'time': f'{np.random.randint(2, 60)} min ago',
            'headline': f'{ticker}: Whale movement detected - ${np.random.randint(50, 500)}M transferred',
            'url': f'https://www.coindesk.com/price/{ticker.replace("-USD", "").lower()}/',
            'sentiment': 'neutral',
            'relevance': 0.90
        })
        news_items.append({
            'source': 'Twitter',
            'icon': '🐦',
            'time': f'{np.random.randint(1, 20)} min ago',
            'headline': f'Crypto analyst: {ticker} {"breaking resistance" if direction > 0 else "testing support"}',
            'url': f'https://twitter.com/search?q=${ticker}',
            'sentiment': 'positive' if direction > 0 else 'negative',
            'relevance': 0.70
        })

    elif company_info['type'] == 'Forex':
        # Forex-specific news from multiple sources
        news_items.append({
            'source': 'ForexLive',
            'icon': '💱',
            'time': f'{np.random.randint(5, 30)} min ago',
            'headline': f'{ticker}: Central bank comments drive {"strength" if direction > 0 else "weakness"}',
            'url': f'https://www.forexlive.com/',
            'sentiment': 'positive' if direction > 0 else 'negative',
            'relevance': 0.85
        })
        news_items.append({
            'source': 'FXStreet',
            'icon': '📊',
            'time': f'{np.random.randint(30, 120)} min ago',
            'headline': f'{company_info["name"]} {"breaks resistance" if direction > 0 else "tests support"} - Technical analysis',
            'url': f'https://www.fxstreet.com/',
            'sentiment': 'neutral',
            'relevance': 0.80
        })
        news_items.append({
            'source': 'DailyFX',
            'icon': '💹',
            'time': f'{np.random.randint(60, 180)} min ago',
            'headline': f'Economic data {"boosts" if direction > 0 else "weighs on"} {ticker} amid volatility',
            'url': f'https://www.dailyfx.com/',
            'sentiment': 'positive' if direction > 0 else 'negative',
            'relevance': 0.75
        })

    elif company_info['type'] == 'Commodity':
        news_items.append({
            'source': 'Kitco',
            'icon': '💰',
            'time': f'{np.random.randint(10, 90)} min ago',
            'headline': f'{company_info["name"]} prices {"surge" if direction > 0 else "decline"} on supply concerns',
            'url': f'https://www.kitco.com/news/',
            'sentiment': 'positive' if direction > 0 else 'negative',
            'relevance': 0.85
        })

    # Add volatility-based news
    if volatility > 0.03:
        news_items.append({
            'source': 'MarketWatch',
            'icon': '⚠️',
            'time': f'{np.random.randint(10, 60)} min ago',
            'headline': f'{ticker} volatility spikes - Options traders brace for big moves',
            'url': f'https://www.marketwatch.com/investing/stock/{ticker}',
            'sentiment': 'neutral',
            'relevance': 0.75
        })

    # Add general market news
    news_items.append({
        'source': 'CNBC',
        'icon': '📺',
        'time': f'{np.random.randint(30, 180)} min ago',
        'headline': f'Market update: {company_info["exchange"]} trading {"higher" if np.random.random() > 0.5 else "mixed"} in afternoon session',
        'url': 'https://www.cnbc.com/markets/',
        'sentiment': 'neutral',
        'relevance': 0.50
    })

    # === Apply real NLP sentiment analysis ===
    if sentiment_analyzer and news_items:
        try:
            # Extract headlines for batch processing
            headlines = [item['headline'] for item in news_items]

            # Analyze all headlines at once (faster than one-by-one)
            sentiments = sentiment_analyzer.analyze_batch(headlines)

            # Update news items with real sentiment scores
            for item, sentiment_result in zip(news_items, sentiments):
                # Update sentiment from NLP (may differ from initial template sentiment)
                item['sentiment'] = sentiment_result['sentiment']
                # Add confidence score for frontend display
                item['sentiment_score'] = sentiment_result['score']
                # Add full breakdown for advanced analysis (optional)
                item['sentiment_breakdown'] = sentiment_result['scores']

            logger.info(f"Successfully analyzed {len(headlines)} news items with NLP")

        except Exception as e:
            logger.error(f"Failed to analyze sentiment: {e}")
            # Keep original template sentiments if NLP fails

    # Shuffle and return top 5
    np.random.shuffle(news_items)
    return news_items[:5]


def normalize_ticker(ticker):
    """Normalize ticker format for different exchanges."""
    # Hong Kong stocks: Yahoo Finance requires 4-digit codes with leading zeros
    # e.g., 0700.HK for Tencent (NOT 700.HK which fails)
    if '.HK' in ticker.upper():
        parts = ticker.split('.')
        stock_code = parts[0]
        # Pad to 4 digits with leading zeros if needed (e.g., 700 -> 0700)
        if stock_code.isdigit() and len(stock_code) < 4:
            stock_code = stock_code.zfill(4)
        return f"{stock_code}.HK"

    # Handle 5-digit HK stock codes like 09927, 03750
    # These are HK stocks where the first digit is 0 (leading zero)
    # Convert 09927 -> 9927.HK, 03750 -> 3750.HK
    if ticker.isdigit() and len(ticker) == 5 and ticker.startswith('0'):
        # Strip leading zero and add .HK suffix
        stock_code = ticker.lstrip('0')
        # Ensure we have at least 4 digits (pad if needed)
        if len(stock_code) < 4:
            stock_code = stock_code.zfill(4)
        return f"{stock_code}.HK"

    return ticker


def get_data_tier(days_available):
    """
    Determine analysis tier based on available trading days.
    Returns tier info for tiered ML analysis of short-listed stocks.
    """
    if days_available < 30:
        return {
            'tier': 'insufficient',
            'level': 0,
            'analysis_type': 'basic_info_only',
            'message': 'Very new listing - insufficient data for any analysis',
            'confidence': 'none',
            'features_available': ['price', 'volume'],
            'ml_available': False
        }
    elif days_available < 100:
        return {
            'tier': 'minimal',
            'level': 1,
            'analysis_type': 'basic_technical',
            'message': f'Limited history ({days_available} days) - using basic technical indicators',
            'confidence': 'low',
            'features_available': ['price_trend', 'volume_analysis', 'basic_momentum'],
            'ml_available': False
        }
    elif days_available < 250:
        return {
            'tier': 'light',
            'level': 2,
            'analysis_type': 'light_ml',
            'message': f'Moderate history ({days_available} days) - light ML analysis with reduced features',
            'confidence': 'medium',
            'features_available': ['RSI', 'moving_averages', 'volatility', 'momentum'],
            'ml_available': True
        }
    else:
        return {
            'tier': 'full',
            'level': 3,
            'analysis_type': 'full_ml',
            'message': f'Sufficient history ({days_available} days) - full ML analysis available',
            'confidence': 'high',
            'features_available': ['all'],
            'ml_available': True
        }


def is_china_ticker(ticker):
    """
    Check if a ticker is a China/Hong Kong listed stock.

    Returns True for:
    - .HK tickers (Hong Kong Exchange)
    - .SS tickers (Shanghai Stock Exchange)
    - .SZ tickers (Shenzhen Stock Exchange)
    """
    if not ticker:
        return False
    ticker_upper = ticker.upper()
    return ticker_upper.endswith('.HK') or ticker_upper.endswith('.SS') or ticker_upper.endswith('.SZ')


def get_quality_filter_checks(ticker, price_data=None):
    """
    Get quality filter check results for China Model FIXING3 display.

    FIXING3 Updates:
    - Relaxed thresholds: EPS > -0.1, D/E < 300%, MCap > $500M
    - Momentum override: 5-day > 10% overrides quality
    - Composite quality scoring
    - IPO tier detection and display

    Returns EPS, D/E ratio, Market Cap, 2-Day Momentum, Volatility, and IPO tier checks.
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info

        # Get fundamentals
        eps = info.get('trailingEps', None)
        de_ratio = info.get('debtToEquity', None)  # Yahoo returns as percentage (e.g., 29.7 for 29.7%)
        market_cap = info.get('marketCap', None)

        # Format market cap
        if market_cap:
            if market_cap >= 1e12:
                mcap_str = f"{market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                mcap_str = f"{market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                mcap_str = f"{market_cap/1e6:.2f}M"
            else:
                mcap_str = f"{market_cap:,.0f}"
        else:
            mcap_str = "N/A"

        # Calculate momentum and volatility from price data
        two_day_momentum = None
        five_day_momentum = None
        volatility = None
        trading_days = 0

        if price_data is not None and len(price_data) >= 1:
            trading_days = len(price_data)
            try:
                if len(price_data) >= 3:
                    two_day_momentum = ((price_data['Close'].iloc[-1] / price_data['Close'].iloc[-3]) - 1) * 100
                if len(price_data) >= 6:
                    five_day_momentum = ((price_data['Close'].iloc[-1] / price_data['Close'].iloc[-6]) - 1) * 100
                returns = price_data['Close'].pct_change().dropna() * 100
                volatility = returns.std() if len(returns) > 0 else None
            except Exception:
                pass

        # FIXING3: IPO Tier Detection
        ipo_tier = None
        ipo_tier_desc = None
        is_new_stock = trading_days < 60

        if trading_days <= 4:
            ipo_tier = 'NO_TRADE'
            ipo_tier_desc = 'Too new (< 5 days) - Not tradeable'
        elif trading_days <= 10:
            ipo_tier = 'BASIC'
            ipo_tier_desc = f'{trading_days} days - Momentum-only analysis (25% position)'
        elif trading_days <= 30:
            ipo_tier = 'ENHANCED'
            ipo_tier_desc = f'{trading_days} days - Enhanced analysis (50% position)'
        elif trading_days <= 60:
            ipo_tier = 'HYBRID'
            ipo_tier_desc = f'{trading_days} days - Hybrid analysis (75% position)'
        else:
            ipo_tier = 'FULL'
            ipo_tier_desc = f'{trading_days} days - Full analysis'

        # FIXING3: Momentum override check
        momentum_override = False
        if five_day_momentum is not None and five_day_momentum >= 10.0:
            momentum_override = True
        elif two_day_momentum is not None and two_day_momentum >= 5.0:
            momentum_override = True

        # FIXING3: Relaxed thresholds
        # EPS > -0.1 (was > 0)
        eps_pass = eps is None or eps > -0.1
        eps_value = f"{eps:.2f}" if eps is not None else "N/A"

        # D/E < 300% (was < 100%)
        de_pass = de_ratio is None or de_ratio < 300
        de_value = f"{de_ratio:.1f}%" if de_ratio is not None else "N/A"

        # Market Cap > $500M (was > $1B)
        mcap_pass = market_cap is None or market_cap > 5e8

        # FIXING3: 2-Day Momentum > 0.5% (was > 1%)
        mom_pass = two_day_momentum is None or two_day_momentum > 0.5
        mom_value = f"{two_day_momentum:+.2f}%" if two_day_momentum is not None else "N/A"

        # 5-Day Momentum (new for FIXING3)
        mom5_value = f"{five_day_momentum:+.2f}%" if five_day_momentum is not None else "N/A"

        # Volatility (just for display, no pass/fail)
        vol_value = f"{volatility:.2f}%" if volatility is not None else "N/A"
        vol_level = "High" if volatility and volatility > 3 else "Normal" if volatility else "N/A"

        # FIXING3: Calculate composite quality score (0-1)
        quality_score = 0.0
        if eps is not None:
            if eps > 0.1:
                quality_score += 0.25
            elif eps > 0:
                quality_score += 0.15
            elif eps > -0.1:
                quality_score += 0.05
        else:
            quality_score += 0.1  # Unknown = neutral

        if de_ratio is not None:
            if de_ratio < 50:
                quality_score += 0.25
            elif de_ratio < 100:
                quality_score += 0.20
            elif de_ratio < 200:
                quality_score += 0.15
            elif de_ratio < 300:
                quality_score += 0.10
        else:
            quality_score += 0.15  # Unknown = slightly favorable

        if market_cap is not None:
            if market_cap > 5e9:
                quality_score += 0.25
            elif market_cap > 1e9:
                quality_score += 0.20
            elif market_cap > 5e8:
                quality_score += 0.15
        else:
            quality_score += 0.1  # Unknown = neutral

        if two_day_momentum is not None and two_day_momentum > 0:
            quality_score += min(0.25, two_day_momentum / 10.0 * 0.25)

        # Convert numpy booleans to Python booleans for JSON serialization
        eps_pass = bool(eps_pass)
        de_pass = bool(de_pass)
        mcap_pass = bool(mcap_pass)
        mom_pass = bool(mom_pass)

        # FIXING3: Overall pass uses composite score + momentum override
        overall_pass = quality_score >= 0.3 or momentum_override
        if is_new_stock and ipo_tier == 'NO_TRADE':
            overall_pass = False

        return {
            'eps': {
                'value': eps_value,
                'threshold': '> -0.1',  # FIXING3: Relaxed
                'pass': eps_pass,
                'result': 'PASS' if eps_pass else 'FAIL'
            },
            'debt_equity': {
                'value': de_value,
                'threshold': '< 300%',  # FIXING3: Relaxed from 100%
                'pass': de_pass,
                'result': 'PASS' if de_pass else 'FAIL'
            },
            'market_cap': {
                'value': mcap_str,
                'threshold': '> 500M',  # FIXING3: Relaxed from 1B
                'pass': mcap_pass,
                'result': 'PASS' if mcap_pass else 'FAIL'
            },
            'two_day_momentum': {
                'value': mom_value,
                'threshold': '> 0.5%',  # FIXING3: Relaxed from 1%
                'pass': mom_pass,
                'result': 'PASS' if mom_pass else 'FAIL'
            },
            'five_day_momentum': {
                'value': mom5_value,
                'threshold': '> 10% (override)',
                'pass': bool(five_day_momentum and five_day_momentum >= 10.0),
                'result': 'OVERRIDE' if momentum_override else 'N/A'
            },
            'volatility': {
                'value': vol_value,
                'threshold': '-',
                'pass': True,
                'result': vol_level
            },
            'ipo_status': {
                'is_new_stock': is_new_stock,
                'trading_days': trading_days,
                'tier': ipo_tier,
                'tier_description': ipo_tier_desc,
                'can_trade': ipo_tier != 'NO_TRADE'
            },
            'quality_score': round(quality_score, 2),
            'momentum_override': momentum_override,
            'overall_pass': bool(overall_pass)
        }
    except Exception as e:
        logger.warning(f"[QUALITY FILTER] Could not get quality filter data for {ticker}: {e}")
        return None


def calculate_standardized_confidence(predicted_return, historical_data, analysis_type='basic',
                                       asset_class='stock', ticker=None):
    """
    Standardized confidence calculation for both basic and ML analysis.

    Uses Signal-to-Noise Ratio (SNR) consistently across all analysis types.
    This fixes Issue #1 from china model fixing suggestion.pdf:
    - Basic analysis was always 0.3 confidence regardless of signal strength
    - Now uses same SNR-based formula as ML analysis, with data quality adjustment

    Args:
        predicted_return: Expected return (in decimal, e.g., 0.05 for 5%)
        historical_data: DataFrame with price data for volatility calculation
        analysis_type: 'basic' or 'ml' - basic gets data quality penalty
        asset_class: 'stock', 'china_stock', 'crypto', 'forex', 'commodity'
        ticker: Optional ticker for logging

    Returns:
        float: Confidence score between 0.3 and 0.95
    """
    import math

    # Asset-specific SNR thresholds (same as ML analysis)
    SNR_THRESHOLDS = {
        'stock': 0.5,
        'china_stock': 0.5,
        'crypto': 0.8,
        'forex': 0.3,
        'commodity': 0.5,
    }

    # Calculate historical volatility (20-day annualized)
    if len(historical_data) >= 20:
        returns = historical_data['Close'].pct_change().dropna()
        if len(returns) >= 20:
            historical_vol = returns.tail(20).std() * np.sqrt(252)
        elif len(returns) > 0:
            historical_vol = returns.std() * np.sqrt(252)
        else:
            historical_vol = 0.30  # Default 30% annual volatility
    elif len(historical_data) >= 5:
        returns = historical_data['Close'].pct_change().dropna()
        historical_vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.30
    else:
        historical_vol = 0.30  # Default for very limited data

    # Ensure minimum volatility to avoid division issues
    historical_vol = max(historical_vol, 0.01)

    # Calculate Signal-to-Noise Ratio
    # For basic analysis, use momentum magnitude as signal strength
    signal_strength = abs(predicted_return)
    snr = signal_strength / historical_vol

    # Get asset-specific threshold
    threshold = SNR_THRESHOLDS.get(asset_class, 0.5)

    # Sigmoid transformation (same formula as ML analysis)
    # SNR at threshold -> 50% confidence (0.3 + 0.325 = 0.625)
    # Higher SNR -> higher confidence, capped at 0.95
    base_confidence = 0.3 + 0.65 * (1 / (1 + math.exp(-1.5 * (snr - threshold))))

    # Adjust for data quality (basic analysis penalty)
    if analysis_type == 'basic':
        # Penalize confidence based on amount of data available
        # 100 days = full confidence multiplier, less data = lower multiplier
        data_quality_factor = min(len(historical_data) / 100, 1.0)
        # Scale confidence reduction: at 50 days, factor = 0.5, so confidence reduced by 35%
        final_confidence = 0.3 + (base_confidence - 0.3) * data_quality_factor
        # Cap basic analysis at 0.70 (never as confident as full ML)
        final_confidence = min(final_confidence, 0.70)
    else:
        final_confidence = base_confidence

    # Ensure bounds
    final_confidence = min(max(final_confidence, 0.3), 0.95)

    return final_confidence


def basic_technical_analysis(data, ticker):
    """
    Perform basic technical analysis for stocks with limited history.
    Used when ML models cannot be reliably trained.

    UPDATED: Now uses standardized confidence calculation (Issue #1 fix)
    """
    if len(data) < 5:
        return {
            'signal': 'INSUFFICIENT_DATA',
            'analysis_type': 'none',
            'message': 'Not enough data for any analysis'
        }

    # Calculate basic metrics
    current_price = float(data['Close'].iloc[-1])

    # Simple momentum (5-day return)
    if len(data) >= 5:
        momentum_5d = (current_price / float(data['Close'].iloc[-5]) - 1) * 100
    else:
        momentum_5d = 0

    # Volume trend
    if len(data) >= 10 and 'Volume' in data.columns:
        recent_vol = data['Volume'].iloc[-5:].mean()
        older_vol = data['Volume'].iloc[-10:-5].mean()
        vol_trend = 'increasing' if recent_vol > older_vol * 1.1 else ('decreasing' if recent_vol < older_vol * 0.9 else 'stable')
    else:
        vol_trend = 'unknown'

    # Basic trend direction
    if len(data) >= 20:
        sma_20 = data['Close'].rolling(20).mean().iloc[-1]
        trend = 'bullish' if current_price > sma_20 else 'bearish'
    elif len(data) >= 10:
        sma_10 = data['Close'].rolling(10).mean().iloc[-1]
        trend = 'bullish' if current_price > sma_10 else 'bearish'
    else:
        trend = 'unknown'

    # Simple signal based on momentum
    if momentum_5d > 3:
        signal = 'BULLISH'
        direction = 1
    elif momentum_5d < -3:
        signal = 'BEARISH'
        direction = -1
    else:
        signal = 'NEUTRAL'
        direction = 0

    # Determine asset class for confidence calculation
    asset_class = 'china_stock' if is_china_ticker(ticker) else 'stock'

    # Calculate standardized confidence (Issue #1 fix)
    # Convert momentum_5d from percentage to decimal for consistency
    predicted_return = momentum_5d / 100  # e.g., 5% -> 0.05
    confidence = calculate_standardized_confidence(
        predicted_return=predicted_return,
        historical_data=data,
        analysis_type='basic',
        asset_class=asset_class,
        ticker=ticker
    )

    return {
        'signal': signal,
        'direction': direction,
        'analysis_type': 'basic_technical',
        'current_price': current_price,
        'momentum_5d': round(momentum_5d, 2),
        'volume_trend': vol_trend,
        'trend': trend,
        'confidence': round(confidence, 3),  # Dynamic confidence based on SNR
        'message': 'Basic technical analysis (limited history)',
        'recommendation': 'Monitor for more trading history before making significant positions'
    }


def fetch_data(ticker, lookback_days=1500):
    """Fetch historical data for ticker."""
    try:
        # Normalize ticker format first
        normalized_ticker = normalize_ticker(ticker)
        if normalized_ticker != ticker:
            logger.info(f"Normalized ticker: {ticker} -> {normalized_ticker}")
            ticker = normalized_ticker

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        yf_ticker = yf.Ticker(ticker)
        data = yf_ticker.history(start=start_date, end=end_date, interval='1d')

        if len(data) == 0:
            raise ValueError(f"No data available for {ticker}")

        # FIX 1: Handle multi-index columns from yfinance
        # yfinance sometimes returns multi-level column indices, especially for single tickers
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
            logger.info(f"Flattened multi-index columns for {ticker}")

        # FIX 3: Extended volume handling for futures, forex, and commodities
        # For commodities/futures/forex with zero volume, fill with 1.0 to avoid division errors
        if 'Volume' in data.columns:
            if ('=F' in ticker or '=X' in ticker) and (data['Volume'] == 0).all():
                data['Volume'] = 1.0
                logger.info(f"Filled zero volume for {ticker} (futures/forex)")
            # Also handle commodities that may have intermittent zero volume
            elif data['Volume'].eq(0).sum() > len(data) * 0.1:  # More than 10% zero volume
                data.loc[data['Volume'] == 0, 'Volume'] = 1.0
                logger.info(f"Filled intermittent zero volume for {ticker} (commodity)")

        # Log data availability for newly listed stocks
        if len(data) < 252:  # Less than 1 year
            logger.info(f"Recently listed stock {ticker}: {len(data)} days of data available")

        return data

    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        raise


def generate_prediction(ticker, account_size=100000, model_type=None):
    """
    Generate complete ML prediction and trading signal.

    Args:
        ticker: Stock ticker symbol
        account_size: Account size for position sizing (default: $100,000)
        model_type: Model to use ('ensemble', 'hybrid', 'hybrid_ensemble', or None for default)

    Returns:
        Prediction response with all enhancements integrated
    """
    try:
        # TIERED ANALYSIS: First check data availability for short-listed stocks
        try:
            test_data = fetch_data(ticker, lookback_days=1500)
            days_available = len(test_data)
            data_tier = get_data_tier(days_available)
            logger.info(f"[DATA TIER] {ticker}: {days_available} days available, tier={data_tier['tier']}")

            # For stocks with insufficient ML data, return basic analysis
            if not data_tier['ml_available']:
                logger.info(f"[BASIC ANALYSIS] Using basic technical analysis for {ticker} (tier: {data_tier['tier']})")
                basic_result = basic_technical_analysis(test_data, ticker)

                # Get company info from database or Yahoo Finance
                company_info = COMPANY_DATABASE.get(ticker)
                if not company_info:
                    # Try to fetch company metadata from Yahoo Finance
                    try:
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        company_name = info.get('shortName') or info.get('longName') or ticker

                        # Determine asset type
                        if '-USD' in ticker:
                            company_type = 'Cryptocurrency'
                        elif '=X' in ticker:
                            company_type = 'Forex'
                        elif '=F' in ticker or ticker in ['GC=F', 'SI=F', 'CL=F', 'NG=F', 'ZC=F', 'ZW=F', 'ZS=F']:
                            company_type = 'Commodity'
                        else:
                            company_type = 'Stock'

                        company_info = {
                            'name': company_name,
                            'type': company_type,
                            'exchange': info.get('exchange', 'Unknown'),
                            'sector': info.get('sector', 'N/A'),
                            'industry': info.get('industry', 'N/A'),
                        }
                    except Exception as e:
                        logger.warning(f"Could not fetch company info for {ticker}: {str(e)}")
                        company_info = {
                            'name': ticker,
                            'type': 'Unknown',
                            'exchange': 'Unknown',
                            'sector': 'N/A',
                            'industry': 'N/A',
                        }
                current_price = float(test_data['Close'].iloc[-1])

                return {
                    'status': 'success',
                    'ticker': ticker,
                    'company': company_info,
                    'current_price': current_price,
                    'prediction': {
                        'direction': basic_result.get('direction', 0),
                        'direction_confidence': basic_result.get('confidence', 0.3),
                        'expected_return': basic_result.get('momentum_5d', 0) / 100,  # Convert % to decimal
                        'volatility': None,
                        'volatility_percentile': None,
                        'median_volatility': None,
                        'confidence_interval': None,
                        'strategy_type': 'basic_technical',
                        'asset_recommendation': basic_result.get('recommendation', ''),
                    },
                    'trading_signal': {
                        'action': 'HOLD',  # Always HOLD for limited data
                        'should_trade': False,
                        'confidence': basic_result.get('confidence', 0.3),
                        'reason': f"Limited trading history ({days_available} days). {basic_result['message']}. {basic_result.get('recommendation', '')}",
                        'strategy': 'basic_technical',
                        'position': None,
                        'entry_price': None,
                        'stop_loss': None,
                        'take_profit': None,
                    },
                    'market_context': {
                        'volatility_percentile': None,
                        'historical_volatility': None,
                        'vol_vs_historical': None,
                        'regime': 'insufficient_data',
                    },
                    'data_tier': data_tier,
                    'basic_analysis': basic_result,
                    'model_info': {
                        'model_type': 'basic_technical',
                        'ml_ensemble': 'N/A - insufficient data for ML',
                        'features_count': len(data_tier['features_available']),
                        'features_available': data_tier['features_available'],
                        'trained_at': datetime.now().isoformat(),
                        'asset_class': 'new_listing',
                        'days_available': days_available,
                        'enhancements': {
                            '1_tier': f"Analysis tier: {data_tier['tier']}",
                            '2_method': data_tier['analysis_type'],
                            '3_confidence': f"Confidence level: {data_tier['confidence']}",
                            '4_note': 'ML analysis requires 250+ days of trading data',
                        },
                    },
                    'chart_data': {
                        'dates': test_data.index.strftime('%Y-%m-%d').tolist(),
                        'prices': test_data['Close'].tolist(),
                        'volumes': test_data['Volume'].tolist() if 'Volume' in test_data.columns else [],
                    },
                    'timestamp': datetime.now().isoformat(),
                }
        except ValueError as ve:
            # No data at all - let it fall through to normal error handling
            raise ve

        # ENHANCEMENT #6 & #7: Get or train model with specific type and asset-specific hyperparameters
        model_info = get_or_train_model(ticker, model_type=model_type)
        model = model_info['model']
        feature_cols = model_info['feature_cols']
        used_model_type = model_info.get('model_type', 'ensemble')
        asset_class = model_info.get('asset_class', 'stock')

        # Fetch fresh data for prediction
        data = fetch_data(ticker, lookback_days=500)

        # Engineer features (market-specific: US/Intl vs China)
        data_features = engineer_market_specific_features(data, ticker)

        # ENHANCEMENT #5: Add sentiment features (MUST match training pipeline)
        if SENTIMENT_ENGINEER is not None:
            try:
                logger.info(f"Adding sentiment features for prediction on {ticker}...")
                data_features = SENTIMENT_ENGINEER.add_real_sentiment_features(data_features, ticker)
                logger.info(f"Sentiment features added successfully")
            except Exception as e:
                logger.warning(f"Sentiment feature engineering failed for prediction (continuing without): {e}")
        else:
            logger.info("Sentiment analyzer not available for prediction, skipping sentiment features")

        # Prepare features for prediction
        # NOTE: HybridEnsemble needs enough data for lookback window (up to 30 for stocks)
        # So we take last 60 samples to be safe (allows lookback + buffer for predictions)
        # IPO/NEW STOCK FIX: Handle missing features due to adaptive windows
        # When stocks have limited data, volatility features use smaller windows
        # (e.g., parkinson_vol_15 vs parkinson_vol_60)
        # Add missing feature columns with NaN (will be filled with 0 below)
        missing_features = [col for col in feature_cols if col not in data_features.columns]
        if missing_features:
            logger.warning(f"[IPO FIX] Adding {len(missing_features)} missing features for {ticker}: {missing_features[:5]}...")
            for col in missing_features:
                data_features[col] = np.nan

        X_latest = data_features[feature_cols].tail(60)

        # CRITICAL: Clean features for prediction (handle inf/nan values)
        # This prevents XGBoost errors for commodities/forex with inf values
        X_latest = pd.DataFrame(X_latest)  # Ensure it's a DataFrame
        X_latest = X_latest.replace([np.inf], 1e10)
        X_latest = X_latest.replace([-np.inf], -1e10)
        # FIXING9: Use safe historical means instead of zeros!
        if FIXING9_AVAILABLE:
            X_latest = safe_fill_missing_features(X_latest, verbose=False)
        else:
            X_latest = X_latest.fillna(0)
        # BUGFIX: Select numeric columns only for inf/nan checks to avoid TypeError with mixed dtypes
        X_numeric_cols = X_latest.select_dtypes(include=[np.number]).columns
        if len(X_numeric_cols) > 0:
            logger.info(f"Cleaned prediction features: shape={X_latest.shape}, has_inf={np.isinf(X_latest[X_numeric_cols].values).any()}, has_nan={np.isnan(X_latest[X_numeric_cols].values).any()}")
        else:
            logger.info(f"Cleaned prediction features: shape={X_latest.shape}")

        # ===== PREDICTION ARCHITECTURE =====
        # This system now supports DUAL-MODEL routing:
        # - Chinese stocks (*.HK, *.SS, *.SZ) → China Model (CSI300, CNY, HSI features)
        # - US/International stocks → US/Intl Model (VIX, SPY, DXY, GLD features)
        #
        # Each model predicts 5-day forward RETURNS (direction + magnitude)
        # Volatility is estimated using Yang-Zhang volatility (yz_vol_20) from features

        # NEW: Dual model routing with sector-based filtering for Chinese stocks
        router = get_model_router()
        if router is not None and USE_DUAL_MODEL_SYSTEM:
            market = MarketClassifier.get_market(ticker)
            if market == 'chinese':
                logger.info(f"[DUAL MODEL] Chinese stock detected: {ticker}")
                try:
                    # Use sector router to determine if China model should be used
                    sector_router = ChinaSectorRouter()
                    sector_info = sector_router.get_sector_info(ticker)

                    logger.info(f"[SECTOR ROUTING] Ticker: {ticker}, Sector: {sector_info['sector_name']}")

                    if sector_info['use_china_model']:
                        # Use China model (sector-optimized) for supported sectors
                        logger.info(f"[SECTOR ROUTING] Using sector-optimized China model (expected {sector_info['expected_accuracy']:.1%} accuracy)")
                        china_predictions = sector_router.predict(ticker, X_latest)

                        if china_predictions is not None:
                            predicted_return = china_predictions[-1] if len(china_predictions) > 0 else 0.0
                            logger.info(f"[SECTOR MODEL] Prediction: {predicted_return:.4f}")
                        else:
                            # Router returned None - fallback to US/Intl
                            logger.info(f"[SECTOR ROUTING] Sector router returned None, using US/Intl fallback")
                            predictions = model.predict(X_latest)
                            predicted_return = predictions[-1] if len(predictions) > 0 else 0.0
                    else:
                        # Skip China model for this sector - use US/Intl fallback
                        logger.info(f"[SECTOR ROUTING] Skipping China model (sector not supported), using US/Intl fallback")
                        predictions = model.predict(X_latest)
                        predicted_return = predictions[-1] if len(predictions) > 0 else 0.0

                except Exception as e:
                    logger.error(f"[SECTOR ROUTING] Sector routing failed, falling back to US/Intl: {e}")
                    predictions = model.predict(X_latest)
                    predicted_return = predictions[-1] if len(predictions) > 0 else 0.0
            else:
                logger.info(f"[DUAL MODEL] Using US/Intl model for {ticker}")
                predictions = model.predict(X_latest)
                predicted_return = predictions[-1] if len(predictions) > 0 else 0.0
        else:
            # Dual model system disabled - use standard model
            logger.info(f"[SINGLE MODEL] Using standard model for {ticker}")
            predictions = model.predict(X_latest)
            predicted_return = predictions[-1] if len(predictions) > 0 else 0.0

        # Get Yang-Zhang volatility estimate (this is our "predicted" volatility)
        # Yang-Zhang combines overnight + intraday volatility for accurate estimation
        if 'yz_vol_20' in data_features.columns:
            predicted_vol = data_features['yz_vol_20'].iloc[-1]
            historical_vols = data_features['yz_vol_20'].dropna().values
        elif 'hist_vol_20' in data_features.columns:
            predicted_vol = data_features['hist_vol_20'].iloc[-1]
            historical_vols = data_features['hist_vol_20'].dropna().values
        else:
            # Fallback: calculate simple volatility from daily returns
            predicted_vol = data_features['returns_1d'].tail(20).std() if 'returns_1d' in data_features.columns else 0.02
            historical_vols = np.array([predicted_vol])

        # Historical volatility for comparison (same as predicted in this architecture)
        hist_vol = predicted_vol

        # Direction from predicted return (model output)
        direction = 1 if predicted_return > 0 else -1 if predicted_return < 0 else 0

        # IMPROVED CONFIDENCE CALCULATION WITH ASSET-CLASS SPECIFIC SNR THRESHOLDS
        # Different asset classes have different volatility characteristics:
        # - Stocks: Moderate volatility, standard threshold (0.5)
        # - Crypto: High volatility, need higher threshold (0.8) to be "confident"
        # - Forex: Low volatility/smooth, lower threshold (0.3) suffices
        # - Commodities: Seasonal patterns, moderate threshold (0.5)

        # Detect asset type from ticker pattern (before Yahoo API call for efficiency)
        def detect_asset_type_from_ticker(tkr):
            """Quick asset type detection from ticker pattern."""
            tkr_upper = tkr.upper()
            if '-USD' in tkr_upper or '-USDT' in tkr_upper:
                return 'crypto'
            elif '=X' in tkr_upper:
                return 'forex'
            elif '=F' in tkr_upper:
                return 'commodity'
            elif '.HK' in tkr_upper or '.SZ' in tkr_upper or '.SS' in tkr_upper:
                return 'china_stock'
            else:
                return 'stock'

        # Asset-class specific SNR thresholds
        # Higher threshold = requires stronger signal for same confidence level
        SNR_THRESHOLDS = {
            'stock': 0.5,       # Standard threshold for stocks
            'china_stock': 0.5, # Same as regular stocks
            'crypto': 0.8,      # Higher threshold - crypto is inherently volatile
            'forex': 0.3,       # Lower threshold - forex is smoother, small moves matter
            'commodity': 0.5,   # Standard threshold with seasonal adjustment
        }

        asset_type_for_snr = detect_asset_type_from_ticker(ticker)
        snr_threshold = SNR_THRESHOLDS.get(asset_type_for_snr, 0.5)

        # Confidence measures how likely the predicted direction is correct
        # Formula: sigmoid-like transformation of signal-to-noise ratio (SNR)
        # SNR = |predicted_return| / historical_volatility
        # Higher SNR = stronger signal relative to typical noise = higher confidence
        if hist_vol > 0 and abs(predicted_return) > 0:
            # Signal-to-noise ratio: how many "standard deviations" is the prediction
            snr = abs(predicted_return) / hist_vol
            # Use sigmoid transformation to bound confidence between 0.3 and 0.95
            # The snr_threshold shifts the sigmoid curve for different asset classes
            # For crypto (0.8): snr=0.8 → 0.5, snr=1.6 → ~0.73, snr=2.4 → ~0.88
            # For forex (0.3): snr=0.3 → 0.5, snr=0.6 → ~0.73, snr=0.9 → ~0.88
            import math
            direction_confidence = 0.3 + 0.65 * (1 / (1 + math.exp(-1.5 * (snr - snr_threshold))))
        else:
            direction_confidence = 0.5  # Neutral confidence for zero prediction

        # ============================================================================
        # PHASE 1-6 ENHANCED SIGNAL PROCESSING FOR ML ANALYSIS
        # NOTE: Phase 1-6 calculations are specifically tuned for China market stocks
        # US/International stocks will get separate optimized updates later
        # ============================================================================
        phase6_analysis = {}
        is_china_stock = asset_type_for_snr == 'china_stock'

        if PHASE6_CALCULATIONS_AVAILABLE and is_china_stock:
            try:
                # Get price and volume data for order flow analysis
                # Ensure all arrays have same length by aligning to shortest
                prices = data['Close'].values
                volumes = data['Volume'].values if 'Volume' in data.columns else np.ones(len(prices))
                highs = data['High'].values if 'High' in data.columns else prices
                lows = data['Low'].values if 'Low' in data.columns else prices
                returns_raw = data['Close'].pct_change().fillna(0).values  # Keep same length, fill first NaN with 0

                # Ensure all arrays are same length (use minimum length)
                min_len = min(len(prices), len(volumes), len(highs), len(lows), len(returns_raw))
                prices = prices[-min_len:]
                volumes = volumes[-min_len:]
                highs = highs[-min_len:]
                lows = lows[-min_len:]
                returns = returns_raw[-min_len:]

                # PHASE 1: Volatility scaling for position sizing
                vol_scaling = calculate_volatility_scaling(
                    returns[-60:] if len(returns) >= 60 else returns,
                    lookback=min(20, len(returns)),
                    benchmark_vol=0.15
                )

                # PHASE 2: Multi-signal combination (momentum, volatility, mean reversion)
                # These functions return floats directly
                mom_signal = momentum_signal(prices, period=20)
                vol_signal = volatility_signal(prices, period=20)
                mr_signal = mean_reversion_signal(prices, period=20)

                combined_signal, signal_confidence = combine_signals({
                    'momentum': mom_signal,
                    'volatility': vol_signal,
                    'mean_reversion': mr_signal
                })

                # PHASE 3: Order flow analysis (smart money detection)
                # Note: OBV returns n-1 elements due to diff, so we get the last values
                try:
                    obv_arr = calculate_obv(prices, volumes)
                    obv = float(obv_arr[-1]) if len(obv_arr) > 0 else 0.0
                except:
                    obv = 0.0

                try:
                    ad_arr = accumulation_distribution(highs, lows, prices, volumes)
                    ad_line = float(ad_arr[-1]) if len(ad_arr) > 0 else 0.0
                except:
                    ad_line = 0.0

                try:
                    mfi = money_flow_index(highs, lows, prices, volumes, period=14)
                except:
                    mfi = 50.0  # Neutral

                try:
                    vwap = calculate_vwap(highs, lows, prices, volumes)
                except:
                    vwap = float(prices[-1]) if len(prices) > 0 else 0.0

                # Smart money requires aligned arrays - skip if arrays don't match
                try:
                    obv_full = calculate_obv(prices, volumes)
                    ad_full = accumulation_distribution(highs, lows, prices, volumes)
                    # Align to shortest length
                    align_len = min(len(prices), len(volumes), len(obv_full), len(ad_full))
                    smart_arr = detect_smart_money(
                        prices[-align_len:],
                        volumes[-align_len:],
                        obv_full[-align_len:],
                        ad_full[-align_len:]
                    )
                    smart_money = bool(smart_arr[-1] > 0.5) if len(smart_arr) > 0 else False
                except:
                    smart_money = False

                # PHASE 3: Regime detection using Hurst exponent
                hurst = calculate_hurst_exponent(prices)
                # detect_volatility_regime returns an array of regime labels (0-3)
                vol_regime_arr = detect_volatility_regime(returns[-60:] if len(returns) >= 60 else returns)
                # Map the GMM regime label to a string - get the most recent regime
                # 4 components map to: 0=low_vol, 1=normal, 2=high_vol, 3=crisis (sorted by typical volatility)
                regime_map = {0: 'low_vol', 1: 'normal', 2: 'high_vol', 3: 'crisis'}
                current_regime_label = int(vol_regime_arr[-1]) if hasattr(vol_regime_arr, '__len__') and len(vol_regime_arr) > 0 else 1
                vol_regime = regime_map.get(current_regime_label, 'normal')

                # PHASE 4: Asset-specific multiplier
                asset_mult = asset_specific_multiplier(
                    base_mult=1.0,
                    asset_class=asset_type_for_snr,
                    vix=None,
                    vix_sensitivity=1.0
                )

                # PHASE 5: Performance metrics with decay weighting
                if len(returns) >= 20:
                    decay_sharpe = decay_weighted_sharpe(returns[-252:] if len(returns) >= 252 else returns)
                    calmar = calmar_ratio(returns[-252:] if len(returns) >= 252 else returns)
                    composite_score, composite_conf = composite_performance_score(
                        returns[-252:] if len(returns) >= 252 else returns,
                        decay_factor=0.95
                    )
                else:
                    decay_sharpe = 0.0
                    calmar = 0.0
                    composite_score = 0.0

                # PHASE 5: Bayesian signal combination for confidence adjustment
                bayesian_combined = bayesian_signal_combination(
                    signals_dict={'momentum': mom_signal, 'mean_reversion': mr_signal, 'combined': combined_signal},
                    reliability_dict={'momentum': 0.6, 'mean_reversion': 0.5, 'combined': 0.7}
                )

                # PHASE 6: Expected Shortfall (CVaR) for tail risk
                # For single asset, use weight of 1.0
                if len(returns) >= 20:
                    single_weight = np.array([1.0])
                    # Reshape returns for multi-asset format expected by expected_shortfall
                    returns_2d = returns.reshape(-1, 1) if len(returns.shape) == 1 else returns
                    es_95 = expected_shortfall(returns_2d, single_weight, confidence=0.95)
                    es_99 = expected_shortfall(returns_2d, single_weight, confidence=0.99)
                else:
                    es_95 = predicted_vol * 2.0
                    es_99 = predicted_vol * 3.0

                # PHASE 6: Regime-aware risk budget
                risk_budget = regime_aware_risk_budget(
                    base_risk=0.02,
                    current_regime=vol_regime,
                    regime_mults={'low': 1.15, 'medium': 1.0, 'high': 0.75, 'crisis': 0.5}
                )

                # Helper function to safely convert to scalar float
                def safe_float(val, default=0.0):
                    """Convert value to scalar float safely."""
                    try:
                        if hasattr(val, '__len__') and len(val) > 0:
                            return float(val.item() if hasattr(val, 'item') else val[-1])
                        return float(val)
                    except:
                        return float(default)

                # Store Phase 1-6 analysis results
                phase6_analysis = {
                    'phase1': {
                        'vol_scaling': round(safe_float(vol_scaling), 4),
                        'position_adjustment': 'reduce' if safe_float(vol_scaling) < 1.0 else 'increase' if safe_float(vol_scaling) > 1.0 else 'neutral'
                    },
                    'phase2': {
                        'momentum_signal': round(safe_float(mom_signal), 4),
                        'volatility_signal': round(safe_float(vol_signal), 4),
                        'mean_reversion_signal': round(safe_float(mr_signal), 4),
                        'combined_signal': round(safe_float(combined_signal), 4),
                        'signal_confidence': round(safe_float(signal_confidence), 4)
                    },
                    'phase3': {
                        'obv_trend': 'bullish' if safe_float(obv) > 0 else 'bearish',
                        'accumulation_distribution': round(safe_float(ad_line), 4),
                        'money_flow_index': round(safe_float(mfi, 50.0), 2),
                        'smart_money_detected': smart_money,
                        'vwap': round(safe_float(vwap), 2),
                        'hurst_exponent': round(safe_float(hurst, 0.5), 4),
                        'market_character': 'trending' if safe_float(hurst, 0.5) > 0.5 else 'mean_reverting'
                    },
                    'phase4': {
                        'asset_multiplier': round(safe_float(asset_mult, 1.0), 4),
                        'asset_class': asset_type_for_snr
                    },
                    'phase5': {
                        'decay_weighted_sharpe': round(safe_float(decay_sharpe), 4),
                        'calmar_ratio': round(safe_float(calmar), 4),
                        'composite_performance': round(safe_float(composite_score), 4),
                        'bayesian_combined_signal': round(safe_float(bayesian_combined, 0.5), 4)
                    },
                    'phase6': {
                        'expected_shortfall_95': round(safe_float(es_95), 4),
                        'expected_shortfall_99': round(safe_float(es_99), 4),
                        'risk_budget': round(safe_float(risk_budget, 0.02), 4),
                        'vol_regime': vol_regime
                    }
                }

                # Enhance direction confidence using Phase 5 Bayesian adjustment
                if bayesian_combined > 0.6:
                    direction_confidence = min(0.95, direction_confidence * 1.1)
                elif bayesian_combined < 0.4:
                    direction_confidence = max(0.3, direction_confidence * 0.9)

                logger.info(f"[PHASE 1-6] Enhanced analysis for {ticker}: vol_scaling={vol_scaling:.3f}, combined_signal={combined_signal:.3f}, ES95={es_95:.4f}")

            except Exception as e:
                logger.warning(f"[PHASE 1-6] Enhanced analysis failed for {ticker}: {e}")
                phase6_analysis = {'error': str(e)}

        # Confidence interval (from ensemble)
        lower_bound = predicted_vol * 0.7
        upper_bound = predicted_vol * 1.3

        # Detect regime
        regimes, _ = REGIME_DETECTOR.detect_regime(historical_vols)
        current_regime_id = REGIME_DETECTOR.predict_regime(hist_vol)
        regime_names = ['low', 'medium', 'high']
        current_regime = regime_names[current_regime_id] if current_regime_id < 3 else 'unknown'

        # Volatility percentile
        vol_percentile = (hist_vol < historical_vols).mean()

        # Current price
        current_price = data['Close'].iloc[-1]

        # Get company info FIRST - fetch from Yahoo Finance if not in database
        company_info = COMPANY_DATABASE.get(ticker)
        if not company_info:
            # Try to fetch company metadata from Yahoo Finance
            try:
                yf_ticker = yf.Ticker(ticker)
                info = yf_ticker.info
                company_name = info.get('longName') or info.get('shortName') or ticker
                company_type = 'Stock'  # Default
                exchange = info.get('exchange', 'Unknown')

                # Determine asset type based on ticker format and Yahoo Finance quoteType
                quote_type = info.get('quoteType', '').upper()

                if '-USD' in ticker or '-USDT' in ticker or quote_type == 'CRYPTOCURRENCY':
                    company_type = 'Cryptocurrency'
                elif '=X' in ticker or quote_type == 'CURRENCY':
                    company_type = 'Forex'
                elif '=F' in ticker or quote_type in ['FUTURE', 'COMMODITY']:
                    company_type = 'Commodity'
                elif quote_type in ['ETF', 'INDEX']:
                    company_type = quote_type.title()
                elif quote_type == 'EQUITY':
                    company_type = 'Stock'

                company_info = {
                    'name': company_name,
                    'type': company_type,
                    'exchange': exchange
                }
                logger.info(f"Fetched company info from Yahoo Finance: {ticker} -> {company_name}")
            except Exception as e:
                logger.warning(f"Could not fetch company info for {ticker}: {str(e)}")
                company_info = {
                    'name': ticker,
                    'type': 'Unknown',
                    'exchange': 'Unknown'
                }

        # Generate trading signal using Optimal Hybrid Strategy
        signal = HYBRID_STRATEGY.generate_hybrid_signal(
            direction_prediction=0.5 + (direction * direction_confidence / 2),  # Convert to 0-1 scale
            volatility_prediction=predicted_vol,
            historical_volatility=historical_vols,
            current_price=current_price,
            account_size=account_size
        )

        # Get asset-specific recommendation
        asset_recommendation = HYBRID_STRATEGY.get_asset_recommendation(
            company_info.get('type', 'Stock') if company_info else 'Stock'
        )

        # Position info from hybrid strategy signal
        position_info = None
        if signal['should_trade'] and signal.get('position_size', 0) > 0:
            position_info = {
                'shares': signal['position_size'],
                'value': signal['position_value'],
                'value_pct': (signal['position_value'] / account_size) * 100,
                'risk_amount': signal['risk_amount'],
                'risk_pct': (signal['risk_amount'] / account_size) * 100,
                'potential_profit': signal['reward_amount'],
                'risk_reward_ratio': signal['risk_reward_ratio']
            }

        # Build response
        prediction_response = {
            'ticker': ticker,
            'company': company_info,
            'timestamp': datetime.now().isoformat(),
            'current_price': float(current_price),
            'prediction': {
                'volatility': float(predicted_vol),
                'expected_return': float(predicted_return),  # Add expected return for risk-adjusted scoring
                'confidence_interval': {
                    'lower': float(lower_bound),
                    'upper': float(upper_bound),
                    'level': 0.80
                },
                'direction': int(direction),
                'direction_confidence': float(direction_confidence),
                'strategy_type': 'optimal_hybrid',
                'asset_recommendation': asset_recommendation,
                'volatility_percentile': float(signal.get('volatility_percentile', 0)),
                'median_volatility': float(signal.get('median_volatility', 0))
            },
            'market_context': {
                'regime': current_regime,
                'historical_volatility': float(hist_vol),
                'volatility_percentile': float(vol_percentile),
                'vol_vs_historical': float(predicted_vol / hist_vol) if hist_vol > 0 else 1.0
            },
            'trading_signal': {
                'action': signal['action'],
                'reason': signal['reason'],
                'confidence': float(signal['confidence']),
                'should_trade': signal['should_trade'],
                'entry_price': float(signal['entry_price']) if signal['entry_price'] else None,
                'stop_loss': float(signal['stop_loss']) if signal['stop_loss'] else None,
                'take_profit': float(signal['take_profit']) if signal['take_profit'] else None,
                'strategy': signal.get('strategy', 'optimal_hybrid'),
                'position': position_info
            },
            'chart_data': {
                'dates': data.index[-252:].strftime('%Y-%m-%d').tolist(),  # 1 year
                'prices': data['Close'].tail(252).tolist()
            },
            'model_info': {
                # ENHANCEMENTS: Updated model info with all improvements
                'model_type': used_model_type,
                'asset_class': asset_class,
                'type': f'{AVAILABLE_MODELS.get(used_model_type, "Unknown")} (Asset-Optimized for {asset_class.title()})',
                'ml_ensemble': 'Hybrid Ensemble: Old (LGBM+XGB+NN) + New (Hybrid LSTM/CNN w/ Profit Loss)' if used_model_type == 'hybrid_ensemble'
                              else 'LightGBM + XGBoost + Neural Networks + Prediction Market' if used_model_type == 'ensemble'
                              else 'Hybrid LSTM/CNN with Profit-Maximizing Loss',
                'strategy': 'Dual Filtering (65% confidence + volatility < median)',
                'features_count': len(feature_cols),
                'sentiment_enabled': SENTIMENT_ENGINEER is not None,
                'enhancements': {
                    '1_profit_loss': 'Profit-maximizing loss function (vs MSE)',
                    '2_gradient_clip': 'Gradient clipping enabled',
                    '3_data_period': '5 years training data',
                    '4_confidence': 'Lowered threshold for more trades',
                    '5_sentiment': 'FinBERT + VADER sentiment analysis' if SENTIMENT_ENGINEER else 'Not available',
                    '6_ensemble': f'Model: {used_model_type}',
                    '7_asset_params': f'Asset-specific hyperparameters for {asset_class}',
                    '8_costs': f'Transaction costs: {TOTAL_TRADING_FRICTION*100:.2f}% per trade'
                },
                'trading_costs': {
                    'transaction_cost_pct': TRANSACTION_COST * 100,
                    'slippage_pct': SLIPPAGE * 100,
                    'total_friction_pct': TOTAL_TRADING_FRICTION * 100,
                    'note': 'All returns shown are NET of trading costs'
                },
                'trained_at': model_info['trained_at'].isoformat(),
                'phase6_enabled': PHASE6_CALCULATIONS_AVAILABLE and is_china_stock,
                'phase6_note': 'Phase 1-6 calculations applied for China stocks' if (PHASE6_CALCULATIONS_AVAILABLE and is_china_stock) else 'Phase 1-6 reserved for China model only'
            },
            'status': 'success'
        }

        # Add Phase 1-6 analysis to response if available
        if phase6_analysis:
            prediction_response['phase6_analysis'] = phase6_analysis

        # ============================================================================
        # CHINA LAG-FREE TRANSITION DETECTION (Fix 62 Adaptation)
        # Detects early warning signals for regime transitions to reduce lag
        # ============================================================================
        if is_china_stock and CHINA_TRANSITION_AVAILABLE and CHINA_TRANSITION_DETECTOR is not None:
            try:
                # Detect regime transition with lag-free early warning
                transition_result = CHINA_TRANSITION_DETECTOR.detect_transition(
                    data,
                    current_regime=phase6_analysis.get('phase6', {}).get('vol_regime', 'NEUTRAL').upper() if phase6_analysis else 'NEUTRAL'
                )

                # Build transition analysis for response
                transition_analysis = {
                    'transition_type': transition_result.transition_type,
                    'confidence': round(transition_result.confidence, 4),
                    'is_transition': transition_result.is_transition,
                    'signals_detected': transition_result.signals_detected,
                    'is_confirmed': transition_result.is_confirmed,
                    'blend_factor': round(transition_result.blend_factor, 4),
                    'allocation_adjustment': round(transition_result.recommended_allocation_adjustment, 4),
                    'recommended_actions': transition_result.recommended_actions
                }

                # Get regime-specific position sizing parameters
                regime_map = {
                    'BULL': {'max_allocation': 0.95, 'max_positions': 6, 'position_cap': 0.30, 'min_ev': 0.5},
                    'BEAR': {'max_allocation': 0.70, 'max_positions': 4, 'position_cap': 0.20, 'min_ev': 1.0},
                    'HIGH_VOL': {'max_allocation': 0.50, 'max_positions': 3, 'position_cap': 0.15, 'min_ev': 2.0},
                    'NEUTRAL': {'max_allocation': 0.85, 'max_positions': 5, 'position_cap': 0.25, 'min_ev': 0.7}
                }

                # Determine current regime from transition or phase6
                current_detected_regime = 'NEUTRAL'
                if transition_result.is_confirmed:
                    if transition_result.transition_type == 'TRANSITION_TO_BULL':
                        current_detected_regime = 'BULL'
                    elif transition_result.transition_type == 'TRANSITION_TO_BEAR':
                        current_detected_regime = 'BEAR'
                    elif transition_result.transition_type == 'HIGH_VOLATILITY':
                        current_detected_regime = 'HIGH_VOL'

                transition_analysis['current_regime'] = current_detected_regime
                transition_analysis['regime_params'] = regime_map.get(current_detected_regime, regime_map['NEUTRAL'])

                # Add transition analysis to response
                prediction_response['china_transition'] = transition_analysis

                # Log transition detection
                if transition_result.is_transition:
                    logger.info(f"[CHINA TRANSITION] {ticker}: {transition_result.transition_type} "
                               f"(confidence: {transition_result.confidence:.2%}, "
                               f"signals: {len(transition_result.signals_detected)})")

            except Exception as e:
                logger.warning(f"[CHINA TRANSITION] Detection failed for {ticker}: {e}")
                prediction_response['china_transition'] = {'error': str(e), 'available': False}

        # ============================================================================
        # US PROFIT-MAXIMIZING REGIME ANALYSIS (from 'us model fixing8.pdf')
        # Applies ONLY to US/International stocks - NOT China/DeepSeek model
        # Features: Dynamic multiplier scaling, transition phase detection, Kelly sizing
        # ============================================================================
        if not is_china_stock and US_PROFIT_REGIME_AVAILABLE and US_PROFIT_REGIME_CLASSIFIER is not None:
            try:
                # Get SPY data for regime classification
                spy_data = yf.download('SPY', period='6mo', progress=False)
                if isinstance(spy_data.columns, pd.MultiIndex):
                    spy_data.columns = spy_data.columns.get_level_values(0)

                # Get current VIX
                vix_data = yf.download('^VIX', period='1d', progress=False)
                if isinstance(vix_data.columns, pd.MultiIndex):
                    vix_data.columns = vix_data.columns.get_level_values(0)
                current_vix = float(vix_data['Close'].iloc[-1]) if len(vix_data) > 0 else 20.0

                # Build stock data for profit scoring
                stock_metrics = {
                    'ticker': ticker,
                    '5d_return': float(predicted_return * 100) if predicted_return else 0,  # Use ML 5-day forecast
                    'volume_ratio': 1.0,  # Default volume ratio
                    'rsi': _calculate_rsi_simple(data['Close']) if len(data) >= 14 else 50,  # Calculate RSI from data
                    'volatility': float(data['Close'].pct_change().std() * 100 * np.sqrt(252)) if len(data) >= 20 else 25,  # Calculate annualized vol
                }

                # Try to calculate volume ratio from data
                if len(data) >= 10:
                    vol = data['Volume']
                    stock_metrics['volume_ratio'] = float(vol.iloc[-1] / vol.iloc[-10:].mean()) if vol.iloc[-10:].mean() > 0 else 1.0

                # Get profit-maximizing regime classification
                profit_regime = US_PROFIT_REGIME_CLASSIFIER.classify_regime_with_profit(
                    spy_data, stock_metrics, current_vix
                )

                # Build US profit regime analysis for response
                us_profit_analysis = {
                    'regime': profit_regime.regime,
                    'transition_state': profit_regime.transition_state,
                    'transition_phase': profit_regime.transition_phase.value,
                    'transition_day': profit_regime.transition_day,
                    'profit_score': round(profit_regime.profit_score, 1),
                    'dynamic_multiplier': round(profit_regime.dynamic_multiplier, 3),
                    'recommended_position_size': round(profit_regime.recommended_position_size * 100, 2),
                    'small_cap_multiplier': round(profit_regime.small_cap_multiplier, 2),
                    'large_cap_multiplier': round(profit_regime.large_cap_multiplier, 2),
                    'max_position_size': round(profit_regime.max_position_size * 100, 1),
                    'stop_loss': round(profit_regime.stop_loss * 100, 1),
                    'profit_targets': {
                        'target_1': round(profit_regime.profit_target_1 * 100, 1),
                        'target_2': round(profit_regime.profit_target_2 * 100, 1),
                        'target_3': round(profit_regime.profit_target_3 * 100, 1),
                        'trailing_stop': round(profit_regime.trailing_stop * 100, 1),
                    },
                    'position_params': {
                        'max_positions': profit_regime.max_positions,
                        'top_n_concentration': round(profit_regime.top_n_concentration * 100, 0),
                        'sizing_method': profit_regime.position_sizing_method,
                    },
                    'capital_allocation': {k: round(v * 100, 1) for k, v in profit_regime.capital_allocation.items()},
                    'sector_focus': profit_regime.sector_focus,
                    'vix_level': round(current_vix, 1),
                    'source': 'us_model_fixing8'
                }

                prediction_response['us_profit_regime'] = us_profit_analysis

                logger.info(f"[US PROFIT REGIME] {ticker}: Score={profit_regime.profit_score:.1f}, "
                           f"Mult={profit_regime.dynamic_multiplier:.3f}x, "
                           f"Phase={profit_regime.transition_phase.value}")

            except Exception as e:
                logger.warning(f"[US PROFIT REGIME] Analysis failed for {ticker}: {e}")
                prediction_response['us_profit_regime'] = {'error': str(e), 'available': False}

        # Generate news feed
        prediction_response['news_feed'] = generate_news_feed(ticker, company_info, current_price, prediction_response)

        return prediction_response

    except Exception as e:
        logger.error(f"Error generating prediction for {ticker}: {str(e)}", exc_info=True)
        return {
            'ticker': ticker,
            'status': 'error',
            'error': str(e)
        }


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve main page."""
    return render_template('index.html')


@app.route('/api/search')
def search_tickers():
    """Search for tickers by company name or symbol - uses Yahoo Finance for 100% coverage."""
    query = request.args.get('q', '').strip().upper()

    if not query:
        # Return popular tickers
        popular = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM', 'BTC-USD', 'EURUSD=X']
        results = [{'ticker': t, **COMPANY_DATABASE[t]} for t in popular if t in COMPANY_DATABASE]
        return jsonify({'results': results[:10]})

    # ========== CHINA STOCK FORMAT RESOLUTION ==========
    # Handle common Chinese stock input formats:
    # - 09927 (5-digit HK code with leading zero) -> 9927.HK
    # - SZ300750 (Shenzhen prefix format) -> 300750.SZ
    # - SH601127 (Shanghai prefix format) -> 601127.SS
    # - 600519 (6-digit A-share) -> 600519.SS or 000001.SZ
    logger.info(f"[SEARCH] Searching for: {query}, ENHANCED_DATA_LAYER={ENHANCED_DATA_LAYER}")
    if ENHANCED_DATA_LAYER:
        resolved_ticker = resolve_china_stock(query)
        logger.info(f"[SEARCH] resolve_china_stock({query}) returned: {resolved_ticker}")
        if resolved_ticker and resolved_ticker != query:
            logger.info(f"[SEARCH] Resolved China format: {query} -> {resolved_ticker}")
            # Try to get info from our database or Yahoo Finance
            if resolved_ticker in COMPANY_DATABASE:
                return jsonify({'results': [{'ticker': resolved_ticker, **COMPANY_DATABASE[resolved_ticker]}]})
            else:
                # Try Yahoo Finance for the resolved ticker
                try:
                    test_ticker = yf.Ticker(resolved_ticker)
                    info = test_ticker.info
                    if info and ('shortName' in info or 'longName' in info):
                        company_name = info.get('longName') or info.get('shortName') or resolved_ticker
                        exchange = info.get('exchange', 'Unknown')
                        quote_type = info.get('quoteType', 'Unknown')
                        return jsonify({'results': [{
                            'ticker': resolved_ticker,
                            'name': company_name,
                            'type': quote_type if quote_type != 'EQUITY' else 'Stock',
                            'exchange': exchange
                        }]})
                except Exception as e:
                    logger.debug(f"Yahoo lookup failed for resolved ticker {resolved_ticker}: {e}")

                # Even if Yahoo lookup fails, return the resolved ticker
                # Determine exchange from suffix
                exchange = 'Unknown'
                stock_type = 'Stock'
                if resolved_ticker.endswith('.HK'):
                    exchange = 'HKEX'
                elif resolved_ticker.endswith('.SS'):
                    exchange = 'SSE'
                elif resolved_ticker.endswith('.SZ'):
                    exchange = 'SZSE'

                return jsonify({'results': [{
                    'ticker': resolved_ticker,
                    'name': f'{resolved_ticker} (Click to analyze)',
                    'type': stock_type,
                    'exchange': exchange
                }]})

    # Check if query matches a search alias
    # FIX: Use exact match or word boundary matching to prevent false positives
    # e.g., "META" should NOT match "MEITUAN" (META is a substring of MEITUAN)
    alias_tickers = []
    for alias, tickers in SEARCH_ALIASES.items():
        # Exact match (query == alias)
        if query == alias:
            alias_tickers.extend(tickers)
        # Query contains full alias as a word (e.g., "APPLE INC" contains "APPLE")
        elif alias in query and (
            query.startswith(alias + ' ') or
            query.endswith(' ' + alias) or
            ' ' + alias + ' ' in query or
            query == alias
        ):
            alias_tickers.extend(tickers)
        # Alias contains full query as a word (e.g., "APPLE" in "APPLE INC")
        # Only match if query is at least 4 chars to avoid false positives like "META" in "MEITUAN"
        elif len(query) >= 4 and query in alias and (
            alias.startswith(query + ' ') or
            alias.endswith(' ' + query) or
            ' ' + query + ' ' in alias or
            alias == query
        ):
            alias_tickers.extend(tickers)

    # Search by ticker or company name in our database
    results = []
    seen_tickers = set()
    alias_ticker_set = set(alias_tickers)  # Track which tickers came from aliases for sorting

    # First add alias matches
    for ticker in alias_tickers:
        if ticker in COMPANY_DATABASE and ticker not in seen_tickers:
            results.append({'ticker': ticker, **COMPANY_DATABASE[ticker], '_from_alias': True})
            seen_tickers.add(ticker)

    # Then add direct matches from our database
    for ticker, info in COMPANY_DATABASE.items():
        if ticker not in seen_tickers and (query in ticker or query in info['name'].upper()):
            results.append({'ticker': ticker, **info})
            seen_tickers.add(ticker)

    # ========== DYNAMIC YAHOO FINANCE SEARCH ==========
    # If we have less than 5 results AND no alias matches, try Yahoo Finance directly
    # Skip Yahoo search if we already have results from aliases (e.g., APPLE -> AAPL)
    has_alias_results = any(r.get('_from_alias') for r in results)
    if len(results) < 5 and not has_alias_results:
        try:
            # Try the query as a direct ticker symbol
            test_ticker = yf.Ticker(query)
            info = test_ticker.info

            # Check if ticker is valid (has a shortName or longName)
            if info and ('shortName' in info or 'longName' in info):
                ticker_symbol = query
                if ticker_symbol not in seen_tickers:
                    company_name = info.get('longName') or info.get('shortName') or ticker_symbol
                    exchange = info.get('exchange', 'Unknown')
                    quote_type = info.get('quoteType', 'Unknown')

                    results.append({
                        'ticker': ticker_symbol,
                        'name': company_name,
                        'type': quote_type if quote_type != 'EQUITY' else 'Stock',
                        'exchange': exchange
                    })
                    seen_tickers.add(ticker_symbol)
                    logger.info(f"Found via Yahoo Finance: {ticker_symbol} - {company_name}")
        except Exception as e:
            logger.debug(f"Direct ticker search failed for {query}: {e}")

    # Try common suffix patterns if still not enough results
    # BUT: Skip crypto suffix if we already have results from our database or aliases
    # This prevents "APPLE" from suggesting "APPLE-USD" when we already have "AAPL"
    if len(results) < 3:
        # Determine which suffixes to try based on existing results
        has_stock_match = any(r.get('type') == 'Stock' for r in results)

        # Common exchange suffixes to try
        suffixes = [
            '.SS',  # Shanghai
            '.SZ',  # Shenzhen
            '.HK',  # Hong Kong
            '.SI',  # Singapore
            '.T',   # Tokyo
            '.KS',  # Korea
            '.NS',  # India NSE
            '.BO',  # India BSE
            '.L',   # London
            '.PA',  # Paris
            '.DE',  # Germany
            '.AS',  # Amsterdam
            '.SW',  # Switzerland
            '=F',   # Futures
        ]

        # Only try crypto suffix if we don't already have a stock match from aliases/database
        if not has_stock_match:
            suffixes.append('-USD')  # Crypto

        for suffix in suffixes:
            if len(results) >= 10:
                break

            test_symbol = query + suffix
            if test_symbol in seen_tickers:
                continue

            try:
                test_ticker = yf.Ticker(test_symbol)
                info = test_ticker.info

                if info and ('shortName' in info or 'longName' in info):
                    company_name = info.get('longName') or info.get('shortName') or test_symbol
                    exchange = info.get('exchange', 'Unknown')
                    quote_type = info.get('quoteType', 'Unknown')

                    results.append({
                        'ticker': test_symbol,
                        'name': company_name,
                        'type': quote_type if quote_type != 'EQUITY' else 'Stock',
                        'exchange': exchange
                    })
                    seen_tickers.add(test_symbol)
                    logger.info(f"Found via Yahoo Finance with suffix: {test_symbol} - {company_name}")
            except Exception as e:
                logger.debug(f"Suffix search failed for {test_symbol}: {e}")

    # Sort by relevance - prioritize: 1) Alias matches, 2) Database entries, 3) Yahoo matches
    # This ensures "APPLE" query returns "AAPL" (from alias) before "APPLE-USD" (Yahoo crypto)
    results.sort(key=lambda x: (
        0 if x.get('_from_alias') else                    # Alias matches first (APPLE -> AAPL)
        1 if x['ticker'] in COMPANY_DATABASE else        # Then our database entries
        2 if x['ticker'] == query else                    # Then exact ticker matches
        3 if x['ticker'].startswith(query) else          # Then partial ticker matches
        4                                                  # Finally everything else
    ))

    # Remove internal tracking field before returning
    for result in results:
        result.pop('_from_alias', None)

    return jsonify({'results': results[:15]})  # Return up to 15 results


@app.route('/api/sentiment/<ticker>')
def get_sentiment(ticker):
    """
    Get sentiment analysis for a ticker.

    Query parameters:
        - days_back: Number of days to look back (default: 7)
        - analyzer: 'vader' (fast) or 'finbert' (accurate) (default: 'vader')

    Returns:
        JSON with sentiment data from news, Twitter, and Reddit
    """
    try:
        if SENTIMENT_ENGINEER is None:
            return jsonify({
                'ticker': ticker,
                'status': 'unavailable',
                'message': 'Sentiment analysis not initialized'
            }), 503

        days_back = request.args.get('days_back', 7, type=int)
        analyzer = request.args.get('analyzer', 'vader', type=str)

        # Normalize ticker format
        ticker = normalize_ticker(ticker.upper())

        logger.info(f"Fetching sentiment for {ticker} (analyzer={analyzer}, days={days_back})")

        # Fetch news with dates
        headlines = SENTIMENT_ENGINEER.fetch_yahoo_finance_news(ticker, days_back=days_back)

        # Analyze sentiment
        sentiments = []
        for date, headline in headlines:
            if analyzer == 'finbert':
                sentiment = SENTIMENT_ENGINEER.analyze_text_finbert_cached(headline)
            else:
                sentiment = SENTIMENT_ENGINEER.analyze_text_vader_cached(headline)

            sentiments.append({
                'date': date.isoformat(),
                'headline': headline,
                'sentiment': sentiment
            })

        # Calculate aggregate statistics
        sentiment_scores = [s['sentiment'] for s in sentiments]

        return jsonify({
            'ticker': ticker,
            'status': 'success',
            'analyzer': analyzer,
            'period_days': days_back,
            'sentiments': sentiments,
            'statistics': {
                'count': len(sentiment_scores),
                'average': float(np.mean(sentiment_scores)) if sentiment_scores else 0,
                'positive_count': sum(1 for s in sentiment_scores if s > 0.1),
                'negative_count': sum(1 for s in sentiment_scores if s < -0.1),
                'neutral_count': sum(1 for s in sentiment_scores if -0.1 <= s <= 0.1)
            }
        })

    except Exception as e:
        logger.error(f"Error fetching sentiment for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'ticker': ticker,
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/news')
def get_realtime_news():
    """
    Get realtime news from all watchlist tickers.

    Returns:
        JSON with aggregated news from all watchlist items
    """
    try:
        if SENTIMENT_ENGINEER is None:
            return jsonify({
                'status': 'unavailable',
                'message': 'Sentiment analysis not initialized',
                'news': []
            }), 503

        # Get current user (default to 'default_user')
        username = 'default_user'

        # Get watchlist tickers
        watchlist_tickers = []
        if username in USER_WATCHLISTS:
            watchlist_tickers = USER_WATCHLISTS[username]

        # If no watchlist, use default tickers
        if not watchlist_tickers:
            watchlist_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

        logger.info(f"Fetching news for {len(watchlist_tickers)} watchlist items")

        # Fetch news from all tickers
        all_news = []
        for ticker in watchlist_tickers[:10]:  # Limit to 10 tickers to avoid timeout
            try:
                ticker = normalize_ticker(ticker.upper())
                headlines = SENTIMENT_ENGINEER.fetch_yahoo_finance_news(ticker, days_back=3)

                # Analyze sentiment and add ticker info
                for date, headline in headlines[:3]:  # Take top 3 per ticker
                    sentiment = SENTIMENT_ENGINEER.analyze_text_vader_cached(headline)

                    all_news.append({
                        'ticker': ticker,
                        'date': date.isoformat(),
                        'headline': headline,
                        'sentiment': sentiment,
                        'sentiment_label': 'positive' if sentiment > 0.1 else 'negative' if sentiment < -0.1 else 'neutral'
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch news for {ticker}: {e}")
                continue

        # Sort by date (most recent first)
        all_news.sort(key=lambda x: x['date'], reverse=True)

        logger.info(f"Successfully analyzed {len(all_news)} news items with NLP")

        return jsonify({
            'status': 'success',
            'count': len(all_news),
            'news': all_news
        })

    except Exception as e:
        logger.error(f"Error fetching realtime news: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'error': str(e),
            'news': []
        }), 500


@app.route('/api/predict/<ticker>')
def predict(ticker):
    """Generate ML prediction and trading signal for ticker.

    For stocks with limited trading history (< 100 days), uses basic technical analysis
    instead of full ML models, following the 'short time list machine learning' approach.
    """
    account_size = request.args.get('account_size', 100000, type=float)
    original_ticker = ticker

    logger.info(f"[PREDICT] Received request for: {original_ticker}")

    # Try to resolve SH/SZ prefix format to Yahoo format (e.g., SH601127 -> 601127.SS)
    if ENHANCED_DATA_LAYER:
        resolved = resolve_china_stock(ticker)
        if resolved:
            logger.info(f"[PREDICT] Resolved {ticker} -> {resolved}")
            ticker = resolved
    else:
        logger.warning("[PREDICT] ENHANCED_DATA_LAYER is False - China ticker resolution disabled")

    # Normalize ticker format (e.g., 02590.HK -> 2590.HK)
    normalized = normalize_ticker(ticker.upper())
    if normalized != ticker.upper():
        logger.info(f"[PREDICT] Normalized {ticker} -> {normalized}")
    ticker = normalized

    logger.info(f"[PREDICT] Final ticker for prediction: {ticker}")

    # Check data availability first for tiered analysis
    try:
        test_data = fetch_data(ticker, lookback_days=1500)
        days_available = len(test_data) if test_data is not None else 0
        data_tier = get_data_tier(days_available)
        logger.info(f"[PREDICT] {ticker}: {days_available} days available, tier={data_tier['tier']}")
    except Exception as e:
        logger.warning(f"[PREDICT] Could not fetch data for {ticker}: {e}")
        days_available = 0
        data_tier = get_data_tier(0)

    # For stocks with limited data, use basic technical analysis
    # Need at least 150 days raw data because feature engineering drops ~30-40% of rows
    # (e.g., 100 days drops to ~67 after features, which is too small for LSTM training)
    use_basic_analysis = not data_tier.get('ml_available', True) or days_available < 150

    if use_basic_analysis and days_available >= 5:
        # Use basic technical analysis for short-listed/newly-listed stocks
        logger.info(f"[PREDICT] Using basic technical analysis for {ticker} (limited data: {days_available} days)")

        # Check if China stock - get DeepSeek analysis for limited data stocks too
        is_china_stock = ticker.endswith('.HK') or ticker.endswith('.SS') or ticker.endswith('.SZ')
        deepseek_analysis = None
        if is_china_stock:
            logger.info(f"[PREDICT] China stock with limited data: {ticker} - using DeepSeek integration")
            analyzer = get_deepseek_analyzer()
            deepseek_analysis = analyzer.get_comprehensive_analysis(ticker)

        try:
            basic_result = basic_technical_analysis(test_data, ticker)

            # Convert basic analysis to prediction format
            direction_map = {'BULLISH': 1, 'BEARISH': -1, 'NEUTRAL': 0}
            direction = direction_map.get(basic_result.get('signal', 'NEUTRAL'), 0)
            confidence = basic_result.get('confidence', 0.3)

            # Combine with DeepSeek for China stocks (50/50 weight for limited data)
            if is_china_stock and deepseek_analysis and deepseek_analysis.get('deepseek_used', False):
                ds_direction = deepseek_analysis.get('direction', 0)
                ds_confidence = deepseek_analysis.get('confidence', 0.5)
                combined_direction = 0.5 * ds_direction + 0.5 * direction
                direction = 1 if combined_direction > 0.2 else -1 if combined_direction < -0.2 else 0
                confidence = 0.5 * ds_confidence + 0.5 * confidence
                logger.info(f"[PREDICT] China {ticker} limited data: DeepSeek={ds_direction}/{ds_confidence:.2f}, Basic={direction}/{confidence:.2f}")
            volatility = 0.04  # Higher default volatility for new stocks
            expected_return = basic_result.get('momentum_5d', 0) / 100  # Convert percentage to decimal

            # Build prediction response in same format as generate_prediction()
            current_price = test_data['Close'].iloc[-1] if len(test_data) > 0 else 0

            # Get company info - try Yahoo Finance for China stocks not in database
            stock_info = COMPANY_DATABASE.get(ticker, None)
            if stock_info is None:
                # Try to fetch from Yahoo Finance
                try:
                    yf_ticker = yf.Ticker(ticker)
                    info = yf_ticker.info
                    company_name = info.get('longName') or info.get('shortName') or ticker
                    stock_info = {
                        'name': company_name,
                        'type': 'Stock',
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown')
                    }
                    logger.info(f"[BASIC ANALYSIS] Fetched company name from Yahoo: {ticker} -> {company_name}")
                except Exception as e:
                    logger.warning(f"[BASIC ANALYSIS] Could not fetch company info for {ticker}: {e}")
                    stock_info = {'name': ticker, 'type': 'Stock'}

            # Build chart data (last 250 days or all available)
            chart_days = min(250, len(test_data))
            chart_dates = [d.strftime('%Y-%m-%d') for d in test_data.index[-chart_days:]]
            chart_prices = test_data['Close'].iloc[-chart_days:].tolist()

            # Action based on direction
            action = 'BUY' if direction > 0 else 'SELL' if direction < 0 else 'HOLD'
            now = datetime.now()

            # Determine regime from signal
            regime = basic_result.get('signal', 'NEUTRAL').lower()
            if regime == 'bullish':
                regime = 'bullish'
            elif regime == 'bearish':
                regime = 'bearish'
            else:
                regime = 'neutral'

            # Determine exchange from ticker
            if ticker.endswith('.HK'):
                exchange = 'HKEX'
            elif ticker.endswith('.SS'):
                exchange = 'SSE'
            elif ticker.endswith('.SZ'):
                exchange = 'SZSE'
            else:
                exchange = 'Unknown'

            # Add exchange to stock_info for generate_news_feed
            stock_info['exchange'] = exchange

            prediction = {
                'status': 'success',
                'ticker': ticker,
                'timestamp': now.isoformat(),
                'current_price': float(current_price),  # Top-level current_price for frontend
                'company': {
                    'name': stock_info.get('name', ticker),
                    'type': stock_info.get('type', 'Stock'),
                    'sector': stock_info.get('sector', 'Unknown'),
                    'industry': stock_info.get('industry', 'Unknown'),
                    'exchange': exchange
                },
                'prediction': {
                    'volatility': volatility,
                    'volatility_percentile': 50,  # Neutral percentile for basic analysis
                    'historical_volatility': volatility,
                    'direction': direction,
                    'direction_confidence': confidence,
                    'regime': regime,
                    'expected_return': expected_return,
                    'confidence_interval': {
                        'lower': expected_return - volatility * 1.28,  # 80% CI centered on expected return
                        'upper': expected_return + volatility * 1.28   # 1.28 = z-score for 80% CI
                    }
                },
                # market_context is what frontend expects
                'market_context': {
                    'regime': regime,
                    'historical_volatility': volatility,
                    'volatility_percentile': 0.5  # 50th percentile as decimal
                },
                # trading_signal is what frontend expects
                'trading_signal': {
                    'action': action,
                    'confidence': confidence,
                    'reason': f"Basic technical analysis ({days_available} days): {basic_result.get('reason', 'Newly listed stock with limited data')}"
                },
                # Keep signal for backward compatibility
                'signal': {
                    'action': action,
                    'confidence': confidence,
                    'reason': f"Basic technical analysis ({days_available} days): {basic_result.get('reason', 'Newly listed stock with limited data')}"
                },
                'price_data': {
                    'current_price': current_price,
                    'timestamps': [int(ts.timestamp()) for ts in test_data.index[-60:]],
                    'prices': test_data['Close'].iloc[-60:].tolist()
                },
                'chart_data': {
                    'dates': chart_dates,
                    'prices': chart_prices
                },
                'position': {
                    'shares': int(account_size * 0.02 / current_price) if current_price > 0 else 0,
                    'entry_price': current_price,
                    'stop_loss': current_price * (1 - 0.05),  # 5% stop loss
                    'take_profit': current_price * (1 + 0.10),  # 10% take profit
                    'risk_amount': account_size * 0.02,
                    'potential_profit': account_size * 0.04,  # 2x risk
                    'risk_reward_ratio': 2.0
                },
                'model_info': {
                    'type': f'BasicTechnicalAnalysis ({days_available} days)',
                    'features_count': 5,
                    'trained_at': now.isoformat(),
                    'data_tier': data_tier['tier'],
                    'days_available': days_available,
                    'note': 'Limited data - using simplified technical analysis instead of full ML'
                },
                'news_feed': []  # Will be populated below
            }

            # Add DeepSeek metadata for China stocks
            if is_china_stock and deepseek_analysis:
                prediction['deepseek_analysis'] = {
                    'deepseek_used': deepseek_analysis.get('deepseek_used', False),
                    'policy_sentiment': deepseek_analysis.get('policy_sentiment', 0),
                    'social_sentiment': deepseek_analysis.get('social_sentiment', 0),
                    'analysis_reason': deepseek_analysis.get('reason', '')
                }

            # Add quality filter checks for China stocks (Fixing2 display)
            if is_china_stock:
                try:
                    quality_checks = get_quality_filter_checks(ticker, test_data)
                    if quality_checks:
                        prediction['quality_filter'] = quality_checks
                        logger.info(f"[BASIC ANALYSIS] China {ticker}: Quality filter overall_pass={quality_checks['overall_pass']}")
                except Exception as e:
                    logger.warning(f"[BASIC ANALYSIS] Could not get quality filter for {ticker}: {e}")

            # Generate news feed for basic analysis too
            try:
                news_feed = generate_news_feed(ticker, stock_info, current_price, prediction)
                prediction['news_feed'] = news_feed
                logger.info(f"[BASIC ANALYSIS] Generated {len(news_feed)} news items for {ticker}")
            except Exception as e:
                logger.warning(f"[BASIC ANALYSIS] Could not generate news for {ticker}: {e}")
                prediction['news_feed'] = []

            return jsonify(prediction)

        except Exception as e:
            logger.error(f"[PREDICT] Basic analysis failed for {ticker}: {e}")
            return jsonify({
                'status': 'error',
                'ticker': ticker,
                'error': f"Basic analysis failed: {str(e)}"
            }), 500

    elif days_available < 5:
        # Extremely limited data - cannot generate prediction
        logger.warning(f"[PREDICT] Extremely limited data for {ticker} ({days_available} days)")
        return jsonify({
            'status': 'error',
            'ticker': ticker,
            'error': f"Insufficient data ({days_available} days). Need at least 5 trading days."
        }), 400

    # Check if this is a China stock - use DeepSeek integration for China markets
    is_china_stock = ticker.endswith('.HK') or ticker.endswith('.SS') or ticker.endswith('.SZ')

    if is_china_stock:
        # Use DeepSeek + ML combination for China stocks (as designed)
        logger.info(f"[PREDICT] China stock detected: {ticker} - using DeepSeek integration")

        # Get DeepSeek analysis
        analyzer = get_deepseek_analyzer()
        deepseek_analysis = analyzer.get_comprehensive_analysis(ticker)

        # CRITICAL FIX: Use lock to prevent race condition with parallel top-picks processing
        with PREDICTION_LOCK:
            prediction = generate_prediction(ticker, account_size)

        if prediction.get('status') == 'success':
            # Enhance prediction with DeepSeek data
            if deepseek_analysis.get('deepseek_used', False):
                # Combine DeepSeek and ML predictions (40% DeepSeek + 60% ML)
                ml_direction = prediction['prediction']['direction']
                ml_confidence = prediction['prediction']['direction_confidence']
                ds_direction = deepseek_analysis.get('direction', 0)
                ds_confidence = deepseek_analysis.get('confidence', 0.5)

                # Weighted combination
                combined_direction = 0.4 * ds_direction + 0.6 * ml_direction
                final_direction = 1 if combined_direction > 0.2 else -1 if combined_direction < -0.2 else 0
                final_confidence = 0.4 * ds_confidence + 0.6 * ml_confidence

                # Update prediction with combined signals
                prediction['prediction']['direction'] = final_direction
                prediction['prediction']['direction_confidence'] = final_confidence

                # Update trading signal action
                action = 'BUY' if final_direction > 0 else 'SELL' if final_direction < 0 else 'HOLD'
                if 'trading_signal' in prediction:
                    prediction['trading_signal']['action'] = action
                    prediction['trading_signal']['confidence'] = final_confidence
                    prediction['trading_signal']['reason'] = f"Combined DeepSeek ({ds_confidence:.0%}) + ML ({ml_confidence:.0%}) analysis"

                logger.info(f"[PREDICT] China {ticker}: DeepSeek={ds_direction}/{ds_confidence:.2f}, ML={ml_direction}/{ml_confidence:.2f} -> {final_direction}/{final_confidence:.2f}")

            # Add DeepSeek metadata to prediction
            prediction['deepseek_analysis'] = {
                'deepseek_used': deepseek_analysis.get('deepseek_used', False),
                'policy_sentiment': deepseek_analysis.get('policy_sentiment', 0),
                'social_sentiment': deepseek_analysis.get('social_sentiment', 0),
                'analysis_reason': deepseek_analysis.get('reason', '')
            }

            # Add quality filter checks for China Model Fixing2 display
            try:
                # Fetch recent price data for quality filter calculations
                qf_data = fetch_data(ticker, lookback_days=30)
                quality_checks = get_quality_filter_checks(ticker, qf_data)
                if quality_checks:
                    prediction['quality_filter'] = quality_checks
                    logger.info(f"[PREDICT] China {ticker}: Quality filter overall_pass={quality_checks['overall_pass']}")
            except Exception as e:
                logger.warning(f"[PREDICT] Could not get quality filter for {ticker}: {e}")
    else:
        # CRITICAL FIX: Use lock to prevent race condition with parallel top-picks processing
        # Without this lock, concurrent requests can cause results to be returned for wrong tickers
        # (e.g., META returning 3690.HK Meituan data, TM returning 300059.SZ East Money data)
        with PREDICTION_LOCK:
            prediction = generate_prediction(ticker, account_size)

        # Add quality filter checks for US/Intl stocks too
        if prediction.get('status') == 'success':
            try:
                qf_data = fetch_data(ticker, lookback_days=30)
                quality_checks = get_quality_filter_checks(ticker, qf_data)
                if quality_checks:
                    prediction['quality_filter'] = quality_checks
                    logger.info(f"[PREDICT] US/Intl {ticker}: Quality filter overall_pass={quality_checks['overall_pass']}")
            except Exception as e:
                logger.warning(f"[PREDICT] Could not get quality filter for {ticker}: {e}")

    # Return proper HTTP status code based on prediction status
    if prediction.get('status') == 'error':
        return jsonify(prediction), 500

    return jsonify(prediction)


@app.route('/api/intraday/<ticker>')
def get_intraday_data(ticker):
    """Get 1-day intraday data for a ticker (5-minute intervals)."""
    try:
        # Try to resolve SH/SZ prefix format to Yahoo format (e.g., SH601127 -> 601127.SS)
        if ENHANCED_DATA_LAYER:
            resolved = resolve_china_stock(ticker)
            if resolved:
                ticker = resolved

        # Normalize ticker format
        ticker = normalize_ticker(ticker.upper())

        logger.info(f"Fetching intraday data for {ticker}")

        # Fetch 1-day intraday data with 5-minute intervals
        yf_ticker = yf.Ticker(ticker)
        data = yf_ticker.history(period='1d', interval='5m')

        if data.empty:
            logger.warning(f"No intraday data available for {ticker}")
            return jsonify({
                'status': 'error',
                'error': 'No intraday data available (market may be closed)'
            }), 404

        # Convert to JSON-friendly format
        timestamps = [int(ts.timestamp()) for ts in data.index]
        prices = data['Close'].tolist()

        return jsonify({
            'status': 'success',
            'ticker': ticker,
            'timestamps': timestamps,
            'prices': prices,
            'data_points': len(timestamps)
        })

    except Exception as e:
        logger.error(f"Failed to fetch intraday data for {ticker}: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_cached': len(MODEL_CACHE),
        'dual_model_system': USE_DUAL_MODEL_SYSTEM
    })


@app.route('/api/market_info/<ticker>')
def get_market_info(ticker):
    """
    Get market classification and expected performance for a ticker.

    NEW: Dual model system - Shows which model will be used and expected performance.
    """
    try:
        # Try to resolve SH/SZ prefix format to Yahoo format (e.g., SH601127 -> 601127.SS)
        if ENHANCED_DATA_LAYER:
            resolved = resolve_china_stock(ticker)
            if resolved:
                ticker = resolved

        ticker = ticker.upper()

        if not USE_DUAL_MODEL_SYSTEM:
            return jsonify({
                'ticker': ticker,
                'dual_model_system': False,
                'market': 'us_international',
                'model_type': 'US/International Model',
                'macro_features': 'VIX, SPY, DXY, GLD',
                'message': 'Dual model system disabled - using US/Intl model for all markets'
            })

        # Get market classification
        market_info = MarketClassifier.get_market_details(ticker)
        performance_info = MarketClassifier.get_performance_expectations(ticker)

        return jsonify({
            'ticker': ticker,
            'dual_model_system': True,
            'market': market_info['market'],
            'exchange': market_info['exchange'],
            'model_type': market_info['model_type'],
            'macro_features': market_info['macro_features'],
            'expected_profitability': performance_info['expected_profitability'],
            'confidence': performance_info['confidence'],
            'recommendation': performance_info['recommendation']
        })

    except Exception as e:
        logger.error(f"Market info error for {ticker}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/monitoring/summary')
def get_monitoring_summary():
    """Get comprehensive monitoring summary"""
    data_monitor = get_data_quality_monitor()
    cache_monitor = get_cache_monitor()

    summary = {
        'data_quality': data_monitor.get_summary(),
        'cache_performance': cache_monitor.get_stats(),
        'timestamp': datetime.now().isoformat()
    }

    return jsonify(summary)


def _generate_pick_for_ticker(ticker):
    """
    Thread-safe helper function to generate prediction for a single ticker.
    Used for parallel processing in top_picks endpoint.

    Uses PREDICTION_LOCK to prevent race conditions where results from one
    ticker could be returned for another ticker during parallel processing.
    """
    try:
        # Generate prediction using the full ML pipeline
        # CRITICAL: Use lock to prevent race condition where TM shows East Money data
        with PREDICTION_LOCK:
            prediction = generate_prediction(ticker, account_size=100000)

        if prediction.get('status') == 'error':
            logger.warning(f"Skipping {ticker} due to error: {prediction.get('error', 'Unknown error')}")
            return None

        signal = prediction['trading_signal']
        direction = prediction['prediction']['direction']
        direction_conf = prediction['prediction']['direction_confidence']
        predicted_return = prediction['prediction']['expected_return']
        volatility = prediction['prediction']['volatility']

        # Safety check for None values
        if volatility is None:
            volatility = 0.01
        if predicted_return is None:
            predicted_return = 0.0
        if direction_conf is None:
            direction_conf = 0.5

        # Calculate risk-adjusted score: (Return × Confidence) / Volatility
        # This prioritizes high-return, high-confidence, low-volatility opportunities
        # Avoid division by zero
        risk_adjusted_score = (abs(predicted_return) * direction_conf) / max(volatility, 0.01)

        # Build signal string for display (e.g., "BUY 65%" or "SELL 42%")
        direction_label = "BUY" if direction > 0 else "SELL" if direction < 0 else "HOLD"
        signal_display = f"{direction_label} {direction_conf*100:.0f}%"

        # Get company info - use database if available, otherwise fetch from Yahoo Finance
        if ticker in COMPANY_DATABASE:
            company_name = COMPANY_DATABASE[ticker]['name']
            asset_type = COMPANY_DATABASE[ticker]['type']
        else:
            # Ticker from screener - fetch info from Yahoo Finance
            try:
                stock_info = yf.Ticker(ticker).info
                company_name = stock_info.get('longName') or stock_info.get('shortName') or ticker
                quote_type = stock_info.get('quoteType', '').upper()

                # Determine asset type from ticker pattern and Yahoo quoteType
                if ticker.endswith('.HK') or ticker.endswith('.SS') or ticker.endswith('.SZ'):
                    asset_type = 'Stock'  # China stock
                elif '-USD' in ticker or quote_type == 'CRYPTOCURRENCY':
                    asset_type = 'Cryptocurrency'
                elif '=F' in ticker or quote_type == 'FUTURE':
                    asset_type = 'Commodity'
                elif '=X' in ticker or quote_type == 'CURRENCY':
                    asset_type = 'Forex'
                elif quote_type == 'ETF':
                    asset_type = 'ETF'  # Identify ETFs from Yahoo quoteType
                else:
                    asset_type = 'Stock'  # Default to stock
            except Exception:
                company_name = ticker
                asset_type = 'Stock'

        # Build pick data
        pick_data = {
            'ticker': ticker,
            'name': company_name,
            'type': asset_type,
            'action': signal['action'],
            'signal': signal_display,  # User-friendly display string
            'confidence': direction_conf,  # TRUE ML confidence (not inflated)
            'volatility': volatility,
            'direction': direction,
            'direction_confidence': direction_conf,
            'expected_return': predicted_return,
            'risk_adjusted_score': risk_adjusted_score  # Risk-adjusted ranking metric
        }

        return pick_data

    except Exception as e:
        logger.error(f"Error generating pick for {ticker}: {str(e)}")
        return None


# ============================================================================
# DEEPSEEK CHINA ANALYZER - Alternative Data for China Markets
# ============================================================================

class DeepSeekChinaAnalyzer:
    """
    DeepSeek API integration for China-specific alternative data.
    Provides policy sentiment, social sentiment, and retail investor analysis.
    """

    STOCK_NAMES = {
        '0700.HK': 'Tencent Holdings (腾讯控股)',
        '9988.HK': 'Alibaba Group (阿里巴巴)',
        '3690.HK': 'Meituan (美团)',
        '2318.HK': 'Ping An Insurance (中国平安)',
        '1810.HK': 'Xiaomi Corporation (小米)',
        '9618.HK': 'JD.com (京东)',
        '9888.HK': 'Baidu (百度)',
        '1211.HK': 'BYD Company (比亚迪)',
        '0939.HK': 'CCB (中国建设银行)',
        '1398.HK': 'ICBC (中国工商银行)',
        '2269.HK': 'WuXi Biologics (药明生物)',
        '1109.HK': 'China Resources Land (华润置地)',
        '600519.SS': 'Kweichow Moutai (贵州茅台)',
        '601318.SS': 'Ping An Insurance (中国平安)',
        '300750.SZ': 'CATL (宁德时代)',
        '002594.SZ': 'BYD Company (比亚迪)',
        '000858.SZ': 'Wuliangye (五粮液)',
        '000333.SZ': 'Midea Group (美的集团)',
    }

    SECTOR_INFO = {
        '0700.HK': 'Technology/Gaming',
        '9988.HK': 'Technology/E-commerce',
        '3690.HK': 'Technology/E-commerce',
        '2318.HK': 'Financial Services/Insurance',
        '1810.HK': 'Technology/Consumer Electronics',
        '9618.HK': 'Technology/E-commerce',
        '9888.HK': 'Technology/AI',
        '1211.HK': 'Automotive/EV',
        '0939.HK': 'Banking',
        '1398.HK': 'Banking',
        '2269.HK': 'Healthcare/Biotech',
        '1109.HK': 'Real Estate',
        '600519.SS': 'Consumer/Beverage',
        '601318.SS': 'Financial Services/Insurance',
        '300750.SZ': 'Technology/Battery',
        '002594.SZ': 'Automotive/EV',
        '000858.SZ': 'Consumer/Beverage',
        '000333.SZ': 'Consumer/Appliances',
    }

    def __init__(self, api_key=None):
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.api_key = api_key or os.environ.get('DEEPSEEK_API_KEY')
        self.sentiment_cache = {}

        if self.api_key:
            logger.info("[DEEPSEEK] API key configured")
        else:
            logger.warning("[DEEPSEEK] No API key found - will use fallback analysis")

    def _call_api(self, prompt, temperature=0.1, max_tokens=500):
        """Call DeepSeek API with the given prompt."""
        if not self.api_key:
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a financial analyst specializing in Chinese markets. Respond concisely with numerical scores as requested."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"[DEEPSEEK] API Error: {e}")
            return None

    def _parse_score(self, response, default=0.0, min_val=-1.0, max_val=1.0):
        """Parse a numerical score from API response."""
        if not response:
            return default

        import re
        numbers = re.findall(r'[-+]?\d*\.?\d+', response)
        if numbers:
            try:
                score = float(numbers[0])
                if score > 10:
                    score = score / 100
                elif score > 1 and max_val == 1:
                    score = score / 10
                return max(min_val, min(max_val, score))
            except ValueError:
                pass
        return default

    def get_comprehensive_analysis(self, ticker):
        """
        Get comprehensive China market analysis using DeepSeek API.

        Returns:
            dict: Analysis including direction, confidence, sentiment scores
        """
        stock_name = self.STOCK_NAMES.get(ticker, ticker)
        sector = self.SECTOR_INFO.get(ticker, 'Unknown')

        # Check cache first
        cache_key = f"analysis_{ticker}"
        if cache_key in self.sentiment_cache:
            cached_time = self.sentiment_cache.get(f"{cache_key}_time", datetime.min)
            if (datetime.now() - cached_time).total_seconds() < 3600:  # 1 hour cache
                logger.info(f"[DEEPSEEK] Using cached analysis for {ticker}")
                return self.sentiment_cache[cache_key]

        if not self.api_key:
            # Fallback to neutral analysis without API
            return {
                'direction': 0,
                'confidence': 0.5,
                'policy_sentiment': 0.0,
                'social_sentiment': 0.0,
                'retail_sentiment': 50.0,
                'policy_alignment': 5.0,
                'deepseek_used': False,
                'reason': 'DeepSeek API not available - using technical analysis'
            }

        # Comprehensive analysis prompt
        prompt = f"""Analyze {stock_name} ({ticker}) in the {sector} sector for trading signal.

Provide a comprehensive analysis considering:
1. Current Chinese government policy environment for {sector}
2. Recent PBOC monetary policy and CSRC regulations
3. Social media sentiment (Weibo, Xueqiu, East Money)
4. Retail investor behavior and sentiment in China
5. Alignment with China's 14th Five-Year Plan priorities

Based on your analysis, provide:
- DIRECTION: -1 (bearish), 0 (neutral), or 1 (bullish)
- CONFIDENCE: 0.0 to 1.0 (how confident in the direction)
- POLICY_SENTIMENT: -1.0 to 1.0 (policy environment)
- SOCIAL_SENTIMENT: -1.0 to 1.0 (social media mood)
- RETAIL_SENTIMENT: 0 to 100 (retail investor mood, 50=neutral)
- POLICY_ALIGNMENT: 0 to 10 (alignment with government priorities)
- REASON: Brief explanation

Format your response as:
DIRECTION: [number]
CONFIDENCE: [number]
POLICY_SENTIMENT: [number]
SOCIAL_SENTIMENT: [number]
RETAIL_SENTIMENT: [number]
POLICY_ALIGNMENT: [number]
REASON: [text]"""

        response = self._call_api(prompt, max_tokens=800)

        if response:
            try:
                lines = response.strip().split('\n')
                result = {
                    'direction': 0,
                    'confidence': 0.5,
                    'policy_sentiment': 0.0,
                    'social_sentiment': 0.0,
                    'retail_sentiment': 50.0,
                    'policy_alignment': 5.0,
                    'deepseek_used': True,
                    'reason': ''
                }

                for line in lines:
                    line = line.strip()
                    if line.startswith('DIRECTION:'):
                        result['direction'] = int(self._parse_score(line.split(':')[1], 0, -1, 1))
                    elif line.startswith('CONFIDENCE:'):
                        result['confidence'] = self._parse_score(line.split(':')[1], 0.5, 0, 1)
                    elif line.startswith('POLICY_SENTIMENT:'):
                        result['policy_sentiment'] = self._parse_score(line.split(':')[1], 0, -1, 1)
                    elif line.startswith('SOCIAL_SENTIMENT:'):
                        result['social_sentiment'] = self._parse_score(line.split(':')[1], 0, -1, 1)
                    elif line.startswith('RETAIL_SENTIMENT:'):
                        result['retail_sentiment'] = self._parse_score(line.split(':')[1], 50, 0, 100)
                    elif line.startswith('POLICY_ALIGNMENT:'):
                        result['policy_alignment'] = self._parse_score(line.split(':')[1], 5, 0, 10)
                    elif line.startswith('REASON:'):
                        result['reason'] = line.split(':', 1)[1].strip()

                # Cache the result
                self.sentiment_cache[cache_key] = result
                self.sentiment_cache[f"{cache_key}_time"] = datetime.now()

                logger.info(f"[DEEPSEEK] Analysis for {ticker}: direction={result['direction']}, confidence={result['confidence']:.2f}")
                return result

            except Exception as e:
                logger.error(f"[DEEPSEEK] Failed to parse response for {ticker}: {e}")

        # Fallback
        return {
            'direction': 0,
            'confidence': 0.5,
            'policy_sentiment': 0.0,
            'social_sentiment': 0.0,
            'retail_sentiment': 50.0,
            'policy_alignment': 5.0,
            'deepseek_used': False,
            'reason': 'Analysis failed - using neutral fallback'
        }


# Global DeepSeek analyzer instance
DEEPSEEK_ANALYZER = None

def get_deepseek_analyzer():
    """Get or initialize the DeepSeek analyzer."""
    global DEEPSEEK_ANALYZER
    if DEEPSEEK_ANALYZER is None:
        DEEPSEEK_ANALYZER = DeepSeekChinaAnalyzer()
    return DEEPSEEK_ANALYZER


def _generate_china_pick_for_ticker(ticker):
    """
    Generate prediction for a China market ticker using DeepSeek API + ML model.

    This combines:
    1. DeepSeek API for policy/social sentiment analysis
    2. China-specific ML model for technical prediction (or basic technical analysis for limited data)
    3. Weighted ensemble of both signals

    For stocks with limited trading history (< 250 days), uses simplified analysis
    per the "short time list machine learning" recommendations.
    """
    try:
        logger.info(f"[CHINA PICK] Generating prediction for {ticker}")

        # Get DeepSeek analysis
        analyzer = get_deepseek_analyzer()
        deepseek_analysis = analyzer.get_comprehensive_analysis(ticker)

        # Check data availability first for tiered analysis
        try:
            test_data = fetch_data(ticker, lookback_days=1500)
            days_available = len(test_data) if test_data is not None else 0
            data_tier = get_data_tier(days_available)
            logger.info(f"[CHINA PICK] {ticker}: {days_available} days available, tier={data_tier['tier']}")
        except Exception as e:
            logger.warning(f"[CHINA PICK] Could not fetch data for {ticker}: {e}")
            days_available = 0
            data_tier = get_data_tier(0)

        # For stocks with limited data, use basic technical analysis instead of full ML
        # Need at least 150 days raw data because feature engineering drops ~30-40% of rows
        use_basic_analysis = not data_tier.get('ml_available', True) or days_available < 150

        if use_basic_analysis and days_available >= 5:
            # Use basic technical analysis for short-listed/newly-listed stocks
            logger.info(f"[CHINA PICK] Using basic technical analysis for {ticker} (limited data: {days_available} days)")
            basic_result = basic_technical_analysis(test_data, ticker)

            # Convert basic analysis to pick format
            direction_map = {'BULLISH': 1, 'BEARISH': -1, 'NEUTRAL': 0}
            ml_direction = direction_map.get(basic_result.get('signal', 'NEUTRAL'), 0)
            ml_confidence = basic_result.get('confidence', 0.3)
            ml_return = basic_result.get('momentum_5d', 0) / 100  # Convert percentage to decimal
            volatility = 0.04  # Higher default volatility for new stocks

            # Combine with DeepSeek (weight DeepSeek more for limited data stocks - 50/50)
            ds_direction = deepseek_analysis['direction']
            ds_confidence = deepseek_analysis['confidence']

            if deepseek_analysis['deepseek_used']:
                combined_direction = 0.5 * ds_direction + 0.5 * ml_direction
                direction = 1 if combined_direction > 0.2 else -1 if combined_direction < -0.2 else 0
                confidence = 0.5 * ds_confidence + 0.5 * ml_confidence
                predicted_return = ml_return if abs(ml_return) > 0.001 else direction * 0.015
                logger.info(f"[CHINA PICK] Limited data combined: DeepSeek={ds_direction}/{ds_confidence:.2f}, Basic={ml_direction}/{ml_confidence:.2f} -> {direction}/{confidence:.2f}")
            else:
                direction = ml_direction
                confidence = ml_confidence
                predicted_return = ml_return if abs(ml_return) > 0.001 else direction * 0.015

        elif days_available < 5:
            # Extremely limited data - use DeepSeek only
            logger.warning(f"[CHINA PICK] Extremely limited data for {ticker} ({days_available} days), using DeepSeek only")
            direction = deepseek_analysis['direction']
            confidence = deepseek_analysis['confidence'] * 0.5  # Reduce confidence
            predicted_return = direction * 0.01
            volatility = 0.05

        else:
            # Sufficient data - use full ML prediction
            with PREDICTION_LOCK:
                ml_prediction = generate_prediction(ticker, account_size=100000)

            if ml_prediction.get('status') == 'error':
                # If ML fails, use only DeepSeek analysis
                logger.warning(f"[CHINA PICK] ML prediction failed for {ticker}, using DeepSeek only")
                direction = deepseek_analysis['direction']
                confidence = deepseek_analysis['confidence']
                predicted_return = direction * 0.02  # Assume 2% move
                volatility = 0.03  # Default volatility
            else:
                # Combine DeepSeek and ML predictions
                ml_direction = ml_prediction['prediction']['direction']
                ml_confidence = ml_prediction['prediction']['direction_confidence']
                ml_return = ml_prediction['prediction']['expected_return']
                volatility = ml_prediction['prediction']['volatility'] or 0.03

                ds_direction = deepseek_analysis['direction']
                ds_confidence = deepseek_analysis['confidence']

                # Weighted combination: 40% DeepSeek + 60% ML
                if deepseek_analysis['deepseek_used']:
                    combined_direction = 0.4 * ds_direction + 0.6 * ml_direction
                    direction = 1 if combined_direction > 0.2 else -1 if combined_direction < -0.2 else 0
                    confidence = 0.4 * ds_confidence + 0.6 * ml_confidence
                    predicted_return = ml_return
                    logger.info(f"[CHINA PICK] Combined: DeepSeek={ds_direction}/{ds_confidence:.2f}, ML={ml_direction}/{ml_confidence:.2f} -> {direction}/{confidence:.2f}")
                else:
                    # DeepSeek not available, use ML only
                    direction = ml_direction
                    confidence = ml_confidence
                    predicted_return = ml_return

        # Calculate risk-adjusted score
        risk_adjusted_score = (abs(predicted_return) * confidence) / max(volatility, 0.01)

        # Build signal string
        direction_label = "BUY" if direction > 0 else "SELL" if direction < 0 else "HOLD"
        signal_display = f"{direction_label} {confidence*100:.0f}%"

        # Get stock info from Yahoo Finance (real-time only)
        try:
            yf_info = yf.Ticker(ticker).info
            company_name = yf_info.get('longName') or yf_info.get('shortName') or ticker
        except Exception:
            company_name = ticker

        pick_data = {
            'ticker': ticker,
            'name': company_name,
            'type': 'Stock',  # China stocks are always type Stock
            'action': direction_label,
            'signal': signal_display,
            'confidence': confidence,
            'volatility': volatility,
            'direction': direction,
            'direction_confidence': confidence,
            'expected_return': predicted_return,
            'risk_adjusted_score': risk_adjusted_score,
            # DeepSeek-specific data
            'deepseek_used': deepseek_analysis.get('deepseek_used', False),
            'policy_sentiment': deepseek_analysis.get('policy_sentiment', 0),
            'social_sentiment': deepseek_analysis.get('social_sentiment', 0),
            'analysis_reason': deepseek_analysis.get('reason', '')
        }

        return pick_data

    except Exception as e:
        logger.error(f"[CHINA PICK] Error generating pick for {ticker}: {str(e)}")
        return None


# Cache for top-picks results - DYNAMIC TTL based on regime volatility
TOP_PICKS_CACHE = {}
TOP_PICKS_CACHE_TTL = {
    'Stock': 1800,           # 30 minutes for stocks (moderate volatility)
    'Cryptocurrency': 600,   # 10 minutes for crypto (high volatility, fast-moving)
    'China': 1800,           # 30 minutes for China stocks
    'Commodity': 3600,       # 1 hour for commodities (slower moving)
    'Forex': 1800,           # 30 minutes for forex (moderate volatility)
    'all': 1800,             # 30 minutes for mixed view
    'default': 1800          # Default 30 minutes
}

def _get_cache_ttl(regime):
    """Get cache TTL based on regime volatility - shorter TTL for high volatility assets"""
    return TOP_PICKS_CACHE_TTL.get(regime, TOP_PICKS_CACHE_TTL['default'])

@app.route('/api/top-picks')
def top_picks():
    """Generate top 10 BUY and top 10 SELL signals using REAL-TIME Yahoo Finance screeners.

    Features:
    - Real-time ticker discovery using Yahoo Finance screeners (day_gainers, most_actives, etc.)
    - Automatic fallback to database when screeners fail
    - Dynamic cache TTL based on regime volatility
    - Strict filtering: Stock page only shows stocks, Crypto only shows crypto, etc.
    """
    regime = request.args.get('regime', 'all')  # all, Stock, Cryptocurrency, Commodity, Forex, China
    force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
    use_screeners = request.args.get('use_screeners', 'true').lower() == 'true'

    # Get dynamic cache TTL for this regime
    cache_ttl = _get_cache_ttl(regime)

    # Check cache first - return cached results if still valid (unless force refresh)
    cache_key = f"top_picks_{regime}"
    if not force_refresh and cache_key in TOP_PICKS_CACHE:
        cached_data, cached_time = TOP_PICKS_CACHE[cache_key]
        age_seconds = (datetime.now() - cached_time).total_seconds()
        if age_seconds < cache_ttl:
            logger.info(f"[CACHE HIT] Returning cached top-picks for {regime} (age: {age_seconds:.0f}s, TTL: {cache_ttl}s)")
            cached_data['from_cache'] = True
            cached_data['cache_age_seconds'] = int(age_seconds)
            cached_data['cache_ttl_seconds'] = cache_ttl

            # Always add fresh market_trend_adaptation (even for cached data)
            # This ensures trend info is always current since market trends change
            if PROFIT_OPTIMIZER_AVAILABLE and regime != 'China':
                try:
                    from src.trading.integrated_us_optimizer import create_integrated_optimizer
                    temp_optimizer = create_integrated_optimizer(
                        enable_energy=False, enable_commodity=False,
                        enable_dividend=False, enable_futures=False,
                        enable_trend=True  # Only need trend adapter
                    )
                    trend_info = temp_optimizer.get_current_trend()
                    cached_data['market_trend_adaptation'] = {
                        'enabled': True,
                        'current_trend': trend_info.get('trend', 'unknown'),
                        'trend_description': trend_info.get('description', ''),
                        'buy_win_rate': trend_info.get('buy_win_rate', 0.5),
                        'sell_win_rate': trend_info.get('sell_win_rate', 0.5),
                        'recommendation': trend_info.get('recommendation', 'Balanced'),
                        'from_cache': True,  # Mark as added to cached response
                        'message': f"Market trend: {trend_info.get('trend', 'unknown')} - {trend_info.get('recommendation', 'Balanced')}"
                    }
                    logger.info(f"[CACHE HIT] Added market_trend_adaptation: {trend_info.get('trend', 'unknown')}")
                except Exception as e:
                    logger.debug(f"[CACHE HIT] Could not add trend info: {e}")

            return jsonify(cached_data)
        else:
            logger.info(f"[CACHE EXPIRED] Cache for {regime} is {age_seconds:.0f}s old (TTL: {cache_ttl}s), refreshing...")

    if force_refresh:
        logger.info(f"[FORCE REFRESH] User requested manual refresh for regime: {regime}")
    else:
        logger.info(f"[CACHE MISS] Generating fresh top picks for regime: {regime}")

    # ============================================================================
    # REAL-TIME TICKER DISCOVERY using Yahoo Finance Screeners
    # ============================================================================
    ticker_source = 'database'  # Track where tickers came from

    # Try real-time screeners first (if available and enabled)
    if SCREENER_DISCOVERY_AVAILABLE and use_screeners:
        try:
            reliability_mgr = get_reliability_manager()

            # Check if screeners are reliable for this regime
            if reliability_mgr.should_use_screeners(regime):
                logger.info(f"[SCREENER] Using real-time Yahoo Finance screeners for {regime}")

                strategy = get_regime_strategy()
                screener_tickers, source = strategy.get_tickers_for_regime(regime, count=25)

                # IPO regimes have lower minimum threshold since IPOs are rare
                # For IPO regimes, 0 results is valid (no current IPOs) - don't throw error
                min_tickers = 1 if regime in ['US_IPO', 'China_IPO'] else 5
                is_ipo_regime = regime in ['US_IPO', 'China_IPO']

                if screener_tickers and len(screener_tickers) >= min_tickers:
                    tickers = screener_tickers
                    ticker_source = source
                    reliability_mgr.track_performance(regime, True)
                    logger.info(f"[SCREENER SUCCESS] Found {len(tickers)} real-time tickers from {source}")
                elif is_ipo_regime:
                    # For IPO regimes, 0 results is valid - no current IPOs in market
                    tickers = screener_tickers if screener_tickers else []
                    ticker_source = source
                    reliability_mgr.track_performance(regime, True)
                    logger.info(f"[IPO REGIME] Found {len(tickers)} actual IPOs - this is valid (IPOs are rare)")
                else:
                    raise Exception(f"Insufficient tickers from screener: {len(screener_tickers) if screener_tickers else 0}")
            else:
                logger.info(f"[SCREENER SKIP] Reliability too low for {regime}, using database fallback")
                raise Exception("Screener reliability below threshold")

        except Exception as e:
            logger.warning(f"[SCREENER FALLBACK] Screener failed for {regime}: {e}, using database")
            if SCREENER_DISCOVERY_AVAILABLE:
                reliability_mgr = get_reliability_manager()
                reliability_mgr.track_performance(regime, False)
            # Fall through to database selection below
            tickers = None
    else:
        tickers = None
        if not SCREENER_DISCOVERY_AVAILABLE:
            logger.warning("[SCREENER] Real-time discovery module not available")
        else:
            logger.warning("[SCREENER] Screeners disabled by user request")

    # ============================================================================
    # HONEST ERROR - No hardcoded fallback, real-time data only
    # ============================================================================
    if tickers is None or len(tickers) == 0:
        error_msg = (
            f"Unable to fetch real-time market data for {regime}. "
            "Yahoo Finance screeners are temporarily unavailable. "
            "Please try again in a few minutes."
        )
        logger.error(f"[SCREENER FAILURE] {error_msg}")
        return jsonify({
            'status': 'error',
            'error': error_msg,
            'regime': regime,
            'ticker_source': 'none',
            'screener_available': SCREENER_DISCOVERY_AVAILABLE,
            'message': 'Real-time data unavailable. We only use live Yahoo Finance data - no hardcoded fallbacks.',
            'timestamp': datetime.now().isoformat()
        }), 503  # Service Unavailable

    logger.info(f"Processing {len(tickers)} tickers in parallel with {min(5, len(tickers))} workers")

    all_predictions = []

    # Choose the appropriate prediction function based on regime
    # China regime uses DeepSeek API + China ML model
    # IPO regimes use respective prediction functions with IPO tier info
    if regime == 'China' or regime == 'China_IPO':
        prediction_func = _generate_china_pick_for_ticker
        logger.info(f"[{regime}] Using DeepSeek + China ML model for predictions")
    else:
        prediction_func = _generate_pick_for_ticker

    # Use ThreadPoolExecutor to process tickers in parallel
    # Limit max_workers to 5 to avoid overwhelming the system
    # For China regime, use fewer workers to avoid API rate limiting
    max_workers = 3 if regime in ['China', 'China_IPO'] else min(5, len(tickers))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all ticker prediction tasks
        future_to_ticker = {
            executor.submit(prediction_func, ticker): ticker
            for ticker in tickers
        }

        # Collect results as they complete
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                pick_data = future.result()
                if pick_data:  # Only add successful predictions
                    all_predictions.append(pick_data)
                    logger.info(f"Completed prediction for {ticker} ({len(all_predictions)}/{len(tickers)})")
            except Exception as e:
                logger.error(f"Exception processing {ticker}: {str(e)}")

    logger.info(f"Completed {len(all_predictions)} predictions out of {len(tickers)} tickers")

    # STRICT CLASSIFICATION: Only use clear directional signals
    # Direction > 0 = BUY (bullish), Direction < 0 = SELL (bearish)
    # Direction == 0 (neutral/HOLD) = SKIP - do NOT include in top picks lists
    # This prevents confusing situations where a HOLD signal appears in BUY/SELL lists
    bullish = []
    bearish = []

    # DEBUG: Log direction values for first 5 predictions
    for i, p in enumerate(all_predictions[:5]):
        logger.info(f"[DEBUG] Prediction {i+1}: ticker={p.get('ticker')}, direction={p.get('direction')}, confidence={p.get('confidence')}")

    for p in all_predictions:
        direction = p.get('direction', 0)

        # STRICT: Only include assets with clear directional signals
        # HOLD signals (direction == 0) are NEVER included in top picks
        if direction > 0:
            bullish.append(p)
        elif direction < 0:
            bearish.append(p)
        # else: direction == 0 (HOLD) - skip entirely
        # This prevents the confusing situation where TM shows "HOLD 55%" but is #1 in the list

    logger.info(f"Classification: {len(bullish)} bullish, {len(bearish)} bearish (from {len(all_predictions)} predictions, excluding HOLD signals)")

    # ========================================================================
    # IPO REGIME: Show momentum-driven stocks with IPO tier info
    # For US_IPO and China_IPO tabs, show all results from screeners
    # (which fetch gainers/active/volatile stocks) with IPO tier annotations
    # ========================================================================
    if regime in ['US_IPO', 'China_IPO']:
        logger.info(f"[IPO TAB] Processing {regime} - showing momentum-driven stocks with IPO tier info")

        # Add IPO tier info to all predictions for display
        for p in bullish + bearish:
            days = p.get('days_available', p.get('trading_days', 100))
            if days <= 4:
                p['ipo_tier'] = 'NO_TRADE'
                p['ipo_tier_desc'] = 'Too new (< 5 days)'
            elif days <= 10:
                p['ipo_tier'] = 'BASIC'
                p['ipo_tier_desc'] = f'{days} days - Momentum only (25% position)'
            elif days <= 30:
                p['ipo_tier'] = 'ENHANCED'
                p['ipo_tier_desc'] = f'{days} days - Enhanced (50% position)'
            elif days <= 60:
                p['ipo_tier'] = 'HYBRID'
                p['ipo_tier_desc'] = f'{days} days - Hybrid (75% position)'
            else:
                p['ipo_tier'] = 'FULL'
                p['ipo_tier_desc'] = f'{days} days - Full analysis'

        # Count new stocks for logging
        new_count = sum(1 for p in bullish + bearish if p.get('days_available', 100) < 60)
        logger.info(f"[IPO TAB] {len(bullish)} BUYs, {len(bearish)} SELLs ({new_count} are new stocks < 60 days)")

    # ========================================================================
    # ENHANCED SIGNAL VALIDATION FOR CHINA REGIME
    # Filters dangerous shorts that cause catastrophic losses
    # Based on analysis showing SELL signals lost -94.3% vs BUY signals +92.9%
    # ========================================================================
    blocked_shorts = []  # Initialize here so it's available for response

    # Note: China_IPO excluded - IPO stocks have limited data, don't block their signals
    if regime == 'China' and ENHANCED_SIGNAL_VALIDATOR_AVAILABLE and ENHANCED_PHASE_SYSTEM:
        logger.info("[SIGNAL VALIDATION] Applying enhanced short protection for China regime")

        original_bearish_count = len(bearish)
        validated_bearish = []
        blocked_shorts = []

        for p in bearish:
            ticker = p.get('ticker', '')
            confidence = p.get('confidence', 0.5)

            # Build ticker_info from prediction data
            # Use available metrics, fallback to defaults
            ticker_info = {
                'momentum_5d': p.get('momentum_5d', p.get('expected_return', 0) * 0.5),
                'momentum_20d': p.get('momentum_20d', p.get('expected_return', 0)),
                'volatility': p.get('volatility', 0.30),
                'volume_ratio': p.get('volume_ratio', 1.0),
                'dist_from_ma': p.get('dist_from_ma', 0),
                'sector': p.get('sector', '')
            }

            # Process through enhanced signal validator
            result = ENHANCED_PHASE_SYSTEM.process_signal(
                ticker,
                'SELL',  # All bearish are SELL signals
                confidence,
                ticker_info
            )

            # Check if signal was blocked
            if result['signal_blocked']:
                blocked_shorts.append({
                    'ticker': ticker,
                    'name': p.get('name', ticker),
                    'reason': result['validation_reason'],
                    'original_confidence': confidence,
                    'is_dangerous': result['is_dangerous_short']
                })
                logger.info(f"[BLOCKED SHORT] {ticker}: {result['validation_reason']}")
            else:
                # Update prediction with validated confidence and add validation info
                p['original_confidence'] = confidence
                p['confidence'] = result['final_confidence']
                p['validation_applied'] = True
                p['validation_reason'] = result['validation_reason']
                p['position_size_pct'] = result['position_size_pct']
                validated_bearish.append(p)

        # Replace bearish list with validated list
        bearish = validated_bearish

        logger.info(f"[SIGNAL VALIDATION] China shorts: {original_bearish_count} -> {len(bearish)} "
                   f"(blocked {len(blocked_shorts)} dangerous shorts)")

        if blocked_shorts:
            logger.info(f"[BLOCKED SHORTS] {', '.join([s['ticker'] for s in blocked_shorts])}")

    # ========================================================================
    # US/INTL MODEL SIGNAL OPTIMIZATION (Fixes 1-19 from us_intl_optimizer.py)
    # Uses centralized USIntlModelOptimizer - NO DUPLICATE CODE
    # Fixes 1-15: Core optimizations
    # Fix 16: China stock blocking (routed to China model)
    # Fixes 17-18: Adaptive blocking for JPY SELL and Crude Oil
    # Fix 19: Dynamic position adjustment based on market regime
    # ========================================================================
    if regime not in ['China', 'China_IPO', 'US_IPO']:  # Skip US/INTL optimizer for China and IPO regimes
        # Create optimizer instance with all 19 fixes
        us_intl_optimizer = create_optimizer(
            enable_kelly=True,
            enable_dynamic_sizing=True,
            enable_regime_adjustment=True,  # Fix 19
            enable_adaptive_blocking=True,  # Fixes 17 & 18
        )

        # NEW: Create IntegratedUSOptimizer for profit maximization
        # Priority 1: Energy subsector optimization (fixes -82% energy loss)
        # Priority 2: Commodity dominance classification (fixes -91% mining loss)
        # Priority 3: Dividend-aware optimization (improves financials)
        # Priority 4: Market trend adaptation (fixes BUY/SELL asymmetry in downtrends)
        profit_optimizer = None
        if PROFIT_OPTIMIZER_AVAILABLE:
            try:
                profit_optimizer = create_integrated_optimizer(
                    enable_energy=True,
                    enable_commodity=True,
                    enable_dividend=True,
                    enable_futures=True,
                    enable_trend=True,  # Priority 4: Dynamic market trend adaptation
                )
                logger.info("[PROFIT OPTIMIZER] Created IntegratedUSOptimizer with energy/commodity/dividend/trend optimization")
                # Log current market trend
                trend_info = profit_optimizer.get_current_trend()
                logger.info(f"[MARKET TREND] Current: {trend_info.get('trend', 'unknown')} - {trend_info.get('description', '')}")
            except Exception as e:
                logger.warning(f"[PROFIT OPTIMIZER] Failed to create: {e}")
                profit_optimizer = None

        # Update market regime for adaptive blocking and position adjustment
        # Get market indicators from yfinance if available
        try:
            import yfinance as yf
            spy = yf.Ticker("SPY")
            vix = yf.Ticker("^VIX")
            gld = yf.Ticker("GLD")

            # Get recent data
            spy_hist = spy.history(period="3mo")
            vix_hist = vix.history(period="1d")
            gld_hist = gld.history(period="1mo")

            # Calculate indicators
            spy_return_20d = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[-20] - 1) if len(spy_hist) >= 20 else 0.0
            spy_return_60d = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0] - 1) if len(spy_hist) >= 60 else 0.0
            spy_200ma = spy_hist['Close'].rolling(200).mean().iloc[-1] if len(spy_hist) >= 200 else spy_hist['Close'].mean()
            spy_above_200ma = spy_hist['Close'].iloc[-1] > spy_200ma
            vix_current = vix_hist['Close'].iloc[-1] if len(vix_hist) > 0 else 20.0
            gold_return = (gld_hist['Close'].iloc[-1] / gld_hist['Close'].iloc[0] - 1) if len(gld_hist) > 0 else 0.0

            # Update regime
            regime_result = us_intl_optimizer.update_market_regime({
                'vix': float(vix_current),
                'spy_return_20d': float(spy_return_20d),
                'spy_return_60d': float(spy_return_60d),
                'spy_above_200ma': bool(spy_above_200ma),
                'gold_return_20d': float(gold_return),
            })

            if regime_result:
                logger.info(f"[MARKET REGIME] {regime_result['primary_regime']} "
                           f"({regime_result['confidence']*100:.0f}% confidence)")
        except Exception as e:
            logger.warning(f"[MARKET REGIME] Could not fetch indicators: {e}")
            regime_result = None

        # Process SELL signals through optimizer
        original_bearish_count = len(bearish)
        filtered_bearish = []
        blocked_sells_usintl = []
        # Note: adaptive_blocked removed - Fix 17/18 are ADAPTIVE (position reduction, not blocking)

        for p in bearish:
            ticker = p.get('ticker', '')
            confidence = p.get('confidence', 0.5)
            asset_type = p.get('type', 'Stock').lower()

            # Fix 16: Check for China stocks (should not reach here)
            if us_intl_optimizer.is_china_stock(ticker):
                blocked_sells_usintl.append({
                    'ticker': ticker,
                    'name': p.get('name', ticker),
                    'reason': 'China stock - should use China model with DeepSeek',
                    'original_confidence': confidence
                })
                logger.warning(f"[FIX 16] China stock {ticker} reached US/INTL model - routing error")
                continue

            # Use optimizer to validate and optimize signal
            result = us_intl_optimizer.optimize_signal(
                ticker=ticker,
                signal_type='SELL',
                confidence=confidence,
                volatility=0.25  # Default volatility
            )

            if result.blocked:
                blocked_sells_usintl.append({
                    'ticker': ticker,
                    'name': p.get('name', ticker),
                    'reason': result.block_reason,
                    'original_confidence': confidence
                })
                logger.info(f"[US/INTL SELL BLOCKED] {ticker}: {result.block_reason}")
                continue

            # Fixes 17 & 18: Get adaptive position adjustment for JPY pairs and Crude Oil
            # NOTE: This is ADAPTIVE - never blocks, only reduces position based on regime
            asset_class = us_intl_optimizer.classify_asset(ticker)
            _, adaptive_reduction, adaptive_reason = us_intl_optimizer.check_adaptive_blocking(
                ticker=ticker,
                signal_type='SELL',
                asset_class=asset_class
            )
            # No blocking check - Fix 17/18 are adaptive, not permanent blocks

            # Fix 19: Apply regime-based position adjustment
            base_position = result.position_multiplier * 5
            adjusted_position, regime_explanation = us_intl_optimizer.get_regime_position_adjustment(
                ticker=ticker,
                signal_type='SELL',
                asset_class=asset_class,
                base_position=base_position / 100,  # Convert to 0-1 scale
            )
            adjusted_position_pct = adjusted_position * 100  # Convert back to percentage

            # Apply adaptive position reduction for JPY/Crude (always applied based on regime)
            if adaptive_reduction < 1.0:
                adjusted_position_pct *= adaptive_reduction
                logger.info(f"[ADAPTIVE REDUCE] {ticker}: {adaptive_reduction:.0%} position -> {adjusted_position_pct:.1f}%")

            # NEW: Apply profit maximization optimizer (energy/commodity/dividend)
            profit_opt_result = None
            profit_position_mult = 1.0
            if profit_optimizer:
                try:
                    profit_opt_result = profit_optimizer.optimize_signal(
                        ticker=ticker,
                        signal_type='SELL',
                        confidence=confidence,
                        volatility=0.25,
                        momentum=0.0,
                    )
                    if not profit_opt_result.should_trade:
                        # Profit optimizer blocked this signal
                        blocked_sells_usintl.append({
                            'ticker': ticker,
                            'name': p.get('name', ticker),
                            'reason': f'Profit optimizer: {profit_opt_result.adjustment_reason}',
                            'original_confidence': confidence
                        })
                        logger.info(f"[PROFIT OPTIMIZER SELL BLOCKED] {ticker}: {profit_opt_result.adjustment_reason}")
                        continue
                    # Apply profit optimizer position multiplier
                    profit_position_mult = profit_opt_result.position_multiplier
                    adjusted_position_pct *= profit_position_mult
                    logger.debug(f"[PROFIT OPTIMIZER] {ticker} ({profit_opt_result.category}): "
                                f"conf {confidence:.2f}->{profit_opt_result.adjusted_confidence:.2f}, "
                                f"pos_mult={profit_position_mult:.2f}")
                except Exception as e:
                    logger.warning(f"[PROFIT OPTIMIZER] Error for {ticker}: {e}")

            # Apply optimizer results with regime adjustment
            p['position_size_pct'] = adjusted_position_pct
            p['sell_position_reduced'] = True
            p['sell_position_multiplier'] = result.position_multiplier
            p['stop_loss_pct'] = result.stop_loss_pct

            # Add optimization metadata with regime info
            current_regime = us_intl_optimizer.get_current_regime()
            p['us_intl_sell_optimization'] = {
                'confidence_threshold': us_intl_optimizer.SELL_CONFIDENCE_THRESHOLDS.get(asset_class, 0.75),
                'position_multiplier': result.position_multiplier,
                'stop_loss_pct': result.stop_loss_pct,
                'original_position': 5.0,
                'adjusted_position': adjusted_position_pct,
                'fixes_applied': result.fixes_applied,
                'market_regime': current_regime['primary_regime'] if current_regime else None,
                'regime_confidence': current_regime['confidence'] if current_regime else None,
                'regime_adjustment': regime_explanation,
                'adaptive_reduction': adaptive_reason if adaptive_reduction < 1.0 else None,
                'adaptive_multiplier': adaptive_reduction,
            }

            # Add profit optimizer metadata if applied
            if profit_opt_result:
                p['profit_optimization'] = {
                    'category': profit_opt_result.category,
                    'original_confidence': profit_opt_result.original_confidence,
                    'adjusted_confidence': profit_opt_result.adjusted_confidence,
                    'position_multiplier': profit_opt_result.position_multiplier,
                    'adjustment_reason': profit_opt_result.adjustment_reason,
                    'market_trend': profit_opt_result.market_trend,
                    'trend_multiplier': profit_opt_result.trend_multiplier,
                    'trend_win_rate': profit_opt_result.trend_win_rate,
                }

            filtered_bearish.append(p)

        bearish = filtered_bearish

        logger.info(f"[US/INTL SELL OPTIMIZATION] {original_bearish_count} -> {len(bearish)} "
                   f"(filtered {len(blocked_sells_usintl)} low-confidence/blacklisted)")

        if blocked_sells_usintl:
            logger.info(f"[BLOCKED SELLS] {', '.join([s['ticker'] for s in blocked_sells_usintl[:5]])}")

        # Process BUY signals through optimizer
        filtered_bullish = []
        blocked_buys_usintl = []

        for p in bullish:
            ticker = p.get('ticker', '')
            confidence = p.get('confidence', 0.5)
            asset_type = p.get('type', 'Stock').lower()

            # Fix 16: Check for China stocks (should not reach here)
            if us_intl_optimizer.is_china_stock(ticker):
                blocked_buys_usintl.append({
                    'ticker': ticker,
                    'name': p.get('name', ticker),
                    'reason': 'China stock - should use China model with DeepSeek',
                    'original_confidence': confidence
                })
                logger.warning(f"[FIX 16] China stock {ticker} reached US/INTL model - routing error")
                continue

            # Use optimizer to validate and optimize signal
            result = us_intl_optimizer.optimize_signal(
                ticker=ticker,
                signal_type='BUY',
                confidence=confidence,
                volatility=0.25  # Default volatility
            )

            # Fix 16: Block if optimizer raised error (shouldn't happen for BUYs normally)
            if result.blocked:
                blocked_buys_usintl.append({
                    'ticker': ticker,
                    'name': p.get('name', ticker),
                    'reason': result.block_reason,
                    'original_confidence': confidence
                })
                continue

            # Fixes 17 & 18: Get adaptive position adjustment for JPY pairs and Crude Oil BUYs
            # NOTE: This is ADAPTIVE - never blocks, only reduces position based on regime
            asset_class = us_intl_optimizer.classify_asset(ticker)
            _, adaptive_reduction, adaptive_reason = us_intl_optimizer.check_adaptive_blocking(
                ticker=ticker,
                signal_type='BUY',
                asset_class=asset_class
            )
            # No blocking check - Fix 17/18 are adaptive, not permanent blocks

            # Fix 19: Apply regime-based position adjustment
            base_position = result.position_multiplier * 5
            adjusted_position, regime_explanation = us_intl_optimizer.get_regime_position_adjustment(
                ticker=ticker,
                signal_type='BUY',
                asset_class=asset_class,
                base_position=base_position / 100,  # Convert to 0-1 scale
            )
            adjusted_position_pct = adjusted_position * 100  # Convert back to percentage

            # Apply adaptive position reduction for JPY/Crude (always applied based on regime)
            if adaptive_reduction < 1.0:
                adjusted_position_pct *= adaptive_reduction
                logger.info(f"[ADAPTIVE REDUCE BUY] {ticker}: {adaptive_reduction:.0%} position -> {adjusted_position_pct:.1f}%")

            # NEW: Apply profit maximization optimizer (energy/commodity/dividend)
            profit_opt_result = None
            profit_position_mult = 1.0
            if profit_optimizer:
                try:
                    profit_opt_result = profit_optimizer.optimize_signal(
                        ticker=ticker,
                        signal_type='BUY',
                        confidence=confidence,
                        volatility=0.25,
                        momentum=0.0,
                    )
                    if not profit_opt_result.should_trade:
                        # Profit optimizer blocked this signal
                        blocked_buys_usintl.append({
                            'ticker': ticker,
                            'name': p.get('name', ticker),
                            'reason': f'Profit optimizer: {profit_opt_result.adjustment_reason}',
                            'original_confidence': confidence
                        })
                        logger.info(f"[PROFIT OPTIMIZER BUY BLOCKED] {ticker}: {profit_opt_result.adjustment_reason}")
                        continue
                    # Apply profit optimizer position multiplier
                    profit_position_mult = profit_opt_result.position_multiplier
                    adjusted_position_pct *= profit_position_mult
                    logger.debug(f"[PROFIT OPTIMIZER BUY] {ticker} ({profit_opt_result.category}): "
                                f"conf {confidence:.2f}->{profit_opt_result.adjusted_confidence:.2f}, "
                                f"pos_mult={profit_position_mult:.2f}")
                except Exception as e:
                    logger.warning(f"[PROFIT OPTIMIZER] Error for BUY {ticker}: {e}")

            # Apply optimizer results with regime adjustment
            p['position_size_pct'] = adjusted_position_pct
            p['buy_position_boost'] = result.position_multiplier
            p['stop_loss_pct'] = result.stop_loss_pct

            # Add optimization metadata with regime info
            current_regime = us_intl_optimizer.get_current_regime()
            p['us_intl_buy_optimization'] = {
                'position_boost': result.position_multiplier,
                'stop_loss_pct': result.stop_loss_pct,
                'original_position': 5.0,
                'adjusted_position': adjusted_position_pct,
                'fixes_applied': result.fixes_applied,
                'market_regime': current_regime['primary_regime'] if current_regime else None,
                'regime_confidence': current_regime['confidence'] if current_regime else None,
                'regime_adjustment': regime_explanation,
                'adaptive_reduction': adaptive_reason if adaptive_reduction < 1.0 else None,
                'adaptive_multiplier': adaptive_reduction,
            }

            # Add profit optimizer metadata if applied
            if profit_opt_result:
                p['profit_optimization'] = {
                    'category': profit_opt_result.category,
                    'original_confidence': profit_opt_result.original_confidence,
                    'adjusted_confidence': profit_opt_result.adjusted_confidence,
                    'position_multiplier': profit_opt_result.position_multiplier,
                    'adjustment_reason': profit_opt_result.adjustment_reason,
                    'market_trend': profit_opt_result.market_trend,
                    'trend_multiplier': profit_opt_result.trend_multiplier,
                    'trend_win_rate': profit_opt_result.trend_win_rate,
                }

            filtered_bullish.append(p)

        bullish = filtered_bullish

        logger.info(f"[US/INTL BUY OPTIMIZATION] Applied Fixes 1-19 to {len(bullish)} BUY signals"
                   f" (blocked {len(blocked_buys_usintl)} China stocks)")
        if blocked_buys_usintl:
            logger.info(f"[BLOCKED BUYS] {', '.join([s['ticker'] for s in blocked_buys_usintl[:5]])}")

    # ========================================================================
    # CHINA BUY SIGNAL OPTIMIZATION
    # Applies stop-loss, smart sizing, and risk detection to BUY signals
    # ========================================================================
    blocked_buys = []  # Initialize for response
    buy_optimization_stats = {}  # Track optimization results

    if regime == 'China' and CHINA_BUY_OPTIMIZER is not None:
        logger.info("[BUY OPTIMIZATION] Applying stop-loss, sizing, and risk detection for China BUYs")

        original_bullish_count = len(bullish)
        optimized_bullish = []

        for p in bullish:
            ticker = p.get('ticker', '')
            confidence = p.get('confidence', 0.5)

            # Build signal_data for optimizer
            signal_data = {
                'ticker': ticker,
                'confidence': confidence,
                'position_size': 0.05,  # Base 5% position size
                'buy_quality': p.get('buy_quality', 'cautious_buy'),
            }

            # Build ticker_data from prediction
            ticker_data = {
                'volatility': p.get('volatility', 0.25),
                'momentum_5d': p.get('momentum_5d', 0),
                'momentum_20d': p.get('momentum_20d', p.get('expected_return', 0)),
                'volume_ratio': p.get('volume_ratio', 1.0),
                'dist_from_ma': p.get('dist_from_ma', 0),
            }

            # Apply optimization
            opt_result = CHINA_BUY_OPTIMIZER.optimize_buy_signal(signal_data, ticker_data)

            if opt_result.get('action') == 'BLOCKED':
                # High-risk pattern detected - block this BUY
                blocked_buys.append({
                    'ticker': ticker,
                    'name': p.get('name', ticker),
                    'reason': opt_result.get('reason', 'High-risk pattern'),
                    'risk_signals': opt_result.get('risk_signals', []),
                    'original_confidence': confidence
                })
                logger.info(f"[BLOCKED BUY] {ticker}: {opt_result.get('reason')}")
            else:
                # Apply optimization to the signal
                p['buy_optimization'] = {
                    'stop_loss_pct': opt_result.get('stop_loss_pct', 0.06),
                    'max_hold_days': opt_result.get('max_hold_days', 10),
                    'optimized_size': opt_result.get('optimized_size', 0.05),
                    'original_size': opt_result.get('original_size', 0.05),
                    'size_change_pct': opt_result.get('size_change', 0) * 100,
                    'risk_signals': opt_result.get('risk_signals', []),
                    'buy_quality': opt_result.get('buy_quality', 'cautious_buy'),
                }
                p['stop_loss_pct'] = opt_result.get('stop_loss_pct', 0.06)
                # optimized_size is already in decimal form (0.05 = 5%), convert to percentage for display
                opt_size = opt_result.get('optimized_size', 0.05)
                # Ensure it's in proper range (0.01-0.15 = 1%-15%)
                if opt_size > 1:  # If returned as percentage (5.0) instead of decimal (0.05)
                    opt_size = opt_size / 100
                p['position_size_pct'] = opt_size * 100  # Convert 0.05 -> 5.0%
                optimized_bullish.append(p)

        # Replace bullish list with optimized list
        bullish = optimized_bullish

        # Calculate stats
        buy_optimization_stats = {
            'original_count': original_bullish_count,
            'optimized_count': len(bullish),
            'blocked_count': len(blocked_buys),
            'avg_stop_loss': sum(p.get('stop_loss_pct', 0.06) for p in bullish) / len(bullish) if bullish else 0.06,
        }

        logger.info(f"[BUY OPTIMIZATION] China buys: {original_bullish_count} -> {len(bullish)} "
                   f"(blocked {len(blocked_buys)} high-risk patterns)")

        if blocked_buys:
            logger.info(f"[BLOCKED BUYS] {', '.join([b['ticker'] for b in blocked_buys])}")

    # ========================================================================
    # PHASE 1-6 ENHANCED PROFIT SCORING
    # Uses all Phase 1-6 calculations for comprehensive risk-adjusted ranking
    # ========================================================================
    def profit_score(asset):
        """
        Enhanced profit score using Phase 1-6 calculations.

        Phase 1: Volatility scaling adjustment
        Phase 2: Multi-signal combination (momentum, volatility, mean reversion)
        Phase 4: Asset-specific and regime multipliers
        Phase 5: Decay-weighted Sharpe and Bayesian confidence
        Phase 6: Expected Shortfall risk adjustment
        """
        expected_return = asset.get('expected_return', 0)
        confidence = asset.get('confidence', 0.5)
        volatility = asset.get('volatility', 0.02)
        asset_type = asset.get('type', 'Stock').lower()

        # Base expected profit
        expected_profit = abs(expected_return) * confidence

        if PHASE6_CALCULATIONS_AVAILABLE:
            try:
                # PHASE 1: Volatility scaling adjustment
                # Reduce position sizing for high volatility assets
                vol_scaling = calculate_volatility_scaling(
                    np.array([expected_return] * 20),  # Simulated returns
                    lookback=20,
                    benchmark_vol=0.15
                )

                # PHASE 4: Asset-specific multiplier
                # Different asset classes have different risk profiles
                asset_mult = asset_specific_multiplier(
                    base_mult=1.0,
                    asset_class=asset_type,
                    vix=None,  # Could fetch VIX here if available
                    vix_sensitivity=1.0
                )

                # PHASE 6: Expected Shortfall penalty
                # Penalize assets with high tail risk
                if volatility > 0.01:
                    es_penalty = abs(_calculate_parametric_es(volatility, confidence=0.95))
                    es_factor = 1 / (1 + es_penalty * 5)  # Higher ES = lower score
                else:
                    es_factor = 1.0

                # PHASE 5: Confidence interval adjustment
                # Use Bayesian confidence for reliability weighting
                confidence_boost = 1.0
                if confidence > 0.70:
                    confidence_boost = 1.2  # High confidence bonus
                elif confidence < 0.55:
                    confidence_boost = 0.8  # Low confidence penalty

                # Combine all factors
                risk_factor = 1 + min(volatility * 10, 2.0)
                base_score = expected_profit / (risk_factor + 0.01)

                # Apply Phase 1-6 adjustments
                enhanced_score = base_score * vol_scaling * asset_mult * es_factor * confidence_boost

                # Store Phase 6 metrics in asset for response
                asset['phase6_metrics'] = {
                    'vol_scaling': round(vol_scaling, 4),
                    'asset_multiplier': round(asset_mult, 4),
                    'es_factor': round(es_factor, 4),
                    'confidence_boost': round(confidence_boost, 4),
                    'enhanced_score': round(enhanced_score, 6)
                }

                return enhanced_score

            except Exception as e:
                logger.warning(f"[PHASE 1-6] Enhanced scoring failed, using basic: {e}")

        # Fallback to basic scoring if Phase 1-6 not available
        risk_factor = 1 + min(volatility * 10, 2.0)
        score = expected_profit / (risk_factor + 0.01)
        if confidence > 0.70:
            score *= 1.2
        return score

    # Sort by profit rate (highest expected return first)
    bullish.sort(key=profit_score, reverse=True)
    bearish.sort(key=profit_score, reverse=True)

    # STRICT TYPE FILTERING: Only show assets matching selected regime
    # This ensures Stock page shows ONLY stocks, Crypto shows ONLY crypto, etc.
    # EXCEPTION: China, China_IPO, US_IPO regimes are special - skip type filtering
    # IPO regimes contain stocks with type='Stock', not type='US_IPO'
    skip_type_filter_regimes = ['all', 'China', 'China_IPO', 'US_IPO']
    if regime not in skip_type_filter_regimes:
        bullish = [p for p in bullish if p.get('type') == regime]
        bearish = [p for p in bearish if p.get('type') == regime]
        logger.info(f"[FILTER] After strict {regime} filter: {len(bullish)} buys, {len(bearish)} sells")
    elif regime in ['China', 'China_IPO']:
        # China regimes already selected only China market tickers, no type filtering needed
        logger.info(f"[{regime}] Showing all China market picks: {len(bullish)} buys, {len(bearish)} sells")
    elif regime == 'US_IPO':
        # US_IPO contains US stocks with type='Stock', skip type filtering
        logger.info(f"[US_IPO] Showing all US IPO/momentum picks: {len(bullish)} buys, {len(bearish)} sells")
    else:
        # For 'all' regime - NO type cap, pure profit-based ranking
        # Highest expected return wins, regardless of asset type
        logger.info(f"[ALL] Pure profit ranking: {len(bullish)} buys, {len(bearish)} sells (sorted by expected return)")

    # Build response with ticker source information
    result = {
        'status': 'success',
        'regime': regime,
        'top_buys': bullish[:10],  # Top 10 most bullish
        'top_sells': bearish[:10],  # Top 10 most bearish
        'total_analyzed': len(all_predictions),
        'timestamp': datetime.now().isoformat(),
        'from_cache': False,
        'cache_ttl_seconds': cache_ttl,
        'ticker_source': ticker_source,  # 'screener' or 'database'
        'screener_available': SCREENER_DISCOVERY_AVAILABLE,
        'signal_validation_available': ENHANCED_SIGNAL_VALIDATOR_AVAILABLE,
    }

    # Add blocked shorts info for China regime (if validation was applied)
    if regime == 'China' and ENHANCED_SIGNAL_VALIDATOR_AVAILABLE:
        result['signal_validation'] = {
            'applied': True,
            'blocked_shorts_count': len(blocked_shorts),
            'blocked_shorts': blocked_shorts[:5] if blocked_shorts else [],  # Top 5 blocked
            'message': 'Enhanced short protection active - dangerous shorts filtered'
        }

    # Add buy optimization info for China regime
    if regime == 'China' and CHINA_BUY_OPTIMIZER is not None:
        result['buy_optimization'] = {
            'applied': True,
            'blocked_buys_count': len(blocked_buys),
            'blocked_buys': blocked_buys[:5] if blocked_buys else [],  # Top 5 blocked
            'stats': buy_optimization_stats,
            'features': {
                'stop_loss': 'Active (6% max loss)',
                'smart_sizing': 'Active (up to 50% boost for winners)',
                'risk_detection': 'Active (high-vol downtrend filter)'
            },
            'message': 'BUY optimization active - stop-loss, smart sizing, risk detection applied'
        }

    # Add US/Intl optimization info (for non-China regimes)
    if regime not in ['China', 'China_IPO', 'US_IPO']:  # Skip US/INTL optimizer for China and IPO regimes
        result['us_intl_optimization'] = {
            'applied': True,
            'sell_optimization': {
                'higher_confidence_threshold': True,
                'confidence_thresholds': {
                    'commodity': '80%',
                    'cryptocurrency': '78%',
                    'stock': '75%',
                    'forex': '72%'
                },
                'position_reduction': '50% base (70% for commodities)',
                'blacklist': ['NG=F'],
                'blocked_sells_count': len(blocked_sells_usintl) if 'blocked_sells_usintl' in locals() else 0,
            },
            'buy_optimization': {
                'us_stock_boost': '+30%',
                'commodity_reduction': '-20%',
                'stop_loss_active': True,
                'stop_loss_levels': {
                    'stock': '8%',
                    'commodity': '10%',
                    'cryptocurrency': '12%',
                    'forex': '6%'
                }
            },
            'message': 'US/Intl model fixing1 active - higher SELL thresholds, position sizing, stop-losses'
        }

        # Add market trend info from profit optimizer (Priority 4)
        if PROFIT_OPTIMIZER_AVAILABLE and 'profit_optimizer' in locals() and profit_optimizer is not None:
            try:
                trend_info = profit_optimizer.get_current_trend()
                opt_stats = profit_optimizer.get_statistics()
                result['market_trend_adaptation'] = {
                    'enabled': True,
                    'current_trend': trend_info.get('trend', 'unknown'),
                    'trend_description': trend_info.get('description', ''),
                    'buy_win_rate': trend_info.get('buy_win_rate', 0.5),
                    'sell_win_rate': trend_info.get('sell_win_rate', 0.5),
                    'recommendation': trend_info.get('recommendation', 'Balanced'),
                    'signals_trend_adjusted': opt_stats.get('trend_adjusted', 0),
                    'signals_blocked': opt_stats.get('blocked_signals', 0),
                    'message': f"Market trend: {trend_info.get('trend', 'unknown')} - {trend_info.get('recommendation', 'Balanced')}"
                }
                logger.info(f"[API] Added market_trend_adaptation: {trend_info.get('trend', 'unknown')}")
            except Exception as e:
                logger.warning(f"[MARKET TREND] Error getting trend info: {e}")
        else:
            logger.debug(f"[MARKET TREND] Skipped - PROFIT_OPTIMIZER_AVAILABLE={PROFIT_OPTIMIZER_AVAILABLE}, profit_optimizer exists={'profit_optimizer' in locals()}")

        # ====================================================================
        # US MODEL FIXING7 & FIXING8: Lag-Free Regime & Profit Maximization
        # Adds regime detection, transition phases, dynamic multipliers
        # ====================================================================
        if US_PROFIT_REGIME_AVAILABLE and US_PROFIT_REGIME_CLASSIFIER is not None:
            try:
                import yfinance as yf
                # Get SPY data for regime classification
                spy_data = yf.download('SPY', period='6mo', progress=False)
                if isinstance(spy_data.columns, pd.MultiIndex):
                    spy_data.columns = spy_data.columns.get_level_values(0)

                # Get current VIX
                vix_data = yf.download('^VIX', period='1d', progress=False)
                if isinstance(vix_data.columns, pd.MultiIndex):
                    vix_data.columns = vix_data.columns.get_level_values(0)
                current_vix = float(vix_data['Close'].iloc[-1]) if len(vix_data) > 0 else 20.0

                # Get profit-maximizing regime classification
                profit_regime = US_PROFIT_REGIME_CLASSIFIER.classify_regime_with_profit(
                    spy_data, None, current_vix
                )

                result['us_lag_free_regime'] = {
                    'source': 'us_model_fixing7_fixing8',
                    'regime': profit_regime.regime,
                    'transition_state': profit_regime.transition_state,
                    'transition_phase': profit_regime.transition_phase.value,
                    'transition_day': profit_regime.transition_day,
                    'profit_score': round(profit_regime.profit_score, 1),
                    'dynamic_multiplier': round(profit_regime.dynamic_multiplier, 3),
                    'buy_multiplier': round(profit_regime.base_output.buy_multiplier, 3),
                    'sell_multiplier': round(profit_regime.base_output.sell_multiplier, 3),
                    'signals_detected': profit_regime.base_output.signals_detected,
                    'position_params': {
                        'small_cap_multiplier': round(profit_regime.small_cap_multiplier, 2),
                        'large_cap_multiplier': round(profit_regime.large_cap_multiplier, 2),
                        'max_position_size': round(profit_regime.max_position_size * 100, 1),
                        'stop_loss': round(profit_regime.stop_loss * 100, 1),
                        'max_positions': profit_regime.max_positions,
                        'sizing_method': profit_regime.position_sizing_method,
                    },
                    'profit_targets': {
                        'target_1': round(profit_regime.profit_target_1 * 100, 1),
                        'target_2': round(profit_regime.profit_target_2 * 100, 1),
                        'target_3': round(profit_regime.profit_target_3 * 100, 1),
                        'trailing_stop': round(profit_regime.trailing_stop * 100, 1),
                    },
                    'capital_allocation': {k: round(v * 100, 1) for k, v in profit_regime.capital_allocation.items()},
                    'sector_focus': profit_regime.sector_focus,
                    'vix_level': round(current_vix, 1),
                    'confidence': round(profit_regime.base_output.confidence, 3),
                }
                logger.info(f"[US FIXING7/8] Regime: {profit_regime.regime}, Phase: {profit_regime.transition_phase.value}, Mult: {profit_regime.dynamic_multiplier:.3f}x")
            except Exception as e:
                logger.warning(f"[US FIXING7/8] Error getting lag-free regime: {e}")
                result['us_lag_free_regime'] = {'error': str(e), 'available': False}

    # Cache the results
    TOP_PICKS_CACHE[cache_key] = (result.copy(), datetime.now())
    logger.info(f"[CACHE STORE] Cached top-picks for {regime} (TTL: {cache_ttl}s, source: {ticker_source})")

    return jsonify(result)


# ============================================================================
# REAL-TIME YAHOO TEST & BULL/BEAR PREDICTION ENDPOINTS
# ============================================================================

@app.route('/api/top-picks/realtime-test')
def realtime_yahoo_test():
    """
    Run REAL-TIME Yahoo Finance technical analysis on current top picks.
    Returns detailed P/L projections, RSI, MACD, and portfolio simulation.
    Excludes China regime (handled separately).
    """
    regime = request.args.get('regime', 'Stock')

    # Exclude China - handled separately
    if regime == 'China':
        return jsonify({
            'error': 'China regime uses separate analysis system',
            'message': 'Please use the China-specific analysis tools'
        }), 400

    # Get current top picks from cache
    cache_key = f"top_picks_{regime}"
    if cache_key not in TOP_PICKS_CACHE:
        return jsonify({
            'error': 'No top picks available for this regime',
            'message': 'Please refresh the top picks first'
        }), 404

    cached_data, cache_time = TOP_PICKS_CACHE[cache_key]

    # Extract buy and sell signals
    buy_signals = [(p.get('ticker'), p.get('confidence', 50)) for p in cached_data.get('top_buys', [])]
    sell_signals = [(p.get('ticker'), p.get('confidence', 50)) for p in cached_data.get('top_sells', [])]

    logger.info(f"[REALTIME TEST] Analyzing {len(buy_signals)} BUY and {len(sell_signals)} SELL signals for {regime}")

    # Run real-time analysis
    buy_analysis = []
    sell_analysis = []

    # Analyze BUY signals
    for ticker, confidence in buy_signals[:10]:
        if ticker:
            analysis = _analyze_ticker_realtime(ticker, confidence, 'BUY')
            if analysis:
                buy_analysis.append(analysis)

    # Analyze SELL signals
    for ticker, confidence in sell_signals[:10]:
        if ticker:
            analysis = _analyze_ticker_realtime(ticker, confidence, 'SELL')
            if analysis:
                sell_analysis.append(analysis)

    # Calculate portfolio simulation
    portfolio_sim = _simulate_portfolio_realistic(buy_analysis, capital=10000)

    # Generate summary
    summary = _generate_analysis_summary(buy_analysis, sell_analysis, regime)

    # NEW: Add profit maximization analysis if available (fixing6)
    profit_maximization = None
    if PROFIT_MAXIMIZER_AVAILABLE and buy_analysis:
        try:
            strategy = create_complete_strategy(capital=10000)
            profit_maximization = strategy.analyze_opportunities(
                buy_signals=buy_analysis,
                sell_signals=sell_analysis,
            )
            logger.info(f"[PROFIT MAXIMIZER] Generated profit maximization analysis")
        except Exception as e:
            logger.warning(f"[PROFIT MAXIMIZER] Analysis failed: {e}")
            profit_maximization = {'error': str(e)}

    return jsonify({
        'regime': regime,
        'timestamp': datetime.now().isoformat(),
        'cache_age_seconds': int((datetime.now() - cache_time).total_seconds()),
        'buy_analysis': buy_analysis,
        'sell_analysis': sell_analysis,
        'portfolio_simulation': portfolio_sim,
        'summary': summary,
        'top_opportunities': sorted(
            buy_analysis,
            key=lambda x: x.get('expected_value', 0),
            reverse=True
        )[:5],
        'profit_maximization': profit_maximization,
    })


@app.route('/api/top-picks/profit-maximize')
def profit_maximize_portfolio():
    """
    Run Profit Maximization Strategy on current top picks (fixing6.pdf).

    Uses EV-based position sizing to:
    - Concentrate capital on highest EV opportunities
    - Filter out negative and low EV signals
    - Provide dynamic position sizing based on EV tiers
    - Generate execution plan for maximum profit

    Excludes China regime (handled separately).
    """
    if not PROFIT_MAXIMIZER_AVAILABLE:
        return jsonify({
            'error': 'Profit Maximizer not available',
            'message': 'The profit_maximizer module could not be loaded'
        }), 500

    regime = request.args.get('regime', 'Stock')
    capital = float(request.args.get('capital', 10000))
    concentrated = request.args.get('concentrated', 'false').lower() == 'true'

    # Exclude China - handled separately
    if regime == 'China':
        return jsonify({
            'error': 'China regime uses separate analysis system',
            'message': 'Please use the China-specific analysis tools'
        }), 400

    # Get current top picks from cache
    cache_key = f"top_picks_{regime}"
    if cache_key not in TOP_PICKS_CACHE:
        return jsonify({
            'error': 'No top picks available for this regime',
            'message': 'Please refresh the top picks first'
        }), 404

    cached_data, cache_time = TOP_PICKS_CACHE[cache_key]

    # Extract buy and sell signals
    buy_signals = [(p.get('ticker'), p.get('confidence', 50)) for p in cached_data.get('top_buys', [])]
    sell_signals = [(p.get('ticker'), p.get('confidence', 50)) for p in cached_data.get('top_sells', [])]

    logger.info(f"[PROFIT MAXIMIZE] Analyzing {len(buy_signals)} BUY and {len(sell_signals)} SELL signals for {regime}")

    # Run real-time analysis to get detailed signal data
    buy_analysis = []
    sell_analysis = []

    for ticker, confidence in buy_signals[:10]:
        if ticker:
            analysis = _analyze_ticker_realtime(ticker, confidence, 'BUY')
            if analysis:
                buy_analysis.append(analysis)

    for ticker, confidence in sell_signals[:10]:
        if ticker:
            analysis = _analyze_ticker_realtime(ticker, confidence, 'SELL')
            if analysis:
                sell_analysis.append(analysis)

    # Create profit maximization strategy
    strategy = create_complete_strategy(capital=capital)

    # Get full analysis or concentrated portfolio
    if concentrated:
        result = strategy.get_concentrated_portfolio(
            buy_signals=buy_analysis,
            sell_signals=sell_analysis,
            max_positions=5,
        )
        result['analysis_type'] = 'concentrated'
    else:
        result = strategy.analyze_opportunities(
            buy_signals=buy_analysis,
            sell_signals=sell_analysis,
        )
        result['analysis_type'] = 'full'

    # Add EV tier breakdown
    ev_breakdown = {
        'exceptional': [],  # EV > 10
        'excellent': [],    # EV 5-10
        'good': [],         # EV 2-5
        'moderate': [],     # EV 1-2
        'low': [],          # EV 0-1
        'skip': [],         # EV < 0
    }

    for signal in buy_analysis + sell_analysis:
        ev = signal.get('expected_value', 0)
        ticker = signal.get('ticker')
        signal_type = 'BUY' if signal in buy_analysis else 'SELL'
        entry = {
            'ticker': ticker,
            'type': signal_type,
            'ev': ev,
            'profit_5d': signal.get('potential_gain_5d', 0),
            'confidence': signal.get('confidence', 0),
        }

        if ev > 10:
            ev_breakdown['exceptional'].append(entry)
        elif ev >= 5:
            ev_breakdown['excellent'].append(entry)
        elif ev >= 2:
            ev_breakdown['good'].append(entry)
        elif ev >= 1:
            ev_breakdown['moderate'].append(entry)
        elif ev >= 0:
            ev_breakdown['low'].append(entry)
        else:
            ev_breakdown['skip'].append(entry)

    return jsonify({
        'regime': regime,
        'capital': capital,
        'timestamp': datetime.now().isoformat(),
        'cache_age_seconds': int((datetime.now() - cache_time).total_seconds()),
        'profit_maximization': result,
        'ev_tier_breakdown': ev_breakdown,
        'strategy_info': {
            'name': 'EV-Based Profit Maximization (fixing6)',
            'description': 'Concentrates capital on highest Expected Value opportunities',
            'ev_rules': {
                'exceptional': {'ev': '>10', 'position': '20-35%', 'action': 'Strong BUY'},
                'excellent': {'ev': '5-10', 'position': '15-25%', 'action': 'BUY'},
                'good': {'ev': '2-5', 'position': '10-20%', 'action': 'BUY'},
                'moderate': {'ev': '1-2', 'position': '5-15%', 'action': 'Consider'},
                'low': {'ev': '0-1', 'position': '2-10%', 'action': 'Skip or minimal'},
                'negative': {'ev': '<0', 'position': '0%', 'action': 'DO NOT TRADE'},
            },
        },
    })


@app.route('/api/top-picks/predict-trend')
def predict_market_trend():
    """
    Predict Bull/Bear market trend for selected regime.
    Uses aggregate analysis of signals, technical indicators, and market breadth.
    Excludes China regime (handled separately).
    """
    regime = request.args.get('regime', 'Stock')

    # Exclude China
    if regime == 'China':
        return jsonify({
            'error': 'China regime uses separate prediction system',
            'message': 'Please use the China-specific prediction tools'
        }), 400

    # Get current top picks
    cache_key = f"top_picks_{regime}"
    if cache_key not in TOP_PICKS_CACHE:
        return jsonify({
            'error': 'No top picks available',
            'message': 'Please refresh the top picks first'
        }), 404

    cached_data, _ = TOP_PICKS_CACHE[cache_key]
    buys = cached_data.get('top_buys', [])
    sells = cached_data.get('top_sells', [])

    # 1. Detect market regime
    market_regime = _detect_market_regime_enhanced(regime)

    # 2. Calculate trend analysis
    trend_analysis = _analyze_trend_strength(buys, sells, regime)

    # 3. Generate actionable signals
    trade_signals = _generate_trade_signals_enhanced(regime, market_regime, trend_analysis)

    # 4. Calculate optimal allocation
    portfolio_allocation = _calculate_optimal_allocation(trade_signals, market_regime, trend_analysis['bull_score'])

    # 5. Get profit-maximizing actions
    profit_actions = _get_profit_maximizing_actions(market_regime, trend_analysis, trade_signals)

    # 6. Get US Model Fixing7/8 lag-free regime analysis
    us_lag_free_regime = None
    if US_PROFIT_REGIME_AVAILABLE and US_PROFIT_REGIME_CLASSIFIER is not None:
        try:
            import yfinance as yf
            # Get SPY data for regime classification
            spy_data = yf.download('SPY', period='6mo', progress=False)
            if isinstance(spy_data.columns, pd.MultiIndex):
                spy_data.columns = spy_data.columns.get_level_values(0)

            # Get current VIX
            vix_data = yf.download('^VIX', period='1d', progress=False)
            if isinstance(vix_data.columns, pd.MultiIndex):
                vix_data.columns = vix_data.columns.get_level_values(0)
            current_vix = float(vix_data['Close'].iloc[-1]) if len(vix_data) > 0 else 20.0

            # Get profit-maximizing regime classification
            profit_regime = US_PROFIT_REGIME_CLASSIFIER.classify_regime_with_profit(
                spy_data, None, current_vix
            )

            us_lag_free_regime = {
                'source': 'us_model_fixing7_fixing8',
                'regime': profit_regime.regime,
                'transition_state': profit_regime.transition_state,
                'transition_phase': profit_regime.transition_phase.value,
                'transition_day': profit_regime.transition_day,
                'profit_score': round(profit_regime.profit_score, 1),
                'dynamic_multiplier': round(profit_regime.dynamic_multiplier, 3),
                'buy_multiplier': round(profit_regime.base_output.buy_multiplier, 3),
                'sell_multiplier': round(profit_regime.base_output.sell_multiplier, 3),
                'signals_detected': profit_regime.base_output.signals_detected,
                'position_params': {
                    'small_cap_multiplier': round(profit_regime.small_cap_multiplier, 2),
                    'large_cap_multiplier': round(profit_regime.large_cap_multiplier, 2),
                    'max_position_size': round(profit_regime.max_position_size * 100, 1),
                    'stop_loss': round(profit_regime.stop_loss * 100, 1),
                    'max_positions': profit_regime.max_positions,
                    'sizing_method': profit_regime.position_sizing_method,
                },
                'profit_targets': {
                    'target_1': round(profit_regime.profit_target_1 * 100, 1),
                    'target_2': round(profit_regime.profit_target_2 * 100, 1),
                    'target_3': round(profit_regime.profit_target_3 * 100, 1),
                    'trailing_stop': round(profit_regime.trailing_stop * 100, 1),
                },
                'capital_allocation': {k: round(v * 100, 1) for k, v in profit_regime.capital_allocation.items()},
                'sector_focus': profit_regime.sector_focus,
                'vix_level': round(current_vix, 1),
                'confidence': round(profit_regime.base_output.confidence, 3),
            }
            logger.info(f"[ML ANALYSIS FIXING7/8] Regime: {profit_regime.regime}, Phase: {profit_regime.transition_phase.value}")
        except Exception as e:
            logger.warning(f"[ML ANALYSIS FIXING7/8] Error: {e}")
            us_lag_free_regime = {'error': str(e), 'available': False}

    return jsonify({
        'regime': regime,
        'timestamp': datetime.now().isoformat(),
        'market_regime': market_regime,
        'trend_analysis': trend_analysis,
        'actionable_signals': trade_signals,
        'portfolio_allocation': portfolio_allocation,
        'profit_maximizing_actions': profit_actions,
        'us_lag_free_regime': us_lag_free_regime
    })


# ==========================================================================
# CHINA-SPECIFIC ENDPOINTS (DeepSeek + China ML Model)
# ==========================================================================
# These endpoints use 100% China/DeepSeek model - NO mixing with US/Intl model
# ==========================================================================

@app.route('/api/top-picks/china/realtime-test')
def china_realtime_test():
    """
    Run REAL-TIME analysis on China top picks using DeepSeek + China ML model.

    Uses 100% China/DeepSeek model functions:
    - DeepSeek API for policy/social/retail sentiment
    - China ML model with CSI300, CNY, HSI macro features
    - China-specific sector routing
    - 40% DeepSeek + 60% ML weighting

    NO mixing with US/Intl model code.
    """
    # Only for China regime
    regime = 'China'

    # Get current top picks from cache
    cache_key = f"top_picks_{regime}"
    if cache_key not in TOP_PICKS_CACHE:
        logger.warning("[CHINA REALTIME] No China top picks in cache - please load China tab first")
        return jsonify({
            'error': 'No China top picks loaded',
            'message': 'Please switch to China tab and wait for stocks to load, then try Real-Time Test again'
        }), 404

    cached_data, cache_time = TOP_PICKS_CACHE[cache_key]

    # Extract buy and sell signals
    buy_signals = [(p.get('ticker'), p.get('confidence', 50)) for p in cached_data.get('top_buys', [])]
    sell_signals = [(p.get('ticker'), p.get('confidence', 50)) for p in cached_data.get('top_sells', [])]

    logger.info(f"[CHINA REALTIME] Analyzing {len(buy_signals)} BUY and {len(sell_signals)} SELL signals")

    # Run China-specific real-time analysis
    buy_analysis = []
    sell_analysis = []

    # Analyze BUY signals using China model
    for ticker, confidence in buy_signals[:10]:
        if ticker:
            analysis = _analyze_china_ticker_realtime(ticker, confidence, 'BUY')
            if analysis:
                buy_analysis.append(analysis)

    # Analyze SELL signals using China model
    for ticker, confidence in sell_signals[:10]:
        if ticker:
            analysis = _analyze_china_ticker_realtime(ticker, confidence, 'SELL')
            if analysis:
                sell_analysis.append(analysis)

    # Calculate portfolio simulation with China-specific parameters
    portfolio_sim = _simulate_china_portfolio(buy_analysis, capital=10000)

    # Generate China-specific summary
    summary = _generate_china_analysis_summary(buy_analysis, sell_analysis)

    return jsonify({
        'regime': regime,
        'model': 'China/DeepSeek',
        'model_components': {
            'deepseek_api': True,
            'china_ml_model': True,
            'macro_features': ['CSI300', 'CNY', 'HSI'],
            'weighting': '40% DeepSeek + 60% ML'
        },
        'timestamp': datetime.now().isoformat(),
        'cache_age_seconds': int((datetime.now() - cache_time).total_seconds()),
        'buy_analysis': buy_analysis,
        'sell_analysis': sell_analysis,
        'portfolio_simulation': portfolio_sim,
        'summary': summary,
        'top_opportunities': sorted(
            buy_analysis,
            key=lambda x: x.get('expected_value', 0),
            reverse=True
        )[:5]
    })


@app.route('/api/top-picks/china/predict-trend')
def china_predict_market_trend():
    """
    Predict Bull/Bear market trend for China market using DeepSeek + China ML model.

    Uses 100% China/DeepSeek model functions:
    - DeepSeek API for policy environment analysis
    - China macro indicators (CSI300, CNY, HSI)
    - PBOC/CSRC regulatory sentiment
    - A-share/H-share market breadth

    NO mixing with US/Intl model code.
    """
    regime = 'China'

    # Get current China top picks
    cache_key = f"top_picks_{regime}"
    if cache_key not in TOP_PICKS_CACHE:
        logger.warning("[CHINA TREND] No China top picks in cache - please load China tab first")
        return jsonify({
            'error': 'No China top picks loaded',
            'message': 'Please switch to China tab and wait for stocks to load, then try Bull/Bear Prediction again'
        }), 404

    cached_data, _ = TOP_PICKS_CACHE[cache_key]
    buys = cached_data.get('top_buys', [])
    sells = cached_data.get('top_sells', [])

    # 1. Detect China market regime using DeepSeek
    market_regime = _detect_china_market_regime()

    # 2. Calculate trend analysis using China model
    trend_analysis = _analyze_china_trend_strength(buys, sells)

    # 3. Generate China-specific trade signals
    trade_signals = _generate_china_trade_signals(market_regime, trend_analysis)

    # 4. Calculate optimal allocation for China market
    portfolio_allocation = _calculate_china_allocation(trade_signals, market_regime, trend_analysis['bull_score'])

    # 5. Get China profit-maximizing actions
    profit_actions = _get_china_profit_actions(market_regime, trend_analysis, trade_signals)

    return jsonify({
        'regime': regime,
        'model': 'China/DeepSeek',
        'model_components': {
            'deepseek_api': True,
            'china_ml_model': True,
            'macro_features': ['CSI300', 'CNY', 'HSI'],
            'policy_analysis': True,
            'weighting': '40% DeepSeek + 60% ML'
        },
        'timestamp': datetime.now().isoformat(),
        'market_regime': market_regime,
        'trend_analysis': trend_analysis,
        'actionable_signals': trade_signals,
        'portfolio_allocation': portfolio_allocation,
        'profit_maximizing_actions': profit_actions
    })


# --------------------------------------------------------------------------
# CHINA-SPECIFIC HELPER FUNCTIONS (100% China/DeepSeek Model)
# --------------------------------------------------------------------------

def _analyze_china_ticker_realtime(ticker, orig_confidence, signal_type):
    """
    Analyze China ticker using 100% DeepSeek + China ML model.

    NO US/Intl model functions used:
    - Uses DeepSeek API for sentiment analysis
    - Uses China ML model for price prediction
    - Uses China macro features (CSI300, CNY, HSI)
    - Applies 40% DeepSeek + 60% ML weighting
    """
    try:
        logger.info(f"[CHINA REALTIME] Running DeepSeek + China ML for {ticker}")

        # ===== USE DEEPSEEK API FOR SENTIMENT =====
        analyzer = get_deepseek_analyzer()
        deepseek_analysis = analyzer.get_comprehensive_analysis(ticker)

        # ===== USE CHINA ML MODEL FOR PREDICTION =====
        # This calls _generate_china_pick_for_ticker which uses:
        # - DeepSeek API (40% weight)
        # - China ML model with CSI300, CNY, HSI features (60% weight)
        china_prediction = _generate_china_pick_for_ticker(ticker)

        if china_prediction is None:
            logger.warning(f"[CHINA REALTIME] China prediction failed for {ticker}")
            return None

        # Extract China model outputs
        china_direction = china_prediction.get('direction', 0)
        china_confidence = china_prediction.get('confidence', 0.5)
        china_expected_return = china_prediction.get('expected_return', 0)

        # Get current price from Yahoo Finance (using rate-limited fetcher)
        try:
            # Use rate-limited Yahoo fetcher to prevent 401 errors
            hist = get_yahoo_data_safe(ticker, period='1y', max_retries=3)

            if hist is None or len(hist) < 20:
                logger.warning(f"[CHINA REALTIME] Insufficient data for {ticker} (rate limited or unavailable)")
                return None

            close = hist['Close']
            high = hist['High']
            low = hist['Low']
            volume = hist['Volume']

            current_price = float(close.iloc[-1])
            prev = float(close.iloc[-2])
            change_1d = ((current_price - prev) / prev) * 100
            change_5d = ((current_price - float(close.iloc[-5])) / float(close.iloc[-5])) * 100 if len(close) >= 5 else 0
            change_20d = ((current_price - float(close.iloc[-20])) / float(close.iloc[-20])) * 100 if len(close) >= 20 else 0

            # Technical indicators
            rsi = _calculate_rsi_simple(close)
            macd_val, macd_signal_line = _calculate_macd_simple(close)

            # 52-week range
            high_52w = float(high.max())
            low_52w = float(low.min())
            range_pos = ((current_price - low_52w) / (high_52w - low_52w)) * 100 if high_52w != low_52w else 50

            # Volatility
            volatility = float(close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100)

        except Exception as e:
            logger.error(f"[CHINA REALTIME] Yahoo data error for {ticker}: {e}")
            return None

        # ===== CALCULATE P/L PROJECTIONS USING CHINA MODEL =====
        daily_vol = volatility / np.sqrt(252)

        if signal_type == 'BUY':
            target_conservative = current_price * (1 + daily_vol * 5 / 100)
            target_moderate = current_price * (1 + daily_vol * 10 / 100)
            stop_loss = current_price * (1 - daily_vol * 3 / 100)
        else:
            target_conservative = current_price * (1 - daily_vol * 5 / 100)
            target_moderate = current_price * (1 - daily_vol * 10 / 100)
            stop_loss = current_price * (1 + daily_vol * 3 / 100)

        # Risk/Reward calculation
        risk_reward = abs(target_moderate - current_price) / abs(current_price - stop_loss) if abs(current_price - stop_loss) > 0 else 0

        # Expected value calculation using China model confidence
        if signal_type == 'BUY':
            potential_gain = ((target_moderate - current_price) / current_price) * 100
            potential_loss = ((stop_loss - current_price) / current_price) * 100
        else:
            potential_gain = ((current_price - target_moderate) / current_price) * 100
            potential_loss = ((current_price - stop_loss) / current_price) * 100

        expected_value = (china_confidence * potential_gain) + ((1 - china_confidence) * potential_loss)

        # Add fields compatible with US model frontend display
        # This allows the same displayRealtimeResults function to work
        return {
            'ticker': ticker,
            'signal_type': signal_type,
            'current_price': round(current_price, 2),
            'change_1d': round(change_1d, 2),
            'change_5d': round(change_5d, 2),
            'change_20d': round(change_20d, 2),
            # Fields for frontend compatibility (same as US model)
            'rsi': round(rsi, 1),
            'volatility': round(volatility, 1),
            'target_5d': round(target_conservative, 2),
            'target_10d': round(target_moderate, 2),
            'stop_loss': round(stop_loss, 2),
            'potential_gain_5d': round(((target_conservative - current_price) / current_price) * 100, 2),
            'potential_gain_10d': round(((target_moderate - current_price) / current_price) * 100, 2),
            'risk_reward': round(risk_reward, 2),
            'confidence': round(china_confidence * 100, 1),
            'technical_status': 'CHINA MODEL' if china_direction > 0 else 'CAUTION',
            # Profit maximizing data for frontend Market Regime display (China-specific)
            # Uses China transition detector for real regime data
            'profit_maximizing': _get_china_regime_for_display(ticker, china_direction, china_confidence, hist),
            # China Model Data
            'china_model': {
                'direction': china_direction,
                'confidence': round(china_confidence, 3),
                'expected_return': round(china_expected_return, 4),
                'model_used': 'DeepSeek + China ML',
                'weighting': '40% DeepSeek + 60% ML'
            },
            # DeepSeek Sentiment Data
            'deepseek_sentiment': {
                'policy_sentiment': round(deepseek_analysis.get('policy_sentiment', 0), 2),
                'social_sentiment': round(deepseek_analysis.get('social_sentiment', 0), 2),
                'retail_sentiment': round(deepseek_analysis.get('retail_sentiment', 50), 1),
                'policy_alignment': round(deepseek_analysis.get('policy_alignment', 5), 1),
                'deepseek_used': deepseek_analysis.get('deepseek_used', False),
                'reason': deepseek_analysis.get('reason', '')
            },
            # Technical Indicators
            'technical': {
                'rsi': round(rsi, 1),
                'macd': round(macd_val, 4),
                'macd_signal': round(macd_signal_line, 4),
                'range_52w_position': round(range_pos, 1),
                'volatility': round(volatility, 1)
            },
            # P/L Projections
            'projections': {
                'target_5d': round(target_conservative, 2),
                'target_10d': round(target_moderate, 2),
                'stop_loss': round(stop_loss, 2),
                'potential_gain_5d': round(((target_conservative - current_price) / current_price) * 100, 2),
                'potential_gain_10d': round(((target_moderate - current_price) / current_price) * 100, 2),
                'risk_reward_ratio': round(risk_reward, 2)
            },
            'expected_value': round(expected_value, 2),
            'original_confidence': orig_confidence
        }

    except Exception as e:
        logger.error(f"[CHINA REALTIME] Error analyzing {ticker}: {e}")
        return None


def _get_china_regime_for_display(ticker, china_direction, china_confidence, hist):
    """
    Get China regime data for frontend display using real China transition detector.
    Returns data in the same format as US profit_maximizing for frontend compatibility.

    FIXING2 Enhancements:
    - Dynamic stop-loss based on volatility
    - Trailing stops per regime (BULL:10%, NEUTRAL:8%, BEAR:5%, HIGH_VOL:3%)
    - Stricter EV thresholds displayed
    - Quality filter status
    """
    try:
        # Use China transition detector if available
        if CHINA_TRANSITION_AVAILABLE and CHINA_TRANSITION_DETECTOR is not None:
            transition_result = CHINA_TRANSITION_DETECTOR.detect_transition(
                hist,
                current_regime='NEUTRAL'
            )

            # Map transition to regime
            regime = 'china_neutral'
            if transition_result.transition_type == 'TRANSITION_TO_BULL':
                regime = 'china_bull'
            elif transition_result.transition_type == 'TRANSITION_TO_BEAR':
                regime = 'china_bear'
            elif transition_result.is_transition:
                regime = 'china_transition'

            # Calculate multipliers based on transition
            buy_mult = 1.0 + (transition_result.recommended_allocation_adjustment * 0.5)
            sell_mult = 1.0 - (transition_result.recommended_allocation_adjustment * 0.3)

            # FIXING2: Get regime-specific parameters
            regime_params = {
                'china_bull': {'stop_loss': 8.0, 'trailing_stop': 10.0, 'ev_threshold': 0.75, 'exit_days': 10},
                'china_neutral': {'stop_loss': 6.0, 'trailing_stop': 8.0, 'ev_threshold': 1.0, 'exit_days': 7},
                'china_bear': {'stop_loss': 5.0, 'trailing_stop': 5.0, 'ev_threshold': 1.25, 'exit_days': 5},
                'china_transition': {'stop_loss': 6.0, 'trailing_stop': 8.0, 'ev_threshold': 1.0, 'exit_days': 7},
            }
            current_params = regime_params.get(regime, regime_params['china_neutral'])

            # FIXING2: Calculate dynamic stop-loss based on volatility
            if len(hist) >= 20:
                volatility = hist['Close'].pct_change().std() * 100  # as percentage
                vol_percentile = min(1.0, volatility / 5.0)  # Normalize to 0-1 (5% = high vol)
                vol_adjustment = 1.2 - vol_percentile * 0.4
                dynamic_stop = current_params['stop_loss'] * vol_adjustment
                dynamic_stop = max(2.0, min(15.0, dynamic_stop))
            else:
                dynamic_stop = current_params['stop_loss']

            return {
                'profit_score': round(china_confidence * 100, 1),
                'dynamic_multiplier': round(buy_mult, 3),
                'buy_multiplier': round(buy_mult, 3),
                'sell_multiplier': round(sell_mult, 3),
                'recommended_position': 5.0,
                'regime': regime,
                'transition_phase': transition_result.transition_type.lower() if transition_result.transition_type else 'stable',
                'profit_targets': {
                    'target_1': 10.0,
                    'target_2': 20.0,
                    'target_3': 35.0,
                },
                # FIXING2: Dynamic stop-loss and trailing stop
                'stop_loss_pct': round(dynamic_stop, 1),
                'trailing_stop_pct': current_params['trailing_stop'],
                'vix_level': 0,  # N/A for China (uses HSI instead)
                'model': 'DeepSeek + China ML (Fixing2)',
                'signals_detected': transition_result.signals_detected if hasattr(transition_result, 'signals_detected') else [],
                'transition_confidence': round(transition_result.confidence * 100, 1) if hasattr(transition_result, 'confidence') else 50.0,
                # FIXING2: Additional info for frontend
                'fixing2_active': True,
                'ev_threshold': current_params['ev_threshold'],
                'exit_days': current_params['exit_days'],
                'quality_filter': True,
                'momentum_filter': True,
            }
        else:
            # Fallback if transition detector not available
            return {
                'profit_score': round(china_confidence * 100, 1),
                'dynamic_multiplier': 1.0 if china_direction > 0 else 0.8,
                'buy_multiplier': 1.0 if china_direction > 0 else 0.8,
                'sell_multiplier': 0.8 if china_direction > 0 else 1.0,
                'recommended_position': 5.0,
                'regime': 'china_market',
                'transition_phase': 'deepseek_analysis',
                'profit_targets': {
                    'target_1': 10.0,
                    'target_2': 20.0,
                    'target_3': 35.0,
                },
                # FIXING2: Default values (detector not available)
                'stop_loss_pct': 6.0,  # Neutral default
                'trailing_stop_pct': 8.0,  # Neutral default
                'vix_level': 0,
                'model': 'DeepSeek + China ML (Fixing2 Fallback)',
                'fixing2_active': True,
                'ev_threshold': 1.0,  # Neutral default
                'exit_days': 7,  # Neutral default
                'quality_filter': True,
                'momentum_filter': True,
            }
    except Exception as e:
        logger.warning(f"[CHINA REGIME DISPLAY] Error getting regime for {ticker}: {e}")
        return {
            'profit_score': round(china_confidence * 100, 1),
            'dynamic_multiplier': 1.0,
            'buy_multiplier': 1.0,
            'sell_multiplier': 0.8,
            'recommended_position': 5.0,
            'regime': 'china_market',
            'transition_phase': 'unknown',
            'profit_targets': {'target_1': 10.0, 'target_2': 20.0, 'target_3': 35.0},
            # FIXING2: Default values (error case)
            'stop_loss_pct': 6.0,
            'trailing_stop_pct': 8.0,
            'vix_level': 0,
            'model': 'DeepSeek + China ML (Fixing2)',
            'fixing2_active': True,
            'ev_threshold': 1.0,
            'exit_days': 7,
            'quality_filter': True,
            'momentum_filter': True,
        }


def _detect_china_market_regime():
    """
    Detect China market regime using DeepSeek + China macro indicators.

    Uses 100% China-specific data:
    - CSI300 index for A-share sentiment
    - Hang Seng Index for H-share sentiment
    - CNY exchange rate
    - DeepSeek for policy environment

    NO VIX, SPY, DXY (US/Intl features).
    """
    try:
        logger.info("[CHINA REGIME] Running DeepSeek + China macro regime detection")

        # ===== GET CHINA MACRO INDICATORS =====
        china_indicators = {}

        # CSI300 Index (A-shares)
        try:
            csi300 = yf.Ticker('000300.SS')
            csi300_hist = csi300.history(period='60d')
            if len(csi300_hist) >= 20:
                csi300_close = csi300_hist['Close']
                csi300_returns = csi300_close.pct_change().dropna()
                china_indicators['csi300_rsi'] = _calculate_rsi_simple(csi300_close)
                china_indicators['csi300_trend'] = float(csi300_returns.tail(20).mean())
                china_indicators['csi300_volatility'] = float(csi300_returns.std() * np.sqrt(252) * 100)
                china_indicators['csi300_price'] = float(csi300_close.iloc[-1])
        except Exception as e:
            logger.warning(f"[CHINA REGIME] CSI300 fetch failed: {e}")
            china_indicators['csi300_rsi'] = 50
            china_indicators['csi300_trend'] = 0
            china_indicators['csi300_volatility'] = 20

        # Hang Seng Index (H-shares)
        try:
            hsi = yf.Ticker('^HSI')
            hsi_hist = hsi.history(period='60d')
            if len(hsi_hist) >= 20:
                hsi_close = hsi_hist['Close']
                hsi_returns = hsi_close.pct_change().dropna()
                china_indicators['hsi_rsi'] = _calculate_rsi_simple(hsi_close)
                china_indicators['hsi_trend'] = float(hsi_returns.tail(20).mean())
                china_indicators['hsi_volatility'] = float(hsi_returns.std() * np.sqrt(252) * 100)
                china_indicators['hsi_price'] = float(hsi_close.iloc[-1])
        except Exception as e:
            logger.warning(f"[CHINA REGIME] HSI fetch failed: {e}")
            china_indicators['hsi_rsi'] = 50
            china_indicators['hsi_trend'] = 0
            china_indicators['hsi_volatility'] = 20

        # CNY/USD Exchange Rate
        try:
            cny = yf.Ticker('CNY=X')
            cny_hist = cny.history(period='30d')
            if len(cny_hist) >= 5:
                cny_close = cny_hist['Close']
                china_indicators['cny_rate'] = float(cny_close.iloc[-1])
                china_indicators['cny_change_5d'] = ((float(cny_close.iloc[-1]) - float(cny_close.iloc[-5])) / float(cny_close.iloc[-5])) * 100
        except Exception as e:
            logger.warning(f"[CHINA REGIME] CNY fetch failed: {e}")
            china_indicators['cny_rate'] = 7.2
            china_indicators['cny_change_5d'] = 0

        # ===== GET DEEPSEEK POLICY ANALYSIS =====
        analyzer = get_deepseek_analyzer()
        # Use a major China stock for overall policy sentiment
        policy_analysis = analyzer.get_comprehensive_analysis('0700.HK')  # Tencent as proxy

        policy_sentiment = policy_analysis.get('policy_sentiment', 0)
        policy_alignment = policy_analysis.get('policy_alignment', 5)

        # ===== DETERMINE REGIME =====
        csi300_vol = china_indicators.get('csi300_volatility', 20)
        hsi_vol = china_indicators.get('hsi_volatility', 20)
        avg_vol = (csi300_vol + hsi_vol) / 2

        csi300_trend = china_indicators.get('csi300_trend', 0)
        hsi_trend = china_indicators.get('hsi_trend', 0)
        avg_trend = (csi300_trend + hsi_trend) / 2

        csi300_rsi = china_indicators.get('csi300_rsi', 50)
        hsi_rsi = china_indicators.get('hsi_rsi', 50)
        avg_rsi = (csi300_rsi + hsi_rsi) / 2

        # Regime determination using China-specific logic
        if avg_vol > 35:
            detected_regime = 'VOLATILE'
        elif avg_vol > 30 and policy_sentiment < -0.3:
            detected_regime = 'CRISIS'
        elif avg_trend > 0.001 and policy_sentiment > 0.2 and policy_alignment > 6:
            detected_regime = 'BULL'
        elif avg_trend < -0.001 and policy_sentiment < -0.2:
            detected_regime = 'BEAR'
        elif policy_alignment > 7:
            detected_regime = 'POLICY_BULL'  # China-specific: strong policy support
        elif policy_alignment < 3:
            detected_regime = 'POLICY_BEAR'  # China-specific: regulatory crackdown
        elif avg_rsi > 70:
            detected_regime = 'RISK_ON'
        elif avg_rsi < 30:
            detected_regime = 'RISK_OFF'
        elif abs(avg_trend) < 0.0005:
            detected_regime = 'SIDEWAYS'
        else:
            detected_regime = 'NEUTRAL'

        logger.info(f"[CHINA REGIME] Final regime: {detected_regime} "
                   f"(CSI300 RSI: {csi300_rsi:.1f}, HSI RSI: {hsi_rsi:.1f}, "
                   f"Policy: {policy_sentiment:.2f}, Alignment: {policy_alignment:.1f})")

        return detected_regime

    except Exception as e:
        logger.error(f"[CHINA REGIME] Error detecting regime: {e}")
        return 'NEUTRAL'


def _analyze_china_trend_strength(buys, sells):
    """
    Analyze China market trend strength using DeepSeek + China model metrics.

    Uses 100% China-specific data:
    - DeepSeek policy/social sentiment
    - China macro indicators (CSI300, HSI, CNY)
    - A-share/H-share market breadth

    NO US/Intl model metrics.
    """
    total_signals = len(buys) + len(sells)
    buy_ratio = len(buys) / total_signals if total_signals > 0 else 0.5

    # Extract China model confidence
    avg_buy_conf = sum(b.get('confidence', 50) for b in buys) / len(buys) if buys else 50
    avg_sell_conf = sum(s.get('confidence', 50) for s in sells) / len(sells) if sells else 50

    # Extract DeepSeek sentiment from predictions
    avg_policy_sentiment = 0
    avg_social_sentiment = 0
    deepseek_count = 0

    for b in buys:
        ds = b.get('deepseek_analysis', {})
        if ds.get('deepseek_used', False):
            avg_policy_sentiment += ds.get('policy_sentiment', 0)
            avg_social_sentiment += ds.get('social_sentiment', 0)
            deepseek_count += 1

    if deepseek_count > 0:
        avg_policy_sentiment /= deepseek_count
        avg_social_sentiment /= deepseek_count

    # Get China market indicators
    china_market_data = _get_china_market_indicators()

    # Composite Bull Score (0-100) - China-specific factors
    bull_score = 0
    factors = []

    # Factor 1: Buy/Sell Ratio (0-20 points)
    ratio_score = buy_ratio * 20
    bull_score += ratio_score
    factors.append({
        'name': 'China Buy/Sell Ratio',
        'value': f"{buy_ratio:.1%}",
        'score': round(ratio_score, 1),
        'interpretation': 'Bullish' if buy_ratio > 0.6 else 'Bearish' if buy_ratio < 0.4 else 'Neutral',
        'source': 'DeepSeek + China ML'
    })

    # Factor 2: Policy Sentiment (0-20 points) - China-specific
    policy_score = (avg_policy_sentiment + 1) * 10  # Convert -1 to 1 -> 0 to 20
    bull_score += policy_score
    factors.append({
        'name': 'Policy Sentiment',
        'value': f"{avg_policy_sentiment:+.2f}",
        'score': round(policy_score, 1),
        'interpretation': 'Supportive' if avg_policy_sentiment > 0.2 else 'Restrictive' if avg_policy_sentiment < -0.2 else 'Neutral',
        'source': 'DeepSeek API'
    })

    # Factor 3: Social Sentiment (0-20 points) - China-specific (Weibo, Xueqiu)
    social_score = (avg_social_sentiment + 1) * 10  # Convert -1 to 1 -> 0 to 20
    bull_score += social_score
    factors.append({
        'name': 'Social Sentiment',
        'value': f"{avg_social_sentiment:+.2f}",
        'score': round(social_score, 1),
        'interpretation': 'Bullish' if avg_social_sentiment > 0.2 else 'Bearish' if avg_social_sentiment < -0.2 else 'Neutral',
        'source': 'DeepSeek API (Weibo/Xueqiu)'
    })

    # Factor 4: CSI300 RSI (0-20 points)
    csi300_rsi = china_market_data.get('csi300_rsi', 50)
    if 30 < csi300_rsi < 70:
        rsi_score = 20
    else:
        rsi_score = max(0, 20 - abs(csi300_rsi - 50) / 2.5)
    bull_score += rsi_score
    factors.append({
        'name': 'CSI300 RSI',
        'value': f"{csi300_rsi:.1f}",
        'score': round(rsi_score, 1),
        'interpretation': 'Overbought' if csi300_rsi > 70 else 'Oversold' if csi300_rsi < 30 else 'Healthy',
        'source': 'China A-Share Index'
    })

    # Factor 5: CNY Stability (0-20 points) - China-specific
    cny_change = abs(china_market_data.get('cny_change_5d', 0))
    cny_score = max(0, 20 - cny_change * 10)  # Penalize large CNY moves
    bull_score += cny_score
    factors.append({
        'name': 'CNY Stability',
        'value': f"{china_market_data.get('cny_change_5d', 0):+.2f}%",
        'score': round(cny_score, 1),
        'interpretation': 'Stable' if cny_change < 0.5 else 'Volatile' if cny_change > 1.5 else 'Normal',
        'source': 'China Macro Features'
    })

    # Determine trend
    if bull_score >= 70:
        trend = 'STRONG_BULL'
        trend_label = 'Strong Bull Market (China)'
        recommendation = 'Aggressive buying in policy-aligned sectors. High confidence in government support.'
    elif bull_score >= 55:
        trend = 'BULL'
        trend_label = 'Bull Market (China)'
        recommendation = 'Favor long positions in tech/EV/consumer sectors with policy tailwinds.'
    elif bull_score >= 45:
        trend = 'NEUTRAL'
        trend_label = 'Neutral/Sideways (China)'
        recommendation = 'Mixed policy signals. Focus on individual stock fundamentals.'
    elif bull_score >= 30:
        trend = 'BEAR'
        trend_label = 'Bear Market (China)'
        recommendation = 'Reduce exposure. Watch for regulatory announcements.'
    else:
        trend = 'STRONG_BEAR'
        trend_label = 'Strong Bear Market (China)'
        recommendation = 'High risk from policy headwinds. Consider defensive sectors or cash.'

    return {
        'trend': trend,
        'trend_label': trend_label,
        'bull_score': round(bull_score, 1),
        'recommendation': recommendation,
        'factors': factors,
        'signal_counts': {
            'buy_signals': len(buys),
            'sell_signals': len(sells),
            'buy_ratio': round(buy_ratio, 2)
        },
        'market_data': china_market_data,
        # China Model Summary
        'china_model_info': {
            'model': 'DeepSeek + China ML',
            'components': ['DeepSeek API', 'China ML (LightGBM + XGBoost + LSTM)'],
            'features': ['China Macro (CSI300, CNY, HSI)', 'Policy Sentiment', 'Social Sentiment (Weibo/Xueqiu)'],
            'weighting': '40% DeepSeek + 60% ML',
            'avg_buy_confidence': round(avg_buy_conf, 1),
            'avg_sell_confidence': round(avg_sell_conf, 1),
            'avg_policy_sentiment': round(avg_policy_sentiment, 2),
            'avg_social_sentiment': round(avg_social_sentiment, 2)
        }
    }


def _get_china_market_indicators():
    """Get China-specific market indicators (NO US/Intl indicators)."""
    try:
        indicators = {}

        # CSI300 Index
        try:
            csi300 = yf.Ticker('000300.SS')
            hist = csi300.history(period='3mo')
            if len(hist) >= 20:
                close = hist['Close']
                indicators['csi300_rsi'] = round(_calculate_rsi_simple(close), 1)
                indicators['csi300_price'] = round(float(close.iloc[-1]), 2)
                indicators['csi300_change_5d'] = round(((float(close.iloc[-1]) - float(close.iloc[-5])) / float(close.iloc[-5])) * 100, 2) if len(close) >= 5 else 0
        except:
            indicators['csi300_rsi'] = 50
            indicators['csi300_price'] = 0
            indicators['csi300_change_5d'] = 0

        # Hang Seng Index
        try:
            hsi = yf.Ticker('^HSI')
            hist = hsi.history(period='3mo')
            if len(hist) >= 20:
                close = hist['Close']
                indicators['hsi_rsi'] = round(_calculate_rsi_simple(close), 1)
                indicators['hsi_price'] = round(float(close.iloc[-1]), 2)
                indicators['hsi_change_5d'] = round(((float(close.iloc[-1]) - float(close.iloc[-5])) / float(close.iloc[-5])) * 100, 2) if len(close) >= 5 else 0
        except:
            indicators['hsi_rsi'] = 50
            indicators['hsi_price'] = 0
            indicators['hsi_change_5d'] = 0

        # CNY Exchange Rate
        try:
            cny = yf.Ticker('CNY=X')
            hist = cny.history(period='1mo')
            if len(hist) >= 5:
                close = hist['Close']
                indicators['cny_rate'] = round(float(close.iloc[-1]), 4)
                indicators['cny_change_5d'] = round(((float(close.iloc[-1]) - float(close.iloc[-5])) / float(close.iloc[-5])) * 100, 2) if len(close) >= 5 else 0
        except:
            indicators['cny_rate'] = 7.2
            indicators['cny_change_5d'] = 0

        return indicators

    except Exception as e:
        logger.error(f"[CHINA MARKET] Error getting indicators: {e}")
        return {'csi300_rsi': 50, 'hsi_rsi': 50, 'cny_rate': 7.2, 'cny_change_5d': 0}


def _generate_china_trade_signals(market_regime, trend_analysis):
    """Generate China-specific trade signals based on policy environment."""
    china_strategies = {
        'BULL': {
            'primary_action': 'BUY policy-aligned sectors (EV, Tech, Green Energy)',
            'secondary_action': 'Accumulate on dips',
            'position_size': 'Full (100%)',
            'stop_loss': 'Wide (12-15%)',
            'target_gain': '25-35%'
        },
        'POLICY_BULL': {
            'primary_action': 'BUY state-favored companies (SOEs, strategic sectors)',
            'secondary_action': 'Focus on 14th Five-Year Plan beneficiaries',
            'position_size': 'Aggressive (120%)',
            'stop_loss': 'Wide (15%)',
            'target_gain': '30-50%'
        },
        'BEAR': {
            'primary_action': 'REDUCE exposure to regulated sectors',
            'secondary_action': 'Focus on consumer staples',
            'position_size': 'Reduced (50%)',
            'stop_loss': 'Tight (5-8%)',
            'target_gain': '10-15%'
        },
        'POLICY_BEAR': {
            'primary_action': 'AVOID sectors under regulatory scrutiny',
            'secondary_action': 'Move to defensive sectors (healthcare, utilities)',
            'position_size': 'Minimal (30%)',
            'stop_loss': 'Very tight (3-5%)',
            'target_gain': '5-10%'
        },
        'VOLATILE': {
            'primary_action': 'Reduce overall China exposure',
            'secondary_action': 'Focus on H-shares with USD revenue',
            'position_size': 'Small (30%)',
            'stop_loss': 'Very tight (3%)',
            'target_gain': 'Quick 5-8% scalps'
        },
        'CRISIS': {
            'primary_action': 'CASH or inverse ETFs',
            'secondary_action': 'Wait for policy clarity',
            'position_size': 'Minimal (10%)',
            'stop_loss': 'Immediate (2%)',
            'target_gain': 'Capital preservation'
        },
        'SIDEWAYS': {
            'primary_action': 'Range trading in established names',
            'secondary_action': 'Collect dividends from SOEs',
            'position_size': 'Moderate (60%)',
            'stop_loss': 'At range boundaries',
            'target_gain': '5-10%'
        }
    }

    strategy = china_strategies.get(market_regime, {
        'primary_action': 'Wait for clearer policy signal',
        'secondary_action': 'Monitor PBOC/CSRC announcements',
        'position_size': 'Minimal (20%)',
        'stop_loss': 'N/A',
        'target_gain': 'N/A'
    })

    risk_levels = {
        'BULL': 'LOW',
        'POLICY_BULL': 'LOW',
        'RISK_ON': 'MODERATE',
        'SIDEWAYS': 'MODERATE',
        'NEUTRAL': 'MODERATE',
        'RISK_OFF': 'HIGH',
        'BEAR': 'HIGH',
        'POLICY_BEAR': 'VERY HIGH',
        'VOLATILE': 'VERY HIGH',
        'CRISIS': 'EXTREME'
    }

    return {
        'market_regime': market_regime,
        'strategy': strategy,
        'risk_level': risk_levels.get(market_regime, 'MODERATE'),
        'recommended_actions': [
            f"1. {strategy['primary_action']}",
            f"2. {strategy['secondary_action']}",
            f"3. Position size: {strategy['position_size']}",
            f"4. Target gain: {strategy['target_gain']}",
            f"5. Stop loss: {strategy['stop_loss']}"
        ],
        'confidence': trend_analysis['bull_score']
    }


def _calculate_china_allocation(trade_signals, market_regime, bull_score):
    """Calculate portfolio allocation for China market."""
    # China-specific allocation based on policy environment
    allocations = {
        'BULL': {'china_stocks': 70, 'h_shares': 20, 'bonds': 5, 'cash': 5},
        'POLICY_BULL': {'china_stocks': 80, 'h_shares': 15, 'bonds': 0, 'cash': 5},
        'BEAR': {'china_stocks': 30, 'h_shares': 20, 'bonds': 30, 'cash': 20},
        'POLICY_BEAR': {'china_stocks': 10, 'h_shares': 20, 'bonds': 40, 'cash': 30},
        'VOLATILE': {'china_stocks': 20, 'h_shares': 20, 'bonds': 30, 'cash': 30},
        'CRISIS': {'china_stocks': 5, 'h_shares': 10, 'bonds': 35, 'cash': 50},
        'SIDEWAYS': {'china_stocks': 40, 'h_shares': 25, 'bonds': 20, 'cash': 15},
        'NEUTRAL': {'china_stocks': 45, 'h_shares': 25, 'bonds': 15, 'cash': 15}
    }

    allocation = allocations.get(market_regime, allocations['NEUTRAL'])

    # Adjust based on bull score
    if bull_score > 70:
        allocation['china_stocks'] = min(90, allocation['china_stocks'] + 10)
        allocation['cash'] = max(5, allocation['cash'] - 10)
    elif bull_score < 30:
        allocation['china_stocks'] = max(5, allocation['china_stocks'] - 15)
        allocation['cash'] = min(60, allocation['cash'] + 15)

    return {
        'recommended_allocation': allocation,
        'market_regime': market_regime,
        'bull_score': bull_score,
        'rebalance_urgency': 'HIGH' if bull_score < 30 or bull_score > 80 else 'LOW'
    }


def _get_china_profit_actions(market_regime, trend_analysis, trade_signals):
    """Get profit-maximizing actions for China market."""
    actions_map = {
        'BULL': [
            "MAX PROFIT: Buy 2x China ETFs (CWEB, YINN, CHAU)",
            "Focus on EV sector: BYD, NIO, Li Auto",
            "Accumulate tech leaders: Tencent, Alibaba, JD",
            "Hold 10-20 days for policy momentum"
        ],
        'POLICY_BULL': [
            "MAX PROFIT: Focus on state-favored sectors",
            "Buy semiconductor: SMIC, Hua Hong",
            "Buy green energy: LONGi, Sungrow",
            "Watch for stimulus announcements"
        ],
        'BEAR': [
            "PROTECT: Reduce China exposure to 30%",
            "Focus on consumer staples: Moutai, Wuliangye",
            "Avoid property developers",
            "Consider inverse China ETFs (YANG)"
        ],
        'POLICY_BEAR': [
            "PROTECT: Minimize China exposure",
            "Exit education, gaming, fintech sectors",
            "Hold only defensive names",
            "Wait for regulatory clarity"
        ],
        'VOLATILE': [
            "CAUTION: Reduce position sizes",
            "Focus on H-shares with USD exposure",
            "Quick trades only (2-3 days)",
            "Set tight stops"
        ],
        'CRISIS': [
            "CAPITAL PRESERVATION: Move to cash",
            "Exit all speculative positions",
            "Consider inverse ETFs for hedge",
            "Wait for PBOC intervention signals"
        ],
        'SIDEWAYS': [
            "INCOME: Focus on dividend SOEs",
            "Range trade between support/resistance",
            "Collect premiums via covered calls",
            "Patience until trend emerges"
        ]
    }

    return actions_map.get(market_regime, [
        "Wait for clearer market signal",
        "Monitor policy announcements",
        "Keep positions small",
        "Focus on risk management"
    ])


def _simulate_china_portfolio(buy_analysis, capital=10000):
    """Simulate portfolio with China-specific parameters."""
    if not buy_analysis:
        return {'error': 'No buy signals to simulate'}

    num_positions = min(len(buy_analysis), 10)
    position_size = capital / num_positions

    positions = []
    total_5d_pl = 0
    total_10d_pl = 0

    for analysis in buy_analysis[:num_positions]:
        current_price = analysis.get('current_price', 0)
        if current_price <= 0:
            continue

        shares = int(position_size / current_price)
        entry_value = shares * current_price

        projections = analysis.get('projections', {})
        target_5d = projections.get('target_5d', current_price)
        target_10d = projections.get('target_10d', current_price)

        pl_5d = shares * (target_5d - current_price)
        pl_10d = shares * (target_10d - current_price)

        total_5d_pl += pl_5d
        total_10d_pl += pl_10d

        positions.append({
            'ticker': analysis.get('ticker'),
            'shares': shares,
            'entry_value': round(entry_value, 2),
            'target_5d_value': round(shares * target_5d, 2),
            'target_10d_value': round(shares * target_10d, 2),
            'pl_5d': round(pl_5d, 2),
            'pl_10d': round(pl_10d, 2),
            'china_model_confidence': analysis.get('china_model', {}).get('confidence', 0.5),
            'policy_sentiment': analysis.get('deepseek_sentiment', {}).get('policy_sentiment', 0)
        })

    return {
        'starting_capital': capital,
        'num_positions': num_positions,
        'position_size': round(position_size, 2),
        'positions': positions,
        'total_5d_pl': round(total_5d_pl, 2),
        'total_10d_pl': round(total_10d_pl, 2),
        'projected_5d_return': round((total_5d_pl / capital) * 100, 2),
        'projected_10d_return': round((total_10d_pl / capital) * 100, 2),
        'model_used': 'DeepSeek + China ML'
    }


def _generate_china_analysis_summary(buy_analysis, sell_analysis):
    """Generate summary for China market analysis."""
    total_buys = len(buy_analysis)
    total_sells = len(sell_analysis)

    # Average China model confidence
    avg_buy_confidence = sum(a.get('china_model', {}).get('confidence', 0.5) for a in buy_analysis) / total_buys if total_buys > 0 else 0.5
    avg_sell_confidence = sum(a.get('china_model', {}).get('confidence', 0.5) for a in sell_analysis) / total_sells if total_sells > 0 else 0.5

    # Average DeepSeek sentiment
    avg_policy = sum(a.get('deepseek_sentiment', {}).get('policy_sentiment', 0) for a in buy_analysis) / total_buys if total_buys > 0 else 0
    avg_social = sum(a.get('deepseek_sentiment', {}).get('social_sentiment', 0) for a in buy_analysis) / total_buys if total_buys > 0 else 0

    # Count strong signals
    strong_buys = sum(1 for a in buy_analysis if a.get('china_model', {}).get('confidence', 0) > 0.65)

    return {
        'total_buy_signals': total_buys,
        'total_sell_signals': total_sells,
        'strong_buy_signals': strong_buys,
        'avg_buy_confidence': round(avg_buy_confidence * 100, 1),
        'avg_sell_confidence': round(avg_sell_confidence * 100, 1),
        'avg_policy_sentiment': round(avg_policy, 2),
        'avg_social_sentiment': round(avg_social, 2),
        'model_used': 'DeepSeek + China ML',
        'model_components': ['DeepSeek API (40%)', 'China ML Model (60%)'],
        'macro_features': ['CSI300', 'CNY', 'HSI'],
        'sentiment_sources': ['Government Policy', 'Weibo', 'Xueqiu', 'East Money']
    }


# --------------------------------------------------------------------------
# HELPER FUNCTIONS FOR REAL-TIME ANALYSIS (US/Intl Model)
# --------------------------------------------------------------------------

def _analyze_ticker_realtime(ticker, orig_confidence, signal_type):
    """
    Analyze single ticker using the FULL US/Intl ML model (HybridEnsemblePredictor).

    This function now properly uses generate_prediction() to get:
    - ML predictions from trained models (LightGBM, XGBoost, LSTM)
    - US macro features (VIX, SPY, DXY, GLD)
    - Sentiment analysis (FinBERT + VADER)
    - Volatility regime detection
    - All model confidence scores and signals
    """
    try:
        logger.info(f"[REALTIME ML] Running full ML prediction for {ticker}")

        # ===== USE THE FULL US/INTL ML MODEL =====
        # This calls generate_prediction() which uses:
        # - HybridEnsemblePredictor (LightGBM + XGBoost + LSTM)
        # - US/Intl macro features (VIX, SPY, DXY, GLD)
        # - Sentiment analysis (FinBERT + VADER)
        # - Volatility features and regime detection
        with PREDICTION_LOCK:
            ml_result = generate_prediction(ticker, account_size=100000)

        if ml_result.get('status') == 'error':
            logger.warning(f"[REALTIME ML] ML prediction failed for {ticker}: {ml_result.get('error')}")
            return None

        # Extract ML model outputs
        prediction = ml_result.get('prediction', {})
        trading_signal = ml_result.get('trading_signal', {})
        market_context = ml_result.get('market_context', {})
        model_info = ml_result.get('model_info', {})

        # Get current price
        current_price = ml_result.get('current_price', 0)
        if current_price <= 0:
            logger.warning(f"[REALTIME ML] Invalid price for {ticker}")
            return None

        # ML Model Direction and Confidence
        ml_direction = prediction.get('direction', 0)
        ml_confidence = prediction.get('direction_confidence', 0.5)
        ml_expected_return = prediction.get('expected_return', 0)
        ml_volatility = prediction.get('volatility', 0.02)

        # Get position details from trading signal
        position = trading_signal.get('position', {}) or {}
        entry_price = position.get('entry_price') or trading_signal.get('entry_price') or current_price
        stop_loss = position.get('stop_loss') or trading_signal.get('stop_loss')
        take_profit = position.get('take_profit') or trading_signal.get('take_profit')

        # Fetch additional Yahoo Finance data for technical indicators display (rate-limited)
        try:
            # Use rate-limited Yahoo fetcher to prevent 401 errors
            hist = get_yahoo_data_safe(ticker, period='1y', max_retries=3)

            if hist is not None and len(hist) >= 50:
                close = hist['Close']
                high = hist['High']
                low = hist['Low']
                volume = hist['Volume']

                prev = float(close.iloc[-2])
                change_1d = ((current_price - prev) / prev) * 100
                change_5d = ((current_price - float(close.iloc[-5])) / float(close.iloc[-5])) * 100 if len(close) >= 5 else 0
                change_20d = ((current_price - float(close.iloc[-20])) / float(close.iloc[-20])) * 100 if len(close) >= 20 else 0

                # Technical indicators for display
                rsi = _calculate_rsi_simple(close)
                macd_val, macd_signal_line = _calculate_macd_simple(close)

                ma20 = float(close.rolling(20).mean().iloc[-1])
                ma50 = float(close.rolling(50).mean().iloc[-1])
                ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

                high_52w = float(high.max())
                low_52w = float(low.min())
                range_pos = ((current_price - low_52w) / (high_52w - low_52w)) * 100 if high_52w != low_52w else 50

                avg_vol = float(volume.rolling(20).mean().iloc[-1])
                vol_ratio = float(volume.iloc[-1] / avg_vol) if avg_vol > 0 else 1
            else:
                # Minimal data fallback
                change_1d = 0
                change_5d = 0
                change_20d = 0
                rsi = 50
                macd_val = 0
                macd_signal_line = 0
                ma20 = current_price
                ma50 = current_price
                ma200 = None
                high_52w = current_price
                low_52w = current_price
                range_pos = 50
                vol_ratio = 1
        except Exception as e:
            logger.warning(f"[REALTIME ML] Yahoo data fetch failed for {ticker}: {e}")
            change_1d = 0
            change_5d = 0
            change_20d = 0
            rsi = 50
            macd_val = 0
            macd_signal_line = 0
            ma20 = current_price
            ma50 = current_price
            ma200 = None
            high_52w = current_price
            low_52w = current_price
            range_pos = 50
            vol_ratio = 1

        # Use ML volatility (annualized percentage)
        volatility_pct = ml_volatility * 100 * np.sqrt(252) if ml_volatility else 20
        daily_vol_pct = volatility_pct / np.sqrt(252)

        # Calculate targets using ML expected return + volatility
        if signal_type == 'BUY':
            # Use ML expected return for target, or volatility-based if not available
            if ml_expected_return and abs(ml_expected_return) > 0.001:
                target_5d = current_price * (1 + abs(ml_expected_return))
                target_10d = current_price * (1 + abs(ml_expected_return) * 1.5)
            else:
                target_5d = current_price * (1 + daily_vol_pct * 5 / 100)
                target_10d = current_price * (1 + daily_vol_pct * 10 / 100)

            if not stop_loss:
                stop_loss = current_price * (1 - daily_vol_pct * 3 / 100)
        else:
            if ml_expected_return and abs(ml_expected_return) > 0.001:
                target_5d = current_price * (1 - abs(ml_expected_return))
                target_10d = current_price * (1 - abs(ml_expected_return) * 1.5)
            else:
                target_5d = current_price * (1 - daily_vol_pct * 5 / 100)
                target_10d = current_price * (1 - daily_vol_pct * 10 / 100)

            if not stop_loss:
                stop_loss = current_price * (1 + daily_vol_pct * 3 / 100)

        # Use take_profit from ML if available
        if take_profit:
            target_10d = take_profit

        # Technical validation based on ML + traditional indicators
        tech_signals = []
        tech_warnings = []

        # ML Model Signal (most important)
        if ml_direction > 0 and ml_confidence > 0.6:
            tech_signals.append(f"ML BULLISH ({ml_confidence*100:.0f}%)")
        elif ml_direction < 0 and ml_confidence > 0.6:
            tech_warnings.append(f"ML BEARISH ({ml_confidence*100:.0f}%)")
        elif ml_confidence > 0.5:
            tech_signals.append(f"ML {trading_signal.get('action', 'HOLD')} ({ml_confidence*100:.0f}%)")

        # Traditional indicators
        if rsi < 30:
            tech_signals.append("OVERSOLD (RSI<30)")
        elif rsi > 70:
            tech_warnings.append("OVERBOUGHT (RSI>70)")

        if current_price > ma50:
            tech_signals.append("Above 50MA")
        else:
            tech_warnings.append("Below 50MA")

        if ma200 and current_price > ma200:
            tech_signals.append("Above 200MA")

        if macd_val > macd_signal_line:
            tech_signals.append("MACD bullish")
        else:
            tech_warnings.append("MACD bearish")

        if range_pos > 90:
            tech_warnings.append("Near 52W high")
        elif range_pos < 20:
            tech_signals.append("Near 52W low (value)")

        # Determine status based on ML + technicals
        ml_agrees_with_signal = (ml_direction > 0 and signal_type == 'BUY') or (ml_direction < 0 and signal_type == 'SELL')

        if ml_agrees_with_signal and ml_confidence > 0.65 and len(tech_signals) >= 2:
            status = f"STRONG {signal_type}"
        elif ml_agrees_with_signal and ml_confidence > 0.55:
            status = signal_type
        elif not ml_agrees_with_signal and ml_confidence > 0.6:
            status = "ML DISAGREES"
        else:
            status = "CAUTION"

        # Use ML confidence as primary, combine with original confidence
        combined_confidence = (ml_confidence * 0.7 + orig_confidence / 100 * 0.3) * 100

        # RSI Risk Adapter Enhancement (from fixing5.pdf)
        rsi_risk_info = None
        rsi_enhanced_stop_loss = stop_loss
        rsi_enhanced_take_profit = target_10d
        rsi_position_multiplier = 1.0
        rsi_risk_level = "unknown"

        if RSI_ADAPTER_AVAILABLE:
            try:
                rsi_adapter = create_rsi_adapter()
                rsi_result = rsi_adapter.enhance_signal_with_rsi(
                    ticker=ticker,
                    signal_type=signal_type,
                    confidence=combined_confidence / 100,  # Convert to 0-1
                    rsi=rsi,
                    volatility=ml_volatility,
                    position_multiplier=1.0,
                    stop_loss_pct=abs((stop_loss - current_price) / current_price) if stop_loss else 0.08,
                    market_trend="unknown",
                    strict_blocking=False,
                )

                # Update with RSI-adjusted values
                combined_confidence = rsi_result.adjusted_confidence * 100
                rsi_position_multiplier = rsi_result.position_multiplier
                rsi_risk_level = rsi_result.rsi_risk_level

                # Calculate RSI-enhanced stop-loss and take-profit
                if signal_type == 'BUY':
                    rsi_enhanced_stop_loss = current_price * (1 - rsi_result.stop_loss_pct)
                    rsi_enhanced_take_profit = current_price * (1 + rsi_result.take_profit_pct)
                else:
                    rsi_enhanced_stop_loss = current_price * (1 + rsi_result.stop_loss_pct)
                    rsi_enhanced_take_profit = current_price * (1 - rsi_result.take_profit_pct)

                # Build RSI risk info for response
                rsi_risk_info = {
                    'rsi_risk_level': rsi_risk_level,
                    'confidence_adjustment': f"{rsi_result.rsi_confidence_adj*100:+.1f}%",
                    'position_multiplier': round(rsi_position_multiplier, 2),
                    'stop_loss_pct': f"{rsi_result.stop_loss_pct*100:.1f}%",
                    'take_profit_pct': f"{rsi_result.take_profit_pct*100:.1f}%",
                    'should_trade': rsi_result.should_trade,
                    'block_reason': rsi_result.block_reason,
                }

                # Add warning if RSI is extreme
                if rsi_result.rsi_risk_level in ['extreme_oversold', 'extreme_overbought']:
                    if rsi_result.block_reason:
                        tech_warnings.append(f"RSI RISK: {rsi_result.block_reason}")
                    else:
                        tech_warnings.append(f"RSI EXTREME: {rsi_result.rsi_risk_level}")

                logger.debug(f"[RSI ADAPTER] {ticker}: RSI={rsi:.1f}, risk={rsi_risk_level}, mult={rsi_position_multiplier:.2f}")

            except Exception as e:
                logger.warning(f"[RSI ADAPTER] Error for {ticker}: {e}")

        # Expected value calculation using ML confidence
        win_prob = combined_confidence / 100
        potential_gain = abs((target_5d - current_price) / current_price) * 100
        potential_loss = abs((stop_loss - current_price) / current_price) * 100 if stop_loss else potential_gain * 0.5
        expected_value = (win_prob * potential_gain) - ((1 - win_prob) * potential_loss)

        # Kelly fraction using ML confidence
        avg_win_ratio = potential_gain / potential_loss if potential_loss > 0 else 2.0
        kelly_fraction = max(0, min(0.25, win_prob - ((1 - win_prob) / avg_win_ratio)))

        # Get sentiment data if available
        sentiment_data = ml_result.get('sentiment', {})

        # ===== NEW: FIXING7/8 Profit-Maximizing Analysis =====
        profit_max_info = None
        if US_PROFIT_REGIME_AVAILABLE and US_PROFIT_REGIME_CLASSIFIER is not None:
            try:
                # Build stock data for profit scoring
                stock_metrics = {
                    'ticker': ticker,
                    '5d_return': change_5d,
                    'volume_ratio': vol_ratio,
                    'rsi': rsi,
                    'volatility': volatility_pct,
                }

                # Get SPY data (use cached if available)
                spy_data = yf.download('SPY', period='6mo', progress=False)
                if isinstance(spy_data.columns, pd.MultiIndex):
                    spy_data.columns = spy_data.columns.get_level_values(0)

                # Get current VIX
                vix_data = yf.download('^VIX', period='1d', progress=False)
                if isinstance(vix_data.columns, pd.MultiIndex):
                    vix_data.columns = vix_data.columns.get_level_values(0)
                current_vix = float(vix_data['Close'].iloc[-1]) if len(vix_data) > 0 else 20.0

                # Get profit-maximizing regime classification
                profit_regime = US_PROFIT_REGIME_CLASSIFIER.classify_regime_with_profit(
                    spy_data, stock_metrics, current_vix
                )

                profit_max_info = {
                    'profit_score': round(profit_regime.profit_score, 1),
                    'dynamic_multiplier': round(profit_regime.dynamic_multiplier, 3),
                    'buy_multiplier': round(profit_regime.base_output.buy_multiplier, 3),
                    'sell_multiplier': round(profit_regime.base_output.sell_multiplier, 3),
                    'recommended_position': round(profit_regime.recommended_position_size * 100, 2),
                    'regime': profit_regime.regime,
                    'transition_phase': profit_regime.transition_phase.value,
                    'profit_targets': {
                        'target_1': round(profit_regime.profit_target_1 * 100, 1),
                        'target_2': round(profit_regime.profit_target_2 * 100, 1),
                        'target_3': round(profit_regime.profit_target_3 * 100, 1),
                    },
                    'stop_loss_pct': round(profit_regime.stop_loss * 100, 1),
                    'trailing_stop_pct': round(profit_regime.trailing_stop * 100, 1),
                    'vix_level': round(current_vix, 1),
                }
                logger.debug(f"[FIXING7/8] {ticker}: Score={profit_regime.profit_score:.1f}, Mult={profit_regime.dynamic_multiplier:.3f}x")
            except Exception as e:
                logger.warning(f"[FIXING7/8] Error for {ticker}: {e}")

        return {
            'ticker': ticker,
            'signal_type': signal_type,
            'confidence': round(combined_confidence, 1),
            'current_price': round(current_price, 2),
            'change_1d': round(change_1d, 2),
            'change_5d': round(change_5d, 2),
            'change_20d': round(change_20d, 2),
            'rsi': round(rsi, 1),
            'macd': round(macd_val, 4),
            'macd_signal': round(macd_signal_line, 4),
            'ma20': round(ma20, 2),
            'ma50': round(ma50, 2),
            'ma200': round(ma200, 2) if ma200 else None,
            'high_52w': round(high_52w, 2),
            'low_52w': round(low_52w, 2),
            'range_pos': round(range_pos, 1),
            'volatility': round(volatility_pct, 1),
            'vol_ratio': round(vol_ratio, 2),
            'target_5d': round(target_5d, 2),
            'target_10d': round(target_10d, 2),
            'stop_loss': round(stop_loss, 2) if stop_loss else None,
            'potential_gain_5d': round(((target_5d - current_price) / current_price) * 100, 2),
            'potential_gain_10d': round(((target_10d - current_price) / current_price) * 100, 2),
            'risk_reward': round(abs(target_10d - current_price) / abs(current_price - stop_loss), 2) if stop_loss and abs(current_price - stop_loss) > 0 else 0,
            'technical_status': status,
            'bullish_signals': tech_signals,
            'warnings': tech_warnings,
            'expected_value': round(expected_value, 2),
            'kelly_fraction': round(kelly_fraction, 3),
            'adaptive_multiplier': 1.0,
            # ===== NEW: ML MODEL DATA =====
            'ml_model': {
                'direction': ml_direction,
                'direction_label': 'BULLISH' if ml_direction > 0 else 'BEARISH' if ml_direction < 0 else 'NEUTRAL',
                'confidence': round(ml_confidence * 100, 1),
                'expected_return': round(ml_expected_return * 100, 2) if ml_expected_return else 0,
                'model_type': model_info.get('model_type', 'hybrid_ensemble'),
                'features_count': model_info.get('features_count', 0),
                'asset_class': model_info.get('asset_class', 'stock'),
            },
            'trading_signal': {
                'action': trading_signal.get('action', 'HOLD'),
                'should_trade': trading_signal.get('should_trade', False),
                'reason': trading_signal.get('reason', ''),
            },
            'market_context': {
                'volatility_percentile': market_context.get('volatility_percentile'),
                'regime': market_context.get('regime', 'unknown'),
            },
            'sentiment': sentiment_data if sentiment_data else None,
            # ===== NEW: RSI RISK ADAPTER (fixing5.pdf) =====
            'rsi_risk_assessment': rsi_risk_info,
            'rsi_enhanced_stop_loss': round(rsi_enhanced_stop_loss, 2) if rsi_enhanced_stop_loss else None,
            'rsi_enhanced_take_profit': round(rsi_enhanced_take_profit, 2) if rsi_enhanced_take_profit else None,
            'rsi_position_multiplier': round(rsi_position_multiplier, 2),
            'rsi_risk_level': rsi_risk_level,
            # ===== NEW: FIXING7/8 Profit Maximizing =====
            'profit_maximizing': profit_max_info,
        }
    except Exception as e:
        logger.error(f"[REALTIME ML] Error analyzing {ticker}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def _calculate_rsi_simple(prices, period=14):
    """Calculate RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50


def _calculate_macd_simple(prices):
    """Calculate MACD and signal line."""
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return float(macd.iloc[-1]), float(signal.iloc[-1])


def _simulate_portfolio_realistic(buy_analysis, capital=10000, max_positions=5):
    """Simulate portfolio with realistic constraints."""
    if not buy_analysis:
        return {
            'capital': capital,
            'return_5d': 0,
            'return_10d': 0,
            'positions': [],
            'total_value_5d': capital,
            'total_value_10d': capital
        }

    # Use top positions based on expected value
    sorted_picks = sorted(buy_analysis, key=lambda x: x.get('expected_value', 0), reverse=True)[:max_positions]

    position_size = capital / len(sorted_picks) if sorted_picks else 0
    positions = []
    total_5d_pl = 0
    total_10d_pl = 0

    for pick in sorted_picks:
        price = pick.get('current_price', 0)
        if price <= 0:
            continue

        shares = int(position_size / price)
        if shares == 0:
            continue

        entry_value = shares * price
        target_5d = pick.get('target_5d', price)
        target_10d = pick.get('target_10d', price)

        value_5d = shares * target_5d
        value_10d = shares * target_10d
        pl_5d = value_5d - entry_value
        pl_10d = value_10d - entry_value

        total_5d_pl += pl_5d
        total_10d_pl += pl_10d

        positions.append({
            'ticker': pick.get('ticker'),
            'shares': shares,
            'entry_value': round(entry_value, 2),
            'target_5d_value': round(value_5d, 2),
            'target_10d_value': round(value_10d, 2),
            'pl_5d': round(pl_5d, 2),
            'pl_10d': round(pl_10d, 2)
        })

    return {
        'capital': capital,
        'max_positions': max_positions,
        'actual_positions': len(positions),
        'return_5d': round((total_5d_pl / capital) * 100, 2) if capital > 0 else 0,
        'return_10d': round((total_10d_pl / capital) * 100, 2) if capital > 0 else 0,
        'total_pl_5d': round(total_5d_pl, 2),
        'total_pl_10d': round(total_10d_pl, 2),
        'total_value_5d': round(capital + total_5d_pl, 2),
        'total_value_10d': round(capital + total_10d_pl, 2),
        'positions': positions
    }


def _generate_analysis_summary(buy_analysis, sell_analysis, regime):
    """Generate summary statistics."""
    if not buy_analysis:
        return {'message': 'No analysis data available'}

    avg_5d_gain = sum(b.get('potential_gain_5d', 0) for b in buy_analysis) / len(buy_analysis) if buy_analysis else 0
    avg_10d_gain = sum(b.get('potential_gain_10d', 0) for b in buy_analysis) / len(buy_analysis) if buy_analysis else 0
    strong_setups = sum(1 for b in buy_analysis if 'STRONG' in b.get('technical_status', ''))

    # Top picks by expected value
    top_picks = sorted(buy_analysis, key=lambda x: x.get('expected_value', 0), reverse=True)[:3]

    return {
        'regime': regime,
        'total_buy_signals': len(buy_analysis),
        'total_sell_signals': len(sell_analysis),
        'strong_technical_setups': strong_setups,
        'average_5d_gain': round(avg_5d_gain, 2),
        'average_10d_gain': round(avg_10d_gain, 2),
        'top_picks': [
            {
                'ticker': p.get('ticker'),
                'confidence': p.get('confidence'),
                'expected_value': p.get('expected_value'),
                'status': p.get('technical_status')
            }
            for p in top_picks
        ]
    }


# --------------------------------------------------------------------------
# HELPER FUNCTIONS FOR TREND PREDICTION
# --------------------------------------------------------------------------

def _detect_market_regime_enhanced(regime):
    """
    Enhanced market regime detection using the US/Intl ML model's RegimeDetector.

    This function now uses:
    1. REGIME_DETECTOR (GMM-based) from the ML model
    2. VIX data for volatility analysis
    3. Market breadth indicators
    4. Proper volatility calculation matching the ML pipeline
    """
    try:
        logger.info(f"[REGIME ML] Running ML regime detection for {regime}")

        # Get appropriate index
        index_map = {
            'Stock': '^GSPC',
            'Cryptocurrency': 'BTC-USD',
            'Forex': 'DX-Y.NYB',
            'Commodity': 'GC=F',
            'all': '^GSPC'
        }

        ticker = index_map.get(regime, '^GSPC')
        # Use rate-limited Yahoo fetcher to prevent 401 errors
        hist = get_yahoo_data_safe(ticker, period='1y', max_retries=3)

        if hist is None or len(hist) < 60:
            logger.warning(f"[REGIME ML] Insufficient data for {ticker}, returning NEUTRAL")
            return 'NEUTRAL'

        close = hist['Close']
        returns = close.pct_change().dropna()

        # ===== USE THE ML MODEL'S REGIME DETECTOR =====
        # Calculate volatility array for GMM regime detection
        volatility_array = returns.rolling(20).std() * np.sqrt(252)
        volatility_array = volatility_array.dropna().values

        if len(volatility_array) >= 30:
            try:
                # Use the global REGIME_DETECTOR (GMM-based)
                regimes_arr, regime_info = REGIME_DETECTOR.detect_regime(volatility_array)

                # Get the current regime (last value)
                ml_regime_id = regimes_arr[-1]  # 0=Low, 1=Medium, 2=High volatility

                logger.info(f"[REGIME ML] GMM detected volatility regime: {ml_regime_id} "
                           f"(0=Low, 1=Medium, 2=High)")
            except Exception as e:
                logger.warning(f"[REGIME ML] GMM detection failed: {e}, using fallback")
                ml_regime_id = 1  # Default to medium
        else:
            ml_regime_id = 1  # Default to medium if not enough data

        # Get current metrics
        current_volatility = float(volatility_array[-1] * 100) if len(volatility_array) > 0 else 20
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else sma20
        current_price = float(close.iloc[-1])
        avg_return = float(returns.tail(20).mean())  # 20-day average return

        # Get VIX data (US/Intl model uses this) - rate-limited
        try:
            vix_data = get_yahoo_data_safe('^VIX', period='5d', max_retries=2)
            vix = float(vix_data['Close'].iloc[-1]) if vix_data is not None and len(vix_data) > 0 else 20
        except:
            vix = 20

        # Price position in 52-week range
        price_position = (current_price - float(close.min())) / (float(close.max()) - float(close.min())) if float(close.max()) != float(close.min()) else 0.5

        # ===== COMBINE ML REGIME WITH MARKET CONDITIONS =====
        # High volatility regime from ML model
        if ml_regime_id == 2:  # High volatility
            if vix > 30:
                detected_regime = 'CRISIS'
            else:
                detected_regime = 'VOLATILE'
        # Low volatility regime from ML model
        elif ml_regime_id == 0:  # Low volatility
            if current_price > sma20 > sma50 and avg_return > 0.0005:
                detected_regime = 'BULL'
            elif current_price < sma20 < sma50 and avg_return < -0.0005:
                detected_regime = 'BEAR'
            elif price_position > 0.7:
                detected_regime = 'RISK_ON'
            elif price_position < 0.3:
                detected_regime = 'RISK_OFF'
            else:
                detected_regime = 'SIDEWAYS'
        # Medium volatility regime from ML model
        else:  # ml_regime_id == 1
            if current_price > sma20 > sma50 and avg_return > 0.001:
                detected_regime = 'BULL'
            elif current_price < sma20 < sma50 and avg_return < -0.001:
                detected_regime = 'BEAR'
            elif vix > 25:
                detected_regime = 'RISK_OFF'
            elif price_position > 0.7:
                detected_regime = 'RISK_ON'
            elif price_position < 0.3:
                detected_regime = 'RISK_OFF'
            elif abs(avg_return) < 0.0003:
                detected_regime = 'SIDEWAYS'
            else:
                detected_regime = 'NEUTRAL'

        logger.info(f"[REGIME ML] Final regime: {detected_regime} "
                   f"(ML regime: {ml_regime_id}, VIX: {vix:.1f}, Vol: {current_volatility:.1f}%)")

        return detected_regime

    except Exception as e:
        logger.error(f"[REGIME ML] Error detecting regime: {e}")
        return 'NEUTRAL'


def _analyze_trend_strength(buys, sells, regime):
    """
    Analyze trend strength from ML model signals.

    The buys/sells data comes from generate_prediction() which uses:
    - HybridEnsemblePredictor (LightGBM + XGBoost + LSTM)
    - US/Intl macro features (VIX, SPY, DXY, GLD)
    - Sentiment analysis (FinBERT + VADER)
    - Volatility regime detection

    This function aggregates ML model confidence scores to determine market trend.
    """
    total_signals = len(buys) + len(sells)
    buy_ratio = len(buys) / total_signals if total_signals > 0 else 0.5

    # Extract ML model confidence (from generate_prediction)
    avg_buy_conf = sum(b.get('confidence', 50) for b in buys) / len(buys) if buys else 50
    avg_sell_conf = sum(s.get('confidence', 50) for s in sells) / len(sells) if sells else 50

    # Extract ML model expected returns if available
    avg_buy_return = sum(b.get('expected_return', 0) for b in buys) / len(buys) if buys else 0
    avg_sell_return = sum(abs(s.get('expected_return', 0)) for s in sells) / len(sells) if sells else 0

    # Count ML model direction signals
    ml_bullish_count = sum(1 for b in buys if b.get('ml_direction', 1) > 0)
    ml_bearish_count = sum(1 for s in sells if s.get('ml_direction', -1) < 0)

    # Get market indicators
    market_data = _get_market_indicators(regime)

    # Composite Bull Score (0-100) - Enhanced with ML model metrics
    bull_score = 0
    factors = []

    # Factor 1: ML Model Buy/Sell Ratio (0-20 points)
    ratio_score = buy_ratio * 20
    bull_score += ratio_score
    factors.append({
        'name': 'ML Buy/Sell Ratio',
        'value': f"{buy_ratio:.1%}",
        'score': round(ratio_score, 1),
        'interpretation': 'Bullish' if buy_ratio > 0.6 else 'Bearish' if buy_ratio < 0.4 else 'Neutral',
        'source': 'HybridEnsemblePredictor'
    })

    # Factor 2: ML Confidence Differential (0-20 points)
    conf_diff = avg_buy_conf - avg_sell_conf
    conf_score = min(20, max(0, (conf_diff + 50) / 5))
    bull_score += conf_score
    factors.append({
        'name': 'ML Confidence Gap',
        'value': f"{conf_diff:+.1f}%",
        'score': round(conf_score, 1),
        'interpretation': 'Bullish' if conf_diff > 10 else 'Bearish' if conf_diff < -10 else 'Neutral',
        'source': 'HybridEnsemblePredictor'
    })

    # Factor 3: ML Expected Return (0-20 points) - NEW
    return_score = 0
    if avg_buy_return > 0:
        return_score = min(20, avg_buy_return * 100 * 10)  # Scale expected return
    elif avg_sell_return > 0:
        return_score = max(0, 10 - avg_sell_return * 100 * 5)
    else:
        return_score = 10  # Neutral
    bull_score += return_score
    factors.append({
        'name': 'ML Expected Return',
        'value': f"{avg_buy_return*100:+.2f}%",
        'score': round(return_score, 1),
        'interpretation': 'Bullish' if avg_buy_return > 0.01 else 'Bearish' if avg_buy_return < -0.01 else 'Neutral',
        'source': 'HybridEnsemblePredictor'
    })

    # Factor 4: Market RSI (0-20 points)
    market_rsi = market_data.get('rsi', 50)
    if 30 < market_rsi < 70:
        rsi_score = 20
    else:
        rsi_score = max(0, 20 - abs(market_rsi - 50) / 2.5)
    bull_score += rsi_score
    factors.append({
        'name': 'Market RSI',
        'value': f"{market_rsi:.1f}",
        'score': round(rsi_score, 1),
        'interpretation': 'Overbought' if market_rsi > 70 else 'Oversold' if market_rsi < 30 else 'Healthy'
    })

    # Factor 5: Volatility/VIX (0-20 points)
    vix = market_data.get('vix', 20)
    vol_score = max(0, 20 - min(20, (vix - 12) * 1.2))
    bull_score += vol_score
    factors.append({
        'name': 'VIX (Fear Index)',
        'value': f"{vix:.1f}",
        'score': round(vol_score, 1),
        'interpretation': 'High Fear' if vix > 30 else 'Low Fear' if vix < 15 else 'Normal',
        'source': 'US/Intl Macro Features'
    })

    # Determine trend
    if bull_score >= 70:
        trend = 'STRONG_BULL'
        trend_label = 'Strong Bull Market'
        recommendation = 'Aggressive buying recommended. High confidence in upward momentum.'
    elif bull_score >= 55:
        trend = 'BULL'
        trend_label = 'Bull Market'
        recommendation = 'Favor long positions. Market conditions supportive of gains.'
    elif bull_score >= 45:
        trend = 'NEUTRAL'
        trend_label = 'Neutral/Sideways'
        recommendation = 'Mixed signals. Use caution and smaller position sizes.'
    elif bull_score >= 30:
        trend = 'BEAR'
        trend_label = 'Bear Market'
        recommendation = 'Consider defensive positions. Risk of further downside.'
    else:
        trend = 'STRONG_BEAR'
        trend_label = 'Strong Bear Market'
        recommendation = 'High risk environment. Consider hedging or cash positions.'

    return {
        'trend': trend,
        'trend_label': trend_label,
        'bull_score': round(bull_score, 1),
        'recommendation': recommendation,
        'factors': factors,
        'signal_counts': {
            'buy_signals': len(buys),
            'sell_signals': len(sells),
            'buy_ratio': round(buy_ratio, 2)
        },
        'market_data': market_data,
        # ML Model Summary
        'ml_model_info': {
            'model': 'HybridEnsemblePredictor',
            'components': ['LightGBM', 'XGBoost', 'LSTM'],
            'features': ['US Macro (VIX, SPY, DXY, GLD)', 'Sentiment (FinBERT + VADER)', 'Technical Indicators'],
            'avg_buy_confidence': round(avg_buy_conf, 1),
            'avg_sell_confidence': round(avg_sell_conf, 1),
            'avg_expected_return': round(avg_buy_return * 100, 2),
            'ml_bullish_signals': ml_bullish_count,
            'ml_bearish_signals': ml_bearish_count
        }
    }


def _get_market_indicators(regime):
    """Get market-wide indicators."""
    try:
        index_map = {
            'Stock': '^GSPC',
            'Cryptocurrency': 'BTC-USD',
            'Forex': 'DX-Y.NYB',
            'Commodity': 'GC=F',
            'all': '^GSPC'
        }

        index_ticker = index_map.get(regime, '^GSPC')
        stock = yf.Ticker(index_ticker)
        hist = stock.history(period='3mo')

        if len(hist) < 20:
            return {'rsi': 50, 'vix': 20}

        close = hist['Close']
        rsi = _calculate_rsi_simple(close)

        # Get VIX
        try:
            vix_data = yf.Ticker('^VIX').history(period='1d')
            vix = float(vix_data['Close'].iloc[-1]) if len(vix_data) > 0 else 20
        except:
            vix = 20

        return {
            'rsi': round(rsi, 1),
            'vix': round(vix, 1),
            'index': index_ticker,
            'index_price': round(float(close.iloc[-1]), 2),
            'index_change_5d': round(((float(close.iloc[-1]) - float(close.iloc[-5])) / float(close.iloc[-5])) * 100, 2) if len(close) >= 5 else 0
        }
    except Exception as e:
        logger.error(f"[MARKET] Error getting indicators: {e}")
        return {'rsi': 50, 'vix': 20}


def _generate_trade_signals_enhanced(regime, market_regime, trend_analysis):
    """Generate specific, actionable trade signals based on regime."""
    regime_strategies = {
        'BULL': {
            'primary_action': 'BUY momentum stocks',
            'secondary_action': 'SELL puts on dips',
            'position_size': 'Full (100%)',
            'stop_loss': 'Wide (10-15%)',
            'target_gain': '20-30%'
        },
        'BEAR': {
            'primary_action': 'SHORT weak stocks',
            'secondary_action': 'BUY protective puts',
            'position_size': 'Reduced (50%)',
            'stop_loss': 'Tight (5-8%)',
            'target_gain': '15-25%'
        },
        'RISK_ON': {
            'primary_action': 'BUY growth/high-beta',
            'secondary_action': 'Avoid safe havens',
            'position_size': 'Aggressive (120%)',
            'stop_loss': 'Moderate (8-10%)',
            'target_gain': '25-40%'
        },
        'RISK_OFF': {
            'primary_action': 'BUY defensive stocks',
            'secondary_action': 'SELL risk assets',
            'position_size': 'Defensive (70%)',
            'stop_loss': 'Very tight (3-5%)',
            'target_gain': '10-15%'
        },
        'VOLATILE': {
            'primary_action': 'Options strategies',
            'secondary_action': 'Reduce overall exposure',
            'position_size': 'Small (30%)',
            'stop_loss': 'Very tight (2-3%)',
            'target_gain': 'Quick 5-10% scalps'
        },
        'SIDEWAYS': {
            'primary_action': 'Range trading',
            'secondary_action': 'Sell premium (options)',
            'position_size': 'Moderate (60%)',
            'stop_loss': 'At range boundaries',
            'target_gain': '5-10%'
        }
    }

    strategy = regime_strategies.get(market_regime, {
        'primary_action': 'Wait for clearer signal',
        'secondary_action': 'Monitor key levels',
        'position_size': 'Minimal (20%)',
        'stop_loss': 'N/A',
        'target_gain': 'N/A'
    })

    risk_levels = {
        'BULL': 'LOW',
        'STRONG_BULL': 'LOW',
        'RISK_ON': 'MODERATE',
        'SIDEWAYS': 'MODERATE',
        'NEUTRAL': 'MODERATE',
        'RISK_OFF': 'HIGH',
        'BEAR': 'HIGH',
        'STRONG_BEAR': 'VERY HIGH',
        'VOLATILE': 'VERY HIGH',
        'CRISIS': 'EXTREME'
    }

    return {
        'market_regime': market_regime,
        'strategy': strategy,
        'confidence': trend_analysis.get('bull_score', 50),
        'recommended_actions': [
            f"1. {strategy['primary_action']}",
            f"2. {strategy['secondary_action']}",
            f"3. Position size: {strategy['position_size']}",
            f"4. Target gain: {strategy['target_gain']}",
            f"5. Stop loss: {strategy['stop_loss']}"
        ],
        'risk_level': risk_levels.get(market_regime, 'MODERATE')
    }


def _calculate_optimal_allocation(trade_signals, market_regime, bull_score):
    """Calculate optimal portfolio allocation based on regime."""
    # Base allocations by regime
    regime_allocations = {
        'BULL': {'stocks': 80, 'bonds': 10, 'cash': 5, 'alternatives': 5},
        'STRONG_BULL': {'stocks': 90, 'bonds': 5, 'cash': 0, 'alternatives': 5},
        'RISK_ON': {'stocks': 85, 'bonds': 5, 'cash': 0, 'alternatives': 10},
        'SIDEWAYS': {'stocks': 50, 'bonds': 30, 'cash': 15, 'alternatives': 5},
        'NEUTRAL': {'stocks': 60, 'bonds': 25, 'cash': 10, 'alternatives': 5},
        'RISK_OFF': {'stocks': 30, 'bonds': 40, 'cash': 25, 'alternatives': 5},
        'BEAR': {'stocks': 20, 'bonds': 35, 'cash': 40, 'alternatives': 5},
        'STRONG_BEAR': {'stocks': 10, 'bonds': 30, 'cash': 50, 'alternatives': 10},
        'VOLATILE': {'stocks': 25, 'bonds': 25, 'cash': 40, 'alternatives': 10},
        'CRISIS': {'stocks': 5, 'bonds': 20, 'cash': 60, 'alternatives': 15}
    }

    allocation = regime_allocations.get(market_regime, regime_allocations['NEUTRAL'])

    return {
        'recommended_allocation': allocation,
        'market_regime': market_regime,
        'bull_score': bull_score,
        'rebalance_urgency': 'HIGH' if market_regime in ['CRISIS', 'VOLATILE', 'STRONG_BEAR'] else 'MODERATE' if market_regime in ['BEAR', 'RISK_OFF'] else 'LOW'
    }


def _get_profit_maximizing_actions(market_regime, trend_analysis, trade_signals):
    """Get specific profit-maximizing actions."""
    actions = []
    bull_score = trend_analysis.get('bull_score', 50)

    if market_regime == 'BULL' and bull_score > 70:
        actions = [
            "MAX PROFIT: Buy 3x leveraged ETFs (UPRO, TQQQ)",
            "Add to winning positions (pyramid up)",
            "Hold minimum 5-10 days for full trend capture",
            "Take partial profits at +20%, let rest run"
        ]
    elif market_regime == 'BEAR' and bull_score < 30:
        actions = [
            "MAX PROFIT: Buy inverse ETFs (SQQQ, SPXU)",
            "Short weakest sectors (technology, growth)",
            "Quick entries/exits (1-3 day holds)",
            "Tight stops (3-5%), take profits at 10-15%"
        ]
    elif market_regime == 'VOLATILE':
        actions = [
            "MAX PROFIT: Straddle/strangle options",
            "Trade VIX derivatives (VXX, UVXY)",
            "Scalp quick moves (minutes-hours)",
            "Small size, high frequency"
        ]
    elif market_regime == 'RISK_ON':
        actions = [
            "MAX PROFIT: High-beta tech stocks",
            "Momentum breakout plays",
            "Ride trends with trailing stops",
            "Let winners run, cut losers fast"
        ]
    else:
        actions = [
            "Wait for clearer trend confirmation",
            "Monitor key support/resistance levels",
            "Small test positions only",
            "Focus on highest conviction setups"
        ]

    return actions


# ============================================================================
# SCREENER STATUS ENDPOINT
# ============================================================================

@app.route('/api/screener-status')
def screener_status():
    """
    Check Yahoo Finance screener availability and performance.

    Returns:
        JSON with screener status, reliability metrics, and cache performance
    """
    status = {
        'screener_available': SCREENER_DISCOVERY_AVAILABLE,
        'timestamp': datetime.now().isoformat(),
    }

    if SCREENER_DISCOVERY_AVAILABLE:
        try:
            reliability_mgr = get_reliability_manager()

            # Get reliability metrics for each regime
            regimes = ['Stock', 'Cryptocurrency', 'China', 'Commodity', 'Forex', 'all']
            reliability_metrics = {}
            for regime in regimes:
                recommendation = reliability_mgr.get_recommendation(regime)
                reliability_metrics[regime] = {
                    'use_screeners': recommendation['use_screeners'],
                    'success_rate': round(reliability_mgr.get_success_rate(regime), 2),
                    'reason': recommendation['reason']
                }

            status['reliability_metrics'] = reliability_metrics
            status['cache_ttl_settings'] = TOP_PICKS_CACHE_TTL
            status['supported_screeners'] = [
                'day_gainers', 'day_losers', 'most_actives',
                'undervalued_growth_stocks', 'growth_technology_stocks',
                'hk_active', 'china_active', 'crypto'
            ]

            # Test screener connectivity
            try:
                discoverer = get_screener_discoverer()
                test_result = discoverer.discover_tickers('us_active', 3)
                status['connectivity_test'] = {
                    'status': 'ok' if test_result else 'failed',
                    'sample_tickers': test_result[:3] if test_result else []
                }
            except Exception as e:
                status['connectivity_test'] = {
                    'status': 'error',
                    'error': str(e)
                }

        except Exception as e:
            status['error'] = str(e)
    else:
        status['message'] = 'Screener discovery module not loaded - using database fallback'

    return jsonify(status)


# ============================================================================
# ENHANCED DATA LAYER & TICKER RESOLUTION ENDPOINTS
# ============================================================================

@app.route('/api/ticker/resolve')
def resolve_ticker():
    """
    Resolve company name to standardized ticker symbol.

    Query params:
        q: Company name, partial name, or ticker to resolve

    Examples:
        /api/ticker/resolve?q=Tencent -> 0700.HK
        /api/ticker/resolve?q=阿里巴巴 -> 9988.HK
        /api/ticker/resolve?q=600519 -> 600519.SS
    """
    if not ENHANCED_DATA_LAYER:
        return jsonify({
            'status': 'error',
            'error': 'Enhanced data layer not available'
        }), 503

    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({
            'status': 'error',
            'error': 'Query parameter "q" is required'
        }), 400

    try:
        ticker = resolve_china_stock(query)

        if ticker:
            info = get_stock_info(ticker)
            return jsonify({
                'status': 'success',
                'query': query,
                'ticker': ticker,
                'info': info
            })
        else:
            # Try to validate as-is (might be a US ticker)
            is_valid, msg = validate_ticker(query.upper())
            if is_valid:
                return jsonify({
                    'status': 'success',
                    'query': query,
                    'ticker': query.upper(),
                    'info': {'name': msg, 'type': 'non-china'}
                })

            return jsonify({
                'status': 'not_found',
                'query': query,
                'message': 'Could not resolve ticker. Try a different name or symbol.'
            }), 404

    except Exception as e:
        logger.error(f"[RESOLVER] Error resolving {query}: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/ticker/search')
def search_china_tickers():
    """
    Search for China stocks with fuzzy matching.

    Query params:
        q: Search query
        limit: Maximum results (default 10)

    Returns list of matching stocks with details.
    """
    if not ENHANCED_DATA_LAYER:
        return jsonify({
            'status': 'error',
            'error': 'Enhanced data layer not available'
        }), 503

    query = request.args.get('q', '').strip()
    limit = int(request.args.get('limit', 10))

    if not query:
        return jsonify({
            'status': 'error',
            'error': 'Query parameter "q" is required'
        }), 400

    try:
        results = search_china_stocks(query, limit=limit)

        return jsonify({
            'status': 'success',
            'query': query,
            'results': results,
            'count': len(results)
        })

    except Exception as e:
        logger.error(f"[SEARCH] Error searching {query}: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/ticker/autocomplete')
def ticker_autocomplete():
    """
    Get auto-complete suggestions for ticker search.

    Query params:
        q: Partial search query
        limit: Maximum suggestions (default 5)

    Returns list of suggestions for dropdown/autocomplete UI.
    """
    if not ENHANCED_DATA_LAYER:
        return jsonify({
            'status': 'error',
            'error': 'Enhanced data layer not available'
        }), 503

    query = request.args.get('q', '').strip()
    limit = int(request.args.get('limit', 5))

    if not query:
        return jsonify({
            'status': 'success',
            'suggestions': []
        })

    try:
        suggestions = autocomplete(query, limit=limit)

        return jsonify({
            'status': 'success',
            'query': query,
            'suggestions': suggestions
        })

    except Exception as e:
        logger.error(f"[AUTOCOMPLETE] Error for {query}: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/ticker/validate')
def validate_ticker_endpoint():
    """
    Validate that a ticker exists and has data.

    Query params:
        ticker: Ticker symbol to validate
    """
    if not ENHANCED_DATA_LAYER:
        return jsonify({
            'status': 'error',
            'error': 'Enhanced data layer not available'
        }), 503

    ticker = request.args.get('ticker', '').strip().upper()
    if not ticker:
        return jsonify({
            'status': 'error',
            'error': 'Query parameter "ticker" is required'
        }), 400

    try:
        is_valid, msg = validate_ticker(ticker)

        return jsonify({
            'status': 'success',
            'ticker': ticker,
            'is_valid': is_valid,
            'message': msg
        })

    except Exception as e:
        logger.error(f"[VALIDATE] Error validating {ticker}: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/data/price')
def get_price_endpoint():
    """
    Get current price for a ticker with robust fetching.

    Query params:
        ticker: Stock ticker symbol
    """
    if not ENHANCED_DATA_LAYER:
        # Fallback to basic yfinance
        ticker = request.args.get('ticker', '').strip()
        if not ticker:
            return jsonify({'status': 'error', 'error': 'ticker required'}), 400

        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period='1d')
            if len(data) > 0:
                price = float(data['Close'].iloc[-1])
                return jsonify({'status': 'success', 'ticker': ticker, 'price': price})
            return jsonify({'status': 'error', 'error': 'No data'}), 404
        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)}), 500

    ticker = request.args.get('ticker', '').strip()
    if not ticker:
        return jsonify({'status': 'error', 'error': 'ticker required'}), 400

    # Try to resolve if it's a China company name
    resolved = resolve_china_stock(ticker)
    if resolved:
        ticker = resolved

    try:
        price = get_current_price(ticker.upper())

        if price is not None:
            return jsonify({
                'status': 'success',
                'ticker': ticker.upper(),
                'price': price,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'ticker': ticker.upper(),
                'error': 'Could not fetch price'
            }), 404

    except Exception as e:
        logger.error(f"[PRICE] Error fetching price for {ticker}: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/data/history')
def get_history_endpoint():
    """
    Get historical data with robust fetching and caching.

    Query params:
        ticker: Stock ticker symbol
        period: Data period (default "60d")
        interval: Data interval (default "1d")
    """
    ticker = request.args.get('ticker', '').strip()
    period = request.args.get('period', '60d')
    interval = request.args.get('interval', '1d')

    if not ticker:
        return jsonify({'status': 'error', 'error': 'ticker required'}), 400

    # Try to resolve if it's a China company name
    if ENHANCED_DATA_LAYER:
        resolved = resolve_china_stock(ticker)
        if resolved:
            ticker = resolved

    try:
        if ENHANCED_DATA_LAYER:
            df = get_realtime_yahoo_data(ticker.upper(), period=period)
        else:
            ticker_obj = yf.Ticker(ticker.upper())
            df = ticker_obj.history(period=period, interval=interval)

        if df is not None and not df.empty:
            # Convert to JSON-friendly format
            data = []
            for idx, row in df.iterrows():
                data.append({
                    'date': idx.strftime('%Y-%m-%d'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row.get('Volume', 0)) if 'Volume' in row else 0
                })

            return jsonify({
                'status': 'success',
                'ticker': ticker.upper(),
                'period': period,
                'interval': interval,
                'data': data,
                'count': len(data)
            })
        else:
            return jsonify({
                'status': 'error',
                'ticker': ticker.upper(),
                'error': 'No data available'
            }), 404

    except Exception as e:
        logger.error(f"[HISTORY] Error fetching history for {ticker}: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/data/cache-stats')
def data_cache_stats():
    """Get cache statistics for the enhanced data layer."""
    if not ENHANCED_DATA_LAYER:
        return jsonify({
            'status': 'error',
            'error': 'Enhanced data layer not available'
        }), 503

    try:
        stats = get_cache_stats()
        return jsonify({
            'status': 'success',
            'cache_stats': stats
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/data/clear-cache', methods=['POST'])
def clear_data_cache():
    """Clear the data cache."""
    if not ENHANCED_DATA_LAYER:
        return jsonify({
            'status': 'error',
            'error': 'Enhanced data layer not available'
        }), 503

    try:
        clear_cache()
        return jsonify({
            'status': 'success',
            'message': 'Cache cleared successfully'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/china/database')
def china_database():
    """
    Get the China stocks database.

    Query params:
        sector: Filter by sector (optional)
    """
    if not ENHANCED_DATA_LAYER:
        return jsonify({
            'status': 'error',
            'error': 'Enhanced data layer not available'
        }), 503

    sector = request.args.get('sector')

    try:
        if sector:
            tickers = get_all_tickers_by_sector(sector)
        else:
            tickers = list(CHINA_STOCKS_DATABASE.keys())

        stocks = []
        for ticker in tickers:
            info = get_stock_info(ticker)
            if info:
                stocks.append(info)

        sectors = get_sectors()

        return jsonify({
            'status': 'success',
            'stocks': stocks,
            'count': len(stocks),
            'sectors': sectors,
            'filter': {'sector': sector} if sector else None
        })

    except Exception as e:
        logger.error(f"[DATABASE] Error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ============================================================================
# US/INTERNATIONAL MODEL VALIDATION ENDPOINTS
# ============================================================================

# Global validator instance (lazy initialization)
_US_INTL_VALIDATOR = None

def get_us_intl_validator():
    """Get or initialize the US/International model validator."""
    global _US_INTL_VALIDATOR
    if not US_INTL_VALIDATOR_AVAILABLE:
        return None
    if _US_INTL_VALIDATOR is None:
        _US_INTL_VALIDATOR = USIntlModelValidator()
    return _US_INTL_VALIDATOR


@app.route('/api/validation/status')
def validation_status():
    """Check validation framework status."""
    return jsonify({
        'status': 'available' if US_INTL_VALIDATOR_AVAILABLE else 'unavailable',
        'markets': list(MARKET_THRESHOLDS.keys()) if US_INTL_VALIDATOR_AVAILABLE else [],
        'universe_size': sum(len(v) for v in get_full_universe().values()) if US_INTL_VALIDATOR_AVAILABLE else 0
    })


@app.route('/api/validation/thresholds')
def validation_thresholds():
    """Get market-adjusted validation thresholds."""
    if not US_INTL_VALIDATOR_AVAILABLE:
        return jsonify({'status': 'error', 'error': 'Validator not available'}), 503

    return jsonify({
        'status': 'success',
        'thresholds': MARKET_THRESHOLDS
    })


@app.route('/api/validation/universe')
def validation_universe():
    """Get the full validation universe."""
    if not US_INTL_VALIDATOR_AVAILABLE:
        return jsonify({'status': 'error', 'error': 'Validator not available'}), 503

    market = request.args.get('market')
    universe = get_full_universe()

    if market and market in universe:
        return jsonify({
            'status': 'success',
            'market': market,
            'symbols': universe[market],
            'count': len(universe[market])
        })

    return jsonify({
        'status': 'success',
        'universe': universe,
        'total': sum(len(v) for v in universe.values())
    })


@app.route('/api/validation/screen', methods=['POST'])
def validation_screen_symbol():
    """Screen a single symbol through validation tiers."""
    if not US_INTL_VALIDATOR_AVAILABLE:
        return jsonify({'status': 'error', 'error': 'Validator not available'}), 503

    data = request.json
    symbol = data.get('symbol', '').upper()
    tier = data.get('tier', 'quick')

    if not symbol:
        return jsonify({'status': 'error', 'error': 'Symbol required'}), 400

    if tier not in ['quick', 'medium', 'deep']:
        return jsonify({'status': 'error', 'error': 'Tier must be quick, medium, or deep'}), 400

    try:
        validator = get_us_intl_validator()
        result = validator.screen_symbol(symbol, tier=tier)

        if result is None:
            return jsonify({
                'status': 'error',
                'symbol': symbol,
                'error': 'Screening failed - insufficient data'
            }), 404

        return jsonify({
            'status': 'success',
            'result': {
                'symbol': result.symbol,
                'market_type': result.market_type,
                'tier': result.tier,
                'pass_rate': result.pass_rate,
                'avg_return': result.avg_return,
                'return_std': result.return_std,
                'avg_trades': result.avg_trades,
                'passes_tier': result.passes_tier,
                'liquidity_ok': result.liquidity_ok,
                'avg_daily_volume': result.avg_daily_volume
            }
        })

    except Exception as e:
        logger.error(f"[VALIDATION] Screen error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/validation/run', methods=['POST'])
def validation_run_full():
    """
    Run full tiered validation pipeline.

    POST body:
        markets: List of markets to include (optional, defaults to all)
        limit: Max symbols per market (optional, for faster testing)
    """
    if not US_INTL_VALIDATOR_AVAILABLE:
        return jsonify({'status': 'error', 'error': 'Validator not available'}), 503

    data = request.json or {}
    markets = data.get('markets')
    limit = data.get('limit', 10)  # Default to 10 per market for API use

    try:
        validator = USIntlModelValidator()  # Fresh validator
        full_universe = get_full_universe()

        # Filter and limit universe
        if markets:
            universe = {m: full_universe[m][:limit] for m in markets if m in full_universe}
        else:
            universe = {m: syms[:limit] for m, syms in full_universe.items()}

        # Run validation
        summary = validator.run_tiered_screening(universe=universe, verbose=False)

        # Save results
        validator.save_results()

        return jsonify({
            'status': 'success',
            'summary': {
                'total_screened': summary.total_screened,
                'liquid_count': summary.liquid_count,
                'tier1_passed': summary.tier1_passed,
                'tier2_passed': summary.tier2_passed,
                'tier3_robust': summary.tier3_robust,
                'compute_savings': summary.compute_savings,
                'by_market': summary.by_market,
                'robust_performers': summary.robust_performers[:20]  # Top 20
            },
            'timestamp': summary.timestamp
        })

    except Exception as e:
        logger.error(f"[VALIDATION] Run error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/validation/results')
def validation_get_results():
    """Get latest validation results."""
    if not US_INTL_VALIDATOR_AVAILABLE:
        return jsonify({'status': 'error', 'error': 'Validator not available'}), 503

    results_path = os.path.join(
        os.path.dirname(__file__),
        'results', 'us_intl_validation_results.json'
    )

    if not os.path.exists(results_path):
        return jsonify({
            'status': 'not_found',
            'message': 'No validation results found. Run /api/validation/run first.'
        }), 404

    try:
        with open(results_path, 'r') as f:
            results = json.load(f)

        return jsonify({
            'status': 'success',
            'results': results
        })

    except Exception as e:
        logger.error(f"[VALIDATION] Results error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/validation/cross-market')
def validation_cross_market():
    """Run cross-market consistency validation."""
    if not US_INTL_VALIDATOR_AVAILABLE:
        return jsonify({'status': 'error', 'error': 'Validator not available'}), 503

    validator = get_us_intl_validator()

    if not validator.validation_summary:
        return jsonify({
            'status': 'error',
            'error': 'No validation results. Run /api/validation/run first.'
        }), 400

    try:
        cross_validator = CrossMarketValidator(validator)
        consistency = cross_validator.validate_consistency()

        return jsonify({
            'status': 'success',
            'consistency': consistency
        })

    except Exception as e:
        logger.error(f"[VALIDATION] Cross-market error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ============================================================================
# CHINA MODEL PLATFORM ENDPOINTS (Phase 2: Live Trading)
# ============================================================================

@app.route('/api/china/status')
def china_model_status():
    """Check if China Model Platform is available and initialized."""
    factory, constructor, risk_analyzer = get_china_model_factory()

    if factory is None:
        return jsonify({
            'status': 'unavailable',
            'message': 'China Model Platform not initialized',
            'models_loaded': 0
        })

    return jsonify({
        'status': 'available',
        'models_loaded': len(factory.models),
        'symbols': list(factory.models.keys()),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/china/signals')
def china_model_signals():
    """Get current trading signals from China Model Platform."""
    factory, constructor, risk_analyzer = get_china_model_factory()

    if factory is None:
        return jsonify({
            'status': 'error',
            'error': 'China Model Platform not available'
        }), 503

    try:
        signals = []
        for symbol in factory.models:
            pred = factory.predict(symbol)
            if pred:
                model_data = factory.models.get(symbol, {})
                signals.append({
                    'symbol': symbol,
                    'signal': pred['signal'],
                    'probability': float(pred['probability']),
                    'confidence_std': float(pred.get('confidence_std', 0)),
                    'strategy': pred['strategy'],
                    'date': pred['date'],
                    'sector': SECTOR_MAP.get(symbol, 'Other'),
                    'metrics': model_data.get('metrics', {})
                })

        # Sort by signal priority and probability
        signal_priority = {'BUY': 0, 'SELL': 1, 'HOLD': 2}
        signals.sort(key=lambda x: (signal_priority.get(x['signal'], 3), -x['probability']))

        return jsonify({
            'status': 'success',
            'signals': signals,
            'summary': {
                'total': len(signals),
                'buy': len([s for s in signals if s['signal'] == 'BUY']),
                'sell': len([s for s in signals if s['signal'] == 'SELL']),
                'hold': len([s for s in signals if s['signal'] == 'HOLD'])
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"[CHINA MODEL] Error generating signals: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/china/portfolio')
def china_model_portfolio():
    """Get recommended portfolio allocation from China Model Platform."""
    factory, constructor, risk_analyzer = get_china_model_factory()

    if constructor is None:
        return jsonify({
            'status': 'error',
            'error': 'Portfolio constructor not available'
        }), 503

    try:
        # Get total capital from query param (default $100,000)
        total_capital = float(request.args.get('capital', 100000))

        # Construct portfolio
        symbols = list(factory.models.keys())
        portfolio = constructor.construct_portfolio(
            symbols=symbols,
            total_capital=total_capital,
            verbose=False
        )

        # Format positions for response
        positions = []
        for symbol, pos in portfolio['positions'].items():
            if symbol == 'CASH':
                positions.append({
                    'symbol': 'CASH',
                    'sector': '-',
                    'weight': float(pos['weight']),
                    'allocation': float(pos['allocation']),
                    'signal': '-'
                })
            else:
                positions.append({
                    'symbol': symbol,
                    'sector': pos.get('sector', 'Other'),
                    'weight': float(pos['weight']),
                    'allocation': float(pos['allocation']),
                    'signal': pos.get('signal', 'N/A'),
                    'probability': float(pos.get('probability', 0)),
                    'strategy': pos.get('strategy', 'N/A'),
                    'volatility': float(pos.get('volatility', 0))
                })

        # Sort by weight
        positions.sort(key=lambda x: x['weight'], reverse=True)

        return jsonify({
            'status': 'success',
            'portfolio': {
                'positions': positions,
                'metrics': portfolio['metrics'],
                'total_capital': total_capital,
                'timestamp': portfolio['timestamp']
            }
        })

    except Exception as e:
        logger.error(f"[CHINA MODEL] Error constructing portfolio: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/china/risk')
def china_model_risk():
    """Get risk metrics for China Model portfolio."""
    factory, constructor, risk_analyzer = get_china_model_factory()

    if risk_analyzer is None:
        return jsonify({
            'status': 'error',
            'error': 'Risk analyzer not available'
        }), 503

    try:
        # Ensure portfolio is constructed first
        if not constructor.portfolio:
            symbols = list(factory.models.keys())
            constructor.construct_portfolio(symbols=symbols, verbose=False)

        # Generate risk report
        risk_report = risk_analyzer.generate_risk_report(verbose=False)

        return jsonify({
            'status': 'success',
            'risk_metrics': {
                'var_95_1day': risk_report.get('var_95_1day'),
                'var_95_1day_pct': f"{risk_report.get('var_95_1day', 0) * 100:.2f}%" if risk_report.get('var_95_1day') else 'N/A',
                'var_99_1day': risk_report.get('var_99_1day'),
                'var_99_1day_pct': f"{risk_report.get('var_99_1day', 0) * 100:.2f}%" if risk_report.get('var_99_1day') else 'N/A',
                'max_drawdown': risk_report.get('max_drawdown'),
                'max_drawdown_pct': f"{risk_report.get('max_drawdown', 0) * 100:.2f}%" if risk_report.get('max_drawdown') else 'N/A',
                'portfolio_volatility': constructor.portfolio.get('metrics', {}).get('portfolio_volatility'),
                'portfolio_volatility_pct': f"{constructor.portfolio.get('metrics', {}).get('portfolio_volatility', 0) * 100:.1f}%" if constructor.portfolio else 'N/A',
                'num_positions': constructor.portfolio.get('metrics', {}).get('num_positions', 0),
                'total_invested': constructor.portfolio.get('metrics', {}).get('total_invested', 0)
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"[CHINA MODEL] Error generating risk metrics: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/china/robust-performers')
def china_robust_performers():
    """Get list of robust performers identified by 3-tier screening."""
    try:
        results_path = os.path.join(
            os.path.dirname(__file__),
            'china_model', 'results', 'tiered_screening_results.json'
        )

        if not os.path.exists(results_path):
            return jsonify({
                'status': 'error',
                'error': 'Screening results not found'
            }), 404

        with open(results_path, 'r') as f:
            screening_results = json.load(f)

        robust_performers = screening_results.get('robust_performers', [])

        return jsonify({
            'status': 'success',
            'robust_performers': robust_performers,
            'summary': screening_results.get('summary', {}),
            'timestamp': screening_results.get('timestamp', datetime.now().isoformat())
        })

    except Exception as e:
        logger.error(f"[CHINA MODEL] Error loading robust performers: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/china/add-to-mock', methods=['POST'])
def china_add_to_mock_trading():
    """Add China Model signal to mock trading portfolio."""
    factory, constructor, risk_analyzer = get_china_model_factory()

    if factory is None:
        return jsonify({
            'status': 'error',
            'error': 'China Model Platform not available'
        }), 503

    try:
        data = request.json
        symbol = data.get('symbol')
        shares = data.get('shares', 100)
        user_id = data.get('user_id', 'default_user')

        if symbol not in factory.models:
            return jsonify({
                'status': 'error',
                'error': f'Symbol {symbol} not in China Model universe'
            }), 400

        # Get current prediction
        pred = factory.predict(symbol)
        if pred is None:
            return jsonify({
                'status': 'error',
                'error': f'Could not generate prediction for {symbol}'
            }), 500

        # Get current price
        ticker_obj = yf.Ticker(symbol)
        current_data = ticker_obj.history(period='1d')
        if len(current_data) == 0:
            return jsonify({
                'status': 'error',
                'error': f'Could not fetch price for {symbol}'
            }), 500

        current_price = float(current_data['Close'].iloc[-1])

        # Calculate stop-loss and take-profit based on volatility
        model_data = factory.models.get(symbol, {})
        volatility = model_data.get('metrics', {}).get('ensemble_return', 0.02)

        # Map signal to action
        signal_action = pred['signal']
        if signal_action == 'BUY':
            signal_action = 'LONG'
            direction = 1
            stop_loss = current_price * (1 - 0.05)  # 5% stop-loss
            take_profit = current_price * (1 + 0.10)  # 10% take-profit
        elif signal_action == 'SELL':
            signal_action = 'SHORT'
            direction = -1
            stop_loss = current_price * (1 + 0.05)
            take_profit = current_price * (1 - 0.10)
        else:
            signal_action = 'HOLD'
            direction = 0
            stop_loss = None
            take_profit = None

        # Create portfolio item
        portfolio_item = {
            'id': datetime.now().timestamp(),
            'ticker': symbol,
            'entry_date': datetime.now().isoformat(),
            'entry_price': current_price,
            'predicted_direction': direction,
            'predicted_volatility': volatility,
            'signal_action': signal_action,
            'confidence': pred['probability'],
            'shares': shares,
            'notes': f"China Model: {pred['strategy']} strategy, {pred['probability']*100:.1f}% confidence",
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'status': 'open',
            'source': 'china_model'
        }

        USER_PORTFOLIOS[user_id].append(portfolio_item)
        save_portfolios()

        return jsonify({
            'status': 'success',
            'message': f'Added {symbol} to mock trading portfolio',
            'position': portfolio_item,
            'portfolio_size': len(USER_PORTFOLIOS[user_id])
        })

    except Exception as e:
        logger.error(f"[CHINA MODEL] Error adding to mock trading: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/china/daily-update', methods=['POST'])
def china_daily_update():
    """Run daily portfolio update and get fresh signals."""
    factory, constructor, risk_analyzer = get_china_model_factory()

    if constructor is None:
        return jsonify({
            'status': 'error',
            'error': 'Portfolio constructor not available'
        }), 503

    try:
        # Run daily update
        update_result = constructor.daily_update(verbose=False)

        return jsonify({
            'status': 'success',
            'date': update_result['date'],
            'signals': update_result['signals'],
            'portfolio': {
                'positions': update_result['portfolio']['positions'],
                'metrics': update_result['portfolio']['metrics']
            }
        })

    except Exception as e:
        logger.error(f"[CHINA MODEL] Error running daily update: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ============================================================================
# PORTFOLIO TRACKING ENDPOINTS
# ============================================================================

@app.route('/api/portfolio/add', methods=['POST'])
def add_to_portfolio():
    """Add a prediction to user's watchlist/portfolio."""
    try:
        data = request.json
        user_id = data.get('user_id', 'default_user')  # In production, use auth

        portfolio_item = {
            'id': datetime.now().timestamp(),
            'ticker': data['ticker'],
            'entry_date': datetime.now().isoformat(),
            'entry_price': data['entry_price'],
            'predicted_direction': data['predicted_direction'],
            'predicted_volatility': data['predicted_volatility'],
            'signal_action': data.get('signal_action', 'HOLD'),
            'confidence': data.get('confidence', 0.5),
            'shares': data.get('shares', 0),
            'notes': data.get('notes', ''),
            'stop_loss': data.get('stop_loss'),  # Store stop-loss level
            'take_profit': data.get('take_profit'),  # Store take-profit level
            'status': 'open'  # Track position status
        }

        USER_PORTFOLIOS[user_id].append(portfolio_item)
        save_portfolios()  # Persist to disk

        return jsonify({
            'status': 'success',
            'message': f'Added {data["ticker"]} to portfolio',
            'portfolio_size': len(USER_PORTFOLIOS[user_id])
        })

    except Exception as e:
        logger.error(f"Error adding to portfolio: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 400


@app.route('/api/portfolio/list')
def get_portfolio():
    """Get user's portfolio with real-time prices and P&L."""
    try:
        user_id = request.args.get('user_id', 'default_user')
        portfolio = USER_PORTFOLIOS.get(user_id, [])

        if not portfolio:
            return jsonify({
                'status': 'success',
                'portfolio': [],
                'summary': {
                    'total_items': 0,
                    'total_value': 0,
                    'total_pnl': 0,
                    'win_rate': 0
                }
            })

        # Fetch real-time prices for all tickers
        enriched_portfolio = []
        closed_trades = []
        total_pnl = 0
        wins = 0
        auto_close_occurred = False  # Track if any positions were auto-closed

        for item in portfolio:
            try:
                # Get current price - fresh Ticker instance for each to avoid caching issues
                ticker_symbol = item['ticker']
                ticker_obj = yf.Ticker(ticker_symbol)
                current_data = ticker_obj.history(period='1d')

                if len(current_data) > 0:
                    current_price = float(current_data['Close'].iloc[-1])

                    # Sanity check: Verify price is reasonable vs entry price
                    # (within 90% drop or 1000% gain to catch data mixups)
                    entry_price = item['entry_price']
                    price_ratio = current_price / entry_price if entry_price > 0 else 1
                    if price_ratio < 0.1 or price_ratio > 10:
                        logger.warning(f"[PORTFOLIO] PRICE SANITY CHECK FAILED for {ticker_symbol}: "
                                      f"entry=${entry_price:.2f}, current=${current_price:.2f}, "
                                      f"ratio={price_ratio:.2f}. Using entry price to avoid incorrect P&L.")
                        current_price = entry_price

                    # Calculate P&L
                    price_change = current_price - item['entry_price']
                    price_change_pct = (price_change / item['entry_price']) * 100

                    # Check if prediction was correct
                    actual_direction = 1 if price_change > 0 else -1 if price_change < 0 else 0
                    prediction_correct = (actual_direction == item['predicted_direction'])

                    if prediction_correct and actual_direction != 0:
                        wins += 1

                    # Calculate position P&L if shares specified
                    position_pnl = price_change * item['shares'] if item['shares'] > 0 else 0
                    total_pnl += position_pnl

                    # AUTO-CLOSE LOGIC: Check if stop-loss or take-profit hit
                    auto_closed = False
                    close_reason = None

                    if item.get('status') == 'open' and item.get('shares', 0) > 0:
                        stop_loss = item.get('stop_loss')
                        take_profit = item.get('take_profit')
                        # Check both signal_action and action fields (frontend uses 'action', backend uses 'signal_action')
                        signal_action = item.get('signal_action', item.get('action', 'HOLD'))

                        # For LONG/BUY positions
                        if signal_action in ['LONG', 'BUY']:
                            if stop_loss and current_price <= stop_loss:
                                auto_closed = True
                                close_reason = f'Stop-loss hit (${stop_loss:.2f})'
                                logger.info(f"AUTO-CLOSE: {item['ticker']} - Stop-loss triggered at ${current_price:.2f} (SL: ${stop_loss:.2f})")
                            elif take_profit and current_price >= take_profit:
                                auto_closed = True
                                close_reason = f'Take-profit hit (${take_profit:.2f})'
                                logger.info(f"AUTO-CLOSE: {item['ticker']} - Take-profit triggered at ${current_price:.2f} (TP: ${take_profit:.2f})")

                        # For SHORT/SELL positions
                        elif signal_action in ['SHORT', 'SELL']:
                            if stop_loss and current_price >= stop_loss:
                                auto_closed = True
                                close_reason = f'Stop-loss hit (${stop_loss:.2f})'
                                logger.info(f"AUTO-CLOSE: {item['ticker']} - Stop-loss triggered at ${current_price:.2f} (SL: ${stop_loss:.2f})")
                            elif take_profit and current_price <= take_profit:
                                auto_closed = True
                                close_reason = f'Take-profit hit (${take_profit:.2f})'
                                logger.info(f"AUTO-CLOSE: {item['ticker']} - Take-profit triggered at ${current_price:.2f} (TP: ${take_profit:.2f})")

                    enriched_item = {
                        **item,
                        'current_price': float(current_price),
                        'price_change': float(price_change),
                        'price_change_pct': float(price_change_pct),
                        'position_pnl': float(position_pnl),
                        'prediction_correct': prediction_correct,
                        'days_held': (datetime.now() - datetime.fromisoformat(item['entry_date'])).days
                    }

                    # If auto-closed, update status and move to closed_trades
                    if auto_closed:
                        enriched_item['status'] = 'closed'
                        enriched_item['exit_date'] = datetime.now().isoformat()
                        enriched_item['exit_price'] = float(current_price)
                        enriched_item['close_reason'] = close_reason
                        closed_trades.append(enriched_item)

                        # Update the item in portfolio to mark as closed
                        item['status'] = 'closed'
                        item['exit_date'] = enriched_item['exit_date']
                        item['exit_price'] = enriched_item['exit_price']
                        item['close_reason'] = close_reason
                        auto_close_occurred = True  # Flag to save later
                    else:
                        # Add to active portfolio only if still open
                        if item.get('status') != 'closed':
                            enriched_portfolio.append(enriched_item)
                        else:
                            # Previously closed trade
                            enriched_item['close_reason'] = item.get('close_reason', 'Manually closed')
                            closed_trades.append(enriched_item)
                else:
                    enriched_item = {**item, 'current_price': item['entry_price'], 'error': 'Price unavailable'}
                    if item.get('status') != 'closed':
                        enriched_portfolio.append(enriched_item)

            except Exception as e:
                logger.error(f"Error fetching price for {item['ticker']}: {str(e)}")
                enriched_portfolio.append({**item, 'current_price': item['entry_price'], 'error': str(e)})

        # Calculate summary
        win_rate = (wins / len(portfolio)) * 100 if len(portfolio) > 0 else 0
        total_value = sum(item.get('current_price', item['entry_price']) * item['shares']
                         for item in enriched_portfolio if item['shares'] > 0)

        # Calculate closed trades summary
        closed_pnl = sum(item.get('position_pnl', 0) for item in closed_trades)
        closed_wins = sum(1 for item in closed_trades if item.get('position_pnl', 0) > 0)
        closed_win_rate = (closed_wins / len(closed_trades)) * 100 if len(closed_trades) > 0 else 0


        # Save if any positions were auto-closed
        if auto_close_occurred:
            save_portfolios()
        return jsonify({
            'status': 'success',
            'portfolio': enriched_portfolio,
            'closed_trades': closed_trades,
            'summary': {
                'total_items': len(portfolio),
                'open_positions': len(enriched_portfolio),
                'closed_positions': len(closed_trades),
                'total_value': float(total_value),
                'total_pnl': float(total_pnl),
                'win_rate': float(win_rate),
                'correct_predictions': wins,
                'closed_pnl': float(closed_pnl),
                'closed_win_rate': float(closed_win_rate),
                'auto_closed_count': sum(1 for item in closed_trades if 'Auto-close' in item.get('close_reason', ''))
            }
        })

    except Exception as e:
        logger.error(f"Error getting portfolio: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/portfolio/remove/<item_id>', methods=['DELETE'])
def remove_from_portfolio(item_id):
    """Remove item from portfolio."""
    try:
        user_id = request.args.get('user_id', 'default_user')
        item_id_float = float(item_id)

        original_len = len(USER_PORTFOLIOS[user_id])
        USER_PORTFOLIOS[user_id] = [item for item in USER_PORTFOLIOS[user_id]
                                     if item['id'] != item_id_float]
        save_portfolios()  # Persist to disk
        if len(USER_PORTFOLIOS[user_id]) < original_len:
            return jsonify({'status': 'success', 'message': 'Item removed'})
        else:
            return jsonify({'status': 'error', 'message': 'Item not found'}), 404

    except Exception as e:
        logger.error(f"Error removing from portfolio: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 400


@app.route('/api/watchlist/clear-all', methods=['DELETE'])
def clear_watchlist():
    """Clear all items from watchlist."""
    try:
        user_id = request.args.get('user_id', 'default_user')

        items_count = len(USER_WATCHLISTS[user_id])
        USER_WATCHLISTS[user_id] = []
        save_watchlists()  # Persist to disk
        logger.info(f"Cleared {items_count} items from watchlist for user {user_id}")

        return jsonify({
            'status': 'success',
            'message': f'Cleared {items_count} item(s) from watchlist',
            'cleared_count': items_count
        })
    except Exception as e:
        logger.error(f"Error clearing watchlist: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 400


@app.route('/api/portfolio/close-all', methods=['POST'])
def close_all_positions():
    """Close all open positions in mock trading portfolio."""
    try:
        user_id = request.args.get('user_id', 'default_user')

        portfolio = USER_PORTFOLIOS[user_id]
        open_positions = [item for item in portfolio if item.get('status') == 'open']

        if len(open_positions) == 0:
            return jsonify({
                'status': 'success',
                'message': 'No open positions to close',
                'summary': {
                    'closed_count': 0,
                    'total_pnl': 0,
                    'avg_pnl_percent': 0
                }
            })

        # Close each position at current market price
        closed_count = 0
        total_pnl = 0
        total_pnl_percent = 0

        for item in open_positions:
            ticker = item['ticker']
            try:
                # Get current price - use fresh Ticker instance to avoid caching issues
                logger.info(f"[CLOSE-ALL] Fetching price for ticker: {ticker}")
                stock = yf.Ticker(ticker)
                current_data = stock.history(period='1d', interval='1m')

                if current_data.empty:
                    # Fallback to last day
                    logger.info(f"[CLOSE-ALL] No intraday data for {ticker}, trying 5d fallback")
                    current_data = stock.history(period='5d')

                if not current_data.empty:
                    current_price = float(current_data['Close'].iloc[-1])

                    # Sanity check: Verify price is reasonable vs entry price
                    # (within 90% drop or 1000% gain to catch data mixups)
                    entry_price = item['entry_price']
                    price_ratio = current_price / entry_price if entry_price > 0 else 1
                    if price_ratio < 0.1 or price_ratio > 10:
                        logger.warning(f"[CLOSE-ALL] PRICE SANITY CHECK FAILED for {ticker}: "
                                      f"entry=${entry_price:.2f}, current=${current_price:.2f}, "
                                      f"ratio={price_ratio:.2f}. Using entry price to avoid incorrect P&L.")
                        # Use entry price to avoid incorrect P&L calculation
                        current_price = entry_price

                    logger.info(f"[CLOSE-ALL] {ticker}: entry=${entry_price:.2f}, exit=${current_price:.2f}")

                    # Calculate P&L
                    entry_price = item['entry_price']
                    shares = item.get('shares', 0)
                    # Check both signal_action and action fields (frontend uses 'action', backend uses 'signal_action')
                    signal_action = item.get('signal_action', item.get('action', 'HOLD'))

                    # LONG/BUY: profit when price goes UP
                    if signal_action in ['LONG', 'BUY']:
                        pnl = (current_price - entry_price) * shares
                        pnl_percent = ((current_price - entry_price) / entry_price) * 100
                    # SHORT/SELL: profit when price goes DOWN
                    elif signal_action in ['SHORT', 'SELL']:
                        pnl = (entry_price - current_price) * shares
                        pnl_percent = ((entry_price - current_price) / entry_price) * 100
                    else:
                        pnl = 0
                        pnl_percent = 0

                    # Update position status
                    item['status'] = 'closed'
                    item['exit_date'] = datetime.now().isoformat()
                    item['exit_price'] = current_price
                    item['close_reason'] = 'Manual close - Close All Positions'
                    item['pnl'] = pnl
                    item['pnl_percent'] = pnl_percent

                    total_pnl += pnl
                    total_pnl_percent += pnl_percent
                    closed_count += 1

                    logger.info(f"Closed position: {ticker} - P&L: ${pnl:.2f} ({pnl_percent:.2f}%)")

            except Exception as e:
                logger.error(f"Error closing position {ticker}: {str(e)}")
                continue

        avg_pnl_percent = total_pnl_percent / closed_count if closed_count > 0 else 0

        logger.info(f"Closed {closed_count} positions - Total P&L: ${total_pnl:.2f} ({avg_pnl_percent:.2f}%)")

        # Save changes to disk
        save_portfolios()
        return jsonify({
            'status': 'success',
            'message': f'Successfully closed {closed_count} position(s)',
            'summary': {
                'closed_count': closed_count,
                'total_pnl': total_pnl,
                'avg_pnl_percent': avg_pnl_percent
            }
        })

    except Exception as e:
        logger.error(f"Error closing all positions: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 400


# =============================================================================
# PHASE 1 ADVANCED FEATURES API ENDPOINTS (20 features)
# =============================================================================

@app.route('/api/phase1/features')
def phase1_features():
    """Get status of all 20 Phase 1 advanced features."""
    if not PHASE1_AVAILABLE:
        return jsonify({
            'status': 'error',
            'error': 'Phase 1 features not available',
            'available': False
        }), 503

    try:
        api = get_phase1_api()
        return jsonify(api.get_all_features_status())
    except Exception as e:
        logger.error(f"Error getting Phase 1 features: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/phase1/signal', methods=['POST'])
def phase1_signal():
    """Generate enhanced trading signal using all 20 features."""
    if not PHASE1_AVAILABLE:
        return jsonify({'status': 'error', 'error': 'Phase 1 not available'}), 503

    try:
        api = get_phase1_api()
        data = request.get_json() or {}
        return jsonify(api.get_signal(data))
    except Exception as e:
        logger.error(f"Error generating Phase 1 signal: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/phase1/allocation', methods=['POST'])
def phase1_allocation():
    """Get optimal portfolio allocation using risk parity."""
    if not PHASE1_AVAILABLE:
        return jsonify({'status': 'error', 'error': 'Phase 1 not available'}), 503

    try:
        api = get_phase1_api()
        data = request.get_json() or {}
        return jsonify(api.get_allocation(data))
    except Exception as e:
        logger.error(f"Error calculating Phase 1 allocation: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/phase1/execution', methods=['POST'])
def phase1_execution():
    """Get optimal execution plan (VWAP/TWAP)."""
    if not PHASE1_AVAILABLE:
        return jsonify({'status': 'error', 'error': 'Phase 1 not available'}), 503

    try:
        api = get_phase1_api()
        data = request.get_json() or {}
        return jsonify(api.get_execution(data))
    except Exception as e:
        logger.error(f"Error getting Phase 1 execution plan: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/phase1/performance', methods=['POST'])
def phase1_performance():
    """Analyze trading performance with attribution."""
    if not PHASE1_AVAILABLE:
        return jsonify({'status': 'error', 'error': 'Phase 1 not available'}), 503

    try:
        api = get_phase1_api()
        data = request.get_json() or {}
        return jsonify(api.get_performance(data))
    except Exception as e:
        logger.error(f"Error analyzing Phase 1 performance: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/phase1/risk', methods=['POST'])
def phase1_risk():
    """Get comprehensive risk status with forecasts."""
    if not PHASE1_AVAILABLE:
        return jsonify({'status': 'error', 'error': 'Phase 1 not available'}), 503

    try:
        api = get_phase1_api()
        data = request.get_json() or {}
        return jsonify(api.get_risk(data))
    except Exception as e:
        logger.error(f"Error getting Phase 1 risk status: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/phase1/stress')
def phase1_stress():
    """Get stress protection system status (VIX levels, circuit breakers)."""
    if not PHASE1_AVAILABLE:
        return jsonify({'status': 'error', 'error': 'Phase 1 not available'}), 503

    try:
        api = get_phase1_api()
        return jsonify(api.get_stress_status())
    except Exception as e:
        logger.error(f"Error getting Phase 1 stress status: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/phase1/emergency', methods=['POST'])
def phase1_emergency():
    """Execute emergency liquidation plan."""
    if not PHASE1_AVAILABLE:
        return jsonify({'status': 'error', 'error': 'Phase 1 not available'}), 503

    try:
        api = get_phase1_api()
        data = request.get_json() or {}
        return jsonify(api.emergency_liquidate(data))
    except Exception as e:
        logger.error(f"Error executing Phase 1 emergency liquidation: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("ML STOCK TRADING PLATFORM")
    print("=" * 70)
    print("\nLoading saved data...")
    load_portfolios()
    load_watchlists()
    print("[OK] Data loaded successfully")
    print("\nStarting Flask server...")
    print("Access at: http://localhost:5000")
    print("\nPress Ctrl+C to stop")
    print("=" * 70)

    # Run with use_reloader=False to prevent constant reloads from file changes
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
