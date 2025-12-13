// ML Stock Trading Platform - JavaScript

let currentTicker = '';
let currentPrediction = null;
let searchTimeout = null;
let currentTopPicks = { buy: [], sell: [] }; // Store current top picks for news feed
let currentRegime = 'all'; // Track current regime filter

// Check API health on load
async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();

        const statusDot = document.getElementById('status-dot');
        const statusText = document.getElementById('status-text');

        if (data.status === 'healthy') {
            statusDot.style.background = '#10b981';
            statusText.textContent = 'System Operational';
        } else {
            statusDot.style.background = '#ef4444';
            statusText.textContent = 'System Error';
        }
    } catch (error) {
        console.error('Health check failed:', error);
        const statusDot = document.getElementById('status-dot');
        const statusText = document.getElementById('status-text');
        statusDot.style.background = '#ef4444';
        statusText.textContent = 'Connection Error';
    }
}

// Update monitoring stats
async function updateMonitoringStats() {
    try {
        const response = await fetch('/api/monitoring/summary');
        const data = await response.json();

        // Update cache hit rate
        const cacheHitRate = data.cache_performance.hit_rate_percent.toFixed(1) + '%';
        document.getElementById('cache-hit-rate').textContent = cacheHitRate;

        // Update avg training time
        const avgTime = data.cache_performance.avg_training_time_seconds.toFixed(2) + 's';
        document.getElementById('avg-training-time').textContent = avgTime;

        // Update models cached (from cache hits + misses)
        const modelsCached = data.cache_performance.cache_hits + data.cache_performance.cache_misses;
        document.getElementById('models-cached').textContent = modelsCached;

        // Update data quality (show tickers monitored)
        const dataQuality = data.data_quality.total_tickers_monitored + ' tracked';
        document.getElementById('data-quality').textContent = dataQuality;
    } catch (error) {
        console.error('Failed to fetch monitoring stats:', error);
    }
}

// Search for tickers
async function searchTickers(query) {
    try {
        console.log('[SEARCH] Searching for:', query);
        const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        const data = await response.json();
        console.log('[SEARCH] Response:', data);
        if (data.results && data.results.length > 0) {
            displaySearchResults(data.results);
        } else {
            console.log('[SEARCH] No results found');
            document.getElementById('search-results').classList.add('hidden');
        }
    } catch (error) {
        console.error('[SEARCH] Search failed:', error);
    }
}

// Display search results
function displaySearchResults(results) {
    const resultsContainer = document.getElementById('search-results');

    if (results.length === 0) {
        resultsContainer.classList.add('hidden');
        return;
    }

    resultsContainer.innerHTML = results.map(item => {
        // For China stocks, show company name prominently
        const isChinaStock = item.ticker && (item.ticker.endsWith('.HK') || item.ticker.endsWith('.SS') || item.ticker.endsWith('.SZ'));
        return `
        <div class="search-result-item" onclick="selectTicker('${item.ticker}')">
            ${isChinaStock ? `
                <div class="result-ticker">${item.name}</div>
                <div class="result-name">${item.ticker}</div>
            ` : `
                <div class="result-ticker">${item.ticker}</div>
                <div class="result-name">${item.name}</div>
            `}
            <div class="result-meta">${item.type} | ${item.exchange}</div>
        </div>
    `}).join('');

    resultsContainer.classList.remove('hidden');
}

// Select ticker from search results
function selectTicker(ticker) {
    currentTicker = ticker;
    document.getElementById('ticker-input').value = ticker;
    document.getElementById('search-results').classList.add('hidden');
}

// Analyze ticker directly (called from Top 10 picks)
// This sets the ticker and immediately runs the analysis
async function analyzeTickerDirectly(ticker) {
    currentTicker = ticker;
    document.getElementById('ticker-input').value = ticker;
    document.getElementById('search-results').classList.add('hidden');

    // Trigger the prediction analysis (loading indicator handled by getPrediction)
    await getPrediction();
}

// Show/hide layer functions
function showAnalysisLayer() {
    document.getElementById('main-view').style.display = 'none';
    document.getElementById('analysis-layer').classList.remove('hidden');
    // Scroll to top of analysis layer
    window.scrollTo(0, 0);
}

function showMainView() {
    document.getElementById('analysis-layer').classList.add('hidden');
    document.getElementById('main-view').style.display = 'block';
    // Scroll to top of main view
    window.scrollTo(0, 0);
}

// Get prediction
async function getPrediction() {
    const ticker = document.getElementById('ticker-input').value.trim().toUpperCase();

    if (!ticker) {
        alert('Please enter a ticker symbol');
        return;
    }

    console.log('[PREDICT] Getting prediction for:', ticker);
    currentTicker = ticker;

    // Show inline loading indicator in search section
    const searchLoading = document.getElementById('search-loading');
    const loadingTicker = document.getElementById('loading-ticker');
    if (searchLoading && loadingTicker) {
        loadingTicker.textContent = ticker;
        searchLoading.classList.remove('hidden');
    }
    document.getElementById('error').classList.add('hidden');

    try {
        console.log('[PREDICT] Fetching /api/predict/' + ticker);
        const response = await fetch(`/api/predict/${ticker}`);
        console.log('[PREDICT] Response status:', response.status);
        const data = await response.json();
        console.log('[PREDICT] Response data:', data);

        if (data.status === 'error') {
            console.error('[PREDICT] Error from server:', data.error);
            showError(data.error);
            return;
        }

        currentPrediction = data;
        displayPrediction(data);

        // Switch to analysis layer after prediction is ready
        showAnalysisLayer();
    } catch (error) {
        console.error('[PREDICT] Prediction failed:', error);
        showError('Failed to get prediction. Please try again. Error: ' + error.message);
    } finally {
        // Hide inline loading indicator
        if (searchLoading) {
            searchLoading.classList.add('hidden');
        }
    }
}

// Display prediction
function displayPrediction(data) {
    // Show results container
    document.getElementById('results').classList.remove('hidden');

    // Company header
    document.getElementById('company-name').textContent = `${data.company.name} (${data.ticker})`;
    document.getElementById('company-meta').textContent = `${data.company.type} | ${data.company.exchange}`;
    document.getElementById('current-price').textContent = `$${data.current_price.toFixed(2)}`;

    // Prediction
    const predVol = (data.prediction.volatility * 100).toFixed(2);
    const ciLower = (data.prediction.confidence_interval.lower * 100).toFixed(2);
    const ciUpper = (data.prediction.confidence_interval.upper * 100).toFixed(2);

    document.getElementById('pred-vol').textContent = `${predVol}%`;
    document.getElementById('ci-lower').textContent = `${ciLower}%`;
    document.getElementById('ci-upper').textContent = `${ciUpper}%`;

    // Market context
    const regime = data.market_context.regime;
    const regimeBadge = document.getElementById('regime-badge');
    regimeBadge.textContent = regime.toUpperCase();
    regimeBadge.className = `badge badge-${regime}`;

    const histVol = (data.market_context.historical_volatility * 100).toFixed(2);
    document.getElementById('hist-vol').textContent = `${histVol}%`;

    const volPercentile = Math.round(data.market_context.volatility_percentile * 100);
    document.getElementById('vol-percentile').textContent = `${volPercentile}th`;

    // Direction - handle NaN and null values properly
    const direction = data.prediction.direction;
    const rawDirConf = data.prediction.direction_confidence;
    const directionConf = (rawDirConf && !isNaN(rawDirConf))
        ? (rawDirConf * 100).toFixed(1)
        : 'N/A';

    const directionIcon = document.getElementById('direction-icon');
    const directionText = document.getElementById('direction-text');

    // Neutral explanations for better user communication
    const neutralExplanations = [
        'Market conditions unclear - waiting for stronger signal',
        'No clear directional edge detected',
        'Model suggests patience - conditions not favorable'
    ];
    const randomNeutralMsg = neutralExplanations[Math.floor(Math.random() * neutralExplanations.length)];

    if (direction > 0) {
        directionIcon.textContent = '📈';
        directionIcon.style.background = '#d1fae5';
        directionIcon.style.color = '#065f46';
        directionText.textContent = 'Upward movement expected';
    } else if (direction < 0) {
        directionIcon.textContent = '📉';
        directionIcon.style.background = '#fee2e2';
        directionIcon.style.color = '#991b1b';
        directionText.textContent = 'Downward movement expected';
    } else {
        // Direction is 0, null, or NaN - show neutral with helpful message
        directionIcon.textContent = '⚪';
        directionIcon.style.background = '#f3f4f6';
        directionIcon.style.color = '#6b7280';
        directionText.textContent = randomNeutralMsg;
    }

    document.getElementById('direction-conf').textContent = `${directionConf}%`;

    // === DUAL SIGNAL DISPLAY ===

    // 1. ML Model Prediction (raw directional signal)
    const modelDirectionBadge = document.getElementById('model-direction-badge');
    const modelDirectionConf = document.getElementById('model-direction-conf');
    const modelPredDesc = document.getElementById('model-prediction-desc');

    // Determine model direction based on prediction_direction (raw model output)
    let modelDirection = 'HOLD';
    if (direction > 0) {
        modelDirection = 'BUY';
    } else if (direction < 0) {
        modelDirection = 'SELL';
    }

    modelDirectionBadge.textContent = modelDirection;
    modelDirectionBadge.className = `model-direction-badge model-direction-${modelDirection.toLowerCase()}`;
    modelDirectionConf.textContent = `${directionConf}%`;

    // Set description based on direction
    if (direction > 0) {
        modelPredDesc.textContent = 'ML ensemble predicts upward movement';
    } else if (direction < 0) {
        modelPredDesc.textContent = 'ML ensemble predicts downward movement';
    } else {
        modelPredDesc.textContent = 'No clear directional signal from ML model';
    }

    // 2. Trading Strategy (risk-filtered recommendation)
    const signal = data.trading_signal;
    const actionBadge = document.getElementById('action-badge');
    actionBadge.textContent = signal.action;
    actionBadge.className = `action-badge action-${signal.action.toLowerCase()}`;

    const rawSignalConf = signal.confidence;
    const signalConf = (rawSignalConf && !isNaN(rawSignalConf))
        ? (rawSignalConf * 100).toFixed(1)
        : 'N/A';
    document.getElementById('signal-conf').textContent = `${signalConf}%`;

    // 3. Show explanation box if signals differ
    const signalDiffBox = document.getElementById('signal-difference-box');
    const signalDiffText = document.getElementById('signal-difference-text');

    if (modelDirection !== signal.action) {
        signalDiffBox.classList.remove('hidden');
        if (signal.action === 'HOLD' && modelDirection !== 'HOLD') {
            signalDiffText.textContent = `The model predicts ${modelDirection}, but the trading strategy recommends HOLD due to risk management filters (e.g., volatility or confidence thresholds). This is why this asset may appear in Top Picks but show HOLD here.`;
        } else {
            signalDiffText.textContent = `Model prediction (${modelDirection}) differs from trading strategy (${signal.action}) due to risk-adjusted filters.`;
        }
    } else {
        signalDiffBox.classList.add('hidden');
    }

    // Enhanced reason display for HOLD signals
    const signalReason = document.getElementById('signal-reason');
    let reasonText = signal.reason;
    if (!reasonText && signal.action === 'HOLD') {
        reasonText = 'No strong trading signal. Market conditions don\'t show clear directional edge - patience recommended.';
    } else if (!reasonText) {
        reasonText = 'Direction confidence is below threshold for actionable signals.';
    }
    signalReason.querySelector('p').textContent = reasonText;

    // Show Execute Trade button for BUY/SELL signals
    const executeTradeBtn = document.getElementById('execute-trade-btn');
    if (signal.action !== 'HOLD' && signal.position) {
        executeTradeBtn.classList.remove('hidden');
        // Remove old listener if exists
        const newTradeBtn = executeTradeBtn.cloneNode(true);
        executeTradeBtn.replaceWith(newTradeBtn);
        // Attach new listener
        document.getElementById('execute-trade-btn').addEventListener('click', executeMockTrade);
    } else {
        executeTradeBtn.classList.add('hidden');
    }

    // Position details - adjusted to user's actual available cash
    const positionDetails = document.getElementById('position-details');
    if (signal.action !== 'HOLD' && signal.position) {
        positionDetails.classList.remove('hidden');

        // Get user's actual available cash from mock portfolio
        const portfolio = JSON.parse(localStorage.getItem('mockPortfolio')) || { cash: 100000 };
        const availableCash = portfolio.cash || 100000;
        const entryPrice = signal.entry_price;

        // Calculate adjusted position size based on available cash (50% max)
        const MAX_POSITION_PCT = 0.50;
        const maxPositionValue = availableCash * MAX_POSITION_PCT;
        const adjustedShares = Math.floor(maxPositionValue / entryPrice);
        const adjustedValue = adjustedShares * entryPrice;

        // Calculate adjusted risk/reward based on adjusted shares
        const riskPerShare = Math.abs(entryPrice - signal.stop_loss);
        const rewardPerShare = Math.abs(signal.take_profit - entryPrice);
        const adjustedRiskAmount = adjustedShares * riskPerShare;
        const adjustedPotentialProfit = adjustedShares * rewardPerShare;

        document.getElementById('pos-shares').textContent = `${adjustedShares} shares`;
        document.getElementById('pos-entry').textContent = `$${entryPrice.toFixed(2)}`;
        document.getElementById('pos-stop').textContent = `$${signal.stop_loss.toFixed(2)}`;
        document.getElementById('pos-target').textContent = `$${signal.take_profit.toFixed(2)}`;

        document.getElementById('risk-amount').textContent = `$${adjustedRiskAmount.toFixed(2)}`;
        document.getElementById('potential-profit').textContent = `$${adjustedPotentialProfit.toFixed(2)}`;

        // Calculate risk/reward ratio
        const rrRatio = (rewardPerShare / riskPerShare).toFixed(1);
        document.getElementById('risk-reward').textContent = `1:${rrRatio}`;

        console.log('📊 Position details adjusted to user cash:', {
            availableCash: availableCash.toFixed(2),
            originalShares: signal.position.shares,
            adjustedShares: adjustedShares,
            adjustedValue: adjustedValue.toFixed(2)
        });
    } else {
        positionDetails.classList.add('hidden');
    }

    // Timestamp
    const timestamp = new Date(data.timestamp);
    document.getElementById('signal-timestamp').textContent = timestamp.toLocaleTimeString();

    // Model info
    document.getElementById('model-type').textContent = data.model_info.type;
    document.getElementById('model-features').textContent = data.model_info.features_count;
    document.getElementById('features-count').textContent = data.model_info.features_count;

    const trainedAt = new Date(data.model_info.trained_at);
    document.getElementById('model-trained').textContent = trainedAt.toLocaleString();

    // News feed (populate the general news feed section)
    displayGeneralNewsFeed(data.news_feed);

    // Price chart
    displayPriceChart(data.ticker, data.chart_data.dates, data.chart_data.prices);

    // Intraday chart (1-day real-time)
    displayIntradayChart(data.ticker);

    // Always attach event listener to "Add to Watchlist" button
    const addToPortfolioBtn = document.getElementById('add-to-portfolio-btn');
    if (addToPortfolioBtn) {
        console.log('🔘 Found "Add to Watchlist" button, attaching listener...');
        // Remove old listener if exists
        const newBtn = addToPortfolioBtn.cloneNode(true);
        addToPortfolioBtn.replaceWith(newBtn);
        // Attach new listener - use the auth.js addToWatchlist function for guest mode support
        document.getElementById('add-to-portfolio-btn').addEventListener('click', async () => {
            console.log('🖱️ "Add to Watchlist" button clicked!');
            console.log('Current prediction:', currentPrediction);
            if (!currentPrediction) {
                console.error('❌ No current prediction available');
                alert('Please generate a prediction first');
                return;
            }
            console.log('✅ Calling addToWatchlist with full prediction data');
            await addToWatchlist(currentPrediction);
        });
        console.log('✅ Listener attached successfully');
    } else {
        console.error('❌ "Add to Watchlist" button not found in DOM');
    }
}

// Display general news feed (ONLY for analysis layer - used when viewing specific asset)
function displayGeneralNewsFeed(newsItems) {
    const analysisNewsFeed = document.getElementById('analysis-news-feed');

    if (!newsItems || newsItems.length === 0) {
        const noNewsHTML = '<p class="text-muted" style="padding: 20px; text-align: center;">No recent news available for this asset</p>';
        if (analysisNewsFeed) analysisNewsFeed.innerHTML = noNewsHTML;
        return;
    }

    const newsHTML = newsItems.map(item => {
        // Sentiment color based on backend sentiment
        const sentimentColor = item.sentiment === 'positive' ? '#10b981' :
                               item.sentiment === 'negative' ? '#ef4444' : '#64748b';

        // Use sentiment score from backend if available, otherwise don't display percentage
        const sentimentDisplay = item.sentiment_score
            ? `<span class="news-sentiment-score" style="color: ${sentimentColor}; font-weight: 600; font-size: 0.7rem;">${Math.round(item.sentiment_score * 100)}%</span>`
            : '';

        return `
            <div class="news-item-small sentiment-${item.sentiment}">
                <span class="news-icon-small">${item.icon}</span>
                <div class="news-content-small">
                    <p class="news-headline-small">
                        ${item.url ? `<a href="${item.url}" target="_blank" rel="noopener noreferrer">${item.headline}</a>` : item.headline}
                    </p>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="news-time-small">${item.time} • ${item.source}</span>
                        ${sentimentDisplay}
                    </div>
                </div>
            </div>
        `;
    }).join('');

    // Only populate analysis layer news feed (NOT entrance page)
    if (analysisNewsFeed) analysisNewsFeed.innerHTML = newsHTML;
}

// Load and display news from watchlist and top picks (entrance page only)
async function loadEntrancePageNews() {
    console.log('📰 Loading entrance page news from watchlist and top picks...');
    const newsFeed = document.getElementById('general-news-feed');

    if (!newsFeed) {
        console.error('❌ general-news-feed element not found');
        return;
    }

    try {
        // Get ALL tickers from watchlist (user's customized choice)
        const watchlist = typeof getLocalWatchlist === 'function' ? getLocalWatchlist() : [];
        const watchlistTickers = watchlist.map(item => item.ticker); // All watchlist tickers, no limit

        // Get ALL tickers from top picks stored in global variable (all 10 buy + all 10 sell)
        let topPicksTickers = [];
        if (currentTopPicks.buy && currentTopPicks.buy.length > 0) {
            // Take ALL buy signals (up to 10)
            topPicksTickers.push(...currentTopPicks.buy.map(item => item.ticker));
        }
        if (currentTopPicks.sell && currentTopPicks.sell.length > 0) {
            // Take ALL sell signals (up to 10)
            topPicksTickers.push(...currentTopPicks.sell.map(item => item.ticker));
        }

        // Combine and deduplicate tickers (watchlist has priority as it's user's choice)
        const allTickers = [...new Set([...watchlistTickers, ...topPicksTickers])];

        if (allTickers.length === 0) {
            newsFeed.innerHTML = '<p class="text-muted" style="padding: 20px; text-align: center;">Add stocks to your watchlist or view Top 10 Trading Opportunities to see relevant news here!</p>';
            return;
        }

        console.log(`📊 Fetching news for ${allTickers.length} tickers (${watchlistTickers.length} from watchlist + ${topPicksTickers.length} from top picks [${currentRegime}]):`, allTickers);

        // Fetch news for all tickers (collect all news items)
        const allNewsItems = [];

        for (const ticker of allTickers) {
            try {
                const response = await fetch(`/api/predict/${ticker}`);
                const data = await response.json();

                if (data.status === 'success' && data.news_feed && data.news_feed.length > 0) {
                    // Add ticker and company name to each news item for context
                    const companyName = data.company?.name || ticker;
                    const newsWithTicker = data.news_feed.map(item => ({
                        ...item,
                        ticker: ticker,
                        companyName: companyName
                    }));
                    allNewsItems.push(...newsWithTicker);
                }
            } catch (error) {
                console.warn(`Failed to fetch news for ${ticker}:`, error);
            }
        }

        // Sort by time (most recent first) and take top 9 (for 3-column layout, 3 items per column)
        allNewsItems.sort((a, b) => {
            // Simple time comparison - adjust if needed based on actual time format
            return b.time.localeCompare(a.time);
        });

        const top9News = allNewsItems.slice(0, 9);

        if (top9News.length === 0) {
            newsFeed.innerHTML = '<p class="text-muted" style="padding: 20px; text-align: center;">No recent news available for your watchlist and top picks</p>';
            return;
        }

        // Display the news
        const newsHTML = top9News.map(item => {
            const sentimentColor = item.sentiment === 'positive' ? '#10b981' :
                                   item.sentiment === 'negative' ? '#ef4444' : '#64748b';

            const sentimentDisplay = item.sentiment_score
                ? `<span class="news-sentiment-score" style="color: ${sentimentColor}; font-weight: 600; font-size: 0.7rem;">${Math.round(item.sentiment_score * 100)}%</span>`
                : '';

            // For China stocks, show company name instead of ticker number
            const isChinaStock = item.ticker && (item.ticker.endsWith('.HK') || item.ticker.endsWith('.SS') || item.ticker.endsWith('.SZ'));
            const displayLabel = isChinaStock && item.companyName && item.companyName !== item.ticker
                ? item.companyName
                : item.ticker;

            return `
                <div class="news-item-small sentiment-${item.sentiment}">
                    <span class="news-icon-small">${item.icon}</span>
                    <div class="news-content-small">
                        <p class="news-headline-small">
                            <strong style="color: #2563eb;">[${displayLabel}]</strong>
                            ${item.url ? `<a href="${item.url}" target="_blank" rel="noopener noreferrer">${item.headline}</a>` : item.headline}
                        </p>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="news-time-small">${item.time} • ${item.source}</span>
                            ${sentimentDisplay}
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        newsFeed.innerHTML = newsHTML;
        console.log('✅ Entrance page news updated with', top9News.length, 'items');

    } catch (error) {
        console.error('❌ Error loading entrance page news:', error);
        newsFeed.innerHTML = '<p class="text-muted" style="padding: 20px; text-align: center;">Error loading news feed</p>';
    }
}

// Display price chart
function displayPriceChart(ticker, dates, prices) {
    const trace = {
        x: dates,
        y: prices,
        type: 'scatter',
        mode: 'lines',
        name: 'Price',
        line: {
            color: '#2563eb',
            width: 2
        }
    };

    const layout = {
        title: `${ticker} - 1 Year Price History`,
        xaxis: {
            title: 'Date',
            showgrid: true,
            gridcolor: '#e5e7eb'
        },
        yaxis: {
            title: 'Price ($)',
            showgrid: true,
            gridcolor: '#e5e7eb'
        },
        margin: { t: 50, r: 20, l: 60, b: 50 },
        hovermode: 'x unified',
        plot_bgcolor: '#f9fafb',
        paper_bgcolor: '#ffffff'
    };

    const config = {
        responsive: true,
        displayModeBar: false
    };

    Plotly.newPlot('price-chart', [trace], layout, config);
}

// Display 1-day intraday chart
async function displayIntradayChart(ticker) {
    try {
        // Fetch 1-day intraday data from our backend API
        const response = await fetch(`/api/intraday/${ticker}`);
        const data = await response.json();

        if (data.status === 'success' && data.timestamps && data.prices) {
            // Convert timestamps to datetime
            const datetimes = data.timestamps.map(ts => new Date(ts * 1000));

            // Calculate tight y-axis range to make movements more visible
            const minPrice = Math.min(...data.prices);
            const maxPrice = Math.max(...data.prices);
            const priceRange = maxPrice - minPrice;

            // Add 2% padding above and below to show movements clearly
            // For very flat days, ensure at least 0.5% range
            const padding = Math.max(priceRange * 0.02, maxPrice * 0.005);
            const yMin = minPrice - padding;
            const yMax = maxPrice + padding;

            const trace = {
                x: datetimes,
                y: data.prices,
                type: 'scatter',
                mode: 'lines',
                name: 'Price',
                line: {
                    color: '#10b981',
                    width: 2
                },
                fill: 'tozeroy',
                fillcolor: 'rgba(16, 185, 129, 0.1)'
            };

            const layout = {
                title: {
                    text: `${ticker} - Today's Price Action (${data.data_points} points)`,
                    font: { size: 14 }
                },
                xaxis: {
                    title: 'Time',
                    showgrid: true,
                    gridcolor: '#e5e7eb',
                    tickformat: '%H:%M'
                },
                yaxis: {
                    title: 'Price ($)',
                    showgrid: true,
                    gridcolor: '#e5e7eb',
                    range: [yMin, yMax],  // Set tight range to zoom into movements
                    fixedrange: false  // Allow user to zoom if needed
                },
                margin: { t: 40, r: 20, l: 50, b: 40 },
                hovermode: 'x unified',
                plot_bgcolor: '#f9fafb',
                paper_bgcolor: '#ffffff',
                height: 350
            };

            const config = {
                responsive: true,
                displayModeBar: false
            };

            Plotly.newPlot('intraday-chart', [trace], layout, config);
        } else {
            // If no intraday data, show message
            document.getElementById('intraday-chart').innerHTML =
                '<p style="padding: 100px 20px; text-align: center; color: #6b7280;">Intraday data not available (market may be closed)</p>';
        }
    } catch (error) {
        console.error('Failed to fetch intraday data:', error);
        document.getElementById('intraday-chart').innerHTML =
            '<p style="padding: 100px 20px; text-align: center; color: #6b7280;">Unable to load intraday chart</p>';
    }
}

// Note: addToPortfolio functionality is now handled by addToWatchlist() in auth.js

// Execute mock trade (with position sizing based on user's actual available cash)
async function executeMockTrade() {
    console.log('🎯 Execute Signal Trade button clicked');
    if (!currentPrediction) {
        alert('Please generate a prediction first');
        return;
    }

    const signal = currentPrediction.trading_signal;
    console.log('Trading signal:', signal);

    if (!signal.position) {
        alert('No position sizing available for this signal');
        return;
    }

    // Get user's actual available cash from mock portfolio
    const portfolio = JSON.parse(localStorage.getItem('mockPortfolio')) || { cash: 100000 };
    const availableCash = portfolio.cash || 100000;
    const entryPrice = currentPrediction.current_price;

    // Position sizing: use 50% of available cash (matching backend strategy)
    const MAX_POSITION_PCT = 0.50;
    const maxPositionValue = availableCash * MAX_POSITION_PCT;

    // Calculate shares based on available cash
    let adjustedShares = Math.floor(maxPositionValue / entryPrice);
    let adjustedValue = adjustedShares * entryPrice;

    // If user has very little cash, warn them
    if (adjustedShares <= 0) {
        alert(`Insufficient cash to execute trade.\n\nAvailable Cash: $${availableCash.toFixed(2)}\nPrice per Share: $${entryPrice.toFixed(2)}\n\nConsider closing some positions or resetting your portfolio.`);
        return;
    }

    // Calculate position concentration
    const totalValue = portfolio.totalValue || availableCash;
    const concentrationPct = (adjustedValue / totalValue * 100).toFixed(1);

    console.log('💰 Position sizing adjusted to user cash:', {
        availableCash: availableCash.toFixed(2),
        originalShares: signal.position.shares,
        adjustedShares: adjustedShares,
        adjustedValue: adjustedValue.toFixed(2),
        concentration: concentrationPct + '%'
    });

    if (!confirm(`Execute mock ${signal.action} trade?\n\nTicker: ${currentPrediction.ticker}\nShares: ${adjustedShares}\nEntry: $${entryPrice.toFixed(2)}\nTotal: $${adjustedValue.toFixed(2)}\n\nAvailable Cash: $${availableCash.toFixed(2)}\nPosition Size: ${concentrationPct}% of portfolio`)) {
        return;
    }

    console.log('✅ Trade confirmed, executing signal trade...');
    console.log('📋 Current prediction object:', currentPrediction);
    console.log('📋 Signal details:', {
        ticker: currentPrediction.ticker,
        action: signal.action,
        shares: adjustedShares,
        entry_price: entryPrice,
        expected_return: currentPrediction.prediction?.expected_return,
        confidence: signal.confidence
    });

    // Use guest mode function from auth.js
    await openMockTrade(
        currentPrediction.ticker,
        signal.action,
        adjustedShares,
        entryPrice,
        currentPrediction.prediction?.expected_return || 0,
        signal.confidence || currentPrediction.model_info?.confidence || 0.5,
        currentPrediction.company?.name  // Pass company name for display
    );
}

// Manual BUY trade (user decides, not signal)
async function manualBuy() {
    console.log('🛒 Manual BUY button clicked');
    if (!currentPrediction) {
        alert('Please generate a prediction first');
        return;
    }

    const shares = prompt(`How many shares of ${currentPrediction.ticker} do you want to BUY?\n\nCurrent price: $${currentPrediction.current_price.toFixed(2)}`);

    if (!shares || isNaN(shares) || shares <= 0) {
        return;
    }

    const totalCost = shares * currentPrediction.current_price;

    if (!confirm(`Confirm manual BUY trade?\n\nTicker: ${currentPrediction.ticker}\nShares: ${shares}\nPrice: $${currentPrediction.current_price.toFixed(2)}\nTotal Cost: $${totalCost.toFixed(2)}`)) {
        return;
    }

    console.log('✅ Trade confirmed, executing...');
    console.log('📋 Current prediction object:', currentPrediction);
    console.log('📋 Prediction details:', {
        ticker: currentPrediction.ticker,
        current_price: currentPrediction.current_price,
        expected_return: currentPrediction.prediction?.expected_return,
        confidence: currentPrediction.model_info?.confidence
    });

    // Use guest mode function from auth.js
    await openMockTrade(
        currentPrediction.ticker,
        'BUY',
        parseInt(shares),
        currentPrediction.current_price,
        currentPrediction.prediction?.expected_return || 0,
        currentPrediction.model_info?.confidence || 0.5,
        currentPrediction.company?.name  // Pass company name for display
    );
}

// Manual SELL trade (user decides, not signal)
async function manualSell() {
    console.log('📉 Manual SELL button clicked');
    if (!currentPrediction) {
        alert('Please generate a prediction first');
        return;
    }

    const shares = prompt(`How many shares of ${currentPrediction.ticker} do you want to SELL (short)?\n\nCurrent price: $${currentPrediction.current_price.toFixed(2)}`);

    if (!shares || isNaN(shares) || shares <= 0) {
        return;
    }

    const totalValue = shares * currentPrediction.current_price;

    if (!confirm(`Confirm manual SELL (short) trade?\n\nTicker: ${currentPrediction.ticker}\nShares: ${shares}\nPrice: $${currentPrediction.current_price.toFixed(2)}\nTotal Value: $${totalValue.toFixed(2)}`)) {
        return;
    }

    console.log('✅ Trade confirmed, executing...');
    console.log('📋 Current prediction object:', currentPrediction);
    console.log('📋 Prediction details:', {
        ticker: currentPrediction.ticker,
        current_price: currentPrediction.current_price,
        expected_return: currentPrediction.prediction?.expected_return,
        confidence: currentPrediction.model_info?.confidence
    });

    // Use guest mode function from auth.js
    await openMockTrade(
        currentPrediction.ticker,
        'SELL',
        parseInt(shares),
        currentPrediction.current_price,
        currentPrediction.prediction?.expected_return || 0,
        currentPrediction.model_info?.confidence || 0.5,
        currentPrediction.company?.name  // Pass company name for display
    );
}

// Note: loadPortfolio, loadWatchlist, and all display functions are now defined in auth.js
// They handle both authenticated and guest modes automatically

// Clear all watchlist items
async function clearWatchlist() {
    if (!confirm('Are you sure you want to clear ALL items from your watchlist?\n\nThis action cannot be undone.')) {
        return;
    }

    try {
        // Clear local watchlist (guest mode)
        if (typeof clearLocalWatchlist === 'function') {
            clearLocalWatchlist();
            console.log('✅ Local watchlist cleared');
        }

        // Also clear server watchlist if authenticated
        const response = await fetch('/api/watchlist/clear-all', {
            method: 'DELETE'
        });

        if (response.ok) {
            console.log('✅ Server watchlist cleared');
        }

        // Refresh the display
        loadWatchlist();
        alert('Watchlist cleared successfully!');
    } catch (error) {
        console.error('Error clearing watchlist:', error);
        alert('Failed to clear watchlist. Please try again.');
    }
}

// Close all mock trading positions - function defined in auth.js
// The auth.js version handles both authenticated and guest (localStorage) modes

// Show error
function showError(message) {
    console.error('[ERROR] Showing error:', message);
    document.getElementById('error-text').textContent = message;
    document.getElementById('error').classList.remove('hidden');
}

// Load top picks
async function loadTopPicks(regime = 'all') {
    const loadingEl = document.getElementById('top-picks-loading');
    const buyList = document.getElementById('top-buy-list');
    const sellList = document.getElementById('top-sell-list');

    // Show loading
    loadingEl.classList.remove('hidden');
    buyList.innerHTML = '';
    sellList.innerHTML = '';

    try {
        const response = await fetch(`/api/top-picks?regime=${regime}`);
        const data = await response.json();

        if (data.status === 'success') {
            displayTopPicks(data.top_buys, data.top_sells);
        } else {
            console.error('Failed to load top picks');
        }
    } catch (error) {
        console.error('Error loading top picks:', error);
    } finally {
        loadingEl.classList.add('hidden');
    }
}

// Refresh top picks (force clear cache and reload)
async function refreshTopPicks() {
    const btn = document.getElementById('refresh-top-picks-btn');
    const loadingEl = document.getElementById('top-picks-loading');
    const buyList = document.getElementById('top-buy-list');
    const sellList = document.getElementById('top-sell-list');

    // Disable button and show loading state
    btn.disabled = true;
    btn.innerHTML = 'Refreshing...';
    loadingEl.classList.remove('hidden');
    buyList.innerHTML = '';
    sellList.innerHTML = '';

    try {
        const response = await fetch(`/api/top-picks?regime=${currentRegime}&force_refresh=true`);
        const data = await response.json();

        if (data.status === 'success') {
            displayTopPicks(data.top_buys, data.top_sells);
            console.log('Top picks refreshed successfully (cache cleared)');
        } else {
            console.error('Failed to refresh top picks');
        }
    } catch (error) {
        console.error('Error refreshing top picks:', error);
    } finally {
        loadingEl.classList.add('hidden');
        btn.disabled = false;
        btn.innerHTML = 'Refresh';
    }
}

// Display top picks
function displayTopPicks(buyList, sellList) {
    const buyContainer = document.getElementById('top-buy-list');
    const sellContainer = document.getElementById('top-sell-list');

    // Store current top picks globally for news feed
    currentTopPicks = { buy: buyList, sell: sellList };

    // Helper function to check if ticker is a China stock
    const isChinaStock = (ticker) => ticker && (ticker.endsWith('.HK') || ticker.endsWith('.SS') || ticker.endsWith('.SZ'));

    // Display buy signals
    if (buyList.length === 0) {
        buyContainer.innerHTML = '<p class="text-muted">No BUY signals found</p>';
    } else {
        buyContainer.innerHTML = buyList.map((item, index) => {
            const isChina = isChinaStock(item.ticker);
            const displayName = item.name && item.name !== item.ticker ? item.name : null;
            return `
            <div class="pick-item buy-pick clickable" onclick="analyzeTickerDirectly('${item.ticker}')" title="Click to view ML Analysis">
                <div class="pick-rank">${index + 1}</div>
                <div class="pick-info">
                    <div class="pick-info-left">
                        ${isChina && displayName ? `
                            <div class="pick-ticker">${displayName}</div>
                            <div class="pick-name">${item.ticker}</div>
                        ` : `
                            <div class="pick-ticker">${item.ticker}</div>
                            <div class="pick-name">${item.name}</div>
                        `}
                    </div>
                    <div class="pick-info-right">
                        <div class="pick-confidence">${(item.confidence * 100).toFixed(1)}%</div>
                        <div class="pick-volatility">Vol: ${(item.volatility * 100).toFixed(1)}%</div>
                    </div>
                </div>
            </div>
        `}).join('');
    }

    // Display sell signals
    if (sellList.length === 0) {
        sellContainer.innerHTML = '<p class="text-muted">No SELL signals found</p>';
    } else {
        sellContainer.innerHTML = sellList.map((item, index) => {
            const isChina = isChinaStock(item.ticker);
            const displayName = item.name && item.name !== item.ticker ? item.name : null;
            return `
            <div class="pick-item sell-pick clickable" onclick="analyzeTickerDirectly('${item.ticker}')" title="Click to view ML Analysis">
                <div class="pick-rank">${index + 1}</div>
                <div class="pick-info">
                    <div class="pick-info-left">
                        ${isChina && displayName ? `
                            <div class="pick-ticker">${displayName}</div>
                            <div class="pick-name">${item.ticker}</div>
                        ` : `
                            <div class="pick-ticker">${item.ticker}</div>
                            <div class="pick-name">${item.name}</div>
                        `}
                    </div>
                    <div class="pick-info-right">
                        <div class="pick-confidence">${(item.confidence * 100).toFixed(1)}%</div>
                        <div class="pick-volatility">Vol: ${(item.volatility * 100).toFixed(1)}%</div>
                    </div>
                </div>
            </div>
        `}).join('');
    }

    // Refresh news feed with new top picks
    loadEntrancePageNews();
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    updateMonitoringStats();  // Load monitoring stats on page load
    loadPortfolio();  // Load portfolio on page load
    loadTopPicks('Stock');  // Load Stock tab by default on page load

    // Load entrance page news on page load
    setTimeout(() => {
        loadEntrancePageNews();
    }, 2000); // Wait 2 seconds for top picks to load first

    // Update monitoring stats every 30 seconds
    setInterval(updateMonitoringStats, 30000);

    // Update entrance page news every 1 minute (60000ms)
    setInterval(loadEntrancePageNews, 60000);

    const tickerInput = document.getElementById('ticker-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const refreshWatchlistBtn = document.getElementById('refresh-watchlist-btn');
    const refreshTradingBtn = document.getElementById('refresh-trading-btn');

    // Search on input
    tickerInput.addEventListener('input', (e) => {
        const query = e.target.value.trim();

        clearTimeout(searchTimeout);

        if (query.length >= 1) {
            searchTimeout = setTimeout(() => {
                searchTickers(query);
            }, 300);
        } else {
            document.getElementById('search-results').classList.add('hidden');
        }
    });

    // Close search results on click outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-input-wrapper')) {
            document.getElementById('search-results').classList.add('hidden');
        }
    });

    // Enter key to analyze
    tickerInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            getPrediction();
        }
    });

    // Analyze button
    analyzeBtn.addEventListener('click', getPrediction);

    // Refresh buttons (check if they exist first)
    if (refreshWatchlistBtn) refreshWatchlistBtn.addEventListener('click', loadPortfolio);
    if (refreshTradingBtn) refreshTradingBtn.addEventListener('click', loadPortfolio);

    // Manual trading buttons
    const manualBuyBtn = document.getElementById('manual-buy-btn');
    const manualSellBtn = document.getElementById('manual-sell-btn');
    if (manualBuyBtn) manualBuyBtn.addEventListener('click', manualBuy);
    if (manualSellBtn) manualSellBtn.addEventListener('click', manualSell);

    // Note: Add to portfolio button listener is attached dynamically in displayPrediction()

    // Regime toggle buttons
    const regimeBtns = document.querySelectorAll('.regime-btn');
    regimeBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            // Update active state
            regimeBtns.forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');

            // Load top picks for selected regime
            const regime = e.target.dataset.regime;
            currentRegime = regime; // Track current regime
            loadTopPicks(regime);
            // News feed will auto-refresh when displayTopPicks() is called
        });
    });
});

// ============================================================================
// 3RD LAYER: REAL-TIME TEST & BULL/BEAR PREDICTION FUNCTIONS
// ============================================================================

/**
 * Run Real-Time Yahoo Finance analysis on current top picks
 * Opens a 3rd layer modal with detailed P/L projections
 */
async function runRealtimeTest() {
    const regime = currentRegime;

    // Handle "all" regime - not supported, need to select specific tab
    if (regime === 'all') {
        alert('Please select a specific asset tab (Stock, Crypto, Commodity, Forex, or China) to run Real-Time Test.');
        return;
    }

    const btn = document.getElementById('realtime-test-btn');
    const modal = document.getElementById('realtime-test-modal');
    const resultsContainer = document.getElementById('realtime-test-results');
    const regimeBadge = document.getElementById('realtime-regime-badge');

    // Update badge and show loading
    const displayRegime = regime === 'China' ? 'China (DeepSeek + ML)' : regime;
    regimeBadge.textContent = displayRegime;
    const chinaTip = regime === 'China'
        ? '<p style="font-size: 12px; color: #B8860B; margin-top: 10px;"><strong>Note:</strong> China analysis uses DeepSeek API + ML model for each stock. This may take 3-5 minutes.</p>'
        : '';
    resultsContainer.innerHTML = `
        <div class="modal-loading">
            <div class="spinner-large"></div>
            <p>Fetching real-time data from Yahoo Finance for ${displayRegime}...</p>
            <p style="font-size: 12px; opacity: 0.7; margin-top: 10px;">Running full ML model predictions...</p>
            ${chinaTip}
        </div>
    `;

    // Show modal and disable button
    modal.classList.remove('hidden');
    btn.disabled = true;
    btn.innerHTML = '⏳ Testing...';

    try {
        // Use different endpoint for China (DeepSeek + China ML model)
        const endpoint = regime === 'China'
            ? '/api/top-picks/china/realtime-test'
            : `/api/top-picks/realtime-test?regime=${regime}`;

        // China takes longer due to DeepSeek API + China ML model (20 min timeout)
        // Other regimes use standard timeout (5 min)
        const timeoutMs = regime === 'China' ? 1200000 : 300000;
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

        const response = await fetch(endpoint, { signal: controller.signal });
        clearTimeout(timeoutId);
        const data = await response.json();

        // Check for error responses (including 404, 503, etc.)
        if (!response.ok || data.error) {
            const errorMsg = data.error || 'Unknown error';
            const helpMsg = data.message || 'Please try again.';
            resultsContainer.innerHTML = `
                <div class="modal-error">
                    <h4>⚠️ ${errorMsg}</h4>
                    <p>${helpMsg}</p>
                    ${regime === 'China' ? '<p style="color:#B8860B; margin-top:10px;"><strong>Tip:</strong> Please wait for China stocks to fully load in the table below, then try again.</p>' : ''}
                </div>
            `;
            return;
        }

        // Display the results
        displayRealtimeResults(data);

    } catch (error) {
        console.error('Real-time test failed:', error);
        const isTimeout = error.name === 'AbortError';
        const errorTitle = isTimeout ? 'Request Timeout' : 'Connection Error';
        const errorMsg = isTimeout
            ? 'The analysis is taking longer than expected. The server is still processing - please try again in a few minutes.'
            : 'Failed to fetch real-time data. Please try again.';
        resultsContainer.innerHTML = `
            <div class="modal-error">
                <h4>⚠️ ${errorTitle}</h4>
                <p>${errorMsg}</p>
                ${regime === 'China' ? '<p style="color:#B8860B; margin-top:10px;"><strong>Tip:</strong> Please wait for China stocks to fully load in the table below, then try again.</p>' : ''}
            </div>
        `;
    } finally {
        btn.disabled = false;
        btn.innerHTML = '📊 Real-Time Test';
    }
}

/**
 * Display real-time analysis results in the modal
 */
function displayRealtimeResults(data) {
    const container = document.getElementById('realtime-test-results');
    const buyAnalysis = data.buy_analysis || [];
    const sellAnalysis = data.sell_analysis || [];
    const portfolio = data.portfolio_simulation || {};
    const summary = data.summary || {};

    let html = `
        <!-- Summary Section -->
        <div class="realtime-summary">
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-value positive">+${portfolio.return_5d || 0}%</div>
                    <div class="summary-label">5-Day Projected Return</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value positive">+${portfolio.return_10d || 0}%</div>
                    <div class="summary-label">10-Day Projected Return</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value">${summary.strong_technical_setups || 0}</div>
                    <div class="summary-label">Strong Setups</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value">${buyAnalysis.length}</div>
                    <div class="summary-label">BUY Signals</div>
                </div>
            </div>
        </div>

        <!-- Portfolio Simulation -->
        <div class="realtime-section">
            <h4>💰 Portfolio Simulation ($10,000 Capital)</h4>
            <div class="portfolio-sim-grid">
                <div class="sim-stat">
                    <span class="sim-label">Capital:</span>
                    <span class="sim-value">$${(portfolio.capital || 10000).toLocaleString()}</span>
                </div>
                <div class="sim-stat">
                    <span class="sim-label">5-Day Value:</span>
                    <span class="sim-value positive">$${(portfolio.total_value_5d || 10000).toLocaleString()}</span>
                </div>
                <div class="sim-stat">
                    <span class="sim-label">10-Day Value:</span>
                    <span class="sim-value positive">$${(portfolio.total_value_10d || 10000).toLocaleString()}</span>
                </div>
                <div class="sim-stat">
                    <span class="sim-label">Positions:</span>
                    <span class="sim-value">${portfolio.actual_positions || 0} / ${portfolio.max_positions || 5}</span>
                </div>
            </div>
        </div>

        <!-- BUY Analysis Table -->
        <div class="realtime-section">
            <h4>📈 BUY Signals Analysis (${buyAnalysis.length} stocks)</h4>
            <div class="table-scroll">
                <table class="realtime-table">
                    <thead>
                        <tr>
                            <th>Ticker</th>
                            <th>Price</th>
                            <th>1D%</th>
                            <th>5D%</th>
                            <th>RSI</th>
                            <th>Status</th>
                            <th>5D Target</th>
                            <th>Gain%</th>
                            <th>R:R</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${buyAnalysis.map(r => `
                            <tr class="${r.technical_status.includes('STRONG') ? 'strong-signal' : ''}">
                                <td><strong>${r.ticker}</strong></td>
                                <td>$${r.current_price.toFixed(2)}</td>
                                <td class="${r.change_1d >= 0 ? 'positive' : 'negative'}">${r.change_1d >= 0 ? '+' : ''}${r.change_1d.toFixed(2)}%</td>
                                <td class="${r.change_5d >= 0 ? 'positive' : 'negative'}">${r.change_5d >= 0 ? '+' : ''}${r.change_5d.toFixed(2)}%</td>
                                <td>${r.rsi.toFixed(1)}</td>
                                <td><span class="status-badge ${r.technical_status.toLowerCase().replace(' ', '-')}">${r.technical_status}</span></td>
                                <td>$${r.target_5d.toFixed(2)}</td>
                                <td class="positive">+${r.potential_gain_5d.toFixed(2)}%</td>
                                <td>${r.risk_reward.toFixed(2)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        </div>
    `;

    // Add SELL analysis if available
    if (sellAnalysis.length > 0) {
        html += `
            <div class="realtime-section">
                <h4>📉 SELL Signals Analysis (${sellAnalysis.length} stocks)</h4>
                <div class="table-scroll">
                    <table class="realtime-table">
                        <thead>
                            <tr>
                                <th>Ticker</th>
                                <th>Price</th>
                                <th>1D%</th>
                                <th>RSI</th>
                                <th>Status</th>
                                <th>5D Target</th>
                                <th>Short Gain%</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${sellAnalysis.map(r => `
                                <tr>
                                    <td><strong>${r.ticker}</strong></td>
                                    <td>$${r.current_price.toFixed(2)}</td>
                                    <td class="${r.change_1d >= 0 ? 'positive' : 'negative'}">${r.change_1d >= 0 ? '+' : ''}${r.change_1d.toFixed(2)}%</td>
                                    <td>${r.rsi.toFixed(1)}</td>
                                    <td><span class="status-badge sell">${r.technical_status}</span></td>
                                    <td>$${r.target_5d.toFixed(2)}</td>
                                    <td class="positive">+${Math.abs(r.potential_gain_5d).toFixed(2)}%</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }

    // Top Picks Summary
    if (summary.top_picks && summary.top_picks.length > 0) {
        html += `
            <div class="realtime-section">
                <h4>🎯 Top Picks (Highest Expected Value)</h4>
                <div class="top-picks-summary">
                    ${summary.top_picks.map((p, i) => `
                        <div class="top-pick-card">
                            <div class="top-pick-rank">#${i + 1}</div>
                            <div class="top-pick-info">
                                <div class="top-pick-ticker">${p.ticker}</div>
                                <div class="top-pick-conf">Confidence: ${p.confidence}%</div>
                                <div class="top-pick-ev">Expected Value: ${p.expected_value > 0 ? '+' : ''}${p.expected_value}%</div>
                            </div>
                            <div class="top-pick-status ${p.status.toLowerCase().replace(' ', '-')}">${p.status}</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    container.innerHTML = html;
}

/**
 * Predict Bull/Bear market trend for current regime
 */
async function predictBullBear() {
    const regime = currentRegime;

    // Handle "all" regime - not supported, need to select specific tab
    if (regime === 'all') {
        alert('Please select a specific asset tab (Stock, Crypto, Commodity, Forex, or China) for Bull/Bear Prediction.');
        return;
    }

    const btn = document.getElementById('predict-trend-btn');
    const modal = document.getElementById('trend-prediction-modal');
    const resultsContainer = document.getElementById('trend-prediction-results');
    const regimeBadge = document.getElementById('trend-regime-badge');

    // Update badge and show loading
    const displayRegime = regime === 'China' ? 'China (DeepSeek + ML)' : regime;
    regimeBadge.textContent = displayRegime;
    resultsContainer.innerHTML = `
        <div class="modal-loading">
            <div class="spinner-large"></div>
            <p>Analyzing market conditions for ${displayRegime}...</p>
            <p style="font-size: 12px; opacity: 0.7; margin-top: 10px;">Running ML regime detection...</p>
        </div>
    `;

    // Show modal and disable button
    modal.classList.remove('hidden');
    btn.disabled = true;
    btn.innerHTML = '⏳ Predicting...';

    try {
        // Use different endpoint for China (DeepSeek + China ML model)
        const endpoint = regime === 'China'
            ? '/api/top-picks/china/predict-trend'
            : `/api/top-picks/predict-trend?regime=${regime}`;
        const response = await fetch(endpoint);
        const data = await response.json();

        // Check for error responses (including 404, 503, etc.)
        if (!response.ok || data.error) {
            const errorMsg = data.error || 'Unknown error';
            const helpMsg = data.message || 'Please try again.';
            resultsContainer.innerHTML = `
                <div class="modal-error">
                    <h4>⚠️ ${errorMsg}</h4>
                    <p>${helpMsg}</p>
                    ${regime === 'China' ? '<p style="color:#B8860B; margin-top:10px;"><strong>Tip:</strong> Please wait for China stocks to fully load in the table below, then try again.</p>' : ''}
                </div>
            `;
            return;
        }

        // Display the prediction
        displayTrendPrediction(data);

    } catch (error) {
        console.error('Trend prediction failed:', error);
        resultsContainer.innerHTML = `
            <div class="modal-error">
                <h4>⚠️ Connection Error</h4>
                <p>Failed to analyze market conditions. Please try again.</p>
                ${regime === 'China' ? '<p style="color:#B8860B; margin-top:10px;"><strong>Tip:</strong> Please wait for China stocks to fully load in the table below, then try again.</p>' : ''}
            </div>
        `;
    } finally {
        btn.disabled = false;
        btn.innerHTML = '🎯 Bull/Bear';
    }
}

/**
 * Display trend prediction results
 */
function displayTrendPrediction(data) {
    const container = document.getElementById('trend-prediction-results');
    const trend = data.trend_analysis || {};
    const signals = data.actionable_signals || {};
    const allocation = data.portfolio_allocation || {};
    const profitActions = data.profit_maximizing_actions || [];

    // Color mapping for trends
    const trendColors = {
        'STRONG_BULL': '#00c853',
        'BULL': '#4caf50',
        'NEUTRAL': '#ff9800',
        'BEAR': '#f44336',
        'STRONG_BEAR': '#b71c1c'
    };

    const trendEmojis = {
        'STRONG_BULL': '🐂',
        'BULL': '📈',
        'NEUTRAL': '➡️',
        'BEAR': '📉',
        'STRONG_BEAR': '🐻'
    };

    const trendColor = trendColors[trend.trend] || '#666';
    const trendEmoji = trendEmojis[trend.trend] || '❓';

    let html = `
        <!-- Main Trend Header -->
        <div class="trend-header" style="background: linear-gradient(135deg, ${trendColor}, ${trendColor}dd)">
            <div class="trend-emoji">${trendEmoji}</div>
            <div class="trend-title">${trend.trend_label || 'Unknown'}</div>
            <div class="trend-score">
                <span class="score-value">${trend.bull_score || 50}</span>
                <span class="score-label">/ 100</span>
            </div>
        </div>

        <!-- Recommendation -->
        <div class="trend-recommendation">
            <p>${trend.recommendation || 'No recommendation available'}</p>
        </div>

        <!-- Analysis Factors -->
        <div class="trend-section">
            <h4>📊 Analysis Factors</h4>
            <div class="factors-grid">
                ${(trend.factors || []).map(f => `
                    <div class="factor-card">
                        <div class="factor-name">${f.name}</div>
                        <div class="factor-value">${f.value}</div>
                        <div class="factor-score-bar">
                            <div class="factor-score-fill" style="width: ${(f.score / 25) * 100}%"></div>
                        </div>
                        <div class="factor-score">${f.score.toFixed(1)} / 25</div>
                        <div class="factor-interp ${f.interpretation.toLowerCase()}">${f.interpretation}</div>
                    </div>
                `).join('')}
            </div>
        </div>

        <!-- Signal Counts -->
        <div class="trend-section">
            <h4>📈 Signal Summary</h4>
            <div class="signal-summary-grid">
                <div class="signal-stat">
                    <div class="signal-stat-value positive">${trend.signal_counts?.buy_signals || 0}</div>
                    <div class="signal-stat-label">BUY Signals</div>
                </div>
                <div class="signal-stat">
                    <div class="signal-stat-value negative">${trend.signal_counts?.sell_signals || 0}</div>
                    <div class="signal-stat-label">SELL Signals</div>
                </div>
                <div class="signal-stat">
                    <div class="signal-stat-value">${((trend.signal_counts?.buy_ratio || 0.5) * 100).toFixed(0)}%</div>
                    <div class="signal-stat-label">Buy Ratio</div>
                </div>
                <div class="signal-stat risk-${signals.risk_level?.toLowerCase() || 'moderate'}">
                    <div class="signal-stat-value">${signals.risk_level || 'MODERATE'}</div>
                    <div class="signal-stat-label">Risk Level</div>
                </div>
            </div>
        </div>

        <!-- Trading Strategy -->
        <div class="trend-section">
            <h4>🎯 Recommended Strategy</h4>
            <div class="strategy-box">
                <ul class="strategy-actions">
                    ${(signals.recommended_actions || []).map(action => `
                        <li>${action}</li>
                    `).join('')}
                </ul>
            </div>
        </div>

        <!-- Profit Maximizing Actions -->
        <div class="trend-section">
            <h4>💰 Profit-Maximizing Actions</h4>
            <div class="profit-actions">
                ${profitActions.map(action => `
                    <div class="profit-action-item">${action}</div>
                `).join('')}
            </div>
        </div>

        <!-- Portfolio Allocation -->
        <div class="trend-section">
            <h4>📊 Recommended Allocation</h4>
            <div class="allocation-grid">
                ${Object.entries(allocation.recommended_allocation || {}).map(([asset, pct]) => `
                    <div class="allocation-bar">
                        <div class="allocation-label">${asset.charAt(0).toUpperCase() + asset.slice(1)}</div>
                        <div class="allocation-bar-bg">
                            <div class="allocation-bar-fill" style="width: ${pct}%"></div>
                        </div>
                        <div class="allocation-pct">${pct}%</div>
                    </div>
                `).join('')}
            </div>
            <div class="rebalance-urgency urgency-${(allocation.rebalance_urgency || 'low').toLowerCase()}">
                Rebalance Urgency: <strong>${allocation.rebalance_urgency || 'LOW'}</strong>
            </div>
        </div>
    `;

    container.innerHTML = html;
}

/**
 * Close the real-time test modal
 */
function closeRealtimeModal() {
    document.getElementById('realtime-test-modal').classList.add('hidden');
}

/**
 * Close the trend prediction modal
 */
function closeTrendModal() {
    document.getElementById('trend-prediction-modal').classList.add('hidden');
}

// Close modals when clicking outside
document.addEventListener('click', (e) => {
    const realtimeModal = document.getElementById('realtime-test-modal');
    const trendModal = document.getElementById('trend-prediction-modal');

    if (e.target === realtimeModal) {
        closeRealtimeModal();
    }
    if (e.target === trendModal) {
        closeTrendModal();
    }
});

// Close modals with Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeRealtimeModal();
        closeTrendModal();
    }
});
