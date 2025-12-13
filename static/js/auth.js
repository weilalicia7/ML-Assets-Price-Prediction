/**
 * Authentication and User Management JavaScript
 * Supports both guest mode (localStorage) and authenticated mode (database)
 */

// ============================================
// PORTFOLIO CONFIGURATION CONSTANTS
// ============================================
const PORTFOLIO_CONFIG = {
    STARTING_CASH: 100000,                    // Initial portfolio value
    CASH_DISCREPANCY_THRESHOLD: 100,          // Auto-fix if cash is off by more than $100
    SEVERE_CORRUPTION_MIN_VALUE: 1000,        // Reset if total value falls below $1,000
    SEVERE_CORRUPTION_MAX_VALUE: 10000000,    // Reset if total value exceeds $10M
    SEVERE_NEGATIVE_CASH: -50000,             // Reset if cash is below -$50,000
    CONCENTRATION_WARNING_THRESHOLD: 15,      // Warn if position > 15% of portfolio
    CONCENTRATION_MAX: 100,                   // Cap concentration display at 100%
    HEALTH_CHECK_THRESHOLD: 0.8,              // Run diagnostics if health < 80%
    MAX_RELOAD_ATTEMPTS: 2                    // Maximum reloads before stopping (prevents infinite loop)
};

// ============================================
// RELOAD LOOP PROTECTION
// ============================================
// Uses sessionStorage to track reload attempts (clears when browser closes)
const RELOAD_KEY = 'portfolioReloadCount';
const RELOAD_TIMESTAMP_KEY = 'portfolioReloadTimestamp';

function getReloadCount() {
    const count = parseInt(sessionStorage.getItem(RELOAD_KEY) || '0', 10);
    const timestamp = parseInt(sessionStorage.getItem(RELOAD_TIMESTAMP_KEY) || '0', 10);
    const now = Date.now();

    // Reset counter if last reload was more than 30 seconds ago
    if (now - timestamp > 30000) {
        sessionStorage.setItem(RELOAD_KEY, '0');
        return 0;
    }
    return count;
}

function incrementReloadCount() {
    const count = getReloadCount() + 1;
    sessionStorage.setItem(RELOAD_KEY, count.toString());
    sessionStorage.setItem(RELOAD_TIMESTAMP_KEY, Date.now().toString());
    return count;
}

function resetReloadCount() {
    sessionStorage.removeItem(RELOAD_KEY);
    sessionStorage.removeItem(RELOAD_TIMESTAMP_KEY);
}

function canReload() {
    return getReloadCount() < PORTFOLIO_CONFIG.MAX_RELOAD_ATTEMPTS;
}

// ============================================
// EARLY CORRUPTION CHECK (runs immediately on script load)
// ============================================
// This runs BEFORE DOMContentLoaded to prevent reload loops
(function earlyCorruptionCheck() {
    const reloadCount = getReloadCount();
    console.log(`[EARLY CHECK] Reload count: ${reloadCount}/${PORTFOLIO_CONFIG.MAX_RELOAD_ATTEMPTS}`);

    if (reloadCount >= PORTFOLIO_CONFIG.MAX_RELOAD_ATTEMPTS) {
        console.warn('[EARLY CHECK] Reload loop detected! Clearing corrupted portfolio data...');

        // Force clear the corrupted portfolio
        const cleanPortfolio = {
            cash: PORTFOLIO_CONFIG.STARTING_CASH,
            totalValue: PORTFOLIO_CONFIG.STARTING_CASH,
            peakValue: PORTFOLIO_CONFIG.STARTING_CASH,
            trades: [],
            closedTrades: [],
            lastUpdated: new Date().toISOString()
        };
        localStorage.setItem('mockPortfolio', JSON.stringify(cleanPortfolio));

        // Reset the reload counter
        resetReloadCount();

        console.log('[EARLY CHECK] Portfolio reset to clean state. Reload loop broken.');
    }
})();

// China stock ticker to company name lookup (for legacy trades without name field)
const CHINA_TICKER_NAMES = {
    // Hong Kong stocks
    '0700.HK': 'Tencent Holdings',
    '9988.HK': 'Alibaba Group',
    '3690.HK': 'Meituan',
    '2318.HK': 'Ping An Insurance',
    '1299.HK': 'AIA Group',
    '0939.HK': 'China Construction Bank',
    '0941.HK': 'China Mobile',
    '1398.HK': 'ICBC',
    '3988.HK': 'Bank of China',
    '0388.HK': 'Hong Kong Exchanges',
    '2382.HK': 'Sunny Optical',
    '0005.HK': 'HSBC Holdings',
    '0001.HK': 'CK Hutchison',
    '0016.HK': 'Sun Hung Kai Properties',
    '1810.HK': 'Xiaomi Corporation',
    '2269.HK': 'Wuxi Biologics',
    '1024.HK': 'Kuaishou Technology',
    '9618.HK': 'JD.com',
    '9888.HK': 'Baidu',
    '9868.HK': 'XPeng',
    '9866.HK': 'NIO',
    '2015.HK': 'Li Auto',
    '1211.HK': 'BYD Company',
    '0003.HK': 'Hong Kong and China Gas',
    '2628.HK': 'China Life Insurance',
    '1109.HK': 'China Resources Land',
    '0002.HK': 'CLP Holdings',
    '1093.HK': 'CSPC Pharmaceutical',
    '0027.HK': 'Galaxy Entertainment',
    // Shanghai stocks
    '600519.SS': 'Kweichow Moutai',
    '601318.SS': 'Ping An Insurance',
    '600036.SS': 'China Merchants Bank',
    '601398.SS': 'ICBC',
    '600028.SS': 'China Petroleum (Sinopec)',
    '601857.SS': 'PetroChina',
    '600000.SS': 'Shanghai Pudong Dev Bank',
    '601012.SS': 'Longi Green Energy',
    '600900.SS': 'China Yangtze Power',
    '601668.SS': 'China State Construction',
    '600887.SS': 'Inner Mongolia Yili',
    '601888.SS': 'China International Travel',
    '601166.SS': 'Industrial Bank',
    '600309.SS': 'Wanhua Chemical',
    '601288.SS': 'Agricultural Bank of China',
    // Shenzhen stocks
    '000858.SZ': 'Wuliangye Yibin',
    '000333.SZ': 'Midea Group',
    '002594.SZ': 'BYD Company',
    '000651.SZ': 'Gree Electric Appliances',
    '002475.SZ': 'Luxshare Precision',
    '300750.SZ': 'Contemporary Amperex (CATL)',
    '000002.SZ': 'China Vanke',
    '002415.SZ': 'Hangzhou Hikvision',
    '000725.SZ': 'BOE Technology',
    '002230.SZ': 'iFLYTEK',
    '000568.SZ': 'Luzhou Laojiao',
    '300059.SZ': 'East Money Information',
    '002920.SZ': 'Yutong Bus',
    '300015.SZ': 'Aier Eye Hospital',
    '000001.SZ': 'Ping An Bank'
};

// Helper function to get company name for China stocks
function getChinaStockName(ticker) {
    return CHINA_TICKER_NAMES[ticker] || null;
}

let isAuthenticated = false;
let currentUser = null;
let guestActionCount = 0;

// Toggle ML Signal panel details visibility
function toggleSignalDetails(index) {
    const panel = document.getElementById(`signal-details-${index}`);
    const arrow = panel?.parentElement?.querySelector('.signal-toggle-arrow');
    if (panel) {
        panel.classList.toggle('collapsed');
        if (arrow) {
            arrow.classList.toggle('collapsed');
        }
    }
}

// Store signal data for quick execution
let storedSignals = {};

// Execute signal trade from portfolio panel (similar to main ML page execute button)
async function executeSignalTrade(ticker) {
    console.log('Executing signal trade for:', ticker);

    const signalData = storedSignals[ticker];
    if (!signalData) {
        showNotification('No signal data available. Click Refresh first.', 'error');
        return;
    }

    const { signal, tradingSignal, currentPrice, confidence, expectedReturn } = signalData;

    // Only allow BUY/SELL signals
    if (signal === 'HOLD' || !tradingSignal.position) {
        showNotification('Cannot execute HOLD signal or no position data available.', 'warning');
        return;
    }

    const entryPrice = currentPrice;
    const action = signal; // 'BUY' or 'SELL'

    // Get user's actual available cash from mock portfolio
    const portfolio = JSON.parse(localStorage.getItem('mockPortfolio')) || { cash: 100000 };
    const availableCash = portfolio.cash || 100000;

    // Position sizing: use 50% of available cash (matching backend strategy)
    const MAX_POSITION_PCT = 0.50;
    const maxPositionValue = availableCash * MAX_POSITION_PCT;

    // Calculate shares based on available cash
    const adjustedShares = Math.floor(maxPositionValue / entryPrice);
    const adjustedValue = adjustedShares * entryPrice;

    // If user has very little cash, warn them
    if (adjustedShares <= 0) {
        showNotification(`Insufficient cash ($${availableCash.toFixed(2)}) to execute trade. Consider closing positions or resetting portfolio.`, 'error');
        return;
    }

    // Calculate position concentration
    const totalValue = portfolio.totalValue || availableCash;
    const concentrationPct = (adjustedValue / totalValue * 100).toFixed(1);

    console.log('💰 Position sizing adjusted to user cash:', {
        availableCash: availableCash.toFixed(2),
        originalShares: tradingSignal.position?.shares,
        adjustedShares: adjustedShares,
        adjustedValue: adjustedValue.toFixed(2),
        concentration: concentrationPct + '%'
    });

    // Confirm trade
    if (!confirm(`Execute ${action} trade?\n\nTicker: ${ticker}\nShares: ${adjustedShares}\nEntry: $${entryPrice.toFixed(2)}\nStop Loss: $${tradingSignal.stop_loss?.toFixed(2) || 'N/A'}\nTake Profit: $${tradingSignal.take_profit?.toFixed(2) || 'N/A'}\nTotal: $${adjustedValue.toFixed(2)}\n\nAvailable Cash: $${availableCash.toFixed(2)}\nPosition Size: ${concentrationPct}% of portfolio`)) {
        return;
    }

    console.log('Trade confirmed, executing with stop/profit levels...');

    // Execute the trade with stop loss and take profit
    await openMockTradeWithLimits(
        ticker,
        action,
        adjustedShares,
        entryPrice,
        expectedReturn,
        confidence,
        tradingSignal.stop_loss,
        tradingSignal.take_profit
    );
}

// Open mock trade with stop loss and take profit limits
async function openMockTradeWithLimits(ticker, action, quantity, entryPrice, predictedReturn, confidence, stopLoss, takeProfit) {
    console.log('openMockTradeWithLimits called:', {ticker, action, quantity, entryPrice, predictedReturn, confidence, stopLoss, takeProfit, isAuthenticated});

    if (isAuthenticated) {
        // Server mode - include stop/profit in request
        console.log('Using server mode (authenticated) with limits');
        try {
            const response = await fetch('/api/mock-trade/open', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    ticker,
                    action,
                    quantity,
                    entry_price: entryPrice,
                    predicted_return: predictedReturn,
                    prediction_confidence: confidence,
                    stop_loss: stopLoss,
                    take_profit: takeProfit
                })
            });

            const data = await response.json();
            if (data.success) {
                showNotification(data.message, 'success');
                await loadPortfolio();
            } else {
                showNotification(data.error || 'Failed to open trade', 'error');
            }
        } catch (error) {
            showNotification('Failed to open trade', 'error');
            console.error(error);
        }
    } else {
        // Local storage mode with limits
        console.log('Using local storage mode (guest) with limits');
        const portfolio = getLocalPortfolio();

        const tradeValue = quantity * entryPrice;
        if (action === 'BUY' && portfolio.cash < tradeValue) {
            showNotification('Insufficient cash!', 'error');
            return;
        }

        const trade = {
            ticker,
            action,
            quantity,
            entry_price: entryPrice,
            currentPrice: entryPrice,
            predicted_return: predictedReturn,
            prediction_confidence: confidence,
            stop_loss: stopLoss,
            take_profit: takeProfit,
            openedAt: new Date().toISOString()
        };

        console.log('Trade with limits created:', trade);

        if (action === 'BUY') {
            portfolio.cash -= tradeValue;
        }

        portfolio.trades.push(trade);
        saveLocalPortfolio(portfolio);
        displayLocalPortfolio();

        showNotification(`${action} ${quantity} shares of ${ticker} at $${entryPrice.toFixed(2)} | SL: $${stopLoss?.toFixed(2) || 'N/A'} | TP: $${takeProfit?.toFixed(2) || 'N/A'} (Guest Mode)`, 'success');
        guestActionCount++;
    }
}

// Increment guest action counter for progressive registration
function incrementActionCounter() {
    if (isAuthenticated) return; // Only for guest mode

    guestActionCount++;
    console.log(`📊 Guest action count: ${guestActionCount}`);

    // Show progressive registration prompts at 1st, 5th, and 10th actions
    if (guestActionCount === 1 || guestActionCount === 5 || guestActionCount === 10) {
        showProgressiveRegistrationPrompt(guestActionCount);
    }
}

// Show progressive registration prompt
function showProgressiveRegistrationPrompt(actionCount) {
    const messages = {
        1: "Great start! Sign in to save your data permanently and access it from any device.",
        5: "You're getting the hang of it! Sign in to ensure your watchlist and trades are never lost.",
        10: "Impressive activity! Sign in now to unlock full features and keep your portfolio safe."
    };

    showNotification(`${messages[actionCount]} Sign in to continue.`, 'info');
}

// Check authentication status on page load
document.addEventListener('DOMContentLoaded', async () => {
    await checkAuth();
    setupAuthUI();
    loadLocalData(); // Load guest data from localStorage
});

// Check if user is authenticated
async function checkAuth() {
    try {
        const response = await fetch('/api/auth/me');
        const data = await response.json();

        if (data.authenticated) {
            isAuthenticated = true;
            currentUser = data.user;
            showAuthenticatedUI(data.user);
            // Migrate local data to server if exists
            await migrateLocalDataToServer();
            // Load server data
            await loadWatchlist();
            await loadPortfolio();
        } else {
            isAuthenticated = false;
            showUnauthenticatedUI();
            // Load from localStorage for guests
            displayLocalWatchlist();
            displayLocalPortfolio();
        }
    } catch (error) {
        console.error('Auth check failed:', error);
        isAuthenticated = false;
        showUnauthenticatedUI();
        displayLocalWatchlist();
        displayLocalPortfolio();
    }
}

// Show UI for authenticated users
function showAuthenticatedUI(user) {
    const userInfo = document.getElementById('user-info');
    if (userInfo) {
        userInfo.innerHTML = `
            <div class="user-profile">
                ${user.avatar_url ? `<img src="${user.avatar_url}" class="user-avatar" style="width: 32px; height: 32px; border-radius: 50%; margin-right: 8px;">` : ''}
                <span class="user-name">${user.full_name || user.username}</span>
                <button onclick="logout()" class="btn-logout">Logout</button>
            </div>
        `;
    }
}

// Show UI for guest users
function showUnauthenticatedUI() {
    const userInfo = document.getElementById('user-info');
    if (userInfo) {
        userInfo.innerHTML = `
            <button onclick="showLoginModal()" class="btn-login">Sign In to Save Data</button>
        `;
    }
}

// Setup auth UI elements
function setupAuthUI() {
    // Create login modal if it doesn't exist
    if (!document.getElementById('login-modal')) {
        const modal = document.createElement('div');
        modal.id = 'login-modal';
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content auth-modal">
                <span class="close-modal" onclick="closeLoginModal()">&times;</span>
                <h2>Sign In to ML Stock Platform</h2>
                <p class="auth-subtitle">💾 Save your watchlist and track mock trading performance across devices!</p>

                <div class="auth-benefits">
                    <div class="benefit-item">✅ Persistent watchlist with price alerts</div>
                    <div class="benefit-item">✅ Track mock trading portfolio history</div>
                    <div class="benefit-item">✅ Get push notifications for stock alerts</div>
                    <div class="benefit-item">✅ Share your watchlist on social media</div>
                </div>

                <div class="oauth-buttons">
                    <a href="/login/google" class="btn-oauth btn-google">
                        <svg class="oauth-icon" viewBox="0 0 24 24">
                            <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                            <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                            <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                            <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                        </svg>
                        Continue with Google
                    </a>

                    <a href="/login/github" class="btn-oauth btn-github">
                        <svg class="oauth-icon" viewBox="0 0 24 24">
                            <path fill="#fff" d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                        </svg>
                        Continue with GitHub
                    </a>

                    <a href="/login/reddit" class="btn-oauth btn-reddit">
                        <svg class="oauth-icon" viewBox="0 0 24 24">
                            <circle fill="#FF4500" cx="12" cy="12" r="11"/>
                            <path fill="#fff" d="M19.5 11.5a1.5 1.5 0 00-3 0 7.5 7.5 0 00-9 0 1.5 1.5 0 10-3 0 1.5 1.5 0 00.89 1.37A3 3 0 005 14a3 3 0 003 3h8a3 3 0 003-3 3 3 0 00-.39-1.13 1.5 1.5 0 00.89-1.37z"/>
                        </svg>
                        Continue with Reddit
                    </a>
                </div>

                <div class="auth-footer">
                    <p>⚠️ <strong>Guest Mode:</strong> Your watchlist and trading data are stored locally and will be lost when you clear browser data.</p>
                    <p class="privacy-note">🔒 Your OAuth credentials are never stored. We only save your username and email.</p>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }
}

// Show/hide login modal
function showLoginModal() {
    document.getElementById('login-modal').style.display = 'block';
}

function closeLoginModal() {
    document.getElementById('login-modal').style.display = 'none';
}

// Logout function
async function logout() {
    try {
        const response = await fetch('/api/auth/logout', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'}
        });

        const data = await response.json();
        if (data.success) {
            window.location.reload();
        }
    } catch (error) {
        console.error('Logout failed:', error);
    }
}

// ========== Local Storage Functions (Guest Mode) ==========

function loadLocalData() {
    const watchlist = localStorage.getItem('guestWatchlist');
    const portfolio = localStorage.getItem('guestPortfolio');

    if (!watchlist) {
        localStorage.setItem('guestWatchlist', JSON.stringify([]));
    }
    if (!portfolio) {
        localStorage.setItem('guestPortfolio', JSON.stringify({
            cash: 100000,
            trades: [],
            closedTrades: [],
            totalReturn: 0,
            winRate: 0
        }));
    }

    // Display the data on the page
    if (!isAuthenticated) {
        displayLocalWatchlist();
        displayLocalPortfolio();
    }
}

function getLocalWatchlist() {
    return JSON.parse(localStorage.getItem('guestWatchlist') || '[]');
}

function saveLocalWatchlist(watchlist) {
    localStorage.setItem('guestWatchlist', JSON.stringify(watchlist));
    promptRegistrationIfNeeded();
}

function clearLocalWatchlist() {
    localStorage.setItem('guestWatchlist', JSON.stringify([]));
    console.log('✅ Local watchlist cleared from localStorage');
}

function getLocalPortfolio() {
    const portfolio = JSON.parse(localStorage.getItem('guestPortfolio') || JSON.stringify({
        cash: 100000,
        trades: [],
        closedTrades: [],
        totalReturn: 0,
        winRate: 0,
        dailySnapshot: null,
        lastSnapshotDate: null,
        peakValue: 100000,  // Track all-time high for drawdown calculation
        peakDate: null      // When peak was reached
    }));

    // CRITICAL FIX: Recalculate cash from positions to fix corrupted data
    // This fixes the bug where SHORT positions incorrectly added to cash
    const recalculatedCash = recalculateCashFromPositions(portfolio);
    if (Math.abs(portfolio.cash - recalculatedCash) > 0.01) {
        console.log(`⚠️ Cash correction: ${portfolio.cash.toFixed(2)} -> ${recalculatedCash.toFixed(2)}`);
        portfolio.cash = recalculatedCash;
        // Save the corrected portfolio
        localStorage.setItem('guestPortfolio', JSON.stringify(portfolio));
    }

    return portfolio;
}

/**
 * Backup portfolio data before major operations or when corruption is detected.
 * Stores backup in localStorage with timestamp for debugging purposes.
 * @param {Object} portfolio - The portfolio object to backup
 * @param {string} reason - The reason for the backup (e.g., 'pre-reset-corruption', 'pre-close-all')
 */
function backupPortfolio(portfolio, reason = 'manual') {
    try {
        const backup = {
            portfolio: JSON.parse(JSON.stringify(portfolio)),
            timestamp: new Date().toISOString(),
            reason: reason,
            version: '1.0'
        };

        // Keep last 5 backups
        let backups = JSON.parse(localStorage.getItem('portfolioBackups') || '[]');
        backups.unshift(backup);
        if (backups.length > 5) {
            backups = backups.slice(0, 5);
        }

        localStorage.setItem('portfolioBackups', JSON.stringify(backups));
        console.log(`Portfolio backed up (reason: ${reason})`);
    } catch (error) {
        console.error('Failed to backup portfolio:', error);
    }
}

/**
 * Restore portfolio from backup
 * @param {number} index - The backup index to restore (0 = most recent)
 * @returns {boolean} - True if restore was successful
 */
function restorePortfolioFromBackup(index = 0) {
    try {
        const backups = JSON.parse(localStorage.getItem('portfolioBackups') || '[]');
        if (backups.length === 0 || index >= backups.length) {
            console.error('No backup available at index:', index);
            return false;
        }

        const backup = backups[index];
        localStorage.setItem('mockPortfolio', JSON.stringify(backup.portfolio));
        console.log(`Portfolio restored from backup (${backup.timestamp}, reason: ${backup.reason})`);
        return true;
    } catch (error) {
        console.error('Failed to restore portfolio:', error);
        return false;
    }
}

/**
 * Calculate portfolio health score (0-1)
 * @param {Object} portfolio - The portfolio object to evaluate
 * @returns {number} - Health score between 0 and 1
 */
function calculatePortfolioHealth(portfolio) {
    let healthScore = 1.0;
    const issues = [];

    // Check if cash is reasonable
    if (portfolio.cash < 0) {
        healthScore -= 0.3;
        issues.push('Negative cash');
    }

    // Check if total value is reasonable
    const totalValue = portfolio.totalValue || portfolio.cash;
    if (totalValue <= 0) {
        healthScore -= 0.4;
        issues.push('Non-positive total value');
    } else if (totalValue < PORTFOLIO_CONFIG.SEVERE_CORRUPTION_MIN_VALUE) {
        healthScore -= 0.3;
        issues.push('Suspiciously low total value');
    }

    // Check position data integrity
    for (const trade of (portfolio.trades || [])) {
        if (!trade.entry_price || trade.entry_price <= 0) {
            healthScore -= 0.1;
            issues.push(`Invalid entry price for ${trade.ticker}`);
        }
        if (!trade.quantity || trade.quantity <= 0) {
            healthScore -= 0.1;
            issues.push(`Invalid quantity for ${trade.ticker}`);
        }
    }

    healthScore = Math.max(0, healthScore);

    if (issues.length > 0) {
        console.warn('Portfolio health issues:', issues);
    }

    return healthScore;
}

/**
 * Recalculate cash from positions using transaction-based accounting.
 *
 * CORRECT MODEL:
 * - LONG (BUY): Cash decreases by entry cost, position has current market value
 * - SHORT (SELL): Cash unchanged (proceeds are collateral), position has unrealized P&L
 *
 * Cash = Starting Cash - LONG entry costs + Realized P&L
 * (SHORT positions don't affect cash until closed)
 */
function recalculateCashFromPositions(portfolio) {
    const STARTING_CASH = 100000;

    console.log('🔧 Recalculating cash from positions...');

    // Calculate cash used for LONG positions (entry cost, not current value)
    let totalLongEntryCost = 0;

    for (const trade of (portfolio.trades || [])) {
        const entryPrice = trade.entry_price || 0;
        const quantity = trade.quantity || 0;
        const entryCost = entryPrice * quantity;

        if (trade.action === 'BUY' || trade.action === 'LONG') {
            totalLongEntryCost += entryCost;
            console.log(`  LONG ${trade.ticker}: entry cost $${entryCost.toFixed(2)}`);
        } else if (trade.action === 'SELL' || trade.action === 'SHORT') {
            // SHORT positions don't use cash - proceeds are held as collateral
            console.log(`  SHORT ${trade.ticker}: $0 cash impact (collateral model)`);
        }
    }

    console.log(`  Total LONG entry cost: $${totalLongEntryCost.toFixed(2)}`);

    // Calculate realized P&L from closed trades
    let realizedPnL = 0;
    for (const closedTrade of (portfolio.closedTrades || [])) {
        realizedPnL += closedTrade.pnl || 0;
    }
    console.log(`  Realized P&L from closed trades: $${realizedPnL.toFixed(2)}`);

    // CORRECT FORMULA: Cash = Starting Cash - LONG entry costs + Realized P&L
    // This is transaction-based: we track what actually went in/out of cash
    const cash = STARTING_CASH - totalLongEntryCost + realizedPnL;

    console.log(`📊 Recalculated cash: $${cash.toFixed(2)}`);
    console.log(`   (Starting $${STARTING_CASH} - LONG costs $${totalLongEntryCost.toFixed(2)} + Realized P&L $${realizedPnL.toFixed(2)})`);

    return cash;
}

/**
 * CRITICAL FIX: Validate portfolio data integrity
 * Cash can NEVER be greater than Total Value
 * Total Value = Cash + Unrealized P&L
 * If Cash > Total Value, portfolio is corrupted and needs fixing
 */
function validateAndFixPortfolio(portfolio) {
    const STARTING_CASH = 100000;

    // Calculate total unrealized P&L
    let totalUnrealizedPnL = 0;
    for (const trade of (portfolio.trades || [])) {
        const currentPrice = trade.currentPrice || trade.entry_price || 0;
        const entryPrice = trade.entry_price || 0;
        const quantity = trade.quantity || 0;

        if (trade.action === 'SELL' || trade.action === 'SHORT') {
            totalUnrealizedPnL += (entryPrice - currentPrice) * quantity;
        } else {
            totalUnrealizedPnL += (currentPrice - entryPrice) * quantity;
        }
    }

    // Calculate what total value should be
    const calculatedTotalValue = portfolio.cash + totalUnrealizedPnL;

    // CRITICAL CHECK: Cash cannot be greater than Total Value
    // This is mathematically impossible in correct accounting
    if (portfolio.cash > calculatedTotalValue) {
        console.error('🚨 PORTFOLIO CORRUPTION DETECTED: Cash > Total Value');
        console.error(`   Cash: $${portfolio.cash.toFixed(2)}`);
        console.error(`   Total Value: $${calculatedTotalValue.toFixed(2)}`);
        console.error(`   Unrealized P&L: $${totalUnrealizedPnL.toFixed(2)}`);

        // Recalculate cash from scratch based on positions
        const correctCash = recalculateCashFromPositions(portfolio);

        // If recalculated cash is still problematic, use conservative fix
        const newTotalValue = correctCash + totalUnrealizedPnL;
        if (correctCash > newTotalValue || correctCash < 0) {
            // Use conservative estimate: Total Value minus position values
            let totalPositionValue = 0;
            for (const trade of (portfolio.trades || [])) {
                const currentPrice = trade.currentPrice || trade.entry_price || 0;
                totalPositionValue += currentPrice * trade.quantity;
            }

            // Cash should be approximately: TotalValue - PositionValue
            // But we use a safer calculation
            const safeCash = Math.max(0, STARTING_CASH - totalPositionValue + totalUnrealizedPnL);

            console.log(`🔧 Conservative cash fix: $${safeCash.toFixed(2)}`);
            portfolio.cash = safeCash;
        } else {
            portfolio.cash = correctCash;
        }

        console.log(`✅ Cash corrected to: $${portfolio.cash.toFixed(2)}`);
        return true; // Portfolio was fixed
    }

    // Also check for negative cash (can happen with margin trading, but not in mock)
    if (portfolio.cash < 0) {
        console.error('🚨 PORTFOLIO CORRUPTION: Negative cash detected');
        portfolio.cash = recalculateCashFromPositions(portfolio);
        if (portfolio.cash < 0) {
            portfolio.cash = 0; // Emergency floor
        }
        return true;
    }

    return false; // Portfolio was already valid
}

function saveLocalPortfolio(portfolio) {
    const STARTING_CASH = 100000;

    // Calculate total unrealized P&L (correctly handles LONG and SHORT positions)
    const totalUnrealizedPnL = portfolio.trades.reduce((sum, t) => {
        const currentPrice = t.currentPrice || t.entry_price || 0;
        const entryPrice = t.entry_price || 0;
        const quantity = t.quantity || 0;

        let positionPnl;
        if (t.action === 'SELL' || t.action === 'SHORT') {
            // SHORT position: profit when price goes DOWN
            positionPnl = (entryPrice - currentPrice) * quantity;
        } else {
            // LONG/BUY position: profit when price goes UP
            positionPnl = (currentPrice - entryPrice) * quantity;
        }
        return sum + positionPnl;
    }, 0);

    const totalValue = portfolio.cash + totalUnrealizedPnL;

    // Update peak value tracking for drawdown calculation
    if (!portfolio.peakValue || totalValue > portfolio.peakValue) {
        portfolio.peakValue = totalValue;
        portfolio.peakDate = new Date().toISOString();
    }

    // Save daily snapshot if it's a new day
    const today = new Date().toDateString();
    if (portfolio.lastSnapshotDate !== today) {
        const closedTrades = portfolio.closedTrades || [];
        const winningTrades = closedTrades.filter(t => t.pnl > 0).length;
        const winRate = closedTrades.length > 0 ? (winningTrades / closedTrades.length) : 0;

        portfolio.dailySnapshot = {
            totalValue: totalValue,
            cash: portfolio.cash,
            totalReturn: totalValue - STARTING_CASH,
            totalReturnPercent: ((totalValue - STARTING_CASH) / STARTING_CASH) * 100,
            winRate: winRate,
            date: today
        };
        portfolio.lastSnapshotDate = today;
    }

    localStorage.setItem('guestPortfolio', JSON.stringify(portfolio));
    promptRegistrationIfNeeded();
}

// Prompt user to register after certain actions
function promptRegistrationIfNeeded() {
    if (isAuthenticated) return;

    guestActionCount++;

    // Show prompt after first action
    if (guestActionCount === 1) {
        showNotification('💡 Tip: Sign in to save your data permanently!', 'info', 5000);
    }

    // Show more urgent prompt after 5 actions
    if (guestActionCount === 5) {
        showNotification('⚠️ You have 5 items in guest mode. Sign in to prevent data loss!', 'warning', 7000);
    }

    // Show strong warning after 10 actions
    if (guestActionCount >= 10) {
        showRegistrationPrompt();
    }
}

function showRegistrationPrompt() {
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.style.display = 'block';
    modal.innerHTML = `
        <div class="modal-content auth-modal" style="max-width: 500px;">
            <h2>⚠️ Important: Your Data is Not Saved</h2>
            <p style="margin: 20px 0; font-size: 1.1em;">You have <strong>${guestActionCount}</strong> actions in guest mode.</p>
            <p style="color: #ef4444; margin-bottom: 20px;">
                <strong>All your watchlist and trading data will be permanently lost if you:</strong>
            </p>
            <ul style="text-align: left; margin: 0 0 20px 40px; color: #ef4444;">
                <li>Close this browser tab/window</li>
                <li>Clear browser data</li>
                <li>Use incognito/private mode</li>
            </ul>
            <button onclick="showLoginModal(); this.parentElement.parentElement.remove();" class="btn-primary" style="width: 100%; padding: 15px; font-size: 1.1em; margin-bottom: 10px;">
                🔐 Sign In Now to Save Data
            </button>
            <button onclick="this.parentElement.parentElement.remove();" class="btn-secondary" style="width: 100%; padding: 10px;">
                Continue in Guest Mode (Risk Data Loss)
            </button>
        </div>
    `;
    document.body.appendChild(modal);
}

// Migrate local data to server when user signs in
async function migrateLocalDataToServer() {
    const localWatchlist = getLocalWatchlist();
    const localPortfolio = getLocalPortfolio();

    if (localWatchlist.length > 0 || localPortfolio.trades.length > 0) {
        const migrate = confirm(`You have ${localWatchlist.length} watchlist items and ${localPortfolio.trades.length} trades in guest mode. Migrate them to your account?`);

        if (migrate) {
            // Migrate watchlist
            for (const item of localWatchlist) {
                try {
                    await addToWatchlist(item.ticker, item.name, item.exchange);
                } catch (e) {
                    console.error('Failed to migrate watchlist item:', e);
                }
            }

            // Clear local storage after migration
            localStorage.removeItem('guestWatchlist');
            localStorage.removeItem('guestPortfolio');

            showNotification('✅ Your guest data has been migrated to your account!', 'success');
        }
    }
}

// ========== Watchlist Functions ==========

async function updateWatchlistPrices(watchlist) {
    console.log('🔄 Updating watchlist prices for', watchlist.length, 'items');

    // Fetch current prices for all tickers
    for (const item of watchlist) {
        try {
            const response = await fetch(`/api/intraday/${item.ticker}`);
            const data = await response.json();

            if (data.status === 'success' && data.prices && data.prices.length > 0) {
                // Get the latest price
                item.current_price = data.prices[data.prices.length - 1];
                console.log(`✅ Updated ${item.ticker}: $${item.current_price.toFixed(2)}`);
            } else {
                console.warn(`⚠️ No price data for ${item.ticker}, keeping entry price`);
            }
        } catch (error) {
            console.error(`❌ Error fetching price for ${item.ticker}:`, error);
        }
    }

    // Save updated watchlist back to localStorage (without prompting registration)
    if (!isAuthenticated) {
        localStorage.setItem('guestWatchlist', JSON.stringify(watchlist));
    }
}

async function loadWatchlist() {
    if (!isAuthenticated) {
        displayLocalWatchlist();
        return;
    }

    try {
        const response = await fetch('/api/watchlist');
        const data = await response.json();

        if (data.success) {
            displayWatchlistItems(data.watchlist);
        }
    } catch (error) {
        console.error('Failed to load watchlist:', error);
    }
}

async function displayLocalWatchlist() {
    const watchlist = getLocalWatchlist();
    // Update prices before displaying
    await updateWatchlistPrices(watchlist);
    displayWatchlistItems(watchlist);
}

function displayWatchlistItems(watchlist) {
    const container = document.getElementById('watchlist-container');
    console.log('🎨 displayWatchlistItems called, container:', container);
    console.log('Watchlist data:', watchlist);

    if (!container) {
        console.error('❌ watchlist-container element not found in DOM!');
        return;
    }

    if (watchlist.length === 0) {
        container.innerHTML = '<p class="empty-state">Your watchlist is empty. Search for a stock to add it!</p>';
        return;
    }

    // Calculate days held and check prediction correctness
    watchlist.forEach((item, index) => {
        if (item.entry_date) {
            const entryDate = new Date(item.entry_date);
            const today = new Date();
            item.days_held = Math.floor((today - entryDate) / (1000 * 60 * 60 * 24));
        }

        // Calculate price changes (in guest mode, prices don't update unless user refreshes)
        if (item.entry_price && item.current_price) {
            item.price_change = item.current_price - item.entry_price;
            item.price_change_pct = (item.price_change / item.entry_price) * 100;

            // Check if prediction was correct
            if (item.predicted_direction === 'UP' && item.price_change > 0) {
                item.prediction_correct = true;
            } else if (item.predicted_direction === 'DOWN' && item.price_change < 0) {
                item.prediction_correct = true;
            } else if (item.price_change !== 0) {
                item.prediction_correct = false;
            }
        }
    });

    container.innerHTML = watchlist.map((item, index) => {
        const priceChange = item.price_change || 0;
        const priceChangePct = item.price_change_pct || 0;
        const isProfitable = priceChange >= 0;
        const isCorrect = item.prediction_correct;
        const confidence = (item.confidence || 0) * 100;

        // For China stocks, show name prominently (tickers are just numbers)
        const isChinaStock = item.ticker && (item.ticker.endsWith('.HK') || item.ticker.endsWith('.SS') || item.ticker.endsWith('.SZ'));
        // Use stored name, or look up from CHINA_TICKER_NAMES for legacy items
        const displayName = item.name && item.name !== item.ticker
            ? item.name
            : (isChinaStock ? getChinaStockName(item.ticker) : null);

        return `
            <div class="portfolio-item">
                <div class="portfolio-item-header">
                    <div>
                        ${displayName ? `
                            <h4 class="portfolio-ticker">${displayName}</h4>
                            <p class="portfolio-ticker-sub">${item.ticker}</p>
                        ` : `
                            <h4 class="portfolio-ticker">${item.ticker}</h4>
                        `}
                        <p class="portfolio-date">Added: ${new Date(item.entry_date).toLocaleDateString()} • Confidence: ${confidence.toFixed(1)}%</p>
                    </div>
                    <button class="btn-remove" onclick="removeFromWatchlist(${isAuthenticated ? item.id : index})">🗑️</button>
                </div>

                <div class="portfolio-item-body">
                    <div class="portfolio-metric">
                        <span class="metric-label">Entry</span>
                        <span class="metric-value">$${(item.entry_price || 0).toFixed(2)}</span>
                    </div>
                    <div class="portfolio-metric">
                        <span class="metric-label">Current</span>
                        <span class="metric-value">$${(item.current_price || 0).toFixed(2)}</span>
                    </div>
                    <div class="portfolio-metric">
                        <span class="metric-label">Change</span>
                        <span class="metric-value ${isProfitable ? 'text-success' : 'text-danger'}">
                            ${isProfitable ? '+' : ''}${priceChangePct.toFixed(2)}%
                        </span>
                    </div>
                </div>

                <div class="portfolio-item-footer">
                    ${isCorrect !== null ? `<div class="prediction-badge ${isCorrect ? 'correct' : 'incorrect'}">
                        ${isCorrect ? '✅ Correct' : '❌ Incorrect'} Prediction
                    </div>` : '<div class="prediction-badge">⏳ Tracking...</div>'}
                    <span class="days-held">${item.days_held || 0} days</span>
                </div>
            </div>
        `;
    }).join('');
    console.log('✅ Watchlist displayed successfully');
}

async function addToWatchlist(predictionData) {
    // Support both old signature (ticker, name, exchange) and new signature (predictionData object)
    let ticker, name, exchange, data;

    if (typeof predictionData === 'string') {
        // Old signature: addToWatchlist(ticker, name, exchange)
        ticker = predictionData;
        name = arguments[1];
        exchange = arguments[2];
        data = {ticker, name, exchange};
    } else {
        // New signature: addToWatchlist(predictionData)
        data = predictionData;
        ticker = data.ticker;
        name = data.company?.name || data.name || ticker;
        exchange = data.exchange || 'US';
    }

    console.log('✅ addToWatchlist called:', {ticker, isAuthenticated});

    if (isAuthenticated) {
        // Server mode
        console.log('📡 Using server mode (authenticated)');
        try {
            const response = await fetch('/api/watchlist', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ticker, name, exchange})
            });

            const apiData = await response.json();
            if (apiData.success) {
                showNotification(`${ticker} added to watchlist!`, 'success');
                await loadWatchlist();
            } else {
                showNotification(apiData.error || 'Failed to add to watchlist', 'error');
            }
        } catch (error) {
            showNotification('Failed to add to watchlist', 'error');
            console.error(error);
        }
    } else {
        // Local storage mode - save full prediction data
        console.log('💾 Using local storage mode (guest)');
        const watchlist = getLocalWatchlist();
        console.log('Current watchlist:', watchlist);

        // Check if already exists
        if (watchlist.some(item => item.ticker === ticker)) {
            console.log('⚠️ Ticker already in watchlist');
            showNotification(`${ticker} is already in your watchlist`, 'warning');
            return;
        }

        console.log('➕ Adding to watchlist...');
        console.log('📊 Full prediction data:', data);

        // Store comprehensive data for guest mode
        // Try multiple paths for confidence (different API response formats)
        const confidence = data.prediction?.direction_confidence ||  // Correct path!
                          data.model_info?.confidence ||
                          data.confidence ||
                          data.signal?.confidence ||
                          0;

        console.log('🎯 Extracted confidence:', confidence, 'from paths:', {
            'prediction.direction_confidence': data.prediction?.direction_confidence,
            'model_info.confidence': data.model_info?.confidence,
            'confidence': data.confidence,
            'signal.confidence': data.signal?.confidence
        });

        // Convert numeric direction to string for correctness checking
        let directionStr = 'N/A';
        const direction = data.prediction?.direction;
        if (direction === 1 || direction === '1' || direction === 'UP') {
            directionStr = 'UP';
        } else if (direction === -1 || direction === '-1' || direction === 'DOWN') {
            directionStr = 'DOWN';
        } else if (direction === 0 || direction === '0') {
            directionStr = 'NEUTRAL';
        }

        const watchlistItem = {
            ticker: ticker,
            name: name,
            exchange: exchange,
            entry_price: data.current_price || 0,
            current_price: data.current_price || 0,
            entry_date: new Date().toISOString(),
            confidence: confidence,
            predicted_direction: directionStr,
            predicted_return: data.prediction?.expected_return || 0,
            shares: 0,  // No shares initially (watchlist only)
            days_held: 0,
            price_change: 0,
            price_change_pct: 0,
            prediction_correct: null
        };

        watchlist.push(watchlistItem);
        saveLocalWatchlist(watchlist);
        console.log('💾 Saved to localStorage:', watchlist);
        displayLocalWatchlist();
        console.log('🎨 Display function called');
        showNotification(`${ticker} added to watchlist! (Guest Mode)`, 'success');

        // Increment action counter for progressive registration
        incrementActionCounter();
    }
}

async function removeFromWatchlist(itemIdOrIndex) {
    if (isAuthenticated) {
        // Server mode
        if (!confirm('Remove from watchlist?')) return;

        try {
            const response = await fetch(`/api/watchlist/${itemIdOrIndex}`, {
                method: 'DELETE'
            });

            const data = await response.json();
            if (data.success) {
                showNotification(data.message, 'success');
                await loadWatchlist();
            }
        } catch (error) {
            showNotification('Failed to remove from watchlist', 'error');
            console.error(error);
        }
    } else {
        // Local storage mode
        if (!confirm('Remove from watchlist?')) return;

        const watchlist = getLocalWatchlist();
        watchlist.splice(itemIdOrIndex, 1);
        saveLocalWatchlist(watchlist);
        displayLocalWatchlist();
        showNotification('Removed from watchlist', 'success');
    }
}

// ========== Portfolio Functions ==========

async function updatePortfolioPrices(portfolio) {
    console.log('🔄 Updating portfolio prices for', portfolio.trades.length, 'trades');

    // Fetch current prices for all trades
    for (const trade of portfolio.trades) {
        try {
            const response = await fetch(`/api/intraday/${trade.ticker}`);
            const data = await response.json();

            if (data.status === 'success' && data.prices && data.prices.length > 0) {
                // Get the latest price
                trade.currentPrice = data.prices[data.prices.length - 1];
                console.log(`✅ Updated ${trade.ticker}: $${trade.currentPrice.toFixed(2)}`);
            } else {
                console.warn(`⚠️ No price data for ${trade.ticker}, keeping entry price`);
                trade.currentPrice = trade.entry_price;
            }
        } catch (error) {
            console.error(`❌ Error fetching price for ${trade.ticker}:`, error);
            trade.currentPrice = trade.entry_price;
        }
    }

    // Save updated portfolio back to localStorage (without prompting registration)
    if (!isAuthenticated) {
        localStorage.setItem('guestPortfolio', JSON.stringify(portfolio));
    }
}

async function loadPortfolio() {
    if (!isAuthenticated) {
        displayLocalPortfolio();
        return;
    }

    try {
        const response = await fetch('/api/portfolio');
        const data = await response.json();

        if (data.success) {
            displayPortfolio(data.portfolio, data.open_positions, data.recent_closed);
        }
    } catch (error) {
        console.error('Failed to load portfolio:', error);
    }
}

async function displayLocalPortfolio() {
    console.log('🎨 displayLocalPortfolio called');
    let portfolio = getLocalPortfolio();
    console.log('Portfolio data:', portfolio);
    console.log('Trades in portfolio:', portfolio.trades);

    // Update prices before displaying
    await updatePortfolioPrices(portfolio);

    // Calculate current values with error handling
    // CORRECT FORMULA (COLLATERAL MODEL):
    // Total Value = Cash + LONG position current values + SHORT unrealized P&L
    // - LONG positions: Cash was spent to buy, so add current market value back
    // - SHORT positions: Cash unchanged (collateral), so only add P&L
    try {
        const STARTING_CASH = 100000;

        // Calculate position values based on position type
        let totalLongValue = 0;       // Current market value of LONG positions (assets)
        let totalShortValue = 0;      // Current market value of SHORT positions (liabilities)
        let totalShortPnL = 0;        // Unrealized P&L from SHORT positions
        let totalUnrealizedPnL = 0;   // Total P&L for display

        for (const t of portfolio.trades) {
            const currentPrice = t.currentPrice || t.entry_price || 0;
            const entryPrice = t.entry_price || 0;
            const quantity = t.quantity || 0;
            const marketValue = currentPrice * quantity;

            if (t.action === 'SELL' || t.action === 'SHORT') {
                // SHORT position: Track both market value (liability) and P&L
                const pnl = (entryPrice - currentPrice) * quantity;
                totalShortValue += marketValue;  // This is a liability
                totalShortPnL += pnl;
                totalUnrealizedPnL += pnl;
                console.log(`SHORT ${t.ticker}: entry=$${entryPrice}, current=$${currentPrice}, qty=${quantity}, liability=$${marketValue.toFixed(2)}, P&L=$${pnl.toFixed(2)}`);
            } else {
                // LONG position: Current market value adds to total
                const pnl = (currentPrice - entryPrice) * quantity;
                totalLongValue += marketValue;
                totalUnrealizedPnL += pnl;
                console.log(`LONG ${t.ticker}: entry=$${entryPrice}, current=$${currentPrice}, qty=${quantity}, value=$${marketValue.toFixed(2)}, P&L=$${pnl.toFixed(2)}`);
            }
        }

        // Net position value = LONG assets - SHORT liabilities
        let netPositionValue = totalLongValue - totalShortValue;

        console.log('Total LONG position value:', totalLongValue.toFixed(2));
        console.log('Total SHORT liability:', totalShortValue.toFixed(2));
        console.log('Net position value:', netPositionValue.toFixed(2));
        console.log('Total SHORT P&L:', totalShortPnL.toFixed(2));
        console.log('Total unrealized P&L:', totalUnrealizedPnL.toFixed(2));

        // Calculate realized P&L from closed trades
        let realizedPnL = 0;
        for (const closedTrade of (portfolio.closedTrades || [])) {
            realizedPnL += closedTrade.pnl || 0;
        }
        console.log('Realized P&L from closed trades:', realizedPnL.toFixed(2));

        // CORRECT Total Value = Starting Cash + Realized P&L + Unrealized P&L
        // This does NOT depend on potentially corrupted portfolio.cash
        let totalValue = STARTING_CASH + realizedPnL + totalUnrealizedPnL;

        // SANITY CHECK: Detect and fix severe data corruption
        // Enhanced debugging log (from math fixing8.pdf recommendation)
        console.log('=== PORTFOLIO SANITY CHECK ===');
        console.log('Cash Calculation Debug:', {
            startingCash: PORTFOLIO_CONFIG.STARTING_CASH,
            realizedPnL: realizedPnL.toFixed(2),
            unrealizedPnL: totalUnrealizedPnL.toFixed(2),
            totalValue: totalValue.toFixed(2),
            netPositionValue: netPositionValue.toFixed(2),
            storedCash: portfolio.cash.toFixed(2),
            longValue: totalLongValue.toFixed(2),
            shortLiability: totalShortValue.toFixed(2),
            shortPnL: totalShortPnL.toFixed(2)
        });

        // Check for impossible values that indicate severe corruption
        const isSeverelyCorrupted = (
            totalValue <= 0 ||
            totalValue < PORTFOLIO_CONFIG.SEVERE_CORRUPTION_MIN_VALUE ||
            portfolio.cash < PORTFOLIO_CONFIG.SEVERE_NEGATIVE_CASH ||
            Math.abs(totalValue) > PORTFOLIO_CONFIG.SEVERE_CORRUPTION_MAX_VALUE
        );

        if (isSeverelyCorrupted) {
            console.error('SEVERE PORTFOLIO CORRUPTION DETECTED!');
            console.error('   Total Value:', totalValue);
            console.error('   Cash:', portfolio.cash);

            // Backup corrupted data before reset (for debugging)
            backupPortfolio(portfolio, 'pre-reset-corruption');

            // Emergency reset to clean state - NO RELOAD (prevents infinite loop)
            const cleanPortfolio = {
                cash: PORTFOLIO_CONFIG.STARTING_CASH,
                totalValue: PORTFOLIO_CONFIG.STARTING_CASH,
                peakValue: PORTFOLIO_CONFIG.STARTING_CASH,
                trades: [],
                closedTrades: [],
                lastUpdated: new Date().toISOString()
            };
            localStorage.setItem('mockPortfolio', JSON.stringify(cleanPortfolio));

            // Show notification but DO NOT reload - just use the clean portfolio
            console.log('Portfolio reset to clean state (no reload)');
            showNotification('Portfolio data was corrupted. Reset to $100,000.', 'warning');

            // Update the portfolio variable to use clean data and continue
            portfolio = cleanPortfolio;
            totalValue = PORTFOLIO_CONFIG.STARTING_CASH;
            realizedPnL = 0;
            totalUnrealizedPnL = 0;
            // Also reset position values since portfolio is now empty
            totalLongValue = 0;
            totalShortValue = 0;
            totalShortPnL = 0;
            netPositionValue = 0;
        }

        // Calculate what cash SHOULD be based on total value and positions
        // Using the formula: Cash = Total Value - Net Position Value
        // Where Net Position Value = LONG values - SHORT liabilities
        const expectedCash = totalValue - netPositionValue;

        // Check if cash is significantly wrong
        const cashDiscrepancy = Math.abs(portfolio.cash - expectedCash);
        if (cashDiscrepancy > PORTFOLIO_CONFIG.CASH_DISCREPANCY_THRESHOLD) {
            console.warn('🔧 Cash discrepancy detected, auto-fixing...');
            console.warn(`   Current cash: $${portfolio.cash.toFixed(2)}`);
            console.warn(`   Expected cash: $${expectedCash.toFixed(2)}`);
            console.warn(`   Discrepancy: $${cashDiscrepancy.toFixed(2)}`);

            // Fix the cash
            portfolio.cash = expectedCash;
            saveLocalPortfolio(portfolio);
            console.log('   ✅ Cash auto-corrected to:', portfolio.cash.toFixed(2));
        } else {
            console.log('✅ Portfolio sanity check passed (cash within tolerance)');
        }
        const totalReturn = totalValue - STARTING_CASH;
        const totalReturnPercent = (totalReturn / STARTING_CASH) * 100;

        const closedTrades = portfolio.closedTrades || [];
        const winningTrades = closedTrades.filter(t => t.pnl > 0).length;
        const winRate = closedTrades.length > 0 ? (winningTrades / closedTrades.length) * 100 : 0;

        // Calculate drawdown from peak
        // First update peak if we're at a new all-time high
        const currentPeakValue = portfolio.peakValue || STARTING_CASH;
        const peakValue = Math.max(currentPeakValue, totalValue);

        // Drawdown should be 0% at all-time highs, negative when below peak
        let drawdown = 0;
        if (totalValue >= peakValue) {
            drawdown = 0;  // At all-time high = 0% drawdown
        } else {
            drawdown = ((totalValue - peakValue) / peakValue) * 100;  // Negative value
        }
        const drawdownAmount = totalValue - peakValue;

        // Calculate position concentrations for alerts
        // Use a minimum totalValue threshold to avoid division issues
        const safeTotal = Math.max(totalValue, 1000);  // Minimum $1000 to avoid extreme %
        const positionConcentrations = portfolio.trades.map(t => {
            const currentPrice = t.currentPrice || t.entry_price || 0;
            const positionValue = Math.abs(currentPrice * t.quantity);
            // Cap concentration at 100% to avoid impossible values
            const concentration = Math.min((positionValue / safeTotal) * 100, 100);
            return {
                ticker: t.ticker,
                value: positionValue,
                concentration: concentration,
                isOverConcentrated: concentration > 15 // Alert threshold: 15%
            };
        });

        // Find positions exceeding concentration limit
        const overConcentratedPositions = positionConcentrations.filter(p => p.isOverConcentrated);

        console.log('Calling displayPortfolio with:', {
            portfolio: {
                total_value: totalValue,
                current_cash: portfolio.cash,
                total_return: totalReturn,
                total_return_percent: totalReturnPercent,
                win_rate: winRate / 100,
                drawdown: drawdown,
                drawdown_amount: drawdownAmount,
                peak_value: peakValue,
                peak_date: portfolio.peakDate
            },
            openPositions: portfolio.trades.length,
            closedTrades: closedTrades.length,
            overConcentratedPositions: overConcentratedPositions.length
        });

        displayPortfolio(
            {
                total_value: totalValue,
                current_cash: portfolio.cash,
                total_return: totalReturn,
                total_return_percent: totalReturnPercent,
                win_rate: winRate / 100,
                drawdown: drawdown,
                drawdown_amount: drawdownAmount,
                peak_value: peakValue,
                peak_date: portfolio.peakDate,
                position_concentrations: positionConcentrations,
                over_concentrated: overConcentratedPositions
            },
            portfolio.trades,
            closedTrades.slice(0, 10)
        );

        // Update peak value in portfolio before saving if we hit a new ATH
        if (totalValue > currentPeakValue) {
            portfolio.peakValue = totalValue;
            portfolio.peakDate = new Date().toISOString();
        }

        // Save updated portfolio with peak tracking
        saveLocalPortfolio(portfolio);
        console.log('✅ displayPortfolio completed successfully');
    } catch (error) {
        console.error('❌ Error in displayLocalPortfolio:', error);
        console.error('Error stack:', error.stack);
    }
}

function displayPortfolio(portfolio, openPositions, closedTrades) {
    console.log('📊 displayPortfolio called with:', {portfolio, openPositions, closedTrades});
    const container = document.getElementById('portfolio-container');
    console.log('Container element:', container);
    if (!container) {
        console.error('❌ portfolio-container not found!');
        return;
    }

    // Ensure portfolio has all required fields with defaults
    portfolio = {
        total_value: 100000,
        current_cash: 100000,
        total_return: 0,
        total_return_percent: 0,
        win_rate: 0,
        drawdown: 0,
        peak_value: 100000,
        over_concentrated: [],
        ...portfolio  // Override defaults with actual values if they exist
    };

    // Ensure arrays are defined
    openPositions = openPositions || [];
    closedTrades = closedTrades || [];

    // Get yesterday's snapshot for comparison
    const portfolioData = getLocalPortfolio();
    const yesterday = portfolioData.dailySnapshot;

    // Calculate changes from yesterday
    const getChange = (current, field) => {
        if (!yesterday) return { value: 0, class: 'neutral', symbol: '' };
        const change = current - yesterday[field];
        if (Math.abs(change) < 0.01) return { value: 0, class: 'neutral', symbol: '' };
        return {
            value: change,
            class: change > 0 ? 'up' : 'down',
            symbol: change > 0 ? '↑' : '↓'
        };
    };

    const valueChange = getChange(portfolio.total_value, 'totalValue');
    const cashChange = getChange(portfolio.current_cash, 'cash');
    const returnChange = getChange(portfolio.total_return_percent, 'totalReturnPercent');
    const winRateChange = getChange(portfolio.win_rate * 100, 'winRate');

    // Calculate visual elements
    const returnBarWidth = Math.min(Math.abs(portfolio.total_return_percent), 100);
    const winRateDots = Array.from({length: 10}, (_, i) => {
        if (i < Math.floor((portfolio.win_rate * 100) / 10)) return 'win';
        return 'empty';
    });

    const html = `
        <div class="portfolio-summary">
            <div class="portfolio-stat">
                <div class="stat-content">
                    <span class="stat-label">Total Value</span>
                    <span class="stat-value">
                        $${portfolio.total_value.toLocaleString('en-US', {minimumFractionDigits: 2})}
                        ${valueChange.value !== 0 ? `<span class="stat-change ${valueChange.class}">${valueChange.symbol} $${Math.abs(valueChange.value).toFixed(2)}</span>` : ''}
                    </span>
                </div>
            </div>
            <div class="portfolio-stat">
                <div class="stat-content">
                    <span class="stat-label">Cash</span>
                    <span class="stat-value">
                        $${portfolio.current_cash.toLocaleString('en-US', {minimumFractionDigits: 2})}
                        ${cashChange.value !== 0 ? `<span class="stat-change ${cashChange.class}">${cashChange.symbol} $${Math.abs(cashChange.value).toFixed(2)}</span>` : ''}
                    </span>
                </div>
            </div>
            <div class="portfolio-stat ${portfolio.total_return >= 0 ? 'positive' : 'negative'}">
                <div class="stat-content">
                    <span class="stat-label">Total Return</span>
                    <span class="stat-value">
                        ${portfolio.total_return >= 0 ? '+' : ''}${portfolio.total_return_percent.toFixed(2)}%
                        ${returnChange.value !== 0 ? `<span class="stat-change ${returnChange.class}">${returnChange.symbol} ${Math.abs(returnChange.value).toFixed(2)}%</span>` : ''}
                    </span>
                    <div class="stat-chart">
                        <div class="progress-bar">
                            <div class="progress-fill ${portfolio.total_return >= 0 ? '' : 'negative'}" style="width: ${returnBarWidth}%"></div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="portfolio-stat">
                <div class="stat-content">
                    <span class="stat-label">Win Rate</span>
                    <span class="stat-value">
                        ${(portfolio.win_rate * 100).toFixed(1)}%
                        ${winRateChange.value !== 0 ? `<span class="stat-change ${winRateChange.class}">${winRateChange.symbol} ${Math.abs(winRateChange.value / 100).toFixed(1)}%</span>` : ''}
                    </span>
                    <div class="stat-chart">
                        <div class="win-rate-visual">
                            ${winRateDots.map(type => `<div class="win-rate-dot ${type}"></div>`).join('')}
                        </div>
                    </div>
                </div>
            </div>
            <div class="portfolio-stat ${(portfolio.drawdown || 0) < -10 ? 'warning' : (portfolio.drawdown || 0) < 0 ? 'negative' : 'neutral'}">
                <div class="stat-content">
                    <span class="stat-label">Drawdown</span>
                    <span class="stat-value">
                        ${(portfolio.drawdown || 0).toFixed(2)}%
                        ${(portfolio.drawdown || 0) < -10 ? '<span class="stat-alert">ALERT</span>' : ''}
                    </span>
                    <div class="stat-chart">
                        <div class="drawdown-info">
                            <small>Peak: $${(portfolio.peak_value || 100000).toLocaleString('en-US', {minimumFractionDigits: 0})}</small>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        ${(portfolio.over_concentrated && portfolio.over_concentrated.length > 0) ? `
        <div class="concentration-alert">
            <strong>Position Concentration Alert:</strong>
            ${portfolio.over_concentrated.map(p => `${p.ticker} (${p.concentration.toFixed(1)}%)`).join(', ')} exceed 15% portfolio concentration.
        </div>
        ` : ''}

        <div class="portfolio-section">
            <h3>Open Positions (${openPositions.length})</h3>
            ${openPositions.length === 0 ? '<p class="empty-state">No open positions. Open your first trade!</p>' :
                openPositions.map((trade, index) => {
                    const positionValue = (trade.currentPrice || trade.entry_price) * trade.quantity;

                    // Calculate P&L correctly for both LONG and SHORT positions
                    let positionPnl;
                    if (trade.action === 'SELL' || trade.action === 'SHORT') {
                        // SHORT position: profit when price goes DOWN
                        positionPnl = (trade.entry_price - (trade.currentPrice || trade.entry_price)) * trade.quantity;
                    } else {
                        // LONG/BUY position: profit when price goes UP
                        positionPnl = ((trade.currentPrice || trade.entry_price) - trade.entry_price) * trade.quantity;
                    }

                    const isProfitable = positionPnl >= 0;
                    const entryDate = trade.openedAt || trade.entry_date || new Date().toISOString();
                    const daysHeld = Math.floor((new Date() - new Date(entryDate)) / (1000 * 60 * 60 * 24));

                    // Determine if LONG or SHORT position
                    const isShort = trade.action === 'SELL' || trade.action === 'SHORT';
                    const positionType = isShort ? 'SHORT' : 'LONG';
                    const positionTypeClass = isShort ? 'position-type-short' : 'position-type-long';

                    // For China stocks, show name prominently (tickers are just numbers)
                    const isChinaStock = trade.ticker && (trade.ticker.endsWith('.HK') || trade.ticker.endsWith('.SS') || trade.ticker.endsWith('.SZ'));
                    // Use stored name, or look up from CHINA_TICKER_NAMES for legacy trades
                    const displayName = trade.name && trade.name !== trade.ticker
                        ? trade.name
                        : (isChinaStock ? getChinaStockName(trade.ticker) : null);

                    return `
                        <div class="portfolio-item portfolio-item-expanded">
                            <div class="portfolio-item-header">
                                <div>
                                    <h4 class="portfolio-ticker">
                                        <span class="position-type-badge ${positionTypeClass}">${positionType}</span>
                                        ${displayName ? displayName : trade.ticker} (${trade.quantity} shares)
                                    </h4>
                                    ${displayName ? `<p class="portfolio-ticker-sub">${trade.ticker}</p>` : ''}
                                    <p class="portfolio-date">Opened: ${new Date(entryDate).toLocaleDateString()} | ${daysHeld} days</p>
                                </div>
                                <button class="btn-remove" onclick="closePosition(${isAuthenticated ? trade.id : index}, '${trade.ticker}', ${(trade.currentPrice || trade.entry_price).toFixed(2)})">Close</button>
                            </div>

                            <div class="portfolio-item-columns">
                                <!-- Left Column: Position Metrics -->
                                <div class="position-metrics-col">
                                    <div class="portfolio-metric">
                                        <span class="metric-label">Entry</span>
                                        <span class="metric-value">$${trade.entry_price.toFixed(2)}</span>
                                    </div>
                                    <div class="portfolio-metric">
                                        <span class="metric-label">Current</span>
                                        <span class="metric-value">$${(trade.currentPrice || trade.entry_price).toFixed(2)}</span>
                                    </div>
                                    <div class="portfolio-metric">
                                        <span class="metric-label">Value</span>
                                        <span class="metric-value">$${positionValue.toFixed(2)}</span>
                                    </div>
                                    <div class="portfolio-metric">
                                        <span class="metric-label">P&L</span>
                                        <span class="metric-value ${isProfitable ? 'text-success' : 'text-danger'}">
                                            ${isProfitable ? '+' : ''}$${positionPnl.toFixed(2)}
                                        </span>
                                    </div>
                                </div>

                                <!-- Right Column: ML Signal Panel -->
                                <div class="position-signal-col" id="signal-panel-${trade.ticker.replace(/[^a-zA-Z0-9]/g, '_')}">
                                    <div class="signal-panel-header">ML Signal</div>
                                    <div class="signal-panel-content">
                                        <button class="btn-get-signal" onclick="getPositionSignal('${trade.ticker}', ${index})">
                                            Get Signal
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                }).join('')
            }
        </div>

        <button onclick="resetPortfolio()" class="btn-reset-portfolio">Reset Portfolio</button>
    `;

    console.log('Setting innerHTML, HTML length:', html.length);
    container.innerHTML = html;
    console.log('✅ HTML set successfully, container now has', container.children.length, 'children');

    // Refresh entrance page news feed with updated watchlist
    if (typeof loadEntrancePageNews === 'function') {
        loadEntrancePageNews();
    }
}

async function openMockTrade(ticker, action, quantity, entryPrice, predictedReturn, confidence, name = null) {
    console.log('🎯 openMockTrade called with:', {ticker, action, quantity, entryPrice, predictedReturn, confidence, name, isAuthenticated});

    if (isAuthenticated) {
        // Server mode
        console.log('📡 Using server mode (authenticated)');
        try {
            const response = await fetch('/api/mock-trade/open', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    ticker,
                    action,
                    quantity,
                    entry_price: entryPrice,
                    predicted_return: predictedReturn,
                    prediction_confidence: confidence
                })
            });

            const data = await response.json();
            if (data.success) {
                showNotification(data.message, 'success');
                await loadPortfolio();
            } else {
                showNotification(data.error || 'Failed to open trade', 'error');
            }
        } catch (error) {
            showNotification('Failed to open trade', 'error');
            console.error(error);
        }
    } else {
        // Local storage mode
        console.log('💾 Using local storage mode (guest)');
        const portfolio = getLocalPortfolio();
        console.log('Current portfolio:', portfolio);

        const tradeValue = quantity * entryPrice;
        console.log(`Trade value: ${tradeValue}, Available cash: ${portfolio.cash}`);

        if (action === 'BUY' && portfolio.cash < tradeValue) {
            console.error('❌ Insufficient cash!');
            showNotification('Insufficient cash!', 'error');
            return;
        }

        const trade = {
            ticker,
            name: name || ticker,  // Store company name for display
            action,
            quantity,
            entry_price: entryPrice,
            currentPrice: entryPrice,
            predicted_return: predictedReturn,
            prediction_confidence: confidence,
            openedAt: new Date().toISOString()
        };

        console.log('Trade object created:', trade);

        if (action === 'BUY') {
            // LONG position: Spend cash to buy shares
            portfolio.cash -= tradeValue;
            console.log(`Cash after BUY: ${portfolio.cash}`);
        } else {
            // SHORT position: Do NOT add to cash - short proceeds are collateral, not spendable
            // The P&L will be calculated when position is closed
            console.log(`SHORT opened - cash unchanged: ${portfolio.cash} (proceeds are collateral)`);
        }

        portfolio.trades.push(trade);
        console.log('Trade added to portfolio, total trades:', portfolio.trades.length);

        saveLocalPortfolio(portfolio);
        console.log('✅ Portfolio saved to localStorage');

        displayLocalPortfolio();
        console.log('🎨 Display function called');

        showNotification(`${action} ${quantity} shares of ${ticker} (Guest Mode)`, 'success');

        // Increment action counter for progressive registration
        incrementActionCounter();
    }
}

async function closePosition(tradeIdOrIndex, ticker = '', currentPrice = 0) {
    const closePrice = prompt(`Close position for ${ticker}\nCurrent price: $${currentPrice}\n\nEnter closing price:`, currentPrice);
    if (!closePrice || isNaN(closePrice)) return;

    if (isAuthenticated) {
        // Server mode
        try {
            const response = await fetch(`/api/mock-trade/close/${tradeIdOrIndex}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({close_price: parseFloat(closePrice)})
            });

            const data = await response.json();
            if (data.success) {
                const pnlText = data.pnl >= 0 ?
                    `Profit: $${data.pnl.toFixed(2)} (+${data.pnl_percent.toFixed(2)}%)` :
                    `Loss: $${Math.abs(data.pnl).toFixed(2)} (${data.pnl_percent.toFixed(2)}%)`;

                showNotification(`Position closed. ${pnlText}`, data.pnl >= 0 ? 'success' : 'warning');
                await loadPortfolio();
            }
        } catch (error) {
            showNotification('Failed to close position', 'error');
            console.error(error);
        }
    } else {
        // Local storage mode
        const portfolio = getLocalPortfolio();
        const trade = portfolio.trades[tradeIdOrIndex];

        if (!trade) {
            showNotification('Trade not found', 'error');
            return;
        }

        const closePriceNum = parseFloat(closePrice);

        // Calculate P&L
        let pnl, pnlPercent;
        if (trade.action === 'BUY') {
            // LONG position: Sell shares to close, receive cash
            pnl = (closePriceNum - trade.entry_price) * trade.quantity;
            portfolio.cash += closePriceNum * trade.quantity;
            console.log(`Closed LONG: Received $${closePriceNum * trade.quantity}, P&L: $${pnl}`);
        } else {
            // SHORT position: Close by buying back shares
            // Using COLLATERAL MODEL: Only P&L affects cash (proceeds were held as collateral)
            // P&L = (entry_price - close_price) * quantity
            pnl = (trade.entry_price - closePriceNum) * trade.quantity;
            portfolio.cash += pnl;  // Add P&L (positive if price dropped, negative if price rose)
            console.log(`Closed SHORT: P&L $${pnl.toFixed(2)} (entry: $${trade.entry_price}, close: $${closePriceNum})`);
        }

        pnlPercent = (pnl / (trade.entry_price * trade.quantity)) * 100;

        // Save to closed trades
        const closedTrade = {
            ...trade,
            closePrice: closePriceNum,
            pnl,
            pnlPercent,
            closedAt: new Date().toISOString()
        };

        if (!portfolio.closedTrades) portfolio.closedTrades = [];
        portfolio.closedTrades.unshift(closedTrade);

        // Remove from open trades
        portfolio.trades.splice(tradeIdOrIndex, 1);

        saveLocalPortfolio(portfolio);
        displayLocalPortfolio();

        const pnlText = pnl >= 0 ?
            `Profit: $${pnl.toFixed(2)} (+${pnlPercent.toFixed(2)}%)` :
            `Loss: $${Math.abs(pnl).toFixed(2)} (${pnlPercent.toFixed(2)}%)`;

        showNotification(`Position closed. ${pnlText} (Guest Mode)`, pnl >= 0 ? 'success' : 'warning');
    }
}

// Get ML signal for an open position
async function getPositionSignal(ticker, index) {
    console.log('🎯 getPositionSignal called:', {ticker, index});
    const panelId = `signal-panel-${ticker.replace(/[^a-zA-Z0-9]/g, '_')}`;
    const panel = document.getElementById(panelId);
    console.log('Looking for panel:', panelId, 'Found:', !!panel);

    if (!panel) {
        console.error('Panel not found for ticker:', ticker, 'Expected ID:', panelId);
        return;
    }

    // Show loading state
    panel.innerHTML = `
        <div class="signal-panel-header">ML Signal</div>
        <div class="signal-panel-content signal-loading">
            <div class="spinner-inline"></div>
            <span>Analyzing...</span>
        </div>
    `;

    try {
        const fetchUrl = `/api/predict/${encodeURIComponent(ticker)}`;
        console.log('Fetching ML signal from:', fetchUrl);
        const response = await fetch(fetchUrl);
        const data = await response.json();
        console.log('ML signal response for', ticker, ':', data);

        if (data.status === 'success') {
            // Extract signal from trading_signal.action or fallback to 'HOLD'
            // API returns: trading_signal.action = 'LONG'/'SHORT'/'HOLD'
            // We need to map: LONG -> BUY, SHORT -> SELL
            let rawSignal = data.signal || (data.trading_signal && data.trading_signal.action) || 'HOLD';
            let signal = rawSignal;
            if (rawSignal === 'LONG') signal = 'BUY';
            else if (rawSignal === 'SHORT') signal = 'SELL';

            // Get confidence from trading_signal or prediction
            const confidence = ((data.trading_signal?.confidence || data.prediction?.direction_confidence || data.confidence || 0) * 100).toFixed(1);
            const expectedReturn = ((data.prediction?.expected_return || data.expected_return || 0) * 100).toFixed(2);
            const direction = data.prediction?.direction || data.prediction_direction || 'NEUTRAL';

            // Determine signal class
            let signalClass = 'signal-hold';
            if (signal.includes('BUY')) signalClass = 'signal-buy';
            else if (signal.includes('SELL')) signalClass = 'signal-sell';

            // Determine direction icon
            let dirIcon = '→';
            if (direction === 'UP' || direction === 1) dirIcon = '↑';
            else if (direction === 'DOWN' || direction === -1) dirIcon = '↓';

            // For HOLD signals, show a clearer message
            const isHold = signal.includes('HOLD');

            // User-friendly direction text
            let directionText = 'No clear trend';
            if (direction === 'UP' || direction === 1) directionText = 'Trending Up';
            else if (direction === 'DOWN' || direction === -1) directionText = 'Trending Down';

            // User-friendly confidence interpretation
            let confLevel = 'Low';
            let confClass = 'conf-low';
            if (parseFloat(confidence) >= 70) { confLevel = 'High'; confClass = 'conf-high'; }
            else if (parseFloat(confidence) >= 50) { confLevel = 'Medium'; confClass = 'conf-medium'; }

            // User-friendly expected return text
            const expRetNum = parseFloat(expectedReturn);
            let expRetText = 'Flat (no expected change)';
            if (expRetNum > 0.5) expRetText = `Expected gain: +${expectedReturn}%`;
            else if (expRetNum < -0.5) expRetText = `Expected loss: ${expectedReturn}%`;

            // Extract position details from trading_signal (for BUY/SELL signals)
            const tradingSignal = data.trading_signal || {};
            const position = tradingSignal.position || {};
            const hasPositionDetails = !isHold && (tradingSignal.entry_price || position.shares);

            // Store signal data for quick execution
            const currentPrice = data.current_price || tradingSignal.entry_price || 0;
            storedSignals[ticker] = {
                signal: signal,
                tradingSignal: tradingSignal,
                currentPrice: currentPrice,
                confidence: parseFloat(confidence) / 100,
                expectedReturn: parseFloat(expectedReturn) / 100
            };
            console.log('Stored signal data for', ticker, ':', storedSignals[ticker]);

            // Format position values
            const entryPrice = tradingSignal.entry_price ? `$${tradingSignal.entry_price.toFixed(2)}` : '--';
            const stopLoss = tradingSignal.stop_loss ? `$${tradingSignal.stop_loss.toFixed(2)}` : '--';
            const takeProfit = tradingSignal.take_profit ? `$${tradingSignal.take_profit.toFixed(2)}` : '--';
            const shares = position.shares || '--';
            const riskAmount = position.risk_amount ? `$${position.risk_amount.toFixed(2)}` : '--';
            const potentialProfit = position.potential_profit ? `$${position.potential_profit.toFixed(2)}` : '--';
            const riskReward = position.risk_reward_ratio ? `1:${position.risk_reward_ratio.toFixed(1)}` : '--';

            // Build position details HTML (only for BUY/SELL signals)
            const positionDetailsHtml = hasPositionDetails ? `
                <div class="position-details-mini">
                    <div class="position-details-title">Position Details</div>
                    <div class="position-grid-mini">
                        <div class="pos-item">
                            <span class="pos-label">Size</span>
                            <span class="pos-value">${shares} shares</span>
                        </div>
                        <div class="pos-item">
                            <span class="pos-label">Entry</span>
                            <span class="pos-value">${entryPrice}</span>
                        </div>
                        <div class="pos-item">
                            <span class="pos-label">Stop Loss</span>
                            <span class="pos-value text-danger">${stopLoss}</span>
                        </div>
                        <div class="pos-item">
                            <span class="pos-label">Take Profit</span>
                            <span class="pos-value text-success">${takeProfit}</span>
                        </div>
                    </div>
                    <div class="risk-metrics-mini">
                        <div class="risk-item">
                            <span class="risk-label">Risk</span>
                            <span class="risk-value text-danger">${riskAmount}</span>
                        </div>
                        <div class="risk-item">
                            <span class="risk-label">Profit</span>
                            <span class="risk-value text-success">${potentialProfit}</span>
                        </div>
                        <div class="risk-item">
                            <span class="risk-label">R/R</span>
                            <span class="risk-value">${riskReward}</span>
                        </div>
                    </div>
                </div>
            ` : '';

            // Execute signal button (only for BUY/SELL signals)
            const executeButtonHtml = !isHold ? `
                <button class="btn-execute-signal ${signal === 'BUY' ? 'btn-execute-buy' : 'btn-execute-sell'}" onclick="executeSignalTrade('${ticker}')">
                    ${signal === 'BUY' ? '&#9650;' : '&#9660;'} Execute ${signal}
                </button>
            ` : '';

            panel.innerHTML = `
                <div class="signal-panel-header-row">
                    <div class="signal-panel-header">ML Signal</div>
                    <div class="signal-badge-mini-header ${signalClass}">${signal}</div>
                    <span class="signal-toggle-arrow" onclick="toggleSignalDetails(${index})">▼</span>
                </div>
                <div class="signal-panel-collapsible" id="signal-details-${index}">
                    <div class="signal-panel-content ${signalClass}">
                        <div class="signal-details-friendly">
                            <div class="signal-row">
                                <span class="signal-label">Trend:</span>
                                <span class="signal-value">${dirIcon} ${directionText}</span>
                            </div>
                            <div class="signal-row">
                                <span class="signal-label">Confidence:</span>
                                <span class="signal-value ${confClass}">${confLevel} (${confidence}%)</span>
                            </div>
                            <div class="signal-row">
                                <span class="signal-label">Forecast:</span>
                                <span class="signal-value">${expRetText}</span>
                            </div>
                        </div>
                        ${isHold ? `<div class="signal-hold-note">The model doesn't see a strong buy or sell opportunity right now. Consider waiting for a clearer signal.</div>` : ''}
                        ${positionDetailsHtml}
                        ${executeButtonHtml}
                    </div>
                    <button class="btn-get-signal-refresh" onclick="getPositionSignal('${ticker}', ${index})">Refresh</button>
                </div>
            `;
        } else {
            panel.innerHTML = `
                <div class="signal-panel-header">ML Signal</div>
                <div class="signal-panel-content signal-error">
                    <span>Error fetching signal</span>
                </div>
                <button class="btn-get-signal" onclick="getPositionSignal('${ticker}', ${index})">Retry</button>
            `;
        }
    } catch (error) {
        console.error('Error fetching position signal:', error);
        panel.innerHTML = `
            <div class="signal-panel-header">ML Signal</div>
            <div class="signal-panel-content signal-error">
                <span>Network error</span>
            </div>
            <button class="btn-get-signal" onclick="getPositionSignal('${ticker}', ${index})">Retry</button>
        `;
    }
}

/**
 * Close all open positions at current market prices
 * Loops through all trades and closes them one by one
 */
async function closeAllPositions() {
    console.log('closeAllPositions() called');

    if (isAuthenticated) {
        // Server mode
        try {
            const response = await fetch('/api/portfolio');
            const data = await response.json();

            if (!data.success || !data.open_positions || data.open_positions.length === 0) {
                showNotification('No open positions to close.', 'info');
                return;
            }

            const positionCount = data.open_positions.length;
            if (!confirm(`Close all ${positionCount} open positions at current market prices?`)) return;

            let closedCount = 0;
            let totalPnL = 0;

            for (const position of data.open_positions) {
                try {
                    const closeResponse = await fetch(`/api/mock-trade/close/${position.id}`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({close_price: position.current_price || position.entry_price})
                    });

                    const closeData = await closeResponse.json();
                    if (closeData.success) {
                        closedCount++;
                        totalPnL += closeData.pnl || 0;
                    }
                } catch (error) {
                    console.error(`Failed to close position ${position.ticker}:`, error);
                }
            }

            const pnlText = totalPnL >= 0 ?
                `Total P&L: +$${totalPnL.toFixed(2)}` :
                `Total P&L: -$${Math.abs(totalPnL).toFixed(2)}`;

            showNotification(`Closed ${closedCount}/${positionCount} positions. ${pnlText}`, totalPnL >= 0 ? 'success' : 'warning');
            await loadPortfolio();

        } catch (error) {
            showNotification('Failed to close positions', 'error');
            console.error(error);
        }
    } else {
        // Local storage mode (Guest)
        const portfolio = getLocalPortfolio();

        if (!portfolio.trades || portfolio.trades.length === 0) {
            showNotification('No open positions to close.', 'info');
            return;
        }

        const positionCount = portfolio.trades.length;
        if (!confirm(`Close all ${positionCount} open positions at current market prices?`)) return;

        let totalPnL = 0;
        const tradesToClose = [...portfolio.trades]; // Copy array since we'll modify it

        for (const trade of tradesToClose) {
            const currentPrice = trade.currentPrice || trade.entry_price;
            let pnl;

            // Calculate P&L
            if (trade.action === 'BUY' || trade.action === 'LONG') {
                // LONG position: Sell shares to close, receive cash
                pnl = (currentPrice - trade.entry_price) * trade.quantity;
                portfolio.cash += currentPrice * trade.quantity;
                console.log(`Closed LONG ${trade.ticker}: Received $${(currentPrice * trade.quantity).toFixed(2)}, P&L: $${pnl.toFixed(2)}`);
            } else {
                // SHORT position: Close by buying back shares
                // Using COLLATERAL MODEL: Only P&L affects cash (proceeds were held as collateral)
                // P&L = (entry_price - close_price) * quantity
                pnl = (trade.entry_price - currentPrice) * trade.quantity;
                portfolio.cash += pnl;  // Add P&L (positive if price dropped, negative if price rose)
                console.log(`Closed SHORT ${trade.ticker}: P&L $${pnl.toFixed(2)} (entry: $${trade.entry_price}, close: $${currentPrice})`);
            }

            const pnlPercent = (pnl / (trade.entry_price * trade.quantity)) * 100;
            totalPnL += pnl;

            // Save to closed trades
            const closedTrade = {
                ...trade,
                closePrice: currentPrice,
                pnl,
                pnlPercent,
                closedAt: new Date().toISOString()
            };

            if (!portfolio.closedTrades) portfolio.closedTrades = [];
            portfolio.closedTrades.unshift(closedTrade);
        }

        // Clear all open trades
        portfolio.trades = [];

        saveLocalPortfolio(portfolio);
        displayLocalPortfolio();

        const pnlText = totalPnL >= 0 ?
            `Total P&L: +$${totalPnL.toFixed(2)}` :
            `Total P&L: -$${Math.abs(totalPnL).toFixed(2)}`;

        showNotification(`Closed ${positionCount} positions. ${pnlText} (Guest Mode)`, totalPnL >= 0 ? 'success' : 'warning');
    }
}

async function resetPortfolio() {
    if (!confirm('Reset portfolio to $100,000? This will close all positions and clear history.')) return;

    if (isAuthenticated) {
        // Server mode
        try {
            const response = await fetch('/api/portfolio/reset', {
                method: 'POST'
            });

            const data = await response.json();
            if (data.success) {
                showNotification(data.message, 'success');
                await loadPortfolio();
            }
        } catch (error) {
            showNotification('Failed to reset portfolio', 'error');
            console.error(error);
        }
    } else {
        // Local storage mode
        const portfolio = {
            cash: 100000,
            trades: [],
            closedTrades: [],
            totalReturn: 0,
            winRate: 0
        };
        saveLocalPortfolio(portfolio);
        displayLocalPortfolio();
        showNotification('Portfolio reset to $100,000 (Guest Mode)', 'success');
        guestActionCount = 0; // Reset action count
    }
}

// ========== Expose functions to global scope for onclick handlers ==========
// These functions are defined inside DOMContentLoaded but need to be accessible from inline HTML onclick attributes
window.closeAllPositions = closeAllPositions;
window.loadWatchlist = loadWatchlist;
// Note: clearWatchlist is defined in app.js, not here
window.loadPortfolio = loadPortfolio;
window.closePosition = closePosition;
window.resetPortfolio = resetPortfolio;
// Note: refreshTopPicks and showMainView are defined in app.js

// ========== Utility Functions ==========

function showNotification(message, type = 'info', duration = 3000) {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => notification.remove(), 300);
    }, duration);
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('login-modal');
    if (event.target === modal) {
        closeLoginModal();
    }
};

// Helper function for app.js to view a stock
function viewStock(ticker) {
    // This will be called by app.js to load stock data
    const tickerInput = document.getElementById('ticker-input');
    if (tickerInput) {
        tickerInput.value = ticker;
        const analyzeBtn = document.getElementById('analyze-btn');
        if (analyzeBtn) {
            analyzeBtn.click();
        }
    }
}

// ========== Push Notifications ==========

async function requestPushNotifications() {
    if (!('Notification' in window)) {
        showNotification('Push notifications are not supported by your browser', 'error');
        return false;
    }

    try {
        const permission = await Notification.requestPermission();

        if (permission === 'granted') {
            showNotification('✅ Push notifications enabled! You\'ll get alerts for sentiment changes.', 'success', 5000);

            // Subscribe to push notifications
            if (isAuthenticated) {
                await fetch('/api/push/subscribe', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({enabled: true})
                });
            }

            return true;
        } else if (permission === 'denied') {
            showNotification('❌ Push notifications blocked. Enable them in browser settings.', 'warning');
            return false;
        }
    } catch (error) {
        console.error('Push notification error:', error);
        return false;
    }
}

function showBrowserNotification(title, message, ticker = null) {
    if ('Notification' in window && Notification.permission === 'granted') {
        const notification = new Notification(title, {
            body: message,
            icon: '/static/img/icon.png',
            badge: '/static/img/badge.png',
            tag: ticker || 'general',
            requireInteraction: true,
            actions: ticker ? [
                { action: 'view', title: `View ${ticker}` },
                { action: 'dismiss', title: 'Dismiss' }
            ] : []
        });

        notification.onclick = function() {
            window.focus();
            if (ticker) {
                viewStock(ticker);
            }
            notification.close();
        };
    }
}

// ========== Social Sentiment Functions ==========

async function getSentiment(ticker) {
    try {
        const response = await fetch(`/api/sentiment/${ticker}`);
        const data = await response.json();

        if (data.success) {
            return data.sentiment;
        }
        return null;
    } catch (error) {
        console.error('Error getting sentiment:', error);
        return null;
    }
}

function displaySentimentAlert(ticker, sentiment) {
    const alertLevel = sentiment.alert_level;
    const overall = sentiment.overall_sentiment;
    const redditVolume = sentiment.reddit.volume;

    const levelEmojis = {
        'critical': '🔴',
        'high': '🟠',
        'medium': '🟡',
        'low': '🟢'
    };

    const sentimentEmojis = {
        'bullish': '🚀',
        'bearish': '📉',
        'neutral': '➡️'
    };

    const title = `${levelEmojis[alertLevel] || ''} ${ticker} Social Sentiment Alert`;
    const message = `${sentimentEmojis[overall]} ${overall.toUpperCase()}\n${redditVolume} Reddit mentions`;

    // Show browser notification
    showBrowserNotification(title, message, ticker);

    // Show in-app notification
    showNotification(`${title}: ${message}`, alertLevel === 'critical' ? 'warning' : 'info', 7000);
}

// Monitor watchlist for sentiment changes
let sentimentCheckInterval = null;

async function startSentimentMonitoring() {
    if (!isAuthenticated) {
        // For guests, check localStorage watchlist
        const watchlist = getLocalWatchlist();
        if (watchlist.length === 0) return;
    }

    // Check sentiment every 15 minutes
    sentimentCheckInterval = setInterval(async () => {
        const watchlist = isAuthenticated ?
            await loadWatchlist() :
            getLocalWatchlist();

        for (const item of watchlist) {
            const sentiment = await getSentiment(item.ticker);

            if (sentiment && sentiment.alert_level !== 'none') {
                // Check if this is a significant change
                const cachedSentiment = localStorage.getItem(`sentiment_${item.ticker}`);

                if (!cachedSentiment || shouldAlertSentimentChange(sentiment, JSON.parse(cachedSentiment))) {
                    displaySentimentAlert(item.ticker, sentiment);
                }

                // Cache current sentiment
                localStorage.setItem(`sentiment_${item.ticker}`, JSON.stringify(sentiment));
            }
        }
    }, 15 * 60 * 1000); // 15 minutes
}

function shouldAlertSentimentChange(current, previous) {
    // Alert on sentiment flip
    if (current.overall_sentiment !== previous.overall_sentiment) {
        return true;
    }

    // Alert on alert level increase
    const levels = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'critical': 4};
    if (levels[current.alert_level] > levels[previous.alert_level]) {
        return true;
    }

    return false;
}

// ========== Social Sharing Functions ==========

async function shareToSocial(platform, contentType, data) {
    try {
        const response = await fetch('/api/social/share', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                platform: platform,
                type: contentType,
                data: data
            })
        });

        const result = await response.json();

        if (result.success) {
            const shareContent = result.share_content;

            if (platform === 'twitter' || platform === 'linkedin' || platform === 'facebook') {
                // Open share URL in new window
                window.open(shareContent.url, '_blank', 'width=600,height=400');
            } else if (platform === 'reddit') {
                // Copy text to clipboard for Reddit
                navigator.clipboard.writeText(shareContent.text).then(() => {
                    showNotification('📋 Copied to clipboard! Paste in your Reddit post.', 'success');
                });
            } else {
                // Use Web Share API if available
                if (navigator.share) {
                    await navigator.share({
                        title: 'ML Stock Trading Platform',
                        text: shareContent.text,
                        url: window.location.href
                    });
                } else {
                    // Fallback: copy to clipboard
                    navigator.clipboard.writeText(shareContent.text).then(() => {
                        showNotification('📋 Copied to clipboard!', 'success');
                    });
                }
            }
        }
    } catch (error) {
        console.error('Share error:', error);
        showNotification('Failed to share content', 'error');
    }
}

function shareWatchlist() {
    const watchlist = isAuthenticated ?
        JSON.parse(localStorage.getItem('serverWatchlist') || '[]') :
        getLocalWatchlist();

    if (watchlist.length === 0) {
        showNotification('Add stocks to your watchlist first!', 'warning');
        return;
    }

    // Show share modal
    showShareModal('watchlist', watchlist);
}

function sharePortfolio() {
    const portfolio = isAuthenticated ?
        JSON.parse(localStorage.getItem('serverPortfolio') || '{}') :
        getLocalPortfolio();

    showShareModal('portfolio', portfolio);
}

function shareTrade(trade) {
    showShareModal('trade', trade);
}

function showShareModal(type, data) {
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.style.display = 'block';
    modal.innerHTML = `
        <div class="modal-content" style="max-width: 500px;">
            <span class="close-modal" onclick="this.parentElement.parentElement.remove()">&times;</span>
            <h2>📤 Share Your ${type === 'watchlist' ? 'Watchlist' : type === 'portfolio' ? 'Portfolio' : 'Trade'}</h2>
            <p style="margin: 20px 0; color: var(--gray);">Choose a platform to share:</p>

            <div class="social-share-buttons" style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <button onclick="shareToSocial('twitter', '${type}', ${JSON.stringify(data).replace(/'/g, "\\'") }); this.parentElement.parentElement.parentElement.remove();" class="btn-oauth" style="background: #1DA1F2; color: white; border: none;">
                    <svg style="width: 20px; height: 20px;" viewBox="0 0 24 24" fill="white">
                        <path d="M23 3a10.9 10.9 0 01-3.14 1.53 4.48 4.48 0 00-7.86 3v1A10.66 10.66 0 013 4s-4 9 5 13a11.64 11.64 0 01-7 2c9 5 20 0 20-11.5a4.5 4.5 0 00-.08-.83A7.72 7.72 0 0023 3z"/>
                    </svg>
                    Twitter / X
                </button>

                <button onclick="shareToSocial('linkedin', '${type}', ${JSON.stringify(data).replace(/'/g, "\\'")}); this.parentElement.parentElement.parentElement.remove();" class="btn-oauth" style="background: #0077B5; color: white; border: none;">
                    <svg style="width: 20px; height: 20px;" viewBox="0 0 24 24" fill="white">
                        <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                    </svg>
                    LinkedIn
                </button>

                <button onclick="shareToSocial('reddit', '${type}', ${JSON.stringify(data).replace(/'/g, "\\'")}); this.parentElement.parentElement.parentElement.remove();" class="btn-oauth" style="background: #FF4500; color: white; border: none;">
                    <svg style="width: 20px; height: 20px;" viewBox="0 0 24 24" fill="white">
                        <circle cx="12" cy="12" r="10"/>
                    </svg>
                    Reddit
                </button>

                <button onclick="shareToSocial('facebook', '${type}', ${JSON.stringify(data).replace(/'/g, "\\'")}); this.parentElement.parentElement.parentElement.remove();" class="btn-oauth" style="background: #1877F2; color: white; border: none;">
                    <svg style="width: 20px; height: 20px;" viewBox="0 0 24 24" fill="white">
                        <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/>
                    </svg>
                    Facebook
                </button>
            </div>

            <p style="margin-top: 20px; font-size: 0.85rem; color: var(--gray); text-align: center;">
                🔒 Share anonymously or with your social accounts
            </p>
        </div>
    `;
    document.body.appendChild(modal);
}

// Start sentiment monitoring when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Request push permissions if authenticated
    if (isAuthenticated && Notification.permission === 'default') {
        setTimeout(() => {
            const banner = document.createElement('div');
            banner.className = 'notification notification-info';
            banner.innerHTML = `
                <span>🔔 Enable push notifications to get social sentiment alerts</span>
                <button onclick="requestPushNotifications(); this.parentElement.remove();" style="margin-left: 10px; padding: 5px 10px; background: var(--primary); color: white; border: none; border-radius: 4px; cursor: pointer;">Enable</button>
                <button onclick="this.parentElement.remove();" style="margin-left: 5px; padding: 5px 10px; background: transparent; border: 1px solid white; border-radius: 4px; cursor: pointer; color: white;">Later</button>
            `;
            document.body.appendChild(banner);

            setTimeout(() => {
                if (banner.parentElement) {
                    banner.style.opacity = '0';
                    setTimeout(() => banner.remove(), 300);
                }
            }, 10000);
        }, 5000); // Show after 5 seconds
    }

    // Start monitoring
    setTimeout(startSentimentMonitoring, 2000);
});
