"""
Authentication routes for login, logout, and user management
"""

from flask import Blueprint, request, jsonify, session, redirect, url_for
from flask_login import login_user, logout_user, login_required, current_user
from flask_dance.contrib.google import google
from flask_dance.contrib.github import github
from flask_dance.contrib.reddit import reddit
from werkzeug.security import generate_password_hash, check_password_hash

from .models import db, User, Watchlist, MockTrade, Portfolio
from .oauth import get_or_create_user

auth_bp = Blueprint('auth', __name__)


# ========== OAuth Login Callbacks ==========

@auth_bp.route('/google-login')
def google_login():
    """Handle Google OAuth callback"""
    if not google.authorized:
        return redirect(url_for('google.login'))

    try:
        resp = google.get('/oauth2/v2/userinfo')
        if not resp.ok:
            return jsonify({'error': 'Failed to fetch user info from Google'}), 400

        info = resp.json()
        user = get_or_create_user(
            oauth_provider='google',
            oauth_id=info['id'],
            email=info.get('email'),
            username=info.get('email', '').split('@')[0],
            full_name=info.get('name'),
            avatar_url=info.get('picture')
        )

        login_user(user)
        return redirect('/')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/github-login')
def github_login():
    """Handle GitHub OAuth callback"""
    if not github.authorized:
        return redirect(url_for('github.login'))

    try:
        resp = github.get('/user')
        if not resp.ok:
            return jsonify({'error': 'Failed to fetch user info from GitHub'}), 400

        info = resp.json()

        # Get email (might need separate request)
        email = info.get('email')
        if not email:
            email_resp = github.get('/user/emails')
            if email_resp.ok:
                emails = email_resp.json()
                email = next((e['email'] for e in emails if e['primary']), None)

        user = get_or_create_user(
            oauth_provider='github',
            oauth_id=str(info['id']),
            email=email,
            username=info.get('login'),
            full_name=info.get('name'),
            avatar_url=info.get('avatar_url')
        )

        login_user(user)
        return redirect('/')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/reddit-login')
def reddit_login():
    """Handle Reddit OAuth callback"""
    if not reddit.authorized:
        return redirect(url_for('reddit.login'))

    try:
        resp = reddit.get('/api/v1/me')
        if not resp.ok:
            return jsonify({'error': 'Failed to fetch user info from Reddit'}), 400

        info = resp.json()
        user = get_or_create_user(
            oauth_provider='reddit',
            oauth_id=info['id'],
            email=None,  # Reddit doesn't provide email
            username=info.get('name'),
            full_name=info.get('name'),
            avatar_url=info.get('icon_img')
        )

        login_user(user)
        return redirect('/')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ========== Traditional Login/Logout ==========

@auth_bp.route('/api/auth/logout', methods=['POST'])
@login_required
def logout():
    """Logout current user"""
    logout_user()
    return jsonify({'success': True, 'message': 'Logged out successfully'})


@auth_bp.route('/api/auth/me', methods=['GET'])
def get_current_user():
    """Get current user info"""
    if current_user.is_authenticated:
        return jsonify({
            'authenticated': True,
            'user': {
                'id': current_user.id,
                'username': current_user.username,
                'email': current_user.email,
                'full_name': current_user.full_name,
                'avatar_url': current_user.avatar_url,
                'oauth_provider': current_user.oauth_provider
            }
        })
    return jsonify({'authenticated': False})


# ========== Watchlist Management ==========

@auth_bp.route('/api/watchlist', methods=['GET'])
@login_required
def get_watchlist():
    """Get user's watchlist"""
    watchlist = Watchlist.query.filter_by(user_id=current_user.id).all()
    return jsonify({
        'success': True,
        'watchlist': [item.to_dict() for item in watchlist]
    })


@auth_bp.route('/api/watchlist', methods=['POST'])
@login_required
def add_to_watchlist():
    """Add stock to watchlist"""
    data = request.json
    ticker = data.get('ticker')

    if not ticker:
        return jsonify({'error': 'Ticker required'}), 400

    # Check if already in watchlist
    existing = Watchlist.query.filter_by(
        user_id=current_user.id,
        ticker=ticker
    ).first()

    if existing:
        return jsonify({'error': 'Already in watchlist'}), 400

    watchlist_item = Watchlist(
        user_id=current_user.id,
        ticker=ticker,
        name=data.get('name'),
        exchange=data.get('exchange'),
        notes=data.get('notes'),
        alert_price_above=data.get('alert_price_above'),
        alert_price_below=data.get('alert_price_below')
    )

    db.session.add(watchlist_item)
    db.session.commit()

    return jsonify({
        'success': True,
        'message': f'{ticker} added to watchlist',
        'item': watchlist_item.to_dict()
    })


@auth_bp.route('/api/watchlist/<int:item_id>', methods=['DELETE'])
@login_required
def remove_from_watchlist(item_id):
    """Remove stock from watchlist"""
    item = Watchlist.query.filter_by(
        id=item_id,
        user_id=current_user.id
    ).first()

    if not item:
        return jsonify({'error': 'Item not found'}), 404

    ticker = item.ticker
    db.session.delete(item)
    db.session.commit()

    return jsonify({
        'success': True,
        'message': f'{ticker} removed from watchlist'
    })


# ========== Mock Trading ==========

@auth_bp.route('/api/mock-trade/open', methods=['POST'])
@login_required
def open_mock_trade():
    """Open a new mock trade"""
    data = request.json

    trade = MockTrade(
        user_id=current_user.id,
        ticker=data['ticker'],
        action=data['action'],  # 'BUY' or 'SELL'
        quantity=data['quantity'],
        entry_price=data['entry_price'],
        current_price=data['entry_price'],
        predicted_return=data.get('predicted_return'),
        prediction_confidence=data.get('prediction_confidence')
    )

    # Update portfolio cash
    portfolio = Portfolio.query.filter_by(user_id=current_user.id).first()
    if not portfolio:
        portfolio = Portfolio(user_id=current_user.id)
        db.session.add(portfolio)

    trade_value = data['quantity'] * data['entry_price']
    if data['action'] == 'BUY':
        if portfolio.current_cash < trade_value:
            return jsonify({'error': 'Insufficient cash'}), 400
        portfolio.current_cash -= trade_value
    else:  # SELL
        portfolio.current_cash += trade_value

    portfolio.total_trades += 1

    db.session.add(trade)
    db.session.commit()

    return jsonify({
        'success': True,
        'message': f'{data["action"]} {data["quantity"]} shares of {data["ticker"]}',
        'trade': trade.to_dict()
    })


@auth_bp.route('/api/mock-trade/close/<int:trade_id>', methods=['POST'])
@login_required
def close_mock_trade(trade_id):
    """Close an open mock trade"""
    trade = MockTrade.query.filter_by(
        id=trade_id,
        user_id=current_user.id,
        is_open=True
    ).first()

    if not trade:
        return jsonify({'error': 'Trade not found'}), 404

    data = request.json
    close_price = data['close_price']

    # Calculate P&L
    if trade.action == 'BUY':
        trade.pnl = (close_price - trade.entry_price) * trade.quantity
    else:  # SELL
        trade.pnl = (trade.entry_price - close_price) * trade.quantity

    trade.pnl_percent = (trade.pnl / (trade.entry_price * trade.quantity)) * 100
    trade.current_price = close_price
    trade.is_open = False
    from datetime import datetime
    trade.closed_at = datetime.utcnow()

    # Update portfolio
    portfolio = Portfolio.query.filter_by(user_id=current_user.id).first()

    # Add closing proceeds to cash
    if trade.action == 'BUY':
        portfolio.current_cash += close_price * trade.quantity
    # For SELL, cash was already added when opening

    portfolio.total_return += trade.pnl
    portfolio.total_return_percent = ((portfolio.current_cash + sum(
        t.current_price * t.quantity for t in MockTrade.query.filter_by(
            user_id=current_user.id, is_open=True
        ).all()
    )) / portfolio.starting_cash - 1) * 100

    if trade.pnl > 0:
        portfolio.winning_trades += 1
    else:
        portfolio.losing_trades += 1

    if portfolio.total_trades > 0:
        portfolio.win_rate = portfolio.winning_trades / portfolio.total_trades

    db.session.commit()

    return jsonify({
        'success': True,
        'message': f'Closed {trade.ticker} position',
        'trade': trade.to_dict(),
        'pnl': trade.pnl,
        'pnl_percent': trade.pnl_percent
    })


@auth_bp.route('/api/portfolio', methods=['GET'])
@login_required
def get_portfolio():
    """Get user's portfolio stats"""
    portfolio = Portfolio.query.filter_by(user_id=current_user.id).first()

    if not portfolio:
        portfolio = Portfolio(user_id=current_user.id)
        db.session.add(portfolio)
        db.session.commit()

    # Get open positions
    open_trades = MockTrade.query.filter_by(
        user_id=current_user.id,
        is_open=True
    ).all()

    # Get closed trades
    closed_trades = MockTrade.query.filter_by(
        user_id=current_user.id,
        is_open=False
    ).order_by(MockTrade.closed_at.desc()).limit(10).all()

    return jsonify({
        'success': True,
        'portfolio': portfolio.to_dict(),
        'open_positions': [trade.to_dict() for trade in open_trades],
        'recent_closed': [trade.to_dict() for trade in closed_trades]
    })


@auth_bp.route('/api/portfolio/reset', methods=['POST'])
@login_required
def reset_portfolio():
    """Reset portfolio to starting state"""
    portfolio = Portfolio.query.filter_by(user_id=current_user.id).first()

    if portfolio:
        # Close all open trades
        MockTrade.query.filter_by(
            user_id=current_user.id,
            is_open=True
        ).delete()

        # Reset portfolio
        portfolio.current_cash = portfolio.starting_cash
        portfolio.total_value = portfolio.starting_cash
        portfolio.total_return = 0.0
        portfolio.total_return_percent = 0.0
        portfolio.win_rate = 0.0
        portfolio.total_trades = 0
        portfolio.winning_trades = 0
        portfolio.losing_trades = 0
        portfolio.max_drawdown = 0.0
        portfolio.sharpe_ratio = 0.0

        db.session.commit()

    return jsonify({
        'success': True,
        'message': 'Portfolio reset to $100,000'
    })


# ========== Social Sentiment & Push Notifications ==========

@auth_bp.route('/api/sentiment/<ticker>', methods=['GET'])
def get_sentiment(ticker):
    """Get social sentiment analysis for a ticker"""
    try:
        from src.sentiment import SocialSentimentMonitor

        monitor = SocialSentimentMonitor()
        analysis = monitor.analyze_ticker(ticker.upper())

        return jsonify({
            'success': True,
            'sentiment': analysis
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@auth_bp.route('/api/push/subscribe', methods=['POST'])
def subscribe_push():
    """Subscribe to push notifications"""
    data = request.json

    if current_user.is_authenticated:
        # Save push subscription to user profile
        # In production, store subscription endpoint and keys
        return jsonify({
            'success': True,
            'message': 'Subscribed to push notifications'
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Please sign in to enable push notifications'
        }), 401


@auth_bp.route('/api/social/share', methods=['POST'])
def share_to_social():
    """Generate shareable content for social media"""
    data = request.json
    platform = data.get('platform')  # twitter, linkedin, reddit, facebook
    content_type = data.get('type')  # watchlist, trade, portfolio

    try:
        share_content = {
            'text': '',
            'url': '',
            'hashtags': []
        }

        if content_type == 'portfolio':
            # Portfolio performance share
            portfolio_data = data.get('data', {})
            total_return = portfolio_data.get('total_return_percent', 0)
            win_rate = portfolio_data.get('win_rate', 0)

            share_content['text'] = f"📊 My ML Stock Trading Results:\n\n"
            share_content['text'] += f"{'📈' if total_return >= 0 else '📉'} Total Return: {total_return:+.2f}%\n"
            share_content['text'] += f"🎯 Win Rate: {win_rate*100:.1f}%\n\n"
            share_content['text'] += f"Powered by ML predictions & social sentiment analysis"
            share_content['hashtags'] = ['stocks', 'trading', 'ML', 'investing']

        elif content_type == 'trade':
            # Individual trade share
            trade_data = data.get('data', {})
            ticker = trade_data.get('ticker')
            action = trade_data.get('action')
            pnl_percent = trade_data.get('pnl_percent', 0)

            share_content['text'] = f"{'🚀' if pnl_percent > 0 else '💸'} ${ticker} {action} Trade Result:\n\n"
            share_content['text'] += f"P&L: {pnl_percent:+.2f}%\n\n"
            share_content['text'] += f"Using ML predictions + social sentiment"
            share_content['hashtags'] = [ticker, action.lower(), 'stocks', 'trading']

        elif content_type == 'watchlist':
            # Watchlist share
            watchlist_data = data.get('data', [])
            tickers = [item.get('ticker') for item in watchlist_data[:5]]

            share_content['text'] = f"👀 My Current Watchlist:\n\n"
            share_content['text'] += '\n'.join([f"${ticker}" for ticker in tickers])
            share_content['text'] += f"\n\nTracking {len(watchlist_data)} stocks with ML sentiment analysis"
            share_content['hashtags'] = ['watchlist', 'stocks', 'investing']

        # Platform-specific formatting
        if platform == 'twitter':
            share_content['url'] = f"https://twitter.com/intent/tweet?text={requests.utils.quote(share_content['text'])}&hashtags={','.join(share_content['hashtags'])}"
        elif platform == 'linkedin':
            share_content['url'] = f"https://www.linkedin.com/sharing/share-offsite/?url=YOUR_APP_URL"
        elif platform == 'reddit':
            share_content['text'] = share_content['text']  # Reddit uses text submission
        elif platform == 'facebook':
            share_content['url'] = f"https://www.facebook.com/sharer/sharer.php?u=YOUR_APP_URL"

        return jsonify({
            'success': True,
            'share_content': share_content
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
