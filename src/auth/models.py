"""
Database models for user authentication and data persistence
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
import json

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """User account model"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=True)  # For email/password login

    # OAuth provider info
    oauth_provider = db.Column(db.String(50), nullable=True)  # 'google', 'github', 'reddit'
    oauth_id = db.Column(db.String(200), nullable=True)  # Provider's user ID

    # Profile info
    full_name = db.Column(db.String(200), nullable=True)
    avatar_url = db.Column(db.String(500), nullable=True)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    watchlists = db.relationship('Watchlist', backref='user', lazy=True, cascade='all, delete-orphan')
    trades = db.relationship('MockTrade', backref='user', lazy=True, cascade='all, delete-orphan')
    portfolio = db.relationship('Portfolio', backref='user', uselist=False, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<User {self.username}>'


class Watchlist(db.Model):
    """User's stock watchlist"""
    __tablename__ = 'watchlists'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    ticker = db.Column(db.String(20), nullable=False)
    name = db.Column(db.String(200), nullable=True)
    exchange = db.Column(db.String(50), nullable=True)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text, nullable=True)

    # Alert settings
    alert_price_above = db.Column(db.Float, nullable=True)
    alert_price_below = db.Column(db.Float, nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'ticker': self.ticker,
            'name': self.name,
            'exchange': self.exchange,
            'added_at': self.added_at.isoformat(),
            'notes': self.notes,
            'alert_price_above': self.alert_price_above,
            'alert_price_below': self.alert_price_below
        }


class MockTrade(db.Model):
    """Mock trading positions"""
    __tablename__ = 'mock_trades'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    ticker = db.Column(db.String(20), nullable=False)
    action = db.Column(db.String(10), nullable=False)  # 'BUY' or 'SELL'
    quantity = db.Column(db.Float, nullable=False)
    entry_price = db.Column(db.Float, nullable=False)
    current_price = db.Column(db.Float, nullable=True)

    # P&L tracking
    pnl = db.Column(db.Float, default=0.0)
    pnl_percent = db.Column(db.Float, default=0.0)

    # Timestamps
    opened_at = db.Column(db.DateTime, default=datetime.utcnow)
    closed_at = db.Column(db.DateTime, nullable=True)
    is_open = db.Column(db.Boolean, default=True)

    # ML prediction context
    predicted_return = db.Column(db.Float, nullable=True)
    prediction_confidence = db.Column(db.Float, nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'ticker': self.ticker,
            'action': self.action,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'opened_at': self.opened_at.isoformat(),
            'closed_at': self.closed_at.isoformat() if self.closed_at else None,
            'is_open': self.is_open,
            'predicted_return': self.predicted_return,
            'prediction_confidence': self.prediction_confidence
        }


class Portfolio(db.Model):
    """User's overall portfolio stats"""
    __tablename__ = 'portfolios'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)

    # Portfolio value
    starting_cash = db.Column(db.Float, default=100000.0)
    current_cash = db.Column(db.Float, default=100000.0)
    total_value = db.Column(db.Float, default=100000.0)

    # Performance metrics
    total_return = db.Column(db.Float, default=0.0)
    total_return_percent = db.Column(db.Float, default=0.0)
    win_rate = db.Column(db.Float, default=0.0)
    total_trades = db.Column(db.Integer, default=0)
    winning_trades = db.Column(db.Integer, default=0)
    losing_trades = db.Column(db.Integer, default=0)

    # Risk metrics
    max_drawdown = db.Column(db.Float, default=0.0)
    sharpe_ratio = db.Column(db.Float, default=0.0)

    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'starting_cash': self.starting_cash,
            'current_cash': self.current_cash,
            'total_value': self.total_value,
            'total_return': self.total_return,
            'total_return_percent': self.total_return_percent,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio
        }


class SharedPost(db.Model):
    """Shared trading posts to social media"""
    __tablename__ = 'shared_posts'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    platform = db.Column(db.String(50), nullable=False)  # 'tiktok', 'instagram', 'x', 'facebook', 'linkedin'

    # Post content
    ticker = db.Column(db.String(20), nullable=False)
    prediction_data = db.Column(db.Text, nullable=True)  # JSON data
    screenshot_url = db.Column(db.String(500), nullable=True)

    # Social media post IDs
    post_id = db.Column(db.String(200), nullable=True)
    post_url = db.Column(db.String(500), nullable=True)

    # Engagement metrics
    likes = db.Column(db.Integer, default=0)
    shares = db.Column(db.Integer, default=0)
    comments = db.Column(db.Integer, default=0)

    shared_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'platform': self.platform,
            'ticker': self.ticker,
            'post_url': self.post_url,
            'likes': self.likes,
            'shares': self.shares,
            'comments': self.comments,
            'shared_at': self.shared_at.isoformat()
        }
