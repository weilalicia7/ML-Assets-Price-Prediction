"""
OAuth authentication setup for Google, GitHub, and Reddit
"""

from flask_dance.contrib.google import make_google_blueprint
from flask_dance.contrib.github import make_github_blueprint
from flask_dance.contrib.reddit import make_reddit_blueprint
from flask_dance.consumer import oauth_authorized
from flask_login import login_user, current_user
from sqlalchemy.orm.exc import NoResultFound
import os

from .models import db, User


def setup_oauth(app):
    """Initialize OAuth blueprints"""

    # Google OAuth
    google_bp = make_google_blueprint(
        client_id=os.getenv('GOOGLE_OAUTH_CLIENT_ID', 'your-google-client-id'),
        client_secret=os.getenv('GOOGLE_OAUTH_CLIENT_SECRET', 'your-google-secret'),
        scope=['profile', 'email'],
        redirect_to='google_login'
    )
    app.register_blueprint(google_bp, url_prefix='/login')

    # GitHub OAuth
    github_bp = make_github_blueprint(
        client_id=os.getenv('GITHUB_OAUTH_CLIENT_ID', 'your-github-client-id'),
        client_secret=os.getenv('GITHUB_OAUTH_CLIENT_SECRET', 'your-github-secret'),
        scope='user:email',
        redirect_to='github_login'
    )
    app.register_blueprint(github_bp, url_prefix='/login')

    # Reddit OAuth
    reddit_bp = make_reddit_blueprint(
        client_id=os.getenv('REDDIT_OAUTH_CLIENT_ID', 'your-reddit-client-id'),
        client_secret=os.getenv('REDDIT_OAUTH_CLIENT_SECRET', 'your-reddit-secret'),
        scope=['identity'],
        redirect_to='reddit_login'
    )
    app.register_blueprint(reddit_bp, url_prefix='/login')

    return google_bp, github_bp, reddit_bp


def get_or_create_user(oauth_provider, oauth_id, email, username, full_name=None, avatar_url=None):
    """Get existing user or create new one from OAuth data"""

    try:
        # Try to find user by OAuth ID
        user = User.query.filter_by(
            oauth_provider=oauth_provider,
            oauth_id=oauth_id
        ).one()

        # Update last login
        from datetime import datetime
        user.last_login = datetime.utcnow()
        db.session.commit()

        return user

    except NoResultFound:
        # Check if user exists with this email
        if email:
            user = User.query.filter_by(email=email).first()
            if user:
                # Link OAuth account to existing user
                user.oauth_provider = oauth_provider
                user.oauth_id = oauth_id
                db.session.commit()
                return user

        # Create new user
        user = User(
            oauth_provider=oauth_provider,
            oauth_id=oauth_id,
            email=email,
            username=username or email.split('@')[0],
            full_name=full_name,
            avatar_url=avatar_url
        )

        db.session.add(user)

        # Create empty portfolio for new user
        from .models import Portfolio
        portfolio = Portfolio(user_id=user.id)
        db.session.add(portfolio)

        db.session.commit()

        return user
