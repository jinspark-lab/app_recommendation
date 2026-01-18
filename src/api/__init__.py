"""Flask API module for recommendation service."""

from .app import app, initialize_models

__all__ = ['app', 'initialize_models']
