"""
fastapi_ml_service.app package

This package exposes the FastAPI application instance so the server
can be started with `uvicorn fastapi_ml_service.app:app`.
"""

# Re-export the FastAPI `app` from the main module for easy imports
from .main import app  # noqa: F401

__all__ = ["app"]