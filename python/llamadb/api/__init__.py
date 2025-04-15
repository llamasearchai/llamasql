"""
LlamaDB API module.

This module provides a REST API for LlamaDB using FastAPI.
"""

from llamadb.api.app import create_app

__all__ = ["create_app"]
