"""
LlamaDB: High-Performance Vector Database with Python/Rust/MLX

LlamaDB is an enterprise-grade vector database built for AI workloads, featuring a hybrid 
architecture that combines Python, Rust, and MLX acceleration. It offers unparalleled 
performance for retrieval augmented generation (RAG), semantic search, and real-time embeddings.

Key Features:
- Hybrid Architecture: Core Python interfaces with Rust extensions
- MLX Acceleration: Uses Apple's MLX framework on Apple Silicon
- Claude AI Integration: First-class integration with Anthropic's Claude models
- Advanced Vector Processing: Optimized SIMD vector operations
- Enterprise Features: Fine-grained access control and monitoring
"""

import os
import sys
from typing import Dict, Any, Optional

# Version information
__version__ = "0.1.0"

# Import core components
from llamadb.core import (
    is_apple_silicon,
    is_mlx_available,
    VectorIndex
)

# Configure logging
from llamadb.utils.logging import configure_root_logger

# Set up logging based on environment variables
log_level = os.environ.get("LLAMADB_LOG_LEVEL", "INFO")
log_file = os.environ.get("LLAMADB_LOG_FILE")
configure_root_logger(level=log_level, log_file=log_file)

# Export public API
__all__ = [
    "VectorIndex",
    "is_apple_silicon",
    "is_mlx_available",
    "__version__"
]
