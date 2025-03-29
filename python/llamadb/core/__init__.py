"""
LlamaDB Core Module

This module provides the core functionality of LlamaDB, including:
- Vector index for similarity search
- Accelerated operations with MLX
- Data structures and algorithms
"""

# Import MLX acceleration functions directly from mlx_acceleration
from llamadb.core.mlx_acceleration import (
    is_apple_silicon,
    is_mlx_available,
    benchmark_matrix_multiply,
    get_system_info,
    install_mlx,
    benchmark_vector_operations
)

# Import vector operations from accelerated_ops
from llamadb.core.accelerated_ops import (
    cosine_similarity,
    l2_distance,
    dot_product,
    matrix_multiply
)

# Import VectorIndex from vector_index
from llamadb.core.vector_index import VectorIndex

# Re-export all public symbols
__all__ = [
    # MLX acceleration related
    "is_apple_silicon",
    "is_mlx_available",
    "get_system_info",
    "install_mlx",
    
    # Vector operations
    "cosine_similarity",
    "l2_distance",
    "dot_product",
    "matrix_multiply",
    
    # Benchmarking functions
    "benchmark_matrix_multiply",
    "benchmark_vector_operations",
    
    # Data structures
    "VectorIndex"
] 