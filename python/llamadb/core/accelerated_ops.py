"""
Accelerated Operations

This module provides accelerated vector operations using MLX on Apple Silicon
with automatic fallback to NumPy for other platforms.
"""

import os
import platform
import numpy as np
from typing import Union, List, Optional, Tuple, Any

from llamadb.utils.logging import get_logger

logger = get_logger(__name__)

# Check if running on Apple Silicon
def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"

# Check if MLX is available
def is_mlx_available() -> bool:
    """Check if MLX is available."""
    if not is_apple_silicon():
        return False
    
    try:
        import mlx.core
        return True
    except ImportError:
        return False

def cosine_similarity(
    query: np.ndarray,
    vectors: np.ndarray,
    use_mlx: bool = True
) -> np.ndarray:
    """
    Calculate cosine similarity between a query vector and a set of vectors.
    
    Args:
        query: Query vector of shape (dimension,)
        vectors: Matrix of vectors of shape (n_vectors, dimension)
        use_mlx: Whether to use MLX acceleration if available
        
    Returns:
        Array of similarities of shape (n_vectors,)
    """
    # Check if MLX should and can be used
    if use_mlx and is_mlx_available():
        try:
            import mlx.core as mx
            
            # Convert to MLX arrays
            query_mx = mx.array(query, dtype=mx.float32)
            vectors_mx = mx.array(vectors, dtype=mx.float32)
            
            # Normalize query
            query_norm = mx.sqrt(mx.sum(query_mx * query_mx))
            if query_norm > 0:
                query_mx = query_mx / query_norm
                
            # Normalize vectors
            vectors_norm = mx.sqrt(mx.sum(vectors_mx * vectors_mx, axis=1, keepdims=True))
            vectors_norm = mx.where(vectors_norm == 0, mx.ones_like(vectors_norm), vectors_norm)
            vectors_mx = vectors_mx / vectors_norm
            
            # Calculate similarities
            similarities = mx.sum(query_mx * vectors_mx, axis=1)
            
            # Convert back to numpy
            return np.array(similarities)
        except Exception as e:
            logger.warning(f"Error using MLX for cosine similarity: {e}. Falling back to NumPy.")
    
    # NumPy implementation
    # Normalize query
    query_norm = np.linalg.norm(query)
    if query_norm > 0:
        query = query / query_norm
        
    # Normalize vectors
    vectors_norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_norm = np.where(vectors_norm == 0, 1.0, vectors_norm)
    vectors = vectors / vectors_norm
    
    # Calculate similarities
    return np.sum(query * vectors, axis=1)

def l2_distance(
    query: np.ndarray,
    vectors: np.ndarray,
    use_mlx: bool = True
) -> np.ndarray:
    """
    Calculate L2 distance between a query vector and a set of vectors.
    
    Args:
        query: Query vector of shape (dimension,)
        vectors: Matrix of vectors of shape (n_vectors, dimension)
        use_mlx: Whether to use MLX acceleration if available
        
    Returns:
        Array of distances of shape (n_vectors,)
    """
    # Check if MLX should and can be used
    if use_mlx and is_mlx_available():
        try:
            import mlx.core as mx
            
            # Convert to MLX arrays
            query_mx = mx.array(query, dtype=mx.float32)
            vectors_mx = mx.array(vectors, dtype=mx.float32)
            
            # Calculate squared distances
            diff = query_mx - vectors_mx
            distances = mx.sum(diff * diff, axis=1)
            
            # Convert to similarities (higher is better)
            similarities = 1.0 / (1.0 + distances)
            
            # Convert back to numpy
            return np.array(similarities)
        except Exception as e:
            logger.warning(f"Error using MLX for L2 distance: {e}. Falling back to NumPy.")
    
    # NumPy implementation
    diff = query - vectors
    distances = np.sum(diff * diff, axis=1)
    
    # Convert to similarities (higher is better)
    return 1.0 / (1.0 + distances)

def dot_product(
    query: np.ndarray,
    vectors: np.ndarray,
    use_mlx: bool = True
) -> np.ndarray:
    """
    Calculate dot product between a query vector and a set of vectors.
    
    Args:
        query: Query vector of shape (dimension,)
        vectors: Matrix of vectors of shape (n_vectors, dimension)
        use_mlx: Whether to use MLX acceleration if available
        
    Returns:
        Array of dot products of shape (n_vectors,)
    """
    # Check if MLX should and can be used
    if use_mlx and is_mlx_available():
        try:
            import mlx.core as mx
            
            # Convert to MLX arrays
            query_mx = mx.array(query, dtype=mx.float32)
            vectors_mx = mx.array(vectors, dtype=mx.float32)
            
            # Calculate dot products
            dot_products = mx.sum(query_mx * vectors_mx, axis=1)
            
            # Convert back to numpy
            return np.array(dot_products)
        except Exception as e:
            logger.warning(f"Error using MLX for dot product: {e}. Falling back to NumPy.")
    
    # NumPy implementation
    return np.sum(query * vectors, axis=1)

def matrix_multiply(
    a: np.ndarray,
    b: np.ndarray,
    use_mlx: bool = True
) -> np.ndarray:
    """
    Multiply two matrices with optional MLX acceleration.
    
    Args:
        a: First matrix
        b: Second matrix
        use_mlx: Whether to use MLX acceleration if available
        
    Returns:
        Result of matrix multiplication
    """
    # Check if MLX should and can be used
    if use_mlx and is_mlx_available():
        try:
            import mlx.core as mx
            
            # Convert to MLX arrays
            a_mx = mx.array(a, dtype=mx.float32)
            b_mx = mx.array(b, dtype=mx.float32)
            
            # Multiply
            result_mx = mx.matmul(a_mx, b_mx)
            
            # Convert back to numpy
            return np.array(result_mx)
        except Exception as e:
            logger.warning(f"Error using MLX for matrix multiplication: {e}. Falling back to NumPy.")
    
    # NumPy implementation
    return np.matmul(a, b)

def benchmark_matrix_multiply(
    size: int = 1000,
    iterations: int = 5
) -> dict:
    """
    Benchmark matrix multiplication with NumPy and MLX.
    
    Args:
        size: Size of square matrices to multiply
        iterations: Number of iterations to run
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    # Create random matrices
    a = np.random.random((size, size)).astype(np.float32)
    b = np.random.random((size, size)).astype(np.float32)
    
    # Benchmark NumPy
    start = time.time()
    for _ in range(iterations):
        c = np.matmul(a, b)
    numpy_time = (time.time() - start) / iterations
    
    # Calculate GFLOPs
    flops = 2 * size**3  # Approximate FLOPs for matrix multiplication
    numpy_gflops = flops / (numpy_time * 1e9)
    
    results = {
        "numpy_time": numpy_time,
        "numpy_gflops": numpy_gflops
    }
    
    # Benchmark MLX if available
    if is_mlx_available():
        try:
            import mlx.core as mx
            
            # Convert to MLX arrays
            a_mx = mx.array(a)
            b_mx = mx.array(b)
            
            # Warm-up
            c_mx = mx.matmul(a_mx, b_mx)
            mx.eval(c_mx)
            
            # Benchmark
            start = time.time()
            for _ in range(iterations):
                c_mx = mx.matmul(a_mx, b_mx)
                mx.eval(c_mx)  # Force computation to complete
            mlx_time = (time.time() - start) / iterations
            
            # Calculate GFLOPs
            mlx_gflops = flops / (mlx_time * 1e9)
            
            # Add to results
            results["mlx_time"] = mlx_time
            results["mlx_gflops"] = mlx_gflops
            results["speedup"] = numpy_time / mlx_time
        except Exception as e:
            logger.warning(f"Error benchmarking MLX: {e}")
    
    return results 