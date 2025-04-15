"""
MLX Acceleration Module for LlamaDB.

This module integrates Apple's MLX framework to accelerate vector and matrix
operations on Apple Silicon hardware (M1/M2/M3). MLX provides significant
performance gains over NumPy for machine learning operations.

On systems without MLX or Apple Silicon, the module gracefully falls back
to NumPy-based implementations.
"""

import logging
import os
import platform
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# MLX availability flag
_MLX_AVAILABLE = False
_APPLE_SILICON = False


# Check if we're on Apple Silicon
def is_apple_silicon() -> bool:
    """Check if the current system is running on Apple Silicon."""
    global _APPLE_SILICON
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        _APPLE_SILICON = True
        return True
    return False


# Try to import MLX if available
if is_apple_silicon():
    try:
        import mlx.core
        import mlx.nn

        _MLX_AVAILABLE = True
        logger.info("🚀 MLX acceleration enabled on Apple Silicon")
    except ImportError:
        logger.warning("⚠️ MLX package not found. Using NumPy fallback.")
        logger.info(
            "🔍 To enable MLX acceleration, install the MLX package: pip install mlx"
        )
else:
    logger.info("⚠️ MLX acceleration not available (requires Apple Silicon)")


def is_mlx_available() -> bool:
    """Check if MLX is available on this system."""
    return _MLX_AVAILABLE


def install_mlx() -> bool:
    """
    Attempt to install MLX if on Apple Silicon.

    Returns:
        bool: True if installation was successful or MLX was already installed
    """
    global _MLX_AVAILABLE

    if not is_apple_silicon():
        logger.warning("Cannot install MLX on non-Apple Silicon hardware")
        return False

    if _MLX_AVAILABLE:
        logger.info("MLX is already installed")
        return True

    try:
        logger.info("Installing MLX library...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "mlx"],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("MLX installed successfully")
        try:
            # Try to import MLX after installation
            import mlx.core
            import mlx.nn

            _MLX_AVAILABLE = True
            return True
        except ImportError:
            logger.error("Failed to import MLX after installation")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install MLX: {e.stderr}")
        return False


# ====== Vector Operations ======


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
    """
    Calculate cosine similarity between vectors.

    Args:
        a: First vector or matrix of vectors (numpy array)
        b: Second vector or matrix of vectors (numpy array)

    Returns:
        Cosine similarity score(s)
    """
    if _MLX_AVAILABLE:
        import mlx.core

        # Convert NumPy arrays to MLX arrays
        a_mx = mlx.core.array(a)
        b_mx = mlx.core.array(b)

        # Normalize the vectors
        a_norm = mlx.core.sqrt(mlx.core.sum(a_mx * a_mx, axis=-1, keepdims=True))
        b_norm = mlx.core.sqrt(mlx.core.sum(b_mx * b_mx, axis=-1, keepdims=True))

        # Avoid division by zero
        a_mx = mlx.core.where(a_norm > 0, a_mx / a_norm, 0)
        b_mx = mlx.core.where(b_norm > 0, b_mx / b_norm, 0)

        # Calculate cosine similarity
        similarity = mlx.core.sum(a_mx * b_mx, axis=-1)

        # Convert back to numpy for compatibility
        return np.array(similarity)
    else:
        # NumPy fallback implementation
        a_norm = np.sqrt(np.sum(a * a, axis=-1, keepdims=True))
        b_norm = np.sqrt(np.sum(b * b, axis=-1, keepdims=True))

        # Avoid division by zero
        a_normalized = np.divide(a, a_norm, out=np.zeros_like(a), where=a_norm != 0)
        b_normalized = np.divide(b, b_norm, out=np.zeros_like(b), where=b_norm != 0)

        return np.sum(a_normalized * b_normalized, axis=-1)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
    """
    Calculate Euclidean distance between vectors.

    Args:
        a: First vector or matrix of vectors (numpy array)
        b: Second vector or matrix of vectors (numpy array)

    Returns:
        Euclidean distance(s)
    """
    if _MLX_AVAILABLE:
        import mlx.core

        # Convert NumPy arrays to MLX arrays
        a_mx = mlx.core.array(a)
        b_mx = mlx.core.array(b)

        # Calculate squared difference
        diff = a_mx - b_mx
        squared_diff = diff * diff

        # Calculate Euclidean distance
        distance = mlx.core.sqrt(mlx.core.sum(squared_diff, axis=-1))

        # Convert back to numpy for compatibility
        return np.array(distance)
    else:
        # NumPy fallback implementation
        return np.sqrt(np.sum((a - b) ** 2, axis=-1))


def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Perform matrix multiplication, accelerated with MLX if available.

    Args:
        a: First matrix (numpy array)
        b: Second matrix (numpy array)

    Returns:
        Resulting matrix as numpy array
    """
    if _MLX_AVAILABLE:
        import mlx.core

        # Convert NumPy arrays to MLX arrays
        a_mx = mlx.core.array(a)
        b_mx = mlx.core.array(b)

        # Perform matrix multiplication
        result = mlx.core.matmul(a_mx, b_mx)

        # Convert back to numpy for compatibility
        return np.array(result)
    else:
        # NumPy fallback implementation
        return np.matmul(a, b)


def batch_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize a batch of vectors to unit length.

    Args:
        vectors: Matrix of vectors to normalize (numpy array)

    Returns:
        Normalized vectors as numpy array
    """
    if _MLX_AVAILABLE:
        import mlx.core

        # Convert NumPy array to MLX array
        vectors_mx = mlx.core.array(vectors)

        # Calculate norms
        norms = mlx.core.sqrt(
            mlx.core.sum(vectors_mx * vectors_mx, axis=1, keepdims=True)
        )

        # Avoid division by zero
        normalized = mlx.core.where(norms > 0, vectors_mx / norms, 0)

        # Convert back to numpy for compatibility
        return np.array(normalized)
    else:
        # NumPy fallback implementation
        norms = np.sqrt(np.sum(vectors * vectors, axis=1, keepdims=True))
        return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)


# ====== Performance Benchmarking ======


def benchmark_matrix_multiply(
    size: int = 1000, iterations: int = 5
) -> Dict[str, float]:
    """
    Benchmark matrix multiplication performance.

    Args:
        size: Size of square matrices to multiply
        iterations: Number of iterations to run

    Returns:
        Dictionary with performance results
    """
    import time

    # Generate random matrices
    a = np.random.random((size, size)).astype(np.float32)
    b = np.random.random((size, size)).astype(np.float32)

    # Benchmark NumPy
    numpy_times = []
    for _ in range(iterations):
        start = time.time()
        _ = np.matmul(a, b)
        numpy_times.append(time.time() - start)
    numpy_avg = sum(numpy_times) / len(numpy_times)

    results = {
        "numpy_time": numpy_avg,
        "numpy_gflops": (2 * size**3) / (numpy_avg * 1e9),
    }

    # Benchmark MLX if available
    if _MLX_AVAILABLE:
        import mlx.core

        a_mx = mlx.core.array(a)
        b_mx = mlx.core.array(b)

        # Warm-up
        _ = mlx.core.matmul(a_mx, b_mx)
        mlx.core.eval()

        mlx_times = []
        for _ in range(iterations):
            start = time.time()
            _ = mlx.core.matmul(a_mx, b_mx)
            mlx.core.eval()  # Force evaluation
            mlx_times.append(time.time() - start)
        mlx_avg = sum(mlx_times) / len(mlx_times)

        results.update(
            {
                "mlx_time": mlx_avg,
                "mlx_gflops": (2 * size**3) / (mlx_avg * 1e9),
                "speedup": numpy_avg / mlx_avg,
            }
        )

    return results


def benchmark_vector_operations(
    dim: int = 1024, batch_size: int = 10000
) -> Dict[str, float]:
    """
    Benchmark vector operations performance.

    Args:
        dim: Dimensionality of vectors
        batch_size: Number of vectors to process

    Returns:
        Dictionary with performance results
    """
    global _MLX_AVAILABLE

    import time

    # Generate random vectors
    vectors = np.random.random((batch_size, dim)).astype(np.float32)
    query = np.random.random(dim).astype(np.float32)

    # Benchmark NumPy implementation
    start = time.time()
    _ = cosine_similarity(query, vectors)
    numpy_time = time.time() - start

    results = {
        "numpy_time": numpy_time,
    }

    # Force MLX path and benchmark if available
    if _MLX_AVAILABLE:
        # Store the original value in a local variable
        original_mlx_available = _MLX_AVAILABLE
        _MLX_AVAILABLE = True

        start = time.time()
        _ = cosine_similarity(query, vectors)
        mlx_time = time.time() - start

        # Restore original flag
        _MLX_AVAILABLE = original_mlx_available

        results.update(
            {
                "mlx_time": mlx_time,
                "speedup": numpy_time / mlx_time,
                "vectors_per_second": batch_size / mlx_time,
            }
        )
    else:
        results.update({"vectors_per_second": batch_size / numpy_time})

    return results


# ====== System Information ======


def get_system_info() -> Dict[str, str]:
    """
    Get system information.

    Returns:
        Dictionary with system information
    """
    import sys

    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "is_apple_silicon": str(is_apple_silicon()),
        "mlx_available": str(_MLX_AVAILABLE),
    }

    if _MLX_AVAILABLE:
        import mlx

        try:
            info["mlx_version"] = mlx.__version__
        except AttributeError:
            # Try alternative ways to get the version
            try:
                import pkg_resources

                info["mlx_version"] = pkg_resources.get_distribution("mlx").version
            except:
                info["mlx_version"] = "unknown"

    return info
