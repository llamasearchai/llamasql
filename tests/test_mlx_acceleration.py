"""
Tests for the MLX acceleration module.

These tests verify the functionality of the MLX acceleration module,
with graceful fallback to NumPy when MLX is not available.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Add the parent directory to the path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.llamadb.core.mlx_acceleration import (
    batch_normalize,
    cosine_similarity,
    euclidean_distance,
    is_apple_silicon,
    is_mlx_available,
    matrix_multiply,
)


# Test data
@pytest.fixture
def vector_data():
    """Create test vector data."""
    # Create two vectors
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

    # Create a batch of vectors
    batch_a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    batch_b = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32)

    return {
        "a": a,
        "b": b,
        "batch_a": batch_a,
        "batch_b": batch_b,
    }


@pytest.fixture
def matrix_data():
    """Create test matrix data."""
    # Create two matrices
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    return {
        "a": a,
        "b": b,
    }


def test_is_apple_silicon():
    """Test the is_apple_silicon function."""
    # This is just a basic test to ensure the function runs
    result = is_apple_silicon()
    assert isinstance(result, bool)

    # The actual result depends on the hardware
    if sys.platform == "darwin" and "arm" in os.uname().machine:
        assert result is True
    else:
        assert result is False


def test_is_mlx_available():
    """Test the is_mlx_available function."""
    # This is just a basic test to ensure the function runs
    result = is_mlx_available()
    assert isinstance(result, bool)

    # The actual result depends on the hardware and whether MLX is installed
    # We can't make assumptions about the result


def test_cosine_similarity(vector_data):
    """Test the cosine_similarity function."""
    a = vector_data["a"]
    b = vector_data["b"]

    # Calculate cosine similarity
    similarity = cosine_similarity(a, b)

    # Calculate expected result using NumPy
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    expected = np.dot(a, b) / (a_norm * b_norm)

    # Check that the result is close to the expected value
    assert np.isclose(similarity, expected)

    # Test with batch
    batch_a = vector_data["batch_a"]
    batch_b = vector_data["batch_b"]

    # Calculate batch cosine similarity
    batch_similarity = cosine_similarity(batch_a[0], batch_b)

    # Check that the result has the expected shape
    assert batch_similarity.shape == (2,)


def test_euclidean_distance(vector_data):
    """Test the euclidean_distance function."""
    a = vector_data["a"]
    b = vector_data["b"]

    # Calculate Euclidean distance
    distance = euclidean_distance(a, b)

    # Calculate expected result using NumPy
    expected = np.sqrt(np.sum((a - b) ** 2))

    # Check that the result is close to the expected value
    assert np.isclose(distance, expected)

    # Test with batch
    batch_a = vector_data["batch_a"]
    batch_b = vector_data["batch_b"]

    # Calculate batch Euclidean distance
    batch_distance = euclidean_distance(batch_a[0], batch_b)

    # Check that the result has the expected shape
    assert batch_distance.shape == (2,)


def test_matrix_multiply(matrix_data):
    """Test the matrix_multiply function."""
    a = matrix_data["a"]
    b = matrix_data["b"]

    # Calculate matrix multiplication
    result = matrix_multiply(a, b)

    # Calculate expected result using NumPy
    expected = np.matmul(a, b)

    # Check that the result is close to the expected value
    assert np.allclose(result, expected)

    # Check that the result has the expected shape
    assert result.shape == expected.shape


def test_batch_normalize(vector_data):
    """Test the batch_normalize function."""
    batch = vector_data["batch_a"]

    # Normalize the batch
    normalized = batch_normalize(batch)

    # Check that the result has the expected shape
    assert normalized.shape == batch.shape

    # Check that each vector has unit norm
    for i in range(len(normalized)):
        norm = np.linalg.norm(normalized[i])
        assert np.isclose(norm, 1.0)


def test_performance_comparison():
    """Test that MLX is faster than NumPy when available."""
    # Skip this test if MLX is not available
    if not is_mlx_available():
        pytest.skip("MLX is not available")

    # Import the benchmark functions
    from python.llamadb.core.mlx_acceleration import (
        benchmark_matrix_multiply,
        benchmark_vector_operations,
    )

    # Run matrix multiplication benchmark
    matrix_result = benchmark_matrix_multiply(size=1000, iterations=1)

    # Run vector operations benchmark
    vector_result = benchmark_vector_operations(dim=128, batch_size=10000)

    # Check that MLX is faster than NumPy
    if "mlx_time" in matrix_result and "numpy_time" in matrix_result:
        assert matrix_result["mlx_time"] < matrix_result["numpy_time"]

    if "mlx_time" in vector_result and "numpy_time" in vector_result:
        assert vector_result["mlx_time"] < vector_result["numpy_time"]
