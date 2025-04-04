#!/usr/bin/env python
"""
Comprehensive test script for LlamaDB functionality
This script validates core functionality of LlamaDB
"""

import argparse
import numpy as np
import time
import os
import sys
import json
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt

# Handle import paths
try:
    from llamadb.core import (
        VectorIndex,
        cosine_similarity,
        dot_product,
        l2_distance,
        matrix_multiply,
        is_mlx_available,
        is_apple_silicon
    )
except ImportError:
    print("Failed to import LlamaDB modules. Make sure LlamaDB is installed.")
    print("Try: pip install -e .")
    sys.exit(1)

# Set up argument parser
parser = argparse.ArgumentParser(description="Test LlamaDB functionality")
parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
parser.add_argument("--plot", action="store_true", help="Generate performance plots")
parser.add_argument("--dimensions", type=int, default=1536, help="Vector dimensions for tests")
parser.add_argument("--count", type=int, default=10000, help="Number of vectors for tests")
args = parser.parse_args()

def print_header(title: str) -> None:
    """Print a formatted header for test sections"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_result(name: str, success: bool, details: str = "") -> None:
    """Print formatted test result"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {name}{': ' + details if details else ''}")

def test_environment() -> None:
    """Test and report on the execution environment"""
    print_header("Environment Information")
    
    # Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    
    # Check if running on Apple Silicon
    apple_silicon = is_apple_silicon()
    print(f"Apple Silicon: {'Yes' if apple_silicon else 'No'}")
    
    # Check if MLX is available
    mlx_available = is_mlx_available()
    print(f"MLX Acceleration: {'Available' if mlx_available else 'Not available'}")
    
    # NumPy version
    numpy_version = np.__version__
    print(f"NumPy version: {numpy_version}")
    
    # OS information
    import platform
    os_info = f"{platform.system()} {platform.release()}"
    print(f"Operating System: {os_info}")

def test_vector_operations() -> None:
    """Test basic vector operations"""
    print_header("Testing Vector Operations")
    
    # Generate random vectors
    dim = 4  # Small dimension for verification
    v1 = np.array([1.0, 2.0, 3.0, 4.0])
    v2 = np.array([5.0, 6.0, 7.0, 8.0])
    
    # Test cosine similarity
    cos_sim = cosine_similarity(v1, v2)
    expected_cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_sim_correct = abs(cos_sim - expected_cos) < 1e-6
    print_result("Cosine Similarity", cos_sim_correct, f"Got {cos_sim:.6f}, Expected {expected_cos:.6f}")
    
    # Test dot product
    dot = dot_product(v1, v2)
    expected_dot = np.dot(v1, v2)
    dot_correct = abs(dot - expected_dot) < 1e-6
    print_result("Dot Product", dot_correct, f"Got {dot:.6f}, Expected {expected_dot:.6f}")
    
    # Test L2 distance
    l2 = l2_distance(v1, v2)
    expected_l2 = np.linalg.norm(v1 - v2)
    l2_correct = abs(l2 - expected_l2) < 1e-6
    print_result("L2 Distance", l2_correct, f"Got {l2:.6f}, Expected {expected_l2:.6f}")
    
    # Test matrix multiply
    m1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    m2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    result = matrix_multiply(m1, m2)
    expected = np.matmul(m1, m2)
    matmul_correct = np.allclose(result, expected)
    print_result("Matrix Multiply", matmul_correct)

def test_vector_index() -> None:
    """Test VectorIndex functionality"""
    print_header("Testing VectorIndex")
    
    # Create a small index
    dim = 4
    index = VectorIndex(dim)
    
    # Add vectors
    vectors = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0])
    ]
    metadata = [{"id": i, "name": f"Vector {i}"} for i in range(len(vectors))]
    
    for i, (vec, meta) in enumerate(zip(vectors, metadata)):
        index.add_item(vec, meta)
    
    print_result("Add Items", index.size() == len(vectors))
    
    # Test search
    query = np.array([1.0, 0.1, 0.1, 0.1])
    results = index.search(query, k=2)
    
    # First result should be vector 0
    search_correct = results[0]["id"] == 0
    print_result("Search Functionality", search_correct)
    
    # Test delete
    index.delete_item(0)
    print_result("Delete Item", index.size() == len(vectors) - 1)
    
    # Test search after delete
    results_after_delete = index.search(query, k=2)
    delete_correct = all(result["id"] != 0 for result in results_after_delete)
    print_result("Search After Delete", delete_correct)

def run_benchmarks() -> Dict[str, Any]:
    """Run performance benchmarks"""
    print_header(f"Running Benchmarks (dim={args.dimensions}, count={args.count})")
    
    results = {}
    
    # Generate random vectors
    vectors = np.random.rand(args.count, args.dimensions).astype(np.float32)
    query = np.random.rand(args.dimensions).astype(np.float32)
    
    # Benchmark index creation
    start_time = time.time()
    index = VectorIndex(args.dimensions)
    for i, vec in enumerate(vectors):
        index.add_item(vec, {"id": i})
    index_creation_time = time.time() - start_time
    results["index_creation"] = index_creation_time
    print(f"Index Creation: {index_creation_time:.4f} seconds")
    
    # Benchmark search
    iterations = 10
    search_times = []
    for _ in range(iterations):
        start_time = time.time()
        results_set = index.search(query, k=10)
        search_time = time.time() - start_time
        search_times.append(search_time)
    avg_search_time = sum(search_times) / len(search_times)
    results["search"] = avg_search_time
    print(f"Average Search Time: {avg_search_time:.6f} seconds")
    
    # Benchmark similarity calculations
    similarity_ops = {
        "cosine_similarity": cosine_similarity,
        "dot_product": dot_product,
        "l2_distance": l2_distance
    }
    
    similarity_times = {}
    iterations = 100
    for name, op in similarity_ops.items():
        start_time = time.time()
        for _ in range(iterations):
            op(query, vectors[0])
        op_time = (time.time() - start_time) / iterations
        similarity_times[name] = op_time
        print(f"{name}: {op_time:.8f} seconds per operation")
    
    results["similarity_ops"] = similarity_times
    
    # Benchmark matrix operations if we have enough vectors
    if args.count >= 100:
        subset_size = 100
        matrix_a = vectors[:subset_size]
        matrix_b = vectors[subset_size:subset_size*2].T if subset_size*2 <= args.count else vectors[:subset_size].T
        
        start_time = time.time()
        matrix_multiply(matrix_a, matrix_b)
        matrix_time = time.time() - start_time
        results["matrix_multiply"] = matrix_time
        print(f"Matrix Multiply ({subset_size}x{args.dimensions} * {args.dimensions}x{subset_size}): {matrix_time:.4f} seconds")
    
    return results

def plot_benchmark_results(results: Dict[str, Any]) -> None:
    """Generate plots for benchmark results"""
    if not args.plot:
        return
        
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Plot similarity operation times
    if "similarity_ops" in results:
        op_names = list(results["similarity_ops"].keys())
        op_times = [results["similarity_ops"][name] for name in op_names]
        
        plt.subplot(2, 1, 1)
        plt.bar(op_names, op_times)
        plt.title("Similarity Operation Performance")
        plt.ylabel("Time (seconds)")
        plt.yscale("log")
        
    # Plot index operations
    plt.subplot(2, 1, 2)
    ops = ["index_creation", "search"]
    times = [results.get(op, 0) for op in ops]
    
    plt.bar(ops, times)
    plt.title("Index Operations Performance")
    plt.ylabel("Time (seconds)")
    
    plt.tight_layout()
    output_file = "benchmark_results.png"
    plt.savefig(output_file)
    print(f"Benchmark plot saved to {output_file}")

def main() -> None:
    """Main function to run all tests"""
    test_environment()
    test_vector_operations()
    test_vector_index()
    
    benchmark_results = {}
    if args.benchmark:
        benchmark_results = run_benchmarks()
        
        # Save benchmark results to JSON
        with open("benchmark_results.json", "w") as f:
            json.dump(benchmark_results, f, indent=2)
        print("Benchmark results saved to benchmark_results.json")
    
    if args.plot and benchmark_results:
        try:
            plot_benchmark_results(benchmark_results)
        except ImportError:
            print("Plotting requires matplotlib. Install with: pip install matplotlib")

if __name__ == "__main__":
    main() 