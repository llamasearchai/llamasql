#!/usr/bin/env python
"""
LlamaDB Vector Index Demo

This script demonstrates the high-performance vector index capabilities of LlamaDB,
including MLX acceleration on Apple Silicon.
"""

import argparse
import os
import time

import numpy as np

# Import LlamaDB components
from llamadb.core import (
    VectorIndex,
    benchmark_matrix_multiply,
    is_apple_silicon,
    is_mlx_available,
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def print_header():
    """Print a stylized header."""
    console.print("\n[bold blue]LlamaDB Vector Index Demo[/bold blue]")
    console.print("Showcasing high-performance vector search with MLX acceleration\n")


def show_system_info():
    """Display system information."""
    console.print("[bold cyan]System Information[/bold cyan]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Property")
    table.add_column("Value")

    table.add_row("Apple Silicon", str(is_apple_silicon()))
    table.add_row("MLX Available", str(is_mlx_available()))

    console.print(table)
    console.print()


def benchmark_matrix_operations():
    """Benchmark matrix operations with and without MLX."""
    console.print("[bold cyan]Matrix Operation Benchmarks[/bold cyan]")

    sizes = [1000, 2000, 5000]

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Matrix Size")
    table.add_column("NumPy Time (ms)")
    table.add_column("NumPy GFLOPs")

    if is_mlx_available():
        table.add_column("MLX Time (ms)")
        table.add_column("MLX GFLOPs")
        table.add_column("Speedup")

    for size in sizes:
        console.print(f"Benchmarking {size}x{size} matrix multiplication...")
        results = benchmark_matrix_multiply(size=size, iterations=3)

        row = [
            f"{size}x{size}",
            f"{results['numpy_time'] * 1000:.2f}",
            f"{results['numpy_gflops']:.2f}",
        ]

        if is_mlx_available() and "mlx_time" in results:
            row.extend(
                [
                    f"{results['mlx_time'] * 1000:.2f}",
                    f"{results['mlx_gflops']:.2f}",
                    f"{results['speedup']:.2f}x",
                ]
            )

        table.add_row(*row)

    console.print(table)
    console.print()


def create_sample_vectors(num_vectors=100000, dimension=128):
    """Create sample vectors for the demo."""
    console.print(f"Creating {num_vectors} sample vectors of dimension {dimension}...")

    vectors = np.random.random((num_vectors, dimension)).astype(np.float32)
    metadata = [
        {"id": i, "category": f"category_{i % 10}", "value": float(np.random.random())}
        for i in range(num_vectors)
    ]

    return vectors, metadata


def benchmark_vector_index(num_vectors=100000, dimension=128, k=10):
    """Benchmark vector index operations."""
    console.print("[bold cyan]Vector Index Benchmarks[/bold cyan]")

    # Create sample data
    vectors, metadata = create_sample_vectors(num_vectors, dimension)

    # Create query vector
    query = np.random.random(dimension).astype(np.float32)

    # Benchmark with MLX
    if is_mlx_available():
        console.print("Benchmarking with MLX acceleration...")

        # Create index
        start_time = time.time()
        index_mlx = VectorIndex(dimension=dimension, metric="cosine", use_mlx=True)
        index_mlx.add(vectors, metadata)
        index_time_mlx = time.time() - start_time

        # Search
        start_time = time.time()
        results_mlx = index_mlx.search(query_vector=query, k=k)
        search_time_mlx = time.time() - start_time

    # Benchmark without MLX
    console.print("Benchmarking without MLX acceleration...")

    # Create index
    start_time = time.time()
    index_numpy = VectorIndex(dimension=dimension, metric="cosine", use_mlx=False)
    index_numpy.add(vectors, metadata)
    index_time_numpy = time.time() - start_time

    # Search
    start_time = time.time()
    results_numpy = index_numpy.search(query_vector=query, k=k)
    search_time_numpy = time.time() - start_time

    # Display results
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Operation")
    table.add_column("NumPy Time (ms)")
    table.add_column("Vectors/sec")

    if is_mlx_available():
        table.add_column("MLX Time (ms)")
        table.add_column("MLX Vectors/sec")
        table.add_column("Speedup")

    # Add index results
    index_row = [
        "Index Creation",
        f"{index_time_numpy * 1000:.2f}",
        f"{num_vectors / index_time_numpy:.2f}",
    ]

    if is_mlx_available():
        index_row.extend(
            [
                f"{index_time_mlx * 1000:.2f}",
                f"{num_vectors / index_time_mlx:.2f}",
                f"{index_time_numpy / index_time_mlx:.2f}x",
            ]
        )

    table.add_row(*index_row)

    # Add search results
    search_row = [
        f"Vector Search (k={k})",
        f"{search_time_numpy * 1000:.2f}",
        f"{num_vectors / search_time_numpy:.2f}",
    ]

    if is_mlx_available():
        search_row.extend(
            [
                f"{search_time_mlx * 1000:.2f}",
                f"{num_vectors / search_time_mlx:.2f}",
                f"{search_time_numpy / search_time_mlx:.2f}x",
            ]
        )

    table.add_row(*search_row)

    console.print(table)
    console.print()

    # Show top results
    console.print("[bold cyan]Top Search Results[/bold cyan]")

    results_table = Table(show_header=True, header_style="bold cyan")
    results_table.add_column("Rank")
    results_table.add_column("ID")
    results_table.add_column("Score")
    results_table.add_column("Category")

    for i, result in enumerate(results_numpy[:5]):
        results_table.add_row(
            str(i + 1),
            str(result["id"]),
            f"{result['score']:.4f}",
            result["metadata"]["category"],
        )

    console.print(results_table)
    console.print()


def main():
    """Run the demo."""
    parser = argparse.ArgumentParser(description="LlamaDB Vector Index Demo")
    parser.add_argument(
        "--num-vectors", type=int, default=100000, help="Number of vectors to create"
    )
    parser.add_argument(
        "--dimension", type=int, default=128, help="Dimension of vectors"
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Number of results to return in search"
    )
    args = parser.parse_args()

    print_header()
    show_system_info()
    benchmark_matrix_operations()
    benchmark_vector_index(
        num_vectors=args.num_vectors, dimension=args.dimension, k=args.k
    )

    console.print("[bold green]Demo completed successfully![/bold green]")
    return 0


if __name__ == "__main__":
    exit(main())
