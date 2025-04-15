"""
Demo module for LlamaDB CLI.

This module implements the 'demo' subcommand for the LlamaDB CLI,
providing interactive demonstrations of LlamaDB's capabilities.
"""

import logging
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import typer
from llamadb.core.mlx_acceleration import (
    benchmark_vector_operations,
    get_system_info,
    is_apple_silicon,
    is_mlx_available,
)
from llamadb.core.vector_index import VectorIndex
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.syntax import Syntax
from rich.table import Table

# Configure logger
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()


def run_demo_command(
    demo_type: str = typer.Option(
        "full", "--type", "-t", help="Type of demo to run (full, acceleration, rag)"
    ),
    dimension: int = typer.Option(
        128, "--dimension", "-d", help="Dimension of vectors for benchmarks"
    ),
    num_vectors: int = typer.Option(
        50000, "--num-vectors", "-n", help="Number of vectors for benchmarks"
    ),
):
    """
    Run an interactive demonstration of LlamaDB's capabilities.
    """
    try:
        if demo_type == "full":
            run_full_demo(dimension, num_vectors)
        elif demo_type == "acceleration":
            run_acceleration_demo(dimension, num_vectors)
        elif demo_type == "rag":
            run_rag_demo(dimension)
        else:
            console.print(f"[bold red]Unknown demo type: {demo_type}[/bold red]")
            console.print("Available demo types: full, acceleration, rag")
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Demo interrupted by user[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Error running demo: {e}[/bold red]")
        logger.exception("Error in demo")
        sys.exit(1)


def show_welcome_banner():
    """Display a welcome banner for the demo."""
    console.print("\n")
    console.print(
        Panel.fit(
            "[bold cyan]Welcome to the LlamaDB Interactive Demo![/bold cyan]\n\n"
            "This demo will showcase LlamaDB's hybrid Python/Rust architecture\n"
            "and MLX acceleration capabilities on Apple Silicon.",
            title="[bold]🦙 LlamaDB[/bold]",
            subtitle="Next-Gen Hybrid Python/Rust Data Platform with MLX",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print("\n")


def show_system_info():
    """Display system information."""
    info = get_system_info()

    table = Table(
        title="System Information", show_header=True, header_style="bold cyan"
    )
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    for key, value in info.items():
        table.add_row(key.replace("_", " ").title(), value)

    console.print(table)
    console.print("\n")


def run_acceleration_demo(dimension: int = 128, num_vectors: int = 50000):
    """
    Run a demonstration of LlamaDB's acceleration capabilities.

    Args:
        dimension: Dimensionality of vectors for benchmarks
        num_vectors: Number of vectors for benchmarks
    """
    show_welcome_banner()

    # Show system information
    console.print("[bold]System Information[/bold]")
    show_system_info()

    # Check for Apple Silicon
    if not is_apple_silicon():
        console.print(
            "[yellow]⚠️ This device is not running on Apple Silicon. MLX acceleration will not be available.[/yellow]"
        )
        console.print(
            "[yellow]The demo will continue using NumPy as the backend.[/yellow]\n"
        )
    elif not is_mlx_available():
        console.print(
            "[yellow]⚠️ MLX not detected on this Apple Silicon device.[/yellow]"
        )
        console.print(
            "[yellow]Would you like to install MLX now for maximum performance?[/yellow]"
        )
        install = typer.confirm("Install MLX?", default=True)

        if install:
            from llamadb.core.mlx_acceleration import install_mlx

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]Installing MLX...[/bold cyan]"),
                BarColumn(),
                TimeElapsedColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("Installing...", total=None)
                success = install_mlx()
                progress.update(task, completed=True)

            if success:
                console.print(
                    "[bold green]✅ MLX successfully installed![/bold green]\n"
                )
            else:
                console.print(
                    "[bold red]❌ Failed to install MLX. Continuing with NumPy backend.[/bold red]\n"
                )

    # Matrix multiplication benchmark
    console.print("[bold cyan]Running Matrix Multiplication Benchmark...[/bold cyan]")
    sizes = [1000, 2000, 3000] if is_mlx_available() else [1000, 1500, 2000]

    matrix_results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Benchmarking [matrix_size]...[/bold cyan]"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Benchmarking...", total=len(sizes))
        for size in sizes:
            progress.update(task, description=f"Benchmarking {size}x{size} matrices...")
            result = benchmark_matrix_multiply(size=size, iterations=3)
            result["size"] = size
            matrix_results.append(result)
            progress.advance(task)

    # Display matrix multiplication results
    table = Table(
        title="Matrix Multiplication Benchmark Results",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Matrix Size", style="cyan")
    table.add_column("NumPy Time (s)", justify="right")
    table.add_column("NumPy GFLOPs", justify="right")

    if is_mlx_available():
        table.add_column("MLX Time (s)", justify="right")
        table.add_column("MLX GFLOPs", justify="right")
        table.add_column("Speedup", justify="right", style="bold green")

    for result in matrix_results:
        size = result["size"]
        numpy_time = f"{result['numpy_time']:.4f}"
        numpy_gflops = f"{result['numpy_gflops']:.2f}"

        if is_mlx_available():
            mlx_time = f"{result['mlx_time']:.4f}"
            mlx_gflops = f"{result['mlx_gflops']:.2f}"
            speedup = f"{result['speedup']:.2f}x"
            table.add_row(
                f"{size}x{size}",
                numpy_time,
                numpy_gflops,
                mlx_time,
                mlx_gflops,
                speedup,
            )
        else:
            table.add_row(f"{size}x{size}", numpy_time, numpy_gflops)

    console.print(table)
    console.print("\n")

    # Vector operations benchmark
    console.print("[bold cyan]Running Vector Operations Benchmark...[/bold cyan]")
    vector_dimensions = [128, 512, 1024]
    batch_sizes = [10000, 50000, 100000]

    vector_results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Benchmarking vectors...[/bold cyan]"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            "Benchmarking...", total=len(vector_dimensions) * len(batch_sizes)
        )
        for dim in vector_dimensions:
            for batch in batch_sizes:
                progress.update(
                    task, description=f"Benchmarking {batch} vectors of dim {dim}..."
                )
                result = benchmark_vector_operations(dim=dim, batch_size=batch)
                result["dimension"] = dim
                result["batch_size"] = batch
                vector_results.append(result)
                progress.advance(task)

    # Display vector operations results
    table = Table(
        title="Vector Operations Benchmark Results",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Dimension", style="cyan")
    table.add_column("Batch Size", style="cyan")
    table.add_column("NumPy Time (s)", justify="right")

    if is_mlx_available():
        table.add_column("MLX Time (s)", justify="right")
        table.add_column("Speedup", justify="right", style="bold green")

    for result in vector_results:
        dim = result["dimension"]
        batch = result["batch_size"]
        numpy_time = f"{result['numpy_time']:.4f}"

        if is_mlx_available():
            mlx_time = f"{result['mlx_time']:.4f}"
            speedup = f"{result['speedup']:.2f}x"
            table.add_row(f"{dim}", f"{batch:,}", numpy_time, mlx_time, speedup)
        else:
            table.add_row(f"{dim}", f"{batch:,}", numpy_time)

    console.print(table)
    console.print("\n")

    # Vector search demonstration
    console.print("[bold cyan]Demonstrating Vector Search Performance...[/bold cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Running vector search demonstration...[/bold cyan]"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Running...", total=None)
        results = demonstrate_acceleration(dimension=dimension, num_vectors=num_vectors)
        progress.update(task, completed=True)

    # Display vector search results
    table = Table(
        title=f"Vector Search Performance ({num_vectors:,} vectors, {dimension} dimensions)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Acceleration Backend", results["acceleration"])
    table.add_row("Index Creation Time", f"{results['index_time']:.4f} seconds")
    table.add_row("Search Time", f"{results['search_time']:.4f} seconds")
    table.add_row("Vectors/Second", f"{results['vectors_per_second']:,.2f}")

    console.print(table)
    console.print("\n")

    # Show code example
    console.print("[bold cyan]Example Code for Vector Search[/bold cyan]")

    code = """
import numpy as np
from llamadb.core.accelerated_ops import VectorIndex

# Create vector index
index = VectorIndex(dimension=128, metric="cosine", use_mlx=True)

# Generate sample vectors and metadata
vectors = np.random.random((10000, 128)).astype(np.float32)
metadata = [{"id": i, "text": f"Document {i}"} for i in range(10000)]

# Add vectors to index
index.add(vectors, metadata)

# Create a query vector
query = np.random.random(128).astype(np.float32)

# Search for similar vectors
results = index.search(query, k=10)

# Print top result
print(f"Top result: ID={results[0]['id']}, Score={results[0]['score']:.4f}")
"""

    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
    console.print(
        Panel(
            syntax, border_style="green", title="Vector Search Example", padding=(1, 2)
        )
    )
    console.print("\n")

    # Show conclusion
    console.print(
        Panel.fit(
            "[bold green]Acceleration Demo Completed![/bold green]\n\n"
            f"You've seen the performance of LlamaDB's acceleration capabilities using "
            f"[bold]{results['acceleration']}[/bold] as the backend.\n\n"
            f"{'With MLX on Apple Silicon, ' if is_mlx_available() else ''}"
            f"LlamaDB can process [bold]{results['vectors_per_second']:,.2f}[/bold] vectors per second "
            f"for similarity search operations.",
            title="[bold]Demo Complete[/bold]",
            border_style="green",
            padding=(1, 2),
        )
    )


def run_rag_demo(dimension: int = 128):
    """
    Run a demonstration of LlamaDB's RAG capabilities.
    """
    show_welcome_banner()

    # Show system information
    console.print("[bold]System Information[/bold]")
    show_system_info()

    console.print(
        Panel.fit(
            "[bold]Retrieval Augmented Generation (RAG) Demo[/bold]\n\n"
            "This demo shows how LlamaDB's vector search capabilities can be used for RAG applications.",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    console.print("[yellow]RAG demo not implemented yet. Coming soon![/yellow]")


def run_full_demo(dimension: int = 128, num_vectors: int = 50000):
    """
    Run the full LlamaDB demonstration.
    """
    show_welcome_banner()

    # Show system information
    console.print("[bold]System Information[/bold]")
    show_system_info()

    # Run acceleration demo
    console.print("[bold cyan]Running Acceleration Demo...[/bold cyan]\n")
    run_acceleration_demo(dimension, num_vectors)

    # Run RAG demo (placeholder)
    console.print("[bold cyan]Running RAG Demo...[/bold cyan]\n")
    run_rag_demo(dimension)

    # Show full conclusion
    console.print(
        Panel.fit(
            "[bold green]Full Demo Completed![/bold green]\n\n"
            "You've experienced the power of LlamaDB's hybrid Python/Rust architecture "
            f"{'with MLX acceleration ' if is_mlx_available() else ''}"
            "for high-performance data processing.\n\n"
            "To learn more, check out the documentation or try building your own applications with LlamaDB.",
            title="[bold]Thank You![/bold]",
            border_style="green",
            padding=(1, 2),
        )
    )


# Add function to implement the missing demonstrate_acceleration functionality
def demonstrate_acceleration(dimension: int = 128, num_vectors: int = 50000) -> dict:
    """
    Demonstrate MLX acceleration by comparing vector operations performance.

    Args:
        dimension: Dimension of vectors to use
        num_vectors: Number of vectors to generate

    Returns:
        Dictionary with performance results
    """
    import time

    import numpy as np
    from llamadb.core.vector_index import VectorIndex

    # Initialize results dictionary
    results = {
        "acceleration": "MLX" if is_mlx_available() else "NumPy",
        "dimension": dimension,
        "num_vectors": num_vectors,
    }

    # Create vector index
    start_time = time.time()
    index = VectorIndex(
        dimension=dimension, metric="cosine", use_mlx=is_mlx_available()
    )

    # Generate random vectors
    vectors = np.random.random((num_vectors, dimension)).astype(np.float32)

    # Create metadata
    metadata = [
        {
            "id": i,
            "text": f"Document {i}",
            "category": f"Category {i % 10}",
            "score": float(np.random.random()),
        }
        for i in range(num_vectors)
    ]

    # Add vectors to index
    index.add(vectors, metadata)
    results["index_time"] = time.time() - start_time

    # Generate query vector
    query = np.random.random(dimension).astype(np.float32)

    # Perform search
    start_time = time.time()
    search_results = index.search(query, k=10)
    results["search_time"] = time.time() - start_time

    # Calculate throughput
    results["vectors_per_second"] = int(num_vectors / results["search_time"])

    return results


# Add the benchmark_matrix_multiply function
def benchmark_matrix_multiply(
    size: int = 1000, iterations: int = 5
) -> Dict[str, float]:
    """
    Benchmark matrix multiplication with MLX and NumPy.

    Args:
        size: Size of square matrices
        iterations: Number of iterations to run

    Returns:
        Dictionary with benchmark results
    """
    import time

    import numpy as np

    # Initialize results
    results = {"size": size, "iterations": iterations}

    # NumPy benchmark
    a_np = np.random.random((size, size)).astype(np.float32)
    b_np = np.random.random((size, size)).astype(np.float32)

    start = time.time()
    for _ in range(iterations):
        c_np = np.matmul(a_np, b_np)
    numpy_time = (time.time() - start) / iterations

    # Calculate GFLOPs for NumPy
    flops = 2 * size**3  # Approximate FLOPs for matrix multiplication
    numpy_gflops = flops / (numpy_time * 1e9)

    results["numpy_time"] = numpy_time
    results["numpy_gflops"] = numpy_gflops

    # MLX benchmark if available
    if is_mlx_available():
        try:
            import mlx.core as mx

            # Convert to MLX arrays
            a_mx = mx.array(a_np)
            b_mx = mx.array(b_np)

            # Warmup
            _ = mx.matmul(a_mx, b_mx)
            mx.eval(_)

            start = time.time()
            for _ in range(iterations):
                c_mx = mx.matmul(a_mx, b_mx)
                mx.eval(c_mx)  # Force computation to complete
            mlx_time = (time.time() - start) / iterations

            # Calculate GFLOPs for MLX
            mlx_gflops = flops / (mlx_time * 1e9)

            results["mlx_time"] = mlx_time
            results["mlx_gflops"] = mlx_gflops
            results["speedup"] = numpy_time / mlx_time
        except Exception as e:
            logger.warning(f"Error in MLX benchmark: {e}")

    return results
