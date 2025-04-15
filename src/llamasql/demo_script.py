#!/usr/bin/env python
"""
LlamaDB Simple Demonstration Script

This script demonstrates the basic functionality of LlamaDB,
showcasing vector operations, MLX acceleration, and vector search.
"""

import os
import sys
import time

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from rich.text import Text

# Set up console for rich output
console = Console()

# Import LlamaDB components
try:
    # Import everything we need from the core module
    from llamadb.core import (
        VectorIndex,
        benchmark_matrix_multiply,
        cosine_similarity,
        is_apple_silicon,
        is_mlx_available,
        matrix_multiply,
    )

    # Confirm imports were successful
    LLAMADB_IMPORTED = True
except ImportError as e:
    console.print(f"[bold red]Error importing LlamaDB components:[/bold red] {e}")
    console.print("[yellow]Attempting fallback import paths...[/yellow]")

    try:
        # Try alternative import paths (in case the user has a different installation structure)
        import os
        import sys

        # Add the parent directory to the path to allow direct imports
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
            console.print(f"[dim]Added {parent_dir} to Python path[/dim]")

        # Try the direct imports from the core subdirectories
        from python.llamadb.core.accelerated_ops import (
            cosine_similarity,
            matrix_multiply,
        )
        from python.llamadb.core.mlx_acceleration import (
            benchmark_matrix_multiply,
            is_apple_silicon,
            is_mlx_available,
        )
        from python.llamadb.core.vector_index import VectorIndex

        console.print("[green]Fallback imports successful[/green]")
        LLAMADB_IMPORTED = True
    except ImportError as e2:
        console.print(
            f"[bold red]Error:[/bold red] Could not import LlamaDB components: {e2}"
        )
        console.print(
            "[yellow]Please ensure LlamaDB is installed correctly with:[/yellow]"
        )
        console.print("  [green]pip install -e .[/green]")
        console.print("Or run from the project root directory.")
        LLAMADB_IMPORTED = False

# Only continue if imports were successful
if not LLAMADB_IMPORTED:
    sys.exit(1)


def print_header():
    """Print a stylized header."""
    title = Text("LlamaDB Demonstration", style="bold blue")
    subtitle = Text(
        "A Next-Gen Hybrid Python/Rust Data Platform with MLX Acceleration",
        style="italic cyan",
    )

    console.print()
    console.print(Panel.fit(f"{title}\n{subtitle}", border_style="blue"))
    console.print()


def show_system_info():
    """Display system information."""
    console.print("[bold cyan]System Information[/bold cyan]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Property")
    table.add_column("Value")

    # System info
    apple_silicon = is_apple_silicon()
    mlx_available = is_mlx_available()

    table.add_row(
        "Apple Silicon",
        f"[green]Yes[/green]" if apple_silicon else "[yellow]No[/yellow]",
    )
    table.add_row(
        "MLX Available",
        f"[green]Yes[/green]" if mlx_available else "[yellow]No[/yellow]",
    )
    table.add_row("Python Version", sys.version.split()[0])
    table.add_row("NumPy Version", np.__version__)
    table.add_row("Operating System", f"{os.name.capitalize()} ({sys.platform})")

    console.print(table)
    console.print()


def demo_vector_operations():
    """Demonstrate vector operations with MLX acceleration."""
    console.print("[bold cyan]Vector Operation Demo[/bold cyan]")

    try:
        # Create example vectors - ensuring proper dimensions for operations
        console.print("Creating example vectors...")

        # Create vectors with explicit dimensions to avoid dimension issues
        vector_a = np.random.random(128).astype(np.float32)
        vector_b = np.random.random(128).astype(np.float32)

        # Normalize vectors for better comparison
        vector_a = vector_a / np.linalg.norm(vector_a)
        vector_b = vector_b / np.linalg.norm(vector_b)

        # Time individual vector operations instead of batch operations
        console.print("Computing vector operations...")
        iterations = 100  # Reduced to avoid too many warnings

        # Pure NumPy implementation
        start = time.time()
        for _ in range(iterations):
            # Compute dot product and normalize
            dot = np.dot(vector_a, vector_b)
            # Since vectors are already normalized, dot product equals cosine similarity
            cos_sim = dot
        numpy_time = (time.time() - start) / iterations

        # Display results
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Operation")
        table.add_column("Time (µs)")
        table.add_column("Backend")

        table.add_row(
            "Cosine Similarity (NumPy)", f"{numpy_time * 1000000:.2f}", "NumPy"
        )

        console.print(table)
        console.print(
            "[yellow]Note: MLX acceleration for vector operations with 1D arrays is currently showing warnings.[/yellow]"
        )
        console.print(
            "[yellow]This is a known issue that will be fixed in a future release.[/yellow]"
        )

    except Exception as e:
        console.print(f"[bold red]Error in vector operations:[/bold red] {str(e)}")

    console.print()


def demo_matrix_operations():
    """Demonstrate matrix operations with MLX acceleration."""
    console.print("[bold cyan]Matrix Operation Demo[/bold cyan]")

    try:
        # Run matrix multiplication benchmark with error handling
        console.print("Benchmarking matrix multiplication...")
        matrix_size = 1000

        try:
            results = benchmark_matrix_multiply(size=matrix_size, iterations=3)

            # Validate results to avoid division by zero
            if "numpy_time" in results and results["numpy_time"] > 0:
                # Display results
                table = Table(show_header=True, header_style="bold cyan")
                table.add_column("Backend")
                table.add_column("Time (ms)")
                table.add_column("GFLOPs")

                table.add_row(
                    "NumPy",
                    f"{results['numpy_time'] * 1000:.2f}",
                    f"{results['numpy_gflops']:.2f}",
                )

                if "mlx_time" in results and results["mlx_time"] > 0:
                    speedup = results["numpy_time"] / results["mlx_time"]
                    table.add_row(
                        "MLX",
                        f"{results['mlx_time'] * 1000:.2f}",
                        f"{results['mlx_gflops']:.2f}",
                    )

                    table.add_row("Speedup", f"{speedup:.2f}x", "")

                console.print(table)
            else:
                # Fallback to manual benchmark if the function returns invalid results
                console.print(
                    "[yellow]Invalid benchmark results. Performing manual benchmark...[/yellow]"
                )
                perform_manual_matrix_benchmark(matrix_size)
        except Exception as inner_e:
            console.print(
                f"[yellow]Error in benchmark function: {str(inner_e)}[/yellow]"
            )
            console.print("[yellow]Performing manual benchmark instead...[/yellow]")
            perform_manual_matrix_benchmark(matrix_size)

    except Exception as e:
        console.print(f"[bold red]Error in matrix operations:[/bold red] {str(e)}")

    console.print()


def perform_manual_matrix_benchmark(size=1000):
    """Perform a manual matrix multiplication benchmark."""
    # Create random matrices
    A = np.random.random((size, size)).astype(np.float32)
    B = np.random.random((size, size)).astype(np.float32)

    # NumPy benchmark
    iterations = 1
    start_time = time.time()
    for _ in range(iterations):
        C_numpy = np.matmul(A, B)
    numpy_time = (time.time() - start_time) / iterations

    # Calculate GFLOPs
    flops = 2 * size**3  # Approximate FLOPs for matrix multiplication
    numpy_gflops = flops / (numpy_time * 1e9)

    # MLX benchmark if available
    mlx_time = None
    mlx_gflops = None

    if is_mlx_available():
        try:
            import mlx.core as mx

            # Convert to MLX arrays
            A_mlx = mx.array(A)
            B_mlx = mx.array(B)

            # Warmup
            _ = mx.matmul(A_mlx, B_mlx)
            mx.eval()

            # Benchmark
            start_time = time.time()
            for _ in range(iterations):
                C_mlx = mx.matmul(A_mlx, B_mlx)
                mx.eval()  # Force evaluation
            mlx_time = (time.time() - start_time) / iterations

            mlx_gflops = flops / (mlx_time * 1e9)
        except Exception as e:
            console.print(
                f"[yellow]Error using MLX for matrix multiplication: {e}[/yellow]"
            )

    # Display results
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Backend")
    table.add_column("Time (ms)")
    table.add_column("GFLOPs")

    table.add_row("NumPy", f"{numpy_time * 1000:.2f}", f"{numpy_gflops:.2f}")

    if mlx_time is not None:
        speedup = numpy_time / mlx_time
        table.add_row("MLX", f"{mlx_time * 1000:.2f}", f"{mlx_gflops:.2f}")

        table.add_row("Speedup", f"{speedup:.2f}x", "")

    console.print(table)


def demo_vector_index():
    """Demonstrate the vector index functionality."""
    console.print("[bold cyan]Vector Index Demo[/bold cyan]")

    try:
        # Create vector index
        console.print("Creating vector index with 10,000 vectors...")
        index = VectorIndex(dimension=128)

        # Generate random vectors and metadata
        vectors = np.random.random((10000, 128)).astype(np.float32)
        categories = ["Technology", "Science", "Health", "Business", "Arts"]
        metadata = [
            {"id": i, "title": f"Document {i}", "category": categories[i % 5]}
            for i in range(10000)
        ]

        # Add vectors to index
        start = time.time()
        for i in track(range(len(vectors)), description="Adding vectors to index"):
            index.add_item(vectors[i], metadata[i])

        index_time = time.time() - start

        # Perform a search
        console.print("Searching for similar vectors...")
        query = np.random.random(128).astype(np.float32)

        start = time.time()
        results = index.search(query, k=5)
        search_time = time.time() - start

        # Display results
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Operation")
        table.add_column("Time (ms)")
        table.add_column("Vectors/sec")

        table.add_row(
            "Index 10,000 vectors",
            f"{index_time * 1000:.2f}",
            f"{10000 / index_time:.2f}",
        )

        table.add_row(
            "Search (k=5)", f"{search_time * 1000:.2f}", f"{10000 / search_time:.2f}"
        )

        console.print(table)

        # Display search results
        console.print("\n[bold cyan]Search Results[/bold cyan]")
        results_table = Table(show_header=True, header_style="bold cyan")
        results_table.add_column("ID")
        results_table.add_column("Score")
        results_table.add_column("Category")
        results_table.add_column("Title")

        for result in results:
            # Extract fields, handling both metadata formats
            result_id = result.get("id", "N/A")
            result_score = result.get("score", 0)

            # Handle both flattened and nested metadata
            if "metadata" in result:
                # Metadata is nested
                metadata = result["metadata"]
                category = metadata.get("category", "N/A")
                title = metadata.get("title", "Untitled")
            else:
                # Metadata is flattened
                category = result.get("category", "N/A")
                title = result.get("title", "Untitled")

            results_table.add_row(
                str(result_id), f"{result_score:.4f}", category, title
            )

        console.print(results_table)
    except Exception as e:
        console.print(f"[bold red]Error in vector index demo:[/bold red] {str(e)}")

    console.print()


def demo_api_compatibility():
    """Demonstrate API vector compatibility solutions."""
    console.print("[bold cyan]API Compatibility Demo[/bold cyan]")

    try:
        # Create vectors with different dimensions
        console.print("Creating sample vectors with different dimensions...")
        small_vector = np.random.random(5).astype(np.float32)
        large_vector = np.random.random(256).astype(np.float32)
        target_dim = 128

        console.print(f"\nOriginal vectors:")
        console.print(f"- Small vector: dimension = {small_vector.shape[0]}")
        console.print(f"- Large vector: dimension = {large_vector.shape[0]}")
        console.print(f"- Target dimension: {target_dim}")

        # Demonstration of padding
        console.print("\n[cyan]Padding a smaller vector:[/cyan]")
        padded_vector = np.zeros(target_dim, dtype=np.float32)
        padded_vector[: small_vector.shape[0]] = small_vector

        console.print(f"- Original: {small_vector.shape[0]} dimensions")
        console.print(f"- Padded: {padded_vector.shape[0]} dimensions")
        console.print(f"- First 5 values preserved: {padded_vector[:5]}")
        console.print(f"- Remaining values padded with zeros")

        # Demonstration of truncating
        console.print("\n[cyan]Truncating a larger vector:[/cyan]")
        truncated_vector = large_vector[:target_dim]

        console.print(f"- Original: {large_vector.shape[0]} dimensions")
        console.print(f"- Truncated: {truncated_vector.shape[0]} dimensions")
        console.print(f"- Only first {target_dim} values kept")

        # Demonstrate normalization
        console.print("\n[cyan]Normalizing vectors:[/cyan]")
        normalized_small = small_vector / np.linalg.norm(small_vector)
        normalized_padded = padded_vector / np.linalg.norm(padded_vector)

        console.print(
            f"- Original small vector norm: {np.linalg.norm(small_vector):.4f}"
        )
        console.print(
            f"- Normalized small vector norm: {np.linalg.norm(normalized_small):.4f}"
        )
        console.print(f"- Padded vector norm: {np.linalg.norm(padded_vector):.4f}")
        console.print(
            f"- Normalized padded vector norm: {np.linalg.norm(normalized_padded):.4f}"
        )

        # Explain why this matters for the API
        console.print("\n[bold cyan]Why This Matters for LlamaDB API:[/bold cyan]")
        console.print("When sending vectors to the LlamaDB API:")
        console.print("1. Vectors must match the dimension of the index (e.g., 128)")
        console.print("2. The API can automatically pad smaller vectors with zeros")
        console.print("3. For larger vectors, you must truncate them before sending")
        console.print(
            "4. For best results, normalize vectors before padding/truncating"
        )

    except Exception as e:
        console.print(f"[bold red]Error in API compatibility demo:[/bold red] {str(e)}")

    console.print()


def main():
    """Run the demo."""
    print_header()
    show_system_info()

    try:
        demo_vector_operations()
        demo_matrix_operations()
        demo_vector_index()
        demo_api_compatibility()

        console.print("[bold green]Demo completed successfully![/bold green]")
        console.print(
            "For more features, try running: [bold]python -m llamadb.cli.main demo[/bold]"
        )
    except Exception as e:
        console.print(
            f"[bold red]An error occurred during the demo:[/bold red] {str(e)}"
        )
        console.print("Please check your installation and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
