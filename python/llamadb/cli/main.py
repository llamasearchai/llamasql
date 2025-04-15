"""
Main CLI module for LlamaDB.

This module implements the main command-line interface for LlamaDB,
providing access to various subcommands for interacting with the platform.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from llamadb.cli.demo import run_demo_command
from llamadb.core.mlx_acceleration import is_apple_silicon, is_mlx_available
from rich.console import Console
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("llamadb")

# Initialize typer app
app = typer.Typer(
    name="llamadb",
    help="LlamaDB: Next-Gen Hybrid Python/Rust Data Platform with MLX",
    add_completion=False,
)

# Initialize rich console
console = Console()


@app.callback()
def callback():
    """
    LlamaDB: Next-Gen Hybrid Python/Rust Data Platform with MLX.

    A cutting-edge data platform that combines the rapid development capabilities
    of Python with the performance advantages of Rust and Apple's MLX acceleration
    framework to create an enterprise-ready solution for modern data analytics challenges.
    """
    pass


@app.command()
def version():
    """
    Show version information.
    """
    from importlib.metadata import version as get_version

    try:
        version = get_version("llamadb")
    except:
        version = "0.1.0.dev"  # Development version

    console.print(f"[bold cyan]LlamaDB[/bold cyan] version [bold]{version}[/bold]")

    # Show acceleration info
    if is_apple_silicon():
        console.print(f"[green]✓[/green] Running on Apple Silicon")
        if is_mlx_available():
            console.print(f"[green]✓[/green] MLX acceleration enabled")
        else:
            console.print(
                f"[yellow]⚠[/yellow] MLX not detected (install with 'pip install mlx')"
            )
    else:
        console.print(
            f"[yellow]⚠[/yellow] Not running on Apple Silicon (MLX acceleration unavailable)"
        )


@app.command()
def demo(
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
    run_demo_command(demo_type, dimension, num_vectors)


@app.command()
def serve(
    host: str = typer.Option(
        "127.0.0.1", "--host", "-h", help="Host to bind the server to"
    ),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind the server to"),
    reload: bool = typer.Option(
        False, "--reload", "-r", help="Enable auto-reload for development"
    ),
    workers: int = typer.Option(
        1, "--workers", "-w", help="Number of worker processes"
    ),
):
    """
    Start the LlamaDB API server.
    """
    try:
        import uvicorn
        from llamadb.api.app import create_app
    except ImportError:
        console.print("[bold red]Error:[/bold red] API dependencies not installed.")
        console.print("Install with: pip install llamadb[api]")
        sys.exit(1)

    console.print(
        f"[bold cyan]Starting LlamaDB API server at http://{host}:{port}[/bold cyan]"
    )

    if is_apple_silicon() and is_mlx_available():
        console.print("[green]✓[/green] MLX acceleration enabled")

    uvicorn.run(
        "llamadb.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
    )


@app.command()
def benchmark(
    benchmark_type: str = typer.Option(
        "all", "--type", "-t", help="Type of benchmark to run (all, matrix, vector)"
    ),
    size: int = typer.Option(2000, "--size", "-s", help="Size for matrix benchmarks"),
    dimension: int = typer.Option(
        128, "--dimension", "-d", help="Dimension for vector benchmarks"
    ),
    num_vectors: int = typer.Option(
        100000, "--num-vectors", "-n", help="Number of vectors for vector benchmarks"
    ),
    iterations: int = typer.Option(
        5, "--iterations", "-i", help="Number of iterations to run"
    ),
):
    """
    Run performance benchmarks.
    """
    from llamadb.core.mlx_acceleration import (
        benchmark_matrix_multiply,
        benchmark_vector_operations,
    )

    console.print("[bold cyan]Running LlamaDB Performance Benchmarks[/bold cyan]\n")

    if benchmark_type in ["all", "matrix"]:
        console.print(f"[bold]Matrix Multiplication Benchmark ({size}x{size})[/bold]")
        result = benchmark_matrix_multiply(size=size, iterations=iterations)

        console.print(
            f"NumPy: {result['numpy_time']:.4f}s ({result['numpy_gflops']:.2f} GFLOPs)"
        )

        if is_mlx_available():
            console.print(
                f"MLX: {result['mlx_time']:.4f}s ({result['mlx_gflops']:.2f} GFLOPs)"
            )
            console.print(f"Speedup: [bold green]{result['speedup']:.2f}x[/bold green]")

        console.print()

    if benchmark_type in ["all", "vector"]:
        console.print(
            f"[bold]Vector Operations Benchmark ({num_vectors} vectors, {dimension} dimensions)[/bold]"
        )
        result = benchmark_vector_operations(dim=dimension, batch_size=num_vectors)

        console.print(f"NumPy: {result['numpy_time']:.4f}s")

        if is_mlx_available():
            console.print(f"MLX: {result['mlx_time']:.4f}s")
            console.print(f"Speedup: [bold green]{result['speedup']:.2f}x[/bold green]")

        console.print()


@app.command()
def shell():
    """
    Start an interactive Python shell with LlamaDB pre-imported.
    """
    try:
        from IPython import start_ipython
    except ImportError:
        console.print("[bold red]Error:[/bold red] IPython not installed.")
        console.print("Install with: pip install ipython")
        sys.exit(1)

    console.print("[bold cyan]Starting LlamaDB Interactive Shell[/bold cyan]")

    # Prepare imports for the shell
    imports = [
        "import numpy as np",
        "from llamadb.core.accelerated_ops import VectorIndex, BatchProcessor",
        "from llamadb.core.mlx_acceleration import is_mlx_available",
    ]

    # Start IPython with pre-defined imports
    start_ipython(argv=[], user_ns={}, exec_lines=imports)


def main():
    """
    Main entry point for the CLI.
    """
    try:
        app()
    except Exception as e:
        logger.exception("Unhandled exception")
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
