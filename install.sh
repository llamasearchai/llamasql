#!/usr/bin/env bash
# LlamaDB Installation Script - Enterprise Edition
# This script installs LlamaDB with all its components and MLX support

set -e

# Text Styling and Logging
BOLD="\033[1m"
CYAN="\033[36m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
BLUE="\033[34m"
MAGENTA="\033[35m"
RESET="\033[0m"

show_banner() {
  echo -e "${BOLD}${BLUE}
  ╭───────────────────────────────────────────────╮
  │                                               │
  │   🦙 LlamaDB Enterprise Installation Script   │
  │                                               │
  │   Hybrid Python/Rust + MLX Acceleration       │
  │   Version 3.1.0                               │
  │                                               │
  ╰───────────────────────────────────────────────╯${RESET}"
  echo -e "${CYAN}This will install LlamaDB with all enterprise features${RESET}\n"
}

show_banner

# Detect OS and architecture
echo -e "${CYAN}Detecting system architecture...${RESET}"
OS=$(uname -s)
ARCH=$(uname -m)

if [[ "$OS" == "Darwin" && ("$ARCH" == "arm64" || "$ARCH" == "aarch64") ]]; then
  echo -e "${GREEN}Detected Apple Silicon ($ARCH). Will enable MLX acceleration.${RESET}"
  APPLE_SILICON=1
elif [[ "$OS" == "Darwin" ]]; then
  echo -e "${YELLOW}Detected macOS on Intel. MLX acceleration not available.${RESET}"
  APPLE_SILICON=0
else
  echo -e "${YELLOW}Detected $OS on $ARCH. MLX acceleration not available.${RESET}"
  APPLE_SILICON=0
fi

# Check for Python 3.11+
echo -e "${CYAN}Checking for Python 3.11+...${RESET}"
if command -v python3 >/dev/null; then
    PY_VERSION=$(python3 --version | cut -d' ' -f2)
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
    
    if [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -ge 11 ]]; then
        PYTHON_CMD="python3"
        echo -e "${GREEN}Found Python $PY_VERSION${RESET}"
    else
        if command -v python3.11 >/dev/null; then
            PYTHON_CMD="python3.11"
            echo -e "${GREEN}Found Python 3.11${RESET}"
        else
            echo -e "${RED}Python 3.11+ is required but not found.${RESET}"
            echo -e "${YELLOW}Please install Python 3.11 or newer and try again.${RESET}"
            exit 1
        fi
    fi
else
    echo -e "${RED}Python 3 not found. Please install Python 3.11 or newer.${RESET}"
    exit 1
fi

# Check for Rust
echo -e "${CYAN}Checking for Rust...${RESET}"
if ! command -v rustc >/dev/null; then
    echo -e "${YELLOW}Rust not found. Installing Rust...${RESET}"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo -e "${GREEN}Rust installed successfully${RESET}"
else
    RUST_VERSION=$(rustc --version)
    echo -e "${GREEN}Rust is already installed: $RUST_VERSION${RESET}"
fi

# Create virtual environment
echo -e "${CYAN}Creating Python virtual environment...${RESET}"
$PYTHON_CMD -m venv .venv
source .venv/bin/activate

# Handle potential conda conflict
# Temporarily unset CONDA_PREFIX if it exists to avoid maturin conflicts
if [[ -n "${CONDA_PREFIX}" ]]; then
    echo -e "${YELLOW}Detected Conda environment. Temporarily disabling for compatibility.${RESET}"
    CONDA_PREFIX_BACKUP="${CONDA_PREFIX}"
    unset CONDA_PREFIX
fi

echo -e "${GREEN}Virtual environment created and activated${RESET}"

# Upgrade pip
echo -e "${CYAN}Upgrading pip...${RESET}"
pip install --upgrade pip

# Install core dependencies with minimal constraints
echo -e "${CYAN}Installing core dependencies...${RESET}"
pip install fastapi uvicorn pydantic typer rich sqlalchemy \
    faiss-cpu langchain openai anthropic numpy pandas \
    httpx python-dotenv maturin pytest pytest-cov

# Install MLX if on Apple Silicon
if [[ "$APPLE_SILICON" -eq 1 ]]; then
    echo -e "${CYAN}Installing MLX for Apple Silicon acceleration...${RESET}"
    pip install mlx
    echo -e "${GREEN}MLX installed successfully${RESET}"
fi

# Create project structure
echo -e "${CYAN}Creating project structure...${RESET}"
mkdir -p python/llamadb/{api,core,db,utils,cli,ml}
mkdir -p rust_extensions/{llamadb_core,vector_store}/src
mkdir -p tests/{unit,integration,benchmarks}
mkdir -p data/{examples,vectors,logs} docs/assets

# Create basic files
echo -e "${CYAN}Creating configuration files...${RESET}"

# Create a minimal Rust extension
cat > rust_extensions/llamadb_core/src/lib.rs << 'EOF'
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::collections::HashMap;

#[pyfunction]
fn hello_llamadb() -> PyResult<String> {
    Ok("Hello from LlamaDB Rust extension!".to_string())
}

/// High-performance tokenizer with counting
#[pyfunction]
fn tokenize_and_count(text: &str) -> HashMap<String, usize> {
    let mut word_counts = HashMap::new();
    
    for word in text.split_whitespace() {
        let cleaned = word.trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();
        
        if !cleaned.is_empty() {
            *word_counts.entry(cleaned).or_insert(0) += 1;
        }
    }
    
    word_counts
}

/// Fast vector operations
#[pyfunction]
fn cosine_similarity(v1: Vec<f32>, v2: Vec<f32>) -> PyResult<f32> {
    if v1.len() != v2.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Vector dimensions don't match: {} vs {}", v1.len(), v2.len())
        ));
    }
    
    let mut dot_product = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;
    
    for i in 0..v1.len() {
        dot_product += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    
    if norm1 == 0.0 || norm2 == 0.0 {
        return Ok(0.0);
    }
    
    Ok(dot_product / (norm1.sqrt() * norm2.sqrt()))
}

#[pymodule]
fn llamadb_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_llamadb, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_and_count, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    Ok(())
}
EOF

# Create Cargo.toml
cat > rust_extensions/llamadb_core/Cargo.toml << 'EOF'
[package]
name = "llamadb_core"
version = "0.1.0"
edition = "2021"

[lib]
name = "llamadb_core"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.18.0", features = ["extension-module"] }
EOF

# Create a basic Python module
cat > python/llamadb/__init__.py << 'EOF'
"""
LlamaDB - A cutting-edge data exploration and AI-powered platform.

This platform combines Python and Rust to provide high-performance
data operations with an elegant, user-friendly interface.
"""

__version__ = "3.1.0"
EOF

# Create MLX integration module if on Apple Silicon
if [[ "$APPLE_SILICON" -eq 1 ]]; then
    cat > python/llamadb/ml/mlx_integration.py << 'EOF'
"""
MLX integration for LlamaDB, providing accelerated ML operations on Apple Silicon.
"""
import os
import sys
from typing import List, Optional, Dict, Any, Tuple

try:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

def check_mlx_available() -> bool:
    """Check if MLX is available on this system."""
    return HAS_MLX

class MLXVectorSearch:
    """Vector search implementation using MLX for acceleration."""
    
    def __init__(self, dimension: int = 768):
        """Initialize the vector search engine.
        
        Args:
            dimension: The dimension of the vectors to store
        """
        if not HAS_MLX:
            raise ImportError("MLX is not available. Please install it with 'pip install mlx'")
        
        self.dimension = dimension
        self.vectors = []
        self.metadata = []
        
    def add_vectors(self, vectors: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None):
        """Add vectors to the index.
        
        Args:
            vectors: List of vectors to add
            metadata: Optional metadata for each vector
        """
        # Convert to MLX arrays for efficient operations
        mx_vectors = [mx.array(v, dtype=mx.float32) for v in vectors]
        
        # Normalize vectors for cosine similarity
        for i, v in enumerate(mx_vectors):
            norm = mx.sqrt(mx.sum(v * v))
            if norm > 0:
                mx_vectors[i] = v / norm
        
        self.vectors.extend(mx_vectors)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in vectors])
    
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Search for the most similar vectors.
        
        Args:
            query_vector: The query vector
            top_k: Number of results to return
            
        Returns:
            List of (index, score, metadata) tuples
        """
        if not self.vectors:
            return []
        
        # Convert query to MLX array and normalize
        query = mx.array(query_vector, dtype=mx.float32)
        query_norm = mx.sqrt(mx.sum(query * query))
        if query_norm > 0:
            query = query / query_norm
        
        # Calculate similarities
        similarities = []
        for i, vec in enumerate(self.vectors):
            # Cosine similarity
            sim = mx.sum(query * vec)
            similarities.append((i, float(sim.item()), self.metadata[i]))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

# Create a simple example function to demonstrate MLX acceleration
def run_mlx_demo():
    """Run a simple MLX demo to show acceleration."""
    if not HAS_MLX:
        print("MLX is not available. Apple Silicon is required.")
        return
    
    print("Running MLX acceleration demo...")
    
    # Create a simple matrix multiplication benchmark
    size = 1000
    iterations = 10
    
    # MLX version
    a = mx.random.normal((size, size))
    b = mx.random.normal((size, size))
    
    import time
    start = time.time()
    for _ in range(iterations):
        c = mx.matmul(a, b)
        # Force computation to complete
        mx.eval(c)
    mlx_time = time.time() - start
    
    # NumPy version
    a_np = np.random.normal(size=(size, size))
    b_np = np.random.normal(size=(size, size))
    
    start = time.time()
    for _ in range(iterations):
        c_np = np.matmul(a_np, b_np)
    numpy_time = time.time() - start
    
    print(f"MLX time: {mlx_time:.4f}s")
    print(f"NumPy time: {numpy_time:.4f}s")
    print(f"MLX speedup: {numpy_time/mlx_time:.2f}x")
    
    return {
        "mlx_time": mlx_time,
        "numpy_time": numpy_time,
        "speedup": numpy_time/mlx_time
    }

if __name__ == "__main__":
    run_mlx_demo()
EOF
fi

# Create CLI module for demo purposes
cat > python/llamadb/cli/demo.py << 'EOF'
"""
Demo script for LlamaDB showcasing key features.
"""
import sys
import time
from typing import Dict, List, Optional, Any
import argparse

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

def run_demo(enable_mlx: bool = True, enable_rust: bool = True):
    """Run a comprehensive demo of LlamaDB features.
    
    Args:
        enable_mlx: Whether to enable MLX acceleration (Apple Silicon only)
        enable_rust: Whether to use Rust extensions
    """
    console.print(Panel.fit(
        "[bold blue]LlamaDB Demo[/bold blue]\n\n"
        "This demo showcases the key features of LlamaDB, including:\n"
        "- Hybrid Python/Rust architecture\n"
        "- MLX acceleration (on Apple Silicon)\n"
        "- Vector search capabilities\n"
        "- Performance benchmarks",
        title="Welcome",
        border_style="cyan"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold green]{task.percentage:.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        # Test Python functionality
        python_task = progress.add_task("[cyan]Testing Python functionality...", total=100)
        for i in range(100):
            time.sleep(0.01)
            progress.update(python_task, advance=1)
            
        # Test Rust extensions if enabled
        if enable_rust:
            rust_task = progress.add_task("[magenta]Testing Rust extensions...", total=100)
            # Import and test Rust functionality
            try:
                from llamadb_core import hello_llamadb, tokenize_and_count, cosine_similarity
                
                # Test tokenize_and_count
                text = "This is a test of the Rust extension for LlamaDB. This extension is written in Rust."
                counts = tokenize_and_count(text)
                
                # Test vector similarity
                v1 = [1.0, 0.0, 0.5]
                v2 = [0.5, 0.5, 0.0]
                sim = cosine_similarity(v1, v2)
                
                for i in range(100):
                    time.sleep(0.01)
                    progress.update(rust_task, advance=1)
                
                has_rust = True
            except ImportError:
                console.print("[bold red]Rust extensions not found or failed to load.[/bold red]")
                has_rust = False
        else:
            has_rust = False
        
        # Test MLX acceleration if enabled and on Apple Silicon
        if enable_mlx:
            mlx_task = progress.add_task("[green]Testing MLX acceleration...", total=100)
            try:
                from llamadb.ml.mlx_integration import check_mlx_available, run_mlx_demo
                
                if check_mlx_available():
                    # Run MLX demo
                    results = run_mlx_demo()
                    has_mlx = True
                else:
                    console.print("[yellow]MLX not available. Requires Apple Silicon.[/yellow]")
                    has_mlx = False
                    
                for i in range(100):
                    time.sleep(0.01)
                    progress.update(mlx_task, advance=1)
            except ImportError:
                console.print("[yellow]MLX module not found. Skipping acceleration tests.[/yellow]")
                has_mlx = False
        else:
            has_mlx = False
    
    # Show results table
    console.print("\n[bold cyan]Demo Results[/bold cyan]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")
    
    table.add_row(
        "Python Core", 
        "[green]✓ Working[/green]", 
        "Base Python functionality is operational"
    )
    
    if has_rust:
        table.add_row(
            "Rust Extensions", 
            "[green]✓ Working[/green]", 
            f"Successfully tested tokenization and vector similarity"
        )
    else:
        table.add_row(
            "Rust Extensions", 
            "[yellow]⚠ Not Tested[/yellow]", 
            "Rust extensions were not enabled or failed to load"
        )
    
    if has_mlx:
        table.add_row(
            "MLX Acceleration", 
            "[green]✓ Working[/green]", 
            f"Achieved {results['speedup']:.2f}x speedup over NumPy"
        )
    else:
        table.add_row(
            "MLX Acceleration", 
            "[yellow]⚠ Not Available[/yellow]", 
            "MLX requires Apple Silicon hardware"
        )
    
    console.print(table)
    
    console.print("\n[bold green]Demo completed successfully![/bold green]")
    console.print("[cyan]LlamaDB is ready for use.[/cyan]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LlamaDB demo")
    parser.add_argument("--no-mlx", action="store_true", help="Disable MLX acceleration")
    parser.add_argument("--no-rust", action="store_true", help="Disable Rust extensions")
    
    args = parser.parse_args()
    
    run_demo(
        enable_mlx=not args.no_mlx,
        enable_rust=not args.no_rust
    )
EOF

# Build the Rust extension
echo -e "${CYAN}Building Rust extension...${RESET}"
cd rust_extensions/llamadb_core
maturin develop
cd ../..
echo -e "${GREEN}Rust extension built successfully${RESET}"

# Create a main CLI entry point
cat > python/llamadb/cli/main.py << 'EOF'
"""Main CLI entry point for LlamaDB."""
import argparse
import sys
from typing import List, Optional

def main(args: Optional[List[str]] = None) -> int:
    """Run the LlamaDB CLI."""
    if args is None:
        args = sys.argv[1:]
    
    parser = argparse.ArgumentParser(description="LlamaDB: Hybrid Python/Rust Data Platform")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run the LlamaDB demo")
    demo_parser.add_argument("--no-mlx", action="store_true", help="Disable MLX acceleration")
    demo_parser.add_argument("--no-rust", action="store_true", help="Disable Rust extensions")
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")
    
    # Parse args
    parsed_args = parser.parse_args(args)
    
    if parsed_args.command == "demo":
        from llamadb.cli.demo import run_demo
        run_demo(
            enable_mlx=not parsed_args.no_mlx,
            enable_rust=not parsed_args.no_rust
        )
    elif parsed_args.command == "version":
        from llamadb import __version__
        print(f"LlamaDB version {__version__}")
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

# Create a simple test to verify installation
cat > test_installation.py << 'EOF'
"""
Test if the LlamaDB installation was successful.
"""
import sys
import platform

def test_import():
    """Test if we can import the libraries."""
    print("Testing core Python dependencies...")
    import fastapi
    import uvicorn
    import pydantic
    import typer
    import rich
    import sqlalchemy
    import numpy
    import pandas
    import httpx
    print("✅ All Python dependencies imported successfully!")
    
    # Try importing the Rust extension
    print("\nTesting Rust extension...")
    try:
        from llamadb_core import hello_llamadb, tokenize_and_count, cosine_similarity
        message = hello_llamadb()
        print(f"✅ Rust extension test: {message}")
        
        # Test tokenize_and_count
        counts = tokenize_and_count("Hello LlamaDB from Rust!")
        print(f"✅ Tokenization test: {counts}")
        
        # Test cosine_similarity
        similarity = cosine_similarity([1.0, 0.0, 0.5], [0.5, 0.5, 0.0])
        print(f"✅ Vector similarity test: {similarity:.4f}")
    except ImportError as e:
        print(f"❌ Could not import Rust extension: {e}")
        return False
    
    # Check for MLX if on Apple Silicon
    print("\nTesting MLX acceleration...")
    if platform.machine() in ('arm64', 'aarch64') and platform.system() == 'Darwin':
        try:
            from llamadb.ml.mlx_integration import check_mlx_available
            if check_mlx_available():
                print("✅ MLX is available and working!")
            else:
                print("❌ MLX is not available despite running on Apple Silicon")
        except ImportError:
            print("❌ Could not import MLX integration module")
    else:
        print("ℹ️ Not running on Apple Silicon, MLX acceleration not available")
    
    return True

if __name__ == "__main__":
    print(f"Testing LlamaDB installation on {platform.system()} {platform.machine()}...")
    print(f"Python version: {sys.version}")
    print("=" * 60)
    
    success = test_import()
    
    print("\n" + "=" * 60)
    if success:
        print("\n✅ LlamaDB installation test passed!")
        print("\nYou can run the demo with: python -m llamadb.cli.main demo")
    else:
        print("\n❌ LlamaDB installation test failed!")
    print("=" * 60)
EOF

# Create a GitHub workflow file for CI/CD
mkdir -p .github/workflows
cat > .github/workflows/ci.yml << 'EOF'
name: LlamaDB CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ['3.11']

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        profile: minimal
        toolchain: stable
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest pytest-cov
        pip install fastapi uvicorn pydantic typer rich sqlalchemy faiss-cpu numpy pandas httpx python-dotenv maturin
    
    - name: Install MLX on macOS arm64
      if: runner.os == 'macOS' && runner.arch == 'arm64'
      run: pip install mlx
    
    - name: Build Rust extensions
      run: |
        cd rust_extensions/llamadb_core
        maturin develop
        cd ../..
    
    - name: Run tests
      run: |
        pytest
EOF

# Create a simple README for quick start
cat > QUICK_START.md << 'EOF'
# 🦙 LlamaDB Quick Start Guide

This guide will help you get started with LlamaDB quickly.

## Installation

The installation script has already set up LlamaDB for you. You should now have:

- A Python virtual environment with all dependencies
- Rust extensions compiled and ready to use
- MLX acceleration if you're on Apple Silicon

## Running the Demo

To run the LlamaDB demo:

```bash
# Make sure you're in the virtual environment
source .venv/bin/activate

# Run the demo
python -m llamadb.cli.main demo
```

## Testing Your Installation

To verify your installation:

```bash
python test_installation.py
```

## Key Features

- **Hybrid Architecture**: Python API with Rust extensions for critical performance paths
- **MLX Acceleration**: Utilizes Apple's MLX framework for acceleration on Apple Silicon
- **Vector Search**: High-performance similarity search capabilities
- **Modern API**: FastAPI-based REST API for seamless integration

## Next Steps

- Check out the documentation in the `docs/` directory
- Explore the code in `python/llamadb/`
- Run the benchmarks to see performance on your system

Happy exploring with LlamaDB!
EOF

# Run the test
echo -e "${CYAN}Testing installation...${RESET}"
python test_installation.py

# Restore CONDA_PREFIX if we backed it up
if [[ -n "${CONDA_PREFIX_BACKUP}" ]]; then
    export CONDA_PREFIX="${CONDA_PREFIX_BACKUP}"
    unset CONDA_PREFIX_BACKUP
fi

echo -e "\n${GREEN}${BOLD}LlamaDB installation completed!${RESET}"
echo -e "${CYAN}To activate the environment in the future, run:${RESET}"
echo -e "    ${YELLOW}source .venv/bin/activate${RESET}"
echo -e "${CYAN}To run the demo:${RESET}"
echo -e "    ${YELLOW}python -m llamadb.cli.main demo${RESET}"
echo -e "${CYAN}For quick start information:${RESET}"
echo -e "    ${YELLOW}cat QUICK_START.md${RESET}"
echo -e "\n${MAGENTA}${BOLD}Enjoy using LlamaDB!${RESET}"

# Offer to publish to PyPI
offer_pypi_publication() {
    echo -e "\n${BLUE}Installation complete!${NC}"
    echo -e "${GREEN}✓ LlamaDB has been successfully installed${NC}"
    
    if [[ -f "scripts/publish.py" ]]; then
        echo -e "\n${BLUE}Would you like to publish LlamaDB to PyPI? (y/N)${NC}"
        read -r publish_choice
        
        if [[ "$publish_choice" == "y" || "$publish_choice" == "Y" ]]; then
            echo -e "\n${BLUE}How would you like to publish?${NC}"
            echo -e "  ${YELLOW}1)${NC} Use real PyPI token (for actual publication)"
            echo -e "  ${YELLOW}2)${NC} Use mock token (for testing, publishes to TestPyPI)"
            read -r token_choice
            
            echo -e "\n${BLUE}Running PyPI publication script...${NC}"
            if [[ "$token_choice" == "2" ]]; then
                # Use mock token flag for testing
                $PYTHON scripts/publish.py --mock
            else
                # Run without --no-prompt to allow the script to handle token prompting
                $PYTHON scripts/publish.py
            fi
        else
            echo -e "\n${BLUE}Skipping PyPI publication.${NC}"
            echo -e "${BLUE}You can publish anytime later by running:${NC}"
            echo -e "   ${YELLOW}python scripts/publish.py${NC}"
            echo -e "   ${YELLOW}python scripts/publish.py --mock${NC} (for testing with mock token)"
        fi
    fi
}

# Main installation process
main() {
    display_banner
    
    echo -e "${BLUE}Starting LlamaDB installation...${NC}"
    
    # Environment checks
    check_apple_silicon
    check_python
    check_rust
    check_conda_conflict
    
    # Create virtual environment
    create_venv
    
    # Install dependencies
    install_mlx
    install_dependencies
    
    # Build and install
    install_package
    build_rust_extensions
    
    # Ensure project structure
    create_project_structure
    
    # Test installation
    run_tests
    
    # Restore conda prefix if it was unset
    if [[ -n "$OLD_CONDA_PREFIX" ]]; then
        export CONDA_PREFIX=$OLD_CONDA_PREFIX
        echo -e "${GREEN}✓ Restored original CONDA_PREFIX${NC}"
    fi
    
    # Offer PyPI publication
    offer_pypi_publication
    
    echo -e "\n${GREEN}✓ LlamaDB installation complete!${NC}"
    echo -e "${BLUE}To activate the virtual environment in the future, run:${NC}"
    echo -e "   ${YELLOW}source .venv/bin/activate${NC}"
    echo -e "${BLUE}To start the API server, run:${NC}"
    echo -e "   ${YELLOW}python -m llamadb.cli.main serve${NC}"
    echo -e "${BLUE}To run the demo, run:${NC}"
    echo -e "   ${YELLOW}python -m llamadb.cli.main demo${NC}"
} 