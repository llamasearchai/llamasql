# LlamaDB Developer Documentation

This document provides detailed information for developers working on or contributing to the LlamaDB project.

## Table of Contents

- [Project Structure](#project-structure)
- [Development Environment Setup](#development-environment-setup)
- [Core Modules](#core-modules)
- [Testing](#testing)
- [Performance Optimization](#performance-optimization)
- [MLX Acceleration](#mlx-acceleration)
- [Publishing](#publishing)
- [Contribution Guidelines](#contribution-guidelines)
- [Debugging](#debugging)

## Project Structure

The LlamaDB project follows this structure:

```
llamasql/
├── python/               # Python package directory
│   ├── llamadb/          # Main package
│   │   ├── __init__.py
│   │   ├── core/         # Core functionality
│   │   │   ├── __init__.py
│   │   │   ├── vector_index.py
│   │   │   ├── accelerated_ops.py
│   │   │   └── mlx_acceleration.py
│   │   ├── api/          # API server
│   │   └── cli/          # Command line interface
│   └── setup.py
├── rust/                 # Rust extensions for performance
├── tests/                # Test suite
├── benchmarks/           # Performance benchmarks
├── docs/                 # Documentation
├── scripts/              # Utility scripts
│   └── publish.py        # Publishing utility
├── .github/              # GitHub configuration
│   └── workflows/        # GitHub Actions workflows
├── quickstart.py         # Quickstart demo script
├── test_llamadb.py       # Comprehensive testing script
├── setup_dev_environment.sh  # Development environment setup
├── commands.sh           # Command-line utilities
└── README.md             # Project README
```

## Development Environment Setup

To set up your development environment, you can use the provided script:

```bash
./setup_dev_environment.sh
```

Alternatively, you can perform the setup manually:

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev,test]"
   ```

3. If on Apple Silicon Mac, install MLX for acceleration:
   ```bash
   pip install mlx
   ```

4. If working with Rust extensions, install Rust:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

## Core Modules

LlamaDB's core functionality is divided into several key modules:

### vector_index.py

This module contains the `VectorIndex` class, which is the primary data structure for vector similarity search. It supports:

- Adding vectors with metadata
- Deleting vectors
- Searching by vector similarity
- Filtering search results
- Batched operations

### accelerated_ops.py

Provides optimized vector operations, including:

- `cosine_similarity`: Calculate cosine similarity between vectors
- `l2_distance`: Calculate Euclidean (L2) distance
- `dot_product`: Calculate dot product between vectors
- `matrix_multiply`: Optimized matrix multiplication

### mlx_acceleration.py

Contains Apple Silicon (M1/M2/M3) specific optimizations using MLX:

- `is_apple_silicon`: Detect if running on Apple Silicon
- `is_mlx_available`: Check if MLX library is available
- `install_mlx`: Utility to install MLX
- `benchmark_matrix_multiply`: Benchmark matrix operations
- `benchmark_vector_operations`: Benchmark vector operations

## Testing

The project uses pytest for testing. Run tests with:

```bash
./commands.sh test
```

Or directly:

```bash
python -m pytest tests/ -v
```

For comprehensive testing and benchmarking:

```bash
python test_llamadb.py --benchmark
```

## Performance Optimization

LlamaDB is optimized for performance in several ways:

1. **Vectorized Operations**: Using NumPy for efficient vector operations
2. **MLX Acceleration**: Using Apple MLX library on Apple Silicon
3. **Rust Extensions**: Critical paths implemented in Rust for maximum performance
4. **Batched Processing**: Support for batched operations to amortize overhead

## MLX Acceleration

On Apple Silicon Macs (M1/M2/M3), LlamaDB can use the MLX library for accelerated performance:

- **Matrix Operations**: Up to 10x faster matrix multiplication
- **Vector Similarity**: Up to 5x faster similarity calculations
- **Batch Processing**: Optimized for Apple Neural Engine

To enable MLX acceleration, install the MLX library:

```bash
pip install mlx
```

## Publishing

To publish the package to PyPI:

```bash
./commands.sh publish
```

This will:
1. Build the package
2. Run tests to ensure quality
3. Prompt for confirmation
4. Upload to PyPI if confirmed

## Contribution Guidelines

When contributing to LlamaDB, please follow these guidelines:

1. **Code Style**: Follow PEP 8 style guidelines for Python code
2. **Documentation**: Document all public functions, classes, and methods
3. **Testing**: Add tests for new functionality
4. **Performance**: Ensure changes don't negatively impact performance
5. **Pull Requests**: Create detailed PRs describing the changes

For the commit process:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for your changes
5. Ensure all tests pass
6. Submit a pull request

## Debugging

For debugging LlamaDB, consider these tips:

1. **Import Issues**: If facing import problems, check that:
   - The package is installed in development mode (`pip install -e .`)
   - The import paths are correct
   - All `__init__.py` files properly export required modules

2. **Performance Issues**: Use the benchmarking tools to identify bottlenecks:
   ```bash
   python test_llamadb.py --benchmark --plot
   ```

3. **CI/GitHub Actions**: For CI issues, check:
   - Secret configuration in repository settings
   - Workflow file syntax
   - Conditional steps based on available secrets