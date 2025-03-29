"""
Setup script for LlamaDB.

This script installs the LlamaDB package and its dependencies.
"""

import os
import sys
from setuptools import setup, find_packages

# Read the version from the package
with open("python/llamadb/__init__.py", "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"\'')
            break
    else:
        version = "0.1.0"

# Read the long description from README.md
with open("README.md", "r") as f:
    long_description = f.read()

# Define dependencies
install_requires = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scipy>=1.10.0",
    "scikit-learn>=1.2.0",
    "fastapi>=0.115.0",
    "uvicorn>=0.27.0",
    "pydantic>=2.5.0",
    "typer>=0.9.0",
    "rich>=13.6.0",
    "sqlalchemy>=2.0.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0.0",
    "tqdm>=4.66.0",
]

# Define optional dependencies
extras_require = {
    "api": [
        "fastapi>=0.115.0",
        "uvicorn>=0.27.0",
        "pydantic>=2.5.0",
        "starlette>=0.35.0",
        "httpx>=0.25.0",
    ],
    "dev": [
        "black>=23.10.0",
        "ruff>=0.1.0",
        "mypy>=1.6.0",
        "pre-commit>=3.5.0",
        "ipython>=8.16.0",
    ],
    "test": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-benchmark>=4.0.0",
    ],
    "docs": [
        "sphinx>=7.2.0",
        "sphinx-rtd-theme>=1.3.0",
        "sphinx-autodoc-typehints>=1.24.0",
        "nbsphinx>=0.9.0",
        "myst-parser>=2.0.0",
    ],
    "mlx": [
        "mlx>=0.5.0",
    ],
}

# Add a full extra that includes all optional dependencies
extras_require["full"] = [
    dep for extra_deps in extras_require.values() for dep in extra_deps
]

# Define entry points
entry_points = {
    "console_scripts": [
        "llamadb=llamadb.cli.main:main",
    ],
}

# Define package data
package_data = {
    "llamadb": ["py.typed"],
}

# Define classifiers
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Rust",
    "Topic :: Database",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

setup(
    name="llamadb-llamasearch",
    version=version,
    description="Next-Gen Hybrid Python/Rust Data Platform with MLX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LlamaSearch AI",
    author_email="nikjois@llamasearch.ai",
    url="https://llamasearch.ai",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    package_data=package_data,
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points=entry_points,
    classifiers=classifiers,
    python_requires=">=3.11",
    zip_safe=False,
) 
# Updated in commit 5 - 2025-04-04 17:07:35

# Updated in commit 13 - 2025-04-04 17:07:36

# Updated in commit 21 - 2025-04-04 17:07:37

# Updated in commit 29 - 2025-04-04 17:07:38

# Updated in commit 5 - 2025-04-05 14:25:01

# Updated in commit 13 - 2025-04-05 14:25:02

# Updated in commit 21 - 2025-04-05 14:25:02

# Updated in commit 29 - 2025-04-05 14:25:02

# Updated in commit 5 - 2025-04-05 15:00:55

# Updated in commit 13 - 2025-04-05 15:00:56

# Updated in commit 21 - 2025-04-05 15:00:56

# Updated in commit 29 - 2025-04-05 15:00:56
