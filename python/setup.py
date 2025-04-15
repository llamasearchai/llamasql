from setuptools import find_packages, setup

setup(
    name="llamadb",
    version="0.1.0",
    description="High-performance vector database optimized for AI workloads",
    author="Nik Jois" "Nik Jois" "Nik Jois" "Nik Jois" "Nik Jois",
    author_email="nikjois@llamasearch.ai"
    "nikjois@llamasearch.ai"
    "nikjois@llamasearch.ai"
    "nikjois@llamasearch.ai"
    "nikjois@llamasearch.ai",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.6.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.2",
        "tqdm>=4.62.0",
        "pandas>=1.3.0",
        "requests>=2.26.0",
    ],
    extras_require={
        "dev": [
            "black>=21.5b2",
            "isort>=5.9.1",
            "mypy>=0.812",
            "flake8>=3.9.2",
        ],
        "test": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
        ],
        "mlx": [
            "mlx>=0.0.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "llamadb=llamadb.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
