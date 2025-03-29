#!/bin/bash
# LlamaDB Command Line Utilities

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Display help information
function show_help {
    echo -e "${BLUE}LlamaDB Command Line Utilities${NC}"
    echo "Usage: ./commands.sh [command]"
    echo
    echo "Available commands:"
    echo "  setup         - Set up development environment"
    echo "  test          - Run all tests"
    echo "  benchmark     - Run performance benchmarks"
    echo "  quickstart    - Run quickstart demo"
    echo "  clean         - Clean build directories"
    echo "  publish       - Build and publish package to PyPI"
    echo "  help          - Show this help message"
    echo
}

# Setup development environment
function setup {
    echo -e "${BLUE}Setting up development environment...${NC}"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv .venv
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Install dependencies
    echo "Installing development dependencies..."
    pip install -e ".[dev,test]"
    
    echo -e "${GREEN}Development environment setup complete!${NC}"
    echo "To activate the environment, run: source .venv/bin/activate"
}

# Run all tests
function run_tests {
    echo -e "${BLUE}Running tests...${NC}"
    
    # Ensure we're in a virtual environment
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
    
    # Run pytest
    python -m pytest tests/ -v
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
    else
        echo -e "${RED}Some tests failed.${NC}"
    fi
}

# Run benchmarks
function run_benchmarks {
    echo -e "${BLUE}Running performance benchmarks...${NC}"
    
    # Ensure we're in a virtual environment
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
    
    # Run the benchmark script
    python test_llamadb.py --benchmark
    
    echo -e "${GREEN}Benchmarks complete!${NC}"
    echo "Results saved to benchmark_results.json"
}

# Run the quickstart demo
function run_quickstart {
    echo -e "${BLUE}Running quickstart demo...${NC}"
    
    # Ensure we're in a virtual environment
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
    
    # Run the quickstart script
    python quickstart.py
}

# Clean build directories
function clean {
    echo -e "${BLUE}Cleaning build directories...${NC}"
    
    # Remove build artifacts
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info/
    rm -rf .pytest_cache/
    rm -rf .coverage
    rm -rf .mypy_cache/
    
    # Remove __pycache__ directories
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type d -name "*.pyc" -delete
    
    echo -e "${GREEN}Clean complete!${NC}"
}

# Publish package to PyPI
function publish {
    echo -e "${BLUE}Building and publishing package to PyPI...${NC}"
    
    # Ensure we're in a virtual environment
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
    
    # Ask for confirmation
    read -p "Are you sure you want to publish to PyPI? (y/n) " choice
    if [[ $choice == "y" ]]; then
        python scripts/publish.py
    else
        echo "Publication cancelled."
        return 1
    fi
}

# Main function to handle command line arguments
function main {
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi

    case "$1" in
        setup)
            setup
            ;;
        test)
            run_tests
            ;;
        benchmark)
            run_benchmarks
            ;;
        quickstart)
            run_quickstart
            ;;
        clean)
            clean
            ;;
        publish)
            publish
            ;;
        help)
            show_help
            ;;
        *)
            echo -e "${RED}Unknown command: $1${NC}"
            show_help
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@" 