#!/bin/bash
# LlamaDB Development Environment Setup
# This script configures and validates the development environment for LlamaDB

echo "===== LlamaDB Development Environment Setup ====="
echo "Setting up a complete development environment for LlamaDB"
echo

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1)
if [[ $PYTHON_VERSION == *"Python 3"* ]]; then
  echo -e "${GREEN}✓ Found $PYTHON_VERSION${NC}"
else
  echo -e "${RED}✗ Python 3 not found, please install Python 3.7 or higher${NC}"
  exit 1
fi

# Check if venv exists
if [ -d ".venv" ]; then
  echo -e "${GREEN}✓ Virtual environment already exists${NC}"
  echo "Activating virtual environment..."
  source .venv/bin/activate || { echo -e "${RED}✗ Failed to activate virtual environment${NC}"; exit 1; }
else
  echo "Creating virtual environment..."
  python3 -m venv .venv || { echo -e "${RED}✗ Failed to create virtual environment${NC}"; exit 1; }
  echo -e "${GREEN}✓ Created virtual environment${NC}"
  echo "Activating virtual environment..."
  source .venv/bin/activate || { echo -e "${RED}✗ Failed to activate virtual environment${NC}"; exit 1; }
fi

echo "Installing dependencies..."
pip install -e ".[dev,test]" || { echo -e "${RED}✗ Failed to install dependencies${NC}"; exit 1; }
echo -e "${GREEN}✓ Installed dependencies${NC}"

# Check for Rust
echo "Checking for Rust installation..."
if command -v rustc &> /dev/null; then
  echo -e "${GREEN}✓ Rust is installed$(rustc --version)${NC}"
else
  echo -e "${YELLOW}⚠ Rust not found${NC}"
  echo "Would you like to install Rust? (y/n)"
  read -r install_rust
  if [[ $install_rust == "y" ]]; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo -e "${GREEN}✓ Rust installed successfully${NC}"
  else
    echo -e "${YELLOW}⚠ Skipping Rust installation. Some features may not work.${NC}"
  fi
fi

# Check for MLX on Apple Silicon
if [[ $(uname -m) == "arm64" && $(uname) == "Darwin" ]]; then
  echo "Detected Apple Silicon Mac"
  
  if python -c "import mlx" &> /dev/null; then
    echo -e "${GREEN}✓ MLX is already installed${NC}"
  else
    echo -e "${YELLOW}⚠ MLX not found${NC}"
    echo "Would you like to install MLX for acceleration? (y/n)"
    read -r install_mlx
    if [[ $install_mlx == "y" ]]; then
      echo "Installing MLX..."
      pip install mlx
      echo -e "${GREEN}✓ MLX installed successfully${NC}"
    else
      echo -e "${YELLOW}⚠ Skipping MLX installation. Acceleration will not be available.${NC}"
    fi
  fi
else
  echo -e "${YELLOW}⚠ Not running on Apple Silicon. MLX acceleration will not be available.${NC}"
fi

# Validate imports
echo "Validating Python imports..."
python -c "from llamadb.core import is_mlx_available, VectorIndex; print('✅ Core imports validated')" || 
  { echo -e "${RED}✗ Failed to validate core imports${NC}"; }

python -c "from llamadb.core.mlx_acceleration import is_apple_silicon; print('✅ MLX acceleration imports validated')" || 
  { echo -e "${RED}✗ Failed to validate MLX acceleration imports${NC}"; }

python -c "from llamadb.core.accelerated_ops import cosine_similarity; print('✅ Accelerated ops imports validated')" || 
  { echo -e "${RED}✗ Failed to validate accelerated ops imports${NC}"; }

python -c "from llamadb.core.vector_index import VectorIndex; print('✅ Vector index imports validated')" || 
  { echo -e "${RED}✗ Failed to validate vector index imports${NC}"; }

echo
echo -e "${GREEN}===== Development Environment Setup Complete =====${NC}"
echo "To activate the environment in the future, run: source .venv/bin/activate"
echo "To run the demo script: python demo_script.py"
echo "To run the CLI: python -m llamadb.cli.main" 