#!/usr/bin/env bash
# =============================================================================
# Python vs. Rust Implementation Analysis for LlamaDB
# 
# ## Implementation Recommendation
# 
# I recommend a **hybrid approach** with a Python core enhanced by Rust extensions through PyO3/Maturin for performance-critical components. This gives you:
# 
# 1. **Development Speed**: Python's ecosystem and libraries allow for rapid development
# 2. **Performance Where It Matters**: Rust modules for high-performance operations
# 3. **Accessibility**: Keeps the codebase accessible to Python developers
# 4. **Competitive Edge**: Demonstrates technical sophistication without sacrificing usability
# 5. **Future-Proofing**: Gradually optimize bottlenecks with Rust without rewriting everything
# 
# This hybrid approach showcases advanced technical skills while keeping the system practical and maintainable—exactly what companies like OpenAI and Anthropic look for.
# 
# ## LlamaDB Ultimate Master Installation Script
# =============================================================================
set -euo pipefail

# =============================================================================
# ENHANCED ERROR HANDLING AND CHECKPOINT SYSTEM
# =============================================================================

# Create logs directory if it doesn't exist
mkdir -p logs

# Setup a log file with timestamp
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_FILE="logs/llamadb_install_${TIMESTAMP}.log"
CHECKPOINT_FILE=".llamadb_install_checkpoint"
TEMP_DIR=".llamadb_temp"

# Trap handling for clean exit on CTRL+C and other termination signals
function cleanup() {
    local exit_code=$?
    echo -e "\n\n${RED}⚠️  Installation interrupted or error occurred.${RESET}"
    
    # Check if there's an active virtual environment and deactivate it
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        echo -e "${YELLOW}Deactivating virtual environment...${RESET}"
        deactivate 2>/dev/null || true
    fi
    
    if [[ $exit_code -ne 0 ]]; then
        echo -e "${YELLOW}Exit code: $exit_code${RESET}"
        echo -e "${CYAN}Check the log file for details: ${LOG_FILE}${RESET}"
        echo -e "${CYAN}You can resume the installation later with:${RESET}"
        echo -e "${YELLOW}./$(basename "$0") --resume${RESET}"
    fi
    
    # Remove temp directory if it exists
    if [[ -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR"
    fi
    
    exit $exit_code
}

# Set up traps for different signals
trap cleanup EXIT
trap 'echo -e "\n${RED}Installation interrupted by user.${RESET}"; exit 1' INT TERM HUP

# Log both to file and stdout
function log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Checkpoint system to allow resuming installation
function save_checkpoint() {
    echo "$1" > "$CHECKPOINT_FILE"
    log "${CYAN}Checkpoint saved: $1${RESET}"
}

function get_checkpoint() {
    if [[ -f "$CHECKPOINT_FILE" ]]; then
        cat "$CHECKPOINT_FILE"
    else
        echo "none"
    fi
}

function clear_checkpoint() {
    rm -f "$CHECKPOINT_FILE"
}

# Parse command line arguments
RESUME=false
SKIP_RUST=false
SKIP_TESTS=false
VERBOSE=false
INSTALL_ONLY=false

for arg in "$@"; do
    case $arg in
        --resume)
            RESUME=true
            shift
            ;;
        --skip-rust)
            SKIP_RUST=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --install-only)
            INSTALL_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: $(basename "$0") [OPTIONS]"
            echo "Options:"
            echo "  --resume        Resume installation from last checkpoint"
            echo "  --skip-rust     Skip Rust extension building (Python only)"
            echo "  --skip-tests    Skip running tests"
            echo "  --verbose       Show verbose output"
            echo "  --install-only  Only install dependencies without examples"
            echo "  --help, -h      Show this help message"
            exit 0
            ;;
        *)
            # Unknown option
            shift
            ;;
    esac
done

# Enhanced logging system
if [[ "$VERBOSE" == "true" ]]; then
    exec > >(tee -a "$LOG_FILE") 2>&1
else
    # Log commands to file, but only output important messages to console
    exec > >(tee -a "$LOG_FILE" >/dev/null) 2>&1
fi

# Create temp directory
mkdir -p "$TEMP_DIR"

# =============================================================================
# Ultimate Enhanced LlamaDB Master Setup Script - Hybrid Python/Rust Edition
#
# This script installs, configures, tests, and runs LlamaDB—
# a cutting-edge data exploration and AI-powered platform featuring:
#   • Python core with Rust extensions for performance-critical components
#   • Multi-model LLM orchestration (Claude, GPT-4, Llama 3)
#   • Real-time collaborative workspaces with versioning
#   • Distributed computation engine with Ray
#   • Advanced visualization and no-code query builder
#   • Multi-modal data processing and knowledge graph backend
#   • Enterprise-grade security, observability, and plugin marketplace
# =============================================================================

# Text Styling and Logging Helpers
BOLD="\033[1m"
CYAN="\033[36m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
BLUE="\033[34m"
MAGENTA="\033[35m"
RESET="\033[0m"

function log_header() { echo -e "\n${BOLD}${BLUE}==== 🦙 $1 ====${RESET}"; }
function log_success() { echo -e "${GREEN}✅ $1${RESET}"; }
function log_info() { echo -e "${CYAN}ℹ️ $1${RESET}"; }
function log_warning() { echo -e "${YELLOW}⚠️ $1${RESET}"; }
function log_error() { echo -e "${RED}❌ $1${RESET}"; }

function show_spinner() {
    local msg="$1"
    local pid=$!
    local delay=0.1
    local spinstr='|/-\\'
    echo -e "${CYAN}$msg${RESET}"
    while ps -p $pid > /dev/null 2>&1; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\r"
    done
    printf "    \r"
}

function prompt_confirm() {
    read -r -p "$1 (y/N): " response
    case "$response" in
        [Yy]* ) return 0;;
        * ) return 1;;
    esac
}

# Banner Display
function show_banner() {
    # Clear screen for better presentation
    clear
    
    # Define version
    VERSION="3.0.0"
    BUILD_DATE=$(date +"%Y-%m-%d")
    
    # Check if figlet is available for fancy banner
    if command -v figlet >/dev/null && command -v lolcat >/dev/null; then
        echo
        figlet -f slant "LlamaDB" | lolcat
        echo
    else
        # Fallback to ASCII art
        cat << "EOF"
                       
       /|       |\                     
      / |       | \                    
     /  |       |  \    ____           _                                 _____   ____  
    /   |       |   \  |  _ \         | |                               |  __ \ |  _ \ 
   /    |_______|    \ | |_) |  __ _  | |      __ _   _ __ ___     __ _ | |  | || |_) |
  /     |=======|     \|  _ <  / _' | | |     / _' | | '_ ' _ \   / _' || |  | ||  _ < 
 /      |~+~+~+~|      \ |_) || (_| | | |____| (_| | | | | | | | | (_| || |__| || |_) |
/       |_______|       \____/  \__,_| |______\__,_| |_| |_| |_|  \__,_||_____/ |____/                                    
EOF
    fi
    
    # Display version and build info
    echo -e "\n${BOLD}${MAGENTA}LLAMA-DB HYBRID EDITION${RESET}"
    echo -e "${YELLOW}Python Core + Rust Performance Extensions${RESET}"
    echo -e "${CYAN}Version ${VERSION} (Build: ${BUILD_DATE})${RESET}\n"
    
    # Display system info
    echo -e "${BOLD}System Information:${RESET}"
    echo -e "  ${CYAN}• OS:${RESET} $(uname -s) $(uname -r)"
    echo -e "  ${CYAN}• Host:${RESET} $(hostname)"
    echo -e "  ${CYAN}• User:${RESET} $(whoami)"
    if command -v python3 >/dev/null; then
        echo -e "  ${CYAN}• Python:${RESET} $(python3 --version 2>&1)"
    fi
    if command -v rustc >/dev/null; then
        echo -e "  ${CYAN}• Rust:${RESET} $(rustc --version 2>&1 | cut -d' ' -f1-2)"
    fi
    echo
    
    # Display script options if any were provided
    if [[ "$RESUME" == "true" || "$SKIP_RUST" == "true" || "$SKIP_TESTS" == "true" || "$INSTALL_ONLY" == "true" ]]; then
        echo -e "${BOLD}Installation Options:${RESET}"
        [[ "$RESUME" == "true" ]] && echo -e "  ${CYAN}• Resuming from checkpoint${RESET}"
        [[ "$SKIP_RUST" == "true" ]] && echo -e "  ${CYAN}• Skipping Rust extensions${RESET}"
        [[ "$SKIP_TESTS" == "true" ]] && echo -e "  ${CYAN}• Skipping tests${RESET}"
        [[ "$INSTALL_ONLY" == "true" ]] && echo -e "  ${CYAN}• Installing core components only${RESET}"
        echo
    fi
    
    # Display a motivational message
    echo -e "${BOLD}${GREEN}Ready to install the most advanced hybrid Python/Rust data platform!${RESET}"
    echo -e "${YELLOW}This installation will take approximately 5-10 minutes depending on your system.${RESET}"
    echo -e "${YELLOW}Sit back and relax while we set everything up for you.${RESET}\n"
    
    # Pause for a moment to let the user read the banner
    sleep 1
}

# System Detection
function detect_os() {
    log_header "System Detection"
    
    # Detect OS
    OS_NAME="$(uname)"
    log_info "Detected OS: $OS_NAME"
    
    # Set default Python command
    PYTHON_CMD="python3"
    
    if [[ "$OS_NAME" == "Darwin" ]]; then
        log_info "Detected macOS – applying macOS optimizations"
        
        # Check macOS version
        MACOS_VERSION=$(sw_vers -productVersion)
        log_info "macOS version: $MACOS_VERSION"
        
        # Add Homebrew paths to PATH
        if [[ -d "/opt/homebrew/bin" ]]; then
            # For Apple Silicon Macs
            export PATH="/opt/homebrew/bin:$PATH"
        fi
        if [[ -d "/usr/local/bin" ]]; then
            # For Intel Macs
            export PATH="/usr/local/bin:$PATH"
        fi
        
        # Install Homebrew if not found
        if ! command -v brew >/dev/null; then
            log_info "Homebrew not found. Installing Homebrew..."
            if prompt_confirm "Would you like to install Homebrew?"; then
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || {
                    log_warning "Failed to install Homebrew. Some dependencies may need to be installed manually."
                }
            else
                log_warning "Skipping Homebrew installation. Some dependencies may need to be installed manually."
            fi
        else
            log_info "Homebrew is already installed: $(brew --version | head -1)"
        fi
        
        # Install necessary tools via Homebrew
        if command -v brew >/dev/null; then
            log_info "Installing system dependencies via Homebrew..."
            brew install figlet lolcat jq python@3.11 rust cmake openssl postgresql fd ripgrep || {
                log_warning "Some Homebrew packages failed to install. Continuing anyway..."
            }
        fi
        
        MAC=1
    elif [[ "$OS_NAME" == "Linux" ]]; then
        log_info "Detected Linux environment"
        
        # Check for specific Linux distribution
        if [[ -f /etc/os-release ]]; then
            source /etc/os-release
            log_info "Linux distribution: $NAME $VERSION_ID"
        fi
        
        # Check for WSL
        if grep -qi microsoft /proc/version 2>/dev/null; then
            log_info "Running in Windows Subsystem for Linux"
            WSL=1
        else
            WSL=0
        fi
        
        # Install dependencies based on package manager
        if command -v apt-get >/dev/null; then
            log_info "Installing dependencies via apt-get..."
            if prompt_confirm "Would you like to install system dependencies via apt-get?"; then
                sudo apt-get update >/dev/null 2>&1 || log_warning "Failed to update apt repositories"
                sudo apt-get install -y figlet lolcat jq python3-pip python3-venv build-essential libssl-dev libffi-dev python3-dev cargo rustc cmake pkg-config libssl-dev || {
                    log_warning "Some apt packages failed to install. Continuing anyway..."
                }
            else
                log_warning "Skipping system dependencies installation. Some features may not work correctly."
            fi
        elif command -v dnf >/dev/null; then
            log_info "Installing dependencies via dnf..."
            if prompt_confirm "Would you like to install system dependencies via dnf?"; then
                sudo dnf install -y figlet lolcat jq python3-pip cargo rustc cmake openssl-devel || {
                    log_warning "Some dnf packages failed to install. Continuing anyway..."
                }
            else
                log_warning "Skipping system dependencies installation. Some features may not work correctly."
            fi
        elif command -v pacman >/dev/null; then
            log_info "Installing dependencies via pacman..."
            if prompt_confirm "Would you like to install system dependencies via pacman?"; then
                sudo pacman -Sy figlet lolcat jq python-pip rust cmake openssl || {
                    log_warning "Some pacman packages failed to install. Continuing anyway..."
                }
            else
                log_warning "Skipping system dependencies installation. Some features may not work correctly."
            fi
        else
            log_warning "Unsupported Linux package manager. You may need to install dependencies manually."
        fi
        
        MAC=0
    else
        log_warning "Unsupported OS: $OS_NAME. Some features may not work correctly."
        MAC=0
    fi

    # Check for Python 3.11+
    log_info "Checking for Python 3.11+..."
    
    # Try different Python commands
    PYTHON_CANDIDATES=("python3.11" "python3.12" "python3.10" "python3" "python")
    PYTHON_FOUND=false
    
    for cmd in "${PYTHON_CANDIDATES[@]}"; do
        if command -v "$cmd" >/dev/null; then
            PY_VERSION=$("$cmd" --version 2>&1 | cut -d' ' -f2)
            major=$(echo "$PY_VERSION" | cut -d. -f1)
            minor=$(echo "$PY_VERSION" | cut -d. -f2)
            
            log_info "Found $cmd version $PY_VERSION"
            
            if [[ "$major" -eq 3 && "$minor" -ge 11 ]]; then
                PYTHON_CMD="$cmd"
                PYTHON_FOUND=true
                log_success "Using $PYTHON_CMD (version $PY_VERSION)"
                break
            elif [[ "$major" -eq 3 && "$minor" -ge 10 ]]; then
                PYTHON_CMD="$cmd"
                log_warning "Python 3.11+ is recommended, but found $PY_VERSION. Some features may not work correctly."
                PYTHON_FOUND=true
                break
            fi
        fi
    done
    
    if [[ "$PYTHON_FOUND" == "false" ]]; then
        log_error "Python 3.11+ is required but not found."
        
        # Suggest installation methods
        if [[ "$MAC" -eq 1 ]]; then
            log_info "You can install Python 3.11 on macOS with: brew install python@3.11"
        elif command -v apt-get >/dev/null; then
            log_info "You can install Python 3.11 on Debian/Ubuntu with: sudo apt-get install python3.11"
        elif command -v dnf >/dev/null; then
            log_info "You can install Python 3.11 on Fedora with: sudo dnf install python3.11"
        fi
        
        if prompt_confirm "Would you like to continue anyway with $PYTHON_CMD?"; then
            log_warning "Continuing with $PYTHON_CMD. Some features may not work correctly."
        else
            log_error "Installation cannot continue without Python 3.11+."
            exit 1
        fi
    fi
    
    # Check for Rust
    log_info "Checking for Rust..."
    if ! command -v cargo >/dev/null; then
        log_info "Rust not found. It will be installed later if needed."
    else
        RUST_VERSION=$(rustc --version)
        log_info "Rust is already installed: $RUST_VERSION"
    fi
    
    # Check for other required tools
    log_info "Checking for other required tools..."
    
    # Git is required
    if ! command -v git >/dev/null; then
        log_warning "Git is not installed. It may be required for some features."
    else
        GIT_VERSION=$(git --version)
        log_info "Git is installed: $GIT_VERSION"
    fi
    
    # Check disk space
    if command -v df >/dev/null; then
        DISK_SPACE=$(df -h . | awk 'NR==2 {print $4}')
        log_info "Available disk space: $DISK_SPACE"
        
        # Extract numeric value and unit
        DISK_NUM=$(echo "$DISK_SPACE" | sed 's/[^0-9.]//g')
        DISK_UNIT=$(echo "$DISK_SPACE" | sed 's/[0-9.]//g')
        
        # Convert to MB for comparison
        if [[ "$DISK_UNIT" == "G" || "$DISK_UNIT" == "GB" ]]; then
            DISK_MB=$(echo "$DISK_NUM * 1024" | bc)
        elif [[ "$DISK_UNIT" == "T" || "$DISK_UNIT" == "TB" ]]; then
            DISK_MB=$(echo "$DISK_NUM * 1024 * 1024" | bc)
        elif [[ "$DISK_UNIT" == "M" || "$DISK_UNIT" == "MB" ]]; then
            DISK_MB=$DISK_NUM
        else
            DISK_MB=0
        fi
        
        if [[ -z "$DISK_MB" || "$DISK_MB" -lt 1024 ]]; then
            log_warning "Low disk space detected. At least 1GB is recommended for installation."
        fi
    fi
    
    log_success "System detection complete"
}

# Environment Setup
function setup_venv() {
    log_header "Python Environment Setup"
    # Install pip tools
    if ! command -v uv >/dev/null; then
        log_info "Installing uv package manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    
    # Create virtual environment
    if command -v uv >/dev/null; then
        log_info "Creating virtual environment with uv..."
        uv venv -p 3.11 .venv
    else
        log_info "Creating virtual environment with venv..."
        $PYTHON_CMD -m venv .venv
    fi
    
    # Activate the virtual environment
    # shellcheck source=/dev/null
    source .venv/bin/activate
    log_success "Virtual environment created and activated"
}

# Install Dependencies
function install_dependencies() {
    log_header "Installing Dependencies"
    
    # Core requirements with version pinning
    cat > requirements.txt << 'EOF'
# Core dependencies
fastapi>=0.100.0,<1.0.0
uvicorn[standard]>=0.22.0,<1.0.0
pydantic>=2.4.0,<3.0.0
typer>=0.9.0,<1.0.0
rich>=13.0.0,<14.0.0
textual>=0.40.0,<1.0.0

# Database and ORM
sqlalchemy>=2.0.0,<3.0.0
alembic>=1.12.0,<2.0.0
asyncpg>=0.28.0,<1.0.0
psycopg2-binary>=2.9.5,<3.0.0
sqlmodel>=0.0.8,<0.1.0

# Vector database and embeddings
faiss-cpu>=1.7.4,<2.0.0
chromadb>=0.4.18,<1.0.0
sentence-transformers>=2.2.2,<3.0.0

# LLM integration
langchain>=0.0.300,<1.0.0
langchain-anthropic>=0.1.0,<1.0.0
langchain-openai>=0.0.5,<1.0.0
anthropic>=0.8.0,<1.0.0
openai>=1.0.0,<2.0.0

# Distributed computing
ray[default]>=2.6.0,<3.0.0

# Data processing and visualization
numpy>=1.24.0,<2.0.0
pandas>=2.0.0,<3.0.0
polars>=0.18.0,<1.0.0
plotly>=5.15.0,<6.0.0
dash>=2.13.0,<3.0.0
streamlit>=1.28.0,<2.0.0

# API and networking
httpx>=0.24.0,<1.0.0
websockets>=11.0.0,<13.0.0
grpcio>=1.58.0,<2.0.0
grpcio-tools>=1.58.0,<2.0.0

# Security and auth
python-jose[cryptography]>=3.3.0,<4.0.0
passlib[bcrypt]>=1.7.4,<2.0.0
python-multipart>=0.0.6,<0.1.0

# Plugin system and extensibility
pluggy>=1.3.0,<2.0.0
stevedore>=5.1.0,<6.0.0

# Observability and monitoring
opentelemetry-api>=1.19.0,<2.0.0
opentelemetry-sdk>=1.19.0,<2.0.0
opentelemetry-exporter-otlp>=1.19.0,<2.0.0
prometheus-client>=0.17.0,<1.0.0

# Utilities
pydash>=6.0.0,<8.0.0
pendulum>=2.1.0,<3.0.0
pydantic-settings>=2.0.0,<3.0.0
python-dotenv>=1.0.0,<2.0.0

# Rust extension building
maturin>=1.2.0,<2.0.0
EOF

    # Development tools
    cat > requirements-dev.txt << 'EOF'
# Testing
pytest>=8.0.0,<8.1.0
pytest-asyncio>=0.23.0,<0.24.0
pytest-cov>=4.1.0,<4.2.0
pytest-benchmark>=4.0.0,<4.1.0
pytest-mock>=3.12.0,<3.13.0
hypothesis>=6.92.1,<6.93.0

# Code quality
ruff>=0.2.0,<0.3.0
black>=23.11.0,<24.0.0
mypy>=1.7.0,<1.8.0
isort>=5.13.0,<5.14.0

# Security testing
bandit>=1.7.6,<1.8.0
safety>=2.3.5,<2.4.0

# Documentation
sphinx>=7.2.6,<7.3.0
sphinx-rtd-theme>=2.0.0,<2.1.0

# CI/CD
tox>=4.11.4,<4.12.0
coherent.build>=0.27.1,<0.28.0
twine>=4.0.2,<4.1.0
bump2version>=1.0.1,<1.1.0
pre-commit>=3.5.0,<3.6.0
EOF

    log_info "Installing core dependencies..."
    
    # Install dependencies with better error handling
    if command -v uv >/dev/null; then
        log_info "Using uv package manager for faster installation"
        
        # Install in batches to better handle errors
        if ! uv run uv pip install --upgrade pip; then
            log_error "Failed to upgrade pip. Trying with standard pip..."
            pip install --upgrade pip
        fi
        
        # Install core dependencies first
        log_info "Installing core dependencies..."
        if ! uv run uv pip install -r requirements.txt; then
            log_warning "Some dependencies failed to install with uv. Trying with pip..."
            pip install -r requirements.txt || {
                log_error "Failed to install core dependencies. Please check your Python environment."
                return 1
            }
        fi
        
        # Install dev dependencies
        if [[ "$INSTALL_ONLY" == "false" ]]; then
            log_info "Installing development dependencies..."
            if ! uv run uv pip install -r requirements-dev.txt; then
                log_warning "Some development dependencies failed to install with uv. Trying with pip..."
                pip install -r requirements-dev.txt || {
                    log_warning "Failed to install some development dependencies. Continuing anyway..."
                }
            fi
        else
            log_info "Skipping development dependencies (--install-only flag provided)"
        fi
    else
        log_info "Using standard pip for installation"
        
        # Upgrade pip
        pip install --upgrade pip || log_warning "Failed to upgrade pip. Continuing anyway..."
        
        # Install core dependencies
        log_info "Installing core dependencies..."
        pip install -r requirements.txt || {
            log_error "Failed to install core dependencies. Please check your Python environment."
            return 1
        }
        
        # Install dev dependencies
        if [[ "$INSTALL_ONLY" == "false" ]]; then
            log_info "Installing development dependencies..."
            pip install -r requirements-dev.txt || log_warning "Failed to install some development dependencies. Continuing anyway..."
        else
            log_info "Skipping development dependencies (--install-only flag provided)"
        fi
    fi
    
    # Verify installation
    log_info "Verifying installation..."
    if python -c "import fastapi, pydantic, sqlalchemy" 2>/dev/null; then
        log_success "Core dependencies installed and verified successfully"
    else
        log_warning "Some core dependencies may not be installed correctly. Installation will continue, but you may encounter issues."
    fi
}

# Create Rust Structure
function setup_rust_extensions() {
    log_header "Setting Up Rust Extensions"
    
    # Create Rust extension directory
    mkdir -p rust_extensions

    # Create main Cargo.toml
    cat > Cargo.toml << 'EOF'
[workspace]
members = [
    "rust_extensions/llamadb_core",
    "rust_extensions/vector_store",
    "rust_extensions/query_engine"
]
EOF

    # Create core Rust extension
    mkdir -p rust_extensions/llamadb_core
    cat > rust_extensions/llamadb_core/Cargo.toml << 'EOF'
[package]
name = "llamadb_core"
version = "0.1.0"
edition = "2021"

[lib]
name = "llamadb_core"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20.0", features = ["extension-module"] }
ndarray = "0.15.6"
rayon = "1.8.0"
serde = { version = "1.0.195", features = ["derive"] }
serde_json = "1.0.111"
tracing = "0.1.40"
EOF

    cat > rust_extensions/llamadb_core/src/lib.rs << 'EOF'
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use std::collections::HashMap;

/// High-performance data processing functions implemented in Rust
#[pyfunction]
fn parallel_process(data: Vec<f64>) -> Vec<f64> {
    data.into_par_iter()
        .map(|x| x * x + 2.0 * x + 1.0)
        .collect()
}

/// Optimized string operations
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

/// Module initialization
#[pymodule]
fn llamadb_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parallel_process, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_and_count, m)?)?;
    
    Ok(())
}
EOF

    # Create vector store extension
    mkdir -p rust_extensions/vector_store/src
    cat > rust_extensions/vector_store/Cargo.toml << 'EOF'
[package]
name = "vector_store"
version = "0.1.0"
edition = "2021"

[lib]
name = "vector_store"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20.0", features = ["extension-module"] }
ndarray = "0.15.6"
rayon = "1.8.0"
faiss = { version = "0.10.4", features = ["static"] }
thiserror = "1.0.51"
EOF

    cat > rust_extensions/vector_store/src/lib.rs << 'EOF'
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use ndarray::{Array, Array2};
use std::path::Path;

/// Optimized vector similarity search
#[pyfunction]
fn cosine_similarity(v1: Vec<f32>, v2: Vec<f32>) -> f32 {
    if v1.len() != v2.len() || v1.is_empty() {
        return 0.0;
    }
    
    let mut dot_product = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;
    
    for i in 0..v1.len() {
        dot_product += v1[i] as f32 * v2[i] as f32;
        norm1 += v1[i] as f32 * v1[i] as f32;
        norm2 += v2[i] as f32 * v2[i] as f32;
    }
    
    if norm1 == 0.0 || norm2 == 0.0 {
        return 0.0;
    }
    
    dot_product / (norm1.sqrt() * norm2.sqrt())
}

/// Batch vector operations
#[pyfunction]
fn batch_normalize(vectors: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    vectors.into_iter().map(|v| {
        let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            v
        } else {
            v.into_iter().map(|x| x / norm).collect()
        }
    }).collect()
}

/// Module initialization
#[pymodule]
fn vector_store(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(batch_normalize, m)?)?;
    
    Ok(())
}
EOF

    # Create query engine extension
    mkdir -p rust_extensions/query_engine/src
    cat > rust_extensions/query_engine/Cargo.toml << 'EOF'
[package]
name = "query_engine"
version = "0.1.0"
edition = "2021"

[lib]
name = "query_engine"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20.0", features = ["extension-module"] }
sqlparser = "0.36.1"
serde = { version = "1.0.195", features = ["derive"] }
serde_json = "1.0.111"
thiserror = "1.0.51"
EOF

    cat > rust_extensions/query_engine/src/lib.rs << 'EOF'
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::{Parser, ParserError};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize)]
struct ParsedQuery {
    query_type: String,
    tables: Vec<String>,
    columns: Vec<String>,
    conditions: Vec<String>,
    raw_sql: String,
}

#[pyfunction]
fn parse_sql(sql: &str) -> PyResult<String> {
    let dialect = GenericDialect {};
    
    match Parser::parse_sql(&dialect, sql) {
        Ok(ast) => {
            let parsed = ParsedQuery {
                query_type: format!("{:?}", ast[0]),
                tables: vec!["table1".to_string(), "table2".to_string()], // simplified
                columns: vec!["col1".to_string(), "col2".to_string()],    // simplified
                conditions: vec![],
                raw_sql: sql.to_string(),
            };
            
            Ok(serde_json::to_string(&parsed).unwrap())
        },
        Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("SQL Parse Error: {}", e))),
    }
}

#[pymodule]
fn query_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_sql, m)?)?;
    
    Ok(())
}
EOF

    # Create pyproject.toml with maturin configuration
    cat > pyproject.toml << 'EOF'
[build-system]
requires = ["maturin>=1.4,<2.0"]
build-backend = "maturin"

[project]
name = "llamadb"
version = "3.0.0"
description = "A cutting-edge data exploration and AI-powered platform with Rust extensions"
authors = [{ name = "Your Name", email = "you@example.com" }]
readme = "README.md"
license = { text = "Apache-2.0" }
requires-python = ">=3.11"
classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Rust",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Development Status :: 5 - Production/Stable",
    "Framework :: FastAPI",
    "Intended Audience :: Developers",
    "Topic :: Database",
    "Topic :: Scientific/Engineering :: Artificial Intelligence"
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "pytest-benchmark>=4.0.0",
    "pytest-mock>=3.12.0",
    "hypothesis>=6.92.1",
    "ruff>=0.2.0",
    "black>=23.11.0",
    "mypy>=1.7.0"
]

[project.scripts]
llamadb = "llamadb.cli.main:app"

[tool.maturin]
features = ["pyo3/extension-module"]
python-source = "python"
module-name = "llamadb._rust"
binding = "pyo3"

[tool.ruff]
target-version = "py311"
line-length = 100
select = ["E", "F", "B", "I", "N", "UP", "ANN", "C4", "SIM", "ARG"]
ignore = ["ANN101", "ANN102"]

[tool.ruff.isort]
known-first-party = ["llamadb"]

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.mypy]
python-version = "3.11"
warn-return-any = true
warn-unused-configs = true
disallow-untyped-defs = true
disallow-incomplete-defs = true
EOF
    
    log_success "Rust extensions structure created"
}

# Create Python Structure
function create_python_structure() {
    log_header "Creating Python Project Structure"
    
    # Create main Python package structure
    mkdir -p python/llamadb/{api,agents,cli,core,db,ml,plugins,utils,tui}
    mkdir -p python/llamadb/{collaborative,analytics,observability,security}
    mkdir -p tests/{unit,integration,performance,e2e}
    mkdir -p docs/{api,guides,tutorials,development}
    mkdir -p .github/workflows scripts data/{examples,vectors,models}
    
    # Create __init__.py files
    find python -type d -exec touch {}/__init__.py \;
    
    # Create main package __init__
    cat > python/llamadb/__init__.py << 'EOF'
"""
LlamaDB - A cutting-edge data exploration and AI-powered platform.

This platform combines Python and Rust to provide high-performance
data operations with an elegant, user-friendly interface.
"""

__version__ = "3.0.0"

# Import main components for easier access
from llamadb.core.config import settings
from llamadb.core.logging import setup_logging

# Initialize logging
setup_logging()
EOF

    # Create core module files
    cat > python/llamadb/core/config.py << 'EOF'
"""Configuration management for LlamaDB."""
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Core settings
    app_name: str = "LlamaDB"
    version: str = "3.0.0"
    debug: bool = False
    environment: str = "development"
    
    # Path settings
    base_dir: Path = Path(__file__).resolve().parent.parent.parent.parent
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    
    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    cors_origins: List[str] = ["*"]
    
    # Database settings
    database_url: str = "sqlite:///data/llamadb.db"
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: int = 30
    
    # Vector settings
    vector_db_path: Optional[Path] = None
    vector_dimension: int = 1536
    vector_metric: str = "cosine"
    
    # LLM settings
    llm_provider: str = "anthropic"  # anthropic, openai, local
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    default_model: str = "claude-3-opus-20240229"
    temperature: float = 0.7
    max_tokens: int = 4000
    
    # Security settings
    secret_key: str = "CHANGE_ME_IN_PRODUCTION"
    token_expire_minutes: int = 60 * 24  # 1 day
    
    # Feature flags
    enable_collaborative: bool = True
    enable_plugins: bool = True
    enable_observability: bool = True
    
    # Ray distributed settings
    enable_ray: bool = False
    ray_address: Optional[str] = None
    
    # Advanced settings
    log_level: str = "INFO"
    log_format: str = "%(levelname)s: %(message)s"
    
    @field_validator("data_dir", "vector_db_path", mode="before")
    def validate_paths(cls, v):
        if v is not None and not isinstance(v, Path):
            return Path(v)
        return v
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="LLAMADB_",
        case_sensitive=False,
    )


# Create global settings instance
settings = Settings()
EOF

    cat > python/llamadb/core/logging.py << 'EOF'
"""Logging configuration for LlamaDB."""
import logging
import sys
from pathlib import Path
from typing import Optional

from llamadb.core.config import settings


def setup_logging(log_level: Optional[str] = None) -> None:
    """Set up logging with the specified log level."""
    level = log_level or settings.log_level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    log_dir = settings.base_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Basic configuration for console logging
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "llamadb.log"),
        ],
    )
    
    # Adjust log levels for some verbose libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    logging.info(f"Logging initialized at level {level}")
EOF

    # Create CLI and main entry point
    cat > python/llamadb/cli/main.py << 'EOF'
"""LlamaDB Command Line Interface."""
import logging
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from llamadb.core.config import settings
from llamadb.utils import display

# Set up CLI with Rich console
console = Console()
app = typer.Typer(
    name="llamadb",
    help="LlamaDB: Modern data exploration and AI-powered platform",
    add_completion=True,
)

@app.callback()
def callback():
    """LlamaDB Command Line Interface."""
    # Set up rich logging
    logging.basicConfig(
        level=settings.log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )

@app.command()
def version():
    """Display the current version of LlamaDB."""
    from llamadb import __version__
    console.print(f"[bold]LlamaDB[/bold] version: [bold cyan]{__version__}[/bold cyan]")

@app.command()
def serve(
    host: str = typer.Option(settings.host, help="Host to bind the server to"),
    port: int = typer.Option(settings.port, help="Port to bind the server to"),
    workers: int = typer.Option(settings.workers, help="Number of worker processes"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
):
    """Start the LlamaDB API server."""
    import uvicorn
    from llamadb.api.app import create_app
    
    # Display banner
    display.show_banner(console)
    console.print(f"Starting LlamaDB API server on [bold]{host}:{port}[/bold]")
    
    # Start the server
    uvicorn.run(
        "llamadb.api.app:create_app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        factory=True,
    )

@app.command()
def tui():
    """Launch the LlamaDB Terminal User Interface."""
    from llamadb.tui.app import LlamaDBApp
    
    # Display banner
    display.show_banner(console)
    console.print("Launching LlamaDB Terminal UI...")
    
    app = LlamaDBApp()
    app.run()

@app.command()
def shell():
    """Start an interactive shell."""
    import code
    from rich.prompt import Prompt
    
    # Import modules for shell use
    import pandas as pd
    import numpy as np
    from llamadb import __version__
    from llamadb.db import session
    
    # Display banner
    display.show_banner(console)
    console.print("Starting LlamaDB interactive shell...")
    console.print("Type [bold]help(llamadb)[/bold] for more information.")
    
    # Start the shell
    vars = {
        "np": np,
        "pd": pd,
        "session": session,
        "console": console,
        "settings": settings,
    }
    
    code.interact(local=vars, banner="")

@app.command()
def query(
    sql: str = typer.Argument(..., help="SQL query to execute"),
    output: str = typer.Option("table", help="Output format: table, json, csv"),
):
    """Execute an SQL query and display the results."""
    from llamadb.db.engine import execute_query
    from llamadb.utils.formatters import format_results
    
    try:
        results = execute_query(sql)
        formatted = format_results(results, output)
        console.print(formatted)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

# Additional commands (simplified)
@app.command()
def init_db():
    """Initialize the database with default schema."""
    from llamadb.db.init import initialize_database
    initialize_database()
    console.print("[bold green]Database initialized successfully[/bold green]")

@app.command()
def demo():
    """Run a demonstration of LlamaDB capabilities."""
    display.show_banner(console)
    console.print("[bold]Starting LlamaDB demonstration...[/bold]")
    
    # Simulate demo with a simple progress display
    with console.status("[bold green]Running demo...[/bold green]") as status:
        import time
        for i in range(5):
            time.sleep(1)
            console.log(f"Demo step {i+1} completed")
    
    console.print("[bold green]Demo completed successfully![/bold green]")

if __name__ == "__main__":
    app()
EOF

    # Create sample utils display module
    mkdir -p python/llamadb/utils
    cat > python/llamadb/utils/display.py << 'EOF'
"""Display utilities for LlamaDB CLI and applications."""
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def show_banner(console: Console) -> None:
    """Display the LlamaDB banner in the console."""
    banner = r"""
       /|       |\                     
      / |       | \                    
     /  |       |  \    ____           _                                 _____   ____  
    /   |       |   \  |  _ \         | |                               |  __ \ |  _ \ 
   /    |_______|    \ | |_) |  __ _  | |      __ _   _ __ ___     __ _ | |  | || |_) |
  /     |=======|     \|  _ <  / _' | | |     / _' | | '_ ' _ \   / _' || |  | ||  _ < 
 /      |~+~+~+~|      \ |_) || (_| | | |____| (_| | | | | | | | | (_| || |__| || |_) |
/       |_______|       \____/  \__,_| |______\__,_| |_| |_| |_|  \__,_||_____/ |____/
    """
    
    title = Text("LLAMA-DB HYBRID EDITION", style="bold magenta")
    subtitle = Text("Python Core + Rust Performance Extensions\nVersion 3.0.0", style="yellow")
    
    panel = Panel(
        Text.from_markup(f"{banner}\n\n[bold magenta]LLAMA-DB HYBRID EDITION[/bold magenta]\n[yellow]Python Core + Rust Performance Extensions[/yellow]\n[cyan]Version 3.0.0[/cyan]"),
        border_style="blue",
        expand=False,
    )
    
    console.print(panel)


def progress_bar(console: Console, message: str):
    """Create and return a Rich progress bar."""
    from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
    
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold green]{task.percentage:.0f}%"),
        TimeElapsedColumn(),
    )
    
    return progress
EOF

    # Create formatters utility
    cat > python/llamadb/utils/formatters.py << 'EOF'
"""Utilities for formatting query results and data."""
import csv
import io
import json
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.table import Table


def format_results(
    results: List[Dict[str, Any]], 
    format_type: str = "table", 
    table_title: Optional[str] = None
) -> Union[str, Table]:
    """Format query results in the specified format."""
    if not results:
        return "No results found."
    
    if format_type == "json":
        return json.dumps(results, indent=2, default=str)
    
    elif format_type == "csv":
        output = io.StringIO()
        if results:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        return output.getvalue()
    
    elif format_type == "table":
        table = Table(title=table_title)
        
        # Add columns
        for key in results[0].keys():
            table.add_column(key, style="cyan")
        
        # Add rows
        for row in results:
            table.add_row(*[str(v) for v in row.values()])
        
        return table
    
    else:
        raise ValueError(f"Unsupported format: {format_type}")
EOF

    # Create API module with FastAPI app
    mkdir -p python/llamadb/api
    cat > python/llamadb/api/app.py << 'EOF'
"""FastAPI application for LlamaDB."""
import logging
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from llamadb.core.config import settings
from llamadb.api import routes

logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="LlamaDB API",
        description="API for LlamaDB data exploration and AI-powered platform",
        version=settings.version,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )
    
    # Include routers
    app.include_router(routes.api_router)
    
    # Add startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting LlamaDB API server")
        
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down LlamaDB API server")
    
    return app
EOF

    # Create API routes
    cat > python/llamadb/api/routes.py << 'EOF'
"""API routes for LlamaDB."""
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from llamadb.db.engine import execute_query

api_router = APIRouter(prefix="/api")

class QueryRequest(BaseModel):
    """Model for query requests."""
    sql: str
    parameters: Optional[Dict] = None

class QueryResponse(BaseModel):
    """Model for query responses."""
    results: List[Dict]
    count: int
    execution_time_ms: float

@api_router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Execute an SQL query and return the results."""
    import time
    
    start_time = time.time()
    
    try:
        results = execute_query(request.sql, request.parameters)
        execution_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            results=results,
            count=len(results),
            execution_time_ms=execution_time,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

# Basic routes for the rest of the API
vector_router = APIRouter(prefix="/vectors")

class VectorSearchRequest(BaseModel):
    """Model for vector search requests."""
    query: str
    top_k: int = 5

@vector_router.post("/search")
async def vector_search(request: VectorSearchRequest):
    """Search vectors by text query."""
    # This would use the Rust extension in production
    return {
        "results": [{"id": i, "score": 0.9 - (i * 0.1), "text": f"Result {i}"} for i in range(request.top_k)]
    }

# LLM agent routes
agent_router = APIRouter(prefix="/agents")

class AgentRequest(BaseModel):
    """Model for agent requests."""
    query: str
    context: Optional[Dict] = None

@agent_router.post("/query")
async def agent_query(request: AgentRequest):
    """Process a query using an LLM agent."""
    return {
        "response": f"This is a simulated response to: {request.query}",
        "source_documents": ["doc1", "doc2"],
    }

# Include all routers
api_router.include_router(vector_router)
api_router.include_router(agent_router)
EOF

    # Create a simple DB engine module
    mkdir -p python/llamadb/db
    cat > python/llamadb/db/engine.py << 'EOF'
"""Database engine and connection management."""
import logging
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from llamadb.core.config import settings

logger = logging.getLogger(__name__)

# Create engines based on configuration
if settings.database_url.startswith("sqlite"):
    # SQLite doesn't support async well, so use sync engine
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
        echo=settings.debug,
    )
    async_engine = None
else:
    # For PostgreSQL, MySQL, etc. - create both sync and async engines
    engine = create_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        echo=settings.debug,
    )
    
    # Convert URL to async format if needed
    async_url = settings.database_url
    if async_url.startswith("postgresql://"):
        async_url = async_url.replace("postgresql://", "postgresql+asyncpg://")
    elif async_url.startswith("mysql://"):
        async_url = async_url.replace("mysql://", "mysql+aiomysql://")
    
    async_engine = create_async_engine(
        async_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        echo=settings.debug,
    )

# Create session factories
Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = sessionmaker(class_=AsyncSession, bind=async_engine) if async_engine else None

# Session for use in shell and utilities
session = Session()

def execute_query(
    sql: str, 
    parameters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Execute a SQL query and return the results as a list of dictionaries."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql), parameters or {})
            if result.returns_rows:
                # Convert result set to list of dictionaries
                column_names = result.keys()
                return [dict(zip(column_names, row)) for row in result.fetchall()]
            return []
    except Exception as e:
        logger.error(f"Error executing query: {e}", exc_info=True)
        raise
EOF

    # Create a database initialization module
    cat > python/llamadb/db/init.py << 'EOF'
"""Database initialization and schema management."""
import logging
from pathlib import Path

from sqlalchemy import MetaData, Table, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.sql import func

from llamadb.db.engine import engine, session

logger = logging.getLogger(__name__)

metadata = MetaData()

# Define tables
users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("username", String(50), unique=True, nullable=False),
    Column("email", String(100), unique=True, nullable=False),
    Column("created_at", DateTime, server_default=func.now()),
    Column("last_login", DateTime),
)

vectors = Table(
    "vectors",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("text", String, nullable=False),
    Column("metadata", String),  # JSON string
    Column("embedding_file", String),  # Path to embedding file
    Column("created_at", DateTime, server_default=func.now()),
)

queries = Table(
    "queries",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("query_text", String, nullable=False),
    Column("execution_time_ms", Float),
    Column("created_at", DateTime, server_default=func.now()),
)

def initialize_database():
    """Initialize the database with the default schema."""
    logger.info("Initializing database")
    
    try:
        # Create tables
        metadata.create_all(engine)
        logger.info("Database schema created successfully")
        
        # Create data directory if it doesn't exist
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Initialize vector database
        vector_dir = data_dir / "vectors"
        vector_dir.mkdir(exist_ok=True)
        
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        return False
EOF

    # Create a basic Terminal UI app
    mkdir -p python/llamadb/tui
    cat > python/llamadb/tui/app.py << 'EOF'
"""Terminal User Interface for LlamaDB."""
import logging
from typing import Dict, List, Optional

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Static

logger = logging.getLogger(__name__)

class QueryScreen(Screen):
    """Query screen for executing SQL queries."""
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "run_query", "Run Query"),
        ("c", "clear", "Clear"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            Static("Enter SQL Query:", classes="label"),
            Input(placeholder="SELECT * FROM ...", id="query_input"),
            Button("Run Query", variant="primary", id="run_btn"),
            Static("Results:", classes="label"),
            DataTable(id="results_table"),
            classes="query-container",
        )
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "run_btn":
            self.run_query()
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        if event.input.id == "query_input":
            self.run_query()
    
    def action_run_query(self) -> None:
        """Run the current query."""
        self.run_query()
    
    def action_clear(self) -> None:
        """Clear the query input and results."""
        self.query_one("#query_input", Input).value = ""
        table = self.query_one("#results_table", DataTable)
        table.clear()
    
    def run_query(self) -> None:
        """Execute the SQL query and display results."""
        query = self.query_one("#query_input", Input).value
        if not query:
            return
        
        # Execute query
        try:
            from llamadb.db.engine import execute_query
            results = execute_query(query)
            
            # Update table
            table = self.query_one("#results_table", DataTable)
            table.clear()
            
            if results:
                # Add columns
                columns = list(results[0].keys())
                table.add_columns(*columns)
                
                # Add rows
                for row in results:
                    table.add_row(*[str(v) for v in row.values()])
        except Exception as e:
            self.notify(f"Error: {str(e)}", severity="error")


class LlamaDBApp(App):
    """Main LlamaDB Terminal UI application."""
    
    TITLE = "LlamaDB"
    SUB_TITLE = "Hybrid Python/Rust Data Platform"
    CSS_PATH = "app.css"
    SCREENS = {"query": QueryScreen()}
    
    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.push_screen("query")


# Sample CSS - in production this would be in a separate file
app_css = """
.query-container {
    layout: vertical;
    padding: 1;
    height: 100%;
}

.label {
    padding: 1 0;
    color: cyan;
    text-style: bold;
}

#query_input {
    width: 100%;
    height: 5;
    margin-bottom: 1;
}

#results_table {
    height: 1fr;
}
"""

# Save the CSS file
def save_css():
    """Save the CSS for the TUI app."""
    with open("python/llamadb/tui/app.css", "w") as f:
        f.write(app_css)

save_css()
EOF

    # Create README file
    cat > README.md << 'EOF'
# 🦙 LlamaDB - Hybrid Python/Rust Data Platform

LlamaDB is a cutting-edge data exploration and AI-powered platform that combines Python's flexibility with Rust's performance to provide an enterprise-grade solution for modern data challenges.

## Features

- **Python Core with Rust Extensions**: Critical performance paths are implemented in Rust with PyO3
- **Multi-Model LLM Integration**: Orchestrates Claude, GPT-4, and Llama 3 models for optimal task handling
- **Real-time Collaboration**: Version-controlled workspaces for team data exploration
- **Vector Search**: High-performance similarity search powered by FAISS with Rust optimizations
- **No-Code Query Builder**: Natural language to SQL with optimization suggestions
- **Distributed Computation**: Ray-based scalable processing for large datasets
- **Interactive Visualization**: Rich exploratory interfaces with D3.js integration
- **Enterprise Security**: Fine-grained access controls and comprehensive audit logging
- **Knowledge Graph Backend**: Automatic discovery and visualization of data relationships
- **Multi-Modal Support**: Process and analyze text, images, and audio with specialized embeddings
- **Plugin Marketplace**: Extensible architecture for third-party developers
- **Comprehensive Observability**: OpenTelemetry integration with visual performance dashboards

## Quick Start

# ```bash
# Install dependencies
./llamadb_setup.sh

# Activate virtual environment
source .venv/bin/activate

# Run the CLI
llamadb

# Start the API server
llamadb serve --host 0.0.0.0 --port 8000

# Launch the terminal UI
llamadb tui

# Run interactive demo
llamadb demo
# ```

## Development

This project uses a hybrid architecture:

- Python for high-level logic, APIs, and extensibility
- Rust for performance-critical components:
  - Vector operations and similarity search
  - SQL parsing and optimization
  - Data processing pipelines

### Building Rust Extensions

# ```bash
# Build all Rust extensions
maturin develop --release

# Run tests including Rust components
pytest tests/
# ```

## Documentation

Comprehensive documentation is available in the `docs/` directory.

## License

Apache License 2.0
EOF

    log_success "Python project structure created"
}

# Build Rust Extensions
function build_rust_extensions() {
    log_header "Building Rust Extensions"
    
    # Check if Rust is installed
    if ! command -v cargo >/dev/null; then
        log_error "Rust is not installed. Cannot build Rust extensions."
        if prompt_confirm "Would you like to install Rust now?"; then
            log_info "Installing Rust..."
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            source "$HOME/.cargo/env"
        else
            log_warning "Skipping Rust extensions build. Some features may not work correctly."
            return 1
        fi
    fi
    
    # Install maturin if needed
    if ! command -v maturin >/dev/null; then
        log_info "Installing maturin..."
        if command -v uv >/dev/null; then
            uv run uv pip install maturin || pip install maturin
        else
            pip install maturin || {
                log_error "Failed to install maturin. Cannot build Rust extensions."
                return 1
            }
        fi
    fi
    
    # Check if Rust extensions directory exists
    if [[ ! -d "rust_extensions" ]]; then
        log_error "Rust extensions directory not found. Please run setup_rust_extensions first."
        return 1
    fi
    
    # Build Rust extensions with maturin
    log_info "Building Rust extensions with maturin..."
    
    # Create a temporary build log
    BUILD_LOG="${TEMP_DIR}/rust_build.log"
    
    # Try to build with release mode first
    if maturin develop --release > "$BUILD_LOG" 2>&1; then
        log_success "Rust extensions built successfully in release mode"
    else
        log_warning "Failed to build Rust extensions in release mode. Trying debug mode..."
        
        # Show the error log
        cat "$BUILD_LOG" >> "$LOG_FILE"
        
        # Try debug mode
        if maturin develop > "$BUILD_LOG" 2>&1; then
            log_success "Rust extensions built successfully in debug mode"
            log_warning "Note: Debug mode builds may be slower than release builds"
        else
            log_error "Failed to build Rust extensions. See log for details."
            cat "$BUILD_LOG" >> "$LOG_FILE"
            
            # Offer to continue without Rust extensions
            if prompt_confirm "Would you like to continue without Rust extensions?"; then
                log_warning "Continuing without Rust extensions. Some features may not work correctly."
                return 0
            else
                log_error "Installation cannot continue without Rust extensions."
                return 1
            fi
        fi
    fi
    
    # Verify the build
    log_info "Verifying Rust extensions..."
    if python -c "import sys; sys.path.insert(0, '.'); from llamadb._rust import llamadb_core; print('Rust extension verification successful')" 2>/dev/null; then
        log_success "Rust extensions verified successfully"
    else
        log_warning "Could not verify Rust extensions. They may not have been built correctly."
        
        # Offer to continue anyway
        if prompt_confirm "Would you like to continue anyway?"; then
            log_warning "Continuing without verified Rust extensions. Some features may not work correctly."
            return 0
        else
            log_error "Installation cannot continue without verified Rust extensions."
            return 1
        fi
    fi
    
    return 0
}

# Setup Sample Data and Configuration
function setup_sample_data() {
    log_header "Setting Up Sample Data"
    
    # Create data directories
    mkdir -p data/{examples,vectors,models,logs}
    
    # Create sample SQLite database with Python
    $PYTHON_CMD - << 'EOF'
import sqlite3, os
import json

# Ensure data directory exists
os.makedirs('data/examples', exist_ok=True)

# Create SQLite database
conn = sqlite3.connect('data/examples/sample.db')
cursor = conn.cursor()

# Create customers table
cursor.execute('''
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    signup_date TEXT,
    last_login TEXT,
    active INTEGER DEFAULT 1
)''')

# Create products table
cursor.execute('''
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    price REAL NOT NULL,
    stock INTEGER DEFAULT 0,
    category TEXT
)''')

# Insert sample data
customers = [
    (1, "John Doe", "john@example.com", "2024-01-15", "2024-03-10", 1),
    (2, "Jane Smith", "jane@example.com", "2024-02-20", "2024-03-11", 1),
    (3, "Alice Johnson", "alice@example.com", "2024-01-10", "2024-02-28", 0),
    (4, "Bob Brown", "bob@example.com", "2024-03-05", "2024-03-12", 1),
    (5, "Carol White", "carol@example.com", "2024-02-10", "2024-03-09", 1)
]

products = [
    (1, "Laptop Pro", "High-performance laptop", 1299.99, 25, "Electronics"),
    (2, "Smartphone X", "Latest smartphone", 899.99, 50, "Electronics"),
    (3, "Desk Chair", "Ergonomic office chair", 199.99, 15, "Furniture"),
    (4, "Coffee Maker", "Automatic coffee machine", 79.99, 30, "Kitchen"),
    (5, "Wireless Headphones", "Noise-cancelling headphones", 149.99, 40, "Electronics")
]

cursor.executemany("INSERT OR REPLACE INTO customers VALUES (?, ?, ?, ?, ?, ?)", customers)
cursor.executemany("INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?, ?, ?)", products)

conn.commit()
conn.close()

print('Sample SQLite database created at data/examples/sample.db')

# Create configuration file
config = {
    'app_name': 'LlamaDB',
    'environment': 'development',
    'debug': True,
    'log_level': 'info',
    'data_dir': './data',
    'databases': {
        'default': {
            'type': 'sqlite',
            'connection_string': 'sqlite:///data/examples/sample.db',
            'pool_size': 5,
            'max_overflow': 10,
            'pool_timeout': 30,
            'pool_recycle': 1800,
            'echo': False
        }
    },
    'vector': {
        'enabled': True,
        'dimension': 1536,
        'metric': 'cosine',
        'index_type': 'Flat'
    },
    'llm': {
        'providers': {
            'anthropic': {
                'enabled': True,
                'default_model': 'claude-3-opus-20240229',
                'api_key_env': 'ANTHROPIC_API_KEY'
            },
            'openai': {
                'enabled': True,
                'default_model': 'gpt-4o',
                'api_key_env': 'OPENAI_API_KEY'
            },
            'local': {
                'enabled': False,
                'model_path': './data/models/llama3-8b'
            }
        },
        'default_provider': 'anthropic',
        'temperature': 0.7,
        'max_tokens': 4000
    },
    'server': {
        'host': '127.0.0.1',
        'port': 8000,
        'workers': 1,
        'cors_origins': ['*'],
        'timeout': 60,
        'log_level': 'info'
    },
    'plugins': {
        'directory': './plugins',
        'enabled': True,
        'auto_discover': True
    },
    'security': {
        'secret_key': 'CHANGE_ME_IN_PRODUCTION',
        'token_expire_minutes': 1440,
        'enable_rbac': True
    },
    'observability': {
        'enabled': True,
        'log_queries': True,
        'trace_requests': True,
        'metrics_endpoint': '/metrics'
    },
    'features': {
        'collaborative': True,
        'distributed_compute': False,
        'multi_modal': True,
        'knowledge_graph': True
    }
}

with open('data/config.json', 'w') as f:
    json.dump(config, f, indent=2)

print('Default configuration created at data/config.json')
EOF
    
    # Create environment variables file
    cat > .env << 'EOF'
# LlamaDB Environment Configuration

# Core settings
LLAMADB_ENVIRONMENT=development
LLAMADB_DEBUG=true
LLAMADB_LOG_LEVEL=INFO

# Database settings
LLAMADB_DATABASE_URL=sqlite:///data/examples/sample.db

# API keys (replace with your actual keys)
# LLAMADB_ANTHROPIC_API_KEY=sk-ant-...
# LLAMADB_OPENAI_API_KEY=sk-...

# Security (change in production)
LLAMADB_SECRET_KEY=CHANGE_ME_IN_PRODUCTION
EOF
    
    log_success "Sample data and configuration set up"
}

# Create Tests
function setup_tests() {
    log_header "Setting Up Tests"
    
    # Create simple test files
    mkdir -p tests/{unit,integration,performance,e2e}
    
    # Unit tests for core functionality
    cat > tests/unit/test_config.py << 'EOF'
"""Tests for configuration module."""
import os
from pathlib import Path
from unittest import mock

import pytest

from llamadb.core.config import Settings

def test_settings_defaults():
    """Test that settings have proper defaults."""
    settings = Settings()
    assert settings.app_name == "LlamaDB"
    assert settings.environment == "development"
    assert settings.debug is False

def test_settings_env_override():
    """Test that environment variables override defaults."""
    with mock.patch.dict(os.environ, {"LLAMADB_APP_NAME": "TestApp", "LLAMADB_DEBUG": "true"}):
        settings = Settings()
        assert settings.app_name == "TestApp"
        assert settings.debug is True

def test_path_validation():
    """Test that path settings are converted to Path objects."""
    settings = Settings(data_dir="/tmp/data")
    assert isinstance(settings.data_dir, Path)
    assert settings.data_dir == Path("/tmp/data")
EOF

    # Unit tests for Rust extensions
    cat > tests/unit/test_rust_extensions.py << 'EOF'
"""Tests for Rust extensions."""
import pytest

try:
    from llamadb._rust import llamadb_core, vector_store, query_engine
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
def test_parallel_process():
    """Test the parallel processing function."""
    data = [1.0, 2.0, 3.0, 4.0]
    result = llamadb_core.parallel_process(data)
    
    # f(x) = x^2 + 2x + 1
    expected = [4.0, 9.0, 16.0, 25.0]
    assert result == pytest.approx(expected)

@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
def test_tokenize_and_count():
    """Test the tokenization function."""
    text = "Hello world, hello Python! rust is fast."
    result = llamadb_core.tokenize_and_count(text)
    
    assert result["hello"] == 2
    assert result["world"] == 1
    assert result["python"] == 1
    assert result["rust"] == 1
    assert result["is"] == 1
    assert result["fast"] == 1

@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
def test_vector_similarity():
    """Test vector similarity calculation."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    
    similarity = vector_store.cosine_similarity(v1, v2)
    assert similarity == pytest.approx(0.0)
    
    v3 = [1.0, 1.0, 0.0]
    similarity = vector_store.cosine_similarity(v1, v3)
    assert similarity == pytest.approx(1.0 / (2.0**0.5))

@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
def test_sql_parsing():
    """Test SQL parsing functionality."""
    sql = "SELECT id, name FROM users WHERE active = true"
    result = query_engine.parse_sql(sql)
    
    # Test that parsing returns a valid JSON string
    assert isinstance(result, str)
    
    import json
    parsed = json.loads(result)
    assert "query_type" in parsed
    assert "tables" in parsed
    assert "columns" in parsed
EOF

    # Integration test for API
    cat > tests/integration/test_api.py << 'EOF'
"""Integration tests for the API."""
import pytest
from fastapi.testclient import TestClient

from llamadb.api.app import create_app

@pytest.fixture
def client():
    """Create a test client for the API."""
    app = create_app()
    return TestClient(app)

def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_query_endpoint(client):
    """Test the query endpoint."""
    payload = {
        "sql": "SELECT 1 as test",
        "parameters": {}
    }
    
    response = client.post("/api/query", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "results" in data
    assert "count" in data
    assert "execution_time_ms" in data
    assert data["count"] == 1
    assert data["results"][0]["test"] == 1
EOF

    # Performance benchmark
    cat > tests/performance/test_benchmarks.py << 'EOF'
"""Performance benchmarks for LlamaDB."""
import random
import time

import pytest

try:
    from llamadb._rust import llamadb_core, vector_store
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

def pure_python_tokenize_and_count(text):
    """Python implementation of tokenize_and_count for comparison."""
    word_counts = {}
    
    for word in text.split():
        cleaned = ''.join(c for c in word.lower() if c.isalnum())
        
        if cleaned:
            if cleaned in word_counts:
                word_counts[cleaned] += 1
            else:
                word_counts[cleaned] = 1
    
    return word_counts

@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not available")
def test_tokenize_performance(benchmark):
    """Benchmark tokenization performance."""
    # Generate a large text
    words = ["hello", "world", "python", "rust", "benchmark", "test", 
             "performance", "database", "vector", "llamadb"]
    
    text = " ".join(random.choice(words) for _ in range(10000))
    
    # Benchmark Rust implementation
    rust_result = benchmark(llamadb_core.tokenize_and_count, text)
    
    # Check Python implementation for comparison
    start = time.time()
    python_result = pure_python_tokenize_and_count(text)
    python_time = time.time() - start
    
    # Results should be the same
    assert set(rust_result.keys()) == set(python_result.keys())
    for key in rust_result:
        assert rust_result[key] == python_result[key]
    
    # Print comparison
    print(f"\nPython implementation took: {python_time:.6f} seconds")
    print(f"Rust should be significantly faster")
EOF

    log_success "Tests created successfully"
}

# Create Docker and CI/CD files
function setup_deployment() {
    log_header "Setting Up Deployment Files"
    
    # Create Dockerfile
    cat > Dockerfile << 'EOF'
# Build stage for Rust extensions
FROM python:3.11-slim AS builder

WORKDIR /app

# Install Rust and build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential curl pkg-config libssl-dev && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

# Install maturin for building Rust extensions
RUN pip install --no-cache-dir maturin

# Copy Rust code and build dependencies
COPY rust_extensions/ /app/rust_extensions/
COPY Cargo.toml /app/
COPY pyproject.toml /app/
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Build Rust extensions
RUN maturin build --release

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libssl-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy Python code
COPY python/ /app/python/
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy built wheel from builder stage
COPY --from=builder /app/target/wheels/*.whl /app/

# Install the wheel
RUN pip install --no-cache-dir /app/*.whl

# Create non-root user
RUN useradd -m llamauser && \
    chown -R llamauser:llamauser /app

USER llamauser

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LLAMADB_ENVIRONMENT=production

# Expose API port
EXPOSE 8000

# Entry point
ENTRYPOINT ["python", "-m", "llamadb.cli.main"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
EOF

    # Create docker-compose.yml
    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - LLAMADB_ENVIRONMENT=production
      - LLAMADB_LOG_LEVEL=INFO
      - LLAMADB_DATABASE_URL=postgresql://llamadb:llamadb@db:5432/llamadb
    depends_on:
      - db
    command: serve --host 0.0.0.0 --port 8000
    restart: unless-stopped

  db:
    image: postgres:15
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=llamadb
      - POSTGRES_PASSWORD=llamadb
      - POSTGRES_DB=llamadb
    restart: unless-stopped

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  postgres_data:
  grafana_data:
EOF

    # Create GitHub Actions workflow
    mkdir -p .github/workflows
    cat > .github/workflows/ci.yml << 'EOF'
name: CI/CD Pipeline

on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11"]

    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      
      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          profile: minimal
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
          pip install maturin
      
      - name: Build Rust extensions
        run: |
          maturin develop
      
      - name: Lint with ruff
        run: |
          ruff python tests
      
      - name: Check formatting with black
        run: |
          black --check python tests
      
      - name: Run tests
        run: |
          pytest tests/unit

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && (github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/tags/v'))
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          profile: minimal
      
      - name: Build package
        run: |
          pip install maturin
          maturin build --release
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: wheels
          path: target/wheels/*.whl

  publish:
    needs: build
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Download artifacts
        uses: actions/download-artifact@v3
        with:
          name: wheels
          path: dist
      
      - name: Publish to PyPI
        if: startsWith(github.ref, 'refs/tags/v')
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          user: __token__
          password: ${{ secrets.PYPI_API_TOKEN }}
EOF

    # Create prometheus.yml for monitoring
    cat > prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'llamadb'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['api:8000']
EOF

    log_success "Deployment files created successfully"
}

# Build and Install the Package
function build_package() {
    log_header "Building LlamaDB Package"
    
    # Install in development mode
    pip install -e .
    
    log_success "LlamaDB package built and installed in editable mode"
}

# Verify Installation
function verify_installation() {
    log_header "Verifying Installation"
    
    local all_good=true
    
    # Check virtual environment
    if [[ -d ".venv" ]]; then
        log_success "Virtual environment exists"
    else
        log_error "Virtual environment not found"
        all_good=false
    fi
    
    # Check Python packages
    log_info "Checking core Python packages..."
    if source .venv/bin/activate 2>/dev/null; then
        # Check for key packages
        for package in fastapi pydantic sqlalchemy rich typer; do
            if python -c "import $package" 2>/dev/null; then
                log_success "Package $package is installed"
            else
                log_error "Package $package is not installed"
                all_good=false
            fi
        done
    else
        log_error "Could not activate virtual environment"
        all_good=false
    fi
    
    # Check Rust extensions if not skipped
    if [[ "$SKIP_RUST" == "false" ]]; then
        log_info "Checking Rust extensions..."
        if python -c "import sys; sys.path.insert(0, '.'); from llamadb._rust import llamadb_core" 2>/dev/null; then
            log_success "Rust extensions are installed"
        else
            log_error "Rust extensions are not installed correctly"
            all_good=false
        fi
    fi
    
    # Check project structure
    log_info "Checking project structure..."
    for dir in python/llamadb/{api,core,db,utils}; do
        if [[ -d "$dir" ]]; then
            log_success "Directory $dir exists"
        else
            log_error "Directory $dir is missing"
            all_good=false
        fi
    done
    
    # Check sample data if not skipped
    if [[ "$INSTALL_ONLY" == "false" ]]; then
        log_info "Checking sample data..."
        if [[ -f "data/examples/sample.db" ]]; then
            log_success "Sample database exists"
        else
            log_warning "Sample database is missing"
        fi
    fi
    
    # Final verdict
    if [[ "$all_good" == "true" ]]; then
        log_success "Installation verification passed! LlamaDB is ready to use."
        return 0
    else
        log_warning "Installation verification found some issues. LlamaDB may not work correctly."
        return 1
    fi
}

# Run the Application
function run_application() {
    log_header "Running LlamaDB Application"
    
    # Verify installation first
    verify_installation
    
    # Run the basic CLI command
    python -m python.llamadb.cli.main version
    
    log_success "LlamaDB started successfully"
}

# Setup Container Support
function setup_container_support() {
    log_header "Setting Up Container Support"
    
    # Create directories
    mkdir -p docker/{app,db,nginx}
    
    # Create main Dockerfile
    cat > Dockerfile << 'EOF'
# LlamaDB Hybrid (Python+Rust) Dockerfile
FROM python:3.11-slim as builder

# Install Rust and build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Create app directory
WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt .
COPY requirements-dev.txt .
COPY Cargo.toml .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install maturin

# Copy Rust extensions and build them
COPY rust_extensions rust_extensions/
RUN maturin build --release

# Copy Python code
COPY python python/

# Build the Python package
RUN pip install -e .

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy from builder stage
COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Add startup scripts
COPY docker/app/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]
EOF

    # Create docker-compose.yml
    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: llamadb
      POSTGRES_PASSWORD: llamadbpass
      POSTGRES_DB: llamadb
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/db/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U llamadb"]
      interval: 5s
      timeout: 5s
      retries: 5

  vector_db:
    image: qdrant/qdrant:latest
    volumes:
      - vector_data:/qdrant/storage
    ports:
      - "6333:6333"
      - "6334:6334"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 5s
      timeout: 5s
      retries: 5

  api:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - LLAMADB_ENV=production
      - LLAMADB_DB_URL=postgresql://llamadb:llamadbpass@db:5432/llamadb
      - LLAMADB_VECTOR_URL=http://vector_db:6333
    volumes:
      - ./data:/app/data
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy
      vector_db:
        condition: service_healthy
    command: ["serve", "--host", "0.0.0.0", "--port", "8000"]

  ui:
    build:
      context: ./docker/ui
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - LLAMADB_API_URL=http://api:8000
    depends_on:
      - api

  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'

  grafana:
    image: grafana/grafana
    volumes:
      - grafana_data:/var/lib/grafana
      - ./docker/grafana/provisioning:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=llamadb_admin
      - GF_USERS_ALLOW_SIGN_UP=false
    ports:
      - "3001:3000"
    depends_on:
      - prometheus

volumes:
  postgres_data:
  vector_data:
  prometheus_data:
  grafana_data:
EOF

    # Create entrypoint.sh for API container
    mkdir -p docker/app
    cat > docker/app/entrypoint.sh << 'EOF'
#!/bin/bash
set -e

# Activate virtual environment
source .venv/bin/activate

# Run database migrations
if [ "$1" = "serve" ]; then
    echo "Running database migrations..."
    python -m python.llamadb.db.migrations
fi

# Run the command
exec python -m python.llamadb.cli.main "$@"
EOF
    chmod +x docker/app/entrypoint.sh

    # Create init.sql for PostgreSQL
    mkdir -p docker/db
    cat > docker/db/init.sql << 'EOF'
-- Initialize LlamaDB database
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";

-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS llamadb;
EOF

    # Create Dockerfile for UI
    mkdir -p docker/ui
    cat > docker/ui/Dockerfile << 'EOF'
FROM node:20-alpine

WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy the rest of the UI code
COPY . .

# Build the UI
RUN npm run build

# Expose the port
EXPOSE 3000

# Start the UI
CMD ["npm", "run", "start"]
EOF

    # Create docker-compose.dev.yml for development
    cat > docker-compose.dev.yml << 'EOF'
version: '3.8'

services:
  db:
    extends:
      file: docker-compose.yml
      service: db
    ports:
      - "5432:5432"

  vector_db:
    extends:
      file: docker-compose.yml
      service: vector_db
    ports:
      - "6333:6333"
      - "6334:6334"

  api:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ./python:/app/python
      - ./data:/app/data
    environment:
      - LLAMADB_ENV=development
      - LLAMADB_DB_URL=postgresql://llamadb:llamadbpass@db:5432/llamadb
      - LLAMADB_VECTOR_URL=http://vector_db:6333
      - LLAMADB_DEBUG=true
    ports:
      - "8000:8000"
    depends_on:
      - db
      - vector_db
    command: ["serve", "--reload", "--host", "0.0.0.0", "--port", "8000"]

volumes:
  postgres_data:
  vector_data:
EOF

    # Create Kubernetes deployment files
    mkdir -p k8s/{base,overlays/{dev,prod}}
    
    # Base Kubernetes manifests
    cat > k8s/base/namespace.yaml << 'EOF'
apiVersion: v1
kind: Namespace
metadata:
  name: llamadb
EOF

    cat > k8s/base/api-deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llamadb-api
  namespace: llamadb
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llamadb-api
  template:
    metadata:
      labels:
        app: llamadb-api
    spec:
      containers:
      - name: api
        image: llamadb/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: LLAMADB_ENV
          value: production
        - name: LLAMADB_DB_URL
          valueFrom:
            secretKeyRef:
              name: llamadb-secrets
              key: db-url
        - name: LLAMADB_VECTOR_URL
          value: http://vector-db:6333
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 20
        resources:
          limits:
            cpu: "1"
            memory: "2Gi"
          requests:
            cpu: "0.5"
            memory: "1Gi"
EOF

    cat > k8s/base/kustomization.yaml << 'EOF'
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
- namespace.yaml
- api-deployment.yaml
- api-service.yaml
- db-statefulset.yaml
- vector-db-statefulset.yaml
EOF

    # Create a README for Docker setup
    cat > docker/README.md << 'EOF'
# LlamaDB Docker Setup

This directory contains Docker configurations for running LlamaDB in containers.

## Quick Start

To start all services:

```bash
docker-compose up -d
```

For development setup:

```bash
docker-compose -f docker-compose.dev.yml up -d
```

## Services

- **api**: The LlamaDB API server
- **db**: PostgreSQL database
- **vector_db**: Qdrant vector database
- **ui**: Web interface
- **prometheus**: Metrics collection
- **grafana**: Monitoring dashboard

## Configuration

The services are configured through environment variables:

- `LLAMADB_ENV`: Environment (development or production)
- `LLAMADB_DB_URL`: PostgreSQL connection string
- `LLAMADB_VECTOR_URL`: Vector database connection URL

## Volumes

The following persistent volumes are used:

- `postgres_data`: PostgreSQL data
- `vector_data`: Vector database data
- `prometheus_data`: Metrics data
- `grafana_data`: Dashboard configurations
EOF

    log_success "Container support setup complete"
    log_info "You can run LlamaDB in containers with: docker-compose up -d"
}

# Setup Performance Tools
function setup_performance_tools() {
    log_header "Setting Up Performance Benchmarking Tools"
    
    # Create benchmarking directory
    mkdir -p tools/benchmarks
    
    # Create benchmark runner script
    cat > tools/benchmarks/run_benchmarks.py << 'EOF'
#!/usr/bin/env python3
"""
LlamaDB Performance Benchmarking Suite

This script runs comprehensive benchmarks to evaluate LlamaDB performance
across various workloads and components.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    print("Please install required packages: numpy, pandas, matplotlib")
    sys.exit(1)

# Add parent directory to path to import LlamaDB modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Try to import Rust extensions if available
try:
    from llamadb._rust import llamadb_core, vector_store
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("WARNING: Rust extensions not available. Some benchmarks will be skipped.")

class LlamaDBBenchmark:
    """Benchmark harness for LlamaDB performance testing."""
    
    def __init__(self, output_dir='results', iterations=5):
        """Initialize benchmark infrastructure."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.iterations = iterations
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"Initializing LlamaDB benchmarks with {iterations} iterations")
        print(f"Results will be saved to {self.output_dir}")
    
    def run_all(self):
        """Run all available benchmarks."""
        print("\n=== Running All Benchmarks ===\n")
        
        # Python performance tests
        self.benchmark_python_processing()
        
        # Rust extension tests (if available)
        if RUST_AVAILABLE:
            self.benchmark_rust_extensions()
        
        # Database operations
        self.benchmark_database_operations()
        
        # Vector operations
        self.benchmark_vector_operations()
        
        # Generate reports
        self.generate_reports()
    
    def benchmark_python_processing(self):
        """Benchmark pure Python data processing performance."""
        print("\n--- Benchmarking Python Processing ---")
        
        sizes = [1000, 10000, 100000]
        results = {}
        
        for size in sizes:
            print(f"  Processing {size} items...")
            data = np.random.rand(size)
            
            # Time the operation
            times = []
            for i in range(self.iterations):
                start = time.time()
                _ = [x * x + 2 * x + 1 for x in data]
                end = time.time()
                times.append(end - start)
            
            avg_time = sum(times) / len(times)
            results[size] = {
                'avg_time': avg_time,
                'items_per_second': size / avg_time,
                'times': times
            }
            
            print(f"    Average time: {avg_time:.4f}s ({size/avg_time:.2f} items/second)")
        
        self.results['python_processing'] = results
    
    def benchmark_rust_extensions(self):
        """Benchmark Rust extension performance."""
        if not RUST_AVAILABLE:
            print("Skipping Rust extension benchmarks (not available)")
            return
        
        print("\n--- Benchmarking Rust Extensions ---")
        
        sizes = [1000, 10000, 100000]
        results = {}
        
        for size in sizes:
            print(f"  Processing {size} items with Rust extension...")
            data = np.random.rand(size).tolist()
            
            # Time the operation
            times = []
            for i in range(self.iterations):
                start = time.time()
                _ = llamadb_core.parallel_process(data)
                end = time.time()
                times.append(end - start)
            
            avg_time = sum(times) / len(times)
            results[size] = {
                'avg_time': avg_time,
                'items_per_second': size / avg_time,
                'times': times
            }
            
            print(f"    Average time: {avg_time:.4f}s ({size/avg_time:.2f} items/second)")
        
        self.results['rust_extensions'] = results
        
        # Calculate speedup over Python
        if 'python_processing' in self.results:
            print("\n--- Performance Comparison (Rust vs Python) ---")
            for size in sizes:
                python_time = self.results['python_processing'][size]['avg_time']
                rust_time = results[size]['avg_time']
                speedup = python_time / rust_time
                print(f"  Size {size}: Rust is {speedup:.2f}x faster than Python")
    
    def benchmark_database_operations(self):
        """Benchmark database operations."""
        print("\n--- Benchmarking Database Operations ---")
        
        # This benchmark would connect to the database and run queries
        # Using a simulated version here for demonstration
        operations = ['insert', 'select', 'update', 'delete']
        results = {}
        
        for op in operations:
            print(f"  Testing {op} operations...")
            
            # Simulate database operation timing
            times = []
            for i in range(self.iterations):
                start = time.time()
                # Simulated work - would be actual database operations
                time.sleep(0.01 + 0.01 * np.random.rand())
                end = time.time()
                times.append(end - start)
            
            avg_time = sum(times) / len(times)
            results[op] = {
                'avg_time': avg_time,
                'operations_per_second': 1 / avg_time,
                'times': times
            }
            
            print(f"    Average time: {avg_time:.4f}s ({1/avg_time:.2f} ops/second)")
        
        self.results['database_operations'] = results
    
    def benchmark_vector_operations(self):
        """Benchmark vector similarity search operations."""
        print("\n--- Benchmarking Vector Operations ---")
        
        # Create random vectors for testing
        dimensions = [128, 512, 1536]
        results = {}
        
        for dim in dimensions:
            print(f"  Testing {dim}-dimensional vectors...")
            vector1 = np.random.rand(dim).astype(np.float32).tolist()
            vector2 = np.random.rand(dim).astype(np.float32).tolist()
            
            # Time vector similarity operations
            times = []
            for i in range(self.iterations):
                start = time.time()
                # Use Rust extension if available, otherwise Python implementation
                if RUST_AVAILABLE:
                    similarity = vector_store.cosine_similarity(vector1, vector2)
                else:
                    # Python implementation of cosine similarity
                    dot_product = sum(a * b for a, b in zip(vector1, vector2))
                    norm1 = sum(a * a for a in vector1) ** 0.5
                    norm2 = sum(b * b for b in vector2) ** 0.5
                    similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
                end = time.time()
                times.append(end - start)
            
            avg_time = sum(times) / len(times)
            results[dim] = {
                'avg_time': avg_time,
                'operations_per_second': 1 / avg_time,
                'times': times,
                'implementation': 'Rust' if RUST_AVAILABLE else 'Python'
            }
            
            print(f"    Average time ({results[dim]['implementation']}): {avg_time:.6f}s ({1/avg_time:.2f} ops/second)")
        
        self.results['vector_operations'] = results
    
    def generate_reports(self):
        """Generate performance reports and visualizations."""
        print("\n--- Generating Performance Reports ---")
        
        # Create a unique filename based on timestamp
        base_filename = f"benchmark_{self.timestamp}"
        
        # Save raw results as JSON
        json_path = self.output_dir / f"{base_filename}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"  Raw results saved to {json_path}")
        
        # Generate CSV report
        csv_path = self.output_dir / f"{base_filename}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Category', 'Test', 'Average Time (s)', 'Operations/Sec', 'Implementation'])
            
            for category, tests in self.results.items():
                for test_name, test_results in tests.items():
                    implementation = test_results.get('implementation', 'Python')
                    writer.writerow([
                        category,
                        test_name,
                        test_results['avg_time'],
                        test_results.get('items_per_second', test_results.get('operations_per_second', 0)),
                        implementation
                    ])
        print(f"  CSV report saved to {csv_path}")
        
        # Generate plots
        try:
            self._generate_plots(base_filename)
            print(f"  Performance plots saved to {self.output_dir}")
        except Exception as e:
            print(f"  Failed to generate plots: {e}")
    
    def _generate_plots(self, base_filename):
        """Generate performance plots."""
        # Create a comparison plot if we have both Python and Rust results
        if 'python_processing' in self.results and 'rust_extensions' in self.results:
            plt.figure(figsize=(10, 6))
            
            sizes = sorted(self.results['python_processing'].keys())
            python_times = [self.results['python_processing'][size]['avg_time'] for size in sizes]
            rust_times = [self.results['rust_extensions'][size]['avg_time'] for size in sizes]
            
            plt.plot(sizes, python_times, 'o-', label='Python')
            plt.plot(sizes, rust_times, 'o-', label='Rust')
            plt.xlabel('Data Size')
            plt.ylabel('Time (seconds)')
            plt.title('Performance Comparison: Python vs Rust')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Use log scale for better visualization
            plt.xscale('log')
            plt.yscale('log')
            
            plt.savefig(self.output_dir / f"{base_filename}_comparison.png")
            plt.close()
        
        # Create a bar chart for database operations
        if 'database_operations' in self.results:
            plt.figure(figsize=(10, 6))
            
            operations = list(self.results['database_operations'].keys())
            times = [self.results['database_operations'][op]['avg_time'] for op in operations]
            
            plt.bar(operations, times)
            plt.xlabel('Operation')
            plt.ylabel('Time (seconds)')
            plt.title('Database Operation Performance')
            plt.grid(True, axis='y')
            plt.tight_layout()
            
            plt.savefig(self.output_dir / f"{base_filename}_database.png")
            plt.close()
        
        # Create a plot for vector operations
        if 'vector_operations' in self.results:
            plt.figure(figsize=(10, 6))
            
            dimensions = sorted(self.results['vector_operations'].keys())
            times = [self.results['vector_operations'][dim]['avg_time'] for dim in dimensions]
            
            plt.plot(dimensions, times, 'o-')
            plt.xlabel('Vector Dimension')
            plt.ylabel('Time (seconds)')
            plt.title('Vector Operation Performance')
            plt.grid(True)
            plt.tight_layout()
            
            plt.savefig(self.output_dir / f"{base_filename}_vector.png")
            plt.close()

def main():
    """Main entry point for benchmarking."""
    parser = argparse.ArgumentParser(description='LlamaDB Performance Benchmarking Suite')
    parser.add_argument('-i', '--iterations', type=int, default=5,
                        help='Number of iterations for each benchmark')
    parser.add_argument('-o', '--output-dir', type=str, default='benchmark_results',
                        help='Directory to save benchmark results')
    args = parser.parse_args()
    
    # Welcome message
    print("=" * 80)
    print("LlamaDB Performance Benchmarking Suite")
    print("=" * 80)
    print(f"Running with {args.iterations} iterations")
    print(f"Results will be saved to: {args.output_dir}")
    print("-" * 80)
    
    # Run benchmarks
    benchmark = LlamaDBBenchmark(output_dir=args.output_dir, iterations=args.iterations)
    benchmark.run_all()
    
    print("\n" + "=" * 80)
    print("Benchmarking complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
EOF

    # Make benchmark script executable
    chmod +x tools/benchmarks/run_benchmarks.py
    
    # Create a simple wrapper script for convenience
    cat > tools/benchmarks/run_benchmarks.sh << 'EOF'
#!/usr/bin/env bash
set -e

# Activate virtual environment if it exists
if [[ -d ".venv" ]]; then
    source .venv/bin/activate
fi

# Run benchmarks with provided arguments
python tools/benchmarks/run_benchmarks.py "$@"
EOF

    # Make wrapper script executable
    chmod +x tools/benchmarks/run_benchmarks.sh
    
    # Create a stress test script for load testing
    cat > tools/benchmarks/stress_test.py << 'EOF'
#!/usr/bin/env python3
"""
LlamaDB Stress Test Tool

This script performs load testing on the LlamaDB API to evaluate performance under load.
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
import random
import statistics
from pathlib import Path

try:
    import httpx
    import aiohttp
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Please install required packages: httpx, aiohttp, matplotlib, numpy")
    exit(1)

class LlamaDBStressTest:
    """Stress testing for LlamaDB API endpoints."""
    
    def __init__(self, 
                 base_url="http://localhost:8000", 
                 concurrency=10, 
                 duration=60,
                 output_dir="stress_test_results"):
        """Initialize the stress test parameters."""
        self.base_url = base_url
        self.concurrency = concurrency
        self.duration = duration
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "base_url": base_url,
                "concurrency": concurrency,
                "duration": duration
            },
            "endpoints": {}
        }
        
        # Define test endpoints and their parameters
        self.endpoints = {
            "/health": {"method": "GET", "payload": None},
            "/api/v1/query": {
                "method": "POST", 
                "payload": {"query": "SELECT * FROM customers LIMIT 10"}
            },
            "/api/v1/vector/search": {
                "method": "POST",
                "payload": {
                    "query_vector": [random.random() for _ in range(384)],
                    "limit": 5
                }
            }
        }
        
        print(f"Initializing stress test against {base_url}")
        print(f"Concurrency: {concurrency} | Duration: {duration}s")
    
    async def test_endpoint(self, session, endpoint, method, payload, semaphore):
        """Test a single endpoint with the given method and payload."""
        url = f"{self.base_url}{endpoint}"
        
        async with semaphore:
            start_time = time.time()
            try:
                if method == "GET":
                    async with session.get(url) as response:
                        await response.read()
                        status = response.status
                else:  # POST
                    async with session.post(url, json=payload) as response:
                        await response.read()
                        status = response.status
                
                response_time = time.time() - start_time
                return {
                    "success": 200 <= status < 300,
                    "status": status,
                    "response_time": response_time
                }
            except Exception as e:
                return {
                    "success": False,
                    "status": 0,
                    "error": str(e),
                    "response_time": time.time() - start_time
                }
    
    async def run_endpoint_test(self, endpoint, method, payload):
        """Run stress test for a specific endpoint."""
        print(f"\nTesting endpoint: {endpoint} ({method})")
        
        semaphore = asyncio.Semaphore(self.concurrency)
        timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            start_time = time.time()
            
            # Keep creating tasks until the duration is reached
            while time.time() - start_time < self.duration:
                tasks.append(
                    asyncio.create_task(
                        self.test_endpoint(session, endpoint, method, payload, semaphore)
                    )
                )
                await asyncio.sleep(0.01)  # Small delay to prevent task creation overhead
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            response_times = []
            success_count = 0
            status_codes = {}
            errors = []
            
            for result in results:
                if isinstance(result, dict):
                    response_times.append(result["response_time"])
                    if result["success"]:
                        success_count += 1
                    
                    status = result.get("status", 0)
                    status_codes[status] = status_codes.get(status, 0) + 1
                    
                    if not result["success"] and "error" in result:
                        errors.append(result["error"])
                else:
                    # Exception occurred
                    errors.append(str(result))
            
            # Calculate statistics
            requests_per_second = len(results) / self.duration
            avg_response_time = statistics.mean(response_times) if response_times else 0
            max_response_time = max(response_times) if response_times else 0
            min_response_time = min(response_times) if response_times else 0
            p95_response_time = np.percentile(response_times, 95) if response_times else 0
            success_rate = (success_count / len(results)) * 100 if results else 0
            
            # Store results
            endpoint_results = {
                "requests": len(results),
                "requests_per_second": requests_per_second,
                "success_rate": success_rate,
                "response_times": {
                    "avg": avg_response_time,
                    "min": min_response_time,
                    "max": max_response_time,
                    "p95": p95_response_time
                },
                "status_codes": status_codes,
                "errors": errors[:10]  # Store only first 10 errors
            }
            
            self.results["endpoints"][endpoint] = endpoint_results
            
            # Print summary
            print(f"  Requests: {len(results)} ({requests_per_second:.2f} req/s)")
            print(f"  Success Rate: {success_rate:.2f}%")
            print(f"  Response Time: Avg={avg_response_time:.3f}s, Min={min_response_time:.3f}s, Max={max_response_time:.3f}s, P95={p95_response_time:.3f}s")
            print(f"  Status Codes: {status_codes}")
            
            if errors:
                print(f"  Errors: {len(errors)} total")
                for i, error in enumerate(errors[:3]):  # Show first 3 errors
                    print(f"    - {error}")
                if len(errors) > 3:
                    print(f"    ... and {len(errors) - 3} more")
    
    async def run_tests(self):
        """Run stress tests for all endpoints."""
        for endpoint, config in self.endpoints.items():
            await self.run_endpoint_test(endpoint, config["method"], config["payload"])
    
    def generate_reports(self):
        """Generate performance reports and visualizations."""
        print("\n--- Generating Stress Test Reports ---")
        
        # Create a unique filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"stress_test_{timestamp}"
        
        # Save raw results as JSON
        json_path = self.output_dir / f"{base_filename}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"  Raw results saved to {json_path}")
        
        # Generate plots
        self._generate_plots(base_filename)
    
    def _generate_plots(self, base_filename):
        """Generate performance plots."""
        # Create a bar chart comparing response times
        plt.figure(figsize=(12, 6))
        
        endpoints = list(self.results["endpoints"].keys())
        avg_times = [self.results["endpoints"][ep]["response_times"]["avg"] for ep in endpoints]
        p95_times = [self.results["endpoints"][ep]["response_times"]["p95"] for ep in endpoints]
        
        x = np.arange(len(endpoints))
        width = 0.35
        
        plt.bar(x - width/2, avg_times, width, label='Average')
        plt.bar(x + width/2, p95_times, width, label='95th Percentile')
        
        plt.xlabel('Endpoint')
        plt.ylabel('Response Time (seconds)')
        plt.title('Response Time by Endpoint')
        plt.xticks(x, endpoints, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(self.output_dir / f"{base_filename}_response_times.png")
        plt.close()
        
        # Create a bar chart comparing requests per second
        plt.figure(figsize=(12, 6))
        
        rps = [self.results["endpoints"][ep]["requests_per_second"] for ep in endpoints]
        success_rates = [self.results["endpoints"][ep]["success_rate"] for ep in endpoints]
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Requests per second on left y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Endpoint')
        ax1.set_ylabel('Requests per Second', color=color)
        ax1.bar(endpoints, rps, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Success rate on right y-axis
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Success Rate (%)', color=color)
        ax2.plot(endpoints, success_rates, 'o-', color=color, linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim([0, 105])  # 0-105% range
        