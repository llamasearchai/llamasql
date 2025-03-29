# PowerShell script to run the RAG web app on Windows

# Check if Python is installed
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "Python is required but could not be found. Please install Python."
    exit 1
}

# Check if virtual environment exists, create if not
if (-not (Test-Path -Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
& "./venv/Scripts/Activate.ps1"

# Install dependencies
Write-Host "Installing dependencies..."
pip install -r update_requirements.txt

# Check if .env file exists
if (-not (Test-Path -Path ".env")) {
    Write-Host "Creating .env file from .env.example..."
    Copy-Item -Path ".env.example" -Destination ".env"
    Write-Host "⚠️ Please edit .env file and add your OpenAI API key before continuing."
    Write-Host "Press Enter to continue once you've updated the .env file..."
    Read-Host
}

# Run the web app
Write-Host "Starting RAG web app..."
$env:FLASK_APP = "rag_web_app.py"
python rag_web_app.py

# End
Write-Host "Web app stopped." 