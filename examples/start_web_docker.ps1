# PowerShell script to start the RAG web app using Docker on Windows

# Check if Docker is installed
$dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
if (-not $dockerCmd) {
    Write-Host "Docker is required but could not be found. Please install Docker Desktop for Windows."
    exit 1
}

# Check if Docker Compose is available
$composeCmd = docker compose version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker Compose is required but could not be found. Please install Docker Desktop with Docker Compose."
    exit 1
}

# Check if .env file exists
if (-not (Test-Path -Path ".env")) {
    Write-Host "Creating .env file from .env.example..."
    Copy-Item -Path ".env.example" -Destination ".env"
    Write-Host "⚠️ Please edit .env file and add your OpenAI API key before continuing."
    Write-Host "Press Enter to continue once you've updated the .env file..."
    Read-Host
}

# Create data directory if it doesn't exist
if (-not (Test-Path -Path "data")) {
    New-Item -Path "data" -ItemType Directory
}

# Start the Docker containers
Write-Host "Starting RAG web app using Docker..."
docker-compose -f docker-compose.web.yml up --build

# End
Write-Host "Docker containers stopped." 