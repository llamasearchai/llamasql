#!/bin/bash
# Script to start the RAG web app using Docker

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is required but could not be found. Please install Docker."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is required but could not be found. Please install Docker Compose."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "⚠️ Please edit .env file and add your OpenAI API key before continuing."
    echo "Press Enter to continue once you've updated the .env file..."
    read
fi

# Create data directory if it doesn't exist
mkdir -p data

# Start the Docker containers
echo "Starting RAG web app using Docker..."
docker-compose -f docker-compose.web.yml up --build

# End
echo "Docker containers stopped." 