#!/bin/bash
# Script to run the RAG web app

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but could not be found. Please install Python 3."
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r update_requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "⚠️ Please edit .env file and add your OpenAI API key before continuing."
    echo "Press Enter to continue once you've updated the .env file..."
    read
fi

# Run the web app
echo "Starting RAG web app..."
export FLASK_APP=rag_web_app.py
python rag_web_app.py

# End
echo "Web app stopped." 