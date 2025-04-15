#!/bin/bash
# Generate diagrams from PlantUML files

set -e

# Check if PlantUML is installed
if ! command -v plantuml &> /dev/null; then
    echo "PlantUML is not installed. Please install it first."
    echo "You can download it from https://plantuml.com/download"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p docs/assets

# Generate architecture diagram
echo "Generating architecture diagram..."
plantuml -tpng docs/assets/architecture.puml -o .
echo "Architecture diagram generated at docs/assets/architecture.png"

# Convert SVG to PNG for hero image
if [ -f "docs/assets/llamadb-hero.svg" ]; then
    echo "Converting hero image from SVG to PNG..."
    if command -v convert &> /dev/null; then
        convert -density 300 docs/assets/llamadb-hero.svg docs/assets/llamadb-hero.png
        echo "Hero image converted to PNG at docs/assets/llamadb-hero.png"
    elif command -v rsvg-convert &> /dev/null; then
        rsvg-convert -w 1200 -h 600 docs/assets/llamadb-hero.svg > docs/assets/llamadb-hero.png
        echo "Hero image converted to PNG at docs/assets/llamadb-hero.png"
    else
        echo "Warning: Neither ImageMagick nor rsvg-convert is installed."
        echo "Could not convert SVG to PNG. Please install one of these tools."
    fi
fi

echo "All diagrams generated successfully!" 