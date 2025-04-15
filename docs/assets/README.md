# LlamaDB Architecture Diagrams

This directory contains architecture diagrams for the LlamaDB system.

## Overview

The diagrams in this directory explain the architecture and components of LlamaDB:

- `llamadb-architecture.png` - High-level overview of the LlamaDB architecture
- `llamadb-detailed-arch.png` - Detailed component-level architecture diagram

## ASCII Diagrams

We include ASCII architecture diagrams in `architecture.txt` as a simple reference.

## Generating Diagrams

These diagrams can be regenerated using the PlantUML source files in this directory:

```bash
# Install required tools
pip install plantuml

# Generate from PlantUML files
plantuml -tpng architecture.puml
plantuml -tpng detailed-architecture.puml
```

## Updating Diagrams

When making significant architectural changes, please update both the PNG files and the PlantUML source files to keep them in sync.

## PlantUML Source

The PlantUML source files are:

- `architecture.puml` - Source for high-level architecture
- `detailed-architecture.puml` - Source for detailed architecture 