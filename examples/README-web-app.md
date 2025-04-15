# LlamaDB RAG Web Application

A web interface for interacting with the LlamaDB RAG (Retrieval-Augmented Generation) system.

## Features

- Interactive chat interface for querying your knowledge base
- Real-time responses generated with context from documents
- Source reference for each response with context highlighting
- Support for custom documents via multiple loader types
- Dockerized deployment for easy installation
- Lightweight and customizable

## Requirements

- Python 3.8 or higher
- LlamaDB 0.3.0 or higher
- OpenAI API key
- Additional dependencies (see requirements.txt)

## Quick Start

### Option 1: Local Installation

#### macOS / Linux

1. Clone the repository:
   ```bash
   git clone https://github.com/llamasearch/llamadb.git
   cd llamadb/examples
   ```

2. Run the setup script:
   ```bash
   chmod +x run_web_app.sh
   ./run_web_app.sh
   ```

3. Open your browser and navigate to http://localhost:8000

#### Windows

1. Clone the repository:
   ```powershell
   git clone https://github.com/llamasearch/llamadb.git
   cd llamadb\examples
   ```

2. Run the setup script:
   ```powershell
   .\run_web_app.ps1
   ```

3. Open your browser and navigate to http://localhost:8000

### Option 2: Docker Deployment

#### macOS / Linux

1. Clone the repository:
   ```bash
   git clone https://github.com/llamasearch/llamadb.git
   cd llamadb/examples
   ```

2. Run the Docker setup script:
   ```bash
   chmod +x start_web_docker.sh
   ./start_web_docker.sh
   ```

3. Open your browser and navigate to http://localhost:8000

#### Windows

1. Clone the repository:
   ```powershell
   git clone https://github.com/llamasearch/llamadb.git
   cd llamadb\examples
   ```

2. Run the Docker setup script:
   ```powershell
   .\start_web_docker.ps1
   ```

3. Open your browser and navigate to http://localhost:8000

## Setup Configuration

1. The first time you run the application, it will create a `.env` file based on `.env.example`
2. Edit the `.env` file to add your OpenAI API key and adjust other settings:
   ```
   OPENAI_API_KEY=your_api_key_here
   EMBEDDING_MODEL=text-embedding-3-small
   COMPLETION_MODEL=gpt-3.5-turbo
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   MAX_TOKENS=500
   ```

## Loading Documents Into Your Knowledge Base

### Using the Web Interface

1. Start the web application
2. Navigate to the "Upload" page by clicking the link in the navigation menu
3. Choose the document type from the dropdown (PDF, Web, CSV, Markdown, JSON)
4. Upload your file or enter a URL (for web documents)
5. Configure additional parameters if needed
6. Click "Upload" to process and add the document to your knowledge base

### Using Command Line Loaders

For bulk loading or automated processes, use the document loaders directly:

```bash
# Load a PDF document
python custom_document_loader.py --source documents/sample.pdf --type pdf --save-path data/knowledge_base.llamadb

# Load multiple documents from a directory
python custom_document_loader.py --source documents/ --type pdf --recursive --save-path data/knowledge_base.llamadb

# Load a website
python custom_document_loader.py --source https://example.com --type web --save-path data/knowledge_base.llamadb
```

See [README-loaders.md](README-loaders.md) for detailed instructions on using document loaders.

## API Endpoints

The web application exposes the following endpoints:

- `GET /`: Main chat interface
- `GET /upload`: Document upload interface
- `POST /query`: API endpoint for submitting queries
  ```json
  {
    "query": "What is LlamaDB?",
    "max_tokens": 500
  }
  ```
- `POST /upload`: API endpoint for uploading documents
  ```json
  {
    "source": "https://example.com",
    "type": "web",
    "metadata": {
      "source_type": "website",
      "category": "documentation"
    }
  }
  ```

## Customization Options

### Styling

The web interface uses simple CSS that can be customized in the `templates/static/style.css` file.

### Adding New Loaders

To add support for additional document types:

1. Extend the `DocumentLoader` class in `custom_document_loader.py`
2. Update the web interface in `rag_web_app.py` to handle the new document type
3. Add appropriate form fields in `templates/upload.html`

Example of adding a new loader to the web app:

```python
@app.route('/upload', methods=['POST'])
def upload_document():
    # ... existing code ...
    
    # Add handling for your new document type
    if doc_type == "my_custom_type":
        from custom_document_loader import MyCustomLoader
        loader = MyCustomLoader(metadata=metadata)
        # Process uploaded file or URL
    
    # ... rest of the function ...
```

## Troubleshooting

### Common Issues

- **"ModuleNotFoundError"**: Make sure all dependencies are installed with `pip install -r update_requirements.txt`
- **"API key not found"**: Check that your `.env` file contains a valid OpenAI API key
- **"Index not found"**: If your knowledge base doesn't exist yet, make sure to add documents first

### Debugging

Set the `FLASK_ENV=development` environment variable for more detailed error messages:

```bash
export FLASK_ENV=development
python rag_web_app.py
```

For Docker deployments, you can check logs with:

```bash
docker-compose -f docker-compose.web.yml logs -f
```

## License

This application is part of LlamaDB and is released under the MIT License. 