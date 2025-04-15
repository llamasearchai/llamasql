# Retrieval-Augmented Generation (RAG) with LlamaDB

This example demonstrates how to build a simple RAG system using LlamaDB for vector storage and retrieval, and OpenAI for embedding generation and text completion.

## Features

- Document loading and preprocessing
- Document chunking with configurable overlap
- Embeddings generation with OpenAI API
- Vector storage and retrieval with LlamaDB
- Context-aware text generation with OpenAI
- MLX acceleration on Apple Silicon
- Filterable vector search
- Persistent storage for indexes

## Requirements

- Python 3.8+
- LlamaDB
- OpenAI API key
- Docker (optional, for containerized deployment)

## Installation

### Option 1: Direct Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your OpenAI API key:

```bash
cp .env.example .env
# Edit .env to add your actual API key
```

### Option 2: Docker

1. Create a `.env` file with your OpenAI API key:

```bash
cp .env.example .env
# Edit .env to add your actual API key
```

2. Build and run with Docker Compose:

```bash
docker-compose up --build
```

## Usage

### Running the Example

```bash
python rag_example.py
```

The example will:

1. Initialize a RAG system using LlamaDB and OpenAI
2. Load sample documents about AI and machine learning
3. Process and index the documents
4. Execute sample queries to demonstrate RAG capabilities
5. Save the knowledge base for later use

### Sample Queries

The example includes the following demonstration queries:

1. "Explain how neural networks work and their components."
2. "What are the main differences between deep learning and traditional machine learning?"
3. "How does the Transformer architecture work in NLP?"
4. "What are the ethical concerns related to AI development?"
5. "How do reinforcement learning agents balance exploration and exploitation?"

It also demonstrates category-filtered search:

- "Explain the concept of attention in natural language processing." (filtered to NLP category)

### Using in Your Own Projects

To use this RAG system in your own project:

1. Initialize the RAG system:

```python
from rag_example import RAGSystem, Document

rag = RAGSystem(
    embedding_model="text-embedding-3-small",
    completion_model="gpt-3.5-turbo",
    chunk_size=600,
    chunk_overlap=100
)
```

2. Add your own documents:

```python
rag.add_document(Document(
    id="unique_id",
    title="Document Title",
    content="Document content...",
    source="Source of the document",
    metadata={"category": "Topic", "author": "Author Name"}
))
```

3. Generate responses to queries:

```python
response, contexts = rag.generate_response(
    query="Your question here?",
    k=3,  # Number of context chunks to retrieve
    filter={"category": "FilterCategory"}  # Optional filtering
)

print(response)
```

4. Save and load the knowledge base:

```python
# Save
rag.save("my_knowledge_base.llamadb")

# Load in a new session
loaded_rag = RAGSystem()
loaded_rag.index = VectorIndex.load("my_knowledge_base.llamadb")
```

## Customization

You can customize the RAG system by adjusting these parameters:

- **embedding_model**: OpenAI embedding model to use
- **completion_model**: OpenAI completion model to use
- **chunk_size**: Size of document chunks in characters
- **chunk_overlap**: Overlap between consecutive chunks in characters
- **max_tokens**: Maximum number of tokens for completions
- **use_mlx**: Whether to use MLX acceleration (Apple Silicon only)

## Advanced Features

### Custom System Prompts

You can provide custom system prompts to control the style and behavior of the generated responses:

```python
custom_prompt = """
You are a helpful assistant that answers questions in a very concise manner.
Always provide factual information based on the context.
If you don't know the answer, say so clearly.
"""

response, _ = rag.generate_response(
    query="Your question?",
    system_prompt=custom_prompt
)
```

### Metadata Filtering

Filter documents by metadata during retrieval:

```python
# Search only within technical documents
response, _ = rag.generate_response(
    query="Your technical question?",
    filter={"category": "Technical"}
)

# Complex filtering
response, _ = rag.generate_response(
    query="Your question?",
    filter={
        "$and": [
            {"category": "Science"},
            {"date": {"$gt": "2022-01-01"}}
        ]
    }
)
```

## Performance Tips

1. **MLX Acceleration**: Enable MLX on Apple Silicon for up to 10x faster vector operations
2. **Batch Processing**: Process documents in batches for faster indexing
3. **HNSW Index**: Use the HNSW index type for large document collections
4. **Chunking Strategy**: Adjust chunk size based on your specific use case
5. **Selective Embeddings**: Only embed relevant parts of documents to reduce token usage

## License

This example is part of LlamaDB and is released under the MIT License. 