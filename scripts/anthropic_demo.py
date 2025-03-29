#!/usr/bin/env python
"""
LlamaDB Anthropic Integration Demo

This script demonstrates the integration between LlamaDB and Anthropic's Claude models,
showcasing RAG capabilities and embeddings.
"""

import os
import time
import argparse
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

# Import LlamaDB components
from llamadb.core.vector_index import VectorIndex
from llamadb.integrations.anthropic import AnthropicClient, ClaudeRAGPipeline, ClaudeEmbeddings

console = Console()

def print_header():
    """Print a stylized header."""
    console.print("\n[bold blue]LlamaDB + Claude Integration Demo[/bold blue]")
    console.print("Showcasing Retrieval-Augmented Generation with Anthropic's Claude\n")

def check_api_key():
    """Check if Anthropic API key is available."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[bold red]Error:[/bold red] ANTHROPIC_API_KEY environment variable not found.")
        console.print("Please set your Anthropic API key with:")
        console.print("  export ANTHROPIC_API_KEY=your_api_key_here")
        return False
    return True

def create_sample_documents():
    """Create sample documents for the demo."""
    console.print("[cyan]Creating sample documents...[/cyan]")
    
    documents = [
        {
            "id": "doc1",
            "text": """
            # LlamaDB: High-Performance Vector Database
            
            LlamaDB is an enterprise-grade vector database built for AI workloads, featuring a hybrid 
            architecture that combines Python, Rust, and MLX acceleration. It offers unparalleled 
            performance for retrieval augmented generation (RAG), semantic search, and real-time embeddings.
            
            ## Key Features
            
            - Hybrid Architecture: Core Python interfaces with Rust extensions
            - MLX Acceleration: Uses Apple's MLX framework on Apple Silicon
            - Claude AI Integration: First-class integration with Anthropic's Claude models
            """,
            "metadata": {
                "title": "LlamaDB Overview",
                "source": "Documentation",
                "category": "product"
            }
        },
        {
            "id": "doc2",
            "text": """
            # Performance Benchmarks
            
            LlamaDB delivers exceptional performance across various operations:
            
            - Vector Search (1M vectors): 22ms with MLX acceleration (40.5x faster)
            - Text Tokenization (10MB): 56ms with Rust (37.9x faster)
            - Large Matrix Multiply (10Kx10K): 74ms with MLX (33.5x faster)
            - Claude Context Processing: 32ms with MLX (21.3x faster)
            - RAG Pipeline End-to-End: 165ms with MLX (26.1x faster)
            """,
            "metadata": {
                "title": "LlamaDB Performance",
                "source": "Benchmarks",
                "category": "performance"
            }
        },
        {
            "id": "doc3",
            "text": """
            # Claude AI Integration
            
            LlamaDB provides first-class integration with Anthropic's Claude models:
            
            ```python
            from llamadb import VectorIndex
            from llamadb.integrations.anthropic import ClaudeRAGPipeline
            
            # Load your data
            index = VectorIndex.load("my_knowledge_base")
            
            # Create a Claude-powered RAG pipeline
            rag = ClaudeRAGPipeline(
                vector_index=index,
                model="claude-3-opus-20240229"
            )
            
            # Query with automatic retrieval and context injection
            response = rag.generate(
                query="What are the key features of LlamaDB?",
                num_documents=5
            )
            ```
            """,
            "metadata": {
                "title": "Claude Integration Guide",
                "source": "Documentation",
                "category": "integration"
            }
        },
        {
            "id": "doc4",
            "text": """
            # Retrieval Augmented Generation (RAG)
            
            LlamaDB excels in RAG applications with these features:
            
            - Context-aware document retrieval with Claude and other LLMs
            - Hybrid vector + keyword search for precise information access
            - Retrieval evaluation frameworks to measure accuracy
            - Automatic context formatting for optimal LLM performance
            - Support for multiple embedding models including Claude, OpenAI, and HuggingFace
            """,
            "metadata": {
                "title": "RAG Capabilities",
                "source": "Documentation",
                "category": "features"
            }
        },
        {
            "id": "doc5",
            "text": """
            # MLX Acceleration
            
            LlamaDB leverages Apple's MLX framework to provide exceptional performance on Apple Silicon:
            
            - Matrix operations up to 40x faster than NumPy
            - Optimized for M1/M2/M3 chips
            - Automatic fallback to CPU for non-Apple Silicon devices
            - Seamless integration with Python and Rust components
            - Specialized kernels for vector similarity calculations
            """,
            "metadata": {
                "title": "MLX Acceleration",
                "source": "Documentation",
                "category": "performance"
            }
        }
    ]
    
    console.print(f"Created {len(documents)} sample documents")
    return documents

def demo_embeddings(api_key):
    """Demonstrate Claude embeddings."""
    console.print("\n[bold cyan]Claude Embeddings Demo[/bold cyan]")
    
    # Create embeddings client
    embeddings = ClaudeEmbeddings(api_key=api_key)
    
    # Create sample texts
    texts = [
        "LlamaDB is a high-performance vector database",
        "Claude is an AI assistant created by Anthropic",
        "Vector databases are essential for modern AI applications",
        "MLX provides acceleration on Apple Silicon devices"
    ]
    
    # Generate embeddings
    console.print("Generating embeddings for sample texts...")
    start_time = time.time()
    embeddings_array = embeddings.embed(texts)
    end_time = time.time()
    
    # Display results
    console.print(f"Generated {len(texts)} embeddings in {(end_time - start_time)*1000:.2f}ms")
    console.print(f"Embedding dimension: {embeddings_array.shape[1]}")
    
    # Calculate similarities
    console.print("\nCalculating similarities between texts:")
    table = Table(show_header=True)
    table.add_column("Text 1")
    table.add_column("Text 2")
    table.add_column("Similarity")
    
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            similarity = embeddings.similarity(embeddings_array[i], embeddings_array[j])
            table.add_row(
                texts[i][:30] + "...",
                texts[j][:30] + "...",
                f"{similarity:.4f}"
            )
    
    console.print(table)
    return embeddings

def demo_rag(api_key, documents):
    """Demonstrate RAG with Claude."""
    console.print("\n[bold cyan]Claude RAG Pipeline Demo[/bold cyan]")
    
    # Create embeddings client
    embeddings = ClaudeEmbeddings(api_key=api_key)
    
    # Create vector index
    console.print("Creating vector index...")
    index = VectorIndex(dimension=1536)  # Claude embeddings are 1536-dimensional
    
    # Extract texts and metadata
    texts = [doc["text"] for doc in documents]
    metadata = [doc.get("metadata", {}) for doc in documents]
    
    # Generate embeddings
    console.print("Generating embeddings for documents...")
    embeddings_array = embeddings.embed(texts)
    
    # Add to index
    console.print("Adding documents to index...")
    for i, (embedding, meta) in enumerate(zip(embeddings_array, metadata)):
        meta["text"] = texts[i]
        index.add_item(embedding, meta)
    
    # Create RAG pipeline
    console.print("Creating Claude RAG pipeline...")
    pipeline = ClaudeRAGPipeline(
        vector_index=index,
        api_key=api_key,
        model="claude-3-sonnet-20240229"  # Using Sonnet for demo to reduce costs
    )
    
    # Run sample queries
    sample_queries = [
        "What are the key features of LlamaDB?",
        "How does LlamaDB integrate with Claude?",
        "What performance improvements does MLX provide?"
    ]
    
    for query in sample_queries:
        console.print(f"\n[bold]Query:[/bold] {query}")
        
        # Generate response
        console.print("Generating response...")
        start_time = time.time()
        response = pipeline.generate(query, num_documents=3, temperature=0.7)
        end_time = time.time()
        
        # Display results
        console.print(f"Response generated in {(end_time - start_time)*1000:.2f}ms")
        console.print(f"Retrieved {len(response.retrieved_documents)} documents")
        
        # Display retrieved documents
        console.print("\n[bold]Retrieved Documents:[/bold]")
        for i, doc in enumerate(response.retrieved_documents, 1):
            title = doc["metadata"].get("title", f"Document {i}")
            source = doc["metadata"].get("source", "Unknown")
            console.print(f"{i}. {title} ({source})")
        
        # Display response
        console.print("\n[bold]Claude Response:[/bold]")
        console.print(Panel(Markdown(response.text), border_style="green"))

def main():
    """Run the demo."""
    parser = argparse.ArgumentParser(description="LlamaDB Anthropic Integration Demo")
    parser.add_argument("--api-key", help="Anthropic API key (overrides environment variable)")
    args = parser.parse_args()
    
    print_header()
    
    # Check for API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[bold red]Error:[/bold red] No Anthropic API key provided.")
        console.print("Please provide an API key with --api-key or set the ANTHROPIC_API_KEY environment variable.")
        return 1
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Run embeddings demo
    embeddings = demo_embeddings(api_key)
    
    # Run RAG demo
    demo_rag(api_key, documents)
    
    console.print("\n[bold green]Demo completed successfully![/bold green]")
    return 0

if __name__ == "__main__":
    exit(main()) 