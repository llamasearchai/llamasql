#!/usr/bin/env python
"""
Custom Document Loader for LlamaDB RAG

This script demonstrates how to load documents from various sources:
- PDFs
- CSV/Excel files
- Websites
- Markdown files
- JSON data

It processes the content, creates Document objects, and adds them to the
LlamaDB RAG system for vector search and retrieval.

Requirements:
- LlamaDB
- All requirements from rag_example.py
- Additional requirements for each loader type (see imports)
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Import RAG components from our example
from rag_example import Chunk, Document, RAGSystem

# Load environment variables
load_dotenv()

# Try to import various document processing libraries
# Each will be optional, with fallbacks for missing dependencies
try:
    import PyPDF2

    HAS_PDF_SUPPORT = True
except ImportError:
    HAS_PDF_SUPPORT = False
    print("PDF support not available. Install PyPDF2 with: pip install PyPDF2")

try:
    import requests
    from bs4 import BeautifulSoup

    HAS_WEB_SUPPORT = True
except ImportError:
    HAS_WEB_SUPPORT = False
    print(
        "Web scraping support not available. Install required packages with: pip install requests beautifulsoup4"
    )

try:
    import markdown

    HAS_MARKDOWN_SUPPORT = True
except ImportError:
    HAS_MARKDOWN_SUPPORT = False
    print("Markdown support not available. Install markdown with: pip install markdown")


class DocumentLoader:
    """Base class for document loaders."""

    def load(self, source: str) -> List[Document]:
        """
        Load documents from the source.

        Args:
            source: Source identifier (file path, URL, etc.)

        Returns:
            List of Document objects
        """
        raise NotImplementedError("Subclasses must implement this method")


class PDFLoader(DocumentLoader):
    """Loader for PDF documents."""

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the PDF loader.

        Args:
            metadata: Additional metadata to add to all documents
        """
        if not HAS_PDF_SUPPORT:
            raise ImportError("PDF support requires PyPDF2 library")
        self.metadata = metadata or {}

    def load(self, source: str) -> List[Document]:
        """Load documents from a PDF file."""
        if not os.path.exists(source):
            raise FileNotFoundError(f"PDF file not found: {source}")

        documents = []

        # Basic metadata from the file
        file_name = os.path.basename(source)
        file_id = re.sub(r"[^\w\-_]", "_", file_name)

        try:
            with open(source, "rb") as file:
                reader = PyPDF2.PdfReader(file)

                # Extract basic metadata
                info = reader.metadata
                if info:
                    title = info.get("/Title", file_name)
                    author = info.get("/Author", "Unknown")
                    creation_date = info.get("/CreationDate", None)
                else:
                    title = file_name
                    author = "Nik Jois"
                    creation_date = None

                # Process each page
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()

                    if not text.strip():
                        continue  # Skip empty pages

                    # Create a document for this page
                    doc = Document(
                        id=f"{file_id}_page_{i+1}",
                        title=f"{title} - Page {i+1}",
                        content=text,
                        source=source,
                        metadata={
                            **self.metadata,
                            "document_type": "pdf",
                            "page_number": i + 1,
                            "total_pages": len(reader.pages),
                            "author": author,
                            "creation_date": creation_date,
                        },
                    )
                    documents.append(doc)

        except Exception as e:
            print(f"Error processing PDF {source}: {e}")

        print(f"Loaded {len(documents)} pages from PDF: {source}")
        return documents


class WebLoader(DocumentLoader):
    """Loader for web pages."""

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the web loader.

        Args:
            metadata: Additional metadata to add to all documents
        """
        if not HAS_WEB_SUPPORT:
            raise ImportError(
                "Web scraping requires requests and beautifulsoup4 libraries"
            )
        self.metadata = metadata or {}

    def load(self, source: str) -> List[Document]:
        """Load documents from a web URL."""
        if not source.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {source}")

        documents = []

        try:
            # Fetch the web page
            response = requests.get(source, timeout=10)
            response.raise_for_status()

            # Parse the HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract title
            title = soup.title.string if soup.title else urlparse(source).netloc

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract text from the main content
            main_content = ""

            # Try to find the main content container
            main_elements = soup.find_all(
                ["article", "main", "div"], class_=re.compile(r"content|main|article")
            )

            if main_elements:
                # Use the largest content container
                main_element = max(main_elements, key=lambda x: len(x.get_text()))
                main_content = main_element.get_text(separator="\n")
            else:
                # Fallback to body text
                main_content = soup.get_text(separator="\n")

            # Clean up the text
            lines = (line.strip() for line in main_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)

            # Create a document
            if text:
                parsed_url = urlparse(source)
                domain = parsed_url.netloc
                path = parsed_url.path
                doc_id = f"{domain.replace('.', '_')}_{path.replace('/', '_')}"

                doc = Document(
                    id=doc_id,
                    title=title,
                    content=text,
                    source=source,
                    metadata={
                        **self.metadata,
                        "document_type": "web",
                        "domain": domain,
                        "path": path,
                        "fetch_time": response.headers.get("Date"),
                    },
                )
                documents.append(doc)

        except Exception as e:
            print(f"Error processing URL {source}: {e}")

        print(f"Loaded {len(documents)} documents from URL: {source}")
        return documents


class CSVLoader(DocumentLoader):
    """Loader for CSV/Excel files."""

    def __init__(
        self,
        text_columns: List[str],
        id_column: Optional[str] = None,
        title_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the CSV/Excel loader.

        Args:
            text_columns: Columns to combine for the document content
            id_column: Column to use as document ID (defaults to index)
            title_column: Column to use as document title (defaults to file name)
            metadata_columns: Columns to add as metadata
            metadata: Additional metadata to add to all documents
        """
        self.text_columns = text_columns
        self.id_column = id_column
        self.title_column = title_column
        self.metadata_columns = metadata_columns or []
        self.metadata = metadata or {}

    def load(self, source: str) -> List[Document]:
        """Load documents from a CSV or Excel file."""
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")

        documents = []

        try:
            # Determine file type from extension
            file_ext = os.path.splitext(source)[1].lower()

            if file_ext == ".csv":
                df = pd.read_csv(source)
            elif file_ext in [".xlsx", ".xls"]:
                df = pd.read_excel(source)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

            file_name = os.path.basename(source)

            # Process each row as a document
            for idx, row in df.iterrows():
                # Extract content from specified columns
                contents = []
                for col in self.text_columns:
                    if col in row and not pd.isna(row[col]):
                        contents.append(str(row[col]))

                content = "\n\n".join(contents)

                if not content.strip():
                    continue  # Skip rows with no content

                # Extract document ID
                if self.id_column and self.id_column in row:
                    doc_id = str(row[self.id_column])
                else:
                    doc_id = f"{file_name.replace('.', '_')}_{idx}"

                # Extract document title
                if self.title_column and self.title_column in row:
                    title = str(row[self.title_column])
                else:
                    title = f"{file_name} - Row {idx+1}"

                # Extract metadata
                doc_metadata = {
                    **self.metadata,
                    "document_type": "csv" if file_ext == ".csv" else "excel",
                    "row_index": idx,
                }

                for col in self.metadata_columns:
                    if col in row and not pd.isna(row[col]):
                        doc_metadata[col] = row[col]

                # Create a document
                doc = Document(
                    id=doc_id,
                    title=title,
                    content=content,
                    source=source,
                    metadata=doc_metadata,
                )
                documents.append(doc)

        except Exception as e:
            print(f"Error processing file {source}: {e}")

        print(f"Loaded {len(documents)} documents from file: {source}")
        return documents


class MarkdownLoader(DocumentLoader):
    """Loader for Markdown files."""

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the Markdown loader.

        Args:
            metadata: Additional metadata to add to all documents
        """
        if not HAS_MARKDOWN_SUPPORT:
            raise ImportError("Markdown support requires markdown library")
        self.metadata = metadata or {}

    def load(self, source: str) -> List[Document]:
        """Load documents from a Markdown file."""
        if not os.path.exists(source):
            raise FileNotFoundError(f"Markdown file not found: {source}")

        documents = []

        try:
            with open(source, "r", encoding="utf-8") as file:
                text = file.read()

            # Extract title from first heading
            title_match = re.search(r"^# (.*?)$", text, re.MULTILINE)
            if title_match:
                title = title_match.group(1)
            else:
                title = os.path.basename(source)

            # Extract sections based on headings
            # This splits the document into sections at each heading
            sections = re.split(r"^## (.*?)$", text, flags=re.MULTILINE)

            if len(sections) <= 1:
                # No second-level headings, use the entire document
                file_name = os.path.basename(source)
                file_id = re.sub(r"[^\w\-_]", "_", file_name)

                doc = Document(
                    id=file_id,
                    title=title,
                    content=text,
                    source=source,
                    metadata={**self.metadata, "document_type": "markdown"},
                )
                documents.append(doc)
            else:
                # First item is content before first heading
                file_name = os.path.basename(source)
                file_id = re.sub(r"[^\w\-_]", "_", file_name)

                # Convert the intro section if it has content
                intro = sections[0].strip()
                if intro:
                    doc = Document(
                        id=f"{file_id}_intro",
                        title=f"{title} - Introduction",
                        content=intro,
                        source=source,
                        metadata={
                            **self.metadata,
                            "document_type": "markdown",
                            "section": "introduction",
                        },
                    )
                    documents.append(doc)

                # Process the heading-content pairs
                for i in range(1, len(sections), 2):
                    if i + 1 < len(sections):
                        section_title = sections[i].strip()
                        section_content = sections[i + 1].strip()

                        if section_content:
                            section_id = re.sub(r"[^\w\-_]", "_", section_title.lower())

                            doc = Document(
                                id=f"{file_id}_{section_id}",
                                title=f"{title} - {section_title}",
                                content=section_content,
                                source=source,
                                metadata={
                                    **self.metadata,
                                    "document_type": "markdown",
                                    "section": section_title,
                                },
                            )
                            documents.append(doc)

        except Exception as e:
            print(f"Error processing Markdown file {source}: {e}")

        print(f"Loaded {len(documents)} sections from Markdown: {source}")
        return documents


class JSONLoader(DocumentLoader):
    """Loader for JSON data."""

    def __init__(
        self,
        content_key: str,
        id_key: Optional[str] = None,
        title_key: Optional[str] = None,
        metadata_mapping: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        items_key: Optional[str] = None,  # For accessing a list of items in the JSON
    ):
        """
        Initialize the JSON loader.

        Args:
            content_key: JSON key for the document content
            id_key: JSON key for the document ID (optional)
            title_key: JSON key for the document title (optional)
            metadata_mapping: Mapping of JSON keys to metadata keys
            metadata: Additional metadata to add to all documents
            items_key: Key to a list of items to process (if JSON contains an array)
        """
        self.content_key = content_key
        self.id_key = id_key
        self.title_key = title_key
        self.metadata_mapping = metadata_mapping or {}
        self.metadata = metadata or {}
        self.items_key = items_key

    def load(self, source: str) -> List[Document]:
        """Load documents from a JSON file or a JSON string."""
        documents = []

        try:
            # Check if source is a file path or a JSON string
            if os.path.exists(source):
                with open(source, "r", encoding="utf-8") as file:
                    data = json.load(file)
                file_name = os.path.basename(source)
            else:
                # Try parsing as a JSON string
                data = json.loads(source)
                file_name = "json_data"

            # Get items to process
            items = data
            if self.items_key:
                if isinstance(data, dict) and self.items_key in data:
                    items = data[self.items_key]
                else:
                    raise ValueError(
                        f"Items key '{self.items_key}' not found in JSON data"
                    )

            if not isinstance(items, list):
                items = [items]  # Handle single item case

            # Process each item
            for idx, item in enumerate(items):
                # Extract content
                if isinstance(item, dict) and self.content_key in item:
                    content = item[self.content_key]
                    if not isinstance(content, str):
                        content = json.dumps(content)
                else:
                    print(
                        f"Skipping item {idx}: content key '{self.content_key}' not found"
                    )
                    continue

                if not content.strip():
                    continue  # Skip items with no content

                # Extract document ID
                if self.id_key and isinstance(item, dict) and self.id_key in item:
                    doc_id = str(item[self.id_key])
                else:
                    doc_id = f"{file_name.replace('.', '_')}_{idx}"

                # Extract document title
                if self.title_key and isinstance(item, dict) and self.title_key in item:
                    title = str(item[self.title_key])
                else:
                    title = f"{file_name} - Item {idx+1}"

                # Extract metadata
                doc_metadata = {
                    **self.metadata,
                    "document_type": "json",
                    "item_index": idx,
                }

                for json_key, meta_key in self.metadata_mapping.items():
                    if isinstance(item, dict) and json_key in item:
                        doc_metadata[meta_key] = item[json_key]

                # Create a document
                doc = Document(
                    id=doc_id,
                    title=title,
                    content=content,
                    source=source if os.path.exists(source) else "json_string",
                    metadata=doc_metadata,
                )
                documents.append(doc)

        except Exception as e:
            print(f"Error processing JSON {source}: {e}")

        print(f"Loaded {len(documents)} documents from JSON source")
        return documents


def load_and_index_documents(
    rag_system: RAGSystem,
    loader: DocumentLoader,
    source: str,
    save_path: Optional[str] = None,
) -> int:
    """
    Load documents using the specified loader and add them to the RAG system.

    Args:
        rag_system: The RAG system to add documents to
        loader: The document loader to use
        source: The source to load documents from
        save_path: Path to save the updated index (optional)

    Returns:
        Number of documents loaded and indexed
    """
    try:
        # Load documents
        documents = loader.load(source)

        if not documents:
            print(f"No documents loaded from {source}")
            return 0

        # Add documents to the RAG system
        for doc in documents:
            rag_system.add_document(doc)

        # Save the updated index if requested
        if save_path:
            rag_system.save(save_path)
            print(f"Saved updated index to {save_path}")

        return len(documents)

    except Exception as e:
        print(f"Error loading and indexing documents: {e}")
        return 0


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Load custom documents into the LlamaDB RAG system"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source to load documents from (file path, URL, etc.)",
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=["pdf", "web", "csv", "markdown", "json"],
        help="Type of document source",
    )
    parser.add_argument(
        "--save-path",
        default="knowledge_base.llamadb",
        help="Path to save the updated index",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=600, help="Chunk size for splitting documents"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap for splitting documents",
    )

    # CSV/Excel specific arguments
    parser.add_argument(
        "--text-columns",
        help="Comma-separated list of columns to use for text content (CSV/Excel only)",
    )
    parser.add_argument(
        "--id-column", help="Column to use as document ID (CSV/Excel only)"
    )
    parser.add_argument(
        "--title-column", help="Column to use as document title (CSV/Excel only)"
    )
    parser.add_argument(
        "--metadata-columns",
        help="Comma-separated list of columns to add as metadata (CSV/Excel only)",
    )

    # JSON specific arguments
    parser.add_argument(
        "--content-key",
        default="content",
        help="JSON key for document content (JSON only)",
    )
    parser.add_argument(
        "--items-key", help="JSON key for list of items to process (JSON only)"
    )

    args = parser.parse_args()

    # Initialize the RAG system
    rag = RAGSystem(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    # Load existing index if it exists
    if os.path.exists(args.save_path):
        try:
            print(f"Loading existing index from {args.save_path}...")
            from llamadb.core import VectorIndex

            rag.index = VectorIndex.load(args.save_path)
            print(f"Index loaded with {rag.index.count()} existing chunks")
        except Exception as e:
            print(f"Error loading existing index: {e}")
            print("Will create a new index.")

    # Create the appropriate document loader
    if args.type == "pdf":
        loader = PDFLoader()

    elif args.type == "web":
        loader = WebLoader()

    elif args.type == "csv":
        if not args.text_columns:
            parser.error("--text-columns is required for CSV/Excel sources")

        text_columns = args.text_columns.split(",")
        metadata_columns = (
            args.metadata_columns.split(",") if args.metadata_columns else []
        )

        loader = CSVLoader(
            text_columns=text_columns,
            id_column=args.id_column,
            title_column=args.title_column,
            metadata_columns=metadata_columns,
        )

    elif args.type == "markdown":
        loader = MarkdownLoader()

    elif args.type == "json":
        loader = JSONLoader(
            content_key=args.content_key,
            id_key=args.id_column,
            title_key=args.title_column,
            items_key=args.items_key,
        )

    # Load and index the documents
    num_docs = load_and_index_documents(
        rag_system=rag, loader=loader, source=args.source, save_path=args.save_path
    )

    print(f"Successfully loaded and indexed {num_docs} documents")


if __name__ == "__main__":
    main()
