# Custom Document Loaders for LlamaDB RAG

This module provides utilities for loading documents from various sources into LlamaDB RAG systems. It includes loaders for PDFs, websites, CSV/Excel files, Markdown, and JSON data.

## Installation

First, install the required dependencies for the loaders you plan to use:

```bash
# Basic requirements
pip install llamadb openai numpy python-dotenv

# PDF support
pip install PyPDF2

# Web scraping support
pip install requests beautifulsoup4

# CSV/Excel support
pip install pandas openpyxl

# Markdown support
pip install markdown
```

## Usage

### Command Line Interface

The simplest way to use the document loaders is through the command line interface:

```bash
# Load a PDF document
python custom_document_loader.py --source documents/sample.pdf --type pdf --save-path knowledge_base.llamadb

# Load a website
python custom_document_loader.py --source https://example.com --type web --save-path knowledge_base.llamadb

# Load a CSV file with specified columns
python custom_document_loader.py --source data.csv --type csv --text-columns "title,content" --id-column "id" --title-column "title" --metadata-columns "category,author,date" --save-path knowledge_base.llamadb

# Load a Markdown file
python custom_document_loader.py --source documentation.md --type markdown --save-path knowledge_base.llamadb

# Load a JSON file with items
python custom_document_loader.py --source data.json --type json --content-key "text" --items-key "articles" --save-path knowledge_base.llamadb
```

### API Usage

You can also use the document loaders programmatically:

```python
from custom_document_loader import PDFLoader, WebLoader, load_and_index_documents
from rag_example import RAGSystem

# Initialize RAG system
rag = RAGSystem()

# Create a loader
pdf_loader = PDFLoader(metadata={"source_type": "research_paper"})

# Load and index documents
num_docs = load_and_index_documents(
    rag_system=rag,
    loader=pdf_loader,
    source="research_paper.pdf",
    save_path="knowledge_base.llamadb"
)

print(f"Added {num_docs} documents to the knowledge base")

# Generate responses using the updated knowledge base
response, contexts = rag.generate_response(
    query="What are the key findings in the research paper?"
)
print(response)
```

## Loader Types

### PDF Loader

Loads each page of a PDF file as a separate document:

```python
from custom_document_loader import PDFLoader

loader = PDFLoader(metadata={"category": "technical_docs"})
documents = loader.load("document.pdf")
```

**Features:**
- Extracts text from each page
- Preserves PDF metadata (title, author, creation date)
- Adds page number and total pages as metadata

### Web Loader

Loads content from a web page:

```python
from custom_document_loader import WebLoader

loader = WebLoader(metadata={"source_type": "website"})
documents = loader.load("https://example.com/article")
```

**Features:**
- Attempts to extract main content, avoiding navigation and boilerplate
- Preserves URL structure as metadata
- Cleans and normalizes extracted text

### CSV/Excel Loader

Loads each row from a CSV or Excel file as a document:

```python
from custom_document_loader import CSVLoader

loader = CSVLoader(
    text_columns=["title", "description", "content"],
    id_column="id",
    title_column="title",
    metadata_columns=["category", "author", "date"],
    metadata={"source_type": "database"}
)
documents = loader.load("data.csv")
```

**Features:**
- Combines multiple columns into document content
- Flexible mapping of columns to document properties
- Supports both CSV and Excel formats

### Markdown Loader

Loads a Markdown file, splitting by sections:

```python
from custom_document_loader import MarkdownLoader

loader = MarkdownLoader(metadata={"source_type": "documentation"})
documents = loader.load("README.md")
```

**Features:**
- Extracts title from first heading
- Splits document into sections based on headings
- Preserves section titles as metadata

### JSON Loader

Loads documents from JSON data:

```python
from custom_document_loader import JSONLoader

loader = JSONLoader(
    content_key="content",
    id_key="id",
    title_key="title",
    metadata_mapping={"category": "category", "publishedAt": "date"},
    items_key="articles"
)
documents = loader.load("data.json")
```

**Features:**
- Flexible mapping of JSON fields to document properties
- Supports both JSON files and JSON strings
- Can process arrays of items within a JSON structure

## Extending the Loaders

You can create custom loaders by extending the `DocumentLoader` base class:

```python
from custom_document_loader import DocumentLoader
from rag_example import Document
from typing import List

class MyCustomLoader(DocumentLoader):
    """Custom document loader implementation."""
    
    def load(self, source: str) -> List[Document]:
        """Load documents from the source."""
        # Implementation goes here
        documents = []
        
        # Create and add documents
        doc = Document(
            id="unique_id",
            title="Document Title",
            content="Document content...",
            source=source,
            metadata={"key": "value"}
        )
        documents.append(doc)
        
        return documents
```

## Handling Large Document Collections

For large document collections, consider these strategies:

1. **Incremental Loading**: Process documents in batches to avoid memory issues

```python
knowledge_base_path = "knowledge_base.llamadb"
rag = RAGSystem()

# Load existing index if available
if os.path.exists(knowledge_base_path):
    from llamadb.core import VectorIndex
    rag.index = VectorIndex.load(knowledge_base_path)

# Process documents in batches
for batch in document_batches:
    for doc in batch:
        rag.add_document(doc)
    
    # Save after each batch
    rag.save(knowledge_base_path)
```

2. **Parallel Processing**: Utilize multiprocessing for faster document loading

```python
import multiprocessing
from functools import partial

def process_document(doc_path, loader_type="pdf"):
    if loader_type == "pdf":
        loader = PDFLoader()
    # ... other loader types
    
    return loader.load(doc_path)

with multiprocessing.Pool(processes=4) as pool:
    document_lists = pool.map(process_document, document_paths)

all_documents = [doc for sublist in document_lists for doc in sublist]
```

## Best Practices

1. **Add Metadata**: Include rich metadata with your documents to enable filtering:

```python
loader = PDFLoader(metadata={
    "source_type": "research_paper",
    "field": "machine_learning",
    "confidential": False,
    "relevance": "high"
})
```

2. **Use Consistent IDs**: Ensure document IDs are consistent if you update the same document:

```python
# If loading the same document again, use the same ID format
doc_id = f"{source_type}_{file_name.replace('.', '_')}"
```

3. **Handle Errors Gracefully**: Wrap document loading in try/except blocks to continue processing if one document fails:

```python
for source in sources:
    try:
        documents = loader.load(source)
        for doc in documents:
            rag.add_document(doc)
    except Exception as e:
        print(f"Error loading {source}: {e}")
        continue
```

4. **Preprocess Text**: Consider text preprocessing to improve quality:

```python
import re

def preprocess_text(text):
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Other preprocessing steps...
    
    return text.strip()

# Then apply to document content:
doc.content = preprocess_text(doc.content)
``` 