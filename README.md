# RAG - Data Ingestion Pipeline

A Retrieval-Augmented Generation (RAG) system for ingesting PDF documents and creating a searchable vector database.

## Overview

`rag_ingestion_pipeline` is a RAG pipeline that processes PDF documents, generates embeddings, and stores them in a vector database for efficient similarity search and retrieval. The system uses LangChain for document processing, Sentence Transformers for embeddings, and ChromaDB for vector storage.

## Features

- 📄 **PDF Document Processing**: Load and process multiple PDF files
- ✂️ **Smart Text Chunking**: Recursive character text splitting with configurable chunk size and overlap
- 🔍 **Semantic Embeddings**: Generate embeddings using Sentence Transformers models
- 💾 **Vector Storage**: Persistent vector database using ChromaDB
- 🔎 **Metadata Preservation**: Maintains document source, page numbers, and content metadata

## Architecture

### Components

1. **Document Loader** (`PyPDFLoader`): Extracts text and metadata from PDF files
2. **Text Splitter** (`RecursiveCharacterTextSplitter`): Splits documents into manageable chunks
3. **Embedding Manager**: Generates semantic embeddings using Sentence Transformers
4. **Vector Store**: Persistent ChromaDB collection for storing and retrieving documents

### Pipeline Flow

```
PDF Documents → Load → Split → Embed → Store in Vector DB
```

## Installation

### Prerequisites

- Python 3.10 or higher (Python 3.11 recommended)
- pip or uv package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/lakshmana/rag_ingestion_pipeline.git
cd rag_ingestion_pipeline
```

2. Install Python 3.11 (if not already installed):
```bash
brew install python@3.11
```

3. Install dependencies:
```bash
# Using uv (recommended)
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

4. Set up Jupyter kernel (for notebook usage):
```bash
python -m ipykernel install --user --name=rag_ingestion_pipeline --display-name "Python (rag_ingestion_pipeline)"
```

## Usage

### Running the Pipeline

1. Place your PDF documents in `data/rag_data/`:
```bash
cp your_documents.pdf data/rag_data/
```

2. Open the Jupyter notebook:
```bash
jupyter notebook notebook/pdf_loader.ipynb
```

3. Select the kernel: **Python (rag_ingestion_pipeline)**

4. Run all cells to:
   - Load PDF documents
   - Split into chunks
   - Generate embeddings
   - Store in vector database

### Notebook Workflow

The pipeline consists of several steps:

1. **Load Documents**: Processes all PDFs in `data/rag_data/`
2. **Split Documents**: Creates text chunks with configurable size (default: 1000 chars, overlap: 200 chars)
3. **Generate Embeddings**: Uses `all-MiniLM-L6-v2` model (384 dimensions)
4. **Store in Vector DB**: Persists embeddings and metadata to `data/vector_store/`

## Project Structure

```
rag_ingestion_pipeline/
├── notebook/
│   ├── pdf_loader.ipynb      # Main RAG pipeline notebook
│   └── document.ipynb         # Document examples
├── data/
│   ├── rag_data/              # Input PDF files
│   │   ├── javanotes5.pdf
│   │   ├── python_intro.pdf
│   │   └── js-intro.pdf
│   └── vector_store/          # ChromaDB storage
│       └── chroma.sqlite3
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project configuration
└── README.md                 # This file
```

## Dependencies

### Core Libraries

- **LangChain** (v0.3.27+): Document loading and text splitting
- **LangChain Community** (v0.3.31+): Additional document loaders
- **Sentence Transformers** (v5.1.2+): Semantic embeddings
- **ChromaDB** (v1.2.2+): Vector database
- **PyPDF** (v6.1.3+): PDF processing
- **PyMuPDF** (v1.26.5+): Advanced PDF processing

### Optional

- **FAISS** (v1.12.0+): Alternative vector search (installed as `faiss-cpu`)

## Configuration

### Embedding Model

Default model: `all-MiniLM-L6-v2`
- Dimension: 384
- Fast and lightweight
- Suitable for most text similarity tasks

To change the model, modify the `EmbeddingManager` initialization:
```python
embeddings_manager = EmbeddingManager(model_name="your-model-name")
```

### Chunking Parameters

Adjust chunk size and overlap in the `split_documents` function:
```python
chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
```

## Vector Store

The system uses ChromaDB for persistent vector storage:

- **Location**: `data/vector_store/`
- **Collection**: `pdf_documents`
- **Storage Type**: Persistent client (SQLite backend)

### Metadata Stored

- `source`: Original PDF file path
- `page`: Page number in the original document
- `doc_index`: Index in the processed batch
- `content_length`: Length of the text chunk
- `file_type`: "pdf"

## Development

### Virtual Environment

The project uses `venv` for dependency management:

```bash
# Activate virtual environment
source venv/bin/activate

# Deactivate
deactivate
```

### Running the Main Script

The `main.py` script provides a simple entry point:

```bash
# Run the main script
python main.py
```

Note: The complete RAG pipeline is implemented in the Jupyter notebook (`notebook/pdf_loader.ipynb`).

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure you're using Python 3.10+ and have installed all dependencies
2. **Kernel Issues**: Select "Python (rag_ingestion_pipeline)" kernel in Jupyter
3. **Import Errors**: Activate the virtual environment before running notebooks

### Python Version Issues

If you encounter ONNX runtime errors:
```bash
# Install Python 3.11
brew install python@3.11

# Update .python-version
echo "3.11" > .python-version

# Reinstall dependencies
pip install -r requirements.txt
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

