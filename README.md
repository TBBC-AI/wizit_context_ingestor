# wizit_context_ingestor

A powerful document processing and ingestion system that leverages AI services for document transcription, analysis, and semantic chunking.

## Features

- Document transcription using AWS and Google Cloud AI services
- Semantic chunking of documents for better context understanding
- Vector storage integration with PostgreSQL
- Support for both local and cloud storage (S3)
- Synthetic data generation capabilities
- RAG (Retrieval-Augmented Generation) implementation

## Prerequisites

- Python 3.11 or higher
- Poetry for dependency management
- AWS credentials (for AWS services)
- Google Cloud credentials (for GCP services)
- PostgreSQL database (for vector storage)
- Supabase account (for data storage)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mega-ingestor.git
cd mega-ingestor
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Set up your environment variables by copying the example.env file:
```bash
cp example.env .env
```

4. Fill in your environment variables in the `.env` file with your credentials and configuration.

## Usage

The project provides several main functionalities:

### Document Transcription

```python
from main import transcribe_document

# Transcribe a document using AWS services
transcribe_document("your-document.pdf")

# Transcribe a document using Google Cloud services
cloud_transcribe_document("your-document.pdf")
```

### Context Chunking

```python
from main import context_chunks_in_document

# Get semantic chunks from a document
context_chunks_in_document("your-document.pdf")
```

## Project Structure

```
mega-ingestor/
├── src/
│   ├── application/
│   ├── infra/
│   └── ...
├── data/
├── credentials/
├── main.py
├── app.py
└── pyproject.toml
```

## Dependencies

- llama-parse
- langchain-experimental
- langchain-google-vertexai
- pymupdf
- supabase
- vecs
- langchain-postgres
- boto3
- langchain-aws

## GENERATE THE PACKAGE WITH POETRY

```
    poetry build
```

## PUBLISH PACKAGE

```
    poetry config repositories.tbbcmegaingestor https://aws:$CODEARTIFACT_AUTH_TOKEN@tbbc-mega-ingestor-411728455297.d.codeartifact.us-east-1.amazonaws.com/pypi/tbbc-mega-ingestor-lib/
```

```
    export CODEARTIFACT_AUTH_TOKEN=`aws codeartifact get-authorization-token --domain tbbc-mega-ingestor --domain-owner 411728455297 --region us-east-1 --query authorizationToken --output text --profile <your-profile>`
```

Finally

```
    poetry publish -r tbbcmegaingestor
```

## License

This project is licensed under the Apache License - see the LICENSE file for details.

# TODO

- Do not transcribe logos
- Support for more cloud providers

## Authors

(Daniel Quesada)[https://github.com/daquesada]
(Jeison Patiño)[https://github.com/jeison-patino]
(Javier Fernandez)[https://github.com/javimaufermu]
(Esteban Cerón)[https://github.com/estebance]
