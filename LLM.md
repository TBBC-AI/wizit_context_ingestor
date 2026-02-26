# wizit_context_ingestor — LLM Developer Skill

> **Purpose of this file:** Provide an LLM assistant (or a new human contributor) with a single, authoritative reference for understanding, extending, and debugging this codebase.  Every section is intentionally dense with facts; avoid paraphrasing when concrete code paths matter.

---

## 1. Project Purpose

`wizit_context_ingestor` is a **Python library** (package name `wizit_context_ingestor`, built with `uv`) for turning raw PDF documents into semantically enriched, vector-indexed chunks ready for **Retrieval-Augmented Generation (RAG)**.

The two core pipelines are:

| Pipeline | Entry point | Output |
|---|---|---|
| **Transcription** | `TranscriptionManager.transcribe_document(file_key)` | A `.md` file in the target storage |
| **Context chunking** | `ChunksManager.gen_context_chunks(file_key, src, dst)` | `list[Document]` with context-wrapped chunks indexed in PostgreSQL pgvector |

The public API surface is intentionally small — three manager classes exported from `src/wizit_context_ingestor/__init__.py`:

```python
from wizit_context_ingestor import (
    TranscriptionManager,
    ChunksManager,
    PgKdbProvisioningManager,
)
```

---

## 2. Repository Layout

```
wizit_context_ingestor_open/
├── src/wizit_context_ingestor/         # Installable library source
│   ├── __init__.py                     # Public exports
│   ├── main_transcription.py           # TranscriptionManager facade
│   ├── main_chunks.py                  # ChunksManager + PgKdbProvisioningManager facades
│   │
│   ├── domain/                         # Pure business logic — NO I/O, NO infra imports
│   │   ├── models.py                   # ParsedDocPage, ParsedDoc dataclasses
│   │   └── services.py                 # ParseDocModelService (pymupdf PDF → base64 pages)
│   │
│   ├── application/                    # Orchestration — depends on domain + interfaces
│   │   ├── interfaces.py               # ABCs: AiApplicationService, PersistenceService,
│   │   │                               #       RagChunker, EmbeddingsManager, TranscriptionService
│   │   ├── transcription_app.py        # TranscriptionApp — drives transcription workflow
│   │   ├── context_chunk_app.py        # ContextChunksInDocumentApp — drives context workflow
│   │   └── kdb_service.py              # KdbService — vector store CRUD
│   │
│   ├── infra/                          # Concrete implementations of interfaces
│   │   ├── vertex_model.py             # VertexModels (AiApplicationService) — Gemini + Claude via Vertex AI
│   │   ├── aws_model.py                # AWS Bedrock model (alternative AI backend)
│   │   ├── persistence/
│   │   │   ├── local_storage.py        # LocalStorageService (PersistenceService)
│   │   │   ├── s3_storage.py           # S3StorageService (PersistenceService)
│   │   │   ├── pg_connection_manager.py
│   │   │   └── pg_engine_manager.py
│   │   ├── rag/
│   │   │   ├── semantic_chunks.py      # SemanticChunks (RagChunker) — langchain-experimental
│   │   │   └── pg_embeddings.py        # PgEmbeddingsManager (EmbeddingsManager) — pgvector
│   │   └── secrets/
│   │       └── aws_secrets_manager.py  # AwsSecretsManager — fetches GCP SA JSON from AWS Secrets Manager
│   │
│   ├── workflows/                      # LangGraph state machines
│   │   ├── transcription_workflow.py   # StateGraph: transcribe → check_transcription → validate
│   │   ├── transcription_nodes.py      # Node functions for transcription graph
│   │   ├── transcription_state.py      # TypedDict states
│   │   ├── transcription_schemas.py    # Pydantic output schemas (Transcription, TranscriptionCheck)
│   │   ├── transcription_tools.py
│   │   ├── context_workflow.py         # StateGraph: gen_context → tools → return_context
│   │   ├── context_nodes.py            # Node functions for context graph
│   │   ├── context_state.py            # TypedDict state (ContextState)
│   │   └── context_tools.py            # think_tool, complete_context_gen LangChain tools
│   │
│   ├── data/
│   │   ├── prompts.py                  # All system prompts + Pydantic output models (ContextChunk, Transcription)
│   │   ├── storage.py                  # StorageServices enum (LOCAL, S3)
│   │   └── kdb.py
│   │
│   └── utils/
│       └── file_utils.py               # validate_file_name_format(file_key) → bool
│
├── test.py                             # CLI runner (transcribe | context | query | provisioning | find_by_name | delete_by_name)
├── context.test.py                     # LLM-as-judge evaluation helper
├── example.env                         # Template for required environment variables
└── pyproject.toml                      # uv build, Python ≥ 3.12, dependency list
```

---

## 3. Architecture: Clean Architecture Layers

```
┌──────────────────────────────────────────────────────┐
│  Facades  (main_transcription.py, main_chunks.py)    │  ← Consumer entry points
├──────────────────────────────────────────────────────┤
│  Application Layer  (application/)                   │  ← Orchestration & use-cases
│    TranscriptionApp, ContextChunksInDocumentApp,     │
│    KdbService                                        │
├──────────────────────────────────────────────────────┤
│  Domain Layer  (domain/)                             │  ← Pure logic, no I/O
│    ParsedDocPage, ParsedDoc, ParseDocModelService    │
├──────────────────────────────────────────────────────┤
│  Infrastructure Layer  (infra/)                      │  ← I/O implementations
│    VertexModels, LocalStorageService, S3Storage,     │
│    PgEmbeddingsManager, SemanticChunks,              │
│    AwsSecretsManager                                 │
└──────────────────────────────────────────────────────┘
```

**Dependency rule:** every layer may only import inward.  Infra implements the ABCs declared in `application/interfaces.py`.  Domain has zero external imports.

---

## 4. End-to-End Data Flows

### 4.1 Transcription Pipeline

```
PDF file (local disk or S3)
  │
  ▼ domain/services.py — ParseDocModelService
  Split into pages → each page serialised to base64 PDF bytes via pymupdf
  │
  ▼ application/transcription_app.py — TranscriptionApp.process_document()
  asyncio.gather() fires one LangGraph workflow per page concurrently
  │
  ▼ workflows/transcription_workflow.py  (StateGraph)
  START
    → "transcribe"  node          (LLM structured output → Transcription pydantic)
    → "check_transcription"  node (LLM structured output → TranscriptionCheck pydantic)
    → "validate_transcription_results"  node
          if accuracy < threshold AND retries left  → back to "transcribe"
          else                                      → END
  │
  ▼ ParseDocModelService.create_md_content(parsed_pages)
  Reassembles sorted pages into one ParsedDoc with markdown:
    "## Page N\n\n{page_text}\n\n"
  │
  ▼ PersistenceService.save_parsed_document(f"{file_key}.md", parsed_doc, tags)
  Written to target_storage_route as <filename>.md
```

### 4.2 Context Chunking Pipeline

```
Markdown file (.md) from target_storage_route
  │
  ▼ PersistenceService.load_markdown_file_content(file_key)
  Raw markdown string
  │
  ▼ infra/rag/semantic_chunks.py — SemanticChunks.gen_chunks_for_document()
  langchain-experimental SemanticChunker produces list[Document]
  Each Document.metadata["source"] = file_key
  │
  ▼ application/context_chunk_app.py — ContextChunksInDocumentApp
  asyncio.gather() fires one ContextWorkflow per chunk concurrently
  │
  ▼ workflows/context_workflow.py  (StateGraph)
  START
    → "gen_context"    node  (LLM with bound tools: think_tool, complete_context_gen)
    → "tools"          node  (executes tool calls)
          if complete_context_gen called → goto "return_context"
          else                           → goto "gen_context"  (think loop)
    → "return_context" node  → END
  │
  ▼ Chunk enrichment
  chunk.page_content = "<context>\n{context}\n</context>\n<content>\n{original}\n</content>"
  File tags (from S3/local) merged into chunk.metadata
  │
  ▼ KdbService.index_documents_in_vector_store(docs)
  infra/rag/pg_embeddings.py — PgEmbeddingsManager
  Uses langchain_postgres PGVectorStore + SQLRecordManager (upsert deduplication)
```

---

## 5. Key Classes Reference

### 5.1 TranscriptionManager (`main_transcription.py`)

```python
TranscriptionManager(
    gcp_project_id: str,
    gcp_project_location: str,
    gcp_secret_name: str,           # AWS Secrets Manager secret name holding GCP SA JSON
    langsmith_api_key: str,
    langsmith_project_name: str,
    storage_service: StorageServices,   # StorageServices.LOCAL or StorageServices.S3
    source_storage_route: str,          # local dir path or S3 bucket name
    target_storage_route: str,
    llm_model_id: str = "claude-sonnet-4-6",
    target_language: str = "es",
    transcription_additional_instructions: str = "",
    transcription_accuracy_threshold: float = 0.90,  # must be 0.0–0.95
    max_transcription_retries: int = 2,              # must be 1–3
)
```

Sole async method: `await manager.transcribe_document(file_key: str) -> str`
Returns the saved markdown file path `f"{file_key}.md"`.

### 5.2 ChunksManager (`main_chunks.py`)

```python
ChunksManager(
    gcp_project_id, gcp_project_location, gcp_secret_name,
    langsmith_api_key, langsmith_project_name,
    storage_service: StorageServices,
    kdb_service_name: Literal["pg"],
    kdb_params: Dict[str, Any],      # see §5.4 for kdb_params schema
    llm_model_id: str = "claude-sonnet-4-6",
    embeddings_model_id: str = "text-multilingual-embedding-002",
    target_language: str = "es",
)
```

Key methods:

| Method | Async | Description |
|---|---|---|
| `gen_context_chunks(file_key, src_route, dst_route)` | ✅ | Full chunk pipeline; returns `list[Document]` |
| `index_documents_in_vector_store(docs)` | ❌ | Indexes Document list into pgvector |
| `search_records(query)` | ❌ | Similarity search; returns `list[Document]` |
| `search_documents_by_file_name(file_name)` | ❌ | Filter by metadata `source` field |
| `delete_documents_by_file_name(file_name)` | ❌ | Delete all chunks for a document |

### 5.3 PgKdbProvisioningManager (`main_chunks.py`)

One-time setup. Creates the pgvector table and HNSW index:

```python
manager = PgKdbProvisioningManager(
    gcp_project_id, gcp_project_location, gcp_secret_name,
    embeddings_model_id="gemini-embedding-001",
    kdb_params={...},
)
manager.provision_vector_store()
```

### 5.4 kdb_params Schema

```python
{
    "pg_connection": "postgresql://user:pass@host:5432/dbname",
    "embeddings_vectors_table_name": "my_collection",   # pgvector table
    "records_manager_table_name": "my_collection",      # SQLRecordManager table
    "content_column": "document",
    "metadata_json_column": "metadata",
    "id_column": "id",
    "vector_size": 3072,   # must match the embeddings model output dimension
}
```

**Vector size by model:**
- `gemini-embedding-001` → **3072**
- `text-multilingual-embedding-002` → **768**

---

## 6. LangGraph Workflows

### 6.1 Transcription Workflow (`workflows/transcription_workflow.py`)

```
TranscriptionInputState  ←  initial messages (HumanMessage with base64 PDF)
         │
    ┌────▼────────────────────────────────────────────────────────┐
    │  StateGraph(TranscriptionState, input_schema=TranscriptionInputState) │
    │                                                             │
    │  START ──► "transcribe" ──► "check_transcription"          │
    │                                    │                        │
    │                        "validate_transcription_results"     │
    │                         ├── accuracy OK ──────────► END    │
    │                         └── retry left ──► "transcribe"    │
    └─────────────────────────────────────────────────────────────┘
```

Configurable per-invocation via `config["configurable"]`:
- `transcription_accuracy_threshold` (float)
- `max_transcription_retries` (int)

**LLM output schemas** (structured output, Pydantic):
- `Transcription` → `{transcription: str, language: str}`
- `TranscriptionCheck` → `{transcription_accuracy: float, transcription_notes: str}`

### 6.2 Context Workflow (`workflows/context_workflow.py`)

```
ContextState  ←  {messages, document_content}
         │
    ┌────▼──────────────────────────────────────────────────────┐
    │  StateGraph(ContextState)                                 │
    │                                                           │
    │  START ──► "gen_context" ──► "tools"                      │
    │                │                  │                       │
    │                │    complete_context_gen called            │
    │                │              ▼                           │
    │                │        "return_context" ──► END          │
    │                │                                          │
    │                │    only think_tool called                │
    │                └──◄──────────────                        │
    └───────────────────────────────────────────────────────────┘
```

Tools:
- `think_tool` — internal reasoning step (does not end the loop)
- `complete_context_gen` — signals the context is complete; its return value becomes `state["context"]`

The LLM is bound to both tools via `llm_model.bind_tools(self.tools)`.  The tool node inspects `tool_call["name"]` to route back to `gen_context` (think loop) or forward to `return_context` (done).

---

## 7. AI Model Backend

All model loading is handled by `infra/vertex_model.py — VertexModels`.

**Authentication flow:**
1. `AwsSecretsManager.get_secret(gcp_secret_name)` fetches a JSON string from AWS Secrets Manager
2. The JSON is parsed into a dict and passed to `google.oauth2.service_account.Credentials.from_service_account_info()`
3. Credentials are injected into both `vertexai.init()` and every model constructor

**Supported LLM backends:**
| `llm_model_id` prefix | Class | Notes |
|---|---|---|
| `gemini` | `ChatGoogleGenerativeAI` | `langchain-google-genai`, uses same GCP credentials |
| `claude` | `ChatAnthropicVertex` | `langchain-google-vertexai`, Claude served via Vertex AI Model Garden |

Default model: `"claude-sonnet-4-6"` (Anthropic on Vertex AI).

**Supported embeddings model:**
- `GoogleGenerativeAIEmbeddings` from `langchain-google-genai`
- Recommended: `"gemini-embedding-001"` (3072-dim) or `"text-multilingual-embedding-002"` (768-dim)

---

## 8. Storage Backends

Both implement `application/interfaces.py — PersistenceService`.

| Method | LocalStorageService | S3StorageService |
|---|---|---|
| `retrieve_raw_file(file_key)` | Checks `{source_route}/{file_key}` on disk | Downloads to `/tmp/{file_key}`, returns local path |
| `load_markdown_file_content(file_key)` | Reads `{source_route}/{file_key}` as UTF-8 | Reads from S3 target bucket |
| `save_parsed_document(file_key, doc, tags)` | Writes `{target_route}/{file_key}` | Uploads to S3 target bucket with optional tags |
| `supports_tagging` | `False` | `True` |

`StorageServices` enum lives in `data/storage.py` — use `StorageServices.LOCAL` or `StorageServices.S3`.

---

## 9. Environment Variables

Copy `example.env` to `.env`. Required variables:

```bash
# Google Cloud
GCP_PROJECT_ID="your-gcp-project-id"
GCP_PROJECT_LOCATION="us-central1"

# AWS Secrets Manager (holds the GCP Service Account JSON)
# No explicit AWS vars needed if the execution environment has an IAM role/profile
# For local dev, ensure ~/.aws/credentials or AWS_PROFILE is set

# PostgreSQL (pgvector)
PG_CONNECTION="postgresql://user:password@host:5432/dbname"

# LangSmith observability (required — Client is always instantiated)
LANGSMITH_API_KEY="ls__..."
LANGCHAIN_PROJECT="your-project-name"
LANGSMITH_TRACING="true"

# S3 (only when using S3 storage backend)
S3_ORIGIN_BUCKET_NAME="your-source-bucket"
S3_TARGET_BUCKET_NAME="your-target-bucket"
```

> **Important:** `LANGSMITH_API_KEY` cannot be empty.  `langsmith.Client(api_key=...)` is instantiated inside `TranscriptionApp` and `ContextChunksInDocumentApp`.  Providing an empty string will raise a runtime error when the first workflow trace is attempted.

---

## 10. CLI Operations (`test.py`)

```bash
# Transcribe a PDF to markdown
python test.py transcribe MyDocument.pdf

# Generate context chunks from a markdown file and index them
python test.py context MyDocument.pdf.md

# Similarity search
python test.py query "What are the main risk factors?"

# Provision a new vector store table + HNSW index
python test.py provisioning my_collection_name

# Find all indexed chunks for a document
python test.py find_by_name MyDocument.pdf.md

# Delete all indexed chunks for a document
python test.py delete_by_name MyDocument.pdf.md
```

All operations are profiled with `pyinstrument` when run via `test.py`.

---

## 11. Extension Patterns

### 11.1 Adding a New Storage Backend

1. Create `src/wizit_context_ingestor/infra/persistence/my_storage.py`
2. Subclass `PersistenceService` and implement all four abstract methods
3. Add a new value to the `StorageServices` enum in `data/storage.py`
4. Handle the new enum value in `PersistenceManager.retrieve_storage_service()` inside both `main_transcription.py` and `main_chunks.py`

```python
# data/storage.py
class StorageServices(str, Enum):
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"   # ← add here

# main_transcription.py  PersistenceManager.retrieve_storage_service()
elif self.storage_service == StorageServices.GCS:
    return GcsStorageService(...)
```

### 11.2 Adding a New LLM Backend

1. Create `src/wizit_context_ingestor/infra/my_model.py`
2. Subclass `AiApplicationService` and implement `load_chat_model()`
3. Add optional `load_embeddings_model()` if your backend supports embeddings
4. Inject the new class into `TranscriptionManager` or `ChunksManager` by passing it as `ai_application_service` — **or** extend the manager constructors with a new `model_backend` parameter

The application layer only types against `AiApplicationService`, so any compliant implementation will work.

### 11.3 Adding a New Workflow Node

All workflow nodes follow the same signature:

```python
def my_node(self, state: MyState, config: dict) -> Command | dict:
    # Read from state
    # Call LLM / tools
    # Return Command(goto="next_node", update={...}) or plain dict
    ...
```

Use `Command(goto=..., update=...)` when you need to both update state AND control routing.  Return a plain `dict` when the graph edges define routing statically.

### 11.4 Modifying Prompts

All prompts live in `src/wizit_context_ingestor/data/prompts.py` as module-level string constants.  They use `.format()` placeholders:

| Prompt constant | Placeholders |
|---|---|
| `AGENT_TRANSCRIPTION_SYSTEM_PROMPT` | `{transcription_additional_instructions}`, `{transcription_notes}` |
| `IMAGE_TRANSCRIPTION_CHECK_SYSTEM_PROMPT` | `{transcription_additional_instructions}`, `{transcription}` |
| `WORKFLOW_CONTEXT_CHUNKS_IN_DOCUMENT_SYSTEM_PROMPT` | `{document_content}`, `{context_additional_instructions}` |

Pydantic output schemas `Transcription` and `TranscriptionCheck` live in `workflows/transcription_schemas.py`; `ContextChunk` lives in `data/prompts.py`.

---

## 12. Common Developer Tasks

### Task: Transcribe a PDF locally (no AWS, no S3)

```python
import asyncio, os
from dotenv import load_dotenv
from wizit_context_ingestor import TranscriptionManager

load_dotenv()

manager = TranscriptionManager(
    gcp_project_id=os.environ["GCP_PROJECT_ID"],
    gcp_project_location=os.environ["GCP_PROJECT_LOCATION"],
    gcp_secret_name="my-gcp-sa-secret",      # must exist in AWS Secrets Manager
    langsmith_api_key=os.environ["LANGSMITH_API_KEY"],
    langsmith_project_name="my-project",
    storage_service="local",
    source_storage_route="data",             # folder containing the PDF
    target_storage_route="tmp",              # folder where .md is written
)
result = asyncio.run(manager.transcribe_document("MyReport.pdf"))
print(result)  # "MyReport.pdf.md"
```

### Task: Chunk a markdown file and index it

```python
import asyncio, os
from dotenv import load_dotenv
from wizit_context_ingestor import ChunksManager

load_dotenv()

manager = ChunksManager(
    gcp_project_id=os.environ["GCP_PROJECT_ID"],
    gcp_project_location=os.environ["GCP_PROJECT_LOCATION"],
    gcp_secret_name="my-gcp-sa-secret",
    langsmith_api_key=os.environ["LANGSMITH_API_KEY"],
    langsmith_project_name="my-project",
    storage_service="local",
    kdb_service_name="pg",
    kdb_params={
        "pg_connection": os.environ["PG_CONNECTION"],
        "embeddings_vectors_table_name": "my_docs",
        "records_manager_table_name": "my_docs",
        "content_column": "document",
        "metadata_json_column": "metadata",
        "id_column": "id",
        "vector_size": 3072,
    },
    embeddings_model_id="gemini-embedding-001",
)

chunks = asyncio.run(manager.gen_context_chunks("MyReport.pdf.md", "tmp", "tmp"))
manager.index_documents_in_vector_store(chunks)
```

### Task: Query the vector store

```python
results = manager.search_records("What are the key financial highlights?")
for doc in results:
    print(doc.page_content[:300])
    print(doc.metadata)
```

### Task: Evaluate chunk quality (LLM-as-judge)

Use `context.test.py` as a template.  It loads a pre-generated `context_chunks.json` and the source markdown, then calls `VertexModels` with the `LLM_AS_JUDGE_SYSTEM_PROMPT` to score each chunk's context quality on a 0–5 scale.

---

## 13. Validation Rules and Guard Rails

| Component | Rule | Exception raised |
|---|---|---|
| `TranscriptionApp.__init__` | `transcription_accuracy_threshold` must be `0.0 – 0.95` | `ValueError` |
| `TranscriptionApp.__init__` | `max_transcription_retries` must be `1 – 3` | `ValueError` |
| `TranscriptionManager.transcribe_document` | `file_key` must pass `validate_file_name_format()` | `ValueError` |
| `ChunksManager.gen_context_chunks` | `file_key` must end with `.md` (enforced in `test.py`, not in library) | `ValueError` |
| `VertexModels.load_chat_model` | `llm_model_id` must contain `"gemini"` or `"claude"` | `ValueError` |

`validate_file_name_format()` in `utils/file_utils.py` rejects file names with spaces or unsupported special characters (only alphanumerics, underscores, hyphens, and dots allowed).

---

## 14. Observability & Tracing

- **LangSmith** is used for workflow tracing.  A `langsmith.Client` is instantiated in both `TranscriptionApp` and `ContextChunksInDocumentApp`.
- All workflow invocations are wrapped with `langsmith.tracing_context(enabled=True, project_name=..., client=...)`.
- Set `LANGSMITH_TRACING=true` in `.env` to activate trace uploads.
- Set `LANGSMITH_API_KEY` and `LANGCHAIN_PROJECT` to route traces to the correct project in the LangSmith dashboard.

---

## 15. Dependency Highlights

| Package | Role |
|---|---|
| `langchain-google-vertexai` | `ChatAnthropicVertex` (Claude via Vertex AI Model Garden) |
| `langchain-google-genai` | `ChatGoogleGenerativeAI`, `GoogleGenerativeAIEmbeddings` |
| `langchain-experimental` | `SemanticChunker` for semantic chunk splitting |
| `langchain-postgres` | `PGVectorStore`, `SQLRecordManager` (upsert deduplication) |
| `langgraph` | `StateGraph` for transcription and context workflows |
| `langsmith` | Tracing and observability |
| `pymupdf` | PDF → per-page base64 serialisation |
| `boto3` | AWS S3 storage + AWS Secrets Manager |
| `sqlalchemy[asyncio]` | Async DB engine for pgvector connection |
| `psycopg2-binary` | Sync PostgreSQL driver |
| `anthropic[vertex]` | Underlying Anthropic SDK for Vertex integration |
| `pillow` | Image processing support |

---

## 16. Known Patterns & Conventions

1. **Async-first orchestration** — `asyncio.gather()` is used to process all pages / chunks concurrently.  Blocking calls (storage, secret retrieval) remain synchronous.

2. **Facade pattern** — `TranscriptionManager` and `ChunksManager` hide all infrastructure wiring from consumers. They are the only objects a library user needs to instantiate.

3. **LangGraph Command routing** — nodes return `Command(goto=..., update=...)` for dynamic routing, and plain `dict` for static edges defined in `gen_workflow()`.

4. **Structured LLM output** — transcription nodes use `llm.with_structured_output(PydanticModel)` to guarantee typed outputs.  Context nodes use `llm.bind_tools(tools)` and tool-based signalling instead.

5. **GCP credentials via AWS** — GCP Service Account JSON is stored in AWS Secrets Manager and fetched at runtime; it is never stored on disk or in environment variables directly.

6. **Storage abstraction** — both `main_transcription.py` and `main_chunks.py` contain a duplicate `PersistenceManager` helper class.  This is intentional to keep each module independently importable without cross-facade imports.

7. **Chunk content format** — after context enrichment every chunk's `page_content` follows this exact template:
   ```
   <context>
   {llm_generated_context}
   </context>
    <content>
   {original_chunk_text}
   </content>
   ```
   Downstream RAG queries receive both the semantic context and the verbatim content in a single vector embedding.