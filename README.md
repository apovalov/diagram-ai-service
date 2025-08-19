# Diagram AI Service

Generate AWS architecture diagrams from natural language descriptions using a production-ready FastAPI service with multiple AI implementations, unified error handling, and intelligent strategy selection.

---

## ✨ What's New (Latest Release)

### 🚀 **Intelligent Execution Strategies**
- **Auto-selection**: Automatically chooses the best AI implementation available
- **Multiple Implementations**: Original Python, LangChain, and LangGraph workflows
- **Smart Fallbacks**: Automatically falls back to working implementations when others fail
- **Zero Downtime**: Graceful degradation ensures service always works

### 🛡️ **Production-Ready Reliability**
- **Unified Error Handling**: Consistent retry logic with exponential backoff across all components
- **100% Backward Compatibility**: All existing configurations continue working
- **Enhanced Observability**: Optional LangSmith integration for monitoring and tracing
- **Comprehensive Testing**: All functionality thoroughly tested for production use

---

## What it does

- **/api/v1/generate-diagram** — turns a plain‑text description into a PNG diagram (base64 in the response) and saves a copy under `/tmp/diagrams/outputs`.
- **/api/v1/assistant** — bonus assistant endpoint that detects intent and (when helpful) triggers diagram generation.
- **Critique-enhanced generation** — optionally analyzes generated diagrams and applies improvements for better quality (with configurable retry attempts).
- **Multiple AI Implementations** — Choose from Original Python, LangChain, or LangGraph implementations with automatic fallbacks.
- Built for **stateless** use; no DB. Temporary files are auto‑cleaned.

---

## Quick start (local)

### Prerequisites

- **Python 3.11+**
- **Graphviz runtime**
  - macOS: `brew install graphviz`
  - Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y graphviz`
  - Windows: install from [https://graphviz.org/download/](https://graphviz.org/download/)

### Setup & run

```bash
# 1) Install uv
pip install uv

# 2) Install deps
uv sync

# 3) Configure env
cp .env.example .env

# NEW: Intelligent execution mode (recommended)
# EXECUTION_MODE=auto              # Automatically selects best available implementation
# ENABLE_FALLBACKS=true           # Enable automatic fallbacks (recommended)

# OpenAI with critique (best quality) - DEFAULT
# OPENAI_API_KEY=your_key
# USE_CRITIQUE_GENERATION=true
# CRITIQUE_MAX_ATTEMPTS=3

# Alternative configurations:
# EXECUTION_MODE=original          # Force original Python implementation
# EXECUTION_MODE=langchain         # Force LangChain implementation  
# EXECUTION_MODE=langgraph         # Force LangGraph workflow

# Optional: LangSmith monitoring
# LANGSMITH_ENABLED=true
# LANGSMITH_API_KEY=your_key
# LANGSMITH_PROJECT=diagram-ai-service

# Legacy options (still work with deprecation warnings):
# USE_LANGCHAIN=true              # DEPRECATED: Use EXECUTION_MODE=langchain
# USE_LANGGRAPH=true              # DEPRECATED: Use EXECUTION_MODE=langgraph

# For local dev/testing
# MOCK_LLM=true

# 4) Run the API
uv run uvicorn app.api.main:app --reload --port 8000
```

Open: `http://localhost:8000/docs` for Swagger UI.

---

## Quick start (Docker)

```bash
docker-compose build
docker-compose up
```

- Source is mounted for fast edit‑reload (`./app -> /app`).
- Diagram images & DOT files are written to `/tmp/diagrams/outputs` (bind‑mounted).
- Ensure `.env` contains `OPENAI_API_KEY` and optionally `EXECUTION_MODE`.

---

## 🎯 Execution Strategies

### **AUTO Mode (Recommended)**
```bash
EXECUTION_MODE=auto
ENABLE_FALLBACKS=true
```
Automatically selects the best available implementation and falls back gracefully:
1. **LangGraph** (if LangSmith enabled and dependencies available)
2. **LangChain** (if dependencies available) 
3. **Original** (always available as final fallback)

### **Specific Strategies**
```bash
# Original Python implementation (fastest, most reliable)
EXECUTION_MODE=original

# LangChain-based implementation (enhanced capabilities)
EXECUTION_MODE=langchain  

# LangGraph workflow implementation (most advanced)
EXECUTION_MODE=langgraph
```

### **Fallback Behavior**
- **Enabled** (`ENABLE_FALLBACKS=true`): Automatically tries alternative implementations if primary fails
- **Disabled** (`ENABLE_FALLBACKS=false`): Fails fast without trying alternatives

---

## Configuration (.env)

### **🆕 New Settings (Recommended)**

| Key                       | Default            | Description                                                   |
| ------------------------- | ------------------ | ------------------------------------------------------------- |
| `EXECUTION_MODE`          | `auto`             | Execution strategy: `auto`, `original`, `langchain`, `langgraph` |
| `ENABLE_FALLBACKS`        | `true`             | Enable automatic fallback to other strategies on failure     |
| `MAX_RETRIES`             | `3`                | Maximum retry attempts for failed operations (0-10)          |
| `BACKOFF_FACTOR`          | `1.5`              | Exponential backoff factor for retries (1.0-3.0)            |
| `LANGSMITH_ENABLED`       | `false`            | Enable LangSmith tracing and monitoring                      |
| `LANGSMITH_API_KEY`       | —                  | LangSmith API key (required if monitoring enabled)           |
| `LANGSMITH_PROJECT`       | `diagram-ai-service` | LangSmith project name                                      |

### **🔧 Core Settings**

| Key                      | Default            | Description                                                           |
| ------------------------ | ------------------ | --------------------------------------------------------------------- |
| `LLM_PROVIDER`           | `openai`           | LLM provider: "openai" or "gemini"                                   |
| `OPENAI_API_KEY`         | —                  | API key for OpenAI (default provider)                                |
| `OPENAI_MODEL`           | `gpt-4o-mini`      | OpenAI model name                                                     |
| `LLM_TIMEOUT`            | `60`               | Request timeout in seconds for LLM calls (10-300)                   |
| `LLM_TEMPERATURE`        | `0.1`              | Temperature for LLM generation (0.0-2.0, lower = more deterministic) |
| `GEMINI_API_KEY`         | —                  | API key for Gemini (rollback option)                                 |
| `GEMINI_MODEL`           | `gemini-2.5-flash` | Gemini model name                                                     |
| `USE_CRITIQUE_GENERATION`| `true`             | Enable critique-enhanced diagram generation for improved quality       |
| `CRITIQUE_MAX_ATTEMPTS`  | `3`                | Maximum critique attempts for better quality (1-5)                    |
| `MOCK_LLM`               | `false`            | If `true`, use deterministic mock analysis (no external calls)        |
| `TMP_DIR`                | `/tmp/diagrams`    | Where images/DOT files are written                                    |

### **⚠️ Legacy Settings (Deprecated but Supported)**

| Key                      | Status             | Migration                                                             |
| ------------------------ | ------------------ | --------------------------------------------------------------------- |
| `USE_LANGCHAIN`          | **Deprecated**     | Use `EXECUTION_MODE=langchain` instead                               |
| `USE_LANGGRAPH`          | **Deprecated**     | Use `EXECUTION_MODE=langgraph` instead                               |
| `LANGCHAIN_FALLBACK`     | **Deprecated**     | Use `ENABLE_FALLBACKS=true` instead                                  |
| `LANGGRAPH_FALLBACK`     | **Deprecated**     | Use `ENABLE_FALLBACKS=true` instead                                  |

> **Note**: Deprecated settings still work but show warning messages. They will be removed in a future version.

---

## 🔍 Monitoring & Observability

### **Request Tracing**
All API requests include:
- **Request ID**: Unique identifier in `X-Request-ID` header
- **Execution Strategy**: Which implementation was used
- **Performance Metrics**: Duration, success/failure rates
- **Error Context**: Detailed error information with operation context

### **LangSmith Integration (Optional)**
```bash
LANGSMITH_ENABLED=true
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=my-project
```

When enabled, provides:
- **Full Request Tracing**: End-to-end visibility into all operations
- **Performance Analytics**: Duration, success rates, error patterns
- **Strategy Performance**: Compare effectiveness of different implementations
- **Error Debugging**: Detailed error context and retry attempts

> **Safe by Design**: LangSmith integration never breaks core functionality. If LangSmith is unavailable, the service continues normally with local logging.

---

## 🛡️ Error Handling & Reliability

### **Unified Error Handling**
- **Consistent Retry Logic**: Exponential backoff with jitter across all components
- **Smart Error Classification**: Distinguishes between retryable and fatal errors
- **Contextual Error Information**: Every error includes operation context and retry count
- **Fallback Mechanisms**: Automatic fallback to alternative implementations

### **Production Safety Features**
- **Graceful Degradation**: Service remains available even when components fail
- **Input Validation**: Comprehensive validation with helpful error messages
- **Timeout Management**: Configurable timeouts prevent hanging requests
- **Memory Management**: Automatic cleanup of temporary files and resources

### **Error Recovery Strategies**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  LangGraph  │───▶│  LangChain  │───▶│  Original   │
│   Strategy  │    │   Strategy  │    │   Strategy  │
└─────────────┘    └─────────────┘    └─────────────┘
      │                    │                    │
      ▼                    ▼                    ▼
   Retry with          Retry with         Always Works
  Backoff Logic      Backoff Logic      (Final Fallback)
```

---

## Critique-Enhanced Generation

When `USE_CRITIQUE_GENERATION=true` (default), the service uses an advanced workflow:

1. **Generate** initial diagram from description
2. **Critique** the rendered image using AI vision analysis
3. **Retry** critique up to `CRITIQUE_MAX_ATTEMPTS` times for better feedback
4. **Adjust** and re-render if improvements are suggested
5. **Return** the improved diagram (or original if no improvements needed)

**Quality vs Speed Trade-offs:**
- `CRITIQUE_MAX_ATTEMPTS=1` — faster, single attempt
- `CRITIQUE_MAX_ATTEMPTS=3` — balanced (default)
- `CRITIQUE_MAX_ATTEMPTS=5` — maximum quality, slower

The `critique_attempts` metadata field shows how many attempts were actually used.

---

## Endpoints

### POST `/api/v1/generate-diagram`

**Request**

```json
{
  "description": "Create a diagram showing a basic web application with an Application Load Balancer, two EC2 instances for the web servers, and an RDS database for storage. The web servers should be in a cluster named 'Web Tier'"
}
```

**Response (shape)**

```json
{
  "success": true,
  "image_data": "<base64 PNG>",
  "image_url": "/tmp/diagrams/outputs/diagram_<uuid>.png",
  "metadata": {
    "nodes_created": 7,
    "clusters_created": 1,
    "connections_made": 8,
    "generation_time": 0.42,
    "timing": {"analysis_s": 0.12, "render_s": 0.30, "total_s": 0.43},
    "analysis_method": "llm|heuristic",
    "execution_mode": "langchain",
    "critique_applied": true,
    "critique": {"done": false, "critique": "Consider adding..."},
    "critique_attempts": 2,
    "adjust_render_s": 0.08,
    "request_id": "uuid-string"
  }
}
```

**New metadata fields:**
- `execution_mode`: Which strategy was used (`original`, `langchain`, `langgraph`)
- `request_id`: Unique identifier for tracing and debugging

**Save the image (one‑liner):**

```bash
curl -s -X POST "http://localhost:8000/api/v1/generate-diagram" \
  -H "Content-Type: application/json" \
  -d '{"description": "<your text here>"}' \
| python -c "import sys,base64,json;d=json.load(sys.stdin);\
  i=d.get('image_data');\
  open('diagram.png','wb').write(base64.b64decode(i)) if i else None;\
  print('saved diagram.png' if i else d)"
```

### POST `/api/v1/assistant`

**Request**

```json
{ "message": "I want to create a diagram for a serverless application" }
```

**Response** — text or image with suggestions.

---

## Supported components

**Canonical types** used in analysis & rendering:

- **Compute:** `ec2`, `lambda`, `service`
- **Data:** `rds`, `dynamodb`, `s3`
- **Networking:** `alb`, `api_gateway`, `vpc`, `internet_gateway`
- **Integration:** `sqs`, `sns`
- **Observability & Identity:** `cloudwatch`, `cognito`

**Aliases**: `apigateway→api_gateway`, `gateway→api_gateway`, `database→rds`, `queue→sqs`, business services like `auth_service` map to `service` (label preserved).

**Clustering rules**: each node belongs to **one cluster max**; prefer **functional** groupings (e.g., *Web Tier*, *Microservices*).

---

## 🧪 Tests & quality

```bash
# Run tests
uv run pytest -q

# Run specific tests
uv run pytest tests/test_api.py -v

# Code quality
uv run ruff check
uv run ruff format

# Test with different execution modes
EXECUTION_MODE=original uv run pytest tests/test_api.py
EXECUTION_MODE=langchain uv run pytest tests/test_api.py
```

**Backward Compatibility Testing:**
```bash
# Test legacy configuration still works
USE_LANGCHAIN=true uv run pytest tests/test_api.py
USE_LANGGRAPH=true uv run pytest tests/test_api.py
```

---

## 🔍 Examples (with pictures)

These are ready‑to‑run inputs plus the expected outputs committed to the repo.

### 1) Serverless CRUD API

**Description**

> Build a serverless CRUD API. The public entrypoint is API Gateway protected by Cognito. Requests invoke a Lambda function that stores items in DynamoDB. The Lambda also publishes events to an SNS topic which fans out to an SQS queue and to a second analytics Lambda. Host static assets in S3. Send logs/metrics from API Gateway and both Lambdas to CloudWatch.

**cURL**

```bash
curl -s -X POST "http://localhost:8000/api/v1/generate-diagram" \
  -H "Content-Type: application/json" \
  -d '{"description": "Build a serverless CRUD API. The public entrypoint is API Gateway protected by Cognito. Requests invoke a Lambda function that stores items in DynamoDB. The Lambda also publishes events to an SNS topic which fans out to an SQS queue and to a second analytics Lambda. Host static assets in S3. Send logs/metrics from API Gateway and both Lambdas to CloudWatch"}' \
| python -c "import sys,base64,json;d=json.load(sys.stdin);i=d.get('image_data');open('serverless_crud.png','wb').write(base64.b64decode(i)) if i else None;print('saved serverless_crud.png' if i else d)"
```

**Output**

<img width="720" height="603" alt="diagram5" src="https://github.com/user-attachments/assets/06a5147b-a806-4ae9-8cfe-cfe0797307fb" />



---

### 2) Event‑Driven File Processing Pipeline

**Description**

> Create an event‑driven file processing pipeline. S3 uploads trigger an event that enqueues a message to SQS. Lambda workers inside a VPC Processing VPC consume from SQS and write metadata to DynamoDB and records to an RDS database. Send notifications via SNS. Monitor Lambdas and DBs with CloudWatch.

**cURL**

```bash
curl -s -X POST "http://localhost:8000/api/v1/generate-diagram" \
  -H "Content-Type: application/json" \
  -d '{"description": "Create an event‑driven file processing pipeline. S3 uploads trigger an event that enqueues a message to SQS. Lambda workers inside a VPC Processing VPC consume from SQS and write metadata to DynamoDB and records to an RDS database. Send notifications via SNS. Monitor Lambdas and DBs with CloudWatch."}' \
| python -c "import sys,base64,json;d=json.load(sys.stdin);i=d.get('image_data');open('file_pipeline.png','wb').write(base64.b64decode(i)) if i else None;print('saved file_pipeline.png' if i else d)"
```

**Output**

<img width="460" height="402" alt="diagram6" src="https://github.com/user-attachments/assets/46ca0ca0-cdaa-485c-97a3-f77f84e99b53" />



---

### 3) Small Web Shop Architecture

**Description**

> Build a small web shop inside a VPC Prod VPC. Place an ALB in front of two EC2 instances grouped as Web Tier to serve the storefront. Use an API Gateway for the public API that routes to a backend service inside the VPC. Use Cognito for user authentication (integrated with API Gateway). Store transactional data in RDS and shopping carts in DynamoDB. Host images in S3. Send logs and metrics from the ALB, EC2 instances, API Gateway, and the backend service to CloudWatch.

**cURL**

```bash
curl -s -X POST "http://localhost:8000/api/v1/generate-diagram" \
  -H "Content-Type: application/json" \
  -d '{"description": "Build a small web shop inside a VPC Prod VPC. Place an ALB in front of two EC2 instances grouped as Web Tier to serve the storefront. Use an API Gateway for the public API that routes to a backend service inside the VPC. Use Cognito for user authentication (integrated with API Gateway). Store transactional data in RDS and shopping carts in DynamoDB. Host images in S3. Send logs and metrics from the ALB, EC2 instances, API Gateway, and the backend service to CloudWatch."}' \
| python -c "import sys,base64,json;d=json.load(sys.stdin);i=d.get('image_data');open('web_shop.png','wb').write(base64.b64decode(i)) if i else None;print('saved web_shop.png' if i else d)"
```

**Output**

<img width="450" height="650" alt="diagram7" src="https://github.com/user-attachments/assets/e316755a-5e8d-424a-a37e-4f037e3181c4" />



---

## 🏗️ Architecture

### **Three Implementation Strategies**

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Original Python   │    │     LangChain       │    │     LangGraph       │
│                     │    │                     │    │                     │
│ • Direct LLM calls  │    │ • Chain-based       │    │ • Workflow-based    │
│ • Fastest execution │    │ • Enhanced features │    │ • Most advanced     │
│ • Most reliable     │    │ • Structured output │    │ • State management  │
│ • Always available  │    │ • Better debugging  │    │ • Complex workflows │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

### **Error Handling Architecture**

```
┌──────────────────┐
│   Request        │
└────────┬─────────┘
         │
┌────────▼─────────┐
│  Strategy        │
│  Factory         │ ← Selects best implementation
└────────┬─────────┘
         │
┌────────▼─────────┐
│  Error Handler   │ ← Unified retry logic
│  with Retries    │
└────────┬─────────┘
         │
┌────────▼─────────┐
│  Fallback        │ ← Alternative strategies
│  Chain           │
└────────┬─────────┘
         │
┌────────▼─────────┐
│  Response        │
└──────────────────┘
```

---

## 🚀 Migration Guide

### **From Legacy Configuration**

**Old `.env` (still works)**:
```bash
USE_LANGCHAIN=true
LANGCHAIN_FALLBACK=true
```

**New `.env` (recommended)**:
```bash
EXECUTION_MODE=langchain
ENABLE_FALLBACKS=true
```

### **Gradual Migration Strategy**

1. **Phase 1**: Add new settings alongside old ones
2. **Phase 2**: Test with `EXECUTION_MODE=auto`
3. **Phase 3**: Remove old settings when ready

> **Zero Downtime**: Old configurations continue working with deprecation warnings.

---

## Output artifacts

- PNG images are saved to: `TMP_DIR/outputs/diagram_<uuid>.png`
- DOT sources are also saved alongside images for reproducibility.
- Old files are auto‑pruned (files older than 24h; keep latest \~50).
- **Request IDs** are included in filenames for tracing.

---

## 📝 Notes & limitations

- Uses a **canonical supported components** set of AWS nodes to keep results consistent and readable.
- When an unsupported service is mentioned, it maps to a close canonical type (e.g., "database" → `rds`, business services → `service`).
- Each node can belong to **one cluster** at most; we prioritize **functional** groupings.
- **Critique generation** analyzes diagrams and applies improvements with configurable retry attempts for higher success rates.
- **Multiple execution strategies** provide redundancy and enhanced capabilities.
- **Automatic fallbacks** ensure service reliability even when individual components fail.
- **Error handling** is unified across all implementations with consistent retry behavior.
- If all strategies fail, a **heuristic fallback** still renders a sensible diagram.
- Layout is automatic; complex meshes may need manual post‑tweaks.

---

## 🔧 Troubleshooting

### **Common Issues**

**Service fails to start:**
```bash
# Check configuration
uv run python -c "from app.core.config import Settings; print(Settings())"

# Test with minimal config
MOCK_LLM=true uv run uvicorn app.api.main:app --reload
```

**Strategy not working:**
```bash
# Force specific strategy
EXECUTION_MODE=original uv run uvicorn app.api.main:app --reload

# Disable fallbacks to see exact errors
ENABLE_FALLBACKS=false uv run uvicorn app.api.main:app --reload
```

**Monitoring issues:**
```bash
# Test without monitoring
LANGSMITH_ENABLED=false uv run uvicorn app.api.main:app --reload
```

### **Debug Mode**
```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
uv run uvicorn app.api.main:app --reload --log-level debug
```

---

## License

MIT

---

## 🎯 Roadmap

- [ ] **Additional AI Providers**: Anthropic Claude, Azure OpenAI support
- [ ] **Enhanced Monitoring**: Prometheus metrics, Grafana dashboards  
- [ ] **Caching Layer**: Redis-based caching for improved performance
- [ ] **Batch Processing**: Generate multiple diagrams in a single request
- [ ] **Custom Templates**: User-defined diagram templates and styles