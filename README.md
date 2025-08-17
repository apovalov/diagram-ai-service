# Diagram AI Service

Generate AWS architecture diagrams from natural‑language descriptions using an async FastAPI service, an agent powered by Gemini, and the `diagrams` Python package.

---

## What it does

- **/api/v1/generate-diagram** — turns a plain‑text description into a PNG diagram (base64 in the response) and saves a copy under `/tmp/diagrams/outputs`.
- **/api/v1/assistant** — bonus assistant endpoint that detects intent and (when helpful) triggers diagram generation.
- **Critique-enhanced generation** — optionally analyzes generated diagrams and applies improvements for better quality (with configurable retry attempts).
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

# Option A: OpenAI with critique (best quality) - DEFAULT
# OPENAI_API_KEY=your_key
# USE_CRITIQUE_GENERATION=true
# CRITIQUE_MAX_ATTEMPTS=3

# Option B: OpenAI without critique (faster)
# OPENAI_API_KEY=your_key
# USE_CRITIQUE_GENERATION=false

# Option C: Gemini rollback (if needed)
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=your_key

# Option D: local dev/testing
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
- Ensure `.env` contains `OPENAI_API_KEY`.

---

## Configuration (.env)

| Key                      | Default            | Description                                                           |
| ------------------------ | ------------------ | --------------------------------------------------------------------- |
| `LLM_PROVIDER`           | `openai`           | LLM provider: "openai" or "gemini"                                   |
| `OPENAI_API_KEY`         | —                  | API key for OpenAI (default provider)                                |
| `OPENAI_MODEL`           | `gpt-4o-mini`      | OpenAI model name                                                     |
| `LLM_TIMEOUT`            | `60`               | Request timeout in seconds for LLM calls                             |
| `LLM_TEMPERATURE`        | `0.1`              | Temperature for LLM generation (0.0-2.0, lower = more deterministic) |
| `GEMINI_API_KEY`         | —                  | API key for Gemini (rollback option)                                 |
| `GEMINI_MODEL`           | `gemini-2.5-flash` | Gemini model name                                                     |
| `USE_CRITIQUE_GENERATION`| `true`             | Enable critique-enhanced diagram generation for improved quality       |
| `CRITIQUE_MAX_ATTEMPTS`  | `3`                | Maximum critique attempts for better quality (1-5)                    |
| `MOCK_LLM`               | `false`            | If `true`, use deterministic mock analysis (no external calls)        |
| `TMP_DIR`                | `/tmp/diagrams`    | Where images/DOT files are written                                    |
| `USE_VERTEX_AI`          | `false`            | Use Vertex AI (needs `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`) |

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
    "critique_applied": true,
    "critique": {"done": false, "critique": "Consider adding..."},
    "critique_attempts": 2,
    "adjust_render_s": 0.08
  }
}
```

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
uv run pytest -q
uv run ruff check
uv run ruff format
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

## Output artifacts

- PNG images are saved to: `TMP_DIR/outputs/diagram_<uuid>.png`
- DOT sources are also saved alongside images for reproducibility.
- Old files are auto‑pruned (files older than 24h; keep latest \~50).

---

## 📝 Notes & limitations

- Uses a **canonical supported components** set of AWS nodes to keep results consistent and readable.
- When an unsupported service is mentioned, it maps to a close canonical type (e.g., "database" → `rds`, business services → `service`).
- Each node can belong to **one cluster** at most; we prioritize **functional** groupings.
- **Critique generation** analyzes diagrams and applies improvements with configurable retry attempts for higher success rates.
- **LLM timeouts** are configurable; adjust `LLM_TIMEOUT` based on your network and complexity needs.
- **Critique retries** improve quality but increase processing time; tune `CRITIQUE_MAX_ATTEMPTS` for your use case.
- If the LLM is unavailable, a **heuristic fallback** still renders a sensible diagram.
- Layout is automatic; complex meshes may need manual post‑tweaks.

---

## License

MIT

