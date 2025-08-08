# Diagram API Service

This project is an async Python API service that generates diagrams from natural language descriptions using LLM agents and the `diagrams` package.

## Setup

### Prerequisits

**Download Graphviz:** [https://graphviz.org/download/](https://graphviz.org/download/)

### Local Development

1.  **Install Python 3.11+**
2.  **Install uv:** `pip install uv`
3.  **Install dependencies:** `uv sync`
4.  **Create a `.env` file with your Gemini API key (or enable mock mode):**

    ```bash
    cp .env.example .env
    # Option A: real LLM
    # If using real LLM
    GEMINI_API_KEY=your_key

    # Option B: local development without external calls
    MOCK_LLM=true
    ```

5.  **Run the application:** `uv run uvicorn app.api.main:app --reload`

### Docker

1.  **Build the image:** `docker-compose build`
2.  **Run the container:** `docker-compose up`

## API Usage

### Generate Diagram

- **POST** `/api/v1/generate-diagram`

**Request Body:**

```json
{
  "description": "Create a diagram showing a basic web application with an Application Load Balancer, two EC2 instances for the web servers, and an RDS database for storage. The web servers should be in a cluster named 'Web Tier'"
}
```

### Assistant

- **POST** `/api/v1/assistant`

**Request Body:**

```json
{
  "message": "I want to create a diagram for a serverless application"
}
```
