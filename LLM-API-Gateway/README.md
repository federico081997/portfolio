# LLM API Gateway

A FastAPI-based gateway for interacting with local and external large
language models through a consistent, validated API.

## Current status

The initial FastAPI service and placeholder health endpoint are implemented.

## Available endpoints

- `GET /`
- `GET /health`

## Local development

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies

```bash
python -m pip install -r requirements.txt
```

Run the development server

```bash
uvicorn app.main:app --reload
```

Open the API documentation

```bash
http://127.0.0.1:8000/docs
```

## Configure pytest

Open:

```bash
code pytest.ini
```
