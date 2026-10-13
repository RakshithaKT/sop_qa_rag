# SOP QA RAG

An SOP question-answering application built with FastAPI.

## Setup

Create and activate a Python virtual environment, then install:

```
pip install -e ".[dev]"
```

## Run

```
uvicorn sop_qa.main:app --reload --app-dir src
```

## API documentation

Visit http://127.0.0.1:8000/docs

## Tests

```
pytest -q
```
