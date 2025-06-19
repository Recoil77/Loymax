# Loymax Milvus Pipeline

Loymax is a collection of Python utilities for building a vector search service with [Milvus](https://milvus.io/) and OpenAI models. The repository contains a small FastAPI application for embeddings and a set of helper scripts for populating and querying a Milvus collection.

## Directory overview

- **indexing/** – FastAPI service exposing text normalisation, embedding and reranking endpoints. The service writes data to a Milvus instance.
- **main_scripts/** – client utilities for loading and fetching records from the service.
- **tests/** – small scripts for creating databases and checking data quality.

## Running with Docker

A `docker-compose.yml` file is provided for running Milvus together with the embedding service:

```bash
docker-compose up -d
```

This starts Milvus, MinIO, ETCD and the FastAPI service. The service listens on port `8500` by default. Environment variables for host names and ports can be adjusted in `indexing/.env` or passed directly when running Docker.

## Local development

1. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r indexing/requirements.txt
```

2. Start the FastAPI app:

```bash
python indexing/main.py
```

The app relies on several environment variables such as `MILVUS_HOST`, `EMBEDDING_URL` and `OPENAI_API_KEY`. Default values are defined inside `indexing/main.py`.

## Indexing data

Use `main_scripts/db_insert_script_concurrency.py` to insert records into Milvus. The script reads from `data/RuBQ_2.0_paragraphs.json` and sends the content to the `/insert_item` endpoint concurrently.

```bash
python main_scripts/db_insert_script_concurrency.py --start 0 --end 100
```

For a simple retrieval test, run `main_scripts/db_fetch_script.py` which streams the `/process_query` response.

## Tests

The `tests/` directory includes small utilities for verifying the embedding endpoints and for initialising a PostgreSQL database with `pgvector`.

## License

This project is distributed under the terms of the GNU General Public License version&nbsp;3. See [LICENSE](LICENSE) for the full text.
