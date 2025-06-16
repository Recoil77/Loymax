# Loymax Milvus Pipeline

This project contains a collection of scripts for building and querying a Milvus
vector database using OpenAI models. The repository is split into two main
parts:

* **indexing/** – tools for populating Milvus with paragraph data.
* **tests/** – helper scripts for preparing a Postgres/Milvus instance and for
  verifying embedding endpoints.

## Indexing

`send_data_to_database.py` performs asynchronous bulk indexing. It reads records
from `data/RuBQ_2.0_paragraphs.json`, cleans the text via an OpenAI endpoint and
stores both the original text and its embedding in Milvus. The script relies on
the following environment variables (see the file for defaults):

```
EMBEDDING_TEXT_URL  # HTTP endpoint that normalises text
EMBEDDING_URL       # Endpoint returning the embedding vector
INPUT_PATH          # Path to the JSON dataset
MILVUS_HOST/MILVUS_PORT
COLLECTION_NAME
```

Running the script creates the collection (if it does not exist) and inserts all
paragraphs. It uses `MAX_CONCURRENCY` workers to speed up indexing.

Several retrieval utilities are provided (`retriever_milvus.py` and
`retriever_milvus_v2.py`/`v3.py`). They take a query, embed it via the same
endpoints and search the Milvus collection. Additional reranking of search
results can be done via BGE or a semantic LLM service.

`main.py` exposes these functions as a FastAPI application with the following
endpoints:

```
/embedding_text      – return a cleaned version of input text
/embedding           – return the embedding vector
/rerank_bge          – rerank using a BGE model
/rerank_semantic     – rerank using an LLM
/assemble_document   – combine retrieved chunks into a document
```

Set `OPENAI_API_KEY` in the environment to use the OpenAI APIs.

## Tests and utilities

The `tests/` directory contains small scripts for creating databases and
checking dataset quality. For example, `create_database.py` initialises a
PostgreSQL table with `pgvector`, while `embedding_text_test_endpoint.py` sends
sample requests to the embedding services.

## License

This project is distributed under the terms of the GNU General Public License
version&nbsp;3. See [LICENSE](LICENSE) for the full text.
