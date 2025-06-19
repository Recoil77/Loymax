#!/usr/bin/env python3
# Поиск ближайших абзацев в Milvus
# usage: python retriever_milvus.py "запрос" 5

import os, asyncio, aiohttp, numpy as np
from dotenv import load_dotenv
from pymilvus import connections, Collection
load_dotenv()

EMBEDDING_TEXT_URL = os.getenv("EMBEDDING_TEXT_URL", "http://192.168.168.10:8500/embedding_text")
EMBEDDING_URL      = os.getenv("EMBEDDING_URL",      "http://192.168.168.10:8500/embedding")
MILVUS_HOST        = os.getenv("MILVUS_HOST",        "192.168.168.11")
MILVUS_PORT        = os.getenv("MILVUS_PORT",        "19530")
COLLECTION_NAME    = os.getenv("COLLECTION_NAME",    "wiki_paragraphs")

def normalize(v: list[float]) -> list[float]:
    arr = np.asarray(v, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()

async def embed(text: str) -> list[float]:
    async with aiohttp.ClientSession() as s:
        async with s.post(EMBEDDING_TEXT_URL, json={"text": text}) as r:
            clean = (await r.json())["embedding_text"]
        async with s.post(EMBEDDING_URL, json={"text": clean}) as r:
            vec = (await r.json())["embedding"]
    return normalize(vec)

def search_milvus(vec: list[float], k: int):
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    col = Collection(COLLECTION_NAME)
    col.load()
    params = {"metric_type": "IP", "params": {"nprobe": 32}}
    return col.search([vec], "embedding", params, limit=k,
                      output_fields=["ru_wiki_pageid", "text"], consistency_level="Strong")[0]

async def main(query: str, k: int):
    vec = await embed(query)
    hits = search_milvus(vec, k)
    print(f"Top {k} для: {query!r}\\n")
    for h in hits:
        print(f"{h.score:.4f}  pageid={h.entity.ru_wiki_pageid}\\n{h.entity.text[:200]}…\\n")

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "1964 — Владимир Федотов (16)."
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    asyncio.run(main(q, k))
