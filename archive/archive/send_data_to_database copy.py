# bulk_index_milvus_single.py — однопоточный индексатор для Milvus

import os
import asyncio
import aiofiles
import json
from pathlib import Path
from dotenv import load_dotenv
import aiohttp
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, list_collections
from tqdm import tqdm
import numpy as np

# ─────────────────────── Настройки ───────────────────────
load_dotenv()
EMBEDDING_TEXT_URL = os.getenv("EMBEDDING_TEXT_URL", "http://192.168.168.10:8500/embedding_text")
EMBEDDING_URL      = os.getenv("EMBEDDING_URL",      "http://192.168.168.10:8500/embedding")
INPUT_PATH         = Path(os.getenv("INPUT_PATH",     "data/RuBQ_2.0_paragraphs.json"))
MILVUS_HOST        = os.getenv("MILVUS_HOST",        "192.168.168.11")
MILVUS_PORT        = os.getenv("MILVUS_PORT",        "19530")
COLLECTION_NAME    = os.getenv("COLLECTION_NAME",    "wiki_paragraphs")
LOG_DUPLICATES     = os.getenv("LOG_DUPLICATES",     "index_duplicates.log")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.02))
EMBEDDING_DIM      = int(os.getenv("EMBEDDING_DIM", 3072))
MIN_TEXT_LENGTH    = int(os.getenv("MIN_TEXT_LENGTH", 20))
MAX_RETRIES        = int(os.getenv("MAX_RETRIES", 3))
BASE_TIMEOUT_SEC   = int(os.getenv("BASE_TIMEOUT_SEC", 30))

def get_or_create_collection() -> Collection:
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    if COLLECTION_NAME in list_collections():
        collection = Collection(COLLECTION_NAME)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="ru_wiki_pageid", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192*2),
            FieldSchema(name="embedding_text", dtype=DataType.VARCHAR, max_length=8192*2),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        ]
        collection = Collection(COLLECTION_NAME, CollectionSchema(fields, "Wiki paragraphs (ru)"))
    if not any(idx.field_name == "embedding" for idx in collection.indexes):
        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "IP",
                "params": {"nlist": 256},
            },
        )
    collection.load()
    return collection

def normalize(vec: list[float]) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()

def uid_exists(collection: Collection, uid: int) -> bool:
    try:
        res = collection.query(expr=f"id == {uid}", output_fields=["id"])
        return bool(res)
    except Exception:
        return True

def is_duplicate_vec(collection: Collection, vector: list[float], threshold: float) -> bool:
    search_params = {"metric_type": "IP", "params": {"nprobe": 32}}
    res = collection.search(
        data=[vector], anns_field="embedding", param=search_params,
        limit=1, output_fields=["id"]
    )
    return bool(res and res[0] and res[0][0].score >= (1 - threshold))

async def load_data() -> list[dict]:
    async with aiofiles.open(INPUT_PATH, "r", encoding="utf-8") as f:
        return json.loads(await f.read())[:10000]

async def fetch_with_retry(session: aiohttp.ClientSession, url: str, payload: dict,
                           max_retries: int = MAX_RETRIES, base_timeout: int = BASE_TIMEOUT_SEC):
    for attempt in range(1, max_retries + 1):
        try:
            async with session.post(url, json=payload, timeout=base_timeout) as resp:
                resp.raise_for_status()
                return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt == max_retries:
                raise
            await asyncio.sleep(2 ** (attempt - 1))

async def main_index_single():
    data       = await load_data()
    collection = get_or_create_collection()
    stats      = {k: 0 for k in ("inserted", "dup_uid", "dup_vec", "errors", "too_short")}

    async with aiofiles.open(LOG_DUPLICATES, "w", encoding="utf-8") as log_file, \
               aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=1)) as session:

        for item in tqdm(data, desc="Индексация", ncols=100):
            uid    = int(item.get("uid"))
            pageid = int(item.get("ru_wiki_pageid"))
            text   = item.get("text", "")

            if len(text) < MIN_TEXT_LENGTH:
                stats["too_short"] += 1
                await log_file.write(f"TOO_SHORT: uid={uid}, pageid={pageid}\n")
                continue

            # СНАЧАЛА ПРОВЕРКА ПО UID:
            if uid_exists(collection, uid):
                stats["dup_uid"] += 1
                await log_file.write(f"UID_DUP: uid={uid}\n")
                continue

            try:
                clean_text = await fetch_with_retry(session, EMBEDDING_TEXT_URL, {"text": text})
                clean_text = clean_text["embedding_text"]
                emb_vec    = await fetch_with_retry(session, EMBEDDING_URL, {"text": clean_text})
                emb_vec    = normalize(emb_vec["embedding"])

                # Только если такого id НЕТ, проверяем на похожий вектор:
                if is_duplicate_vec(collection, emb_vec, SIMILARITY_THRESHOLD):
                    stats["dup_vec"] += 1
                    await log_file.write(f"VEC_DUP: uid={uid}\n")
                    continue

                collection.insert([[uid], [pageid], [text], [clean_text], [emb_vec]])
                stats["inserted"] += 1

            except Exception as exc:
                stats["errors"] += 1
                await log_file.write(f"ERROR: uid={uid}, err={repr(exc)}\n")


    collection.flush(); collection.load()
    print(
        f"Всего {len(data)} → inserted {stats['inserted']} dup_uid {stats['dup_uid']} dup_vec {stats['dup_vec']} "
        f"errors {stats['errors']} short {stats['too_short']}"
    )

if __name__ == "__main__":
    asyncio.run(main_index_single())
