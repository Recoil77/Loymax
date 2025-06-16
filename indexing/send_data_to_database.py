"""
# bulk_index_milvus_multi.py — многопоточный (8 параллельных задач) индексатор для Milvus
# -----------------------------------------------------------------------------
# Ключевые отличия от однопоточной версии:
#   • Параллельность управляется asyncio-семафором (MAX_CONCURRENCY = 8).
#   • Блокирующие операции Milvus выполняются в ThreadPoolExecutor (run_in_executor).
#   • aiohttp TCPConnector(limit=MAX_CONCURRENCY) − до 8 одновременных HTTP-запросов.
#   • tqdm-прогресс создаётся один раз и закрывается по завершении (без async-контекста).
# -----------------------------------------------------------------------------
"""

import os
import asyncio
import aiofiles
import json
from pathlib import Path
from functools import partial
from typing import Any
from dotenv import load_dotenv
import aiohttp
import numpy as np
from tqdm import tqdm
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    list_collections,
)

# ─────────────────────── Настройки ───────────────────────
load_dotenv()
EMBEDDING_TEXT_URL     = os.getenv("EMBEDDING_TEXT_URL", "http://192.168.168.10:8500/embedding_text")
EMBEDDING_URL          = os.getenv("EMBEDDING_URL",      "http://192.168.168.10:8500/embedding")
INPUT_PATH             = Path(os.getenv("INPUT_PATH",    "data/RuBQ_2.0_paragraphs.json"))
MILVUS_HOST            = os.getenv("MILVUS_HOST",        "192.168.168.11")
MILVUS_PORT            = os.getenv("MILVUS_PORT",        "19530")
COLLECTION_NAME        = os.getenv("COLLECTION_NAME",    "wiki_paragraphs")
LOG_DUPLICATES         = os.getenv("LOG_DUPLICATES",     "index_duplicates.log")
SIMILARITY_THRESHOLD   = float(os.getenv("SIMILARITY_THRESHOLD", 0.02))
EMBEDDING_DIM          = int(os.getenv("EMBEDDING_DIM", 3072))
MIN_TEXT_LENGTH        = int(os.getenv("MIN_TEXT_LENGTH", 20))
MAX_RETRIES            = int(os.getenv("MAX_RETRIES", 3))
BASE_TIMEOUT_SEC       = int(os.getenv("BASE_TIMEOUT_SEC", 30))
MAX_CONCURRENCY        = 16  # ← главная настройка параллельности

# ─────────────────────── Milvus helpers ───────────────────────

def get_or_create_collection() -> Collection:
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    if COLLECTION_NAME in list_collections():
        collection = Collection(COLLECTION_NAME)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="ru_wiki_pageid", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192 * 2),
            FieldSchema(name="embedding_text", dtype=DataType.VARCHAR, max_length=8192 * 2),
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

# ─────────────────────── Вспомогательные функции ───────────────────────

def normalize(vec: list[float]) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()

# Блокирующие вызовы Milvus в пуле потоков

def uid_exists_sync(collection: Collection, uid: int) -> bool:
    try:
        res = collection.query(expr=f"id == {uid}", output_fields=["id"])
        return bool(res)
    except Exception:
        return True

def is_duplicate_vec_sync(collection: Collection, vector: list[float], threshold: float) -> bool:
    search_params = {"metric_type": "IP", "params": {"nprobe": 32}}
    res = collection.search(
        data=[vector], anns_field="embedding", param=search_params,
        limit=1, output_fields=["id"],
    )
    return bool(res and res[0] and res[0][0].score >= (1 - threshold))

# ─────────────────────── I/O helpers ───────────────────────

async def load_data() -> list[dict]:
    async with aiofiles.open(INPUT_PATH, "r", encoding="utf-8") as f:
        return json.loads(await f.read())[:10000]

async def fetch_with_retry(session: aiohttp.ClientSession, url: str, payload: dict,
                           max_retries: int = MAX_RETRIES,
                           base_timeout: int = BASE_TIMEOUT_SEC):
    for attempt in range(1, max_retries + 1):
        try:
            async with session.post(url, json=payload, timeout=base_timeout) as resp:
                resp.raise_for_status()
                return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt == max_retries:
                raise
            await asyncio.sleep(2 ** (attempt - 1))

# ─────────────────────── Обработка одного параграфа ───────────────────────

async def process_item(item: dict,
                       collection: Collection,
                       session: aiohttp.ClientSession,
                       stats: dict[str, int],
                       log_lock: asyncio.Lock,
                       log_file: Any,
                       sem: asyncio.Semaphore,
                       pbar: tqdm):

    async with sem:
        uid    = int(item.get("uid"))
        pageid = int(item.get("ru_wiki_pageid"))
        text   = item.get("text", "")

        if len(text) < MIN_TEXT_LENGTH:
            stats["too_short"] += 1
            async with log_lock:
                await log_file.write(f"TOO_SHORT: uid={uid}, pageid={pageid}\n")
            pbar.update(1)
            return

        loop = asyncio.get_running_loop()

        exists = await loop.run_in_executor(None, partial(uid_exists_sync, collection, uid))
        if exists:
            stats["dup_uid"] += 1
            async with log_lock:
                await log_file.write(f"UID_DUP: uid={uid}\n")
            pbar.update(1)
            return

        try:
            clean_json = await fetch_with_retry(session, EMBEDDING_TEXT_URL, {"text": text})
            clean_text = clean_json["embedding_text"]

            emb_json = await fetch_with_retry(session, EMBEDDING_URL, {"text": clean_text})
            emb_vec  = normalize(emb_json["embedding"])

            dup_vec = await loop.run_in_executor(
                None,
                partial(is_duplicate_vec_sync, collection, emb_vec, SIMILARITY_THRESHOLD),
            )
            if dup_vec:
                stats["dup_vec"] += 1
                async with log_lock:
                    await log_file.write(f"VEC_DUP: uid={uid}\n")
                pbar.update(1)
                return

            await loop.run_in_executor(
                None,
                partial(collection.insert, [[uid], [pageid], [text], [clean_text], [emb_vec]]),
            )
            stats["inserted"] += 1

        except Exception as exc:
            stats["errors"] += 1
            async with log_lock:
                await log_file.write(f"ERROR: uid={uid}, err={repr(exc)}\n")

        pbar.update(1)

# ─────────────────────── Главная функция ───────────────────────

async def main_index_multi():
    data       = await load_data()
    collection = get_or_create_collection()
    stats      = {k: 0 for k in ("inserted", "dup_uid", "dup_vec", "errors", "too_short")}

    sem       = asyncio.Semaphore(MAX_CONCURRENCY)
    log_lock  = asyncio.Lock()

    async with aiofiles.open(LOG_DUPLICATES, "w", encoding="utf-8") as log_file, \
               aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=MAX_CONCURRENCY)) as session:

        pbar = tqdm(total=len(data), desc="Индексация", ncols=100)
        try:
            tasks = [
                asyncio.create_task(
                    process_item(item, collection, session, stats, log_lock, log_file, sem, pbar)
                )
                for item in data
            ]
            await asyncio.gather(*tasks)
        finally:
            pbar.close()

    collection.flush(); collection.load()

    print(
        f"Всего {len(data)} → inserted {stats['inserted']} dup_uid {stats['dup_uid']} "
        f"dup_vec {stats['dup_vec']} errors {stats['errors']} short {stats['too_short']}"
    )

if __name__ == "__main__":
    asyncio.run(main_index_multi())
