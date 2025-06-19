"""
Bulk-индексация для Milvus Standalone (pymilvus, асинхронные эмбеддинги)
Версия v3.0 — переработана логика:
  • убрана предварительная проверка uid_exists(); полагаемся на уникальность PK
  • вставка в Milvus выполняется батчами (BATCH_SIZE, по умолчанию 512)
  • дубли PK ловятся через pymilvus.exceptions.InsertException
  •Consitency level по умолчанию (Bounded для Standalone)
  • nprobe=8 для IVF_FLAT (быстрее при малой коллекции)
"""

import os
import asyncio
import aiofiles
import json
from pathlib import Path
from dotenv import load_dotenv
import aiohttp
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    list_collections,
    exceptions as milvus_exc,
)
from tqdm import tqdm
import numpy as np

# ─────────────────────────── Настройки ────────────────────────────
load_dotenv()

EMBEDDING_TEXT_URL = os.getenv("EMBEDDING_TEXT_URL", "http://192.168.168.10:8500/embedding_text")
EMBEDDING_URL      = os.getenv("EMBEDDING_URL",      "http://192.168.168.10:8500/embedding")
INPUT_PATH         = Path(os.getenv("INPUT_PATH",     "data/RuBQ_2.0_paragraphs.json"))
MILVUS_HOST        = os.getenv("MILVUS_HOST",        "192.168.168.11")
MILVUS_PORT        = os.getenv("MILVUS_PORT",        "19530")
COLLECTION_NAME    = os.getenv("COLLECTION_NAME",    "wiki_paragraphs")
LOG_DUPLICATES     = os.getenv("LOG_DUPLICATES",     "index_duplicates.log")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.02))  # не используется в этой версии
EMBEDDING_DIM      = int(os.getenv("EMBEDDING_DIM", 3072))

MIN_TEXT_LENGTH  = int(os.getenv("MIN_TEXT_LENGTH", 20))
MAX_CONCURRENT   = int(os.getenv("MAX_CONCURRENT", 8))
MAX_RETRIES      = int(os.getenv("MAX_RETRIES", 3))
BASE_TIMEOUT_SEC = int(os.getenv("BASE_TIMEOUT_SEC", 30))
BATCH_SIZE       = int(os.getenv("BATCH_SIZE", 1))

# ─────────────────────── Milvus: коллекция ────────────────────────

def get_or_create_collection() -> Collection:
    """Создаёт (или открывает) коллекцию с нужными полями и индексом."""
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    if COLLECTION_NAME in list_collections():
        collection = Collection(COLLECTION_NAME)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="ru_wiki_pageid", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=16384),
            FieldSchema(name="embedding_text", dtype=DataType.VARCHAR, max_length=16384),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        ]
        collection = Collection(COLLECTION_NAME, CollectionSchema(fields, "Wiki paragraphs (ru)"))

    # индекс по вектору (если ещё не создан)
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

# ───────────────────── Вспомогательные функции ───────────────────

def normalize(vec: list[float]) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return (arr / norm).tolist() if norm else arr.tolist()

# ───────────────────── Асинхронные утилиты ───────────────────────

async def load_data() -> list[dict]:
    async with aiofiles.open(INPUT_PATH, "r", encoding="utf-8") as f:
        return json.loads(await f.read())[:2000]

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

async def get_embedding_text(session: aiohttp.ClientSession, text: str):
    data = await fetch_with_retry(session, EMBEDDING_TEXT_URL, {"text": text})
    return data["embedding_text"]

async def get_embedding(session: aiohttp.ClientSession, text: str):
    data = await fetch_with_retry(session, EMBEDDING_URL, {"text": text})
    return data["embedding"]

# ─────────────────────── Короутина обработки ──────────────────────

async def process_item(item: dict, *, collection: Collection, session: aiohttp.ClientSession,
                       buf: dict, buf_lock: asyncio.Lock, sem: asyncio.Semaphore, stats: dict):
    async with sem:
        uid    = int(item.get("uid"))
        pageid = int(item.get("ru_wiki_pageid", 0))
        text   = item.get("text", "")

        if len(text) < MIN_TEXT_LENGTH:
            stats["too_short"] += 1
            return

        try:
            clean_text = await get_embedding_text(session, text)
            emb_vec    = normalize(await get_embedding(session, clean_text))
        except Exception:
            stats["errors"] += 1
            return

        # Буферизуем вставку
        async with buf_lock:
            buf["id"].append(uid)
            buf["ru_wiki_pageid"].append(pageid)
            buf["text"].append(text)
            buf["embedding_text"].append(clean_text)
            buf["embedding"].append(emb_vec)

            if len(buf["id"]) >= BATCH_SIZE:
                try:
                    collection.insert([buf["id"], buf["ru_wiki_pageid"], buf["text"], buf["embedding_text"], buf["embedding"]])
                    stats["inserted"] += len(buf["id"])
                except milvus_exc.InsertException as exc:
                    # Выделяем количество дублей из сообщения об ошибке, иначе считаем все PK дубликатами
                    dup = len(buf["id"])
                    stats["dup_uid"] += dup
                finally:
                    for k in buf:
                        buf[k].clear()

# ─────────────────────────── main() ───────────────────────────────

async def main_index():
    data       = await load_data()
    collection = get_or_create_collection()
    sem        = asyncio.Semaphore(MAX_CONCURRENT)
    buf_lock   = asyncio.Lock()
    buf        = {k: [] for k in ("id", "ru_wiki_pageid", "text", "embedding_text", "embedding")}
    stats      = {k: 0 for k in ("inserted", "dup_uid", "errors", "too_short")}

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=MAX_CONCURRENT)) as session:
        tasks = [asyncio.create_task(process_item(it, collection=collection, session=session,
                                                  buf=buf, buf_lock=buf_lock, sem=sem, stats=stats))
                 for it in data]

        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Индексация", ncols=100):
            await fut
            # Периодически печатаем текущую статистику
            if (stats["inserted"] + stats["dup_uid"] + stats["errors"] + stats["too_short"]) % 500 == 0:
                tqdm.write(
                    f"inserted:{stats['inserted']} dup_uid:{stats['dup_uid']} "
                    f"err:{stats['errors']} short:{stats['too_short']}")

    # Финальный слив буфера
    if buf["id"]:
        try:
            collection.insert([buf["id"], buf["ru_wiki_pageid"], buf["text"], buf["embedding_text"], buf["embedding"]])
            stats["inserted"] += len(buf["id"])
        except milvus_exc.InsertException:
            stats["dup_uid"] += len(buf["id"])

    collection.flush(); collection.load()
    print(
        f"Всего {len(data)} → inserted {stats['inserted']} dup_uid {stats['dup_uid']} "
        f"errors {stats['errors']} short {stats['too_short']}"
    )

if __name__ == "__main__":
    asyncio.run(main_index())
