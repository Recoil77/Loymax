# Bulk‑индексация для Milvus Standalone (pymilvus, асинхронные эмбеддинги)
# Файл: bulk_index_milvus.py
# v2.3 — двойная защита от дублей: по uid **до** обращения к OpenAI + существующая проверка по вектору;
#        а также EMBEDDING_DIM по умолчанию 3072 (1536 * 2).

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

# ─────────────────────────── Настройки ────────────────────────────
load_dotenv()

EMBEDDING_TEXT_URL = os.getenv("EMBEDDING_TEXT_URL", "http://192.168.168.10:8500/embedding_text")
EMBEDDING_URL      = os.getenv("EMBEDDING_URL",      "http://192.168.168.10:8500/embedding")
INPUT_PATH         = Path(os.getenv("INPUT_PATH",     "data/RuBQ_2.0_paragraphs.json"))
MILVUS_HOST        = os.getenv("MILVUS_HOST",        "192.168.168.11")
MILVUS_PORT        = os.getenv("MILVUS_PORT",        "19530")
COLLECTION_NAME    = os.getenv("COLLECTION_NAME",    "wiki_paragraphs")
LOG_DUPLICATES     = os.getenv("LOG_DUPLICATES",     "index_duplicates.log")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.02))
EMBEDDING_DIM      = int(os.getenv("EMBEDDING_DIM", 3072))  # 1536 * 2

MIN_TEXT_LENGTH  = int(os.getenv("MIN_TEXT_LENGTH", 20))
MAX_CONCURRENT   = int(os.getenv("MAX_CONCURRENT", 8))
MAX_RETRIES      = int(os.getenv("MAX_RETRIES", 3))
BASE_TIMEOUT_SEC = int(os.getenv("BASE_TIMEOUT_SEC", 30))

# ─────────────────────── Milvus: коллекция ────────────────────────

def get_or_create_collection() -> Collection:
    """Создаёт (или открывает) коллекцию с индексом IVF_FLAT/IP."""
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    if COLLECTION_NAME in list_collections():
        collection = Collection(COLLECTION_NAME)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="ru_wiki_pageid", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
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

# ───────────────────── Вспомогательные функции ───────────────────

def normalize(vec: list[float]) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()

def uid_exists(collection: Collection, uid: int) -> bool:
    """Быстрая проверка PK через `collection.get([uid])` (использует внутренний PK‑индекс)."""
    try:
        return bool(collection.get([uid]))  # пустой список, если PK не найден
    except Exception:
        return False

# ───────────────────── Асинхронные утилиты ───────────────────────

async def load_data() -> list[dict]:
    async with aiofiles.open(INPUT_PATH, "r", encoding="utf-8") as f:
        return json.loads(await f.read())[:10]

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

# ─── Milvus: поиск дублей по вектору ──────────────────────────────

def is_duplicate_vec(collection: Collection, vector: list[float], threshold: float) -> bool:
    search_params = {"metric_type": "IP", "params": {"nprobe": 32}}
    res = collection.search(
        data=[vector], anns_field="embedding", param=search_params,
        limit=1, output_fields=["id"], consistency_level="Strong"
    )
    return bool(res and res[0] and res[0][0].score >= (1 - threshold))

# ─────────────────────── Короутина обработки ──────────────────────

async def process_item(item: dict, collection: Collection, session: aiohttp.ClientSession,
                       log_file, sem: asyncio.Semaphore, stats: dict):
    async with sem:
        uid    = int(item.get("uid"))
        pageid = int(item.get("ru_wiki_pageid"))
        text   = item.get("text", "")

        # 0) минимальная длина
        if len(text) < MIN_TEXT_LENGTH:
            stats["too_short"] += 1
            await log_file.write(f"TOO_SHORT: uid={uid}, pageid={pageid}, len={len(text)}\n")
            return

        # 1) проверка дубля по uid до тяжёлых операций
        if uid_exists(collection, uid):
            stats["dup_uid"] += 1
            await log_file.write(f"UID_DUP: uid={uid} pageid={pageid}\n")
            return

        try:
            # 2) эмбеддинг
            emb_text = await get_embedding_text(session, text)
            emb      = normalize(await get_embedding(session, emb_text))

            # 3) проверка дубля по вектору
            if is_duplicate_vec(collection, emb, SIMILARITY_THRESHOLD):
                stats["dup_vec"] += 1
                await log_file.write(f"VEC_DUP: uid={uid} pageid={pageid}\n")
                return

            # 4) вставка
            collection.insert([[uid], [pageid], [text], [emb]])
            stats["inserted"] += 1

        except Exception as exc:
            stats["errors"] += 1
            await log_file.write(f"ERROR: uid={uid}, pageid={pageid}, err={repr(exc)}\n")

# ─────────────────────────── main() ───────────────────────────────

async def main_index():
    data       = await load_data()
    collection = get_or_create_collection()
    sem        = asyncio.Semaphore(MAX_CONCURRENT)
    stats      = {k: 0 for k in ("inserted", "dup_uid", "dup_vec", "errors", "too_short")}

    async with aiofiles.open(LOG_DUPLICATES, "w", encoding="utf-8") as log_file, \
               aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=MAX_CONCURRENT)) as session:

        tasks = [asyncio.create_task(process_item(item, collection, session, log_file, sem, stats)) for item in data]
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Индексация", ncols=100):
            await fut
            if sum(stats.values()) % 100 == 0:
                tqdm.write(
                    f"inserted: {stats['inserted']}  dup_uid: {stats['dup_uid']}  dup_vec: {stats['dup_vec']}  "
                    f"errors: {stats['errors']}  too_short: {stats['too_short']}"
                )

    # ── Финальный flush, чтобы PK‑индекс гарантированно обновился ──
    collection.flush()
    collection.load()  # подгружаем запечатанные сегменты, чтобы следующий запуск видел PK

    print(
        f"Всего {len(data)} → "
        f"inserted {stats['inserted']}, dup_uid {stats['dup_uid']}, dup_vec {stats['dup_vec']}, "
        f"errors {stats['errors']}, short {stats['too_short']}"
    )

if __name__ == "__main__":
    asyncio.run(main_index())
