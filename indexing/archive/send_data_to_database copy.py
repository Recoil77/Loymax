#!/usr/bin/env python3
"""
Async bulk indexer that loads Wikipedia paragraphs to a **pgvector** column
using clean SQLAlchemy ORM calls.  Works with Python **list[float]** objects –
no manual casting, no raw SQL – because the pgvector codec is registered once
for every asyncpg connection.
"""
import os
import asyncio
import json
import aiofiles
import aiohttp
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, Text, text as sql_text, event
from pgvector.sqlalchemy import Vector
from pgvector.asyncpg import register_vector

# ─────────────────────── CONFIG ──────────────────────────────────────────────
load_dotenv()
DATABASE_URL: str = os.getenv("DATABASE_URL_ASYNC")  # postgresql+asyncpg://...
EMBEDDING_TEXT_URL = "http://192.168.168.10:8500/embedding_text"
EMBEDDING_URL = "http://192.168.168.10:8500/embedding"
INPUT_PATH = "data/RuBQ_2.0_paragraphs.json"
LOG_DUPLICATES = "index_duplicates.log"
SIMILARITY_THRESHOLD = 0.01  # cosine distance threshold for duplicate check

# ────────────────── SQLAlchemy model ─────────────────────────────────────────
Base = declarative_base()

class WikiParagraph(Base):
    __tablename__ = "wiki_paragraphs"

    id             = Column(Integer, primary_key=True)
    ru_wiki_pageid = Column(Integer, index=True)
    text           = Column(Text, nullable=False)
    embedding      = Column(Vector(1536), nullable=False)

# ─────────────────── Helper I/O ──────────────────────────────────────────────
async def load_json(path: str = INPUT_PATH):
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        return json.loads(await f.read())

async def post_json(http: aiohttp.ClientSession, url: str, payload: dict, key: str):
    async with http.post(url, json=payload) as resp:
        resp.raise_for_status()
        return (await resp.json())[key]

# ────────────────── Duplicate check ──────────────────────────────────────────
async def is_duplicate(db: AsyncSession, vec, thr: float = SIMILARITY_THRESHOLD) -> bool:
    row = (
        await db.execute(
            sql_text(
                """
                SELECT (embedding <=> :v) AS dist
                FROM wiki_paragraphs
                ORDER BY dist
                LIMIT 1
                """
            ),
            {"v": vec},
        )
    ).first()
    return bool(row and row.dist is not None and row.dist < thr)

# ─────────────────────── Main routine ────────────────────────────────────────
async def main():
    data = await load_json()

    engine = create_async_engine(DATABASE_URL, echo=False)

    # Register pgvector codec for every new asyncpg connection
    @event.listens_for(engine.sync_engine, "connect")
    def _register_vector_codec(dbapi_conn, conn_record):
        register_vector(dbapi_conn)

    AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    log = await aiofiles.open(LOG_DUPLICATES, "w", encoding="utf-8")

    async with aiohttp.ClientSession() as http, AsyncSessionLocal() as db:
        inserted = skipped = errors = 0

        for item in tqdm(data, desc="Индексация", ncols=100):
            uid     = item["uid"]
            pageid  = item["ru_wiki_pageid"]
            text    = item.get("text", "")

            try:
                # 1) embedding_text и embedding
                emb_text = await post_json(http, EMBEDDING_TEXT_URL, {"text": text}, "embedding_text")
                vec      = await post_json(http, EMBEDDING_URL,       {"text": emb_text}, "embedding")

                # 2) проверка на дубликат
                if await is_duplicate(db, vec):
                    skipped += 1
                    await log.write(f"SKIP uid={uid} pageid={pageid}\n")
                    continue

                # 3) вставка ORM‑способом
                db.add(
                    WikiParagraph(
                        id=uid,
                        ru_wiki_pageid=pageid,
                        text=text,
                        embedding=[float(x) for x in vec],
                    )
                )
                await db.commit()
                inserted += 1

            except Exception as exc:
                await db.rollback()
                errors += 1
                await log.write(f"ERROR uid={uid} pageid={pageid} err={exc!r}\n")

    await log.close()
    await engine.dispose()

    print(
        f"Готово! обработано={len(data)}, вставлено={inserted}, "
        f"дубликаты={skipped}, ошибки={errors}"
    )


if __name__ == "__main__":
    asyncio.run(main())
