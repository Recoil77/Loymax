import os
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, Text, text
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL_ASYNC")
print(f"DATABASE_URL: {DATABASE_URL}")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL_ASYNC не найден в окружении!")
Base = declarative_base()

class WikiParagraph(Base):
    __tablename__ = "wiki_paragraphs"

    id = Column(Integer, primary_key=True)
    ru_wiki_pageid = Column(Integer, index=True)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=False)

async def prepare_database():
    engine = create_async_engine(DATABASE_URL, echo=True)
    async with engine.begin() as conn:
        # 1. Включить расширение pgvector
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
        # 2. Создать таблицу
        await conn.run_sync(Base.metadata.create_all)
        # 3. Создать индекс на embedding (ivfflat, cosine)
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS wiki_paragraphs_embedding_idx "
            "ON wiki_paragraphs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
        ))
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(prepare_database())
