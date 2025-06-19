import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, Text
from pgvector.sqlalchemy import Vector

DATABASE_URL = os.getenv("DATABASE_URL_ASYNC")

Base = declarative_base()

class WikiParagraph(Base):
    __tablename__ = "wiki_paragraphs"

    id = Column(Integer, primary_key=True)
    ru_wiki_pageid = Column(Integer, index=True)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=False)

async def create_tables():
    engine = create_async_engine(DATABASE_URL, echo=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(create_tables())