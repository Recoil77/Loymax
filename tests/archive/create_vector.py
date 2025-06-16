import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

DATABASE_URL = "postgresql+asyncpg://postgres:Recoil_post_2002#@db-dev.fullnode.pro/loymax"

async def enable_pgvector():
    engine = create_async_engine(DATABASE_URL, echo=True)
    async with engine.begin() as conn:
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(enable_pgvector())