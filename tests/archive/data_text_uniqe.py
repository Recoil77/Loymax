import aiofiles
import asyncio
import json
import hashlib

INPUT_PATH = "data/RuBQ_2.0_paragraphs.json"

def text_hash(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()

async def main():
    async with aiofiles.open(INPUT_PATH, "r", encoding="utf-8") as f:
        content = await f.read()
        data = json.loads(content)

    hashes = set()
    duplicates = 0

    for i, item in enumerate(data):
        h = text_hash(item.get("text", ""))
        if h in hashes:
            print(f"❌ Дубликат text (hash={h}) найден в записи {i}")
            duplicates += 1
        hashes.add(h)

    print(f"\nВсего записей: {len(data)}")
    print(f"Дубликатов по text (hash): {duplicates}")

if __name__ == "__main__":
    asyncio.run(main())
