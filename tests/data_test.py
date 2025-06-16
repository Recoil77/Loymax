import aiofiles
import asyncio
import json
import hashlib

INPUT_PATH = "data/RuBQ_2.0_paragraphs.json"

BAD_CHARS = {
    "\u200e", "\u200f", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
    "�", "\x00", "\ufffd"
}

def text_hash(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()

async def main():
    async with aiofiles.open(INPUT_PATH, "r", encoding="utf-8") as f:
        content = await f.read()
        data = json.loads(content)

    hashes = set()
    uids = set()
    duplicate_count = 0
    duplicate_uid_count = 0
    empty_count = 0
    bad_char_count = 0

    text_lengths = []

    for item in data:
        text = item.get("text", "").strip()
        uid = item.get("uid")
        h = text_hash(text)

        # Статистика длин
        text_lengths.append(len(text))

        # Пустой text
        if not text:
            empty_count += 1

        # Дубли по text
        if h in hashes:
            duplicate_count += 1
        hashes.add(h)

        # Дубли по uid
        if uid in uids:
            duplicate_uid_count += 1
        uids.add(uid)

        # Битые символы
        if any(bad in text for bad in BAD_CHARS):
            bad_char_count += 1

    min_len = min(text_lengths) if text_lengths else 0
    max_len = max(text_lengths) if text_lengths else 0
    avg_len = sum(text_lengths) / len(text_lengths) if text_lengths else 0

    print("=== Статистика по датасету ===")
    print(f"Всего записей: {len(data)}")
    print(f"Пустых text: {empty_count}")
    print(f"Дубликатов по text (hash): {duplicate_count}")
    print(f"Дубликатов по uid: {duplicate_uid_count}")
    print(f"С битым символом: {bad_char_count}")
    print(f"Мин. длина text: {min_len}")
    print(f"Макс. длина text: {max_len}")
    print(f"Средняя длина text: {avg_len:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
