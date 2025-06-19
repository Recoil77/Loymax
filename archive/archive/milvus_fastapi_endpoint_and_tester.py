# milvus_insert_test.py
"""
Sequential test driver for the `/insert_item` endpoint defined in
`milvus_insert_api.py`.

• Читает исходный JSON (`INPUT_PATH`)
• Берёт диапазон записей --start/--end
• Посылает их по одному в API, пишет результат в консоль
• Лог дублей/ошибок дописывает в тот же `index_duplicates.log`
"""

import asyncio, aiohttp, json, argparse, aiofiles, sys
from pathlib import Path
from tqdm import tqdm

# -------------------------------------------------------------
API_URL    = "http://0.0.0.0:8500/insert_item"  # ⚠️ меняй порт/хост при необходимости
INPUT_PATH = Path("data/RuBQ_2.0_paragraphs.json")
LOG_FILE   = "index_duplicates.log"
# -------------------------------------------------------------

async def _load_slice(start: int, end: int):
    async with aiofiles.open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = await f.read()
    records = json.loads(data)
    return records[start:end]

async def run_test(start: int, end: int):
    records = await _load_slice(start, end)
    async with aiohttp.ClientSession() as session, aiofiles.open(LOG_FILE, "a", encoding="utf-8") as log:
        pbar = tqdm(total=len(records), desc="Upload", ncols=100)
        for rec in records:
            uid = rec.get("uid")
            try:
                resp = await session.post(API_URL, json=rec)
                data = await resp.json()
                if data.get("success"):
                    print(f"✔ INSERTED uid={uid}")
                else:
                    reason = data.get("reason")
                    await log.write(f"{reason.upper()}: uid={uid}\n")
                    print(f"✖ SKIP uid={uid} ({reason})")
            except Exception as exc:
                await log.write(f"ERROR: uid={uid} | {exc!r}\n")
                print(f"‼ ERROR uid={uid}: {exc}")
            pbar.update(1)
        pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test driver for /insert_item")
    parser.add_argument("--start", type=int, default=100, help="Начальный индекс (включительно)")
    parser.add_argument("--end", type=int, default=110, help="Конечный индекс (исключительно)")
    args = parser.parse_args()

    try:
        asyncio.run(run_test(args.start, args.end))
    except KeyboardInterrupt:
        sys.exit(0)
