# milvus_insert_test_sem8.py
"""
Robust concurrent test driver for the `/insert_item` endpoint.
• Одновременно выполняет не более 8 запросов (asyncio.Semaphore).
• Логирует и показывает прогресс так же, как оригинал.
"""

import asyncio, aiohttp, json, argparse, aiofiles, sys
from pathlib import Path
from tqdm import tqdm

# -------------------------------------------------------------
API_URL    = "http://0.0.0.0:8500/insert_item"        # <– при необходимости измените
INPUT_PATH = Path("data/RuBQ_2.0_paragraphs.json")
LOG_FILE   = "log/index_duplicates.log"
TIMEOUT    = aiohttp.ClientTimeout(total=30)
MAX_CONCURRENCY = 8                                   # размер семафора
# -------------------------------------------------------------


async def _load_slice(start: int, end: int):
    async with aiofiles.open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = await f.read()
    return json.loads(data)[start:end]


async def _insert_record(rec: dict,
                         session: aiohttp.ClientSession,
                         log,
                         sem: asyncio.Semaphore,
                         pbar: tqdm,
                         log_lock: asyncio.Lock):
    """Отправить один рекорд, соблюдая лимит concurrency."""
    uid = rec.get("uid")
    async with sem:                       # блокируем, если уже 8 активных задач
        try:
            resp = await session.post(API_URL, json=rec)
            try:
                data = await resp.json(content_type=None)  # не падаем на text/html
            except Exception:
                data = {}

            if resp.status == 200 and data.get("success"):
                #print(f"✔ INSERTED uid={uid}")
                pass

            else:
                reason = data.get("reason") or data.get("detail") or "unknown_error"
                async with log_lock:
                    await log.write(f"{reason.upper()}: uid={uid} | HTTP {resp.status}\n")
                #print(f"✖ SKIP uid={uid} – {reason} (HTTP {resp.status})")
        except Exception as exc:
            async with log_lock:
                await log.write(f"ERROR: uid={uid} | {exc!r}\n")
            print(f"‼ ERROR uid={uid}: {exc}")
        finally:
            pbar.update(1)


async def run_test(start: int, end: int):
    records = await _load_slice(start, end)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)  # ограничитель параллелизма
    log_lock = asyncio.Lock()                 # чтобы записи в лог не мешали друг другу

    connector = aiohttp.TCPConnector(limit=None)  # общий коннектор; лимитируем через семафор
    async with aiohttp.ClientSession(timeout=TIMEOUT, connector=connector) as session, \
               aiofiles.open(LOG_FILE, "a", encoding="utf-8") as log:

        pbar = tqdm(total=len(records), desc="Upload", ncols=100)

        tasks = [asyncio.create_task(
                    _insert_record(rec, session, log, sem, pbar, log_lock)
                 ) for rec in records]

        # ждём завершения всех задач
        await asyncio.gather(*tasks)

        pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Concurrent test driver for /insert_item")
    parser.add_argument("--start", type=int, default=39, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=50, help="End index (exclusive)")
    args = parser.parse_args()

    try:
        asyncio.run(run_test(args.start, args.end))
    except KeyboardInterrupt:
        sys.exit(0)
