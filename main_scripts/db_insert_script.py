# milvus_insert_test.py
"""
Robust sequential test driver for the `/insert_item` endpoint.
Improvements over the previous version:
• Gracefully handles non‑200 responses and missing keys in JSON (no more NoneType.upper()).
• Clearly prints/records HTTP status codes and Milvus‑API reasons.
• Keeps the same CLI interface:  --start / --end
"""

import asyncio, aiohttp, json, argparse, aiofiles, sys
from pathlib import Path
from tqdm import tqdm

# -------------------------------------------------------------
API_URL    = "http://0.0.0.0:8500/insert_item"  # ← change host/port if necessary
INPUT_PATH = Path("data/RuBQ_2.0_paragraphs.json")
LOG_FILE   = "log/index_duplicates.log"
TIMEOUT    = aiohttp.ClientTimeout(total=30)
# -------------------------------------------------------------

async def _load_slice(start: int, end: int):
    async with aiofiles.open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = await f.read()
    return json.loads(data)[start:end]

async def run_test(start: int, end: int):
    records = await _load_slice(start, end)
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session, aiofiles.open(LOG_FILE, "a", encoding="utf-8") as log:
        pbar = tqdm(total=len(records), desc="Upload", ncols=100)
        for rec in records:
            uid = rec.get("uid")
            try:
                resp = await session.post(API_URL, json=rec)
                try:
                    data = await resp.json(content_type=None)  # don't fail on text/html
                except Exception:
                    data = {}

                if resp.status == 200 and data.get("success"):
                    print(f"✔ INSERTED uid={uid}")
                else:
                    reason = data.get("reason") or data.get("detail") or "unknown_error"
                    await log.write(f"{reason.upper()}: uid={uid} | HTTP {resp.status}\n")
                    print(f"✖ SKIP uid={uid} – {reason} (HTTP {resp.status})")
            except Exception as exc:
                await log.write(f"ERROR: uid={uid} | {exc!r}\n")
                print(f"‼ ERROR uid={uid}: {exc}")
            pbar.update(1)
        pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test driver for /insert_item")
    parser.add_argument("--start", type=int, default=1, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=10, help="End index (exclusive)")
    args = parser.parse_args()

    try:
        asyncio.run(run_test(args.start, args.end))
    except KeyboardInterrupt:
        sys.exit(0)
