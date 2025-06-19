"""
Simple one‑shot client for the /process_query streaming endpoint.

Edit QUESTION and K below, then just run the script – никаких аргументов командной строки не требуется.

Usage:
    python test_stream_endpoint.py
"""

import asyncio

import httpx

# ---------------------------------------------------------------------------
# USER CONFIGURATION – поменяй при необходимости
# ---------------------------------------------------------------------------
API_URL: str = "http://0.0.0.0:8500/process_query"  # Твой FastAPI хост
QUESTION: str = "с каким четом Команда КФС выиграла В сезоне 1922 года ?"            # Тестовый вопрос
K: int = 8                                              # Кол‑во кандидатов из Milvus
# ---------------------------------------------------------------------------


async def main() -> None:
    """Отправляем запрос и выводим построчный стрим‑ответ."""
    payload = {"question": QUESTION, "k": K}

    print(f"→ POST {API_URL}\n→ payload: {payload}\n")

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", API_URL, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:  # пропускаем keep‑alive пустые строки
                    print(line)


if __name__ == "__main__":
    asyncio.run(main())
