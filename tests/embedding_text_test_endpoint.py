import aiohttp
import asyncio

ENDPOINT_TEXT = "http://192.168.168.10:8500/embedding_text"
ENDPOINT_EMB = "http://192.168.168.10:8500/embedding"

async def test_embedding_pipeline(text):
    async with aiohttp.ClientSession() as session:
        # 1. Получить "embedding_text"
        payload = {"text": text}
        async with session.post(ENDPOINT_TEXT, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            print("\n[embedding_text]:")
            print(data)
            embedding_text = data["embedding_text"]
            print(embedding_text)

        # 2. Получить embedding
        payload_emb = {"text": embedding_text}
        async with session.post(ENDPOINT_EMB, json=payload_emb) as resp:
            resp.raise_for_status()
            data_emb = await resp.json()
            print("\n[embedding] — первые 5 значений:")
            print(data_emb["embedding"][:5])  # первые 5 элементов эмбеддинга

if __name__ == "__main__":
    test_text = "Это тестовый текст на русском с [markdown](https://example.com) и спецсимволами ©™!"
    asyncio.run(test_embedding_pipeline(test_text))
