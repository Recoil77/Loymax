import os, asyncio, aiohttp, numpy as np
from dotenv import load_dotenv
from pymilvus import connections, Collection

load_dotenv()
EMBEDDING_TEXT_URL = os.getenv("EMBEDDING_TEXT_URL", "http://192.168.168.10:8500/embedding_text")
EMBEDDING_URL      = os.getenv("EMBEDDING_URL",      "http://192.168.168.10:8500/embedding")
MILVUS_HOST        = os.getenv("MILVUS_HOST",        "192.168.168.11")
MILVUS_PORT        = os.getenv("MILVUS_PORT",        "19530")
COLLECTION_NAME    = os.getenv("COLLECTION_NAME",    "wiki_paragraphs")
RERANK_BGE_URL     = os.getenv("RERANK_BGE_URL",     "http://192.168.168.10:8500/rerank_bge")
RERANK_SEMANTIC_URL= os.getenv("RERANK_SEMANTIC_URL","http://192.168.168.10:8500/rerank_semantic")
ASSEMBLE_DOCUMENT  = os.getenv("ASSEMBLE_DOCUMENT",  "http://192.168.168.10:8500/assemble_document")
MAX_LLM_BLOCKS     = int(os.getenv("MAX_LLM_BLOCKS", 8))
THRESHOLD          = float(os.getenv("RERANK_THRESHOLD", 0.25))

def normalize(v: list[float]) -> list[float]:
    arr = np.asarray(v, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()

async def embed(text: str) -> list[float]:
    async with aiohttp.ClientSession() as s:
        print("[embed] → очищаем и переводим текст...")
        async with s.post(EMBEDDING_TEXT_URL, json={"text": text}) as r:
            clean = (await r.json())["embedding_text"]
        print("[embed] → получаем embedding...")
        async with s.post(EMBEDDING_URL, json={"text": clean}) as r:
            vec = (await r.json())["embedding"]
    print("[embed] → нормализация embedding")
    return normalize(vec)


def search_milvus(vec: list[float], k: int):
    print(f"[milvus] Подключение к Milvus {MILVUS_HOST}:{MILVUS_PORT} ...")
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    col = Collection(COLLECTION_NAME)
    print(f"[milvus] Загружаем коллекцию {COLLECTION_NAME} ...")
    col.load()
    print("[milvus] Параметры поиска: metric_type=IP, nprobe=32")
    params = {"metric_type": "IP", "params": {"nprobe": 32}}
    print("[milvus] Выполняем поиск ...")
    hits = col.search([vec], "embedding", params, limit=k,
                      output_fields=["id", "ru_wiki_pageid", "text", "embedding_text"], consistency_level="Strong")[0]
    print(f"[milvus] Получено кандидатов: {len(hits)}")
    # Кандидаты: id (uid), ru_wiki_pageid, text, embedding_text
    candidates = [
        {
            "id": h.entity.id,
            "text": h.entity.text,
            "embedding_text": getattr(h.entity, "embedding_text", None) or ""
        }
        for h in hits
    ]
    return hits, candidates

async def rerank_bge(question: str, answers: list[str], threshold=THRESHOLD):
    print(f"[bge_rerank] Запрос к BGE reranker для {len(answers)} кандидатов...")
    payload = {"question": question, "answers": answers, "threshold": threshold}
    async with aiohttp.ClientSession() as session:
        async with session.post(RERANK_BGE_URL, json=payload) as resp:
            resp.raise_for_status()
            result = await resp.json()
    print("[bge_rerank] Готово!")
    return result

async def rerank_semantic(question: str, candidates: list[dict], threshold=THRESHOLD):
    print(f"[llm_rerank] Запрос к LLM reranker для {len(candidates)} кандидатов...")
    payload = {"question": question, "candidates": candidates, "threshold": threshold}
    async with aiohttp.ClientSession() as session:
        async with session.post(RERANK_SEMANTIC_URL, json=payload) as resp:
            resp.raise_for_status()
            result = await resp.json()
    print("[llm_rerank] Готово!")
    return result

async def main(query: str, k: int):
    print(f"=== START RETRIEVAL: '{query}' ===")
    # 1. Векторизация запроса (embedding_text)
    vec = await embed(query)  # Твой embed уже использует "очищение" + перевод

    # 2. Поиск кандидатов в Milvus
    hits, candidates = search_milvus(vec, k)  # кандидаты теперь имеют "embedding_text" и "text"

    print(f"\nTop {k} (Milvus ANN) для: {query!r}\n")
    for h in hits:
        print(f"{h.score:.4f}  id={h.entity.id} pageid={getattr(h.entity, 'ru_wiki_pageid', '-')}\n{h.entity.text[:200]}…\n")

    # 3. BGE rerank по embedding_text
    bge_result = await rerank_bge(query, [c["embedding_text"] for c in candidates], threshold=THRESHOLD)
    top_bge = bge_result["results"][:MAX_LLM_BLOCKS]
    print("\n=== BGE top-N ===")
    for r in top_bge:
        print(f"score={r['score']:.3f}  idx={r['index']}\n{r['text'][:200]}…\n")

    # 4. Semantic LLM rerank по text (как раньше)
    top_bge_indices = [r["index"] for r in top_bge]
    semantic_candidates = [{"block_id": candidates[i]["id"], "text": candidates[i]["text"]} for i in top_bge_indices]
    semantic_result = await rerank_semantic(query, semantic_candidates, threshold=THRESHOLD)
    print("\n=== Semantic LLM Rerank ===")
    for r in semantic_result:
        text = next((c["text"] for c in semantic_candidates if c["block_id"] == r["block_id"]), "")
        print(f"score={r['score']:.3f}  id={r['block_id']}\n{text[:200]}…\n")

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "33 лучших"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    asyncio.run(main(q, k))
