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
ASSEMBLE_DOCUMENT  = os.getenv("ASSEMBLE_DOCUMENT","http://192.168.168.10:8500/assemble_document")
MAX_LLM_BLOCKS     = int(os.getenv("MAX_LLM_BLOCKS", 8))
THRESHOLD          = float(os.getenv("RERANK_THRESHOLD", 0.1))

def normalize(v: list[float]) -> list[float]:
    arr = np.asarray(v, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()

async def get_embedding_text(text: str) -> str:
    async with aiohttp.ClientSession() as s:
        async with s.post(EMBEDDING_TEXT_URL, json={"text": text}) as r:
            return (await r.json())["embedding_text"]


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

async def assemble_document(question: str, blocks: list[str]) -> str:
    payload = {
        "question": question,
        "chunks": blocks
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(ASSEMBLE_DOCUMENT, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
    return data.get("assembled_document", "")

# --- Основная логика ---
async def main(query: str, k: int):
    print(f"=== START RETRIEVAL: '{query}' ===")
    # 1. Получаем очищенный embedding_text для запроса
    query_embedding_text = await get_embedding_text(query)
    print(f"[step] Очищенный текст запроса:\n{query_embedding_text[:150]}\n{'-'*40}")

    # 2. Векторизация для поиска (embedding_text)
    vec = await embed(query)

    # 3. Поиск кандидатов в Milvus (embedding_text)
    hits, candidates = search_milvus(vec, k)
    print(f"\n[step] TOP-{k} Кандидатов из Milvus (embedding_text):")
    for i, c in enumerate(candidates):
        print(f"[{i}] {c['embedding_text'][:150]} ...\n")

    # 4. BGE rerank по embedding_text
    bge_result = await rerank_bge(query_embedding_text, [c["embedding_text"] for c in candidates], threshold=-5)
    top_bge = bge_result["results"][:MAX_LLM_BLOCKS]
    print(f"\n[step] TOP-{MAX_LLM_BLOCKS} после BGE rerank (embedding_text): {query_embedding_text}")
    for r in top_bge:
        text = r['text'][:150]
        print(f"score={r['score']:.3f} idx={r['index']}:\n{text} ...\n")

    # 5. LLM rerank по исходному вопросу и по text
    top_bge_indices = [r["index"] for r in top_bge]
    semantic_candidates = [{"block_id": candidates[i]["id"], "text": candidates[i]["text"]} for i in top_bge_indices]
    semantic_result = await rerank_semantic(query, semantic_candidates, threshold=THRESHOLD)

    # Сортируем блоки по score по убыванию
    sorted_semantic = sorted(semantic_result, key=lambda r: r["score"], reverse=True)

    print(f"\n[step] Блоки после LLM rerank (отсортировано по score):")
    for r in sorted_semantic:
        text = next((c["text"] for c in semantic_candidates if c["block_id"] == r["block_id"]), "")
        print(f"score={r['score']:.3f}  id={r['block_id']}\n{text[:150]} ...\n")

    # Для assemble_document можешь использовать отсортированный список (если хочешь)
    top_blocks = [next((c["text"] for c in semantic_candidates if c["block_id"] == r["block_id"]), "")
                for r in sorted_semantic]
    print(f"\n[step] TOP блоки для assemble_document (text):")
    for i, t in enumerate(top_blocks):
        print(f"[{i}]:\n{t[:150]} ...\n")

    assembled = await assemble_document(query, top_blocks)
    print("\n=== Финальный сгенерированный ответ ===")
    print(assembled)

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "1964 — Владимир Федотов (16)."
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    asyncio.run(main(q, k))
