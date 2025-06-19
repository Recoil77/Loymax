import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import json
import asyncio
import torch
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()
client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class EmbeddingTextRequest(BaseModel):
    text: str

class EmbeddingTextResponse(BaseModel):
    embedding_text: str

# ---------------------------------------------------------------------------
# Deterministic normalisation prompt
# ---------------------------------------------------------------------------
PROMPT = """
You are a professional, literal English translator and text normaliser.

Your task: Given any text in any language, output an English version that is as precise and faithful as possible, with no additions, omissions, or stylistic rephrasing.

**Rules:**
1. Always translate the input into clear, natural, literal English. Do not paraphrase or summarize.
2. Retain all original facts, names, numbers, dates, and word order wherever possible.
3. Remove markdown, HTML, links, and all special formatting. Output only plain English text.
4. Do not add explanations, synonyms, or extra commentary.
5. Output pure UTF-8 English text, with single spaces between words and no leading or trailing spaces or newlines.
6. If the input is already in English, simply clean it up as above, but do not change any meaning.

**Respond only as a JSON object in this format:**
{{"embedding_text": "<the precise English translation here>"}}

Text to translate and normalise (between the lines):
----------------------------------------
{input_text}
----------------------------------------
"""


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.post("/embedding_text", response_model=EmbeddingTextResponse)
async def embedding_text_endpoint(req: EmbeddingTextRequest):
    """Normalise input text for stable embeddings."""
    try:
       

        chat_response = await client.chat.completions.create(
            model="gpt-4.1-2025-04-14",  # ↔ use your preferred deterministic model ID
            messages=[
                {"role": "system", "content": "You are an expert English normaliser for embeddings."},
                {"role": "user", "content": PROMPT.format(input_text=req.text)},
            ],
            max_tokens=2048,
            temperature=0.0,  # ← remove stochasticity
            top_p=0.0,        # ← even stricter determinism
            response_format={"type": "json_object"},
        )
        result = chat_response.choices[0].message.content
        result_json = json.loads(result)
        return EmbeddingTextResponse(**result_json)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: list

@app.post("/embedding", response_model=EmbeddingResponse)
async def embedding_endpoint(req: EmbeddingRequest):
    try:
        response = await openai.AsyncOpenAI().embeddings.create(
            model="text-embedding-3-large",  # или твоя модель
            input=req.text
        )
        embedding = response.data[0].embedding
        return EmbeddingResponse(embedding=embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from typing import List


def _add_prefixes(query: str, docs: List[str]) -> tuple[str, List[str]]:
    """Return query/document strings with the prefixes expected by BGE‑reranker."""
    q_prefixed = f"query: {query}"
    d_prefixed = [f"passage:: {t}" for t in docs]
    return q_prefixed, d_prefixed

# --------------------------------------------------------------------------------------
# Re‑ranker class
# --------------------------------------------------------------------------------------

class BGERerankFunction:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        # Dynamic device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        # Thread pool for parallel batch processing
        self.executor = ThreadPoolExecutor()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_batch(self, query: str, docs: List[str]) -> List[float]:
        """Tokenize & infer a batch → return *logits* (no sigmoid)."""
        # 1) Prefixes
        q, d = _add_prefixes(query, docs)
        # 2) Tokenize — truncate *only second* sequence
        inputs = self.tokenizer(
            [q] * len(d),
            d,
            padding=True,
            truncation="only_second",
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        # 3) Forward pass
        with torch.inference_mode():
            logits = self.model(**inputs).logits.squeeze(-1)
        return logits.float().cpu().tolist()

    # ------------------------------------------------------------------
    # Public async callable
    # ------------------------------------------------------------------

    async def __call__(
        self, query: str, docs: List[str], batch_size: int = 8
    ) -> List[float]:
        tasks = []
        loop = asyncio.get_running_loop()
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            tasks.append(
                loop.run_in_executor(self.executor, self._score_batch, query, batch)
            )
        # Gather & flatten
        results = await asyncio.gather(*tasks)
        scores = [s for batch in results for s in batch]
        return scores

# Singleton instance
ger_reranker = BGERerankFunction()

# --------------------------------------------------------------------------------------
# Pydantic models
# --------------------------------------------------------------------------------------

class RerankRequest(BaseModel):
    question: str
    answers: List[str]
    threshold: float = 0.0  # сырые logits > 0 ≈ «скорее релевантно»

class RerankResult(BaseModel):
    index: int
    score: float
    text: str

# --------------------------------------------------------------------------------------
# FastAPI endpoint
# --------------------------------------------------------------------------------------

@app.post("/rerank_bge", response_model=dict)
async def rerank_bge_endpoint(req: RerankRequest):
    try:
        # 1) Получаем оценки (logits)
        scores = await ger_reranker(req.question, req.answers)

        # 2) Фильтруем по порогу и собираем результаты
        filtered = [
            RerankResult(index=i, score=score, text=req.answers[i])
            for i, score in enumerate(scores)
            if score >= req.threshold
        ]
        # 3) Сортируем по убыванию
        sorted_results = sorted(filtered, key=lambda r: r.score, reverse=True)
        return {"results": [r.dict() for r in sorted_results]}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))




class RerankBlockCandidate(BaseModel):
    block_id: int
    text: str

class RerankSemanticV5Request(BaseModel):
    question: str
    candidates: list[RerankBlockCandidate]
    threshold: float = 0.25

class RerankBlockCandidate(BaseModel):
    block_id: int
    text: str


class RerankMinimalResult(BaseModel):
    block_id: int
    score: float


# Configuration
MAX_CONCURRENT_RERANK = 8
semaphore = asyncio.Semaphore(MAX_CONCURRENT_RERANK)

def make_system_prompt(threshold: float) -> str:
    return (
        "You are a semantic relevance assistant.\n"
        "Your task is to evaluate how well a candidate text fragment answers or supports the given user question.\n"
        "Return a JSON object with a single field 'score' between 0.0 and 1.0.\n"
        "If the score is below the threshold (" + str(threshold) + "), return exactly {\"score\": 0.0}.\n"
        "Do not include explanations or extra fields."
    )

RERANKER_USER_PROMPT = (
    "Question:\n{question}\n\n"
    "Candidate Text:\n{candidate_text}"
)


@app.post("/rerank_semantic", response_model=list[RerankMinimalResult])
async def rerank_semantic_v5(request: RerankSemanticV5Request):
    system_prompt = make_system_prompt(request.threshold)

    async def score_candidate(candidate: RerankBlockCandidate) -> dict:
        user_prompt = RERANKER_USER_PROMPT.format(
            question=request.question,
            candidate_text=candidate.text
        )
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model="gpt-4.1-2025-04-14",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0
                )
            content = response.choices[0].message.content.strip()
            parsed = json.loads(content)
            score = float(parsed.get("score", 0.0))
            if not (0.0 <= score <= 1.0):
                score = 0.0
        except Exception as e:
            print(f"❌ Error scoring block {candidate.block_id}: {e}")
            score = 0.0
        return {"block_id": candidate.block_id, "score": score}

    # Score all candidates with concurrency control
    results = await asyncio.gather(*(score_candidate(c) for c in request.candidates))
    # Filter and sort
    filtered = [r for r in results if r["score"] >= request.threshold]
    sorted_results = sorted(filtered, key=lambda r: r["score"], reverse=True)
    return sorted_results


ASSEMBLE_DOCUMENT_PROMPT = """You are an assistant that assembles a relevant document from a list of text fragments (chunks).

Your task:
- Focus strictly on the provided question.
- Use only the information contained in the provided chunks.
- Remove duplicates and repeated ideas.
- Merge overlapping or similar content into a single clear explanation.
- Do not invent new facts or add external knowledge.
- Do not copy sentences verbatim multiple times — if similar sentences appear in several chunks, summarize or rephrase to avoid repetition.

Write clearly, concisely, and naturally.

Structure:
- Start with the key information that directly answers the question.
- Follow with supporting details, logically grouped.
- If the information includes examples, lists, or structured data (such as prices, terms, names, dates, times or timelines), preserve these formats.
- Do not add extra headings or formatting unless explicitly present in the chunks.

Your output must be the assembled document text only. Do not include comments or explanations about your process.
Return your answer as a valid JSON object with key "assembled_document".
""".strip()

class AssembleDocumentRequest(BaseModel):
    question: str
    chunks: list[str]

class AssembleDocumentResponse(BaseModel):
    assembled_document: str



@app.post("/assemble_document", response_model=AssembleDocumentResponse)
async def assemble_document(request: AssembleDocumentRequest):
    

    user_prompt = f"""User Question:\n{request.question}\n\nRetrieved Chunks:\n""" + \
                  "\n".join([f"{idx+1}. {chunk}" for idx, chunk in enumerate(request.chunks)])

    messages = [
        {"role": "system", "content": ASSEMBLE_DOCUMENT_PROMPT.strip()},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-2025-04-14", #   "o3-mini-2025-01-31" "o3-2025-04-16"   "gpt-4.1-nano-2025-04-14" "gpt-4.1-mini-2025-04-14" 
            messages=messages,
            temperature=0.2,
        )
        result = json.loads(response.choices[0].message.content)
        return  result

    except Exception as e:
        return {"assembled_document": f"Error: {str(e)}"}


import os
import asyncio
from typing import List, Dict, AsyncIterator

import aiohttp
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pymilvus import connections, Collection


class QueryRequest(BaseModel):
    question: str
    k: int = 8



# ---------------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------------
EMBEDDING_TEXT_URL = os.getenv("EMBEDDING_TEXT_URL", "http://192.168.168.10:8500/embedding_text")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://192.168.168.10:8500/embedding")
MILVUS_HOST = os.getenv("MILVUS_HOST", "192.168.168.11")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "wiki_paragraphs")
RERANK_BGE_URL = os.getenv("RERANK_BGE_URL", "http://192.168.168.10:8500/rerank_bge")
RERANK_SEMANTIC_URL = os.getenv("RERANK_SEMANTIC_URL", "http://192.168.168.10:8500/rerank_semantic")
ASSEMBLE_DOCUMENT_URL = os.getenv("ASSEMBLE_DOCUMENT", "http://192.168.168.10:8500/assemble_document")
MAX_LLM_BLOCKS = int(os.getenv("MAX_LLM_BLOCKS", 8))
THRESHOLD = float(os.getenv("RERANK_THRESHOLD", 0.1))

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def normalize(v: List[float]) -> List[float]:
    arr = np.asarray(v, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()


async def get_embedding_text(text: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.post(EMBEDDING_TEXT_URL, json={"text": text}) as resp:
            resp.raise_for_status()
            return (await resp.json())["embedding_text"]


async def embed(text: str) -> List[float]:
    async with aiohttp.ClientSession() as session:
        # 1. Clean / translate text
        async with session.post(EMBEDDING_TEXT_URL, json={"text": text}) as resp:
            resp.raise_for_status()
            clean = (await resp.json())["embedding_text"]

        # 2. Get embedding
        async with session.post(EMBEDDING_URL, json={"text": clean}) as resp:
            resp.raise_for_status()
            vec = (await resp.json())["embedding"]

    return normalize(vec)


def search_milvus(vec: List[float], k: int):
    """Blocking call → executed in threadpool via asyncio.to_thread"""
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    col = Collection(COLLECTION_NAME)
    col.load()

    params = {"metric_type": "IP", "params": {"nprobe": 32}}
    hits = col.search([vec], "embedding", params, limit=k,
                      output_fields=["id", "ru_wiki_pageid", "text", "embedding_text"],
                      consistency_level="Strong")[0]

    candidates = [
        {
            "id": h.entity.id,
            "text": h.entity.text,
            "embedding_text": getattr(h.entity, "embedding_text", "")
        }
        for h in hits
    ]
    return candidates


async def rerank_bge(question: str, answers: List[str], threshold: float = THRESHOLD):
    payload = {"question": question, "answers": answers, "threshold": threshold}
    async with aiohttp.ClientSession() as session:
        async with session.post(RERANK_BGE_URL, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def rerank_semantic(question: str, candidates: List[Dict], threshold: float = THRESHOLD):
    payload = {"question": question, "candidates": candidates, "threshold": threshold}
    async with aiohttp.ClientSession() as session:
        async with session.post(RERANK_SEMANTIC_URL, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def assemble_document(question: str, blocks: List[str]) -> str:
    payload = {"question": question, "chunks": blocks}
    async with aiohttp.ClientSession() as session:
        async with session.post(ASSEMBLE_DOCUMENT_URL, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data.get("assembled_document", "")

# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------




@app.post("/process_query")
async def process_query(request: QueryRequest):
    async def event_stream() -> AsyncIterator[str]:
        try:
            # 1️⃣ Clean question
            yield f"=== START RETRIEVAL: '{request.question}' ===\n"
            query_embedding_text = await get_embedding_text(request.question)
            yield f"[step] Очищенный текст запроса:\n{query_embedding_text[:150]}\n{'-'*40}\n"

            # 2️⃣ Vectorize
            vec = await embed(request.question)
            
            # 3️⃣ Milvus search (run in thread)
            candidates = await asyncio.to_thread(search_milvus, vec, request.k)
            yield f"\n[step] TOP-{request.k} Кандидатов из Milvus (embedding_text):\n"
            for i, c in enumerate(candidates):
                yield f"[{i}] {c['embedding_text'][:150]} ...\n"

            # 4️⃣ BGE rerank
            bge_result = await rerank_bge(query_embedding_text, [c["embedding_text"] for c in candidates], threshold=-5)
            top_bge = bge_result["results"][:MAX_LLM_BLOCKS]
            yield f"\n[step] TOP-{MAX_LLM_BLOCKS} после BGE rerank (embedding_text):\n"
            for r in top_bge:
                snippet = r['text'][:150]
                yield f"score={r['score']:.3f} idx={r['index']}: {snippet} ...\n"

            # 5️⃣ LLM semantic rerank
            top_indices = [r["index"] for r in top_bge]
            semantic_candidates = [{"block_id": candidates[i]["id"], "text": candidates[i]["text"]} for i in top_indices]
            semantic_result = await rerank_semantic(request.question, semantic_candidates, threshold=THRESHOLD)
            semantic_sorted = sorted(semantic_result, key=lambda r: r["score"], reverse=True)
            yield "\n[step] Блоки после LLM rerank (отсортировано по score):\n"
            for r in semantic_sorted:
                text = next(c["text"] for c in semantic_candidates if c["block_id"] == r["block_id"])
                yield f"score={r['score']:.3f} id={r['block_id']}\n{text[:150]} ...\n"

            # 6️⃣ Assemble document
            top_blocks = [next(c["text"] for c in semantic_candidates if c["block_id"] == r["block_id"]) for r in semantic_sorted]
            assembled = await assemble_document(request.question, top_blocks)
            yield "\n=== Финальный сгенерированный ответ ===\n"
            yield assembled + "\n"
        except Exception as exc:
            # Surface errors to client
            yield f"ERROR: {exc}\n"

    return StreamingResponse(event_stream(), media_type="text/plain")



###############################################################################
# ───────────────────────────────  SETTINGS  ──────────────────────────────── #
###############################################################################

EMBEDDING_TEXT_URL   = "http://192.168.168.10:8500/embedding_text"
EMBEDDING_URL        = "http://192.168.168.10:8500/embedding"
LOG_DUPLICATES       = "index_duplicates.log"
SIMILARITY_THRESHOLD = 0.02
EMBEDDING_DIM        = 3072
MIN_TEXT_LENGTH      = 20
MAX_RETRIES          = 3
BASE_TIMEOUT_SEC     = 30

MILVUS_HOST          = "192.168.168.10"
MILVUS_PORT          = "19530"
COLLECTION_NAME      = "wiki_paragraphs"

###############################################################################
# ──────────────────────────────  DEPENDENCIES  ───────────────────────────── #
###############################################################################

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field
import aiohttp, aiofiles, asyncio, numpy as np
from functools import partial
from typing import Any, Dict, Literal

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    list_collections,
)

###############################################################################
# ────────────────────────────  MILVUS HELPERS  ───────────────────────────── #
###############################################################################

def _get_or_create_collection() -> Collection:
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    if COLLECTION_NAME in list_collections():
        coll = Collection(COLLECTION_NAME)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="ru_wiki_pageid", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192 * 2),
            FieldSchema(name="embedding_text", dtype=DataType.VARCHAR, max_length=8192 * 2),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        ]
        coll = Collection(COLLECTION_NAME, CollectionSchema(fields, "Wiki paragraphs (ru)"))

    if not any(idx.field_name == "embedding" for idx in coll.indexes):
        coll.create_index(
            field_name="embedding",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "IP",
                "params": {"nlist": 256},
            },
        )
    coll.load()
    return coll

collection: Collection = _get_or_create_collection()
loop = asyncio.get_event_loop()

###############################################################################
# ─────────────────────────────  UTILITIES  ───────────────────────────────── #
###############################################################################

def normalize(vec: list[float]) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()

async def fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    payload: Dict[str, Any],
    max_retries: int = MAX_RETRIES,
    base_timeout: int = BASE_TIMEOUT_SEC,
):
    """POST `payload` and return `await resp.json()` with exponential back‑off."""
    for attempt in range(1, max_retries + 1):
        try:
            async with session.post(url, json=payload, timeout=base_timeout) as resp:
                resp.raise_for_status()
                return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt == max_retries:
                raise
            await asyncio.sleep(2 ** (attempt - 1))

# Thread‑pool wrappers for Milvus calls -------------------------------------------------

def _uid_exists(uid: int) -> bool:
    try:
        return bool(collection.query(expr=f"id == {uid}", output_fields=["id"]))
    except Exception:
        return True

def _is_duplicate_vec(vec: list[float]) -> bool:
    search_params = {"metric_type": "IP", "params": {"nprobe": 32}}
    res = collection.search([vec], anns_field="embedding", param=search_params, limit=1, output_fields=["id"])
    return bool(res and res[0] and res[0][0].score >= (1 - SIMILARITY_THRESHOLD))

async def _log(kind: str, uid: int, msg: str | None = None):
    async with aiofiles.open(LOG_DUPLICATES, "a", encoding="utf-8") as f:
        await f.write(f"{kind.upper()}: uid={uid}" + (f" | {msg}" if msg else "") + "\n")

###############################################################################
# ─────────────────────────────  DATA MODELS  ─────────────────────────────── #
###############################################################################

class Paragraph(BaseModel):
    uid: int = Field(..., gt=0)
    ru_wiki_pageid: int = Field(..., gt=0)
    text: str

class InsertResponse(BaseModel):
    success: bool
    reason: Literal[
        "inserted",
        "uid_duplicate",
        "vector_duplicate",
        "too_short",
        "internal_error",
    ]

###############################################################################
# ────────────────────────────────  ROUTER  ───────────────────────────────── #
###############################################################################

@app.post("/insert_item", response_model=InsertResponse)
async def insert_item(item: Paragraph):
    # ----------- 1. длина текста -----------
    if len(item.text) < MIN_TEXT_LENGTH:
        await _log("too_short", item.uid)
        return InsertResponse(success=False, reason="too_short")

    # ----------- 2. дубликат UID -----------
    if await loop.run_in_executor(None, partial(_uid_exists, item.uid)):
        await _log("uid_dup", item.uid)
        return InsertResponse(success=False, reason="uid_duplicate")

    try:
        async with aiohttp.ClientSession() as session:
            # --- 3. очищенный текст + embedding ---
            clean_json = await fetch_with_retry(session, EMBEDDING_TEXT_URL, {"text": item.text})
            clean_text = clean_json["embedding_text"]

            emb_json  = await fetch_with_retry(session, EMBEDDING_URL, {"text": clean_text})
            emb_vec   = normalize(emb_json["embedding"])

        # ----------- 4. дубликат по вектору -----------
        if await loop.run_in_executor(None, partial(_is_duplicate_vec, emb_vec)):
            await _log("vec_dup", item.uid)
            return InsertResponse(success=False, reason="vector_duplicate")

        # ----------- 5. вставка -----------
        await loop.run_in_executor(
            None,
            partial(collection.insert, [[item.uid], [item.ru_wiki_pageid], [item.text], [clean_text], [emb_vec]]),
        )
        return InsertResponse(success=True, reason="inserted")

    except Exception as exc:
        await _log("error", item.uid, repr(exc))
        raise HTTPException(status_code=500, detail="internal_error")