"""
baselines/lightrag_runner.py
============================
Baseline 4: LightRAG (graph-based RAG)
문서를 LightRAG에 삽입하고 hybrid 모드로 질문에 답합니다.
- naive 모드: 전통적 벡터 검색 (Standard RAG와 비교 기준)
- hybrid 모드: 그래프 + 벡터 통합 (LightRAG 대표 모드)

각 문서별로 독립적인 working_dir을 사용해 인덱스를 캐시합니다.
"""

import asyncio
import os
import time
from pathlib import Path


def _get_working_dir(doc_id: str) -> str:
    base = Path("data/lightrag_indexes")
    base.mkdir(parents=True, exist_ok=True)
    return str(base / doc_id)


def _build_doc_text(doc: dict) -> str:
    parts = []
    if doc.get("abstract"):
        parts.append(doc["abstract"])
    for section in doc.get("sections", []):
        parts.append(f"[{section['section_title']}]\n{section['text']}")
    return "\n\n".join(parts)


async def _ensure_index(doc: dict, working_dir: str):
    """문서가 아직 인덱싱되지 않았으면 삽입합니다."""
    from lightrag import LightRAG
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

    done_flag = Path(working_dir) / ".indexed"
    if done_flag.exists():
        return

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
    )
    await rag.initialize_storages()
    await rag.ainsert(_build_doc_text(doc))
    await rag.finalize_storages()
    done_flag.touch()


async def _query_lightrag(question: str, working_dir: str, mode: str) -> str:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
    )
    await rag.initialize_storages()
    result = await rag.aquery(question, param=QueryParam(mode=mode))
    await rag.finalize_storages()

    return result.content if hasattr(result, "content") else str(result)


def run_lightrag(
    question: dict,
    doc: dict,
    config: dict,
    mode: str = "hybrid",
) -> dict:
    working_dir = _get_working_dir(question["doc_id"])

    start = time.time()

    async def _run():
        await _ensure_index(doc, working_dir)
        return await _query_lightrag(question["question"], working_dir, mode)

    answer = asyncio.run(_run())
    elapsed = time.time() - start

    return {
        "question_id": question["question_id"],
        "doc_id": question["doc_id"],
        "question_type": question["question_type"],
        "baseline": f"lightrag_{mode}",
        "question": question["question"],
        "answer": answer,
        "retrieved_context": [],   # LightRAG는 내부 그래프 검색이라 chunk 미노출
        "intermediate_summary": None,
        "runtime_sec": round(elapsed, 2),
        "notes": f"LightRAG mode={mode}",
    }
