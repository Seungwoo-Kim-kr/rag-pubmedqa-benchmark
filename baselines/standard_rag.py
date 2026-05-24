"""
baselines/standard_rag.py
=========================
Baseline 2: Standard RAG
chunk retrieval → retrieved chunks + question → LLM answer generation.
"""

import os
import time

import faiss
from openai import OpenAI

from retrieval.retrieve import retrieve_top_k

PROMPT_TEMPLATE = """\
You are a scientific paper assistant. Use only the retrieved passages below to answer the question.
If the answer is not in the passages, say "Not found in retrieved context."

--- RETRIEVED PASSAGES ---
{passages}
--- END OF PASSAGES ---

Question: {question}

Answer:"""


def _format_passages(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Passage {i} | Section: {chunk['section_title']} | Score: {chunk['retrieval_score']:.3f}]\n{chunk['text']}")
    return "\n\n".join(parts)


def run_standard_rag(
    question: dict,
    index: faiss.Index,
    chunks: list[dict],
    config: dict,
) -> dict:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    llm_cfg = config["llm"]
    ret_cfg = config["retrieval"]
    emb_cfg = config["embedding"]

    start = time.time()

    # Step 1: retrieval
    retrieved = retrieve_top_k(
        query=question["question"],
        index=index,
        chunks=chunks,
        top_k=ret_cfg["top_k"],
        embedding_model=emb_cfg["model"],
    )

    # Step 2: answer generation
    passages_text = _format_passages(retrieved)
    prompt = PROMPT_TEMPLATE.format(passages=passages_text, question=question["question"])

    response = client.chat.completions.create(
        model=llm_cfg["model"],
        temperature=llm_cfg["temperature"],
        max_tokens=llm_cfg["max_tokens"],
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.time() - start

    answer = response.choices[0].message.content.strip()

    return {
        "question_id": question["question_id"],
        "doc_id": question["doc_id"],
        "question_type": question["question_type"],
        "baseline": "standard_rag",
        "question": question["question"],
        "answer": answer,
        "retrieved_context": [
            {"chunk_id": c["chunk_id"], "score": c["retrieval_score"], "text": c["text"]}
            for c in retrieved
        ],
        "intermediate_summary": None,
        "runtime_sec": round(elapsed, 2),
        "notes": "",
    }
