"""
baselines/summary_mediated_qa.py
=================================
Baseline 3: Summary-mediated QA
chunk retrieval → question-guided summary → LLM answer generation.
이 baseline이 Direct QA / Standard RAG와 어떻게 다른지를 보는 것이 핵심 실험 목표입니다.
"""

import os
import time

import faiss
from openai import OpenAI

from retrieval.retrieve import retrieve_top_k

SUMMARY_PROMPT = """\
You are a research assistant. Given the retrieved passages and a question, write a concise summary (3-5 sentences) that captures only the information most relevant to answering the question.
Do not answer the question yet — just summarize the relevant evidence.

Question: {question}

--- RETRIEVED PASSAGES ---
{passages}
--- END OF PASSAGES ---

Relevant summary:"""

ANSWER_PROMPT = """\
You are a scientific paper assistant. Use the summary of retrieved evidence below to answer the question concisely.
If the answer is not in the summary, say "Not found in retrieved context."

Summary of relevant evidence:
{summary}

Question: {question}

Answer:"""


def _format_passages(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Passage {i} | Section: {chunk['section_title']}]\n{chunk['text']}")
    return "\n\n".join(parts)


def run_summary_mediated_qa(
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

    # Step 2: question-guided summarization
    passages_text = _format_passages(retrieved)
    summary_prompt = SUMMARY_PROMPT.format(
        question=question["question"],
        passages=passages_text,
    )
    summary_response = client.chat.completions.create(
        model=llm_cfg["model"],
        temperature=llm_cfg["temperature"],
        max_tokens=256,
        messages=[{"role": "user", "content": summary_prompt}],
    )
    intermediate_summary = summary_response.choices[0].message.content.strip()

    # Step 3: answer generation from summary
    answer_prompt = ANSWER_PROMPT.format(
        summary=intermediate_summary,
        question=question["question"],
    )
    answer_response = client.chat.completions.create(
        model=llm_cfg["model"],
        temperature=llm_cfg["temperature"],
        max_tokens=llm_cfg["max_tokens"],
        messages=[{"role": "user", "content": answer_prompt}],
    )
    elapsed = time.time() - start

    answer = answer_response.choices[0].message.content.strip()

    return {
        "question_id": question["question_id"],
        "doc_id": question["doc_id"],
        "question_type": question["question_type"],
        "baseline": "summary_mediated_qa",
        "question": question["question"],
        "answer": answer,
        "retrieved_context": [
            {"chunk_id": c["chunk_id"], "score": c["retrieval_score"], "text": c["text"]}
            for c in retrieved
        ],
        "intermediate_summary": intermediate_summary,
        "runtime_sec": round(elapsed, 2),
        "notes": "",
    }
