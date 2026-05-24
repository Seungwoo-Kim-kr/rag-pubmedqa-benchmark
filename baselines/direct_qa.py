"""
baselines/direct_qa.py
======================
Baseline 1: Direct QA
문서 전체(section 합본)를 LLM에 직접 넣고 질문에 답하게 합니다.
"""

import os
import time

from openai import OpenAI

PROMPT_TEMPLATE = """\
You are a scientific paper assistant. Read the paper below and answer the question as accurately and concisely as possible.
If the answer cannot be found in the paper, say "Not found in the paper."

--- PAPER ---
{context}
--- END OF PAPER ---

Question: {question}

Answer:"""

MAX_CONTEXT_TOKENS = 12000  # gpt-4o-mini context 여유분


def _build_context(doc: dict, max_tokens: int) -> str:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    parts = []
    if doc.get("abstract"):
        parts.append(f"Abstract:\n{doc['abstract']}")

    total = len(enc.encode("\n\n".join(parts)))

    for section in doc.get("sections", []):
        section_text = f"[{section['section_title']}]\n{section['text']}"
        section_tokens = len(enc.encode(section_text))
        if total + section_tokens > max_tokens:
            break
        parts.append(section_text)
        total += section_tokens

    return "\n\n".join(parts)


def run_direct_qa(question: dict, doc: dict, config: dict) -> dict:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    llm_cfg = config["llm"]

    context = _build_context(doc, MAX_CONTEXT_TOKENS)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question["question"])

    start = time.time()
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
        "baseline": "direct_qa",
        "question": question["question"],
        "answer": answer,
        "retrieved_context": [context[:500] + "..."],  # 저장용 preview
        "intermediate_summary": None,
        "runtime_sec": round(elapsed, 2),
        "notes": "",
    }
