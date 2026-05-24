"""
retrieval/retrieve.py
=====================
질문 임베딩으로 FAISS 인덱스에서 top-k chunk를 검색합니다.
"""

import os

import faiss
import numpy as np
from openai import OpenAI


def embed_query(query: str, model: str) -> np.ndarray:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.embeddings.create(model=model, input=[query])
    vec = np.array([response.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(vec)
    return vec


def retrieve_top_k(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    top_k: int,
    embedding_model: str,
) -> list[dict]:
    """질문과 가장 유사한 top-k chunk를 반환합니다."""
    query_vec = embed_query(query, embedding_model)
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = dict(chunks[idx])
        chunk["retrieval_score"] = float(score)
        results.append(chunk)

    return results
