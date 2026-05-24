"""
retrieval/embed_index.py
========================
OpenAI Embedding API로 chunk를 임베딩하고 FAISS 인덱스를 생성합니다.
GPU 불필요 — 모든 연산은 OpenAI API + CPU FAISS.
"""

import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI
from tqdm import tqdm


def get_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def embed_texts(texts: list[str], model: str, batch_size: int = 32) -> np.ndarray:
    """텍스트 리스트를 임베딩 벡터 배열로 변환합니다."""
    client = get_client()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="임베딩 생성"):
        batch = texts[i : i + batch_size]
        # 빈 텍스트 방지
        batch = [t if t.strip() else " " for t in batch]
        response = client.embeddings.create(model=model, input=batch)
        vectors = [item.embedding for item in response.data]
        all_embeddings.extend(vectors)
        time.sleep(0.1)  # rate limit 방지

    return np.array(all_embeddings, dtype=np.float32)


def build_index(chunks: list[dict], embedding_model: str, batch_size: int) -> tuple[faiss.Index, list[dict]]:
    """chunk 리스트로 FAISS 인덱스를 생성합니다."""
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, embedding_model, batch_size)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product (코사인 유사도용)

    # L2 정규화 후 Inner Product = 코사인 유사도
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    return index, chunks


def save_index(index: faiss.Index, chunks: list[dict], doc_id: str, save_dir: str):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(Path(save_dir) / f"{doc_id}.faiss"))
    with open(Path(save_dir) / f"{doc_id}_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_index(doc_id: str, save_dir: str) -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(str(Path(save_dir) / f"{doc_id}.faiss"))
    with open(Path(save_dir) / f"{doc_id}_chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks
