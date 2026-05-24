"""
retrieval/chunking.py
=====================
문서(section 리스트)를 token 기반 chunk로 나눕니다.
"""

import tiktoken


def get_tokenizer():
    return tiktoken.get_encoding("cl100k_base")


def chunk_document(doc: dict, chunk_size: int = 400, overlap: int = 80) -> list[dict]:
    """
    한 문서의 section 텍스트를 chunk로 분할합니다.
    section 경계를 먼저 존중하고, section이 chunk_size를 초과하면 sliding window로 자릅니다.
    """
    enc = get_tokenizer()
    chunks = []
    chunk_idx = 0

    for section in doc.get("sections", []):
        section_title = section["section_title"]
        text = section["text"]
        tokens = enc.encode(text)

        if len(tokens) <= chunk_size:
            chunks.append({
                "chunk_id": f"{doc['doc_id']}_c{chunk_idx:03d}",
                "doc_id": doc["doc_id"],
                "section_title": section_title,
                "text": text,
                "token_count": len(tokens),
            })
            chunk_idx += 1
        else:
            # sliding window
            start = 0
            while start < len(tokens):
                end = min(start + chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = enc.decode(chunk_tokens)
                chunks.append({
                    "chunk_id": f"{doc['doc_id']}_c{chunk_idx:03d}",
                    "doc_id": doc["doc_id"],
                    "section_title": section_title,
                    "text": chunk_text,
                    "token_count": len(chunk_tokens),
                })
                chunk_idx += 1
                if end == len(tokens):
                    break
                start += chunk_size - overlap

    return chunks


def chunk_all_documents(docs: list[dict], chunk_size: int, overlap: int) -> dict[str, list[dict]]:
    """doc_id → chunks 매핑을 반환합니다."""
    return {doc["doc_id"]: chunk_document(doc, chunk_size, overlap) for doc in docs}
