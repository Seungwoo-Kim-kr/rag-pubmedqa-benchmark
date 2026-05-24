"""
run_lightrag_standalone.py
==========================
LightRAG baseline 독립 실행 스크립트.
Python 3.10+ 환경에서만 실행 가능합니다.

사용법:
    python3.10 run_lightrag_standalone.py
    python3.10 run_lightrag_standalone.py --limit 10

준비:
    python3.10 -m pip install lightrag-hku python-dotenv pyyaml tqdm
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm


def check_python_version():
    if sys.version_info < (3, 10):
        print(f"오류: Python 3.10+ 필요. 현재: {sys.version}")
        sys.exit(1)


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def load_data(config, qa_file=None):
    with open(Path(config["paths"]["data_processed"]) / "documents.json", encoding="utf-8") as f:
        docs = json.load(f)
    qa_filename = qa_file if qa_file else "questions.json"
    with open(Path(config["paths"]["qa_sets"]) / qa_filename, encoding="utf-8") as f:
        questions = json.load(f)
    return {d["doc_id"]: d for d in docs}, questions


def get_working_dir(doc_id: str) -> str:
    p = Path("data/lightrag_indexes") / doc_id
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def build_doc_text(doc: dict) -> str:
    parts = []
    if doc.get("abstract"):
        parts.append(doc["abstract"])
    for s in doc.get("sections", []):
        parts.append(f"[{s['section_title']}]\n{s['text']}")
    return "\n\n".join(parts)


async def run_one(question: dict, doc: dict, mode: str = "hybrid") -> dict:
    import time
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

    working_dir = get_working_dir(question["doc_id"])
    done_flag = Path(working_dir) / ".indexed"

    start = time.time()
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
    )
    await rag.initialize_storages()

    if not done_flag.exists():
        await rag.ainsert(build_doc_text(doc))
        done_flag.touch()

    result = await rag.aquery(question["question"], param=QueryParam(mode=mode))
    await rag.finalize_storages()

    answer = result.content if hasattr(result, "content") else str(result)
    elapsed = time.time() - start

    return {
        "question_id": question["question_id"],
        "doc_id": question["doc_id"],
        "question_type": question["question_type"],
        "baseline": f"lightrag_{mode}",
        "question": question["question"],
        "answer": answer,
        "retrieved_context": [],
        "intermediate_summary": None,
        "runtime_sec": round(elapsed, 2),
        "notes": f"LightRAG mode={mode}",
    }


async def main_async(questions, doc_map, mode, limit):
    results = []
    if limit:
        questions = questions[:limit]
    for q in tqdm(questions, desc=f"LightRAG ({mode})"):
        doc = doc_map.get(q["doc_id"])
        if not doc:
            continue
        try:
            r = await run_one(q, doc, mode)
            results.append(r)
        except Exception as e:
            print(f"  오류 ({q['question_id']}): {e}")
    return results


def main():
    check_python_version()
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="hybrid", choices=["naive", "local", "global", "hybrid", "mix"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--questions", type=str, default=None,
                        help="질문셋 파일명 (예: questions_balanced.json)")
    args = parser.parse_args()

    config = load_config()
    doc_map, questions = load_data(config, qa_file=args.questions)

    print(f"LightRAG ({args.mode}) 실행: {len(questions)}개 질문")
    results = asyncio.run(main_async(questions, doc_map, args.mode, args.limit))

    out_dir = Path(config["paths"]["results_raw"])
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"results_lightrag_{args.mode}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {out_path}  ({len(results)}개)")


if __name__ == "__main__":
    main()
