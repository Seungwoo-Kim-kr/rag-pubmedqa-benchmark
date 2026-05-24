"""
run_experiment.py
=================
메인 실험 파이프라인.
모든 baseline을 같은 질문셋에 대해 실행하고 결과를 저장합니다.

실행:
    python run_experiment.py
    python run_experiment.py --baselines direct_qa standard_rag   # 특정 baseline만
    python run_experiment.py --limit 5                             # 질문 5개만 테스트
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from baselines.direct_qa import run_direct_qa
from baselines.lightrag_runner import run_lightrag
from baselines.standard_rag import run_standard_rag
from baselines.summary_mediated_qa import run_summary_mediated_qa
from retrieval.chunking import chunk_all_documents
from retrieval.embed_index import build_index, load_index, save_index


def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def load_processed_data(config: dict, qa_file: str = None, docs_file: str = None) -> tuple[list[dict], list[dict]]:
    proc_dir = Path(config["paths"]["data_processed"])
    qa_dir = Path(config["paths"]["qa_sets"])

    docs_filename = docs_file if docs_file else "documents.json"
    docs_path = proc_dir / docs_filename
    qa_filename = qa_file if qa_file else "questions.json"
    qa_path = qa_dir / qa_filename

    if not docs_path.exists() or not qa_path.exists():
        print("전처리된 데이터가 없습니다. 먼저 prepare_data.py를 실행하세요.")
        sys.exit(1)

    with open(docs_path, encoding="utf-8") as f:
        docs = json.load(f)
    with open(qa_path, encoding="utf-8") as f:
        questions = json.load(f)

    return docs, questions


def prepare_rag_indexes(docs: list[dict], config: dict) -> dict[str, tuple]:
    """각 문서에 대한 FAISS 인덱스를 빌드(또는 캐시에서 로드)합니다."""
    index_dir = "data/indexes"
    emb_cfg = config["embedding"]
    ret_cfg = config["retrieval"]

    doc_chunks = chunk_all_documents(
        docs,
        chunk_size=ret_cfg["chunk_size"],
        overlap=ret_cfg["chunk_overlap"],
    )

    indexes = {}
    for doc in tqdm(docs, desc="FAISS 인덱스 준비"):
        doc_id = doc["doc_id"]
        faiss_path = Path(index_dir) / f"{doc_id}.faiss"

        if faiss_path.exists():
            index, chunks = load_index(doc_id, index_dir)
        else:
            chunks = doc_chunks[doc_id]
            if not chunks:
                continue
            index, chunks = build_index(chunks, emb_cfg["model"], emb_cfg["batch_size"])
            save_index(index, chunks, doc_id, index_dir)

        indexes[doc_id] = (index, chunks)

    return indexes


def run_all_baselines(
    questions: list[dict],
    docs: list[dict],
    indexes: dict,
    config: dict,
    active_baselines: list[str],
) -> list[dict]:
    doc_map = {d["doc_id"]: d for d in docs}
    all_results = []

    for question in tqdm(questions, desc="질문 처리"):
        doc_id = question["doc_id"]
        doc = doc_map.get(doc_id)
        if not doc:
            continue

        if "direct_qa" in active_baselines:
            result = run_direct_qa(question, doc, config)
            all_results.append(result)

        if doc_id in indexes:
            index, chunks = indexes[doc_id]

            if "standard_rag" in active_baselines:
                result = run_standard_rag(question, index, chunks, config)
                all_results.append(result)

            if "summary_mediated_qa" in active_baselines:
                result = run_summary_mediated_qa(question, index, chunks, config)
                all_results.append(result)

        if "lightrag" in active_baselines:
            result = run_lightrag(question, doc, config)
            all_results.append(result)

    return all_results


def save_results(results: list[dict], config: dict):
    out_dir = Path(config["paths"]["results_raw"])
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"results_{timestamp}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {out_path}  ({len(results)}개 항목)")
    return out_path


def print_summary(results: list[dict]):
    from collections import defaultdict
    by_baseline = defaultdict(list)
    for r in results:
        by_baseline[r["baseline"]].append(r)

    print("\n" + "=" * 50)
    print("실험 완료 요약")
    print("=" * 50)
    for baseline, items in by_baseline.items():
        avg_time = sum(i["runtime_sec"] for i in items) / len(items)
        print(f"  {baseline}: {len(items)}개 질문, 평균 {avg_time:.2f}초")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", nargs="+", default=None,
                        help="실행할 baseline (default: config.yaml의 baselines 항목)")
    parser.add_argument("--limit", type=int, default=None,
                        help="질문 수 제한 (테스트용)")
    parser.add_argument("--questions", type=str, default=None,
                        help="질문셋 파일명 (default: questions.json, 예: questions_balanced.json)")
    parser.add_argument("--documents", type=str, default=None,
                        help="문서 파일명 (default: documents.json, 예: documents_qasper.json)")
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("오류: OPENAI_API_KEY가 없습니다. .env 파일을 확인하세요.")
        sys.exit(1)

    config = load_config()
    active_baselines = args.baselines or config["baselines"]

    print("데이터 로드 중...")
    docs, questions = load_processed_data(config, qa_file=args.questions, docs_file=args.documents)

    if args.limit:
        questions = questions[: args.limit]

    print(f"문서: {len(docs)}개 | 질문: {len(questions)}개 | Baselines: {active_baselines}")

    # RAG baseline이 필요할 때만 인덱스 빌드
    need_index = any(b in active_baselines for b in ["standard_rag", "summary_mediated_qa", "lightrag"])
    indexes = {}
    if need_index:
        print("\nFAISS 인덱스 준비 중...")
        indexes = prepare_rag_indexes(docs, config)

    print("\n실험 실행 중...")
    results = run_all_baselines(questions, docs, indexes, config, active_baselines)

    out_path = save_results(results, config)
    print_summary(results)
    print(f"\n다음 단계: python analyze_results.py --input {out_path}")


if __name__ == "__main__":
    main()
