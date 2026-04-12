from __future__ import annotations

import json
import statistics
from pathlib import Path

from bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path()

from src.api.service import TeleRAGService
from src.pipeline.index_pipeline import build_chunks_from_file


def keyword_hit(answer: str, expected_keywords: list[str]) -> float:
    lowered = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in lowered)
    return hits / len(expected_keywords) if expected_keywords else 0.0


def retrieval_hit(sources: list[dict], source_doc: str) -> float:
    return 1.0 if any(item.get("source") == source_doc for item in sources) else 0.0


def run_suite(service: TeleRAGService, dataset: list[dict], enable_rerank: bool, top_k: int) -> dict:
    answer_scores = []
    retrieval_scores = []
    total_latencies = []

    for row in dataset:
        result = service.ask(
            query=row["question"],
            top_k=top_k,
            enable_rerank=enable_rerank,
        )
        answer_scores.append(keyword_hit(result["answer"], row["expected_keywords"]))
        retrieval_scores.append(retrieval_hit(result["sources"], row["source_doc"]))
        total_latencies.append(result["timings"]["total_ms"])

    return {
        "enable_rerank": enable_rerank,
        "top_k": top_k,
        "retrieval_hit_rate": round(statistics.mean(retrieval_scores), 4),
        "answer_keyword_coverage": round(statistics.mean(answer_scores), 4),
        "avg_total_ms": round(statistics.mean(total_latencies), 2),
    }


def build_benchmark_corpus(service: TeleRAGService) -> list[str]:
    source_paths = [
        PROJECT_ROOT / "data" / "raw" / "beamforming.pdf",
        PROJECT_ROOT / "data" / "raw" / "wireless_systems_overview.md",
        PROJECT_ROOT / "data" / "raw" / "communications_standards_notes.txt",
    ]
    all_chunks = []
    for path in source_paths:
        all_chunks.extend(
            build_chunks_from_file(
                str(path),
                chunk_size=service.config.chunk_size,
                overlap=service.config.overlap,
            )
        )

    service.pipeline.build_knowledge_base(all_chunks)
    service.chunk_count = len(all_chunks)
    return [path.name for path in source_paths]


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "eval" / "communications_eval.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    service = TeleRAGService()
    source_docs = build_benchmark_corpus(service)

    runs = [
        run_suite(service, dataset, enable_rerank=False, top_k=2),
        run_suite(service, dataset, enable_rerank=True, top_k=3),
    ]

    output = {
        "dataset_size": len(dataset),
        "source_docs": source_docs,
        "runs": runs,
    }
    output_path = PROJECT_ROOT / "docs" / "benchmark_results.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
