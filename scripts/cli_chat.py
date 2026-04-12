from __future__ import annotations

from pathlib import Path

from bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path()

from src.api.service import TeleRAGService
from src.config import load_config


APP_CONFIG = load_config()
DEFAULT_SAMPLE_PATHS = [
    PROJECT_ROOT / "data" / "raw" / "beamforming.pdf",
    PROJECT_ROOT / "data" / "raw" / "wireless_systems_overview.md",
    PROJECT_ROOT / "data" / "raw" / "communications_standards_notes.txt",
]


def _existing_sample_paths() -> list[Path]:
    return [path for path in DEFAULT_SAMPLE_PATHS if path.exists()]


def _bootstrap_service() -> TeleRAGService:
    service = TeleRAGService(config=APP_CONFIG)
    if service.pipeline.is_ready:
        return service

    sample_paths = _existing_sample_paths()
    if not sample_paths:
        raise SystemExit(
            "No persisted knowledge base was found and no sample documents are available under data/raw/."
        )

    service.index_document(file_paths=[str(path) for path in sample_paths], persist=False)
    return service


def main() -> None:
    service = _bootstrap_service()

    print("TeleRAG communications knowledge base ready.")
    print(f"Generator model will be loaded on first question: {service.pipeline.llm_client.model_name}")
    print("Suggested queries: What is beamforming? / Explain MIMO. / Which standards organizations are covered?")

    while True:
        query = input("\n请输入问题（exit 退出）：")
        if query.lower() == "exit":
            break

        result = service.ask(query, top_k=APP_CONFIG.top_k)

        print("\n" + "=" * 60)
        print("回答：")
        print(result["answer"])

        print("\n来源：")
        for item in result["sources"]:
            print(
                f"- source={item['source']}, "
                f"chunk_id={item['chunk_id']}, "
                f"score={item['score']:.4f}"
            )

        print("\n检索到的文本片段：")
        for ctx in result["retrieved_contexts"]:
            print("-" * 40)
            print(f"{ctx['chunk_id']} | {ctx['source']} | score={ctx['score']:.4f}")
            print(ctx["text"])


if __name__ == "__main__":
    main()
