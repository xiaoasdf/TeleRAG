import os

from bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path()

from src.config import load_config
from src.generation.llm_client import DEFAULT_LLM_MODEL
from src.pipeline.index_pipeline import build_chunks_from_file
from src.pipeline.qa_pipeline import QAPipeline


BASE_DIR = str(PROJECT_ROOT)
file_path = os.path.join(BASE_DIR, "data", "raw", "beamforming.pdf")
APP_CONFIG = load_config()

pipeline = QAPipeline(
    model_name=APP_CONFIG.retriever_model,
    llm_mode="hf",
    llm_model_name=APP_CONFIG.default_llm_model or DEFAULT_LLM_MODEL,
    reranker_model_name=APP_CONFIG.reranker_model,
    enable_rerank=APP_CONFIG.enable_rerank,
    candidate_k=APP_CONFIG.candidate_k,
    rerank_top_n=APP_CONFIG.rerank_top_n,
    prompt_char_budget=APP_CONFIG.prompt_char_budget,
    max_new_tokens=APP_CONFIG.max_new_tokens,
)

chunks = build_chunks_from_file(file_path, chunk_size=APP_CONFIG.chunk_size, overlap=APP_CONFIG.overlap)
pipeline.build_knowledge_base(chunks)

print("TeleRAG communications knowledge base ready.")
print(f"Generator model will be loaded on first question: {pipeline.llm_client.model_name}")
print("Suggested queries: What is beamforming? / Explain MIMO. / Which standards organizations are covered?")

while True:
    query = input("\n请输入问题（exit 退出）：")
    if query.lower() == "exit":
        break

    result = pipeline.ask(query, top_k=APP_CONFIG.top_k)

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
