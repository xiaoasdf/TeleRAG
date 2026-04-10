from typing import Dict, List

from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import build_prompt
from src.retrieval.retriever import Retriever


class QAPipeline:
    def __init__(self, model_name: str = "BAAI/bge-small-en", llm_mode: str = "hf"):
        self.retriever = Retriever(model_name=model_name)
        self.llm_client = LLMClient(mode=llm_mode)
        self.is_ready = False

    def build_knowledge_base(self, chunks: List[Dict]) -> None:
        if not chunks:
            raise ValueError("Chunks cannot be empty")

        self.retriever.build_index(chunks)
        self.is_ready = True

    def ask(self, query: str, top_k: int = 3) -> Dict:
        if not self.is_ready:
            raise ValueError("Knowledge base has not been built yet")

        if not query.strip():
            raise ValueError("Query cannot be empty")

        retrieved_contexts = self.retriever.retrieve(query, top_k=top_k)
        prompt = build_prompt(query, retrieved_contexts)
        answer = self.llm_client.generate(prompt)

        sources = []
        for ctx in retrieved_contexts:
            sources.append(
                {
                    "chunk_id": ctx.get("chunk_id"),
                    "source": ctx.get("source"),
                    "score": ctx.get("score"),
                }
            )

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "retrieved_contexts": retrieved_contexts,
            "prompt": prompt,
        }