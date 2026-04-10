from typing import Dict, List

from src.generation.llm_client import DEFAULT_LLM_MODEL, LLMClient
from src.generation.prompt_builder import build_prompt
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever


class QAPipeline:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        llm_mode: str = "hf",
        llm_model_name: str = DEFAULT_LLM_MODEL,
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
    ):
        self.retriever = Retriever(model_name=model_name)
        self.llm_client = LLMClient(mode=llm_mode, model_name=llm_model_name)
        self.is_ready = False
        self.reranker = Reranker(model_name=reranker_model_name)

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
        retrieved_contexts = self.reranker.rerank(query, retrieved_contexts)
        prompt_contexts = self._select_prompt_contexts(retrieved_contexts)
        prompt = build_prompt(query, prompt_contexts)
        answer = self.llm_client.generate(prompt)

        sources = []
        for ctx in retrieved_contexts:
            sources.append(
                {
                    "chunk_id": ctx.get("chunk_id"),
                    "source": ctx.get("source"),
                    "score": ctx.get("score"),
                    "rerank_score": ctx.get("rerank_score"),
                }
            )

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "retrieved_contexts": retrieved_contexts,
            "prompt": prompt,
        }

    def _select_prompt_contexts(self, contexts: List[Dict]) -> List[Dict]:
        if len(contexts) <= 2:
            return contexts

        return contexts[:2]
