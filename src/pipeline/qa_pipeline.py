from typing import Dict, List
import time
from pathlib import Path
from typing import Dict, List

from src.generation.llm_client import DEFAULT_LLM_MODEL, DEFAULT_MAX_NEW_TOKENS, LLMClient
from src.generation.prompt_builder import build_prompt
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever
from src.runtime import get_compute_device


class QAPipeline:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        llm_mode: str = "hf",
        llm_model_name: str = DEFAULT_LLM_MODEL,
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        enable_rerank: bool = True,
        candidate_k: int = 5,
        rerank_top_n: int = 3,
        prompt_char_budget: int = 1200,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ):
        self.device = device or get_compute_device()
        self.retriever = Retriever(model_name=model_name, device=self.device)
        self.llm_client = LLMClient(
            mode=llm_mode,
            model_name=llm_model_name,
            max_new_tokens=max_new_tokens,
            device=self.device,
        )
        self.is_ready = False
        self.reranker = Reranker(model_name=reranker_model_name, device=self.device)
        self.enable_rerank = enable_rerank
        self.candidate_k = max(candidate_k, 1)
        self.rerank_top_n = max(rerank_top_n, 1)
        self.prompt_char_budget = max(prompt_char_budget, 1)

    def build_knowledge_base(self, chunks: List[Dict]) -> None:
        if not chunks:
            raise ValueError("Chunks cannot be empty")

        self.retriever.build_index(chunks)
        self.is_ready = True

    def save_knowledge_base(self, output_dir: str | Path) -> None:
        if not self.is_ready:
            raise ValueError("Knowledge base has not been built yet")
        self.retriever.save_index(output_dir)

    def load_knowledge_base(self, input_dir: str | Path) -> None:
        self.retriever.load_index(input_dir)
        self.is_ready = True

    def ask(
        self,
        query: str,
        top_k: int = 3,
        enable_rerank: bool | None = None,
        max_new_tokens: int | None = None,
    ) -> Dict:
        if not self.is_ready:
            raise ValueError("Knowledge base has not been built yet")
        if not query.strip():
            raise ValueError("Query cannot be empty")

        total_started_at = time.perf_counter()
        should_rerank = self._should_rerank(top_k=top_k, enable_rerank=enable_rerank)
        candidate_k = max(top_k, self.candidate_k) if should_rerank else top_k

        retrieve_started_at = time.perf_counter()
        retrieved_contexts = self.retriever.retrieve(query, top_k=candidate_k)
        retrieve_ms = self._elapsed_ms(retrieve_started_at)

        rerank_started_at = time.perf_counter()
        retrieved_contexts = self._maybe_rerank(query, retrieved_contexts, should_rerank)
        rerank_ms = self._elapsed_ms(rerank_started_at) if should_rerank else 0.0
        retrieved_contexts = retrieved_contexts[:top_k]

        prompt_started_at = time.perf_counter()
        prompt_contexts = self._select_prompt_contexts(retrieved_contexts)
        prompt = build_prompt(query, prompt_contexts)
        prompt_ms = self._elapsed_ms(prompt_started_at)

        if max_new_tokens is not None:
            self.llm_client.max_new_tokens = max_new_tokens

        generate_started_at = time.perf_counter()
        answer = self.llm_client.generate(prompt)
        generate_ms = self._elapsed_ms(generate_started_at)
        total_ms = self._elapsed_ms(total_started_at)

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
            "device": self.device,
            "vector_backend": self.retriever.vector_store.index_backend if self.retriever.vector_store else "cpu",
            "timings": {
                "retrieve_ms": retrieve_ms,
                "rerank_ms": rerank_ms,
                "prompt_ms": prompt_ms,
                "generate_ms": generate_ms,
                "total_ms": total_ms,
            },
            "config": {
                "top_k": top_k,
                "candidate_k": candidate_k,
                "enable_rerank": should_rerank,
                "rerank_top_n": self.rerank_top_n,
                "prompt_char_budget": self.prompt_char_budget,
                "max_new_tokens": self.llm_client.max_new_tokens,
            },
        }

    def _select_prompt_contexts(self, contexts: List[Dict]) -> List[Dict]:
        selected_contexts = []
        remaining_budget = self.prompt_char_budget

        for ctx in contexts:
            if len(selected_contexts) >= 2 or remaining_budget <= 0:
                break

            text = ctx.get("text", "").strip()
            if not text:
                continue

            truncated_text = text[:remaining_budget].strip()
            if not truncated_text:
                continue

            selected_context = ctx.copy()
            selected_context["text"] = truncated_text
            selected_contexts.append(selected_context)
            remaining_budget -= len(truncated_text)

        if selected_contexts:
            return selected_contexts

        return contexts[:1]

    def _should_rerank(self, top_k: int, enable_rerank: bool | None) -> bool:
        if enable_rerank is False:
            return False
        if enable_rerank is True:
            return True
        return top_k > 2 and self.enable_rerank

    def _maybe_rerank(self, query: str, contexts: List[Dict], should_rerank: bool) -> List[Dict]:
        if not should_rerank or not contexts:
            return contexts

        rerank_limit = min(len(contexts), self.rerank_top_n)
        reranked_subset = self.reranker.rerank(query, contexts[:rerank_limit])
        return reranked_subset + contexts[rerank_limit:]

    def _elapsed_ms(self, started_at: float) -> float:
        return round((time.perf_counter() - started_at) * 1000, 2)
