from src.generation.llm_client import DEFAULT_LLM_MODEL
from src.pipeline.qa_pipeline import QAPipeline


def test_qa_pipeline_default_models():
    assert QAPipeline.__init__.__defaults__ == (
        "BAAI/bge-m3",
        "hf",
        DEFAULT_LLM_MODEL,
        "BAAI/bge-reranker-v2-m3",
        None,
        True,
        5,
        3,
        1200,
        128,
    )


def test_qa_pipeline_build_does_not_load_llm(monkeypatch):
    state = {"llm_inits": 0, "llm_generates": 0}

    monkeypatch.setattr("src.pipeline.qa_pipeline.get_compute_device", lambda: "cpu")

    class FakeRetriever:
        def __init__(self, model_name, device=None):
            self.model_name = model_name
            self.device = device
            self.vector_store = type("VectorStore", (), {"index_backend": "gpu"})()

        def build_index(self, chunks):
            self.chunks = chunks

        def retrieve(self, query, top_k=3):
            return [
                {
                    "chunk_id": "c1",
                    "text": "Beamforming improves signal quality in wireless systems.",
                    "source": "doc1.txt",
                    "score": 0.9,
                }
            ]

    class FakeReranker:
        def __init__(self, model_name, device=None):
            self.model_name = model_name
            self.device = device

        def rerank(self, query, docs):
            for doc in docs:
                doc["rerank_score"] = 1.0
            return docs

    class FakeLLMClient:
        def __init__(self, mode="hf", model_name=DEFAULT_LLM_MODEL, max_new_tokens=128, device=None):
            state["llm_inits"] += 1
            self.mode = mode
            self.model_name = model_name
            self.max_new_tokens = max_new_tokens
            self.device = device

        def generate(self, prompt):
            state["llm_generates"] += 1
            return "answer"

    monkeypatch.setattr("src.pipeline.qa_pipeline.Retriever", FakeRetriever)
    monkeypatch.setattr("src.pipeline.qa_pipeline.Reranker", FakeReranker)
    monkeypatch.setattr("src.pipeline.qa_pipeline.LLMClient", FakeLLMClient)

    chunks = [
        {
            "chunk_id": "c1",
            "text": "Beamforming improves signal quality in wireless systems.",
            "source": "doc1.txt",
            "start_idx": 0,
            "end_idx": 60,
        }
    ]

    pipeline = QAPipeline()
    assert state == {"llm_inits": 1, "llm_generates": 0}

    pipeline.build_knowledge_base(chunks)
    assert state == {"llm_inits": 1, "llm_generates": 0}

    result = pipeline.ask("What is beamforming?", top_k=2)

    assert state == {"llm_inits": 1, "llm_generates": 1}
    assert "answer" in result
    assert "sources" in result
    assert "retrieved_contexts" in result
    assert "timings" in result
    assert "config" in result
    assert result["device"] == "cpu"
    assert result["vector_backend"] == "gpu"
    assert len(result["sources"]) > 0
    assert result["timings"]["rerank_ms"] == 0.0
    assert result["config"]["enable_rerank"] is False


def test_qa_pipeline_uses_top_two_reranked_contexts_for_prompt(monkeypatch):
    prompts = []

    monkeypatch.setattr("src.pipeline.qa_pipeline.get_compute_device", lambda: "cpu")

    class FakeRetriever:
        def __init__(self, model_name, device=None):
            self.model_name = model_name
            self.device = device
            self.vector_store = type("VectorStore", (), {"index_backend": "cpu"})()

        def build_index(self, chunks):
            self.chunks = chunks

        def retrieve(self, query, top_k=3):
            return [
                {"chunk_id": "c1", "text": "beamforming details", "source": "doc1.txt", "score": 0.95},
                {"chunk_id": "c2", "text": "antenna arrays", "source": "doc2.txt", "score": 0.85},
                {"chunk_id": "c3", "text": "HRMS software", "source": "doc3.txt", "score": 0.10},
            ]

    class FakeReranker:
        def __init__(self, model_name, device=None):
            self.model_name = model_name
            self.device = device

        def rerank(self, query, docs):
            reranked = []
            for doc, score in zip(docs, [3.2, 2.8, -1.0]):
                updated = doc.copy()
                updated["rerank_score"] = score
                reranked.append(updated)
            return reranked

    class FakeLLMClient:
        def __init__(self, mode="hf", model_name=DEFAULT_LLM_MODEL, max_new_tokens=128, device=None):
            self.mode = mode
            self.model_name = model_name
            self.max_new_tokens = max_new_tokens
            self.device = device

        def generate(self, prompt):
            prompts.append(prompt)
            return "answer"

    monkeypatch.setattr("src.pipeline.qa_pipeline.Retriever", FakeRetriever)
    monkeypatch.setattr("src.pipeline.qa_pipeline.Reranker", FakeReranker)
    monkeypatch.setattr("src.pipeline.qa_pipeline.LLMClient", FakeLLMClient)

    pipeline = QAPipeline()
    pipeline.build_knowledge_base(
        [
            {"chunk_id": "c1", "text": "beamforming details", "source": "doc1.txt"},
            {"chunk_id": "c2", "text": "antenna arrays", "source": "doc2.txt"},
            {"chunk_id": "c3", "text": "HRMS software", "source": "doc3.txt"},
        ]
    )

    result = pipeline.ask("What is beamforming?", top_k=3)

    assert prompts
    assert "beamforming details" in prompts[0]
    assert "antenna arrays" in prompts[0]
    assert "HRMS software" not in prompts[0]
    assert len(result["retrieved_contexts"]) == 3


def test_qa_pipeline_passes_device_to_components(monkeypatch):
    seen = {}

    class FakeRetriever:
        def __init__(self, model_name, device=None):
            seen["retriever"] = (model_name, device)
            self.vector_store = type("VectorStore", (), {"index_backend": "gpu"})()

        def build_index(self, chunks):
            self.chunks = chunks

        def retrieve(self, query, top_k=3):
            return [{"chunk_id": "c1", "text": "beamforming", "source": "doc1.txt", "score": 0.9}]

    class FakeReranker:
        def __init__(self, model_name, device=None):
            seen["reranker"] = (model_name, device)

        def rerank(self, query, docs):
            for doc in docs:
                doc["rerank_score"] = 1.0
            return docs

    class FakeLLMClient:
        def __init__(self, mode="hf", model_name=DEFAULT_LLM_MODEL, max_new_tokens=128, device=None):
            seen["llm"] = (mode, model_name, max_new_tokens, device)
            self.max_new_tokens = max_new_tokens

        def generate(self, prompt):
            return "answer"

    monkeypatch.setattr("src.pipeline.qa_pipeline.Retriever", FakeRetriever)
    monkeypatch.setattr("src.pipeline.qa_pipeline.Reranker", FakeReranker)
    monkeypatch.setattr("src.pipeline.qa_pipeline.LLMClient", FakeLLMClient)

    pipeline = QAPipeline(device="cuda")
    pipeline.build_knowledge_base([{"chunk_id": "c1", "text": "beamforming", "source": "doc1.txt"}])
    result = pipeline.ask("What is beamforming?")

    assert seen["retriever"] == ("BAAI/bge-m3", "cuda")
    assert seen["reranker"] == ("BAAI/bge-reranker-v2-m3", "cuda")
    assert seen["llm"] == ("hf", DEFAULT_LLM_MODEL, 128, "cuda")
    assert result["device"] == "cuda"
    assert result["vector_backend"] == "gpu"


def test_qa_pipeline_reranks_only_top_subset(monkeypatch):
    seen = {}

    class FakeRetriever:
        def __init__(self, model_name, device=None):
            self.vector_store = type("VectorStore", (), {"index_backend": "cpu"})()

        def build_index(self, chunks):
            self.chunks = chunks

        def retrieve(self, query, top_k=3):
            seen["retrieve_top_k"] = top_k
            return [
                {"chunk_id": "c1", "text": "one", "source": "doc1.txt", "score": 0.91},
                {"chunk_id": "c2", "text": "two", "source": "doc2.txt", "score": 0.90},
                {"chunk_id": "c3", "text": "three", "source": "doc3.txt", "score": 0.89},
                {"chunk_id": "c4", "text": "four", "source": "doc4.txt", "score": 0.88},
                {"chunk_id": "c5", "text": "five", "source": "doc5.txt", "score": 0.87},
            ]

    class FakeReranker:
        def __init__(self, model_name, device=None):
            pass

        def rerank(self, query, docs):
            seen["rerank_docs"] = [doc["chunk_id"] for doc in docs]
            reranked = list(reversed([doc.copy() for doc in docs]))
            for idx, doc in enumerate(reranked, start=1):
                doc["rerank_score"] = float(10 - idx)
            return reranked

    class FakeLLMClient:
        def __init__(self, mode="hf", model_name=DEFAULT_LLM_MODEL, max_new_tokens=128, device=None):
            self.max_new_tokens = max_new_tokens

        def generate(self, prompt):
            return "answer"

    monkeypatch.setattr("src.pipeline.qa_pipeline.Retriever", FakeRetriever)
    monkeypatch.setattr("src.pipeline.qa_pipeline.Reranker", FakeReranker)
    monkeypatch.setattr("src.pipeline.qa_pipeline.LLMClient", FakeLLMClient)

    pipeline = QAPipeline(candidate_k=5, rerank_top_n=3)
    pipeline.build_knowledge_base([{"chunk_id": "c1", "text": "one", "source": "doc1.txt"}])

    result = pipeline.ask("What is beamforming?", top_k=3)

    assert seen["retrieve_top_k"] == 5
    assert seen["rerank_docs"] == ["c1", "c2", "c3"]
    assert [item["chunk_id"] for item in result["retrieved_contexts"]] == ["c3", "c2", "c1"]
    assert result["config"]["candidate_k"] == 5
    assert result["config"]["enable_rerank"] is True


def test_qa_pipeline_respects_prompt_char_budget(monkeypatch):
    prompts = []

    class FakeRetriever:
        def __init__(self, model_name, device=None):
            self.vector_store = type("VectorStore", (), {"index_backend": "cpu"})()

        def build_index(self, chunks):
            self.chunks = chunks

        def retrieve(self, query, top_k=3):
            return [
                {"chunk_id": "c1", "text": "A" * 80, "source": "doc1.txt", "score": 0.9},
                {"chunk_id": "c2", "text": "B" * 80, "source": "doc2.txt", "score": 0.8},
            ]

    class FakeReranker:
        def __init__(self, model_name, device=None):
            pass

        def rerank(self, query, docs):
            return docs

    class FakeLLMClient:
        def __init__(self, mode="hf", model_name=DEFAULT_LLM_MODEL, max_new_tokens=128, device=None):
            self.max_new_tokens = max_new_tokens

        def generate(self, prompt):
            prompts.append(prompt)
            return "answer"

    monkeypatch.setattr("src.pipeline.qa_pipeline.Retriever", FakeRetriever)
    monkeypatch.setattr("src.pipeline.qa_pipeline.Reranker", FakeReranker)
    monkeypatch.setattr("src.pipeline.qa_pipeline.LLMClient", FakeLLMClient)

    pipeline = QAPipeline(prompt_char_budget=100)
    pipeline.build_knowledge_base([{"chunk_id": "c1", "text": "seed", "source": "doc1.txt"}])
    pipeline.ask("What is beamforming?", top_k=2, enable_rerank=False)

    assert prompts
    assert "A" * 80 in prompts[0]
    assert "B" * 21 not in prompts[0]
