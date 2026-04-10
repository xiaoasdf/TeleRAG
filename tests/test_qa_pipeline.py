from src.generation.llm_client import DEFAULT_LLM_MODEL
from src.pipeline.qa_pipeline import QAPipeline


def test_qa_pipeline_default_models():
    assert QAPipeline.__init__.__defaults__ == (
        "BAAI/bge-m3",
        "hf",
        DEFAULT_LLM_MODEL,
        "BAAI/bge-reranker-v2-m3",
    )


def test_qa_pipeline_build_does_not_load_llm(monkeypatch):
    state = {"llm_inits": 0, "llm_generates": 0}

    class FakeRetriever:
        def __init__(self, model_name):
            self.model_name = model_name

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
        def __init__(self, model_name):
            self.model_name = model_name

        def rerank(self, query, docs):
            for doc in docs:
                doc["rerank_score"] = 1.0
            return docs

    class FakeLLMClient:
        def __init__(self, mode="hf", model_name=DEFAULT_LLM_MODEL):
            state["llm_inits"] += 1
            self.mode = mode
            self.model_name = model_name

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
    assert len(result["sources"]) > 0


def test_qa_pipeline_uses_top_two_reranked_contexts_for_prompt(monkeypatch):
    prompts = []

    class FakeRetriever:
        def __init__(self, model_name):
            self.model_name = model_name

        def build_index(self, chunks):
            self.chunks = chunks

        def retrieve(self, query, top_k=3):
            return [
                {"chunk_id": "c1", "text": "beamforming details", "source": "doc1.txt", "score": 0.95},
                {"chunk_id": "c2", "text": "antenna arrays", "source": "doc2.txt", "score": 0.85},
                {"chunk_id": "c3", "text": "HRMS software", "source": "doc3.txt", "score": 0.10},
            ]

    class FakeReranker:
        def __init__(self, model_name):
            self.model_name = model_name

        def rerank(self, query, docs):
            reranked = []
            for doc, score in zip(docs, [3.2, 2.8, -1.0]):
                updated = doc.copy()
                updated["rerank_score"] = score
                reranked.append(updated)
            return reranked

    class FakeLLMClient:
        def __init__(self, mode="hf", model_name=DEFAULT_LLM_MODEL):
            self.mode = mode
            self.model_name = model_name

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
