from src.generation.llm_client import LLMClient


def test_llm_client_mock():
    client = LLMClient(mode="mock")

    prompt = """You are a helpful AI assistant.

[Context 1] (source: doc1.txt)
Beamforming improves signal quality in wireless systems.

[Question]
What is beamforming?

[Answer]
"""

    answer = client.generate(prompt)

    assert isinstance(answer, str)
    assert len(answer) > 0