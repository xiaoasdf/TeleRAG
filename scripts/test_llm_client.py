from src.generation.llm_client import LLMClient


prompt = """You are a helpful AI assistant for telecommunications knowledge QA.

Please answer the user's question based only on the provided context.
If the answer cannot be found in the context, say you do not know.

[Context 1] (source: doc1.txt)
Beamforming improves signal quality in wireless systems.

[Context 2] (source: doc2.txt)
Dynamic TDD allows flexible uplink and downlink allocation.

[Question]
What is beamforming?

[Answer]
"""

client = LLMClient(mode="hf")
answer = client.generate(prompt)

print(answer)