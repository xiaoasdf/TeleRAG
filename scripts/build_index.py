import os
from src.pipeline.qa_pipeline import QAPipeline
from src.pipeline.index_pipeline import build_chunks_from_file

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(BASE_DIR, "data", "raw", "test.txt")

pipeline = QAPipeline()
chunks = build_chunks_from_file(file_path)
pipeline.build_knowledge_base(chunks)

result = pipeline.ask("What is beamforming?")

print("=" * 50)
print("ANSWER:")
print(result["answer"])