from bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path()

from src.pipeline.index_pipeline import build_chunks_from_file
from src.pipeline.qa_pipeline import QAPipeline


file_path = PROJECT_ROOT / "data" / "raw" / "wireless_systems_overview.md"

pipeline = QAPipeline()
chunks = build_chunks_from_file(str(file_path))
pipeline.build_knowledge_base(chunks)

result = pipeline.ask("What topics are summarized in the wireless systems overview note?")

print("=" * 50)
print("ANSWER:")
print(result["answer"])
