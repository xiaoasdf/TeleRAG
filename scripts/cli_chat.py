from src.pipeline.qa_pipeline import QAPipeline
from src.pipeline.index_pipeline import build_chunks_from_file


import os
pipeline = QAPipeline()
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(BASE_DIR, "data", "raw", "test.txt")
chunks = build_chunks_from_file(file_path)
pipeline.build_knowledge_base(chunks)

while True:
    query = input("\n请输入问题（exit退出）：")

    if query == "exit":
        break

    result = pipeline.ask(query)

    print("\n回答：")
    print(result["answer"])