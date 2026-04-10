import os
from src.pipeline.qa_pipeline import QAPipeline
from src.pipeline.index_pipeline import build_chunks_from_file

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(BASE_DIR, "data", "raw", "test.pdf")

pipeline = QAPipeline(llm_mode="hf")

chunks = build_chunks_from_file(file_path)
pipeline.build_knowledge_base(chunks)

while True:
    query = input("\n请输入问题（exit退出）：")

    if query.lower() == "exit":
        break

    result = pipeline.ask(query, top_k=3)

    print("\n" + "=" * 60)
    print("回答：")
    print(result["answer"])

    print("\n来源：")
    for item in result["sources"]:
        print(
            f"- source={item['source']}, "
            f"chunk_id={item['chunk_id']}, "
            f"score={item['score']:.4f}"
        )

    print("\n检索到的文本片段：")
    for ctx in result["retrieved_contexts"]:
        print("-" * 40)
        print(f"{ctx['chunk_id']} | {ctx['source']} | score={ctx['score']:.4f}")
        print(ctx["text"])