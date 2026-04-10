from typing import Dict, List


def build_prompt(query: str, contexts: List[Dict]) -> str:
    if not query.strip():
        raise ValueError("Query cannot be empty")

    if not contexts:
        raise ValueError("Contexts cannot be empty")

    context_blocks = []
    for i, ctx in enumerate(contexts, start=1):
        source = ctx.get("source", "unknown")
        text = ctx.get("text", "").strip()
        context_blocks.append(f"[Document {i}] source={source}\n{text}")

    context_text = "\n\n".join(context_blocks)

    return (
        "You are a helpful bilingual assistant for technical document question answering.\n"
        "Answer the user's question using only the context below.\n"
        "If the answer is not in the context, reply with: I cannot answer from the provided context.\n"
        "Use the same language as the user's question when possible.\n"
        "Ignore context that is unrelated to the user's question.\n"
        "Only answer the specific question asked, and do not add unrelated topics or examples.\n"
        "Write in complete sentences with natural sentence boundaries and proper punctuation.\n"
        "Do not output sentence fragments, broken line-by-line phrases, or unnatural line breaks.\n"
        "Do not output special tokens, placeholders, or the prompt itself.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
