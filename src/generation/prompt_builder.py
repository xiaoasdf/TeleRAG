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

        block = f"[Context {i}] (source: {source})\n{text}"
        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful AI assistant for telecommunications knowledge QA.

Please answer the user's question based only on the provided context.
If the answer cannot be found in the context, say you do not know.
Do not make up facts.
When possible, mention the source file name in your answer.

{context_text}

[Question]
{query}

[Answer]
"""
    return prompt