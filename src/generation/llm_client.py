class LLMClient:
    def __init__(self, mode: str = "mock"):
        self.mode = mode

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if self.mode == "mock":
            return self._mock_generate(prompt)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _mock_generate(self, prompt: str) -> str:
        """
        简单 mock 版本：
        从 prompt 中提取一点上下文，返回一个模拟回答。
        目的是先把 QA 流程打通。
        """
        lines = prompt.splitlines()
        context_lines = []

        for line in lines:
            if line.startswith("[Context"):
                continue
            if line.startswith("[Question]") or line.startswith("[Answer]"):
                continue
            if "(source:" in line:
                continue
            if line.strip():
                context_lines.append(line.strip())

        if context_lines:
            context_summary = context_lines[:2]
            joined = " ".join(context_summary)
            return f"Based on the retrieved context, the answer is: {joined}"
        return "I do not know based on the provided context."