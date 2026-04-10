from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class LLMClient:
    def __init__(self, mode: str = "hf"):
        self.mode = mode

        if self.mode == "hf":
            model_name = "google/flan-t5-base"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if self.mode == "hf":
            return self._hf_generate(prompt)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _hf_generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer