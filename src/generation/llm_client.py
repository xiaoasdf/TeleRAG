from __future__ import annotations

import os
import re
from pathlib import Path

from src.runtime import get_compute_device

try:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )
except Exception:  # pragma: no cover
    AutoConfig = None
    AutoModelForCausalLM = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None


DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FAST_LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BALANCED_LLM_MODEL = DEFAULT_LLM_MODEL
DEFAULT_MAX_NEW_TOKENS = 128
STOPWORDS = {
    "about", "after", "also", "and", "are", "been", "being", "between", "both", "but",
    "can", "could", "does", "each", "from", "have", "helps", "into", "its", "many",
    "more", "most", "much", "only", "other", "over", "same", "such", "than", "that",
    "their", "them", "then", "there", "these", "they", "this", "those", "through",
    "used", "using", "various", "what", "when", "where", "which", "while", "with",
    "within", "would", "your",
}


class LLMClient:
    def __init__(
        self,
        mode: str = "hf",
        model_name: str = DEFAULT_LLM_MODEL,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        device: str | None = None,
        api_key_env: str | None = None,
        base_url: str | None = None,
    ):
        self.mode = mode
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device or get_compute_device()
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.tokenizer = None
        self.model = None
        self.config = None
        self.model_kind = None
        self._is_loaded = False
        self._remote_client = None

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if self.mode == "hf":
            return self._hf_generate(prompt)
        if self.mode == "openai_compatible":
            return self._openai_compatible_generate(prompt)
        if self.mode == "mock":
            return "This is a mock answer used for testing."

        raise ValueError(f"Unsupported mode: {self.mode}")

    def _ensure_hf_model_loaded(self) -> None:
        if self._is_loaded:
            return

        AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer = self._import_transformers()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(self._build_load_error_message("tokenizer", exc)) from exc

        try:
            self.config = AutoConfig.from_pretrained(self.model_name)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(self._build_load_error_message("config", exc)) from exc

        try:
            if getattr(self.config, "is_encoder_decoder", False):
                self.model_kind = "seq2seq"
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            elif self._is_causal_lm_config():
                self.model_kind = "causal"
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            else:
                architectures = getattr(self.config, "architectures", None)
                raise ValueError(f"Unsupported model architecture: {architectures}")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(self._build_load_error_message("model", exc)) from exc

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)

        self._is_loaded = True

    def _import_transformers(self):
        global AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        if None in (AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer):
            try:
                from transformers import (
                    AutoConfig as ImportedAutoConfig,
                    AutoModelForCausalLM as ImportedAutoModelForCausalLM,
                    AutoModelForSeq2SeqLM as ImportedAutoModelForSeq2SeqLM,
                    AutoTokenizer as ImportedAutoTokenizer,
                )
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "The 'transformers' package is required for local model generation. "
                    "Install local dependencies or switch to a compatible online API provider."
                ) from exc

            AutoConfig = ImportedAutoConfig
            AutoModelForCausalLM = ImportedAutoModelForCausalLM
            AutoModelForSeq2SeqLM = ImportedAutoModelForSeq2SeqLM
            AutoTokenizer = ImportedAutoTokenizer

        return AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    def _is_causal_lm_config(self) -> bool:
        architectures = getattr(self.config, "architectures", []) or []
        return any("CausalLM" in arch or "ForCausalLM" in arch for arch in architectures)

    def _build_load_error_message(self, component: str, exc: Exception) -> str:
        location_hint = self._build_location_hint()
        return (
            f"Failed to load the HuggingFace {component} for '{self.model_name}'. "
            f"Possible causes: missing dependencies such as 'sentencepiece' or 'protobuf'; "
            f"the model is not cached locally and this environment cannot download from Hugging Face; "
            f"or a local directory with the same name is shadowing the remote model. "
            f"{location_hint}Original error: {exc}"
        )

    def _build_location_hint(self) -> str:
        model_path = Path(self.model_name)
        if model_path.exists():
            return (
                f"The configured model path exists locally at '{model_path}'. "
                "Make sure it contains the full tokenizer/model files expected by Transformers. "
            )

        return (
            "If you are using a remote model ID, make sure network access is available or pre-download "
            "the model into the local Hugging Face cache. "
        )

    def _hf_generate(self, prompt: str) -> str:
        self._ensure_hf_model_loaded()

        if self.model_kind == "seq2seq":
            answer = self._generate_seq2seq(prompt)
        elif self.model_kind == "causal":
            answer = self._generate_causal(prompt)
        else:
            raise RuntimeError(f"Unsupported model kind: {self.model_kind}")

        return self._clean_answer(answer, prompt)

    def _generate_seq2seq(self, prompt: str) -> str:
        inputs = self._move_inputs_to_device(
            self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            )
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _generate_causal(self, prompt: str) -> str:
        inputs = self._move_inputs_to_device(
            self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            )
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        input_length = self._get_input_length(inputs["input_ids"])
        generated_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def _get_input_length(self, input_ids) -> int:
        if hasattr(input_ids, "shape"):
            return int(input_ids.shape[-1])
        if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)

    def _move_inputs_to_device(self, inputs):
        if self.device == "cpu":
            return inputs

        moved_inputs = {}
        for key, value in inputs.items():
            moved_inputs[key] = value.to(self.device) if hasattr(value, "to") else value
        return moved_inputs

    def _clean_answer(self, answer: str, prompt: str) -> str:
        cleaned = re.sub(r"<extra_id_\d+>", "", answer)
        cleaned = cleaned.replace("<pad>", "").replace("</s>", "")
        cleaned = cleaned.strip()

        if cleaned.startswith(prompt):
            cleaned = cleaned[len(prompt):].strip()

        cleaned = re.sub(r"^(Answer|回答)\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = self._remove_prompt_artifacts(cleaned)
        cleaned = self._normalize_whitespace(cleaned)
        cleaned = self._normalize_punctuation(cleaned)
        cleaned = self._format_sentences(cleaned)
        cleaned = self._filter_answer_by_topic(cleaned, prompt)

        if not cleaned:
            return "Unable to answer from the provided context."

        return cleaned

    def _openai_compatible_generate(self, prompt: str) -> str:
        client = self._get_openai_compatible_client()
        try:
            response = client.responses.create(
                model=self.model_name,
                input=prompt,
                max_output_tokens=self.max_new_tokens,
            )
            answer = getattr(response, "output_text", None)
            if not answer:
                payload = response.model_dump() if hasattr(response, "model_dump") else {}
                answer = self._extract_output_text(payload)
            return self._clean_answer(answer or "", prompt)
        except Exception as exc:
            raise RuntimeError(
                f"Compatible API generation failed for '{self.model_name}'. "
                f"Check your API key, base_url, quota, and provider availability. Original error: {exc}"
            ) from exc

    def _get_openai_compatible_client(self):
        if self._remote_client is not None:
            return self._remote_client

        if not self.api_key_env:
            raise RuntimeError("Missing api_key_env for openai_compatible mode")

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {self.api_key_env}")

        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("The 'openai' package is required for compatible API generation.") from exc

        client_kwargs = {"api_key": api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._remote_client = OpenAI(**client_kwargs)
        return self._remote_client

    def _extract_output_text(self, payload: dict) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        parts = []
        for item in payload.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    parts.append(content["text"])
        return "\n".join(parts).strip()

    def _remove_prompt_artifacts(self, text: str) -> str:
        markers = [
            r"\[Document\s+\d+\]\s*source=",
            r"\[Context\s+\d+\]",
            r"User Question\s*:",
            r"Question\s*:",
            r"Context\s*:",
            r"Answer\s*:",
            r"\bAssistant\s*:",
            r"\bUser\s*:",
            r"\bSystem\s*:",
            r"\bHuman resources department\s*:",
        ]

        earliest = None
        for pattern in markers:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match and match.start() > 0:
                earliest = match.start() if earliest is None else min(earliest, match.start())

        if earliest is not None:
            return text[:earliest].rstrip()

        return text

    def _normalize_whitespace(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = re.sub(r"(?<![.!?。！？；;:：])\n(?!\n)", " ", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
        return text.strip()

    def _normalize_punctuation(self, text: str) -> str:
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"\s+([，。；：！？])", r"\1", text)
        text = re.sub(r"([，。；：！？])([^\s\n])", r"\1\2", text)
        text = re.sub(r"([,.;:!?])([A-Za-z0-9])", r"\1 \2", text)
        text = re.sub(r"([。！？；])\s+", r"\1", text)
        text = re.sub(r"([。！？；])(?=[^\s\n])", r"\1", text)
        text = re.sub(r"([.!?])\s*(\n)", r"\1\2", text)
        text = re.sub(r"([.!?])(?=[A-Z])", r"\1 ", text)
        text = re.sub(r"([.!?]) {2,}", r"\1 ", text)
        return text.strip()

    def _format_sentences(self, text: str) -> str:
        paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
        formatted = []
        for paragraph in paragraphs:
            normalized = paragraph.replace("\n", " ").strip()
            normalized = re.sub(r"\s+", " ", normalized)
            normalized = self._trim_trailing_incomplete_text(normalized)
            sentences = self._split_sentences(normalized)
            merged = self._merge_short_fragments(sentences)
            filtered = self._trim_off_topic_tail(merged)
            filtered = self._drop_unfinished_trailing_sentence(filtered)
            formatted.append(self._join_sentences(filtered))
        return "\n\n".join(part for part in formatted if part).strip()

    def _filter_answer_by_topic(self, text: str, prompt: str) -> str:
        paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
        if not paragraphs:
            return text

        topic_keywords = self._extract_topic_keywords(prompt)
        if not topic_keywords:
            return text

        filtered_paragraphs = []
        for paragraph in paragraphs:
            sentences = self._split_sentences(paragraph)
            filtered_sentences = []
            for idx, sentence in enumerate(sentences):
                if idx == 0:
                    filtered_sentences.append(sentence)
                    continue

                if self._looks_like_topic_match(sentence, topic_keywords):
                    filtered_sentences.append(sentence)
                    continue

                # Once the answer drifts off-topic, drop the remaining tail.
                break

            if filtered_sentences:
                filtered_paragraphs.append(self._join_sentences(filtered_sentences))

        return "\n\n".join(filtered_paragraphs).strip() or text

    def _extract_topic_keywords(self, prompt: str) -> set[str]:
        question_match = re.search(r"Question:\s*(.+?)\nAnswer:", prompt, flags=re.DOTALL | re.IGNORECASE)
        context_match = re.search(r"Context:\n(.+?)\n\nQuestion:", prompt, flags=re.DOTALL | re.IGNORECASE)

        keywords = set()
        if question_match:
            keywords.update(self._keywordize(question_match.group(1)))
        if context_match:
            keywords.update(self._keywordize(context_match.group(1), limit=40))
        return keywords

    def _keywordize(self, text: str, limit: int | None = None) -> set[str]:
        tokens = re.findall(r"[A-Za-z]{4,}", text.lower())
        keywords = []
        seen = set()
        for token in tokens:
            token = self._normalize_keyword(token)
            if token in STOPWORDS or token in seen:
                continue
            seen.add(token)
            keywords.append(token)
            if limit is not None and len(keywords) >= limit:
                break
        return set(keywords)

    def _normalize_keyword(self, token: str) -> str:
        if len(token) > 5 and token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
        return token

    def _looks_like_topic_match(self, sentence: str, topic_keywords: set[str]) -> bool:
        lowered = sentence.lower()
        if self._looks_like_chat_or_role_artifact(sentence):
            return False

        sentence_keywords = self._keywordize(sentence)
        overlap = sentence_keywords & topic_keywords
        if len(overlap) >= 2:
            return True

        if overlap and re.search(r"\b(it|this|these|they|technology|technique|method)\b", lowered):
            return True

        if re.match(r"^(it|this|these|they)\b", lowered):
            return True

        if re.match(r"^this (technology|technique|method)\b", lowered):
            return True

        return False

    def _split_sentences(self, text: str) -> list[str]:
        matches = re.findall(r".*?(?:[。！？；]|[.!?](?=\s|$)|$)", text)
        sentences = []
        for item in matches:
            sentence = item.strip()
            if sentence:
                sentences.append(sentence)
        return sentences

    def _merge_short_fragments(self, sentences: list[str]) -> list[str]:
        merged = []
        for sentence in sentences:
            if (
                merged
                and not self._ends_with_terminal_punctuation(merged[-1])
            ):
                merged[-1] = f"{merged[-1]} {sentence}".strip()
                continue

            if merged and self._is_short_fragment(sentence):
                merged[-1] = f"{merged[-1]} {sentence}".strip()
                continue

            merged.append(sentence)

        return [sentence for sentence in merged if sentence]

    def _join_sentences(self, sentences: list[str]) -> str:
        result = ""
        for sentence in sentences:
            if not result:
                result = sentence
                continue

            if self._should_join_without_space(result, sentence):
                result = f"{result}{sentence}"
            else:
                result = f"{result} {sentence}"

        return result.strip()

    def _should_join_without_space(self, previous: str, current: str) -> bool:
        return bool(
            re.search(r"[。！？；]$", previous)
            or re.match(r"^[\u4e00-\u9fff]", current)
        )

    def _is_short_fragment(self, text: str) -> bool:
        if re.search(r"[\u4e00-\u9fff]", text):
            return False
        words = text.split()
        if len(words) <= 3 and not self._ends_with_terminal_punctuation(text):
            return True
        return len(text) <= 12 and not self._ends_with_terminal_punctuation(text)

    def _ends_with_terminal_punctuation(self, text: str) -> bool:
        return bool(re.search(r"[.!?。！？；;]$", text))

    def _trim_off_topic_tail(self, sentences: list[str]) -> list[str]:
        trimmed = []
        for sentence in sentences:
            if self._looks_like_chat_or_role_artifact(sentence):
                break
            trimmed.append(sentence)
        return trimmed

    def _looks_like_chat_or_role_artifact(self, sentence: str) -> bool:
        lowered = sentence.lower()
        artifact_phrases = [
            "assistant:",
            "user:",
            "system:",
            "human resources department",
            "could you please specify your location",
            "different countries have varying laws",
            "employee rights",
            "labor market conditions",
        ]
        return any(phrase in lowered for phrase in artifact_phrases)

    def _drop_unfinished_trailing_sentence(self, sentences: list[str]) -> list[str]:
        if len(sentences) <= 1:
            return sentences

        last_sentence = sentences[-1]
        if (
            not self._ends_with_terminal_punctuation(last_sentence)
            and self._looks_like_unfinished_fragment(last_sentence)
        ):
            return sentences[:-1]

        return sentences

    def _looks_like_unfinished_fragment(self, sentence: str) -> bool:
        if re.search(r"[\u4e00-\u9fff]", sentence):
            return len(sentence) <= 12

        words = sentence.split()
        if not words:
            return True

        return len(words) <= 4

    def _trim_trailing_incomplete_text(self, text: str) -> str:
        if self._ends_with_terminal_punctuation(text):
            return text

        if re.search(r"[,，;；:：]\s*$", text):
            matches = list(re.finditer(r"[.!?。！？；;](?=\s|$)", text))
            if matches:
                return text[: matches[-1].end()].rstrip()
            return re.sub(r"[,，;；:：]\s*$", "", text).rstrip()

        matches = list(re.finditer(r"[.!?。！？；;](?=\s|$)", text))
        if not matches:
            return text

        last_match = matches[-1]
        trailing = text[last_match.end():].strip()
        if trailing and self._looks_like_unfinished_fragment(trailing):
            return text[: last_match.end()].rstrip()

        return text
