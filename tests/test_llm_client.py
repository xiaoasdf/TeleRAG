import pytest

from src.generation.llm_client import DEFAULT_LLM_MODEL, LLMClient


def test_llm_client_default_model_name():
    assert LLMClient.__init__.__defaults__ == ("hf", DEFAULT_LLM_MODEL, 192, None)


def test_llm_client_mock():
    client = LLMClient(mode="mock")
    answer = client.generate("hello")
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_llm_client_moves_model_and_inputs_to_device(monkeypatch):
    class FakeTensor:
        def __init__(self):
            self.moved_to = None
            self.shape = (1, 3)

        def to(self, device):
            self.moved_to = device
            return self

        def __len__(self):
            return 3

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        pad_token = "<pad>"
        eos_token = "</s>"

        def __call__(self, prompt, return_tensors=None, truncation=None, max_length=None):
            return {"input_ids": FakeTensor()}

        def decode(self, tokens, skip_special_tokens=True):
            return "generated answer"

    class FakeSeq2SeqModel:
        def __init__(self):
            self.device = None

        def to(self, device):
            self.device = device
            return self

        def generate(self, **kwargs):
            return [[101, 102]]

    class FakeConfig:
        is_encoder_decoder = True
        architectures = ["MT5ForConditionalGeneration"]

    fake_model = FakeSeq2SeqModel()

    monkeypatch.setattr(
        "src.generation.llm_client.AutoTokenizer.from_pretrained",
        lambda model_name: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "src.generation.llm_client.AutoConfig.from_pretrained",
        lambda model_name: FakeConfig(),
    )
    monkeypatch.setattr(
        "src.generation.llm_client.AutoModelForSeq2SeqLM.from_pretrained",
        lambda model_name: fake_model,
    )

    client = LLMClient(mode="hf", device="cuda")
    assert client.generate("hello") == "generated answer"
    assert fake_model.device == "cuda"


def test_llm_client_seq2seq_is_lazy_loaded(monkeypatch):
    calls = {"tokenizer": 0, "config": 0, "seq2seq": 0, "causal": 0}

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        pad_token = "<pad>"
        eos_token = "</s>"

        def __call__(self, prompt, return_tensors=None, truncation=None, max_length=None):
            return {"input_ids": [[1, 2, 3]]}

        def decode(self, tokens, skip_special_tokens=True):
            return "generated answer"

    class FakeSeq2SeqModel:
        def generate(self, **kwargs):
            return [[101, 102]]

    class FakeConfig:
        is_encoder_decoder = True
        architectures = ["MT5ForConditionalGeneration"]

    def fake_load_tokenizer(model_name):
        calls["tokenizer"] += 1
        return FakeTokenizer()

    def fake_load_config(model_name):
        calls["config"] += 1
        return FakeConfig()

    def fake_load_seq2seq(model_name):
        calls["seq2seq"] += 1
        return FakeSeq2SeqModel()

    def fake_load_causal(model_name):
        calls["causal"] += 1
        raise AssertionError("causal path should not be used")

    monkeypatch.setattr("src.generation.llm_client.AutoTokenizer.from_pretrained", fake_load_tokenizer)
    monkeypatch.setattr("src.generation.llm_client.AutoConfig.from_pretrained", fake_load_config)
    monkeypatch.setattr("src.generation.llm_client.AutoModelForSeq2SeqLM.from_pretrained", fake_load_seq2seq)
    monkeypatch.setattr("src.generation.llm_client.AutoModelForCausalLM.from_pretrained", fake_load_causal)

    client = LLMClient(mode="hf")

    assert client._is_loaded is False
    assert client.generate("hello") == "generated answer"
    assert client.generate("hello again") == "generated answer"
    assert calls == {"tokenizer": 1, "config": 1, "seq2seq": 1, "causal": 0}


def test_llm_client_causal_lm_returns_only_new_tokens(monkeypatch):
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        pad_token = "<pad>"
        eos_token = "</s>"

        def __call__(self, prompt, return_tensors=None, truncation=None, max_length=None):
            return {"input_ids": [[10, 20, 30]]}

        def decode(self, tokens, skip_special_tokens=True):
            if tokens == [40, 50]:
                return "direct answer"
            return "unexpected"

    class FakeCausalModel:
        def generate(self, **kwargs):
            return [[10, 20, 30, 40, 50]]

    class FakeConfig:
        is_encoder_decoder = False
        architectures = ["Qwen2ForCausalLM"]

    monkeypatch.setattr(
        "src.generation.llm_client.AutoTokenizer.from_pretrained",
        lambda model_name: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "src.generation.llm_client.AutoConfig.from_pretrained",
        lambda model_name: FakeConfig(),
    )
    monkeypatch.setattr(
        "src.generation.llm_client.AutoModelForCausalLM.from_pretrained",
        lambda model_name: FakeCausalModel(),
    )

    client = LLMClient(mode="hf")
    assert client.generate("hello") == "direct answer"


def test_llm_client_unsupported_architecture(monkeypatch):
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        pad_token = "<pad>"
        eos_token = "</s>"

    class FakeConfig:
        is_encoder_decoder = False
        architectures = ["UnknownModel"]

    monkeypatch.setattr(
        "src.generation.llm_client.AutoTokenizer.from_pretrained",
        lambda model_name: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "src.generation.llm_client.AutoConfig.from_pretrained",
        lambda model_name: FakeConfig(),
    )

    client = LLMClient(mode="hf")
    with pytest.raises(RuntimeError) as exc_info:
        client.generate("hello")

    assert "Unsupported model architecture" in str(exc_info.value)


def test_llm_client_tokenizer_error_message(monkeypatch):
    def fail_tokenizer(model_name):
        raise OSError("Can't load tokenizer")

    monkeypatch.setattr("src.generation.llm_client.AutoTokenizer.from_pretrained", fail_tokenizer)

    client = LLMClient(mode="hf")
    with pytest.raises(RuntimeError) as exc_info:
        client.generate("hello")

    message = str(exc_info.value)
    assert "Failed to load the HuggingFace tokenizer" in message
    assert "sentencepiece" in message
    assert "protobuf" in message
    assert "local directory with the same name" in message
    assert "Can't load tokenizer" in message


def test_llm_client_removes_special_tokens(monkeypatch):
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        pad_token = "<pad>"
        eos_token = "</s>"

        def __call__(self, prompt, return_tensors=None, truncation=None, max_length=None):
            return {"input_ids": [[1, 2, 3]]}

        def decode(self, tokens, skip_special_tokens=True):
            return "<extra_id_0> beamforming </s>"

    class FakeSeq2SeqModel:
        def generate(self, **kwargs):
            return [[101, 102]]

    class FakeConfig:
        is_encoder_decoder = True
        architectures = ["MT5ForConditionalGeneration"]

    monkeypatch.setattr(
        "src.generation.llm_client.AutoTokenizer.from_pretrained",
        lambda model_name: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "src.generation.llm_client.AutoConfig.from_pretrained",
        lambda model_name: FakeConfig(),
    )
    monkeypatch.setattr(
        "src.generation.llm_client.AutoModelForSeq2SeqLM.from_pretrained",
        lambda model_name: FakeSeq2SeqModel(),
    )

    client = LLMClient(mode="hf")
    assert client.generate("hello") == "beamforming"


def test_llm_client_normalizes_broken_english_sentences(monkeypatch):
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        pad_token = "<pad>"
        eos_token = "</s>"

        def __call__(self, prompt, return_tensors=None, truncation=None, max_length=None):
            return {"input_ids": [[1, 2, 3]]}

        def decode(self, tokens, skip_special_tokens=True):
            return "Answer: Beamforming focuses energy\n toward a target.It improves signal quality\nand reduces interference"

    class FakeSeq2SeqModel:
        def generate(self, **kwargs):
            return [[101, 102]]

    class FakeConfig:
        is_encoder_decoder = True
        architectures = ["MT5ForConditionalGeneration"]

    monkeypatch.setattr(
        "src.generation.llm_client.AutoTokenizer.from_pretrained",
        lambda model_name: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "src.generation.llm_client.AutoConfig.from_pretrained",
        lambda model_name: FakeConfig(),
    )
    monkeypatch.setattr(
        "src.generation.llm_client.AutoModelForSeq2SeqLM.from_pretrained",
        lambda model_name: FakeSeq2SeqModel(),
    )

    client = LLMClient(mode="hf")
    assert (
        client.generate("hello")
        == "Beamforming focuses energy toward a target. It improves signal quality and reduces interference"
    )


def test_llm_client_normalizes_broken_chinese_sentences(monkeypatch):
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        pad_token = "<pad>"
        eos_token = "</s>"

        def __call__(self, prompt, return_tensors=None, truncation=None, max_length=None):
            return {"input_ids": [[1, 2, 3]]}

        def decode(self, tokens, skip_special_tokens=True):
            return "回答：波束成形 可以把信号能量集中到特定方向 。\n这样 能增强目标方向信号 并减少干扰"

    class FakeSeq2SeqModel:
        def generate(self, **kwargs):
            return [[101, 102]]

    class FakeConfig:
        is_encoder_decoder = True
        architectures = ["MT5ForConditionalGeneration"]

    monkeypatch.setattr(
        "src.generation.llm_client.AutoTokenizer.from_pretrained",
        lambda model_name: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "src.generation.llm_client.AutoConfig.from_pretrained",
        lambda model_name: FakeConfig(),
    )
    monkeypatch.setattr(
        "src.generation.llm_client.AutoModelForSeq2SeqLM.from_pretrained",
        lambda model_name: FakeSeq2SeqModel(),
    )

    client = LLMClient(mode="hf")
    assert client.generate("hello") == "波束成形可以把信号能量集中到特定方向。这样能增强目标方向信号并减少干扰"


def test_llm_client_strips_prompt_artifacts_from_answer(monkeypatch):
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        pad_token = "<pad>"
        eos_token = "</s>"

        def __call__(self, prompt, return_tensors=None, truncation=None, max_length=None):
            return {"input_ids": [[1, 2, 3]]}

        def decode(self, tokens, skip_special_tokens=True):
            return (
                "Beamforming directs signals toward a target. "
                "[Document 2] source=example.pdf User Question: What does beamforming mean?"
            )

    class FakeSeq2SeqModel:
        def generate(self, **kwargs):
            return [[101, 102]]

    class FakeConfig:
        is_encoder_decoder = True
        architectures = ["MT5ForConditionalGeneration"]

    monkeypatch.setattr(
        "src.generation.llm_client.AutoTokenizer.from_pretrained",
        lambda model_name: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "src.generation.llm_client.AutoConfig.from_pretrained",
        lambda model_name: FakeConfig(),
    )
    monkeypatch.setattr(
        "src.generation.llm_client.AutoModelForSeq2SeqLM.from_pretrained",
        lambda model_name: FakeSeq2SeqModel(),
    )

    client = LLMClient(mode="hf")
    assert client.generate("hello") == "Beamforming directs signals toward a target."


def test_llm_client_trims_hr_chat_tail(monkeypatch):
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        pad_token = "<pad>"
        eos_token = "</s>"

        def __call__(self, prompt, return_tensors=None, truncation=None, max_length=None):
            return {"input_ids": [[1, 2, 3]]}

        def decode(self, tokens, skip_special_tokens=True):
            return (
                "Beamforming is a signal processing technique used to focus waves toward a desired direction. "
                "It is widely used in radar and wireless communications. "
                "Human resources department: Assistant: To provide you with accurate information about human resource management practices, "
                "could you please specify your location?"
            )

    class FakeSeq2SeqModel:
        def generate(self, **kwargs):
            return [[101, 102]]

    class FakeConfig:
        is_encoder_decoder = True
        architectures = ["MT5ForConditionalGeneration"]

    monkeypatch.setattr(
        "src.generation.llm_client.AutoTokenizer.from_pretrained",
        lambda model_name: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "src.generation.llm_client.AutoConfig.from_pretrained",
        lambda model_name: FakeConfig(),
    )
    monkeypatch.setattr(
        "src.generation.llm_client.AutoModelForSeq2SeqLM.from_pretrained",
        lambda model_name: FakeSeq2SeqModel(),
    )

    client = LLMClient(mode="hf")
    assert (
        client.generate("hello")
        == "Beamforming is a signal processing technique used to focus waves toward a desired direction. It is widely used in radar and wireless communications."
    )


def test_llm_client_drops_unfinished_trailing_sentence(monkeypatch):
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        pad_token = "<pad>"
        eos_token = "</s>"

        def __call__(self, prompt, return_tensors=None, truncation=None, max_length=None):
            return {"input_ids": [[1, 2, 3]]}

        def decode(self, tokens, skip_special_tokens=True):
            return (
                "Beamforming is a signal processing technique used to steer energy in a desired direction. "
                "It improves signal strength and reduces interference. These features"
            )

    class FakeSeq2SeqModel:
        def generate(self, **kwargs):
            return [[101, 102]]

    class FakeConfig:
        is_encoder_decoder = True
        architectures = ["MT5ForConditionalGeneration"]

    monkeypatch.setattr(
        "src.generation.llm_client.AutoTokenizer.from_pretrained",
        lambda model_name: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "src.generation.llm_client.AutoConfig.from_pretrained",
        lambda model_name: FakeConfig(),
    )
    monkeypatch.setattr(
        "src.generation.llm_client.AutoModelForSeq2SeqLM.from_pretrained",
        lambda model_name: FakeSeq2SeqModel(),
    )

    client = LLMClient(mode="hf")
    assert (
        client.generate("hello")
        == "Beamforming is a signal processing technique used to steer energy in a desired direction. It improves signal strength and reduces interference."
    )


def test_llm_client_drops_comma_terminated_tail(monkeypatch):
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        pad_token = "<pad>"
        eos_token = "</s>"

        def __call__(self, prompt, return_tensors=None, truncation=None, max_length=None):
            return {"input_ids": [[1, 2, 3]]}

        def decode(self, tokens, skip_special_tokens=True):
            return (
                "Beamforming improves signal directionality in wireless systems. "
                "It helps focus energy toward desired users. Additionally, HRMS often includes modules for payroll administration,"
            )

    class FakeSeq2SeqModel:
        def generate(self, **kwargs):
            return [[101, 102]]

    class FakeConfig:
        is_encoder_decoder = True
        architectures = ["MT5ForConditionalGeneration"]

    monkeypatch.setattr(
        "src.generation.llm_client.AutoTokenizer.from_pretrained",
        lambda model_name: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "src.generation.llm_client.AutoConfig.from_pretrained",
        lambda model_name: FakeConfig(),
    )
    monkeypatch.setattr(
        "src.generation.llm_client.AutoModelForSeq2SeqLM.from_pretrained",
        lambda model_name: FakeSeq2SeqModel(),
    )

    client = LLMClient(mode="hf")
    assert (
        client.generate("hello")
        == "Beamforming improves signal directionality in wireless systems. It helps focus energy toward desired users."
    )
