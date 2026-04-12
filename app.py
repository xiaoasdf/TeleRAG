import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from src.config import load_config
from src.generation.llm_client import DEFAULT_LLM_MODEL, FAST_LLM_MODEL
from src.pipeline.index_pipeline import build_chunks_from_file
from src.pipeline.qa_pipeline import QAPipeline


def is_discouraged_model_name(model_name: str) -> bool:
    lowered = model_name.lower()
    return "mt5-base" in lowered or "t5-base" in lowered


def build_pipeline(*, generation_backend: str, llm_model_name: str, enable_rerank: bool, max_new_tokens: int) -> QAPipeline:
    return QAPipeline(
        llm_mode="openai_compatible" if generation_backend == "dashscope_compatible" else "hf",
        llm_model_name=llm_model_name,
        model_name=APP_CONFIG.retriever_model,
        reranker_model_name=APP_CONFIG.reranker_model,
        enable_rerank=enable_rerank,
        candidate_k=APP_CONFIG.candidate_k,
        rerank_top_n=APP_CONFIG.rerank_top_n,
        prompt_char_budget=APP_CONFIG.prompt_char_budget,
        max_new_tokens=max_new_tokens,
        generation_backend=generation_backend,
        compatible_api_key_env=APP_CONFIG.dashscope_api_key_env,
        compatible_base_url=APP_CONFIG.dashscope_base_url,
    )


def get_vector_store_path(kb_slot: str) -> Path:
    if kb_slot == "上传库":
        return APP_CONFIG.vector_store_path.parent / "uploads"
    return APP_CONFIG.vector_store_path


def restore_persisted_pipeline(kb_slot: str) -> tuple[QAPipeline | None, str | None, int]:
    vector_store_path = get_vector_store_path(kb_slot)
    index_file = vector_store_path / "index.faiss"
    metadata_file = vector_store_path / "metadata.json"
    if not index_file.exists() or not metadata_file.exists():
        return None, None, 0

    pipeline = build_pipeline(
        generation_backend=st.session_state.get("generation_backend", APP_CONFIG.rag_provider),
        llm_model_name=st.session_state.get("llm_model_name", APP_CONFIG.dashscope_model),
        enable_rerank=st.session_state.get("enable_rerank", APP_CONFIG.enable_rerank),
        max_new_tokens=st.session_state.get("max_new_tokens", APP_CONFIG.max_new_tokens),
    )
    pipeline.load_knowledge_base(vector_store_path)
    summary = pipeline.get_source_summary()
    return pipeline, summary["label"], int(summary["chunk_count"])


def _read_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_corpus_sources(metadata: dict | None) -> list[str]:
    if not metadata:
        return []
    sources = []
    seen = set()
    for item in metadata.get("metadata", []):
        source = item.get("source")
        if not source or source in seen:
            continue
        seen.add(source)
        sources.append(source)
    return sources


def get_current_knowledge_base_summary(kb_slot: str) -> dict:
    if st.session_state.pipeline:
        summary = st.session_state.pipeline.get_source_summary()
        if summary["label"]:
            return summary

    persisted_metadata = _read_json_file(get_vector_store_path(kb_slot) / "metadata.json")
    sources = _extract_corpus_sources(persisted_metadata)
    chunk_count = len(persisted_metadata.get("metadata", [])) if persisted_metadata else 0
    return {
        "source_count": len(sources),
        "sources": sources,
        "chunk_count": chunk_count,
        "label": sources[0] if len(sources) == 1 and sources else (f"{len(sources)} sources" if sources else None),
    }


def _count_standard_raw_files(root: Path) -> int:
    if not root.exists():
        return 0
    supported = {".pdf", ".txt", ".md", ".doc", ".docx", ".zip"}
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in supported)


def _format_source_org_counts(ingest_state: dict | None) -> str:
    if not ingest_state:
        return "未生成"
    counts = {}
    for record in ingest_state.get("records", []):
        source_org = record.get("source_org", "unknown")
        counts[source_org] = counts.get(source_org, 0) + 1
    if not counts:
        return "未生成"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def get_standards_status_summary() -> dict:
    standards_root = APP_CONFIG.standards_source_path
    ingest_state = _read_json_file(APP_CONFIG.raw_data_path / "index_ready" / "standards_ingest_state.json")
    build_state = _read_json_file(APP_CONFIG.raw_data_path / "index_ready" / "standards_build_state.json")
    metadata = _read_json_file(APP_CONFIG.vector_store_path / "metadata.json")
    corpus_sources = _extract_corpus_sources(metadata)
    failed_doc_count = 0
    if ingest_state:
        for record in ingest_state.get("records", []):
            rel_path = str(record.get("source_rel_path", "")).lower()
            if rel_path.endswith(".doc") and record.get("status") == "failed":
                failed_doc_count += 1

    return {
        "standards_root": standards_root,
        "raw_files": _count_standard_raw_files(standards_root),
        "index_ready_docs": ingest_state.get("summary", {}).get("extracted_documents", 0) if ingest_state else 0,
        "persisted_chunks": len(metadata.get("metadata", [])) if metadata else 0,
        "corpus_sources": len(corpus_sources),
        "source_org_counts": _format_source_org_counts(ingest_state),
        "ingest_state": ingest_state or {},
        "build_state": build_state or {},
        "failed_doc_count": failed_doc_count,
    }


def _get_runtime_mode_hint(build_state: dict | None) -> str:
    if not build_state:
        return "当前 Windows 方案默认使用 GPU embeddings + CPU FAISS。"

    device = build_state.get("device", "unknown")
    index_backend = build_state.get("index_backend", "unknown")
    faiss_status = build_state.get("faiss_gpu_status", "unknown")
    return f"当前运行模式：Device={device} | FAISS={index_backend} | faiss_gpu_status={faiss_status}"


def current_llm_presets(generation_backend: str) -> dict[str, str | None]:
    if generation_backend == "dashscope_compatible":
        return {
            "百炼": APP_CONFIG.dashscope_model,
            "自定义": None,
        }
    return {
        "快速": APP_CONFIG.fast_llm_model,
        "平衡": APP_CONFIG.balanced_llm_model,
        "自定义": None,
    }


def default_model_name_for_backend(generation_backend: str) -> str:
    if generation_backend == "dashscope_compatible":
        return APP_CONFIG.dashscope_model
    return APP_CONFIG.default_llm_model


APP_CONFIG = load_config()

st.set_page_config(page_title="TeleRAG", page_icon="📚")

st.markdown(
    """
    <style>
    .app-shell {
        padding: 1rem 0 0.5rem 0;
    }
    .hero-card {
        background: linear-gradient(135deg, #f3f8ff 0%, #eef6ee 100%);
        border: 1px solid #d8e6dc;
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #17324d;
    }
    .hero-subtitle {
        color: #496176;
        font-size: 0.98rem;
    }
    .meta-text {
        color: #566b78;
        font-size: 0.92rem;
    }
    .answer-card {
        background: #ffffff;
        border: 1px solid #e5e8eb;
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        margin-top: 0.8rem;
        margin-bottom: 0.75rem;
    }
    .thinking-text {
        color: #667985;
        font-size: 0.9rem;
        margin-top: 0.7rem;
    }
    div[data-testid="stForm"] button[kind="secondaryFormSubmit"] {
        min-height: 2.8rem;
        border-radius: 999px;
        font-size: 1.1rem;
        padding: 0;
    }
    </style>
    <div class="app-shell"></div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">TeleRAG</div>
        <div class="hero-subtitle">面向通信知识库、标准资料与无线系统讲义的 RAG 问答工具，支持来源追踪、时延分析与可演示的本地实验链路。</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "generation_backend" not in st.session_state:
    st.session_state.generation_backend = APP_CONFIG.rag_provider
if "kb_slot" not in st.session_state:
    st.session_state.kb_slot = "标准库"
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "llm_model_name" not in st.session_state:
    st.session_state.llm_model_name = default_model_name_for_backend(st.session_state.generation_backend)
if "model_preset" not in st.session_state:
    st.session_state.model_preset = "百炼" if st.session_state.generation_backend == "dashscope_compatible" else "平衡"
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "last_submitted_query" not in st.session_state:
    st.session_state.last_submitted_query = None
if "enable_rerank" not in st.session_state:
    st.session_state.enable_rerank = APP_CONFIG.enable_rerank
if "max_new_tokens" not in st.session_state:
    st.session_state.max_new_tokens = APP_CONFIG.max_new_tokens
if "pipeline" not in st.session_state:
    restored_pipeline, restored_file, restored_chunk_count = restore_persisted_pipeline(st.session_state.kb_slot)
    st.session_state.pipeline = restored_pipeline
    st.session_state.current_file = restored_file
    st.session_state.chunk_count = restored_chunk_count

current_kb = get_current_knowledge_base_summary(st.session_state.kb_slot)
if current_kb["label"]:
    st.session_state.current_file = current_kb["label"]
    st.session_state.chunk_count = current_kb["chunk_count"]

st.caption(
    f"当前知识库槽位：{st.session_state.kb_slot} | 持久化目录：`{get_vector_store_path(st.session_state.kb_slot)}`"
)

standards_status = get_standards_status_summary()
with st.expander("标准库状态面板", expanded=False):
    st.caption(
        " | ".join(
            [
                f"source_root={standards_status['standards_root']}",
                f"raw_files={standards_status['raw_files']}",
                f"index_ready_docs={standards_status['index_ready_docs']}",
                f"persisted_chunks={standards_status['persisted_chunks']}",
                f"corpus_sources={standards_status['corpus_sources']}",
            ]
        )
    )
    st.caption(f"来源分布：{standards_status['source_org_counts']}")
    st.caption(_get_runtime_mode_hint(standards_status["build_state"]))

    ingest_state = standards_status["ingest_state"]
    ingest_summary = ingest_state.get("summary", {})
    if ingest_summary:
        st.caption(
            "入库层："
            f" processed={ingest_summary.get('processed_sources', 0)}"
            f" | new_or_updated={ingest_summary.get('new_or_updated_sources', 0)}"
            f" | failed={ingest_summary.get('failed_sources', 0)}"
            f" | extracted_docs={ingest_summary.get('extracted_documents', 0)}"
            f" | failed_doc={standards_status['failed_doc_count']}"
        )
    else:
        st.caption("入库层：尚未生成 standards ingest state。")

    build_state = standards_status["build_state"]
    if build_state:
        st.caption(
            "索引层："
            f" status={build_state.get('status', 'unknown')}"
            f" | stage={build_state.get('current_stage', 'unknown')}"
            f" | processed_files={build_state.get('processed_files', 0)}/{build_state.get('total_files', 0)}"
            f" | processed_batches={build_state.get('processed_batches', 0)}"
            f" | index_backend={build_state.get('index_backend', 'unknown')}"
            f" | index_backend_reason={build_state.get('index_backend_reason', 'unknown')}"
            f" | embedding_batch_size={build_state.get('embedding_batch_size', APP_CONFIG.embedding_batch_size)}"
        )
    else:
        st.caption("索引层：尚未生成 standards build state。")


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def vector_store_exists(path: Path) -> bool:
    return (path / "index.faiss").exists() and (path / "metadata.json").exists()


with st.sidebar:
    st.header("通信资料导入")

    kb_slot = st.radio(
        "知识库",
        options=["标准库", "上传库"],
        index=0 if st.session_state.kb_slot == "标准库" else 1,
        horizontal=True,
    )
    if kb_slot != st.session_state.kb_slot:
        st.session_state.kb_slot = kb_slot
        restored_pipeline, restored_file, restored_chunk_count = restore_persisted_pipeline(kb_slot)
        st.session_state.pipeline = restored_pipeline
        st.session_state.current_file = restored_file
        st.session_state.chunk_count = restored_chunk_count
        st.session_state.last_result = None
        st.session_state.last_submitted_query = None
        st.rerun()

    provider_label = st.radio(
        "回答后端",
        options=["百炼API", "本地模型"],
        index=0 if st.session_state.generation_backend == "dashscope_compatible" else 1,
        horizontal=True,
    )
    selected_backend = "dashscope_compatible" if provider_label == "百炼API" else "local"
    if selected_backend != st.session_state.generation_backend:
        st.session_state.generation_backend = selected_backend
        st.session_state.llm_model_name = default_model_name_for_backend(selected_backend)
        st.session_state.model_preset = "百炼" if selected_backend == "dashscope_compatible" else "平衡"
        if st.session_state.pipeline is not None:
            st.session_state.pipeline.switch_generation_backend(
                generation_backend=selected_backend,
                llm_model_name=st.session_state.llm_model_name,
                llm_mode="openai_compatible" if selected_backend == "dashscope_compatible" else "hf",
            )
        st.rerun()

    if st.session_state.generation_backend == "dashscope_compatible":
        if os.getenv(APP_CONFIG.dashscope_api_key_env):
            st.caption("已检测到百炼 API Key。")
        else:
            st.warning(f"未检测到 `{APP_CONFIG.dashscope_api_key_env}`，百炼问答会失败，可手动切到本地模型。")

    uploaded_files = st.file_uploader(
        "上传通信资料（支持 txt / md / pdf）",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        selected_names = [uploaded_file.name for uploaded_file in uploaded_files]
        preview_names = selected_names[:5]
        st.caption(f"已选择 {len(selected_names)} 个文件")
        for name in preview_names:
            st.caption(f"- {name}")
        hidden_count = len(selected_names) - len(preview_names)
        if hidden_count > 0:
            st.caption(f"其余 {hidden_count} 个文件未展开。")

    st.subheader("回答速度")
    fast_mode = False
    if st.session_state.generation_backend == "local":
        fast_mode = st.toggle(
            "fast_mode",
            value=APP_CONFIG.default_llm_model == FAST_LLM_MODEL,
            help="开启后默认使用更轻量的本地生成模型。",
        )
    model_presets = current_llm_presets(st.session_state.generation_backend)
    default_preset = ("快速" if fast_mode else st.session_state.model_preset)
    fallback_preset = "百炼" if st.session_state.generation_backend == "dashscope_compatible" else "平衡"
    model_preset = st.radio(
        "模型档位",
        options=list(model_presets.keys()),
        index=list(model_presets.keys()).index(default_preset if default_preset in model_presets else fallback_preset),
        horizontal=True,
    )
    preset_model_name = model_presets[model_preset]
    if preset_model_name is not None:
        llm_model_name = preset_model_name
        st.caption(f"当前预设模型：`{llm_model_name}`")
    else:
        llm_model_name = st.session_state.llm_model_name

    with st.expander("高级设置", expanded=False):
        if preset_model_name is None:
            llm_model_name = st.text_input(
                "自定义生成模型名称",
                value=st.session_state.llm_model_name,
                help="百炼模式填写兼容接口模型名；本地模式填写 HuggingFace 模型名或本地目录。",
            )
        else:
            st.caption("当前使用预设模型，无需填写自定义模型名称。")

        enable_rerank = st.toggle("启用 rerank", value=st.session_state.enable_rerank)
        max_new_tokens = st.slider("max_new_tokens", min_value=32, max_value=256, value=st.session_state.max_new_tokens, step=16)
        top_k = st.slider("top_k", min_value=1, max_value=5, value=APP_CONFIG.top_k)
        chunk_size = st.number_input("chunk_size", min_value=100, max_value=1000, value=APP_CONFIG.chunk_size, step=50)
        overlap = st.number_input("overlap", min_value=0, max_value=300, value=APP_CONFIG.overlap, step=10)

    if st.session_state.generation_backend == "dashscope_compatible":
        st.caption("当前使用百炼兼容 API 生成答案，本地负责建库和检索。Windows 方案默认是 GPU embeddings + CPU FAISS。")
    elif llm_model_name == DEFAULT_LLM_MODEL:
        st.caption("当前使用推荐本地模型。Windows 方案默认是 GPU embeddings + CPU FAISS。")
    elif is_discouraged_model_name(llm_model_name):
        st.warning("当前模型更偏预训练填空，不太适合直接做 RAG 问答。")
    else:
        st.caption("当前使用自定义本地模型。Windows 方案默认是 GPU embeddings + CPU FAISS。")

    build_button = st.button("构建知识库")
    clear_button = st.button("清空知识库")

    if clear_button:
        restored_pipeline, restored_file, restored_chunk_count = restore_persisted_pipeline(st.session_state.kb_slot)
        st.session_state.pipeline = restored_pipeline
        st.session_state.current_file = restored_file
        st.session_state.chunk_count = restored_chunk_count
        st.session_state.llm_model_name = llm_model_name
        st.session_state.model_preset = model_preset
        st.session_state.enable_rerank = enable_rerank
        st.session_state.max_new_tokens = max_new_tokens
        st.session_state.last_result = None
        st.session_state.last_submitted_query = None
        if restored_pipeline is not None:
            st.success("已清空当前问答状态，并恢复本地持久化知识库。")
        else:
            st.success("当前会话知识库已清空。")

    if build_button:
        if not uploaded_files:
            st.warning("请先上传至少一个文档。")
        else:
            try:
                with st.spinner("正在解析文档并构建知识库..."):
                    target_slot = "上传库"
                    target_store_path = get_vector_store_path(target_slot)
                    pipeline = build_pipeline(
                        generation_backend=st.session_state.generation_backend,
                        llm_model_name=llm_model_name,
                        enable_rerank=enable_rerank,
                        max_new_tokens=max_new_tokens,
                    )
                    if vector_store_exists(target_store_path):
                        pipeline.load_knowledge_base(target_store_path)
                    chunks = []
                    uploaded_names = []
                    for uploaded_file in uploaded_files:
                        temp_path = save_uploaded_file(uploaded_file)
                        uploaded_names.append(uploaded_file.name)
                        chunks.extend(
                            build_chunks_from_file(
                                temp_path,
                                chunk_size=chunk_size,
                                overlap=overlap,
                            )
                        )
                    existing_chunk_count = len(pipeline.retriever.chunks)
                    if pipeline.is_ready:
                        pipeline.add_knowledge_chunks(chunks)
                    else:
                        pipeline.build_knowledge_base(chunks)
                    pipeline.save_knowledge_base(target_store_path)

                st.session_state.pipeline = pipeline
                st.session_state.kb_slot = target_slot
                updated_summary = pipeline.get_source_summary()
                st.session_state.current_file = updated_summary["label"]
                st.session_state.chunk_count = updated_summary["chunk_count"]
                st.session_state.llm_model_name = llm_model_name
                st.session_state.model_preset = model_preset
                st.session_state.enable_rerank = enable_rerank
                st.session_state.max_new_tokens = max_new_tokens
                st.session_state.last_result = None
                st.session_state.last_submitted_query = None

                if len(uploaded_names) == 1:
                    st.success(f"知识库已增量更新：{uploaded_names[0]}")
                else:
                    st.success(f"知识库已增量更新：{len(uploaded_names)} 个文件")
                st.info(
                    f"本次新增 {len(chunks)} 个 chunks，当前上传库累计 {st.session_state.chunk_count} 个 chunks。"
                )
            except Exception as e:
                st.error(f"构建失败：{e}")

if st.session_state.pipeline is None:
    st.info("请先在左侧上传通信资料，并点击“构建知识库”。")

with st.form("qa_form", clear_on_submit=True):
    input_col, button_col = st.columns([14, 1.2], gap="small")
    with input_col:
        query = st.text_input(
            "请输入你的问题",
            placeholder="给 TeleRAG 发送消息",
            key="query_input",
            label_visibility="collapsed",
        )
    with button_col:
        ask_button = st.form_submit_button("↑", type="secondary", use_container_width=True)

if ask_button:
    if st.session_state.pipeline is None:
        st.warning("请先上传通信资料并构建知识库。")
    elif not query.strip():
        st.warning("请输入问题。")
    elif st.session_state.last_submitted_query == query.strip():
        st.info("相同问题已经提交过。修改问题后再提交，避免重复执行。")
    else:
        try:
            expected_model = st.session_state.llm_model_name
            if st.session_state.pipeline.generation_backend != st.session_state.generation_backend or st.session_state.pipeline.llm_client.model_name != expected_model:
                st.session_state.pipeline.switch_generation_backend(
                    generation_backend=st.session_state.generation_backend,
                    llm_model_name=expected_model,
                    llm_mode="openai_compatible" if st.session_state.generation_backend == "dashscope_compatible" else "hf",
                )

            with st.spinner("模型正在思考中..."):
                submitted_query = query.strip()
                st.session_state.last_result = st.session_state.pipeline.ask(
                    submitted_query,
                    top_k=top_k,
                    enable_rerank=enable_rerank,
                    max_new_tokens=max_new_tokens,
                )
                st.session_state.last_submitted_query = submitted_query
        except Exception as e:
            st.error(f"提问失败：{e}")

if st.session_state.last_result:
    result = st.session_state.last_result
    timings = result.get("timings", {})
    total_seconds = timings.get("total_ms", 0) / 1000

    with st.container():
        st.markdown('<div class="answer-card">', unsafe_allow_html=True)
        st.write(result["answer"])
        st.markdown(f'<div class="thinking-text">已思考 {total_seconds:.2f} 秒</div></div>', unsafe_allow_html=True)

    with st.expander("检索来源", expanded=False):
        for item in result["sources"]:
            source_line = f"source={item['source']} | chunk_id={item['chunk_id']} | score={item['score']:.4f}"
            if item.get("rerank_score") is not None:
                source_line += f" | rerank={item['rerank_score']:.4f}"
            st.markdown(f'<div class="meta-text">{source_line}</div>', unsafe_allow_html=True)
