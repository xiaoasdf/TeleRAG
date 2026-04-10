import tempfile
from pathlib import Path

import streamlit as st

from src.generation.llm_client import DEFAULT_LLM_MODEL
from src.pipeline.index_pipeline import build_chunks_from_file
from src.pipeline.qa_pipeline import QAPipeline
from src.runtime import get_compute_device


def is_discouraged_model_name(model_name: str) -> bool:
    lowered = model_name.lower()
    return "mt5-base" in lowered or "t5-base" in lowered


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
    .status-strip {
        background: #fbfcf7;
        border: 1px solid #e2e7d7;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        margin: 0.8rem 0 1rem 0;
    }
    .section-card {
        background: #ffffff;
        border: 1px solid #e5e8eb;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.9rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1c3342;
        margin-bottom: 0.55rem;
    }
    .meta-text {
        color: #566b78;
        font-size: 0.92rem;
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
        <div class="hero-subtitle">面向技术文档的 RAG 问答工具，支持更快的 GPU 自动检测与更紧凑的三段式结果展示。</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "llm_model_name" not in st.session_state:
    st.session_state.llm_model_name = DEFAULT_LLM_MODEL
if "last_result" not in st.session_state:
    st.session_state.last_result = None


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


with st.sidebar:
    st.header("文档导入")

    uploaded_file = st.file_uploader(
        "上传文档（支持 txt / md / pdf）",
        type=["txt", "md", "pdf"],
    )

    chunk_size = st.number_input("chunk_size", min_value=100, max_value=1000, value=300, step=50)
    overlap = st.number_input("overlap", min_value=0, max_value=300, value=50, step=10)
    top_k = st.slider("top_k", min_value=1, max_value=5, value=3)
    llm_model_name = st.text_input(
        "生成模型名称",
        value=st.session_state.llm_model_name,
        help="推荐使用指令微调模型，例如 Qwen/Qwen2.5-1.5B-Instruct，也支持本地模型目录。",
    )

    if llm_model_name == DEFAULT_LLM_MODEL:
        st.caption("当前使用推荐模型。")
    elif is_discouraged_model_name(llm_model_name):
        st.warning("当前模型更偏预训练填空，不太适合直接做 RAG 问答。")
    else:
        st.caption("当前使用自定义模型。")

    build_button = st.button("构建知识库")
    clear_button = st.button("清空知识库")

    if clear_button:
        st.session_state.pipeline = None
        st.session_state.current_file = None
        st.session_state.chunk_count = 0
        st.session_state.llm_model_name = llm_model_name
        st.session_state.last_result = None
        st.success("知识库已清空。")

    if build_button:
        if uploaded_file is None:
            st.warning("请先上传一个文档。")
        else:
            try:
                with st.spinner("正在解析文档并构建知识库..."):
                    temp_path = save_uploaded_file(uploaded_file)
                    pipeline = QAPipeline(llm_mode="hf", llm_model_name=llm_model_name)
                    chunks = build_chunks_from_file(
                        temp_path,
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )
                    pipeline.build_knowledge_base(chunks)

                st.session_state.pipeline = pipeline
                st.session_state.current_file = uploaded_file.name
                st.session_state.chunk_count = len(chunks)
                st.session_state.llm_model_name = llm_model_name
                st.session_state.last_result = None

                st.success(f"知识库构建完成：{uploaded_file.name}")
                st.info(f"共生成 {len(chunks)} 个 chunks。首次提问时才会加载生成模型。")
            except Exception as e:
                st.error(f"构建失败：{e}")

st.subheader("问答区域")

if st.session_state.pipeline is None:
    st.info("请先在左侧上传文档，并点击“构建知识库”。")

device_label = get_compute_device()
vector_backend = "auto"
if st.session_state.pipeline and st.session_state.pipeline.retriever.vector_store:
    device_label = st.session_state.pipeline.device
    vector_backend = st.session_state.pipeline.retriever.vector_store.index_backend

if st.session_state.current_file:
    st.markdown(
        (
            '<div class="status-strip">'
            f"<strong>当前知识库</strong>: {st.session_state.current_file} &nbsp;|&nbsp; "
            f"<strong>chunks</strong>: {st.session_state.chunk_count} &nbsp;|&nbsp; "
            f"<strong>LLM</strong>: {st.session_state.llm_model_name} &nbsp;|&nbsp; "
            f"<strong>Device</strong>: {device_label} &nbsp;|&nbsp; "
            f"<strong>FAISS</strong>: {vector_backend}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        (
            '<div class="status-strip">'
            f"<strong>Device</strong>: {device_label} &nbsp;|&nbsp; "
            "<strong>FAISS</strong>: auto"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

with st.form("qa_form", clear_on_submit=False):
    query = st.text_input(
        "请输入你的问题：",
        placeholder="例如：what is beamforming / 什么是波束成形？",
    )
    ask_button = st.form_submit_button("提问")

if ask_button:
    if st.session_state.pipeline is None:
        st.warning("请先上传文档并构建知识库。")
    elif not query.strip():
        st.warning("请输入问题。")
    else:
        try:
            with st.spinner("模型正在思考中..."):
                st.session_state.last_result = st.session_state.pipeline.ask(query, top_k=top_k)
        except Exception as e:
            st.error(f"提问失败：{e}")

if st.session_state.last_result:
    result = st.session_state.last_result

    with st.container(border=True):
        st.markdown('<div class="section-title">回答</div>', unsafe_allow_html=True)
        st.write(result["answer"])

    with st.container(border=True):
        st.markdown('<div class="section-title">检索到的文本片段</div>', unsafe_allow_html=True)
        for ctx in result["retrieved_contexts"]:
            rerank_score = ctx.get("rerank_score")
            title = f"{ctx['chunk_id']} | {ctx['source']} | score={ctx['score']:.4f}"
            if rerank_score is not None:
                title += f" | rerank={rerank_score:.4f}"

            with st.expander(title):
                st.write(ctx["text"])

    with st.container(border=True):
        st.markdown('<div class="section-title">来源</div>', unsafe_allow_html=True)
        for item in result["sources"]:
            source_line = f"source={item['source']} | chunk_id={item['chunk_id']} | score={item['score']:.4f}"
            if item.get("rerank_score") is not None:
                source_line += f" | rerank={item['rerank_score']:.4f}"
            st.markdown(f'<div class="meta-text">{source_line}</div>', unsafe_allow_html=True)
