import tempfile
from pathlib import Path

import streamlit as st

from src.generation.llm_client import DEFAULT_LLM_MODEL
from src.pipeline.index_pipeline import build_chunks_from_file
from src.pipeline.qa_pipeline import QAPipeline


def is_discouraged_model_name(model_name: str) -> bool:
    lowered = model_name.lower()
    return "mt5-base" in lowered or "t5-base" in lowered


st.set_page_config(page_title="TeleRAG", page_icon="📚")

st.title("📚 TeleRAG")
st.caption("RAG 问答系统，默认使用更适合中英技术问答的指令模型。")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "llm_model_name" not in st.session_state:
    st.session_state.llm_model_name = DEFAULT_LLM_MODEL


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

                st.success(f"知识库构建完成：{uploaded_file.name}")
                st.info(f"共生成 {len(chunks)} 个 chunks。首次提问时才会加载生成模型。")
            except Exception as e:
                st.error(f"构建失败：{e}")

st.subheader("问答区域")

if st.session_state.pipeline is None:
    st.info("请先在左侧上传文档，并点击“构建知识库”。")

if st.session_state.current_file:
    st.success(
        f"当前知识库：{st.session_state.current_file} | "
        f"chunks: {st.session_state.chunk_count} | "
        f"LLM: {st.session_state.llm_model_name}"
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
                result = st.session_state.pipeline.ask(query, top_k=top_k)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 回答")
                st.write(result["answer"])

                st.markdown("### 检索到的文本片段")
                for ctx in result["retrieved_contexts"]:
                    rerank_score = ctx.get("rerank_score")
                    title = (
                        f"{ctx['chunk_id']} | {ctx['source']} | score={ctx['score']:.4f}"
                    )
                    if rerank_score is not None:
                        title += f" | rerank={rerank_score:.4f}"

                    with st.expander(title):
                        st.write(ctx["text"])

            with col2:
                st.markdown("### 来源")
                for item in result["sources"]:
                    text = (
                        f"- **source**: {item['source']}\n"
                        f"  \n**chunk_id**: {item['chunk_id']}\n"
                        f"  \n**score**: {item['score']:.4f}"
                    )
                    if item.get("rerank_score") is not None:
                        text += f"\n  \n**rerank_score**: {item['rerank_score']:.4f}"

                    st.write(text)

                st.markdown("### Prompt（调试用）")
                with st.expander("查看 Prompt"):
                    st.text(result["prompt"])
        except Exception as e:
            st.error(f"提问失败：{e}")
