# TeleRAG

面向技术文档问答的轻量级 RAG 项目。当前版本提供从文档导入、切块、向量检索、重排序到答案生成的一条完整链路，并带有可直接运行的 Streamlit 页面。

## Highlights

- 支持 `txt`、`md`、`pdf` 文档导入
- 支持可调 `chunk_size` 与 `overlap`
- 使用 `SentenceTransformer` 做文本向量化
- 使用 FAISS 做 Top-K 相似度检索
- 使用 CrossEncoder 做检索结果 rerank
- 使用 `google/flan-t5-base` 生成最终答案
- 返回答案、来源、检索分数和 rerank 分数
- 提供 Web UI 和命令行示例

## Demo Flow

```text
Document Upload
    -> Document Loader
    -> Text Chunking
    -> Embedding
    -> FAISS Retrieval
    -> CrossEncoder Rerank
    -> Prompt Builder
    -> LLM Generation
    -> Answer + Sources
```

## Tech Stack

- Python
- Streamlit
- sentence-transformers
- FAISS
- HuggingFace Transformers
- PyPDF

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Run Web App

```bash
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

### 3. Run CLI Demo

```bash
python scripts/cli_chat.py
```

CLI 默认读取 `data/raw/test.pdf` 并进入交互式问答。

## How It Works

1. 用户上传一个 `txt`、`md` 或 `pdf` 文档
2. `loaders.py` 解析文档内容
3. `chunker.py` 按字符窗口切成多个 chunk
4. `embedder.py` 将 chunk 编码为向量
5. `vector_store.py` 用 FAISS 建立内存索引
6. `retriever.py` 召回 Top-K 相关片段
7. `reranker.py` 对召回结果重新排序
8. `prompt_builder.py` 组装上下文和问题
9. `llm_client.py` 生成答案
10. 页面展示答案、来源和检索片段

## Project Structure

```text
TeleRAG/
├── app.py
├── requirements.txt
├── README.md
├── config/
│   └── settings.yaml
├── data/
│   ├── raw/
│   └── vector_store/
├── docs/
│   └── plan.md
├── scripts/
│   ├── build_index.py
│   ├── cli_chat.py
│   ├── test_chunking.py
│   ├── test_faiss.py
│   ├── test_llm_client.py
│   ├── test_prompt.py
│   ├── test_qa_pipeline.py
│   └── test_retriever.py
├── src/
│   ├── config.py
│   ├── main.py
│   ├── generation/
│   │   ├── llm_client.py
│   │   └── prompt_builder.py
│   ├── ingestion/
│   │   ├── chunker.py
│   │   └── loaders.py
│   ├── pipeline/
│   │   ├── index_pipeline.py
│   │   └── qa_pipeline.py
│   └── retrieval/
│       ├── embedder.py
│       ├── reranker.py
│       ├── retriever.py
│       └── vector_store.py
└── tests/
    ├── test_chunker.py
    ├── test_embedder.py
    ├── test_llm_client.py
    ├── test_loaders.py
    ├── test_prompt_builder.py
    ├── test_qa_pipeline.py
    ├── test_retriever.py
    └── test_vector_store.py
```

## Core Modules

### `app.py`

Streamlit 前端入口，负责文件上传、参数输入、知识库构建和问答结果展示。

### `src/ingestion`

- `loaders.py`：读取 `txt / md / pdf`
- `chunker.py`：切分文本并生成带 metadata 的 chunks

### `src/retrieval`

- `embedder.py`：生成文本和查询向量
- `vector_store.py`：管理 FAISS 内存索引
- `retriever.py`：封装检索逻辑
- `reranker.py`：对初检结果进行重排

### `src/generation`

- `prompt_builder.py`：构造 Prompt
- `llm_client.py`：调用 HuggingFace 模型生成答案

### `src/pipeline`

- `index_pipeline.py`：从文件构建 chunks
- `qa_pipeline.py`：串联检索、重排、Prompt 和生成流程

## Testing

项目包含基础测试，可在本地运行：

```bash
pytest tests
```

## Current Scope

- 当前主要面向单文档问答
- 向量索引为内存实现，未做持久化
- `config/settings.yaml`、`src/config.py`、`src/main.py` 目前仍是预留文件
- 仓库中有 `src/api/`、`src/utils/` 目录，但当前没有实际代码

## Roadmap

- 多文档知识库
- 持久化向量存储
- 可配置模型与参数管理
- 基于 FastAPI 的接口层
- 多轮对话与上下文记忆
- 更强的中文或领域模型支持
