# TeleRAG

轻量级技术文档 RAG 项目，提供从文档导入、切块、向量检索、重排序到答案生成的一条完整链路，并附带可直接运行的 Streamlit 页面。

## Highlights

- 支持 `txt`、`md`、`pdf` 文档导入
- 使用 `BAAI/bge-m3` 做向量检索，`BAAI/bge-reranker-v2-m3` 做重排
- 默认使用 `Qwen/Qwen2.5-1.5B-Instruct` 生成答案，更适合中英双语问答
- 生成层同时兼容 Seq2Seq 与 Causal LM
- 首次提问时才会懒加载生成模型

## Recommended Models

- 默认推荐：`Qwen/Qwen2.5-1.5B-Instruct`
- 可替换为其他 HuggingFace 指令模型或本地模型目录
- 不推荐继续使用 `google/mt5-base` 作为默认问答模型，它更偏预训练填空，容易输出 `<extra_id_*>`

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

如果当前环境不能联网下载 Hugging Face 模型，请先手动缓存模型，或把界面中的模型名称改成本地模型目录。

### 2. Run Web App

```bash
python -m streamlit run app.py
```

打开 `http://localhost:8501`。

Windows 也可以直接双击项目根目录下的 `run_app.bat` 启动页面。

### 3. Run CLI Demo

```bash
python scripts/cli_chat.py
```

## Notes

- 默认模型是指令微调的双语模型，通常比 `mT5-base` 更适合 RAG 问答
- 首次提问会触发模型下载或加载，耗时取决于网络、磁盘和模型体积
- Windows 如果不支持 symlink，Hugging Face 仍可正常缓存，只是更占磁盘
