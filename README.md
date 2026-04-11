# TeleRAG

TeleRAG 是一个面向技术文档问答的 RAG 系统，目标不是只做“能跑通的 demo”，而是实现一条可评测、可优化、可展示的 LLM 工程链路：文档导入、切块、向量检索、可选重排、答案生成、来源回溯、性能统计、离线 benchmark，以及最小 API 服务化能力。

这个项目适合在简历中描述为：

> 实现并优化了一套面向技术文档的 RAG 问答系统，支持多格式文档导入、检索重排、来源追踪与时延分析，并通过离线评测对检索命中率和响应耗时进行对比验证。

## Problem And System Design

### 输入

- `txt`
- `md`
- `pdf`

### 核心问题

- 如何从技术文档中检索到足够相关的上下文
- 如何在回答质量和响应时延之间做平衡
- 如何让 RAG 项目具备可量化结果，而不是停留在 UI demo

### 系统链路

```text
Document -> Chunking -> Embedding Retrieval -> Optional Rerank -> Prompt Builder
-> LLM Generation -> Answer Cleanup -> Sources + Stage Timings
```

## Resume-Friendly Highlights

- 支持技术文档导入、切块、向量检索、重排、生成回答的完整 RAG 流程
- 使用 `BAAI/bge-m3` 进行检索，`BAAI/bge-reranker-v2-m3` 进行可选重排
- 支持快档与平衡档生成模型，默认面向低延迟问答场景
- 支持中英文技术问答与回答后处理，减少 prompt 泄漏和主题漂移
- 暴露分阶段时延指标：`retrieve_ms`、`rerank_ms`、`prompt_ms`、`generate_ms`、`total_ms`
- 新增索引持久化，避免每次启动都重复构建向量库
- 提供 Streamlit 页面、CLI demo、FastAPI 服务三种入口
- 提供离线 benchmark 脚本与评测集，支持效果与性能对比

## Key Project Outputs

- Web app: `app.py`
- API service: `src/main.py`
- Benchmark script: `scripts/run_benchmark.py`
- Evaluation set: `data/eval/beamforming_eval.json`
- Experiment guide: `docs/benchmark.md`

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Run Streamlit

```bash
python -m streamlit run app.py
```

访问 `http://localhost:8501`

### 3. Run API

```bash
python scripts/run_api.py
```

接口：

- `GET /health`
- `POST /index`
- `POST /query`

### 4. Run CLI Demo

```bash
python scripts/cli_chat.py
```

### 5. Run Benchmark

```bash
python scripts/run_benchmark.py
```

输出文件：

- `docs/benchmark_results.json`

## Example API Usage

### Build Index

```bash
curl -X POST http://127.0.0.1:8000/index ^
  -H "Content-Type: application/json" ^
  -d "{\"file_path\":\"data/raw/test.pdf\",\"persist\":true}"
```

### Query

```bash
curl -X POST http://127.0.0.1:8000/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is beamforming?\",\"top_k\":3,\"enable_rerank\":true}"
```

## Evaluation And Benchmarking

项目内置了一个轻量评测闭环：

- 样例评测集：`data/eval/beamforming_eval.json`
- 评测指标：
  - `retrieval_hit_rate`
  - `answer_keyword_coverage`
  - `avg_total_ms`
- 对比配置：
  - 无重排、低延迟基线
  - 启用重排的平衡配置

更详细的实验说明见 `docs/benchmark.md`。

## Engineering Notes

- 配置集中在 `config/settings.yaml`
- API、CLI、评测脚本复用同一个服务层，避免逻辑分叉
- 向量索引可持久化到 `data/vector_store/default`
- 模型首次推理时懒加载，适合本地实验环境

> 自主实现面向技术文档的 RAG 问答系统，基于 BGE 检索与重排、Qwen 指令模型生成回答，支持 PDF/Markdown/TXT 导入、来源追踪、阶段耗时分析、索引持久化与离线 benchmark，对检索效果与响应延迟进行对比评测。
