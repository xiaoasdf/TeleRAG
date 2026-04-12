# TeleRAG

TeleRAG 是一个面向通信知识库的 RAG 系统，重点服务于无线通信基础概念、通信标准资料和部分科研讲义的本地问答场景。项目目标不是只做“能跑通的 demo”，而是实现一条可评测、可优化、可展示的 LLM 工程链路：文档导入、切块、向量检索、可选重排、答案生成、来源回溯、性能统计、离线 benchmark，以及最小 API 服务化能力。

这个项目适合在简历中描述为：

> 实现并优化了一套面向通信知识库的 RAG 问答系统，支持无线通信资料与标准文档导入、检索重排、来源追踪与时延分析，并通过离线评测对检索命中率和响应耗时进行对比验证。

## Problem And System Design

### 输入

- `txt`
- `md`
- `pdf`
- `docx`
- 部分 `doc` 会在 Windows 环境尝试抽取，失败时会显式记录

### 核心问题

- 如何从通信资料、标准说明和无线系统讲义中检索到足够相关的上下文
- 如何在回答质量和响应时延之间做平衡
- 如何让通信领域 RAG 项目具备可量化结果，而不是停留在 UI demo

### 系统链路

```text
Document -> Chunking -> Embedding Retrieval -> Optional Rerank -> Prompt Builder
-> LLM Generation -> Answer Cleanup -> Sources + Stage Timings
```

## Resume-Friendly Highlights

- 支持通信讲义、标准笔记、PDF 论文等资料导入、切块、向量检索、重排、生成回答的完整 RAG 流程
- 使用 `BAAI/bge-m3` 进行检索，`BAAI/bge-reranker-v2-m3` 进行可选重排
- 支持快档与平衡档生成模型，默认面向低延迟问答场景
- 支持中英文通信问答与回答后处理，减少 prompt 泄漏和主题漂移
- 暴露分阶段时延指标：`retrieve_ms`、`rerank_ms`、`prompt_ms`、`generate_ms`、`total_ms`
- 新增索引持久化，避免每次启动都重复构建向量库
- 提供 Streamlit 页面、CLI demo、FastAPI 服务三种入口
- 提供离线 benchmark 脚本与通信领域评测集，支持效果与性能对比
- 支持从外部标准资料目录构建标准知识库，当前默认源为 `D:\project\pythoncrawler\downloads`

## Key Project Outputs

- Web app: `app.py`
- API service: `src/main.py`
- Benchmark script: `scripts/run_benchmark.py`
- Standards downloader: `scripts/download_standards.py`
- Standards indexer: `scripts/index_standards.py`
- Evaluation set: `data/eval/communications_eval.json`
- Experiment guide: `docs/benchmark.md`

## Quick Start

### 1. Install

推荐先创建固定位置的 Conda 环境：

```bash
conda create --prefix D:\conda_envs\telerag python=3.10 -y
conda activate D:\conda_envs\telerag
```

Windows 下也可以直接运行：

```bat
setup_conda.bat
```

它会自动在 `D:\conda_envs\telerag` 创建环境并安装 `requirements-managed.txt`。

推荐优先使用轻量管理版依赖：

```bash
pip install -r requirements-managed.txt
```

然后设置：

```bash
export DASHSCOPE_API_KEY=your_key
```

或在 Windows PowerShell 中：

```powershell
$env:DASHSCOPE_API_KEY="your_key"
```

如需旧的本地重依赖组合：

```bash
pip install -r requirements.txt
```

Windows 路线当前推荐形态：

- `torch` 使用 CUDA 版
- `FAISS` 使用 CPU 版
- 标准库重建时通常表现为 `GPU embeddings + CPU FAISS`

### 2. Run Streamlit

```bash
python -m streamlit run app.py
```

或直接运行：

```bat
run_app.bat
```

访问 `http://localhost:8501`

### 3. Run API

```bash
python scripts/run_api.py
```

或直接运行：

```bat
run_api.bat
```

这些脚本支持直接运行，不依赖手工设置 IDE 的 working directory。

接口：

- `GET /health`
- `POST /index`
- `POST /index/standards`
- `POST /query`

### 4. Run CLI Demo

```bash
python scripts/cli_chat.py
```

### 5. Run Benchmark

```bash
python scripts/run_benchmark.py
```

推荐从仓库根目录执行上述命令，但脚本入口本身也会自动解析项目根目录。

输出文件：

- `docs/benchmark_results.json`

### 6. Check Runtime

```bash
python scripts/check_runtime.py
```

这个脚本会输出当前环境的：

- `device`
- `faiss_gpu_status`
- `/health` 摘要

Windows 正常情况下应看到类似：

- `device = cuda`
- `faiss_gpu_status = faiss_gpu_symbols_missing`

这表示嵌入和本地模型走 GPU，但 `FAISS` 仍在 CPU。

### 7. Download Standards Corpus

```bash
python scripts/download_standards.py --dry-run
python scripts/download_standards.py --source-org 3GPP --limit 20
```

下载脚本仍然保留，默认下载目标位于 `config/standards_targets.json`，原始标准资料会落到：

- `data/raw/standards/3gpp/...`
- `data/raw/standards/itu/...`

但当前项目默认标准库来源已经切到外部目录：

- `D:\project\pythoncrawler\downloads`

该目录下的 `pdf / docx / doc` 会先抽取到 `data/raw/index_ready/standards/`，再构建标准知识库。

### 8. Build Standards Knowledge Base

```bash
python scripts/index_standards.py
python scripts/index_standards.py --source-org 3GPP
python scripts/index_standards.py --source-org ITU
```

标准全文入库时会：

- 扫描 `D:\project\pythoncrawler\downloads` 指向的标准源目录
- 清空旧标准库索引和旧 `index_ready` 中间产物
- 将可索引正文整理到 `data/raw/index_ready/standards/`
- 生成状态文件：
  - `data/raw/index_ready/standards_ingest_state.json`
  - `data/raw/index_ready/standards_build_state.json`
- 分批嵌入并持久化向量索引到 `data/vector_store/default`

说明：

- 当前标准源覆盖 `3GPP / ITU / ETSI / DVB`
- `docx` 会直接抽取为文本
- `doc` 会在 Windows 下尝试调用 Word 自动化；如果失败，会在状态文件里明确记为 `failed`

## Example API Usage

### Build Index

```bash
curl -X POST http://127.0.0.1:8000/index ^
  -H "Content-Type: application/json" ^
  -d "{\"file_path\":\"data/raw/beamforming.pdf\",\"persist\":true}"
```

### Build Standards Corpus

```bash
curl -X POST http://127.0.0.1:8000/index/standards ^
  -H "Content-Type: application/json" ^
  -d "{\"download_first\":false,\"source_orgs\":[\"3GPP\"],\"persist\":true}"
```

### Query

```bash
curl -X POST http://127.0.0.1:8000/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What does beamforming mean in wireless systems?\",\"top_k\":3,\"enable_rerank\":true}"
```

## Evaluation And Benchmarking

项目内置了一个通信领域的轻量评测闭环：

- 样例评测集：`data/eval/communications_eval.json`
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
- 默认 RAG provider 已切到 `dashscope_compatible`，文件上传、切块和检索仍在本地，答案生成走百炼兼容 API
- API、CLI、评测脚本复用同一个服务层，避免逻辑分叉
- 向量索引可持久化到 `data/vector_store/default`
- 模型首次推理时懒加载，适合本地实验环境
- Windows 当前推荐路线不是 GPU FAISS，而是 `GPU embeddings + CPU FAISS`
- 标准库状态面板会展示外部标准源目录、来源分布、`.doc` 失败数和当前索引阶段
- 标准资料下载脚本会把原始包和可索引文档分层存放，避免 zip、Word/PDF 源文件与索引目录混杂

> 自主实现面向通信知识库的 RAG 问答系统，基于 BGE 检索与重排、Qwen 指令模型生成回答，支持无线通信资料与标准文档导入、来源追踪、阶段耗时分析、索引持久化与离线 benchmark，对检索效果与响应延迟进行对比评测。
