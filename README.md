# TeleRAG

TeleRAG 是一个面向通信知识库的本地 RAG 项目，覆盖文档导入、切块、向量检索、可选重排、答案生成、来源回溯、时延统计、离线 benchmark，以及最小 FastAPI 服务化能力。项目默认保留一组通信领域样例，支持从仓库内标准目录或外部目录构建标准知识库。

适合简历描述为：

> 实现了一套面向通信知识库的 RAG 问答系统，支持通信资料与标准文档导入、来源追踪、时延分析、离线评测和 API 服务化，并围绕检索质量与响应耗时做对比验证。

## Features

- 支持 `txt`、`md`、`pdf`、`docx`，以及部分 Windows 环境下可抽取的 `doc`
- 检索模型使用 `BAAI/bge-m3`，可选重排模型使用 `BAAI/bge-reranker-v2-m3`
- 默认提供 Streamlit、CLI、FastAPI 三个入口
- 提供 `retrieve_ms`、`rerank_ms`、`prompt_ms`、`generate_ms`、`total_ms` 分阶段耗时
- 支持持久化本地向量索引并在启动时恢复
- 提供通信领域样例评测集与 benchmark 脚本
- 支持标准资料下载、抽取、入库和状态跟踪

## Project Layout

- Web app: `app.py`
- API service: `src/main.py`
- CLI demo: `scripts/cli_chat.py`
- Benchmark: `scripts/run_benchmark.py`
- Standards downloader: `scripts/download_standards.py`
- Standards indexer: `scripts/index_standards.py`
- Sample eval set: `data/eval/communications_eval.json`
- Benchmark guide: `docs/benchmark.md`

## Install

主依赖文件是 `requirements-managed.txt`。

### Windows / Conda

```bat
setup_conda.bat
```

或手动：

```bash
conda create --prefix D:\conda_envs\telerag python=3.10 -y
conda activate D:\conda_envs\telerag
pip install -r requirements-managed.txt
```

### Linux / WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-managed.txt
```

如果使用 DashScope 兼容接口，还需要设置：

```bash
export DASHSCOPE_API_KEY=your_key
```

PowerShell:

```powershell
$env:DASHSCOPE_API_KEY="your_key"
```

## Run

### Streamlit

```bash
python -m streamlit run app.py
```

Windows:

```bat
run_app.bat
```

### API

```bash
python scripts/run_api.py
```

Windows:

```bat
run_api.bat
```

接口：

- `GET /health`
- `POST /index`
- `POST /index/standards`
- `POST /query`

### CLI

```bash
python scripts/cli_chat.py
```

CLI 会优先恢复已持久化知识库；如果没有可恢复索引，则回退到仓库内三份通信样例。

### Runtime Check

```bash
python scripts/check_runtime.py
```

## Sample Corpus And Benchmark

仓库内默认样例语料：

- `data/raw/beamforming.pdf`
- `data/raw/wireless_systems_overview.md`
- `data/raw/communications_standards_notes.txt`

运行 benchmark：

```bash
python scripts/run_benchmark.py
```

输出：

- `docs/benchmark_results.json`

评测集：

- `data/eval/communications_eval.json`

## Standards Corpus

默认标准源目录为仓库内：

- `data/raw/standards`

也可以通过配置文件或环境变量覆盖：

```bash
export TELERAG_STANDARDS_SOURCE_DIR=/path/to/standards
```

Windows / WSL 混合使用时，配置中的 Windows 盘符路径会自动尝试映射到 `/mnt/<drive>/...`。

### Download Standards

```bash
python scripts/download_standards.py --dry-run
python scripts/download_standards.py --source-org 3GPP --limit 20
```

下载目标来自 `config/standards_targets.json`，文件默认落到 `data/raw/standards/`。

### Build Standards Knowledge Base

```bash
python scripts/index_standards.py
python scripts/index_standards.py --source-org 3GPP
python scripts/index_standards.py --source-org ITU
```

标准入库流程会：

- 扫描标准源目录
- 将可索引正文整理到 `data/raw/index_ready/standards/`
- 生成状态文件：
  - `data/raw/index_ready/standards_ingest_state.json`
  - `data/raw/index_ready/standards_build_state.json`
- 将持久化索引写入 `data/vector_store/default`

说明：

- 当前目标源覆盖 `3GPP`、`ITU`、`ETSI`、`DVB`
- `docx` 会抽取成文本
- `doc` 在 Windows 下会尝试调用 Word 自动化；失败会在状态文件中标记

## Example API Usage

### Build Sample Index

```bash
curl -X POST http://127.0.0.1:8000/index \
  -H "Content-Type: application/json" \
  -d "{\"file_paths\":[\"data/raw/beamforming.pdf\",\"data/raw/wireless_systems_overview.md\",\"data/raw/communications_standards_notes.txt\"],\"persist\":true}"
```

### Build Standards Corpus

```bash
curl -X POST http://127.0.0.1:8000/index/standards \
  -H "Content-Type: application/json" \
  -d "{\"download_first\":false,\"source_orgs\":[\"3GPP\"],\"persist\":true}"
```

### Query

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"What does beamforming mean in wireless systems?\",\"top_k\":3,\"enable_rerank\":true}"
```

## Notes

- 配置集中在 `config/settings.yaml`
- 生成索引、中间抽取结果、IDE 文件和缓存默认不纳入版本控制
- `docs/benchmark_results.json` 是保留在仓库中的一次样例 benchmark 快照，不代表每次运行都会自动提交
