# TeleRAG Progress

## 项目概述

TeleRAG 从一个本地可运行的技术文档 RAG Demo，逐步演进为一个更适合写进简历和在面试中讲清楚的通信领域 LLM 工程项目。整个过程的重点不是单纯增加功能，而是持续补齐以下几个方向：

- 可用性：让页面、脚本、问答链路真正稳定可跑
- 可解释性：答案带来源、带阶段耗时、带配置回显
- 可优化性：可以定位性能瓶颈，并围绕检索、重排、生成做调优
- 可工程化：引入配置层、API 服务层、索引持久化和统一入口
- 可展示性：补 benchmark、实验说明、README 和简历叙事

## 阶段进展

### 1. 初始版本：打通基础 RAG 链路

完成内容：

- 支持 `txt`、`md`、`pdf` 文档导入
- 实现文档切块、向量检索、重排、答案生成的基础流程
- 提供 Streamlit 页面用于文档上传、构建知识库和提问
- 支持 CLI 问答脚本，便于脱离页面快速验证

项目状态：

- 已经可以完成“上传文档 -> 提问 -> 返回答案和来源”的端到端流程
- 但整体更偏课程作业 / 技术 demo，缺少评测、持久化、服务化和工程叙事

### 2. 回答质量与展示优化

完成内容：

- 调整默认生成模型，避免继续使用更不适合直接问答的 `mT5-base`
- 增强答案清洗逻辑，减少 prompt 泄漏、特殊 token 残留和跑题回答
- 给检索结果展示补充 `rerank_score`
- 改善页面展示样式，使回答、来源和检索片段更清晰

结果：

- 回答更自然，杂质更少
- 检索与重排结果更容易解释
- 页面从“功能堆叠”变成更适合演示的结构化展示

### 3. 性能分析与问答链路提速

完成内容：

- 在 `QAPipeline.ask()` 中增加阶段耗时统计：
  - `retrieve_ms`
  - `rerank_ms`
  - `prompt_ms`
  - `generate_ms`
  - `total_ms`
- 引入条件 rerank 策略，不再每次都强制重排
- 增加 `candidate_k`、`rerank_top_n`、`prompt_char_budget`、`max_new_tokens` 等参数
- 在 UI 中增加模型档位、`fast_mode`、`max_new_tokens` 和 rerank 开关
- 引入快档与平衡档模型配置，支持更低延迟的交互问答

结果：

- 项目开始具备“可分析性能瓶颈”的基础设施
- 可以讨论质量与延迟之间的权衡，而不是只说“能不能跑”
- 页面上可以直接看到问答每个阶段的耗时

### 4. Streamlit 交互稳定性修复

完成内容：

- 修复了表单内 `on_change` 回调导致的 `StreamlitInvalidFormCallbackError`
- 清理 `query_dirty` 相关状态逻辑
- 改用 `clear_on_submit=True` 处理输入框清空，避免 `st.session_state.query_input cannot be modified` 报错

结果：

- 页面提交流程更符合 Streamlit 的约束
- 减少了前端交互层的运行时错误
- 问答表单可以稳定演示，不容易因为状态管理问题中断

### 5. 简历项目化升级

完成内容：

- 新增统一配置层：
  - `src/config.py`
  - `config/settings.yaml`
- 新增最小 API 服务层：
  - `GET /health`
  - `POST /index`
  - `POST /query`
- 新增服务封装层 `TeleRAGService`
- 给向量索引增加保存与恢复能力，支持索引持久化
- 让 Streamlit、CLI、API 共用同一套核心服务与配置

结果：

- 项目从“本地 UI demo”升级为“有 Web、CLI、API 三个入口的系统”
- 支持知识库索引持久化，减少重复构建成本
- 具备更强的工程化形态，适合在简历里描述为系统项目

### 6. 评测闭环与实验文档补齐

完成内容：

- 新增样例评测集：
  - `data/eval/communications_eval.json`
- 新增 benchmark 脚本：
  - `scripts/run_benchmark.py`
- 新增实验说明文档：
  - `docs/benchmark.md`
- 新增 benchmark 结果输出：
  - `docs/benchmark_results.json`

当前评测指标：

- `retrieval_hit_rate`
- `answer_keyword_coverage`
- `avg_total_ms`

结果：

- 项目开始具备“可量化结果”的基础
- 后续可以直接把 benchmark 结果写进 README、简历和面试讲解

### 7. 脚本入口稳定性修复

完成内容：

- 新增 `scripts/bootstrap.py`
- 统一 `run_benchmark.py`、`run_api.py`、`cli_chat.py`、`build_index.py` 的入口行为
- 自动把项目根目录加入 `sys.path`
- 统一脚本内的数据路径、输出路径和配置路径解析
- 新增脚本入口测试 `tests/test_script_bootstrap.py`

结果：

- 修复了 `ModuleNotFoundError: No module named 'src'`
- 脚本不再依赖 IDE 手工设置 working directory
- 入口体验更稳定，适合演示和交付

### 8. 通信领域化重构

完成内容：

- 将项目主叙事从“通用技术文档 RAG”收束为“通信知识库助手”
- 重写 README、benchmark 文档和项目进展文档中的对外描述
- 将样例数据与评测集切换为通信领域语义：
  - `data/raw/wireless_systems_overview.md`
  - `data/raw/communications_standards_notes.txt`
  - `data/eval/communications_eval.json`
- 调整 Streamlit、CLI、API 示例中的默认问题与文案，使其围绕通信概念和标准资料展开

结果：

- 项目不再只是“用 beamforming 举例的通用 RAG demo”
- 通信领域定位变成贯穿 README、样例、评测、页面体验的一条主线
- 更适合在简历和面试中被讲成通信方向的 LLM 工程项目

### 9. 标准资料下载与协议全文入库

完成内容：

- 新增公开标准资料下载脚本：
  - `scripts/download_standards.py`
- 新增标准资料下载配置：
  - `config/standards_targets.json`
- 支持从公开 `3GPP` 目录批量展开下载任务，并补充精选 `ITU-R` 无线通信相关 Recommendation
- 新增标准资料处理与入库模块：
  - `src/standards/downloader.py`
  - `src/standards/ingest.py`
- 新增标准全文入库脚本：
  - `scripts/index_standards.py`
- 新增标准库 API：
  - `POST /index/standards`
- 将原始标准包与可索引正文分层存放：
  - `data/raw/standards/`
  - `data/raw/index_ready/standards/`
- 增加标准入库状态文件，支持增量识别与重复执行跳过

结果：

- 项目从“能问通信资料”进一步升级为“能把公开标准资料真正沉淀进知识库”
- `3GPP zip` 不再只是下载到本地，而是可以抽取正文后进入向量知识库
- 标准知识库的构建流程具备可复用、可增量更新、可 API 化触发的工程能力

### 10. Streamlit 自动恢复已持久化知识库

完成内容：

- 在 `app.py` 中补充本地索引恢复逻辑
- 启动页面时自动检测 `data/vector_store/default` 下的持久化索引
- 如果恢复成功，页面直接展示已加载知识库状态，并允许用户立即提问
- 增加恢复成功提示，明确当前知识库来源和 chunk 数
- 增加标准库状态面板，展示原始标准包、正文抽取结果和当前知识库 source 预览
- 修正页面状态统计逻辑，优先以持久化索引中的真实 source / chunk 数为准，避免旧 session 信息误导

结果：

- Streamlit 不再每次启动都必须重新上传文档才能问答
- 页面行为和 API 服务层的“持久化索引恢复”能力更一致
- 演示体验更顺畅，更接近真正可用的知识库助手
- 页面可以更直观地区分“标准包已下载”“正文已抽取”“知识库已更新”三个阶段

### 11. 大规模 3GPP 标准知识库构建完成

完成内容：

- 基于当前已下载的公开 `3GPP` 与 `ITU-R` 标准资料运行标准全文入库脚本
- 将可索引正文统一抽取到：
  - `data/raw/index_ready/standards/`
- 以标准正文重建持久化向量知识库：
  - `data/vector_store/default/index.faiss`
  - `data/vector_store/default/metadata.json`

本次构建结果：

- `processed_sources = 589`
- `new_or_updated_sources = 586`
- `skipped_unchanged_sources = 3`
- `failed_sources = 0`
- `extracted_documents = 509`
- `chunk_count = 591852`
- `source_count = 509`

结果：

- 项目已经从“小样本通信资料演示库”升级为“可查询数百份 3GPP 标准正文的通信知识库”
- 当前知识库不再局限于 `38101-*` 等少量文件，而是覆盖大量 `23_series`、`38_series` 等标准正文
- 标准资料入库能力已经从工程链路验证，推进到真正可用的大规模知识库构建结果

## 关键里程碑

### 代码层

- 完整 RAG 流程跑通
- 回答清洗与重排结果展示补齐
- 问答链路阶段耗时统计落地
- 配置层、API 层、索引持久化落地
- benchmark 与脚本入口治理完成
- 通信领域化叙事、样例与评测集收束完成
- 公开标准资料下载、协议全文抽取与批量入库链路落地
- Streamlit 支持自动恢复已持久化知识库
- 已完成一次大规模标准知识库构建，知识库规模达到 `509` 个 source / `591852` 个 chunks

### 文档层

- README 从“使用说明”升级为“简历友好叙事”
- 增加 benchmark 文档
- 增加本项目进展记录
- README 已补标准下载与标准全文入库的使用说明

### 测试层

- 增补 `qa_pipeline`、`llm_client`、`retriever`、`vector_store`、`api_service`、`script_bootstrap` 相关测试
- 增补 `standards_downloader`、`standards_ingest`、`standards_api` 相关测试
- 当前核心回归测试可通过

## 当前项目形态

截至目前，TeleRAG 已经具备以下特征：

- 是一个面向通信知识库问答的 RAG 系统，而不仅是简单 UI demo
- 支持多格式文档导入、检索、重排、回答生成、来源追踪
- 支持公开通信标准资料下载、正文抽取与批量入库
- 当前持久化知识库已经包含数百份 3GPP 标准正文
- 支持阶段耗时分析和部分性能调优
- 支持配置化、最小 API 服务、索引持久化和启动自动恢复
- 支持离线 benchmark 和实验文档沉淀
- 支持更稳定的脚本直接运行

## 当前遗留问题与后续方向

仍待继续推进的点：

- benchmark 仍依赖本地模型可用，离线无网环境下还需要更好的降级策略
- benchmark 数据集目前规模较小，更适合演示，不足以支撑系统化评测
- 标准全文入库目前优先支持 `pdf`、`txt`、`md` 和 `docx`，对更老的 `doc` 等格式还没有覆盖
- 大规模标准知识库构建目前仍采用全量重建模式，耗时较长，缺少真正的分批建库与更细粒度进度输出
- API 目前仍偏最小版本，缺少更完整的部署文档和更细的错误码设计
- README 里还可以继续补架构图、实验截图和更强的结果对比表

建议的下一阶段方向：

1. 扩大评测集到 20 到 50 条样本，覆盖定义类、流程类、参数类问题
2. 基于当前大规模标准知识库补一版真正可复现的 benchmark 结果表，并同步回 README
3. 将标准知识库构建改造成分批建库 / 分批持久化模式，降低长时间全量重建成本
4. 扩展标准入库对更多协议正文格式的支持，并评估索引规模控制策略
5. 增加 API 层更完整的自动化测试、部署说明和错误码设计
6. 补项目架构图、时延对比图和简历 bullet
7. 增加离线运行策略，降低对 Hugging Face 在线拉取的依赖

## 一句话总结

TeleRAG 已经从“本地可运行的技术文档 RAG Demo”演进为“具备配置化、服务化、持久化、评测能力、数百份 3GPP 标准正文入库能力和通信领域叙事基础的 LLM 工程项目”，目前最适合继续往“更强标准知识库工程化 + 可复现 benchmark + 可投递简历材料”方向收尾。
