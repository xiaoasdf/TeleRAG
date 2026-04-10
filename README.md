# 📡 TeleRAG：通信文档问答系统（RAG）

基于 Retrieval-Augmented Generation（RAG）的通信领域文档问答系统。  
支持文档上传、向量检索、上下文增强生成，并提供答案来源追溯。

---

## 🚀 项目简介

本项目实现了一个完整的 RAG 问答流程：

用户输入问题 → 检索相关文档片段 → 构建 Prompt → 调用大模型生成答案 → 返回答案 + 来源

适用于：

- 通信 / AI / 技术文档问答
- 知识库问答系统
- LLM + 检索增强应用场景

---

## 🧠 系统架构

```
用户问题  
↓  
向量检索（Retriever）  
↓  
Top-K 相关文本片段  
↓  
Prompt 构建（Prompt Builder）  
↓  
大模型生成（LLM）  
↓  
答案 + 来源返回  
```

---

## 🧩 核心功能

- 📄 文档上传（支持 txt / md / pdf）
- 🔍 文本切分（chunk + overlap）
- 🧠 向量化（BGE embedding）
- 📚 相似度检索（Top-K）
- 🤖 大模型生成（Flan-T5）
- 📌 来源追溯（chunk_id / score）
- 🌐 Web 界面（Streamlit）

---

## 🖥️ 使用方式

### 1️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

### 2️⃣ 启动 Web 页面

```bash
python -m streamlit run app.py
```

打开浏览器访问：

```
http://localhost:8501
```

### 3️⃣ 使用流程

1. 上传文档（txt / md / pdf）
2. 点击「构建知识库」
3. 输入问题
4. 查看回答 + 来源 + 检索片段

---

## 📂 项目结构

```
TeleRAG/
├── app.py                  # Streamlit 前端
├── data/                  # 原始文档
├── scripts/
│   ├── cli_chat.py        # 命令行问答
│   └── build_index.py     # 构建索引
├── src/
│   ├── pipeline/          # RAG 主流程
│   ├── retrieval/         # 检索模块
│   ├── generation/        # LLM 调用
│   └── ingestion/         # 文档处理
```

---

## 🔍 示例

**问题：**

> What is beamforming?

**回答：**

> Beamforming 是一种信号处理技术，通过控制天线阵列的相位和幅度，实现信号在特定方向上的增强。

**来源：**

- chunk_id: xxx | score: 0.89  
- chunk_id: xxx | score: 0.86  

---

## ⚙️ 技术栈

- Python
- HuggingFace Transformers
- SentenceTransformers（BGE）
- Streamlit
- 向量检索（FAISS / 内存索引）

---

## 🎯 项目亮点（面试重点）

- ✅ 完整实现 RAG Pipeline（检索 + 生成）
- ✅ 支持文档级知识库构建
- ✅ 引入 Top-K 检索提升回答准确性
- ✅ 提供答案来源追溯（可解释性）
- ✅ 使用 Prompt Engineering 控制模型行为
- ✅ 搭建 Web UI 提升交互体验

---

## 🚧 可扩展方向

- 多文档知识库
- GPU 加速 embedding
- 更强模型（Qwen / Mistral / LLaMA）
- 部署（Docker / 云服务）
- 多轮对话

---

## 👤 作者

- 你的名字