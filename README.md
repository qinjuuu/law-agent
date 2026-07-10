# 消费维权智能助手

基于 LangChain + RAG + 多智能体架构的消费者权益保护智能体系统。

## 功能概览

| 功能 | 说明 |
|------|------|
| 消费法律问答 | 基于 RAG 检索法律条文和案例，回答消费者权益问题 |
| 投诉信起草 | 根据纠纷描述检索法条依据，生成正式投诉信 Word 文档 |
| 格式条款审查 | 分析消费合同/格式条款中的不公平条款，生成审查报告 |
| 意图路由 | 自动识别用户意图，分发到对应专业智能体 |

## 技术栈

- **LangChain** — 多智能体编排框架，ReAct Agent 模式
- **FAISS** — 向量检索引擎，法律条文语义检索
- **BM25 + jieba** — 关键词检索，中文分词
- **混合检索** — 向量检索 70% + BM25 30% 加权融合
- **Gradio** — Web 交互界面
- **火山引擎 ARK (doubao)** — 大语言模型
- **BAAI/bge-small-zh** — 中文文本嵌入模型
- **python-docx** — Word 文档生成

## 项目结构

```
agent law/
├── app.py                      # Gradio 主入口（多 Tab 界面）
├── config.py                   # 配置管理
├── requirements.txt            # 依赖列表
├── .env.example                # 环境变量模板
│
├── agents/                     # 多智能体模块
│   ├── router.py               # 意图路由器
│   ├── qa_agent.py             # 消费法律问答智能体
│   ├── complaint_agent.py      # 投诉信起草智能体
│   ├── review_agent.py         # 格式条款审查智能体
│   └── intake_agent.py         # 信息采集向导
│
├── rag/                        # RAG 检索增强生成
│   ├── chunking.py             # 中文法律文本分块
│   ├── embedder.py             # 嵌入模型封装
│   ├── vector_store.py         # FAISS 向量存储
│   ├── bm25_retriever.py       # BM25 关键词检索
│   └── retriever.py            # 混合检索器
│
├── tools/                      # Agent 工具模块
│   ├── file_tools.py           # 文件操作（创建/读取/删除/压缩）
│   ├── word_tools.py           # Word 文档生成（投诉信/审查报告）
│   └── search_tools.py         # 法律检索（法条/案例）
│
├── data/                       # 知识库数据
│   ├── laws/                   # 法律条文
│   │   ├── 消费者权益保护法.txt
│   │   ├── 产品质量法.txt
│   │   ├── 电子商务法.txt
│   │   └── 食品安全法.txt
│   ├── cases/                  # 典型案例
│   ├── templates/              # 文书模板
│   └── vectors/                # 向量索引（自动生成）
│
├── workspace/                  # 运行时工作目录
│   ├── files/                  # Agent 文件操作沙箱
│   └── reports/                # 生成的 Word 文档
│
└── scripts/                    # 脚本
    └── ingest_data.py          # 数据向量化入库
```

## 快速开始

### 1. 安装依赖

```bash
cd "C:\Users\Q\PycharmProjects\agent law"
pip install -r requirements.txt
```
cd "C:\Users\Q\PycharmProjects\agent law"
.venv\Scripts\activate
pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple langchain langchain-openai langchain-community gradio python-docx faiss-cpu sentence-transformers jieba numpy python-dotenv

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填入火山引擎 ARK API Key:

```env
ARK_API_KEY=你的API密钥
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
MODEL_ID=doubao-seed-2-0-pro-260215
```

### 3. 数据入库（构建 RAG 索引）

```bash
python scripts/ingest_data.py
```

首次运行会下载嵌入模型（约 100MB），后续直接加载本地缓存。

### 4. 启动应用

```bash
python app.py
```

浏览器访问 `http://127.0.0.1:7860`

## 智能体架构

```
用户输入
    │
    ▼
┌──────────────┐
│ IntentRouter │  ← LLM 意图识别
└──────┬───────┘
       │
   ┌───┼───┐
   ▼   ▼   ▼
┌──┐ ┌──┐ ┌──┐
│QA│ │CP│ │RV│  ← 三个专业智能体
└┬─┘ └┬─┘ └┬─┘
 │    │    │
 ▼    ▼    ▼
┌──────────────┐
│  RAG 混合检索  │  ← FAISS 向量 + BM25 关键词
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  工具调用      │  ← search_law / create_complaint_report / ...
└──────────────┘
```

## 知识库内容

- 《消费者权益保护法》— 核心法条（退货、赔偿、格式条款等）
- 《产品质量法》— 产品质量责任、缺陷赔偿
- 《电子商务法》— 网购维权、平台责任
- 《食品安全法》— 食品安全十倍赔偿
- 7 个典型消费维权案例（网购退货、食品安全、虚假宣传等）
