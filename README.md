# 医疗问诊机器人

基于 **LangChain + LangGraph + Neo4j 知识图谱 + RAG 检索增强 + 大语言模型** 的智能医疗问答系统。

---

## 系统架构

```
用户（Gradio 网页 / FastAPI 接口）
       │
       ▼
   Service 层
       │
       ▼
   LangGraph Agent（DeepSeek-V4-Pro）
       │
       ├── retrival_func ──→ Chroma 向量库（寻医问药网文档检索）
       ├── graph_func    ──→ Neo4j 知识图谱（疾病/症状/药物/科室查询）
       └── search_func   ──→ 博世搜索 API（联网获取最新信息）
```

Agent 根据用户意图自动选择工具：医疗专业知识走知识图谱，平台文档走向量检索，实时信息走网络搜索。

---

## 功能特性

| 能力 | 说明 |
|------|------|
| 知识图谱问答 | 11 类 Cypher 模板覆盖疾病定义、病因、症状、治疗、药物、预防、并发症、检查、科室、饮食禁忌、治愈率 |
| RAG 文档检索 | 基于 Chroma 向量库的寻医问药网平台文档检索（PDF/CSV/TXT） |
| 联网搜索 | 博世 API 实时搜索，补充最新公开信息 |
| 流式输出 | SSE token 流式返回，打字机效果 |
| 日志系统 | 按天轮转日志（`logger/YYYY-MM-DD.log`），保留 30 天，北京时间毫秒级时间戳 |
| 双入口 | Gradio 网页对话 + FastAPI RESTful 接口 |
| 文件热重载 | Watchdog 监控 `data/inputs/`，文档变动自动重建向量库 |

---

## 技术栈

| 组件 | 选型 |
|------|------|
| LLM | DeepSeek-V4-Pro（通过 OpenAI 兼容接口调用） |
| Embedding | Qwen text-embedding-v4（DashScope） / BGE（本地 HuggingFace） |
| Agent 框架 | LangGraph + LangChain 1.x |
| 知识图谱 | Neo4j（py2neo） |
| 向量数据库 | Chroma（langchain-chroma） |
| 联网搜索 | 博世搜索 API |
| 前端 | Gradio ChatInterface + FastAPI |
| 日志 | Python logging + TimedRotatingFileHandler |

---

## 项目结构

```
chat-with-doctor/
├── main.py              # FastAPI 入口（RESTful API）
├── app.py               # Gradio 入口（网页对话）
├── agent.py             # LangGraph Agent 定义
├── service.py           # 业务服务层
├── utils.py             # 工具函数（LLM/Embedding/Neo4j 连接）
├── tools.py             # Agent 工具集（retrival_func / graph_func / search_func）
├── config.py            # 知识图谱 Cypher 查询模板
├── prompt.py            # System Prompt
├── data_process.py      # 文档向量化 → Chorma 向量库构建
├── file_watcher.py      # data/inputs 文件变动监控
├── logger_config.py     # 统一日志配置
├── .env                 # 环境变量（API Key / 数据库 / 模型配置）
├── data/
│   ├── inputs/          # 待向量化的文档（PDF/CSV/TXT）
│   └── db/              # Chroma 向量库持久化目录
└── logger/              # 日志文件（YYYY-MM-DD.log）
```

---

## 快速开始

### 1. 环境要求

- Python 3.10+
- Neo4j 数据库（已启动并导入医疗知识图谱）
- Conda 虚拟环境（推荐）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

核心依赖清单：

```
langchain langchain-openai langchain-community langchain-chroma langgraph
fastapi uvicorn gradio py2neo dashscope requests python-dotenv watchdog
```

### 3. 配置 `.env`

复制 `.env.example`（如有）或直接编辑 `.env`：

```env
# LLM 模型
LLM_MODEL = "deepseek"
DEEPSEEK_API_KEY = "sk-xxxxxxxx"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_LLM_MODEL = "deepseek-v4-pro"

# Embedding 模型（qwen / bge）
EMBEDDING_MODEL = "qwen"
QWEN_EMBEDDING_MODEL = "text-embedding-v4"
QWEN_API_KEY = "sk-xxxxxxxx"

# 博世搜索
SEARCH_URL = "https://api.bocha.cn/v1/web-search"
SEARCH_API_KEY = "sk-xxxxxxxx"

# Neo4j
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your-password"

# 日志级别（DEBUG / INFO / WARNING / ERROR）
LOG_LEVEL = INFO

# 检索参数（可选，有默认值）
RETRIEVAL_TOP_K = 5
RETRIEVAL_SCORE_THRESHOLD = 0.5
SEARCH_TIMEOUT = 10
```

### 4. 准备向量库

将文档放入 `data/inputs/` 目录（支持 PDF / CSV / TXT），然后运行：

```bash
python data_process.py
```

> **注意**：每次切换 Embedding 模型后必须重建向量库，因为不同模型的向量维度不兼容。

### 5. 启动服务

**Gradio 网页版（推荐）：**

```bash
python app.py
```

浏览器访问 `http://127.0.0.1:7860`。

**FastAPI 接口版：**

```bash
python main.py
```

访问 `http://127.0.0.1:8000/docs` 查看 Swagger 文档。

---

## API 接口

### POST /ask

```json
// Request
{ "message": "鼻炎是一种什么病？" }

// Response
{ "answer": "【温馨提示】我是AI助手…鼻炎是鼻黏膜的炎症…" }
```

### GET /health

```json
{ "status": "ok" }
```

---

## 知识图谱

系统支持 11 类医疗知识查询，通过 Neo4j Cypher 实现：

| 查询类别 | 示例问题 | 对应槽位 |
|----------|---------|----------|
| 疾病定义 | X 是一种什么病？ | disease |
| 病因 | X 由什么引起？ | disease |
| 症状 | X 有什么症状？ | disease |
| 诊断 | 出现 Y 症状可能是什么病？ | symptom |
| 治疗 | X 怎么治？ | disease |
| 科室 | X 挂什么科？ | disease |
| 预防 | X 如何预防？ | disease |
| 饮食禁忌 | X 不能吃什么？ | disease |
| 检查 | X 需要做什么检查？ | disease |
| 治愈率 | X 能治好吗？ | disease |
| 并发症 | X 有什么并发症？ | disease |
| 药物适应症 | 药 Y 能治什么病？ | drug |

Cypher 模板定义在 [config.py](config.py)。

---

## 日志系统

所有日志输出到 **`logger/`** 目录，按天命名 `YYYY-MM-DD.log`，保留 30 天。

日志格式：`时间(北京) | 级别 | 模块名 | 消息`

```log
2026-06-11 21:35:12.034 | INFO  | service    | [user_ZDQNFU] 收到用户请求 | query=鼻炎是什么病
2026-06-11 21:35:12.512 | INFO  | utils      | [user_ZDQNFU] Agent 开始流式处理 | query_len=8
2026-06-11 21:35:14.891 | INFO  | utils      | [user_ZDQNFU] Agent 流式处理完成 | latency=2.38s chunks=47
```

在 `.env` 中调节级别：

```env
LOG_LEVEL=DEBUG    # 开发调试：显示 Cypher 语句、API 请求体
LOG_LEVEL=INFO     # 日常使用：请求/响应/耗时
LOG_LEVEL=WARNING  # 生产环境：仅记录异常
```

---

## 常见问题

### Q: 切换 Embedding 模型后检索不到结果？
**A:** 重建向量库：先删除 `data/db/` 目录，再执行 `python data_process.py`。不同模型的向量维度不兼容。

### Q: Neo4j 连接失败？
**A:** 检查 Neo4j 服务是否启动，确认 `.env` 中 `NEO4J_URI`、用户名、密码正确。

### Q: `DashScopeEmbeddings` 报 400 错误？
**A:** 确保 `.env` 中 `QWEN_API_KEY` 已正确配置，且安装了 `dashscope` 包：`pip install dashscope`。

### Q: `retrival_func` 返回"未找到相关文档"但文档已放入 `data/inputs/`？
**A:** 运行 `python data_process.py` 重建向量库。如果问题持续，将 `.env` 中 `LOG_LEVEL=DEBUG`，重启后查看 Chroma 检索日志。

### Q: 日志文件在哪里？
**A:** `logger/YYYY-MM-DD.log`。该目录已加入 `.gitignore`，不会被提交到仓库。
