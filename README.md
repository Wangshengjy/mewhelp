# MewHelp 电商智能客服

LangGraph Workflow + Agent 混合架构的电商智能客服系统：混合检索问答、意图分流、工具办事、可观测、数据飞轮与旁路主题分类微调。业务铁律（强制检索、强制取证、转人工）由流程代码守住，开放推理交给模型；全程 TDD。

## 核心特性

- **混合检索 + 幻觉治理**：Milvus 2.5 BM25 与向量双路 RRF 融合、bge-reranker 精排；自建 300 题五桶评估集实测 MRR@5 0.735（纯向量 0.711、纯 BM25 0.610）；生成前四信号置信度闸，阈值经 300 题校准（拦 86.7% 库外题、误拦 10%），库外题拒答率 1.000
- **Workflow + Agent 混合架构**：LangGraph 编排指代消解、意图识别（八类五出口）、置信度校验、Agent 生成与审计；退款子流程缺单号时 interrupt 暂停，前端订单选择器回填后续跑
- **SSE 流式对话协议**：conversation / token / tool / done / interrupt / error 六类事件帧，逐 token 打字机输出，回答带引用角标可展开出处
- **工具系统**：ToolSpec 注册中心统一登记内置与 MCP 工具；执行引擎覆盖 Schema 校验回灌、权限门、超时重试、失败分诊与审计；写操作须经前端确认令牌
- **分层上下文**：最近轮原文、中段压缩、远段滚动摘要三层，额度从模型窗口倒推，37 轮长对话实测不丢早期信息
- **可观测与数据飞轮**：自部署 Langfuse 零侵入还原调用树、token 按意图分账；知识缺口经检索不足、自评未过、用户点踩三入口落池，标准化查重后按频次审核回填
- **旁路主题分类**：17 类多标签 RoBERTa-wwm-ext 微调管线，训练 → ONNX 导出（logits 一致性偏差 1.19e-06）→ 推理服务 → 批量归类落表（幂等）；正式规模训练待 GPU，链路已用冒烟权重全量验证
- **TDD**：pytest 410 个测试全通过（内存 SQLite + 假模型，不依赖网络与 MySQL）

## 技术栈

Python 3.10+ · FastAPI · LangChain · LangGraph · Milvus 2.5（BM25+dense 混合）· MCP · Langfuse · MySQL 8.4 · SSE · Pydantic · ONNX Runtime · pytest · Docker Compose

## 架构

```
浏览器 (static/*.html, 原生 fetch 读 SSE)
   │  POST /api/chat/stream
   ▼
FastAPI (app/api/chat.py)
   │
   ├─ LangGraph 主图 (app/graph/)              ← coref → intent → 路由分流
   │     ├─ 知识线: 强制 RAG → 四信号置信度闸 → 生成（带引用角标）
   │     ├─ 退款线: fetch_order →(interrupt 选单) retrieve_policy → Agent 判定
   │     ├─ 投诉线: 强制转人工 + 建工单提议
   │     └─ 闲聊线: 脚本话术零模型调用
   │
   ├─ ToolSpec 注册中心 (app/tools/)           ← 内置工具 + MCP 工具统一登记
   │     └─ 执行引擎六段管道 + 两台 MCP Server (mcp_servers/)
   │
   ├─ 分层上下文 (app/core/memory.py)          ← 原文/压缩/滚动摘要，窗口预算倒推
   │
   ├─ 检索与评估 (app/kb/, app/eval/)          ← Milvus 混合检索四策略 + 裁判模型
   │
   ├─ 数据飞轮 (app/core/flywheel.py)          ← 落池 → 标准化查重 → 审核 → 写回向量化
   │
   ├─ 可观测 (app/core/observability.py)       ← Langfuse 回调，token 按意图分账
   │
   └─ MySQL (conversations/messages/tickets/kb_chunks/
             low_confidence_questions/review_queue/
             faith_cases/eval_runs/topic_classifications)
       Milvus (知识向量库)
       分类器服务 :8110 (ONNX, 旁路批量归类)
```

## 快速开始

需要 Python 3.10+ 和 Docker。

```powershell
cd mewhelp
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`.env` 只需在不存在时从模板复制一次；**不要用模板覆盖已有的 `.env`**，**不要把真实密钥提交到仓库**：

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

编辑 `.env` 填写上游与数据库配置，最小必填三项 `CHAT_BASE_URL` / `CHAT_MODEL` / `CHAT_API_KEY`（任何 OpenAI 兼容服务均可）。RAG 与飞轮需要 `EMBED_API_KEY`（硅基流动 BGE-M3）与 `RERANK_API_KEY`；可观测需要 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` 三项齐全才挂回调，缺失则静默不挂、功能不受影响。

### 启动基础设施（MySQL + Milvus）

```powershell
docker compose up -d --wait     # mysql / etcd / minio / milvus 四容器全部 healthy 即就绪
```

### 启动主应用

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

服务监听 `http://127.0.0.1:8000`，接口文档在 `/docs`。**最小可用即以上两步**；以下为可选增强：

```powershell
# 可选：Langfuse 可观测（调用树、token 分账），界面 http://localhost:3000
docker compose -f docker-compose.langfuse.yml up -d

# 可选：两台 MCP Server（售后/物流），不跑则 MCP 工具清单为 0、内置工具不受影响
.\.venv\Scripts\python.exe -m scripts.mcp_up

# 可选：主题分类器服务 :8110（飞轮批量归类用）
.\.venv\Scripts\python.exe -m scripts.ch10.classifier_ctl start
```

## 页面一览

| 页面 | 地址 | 说明 |
| --- | --- | --- |
| 对话页 | `/` | 聊天、工具轨迹徽章、引用角标、订单选择器、工单/退款表单确认 |
| 知识库 | `/kb` | 四策略检索自测、就地修块、重新向量化 |
| RAG 评估台 | `/rag-eval` | 四策略对照报告、编造个案台账（跨轮累计+处置）、就地重跑 |
| 飞轮审核后台 | `/review` | 待审队列、原话+检索快照详情、通过写回/驳回 |
| 观测与成本 | `/observability` | Langfuse 调用树报表、按意图 token 分账 |
| 主题分布 | `/topics` | 17 类条形图，下钻类目问题列表 |

## 评估与验收脚本

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest-tmp    # 410 个测试
.\.venv\Scripts\python.exe -m compileall app scripts

# 真实上游评估（需 .env 可用、MySQL/Milvus 运行中）
.\.venv\Scripts\python.exe -m scripts.eval_rag            # 300 题 × 4 策略全量评估
.\.venv\Scripts\python.exe -m scripts.calibrate_confidence # 置信度阈值校准
.\.venv\Scripts\python.exe -m scripts.judge_check          # 编造台账裁判一致性重放
.\.venv\Scripts\python.exe -m scripts.e2e_ch11             # 全链路串联两条样例
.\.venv\Scripts\python.exe -m scripts.flywheel_pipeline    # 飞轮标准化查重批处理
```

## 清理

```powershell
docker compose down                       # 停止基础设施，保留数据卷
docker compose -f docker-compose.langfuse.yml down   # 停止 Langfuse
docker compose down -v                    # 连同数据卷一起删除（慎用）
```

## 常见问题

- **3306 被占用**：停掉本机 MySQL 服务，或改容器端口映射并让 `.env` 的 `MYSQL_PORT` 跟过去。
- **启动即报数据库连接失败**：确认 `docker compose ps` 全部 healthy 且 `.env` 与容器凭据一致；凭据变更过需 `docker compose down -v` 重建。
- **Milvus 未就绪**：Milvus 依赖 etcd/minio，`docker compose up -d --wait` 会等全部 healthy；首次启动较慢属正常。
- **Langfuse 未连上**：确认三 env 齐全且容器在跑；不齐全时应用会静默跳过回调，主流程不受影响。
- **答复较慢**：单轮耗时取决于上游模型；知识类问题多一次检索与证据生成调用。
