# MewHelp 电商智能客服

基于 LLM Function Calling 的电商智能客服系统：用户用自然语言即可查订单、查物流、查 FAQ、转人工建单。模型每轮**最多调用一个**工具（订单、商品、物流、FAQ、工单五个 LangChain 工具），工具结果写回 MySQL 后，再用未绑定工具的模型逐步流式生成最终答复。

## 核心特性

- **Function Calling 工具链**：模型决策 → 参数 Schema 校验（Pydantic）→ 工具执行 → 结果以 `role=tool` 回灌 → 流式收敛，五步闭环让对话直接"办事"
- **SSE 流式对话协议**：`conversation / token / tool / done` 四类事件帧，回答逐 token 打字机输出，工具执行实时推送状态，前端气泡展示工具调用轨迹徽章
- **多轮上下文管理**：基于 token 预算的对话裁剪——始终保留 system prompt，问答轮与工具交互轮整存整裁，防止工具申请与结果被拆散导致上下文错乱，同时控制长会话成本
- **四表持久化**：`faq / conversations / messages / tickets`，以 `tool_call_id` 关联工具申请与结果；工单表唯一键约束保证超时重试**幂等**（同一轮重复触发只建一张工单）
- **TDD 开发**：pytest 69 个测试全通过（内存 SQLite + 假模型，不依赖网络与 MySQL）
- **Docker Compose 一键部署**：MySQL 8.4 容器化，服务启动自动建表并幂等写入演示数据

## 技术栈

Python 3.10+ · FastAPI · LangChain · SQLAlchemy · MySQL 8.4 · SSE · Pydantic · pytest · Docker Compose

## 架构

```
浏览器 (static/index.html, 原生 fetch 读 SSE)
   │  POST /api/chat/stream
   ▼
FastAPI (app/api/chat.py)
   │
   ├─ ChatService (app/services/chat.py)      ← Function Calling 循环 + 上下文裁剪
   │     ├─ ToolRegistry (app/tools/)         ← 5 个 @tool 业务工具 + 参数 Schema
   │     └─ ToolExecutor (app/services/)      ← 超时控制 / 异常归一
   │
   ├─ Repositories (app/repositories/)        ← SQLAlchemy 数据访问层
   │
   └─ MySQL (conversations / messages / faq / tickets)
```

## 快速开始

需要 Python 3.10+ 和 Docker（用于运行 MySQL 8.4）。

```powershell
cd mewhelp
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`.env` 只需在不存在时从模板复制一次；**不要用模板覆盖已有的 `.env`**：

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

编辑 `.env` 填写上游和数据库配置：

```dotenv
CHAT_BASE_URL=https://api.openai.com/v1
CHAT_MODEL=your-model-name
CHAT_API_KEY=your-api-key
CHAT_TEMPERATURE=0.2
CHAT_CONTEXT_TOKEN_BUDGET=6000
CHAT_TOKEN_SAFETY_MARGIN=500
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_DATABASE=mewhelp
MYSQL_USER=mewhelp
MYSQL_PASSWORD=mewhelp
TOOL_TIMEOUT_SECONDS=10
```

`CHAT_BASE_URL`、`CHAT_MODEL`、`CHAT_API_KEY` 必填，缺失时启动即报错。任何 OpenAI 兼容服务均可（例如 DeepSeek `https://api.deepseek.com`、Ollama `http://localhost:11434/v1/`），要求模型支持流式输出与 function calling。详见 [DeepSeek API](https://api-docs.deepseek.com/api/create-chat-completion) 与 [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)。**不要把真实密钥提交到仓库。**

### 启动 MySQL

```powershell
docker compose up -d --wait
```

`docker compose ps` 显示 `healthy` 即就绪。应用启动时用 SQLAlchemy `create_all` 建表并幂等写入演示夹具，重复启动不会重复插入。

### 启动 API

```powershell
fastapi dev app/main.py
```

服务默认监听 `http://127.0.0.1:8000`，接口文档在 `/docs`，聊天页在 `/`。

## 使用示例

浏览器打开 `http://127.0.0.1:8000/` 即可聊天；工具调用会在 assistant 气泡顶部显示徽章（工具名 + 运行中/已完成/失败）。

命令行验证 SSE 流（`curl.exe -N` 关闭客户端缓冲）：

```powershell
$body = "$env:TEMP\chat.json"
'{"message":"订单 1001 的物流到哪了"}' | Set-Content -LiteralPath $body -Encoding UTF8
curl.exe -N -X POST http://127.0.0.1:8000/api/chat/stream `
  -H "Content-Type: application/json" `
  --data-binary "@$body"
```

命中工具时的事件顺序：

```text
event: conversation
data: {"conversation_id": "29c8e4ab-..."}

event: tool
data: {"tool_call_id": "call_00_...", "name": "query_logistics", "status": "running"}

event: tool
data: {"tool_call_id": "call_00_...", "name": "query_logistics", "status": "completed"}

event: token
data: {"delta": "订单"}

event: done
data: [DONE]
```

把 `conversation_id` 放进下一轮请求即可续用同一会话。未知会话 ID 返回 404、空白消息返回 422、上下文超预算返回 413；上游失败会发送带稳定 `code` 的 `error` 事件（不泄露异常原文与密钥），已写入的消息和工具轨迹保留，服务端不伪造答复。

## 五个演示工具

| 工具 | 用途 | 说明 |
| --- | --- | --- |
| `query_order` | 查询订单 | 演示数据，不连接真实订单系统 |
| `query_product` | 查询商品 | 同上 |
| `query_logistics` | 查询物流 | 返回承运商、运单号、当前状态与轨迹 |
| `query_faq` | 查询 FAQ | 参数化 SQL `LIKE`，按关键词匹配问题/答案/分类 |
| `create_ticket` | 创建人工工单 | 仅在用户明确要求转人工时调用；(会话, tool_call_id) 唯一，超时重试只建一张 |

所有订单/商品/物流/工单均为虚构演示数据（响应带 `demo: true`）。FAQ 演示数据刻意只含"退货"与"服务"两条——"邮费是多少"必然查不到，用于验证客服在无命中时**如实说明而不是编造**（漏召回样例）。

## 自动化验证

```powershell
python -m pytest -q        # 69 个测试（内存 SQLite + 假模型）
python -m compileall app tests
```

针对真实上游的两组标注评估单独运行（需要 `.env` 里的 `CHAT_*` 可用）：

```powershell
python tests/eval/run_after_sales_eval.py
python tests/eval/run_tool_call_eval.py
```

## 清理

```powershell
docker compose down      # 停止容器，保留数据卷
docker compose down -v   # 连同演示数据一起删除
```

## 常见问题

- **3306 被占用**：停掉本机占用 3306 的 MySQL 服务，或把容器映射到其他端口并让应用连过去（`$env:MYSQL_PORT = "3307"`）。
- **启动即报数据库连接失败**：确认 MySQL 已运行且 `.env` 的 `MYSQL_*` 与容器一致。数据卷一旦用别的凭据初始化过，需要 `docker compose down -v` 后重建。
- **答复较慢**：单轮耗时取决于上游模型响应速度；工具轮要多一次模型调用。
