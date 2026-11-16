# MewHelp 电商智能客服（第 2 章）

在第 1 章纯对话的基础上，本章把会话历史放进 MySQL，并接入五个 LangChain 工具：模型每轮**最多调用一个**工具（订单、商品、物流、FAQ、工单），工具结果写回数据库后，再用**未绑定工具**的模型逐步流式生成最终答复。

- 会话、消息、FAQ、工单四张表由服务启动时自动创建，并幂等写入演示夹具。
- 聊天页仍由 FastAPI 托管单文件 `static/index.html`；触发工具的那一轮，assistant 气泡先显示工具徽章，再显示答复文本。
- 订单、商品、物流、工单全部是虚构的演示数据（结果里带 `demo: true`），不连接任何真实业务系统。

## 环境准备

需要 Python 3.10 或更高版本，以及用于运行 MySQL 8.4 的 Docker。Windows PowerShell 示例：

```powershell
cd mewhelp
# 使用本工作区的 Python 3.12；其他环境请替换成已安装的 Python 3.10+ 路径。
$python = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $python -m venv .venv
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

`CHAT_BASE_URL`、`CHAT_MODEL`、`CHAT_API_KEY` 必填，缺失时启动即报错；其余项有示例默认值，`MYSQL_*` 的默认值与 `compose.yaml` 一致，仅用于本地演示。进程环境变量优先于 `.env`，因此临时换端口不需要改文件（例如 `$env:MYSQL_PORT = "3307"`）。不要把真实密钥提交到仓库。

聊天需要 Chat Completions 的流式能力，工具调用需要模型支持 function calling。同一组配置可指向不同 OpenAI 兼容服务（例如 DeepSeek `https://api.deepseek.com`、Ollama `http://localhost:11434/v1/`）。上游不支持时接口返回通用错误提示。详见 [DeepSeek API](https://api-docs.deepseek.com/api/create-chat-completion) 与 [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)。

## 启动 MySQL

```powershell
docker compose up -d --wait
```

首次启动自动创建 `mewhelp` 库；应用启动时再用 SQLAlchemy `create_all` 建表并幂等写入四表演示夹具，重复启动不会重复插入。`docker compose ps` 显示 `healthy` 即就绪，`docker compose logs mysql` 可查看初始化日志。

## 启动 API

```powershell
fastapi dev app/main.py
```

等价写法：`python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`。服务默认监听 `http://127.0.0.1:8000`，接口文档在 `http://127.0.0.1:8000/docs`。

第 1 章的 `start.bat` 也可以继续用：它启动同一个服务，轮询就绪后打开浏览器。**它不负责启动 MySQL**，请先执行上面的 Compose 命令。

## 浏览器聊天页

启动服务后打开 `http://127.0.0.1:8000/`。页面是 `static/index.html` 单文件，不依赖构建工具或外部前端库。

- 提交消息后用原生 `fetch` 读取 POST SSE 流，答复逐段显示。
- 某轮触发工具时，当前 assistant 气泡顶部先出现工具徽章（工具名 + 运行中/已完成/失败），随后同一气泡继续显示答复文本；徽章只显示工具名和状态，不暴露工具参数或原始结果。
- 页面在本次页面生命周期内保留 `conversation_id`，续问沿用同一会话，服务端从 MySQL 读取已完成的问答轮作为上下文；刷新页面或点“新对话”会清空页面会话状态并开始新会话。
- 按 Enter 发送，Shift + Enter 换行。

## 多轮 SSE 对话

用 `curl.exe -N` 关闭客户端缓冲。JSON 放进文件传输，避免 Windows 控制台代码页损坏命令行里的中文参数：

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
data: {"conversation_id": "29c8e4ab-1393-4335-80b0-29bcff0df924"}

event: tool
data: {"tool_call_id": "call_00_wtDwcrPXd0kBTfh2i1xH4954", "name": "query_logistics", "status": "running"}

event: tool
data: {"tool_call_id": "call_00_wtDwcrPXd0kBTfh2i1xH4954", "name": "query_logistics", "status": "completed"}

event: token
data: {"delta": "订单"}

event: done
data: [DONE]
```

把 `conversation_id` 放进下一轮请求即可续用同一会话（只有省略 ID 时才新建会话并先发 `conversation` 事件）：

```powershell
'{"message":"退货政策是什么","conversation_id":"粘贴这里的 ID"}' | Set-Content -LiteralPath $body -Encoding UTF8
curl.exe -N -X POST http://127.0.0.1:8000/api/chat/stream `
  -H "Content-Type: application/json" `
  --data-binary "@$body"
```

`token` 只包含最终未绑定工具模型的文本增量；`done` 的原始数据是 `[DONE]`。未知 `conversation_id` 返回 HTTP 404、空白消息返回 422、固定提示加本轮输入超出预算返回 413，三者都在 SSE 开始前返回。上游失败或数据库写入失败会发送带稳定 `code` 的 `error` 事件（不含异常原文、SQL 或密钥）；此时已经写入的用户消息和工具轨迹会保留，服务端不会伪造答复。

## 五个演示工具与预期行为

| 工具 | 用途 | 说明 |
| --- | --- | --- |
| `query_order` | 查询订单 | 由固定种子派生，不连接真实订单系统 |
| `query_product` | 查询商品 | 同上，`P-1001` 这类演示商品号才有内容 |
| `query_logistics` | 查询物流 | 返回承运商、运单号、当前状态与轨迹 |
| `query_faq` | 查询 FAQ | 只做参数化 SQL `LIKE`，按关键词匹配问题/答案/分类 |
| `create_ticket` | 创建人工工单 | 仅在用户明确要求转人工或建单时调用；(会话, tool_call_id) 唯一，超时重试也只建一张 |

FAQ 演示数据只有“退货”和“服务”两条，因此**“退货政策是什么”能命中，“邮费是多少”必然查不到**——这是刻意保留的漏召回样例，用来验证客服在无命中时如实说明、而不是编造运费规则。请把工单号、演示订单/商品/物流都当作虚构数据。

## 售后结构化提取

```powershell
'{"description":"订单 12345 的鞋开胶了，希望退款，最好退回原支付方式。"}' | Set-Content -LiteralPath $body -Encoding UTF8
curl.exe -X POST http://127.0.0.1:8000/api/after-sales/extract `
  -H "Content-Type: application/json" `
  --data-binary "@$body"
```

响应固定包含 `order_number`、`request_type`、`expected_solution` 三个字段，未明确提供时为 JSON `null`；`request_type` 只取 `refund`、`exchange`、`logistics_query`、`quality_issue` 之一。

## 自动化验证

```powershell
python -m pytest -q
python -m compileall app tests
```

测试使用内存 SQLite 和脚本化的假模型，不访问 MySQL、也不访问网络。真实上游的两组标注评估单独运行（需要 `.env` 里的 `CHAT_*` 可用）：

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

- **3306 被占用、Compose 起不来**：本机若已有 MySQL 服务占用 3306，容器无法绑定该端口。可以停掉占用该端口的服务，或把容器映射到别的端口并让应用连过去：

  ```powershell
  docker run -d --name mewhelp-mysql-verify `
    -e MYSQL_DATABASE=mewhelp -e MYSQL_USER=mewhelp `
    -e MYSQL_PASSWORD=mewhelp -e MYSQL_ROOT_PASSWORD=mewhelp-root `
    -p 127.0.0.1:3307:3306 mysql:8.4
  $env:MYSQL_PORT = "3307"
  fastapi dev app/main.py
  ```

- **启动后立刻报数据库连接失败**：确认 MySQL 已在运行，且 `.env` 的 `MYSQL_*` 与容器一致。数据卷一旦用别的凭据初始化过，改环境变量不会生效，需要 `docker compose down -v` 后重建。
- **答复较慢**：单轮耗时取决于上游模型的响应速度和是否开启思考模式，工具轮还要多一次模型调用。
