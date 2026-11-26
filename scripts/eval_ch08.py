"""ch08 · 端到端验收(脚本层,真上游 + 真 MySQL + 真 MCP Server)。

前置:
    1) MySQL(dev 库 13306)与上游模型(.env 硅基 Qwen2.5-14B)可用;
    2) 先起 MCP Server:.\\.venv\\Scripts\\python.exe -m scripts.mcp_up
    3) .\\.venv\\Scripts\\python.exe -m scripts.eval_ch08
    4) 结束后:.\\.venv\\Scripts\\python.exe -m scripts.mcp_down
六条验收(输出审计核对明细):
    1 新工具注册即用:promotions 草稿 import 进 builtin 包 → 注册中心即刻可见,
      主力 Agent 当轮可调(import 即注册,连重启都不用);测完删干净不留痕。
    2 MCP 查物流:问「订单1001的物流到哪了」→ agent 先 query_order 拿 tracking_no
      再调 MCP query_logistics → 内部枚举码翻成中文状态出现在答复里。
    3 Server 加工具客服侧不动:不动本进程,Server 侧可用性变化 → 现问现拿清单
      随之变化(mcp_down 后只上内置;mcp_up 后 MCP 工具回来)。演示「加工具只重启
      Server」= 同一机制(Server 侧加工具,客服侧重拉清单即见)。
    4 确认建工单:uvicorn 子进程 + httpx 直连 /api/chat/stream → interrupt 预览卡 →
      POST /api/actions/resume {"confirmed":true} → 落 tickets(带工单号) + 审计「成功」。
    5 取消建工单:resume {"confirmed":false} → 不落 tickets + 审计「权限拒绝」。
    6 超时与不重试:6a 写操作人为变慢(DEMO_TICKET_DELAY_SECONDS=12 重启客服)→
      引擎超时、写恒不重试(审计 1 行「超时」retry=0;晚到的工单行反而证明
      「超时未必没执行」);6b 读操作人为变慢(MOCK_DELAY_SECONDS=12 起物流 Server)→
      超时重试(审计 3 行「超时」retry=0/1/2),坏消息如实回灌。
"""

import asyncio
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402
import pymysql  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

API_PORT = 8200
BASE_URL = f"http://127.0.0.1:{API_PORT}"
LOGISTICS_PORT = 8101

RESULTS: list[tuple[str, bool, str]] = []


# ── DB 工具(审计/工单核对) ──────────────────────────────────────────────────


def _connect():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "13306")),
        user=os.getenv("MYSQL_USER", "mewhelp"),
        password=os.getenv("MYSQL_PASSWORD", "mewhelp"),
        database=os.getenv("MYSQL_DATABASE", "mewhelp"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def audit_rows(tool_name: str, conversation_id: str | None = None) -> list[dict]:
    """按工具名(+可选会话)查审计,时间升序。按会话过滤避免跨验收/跨次运行的行串扰。"""
    query = (
        "SELECT tool_name, tool_source, mcp_server, status, retry_count, duration_ms, tool_call_id, "
        "conversation_id FROM tool_audit_logs WHERE tool_name=%s"
    )
    params: list = [tool_name]
    if conversation_id:
        query += " AND conversation_id=%s"
        params.append(conversation_id)
    query += " ORDER BY id ASC LIMIT 24"
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def ticket_rows(conversation_id: str) -> list[dict]:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT ticket_no, ticket_type, description, status FROM tickets "
            "WHERE conversation_id=%s ORDER BY ticket_no",
            (conversation_id,),
        )
        return cur.fetchall()


def conversations_by_user(user_id: str) -> list[str]:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM conversations WHERE user_id=%s ORDER BY id DESC LIMIT 4", (user_id,))
        return [row["id"] for row in cur.fetchall()]


def answer_of(state: dict) -> str:
    """agent 路的最终答复在最后一条 AIMessage 里(不在 state['answer']);确定性路才写 answer。"""
    from app.graph.nodes import resolve_answer

    return resolve_answer(state or {})


# ── 图级验收(进程内直连图;真上游/真 MCP) ────────────────────────────────────


async def _init_graph() -> None:
    from app.graph import runtime

    await runtime.init_graph()


async def acceptance_1_new_tool_without_restart() -> tuple[bool, str]:
    """验收1(确定性口径):注册即定义 → 模型当轮可选新工具 → 引擎可执行 → 审计可查。
    直连 agent_llm(带真模型)避开意图路由抽签(促销话术会被分类到知识路,那是 ch06 语义)。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    from app.graph import nodes, runtime
    from app.prompts import CHAT_SYSTEM_PROMPT
    from app.tools import engine, registry as tool_registry

    draft = PROJECT_ROOT / "scripts" / "demo_ch08_promotions.py.txt"
    target = PROJECT_ROOT / "app" / "tools" / "builtin" / "promotions.py"
    shutil.copyfile(draft, target)
    try:
        importlib.import_module("app.tools.builtin.promotions")  # 注册即定义:导入即登记(未重启)
        spec = tool_registry.get_builtin_spec("query_promotion")
        assert spec is not None, "注册中心未登记 query_promotion"

        deps = runtime.get_deps()
        settings = deps.settings
        specs = await tool_registry.get_all_specs(settings)
        assert "query_promotion" in specs, "主力清单未合并新工具"

        model = deps.chat_model_factory(settings).bind_tools([s.tool for s in specs.values()])
        # 用户明示「用工具查」:新工具注册后模型必须能选中并执行成功
        ai = await model.ainvoke(
            [
                SystemMessage(content=CHAT_SYSTEM_PROMPT),
                HumanMessage(content="请用工具帮我查一下,现在有什么促销活动?"),
            ]
        )
        call = next((c for c in (ai.tool_calls or []) if c["name"] == "query_promotion"), None)
        if call is None:
            return False, f"模型未选中新工具 tool_calls={[c['name'] for c in (ai.tool_calls or [])]}"

        run = await engine.execute_tool_call(
            call, "eval-ch08-promo", specs, session_factory=deps.session_factory, settings=settings
        )
        rows = audit_rows("query_promotion", "eval-ch08-promo")
        ok = run.ok and run.status == "成功" and rows and rows[-1]["status"] == "成功"
        detail = (
            f"模型选中 query_promotion args={call.get('args')}; 引擎={run.status}; "
            f"回灌前80字={run.tool_message.content[:80]!r}; 审计={[(r['status'], r['retry_count']) for r in rows]}"
        )
        return bool(ok), detail
    finally:
        target.unlink(missing_ok=True)
        tool_registry._BUILTIN.pop("query_promotion", None)  # demo 清理:注册中心摘除
        sys.modules.pop("app.tools.builtin.promotions", None)


async def acceptance_2_mcp_logistics() -> tuple[bool, str]:
    from app.graph import runtime

    turn = await runtime.run_turn(None, "订单1001的物流到哪了?", user_id="eval-ch08")
    answer = answer_of(turn["state"])
    conv_id = turn["conversation_id"]
    rows = audit_rows("query_logistics", conv_id)
    log_rows = audit_rows("query_order", conv_id)
    ok = (
        turn.get("interrupt") is None
        and any(r["status"] == "成功" for r in rows)
        and any(r["tool_source"] == "mcp" for r in rows)
        and any(r["status"] == "成功" for r in log_rows)
        and any(kw in answer for kw in ("运输中", "已揽收", "派送中", "已签收"))
    )
    detail = (
        f"MCP审计={[(r['tool_source'], r['status'], r['retry_count']) for r in rows]}; "
        f"query_order审计={[(r['status']) for r in log_rows]}; "
        f"answer含中文状态={any(kw in answer for kw in ('运输中','已揽收','派送中','已签收'))!r}; "
        f"answer前60字={answer[:60]!r}"
    )
    return bool(ok), detail


async def acceptance_3_live_discovery() -> tuple[bool, str]:
    """不重启本进程,验证清单随 Server 侧变化:down → 只内置;up → MCP 回来。"""
    from app.tools import registry as tool_registry

    subprocess.run([sys.executable, "-m", "scripts.mcp_down"], cwd=PROJECT_ROOT, capture_output=True)
    await asyncio.sleep(1.0)
    specs_down = await tool_registry.get_all_specs()
    has_mcp_down = any(s.source == "mcp" for s in specs_down.values())

    subprocess.run([sys.executable, "-m", "scripts.mcp_up"], cwd=PROJECT_ROOT, capture_output=True)
    specs_up = await tool_registry.get_all_specs()
    mcp_names = sorted(s.name for s in specs_up.values() if s.source == "mcp")

    ok = (not has_mcp_down) and mcp_names == ["query_logistics", "query_return_status", "query_warranty"]
    detail = (
        f"Server停: 清单MCP工具={sum(1 for s in specs_down.values() if s.source=='mcp')}个(应为0); "
        f"Server起: MCP工具={mcp_names}(客服进程全程未重启,现问现拿)"
    )
    return bool(ok), detail


# ── API 层验收(httpx 直连 uvicorn 子进程) ───────────────────────────────────


def _start_server(env_extra: dict | None = None) -> subprocess.Popen:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", **(env_extra or {})}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(API_PORT)],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            response = httpx.get(f"{BASE_URL}/", timeout=1.0)
            if response.status_code == 200 and response.status_code != 502:
                return proc
        except httpx.TransportError:
            pass
        time.sleep(0.4)
    proc.terminate()
    raise RuntimeError("uvicorn 30s 未就绪")


def _stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _stream_chat(message: str) -> tuple[str | None, dict | None, list[str]]:
    """POST /api/chat/stream,聚合出 conversation_id / interrupt 载荷 / 全部 delta 文本。"""
    with httpx.stream(
        "POST",
        f"{BASE_URL}/api/chat/stream",
        json={"message": message},
        headers={"Accept": "text/event-stream"},
        timeout=120,
    ) as response:
        response.raise_for_status()
        conversation_id = None
        interrupt_payload = None
        texts: list[str] = []
        event_name = ""
        data_lines: list[str] = []
        for line in response.iter_lines():
            if line == "":
                if data_lines and event_name:
                    data = "\n".join(data_lines)
                    payload = json.loads(data) if data and data != "[DONE]" else {}
                    if event_name == "conversation":
                        conversation_id = payload.get("conversation_id")
                    elif event_name == "interrupt":
                        interrupt_payload = payload
                    elif event_name == "token":
                        texts.append(payload.get("delta") or "")
                event_name, data_lines = "", []
                continue
            if ":" in line:
                field, _, value = line.partition(":")
                value = value.lstrip(" ")
                if field == "event":
                    event_name = value
                elif field == "data":
                    data_lines.append(value)
        return conversation_id, interrupt_payload, texts


def _resume_api(conversation_id: str, body: dict) -> tuple[dict | None, list[str]]:
    interrupt_payload = None
    texts: list[str] = []
    with httpx.stream(
        "POST",
        f"{BASE_URL}/api/actions/resume",
        json={"conversation_id": conversation_id, **body},
        headers={"Accept": "text/event-stream"},
        timeout=120,
    ) as response:
        response.raise_for_status()
        event_name = ""
        data_lines: list[str] = []
        for line in response.iter_lines():
            if line == "":
                if data_lines and event_name:
                    data = "\n".join(data_lines)
                    payload = json.loads(data) if data and data != "[DONE]" else {}
                    if event_name == "interrupt":
                        interrupt_payload = payload
                    elif event_name == "token":
                        texts.append(payload.get("delta") or "")
                event_name, data_lines = "", []
                continue
            if ":" in line:
                field, _, value = line.partition(":")
                value = value.lstrip(" ")
                if field == "event":
                    event_name = value
                elif field == "data":
                    data_lines.append(value)
    return interrupt_payload, texts


def _latest_conversation(user_id: str) -> str | None:
    ids = conversations_by_user(user_id)
    return ids[0] if ids else None


def acceptance_4_confirm_ticket() -> tuple[bool, str]:
    proc = _start_server()
    try:
        conversation_id, interrupt_payload, _texts = _stream_chat("我要建个售后工单,收到的猫粮包装破损了")
        if not conversation_id or not interrupt_payload:
            return False, f"未收到 interrupt 帧 conv={conversation_id} payload={interrupt_payload}"
        preview = interrupt_payload.get("preview") or {}
        assert preview.get("ticket_type"), f"预览缺工单类型: {interrupt_payload}"
        assert preview.get("description"), f"预览缺问题描述: {interrupt_payload}"

        _interrupt2, texts = _resume_api(conversation_id, {"confirmed": True})
        answer = "".join(texts)
        rows = ticket_rows(conversation_id)
        audits = audit_rows("create_ticket", conversation_id)
        ok = (
            len(rows) == 1
            and rows[0]["status"] == "open"
            and rows[0]["ticket_type"] == "售后"
            and any(r["status"] == "成功" and r["tool_source"] == "builtin" for r in audits)
            and any(r["ticket_no"] in answer for r in rows)  # 回复带工单号
        )
        detail = (
            f"preview={preview}; tickets={rows}; "
            f"审计={[(r['status'], r['tool_source'], r['retry_count']) for r in audits]}; "
            f"回复含工单号={any(r['ticket_no'] in answer for r in rows)!r}"
        )
        return bool(ok), detail
    finally:
        _stop_server(proc)


def acceptance_5_cancel_ticket() -> tuple[bool, str]:
    proc = _start_server()
    try:
        conversation_id, _interrupt_payload, _texts = _stream_chat("帮我建个工单,商品还没收到,先登记一下")
        if not conversation_id:
            return False, "未收到 conversation 帧"
        _resume_api(conversation_id, {"confirmed": False})
        rows = ticket_rows(conversation_id)
        audits = audit_rows("create_ticket", conversation_id)
        ok = not rows and any(r["status"] == "权限拒绝" for r in audits)
        detail = f"tickets(应为空)={rows}; 审计={[(r['status'], r['retry_count']) for r in audits]}"
        return bool(ok), detail
    finally:
        _stop_server(proc)


def acceptance_6a_write_timeout_no_retry() -> tuple[bool, str]:
    """写操作人为变慢:DEMO_TICKET_DELAY_SECONDS=12 重启客服 → 引擎 5s 超时,写恒不重试。"""
    proc = _start_server({"DEMO_TICKET_DELAY_SECONDS": "12"})
    try:
        conversation_id, _interrupt_payload, _texts = _stream_chat("帮我建个工单,发票没开出来")
        if not conversation_id:
            return False, "未收到 conversation 帧"
        started = time.time()
        _resume_api(conversation_id, {"confirmed": True})
        elapsed = time.time() - started
        audits = audit_rows("create_ticket", conversation_id)
        timeout_rows = [r for r in audits if r["status"] == "超时"]
        ok = (
            len(timeout_rows) == 1
            and timeout_rows[0]["retry_count"] == 0
            and 3500 <= (timeout_rows[0]["duration_ms"] or 0) <= 11000  # ≈5s 超时,未重试第二次
        )
        detail = (
            f"resume耗时={elapsed:.1f}s(含模型回灌轮); "
            f"超时审计行={[(r['status'], r['retry_count'], r['duration_ms']) for r in timeout_rows]}"
            "(写不重试→恒1行retry=0;工单行约12s后由未被打断的线程晚到提交,更证「超时未必没执行」)"
        )
        return bool(ok), detail
    finally:
        _stop_server(proc)


async def acceptance_6b_read_timeout_with_retries() -> tuple[bool, str]:
    """读操作人为变慢:MOCK_DELAY_SECONDS=12 起物流 Server → 10s 超时 × 3 次尝试。"""
    subprocess.run([sys.executable, "-m", "scripts.mcp_down"], cwd=PROJECT_ROOT, capture_output=True)
    log_file = PROJECT_ROOT / "log" / "mcp-delayed.log"
    log_file.parent.mkdir(exist_ok=True)
    env = {**os.environ, "PORT": str(LOGISTICS_PORT), "MOCK_DELAY_SECONDS": "12", "PYTHONIOENCODING": "utf-8"}
    delayed = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "mcp_servers" / "logistics_server.py")],
        cwd=PROJECT_ROOT, env=env, stdout=log_file.open("ab"), stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                probe = httpx.get(f"http://127.0.0.1:{LOGISTICS_PORT}/mcp", timeout=1.0)
                if probe.status_code not in (502, 503, 504):
                    break
            except httpx.TransportError:
                pass
            time.sleep(0.3)
        from app.graph import runtime

        turn = await runtime.run_turn(None, "订单1001的物流到哪了?", user_id="eval-ch08")
        answer = answer_of(turn["state"])
        conv_id = turn["conversation_id"]
        # 引擎一次调用只落一行审计:写最终 retry_count(3 次尝试 = 1 行「超时」retry=2,≈30s)
        audits = [r for r in audit_rows("query_logistics", conv_id) if r["status"] == "超时"]
        ok = (
            len(audits) == 1
            and audits[0]["retry_count"] == 2
            and 20000 <= (audits[0]["duration_ms"] or 0) <= 45000
            and all(r["tool_source"] == "mcp" for r in audits)
            and bool(answer)
        )
        detail = (
            f"超时审计行={[(r['retry_count'], r['duration_ms']) for r in audits]}"
            "(一次调用一行:3 次尝试后终止,retry=2≈30s); "
            f"answer前60字={answer[:60]!r}"
        )
        return bool(ok), detail
    finally:
        delayed.terminate()
        delayed.wait(timeout=10)
        subprocess.run([sys.executable, "-m", "scripts.mcp_down"], cwd=PROJECT_ROOT, capture_output=True)


async def main() -> None:
    # 本会话/演示终端的 Job 生命周期决定后台进程存活:由 eval 自己拉起 MCP Server
    # (eval 的子进程,与本进程同 Job,全程存活),不再依赖外部 mcp_up。
    subprocess.run([sys.executable, "-m", "scripts.mcp_up"], cwd=PROJECT_ROOT, capture_output=True)
    await _init_graph()
    async def _run(name, coro):
        try:
            ok, detail = await coro
        except Exception as exc:  # noqa: BLE001 验收脚本绝不半途而废,逐条给结论
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        RESULTS.append((name, ok, detail))
        print(json.dumps({"验收": name, "通过": ok, "明细": detail}, ensure_ascii=False))

    await _run("验收1 新工具注册即用(promotions)", acceptance_1_new_tool_without_restart())
    await _run("验收2 MCP 查物流(枚举码翻人话)", acceptance_2_mcp_logistics())
    await _run("验收3 Server 侧变化客服侧现问现拿", acceptance_3_live_discovery())
    await _run("验收4 确认建工单(API 层)", asyncio.to_thread(acceptance_4_confirm_ticket))
    await _run("验收5 取消=权限拒绝+审计(API 层)", asyncio.to_thread(acceptance_5_cancel_ticket))
    await _run("验收6a 写超时不重试(API 层)", asyncio.to_thread(acceptance_6a_write_timeout_no_retry))
    await _run("验收6b 读超时重试(图级)", acceptance_6b_read_timeout_with_retries())

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"ch08 端到端验收:{passed}/{len(RESULTS)} 通过")
    results_path = PROJECT_ROOT / "log" / "eval_ch08-results.json"
    results_path.parent.mkdir(exist_ok=True)
    results_path.write_text(
        json.dumps(
            {"passed": passed, "total": len(RESULTS), "results": RESULTS}, ensure_ascii=False, indent=1
        ),
        encoding="utf-8",
    )
    print(f"结果已落盘:{results_path}")
    from app.graph import runtime

    await runtime.close_graph()
    raise SystemExit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    asyncio.run(main())
