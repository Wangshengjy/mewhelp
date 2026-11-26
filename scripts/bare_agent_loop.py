"""ch05/ch08 · 祛魅:不用任何框架,手写最裸的 Agent 循环。

看清 Agent 的本质:它就是「带工具清单的 for 循环」——模型每轮决定调不调工具,
不调就收敛;LangGraph 只是把这个循环放进更大的图(意图分流/证据闸/审计)。
本脚本不引入 LangGraph,模型工厂与统一执行引擎直接复用本仓库模块:
- 模型工厂:app/graph/llm.build_chat_model(Settings)(课程参考是 app.core.llm.get_chat_model);
- 工具执行:app/tools/engine.execute_tool_call(返回 ToolRun{tool_message},六段管道齐);
- 工具清单:app/tools/registry.get_all_specs()(内置 + MCP 现问现拿,MCP_DOWN 时只上内置);
- system prompt:app/prompts.CHAT_SYSTEM_PROMPT。

需 MySQL + 真实上游可用。
用法:
    .\\.venv\\Scripts\\python.exe -m scripts.bare_agent_loop "订单1001的物流到哪了"
"""
import asyncio
import sys

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.llm import build_chat_model
from app.prompts import CHAT_SYSTEM_PROMPT
from app.settings import Settings
from app.tools.engine import execute_tool_call
from app.tools.registry import get_all_specs

MAX_TURNS = 6


async def run_agent(query: str, max_turns: int = MAX_TURNS) -> str:
    """裸 ReAct 循环:bind_tools + for 循环,步数封顶打提示。"""
    settings = Settings.load()
    from scripts._common import session_factory_from  # noqa: PLC0415 复用脚本土著引导

    session_factory = session_factory_from(settings)
    specs = await get_all_specs(settings)
    model = build_chat_model(settings).bind_tools([spec.tool for spec in specs.values()])

    messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT), HumanMessage(content=query)]
    for step in range(1, max_turns + 1):
        ai = await model.ainvoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            print(f"[第{step}步] 无工具调用 → 收敛")
            return ai.content
        for call in ai.tool_calls:
            print(f"[第{step}步] 调用工具 {call['name']} args={call['args']}")
            run = await execute_tool_call(
                call,
                "bare-loop-demo",
                specs,
                session_factory=session_factory,
                settings=settings,
            )
            preview = run.tool_message.content[:120]
            print(f"        -> {run.status}: {preview}")
            messages.append(run.tool_message)
    print(f"[封顶] max_turns={max_turns} 用尽仍未收敛,生产环境此处应走兜底话术")
    return "(未收敛)"


async def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "订单1001的物流到哪了"
    print("问题:", query)
    print("答复:", await run_agent(query))


if __name__ == "__main__":
    asyncio.run(main())
