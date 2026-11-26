"""ch06 红线冒烟:interrupt/Command(resume) 在当前 langgraph 上的中断 surface 形状。

需要打上游吗?不需要——只用一个纯 interrupt 节点。
用法:.\\.venv\\Scripts\\python.exe -m scripts.smoke_interrupt
钉死三件事(结论写 dev-notes/ch06.md,供 fetch_order 单测与 runtime 断言引用):
  A) ainvoke 首跑返回态 __interrupt__ 键形状;
  B) Command(resume=...) 续跑语义(被中断节点从头重跑);
  C) astream(stream_mode=["messages","updates"]) 下中断出现在哪种 chunk;
  D) 图外直接调用被中断节点抛的异常类型与载荷取法。
"""
import asyncio

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import Annotated, TypedDict

from langgraph.graph.message import add_messages


class S(TypedDict):
    messages: Annotated[list, add_messages]
    picked: str


async def ask_order(state: S):
    picked = interrupt({"type": "select_order", "orders": [{"order_id": "1001"}, {"order_id": "2002"}]})
    return {"picked": picked}


async def confirm(state: S):
    return {"messages": [("ai", f"已选订单 {state['picked']}")]}


async def main():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as cp:
        await cp.setup()
        b = StateGraph(S)
        b.add_node("ask_order", ask_order)
        b.add_node("confirm", confirm)
        b.add_edge(START, "ask_order")
        b.add_edge("ask_order", "confirm")
        b.add_edge("confirm", END)
        graph = b.compile(checkpointer=cp)
        config = {"configurable": {"thread_id": "smoke-int-1"}}

        # A) 非流式:首跑应带 __interrupt__
        out = await graph.ainvoke({"messages": [("human", "我要退款")]}, config)
        print("A ainvoke keys =", list(out.keys()))
        intr = out.get("__interrupt__")
        print("A __interrupt__ =", intr)
        if intr:
            print("A [0].value =", getattr(intr[0], "value", None))

        # B) 非流式 resume
        out2 = await graph.ainvoke(Command(resume="1001"), config)
        print("B resume picked =", out2.get("picked"), "msgs =", [m.content for m in out2["messages"]])

        # C) 流式:中断信息出现在哪种 chunk
        config2 = {"configurable": {"thread_id": "smoke-int-2"}}
        print("C astream chunks:")
        async for mode, chunk in graph.astream(
            {"messages": [("human", "我要退款")]}, config2, stream_mode=["messages", "updates"]
        ):
            if mode == "updates":
                print("  UPD keys =", list(chunk.keys()), "val =", chunk)
        # C') 流式后探 pending(备用探测路径)
        snap = await graph.aget_state(config2)
        print("C' aget_state .next =", snap.next, "interrupts =", getattr(snap, "interrupts", None),
              "tasks_interrupts =", [t.interrupts for t in snap.tasks])
        # C'') 流式 resume
        print("C'' astream resume:")
        async for mode, chunk in graph.astream(
            Command(resume="2002"), config2, stream_mode=["messages", "updates"]
        ):
            print("  ", mode, chunk if mode == "updates" else (chunk[0].content, chunk[1].get("langgraph_node")))

        # D) 直接调用被中断节点(无 resume 上下文)——fetch_order 缺单单测要断言这个形状
        from langgraph.errors import GraphInterrupt

        try:
            await ask_order({"messages": []})
        except GraphInterrupt as e:
            print("D GraphInterrupt args =", e.args, "first.value =",
                  e.args[0][0].value if e.args else None)
        except Exception as e:
            print("D other exception =", type(e).__name__, e)


if __name__ == "__main__":
    asyncio.run(main())
