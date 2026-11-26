"""ch05 · Workflow + Agent 混合架构(LangGraph 确定性骨架 + 主力 Agent)。

包结构:
- state.py    ConversationState + reducer
- deps.py     GraphDeps 依赖注入(configurable)
- routing.py  分流规则 / ReAct 停止条件
- intent.py   七类意图分类器(structured output)
- nodes.py    各节点函数
- build.py    StateGraph 组装
- runtime.py  checkpointer 生命周期 + 图单例 + run_turn/stream_turn
"""
