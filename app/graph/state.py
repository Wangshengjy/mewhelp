"""ch05 · 会话状态:ConversationState + reducer。

- ``merge_dict``:trace 字典的合并 reducer。各节点只写自己关心的键,
  LangGraph 在通道层浅合并,谁也不覆盖谁留下的痕迹。
- ``messages`` 用 LangGraph 官方 ``add_messages``:按消息 id 追加/去重,
  配合 AsyncSqliteSaver 才有跨轮历史。
- 其余字段都是"最后写入者胜"(replace);需要累加的(steps/tokens_used/
  suggested_actions)由节点自己读旧值再写新值。
"""
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def merge_dict(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """浅合并两个字典(reducer):右值覆盖同键,其余保留。"""
    merged: dict[str, Any] = dict(left or {})
    merged.update(right or {})
    return merged


class ConversationState(TypedDict, total=False):
    """一轮对话在图里流动的全部状态(total=False:字段均可缺省)。"""

    conversation_id: str  # 本仓库会话主键是 UUID 字符串(课程参考是 int)
    user_id: str
    query: str  # 原始用户问句(coref 透传填充)
    resolved_query: str  # ch06 coref 改写后的问句(指代消解;原 query 保持原文)
    intent: str  # 八类单标签:物流/订单/商品咨询/退款退货/售后/投诉/闲聊/其他
    intent_confidence: float  # ch06 意图置信度(0~1;低置信落池在 forced_rag 前)
    order_id: str  # ch06 退款路命中的订单号(interrupt 选单后回填)
    order_data: dict[str, Any]  # ch06 订单快照(build_order 同源)
    route: str  # knowledge | refund_flow | business | complaint | fallback_script
    evidence: str  # 知识路编号证据文本(拼进 system)
    evidence_strong: bool  # 生成前证据闸信号(置信度闸+自评闸双闸结论)
    citations: list[dict[str, Any]]  # 编号证据元数据(角标渲染用)
    # ch09 · 置信度闸与飞轮:
    evidence_confidence: float  # 四信号加权总分 0~1(置信度闸判据,阈值校准回填)
    fallback_source: str  # 拒答来源标签:retrieval_low_conf | self_check(落池 source 同名)
    retrieved_snapshot: list[dict[str, Any]]  # 落池/反馈共用的召回快照 Top3(闸都过留给 👎 回捞)
    trace: Annotated[dict[str, Any], merge_dict]  # 各节点留痕(浅合并)
    steps: int  # ReAct 已走步数
    tokens_used: int  # token 累计(仅记账,不是停止条件)
    suggested_actions: list[dict[str, Any]]  # 后端建议动作(转人工/建工单/退款表单)
    answer: str  # 确定性出口节点的固定话术
    messages: Annotated[list[AnyMessage], add_messages]
    # ch07 · 三层上下文:每轮入口从 MySQL 会话行读入(最后写入者胜),轮末 settle_layers 只写 DB 不回写 State
    summary: str  # 最近几段梗概投影(跟证据一起挂在用户侧,不进 system)
    summary_upto_msg_id: int  # 摘要覆盖到哪条消息(滑窗从其后接原文)
    layer1_from_msg_id: int  # 层 1 起点锚点(id≤此值为层 2 半压,>此值原样)
