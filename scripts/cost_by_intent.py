"""ch09 · Cost Control 统计:按意图分堆算 token 账(Langfuse traces + observations)。

前提:Langfuse 三 env 齐全(服务起了,有真实 trace);tag_intent 已把意图打成
trace tag `intent:xxx` + metadata。脚本用公开 REST(API key 认证)分页拉 traces,
逐 trace 拉 observations 累计 usage,按意图 tag 分组汇总。

用法:python -m scripts.cost_by_intent [--session-id xxx] [--max-traces 500]
产物:data/ch09/reports/cost_by_intent.txt / .json
Langfuse 不可用时:打印 BLOCKED 原因,退出码 2(不伪装成功)。
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._common import load_settings  # noqa: E402

REPORT_DIR = ROOT / "data" / "ch09" / "reports"
PAGE_SIZE = 100


def _auth(pk: str, sk: str) -> dict:
    token = base64.b64encode(f"{pk}:{sk}".encode()).decode()
    return {"Authorization": "Basic " + token}


def _get(url: str, headers: dict) -> dict:
    import httpx
    resp = httpx.get(url, headers=headers, timeout=30.0, trust_env=False)  # 绕过系统代理,直连自部署栈
    resp.raise_for_status()
    return resp.json()


def aggregate(base_url: str, pk: str, sk: str, session_id: str | None = None,
              max_traces: int = 500) -> dict:
    """拉 traces → 按 intent:xxx tag 分组 → 逐 trace 累计 observations usage。

    无 intent tag 的 trace 归 unknown(闸外路由/脚本直连);token 单位按
    Langfuse usageDetails(input/output/total)统计,无价格表不算钱只算 token。"""
    headers = _auth(pk, sk)
    traces: list[dict] = []
    page = 1
    while len(traces) < max_traces:
        params = [f"page={page}", f"limit={PAGE_SIZE}"]
        if session_id:
            params.append(f"sessionId={session_id}")
        data = _get(f"{base_url.rstrip('/')}/api/public/traces?{'&'.join(params)}", headers)
        batch = data.get("data", [])
        traces.extend(batch)
        meta = data.get("meta", {})
        if not batch or len(traces) >= int(meta.get("totalItems") or 0) or page * PAGE_SIZE >= int(meta.get("totalItems") or 0):
            break
        page += 1
    traces = traces[:max_traces]

    per_intent: dict[str, dict] = {}

    def _bucket(intent: str) -> dict:
        return per_intent.setdefault(intent, {
            "traces": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

    for trace in traces:
        tags = trace.get("tags") or []
        intent = next((t.split(":", 1)[1] for t in tags if t.startswith("intent:")), None)
        obs = _get(f"{base_url.rstrip('/')}/api/public/observations?traceId={trace['id']}", headers)
        if intent is None:
            # LangGraph 1.x 会覆盖 root run 的 config metadata,tag_intent 的
            # update_current_trace 也挂不进回调创建的 trace——兜底:从 trace 的
            # classify_intent generation 的 output 解析意图(LangGraph 节点 run name 稳定)
            intent = _intent_from_observations(obs.get("data", [])) or "unknown"
        bucket = _bucket(intent)
        bucket["traces"] += 1
        for item in obs.get("data", []):
            usage = item.get("usageDetails") or {}
            bucket["input_tokens"] += int(usage.get("input") or 0)
            bucket["output_tokens"] += int(usage.get("output") or 0)
            # usageDetails.total = input+output(勿三者累加重复计费);仅 total 缺失时回退
            bucket["total_tokens"] += int(usage.get("total")
                                         or (usage.get("input") or 0) + (usage.get("output") or 0))

    grand = {
        "traces": sum(b["traces"] for b in per_intent.values()),
        "input_tokens": sum(b["input_tokens"] for b in per_intent.values()),
        "output_tokens": sum(b["output_tokens"] for b in per_intent.values()),
        "total_tokens": sum(b["total_tokens"] for b in per_intent.values()),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url, "session_id": session_id,
        "trace_total": len(traces), "grand_total": grand,
        "per_intent": dict(sorted(per_intent.items(), key=lambda kv: -kv[1]["total_tokens"])),
    }


def _intent_from_observations(observations: list[dict]) -> str | None:
    """从 trace 的 generation output 里解析意图。

    classify_intent 用 with_structured_output → tool_calls 路径,意图在
    output.tool_calls[0].args.intent(IntentLabel);文本 JSON 是旧形态兜底。
    一个 trace 可能出现多个 IntentLabel(路由节点重跑),取第一个。"""
    import re

    for item in observations:
        if item.get("type") != "GENERATION":
            continue
        output = item.get("output")
        if not isinstance(output, dict):
            continue
        for call in output.get("tool_calls") or []:
            args = call.get("args") or {}
            if call.get("name") == "IntentLabel" and args.get("intent"):
                return str(args["intent"])
        text = output.get("content", "") or ""
        if '"intent"' in text:
            m = re.search(r'"intent"\s*:\s*"([^"]+)"', text)
            if m:
                return m.group(1)
    return None


def render(report: dict) -> str:
    lines = [
        "ch09 · Cost Control · 按意图 token 账",
        f"来源: {report['base_url']} | traces={report['trace_total']}"
        + (f" | sessionId={report['session_id']}" if report.get("session_id") else ""),
        f"总计: {report['grand_total']['total_tokens']} tokens"
        f" (in {report['grand_total']['input_tokens']} / out {report['grand_total']['output_tokens']})",
        "",
        f"{'意图':<14}{'traces':>8}{'input':>10}{'output':>10}{'total':>10}",
    ]
    for intent, b in report["per_intent"].items():
        lines.append(f"{intent:<14}{b['traces']:>8}{b['input_tokens']:>10}{b['output_tokens']:>10}{b['total_tokens']:>10}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="ch09 按意图 token 账")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--max-traces", type=int, default=500)
    args = parser.parse_args()

    settings = load_settings()
    pk, sk, base = settings.langfuse_public_key, settings.langfuse_secret_key, settings.langfuse_base_url
    if not (pk and sk and base):
        print("BLOCKED:LANGFUSE_PUBLIC_KEY / SECRET_KEY / BASE_URL 未配置,无法统计(系统本身不受影响)。")
        return 2
    try:
        report = aggregate(base, pk, sk, args.session_id, args.max_traces)
    except Exception as exc:
        print(f"BLOCKED:Langfuse 拉取失败 {type(exc).__name__}: {exc}")
        return 2
    if report["trace_total"] == 0:
        print("BLOCKED:Langfuse 里没有 trace(回调未挂或无流量)。")
        return 2

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "cost_by_intent.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "cost_by_intent.txt").write_text(render(report), encoding="utf-8")
    print(render(report))
    print(f"\n产物:{REPORT_DIR / 'cost_by_intent.txt'} / .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
