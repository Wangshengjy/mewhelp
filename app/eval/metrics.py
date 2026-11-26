"""ch04 · RAG 评估指标（纯函数，TDD）。

口径与课程一致：
- 检索段：expect_section 去 `+` 分组（都得命中）、组内 `|` 或 `/` 任一命中，
  按 section_path 子串匹配算 MRR@K / Recall@K；
- 覆盖度：expect_points 去空白子串机械匹配（确定性可复现）。
"""

from __future__ import annotations

from typing import Any, Iterable


def normalize(text: str | None) -> str:
    """去所有空白，兼容「每 2 周」↔「每2周」。"""
    return "".join((text or "").split())


def split_expect_groups(expect_section: Iterable[str] | None) -> list[list[str]]:
    """[[全组1备选...], [全组2备选...]]：外层 + 都得命中，内层任一命中。"""
    return [list(group) for group in (expect_section or []) if group]


def section_groups_hit(section_paths: Iterable[str], groups: list[list[str]]) -> bool:
    """Top-K 召回的 section_path 集合是否覆盖全部期望分组（E_multi 口径）。"""
    if not groups:
        return False
    normalized_paths = [normalize(path) for path in section_paths if path]
    for group in groups:
        if not any(
            normalize(alt) and normalize(alt) in path for alt in group for path in normalized_paths
        ):
            return False
    return True


def first_relevant_rank(section_paths: Iterable[str], groups: list[list[str]]) -> int | None:
    """首个「命中至少一个分组」的 rank（1 起）；E_multi 用全组覆盖完成位。

    返回 None 表示 Top-K 内未命中。
    """
    normalized_paths = [normalize(path) for path in section_paths if path]
    if not groups or not normalized_paths:
        return None
    if len(groups) == 1:
        for rank, path in enumerate(normalized_paths, start=1):
            if any(normalize(alt) and normalize(alt) in path for alt in groups[0]):
                return rank
        return None
    covered: set[int] = set()
    for rank, path in enumerate(normalized_paths, start=1):
        for index, group in enumerate(groups):
            if index in covered:
                continue
            if any(normalize(alt) and normalize(alt) in path for alt in group):
                covered.add(index)
                if len(covered) == len(groups):
                    return rank
                break
    return None


def mrr_at_k(section_paths: Iterable[str], groups: list[list[str]], k: int) -> float:
    rank = first_relevant_rank(section_paths, groups)
    if rank is None or rank > k:
        return 0.0
    return 1.0 / rank


def recall_at_k(section_paths: Iterable[str], groups: list[list[str]], k: int) -> float:
    """单组：Top-K 内命中任一 → 1；多组：全部组都被覆盖 → 1。"""
    normalized_paths = list(section_paths)[:k] if section_paths else []
    if not groups:
        return 0.0
    if len(groups) == 1:
        return 1.0 if first_relevant_rank(normalized_paths, groups) is not None else 0.0
    if not section_groups_hit(normalized_paths, groups):
        return 0.0
    return 1.0


def evidence_coverage(expect_points: Iterable[str], evidence_answers: Iterable[str]) -> float | None:
    """Top-K 证据答案拼接后，expect_points 去空白子串命中的比例；无要点返回 None。"""
    points = [point for point in (expect_points or []) if normalize(point)]
    if not points:
        return None
    haystack = normalize("\n".join(evidence_answers or []))
    return sum(1 for point in points if normalize(point) in haystack) / len(points)


def aggregate(scores: Iterable[float | None]) -> float | None:
    """均值；空/全 None 返回 None（报告显示「—」）。"""
    values = [float(value) for value in scores if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def bucket_aggregate(records: list[dict], metric: str) -> dict[str, Any]:
    """按 bucket 聚合某指标：{bucket: mean, "overall": mean}。"""
    by_bucket: dict[str, list[float | None]] = {}
    for record in records:
        by_bucket.setdefault(record["bucket"], []).append(record.get(metric))
    result: dict[str, Any] = {
        bucket: aggregate(values) for bucket, values in by_bucket.items()
    }
    result["overall"] = aggregate([value for values in by_bucket.values() for value in values])
    return result


REFUSAL_PATTERNS = (
    "没有查到",
    "未查到",
    "暂时没有",
    "暂时无法",
    "无法查询",
    "无法提供",
    "不确定",
    "没有相关",
    "以平台实际",
    "以商品页公示",
    "以页面公示",
    "人工客服",
    "联系客服",
    "抱歉",
)


def is_refusal(answer: str | None) -> bool:
    """机械判拒答：D 桶期望拒答，命中任一话术模式即视为拒答。"""
    text = normalize(answer)
    return any(normalize(pattern) in text for pattern in REFUSAL_PATTERNS)
