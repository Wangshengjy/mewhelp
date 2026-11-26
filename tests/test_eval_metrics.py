"""ch04 · 评估指标纯函数测试（固定样例，TDD）。

口径：expect_section `+` 分组（都得命中）、组内任一命中、section_path 子串匹配；
MRR@K / Recall@K（K=5）；expect_points 去空白子串机械匹配。
"""

import pytest

from app.eval.metrics import (
    aggregate,
    bucket_aggregate,
    evidence_coverage,
    first_relevant_rank,
    is_refusal,
    mrr_at_k,
    normalize,
    recall_at_k,
    section_groups_hit,
    split_expect_groups,
)


def test_normalize_strips_all_whitespace():
    assert normalize("每 2 周\t包邮\n") == "每2周包邮"
    assert normalize(None) == ""


def test_split_expect_groups_drops_empty_groups():
    assert split_expect_groups([["退货"], [], ["换货", "退换"]]) == [["退货"], ["换货", "退换"]]
    assert split_expect_groups(None) == []


def test_section_groups_hit_requires_every_group_covered():
    paths = ["A > 退货运费", "B > 换货时效"]
    assert section_groups_hit(paths, [["退货运费"], ["换货"]]) is True
    assert section_groups_hit(paths, [["退货运费"], ["赔付"]]) is False
    assert section_groups_hit(paths, []) is False


def test_first_relevant_rank_single_group():
    groups = [["换货运费"]]
    assert first_relevant_rank(["无关", "退货退款政策 > 换货运费"], groups) == 2
    assert first_relevant_rank(["无关", "也无关"], groups) is None


def test_first_relevant_rank_multi_group_completes_full_coverage():
    groups = [["退货运费"], ["换货"]]
    paths = ["退货退款政策 > 换货政策", "退货退款政策 > 退货运费", "尾部垫底"]
    assert first_relevant_rank(paths, groups) == 2, "两段各覆盖一组，第 2 名完成全组覆盖"


def test_mrr_at_k_fixed_samples():
    groups = [["包邮"]]
    assert mrr_at_k(["满99包邮"], groups, k=5) == 1.0
    assert mrr_at_k(["无关", "无关", "满99包邮"], groups, k=5) == pytest.approx(1 / 3)
    assert mrr_at_k(["无关"] * 5, groups, k=5) == 0.0
    assert mrr_at_k(["无关"] * 5 + ["满99包邮"], groups, k=5) == 0.0, "rank 6 > K=5 计 0"
    assert mrr_at_k(["无关"] * 4 + ["满99包邮"], groups, k=5) == pytest.approx(0.2)


def test_recall_at_k_single_and_multi_group():
    assert recall_at_k(["无关", "满99包邮"], [["包邮"]], k=5) == 1.0
    assert recall_at_k(["无关", "无关"], [["包邮"]], k=5) == 0.0
    groups = [["退货运费"], ["换货"]]
    assert recall_at_k(["换货", "退货运费"], groups, k=5) == 1.0
    assert recall_at_k(["换货", "无关"], groups, k=5) == 0.0, "多组必须全部覆盖"
    assert recall_at_k(["无关", "退货运费"], groups, k=1) == 0.0, "K 截断后再判覆盖"


def test_evidence_coverage_is_substring_and_none_without_points():
    answers = ["单笔订单满 99 元包邮,未满收取 10 元运费。"]
    assert evidence_coverage(["满 99 元包邮", "10 元运费", "7 天无理由"], answers) == 2 / 3
    assert evidence_coverage([], answers) is None


def test_aggregate_skips_none_and_handles_empty():
    assert aggregate([0.5, None, 1.0]) == 0.75
    assert aggregate([None]) is None
    assert aggregate([]) is None


def test_bucket_aggregate_reports_per_bucket_and_overall():
    records = [
        {"bucket": "A_policy", "mrr": 1.0},
        {"bucket": "A_policy", "mrr": 0.0},
        {"bucket": "B_model", "mrr": None},
    ]
    report = bucket_aggregate(records, "mrr")
    assert report["A_policy"] == 0.5
    assert report["B_model"] is None
    assert report["overall"] == 0.5


def test_is_refusal_matches_refusal_phrases_only():
    assert is_refusal("暂时没有查到相关信息，请联系人工客服。") is True
    assert is_refusal("以平台实际处理进度为准") is True
    assert is_refusal("单笔订单满 99 元包邮。") is False
    assert is_refusal(None) is False
