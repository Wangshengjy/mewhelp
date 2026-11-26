"""ch10 · topic_classifications：池捞取 / 写表 / 分布聚合 / 类目问题列表（TDD）。

口径：
- 分类旁路只处理已归并的问题（matched_review_id 非空），文本取 review_queue.normalized_question；
  未归并的不进分类，留到下一轮归并后再归——分类器只吃语义完整的标准化问法，不喂原话。
- 训练语料捞取（list_pool_texts）保留原话兜底：建训练集是一次性全量、模型冻结，无标准化问法时退回原话。
- labels 是 JSON 数组，聚合/筛选/分页都在 Python 侧做：归类总量是问题池规模（百级），
  换来不依赖 MySQL 的 JSON 函数，测试库上跑得到同样结果。
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.taxonomy import TOPIC_NAMES
from app.models import LowConfidenceQuestion, ReviewQueue, TopicClassification

# ---------------------------------------------------------------- 池捞取


def _pool_text_stmt():
    """池问题取文本的公共查询：优先归并后的标准化问法，没归并退回原话。"""
    return (
        select(
            LowConfidenceQuestion.id,
            LowConfidenceQuestion.raw_question,
            ReviewQueue.normalized_question,
            LowConfidenceQuestion.created_at,
        )
        .outerjoin(ReviewQueue, LowConfidenceQuestion.matched_review_id == ReviewQueue.id)
        .order_by(LowConfidenceQuestion.id)
    )


def list_pool_texts(session: Session) -> list[dict]:
    """ch10 训练语料捞取：全量低置信度问题（question_id 溯源）。"""
    rows = session.execute(_pool_text_stmt()).all()
    return [
        {"question_id": qid, "text": norm or raw, "created_at": created}
        for qid, raw, norm, created in rows
    ]


def list_unclassified_questions(session: Session, limit: int = 500) -> list[dict]:
    """ch10 旁路批处理捞取：尚未归类、且已归并的低置信度问题。

    只归有 matched_review_id 的问题——分类器只吃归并阶段产出的标准化问法，
    未归并的留到下一轮归并后再归，不喂原话。
    """
    stmt = (
        _pool_text_stmt()
        .outerjoin(
            TopicClassification,
            TopicClassification.question_id == LowConfidenceQuestion.id,
        )
        .where(TopicClassification.id.is_(None))
        .where(LowConfidenceQuestion.matched_review_id.is_not(None))
        .limit(limit)
    )
    rows = session.execute(stmt).all()
    return [{"question_id": qid, "text": norm or raw} for qid, raw, norm, _ in rows]


# ---------------------------------------------------------------- 写表


def insert_topic_classifications(session: Session, rows: list[dict]) -> int:
    """批量写归类结果；rows: [{question_id, labels}]。同一问题重复写靠 uk_question_id 拦下。"""
    for r in rows:
        session.add(
            TopicClassification(question_id=r["question_id"], labels=list(r["labels"]))
        )
    session.flush()
    return len(rows)


# ---------------------------------------------------------------- 读：分布与类目列表


def topic_distribution(session: Session, samples_per_class: int = 3) -> dict:
    """主题分布：17 类各自问题量 + 每类样例（样例按文本去重，同一句标准化问法
    对应池里好几行时三条样例不能都一样；计数照旧按条算）。

    返回 {total, latest, classes: [{label, count, samples}]}，17 类全出、顺序同 TOPIC_NAMES。
    """
    stmt = (
        select(
            TopicClassification.labels,
            LowConfidenceQuestion.raw_question,
            ReviewQueue.normalized_question,
            TopicClassification.classified_at,
        )
        .join(LowConfidenceQuestion, TopicClassification.question_id == LowConfidenceQuestion.id)
        .outerjoin(ReviewQueue, LowConfidenceQuestion.matched_review_id == ReviewQueue.id)
        .order_by(TopicClassification.id)
    )
    rows = session.execute(stmt).all()
    counts = {name: 0 for name in TOPIC_NAMES}
    samples: dict[str, list[str]] = {name: [] for name in TOPIC_NAMES}
    latest = None
    for labels, raw, norm, ts in rows:
        text = norm or raw
        latest = ts if latest is None or ts > latest else latest
        for lb in labels or []:
            if lb in counts:
                counts[lb] += 1
                if len(samples[lb]) < samples_per_class and text not in samples[lb]:
                    samples[lb].append(text)
    return {
        "total": len(rows),
        "latest": latest.isoformat() if latest else None,
        "classes": [
            {"label": name, "count": counts[name], "samples": samples[name]}
            for name in TOPIC_NAMES
        ],
    }


def topic_questions(session: Session, label: str, page: int = 1, size: int = 10) -> dict:
    """某一类目下归类到的全部问题，分页；多标签问题在它命中的每个类目下都出现。

    未归并的显示原话并标 normalized=false、审核状态为空；归并过的显示标准化问法、
    另给一份原话、审核状态跟着 review_queue。类目名合法性由 API 层先校验（回 400）。
    """
    stmt = (
        select(
            TopicClassification.question_id,
            TopicClassification.labels,
            TopicClassification.classified_at,
            LowConfidenceQuestion.raw_question,
            LowConfidenceQuestion.source,
            LowConfidenceQuestion.created_at,
            ReviewQueue.normalized_question,
            ReviewQueue.occurrence_count,
            ReviewQueue.review_status,
        )
        .join(LowConfidenceQuestion, TopicClassification.question_id == LowConfidenceQuestion.id)
        .outerjoin(ReviewQueue, LowConfidenceQuestion.matched_review_id == ReviewQueue.id)
        .order_by(TopicClassification.id)
    )
    rows = session.execute(stmt).all()
    items: list[dict] = []
    for qid, labels, classified_at, raw, source, created_at, norm, occ, status in rows:
        hit = [lb for lb in (labels or []) if lb]
        if label not in hit:
            continue
        items.append({
            "question_id": qid,
            "text": norm or raw,                # 展示口径：标准化问法优先
            "raw_question": raw,                # 原话另给一份
            "normalized": norm is not None,
            "labels": hit,
            "peer_labels": [lb for lb in hit if lb != label],
            "source": source,
            "occurrence_count": occ if occ is not None else 1,  # 同义合并条数
            "review_status": status,
            "classified_at": classified_at.isoformat() if classified_at else None,
            "created_at": created_at.isoformat() if created_at else None,
        })
    total = len(items)
    pages = max(1, (total + size - 1) // size)
    page = max(1, min(page, pages))
    start = (page - 1) * size
    return {
        "label": label,
        "total": total,
        "page": page,
        "pages": pages,
        "size": size,
        "items": items[start:start + size],
    }
