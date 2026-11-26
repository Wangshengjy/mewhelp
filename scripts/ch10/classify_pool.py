"""ch10 旁路批处理：低置信度问题攒够一批，整批喂分类器归一次类，结果写 topic_classifications。

实时对话主链路不调它。运行：python -m scripts.ch10.classify_pool [--min-batch 10] [--force]
（需 mysql + 分类器服务 :8110）。幂等：已归类的行（LEFT JOIN 命中）不重复归。
定时跑示例（Windows 计划任务/其他平台 cron 同理）：
  python -m scripts.ch10.classify_pool >> log/classify-pool.log 2>&1
运行报告 classify_run.json 区分 done / empty / below_batch——第二遍跑出 empty 就是幂等的实证。
"""
import argparse
import asyncio
import json
import pathlib
import sys
from collections import Counter
from datetime import datetime, timezone

import httpx

from app.repositories.topic_classification import (insert_topic_classifications,
                                                   list_unclassified_questions)
from scripts._common import load_settings, session_factory_from

SERVICE = "http://127.0.0.1:8110"
REPORTS = pathlib.Path("data/ch10/reports")


async def main(min_batch: int, force: bool) -> int:
    settings = load_settings()
    session_factory = session_factory_from(settings)
    with session_factory() as session:
        rows = list_unclassified_questions(session, limit=500)
    status, written = "empty", 0
    counts: dict[str, int] = {}
    if not rows:
        print("池里没有待归类问题")
    elif len(rows) < min_batch and not force:
        print(f"待归类 {len(rows)} 条，不足一批（{min_batch} 条）；--force 可强跑")
        status = "below_batch"
    else:
        # trust_env=False：本机服务不走系统代理（Windows 上 httpx 会读注册表代理设置，
        # 把 127.0.0.1 请求送进代理导致 ReadError）
        async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
            r = await client.post(f"{SERVICE}/classify",
                                  json={"texts": [x["text"] for x in rows]})
            r.raise_for_status()
            results = r.json()["results"]
        with session_factory() as session:
            written = insert_topic_classifications(
                session, [{"question_id": x["question_id"], "labels": res["labels"]}
                          for x, res in zip(rows, results)])
            session.commit()
        counter = Counter()
        for res in results:
            for lb in res["labels"]:
                counter[lb] += 1
        counts = dict(counter.most_common())
        status = "done"
        print(f"归类完成：{written} 条写入 topic_classifications；"
              + " ".join(f"{k}={v}" for k, v in counts.items()))

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "classify_run.json").write_text(json.dumps({
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "pending": len(rows),
        "written": written,
        "counts": counts,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"运行报告已落 {REPORTS / 'classify_run.json'}（status={status}）")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch10 旁路批量归类")
    ap.add_argument("--min-batch", type=int, default=10, help="攒够多少条才归一次")
    ap.add_argument("--force", action="store_true", help="不足一批也强跑")
    a = ap.parse_args()
    sys.exit(asyncio.run(main(a.min_batch, a.force)))
