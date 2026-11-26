"""清空 ch09 测试残留（dev 库）：LCQ / review_queue / eval_runs / 知识库写回条目（含 Milvus 向量）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pymysql  # noqa: E402

from scripts._common import load_settings  # noqa: E402


def main() -> None:
    u = load_settings().database_url
    c = pymysql.connect(host=u.host, port=int(u.port or 3306), user=u.username,
                        password=u.password, database=u.database, charset="utf8mb4")
    cur = c.cursor()
    cur.execute("SELECT id, questions, vector_id FROM knowledge_chunks "
                "WHERE questions LIKE %s OR questions LIKE %s", ("%保质期%", "%注销%"))
    polluted = cur.fetchall()
    cur.execute("DELETE FROM low_confidence_questions")
    cur.execute("DELETE FROM review_queue")
    cur.execute("DELETE FROM eval_runs")
    c.commit()
    for table in ("low_confidence_questions", "review_queue", "eval_runs"):
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        print(table, "=", cur.fetchone()[0])
    print("知识库污染行:", polluted)
    if polluted:
        s = load_settings()
        from app.kb.milvus_store import MilvusStore
        ids = [row[0] for row in polluted]
        cur.execute("DELETE FROM knowledge_chunks WHERE id IN (%s)" % ",".join(str(i) for i in ids))
        c.commit()
        MilvusStore(s.milvus_uri).delete(ids)
        print(f"已删 knowledge_chunks 行 + Milvus 向量: {ids}")
    c.close()


if __name__ == "__main__":
    main()
