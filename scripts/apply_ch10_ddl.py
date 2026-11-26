"""ch10 · 一次性把 sql/ch10-ddl.sql 应用到 dev MySQL（pymysql，幂等跳过已存在的对象）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pymysql  # noqa: E402

from scripts._common import load_settings  # noqa: E402


def main() -> int:
    settings = load_settings()
    url = settings.database_url
    conn = pymysql.connect(
        host=url.host, port=int(url.port or 3306), user=url.username,
        password=url.password, database=url.database, charset="utf8mb4",
    )
    ddl = (ROOT / "sql" / "ch10-ddl.sql").read_text(encoding="utf-8")
    body = "\n".join(line for line in ddl.splitlines() if not line.strip().startswith("--"))
    statements = [s.strip() for s in body.split(";") if s.strip()]
    applied, skipped = [], []
    with conn.cursor() as cur:
        for stmt in statements:
            try:
                cur.execute(stmt)
                applied.append(stmt.splitlines()[0][:60])
            except pymysql.err.OperationalError as exc:
                code = exc.args[0] if exc.args else None
                if code in (1050, 1060, 1061):  # table/column/key already exists
                    skipped.append(f"{stmt.splitlines()[0][:50]} → {exc.args[1]}")
                else:
                    raise
        conn.commit()
    print("已应用：")
    for item in applied:
        print(f"  + {item}")
    if skipped:
        print("已存在跳过：")
        for item in skipped:
            print(f"  - {item}")
    with conn.cursor() as cur:
        cur.execute("SHOW CREATE TABLE topic_classifications")
        print(cur.fetchone()[1])
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
