"""ch08 · 停两台业务 MCP Server(读 data/mcp-*..pid 杀进程)。

用法:.\\.venv\\Scripts\\python.exe scripts/mcp_down.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SERVERS = ["logistics", "aftersales"]


def down() -> None:
    for name in SERVERS:
        pid_file = ROOT / "data" / f"mcp-{name}.pid"
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            print(f"mcp-{name}: 无 pid 记录,跳过")
            continue
        if sys.platform == "win32":
            result = subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, text=True)
            print(f"mcp-{name}: pid={pid} taskkill {'OK' if result.returncode == 0 else '失败(可能已退出)'}")
        else:
            subprocess.run(["kill", str(pid)], check=False)
            print(f"mcp-{name}: pid={pid} killed")
        pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    down()
