"""ch08 · 起两台业务 MCP Server(独立后台进程,Windows)。

用法:.\\.venv\\Scripts\\python.exe scripts/mcp_up.py
- 后台进程 stdout/stderr 落 log/mcp-*.log,pid 落 data/mcp-*.pid;
- MOCK_DELAY_SECONDS 可为工具调用注入人为延迟(验收 6);
- 已在跑(端口被占)则跳过,幂等。
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SERVERS = [
    ("logistics", ROOT / "mcp_servers" / "logistics_server.py", 8101),
    ("aftersales", ROOT / "mcp_servers" / "aftersales_server.py", 8102),
]

# Windows:独立进程组脱离父进程,关掉控制台窗口也不死
_CREATIONFLAGS = 0
if sys.platform == "win32":
    _CREATIONFLAGS = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _read_pid(pid_file: Path) -> int | None:
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def up() -> None:
    (ROOT / "log").mkdir(exist_ok=True)
    (ROOT / "data").mkdir(exist_ok=True)
    for name, script, port in SERVERS:
        pid_file = ROOT / "data" / f"mcp-{name}.pid"
        if _port_open(port):
            print(f"mcp-{name}: 端口 {port} 已被占用(可能已在跑),跳过")
            continue
        log_file = ROOT / "log" / f"mcp-{name}.log"
        env = {**os.environ, "PORT": str(port), "PYTHONIOENCODING": "utf-8"}
        with log_file.open("ab") as log_handle:
            proc = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(ROOT),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=_CREATIONFLAGS,
            )
        pid_file.write_text(str(proc.pid), encoding="utf-8")
        print(f"mcp-{name}: pid={proc.pid} port={port} log={log_file.name}")
    # 起完探活 5s,失败提示看日志
    for _name, _script, port in SERVERS:
        deadline = time.time() + 5
        while time.time() < deadline and not _port_open(port):
            time.sleep(0.2)
        if _port_open(port):
            print(f"  :{port} 就绪")
        else:
            print(f"  :{port} 5s 内未就绪——请查看 log/mcp-*.log")


if __name__ == "__main__":
    up()
