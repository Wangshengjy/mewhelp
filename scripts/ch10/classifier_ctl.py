"""ch10 推理服务起停（Windows PowerShell 5，无 nohup/make）。

python -m scripts.ch10.classifier_ctl start|stop|status
start：后台起 serve.py（uvicorn :8110），pid 落 data/classifier.pid，日志追加 log/classifier.log；
      起完探 /healthz 直到就绪（默认 30s 超时）。
stop：读 pid 文件 os.kill 终止（Windows 上 TerminateProcess），删 pid 文件。
"""
import argparse
import pathlib
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PID_FILE = ROOT / "data" / "classifier.pid"
LOG_FILE = ROOT / "log" / "classifier.log"
HEALTHZ = "http://127.0.0.1:8110/healthz"


def _probe(seconds: int = 30) -> bool:
    import httpx

    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            # trust_env=False：本机探活不走系统代理（Windows 注册表代理会把 127.0.0.1 也拦下）
            r = httpx.get(HEALTHZ, timeout=2, trust_env=False)
            if r.status_code == 200:
                return True
        except Exception:
            time.sleep(0.5)
    return False


def start() -> int:
    if PID_FILE.exists() and _probe(3):
        print(f"分类器服务已在跑：:8110（pid {PID_FILE.read_text().strip()}）")
        return 0
    if PID_FILE.exists():
        PID_FILE.unlink()                        # 残留 pid（上次没停干净）
    LOG_FILE.parent.mkdir(exist_ok=True)
    # CIM Win32_Process.Create 起服务：父进程是 WmiPrvSE，不属于当前 shell 的进程树/作业对象，
    # 命令结束后服务照常存活（Popen 子进程会被调用方回收，实测起完即死）。
    # 外面再包一层 powershell（提供有效 std 句柄并把崩溃栈留到 stderr 日志）；
    # 真正的 python 进程自写 pid 文件（serve.py 启动即写），stop 直接按 pid 收。
    launcher_ps1 = ROOT / "data" / "classifier-launch.ps1"
    launcher_ps1.write_text(
        f'& "{sys.executable}" "{ROOT / "scripts" / "ch10" / "serve.py"}"'
        f' *>> "{ROOT / "log" / "classifier-stderr.log"}"\n',
        encoding="utf-8")
    commandline = (f'powershell.exe -NoProfile -ExecutionPolicy Bypass '
                   f'-File "{launcher_ps1}"')
    code, _ = _cim_create(commandline)
    if code != 0:
        print(f"启动失败：CIM Win32_Process.Create 返回 {code}")
        return 1
    if _probe():
        pid = PID_FILE.read_text().strip() if PID_FILE.exists() else "?"
        print(f"分类器服务已拉起：:8110（pid {pid}，日志 {LOG_FILE}）")
        return 0
    stderr_log = ROOT / "log" / "classifier-stderr.log"
    print(f"启动失败，看 {LOG_FILE} 与 {stderr_log}")
    return 1


def _cim_create(commandline: str):
    """经 PowerShell Invoke-CimMethod 起独立进程（父进程 WmiPrvSE，脱离当前 shell 进程树）。
    返回 (returncode, pid)。不用 pywin32 的 WMI 后期绑定（.Create 与属性绑定冲突）。"""
    ps = (
        "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
        f"-Arguments @{{ CommandLine = '{commandline}'; CurrentDirectory = '{ROOT}' }}; "
        "Write-Output ($r.ReturnValue.ToString() + ' ' + $r.ProcessId.ToString())"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True, text=True, timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    out = (proc.stdout or "").strip().splitlines()
    if not out or " " not in out[-1]:
        return 1, -1
    code, pid = out[-1].split(" ", 1)
    return int(code), int(pid)


def stop() -> int:
    if not PID_FILE.exists():
        print("分类器服务未在跑（无 pid 文件）")
        return 0
    pid = int(PID_FILE.read_text().strip())
    try:
        import os

        os.kill(pid, signal.SIGTERM)             # Windows → TerminateProcess
        print(f"分类器服务已停（pid {pid}）")
    except ProcessLookupError:
        print(f"pid {pid} 已不存在（清理残留 pid 文件）")
    finally:
        PID_FILE.unlink()
    return 0


def status() -> int:
    alive = _probe(3)
    pid = PID_FILE.read_text().strip() if PID_FILE.exists() else "-"
    print(f":8110 {'在线' if alive else '离线'}（pid {pid}）")
    return 0 if alive else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch10 分类器服务起停")
    ap.add_argument("action", choices=["start", "stop", "status"])
    a = ap.parse_args()
    sys.exit({"start": start, "stop": stop, "status": status}[a.action]())
