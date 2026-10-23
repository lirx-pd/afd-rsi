"""Bounded Linux subprocess group execution; no shell, no global cleanup."""

import os
import signal
import subprocess
import time
from pathlib import Path

from .evidence import write_json


def group_alive(group: int) -> bool:
    # Linux zombies hold no executable work or device resources.
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group and fields[0] != "Z":
                return True
        except (FileNotFoundError, ProcessLookupError):
            continue
    return False


def execute(argv: list[str], cwd: Path, output: Path, timeout: float, env: dict) -> dict:
    # shortcut: trusted launchers keep descendants in this group; use cgroups before supporting daemonized workers.
    if not Path("/proc").is_dir():
        raise ValueError("local executor requires Linux /proc")
    output.mkdir(parents=True, exist_ok=False)
    record = {"argv": argv, "cwd": str(cwd), "status": "running", "cleanup_complete": False}
    started = time.monotonic()
    process = None
    with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
        try:
            process = subprocess.Popen(argv, cwd=cwd, env={**os.environ, **env},
                                       stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
            record["pid"] = process.pid
            write_json(output / "process.json", record)
            try:
                code = process.wait(timeout=max(0.001, timeout))
                record.update(status="completed" if code == 0 else "failed", exit_code=code)
            except subprocess.TimeoutExpired:
                record["status"] = "timeout"
            except BaseException:
                record["status"] = "interrupted"
                raise
        except OSError as error:
            record.update(status="launch_error", error=str(error))
        finally:
            if process is not None:
                remaining = group_alive(process.pid)
                if remaining:
                    if record["status"] == "completed":
                        record["status"] = "orphaned_children"
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    stop = time.monotonic() + 0.5
                    while group_alive(process.pid) and time.monotonic() < stop:
                        time.sleep(0.01)
                    if group_alive(process.pid):
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    record["status"] = "cleanup_timeout"
                stop = time.monotonic() + 1
                while group_alive(process.pid) and time.monotonic() < stop:
                    time.sleep(0.01)
                record["cleanup_complete"] = not group_alive(process.pid)
            else:
                record["cleanup_complete"] = True
            record["elapsed_seconds"] = time.monotonic() - started
            write_json(output / "process.json", record)
    return record
