"""Bounded local execution and PID-safe recovery of experiment-owned processes."""

import os
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path

from .evidence import write_json


def host_identity():
    return {"host": socket.gethostname(), "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def group_alive(group: int) -> bool:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group and fields[0] not in ("Z", "X"):
                return True
        except (FileNotFoundError, ProcessLookupError):
            continue
    return False


def tagged_processes(tag):
    """Return pidfds bound to live processes carrying this unguessable trial tag."""
    handles = []
    marker = f"AFD_RSI_PROCESS_TAG={tag}".encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        fd = None
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            fd = os.pidfd_open(int(entry.name))
            fields = (entry / "stat").read_bytes().rsplit(b")", 1)[1].split()
            environment = (entry / "environ").read_bytes().split(b"\0")
            ancestry = next((x.split(b"=", 1)[1].split(b":") for x in environment
                             if x.startswith(b"AFD_RSI_ANCESTOR_TAGS=")), [])
            if fields[0] in (b"Z", b"X") or (marker not in environment and tag.encode() not in ancestry):
                continue
            handles.append((int(entry.name), fd))
            fd = None
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
        finally:
            if fd is not None:
                os.close(fd)
    return handles


def signal_owned(tag, sig, *, leader=None):
    handles = tagged_processes(tag)
    try:
        for pid, fd in handles:
            if leader is not None and pid != leader:
                continue
            try:
                signal.pidfd_send_signal(fd, sig)
            except ProcessLookupError:
                pass
        return len(handles)
    finally:
        for _, fd in handles:
            os.close(fd)


def owned_alive(tag):
    handles = tagged_processes(tag)
    for _, fd in handles:
        os.close(fd)
    return bool(handles)


def stop_owned(record, grace=0.5):
    if record.get("host_identity") != host_identity() or not record.get("tag"):
        raise ValueError("cannot recover a process without matching host/boot and ownership tag")
    tag = record["tag"]
    signal_owned(tag, signal.SIGTERM, leader=record.get("pid"))
    deadline = time.monotonic() + grace
    while owned_alive(tag) and time.monotonic() < deadline:
        time.sleep(0.05)
    forced = owned_alive(tag)
    if forced:
        signal_owned(tag, signal.SIGTERM)
        deadline = time.monotonic() + 0.25
        while owned_alive(tag) and time.monotonic() < deadline:
            time.sleep(0.01)
        signal_owned(tag, signal.SIGKILL)
    deadline = time.monotonic() + 2
    while owned_alive(tag) and time.monotonic() < deadline:
        time.sleep(0.02)
    return {"cleanup_complete": not owned_alive(tag), "forced_cleanup": forced}


def execute(argv: list[str], cwd: Path, output: Path, timeout: float, env: dict,
            *, grace_seconds=0.5) -> dict:
    if not Path("/proc").is_dir():
        raise ValueError("local executor requires Linux /proc")
    try:
        if not hasattr(signal, "pidfd_send_signal"):
            raise OSError("pidfd signals unavailable")
        fd = os.pidfd_open(os.getpid())
        os.close(fd)
    except (AttributeError, OSError) as error:
        raise ValueError("executor requires working Linux pidfds before launching a process") from error
    output.mkdir(parents=True, exist_ok=False)
    record = {"argv": argv, "cwd": str(cwd), "status": "running", "cleanup_complete": False,
              "started_at": time.time(), "host_identity": host_identity(), "tag": uuid.uuid4().hex,
              "grace_seconds": grace_seconds}
    write_json(output / "process.json", record)
    started = time.monotonic()
    process = None
    with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
        try:
            ancestry = ":".join(filter(None, (os.environ.get("AFD_RSI_ANCESTOR_TAGS"), os.environ.get("AFD_RSI_PROCESS_TAG"))))
            process = subprocess.Popen(argv, cwd=cwd, env={**os.environ, **env, "AFD_RSI_PROCESS_TAG": record["tag"],
                                                         "AFD_RSI_ANCESTOR_TAGS": ancestry},
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
                if owned_alive(record["tag"]):
                    if record["status"] == "completed":
                        record["status"] = "orphaned_children"
                    record.update(stop_owned(record, grace_seconds))
                else:
                    record["cleanup_complete"] = True
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    record["status"] = "cleanup_timeout"
                    record["cleanup_complete"] = False
            else:
                record["cleanup_complete"] = True
            record["elapsed_seconds"] = time.monotonic() - started
            write_json(output / "process.json", record)
    return record
