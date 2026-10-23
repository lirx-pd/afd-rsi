import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def invoke(*args):
    return subprocess.run([sys.executable, "-m", "afd_rsi", "service", *args], cwd=ROOT,
                          env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                          text=True, capture_output=True, timeout=30)


@pytest.mark.parametrize('execution', ['eager', 'graph'])
def test_v4_megamoe_plan_uses_actual_afd_serve_templates(tmp_path, execution):
    result = invoke("plan", "--repo", str(tmp_path / "target"), "--model", "v4-flash",
                    "--model-path", str(tmp_path / "weights"), "--device", ",".join(map(str, range(16))),
                    "--comm", "async", "--mrv", "mrv2", "--layered-gmm", "on",
                    "--moe-ubatching", "off", "--execution", execution, "--dbo", "off",
                    "--serve-env", "AFD_ASYNC_CAM_MEGA_MOE=1", "--test", "smoke",
                    "--nic", "lo", "--afd-host", "127.0.0.1")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["configuration"]["topology"] == "dp2tp4-ep8"
    assert plan["attention"] == {"ranks": 8, "dp": 2, "tp": 4}
    assert plan["ffn"] == {"ranks": 8, "dp": 8, "tp": 1, "ep": 8}
    services = {service["role"]: service for service in plan["services"]}
    assert set(services) == {"attention", "ffn"}
    assert services["attention"]["devices"] == list(map(str, range(8)))
    assert services["ffn"]["devices"] == list(map(str, range(8, 16)))
    assert services["ffn"]["environment"]["VLLM_PLUGINS"] == "ascend,afd"
    assert services["ffn"]["environment"]["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert services["ffn"]["environment"]["AFD_ASYNC_CAM_LAYERED_GMM"] == "1"
    assert services["ffn"]["environment"]["AFD_ASYNC_CAM_MEGA_MOE"] == "1"
    for service in services.values():
        command = service["command"]
        assert ('--enforce-eager' in command) == (execution == 'eager' or service['role'] == 'attention')
        if execution == 'graph' and service['role'] == 'ffn':
            compilation = json.loads(command[command.index('--compilation-config') + 1])
            assert compilation['cudagraph_mode'] == 'FULL'
            assert compilation['cudagraph_capture_sizes']
        additional = json.loads(command[command.index("--additional-config") + 1])
        extra = additional["afd"]["connector_extra_config"]
        assert set(extra) == {"hccl_buffer_size", "dynamicQuant", "attn_ranks_per_dp"}
        assert extra["dynamicQuant"] == 1 and extra["attn_ranks_per_dp"] == 4
        assert command[command.index("--max-model-len") + 1] == "32768"
    assert all(service["command"][:2] == ["vllm", "serve"] for service in services.values())
    text = json.dumps(plan)
    assert "serve" in text
    assert "ascend,afd" in text
    assert "AFD_ASYNC_CAM_MEGA_MOE" in text
    assert "AFD_ASYNC_CAM_LAYERED_GMM" in text
    assert "v4-flash" in text


def test_service_requires_explicit_target_repo():
    result = invoke("plan", "--device", "0,1,2,3", "--nic", "lo", "--afd-host", "127.0.0.1")
    assert result.returncode != 0
    assert "explicit --repo" in result.stderr


def test_service_rejects_removed_async_configuration_before_starting():
    result = invoke("plan", "--repo", "/work/target", "--comm", "async", "--moe-ubatching", "on")
    assert result.returncode == 2
    assert "no async_moe_ubatching field" in result.stderr


def test_pinned_service_waits_for_actual_models_http(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks/service"))
    from afd_e2e.cli import parser
    from afd_e2e.config import from_args
    from afd_e2e.runner import wait_ready

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/v1/models"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"afd-e2e-baseline"}]}')

        def log_message(self, *args):
            pass

    try:
        server = HTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError:
        pytest.skip("local socket binding unavailable; real HTTP readiness not validated")
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        case = from_args(parser().parse_args(["run", "--baseline", "--topology", "dp1",
                         "--device", "0", "--api-port", str(server.server_port),
                         "--nic", "lo", "--afd-host", "127.0.0.1", "--startup-timeout", "3"]))
        wait_ready(case, [SimpleNamespace(ensure_alive=lambda: None)])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_pinned_runner_rejects_exited_service_before_http(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks/service"))
    from afd_e2e.cli import parser
    from afd_e2e.config import from_args
    from afd_e2e.runner import wait_ready

    def exited():
        raise RuntimeError("FFN process exited")

    case = from_args(parser().parse_args(["run", "--baseline", "--topology", "dp1", "--device", "0"]))
    with pytest.raises(RuntimeError, match="FFN process exited"):
        wait_ready(case, [SimpleNamespace(ensure_alive=exited)])
