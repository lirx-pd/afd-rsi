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
                    "--ffn-mega-moe", "on", "--test", "smoke",
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
    assert plan["configuration"]["layered_gmm"] is True
    assert services["ffn"]["environment"]["BATCH_SIZE_FACTOR"] == "0.125"
    for service in services.values():
        command = service["command"]
        assert command[command.index('--data-parallel-rpc-port') + 1] == str(19282 + int(service['role'] == 'ffn'))
        assert ('--enforce-eager' in command) == (execution == 'eager' or service['role'] == 'attention')
        if execution == 'graph' and service['role'] == 'ffn':
            compilation = json.loads(command[command.index('--compilation-config') + 1])
            assert compilation['cudagraph_mode'] == 'FULL'
            assert compilation['cudagraph_capture_sizes']
        additional = json.loads(command[command.index("--additional-config") + 1])
        extra = additional["afd"]["connector_extra_config"]
        assert set(extra) == {"hccl_buffer_size", "dynamicQuant", "attn_ranks_per_dp", "ffn_mega_moe"}
        assert extra["dynamicQuant"] == 1 and extra["attn_ranks_per_dp"] == 4
        # The Mega switch lives in the connector config now, not in the environment.
        assert extra["ffn_mega_moe"] is True
        assert command[command.index("--max-model-len") + 1] == "32768"
    assert all(service["command"][:2] == ["vllm", "serve"] for service in services.values())
    text = json.dumps(plan)
    assert "serve" in text
    assert "ascend,afd" in text
    assert "ffn_mega_moe" in text
    assert "AFD_ASYNC_CAM_LAYERED_GMM" not in text
    assert "v4-flash" in text


@pytest.mark.parametrize('option', [[], ['--layered-gmm', 'on'], ['--layered-gmm', 'off']])
def test_async_plan_reports_mandatory_layered_execution(tmp_path, monkeypatch, option):
    monkeypatch.syspath_prepend(str(ROOT / 'benchmarks/service'))
    from afd_e2e.cli import parser
    from afd_e2e.config import from_args
    from afd_e2e.planner import describe
    args = ['plan', '--repo', str(tmp_path), '--model', 'v4-flash', '--model-path', '/prepared/model',
            '--comm', 'async', '--test', 'smoke', '--nic', 'lo', '--afd-host', '127.0.0.1']
    plan = describe(from_args(parser().parse_args(args + option)))
    assert plan['configuration']['layered_gmm'] is True
    assert all('AFD_ASYNC_CAM_LAYERED_GMM' not in item['environment'] for item in plan['services'])


@pytest.mark.parametrize('mega', ['on', 'off'])
def test_async_plan_rejects_unquantized_execution_before_startup(tmp_path, mega):
    result = invoke('plan', '--repo', str(tmp_path), '--model', 'v4-flash',
                    '--model-path', '/prepared/model', '--comm', 'async',
                    '--ffn-mega-moe', mega, '--dynamic-quant', '0', '--test', 'smoke')
    assert result.returncode == 2 and 'dynamicQuant=1' in result.stderr


def test_service_requires_explicit_target_repo():
    result = invoke("plan", "--device", "0,1,2,3", "--nic", "lo", "--afd-host", "127.0.0.1")
    assert result.returncode != 0
    assert "explicit --repo" in result.stderr


def test_gsm8k_probe_checks_dependencies_before_startup(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'benchmarks/service'))
    from afd_e2e import probe

    monkeypatch.setattr(probe, 'check_ports', lambda *args: None)
    args = SimpleNamespace(api_host='127.0.0.1', ports=(), nic=None,
                           model=tmp_path / 'missing', devices='0',
                           gsm8k=True, local_dataset=False)
    def missing(name):
        raise ModuleNotFoundError(f"No module named '{name}'")
    monkeypatch.setattr(probe.importlib, 'import_module', missing)
    with pytest.raises(RuntimeError, match='GSM8K requires lm_eval'):
        probe.probe(args)
    args.gsm8k = False
    assert probe.probe(args)['requested_devices'] == ['0']


def test_preflight_requests_gsm8k_only_for_accuracy_run(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'benchmarks/service'))
    from afd_e2e.cli import parser
    from afd_e2e.config import from_args
    from afd_e2e.runner import _preflight_command

    base = ['run', '--model', 'v4-flash', '--model-path', '/prepared/model',
            '--repo', str(tmp_path), '--device', ','.join(map(str, range(16)))]
    accuracy = from_args(parser().parse_args(base + ['--test', 'gsm8k']))
    smoke = from_args(parser().parse_args(base + ['--test', 'smoke']))
    assert '--gsm8k' in _preflight_command(accuracy, accuracy.devices)
    assert '--gsm8k' not in _preflight_command(smoke, smoke.devices)


def test_service_rejects_removed_async_configuration_before_starting():
    result = invoke("plan", "--repo", "/work/target", "--comm", "async", "--moe-ubatching", "on")
    assert result.returncode == 2
    assert "no async_moe_ubatching field" in result.stderr


def test_pull_plan_checks_checkpoint_topk_source_capacity(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "config.json").write_text('{"num_experts_per_tok": 8}')
    common = ("plan", "--repo", str(tmp_path / "target"), "--model", "v4-flash",
              "--model-path", str(weights), "--device", ",".join(map(str, range(16))),
              "--comm", "async", "--ffn-mega-moe", "on", "--test", "smoke")
    bad = invoke(*common, "--max-batched-tokens", "32768")
    assert bad.returncode == 2 and "top-k uint16 source row limit" in bad.stderr
    good = invoke(*common, "--max-batched-tokens", "32764")
    assert good.returncode == 0, good.stderr


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


@pytest.mark.parametrize("model, default", [("v2-lite", 0.27), ("v4-flash", 284 / 300)])
@pytest.mark.parametrize("override", [None, 0.0, 0.8])
def test_gsm8k_threshold_defaults_follow_model(monkeypatch, model, default, override):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks/service"))
    from afd_e2e.cli import parser
    from afd_e2e.config import from_args

    args = ["plan", "--model", model, "--model-path", "/work/weights"]
    if override is not None:
        args += ["--threshold", str(override)]
    case = from_args(parser().parse_args(args))
    assert case.threshold == (default if override is None else override)


@pytest.mark.parametrize("extra", [("--api-port", "65533"), ("--afd-port", "19282")])
def test_service_rejects_rpc_ports_out_of_range_or_overlapping(extra):
    result = invoke("plan", "--repo", "/work/target", "--device", "0,1,2,3", *extra)
    assert result.returncode != 0
    assert "ports must be distinct and in range" in result.stderr


@pytest.mark.parametrize('option', ['--test-interval','--perf-point-interval'])
@pytest.mark.parametrize('value', ['-1','nan','inf'])
def test_service_rejects_invalid_settling_intervals(option, value):
    result=invoke('plan','--repo','/work/target','--device','0,1,2,3',option,value)
    assert result.returncode==2 and 'intervals' in result.stderr


def test_one_startup_runs_accuracy_and_performance_matrix(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'benchmarks/service'))
    from types import SimpleNamespace
    from afd_e2e import runner
    from afd_e2e.cli import parser
    from afd_e2e.config import from_args
    case=from_args(parser().parse_args(['run','--model','v4-flash','--model-path','/prepared/model',
        '--repo',str(tmp_path),'--device',','.join(map(str,range(16))), '--nic','lo',
        '--afd-host','127.0.0.1','--test','all,perf','--limit','30',
        '--perf-phase','prefill','--perf-mode','matrix','--perf-input-lens','128,512,2048','--perf-concurrencies','1,4,8',
        '--test-interval','5','--perf-point-interval','3','--output',str(tmp_path/'result')]))
    seen=[]; checks=[]; clock=[0.0]
    def sleep(seconds): clock[0]+=seconds
    monkeypatch.setattr(runner,'time',SimpleNamespace(time=lambda:clock[0],monotonic=lambda:clock[0],sleep=sleep))
    class Process:
        def __init__(self,name,command,env,directory):
            self.name=name; self.diagnostics={'inference':[],'shutdown':[]}
            self.root_errors={'inference':None,'shutdown':None};self.advisories=[]
            seen.append(name)
            if name=='preflight':(directory/'preflight.log').write_text('AFD_E2E_PROBE {}\n')
        def ensure_alive(self):checks.append(self.name)
        def wait(self,*args): pass
    monkeypatch.setattr(runner,'ManagedProcess',Process)
    monkeypatch.setattr(runner,'source_manifest',lambda case:{})
    monkeypatch.setattr(runner,'wait_ready',lambda *args:None)
    monkeypatch.setattr(runner,'capture_metrics',lambda *args:{})
    monkeypatch.setattr(runner,'_stop_children',lambda *args:[])
    monkeypatch.setattr(runner,'find_processes_matching_environment',lambda *args:[])
    monkeypatch.setattr(runner,'report_text',lambda *args:'fixture-only\n')
    monkeypatch.setattr(runner.evaluation,'read_results',lambda *args:dict(samples=30,strict_match=0.9))
    result=runner.execute(case)
    assert result['status']=='passed' and result['cleanup_status']=='passed'
    assert seen.count('attention')==seen.count('ffn')==1
    assert seen.index('smoke')<seen.index('evaluation')<seen.index('performance')
    manifest=json.loads((tmp_path/'result/manifest.json').read_text())
    assert len(manifest['performance_points'])==9
    assert clock[0]==10 and checks.count('ffn')>5


@pytest.mark.parametrize('fail_first', [False, True])
def test_point_settling_is_outside_measured_duration(tmp_path,monkeypatch,fail_first):
    monkeypatch.syspath_prepend(str(ROOT/'benchmarks/service'))
    import asyncio
    from types import SimpleNamespace
    from afd_e2e import performance
    from afd_e2e.cli import parser
    from afd_e2e.config import from_args
    settings=from_args(parser().parse_args(['plan','--baseline','--topology','dp1','--repo',str(tmp_path),
        '--test','perf','--perf-phase','prefill','--perf-input-lens','128,512','--perf-concurrencies','1',
        '--perf-requests','1','--perf-warmup','0','--perf-mode','matrix','--perf-point-interval','3'])).to_dict()
    delays=[]
    class Client:
        def __init__(self,**kwargs): pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def post(self,*args,**kwargs):
            return SimpleNamespace(json=lambda:{'tokens':[1,2,3]},raise_for_status=lambda:None)
    async def batch(client,endpoints,model,vocabulary,point,count,phase,timeout,records,raw,state):
        if count and fail_first and point['input_lens']==[128]:
            raise ValueError('fixture failed point')
        if count:
            records.append(dict(status='passed',input_tokens=point['input_lens'][0],output_tokens=1,
                                input_target=point['input_lens'][0],output_target=1,sent_offset_s=0,ttft_ms=10,e2e_ms=10,tpot_ms=0))
            state.update(duration_seconds=1,peak_concurrency=1)
    async def sleep(seconds):delays.append(seconds)
    monkeypatch.setitem(sys.modules,'httpx',SimpleNamespace(AsyncClient=Client,Limits=lambda **kwargs:None))
    monkeypatch.setattr(performance,'batch',batch)
    monkeypatch.setattr(performance.asyncio,'sleep',sleep)
    result=asyncio.run(performance.evaluate(settings,tmp_path))
    assert result['status']==('failed' if fail_first else 'passed') and delays==[3]
    assert result['points'][-1]['duration_seconds']==1
    assert len(result['points'])==2
    if fail_first:
        assert result['points'][0]['status']=='failed' and result['errors']
    else:
        assert result['points'][0]['duration_seconds']==1
