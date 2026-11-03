"""The generated serve config must satisfy the plugin that will parse it.

This is the guard that keeps afd_e2e honest across afd-plugin revisions: the
planner writes `additional_config` and connector extra fields by name, and the
plugin validates those names at startup. Point AFD_PLUGIN_ROOT at a checkout to
run it; the connector half additionally needs torch, so it runs on an NPU host.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = os.environ.get('AFD_PLUGIN_ROOT')


def plugin_config():
    if not PLUGIN or not (Path(PLUGIN) / 'afd_plugin/config.py').is_file():
        pytest.skip('set AFD_PLUGIN_ROOT to an afd-plugin checkout')
    sys.path.insert(0, str(PLUGIN))
    import afd_plugin.config as config
    return config


def plan(tmp_path):
    result = subprocess.run(
        [sys.executable, '-m', 'afd_rsi', 'service', 'plan',
         '--model', 'v4-flash', '--model-path', str(tmp_path / 'weights'),
         '--repo', str(tmp_path / 'source'), '--device', ','.join(map(str, range(16))),
         '--comm', 'async', '--mrv', 'mrv2', '--layered-gmm', 'on',
         '--moe-ubatching', 'off', '--execution', 'eager', '--dbo', 'off',
         '--test', 'smoke', '--nic', 'lo', '--afd-host', '127.0.0.1',
         '--ffn-mega-moe', 'on'],
        cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')},
        text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def additional(document, role):
    for service in document['services']:
        if service['role'] == role:
            command = service['command']
            return json.loads(command[command.index('--additional-config') + 1])
    raise AssertionError(f'no {role} service')


def test_generated_afd_config_is_accepted_by_the_current_plugin(tmp_path):
    config = plugin_config()
    document = plan(tmp_path)
    for role in ('attention', 'ffn'):
        parsed = config.afd_config_from_mapping(additional(document, role)['afd'])
        extra = parsed.connector_extra_config
        # Every name below is validated by the plugin itself; a rename there fails
        # here instead of at service startup on the NPU host.
        assert extra['ffn_mega_moe'] is True
        assert extra['dynamicQuant'] == 1
        assert extra['attn_ranks_per_dp'] == 4
        assert 'hccl_buffer_size' in extra


def test_connector_accepts_the_generated_extra_config(tmp_path):
    config = plugin_config()
    pytest.importorskip('torch', reason='connector parsing imports the NPU runtime')
    from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector
    document = plan(tmp_path)
    info = CAMAsyncAFDConnector.parse_extra_info(
        config.connector_extra_config_from_mapping(additional(document, 'ffn')['afd']))
    assert info.ffn_mega_moe is True and info.dynamic_quant == 1
