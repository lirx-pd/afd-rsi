"""One campaign owns one finite, resumable Codex conversation."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from afd_rsi.evidence import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
SID = '12345678-1234-4234-8234-123456789abc'


def adapter():
    spec = importlib.util.spec_from_file_location('codex_adapter', ROOT / 'benchmarks/agents/codex_agent.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analysis_coding_and_next_round_resume_same_thread(tmp_path, monkeypatch):
    module = adapter()
    commands = []

    def run(argv, *, stdout, **kwargs):
        commands.append(argv)
        assert '--ephemeral' not in argv
        if len(commands) == 1:
            assert 'resume' not in argv
        else:
            assert argv[argv.index('resume') + 1] == SID
        stdout.write((json.dumps(dict(type='thread.started', thread_id=SID)) + '\n').encode())
        stdout.write((json.dumps(dict(type='turn.completed', usage=dict(input_tokens=1))) + '\n').encode())
        output = Path(argv[argv.index('--output-last-message') + 1])
        output.write_text('{"stop": true}')
        return type('Result', (), {'returncode': 0})()

    monkeypatch.setattr(module.subprocess, 'run', run)
    session = tmp_path / 'session.json'
    write_json(session, dict(campaign={'id': 'first'}, thread_id=None))
    for index, model in enumerate(('gpt-6-astra', 'gpt-6.1-sol', 'gpt-6-astra')):
        module.invoke('codex', tmp_path, tmp_path / 'schema.json', tmp_path / f'out{index}.json',
                      'current evidence only', model, 'xhigh', tmp_path / f'events{index}', session)
    assert read_json(session)['thread_id'] == SID
    assert len(commands) == 3


def test_different_thread_is_rejected(tmp_path, monkeypatch):
    module = adapter()
    session = tmp_path / 'session.json'
    write_json(session, dict(campaign={}, thread_id=SID))

    def run(argv, *, stdout, **kwargs):
        stdout.write(b'{"type":"thread.started","thread_id":"12345678-1234-4234-8234-000000000000"}\n')
        return type('Result', (), {'returncode': 0})()

    monkeypatch.setattr(module.subprocess, 'run', run)
    with pytest.raises(ValueError, match='different or duplicate thread'):
        module.invoke('codex', tmp_path, tmp_path / 'schema', tmp_path / 'out', '', 'gpt-6-astra',
                      'xhigh', tmp_path / 'events', session)
    assert read_json(session)['thread_id'] == SID


@pytest.mark.parametrize('state', [dict(campaign={'fingerprint': 'other'}, thread_id=SID),
                                       dict(campaign={'fingerprint': None, 'deadline': None, 'provider_hash': None}, thread_id=None)])
def test_cross_campaign_or_uncaptured_session_does_not_start_model(tmp_path, monkeypatch, state):
    module = adapter()
    request = tmp_path / 'request.json'
    write_json(request, {})
    write_json(tmp_path / 'session.json', state)
    monkeypatch.setattr(module.shutil, 'which', lambda _: 'codex')
    monkeypatch.setattr(module, 'invoke', lambda *a: pytest.fail('must fail before model call'))
    args = type('Args', (), dict(request=request, output=tmp_path / 'response.json', session_dir=tmp_path))()
    with pytest.raises(ValueError, match='another campaign or was interrupted'):
        module.propose(args)


def test_adapter_refreshes_request_in_shared_workspace_across_dispatches(tmp_path, monkeypatch):
    module = adapter()
    monkeypatch.setattr(module.shutil, 'which', lambda _: 'codex')
    calls = []

    def invoke(executable, workspace, schema, output, prompt, model, effort, prefix, session):
        request = read_json(workspace / 'request.json')
        calls.append((workspace, model, request['round'], read_json(session)['thread_id']))
        write_json(session, {**read_json(session), 'thread_id': SID})
        value = {'stop': True, 'reason': 'fixture', 'proposals': []}
        write_json(output, value)
        return value, []

    monkeypatch.setattr(module, 'invoke', invoke)
    session_dir = tmp_path / 'conversation'
    session_dir.mkdir()
    for number in (1, 2):
        request = tmp_path / f'request-{number}.json'
        write_json(request, {'fingerprint': 'same', 'deadline': 123, 'provider_hash': 'same',
                             'round': number, 'evidence': [{'id': 'baseline'}]})
        args = type('Args', (), dict(request=request, output=tmp_path / str(number) / 'response.json',
                                    session_dir=session_dir))()
        module.propose(args)
    assert calls == [(session_dir / 'workspace', 'gpt-6-astra', 1, None),
                     (session_dir / 'workspace', 'gpt-6-astra', 2, SID)]
