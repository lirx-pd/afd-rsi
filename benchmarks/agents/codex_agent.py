"""Read-only Codex proposer; request/response files also work across machines."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from afd_rsi.evidence import read_json, write_json


def invoke(executable, workspace, schema, output, prompt, model, effort, prefix, session):
    state = read_json(session)
    command = [executable, 'exec', '--model', model,
               '-c', f'model_reasoning_effort="{effort}"',
               '--sandbox', 'read-only', '--skip-git-repo-check',
               '--json', '--output-schema', str(schema),
               '--output-last-message', str(output), '--cd', str(workspace)]
    if state['thread_id']:
        command += ['resume', state['thread_id']]
    command.append(prompt)
    prefix.with_suffix('.command.json').write_text(json.dumps(command) + '\n')
    with prefix.with_suffix('.jsonl').open('wb') as events, prefix.with_suffix('.stderr.log').open('wb') as errors:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=events, stderr=errors,
                                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    usages = []
    threads = []
    for line in prefix.with_suffix('.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event.get('type') == 'thread.started':
            threads.append(str(UUID(event['thread_id'])))
        if event.get('type') == 'turn.completed' and 'usage' in event:
            usages.append({'model': model, 'effort': effort, **event['usage']})
    if threads:
        if len(threads) != 1 or (state['thread_id'] and threads[0] != state['thread_id']):
            raise ValueError('Codex returned a different or duplicate thread; refusing context loss')
        write_json(session, {**state, 'thread_id': threads[0]})
    if result.returncode:
        raise RuntimeError(f'Codex exited {result.returncode}; inspect {prefix}.stderr.log')
    if not threads or len(usages) != 1:
        raise ValueError('Codex did not report one persistent thread and completed turn')
    return json.loads(output.read_text()), usages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--session-dir', type=Path, default=os.environ.get('AFD_RSI_CODEX_SESSION_DIR'))
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.session_dir = (args.session_dir or args.output.parent / 'codex-session').resolve()
    args.session_dir.mkdir(parents=True, exist_ok=True)
    with (args.session_dir / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        propose(args)


def propose(args):
    request = json.loads(args.request.read_text())
    executable = shutil.which('codex')
    if not executable:
        raise ValueError('Codex CLI is not installed; use an authenticated proposer host and file exchange')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    session = args.session_dir / 'session.json'
    campaign = {key: request.get(key) for key in ('fingerprint', 'deadline', 'provider_hash')}
    if session.exists():
        state = read_json(session)
        if state['campaign'] != campaign or not state['thread_id']:
            raise ValueError('Codex session belongs to another campaign or was interrupted before thread capture')
    else:
        write_json(session, {'campaign': campaign, 'thread_id': None})
    workspace = args.session_dir / 'workspace'
    workspace.mkdir(exist_ok=True)
    (workspace / 'analysis.json').unlink(missing_ok=True)
    (workspace / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2) + '\n')
    (workspace / 'AGENTS.md').write_text(
        'Propose a minimal candidate patch using only request.json evidence and repository-local knowledge.\n'
        'Do not change files, gates, validators, budgets, or knowledge. Do not read credentials or external repositories.\n'
        'Return JSON matching the supplied schema. Analysis omits patches; coding supplies complete git-apply unified diffs.\n'
        'The evidence array must contain only exact IDs from request.json, such as baseline, never explanations.\n'
        'This conversation persists across rounds. The current request.json is authoritative; old source and evidence may be stale.\n'
        'Label a hypothesis as a hypothesis.\n'
        'Follow request.strategy: early deep changes may span allowed files; late changes should be localized.\n'
        'Choose up to three distinct proposals; stop when there is no justified change.\n')
    schema = args.output.parent / 'proposal.schema.json'
    contract = json.loads(Path(__file__).with_name('proposal.schema.json').read_text())
    if request.get('proposal_requirements'):
        proposal = contract['properties']['proposals']['items']
        fields = request['proposal_requirements']['experiment_fields']
        proposal['required'].append('experiment')
        proposal['properties']['experiment'] = {
            'type': 'object', 'additionalProperties': False, 'required': fields,
            'properties': {name: {'type': 'string', 'minLength': 1} for name in fields},
        }
    contract['properties']['proposals']['items']['properties']['evidence']['items']['enum'] = [
        item['id'] for item in request['evidence']]
    schema.write_text(json.dumps(contract) + '\n')
    analysis_contract = json.loads(json.dumps(contract))
    proposal = analysis_contract['properties']['proposals']['items']
    proposal['required'].remove('patch')
    del proposal['properties']['patch']
    analysis_schema = args.output.parent / 'analysis.schema.json'
    analysis_schema.write_text(json.dumps(analysis_contract) + '\n')
    analysis_path = args.output.parent / 'analysis.json'
    analysis, usages = invoke(executable, workspace, analysis_schema, analysis_path,
        'Read request.json and AGENTS.md. Analyze evidence, profiling and history. '
        'Choose falsifiable optimization hypotheses and controlled experiments. '
        'When proposal_requirements is present, provide the complete experiment and required evidence IDs; '
        'use measured waits/stages, explain how the source change hides or removes them, and predict an observable '
        'change with a rejection criterion. Stop rather than guess. Do not write patches.',
        'gpt-6-astra', 'xhigh', args.output.parent / 'analysis-events', session)
    (workspace / 'analysis.json').write_text(json.dumps(analysis) + '\n')
    (args.output.parent / 'usage.json').write_text(json.dumps({'turns': usages}) + '\n')
    if analysis['stop']:
        args.output.write_text(json.dumps(analysis) + '\n')
        return
    requirements = request.get('proposal_requirements')
    if requirements:
        for proposal in analysis['proposals']:
            experiment = proposal.get('experiment', {})
            if (set(experiment) != set(requirements['experiment_fields'])
                    or not all(isinstance(v, str) and v.strip() for v in experiment.values())
                    or not set(requirements['required_evidence_ids']) <= set(proposal['evidence'])):
                raise ValueError('analysis lacks a falsifiable experiment or required incumbent evidence; no coding call')
    _, code_usages = invoke(executable, workspace, schema, args.output,
        'Read request.json, AGENTS.md and analysis.json. Implement only the justified hypotheses '
        'as minimal complete patches. Preserve their evidence IDs and explain any implementation limits. '
        'Return stop=true if no safe complete patch can be produced.',
        'gpt-6.1-sol', 'xhigh', args.output.parent / 'codex-events', session)
    (args.output.parent / 'usage.json').write_text(json.dumps({'turns': usages + code_usages}) + '\n')


if __name__ == '__main__':
    main()
