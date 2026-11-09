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


def write_workspace(workspace, request):
    """Keep the transport self-contained, but let Codex read individual contracts."""
    view = json.loads(json.dumps(request))
    (workspace / 'knowledge.md').unlink(missing_ok=True)
    for directory in ('source', 'dependencies', 'evidence'):
        target = workspace / directory
        if target.exists():
            shutil.rmtree(target)
    def save(name, content):
        target = (workspace / name).resolve()
        if not target.is_relative_to(workspace.resolve()):
            raise ValueError('context path escapes proposer workspace')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return name
    readonly = set(request.get('read_only_source_files', []))
    for name, content in request.get('source_files', {}).items():
        view['source_files'][name] = dict(path=save('source/' + name, content), read_only=name in readonly)
    for name, item in request.get('dependency_context', {}).get('files', {}).items():
        entry = view['dependency_context']['files'][name]
        entry['path'] = save('dependencies/' + name, entry.pop('content'))
    for item in view.get('evidence', []):
        item['facts_file'] = save('evidence/' + item['id'] + '.json', json.dumps(item.pop('facts', {}), indent=2) + '\n')
    if view.get('knowledge'):
        view['knowledge'] = save('knowledge.md', view['knowledge'])
    (workspace / 'request.json').write_text(json.dumps(view, ensure_ascii=False, indent=2) + '\n')


def propose(args):
    request = json.loads(args.request.read_text())
    global_redesign = request.get('strategy', {}).get('mode') == 'global-redesign'
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
    write_workspace(workspace, request)
    (workspace / 'AGENTS.md').write_text(
        'Propose a complete candidate patch using only request.json evidence and repository-local knowledge.\n'
        'Do not change files, gates, validators, budgets, or knowledge. Do not read credentials or external repositories.\n'
        'Return JSON matching the supplied schema. Analysis omits patches; coding supplies complete git-apply unified diffs.\n'
        'The evidence array must contain only exact IDs from request.json, such as baseline, never explanations.\n'
        'This conversation persists across rounds. The current request.json is authoritative; old source and evidence may be stale.\n'
        'Label a hypothesis as a hypothesis.\n'
        'request.json is a manifest: read the named source/dependency files and evidence facts_file as needed.\n'
        'Avoid dumping the entire workspace; inspect relevant functions and the supplied overlap/wait summaries.\n'
        'Before choosing a design, verify its completion/Finalize, idle-core drain and activation slicing contracts.\n'
        'If those contracts are missing, return stop=true, proposals=[], needs_context=[exact missing contracts].\n'
        'Follow request.strategy: a global redesign may replace or add allowed files; late changes should be localized.\n'
        'Choose up to three distinct proposals; stop when there is no justified change.\n'
        + ('For this global-redesign round, analyze the whole FFN pipeline after input readiness and choose one coherent architecture, '
           'not a local parameter tweak. Large source changes are allowed when the evidence supports them.\n'
           if global_redesign else ''))
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
    analysis_prompt = (
        'Read request.json and AGENTS.md. Analyze evidence, profiling and history. '
        'Choose falsifiable optimization hypotheses and controlled experiments. '
        'When proposal_requirements is present, provide the complete experiment and required evidence IDs; '
        'use measured waits/stages, explain how the source change hides or removes them, and predict an observable '
        'change with a rejection criterion. Verify the implementation contracts before selecting a design; '
        'return needs_context before coding if any required contract is unavailable. '
        'Account for measured stage overlap and remaining unhidden work; '
        'do not add concurrent stage durations. Stop rather than guess. Do not write patches.')
    if global_redesign:
        analysis_prompt += (' Compare at least two FFN architectures across RX/pull, packing, GMM1, '
            'activation, GMM2 and output/TX. Choose one cohesive redesign using FFN work after source readiness; '
            'exclude Attention submission waits and do not stop merely because earlier local tweaks failed.')
    analysis, usages = invoke(executable, workspace, analysis_schema, analysis_path,
        analysis_prompt,
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
    code_prompt = ('Read request.json, AGENTS.md and analysis.json. Implement only the justified hypotheses '
        'as complete patches. Preserve their evidence IDs and explain any implementation limits. '
        'Return stop=true if no safe complete patch can be produced.')
    if global_redesign:
        code_prompt += (' Implement the selected coherent redesign, including all affected allowed files and '
            'synchronization contracts. Rewriting a kernel or adding a new implementation is permitted; '
            'do not shrink the architecture to a local tweak merely to minimize the diff.')
    _, code_usages = invoke(executable, workspace, schema, args.output,
        code_prompt,
        'gpt-6.1-sol', 'xhigh', args.output.parent / 'codex-events', session)
    (args.output.parent / 'usage.json').write_text(json.dumps({'turns': usages + code_usages}) + '\n')


if __name__ == '__main__':
    main()
