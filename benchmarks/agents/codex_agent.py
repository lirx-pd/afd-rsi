"""Read-only Codex proposer; request/response files also work across machines."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    executable = shutil.which('codex')
    if not executable:
        raise ValueError('Codex CLI is not installed; use an authenticated proposer host and file exchange')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    workspace = args.output.parent / 'codex-workspace'
    workspace.mkdir(exist_ok=False)
    (workspace / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2) + '\n')
    (workspace / 'AGENTS.md').write_text(
        'Propose a minimal candidate patch using only request.json evidence and repository-local knowledge.\n'
        'Do not change files, gates, validators, budgets, or knowledge. Do not read credentials or external repositories.\n'
        'Return JSON matching the supplied schema. A patch must be a complete git-apply unified diff.\n'
        'The evidence array must contain only exact IDs from request.json, such as baseline, never explanations.\n'
        'Label a hypothesis as a hypothesis.\n'
        'Choose up to three distinct proposals; stop when there is no justified change.\n')
    schema = workspace / 'proposal.schema.json'
    contract = json.loads(Path(__file__).with_name('proposal.schema.json').read_text())
    contract['properties']['proposals']['items']['properties']['evidence']['items']['enum'] = [
        item['id'] for item in request['evidence']]
    schema.write_text(json.dumps(contract) + '\n')
    command = [executable, 'exec', '--sandbox', 'read-only', '--skip-git-repo-check',
               '--ephemeral', '--json', '--output-schema', str(schema),
               '--output-last-message', str(args.output), '--cd', str(workspace),
               'Read request.json and AGENTS.md. Propose the next controlled optimization experiment.']
    with (args.output.parent / 'codex-events.jsonl').open('wb') as events, (args.output.parent / 'codex-stderr.log').open('wb') as errors:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=events, stderr=errors,
                                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    if result.returncode:
        raise RuntimeError(f'Codex exited {result.returncode}; inspect codex-stderr.log')
    json.loads(args.output.read_text())
    usages = []
    for line in (args.output.parent / 'codex-events.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event.get('type') == 'turn.completed' and 'usage' in event:
            usages.append(event['usage'])
    (args.output.parent / 'usage.json').write_text(json.dumps({'turns': usages}) + '\n')


if __name__ == '__main__':
    main()
