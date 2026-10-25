"""Read-only Codex proposer; request/response files also work across machines."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def invoke(executable, workspace, schema, output, prompt, model, effort, prefix):
    command = [executable, 'exec', '--model', model,
               '-c', f'model_reasoning_effort="{effort}"',
               '--sandbox', 'read-only', '--skip-git-repo-check',
               '--ephemeral', '--json', '--output-schema', str(schema),
               '--output-last-message', str(output), '--cd', str(workspace), prompt]
    prefix.with_suffix('.command.json').write_text(json.dumps(command) + '\n')
    with prefix.with_suffix('.jsonl').open('wb') as events, prefix.with_suffix('.stderr.log').open('wb') as errors:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=events, stderr=errors,
                                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    if result.returncode:
        raise RuntimeError(f'Codex exited {result.returncode}; inspect {prefix}.stderr.log')
    value = json.loads(output.read_text())
    usages = []
    for line in prefix.with_suffix('.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event.get('type') == 'turn.completed' and 'usage' in event:
            usages.append({'model': model, 'effort': effort, **event['usage']})
    return value, usages


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
        'Return JSON matching the supplied schema. Analysis omits patches; coding supplies complete git-apply unified diffs.\n'
        'The evidence array must contain only exact IDs from request.json, such as baseline, never explanations.\n'
        'Label a hypothesis as a hypothesis.\n'
        'Choose up to three distinct proposals; stop when there is no justified change.\n')
    schema = workspace / 'proposal.schema.json'
    contract = json.loads(Path(__file__).with_name('proposal.schema.json').read_text())
    contract['properties']['proposals']['items']['properties']['evidence']['items']['enum'] = [
        item['id'] for item in request['evidence']]
    schema.write_text(json.dumps(contract) + '\n')
    analysis_contract = json.loads(json.dumps(contract))
    proposal = analysis_contract['properties']['proposals']['items']
    proposal['required'].remove('patch')
    del proposal['properties']['patch']
    analysis_schema = workspace / 'analysis.schema.json'
    analysis_schema.write_text(json.dumps(analysis_contract) + '\n')
    analysis_path = workspace / 'analysis.json'
    analysis, usages = invoke(executable, workspace, analysis_schema, analysis_path,
        'Read request.json and AGENTS.md. Analyze evidence, profiling and history. '
        'Choose falsifiable optimization hypotheses and controlled experiments. Do not write patches.',
        'gpt-6-astra', 'xhigh', args.output.parent / 'analysis-events')
    (args.output.parent / 'usage.json').write_text(json.dumps({'turns': usages}) + '\n')
    if analysis['stop']:
        args.output.write_text(json.dumps(analysis) + '\n')
        return
    _, code_usages = invoke(executable, workspace, schema, args.output,
        'Read request.json, AGENTS.md and analysis.json. Implement only the justified hypotheses '
        'as minimal complete patches. Preserve their evidence IDs and explain any implementation limits. '
        'Return stop=true if no safe complete patch can be produced.',
        'gpt-6.1-sol', 'xhigh', args.output.parent / 'codex-events')
    (args.output.parent / 'usage.json').write_text(json.dumps({'turns': usages + code_usages}) + '\n')


if __name__ == '__main__':
    main()
