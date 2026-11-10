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
from afd_rsi.proposal import DEBUG_TASK, require_patch_tools, validate_patches
from afd_rsi.model_protocol import ModelContext, MODEL_RULES, DEBUG_EDIT_TASK, code_schema, compact
from afd_rsi.search_policy import direction_schema
from afd_rsi.model_recovery import Journal, ModelExhausted, ContextFull, failure_response, seed_analysis, check_size


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
    write_json(prefix.with_suffix('.usage.json'), {'turns': usages})
    if result.returncode:
        detail = prefix.with_suffix('.stderr.log').read_text(errors='replace').lower()
        if any(marker in detail for marker in ('context window', 'context_length_exceeded', 'maximum context length')):
            raise ContextFull('Codex context budget exceeded; reload the complete canonical request in a fresh context')
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
        try:
            propose(args)
        except ModelExhausted as error:
            response = failure_response(read_json(args.request), error)
            write_json(args.output.parent / 'model-failure.json', response)
            if not read_json(args.request).get('model_recovery'):
                raise
            write_json(args.output, response)


def recover_stage(journal, stage, call, validate, output, prefix, reset_context):
    cached = journal.cached(stage)
    if cached is not None:
        validate(cached)
        return cached, []
    usages = []
    usage_path = output.parent / 'usage.json'
    previous = read_json(usage_path).get('turns', []) if usage_path.exists() else []
    while len(journal.stage(stage)['attempts']) < journal.limit:
        index = journal.reserve(stage)
        attempt_prefix = prefix.with_name(prefix.name + (f'-retry{index}' if index else ''))
        reported = []
        try:
            value, reported = call(journal.stage(stage)['hint'], attempt_prefix)
            write_json(attempt_prefix.with_suffix('.response.json'), value)
            validate(value)
        except (ValueError, KeyError, TypeError, RuntimeError) as error:
            if 'different or duplicate thread' in str(error):
                raise
            if attempt_prefix.with_suffix('.usage.json').exists():
                reported = read_json(attempt_prefix.with_suffix('.usage.json'))['turns']
            if output.exists():
                attempt_prefix.with_suffix('.answer.txt').write_bytes(output.read_bytes())
            journal.rejected(stage, error, truncated=isinstance(error, json.JSONDecodeError) or 'budget' in str(error))
            if isinstance(error, ContextFull):
                reset_context(attempt_prefix)
            if not reported:
                reported = [dict(stage=stage, attempt=index, usage_unknown=True)]
            usages += reported
            write_json(usage_path, {'turns': previous + usages, 'model_calls': len(previous + usages)})
            continue
        usages += reported
        write_json(usage_path, {'turns': previous + usages, 'model_calls': len(previous + usages)})
        journal.complete(stage, value)
        return value, usages
    raise ModelExhausted(stage, journal.stage(stage)['hint'])


def write_workspace(workspace, request):
    """Keep the transport self-contained, but let Codex read individual contracts."""
    view = json.loads(json.dumps(request))
    (workspace / 'knowledge.md').unlink(missing_ok=True)
    for directory in ('source', 'dependencies', 'evidence', 'knowledge-files'):
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
    for name, content in request.get('knowledge_files', {}).items():
        view['knowledge_files'][name] = save('knowledge-files/' + name, content)
    (workspace / 'request.json').write_text(json.dumps(view, ensure_ascii=False, indent=2) + '\n')


def propose(args):
    request = json.loads(args.request.read_text())
    journal = Journal(args.output.parent, request) if request.get('model_recovery') else None
    if journal is not None:
        seed_analysis(journal, request)
    require_patch_tools(request)
    global_redesign = request.get('strategy', {}).get('mode') == 'global-redesign'
    executable = shutil.which('codex')
    if not executable:
        raise ValueError('Codex CLI is not installed; use an authenticated proposer host and file exchange')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    session = args.session_dir / 'session.json'
    campaign = {key: request.get(key) for key in ('fingerprint', 'deadline', 'provider_hash')}
    if session.exists():
        state = read_json(session)
        if state['campaign'] != campaign or (not state['thread_id'] and journal is None):
            raise ValueError('Codex session belongs to another campaign or was interrupted before thread capture')
    else:
        write_json(session, {'campaign': campaign, 'thread_id': None})
    workspace = args.session_dir / 'workspace'
    workspace.mkdir(exist_ok=True)
    (workspace / 'analysis.json').unlink(missing_ok=True)
    write_workspace(workspace, request)
    wire = ModelContext(request) if request.get('model_protocol') == 2 else None
    def call_stage(stage, contract_path, output, prompt, model, prefix, analysis=None):
        if journal is None:
            return invoke(executable, workspace, contract_path, output, prompt, model, 'xhigh', prefix, session)
        def validate(value):
            from afd_rsi.search import validate_proposal
            check_size(value, request)
            if stage == 'analysis':
                normalized = wire.normalize_analysis(value) if wire is not None else value
                shaped = {**normalized, 'proposals': [
                    {**{k: v for k, v in p.items() if k != 'files'}, 'patch': 'analysis only'} for p in normalized['proposals']]}
                validate_proposal(shaped, {e['id'] for e in request['evidence']}, request.get('proposal_requirements'))
                if wire is not None and not value['stop']:
                    wire.coding(normalized, inline=False)
            else:
                decoded = wire.decode(value, analysis) if wire is not None else value
                check_size(decoded, request)
                validate_proposal(decoded, {e['id'] for e in request['evidence']}, request.get('proposal_requirements'))
                validate_patches(decoded, request)
        def reset_context(where):
            old = read_json(session)
            write_json(where.with_suffix('.context-reset.json'), dict(
                previous_session=old, request_hash=journal.request_hash,
                reason='Provider context window exhausted; current request/code manifest remains authoritative'))
            write_json(session, {**old, 'thread_id': None})
        return recover_stage(journal, stage,
            lambda hint, where: invoke(executable, workspace, contract_path, output,
                prompt + '\n' + hint, model, 'xhigh', where, session), validate, output, prefix, reset_context)
    if wire is not None:
        view = json.loads(wire.initial(inline=False))
        view['file_paths'] = wire.paths()
        for item in wire.evidence():
            (workspace / 'evidence' / (item['id'] + '.summary.json')).write_text(compact(item['facts']))
        if request.get('debug'):
            view.update(failure=request['debug']['failure'], logs=request.get('logs', []))
        (workspace / 'request.json').write_text(compact(view))
    (workspace / 'AGENTS.md').write_text(
        'Propose a complete candidate patch using only request.json evidence and repository-local knowledge.\n'
        'Do not change files, gates, validators, budgets, or knowledge. Do not read credentials or external repositories.\n'
        + ('Return JSON matching the supplied schema. Analysis omits patches; coding supplies complete git-apply unified diffs.\n'
         if wire is None else MODEL_RULES + '\nRead source/SDK via request.json file_paths; never dump the whole SDK.\n')
        + 'The evidence array must contain only exact IDs from request.json, such as baseline, never explanations.\n'
        'This conversation persists across rounds. The current request.json is authoritative; old source and evidence may be stale.\n'
        'Label a hypothesis as a hypothesis.\n'
        'request.json is a manifest: read the named source/dependency files and evidence facts_file as needed.\n'
        'Avoid dumping the entire workspace; inspect relevant functions and the supplied overlap/wait summaries.\n'
        'Before choosing a design, verify its completion/Finalize, idle-core drain and activation slicing contracts.\n'
        'If those contracts are missing, return stop=true, proposals=[], needs_context=[exact missing contracts].\n'
        'Follow request.strategy and its evidence-driven scope; use local skills as references and allow justified new mechanisms.\n'
        'Choose up to three distinct proposals; stop when there is no justified change.\n'
        + ('For global review, reassess the whole FFN pipeline after input readiness and earlier failed predictions. '
           'Compare architectures and choose the justified scope; a rewrite is permitted when supported.\n'
           if global_redesign else '')
        + ('The legacy explicit redesign requests a coherent architecture, not a local parameter tweak.\n'
           if global_redesign and 'anchor' not in request.get('strategy', {}) else ''))
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
        if 'knowledge_refs' in request['proposal_requirements']:
            proposal['required'].append('direction')
            proposal['properties']['direction'] = direction_schema()
    contract['properties']['proposals']['items']['properties']['evidence']['items']['enum'] = [
        item['id'] for item in request['evidence']]
    schema.write_text(json.dumps(contract) + '\n')
    if request.get('debug'):
        analysis = wire.debug_analysis() if wire is not None else None
        if wire is not None:
            view = json.loads(wire.coding(analysis, inline=False))
            view.update(failure=request['debug']['failure'], logs=request.get('logs', []))
            (workspace / 'code.json').write_text(compact(view))
            write_json(schema, code_schema(wire.base, 1))
        value, usages = call_stage('code', schema, args.output,
            ('Read request.json and AGENTS.md. ' + DEBUG_TASK if wire is None
             else 'Read code.json and AGENTS.md. Return id/edits. ' + DEBUG_EDIT_TASK),
            'gpt-6.1-sol', args.output.parent / 'codex-events', analysis)
        if journal is None:
            write_json(args.output.parent / 'usage.json', {'turns': usages, 'model_calls': len(usages)})
        if wire is not None:
            value = wire.decode(value, analysis)
            write_json(args.output, value)
        validate_patches(value, request)
        return
    analysis_contract = json.loads(json.dumps(contract))
    proposal = analysis_contract['properties']['proposals']['items']
    proposal['required'].remove('patch')
    del proposal['properties']['patch']
    if wire is not None:
        proposal['required'].append('files')
        proposal['properties']['files'] = dict(type='array', minItems=1, items=dict(type='string'))
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
            'activation, GMM2 and output/TX. Review failed predictions and choose justified changes after source readiness; '
            'exclude Attention submission waits and do not stop merely because earlier local tweaks failed.')
    analysis, usages = call_stage('analysis', analysis_schema, analysis_path,
        analysis_prompt,
        'gpt-6-astra', args.output.parent / 'analysis-events')
    if journal is None:
        write_json(args.output.parent / 'usage.json', {'turns': usages, 'model_calls': len(usages)})
    if wire is not None:
        analysis = wire.normalize_analysis(analysis)
    (workspace / 'analysis.json').write_text(json.dumps(analysis) + '\n')
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
    if wire is not None:
        view = json.loads(wire.coding(analysis, inline=False))
        (workspace / 'code.json').write_text(compact(view))
        write_json(schema, code_schema(wire.base, len(analysis['proposals'])))
        code_prompt = 'Read code.json and AGENTS.md. Implement the analyzed hypotheses. Return id/edits; metadata is inherited.'
    if global_redesign:
        code_prompt += (' Implement the selected mechanisms, including all affected allowed files and '
            'synchronization contracts. Rewriting a kernel or adding a new implementation is permitted; '
            'choose scope from evidence and request.strategy, rather than round number.')
    value, code_usages = call_stage('code', schema, args.output,
        code_prompt,
        'gpt-6.1-sol', args.output.parent / 'codex-events', analysis)
    usages += code_usages
    if journal is None:
        write_json(args.output.parent / 'usage.json', {'turns': usages, 'model_calls': len(usages)})
    try:
        if wire is not None:
            write_json(args.output.parent / 'model-response.json', value)
            value = wire.decode(value, analysis)
            write_json(args.output, value)
        validate_patches(value, request)
    except ValueError as error:
        if journal is not None:
            raise
        write_json(args.output.parent / 'proposal-before-repair.json', value)
        value, repaired_usage = invoke(executable, workspace, schema, args.output,
            'The coding response failed local patch preflight: ' + str(error)
            + ('. Read request.json and return corrected complete patches against its parent source. ' if wire is None
               else '. Read code.json and return corrected id/edits against the original parent. ')
            + 'Keep the hypotheses and validation gates; this is the only patch repair call.',
            'gpt-6.1-sol', 'xhigh', args.output.parent / 'codex-repair-events', session)
        usages += repaired_usage
        write_json(args.output.parent / 'usage.json', {'turns': usages, 'model_calls': len(usages)})
        if wire is not None:
            write_json(args.output.parent / 'model-response-repair.json', value)
            value = wire.decode(value, analysis)
            write_json(args.output, value)
        validate_patches(value, request)


if __name__ == '__main__':
    main()
