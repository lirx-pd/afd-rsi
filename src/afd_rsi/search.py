"""Bounded proposal search over frozen campaigns; the model never owns acceptance."""

import fnmatch
import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from . import campaign, skills
from .evidence import digest, read_json, write_json
from .runner import execute
from .workspace import copy_source, inventory


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def fingerprint(plan):
    keys = ('source_snapshot', 'foundation_hash', 'controller_hash', 'backend', 'workload_id',
            'scope', 'unit', 'metric', 'ranks', 'primary_rank', 'repetitions', 'min_gain_pct',
            'commands', 'artifacts', 'allowed_files', 'native', 'service', 'validation')
    return identity({key: plan.get(key) for key in keys})


def verified_verdict(run, name, root):
    attempt = run / 'attempts' / name
    if read_json(attempt / 'progress.json')['status'] != 'completed':
        raise ValueError('best candidate points to an unfinished attempt')
    observations = campaign.audit_attempt(run, attempt, root)
    verdict = read_json(attempt / 'verdict.json')
    if campaign.evaluate_attempt(campaign.load(run, root), attempt, observations) != verdict:
        raise ValueError('candidate verdict inconsistent with evidence')
    return verdict


def best_candidate(run, root):
    best = read_json(run / 'search/best.json')
    source = (run / best['source']).resolve()
    if not source.is_relative_to(run) or inventory(source)['hash'] != best['source_hash']:
        raise ValueError('best candidate source changed')
    if best['confirmation'] is None:
        attempt = run / 'attempts' / best['baseline']
        campaign.audit_attempt(run, attempt, root)
        qualification = read_json(attempt / 'baseline.json')
        progress = read_json(attempt / 'progress.json')
        if not qualification['complete'] or progress['status'] != 'completed':
            raise ValueError('initial best has not passed baseline qualification')
        if digest(attempt / 'baseline.json') != best['baseline_hash']:
            raise ValueError('baseline qualification changed')
    else:
        for stage in ('screen', 'confirmation'):
            record = best[stage]
            verdict = verified_verdict(run, record['attempt'], root)
            attempt = run / 'attempts' / record['attempt']
            sources = read_json(attempt / 'attempt.lock.json')['sources']
            expected_level = ('operator' if stage == 'screen' and campaign.load(run, root).get('validation') == 'operator-first'
                              else 'full')
            if (verdict['verdict'] != 'PASS' or digest(attempt / 'verdict.json') != record['hash']
                    or verdict['validation_level'] != expected_level
                    or sources['candidate']['hash'] != best['source_hash']
                    or sources['baseline']['hash'] != best['parent_hash']):
                raise ValueError('best candidate lacks matching screen and confirmation evidence')
        if best['screen']['attempt'] == best['confirmation']['attempt']:
            raise ValueError('promotion requires a fresh confirmation attempt')
    return best


def load_experiences(run, plan):
    records = []
    for directory in [run / 'search/experience', *map(Path, plan['search']['experience_dirs'])]:
        for path in sorted(directory.glob('*.json')):
            record = read_json(path)
            if record.get('record_hash') != identity({key: value for key, value in record.items() if key != 'record_hash'}):
                raise ValueError(f'experience changed: {path}')
            if record.get('schema_version') != 1:
                raise ValueError(f'unsupported experience schema: {path}')
            if record.get('fingerprint') != fingerprint(plan):
                continue
            evidence_root = directory.parent.parent
            for name, value in record['evidence'].items():
                target = (evidence_root / name).resolve()
                if not target.is_relative_to(evidence_root.resolve()) or digest(target) != value:
                    raise ValueError(f'experience evidence changed: {path}')
            records.append(record)
    # The same archived run may be configured twice; do not count its lessons twice.
    return list({identity(record): record for record in records}.values())


def request_context(run, plan, best, records, number):
    source = run / best['source']
    evidence = [{'id': 'baseline', 'facts': {'source_hash': best['source_hash'],
                  'qualification': best['confirmation'] or best['baseline'], 'backend': plan['backend']}}]
    if best['confirmation'] is None:
        evidence[0]['facts']['observations'] = read_json(run / 'attempts' / best['baseline'] / 'baseline.json')['observations']
    else:
        evidence[0]['facts']['verdict'] = read_json(run / 'attempts' / best['confirmation']['attempt'] / 'verdict.json')
        evidence[0]['facts']['observations'] = read_json(run / 'attempts' / best['confirmation']['attempt'] / 'observations.json')
    # Keep only phase summaries here; full receipts and logs stay in the evidence archive.
    if 'observations' in evidence[0]['facts']:
        evidence[0]['facts']['observations'] = [{key: value for key, value in item.items()
            if key in ('side', 'phase', 'status', 'checks_passed', 'value', 'reason', 'service_metrics')}
            for item in evidence[0]['facts']['observations']]
    incumbent = run / 'attempts' / (best['baseline'] if best['confirmation'] is None else best['confirmation']['attempt'])
    for step in read_json(incumbent / 'progress.json')['steps']:
        item = step.get('result', {})
        if item.get('profiling'):
            evidence.append({'id': 'profile-incumbent', 'facts': {
                'source_hash': best['source_hash'], **item['profiling']}})
    for index, record in enumerate(records):
        evidence.append({'id': f'history-{index}', 'facts': {
            key: record[key] for key in ('category', 'decision', 'reason', 'parent_hash', 'candidate_hash', 'hypothesis', 'experiment')
            if key in record}})
    for index, record in enumerate(records[-2:]):
        for name in record['evidence']:
            if name.endswith('/observation.json') and (run / name).is_file():
                observation = read_json(run / name)
                if observation.get('profiling'):
                    profiling = copy.deepcopy(observation['profiling'])
                    # Keep current detailed timelines; history retains totals and archived raw evidence.
                    for rank in (profiling.get('pipeline') or {}).get('ffn_ranks', []):
                        rank.pop('selected_cores', None)
                    evidence.append({'id': f'profile-history-{index}-{len(evidence)}', 'facts': {
                        'source_hash': record['candidate_hash'], 'decision': record['decision'], **profiling}})
    # Later observations are most useful after a failed compiler/correctness/service gate.
    snippets = []
    for record in records[-4:]:
        for name in record['evidence']:
            if name.endswith(('stderr.log', 'stdout.log', 'verdict.json', 'summary.json')):
                file = run / name
                if file.is_file():
                    snippets.append({'file': name, 'tail': file.read_text(errors='replace')[-4000:]})
    total = plan['search']['max_rounds']
    deep_rounds = 1 if total == 1 else min(total - 1, (2 * total + 2) // 3)
    strategy = {
        'mode': 'deep' if number <= deep_rounds else 'incremental',
        'round': number, 'max_rounds': total, 'deep_rounds': deep_rounds,
        'guidance': ('Prioritize substantial source-backed bottleneck changes, including tile scheduling, '
            'buffer lifetimes or coordinated tiling. A coherent hypothesis may need multiple allowed files. '
            'Prove ownership, synchronization and caller compatibility; do not shrink a justified change to one line.'
            if number <= deep_rounds else
            'Converge with localized changes to the current best. Prefer one bounded mechanism or tuning parameter; '
            'avoid new protocol, ABI or broad pipeline redesigns.'),
    }
    context = {'schema_version': 1, 'round': number, 'strategy': strategy, 'fingerprint': fingerprint(plan),
               'provider_hash': digest(run / 'foundation/benchmarks/agents/codex_agent.py'),
               'deadline': plan['expires_at'], 'agent_timeout_seconds': plan['search']['agent_timeout_seconds'],
               'max_response_bytes': plan['search']['max_context_bytes'] * 2,
               'source_hash': best['source_hash'], 'allowed_files': plan['allowed_files'],
               'execution': {key: plan.get(key) for key in ('ranks', 'primary_rank', 'repetitions', 'native', 'service')},
               'objective': {'metric': plan['metric'], 'scope': plan['scope'], 'unit': plan['unit'],
                             'min_gain_pct': plan['min_gain_pct'], 'backend': plan['backend']},
               'evidence': evidence, 'logs': [], 'source_files': {}, 'knowledge': '',
               'instructions': 'Use only supplied evidence and local knowledge. Return complete unified diffs. '
                   'Validators, gates, framework and original source are immutable. Hypotheses are not measured facts. '
                   'Only a confirmed PASS replaces best. Avoid already tested patches and failed approaches. '
                   'When profile evidence exists, cite it and explain a source-backed bottleneck hypothesis. '
                   'Fused-kernel duration does not prove an internal hardware bound. Profile timings are diagnostic only. '
                   'Use GSM8K and prefill TTFT evidence; do not equate HTTP TTFT with native synthetic kernel time. '
                   'Follow the supplied deep/incremental strategy without changing allowed_files or acceptance gates.'}
    if plan['search'].get('goal'):
        context['objective']['goal'] = plan['search']['goal']
    if plan.get('validation') == 'operator-first':
        context['proposal_requirements'] = {
            'experiment_fields': ['bottleneck', 'mechanism', 'prediction', 'falsification', 'risk'],
            'required_evidence_ids': ['baseline'] + (['profile-incumbent'] if any(
                item['id'] == 'profile-incumbent' for item in evidence) else []),
        }
        context['instructions'] += (' Each proposal needs an experiment: measured bottleneck, source-backed mechanism, '
            'observable prediction, falsification criterion and numerical/synchronization risks. '
            'Cite baseline and the incumbent profile when supplied. Reject speculative parameter guessing. '
            'The controller screens native numerics and paired timing first; only independently repeated winners '
            'reach expensive E2E accuracy/performance and candidate profiling. Stop if evidence is insufficient.')
    limit = plan['search']['max_context_bytes']
    # Match write_json's actual artifact size, including indentation and the newline.
    def size():
        return len((json.dumps(context, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode())

    while size() > limit // 2 and len(context['evidence']) > 1:
        # Keep the incumbent profile ahead of older history when even facts exceed the budget.
        required = context.get('proposal_requirements', {}).get('required_evidence_ids', ['baseline'])
        index = next((i for i, item in enumerate(context['evidence']) if item['id'].startswith('history-')), None)
        if index is None:
            index = next((i for i in reversed(range(len(context['evidence'])))
                          if context['evidence'][i]['id'] not in required), None)
        if index is None:
            break
        del context['evidence'][index]
    files = inventory(source)['files']
    allowed = sorted(name for name in files
                     if any(fnmatch.fnmatchcase(name, pattern) for pattern in plan['allowed_files']))
    queue = list(allowed)
    seen = set(queue)
    context['read_only_source_files'] = []
    context['instructions'] += ' Included dependencies outside allowed_files are read-only; never patch them.'
    for name in queue:
        try:
            content = (source / name).read_text()
        except UnicodeError:
            continue
        context['source_files'][name] = content
        if size() > limit * 3 // 4:
            del context['source_files'][name]
            context.setdefault('omitted_source_files', []).append(name)
            continue
        if name not in allowed:
            context['read_only_source_files'].append(name)
        # Resolve only frozen repository files, never SDK headers or another checkout.
        for include in re.findall(r'^\s*#\s*include\s*"([^"\n]+)"', content, re.MULTILINE):
            local = (source / name).parent.joinpath(include).resolve()
            relative = local.relative_to(source.resolve()).as_posix() if local.is_relative_to(source.resolve()) else None
            matches = [relative] if relative in files else [p for p in files if p == include or p.endswith('/' + include)]
            if len(matches) == 1 and matches[0] not in seen:
                seen.add(matches[0])
                queue.append(matches[0])
    knowledge = skills.context(run / 'foundation', 'ascendc-perf-optimize') + '\n\n' + skills.context(run / 'foundation', 'afd-megamoe')
    # Context is a bounded view; the complete frozen foundation remains in the campaign.
    while knowledge:
        context['knowledge'] = knowledge
        if size() <= limit:
            break
        knowledge = knowledge[:max(0, len(knowledge) - 2000)]
    if not knowledge:
        context['knowledge'] = ''
    if size() > limit:
        raise ValueError('request metadata exceeds the frozen context budget')
    # Logs use only space left after source, measured facts and local knowledge.
    for snippet in reversed(snippets):
        context['logs'].insert(0, snippet)
        if size() > limit:
            del context['logs'][0]
            break
    return context


def propose(request: Path, output: Path, root: Path, session_dir: Path = None):
    """Run only the proposer on an authenticated host; no target checkout is required."""
    skills.validate(root)
    context = read_json(request)
    driver = root / 'benchmarks/agents/codex_agent.py'
    if digest(driver) != context['provider_hash']:
        raise ValueError('proposer differs from the frozen campaign adapter')
    remaining = context['deadline'] - time.time()
    if remaining <= 0:
        raise ValueError('campaign budget exhausted before proposal')
    process = execute([sys.executable, str(driver), '--request', str(request),
                       '--output', str(output / 'proposal.json')], request.parent, output,
                      min(context['agent_timeout_seconds'], remaining),
                      {'AFD_RSI_CODEX_SESSION_DIR': str((session_dir or output / 'codex-session').resolve())})
    if process['status'] != 'completed' or not process['cleanup_complete']:
        raise ValueError('Codex proposal failed; inspect proposer logs and receipt')
    response = output / 'proposal.json'
    if response.stat().st_size > context['max_response_bytes']:
        raise ValueError('Codex response exceeds the frozen request bound')
    validate_proposal(read_json(response), {item['id'] for item in context['evidence']}, context.get('proposal_requirements'))
    return response


def validate_proposal(value, evidence_ids, requirements=None):
    if not isinstance(value, dict) or set(value) != {'stop', 'reason', 'proposals'}:
        raise ValueError('invalid agent response keys')
    if type(value['stop']) is not bool or not isinstance(value['reason'], str) or not value['reason'].strip():
        raise ValueError('agent must provide a stop flag and reason')
    proposals = value['proposals']
    if not isinstance(proposals, list) or len(proposals) > 3 or (not value['stop'] and not proposals) or (value['stop'] and proposals):
        raise ValueError('provide one to three proposals, or stop with no proposals')
    for proposal in proposals:
        keys = {'hypothesis', 'category', 'evidence', 'priority', 'patch'} | ({'experiment'} if requirements else set())
        if not isinstance(proposal, dict) or set(proposal) != keys:
            raise ValueError('invalid proposal keys')
        if not all(isinstance(proposal[key], str) and proposal[key].strip() for key in ('hypothesis', 'category', 'patch')):
            raise ValueError('proposal requires hypothesis, category and patch')
        if type(proposal['priority']) is not int or proposal['priority'] not in (1, 2, 3):
            raise ValueError('proposal priority must be 1..3')
        if (not isinstance(proposal['evidence'], list) or not proposal['evidence']
                or not all(isinstance(item, str) and item in evidence_ids for item in proposal['evidence'])):
            raise ValueError('proposal must cite existing evidence IDs')
        if requirements:
            experiment = proposal['experiment']
            if (not isinstance(experiment, dict) or set(experiment) != set(requirements['experiment_fields'])
                    or not all(isinstance(v, str) and v.strip() for v in experiment.values())):
                raise ValueError('proposal requires a complete falsifiable experiment')
            if not set(requirements['required_evidence_ids']) <= set(proposal['evidence']):
                raise ValueError('proposal must cite the incumbent measurement and profile evidence')
    return value


def select_proposal(proposals, records, parent):
    seen = {record['patch_hash'] for record in records if record['parent_hash'] == parent}
    ranked = []
    for index, proposal in enumerate(proposals):
        patch_hash = identity(proposal['patch'].strip())
        if patch_hash in seen:
            continue
        successes = sum(r['decision'] == 'promoted' for r in records if r['category'] == proposal['category'])
        failures = sum(r['decision'] in ('rejected', 'invalid_patch') for r in records if r['category'] == proposal['category'])
        score = proposal['priority'] + successes - failures
        ranked.append((score, -index, proposal))
    return max(ranked, key=lambda item: item[:2])[2] if ranked else None


def _save_state(run, state):
    write_json(run / 'search/state.json', state)


def _start_attempt(run, root, state, record, stage, round_dir=None):
    if stage not in record:
        record[stage] = uuid.uuid4().hex
        target = run / 'search/state.json' if round_dir is None else round_dir / 'state.json'
        write_json(target, record if round_dir is not None else state)
    attempt = run / 'attempts' / record[stage]
    if not attempt.exists():
        level = ('operator' if stage == 'screen' and campaign.load(run, root).get('validation') == 'operator-first' else 'full')
        campaign._new_attempt(run, root, 'baseline' if stage == 'baseline' else 'comparison',
                              name=record[stage], candidate_source=None if round_dir is None else round_dir / 'source',
                              baseline_source=None if round_dir is None else run / record['parent']['source'], level=level)
    if read_json(attempt / 'progress.json')['status'] != 'completed':
        campaign._continue(run, attempt, root)
    return attempt


def _lesson(run, plan, round_dir, record):
    proposal = record.get('selected', {})
    evidence = {}
    for stage in ('screen', 'confirmation'):
        if stage not in record:
            continue
        attempt = run / 'attempts' / record[stage]
        for name in ('verdict.json', 'observations.json', 'attempt.lock.json', 'attempt.sha256', 'progress.json', 'sources.json'):
            path = attempt / name
            if path.is_file():
                evidence[str(path.relative_to(run))] = digest(path)
        for step in read_json(attempt / 'progress.json')['steps']:
            if 'result' in step:
                trial = run / step['result']['trial']
                for path in trial.rglob('*'):
                    if path.is_file():
                        evidence[str(path.relative_to(run))] = digest(path)
    for name in ('request.json', 'proposal.json', 'change.patch'):
        path = round_dir / name
        if path.is_file():
            evidence[str(path.relative_to(run))] = digest(path)
    lesson = {'schema_version': 1, 'fingerprint': fingerprint(plan),
              'parent_hash': record['parent']['source_hash'], 'candidate_hash': record.get('candidate_hash'),
              'patch_hash': identity(proposal.get('patch', '').strip()), 'category': proposal.get('category', 'agent-protocol'),
              'hypothesis': proposal.get('hypothesis', ''), 'decision': record['decision'],
              'reason': record['reason'], 'evidence': evidence}
    if 'experiment' in proposal:
        lesson['experiment'] = proposal['experiment']
    destination = run / 'search/experience' / f'{round_dir.name}.json'
    write_json(destination, {**lesson, 'record_hash': identity(lesson)})


def search(run: Path, root: Path, *, external=False):
    run = run.resolve()
    with campaign.exclusive(run, recovery=True):
        plan = campaign.load(run, root)
        if 'search' not in plan:
            raise ValueError('recipe has no frozen search settings')
        campaign.resume(run, root, locked=True)
        state_path = run / 'search/state.json'
        if state_path.is_file():
            state = read_json(state_path)
        else:
            state = {'schema_version': 1, 'status': 'running', 'external_agent': external,
                     'calls': 0, 'rounds': [], 'active': None, 'reason': ''}
            _save_state(run, state)
        if state['status'] != 'running':
            (run / 'search/pending.json').unlink(missing_ok=True)
            return search_report(run, root)
        baseline = _start_attempt(run, root, state, state, 'baseline')
        qualification = read_json(baseline / 'baseline.json')
        campaign.audit_attempt(run, baseline, root)
        if not qualification['complete']:
            state.update(status='baseline_failed', reason='Frozen baseline gates did not all pass; no model call was made.')
            _save_state(run, state)
            return search_report(run, root)
        if not (run / 'search/best.json').exists():
            write_json(run / 'search/best.json', {'source': 'workspaces/baseline',
                       'source_hash': plan['source_snapshot']['hash'], 'baseline': baseline.name,
                       'baseline_hash': digest(baseline / 'baseline.json'), 'confirmation': None})
        settings = plan['search']
        while state['status'] == 'running':
            best = best_candidate(run, root)
            records = load_experiences(run, plan)
            if time.time() >= plan['expires_at']:
                state.update(status='budget_exhausted', reason='Frozen wall-time budget exhausted.')
                _save_state(run, state)
                (run / 'search/pending.json').unlink(missing_ok=True)
                break
            if state['active'] is None:
                if time.time() >= plan['expires_at'] or len(state['rounds']) >= settings['max_rounds'] or state['calls'] >= settings['max_calls']:
                    state.update(status='budget_exhausted', reason='Frozen wall time, round or Agent-dispatch budget exhausted.')
                    _save_state(run, state)
                    break
                name = f'{len(state["rounds"]) + 1:04d}'
                state['active'] = name
                state['rounds'].append(name)
                _save_state(run, state)
            round_dir = run / 'search/rounds' / state['active']
            round_dir.mkdir(parents=True, exist_ok=True)
            path = round_dir / 'state.json'
            if not path.exists():
                context = request_context(run, plan, best, records, len(state['rounds']))
                write_json(round_dir / 'request.json', context)
                record = {'phase': 'request', 'parent': best, 'request_hash': digest(round_dir / 'request.json')}
                write_json(path, record)
            record = read_json(path)
            if digest(round_dir / 'request.json') != record['request_hash']:
                raise ValueError('agent request changed')
            context = read_json(round_dir / 'request.json')
            response = round_dir / 'proposal.json'
            if record['phase'] == 'request':
                if time.time() >= plan['expires_at']:
                    state.update(status='budget_exhausted', reason='Wall time exhausted before model call.')
                    _save_state(run, state)
                    break
                # Charge before dispatch; interrupted/failed calls are never refunded on resume.
                if state.get('dispatch', {}).get('round') != state['active']:
                    if state['calls'] >= settings['max_calls']:
                        state.update(status='budget_exhausted', reason='Agent-dispatch budget exhausted.')
                        _save_state(run, state)
                        break
                    state['calls'] += 1
                    state['dispatch'] = {'round': state['active'], 'call': state['calls']}
                    _save_state(run, state)
                record.update(phase='awaiting_agent', call=state['dispatch']['call'])
                write_json(path, record)
            if record['phase'] == 'awaiting_agent':
                if state['external_agent'] and not response.exists():
                    write_json(run / 'search/pending.json', {'request': str((round_dir / 'request.json').relative_to(run)),
                               'response': str(response.relative_to(run)), 'call': record['call'],
                               'request_hash': record['request_hash']})
                    return search_report(run, root)
                if not response.exists() and not (round_dir / 'agent').exists():
                    replacements = {'{python}': sys.executable, '{foundation}': str(run / 'foundation'),
                                    '{request}': str(round_dir / 'request.json'), '{response}': str(response)}
                    argv = settings['agent_command'][:]
                    for marker, replacement in replacements.items():
                        argv = [part.replace(marker, replacement) for part in argv]
                    remaining = plan['expires_at'] - time.time()
                    if remaining > 0:
                        process = execute(argv, round_dir, round_dir / 'agent', min(settings['agent_timeout_seconds'], remaining),
                                          {'AFD_RSI_CODEX_SESSION_DIR': str(run / 'search/codex-session')})
                        record['agent_process'] = process
                        write_json(path, record)
                try:
                    if response.stat().st_size > settings['max_context_bytes'] * 2:
                        raise ValueError('agent response exceeds context bound')
                    value = validate_proposal(read_json(response), {item['id'] for item in context['evidence']}, context.get('proposal_requirements'))
                    if not state['external_agent']:
                        receipt = read_json(round_dir / 'agent/process.json')
                        if receipt['status'] != 'completed' or not receipt['cleanup_complete']:
                            raise ValueError('agent did not complete cleanly')
                    record['response_hash'] = digest(response)
                    if value['stop']:
                        record.update(phase='decided', decision='stopped', reason=value['reason'])
                        state.update(status='agent_stopped', reason=value['reason'])
                    else:
                        selected = select_proposal(value['proposals'], records, record['parent']['source_hash'])
                        if selected is None:
                            record.update(phase='decided', decision='duplicate', reason='Every proposal was already tested against this parent.')
                        else:
                            record.update(phase='proposed', selected=selected)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    record.update(phase='decided', decision='invalid_agent', reason=str(error))
                write_json(path, record)
            if 'response_hash' in record and digest(response) != record['response_hash']:
                raise ValueError('agent response changed')
            if record['phase'] == 'proposed':
                patch = round_dir / 'change.patch'
                patch.write_text(record['selected']['patch'])
                source, staging = round_dir / 'source', round_dir / 'staging'
                if staging.exists():
                    shutil.rmtree(staging)
                parent_source = run / record['parent']['source']
                copy_source(parent_source, staging, inventory(parent_source))
                try:
                    campaign.apply_patch(run, patch, root, source=staging, locked=True)
                    if source.exists():
                        if inventory(source) != inventory(staging):
                            raise ValueError('interrupted patch source differs from the validated patch')
                        shutil.rmtree(staging)
                    else:
                        staging.rename(source)
                    record.update(phase='screening', candidate_hash=inventory(source)['hash'])
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    record.update(phase='decided', decision='invalid_patch', reason=str(error))
                write_json(path, record)
            if record['phase'] in ('screening', 'confirming'):
                if inventory(round_dir / 'source')['hash'] != record['candidate_hash']:
                    raise ValueError('search candidate changed after applying proposal')
                if record['phase'] == 'screening':
                    attempt = _start_attempt(run, root, state, record, 'screen', round_dir)
                    verdict = verified_verdict(run, attempt.name, root)
                    if verdict['verdict'] == 'PASS':
                        record['phase'] = 'confirming'
                    else:
                        record.update(phase='decided', decision='rejected', reason=verdict)
                    write_json(path, record)
                if record['phase'] == 'confirming':
                    attempt = _start_attempt(run, root, state, record, 'confirmation', round_dir)
                    verdict = verified_verdict(run, attempt.name, root)
                    if verdict['verdict'] == 'PASS':
                        promoted = {'source': str((round_dir / 'source').relative_to(run)),
                                    'source_hash': record['candidate_hash'], 'parent_hash': record['parent']['source_hash'],
                                    'screen': {'attempt': record['screen'], 'hash': digest(run / 'attempts' / record['screen'] / 'verdict.json')},
                                    'confirmation': {'attempt': record['confirmation'], 'hash': digest(attempt / 'verdict.json')}}
                        write_json(run / 'search/best.json', promoted)
                        record.update(phase='decided', decision='promoted', reason=verdict)
                    else:
                        record.update(phase='decided', decision='rejected', reason={'confirmation': verdict})
                    write_json(path, record)
            if record['phase'] == 'decided':
                _lesson(run, plan, round_dir, record)
                if record['decision'] == 'stopped':
                    state.update(status='agent_stopped', reason=record['reason'])
                state['active'] = None
                state.pop('dispatch', None)
                _save_state(run, state)
                (run / 'search/pending.json').unlink(missing_ok=True)
                if (run / 'recovery.json').is_file() and read_json(run / 'recovery.json').get('health') in ('required', 'failed'):
                    campaign.recover(run, root)
        (run / 'search/pending.json').unlink(missing_ok=True)
        return search_report(run, root)


def search_report(run, root):
    plan = campaign.load(run, root)
    state = read_json(run / 'search/state.json')
    best = best_candidate(run, root) if (run / 'search/best.json').is_file() else None
    lines = [f'# {plan["name"]}', '', f'Search status: {state["status"]}', '',
             f'Reason: {state["reason"]}', '', f'Agent dispatches charged: {state["calls"]} / {plan["search"]["max_calls"]}', '',
             f'Backend: {plan["backend"]}; scope: {plan["scope"]}; unit: {plan["unit"]}', '',
             f'Require clean service shutdown: {plan.get("service", {}).get("require_clean_shutdown", True)}; raw lifecycle results are retained.', '',
             f'Best source: {best["source_hash"] if best else "none: baseline unqualified"}', '',
             'Promotion requires a native screen and a fresh confirmation with every configured final service gate. '
             'Operator-only PASS is not service qualification.', '',
             '| Round | Decision | Evidence |', '| --- | --- | --- |']
    for name in state['rounds']:
        path = run / 'search/rounds' / name / 'state.json'
        record = read_json(path) if path.is_file() else {}
        lines.append(f'| {name} | {record.get("decision", record.get("phase", "preparing"))} | search/rounds/{name}/ |')
    lines.extend(['', 'Synthetic results qualify controller contracts only.' if plan['backend'] == 'cpu-synthetic'
                  else 'NPU claims are limited to the frozen workload and configured native/service gates.', ''])
    destination = run / 'search-report.md'
    destination.write_text('\n'.join(lines))
    return destination
