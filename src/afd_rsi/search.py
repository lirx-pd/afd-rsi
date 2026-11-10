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

from . import campaign, skills, search_policy as adaptive
from .contracts import agent_driver
from .evidence import digest, read_json, write_json
from .router import Ledger, Router, normalized_score
from .runner import execute
from .workspace import copy_source, inventory

# Pre-router campaigns and their in-flight requests carry no provider field; the
# digest they recorded was always this adapter, so keep it as the fallback.
DEFAULT_PROVIDER = 'benchmarks/agents/codex_agent.py'


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def fingerprint(plan):
    keys = ('source_snapshot', 'foundation_hash', 'controller_hash', 'backend', 'workload_id',
            'scope', 'unit', 'metric', 'ranks', 'primary_rank', 'repetitions', 'min_gain_pct',
            'commands', 'artifacts', 'allowed_files', 'native', 'service', 'validation')
    payload = {key: plan.get(key) for key in keys}
    # Router settings decide how directions are ranked, so declaring different
    # ones must invalidate cross-campaign experience reuse. Campaigns frozen
    # before routing existed keep their original fingerprint byte for byte, so
    # their recorded experience stays readable.
    routing = {key: plan['search'][key]
               for key in ('router', 'features', 'info_floor', 'margin', 'explore_max_fraction', 'policy', 'max_model_calls')
               if key in plan['search']}
    if routing:
        payload['routing'] = routing
    # A declared provider is part of the method: directions ranked from one
    # model's history must not be silently reused for another. Recipes that do
    # not declare it keep their historical digest, so old experience still loads.
    if 'provider' in plan['search']:
        payload['provider'] = plan['search']['provider']
    if 'models' in plan['search']:
        payload['models'] = plan['search']['models']
    return identity(payload)


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


def profile_context(profiling):
    """Bound display timelines; measured totals and overlap still use every raw event."""
    view = copy.deepcopy(profiling)
    for rank in (view.get('pipeline') or {}).get('ffn_ranks', []):
        for core in rank.get('selected_cores', []):
            events = core.get('events', [])
            if len(events) <= 128:
                continue
            stages = {}
            for index, event in enumerate(events):
                stages.setdefault(event[2], []).append(index)
            chosen = {index for indices in stages.values() for index in (indices[0], indices[-1])}
            if len(chosen) > 128:
                raise ValueError('too many stage types for the request timeline sample')
            remaining = [index for index in range(len(events)) if index not in chosen]
            slots = 128 - len(chosen)
            chosen.update(remaining[i * len(remaining) // slots] for i in range(slots))
            core.update(events=[events[i] for i in sorted(chosen)],
                        events_total=len(events), events_omitted=len(events) - len(chosen),
                        sampling='first/last occurrence of every stage, then evenly spaced events; full trace archived')
    return view


def request_context(run, plan, best, records, number, router=None, *, strategy=None, parent=None):
    source_parent = parent or best
    source = run / source_parent['source']
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
    build_attempt = run / 'attempts' / parent['attempt'] if parent else incumbent
    dependencies = None
    for step in read_json(build_attempt / 'progress.json')['steps']:
        item = step.get('result', {})
        if item.get('dependency_context_hash') and item.get('side') == ('candidate' if parent or best['confirmation'] else 'baseline'):
            path = run / item['trial'] / 'dependencies.json'
            if digest(path) != item['dependency_context_hash']:
                raise ValueError('compiled dependency context changed')
            dependencies = read_json(path)
            from .build_context import verify_context
            verify_context(dependencies, source_parent['source_hash'], dependencies['artifacts'])
            manifest_hash = hashlib.sha256((json.dumps(dependencies['artifacts'], indent=2,
                ensure_ascii=False, allow_nan=False) + '\n').encode()).hexdigest()
            if manifest_hash != item['artifacts'].get('native/artifacts.json'):
                raise ValueError('dependency context artifact identity differs from the completed build')
    for step in read_json(incumbent / 'progress.json')['steps']:
        item = step.get('result', {})
        if item.get('profiling'):
            profile = profile_context(item['profiling'])
            profile['pipeline_trace_files'] = [
                {'file': item['trial'] + '/' + name, 'sha256': sha}
                for name, sha in item.get('profile_files', {}).items()
                if '/pipeline/rank-' in name and name.endswith('.json')]
            evidence.append({'id': 'profile-incumbent', 'facts': {
                'source_hash': best['source_hash'], **profile}})
    for index, record in enumerate(records):
        evidence.append({'id': f'history-{index}', 'facts': {
            key: record[key] for key in ('category', 'decision', 'reason', 'parent_hash', 'candidate_hash', 'hypothesis', 'experiment', 'debug', 'direction', 'comparison_hash')
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
    global_redesign = number == plan['search'].get('global_redesign_round')
    strategy = strategy or {
        'mode': 'global-redesign' if global_redesign else 'deep' if number <= deep_rounds else 'incremental',
        'round': number, 'max_rounds': total, 'deep_rounds': deep_rounds,
        'guidance': ('Analyze FFN execution after Attention source readiness: RX/pull, packing and '
            'quantization, GMM1, activation, GMM2, output/TX, synchronization and overlap. Exclude Attention '
            'submission waits from FFN hotspot rankings. Compare architectural alternatives and propose one cohesive redesign '
            'across allowed files; replacing a kernel or adding a new implementation is permitted. Do not '
            'default to a local tuning change. Preserve protocol, graph, quantization and numerical contracts; '
            'state exact missing evidence if no safe redesign can be justified.'
            if global_redesign else
            'Prioritize substantial source-backed bottleneck changes, including tile scheduling, '
            'buffer lifetimes or coordinated tiling. A coherent hypothesis may need multiple allowed files. '
            'Prove ownership, synchronization and caller compatibility; do not shrink a justified change to one line.'
            if number <= deep_rounds else
            'Converge with localized changes to the current best. Prefer one bounded mechanism or tuning parameter; '
            'avoid new protocol, ABI or broad pipeline redesigns.'),
    }
    provider = agent_driver(plan['search']['agent_command'])
    primary = plan.get('service', {}).get('primary')
    components = primary if isinstance(primary, list) else [primary] if primary else []
    objective = {'metric': plan['metric'], 'scope': plan['scope'], 'unit': plan['unit'],
                 'min_gain_pct': plan['min_gain_pct'], 'backend': plan['backend']}
    if components:
        objective.update(source='real model service' if plan['backend'] == 'ascend' else 'synthetic service fixture',
                         metric=components[0]['metric'] if len(components) == 1 else 'mean_gain_pct',
                         scope=components[0]['id'] if len(components) == 1 else 'service-scenario-mean',
                         unit=('tokens/s' if components[0]['metric'].endswith('_per_s') else 'ms')
                         if len(components) == 1 else 'percent gain')
        if len(components) > 1:
            objective['components'] = components
    context = {'schema_version': 1, 'model_protocol': 2, 'round': number, 'strategy': strategy, 'fingerprint': fingerprint(plan),
               'provider': provider,
               'provider_hash': digest(run / 'foundation' / provider),
               'deadline': plan['expires_at'], 'agent_timeout_seconds': plan['search']['agent_timeout_seconds'],
               'max_context_bytes': plan['search']['max_context_bytes'],
               'max_response_bytes': plan['search']['max_context_bytes'] * 2,
               'source_hash': source_parent['source_hash'], 'allowed_files': plan['allowed_files'],
               'execution': {key: plan.get(key) for key in ('ranks', 'primary_rank', 'repetitions', 'native', 'service')},
               'objective': objective,
               'evidence': evidence, 'logs': [], 'source_files': {}, 'knowledge': '',
               'instructions': 'Use only supplied evidence and local knowledge. Return complete unified diffs. '
                   'Validators, gates, framework and original source are immutable. Hypotheses are not measured facts. '
                   'Only a confirmed PASS replaces best. Avoid already tested patches and failed approaches. '
                   'When profile evidence exists, cite it and explain a source-backed bottleneck hypothesis. '
                   'For FFN profiling use active shares and FFN execution spans after excluding Attention readiness '
                   'and sync_in; do not optimize or rank the deliberate producer wait as FFN work. '
                   'Inspect sampled-core overlap bounds and internal dependency waits for Pull/RX, GMM1, activation, GMM2, '
                   'output and TX before ranking hotspots; a stage can be hidden by another. '
                   'The bounds assume stable core clocks; TX call spans do not measure asynchronous link occupancy. '
                   'Sampled call-span overlap is not hardware utilization or a complete critical path. '
                   'If the source-wait breakdown is unavailable, do not infer FFN hotspots from raw kernel duration. '
                   'Fused-kernel duration does not prove an internal hardware bound. Profile timings are diagnostic only. '
                   'Use GSM8K and prefill TTFT evidence; do not equate HTTP TTFT with native synthetic kernel time. '
                   'Follow the supplied strategy without changing allowed_files or acceptance gates.'}
    if dependencies:
        context['dependency_context'] = dependencies
    if adaptive.enabled(plan):
        context['direction_catalog'] = read_json(run / 'foundation/knowledge/performance/search-directions.json')
        context['instructions'] += (' Use local skills and the direction catalog; these are starting points, not a whitelist. '
            'Each proposal.direction names id, mechanism, step (1 local/2 coordinated/3 architectural), refs '
            '(exact supplied knowledge paths), use (applicability, extension or a new mechanism), and next '
            '(a concrete continuation after a valid slower intermediate, otherwise empty). '
            'Use distinct mechanisms. Existing pool candidates are reusable; explain any superseded assumptions. '
            'Global review compares alternatives and failed predictions; it need not produce a large rewrite.')
        if parent:
            context['exploration_parent'] = {'source_hash': parent['source_hash'], 'baseline_hash': best['source_hash'],
                'seed': parent['seed'], 'instruction': 'Code against source_files; all gains are measured against the qualified best, not this slower parent.'}
    from .build_context import PULL_KERNEL
    if (source / PULL_KERNEL).is_dir() and dependencies is None:
        context['missing_context'] = ['exact compiled MC2/Catlass headers and generated activation; run a new baseline with dependency capture']
    context['instructions'] += (' Before selecting a redesign, read its GMM completion/Finalize and activation '
        'contracts from dependency_context. Report stop=true with needs_context naming missing contracts '
        'before coding; missing dependencies are not evidence that optimization is exhausted. '
        'A cohort redesign must be tested on native cases with multiple cohorts and boundary tails.')
    if plan['search'].get('models'):
        # Frozen per-stage models; an adapter that owns its policy ignores this.
        context['models'] = plan['search']['models']
    if plan['search'].get('goal'):
        context['objective']['goal'] = plan['search']['goal']
    if router is not None and router.enabled:
        # Exploration is a request, never a forced choice: the controller cannot
        # author patches, so an untested family is named for the model to weigh.
        directive = router.directive()
        context['routing'] = directive or {'preferred_category': None,
            'reason': 'every known change family already has evidence under this foundation'}
        if directive:
            context['instructions'] += (f" routing.preferred_category = {directive['preferred_category']!r} has no "
                'evidence under this foundation; weigh it against your default, and do not follow it if the '
                'evidence points elsewhere. Name the change family in each proposal category.')
    if plan.get('validation') == 'operator-first' or adaptive.enabled(plan):
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
    if adaptive.enabled(plan):
        context['proposal_requirements']['knowledge_refs'] = sorted(set(context['direction_catalog']['skills'] + [
            ref for direction in context['direction_catalog']['directions'] for ref in direction['refs']]))
        context['knowledge_files'] = {ref: (run / 'foundation' / ref).read_text()
                                      for ref in context['proposal_requirements']['knowledge_refs']}
    context['protected_evidence_ids'] = list(dict.fromkeys(
        ['baseline'] + [item['id'] for item in evidence if item['id'] == 'profile-incumbent']
        + [item['id'] for item in evidence if item['id'].startswith('history-')][-2:]))
    limit = plan['search']['max_context_bytes']
    # Match write_json's actual artifact size, including indentation and the newline.
    def size():
        return len((json.dumps(context, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode())

    while size() > limit // 2 and len(context['evidence']) > 1:
        # Keep the incumbent profile ahead of older history when even facts exceed the budget.
        required = context['protected_evidence_ids']
        index = next((i for i, item in enumerate(context['evidence'])
                      if item['id'].startswith('history-') and item['id'] not in required), None)
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
        if size() > limit and name not in allowed:
            del context['source_files'][name]
            context.setdefault('omitted_source_files', []).append(name)
            continue
        if name not in allowed:
            context['read_only_source_files'].append(name)
        # Repository closure is separate from the hashed contracts captured by the actual build.
        for include in re.findall(r'^\s*#\s*include\s*"([^"\n]+)"', content, re.MULTILINE):
            local = (source / name).parent.joinpath(include).resolve()
            relative = local.relative_to(source.resolve()).as_posix() if local.is_relative_to(source.resolve()) else None
            matches = [relative] if relative in files else [p for p in files if p == include or p.endswith('/' + include)]
            if len(matches) == 1 and matches[0] not in seen:
                seen.add(matches[0])
                queue.append(matches[0])
    knowledge = skills.context(run / 'foundation', 'ascendc-perf-optimize') + '\n\n' + skills.context(run / 'foundation', 'afd-megamoe')
    if adaptive.enabled(plan):
        knowledge += '\n\n' + skills.context(run / 'foundation', 'ascendc-performance-best-practices')
    # Context is a bounded view; the complete frozen foundation remains in the campaign.
    while knowledge:
        context['knowledge'] = knowledge
        if size() <= limit:
            break
        knowledge = knowledge[:max(0, len(knowledge) - 2000)]
    if not knowledge:
        context['knowledge'] = ''
    if size() > limit:
        raise ValueError(f'request metadata exceeds the frozen context budget: {size()} > {limit} bytes')
    # Logs use only space left after source, measured facts and local knowledge.
    for snippet in reversed(snippets):
        context['logs'].insert(0, snippet)
        if size() > limit:
            del context['logs'][0]
            break
    return context


def session_env(directory: Path) -> dict:
    """Both adapters read their own session variable; the neutral one is shared.

    The directory keeps its historical ``codex-session`` name so a campaign that
    is mid-flight when the provider is introduced resumes the same conversation;
    it now holds whichever provider the frozen recipe selected.
    """
    directory = directory.resolve()
    return {'AFD_RSI_AGENT_SESSION_DIR': str(directory),
            'AFD_RSI_CODEX_SESSION_DIR': str(directory)}


def propose(request: Path, output: Path, root: Path, session_dir: Path = None,
            provider_override: dict = None):
    """Run only the proposer on an authenticated host; no target checkout is required."""
    skills.validate(root)
    context = read_json(request)
    # The request names the adapter it was frozen against; proposals from any
    # other adapter would come from a different model and must not be accepted.
    # A restart may replace a defective adapter, but only through a recorded
    # override naming the digest it replaces; the swap is written as evidence.
    provider = context.get('provider', DEFAULT_PROVIDER)
    driver = root / provider
    if not driver.is_file():
        raise ValueError('proposer adapter is missing from the controller')
    changed = digest(driver) != context['provider_hash']
    if changed:
        replaces = (provider_override or {}).get('replaces')
        if replaces != context['provider_hash'] or (provider_override or {}).get('digest') != digest(driver):
            raise ValueError('proposer differs from the frozen campaign adapter')
        write_json(output / 'provider-change.json', {
            'provider': provider, 'replaces': replaces, 'digest': digest(driver),
            'reason': (provider_override or {}).get('reason', ''),
            'campaign_fingerprint': context.get('fingerprint')})
    remaining = context['deadline'] - time.time()
    if remaining <= 0:
        raise ValueError('campaign budget exhausted before proposal')
    process = execute([sys.executable, str(driver), '--request', str(request),
                       '--output', str(output / 'proposal.json')], request.parent, output,
                      min(context['agent_timeout_seconds'], remaining),
                      session_env(session_dir or output / 'codex-session'))
    if process['status'] != 'completed' or not process['cleanup_complete']:
        raise ValueError(f'{Path(provider).stem} proposal failed; inspect proposer logs and receipt')
    response = output / 'proposal.json'
    if response.stat().st_size > context['max_response_bytes']:
        raise ValueError('proposer response exceeds the frozen request bound')
    validate_proposal(read_json(response), {item['id'] for item in context['evidence']}, context.get('proposal_requirements'))
    return response


def validate_proposal(value, evidence_ids, requirements=None):
    if not isinstance(value, dict) or set(value) - {'needs_context'} != {'stop', 'reason', 'proposals'}:
        raise ValueError('invalid agent response keys')
    if type(value['stop']) is not bool or not isinstance(value['reason'], str) or not value['reason'].strip():
        raise ValueError('agent must provide a stop flag and reason')
    proposals = value['proposals']
    missing = value.get('needs_context', [])
    if (not isinstance(missing, list) or not all(isinstance(p, str) and p.strip() for p in missing)
            or (missing and not value['stop'])):
        raise ValueError('needs_context requires a stop response and named missing contracts')
    if not isinstance(proposals, list) or len(proposals) > 3 or (not value['stop'] and not proposals) or (value['stop'] and proposals):
        raise ValueError('provide one to three proposals, or stop with no proposals')
    for proposal in proposals:
        keys = {'hypothesis', 'category', 'evidence', 'priority', 'patch'} | ({'experiment'} if requirements else set())
        if requirements and 'knowledge_refs' in requirements:
            keys.add('direction')
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
            if 'knowledge_refs' in requirements:
                adaptive.validate_direction(proposal['direction'], requirements)
            experiment = proposal['experiment']
            if (not isinstance(experiment, dict) or set(experiment) != set(requirements['experiment_fields'])
                    or not all(isinstance(v, str) and v.strip() for v in experiment.values())):
                raise ValueError('proposal requires a complete falsifiable experiment')
            if not set(requirements['required_evidence_ids']) <= set(proposal['evidence']):
                raise ValueError('proposal must cite the incumbent measurement and profile evidence')
    return value


def route_proposals(proposals, records, parent, router=None):
    """Return ``(selected proposal or None, routing decision or None)``.

    Without a router this is the historical rule, unchanged: deduplicate against
    patches already tried from this parent, then rank by
    ``priority + category promotions - category failures``. With a router the
    categories themselves are ordered by estimated ability and priority only
    breaks ties inside a category, so a family is no longer punished for
    difficulty or rewarded for being tried first.
    """
    seen = {record['patch_hash'] for record in records if record['parent_hash'] == parent}
    seen.update(record['debug']['patch_hash'] for record in records if record['parent_hash'] == parent
                and record.get('debug', {}).get('patch_hash'))
    fresh = {index: proposal for index, proposal in enumerate(proposals)
             if identity(proposal['patch'].strip()) not in seen}
    if not fresh:
        return None, None
    if router is None or not router.enabled:
        ranked = []
        for index, proposal in fresh.items():
            successes = sum(r['decision'] == 'promoted' for r in records if r['category'] == proposal['category'])
            failures = sum(r['decision'] in ('rejected', 'invalid_patch') for r in records if r['category'] == proposal['category'])
            ranked.append((proposal['priority'] + successes - failures, -index, proposal))
        return max(ranked, key=lambda item: item[:2])[2], None
    items = [{'category': proposal['category'], 'index': index,
              'priority': proposal['priority'], 'features': router.features(proposal['patch'])}
             for index, proposal in fresh.items()]
    decision = router.rank(items)
    return fresh[decision['index']], decision


def select_proposal(proposals, records, parent, router=None):
    return route_proposals(proposals, records, parent, router)[0]


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
                              name=record[stage], candidate_source=None if round_dir is None else run / record.get(
                                  'source', str((round_dir / 'source').relative_to(run))),
                              baseline_source=None if round_dir is None else run / record.get('comparison', record['parent'])['source'], level=level)
    if read_json(attempt / 'progress.json')['status'] != 'completed':
        campaign._continue(run, attempt, root)
    return attempt


def _debug_failure(run, record):
    """Only candidate execution/numerical failures earn a repair, never a slow result."""
    for stage in ('confirmation', 'screen'):
        if stage not in record:
            continue
        attempt = run / 'attempts' / record[stage]
        for step in read_json(attempt / 'progress.json')['steps']:
            result = step.get('result', {})
            failed = result.get('status') == 'invalid' or result.get('checks_passed') is False
            if not failed:
                continue
            process = result.get('process', {})
            # Service execution can fail before run_service returns its observation.
            receipt = run / result.get('trial', '') / 'execution/process.json'
            if not process and receipt.is_file():
                process = read_json(receipt)
            if step['side'] != 'candidate' or not process.get('cleanup_complete'):
                return None
            crashed = result.get('status') == 'invalid' and process.get('status') in ('failed', 'timeout')
            trial = run / result.get('trial', '')
            native = [read_json(trial / name / 'process.json') for name in ('transaction', 'transaction-dense')
                      if (trial / name / 'process.json').is_file()]
            if native:
                # Completed kernels followed by noise/identity validation errors need no code repair.
                crashed = crashed and any(item['status'] in ('failed', 'timeout') for item in native)
            service_crash = (step['phase'] == 'service' and process.get('status') in ('failed', 'timeout')
                             and result.get('summary', {}).get('inference_status') == 'failed')
            numerical = step['phase'] in ('correctness', 'benchmark') and result.get('checks_passed') is False
            return result if crashed or service_crash or numerical else None
    return None


def _prepare_debug(run, plan, round_dir, record, failure):
    """Persist the failed attempt and a code-only request before reserving its call."""
    from .proposal import DEBUG_TASK
    context = read_json(round_dir / 'request.json')
    original = copy.deepcopy(record)
    context['debug'] = {'original_proposal': record['selected'], 'failure': failure,
                        'patch_base': 'original parent source_files; return a complete replacement patch'}
    context['instructions'] = DEBUG_TASK
    context['strategy'] = {'mode': 'debug', 'guidance': 'One repair of this implementation, not a new optimization.'}
    context['knowledge'] = ''
    context['logs'] = []
    required = context.get('proposal_requirements', {}).get('required_evidence_ids', ['baseline'])
    context['evidence'] = [{'id': item['id'], 'facts': {'reference': str((round_dir / 'request.json').relative_to(run)),
        'request_hash': original['request_hash'], 'source_hash': context['source_hash']}}
        for item in context['evidence'] if item['id'] in required]
    context['evidence'].append({'id': 'debug-failure', 'facts': failure})
    context['protected_evidence_ids'] = [item['id'] for item in context['evidence']]
    trial = run / failure['trial']
    paths = [trial / name for name in ('stderr.log', 'stdout.log', 'summary.json', 'execution/stderr.log',
        'execution/stdout.log', 'service/summary.json', 'transaction/stderr.log', 'transaction/stdout.log',
        'transaction-dense/stderr.log', 'transaction-dense/stdout.log')]
    attempt = next(run / 'attempts' / record[stage] for stage in ('confirmation', 'screen')
                   if stage in record and trial.is_relative_to(run / 'attempts' / record[stage]))
    paths += [attempt / 'build/candidate/native/build' / name for name in ('stderr.log', 'stdout.log')]
    for path in paths:
        if path.is_file():
            context['logs'].append({'file': str(path.relative_to(run)), 'sha256': digest(path),
                                    'tail': path.read_text(errors='replace')[-12000:]})
    if len((json.dumps(context, indent=2, ensure_ascii=False) + '\n').encode()) > plan['search']['max_context_bytes']:
        record['debug_skipped'] = 'Repair evidence exceeds the frozen context budget.'
        return False
    debug_dir = round_dir / 'debug'
    debug_dir.mkdir(exist_ok=True)
    write_json(debug_dir / 'request.json', context)
    record.clear()
    record.update(phase='request', parent=original['parent'], selected=original['selected'], debug=original,
                  request_hash=digest(debug_dir / 'request.json'))
    for key in ('comparison', 'reuse', 'branch_depth', 'branch_seed', 'routing'):
        if key in original:
            record[key] = original[key]
    write_json(round_dir / 'state.json', record)
    return True


def _lesson(run, plan, round_dir, record):
    proposal = record.get('selected', {})
    evidence = {}
    attempts = {item[stage] for item in (record, record.get('debug', {}))
                for stage in ('screen', 'confirmation') if stage in item}
    for name in attempts:
        attempt = run / 'attempts' / name
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
    for name in ('request.json', 'proposal.json', 'change.patch', 'usage.json',
                 'debug/request.json', 'debug/proposal.json', 'debug/change.patch', 'debug/usage.json'):
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
    if 'direction' in proposal:
        lesson['direction'] = proposal['direction']
        lesson['comparison_hash'] = record.get('comparison', record['parent'])['source_hash']
    if record.get('debug'):
        for item in read_json(round_dir / 'debug/request.json')['logs']:
            if digest(run / item['file']) != item['sha256']:
                raise ValueError('debug failure log changed')
            evidence[item['file']] = item['sha256']
        lesson['debug'] = {key: record['debug'][key] for key in
                           ('decision', 'reason', 'candidate_hash', 'screen', 'confirmation') if key in record['debug']}
        lesson['debug']['patch_hash'] = identity(record['debug']['selected']['patch'].strip())
    destination = run / 'search/experience' / f'{round_dir.name}.json'
    write_json(destination, {**lesson, 'record_hash': identity(lesson)})


def _record_ledger(run, plan, round_dir, record, ledger, router):
    """Append one evaluated direction to the routing ledger.

    Only candidates that actually reached a verdict are recorded, so a duplicate
    or protocol failure never enters the ability estimate. The score is the
    measured gain mapped onto (0, 1), not the PASS flag, so a hard direction that
    pays off is not permanently outranked by an easy one.
    """
    proposal = record.get('selected') or {}
    if not proposal or record.get('decision') not in ('promoted', 'rejected'):
        return
    # A crash between the lesson and the state save replays this round; the
    # ledger is append-only, so the replay must not add a second row.
    if any(row.get('round') == round_dir.name for row in ledger.rows):
        return
    verdict = None
    stage_used = None
    for stage in ('confirmation', 'screen'):
        attempt = record.get(stage)
        if not attempt:
            continue
        path = run / 'attempts' / attempt / 'verdict.json'
        if path.is_file():
            verdict = read_json(path)
            stage_used = stage
            if verdict.get('gain_pct') is not None:
                break
    if verdict is None:
        return
    ledger.add(category=proposal.get('category', 'agent-protocol'),
               features=router.features(proposal['patch']),
               score=normalized_score(verdict, plan['min_gain_pct']),
               decision=record['decision'], foundation_hash=plan['foundation_hash'],
               round_name=round_dir.name, gain_pct=verdict.get('gain_pct'),
               stage=stage_used)


def _prepare_adaptive_round(run, plan, state, best, records, router, directory):
    items = adaptive.rounds(run, state)
    rows = [row for item in items if (row := adaptive.measured(run, item)) is not None]
    strategy = adaptive.strategy(plan, items, rows, best)
    exploration_used = sum(bool(item['record'].get('branch_seed') or
        item['record'].get('routing', {}).get('explored')) for item in items)
    seen = {identity(item['selected']['patch'].strip())
            for entry in items for item in (entry['record'], entry['record'].get('debug', {}))
            if item.get('parent', {}).get('source_hash') == best['source_hash'] and item.get('selected')}
    candidates = adaptive.pending(run, items, best['source_hash'], seen)
    if not candidates and exploration_used < int(plan['search']['max_rounds'] * plan['search']['explore_max_fraction']):
        for item in items:
            original = item['original']
            if original.get('branch_seed') and original['comparison']['source_hash'] == best['source_hash']:
                parent_hash = original['parent']['source_hash']
                tried = {identity(entry['record']['selected']['patch'].strip()) for entry in items
                         if entry['record'].get('selected') and entry['record']['parent']['source_hash'] == parent_hash}
                candidates.extend(adaptive.pending(run, [item], parent_hash, tried))
    slots = adaptive.completion_slots(plan)
    can_call = (state['calls'] < plan['search']['max_calls'] and
                state.get('model_calls_reserved', 0) + slots <= plan['search']['max_model_calls'])
    can_call = can_call and time.time() + 2 * adaptive.confirmation_reserve(plan) + plan['search']['agent_timeout_seconds'] < plan['expires_at']
    if time.time() + 2 * adaptive.confirmation_reserve(plan) >= plan['expires_at']:
        state['reason'] = 'Remaining time cannot reserve independent full confirmation.'
        return None
    if candidates and (not strategy['review_due'] or not can_call):
        decision = adaptive.rank(candidates, rows, best['source_hash'], plan, exploration_used)
        chosen = candidates[decision['index']]
        origin = run / 'search/rounds' / chosen['batch']
        write_json(directory / 'request.json', read_json(origin / 'request.json'))
        write_json(directory / 'proposal.json', read_json(origin / 'proposal.json'))
        write_json(directory / 'routing.json', decision)
        result = dict(phase='proposed', parent=chosen['parent'], comparison=best,
                    request_hash=digest(directory / 'request.json'), response_hash=digest(directory / 'proposal.json'),
                    selected=chosen['proposal'], routing=decision,
                    reuse={'batch': chosen['batch'], 'index': chosen['index'],
                           'request_hash': digest(origin / 'request.json'), 'response_hash': digest(origin / 'proposal.json')})
        for key in ('branch_depth', 'branch_seed'):
            if key in chosen['record']:
                result[key] = chosen['record'][key]
        return result
    if not can_call:
        state['reason'] = 'Proposal pool exhausted for this parent and model budget or analysis/confirmation time exhausted.'
        return None
    parent = None if strategy['review_due'] else adaptive.continuation(run, items, rows, best, plan, exploration_used)
    if parent:
        strategy.update(reason='bounded continuation of a valid slower intermediate', step=2, mode='deep')
    context = request_context(run, plan, best, records, len(state['rounds']), router, strategy=strategy, parent=parent)
    context['candidate_pool'] = [dict(id=f"{c['batch']}:{c['index']}", direction=c['proposal']['direction'],
        hypothesis=c['proposal']['hypothesis'], parent_hash=c['parent']['source_hash'])
        for item in items for c in adaptive.pending(run, [item], item['original']['parent']['source_hash'], seen)][:12]
    context['search_feedback'] = rows[-12:]
    def context_size():
        return len((json.dumps(context, indent=2, ensure_ascii=False) + '\n').encode())
    while context['knowledge'] and context_size() > plan['search']['max_context_bytes']:
        context['knowledge'] = context['knowledge'][:-2000]
    if context_size() > plan['search']['max_context_bytes']:
        raise ValueError('adaptive feedback exceeds frozen context budget')
    write_json(directory / 'request.json', context)
    record = dict(phase='request', parent=parent or best, comparison=best,
                  request_hash=digest(directory / 'request.json'))
    if parent:
        record.update(branch_depth=2, branch_seed=parent['seed'])
    return record


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
        ledger = Ledger(run / 'search/ledger.jsonl').load()
        router = Router(settings, ledger, foundation_hash=plan['foundation_hash'])
        while state['status'] == 'running':
            best = best_candidate(run, root)
            records = load_experiences(run, plan)
            if time.time() >= plan['expires_at']:
                state.update(status='budget_exhausted', reason='Frozen wall-time budget exhausted.')
                _save_state(run, state)
                (run / 'search/pending.json').unlink(missing_ok=True)
                break
            if state['active'] is None:
                if (time.time() >= plan['expires_at'] or len(state['rounds']) >= settings['max_rounds']
                        or (not adaptive.enabled(plan) and state['calls'] >= settings['max_calls'])):
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
                if adaptive.enabled(plan):
                    record = _prepare_adaptive_round(run, plan, state, best, records, router, round_dir)
                    if record is None:
                        state.update(status='budget_exhausted', active=None)
                        state['rounds'].pop()
                        _save_state(run, state)
                        break
                else:
                    context = request_context(run, plan, best, records, len(state['rounds']), router)
                    write_json(round_dir / 'request.json', context)
                    record = {'phase': 'request', 'parent': best, 'request_hash': digest(round_dir / 'request.json')}
                write_json(path, record)
            record = read_json(path)
            if record.get('debug'):
                for filename, key in (('request.json', 'request_hash'), ('proposal.json', 'response_hash')):
                    if digest(round_dir / filename) != record['debug'][key]:
                        raise ValueError('original failed proposal evidence changed')
            work_dir = round_dir / 'debug' if record.get('debug') else round_dir
            if digest(work_dir / 'request.json') != record['request_hash']:
                raise ValueError('agent request changed')
            context = read_json(work_dir / 'request.json')
            response = work_dir / 'proposal.json'
            if adaptive.enabled(plan):
                original_context = read_json(round_dir / 'request.json')
                if (record['parent']['source_hash'] != original_context['source_hash']
                        or record['comparison']['source_hash'] != original_context['strategy']['anchor']):
                    raise ValueError('candidate parent or comparison identity changed')
                accepted = (record['phase'] in ('proposed', 'screening', 'confirming') or
                            record.get('decision') in ('promoted', 'rejected', 'invalid_patch'))
                if record.get('selected') and accepted:
                    if record['selected'] not in read_json(response)['proposals']:
                        raise ValueError('selected candidate differs from the sealed response')
                comparison = (run / record['comparison']['source']).resolve()
                if not comparison.is_relative_to(run) or inventory(comparison)['hash'] != record['comparison']['source_hash']:
                    raise ValueError('comparison baseline source changed')
            if record.get('reuse'):
                origin = run / 'search/rounds' / record['reuse']['batch']
                if any(digest(origin / file) != record['reuse'][key] for file, key in
                       (('request.json', 'request_hash'), ('proposal.json', 'response_hash'))):
                    raise ValueError('reused proposal origin changed')
                selected = record.get('debug', record)['selected']
                if selected != read_json(origin / 'proposal.json')['proposals'][record['reuse']['index']]:
                    raise ValueError('reused proposal index changed')
            parent_source = (run / record['parent']['source']).resolve()
            if not parent_source.is_relative_to(run) or inventory(parent_source)['hash'] != record['parent']['source_hash']:
                raise ValueError('patch parent source changed')
            if context.get('missing_context'):
                state.update(status='context_required', reason='; '.join(context['missing_context']))
                _save_state(run, state)
                break
            if record['phase'] == 'request':
                if time.time() >= plan['expires_at']:
                    state.update(status='budget_exhausted', reason='Wall time exhausted before model call.')
                    _save_state(run, state)
                    break
                # Charge before dispatch; interrupted/failed calls are never refunded on resume.
                kind = 'debug' if record.get('debug') else 'proposal'
                if (state.get('dispatch', {}).get('round') != state['active']
                        or state['dispatch'].get('kind', 'proposal') != kind):
                    if state['calls'] >= settings['max_calls']:
                        state.update(status='budget_exhausted', reason='Agent-dispatch budget exhausted.')
                        _save_state(run, state)
                        break
                    if adaptive.enabled(plan):
                        slots = adaptive.completion_slots(plan, kind == 'debug')
                        if state.get('model_calls_reserved', 0) + slots > settings['max_model_calls']:
                            state.update(status='budget_exhausted', reason='Frozen model-completion reservation budget exhausted.')
                            _save_state(run, state)
                            break
                        state['model_calls_reserved'] = state.get('model_calls_reserved', 0) + slots
                    state['calls'] += 1
                    state['dispatch'] = {'round': state['active'], 'call': state['calls'], 'kind': kind}
                    _save_state(run, state)
                record.update(phase='awaiting_agent', call=state['dispatch']['call'])
                write_json(path, record)
            if record['phase'] == 'awaiting_agent':
                if state['external_agent'] and not response.exists():
                    write_json(run / 'search/pending.json', {'request': str((work_dir / 'request.json').relative_to(run)),
                               'response': str(response.relative_to(run)), 'call': record['call'],
                               'request_hash': record['request_hash']})
                    return search_report(run, root)
                if not response.exists() and not (work_dir / 'agent').exists():
                    replacements = {'{python}': sys.executable, '{foundation}': str(run / 'foundation'),
                                    '{request}': str(work_dir / 'request.json'), '{response}': str(response)}
                    argv = settings['agent_command'][:]
                    for marker, replacement in replacements.items():
                        argv = [part.replace(marker, replacement) for part in argv]
                    remaining = plan['expires_at'] - time.time()
                    if remaining > 0:
                        process = execute(argv, work_dir, work_dir / 'agent', min(settings['agent_timeout_seconds'], remaining),
                                          session_env(run / 'search/codex-session'))
                        record['agent_process'] = process
                        write_json(path, record)
                try:
                    if response.stat().st_size > settings['max_context_bytes'] * 2:
                        raise ValueError('agent response exceeds context bound')
                    value = validate_proposal(read_json(response), {item['id'] for item in context['evidence']}, context.get('proposal_requirements'))
                    if not state['external_agent']:
                        receipt = read_json(work_dir / 'agent/process.json')
                        if receipt['status'] != 'completed' or not receipt['cleanup_complete']:
                            raise ValueError('agent did not complete cleanly')
                    record['response_hash'] = digest(response)
                    if value['stop']:
                        missing = value.get('needs_context', [])
                        record.update(phase='decided', decision='context_required' if missing else 'stopped',
                                      reason=value['reason'], needs_context=missing)
                        state.update(status='context_required' if missing else 'agent_stopped', reason=value['reason'])
                    else:
                        if record.get('debug'):
                            if len(value['proposals']) != 1:
                                raise ValueError('debug must return exactly one repair proposal')
                            selected, decision = value['proposals'][0], None
                            original = record['debug']['selected']
                            if any(selected.get(key) != original.get(key)
                                   for key in ('hypothesis', 'category', 'experiment', 'direction')):
                                raise ValueError('debug changed the original hypothesis, category or experiment')
                        else:
                            if adaptive.enabled(plan):
                                items = adaptive.rounds(run, state)
                                rows = [row for item in items if (row := adaptive.measured(run, item)) is not None]
                                tried = {r['patch_hash'] for r in records if r['parent_hash'] == record['parent']['source_hash']}
                                fresh = [p for p in value['proposals'] if identity(p['patch'].strip()) not in tried]
                                if fresh:
                                    used = sum(bool(i['record'].get('branch_seed') or i['record'].get('routing', {}).get('explored')) for i in items)
                                    decision = adaptive.rank([{'proposal': p} for p in fresh], rows,
                                        record['comparison']['source_hash'], plan, used)
                                    selected = fresh[decision['index']]
                                    record['routing'] = decision
                                else:
                                    selected, decision = None, None
                            else:
                                selected, decision = route_proposals(
                                    value['proposals'], records, record['parent']['source_hash'], router)
                        if decision is not None:
                            write_json(round_dir / 'routing.json', decision)
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
                patch = work_dir / 'change.patch'
                patch.write_text(record['selected']['patch'])
                source, staging = work_dir / 'source', work_dir / 'staging'
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
                    record.update(phase='screening', candidate_hash=inventory(source)['hash'],
                                  source=str(source.relative_to(run)))
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    record.update(phase='decided', decision='invalid_patch', reason=str(error))
                write_json(path, record)
            if record['phase'] in ('screening', 'confirming'):
                if inventory(run / record.get('source', str((work_dir / 'source').relative_to(run))))['hash'] != record['candidate_hash']:
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
                        promoted = {'source': record.get('source', str((work_dir / 'source').relative_to(run))),
                                    'source_hash': record['candidate_hash'], 'parent_hash': record.get('comparison', record['parent'])['source_hash'],
                                    'screen': {'attempt': record['screen'], 'hash': digest(run / 'attempts' / record['screen'] / 'verdict.json')},
                                    'confirmation': {'attempt': record['confirmation'], 'hash': digest(attempt / 'verdict.json')}}
                        write_json(run / 'search/best.json', promoted)
                        record.update(phase='decided', decision='promoted', reason=verdict)
                    else:
                        record.update(phase='decided', decision='rejected', reason={'confirmation': verdict})
                    write_json(path, record)
            if record['phase'] == 'decided':
                failure = _debug_failure(run, record) if record['decision'] == 'rejected' and not record.get('debug') else None
                if failure:
                    can_debug = state['calls'] < settings['max_calls'] and time.time() < plan['expires_at']
                    if adaptive.enabled(plan):
                        can_debug = can_debug and state.get('model_calls_reserved', 0) < settings['max_model_calls']
                        can_debug = can_debug and time.time() + 2 * adaptive.confirmation_reserve(plan) < plan['expires_at']
                    if can_debug:
                        if _prepare_debug(run, plan, round_dir, record, failure):
                            if ((run / 'recovery.json').is_file()
                                    and read_json(run / 'recovery.json').get('health') in ('required', 'failed')):
                                campaign.recover(run, root)
                            continue
                    else:
                        record['debug_skipped'] = 'Frozen dispatch or wall-time budget exhausted.'
                    write_json(path, record)
                _lesson(run, plan, round_dir, record)
                _record_ledger(run, plan, round_dir, record, ledger, router)
                if record['decision'] == 'stopped':
                    state.update(status='agent_stopped', reason=record['reason'])
                elif record['decision'] == 'context_required':
                    state.update(status='context_required', reason=record['reason'])
                state['active'] = None
                state.pop('dispatch', None)
                _save_state(run, state)
                (run / 'search/pending.json').unlink(missing_ok=True)
                if (run / 'recovery.json').is_file() and read_json(run / 'recovery.json').get('health') in ('required', 'failed'):
                    campaign.recover(run, root)
        (run / 'search/pending.json').unlink(missing_ok=True)
        return search_report(run, root)


def _routing_summary(run, plan):
    """One honest line about the direction router; no estimate is called a result."""
    if adaptive.enabled(plan):
        return ('Direction routing: adaptive-v1, recent measured-gain momentum; information exploration '
                'requires >6 valid evaluations, empirical score uncertainty and shared exploration budget. '
                'Scores guide experiments and cannot qualify a candidate.')
    ledger = Ledger(run / 'search/ledger.jsonl').load()
    router = Router(plan['search'], ledger, foundation_hash=plan['foundation_hash'])
    if not router.enabled:
        return 'Direction routing: off (frozen category counts only).'
    return (f'Direction routing: {router.kind()} ability over {len(ledger.rows)} evaluated '
            f'direction(s); scores order proposals only and never substitute for the two-PASS promotion gate.')


def search_report(run, root):
    plan = campaign.load(run, root)
    state = read_json(run / 'search/state.json')
    best = best_candidate(run, root) if (run / 'search/best.json').is_file() else None
    primary = plan.get('service', {}).get('primary')
    components = primary if isinstance(primary, list) else [primary] if primary else []
    scope = components[0]['id'] if len(components) == 1 else 'service-scenario-mean' if components else plan['scope']
    unit = (('tokens/s' if components[0]['metric'].endswith('_per_s') else 'ms') if len(components) == 1
            else 'percent gain' if components else plan['unit'])
    lines = [f'# {plan["name"]}', '', f'Search status: {state["status"]}', '',
             f'Reason: {state["reason"]}', '', f'Agent dispatches charged: {state["calls"]} / {plan["search"]["max_calls"]}', '',
             f'Backend: {plan["backend"]}; scope: {scope}; unit: {unit}', '',
             f'Require clean service shutdown: {plan.get("service", {}).get("require_clean_shutdown", True)}; raw lifecycle results are retained.', '',
             f'Best source: {best["source_hash"] if best else "none: baseline unqualified"}', '',
             ('Promotion requires two model-service gains and a fresh confirmation with every final service gate. '
              if primary else 'Promotion requires a native screen and a fresh confirmation with every configured final service gate. ')
             + 'Operator-only PASS is not final qualification.', '',
             _routing_summary(run, plan), '',
             'A candidate compile/runtime/numerical failure gets at most one code-only repair within the dispatch budget. '
             'A repaired candidate restarts screening and independent confirmation; slow valid results do not trigger debug.', '',
             '| Round | Decision | Debug | Evidence |', '| --- | --- | --- | --- |']
    if adaptive.enabled(plan):
        items = adaptive.rounds(run, state)
        valid = sum(adaptive.measured(run, item) is not None for item in items)
        reused = sum(bool(item['record'].get('reuse')) for item in items)
        branches = sum(bool(item['record'].get('branch_seed')) for item in items)
        reviews = sum(item['request']['strategy']['mode'] == 'global-redesign' and not item['record'].get('reuse') for item in items)
        usage = [read_json(path) for item in items for path in (
            run / 'search/rounds' / item['name'] / 'usage.json',
            run / 'search/rounds' / item['name'] / 'debug/usage.json') if path.is_file()]
        lines[12:12] = [f'Adaptive policy: reused candidates={reused}; valid evaluations={valid}; '
            f'global reviews={reviews}; exploration continuations={branches}.', '',
            f'Model completion slots reserved: {state.get("model_calls_reserved", 0)} / {plan["search"]["max_model_calls"]}; '
            f'reported completions={sum(u.get("model_calls", 0) for u in usage)}; '
            f'dispatches without usage={state["calls"] - len(usage)}. Reservations are conservative and never refunded.', '']
    for name in state['rounds']:
        path = run / 'search/rounds' / name / 'state.json'
        record = read_json(path) if path.is_file() else {}
        debug = 'used once' if record.get('debug') else record.get('debug_skipped', '')
        lines.append(f'| {name} | {record.get("decision", record.get("phase", "preparing"))} | {debug} | search/rounds/{name}/ |')
    lines.extend(['', 'Synthetic results qualify controller contracts only.' if plan['backend'] == 'cpu-synthetic'
                  else 'NPU claims are limited to the frozen workload and configured native/service gates.', ''])
    destination = run / 'search-report.md'
    destination.write_text('\n'.join(lines))
    return destination
