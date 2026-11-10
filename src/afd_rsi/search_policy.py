"""Adaptive decisions derived from sealed rounds, never from a mutable queue."""

import math
import re
from statistics import median, stdev

from .evidence import digest, read_json
from .workspace import inventory

VERSION = 'adaptive-v1'
PERFORMANCE_REASONS = {
    'positive gain meeting the frozen threshold required',
    'positive model service gain meeting the frozen threshold required',
}


def enabled(plan):
    return plan['search'].get('policy') == VERSION


def direction_schema():
    return dict(type='object', additionalProperties=False,
        required=['id', 'mechanism', 'step', 'refs', 'use', 'next'], properties={
            'id': dict(type='string', pattern=r'^[a-z][a-z0-9_-]{0,63}$'),
            'mechanism': dict(type='string', minLength=1),
            'step': dict(type='integer', minimum=1, maximum=3),
            'refs': dict(type='array', minItems=1, items=dict(type='string')),
            'use': dict(type='string', minLength=1),
            'next': dict(type='string', description='Concrete continuation if a valid intermediate is slower; empty otherwise.')})


def validate_direction(value, requirements):
    if (not isinstance(value, dict) or set(value) != set(direction_schema()['required'])
            or any(not isinstance(value[k], str) or not value[k].strip() for k in ('id', 'mechanism', 'use'))
            or not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', value['id'])
            or type(value['step']) is not int or value['step'] not in (1, 2, 3)
            or not isinstance(value['next'], str)
            or not isinstance(value['refs'], list) or not value['refs']
            or not all(isinstance(ref, str) and ref in requirements['knowledge_refs'] for ref in value['refs'])):
        raise ValueError('direction needs a mechanism, step, frozen knowledge references and applicability explanation')


def rounds(run, state):
    """The original request/response are also the durable proposal pool."""
    result = []
    for name in state['rounds']:
        directory = run / 'search/rounds' / name
        if not (directory / 'state.json').exists():
            continue
        record = read_json(directory / 'state.json')
        original = record.get('debug', record)
        if digest(directory / 'request.json') != original['request_hash']:
            raise ValueError('pool request changed')
        response = directory / 'proposal.json'
        if 'response_hash' in original and digest(response) != original['response_hash']:
            raise ValueError('pool response changed')
        result.append(dict(name=name, record=record, original=original,
                           request=read_json(directory / 'request.json')))
    return result


def measured(run, item):
    """Reject incomplete/incorrect runs; do not reinterpret a FAIL as a PASS."""
    record = item['record']
    if record.get('phase') != 'decided' or record.get('decision') not in ('promoted', 'rejected'):
        return None
    for stage in ('confirmation', 'screen'):
        if stage not in record:
            continue
        attempt = run / 'attempts' / record[stage]
        verdict = read_json(attempt / 'verdict.json')
        observations = read_json(attempt / 'observations.json')
        gain = verdict.get('gain_pct')
        if (gain is None or not math.isfinite(gain) or verdict['verdict'] == 'INCONCLUSIVE'
                or set(verdict['reasons']) - PERFORMANCE_REASONS
                or any(o['status'] != 'valid' or not o.get('checks_passed') for o in observations)):
            return None
        noise = 0.0
        if verdict.get('primary_source') != 'service':
            for samples in verdict['samples'].values():
                if len(samples) > 1:
                    center = median(samples)
                    noise = max(noise, 100 * median(abs(v - center) for v in samples) / center)
        return dict(round=item['name'], gain=gain, noise=noise,
                    direction=record['selected']['direction'],
                    baseline=record.get('comparison', record['parent'])['source_hash'],
                    promoted=record['decision'] == 'promoted', stage=stage)
    return None


def pending(run, items, parent_hash, seen):
    candidates, used = [], set(seen)
    for item in items:
        original = item['original']
        if (item['record'].get('reuse') or 'response_hash' not in original
                or original['parent']['source_hash'] != parent_hash):
            continue
        value = read_json(run / 'search/rounds' / item['name'] / 'proposal.json')
        if value.get('stop') or value.get('model_failure') or original.get('decision') == 'invalid_agent':
            continue
        for index, proposal in enumerate(value['proposals']):
            from .search import identity
            key = identity(proposal['patch'].strip())
            if key not in used:
                used.add(key)
                candidates.append(dict(batch=item['name'], index=index, proposal=proposal,
                                       parent=original['parent'], record=original))
    return candidates


def strategy(plan, items, rows, best):
    recent = [row for row in rows if row['baseline'] == best['source_hash']]
    reviews = [item['request']['strategy'].get('review_count', 0) for item in items
               if not item['record'].get('reuse')
               and item['request']['strategy'].get('anchor') == best['source_hash']
               and item['request']['strategy']['mode'] == 'global-redesign']
    since = recent[(max(reviews) if reviews else 0):]
    stalled = []
    for row in reversed(since):
        if row['gain'] >= max(plan['min_gain_pct'], row['noise']):
            break
        stalled.append(row)
    review = (not reviews and not recent) or (len(stalled) >= 3 and len({r['direction']['mechanism'] for r in stalled}) >= 2)
    last = recent[-1] if recent else rows[-1] if rows and rows[-1]['promoted'] else None
    step = 2 if last is None else last['direction']['step']
    if last:
        step = min(3, step + 1) if last['promoted'] else max(1, step - 1)
        if len(rows) >= 2 and rows[-1]['promoted'] and rows[-2]['promoted'] and rows[-1]['gain'] < rows[-2]['gain'] / 2:
            step = max(1, last['direction']['step'] - 1)
    return dict(mode='global-redesign' if review else 'deep' if step > 1 else 'incremental',
        anchor=best['source_hash'], step=step, review_due=review, review_count=len(recent), valid_evaluations=len(rows),
        stagnation=len(stalled), reason='initial global diagnosis' if not items else
        'multiple mechanisms stalled' if review else 'measured trend and current parent',
        guidance='Use the direction catalog and local skills as evidence-backed starting points, not a whitelist. '
        'Review RX/pull, GMM1, activation, GMM2 and output/TX after source readiness, including hidden work. '
        'Explain failed predictions and compare alternatives during global review. Step 1/2/3 means local/coordinated/architectural scope; '
        'choose a different justified scope when needed. A global review does not require a rewrite. '
        'Cite frozen knowledge, explain applicability or departure, and name a falsifiable new mechanism when existing methods do not fit.')


def rank(candidates, rows, baseline, plan, exploration_used):
    """Small-sample momentum; information exploration needs empirical variability."""
    scores = []
    scale = max(plan['min_gain_pct'], .5)
    for index, candidate in enumerate(candidates):
        proposal = candidate['proposal']
        direction = proposal['direction']
        history = [row['gain'] for row in rows if row['baseline'] == baseline
                   and row['direction']['id'] == direction['id']][-3:]
        weights = [2 ** i for i in range(len(history))]
        momentum = sum(w * g for w, g in zip(weights, history)) / sum(weights) if history else 0.0
        expected = .5 + .5 * math.tanh(momentum / (2 * scale))
        error = None
        if len(history) >= 3:
            # Delta method maps gain variability to the same score scale.
            error = .25 / scale * (1 - math.tanh(momentum / (2 * scale)) ** 2) * stdev(history) / math.sqrt(len(history))
        scores.append(dict(index=index, expected=expected, se=error, momentum=momentum,
                           priority=proposal['priority']))
    selected = max(scores, key=lambda s: (s['expected'], s['priority'], -s['index']))
    explored = False
    settings = plan['search']
    if (len(rows) > 6 and settings['info_floor'] > 0 and
            exploration_used < int(settings['max_rounds'] * settings['explore_max_fraction'])):
        contenders = [s for s in scores if s['index'] != selected['index'] and s['se'] is not None
                      and s['expected'] * (1 - s['expected']) >= settings['info_floor']
                      and s['expected'] + 1.96 * s['se'] > selected['expected'] + settings['margin']]
        if contenders:
            selected = max(contenders, key=lambda s: s['expected'] + 1.96 * s['se'])
            explored = True
    return dict(index=selected['index'], explored=explored, kind=VERSION, scores=scores)


def continuation(run, items, rows, best, plan, exploration_used):
    if exploration_used >= int(plan['search']['max_rounds'] * plan['search']['explore_max_fraction']):
        return None
    for row in reversed(rows):
        item = next(item for item in items if item['name'] == row['round'])
        record = item['record']
        if (row['baseline'] != best['source_hash'] or row['promoted'] or not -5 <= row['gain'] <= 0
                or not row['direction']['next'].strip() or record.get('branch_depth', 1) >= 2
                or any(other['record'].get('branch_seed') == item['name'] for other in items)):
            continue
        if inventory(run / record['source'])['hash'] != record['candidate_hash']:
            raise ValueError('exploration parent changed')
        return dict(source=record['source'], source_hash=record['candidate_hash'],
                    attempt=record['screen'], seed=item['name'])
    return None


def confirmation_reserve(plan):
    # A cold estimate uses the frozen per-phase ceilings; no historical latency is promised.
    sides = 2
    seconds = sides * (2 + plan['repetitions']) * plan['timeout_seconds']
    if 'service' in plan:
        seconds += sides * (plan['service']['timeout_seconds'] + plan['service']['cleanup_seconds'])
    if plan.get('native', {}).get('profile'):
        seconds += plan['timeout_seconds']
    return seconds


def completion_slots(plan, debug=False):
    return 3 if debug else 6


def evaluation_count(run, state):
    return sum(bool(item['record'].get('selected')) for item in rounds(run, state))
