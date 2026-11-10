"""Bounded response repair and source-bound, atomic model-stage checkpoints."""

import hashlib
import json
import time
from pathlib import Path

from .evidence import digest, read_json, write_json


class ModelExhausted(ValueError):
    def __init__(self, stage, reason):
        self.stage = stage
        super().__init__(f'Model {stage} response recovery exhausted: {str(reason)[:2000]}')


class ContextFull(RuntimeError):
    pass


def check_size(value, request):
    limit = request.get('max_response_bytes', request.get('max_context_bytes', 2097152) * 2)
    if len((json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode()) > limit:
        raise ValueError('response exceeds frozen output budget; produce one concise proposal')


def attempts(request):
    return 3 if request.get('model_recovery') == 1 else 1 if request.get('debug') else 2


def repair_hint(stage, number, error, *, truncated=False):
    return (f'Repair attempt {number}: {str(error)[:1000]}. Return only valid schema JSON; no Markdown or explanation. '
        'Use one highest-priority proposal only. Keep every required evidence and experiment field; '
        'make descriptions concise. Do not repeat source/SDK text. '
        + ('Stop extended reasoning and produce the answer now. ' if truncated else '')
        + ('Use exact short old/new edits; include only required changed spans, not entire files. '
           'Keep the analyzed hypothesis, source base and metadata unchanged.' if stage == 'code' else
           'Select only the files needed for one falsifiable mechanism. Do not write code.'))


class Journal:
    def __init__(self, directory, request):
        self.directory = Path(directory)
        self.path = self.directory / 'model-state.json'
        self.request_hash = request_hash(request)
        self.limit = attempts(request)
        self.state = read_json(self.path) if self.path.exists() else dict(request_hash=self.request_hash, stages={},
            deadline=min(float(request.get('deadline') or time.time() + 3600),
                         time.time() + float(request.get('agent_timeout_seconds') or 3600)))
        if self.state['request_hash'] != self.request_hash:
            raise ValueError('model checkpoint belongs to another request; preserve it and use a new proposal directory')

    def stage(self, name):
        return self.state['stages'].setdefault(name, dict(attempts=[], hint=''))

    def cached(self, name):
        entry = self.stage(name).get('result')
        if entry is None:
            return None
        path = self.directory / entry['file']
        if not path.resolve().is_relative_to(self.directory.resolve()):
            raise ValueError('model checkpoint result escapes proposal directory')
        if digest(path) != entry['sha256']:
            raise ValueError('completed model stage changed')
        return read_json(path)

    def reserve(self, name):
        stage = self.stage(name)
        if time.time() >= self.state['deadline']:
            raise ModelExhausted(name, 'frozen model-stage time budget exhausted')
        index = len(stage['attempts'])
        if index >= self.limit:
            raise ModelExhausted(name, stage['hint'] or 'reserved attempts already consumed')
        stage['attempts'].append(dict(index=index, status='reserved'))
        write_json(self.path, self.state)
        return index

    def rejected(self, name, error, *, truncated=False):
        stage = self.stage(name)
        stage['attempts'][-1].update(status='rejected', reason=str(error)[:2000])
        stage['truncated'] = truncated
        stage['hint'] = repair_hint(name, len(stage['attempts']), error, truncated=truncated)
        write_json(self.path, self.state)

    def complete(self, name, value, **extra):
        path = self.directory / f'model-{name}-complete.json'
        write_json(path, value)
        stage = self.stage(name)
        stage.update(result=dict(file=path.name, sha256=digest(path)), **extra)
        if stage['attempts']:
            stage['attempts'][-1]['status'] = 'completed'
        write_json(self.path, self.state)


def failure_response(request, error):
    return dict(stop=False, proposals=[], reason=str(error), model_failure=dict(
        request_hash=request_hash(request), stage=error.stage))


def request_hash(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_failure(value, request=None):
    if (set(value) != {'stop', 'proposals', 'reason', 'model_failure'} or value['stop'] is not False
            or value['proposals'] != [] or not isinstance(value['reason'], str) or not value['reason'].strip()
            or not isinstance(value['model_failure'], dict)
            or set(value['model_failure']) != {'request_hash', 'stage'}
            or value['model_failure']['stage'] not in ('analysis', 'code')
            or not isinstance(value['model_failure']['request_hash'], str)
            or (request is not None and value['model_failure']['request_hash'] != request_hash(request))):
        raise ValueError('invalid model failure receipt')
    return value


def resumable(directory, request):
    """Only retry a known request after all owned proposer processes were cleaned."""
    directory = Path(directory)
    if not request.get('model_recovery'):
        return False
    if (directory / 'model-state.json').exists():
        journal = Journal(directory, request)
        for stage in ('analysis', 'code'):
            journal.cached(stage)
    elif (directory / 'model-start.json').exists():
        if read_json(directory / 'model-start.json')['request_hash'] != request_hash(request):
            raise ValueError('proposer launch belongs to another request')
    else:
        return False
    receipts = list(directory.glob('launches/*/process.json'))
    if (directory / 'process.json').exists():
        receipts.append(directory / 'process.json')
    return bool(receipts) and all(read_json(path).get('cleanup_complete') is True for path in receipts)


def seed_analysis(journal, request):
    seed = request.get('analysis_seed')
    if seed and journal.cached('analysis') is None:
        journal.complete('analysis', seed['analysis'], origin=seed['origin'], reads=seed.get('reads'))
