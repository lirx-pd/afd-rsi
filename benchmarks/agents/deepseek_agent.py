"""Read-only DeepSeek proposer using the same request/response contract as Codex.

The adapter never touches the target checkout and never owns acceptance: it reads
request.json, calls the chat API, and writes proposal.json for the controller to
validate. Everything it needs is stdlib, so the proposer host needs no extra
package beyond a Python interpreter.

Configuration is taken from the environment only, so no credential or endpoint is
ever written into a campaign, a request, or the repository:

    DEEPSEEK_API_KEY                 required unless AFD_RSI_DEEPSEEK_KEY_ENV renames it
    AFD_RSI_DEEPSEEK_KEY_ENV         name of the variable holding the key
    AFD_RSI_DEEPSEEK_BASE_URL        required; OpenAI-compatible gateway root, e.g. ending /v1
    AFD_RSI_DEEPSEEK_MODEL           default DeepSeek-V4.1-Flash-1017, used by both stages
    AFD_RSI_DEEPSEEK_ANALYSIS_MODEL  optional per-stage override
    AFD_RSI_DEEPSEEK_CODE_MODEL      optional per-stage override
    AFD_RSI_DEEPSEEK_JSON_MODE       set 0 when the endpoint rejects JSON mode
    AFD_RSI_DEEPSEEK_EFFORT_ANALYSIS default max; thinking level for direction analysis
    AFD_RSI_DEEPSEEK_EFFORT_CODE     default high; thinking level for patch writing
    AFD_RSI_DEEPSEEK_MAX_TOKENS      output budget for both stages (analysis default 32768)
    AFD_RSI_DEEPSEEK_MAX_TOKENS_ANALYSIS / _CODE   per-stage output budget
    AFD_RSI_DEEPSEEK_TOKEN_CAP       ceiling for the truncated-reply escalation, default 65536
    AFD_RSI_DEEPSEEK_MAX_CHARS       prompt budget in characters; defaults to frozen request limit, else 200000

Neither the endpoint nor the credential has a built-in value: an invented default is
how a campaign ends up calling a host that never serves the requested model. Run
``--check`` on the proposer host first; it reports the missing piece and the gateway's
own model list instead of failing mid-round.

A recipe may freeze the per-stage models itself through ``[search.models]``; those
values win over the environment so a campaign always runs what it declared.
Analysis runs first and must not emit patches; the coding call runs only when the
analysis produced a falsifiable direction, mirroring the Codex adapter's policy.
Both stages default to one model because it is capable of both, but they stay
separately configurable so a future model can replace just one of them.
"""

import argparse
import fnmatch
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from afd_rsi.evidence import read_json, write_json
from afd_rsi.proposal import DEBUG_TASK, require_patch_tools, validate_patches
from afd_rsi.model_protocol import ModelContext, MODEL_RULES, READ_RULES, EDIT_TASK, DEBUG_EDIT_TASK, code_schema, compact
from afd_rsi.search_policy import direction_schema, validate_direction
from afd_rsi.model_recovery import Journal, ModelExhausted, failure_response, seed_analysis, check_size

DEFAULT_MODEL = 'DeepSeek-V4.1-Flash-1017'
DEFAULT_EFFORT = {'analysis': 'max', 'code': 'high'}
# The reasoning model consumes the output budget before it writes anything, so the
# analysis stage needs headroom well beyond the JSON it returns.
DEFAULT_MAX_TOKENS = {'analysis': 32768, 'code': 16384}
DEFAULT_TOKEN_CAP = 65536
# One recovery per stage, shared by truncation, schema and patch failures.
MAX_ATTEMPTS = 2


class Truncated(ValueError):
    """The model spent the output budget before writing an answer."""


class UsageLog:
    """Token record written after every call, so a failed round keeps its usage."""

    def __init__(self, path=None, *, resume=False):
        self.path = Path(path) if path else None
        self.turns = read_json(self.path)['turns'] if resume and self.path and self.path.exists() else []

    def add(self, turn):
        self.turns.append(turn)
        if self.path:
            write_json(self.path, {'turns': self.turns, 'model_calls': len(self.turns)})

    def record(self, turn, path=None):
        if path and not self.path:
            self.path = Path(path)
        self.add(turn)
DEFAULT_KEY_ENV = 'DEEPSEEK_API_KEY'
DEFAULT_MAX_CHARS = 200_000

SYSTEM_RULES = (
    'Propose a complete candidate patch using only the supplied evidence and repository-local knowledge.\n'
    'Do not change files, gates, validators, budgets, or knowledge. Do not read credentials or external repositories.\n'
    'Return one JSON object matching the supplied schema and nothing else.\n'
    'The evidence array must contain only exact IDs from the request, such as baseline, never explanations.\n'
    'The current request is authoritative; older history and profiling may be stale.\n'
    'Label a hypothesis as a hypothesis.\n'
    'Follow evidence-driven strategy; use supplied skills and allow justified new mechanisms.\n'
    'Choose up to three distinct proposals; stop when there is no justified change.'
)

ANALYSIS_TASK = (
    'Analyze evidence, profiling and history. Choose falsifiable optimization hypotheses and controlled experiments. '
    'When proposal_requirements is present, provide the complete experiment and required evidence IDs; use measured '
    'waits/stages, explain how the source change hides or removes them, and predict an observable change with a '
    'rejection criterion. Account for stage overlap and unhidden work; do not add concurrent durations. '
    'Stop rather than guess. Do not write patches.'
)

CODE_TASK = (
    'Implement only the justified hypotheses from your analysis as complete patches. Preserve their evidence '
    'IDs and explain any implementation limits. Return stop=true if no safe complete patch can be produced.'
)


def key_env() -> str:
    """Name of the environment variable holding the API key.

    The variable is named rather than assumed so a gateway that issues its own
    credential (for example an internal one) needs no renaming or copying.
    """
    return os.environ.get('AFD_RSI_DEEPSEEK_KEY_ENV') or DEFAULT_KEY_ENV


def api_key() -> str:
    name = key_env()
    key = os.environ.get(name)
    if not key:
        raise ValueError(f'{name} is not set on the proposer host')
    return key


def base_url() -> str:
    """Gateway root, including the API prefix such as /v1.

    There is deliberately no built-in default: an invented endpoint is exactly
    how a campaign ends up calling a host that never serves the requested model.
    ``--check`` reports the missing configuration before any round is spent.
    """
    value = (os.environ.get('AFD_RSI_DEEPSEEK_BASE_URL') or '').strip().rstrip('/')
    if not value:
        raise ValueError(
            'AFD_RSI_DEEPSEEK_BASE_URL is not set; point it at an OpenAI-compatible '
            'gateway root that serves the configured model')
    return value


def endpoint(path: str = '/chat/completions') -> str:
    return base_url() + path


def effort_for(stage: str) -> str:
    """Thinking level for a stage.

    Choosing an optimization direction is the reasoning-heavy task, so it gets
    the highest level; writing the patch from an already-decided direction is
    narrower work and runs one level down. Both are overridable per host.
    """
    key = 'AFD_RSI_DEEPSEEK_EFFORT_ANALYSIS' if stage == 'analysis' else 'AFD_RSI_DEEPSEEK_EFFORT_CODE'
    return (os.environ.get(key) or DEFAULT_EFFORT[stage]).strip()


def model_for(stage: str, request: dict | None = None) -> str:
    """Frozen per-stage model first, then environment overrides, then the default.

    A recipe that declares ``[search.models]`` owns the choice: ambient
    environment cannot silently change what a frozen campaign asked for. The
    environment stays useful for ad-hoc runs that declare nothing.
    """
    declared = (request or {}).get('models') or {}
    if isinstance(declared, dict) and isinstance(declared.get(stage), str) and declared[stage].strip():
        return declared[stage]
    shared = os.environ.get('AFD_RSI_DEEPSEEK_MODEL')
    if shared:
        return shared
    key = 'AFD_RSI_DEEPSEEK_ANALYSIS_MODEL' if stage == 'analysis' else 'AFD_RSI_DEEPSEEK_CODE_MODEL'
    return os.environ.get(key) or DEFAULT_MODEL


def json_mode_enabled() -> bool:
    """Whether to ask the endpoint for native JSON output and sampling control."""
    return (os.environ.get('AFD_RSI_DEEPSEEK_JSON_MODE') or '1') != '0'


def truncate(request: dict, budget: int) -> tuple[dict, list]:
    """Fit the request into the prompt budget without silently dropping evidence.

    Removal order follows the controller's own priority: logs first, then raw
    knowledge text, then the longest source files, then older history. Every
    removal is reported so the model knows what it is missing.
    """
    view = json.loads(json.dumps(request))
    dropped = []

    def size() -> int:
        return len(json.dumps(view, ensure_ascii=False))

    if size() > budget and view.get('logs') and not view.get('debug'):
        view['logs'] = []
        dropped.append('logs')

    if size() > budget and view.get('knowledge'):
        view['knowledge'] = ''
        dropped.append('knowledge')

    while size() > budget and view.get('source_files'):
        removable = [name for name in view['source_files'] if not any(
            fnmatch.fnmatchcase(name, pattern) for pattern in view.get('allowed_files', []))]
        if not removable:
            break
        longest = max(removable, key=lambda name: len(view['source_files'][name]))
        del view['source_files'][longest]
        view.setdefault('omitted_source_files', []).append(longest)
        if 'source_files' not in dropped:
            dropped.append('source_files')

    while size() > budget and len(view.get('evidence', [])) > 1:
        required = set(view.get('protected_evidence_ids', [])) | set(
            (view.get('proposal_requirements') or {}).get('required_evidence_ids', ['baseline']))
        index = next((i for i, item in enumerate(view['evidence'])
                      if item['id'] not in required), None)
        if index is None:
            break
        view['evidence'].pop(index)
        if 'evidence' not in dropped:
            dropped.append('evidence')

    if dropped:
        view['context_truncation'] = {
            'dropped': dropped,
            'note': 'Prompt budget was exceeded; the listed parts of the request were omitted.'}
    if size() > budget:
        view['missing_context'] = [*view.get('missing_context', []),
            'Required source, SDK contracts and evidence exceed AFD_RSI_DEEPSEEK_MAX_CHARS; '
            'raise the prompt budget or use Codex selective file reading']
    return view, dropped


def redact(text: str) -> str:
    """Strip key material from gateway messages before it can reach a log.

    Gateways sometimes echo the presented token in an auth error. A credential
    must never land in a campaign receipt, a probe report, or a test failure.
    """
    text = re.sub(r'sk-[A-Za-z0-9_\-]{6,}', 'sk-<redacted>', text)
    try:
        secret = os.environ.get(key_env())
    except Exception:                                     # pragma: no cover - defensive
        secret = None
    if secret:
        text = text.replace(secret, '<redacted>')
    return text


TRANSIENT_HTTP = frozenset({429, 500, 502, 503, 504})
RETRY_SLEEPS = (5, 15, 45, 90)


def retry_delay(failures: int) -> int:
    return RETRY_SLEEPS[min(failures - 1, len(RETRY_SLEEPS) - 1)]


def request_json(url: str, payload: dict | None, timeout: float,
                 attempts: int | None = None) -> dict:
    """One JSON round trip, retried while the gateway says it is unavailable.

    A proposal round is expensive and the gateway is a shared service, so a 503 or
    a connection reset is retried here with backoff. Client errors (bad key, bad
    request, unknown model) are raised immediately: retrying cannot fix those.
    """
    attempts = attempts or int(os.environ.get('AFD_RSI_DEEPSEEK_ATTEMPTS') or 5)
    key = api_key()
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, method='GET' if body is None else 'POST',
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
    failures = 0
    while True:
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            detail = redact(error.read().decode('utf-8', 'replace')[:2000])
            failures += 1
            if error.code not in TRANSIENT_HTTP or failures >= attempts:
                raise RuntimeError(f'gateway HTTP {error.code}: {detail}') from error
            time.sleep(retry_delay(failures))
        except urllib.error.URLError as error:
            failures += 1
            if failures >= attempts:
                raise RuntimeError(f'gateway request failed: {error.reason}') from error
            time.sleep(retry_delay(failures))


def post(payload: dict, timeout: float) -> dict:
    return request_json(endpoint(), payload, timeout)


# Direct connection only. The gateway is reachable from the proposer host, and a
# proxied hop would add its own credentials, buffering and truncation risk to a
# long JSON body, so environment proxy settings are deliberately ignored here.
PROXY_HANDLER = urllib.request.ProxyHandler({})
OPENER = urllib.request.build_opener(PROXY_HANDLER)

# Kept as a module attribute so tests can substitute a transport without network.
urlopen = OPENER.open


def extract_json(text: str) -> dict:
    """Accept a bare object, a fenced block, or one object inside prose."""
    body = text.strip()
    if body.startswith('```'):
        body = body.split('```')[1] if '```' in body[3:] else body[3:]
        body = body.split('\n', 1)[1] if '\n' in body else body
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    start = body.find('{')
    while start != -1:
        for end in range(len(body), start, -1):
            chunk = body[start:end]
            if not chunk.endswith('}'):
                continue
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                continue
        start = body.find('{', start + 1)
    raise ValueError('model did not return a JSON object')


HUNK_HEADER = re.compile(r'^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@')


def patch_problem(patch: str) -> str:
    """Reject a diff that git cannot apply even with --recount.

    Line counts are tolerated (the controller applies with --recount), but a hunk
    header without ranges is corrupt for every git version, so it must be repaired
    here instead of spending a device round on it. Everything else is left to the
    controller, which validates the applied result.
    """
    for line in patch.splitlines():
        if line.startswith('@@') and not HUNK_HEADER.match(line):
            return (f'hunk header {line[:24]!r} has no ranges; '
                    f'write @@ -start,count +start,count @@')
    return ''


def check_shape(value: dict, request: dict, *, analysis: bool) -> list[str]:
    """Advisory shape check; the controller re-validates authoritatively.

    Unknown keys are ignored and returned rather than fatal. A reasoning model
    sometimes echoes a request field (``type: json_object``) or appends notes;
    failing the whole round for that discards an expensive device experiment.
    Missing or ill-typed required content is still rejected.
    """
    if not isinstance(value, dict):
        raise ValueError('response must be a JSON object')
    required = {'stop', 'reason', 'proposals'}
    missing = required - set(value)
    if missing:
        raise ValueError(f'response is missing {sorted(missing)}')
    ignored = sorted(set(value) - required - {'needs_context'})
    missing_context = value.get('needs_context', [])
    if (not isinstance(missing_context, list)
            or not all(isinstance(item, str) and item.strip() for item in missing_context)
            or (missing_context and not value['stop'])):
        raise ValueError('needs_context requires stop=true and exact missing contracts')
    if type(value['stop']) is not bool or not isinstance(value['reason'], str):
        raise ValueError('stop must be boolean and reason must be a string')
    proposals = value['proposals']
    if not isinstance(proposals, list) or len(proposals) > 3:
        raise ValueError('proposals must be a list of at most three items')
    if value['stop'] and proposals:
        raise ValueError('a stop response must not carry proposals')
    if not value['stop'] and not proposals:
        raise ValueError('a non-stop response needs at least one proposal')
    allowed = {item['id'] for item in request.get('evidence', [])}
    requirements = request.get('proposal_requirements')
    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise ValueError('each proposal must be an object')
        expected = {'hypothesis', 'category', 'evidence', 'priority'}
        expected |= set() if analysis else {'patch'}
        if requirements:
            expected |= {'experiment'}
            if 'knowledge_refs' in requirements:
                expected.add('direction')
        if analysis and request.get('model_protocol') == 2:
            expected |= {'files'}
        missing = expected - set(proposal)
        if missing:
            raise ValueError(f'proposal is missing {sorted(missing)}')
        ignored += [f'proposal.{name}' for name in sorted(set(proposal) - expected)]
        if not all(isinstance(proposal[name], str) and proposal[name].strip()
                   for name in ('hypothesis', 'category')):
            raise ValueError('proposal needs a hypothesis and a category')
        if not analysis and not (isinstance(proposal['patch'], str) and proposal['patch'].strip()):
            raise ValueError('coding proposals need a patch')
        if not analysis:
            problem = patch_problem(proposal['patch'])
            if problem:
                raise ValueError(problem)
        if type(proposal['priority']) is not int or proposal['priority'] not in (1, 2, 3):
            raise ValueError('priority must be 1, 2 or 3')
        cited = proposal['evidence']
        if not isinstance(cited, list) or not cited or not set(cited) <= allowed:
            raise ValueError('evidence must cite exact IDs from the request')
        if requirements:
            if 'knowledge_refs' in requirements:
                validate_direction(proposal.get('direction'), requirements)
            experiment = proposal['experiment']
            fields = set(requirements['experiment_fields'])
            if (not isinstance(experiment, dict) or set(experiment) != fields
                    or not all(isinstance(v, str) and v.strip() for v in experiment.values())):
                raise ValueError('proposal needs a complete falsifiable experiment')
            if not set(requirements['required_evidence_ids']) <= set(cited):
                raise ValueError('proposal must cite the incumbent measurement and profile evidence')
    return ignored


def optional_parameters(model: str) -> dict:
    """JSON mode and deterministic sampling, unless the caller or endpoint disagrees."""
    if not json_mode_enabled():
        return {}
    return {'response_format': {'type': 'json_object'}, 'temperature': 0}


def send(payload, timeout, prefix_name=''):
    """Post, falling back once if the endpoint rejects an optional parameter.

    A gateway that fronts the model may not accept JSON mode, a temperature
    field, or a reasoning level. Dropping them is safe: the response is parsed
    and validated locally either way, so this only removes an optimisation,
    never a check. The drop is recorded so a silent downgrade stays visible.
    """
    try:
        return post(payload, timeout)
    except RuntimeError as error:
        message = str(error)
        optional = [key for key in ('response_format', 'temperature', 'reasoning_effort')
                    if key in payload]
        if not optional or not any(key in message for key in optional):
            raise
        # Drop only what the gateway actually named, so rejecting one parameter
        # does not quietly downgrade the others; fall back to all of them when the
        # message names none.
        named = [key for key in optional if key in message]
        dropped = named or optional
        reduced = {key: value for key, value in payload.items() if key not in dropped}
        if prefix_name:
            (prefix_name.parent / f'{prefix_name.name}.unsupported-parameters.json').write_text(
                json.dumps({'dropped': dropped, 'error': message}) + '\n')
        return post(reduced, timeout)


def token_budget(stage: str) -> int:
    """Output budget for a stage, before any truncation escalation."""
    key = 'AFD_RSI_DEEPSEEK_MAX_TOKENS_ANALYSIS' if stage == 'analysis' else 'AFD_RSI_DEEPSEEK_MAX_TOKENS_CODE'
    shared = os.environ.get('AFD_RSI_DEEPSEEK_MAX_TOKENS')
    value = os.environ.get(key) or shared
    return int(value) if value else DEFAULT_MAX_TOKENS[stage]


def token_cap() -> int:
    return int(os.environ.get('AFD_RSI_DEEPSEEK_TOKEN_CAP') or DEFAULT_TOKEN_CAP)


def ask(stage, messages, schema, request, *, timeout, prefix, analysis, usage=None, wire=None, parent_analysis=None, journal=None):
    """One recovery per ordinary stage; a candidate debug gets one coding call."""
    usage = usage or UsageLog()
    model = model_for(stage, request)
    effort = effort_for(stage)
    budget = min(token_budget(stage), token_cap())
    payload = {'model': model, 'messages': messages, 'max_tokens': budget, 'stream': False,
               'reasoning_effort': effort, **optional_parameters(model)}
    if journal is not None:
        cached = journal.cached(stage)
        if cached is not None:
            check_shape(cached, request, analysis=analysis)
            if wire is not None and analysis and journal.stage(stage).get('reads'):
                wire.restore_reads(journal.stage(stage)['reads'])
            if not analysis:
                validate_patches(cached, request)
            return cached, {'kind': 'cached'}
        read = journal.stage(stage).get('read')
        if read is not None:
            content = wire.read(read['read'])
            messages = messages + [dict(role='assistant', content=compact(read)), dict(role='user', content=content)]
    (prefix.parent / f'{prefix.name}.request.json').write_text(
        json.dumps({key: value for key, value in payload.items() if key != 'messages'}) + '\n')
    errors = []
    keys = sorted(schema['properties'])
    limit = journal.limit if journal is not None else 1 if request.get('debug') else MAX_ATTEMPTS
    start = len(journal.stage(stage)['attempts']) if journal is not None else 0
    for attempt in range(start, limit):
        if journal is not None:
            journal.reserve(stage)
            entry = journal.stage(stage)
            payload['messages'] = messages + ([dict(role='user', content=entry['hint'])] if entry['hint'] else [])
            if entry.get('truncated'):
                payload['max_tokens'] = min(budget * 2, token_cap())
                if attempt >= 2:
                    payload['reasoning_effort'] = 'medium' if analysis else 'low'
        try:
            response = send(payload, min(timeout, max(.001, journal.state['deadline'] - time.time())) if journal is not None else timeout, prefix)
        except (RuntimeError, OSError) as error:
            if journal is None:
                raise
            journal.rejected(stage, redact(str(error)))
            usage.add(dict(model=model, stage=stage, attempt=attempt, error=redact(str(error)), usage_unknown=True))
            continue
        suffix = '' if attempt == 0 else f'-retry{attempt}'
        (prefix.parent / f'{prefix.name}.response{suffix}.json').write_text(
            json.dumps(response) + '\n')
        reported = response.get('usage') or {}
        choices = response.get('choices') or []
        finish = (choices[0].get('finish_reason') if choices else None) or ''
        text = ''
        if choices:
            text = (choices[0].get('message') or {}).get('content') or ''
        turn = {'model': model, 'stage': stage, 'effort': payload['reasoning_effort'], 'attempt': attempt,
                'max_tokens': payload['max_tokens'], 'finish_reason': finish,
                'prompt_tokens': reported.get('prompt_tokens'),
                'completion_tokens': reported.get('completion_tokens'),
                'cached_tokens': (reported.get('prompt_tokens_details') or {}).get('cached_tokens',
                                    reported.get('prompt_cache_hit_tokens')),
                'reasoning_tokens': (reported.get('completion_tokens_details') or {}).get('reasoning_tokens'),
                'input_utf8_bytes': sum(len(item['content'].encode()) for item in payload['messages']),
                'output_utf8_bytes': len(text.encode()),
                'ignored_keys': []}
        try:
            if not choices:
                raise ValueError('response carried no choices')
            if finish == 'length' and (journal is not None or not text.strip()):
                raise Truncated(
                    f'output budget {payload["max_tokens"]} was spent before any answer '
                    f'(finish_reason=length)')
            value = extract_json(text)
            if wire is not None and analysis and 'read' in value:
                if set(value) != {'read'}:
                    raise ValueError('read must be the only response field')
                if (journal is not None and journal.stage(stage).get('read')) or attempt + 1 >= MAX_ATTEMPTS:
                    if journal is not None:
                        raise ValueError('Context read allowance exhausted; use the supplied context and return final analysis JSON')
                    usage.add(turn)
                    return dict(stop=True, reason='Context-read allowance exhausted', proposals=[],
                                needs_context=['additional SDK/source reads require a new context']), turn
                content = wire.read(value['read'])
                if sum(len(item['content']) for item in messages) + len(content) > wire.max_chars:
                    usage.add(turn)
                    return dict(stop=True, reason='Requested context exceeds the prompt budget', proposals=[],
                                needs_context=['raise the prompt budget for the requested implementation contracts']), turn
                usage.add({**turn, 'kind': 'context_read'})
                if journal is not None:
                    journal.stage(stage)['read'] = value
                    write_json(journal.path, journal.state)
                payload = {**payload, 'messages': messages + [
                    {'role': 'assistant', 'content': compact(value)},
                    {'role': 'user', 'content': content + '\nReturn the final analysis JSON now; no further reads.'}]}
                if journal is not None:
                    messages = payload['messages']
                continue
            if wire is not None and not analysis:
                value = wire.decode(value, parent_analysis)
            if journal is not None:
                check_size(value, request)
            turn['ignored_keys'] = check_shape(value, request, analysis=analysis)
            if wire is not None and analysis:
                value = wire.normalize_analysis(value)
                if not value['stop']:
                    document = wire.coding(value, inline=journal is not None)
                    if journal is not None and len(document) > wire.max_chars:
                        raise ValueError('Selected coding context exceeds budget; choose one smaller complete mechanism and fewer files')
            if not analysis:
                validate_patches(value, request)
            usage.add(turn)
            if journal is not None:
                journal.complete(stage, value, reads=wire.reads() if wire is not None else None)
            return value, turn
        except Truncated as error:
            errors.append(str(error))
            usage.add(turn)
            if journal is not None:
                journal.rejected(stage, error, truncated=True)
                continue
            escalated = min(payload['max_tokens'] * 2, token_cap())
            if escalated <= payload['max_tokens']:
                break
            payload = {**payload, 'max_tokens': escalated, 'messages': messages + [
                {'role': 'user', 'content':
                 'Answer within a tighter budget: stop reasoning and reply with one JSON object '
                 f'only, holding exactly the keys {keys}.'}]}
        except (KeyError, IndexError, TypeError, ValueError) as error:
            errors.append(str(error))
            usage.add(turn)
            if journal is not None:
                journal.rejected(stage, error)
                continue
            payload = {**payload, 'messages': messages + [
                {'role': 'assistant', 'content': text},
                {'role': 'user', 'content':
                 f'That response was rejected: {error}. Reply with one JSON object and nothing else. '
                 f'It must contain exactly these keys: {keys}. '
                 f'The schema is: {json.dumps(schema)}'}]}
    if journal is not None:
        raise ModelExhausted(stage, journal.stage(stage)['hint'])
    raise ValueError(f'DeepSeek {stage} response rejected: {errors[-1]}')


def propose_compact(args, request, analysis_contract):
    journal = Journal(args.output.parent, request) if request.get('model_recovery') else None
    if journal is not None:
        seed_analysis(journal, request)
    wire = ModelContext(request)
    wire.max_chars = int(os.environ.get('AFD_RSI_DEEPSEEK_MAX_CHARS')
                         or request.get('max_context_bytes') or DEFAULT_MAX_CHARS)
    usage = UsageLog(args.output.parent / 'usage.json', resume=journal is not None)
    remaining = float(request.get('deadline') or 0) - time.time() if request.get('deadline') else float('inf')
    limit = min(float(request.get('agent_timeout_seconds') or 600), max(1.0, remaining))
    item = analysis_contract['properties']['proposals']['items']
    item['required'].append('files')
    item['properties']['files'] = dict(type='array', minItems=1, items=dict(type='string'))
    saved = args.session_dir / 'sdk-reads.json'
    if request.get('debug'):
        if saved.is_file():
            wire.restore_reads(read_json(saved))
        analysis = wire.debug_analysis()
    else:
        document = wire.initial()
        (args.output.parent / 'model-input.txt').write_text(document)
        if request.get('missing_context') or len(document) > wire.max_chars:
            write_json(args.output, dict(stop=True, reason='Required model context exceeds the prompt budget', proposals=[],
                needs_context=request.get('missing_context') or ['source/contract seed exceeds prompt budget']))
            write_json(args.output.parent / 'usage.json', {'turns': [], 'model_calls': 0})
            return
        analysis, _ = ask('analysis', [dict(role='system', content=SYSTEM_RULES + MODEL_RULES + READ_RULES),
            dict(role='user', content=ANALYSIS_TASK + '\n' + document + '\nSchema:' + compact(analysis_contract))],
            analysis_contract, request, timeout=limit, prefix=args.output.parent / 'analysis-events',
            analysis=True, usage=usage, wire=wire, journal=journal)
        write_json(args.output.parent / 'analysis.json', analysis)
        write_json(saved, wire.reads())
        if analysis['stop']:
            write_json(args.output, analysis)
            return
    document = wire.coding(analysis)
    if request.get('debug'):
        document += '\nFAILURE ' + compact(request['debug']['failure']) + '\nLOGS ' + compact(request.get('logs', []))
    (args.output.parent / 'model-code-input.txt').write_text(document)
    if len(document) > wire.max_chars:
        write_json(args.output, dict(stop=True, reason='Coding context exceeds prompt budget', proposals=[],
                                    needs_context=['selected source/SDK exceeds prompt budget']))
        if not usage.turns:
            write_json(args.output.parent / 'usage.json', {'turns': [], 'model_calls': 0})
        return
    contract = code_schema(wire.base, len(analysis['proposals']))
    code, _ = ask('code', [dict(role='system', content=SYSTEM_RULES + MODEL_RULES),
        dict(role='user', content=(DEBUG_EDIT_TASK if request.get('debug') else EDIT_TASK)
             + '\nReturn id/edits, not patches.\n' + document + '\nSchema:' + compact(contract))],
        contract, request, timeout=limit, prefix=args.output.parent / 'deepseek-events', analysis=False,
        usage=usage, wire=wire, parent_analysis=analysis, journal=journal)
    write_json(args.output, code)


def propose(args):
    request = json.loads(args.request.read_text())
    require_patch_tools(request)
    # Fail before any other work, naming the variable this host actually exports.
    api_key()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.session_dir.mkdir(parents=True, exist_ok=True)
    session = args.session_dir / 'session.json'
    campaign = {key: request.get(key) for key in ('fingerprint', 'deadline', 'provider_hash')}
    if session.exists() and read_json(session).get('campaign') != campaign:
        raise ValueError('DeepSeek session belongs to another campaign; use a separate session directory')
    write_json(session, {'campaign': campaign, 'provider': 'deepseek', 'base_url': endpoint(),
                         'models': {stage: model_for(stage, request) for stage in ('analysis', 'code')},
                         'effort': {stage: effort_for(stage) for stage in ('analysis', 'code')}})

    contract = json.loads(Path(__file__).with_name('proposal.schema.json').read_text())
    proposal = contract['properties']['proposals']['items']
    if request.get('proposal_requirements'):
        fields = request['proposal_requirements']['experiment_fields']
        proposal['required'].append('experiment')
        proposal['properties']['experiment'] = {
            'type': 'object', 'additionalProperties': False, 'required': fields,
            'properties': {name: {'type': 'string', 'minLength': 1} for name in fields}}
        if 'knowledge_refs' in request['proposal_requirements']:
            proposal['required'].append('direction')
            proposal['properties']['direction'] = direction_schema()
    proposal['properties']['evidence']['items']['enum'] = [item['id'] for item in request['evidence']]
    analysis_contract = json.loads(json.dumps(contract))
    analysis_item = analysis_contract['properties']['proposals']['items']
    analysis_item['required'].remove('patch')
    del analysis_item['properties']['patch']

    if request.get('model_protocol') == 2:
        return propose_compact(args, request, analysis_contract)

    budget = int(os.environ.get('AFD_RSI_DEEPSEEK_MAX_CHARS')
                 or request.get('max_context_bytes') or DEFAULT_MAX_CHARS)
    view, _ = truncate(request, budget)
    if view.get('missing_context'):
        write_json(args.output, dict(stop=True, reason='; '.join(view['missing_context']),
                                    proposals=[], needs_context=view['missing_context']))
        write_json(args.output.parent / 'usage.json', {'turns': [], 'model_calls': 0})
        return
    document = json.dumps(view, ensure_ascii=False)
    deadline = float(request.get('deadline') or 0)
    remaining = deadline - time.time() if deadline else float('inf')
    limit = min(float(request.get('agent_timeout_seconds') or 600), max(1.0, remaining))

    journal = Journal(args.output.parent, request) if request.get('model_recovery') else None
    usage = UsageLog(args.output.parent / 'usage.json', resume=journal is not None)
    if request.get('debug'):
        code, _ = ask('code',
            [{'role': 'system', 'content': SYSTEM_RULES},
             {'role': 'user', 'content': f'{DEBUG_TASK}\n\nSchema:\n{json.dumps(contract)}\n\nRequest:\n{document}'}],
            contract, request, timeout=limit, prefix=args.output.parent / 'deepseek-events',
            analysis=False, usage=usage, journal=journal)
        write_json(args.output, code)
        return
    analysis, first = ask('analysis',
        [{'role': 'system', 'content': SYSTEM_RULES},
         {'role': 'user', 'content': f'{ANALYSIS_TASK}\n\nSchema:\n{json.dumps(analysis_contract)}\n\nRequest:\n{document}'}],
        analysis_contract, request, timeout=limit, prefix=args.output.parent / 'analysis-events',
        analysis=True, usage=usage, journal=journal)
    write_json(args.output.parent / 'analysis.json', analysis)
    if analysis['stop']:
        args.output.write_text(json.dumps(analysis) + '\n')
    else:
        requirements = request.get('proposal_requirements')
        if requirements:
            for item in analysis['proposals']:
                experiment = item.get('experiment') or {}
                if (set(experiment) != set(requirements['experiment_fields'])
                        or not all(isinstance(v, str) and v.strip() for v in experiment.values())
                        or not set(requirements['required_evidence_ids']) <= set(item['evidence'])):
                    raise ValueError('analysis lacks a falsifiable experiment or required incumbent evidence; no coding call')
        code, second = ask('code',
            [{'role': 'system', 'content': SYSTEM_RULES},
             {'role': 'user', 'content': f'{CODE_TASK}\n\nSchema:\n{json.dumps(contract)}\n\nAnalysis:\n'
                                         f'{json.dumps(analysis, ensure_ascii=False)}\n\nRequest:\n{document}'}],
            contract, request, timeout=limit, prefix=args.output.parent / 'deepseek-events',
            analysis=False, usage=usage, journal=journal)
        args.output.write_text(json.dumps(code) + '\n')


def exclusive_lock(path: Path):
    """Hold an exclusive lock on the session directory, portably.

    The proposer host is often a developer workstation rather than the Linux
    execution host, so the lock must not assume a POSIX platform.
    """
    handle = path.open('a')
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise ValueError(f'another proposer holds {path}; refusing concurrent use') from error
    return handle


def probe(timeout: float = 30.0) -> tuple[list, bool]:
    """Verify the endpoint can actually serve the configured model.

    Nothing here can guarantee an external service stays up; the point is that a
    broken configuration is reported as a specific, actionable failure before a
    campaign spends any round on it. Each step is independent so the first
    failure names the one thing to change.
    """
    steps = []

    def step(name: str, ok: bool, detail: str, remedy: str = '') -> bool:
        steps.append({'step': name, 'ok': ok, 'detail': detail, 'remedy': remedy})
        return ok

    try:
        key_name, key, base = key_env(), api_key(), base_url()
    except ValueError as error:
        step('configuration', False, str(error),
             'export the named variable on the proposer host and set '
             'AFD_RSI_DEEPSEEK_BASE_URL to the gateway root')
        return steps, False
    step('configuration', True,
         f'endpoint {base}, key from {key_name} ({len(key)} chars), '
         f'models {model_for("analysis")} / {model_for("code")}, '
         f'effort {effort_for("analysis")} / {effort_for("code")}')

    try:
        listing = request_json(base + '/models', None, timeout)
    except RuntimeError as error:
        message = str(error)
        code = message.split('HTTP ')[1].split(':')[0] if 'HTTP ' in message else ''
        remedy = {
            '401': f'{key_name} was rejected by the gateway; refresh the credential',
            '403': f'{key_name} is valid but lacks access to this model',
            '404': 'the gateway has no /models route; check AFD_RSI_DEEPSEEK_BASE_URL '
                   'includes the API prefix such as /v1',
            '503': 'the gateway is reachable but reports no healthy upstream; this is '
                   'an availability problem, not a configuration one, so retry later',
        }.get(code, 'check that AFD_RSI_DEEPSEEK_BASE_URL points at an '
                    'OpenAI-compatible gateway and that the network allows it')
        step('endpoint', False, message[:300], remedy)
        return steps, False
    step('endpoint', True, f'{base}/models answered')

    served = [item.get('id') for item in (listing.get('data') or []) if isinstance(item, dict)]
    wanted = sorted({model_for('analysis'), model_for('code')})
    missing = [name for name in wanted if name not in served]
    if missing:
        step('model', False, f'gateway does not list {missing}; it serves {served[:12]}',
             'set AFD_RSI_DEEPSEEK_MODEL or the recipe [search.models] to one of the '
             'listed identifiers')
        return steps, False
    step('model', True, f'gateway lists {wanted}')

    payload = {'model': model_for('analysis'), 'max_tokens': 16, 'stream': False,
               'reasoning_effort': effort_for('analysis'),
               'messages': [{'role': 'user', 'content': 'Reply with the single word ok.'}]}
    try:
        answer = post(payload, timeout)
    except RuntimeError as error:
        step('completion', False, str(error)[:300],
             'the gateway accepts listing but not generation for this model; verify the '
             'model is loaded and that the key may run completions')
        return steps, False
    choices = answer.get('choices') or []
    text = ((choices[0].get('message') or {}).get('content') or '') if choices else ''
    if not choices or not text.strip():
        step('completion', False, f'no usable content in {json.dumps(answer)[:200]}',
             'the endpoint answered without a chat completion; a plain completion-only '
             'gateway needs /chat/completions support')
        return steps, False
    step('completion', True, f'generated {len(text)} chars')

    payload['response_format'] = {'type': 'json_object'}
    payload['messages'] = [{'role': 'user', 'content': 'Reply with {"ok": true} as JSON.'}]
    try:
        post(payload, timeout)
        step('json mode', True, 'accepted')
    except RuntimeError as error:
        step('json mode', False, str(error)[:200],
             'set AFD_RSI_DEEPSEEK_JSON_MODE=0; responses are still parsed and '
             'validated locally, so this only removes an optimisation')
    return steps, all(item['ok'] for item in steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true',
                        help='verify the endpoint, model and JSON mode, then exit')
    parser.add_argument('--session-dir', type=Path,
                        default=os.environ.get('AFD_RSI_AGENT_SESSION_DIR')
                        or os.environ.get('AFD_RSI_DEEPSEEK_SESSION_DIR'))
    args = parser.parse_args()
    if args.check:
        steps, ok = probe()
        for item in steps:
            print(f'{"ok  " if item["ok"] else "FAIL"} {item["step"]}: {item["detail"]}')
            if not item['ok'] and item['remedy']:
                print(f'     fix: {item["remedy"]}')
        if args.output:
            write_json(args.output.resolve(), {'ok': ok, 'steps': steps})
        raise SystemExit(0 if ok else 2)
    if not args.request or not args.output:
        parser.error('--request and --output are required unless --check is used')
    args.output = args.output.resolve()
    args.session_dir = (args.session_dir or args.output.parent / 'deepseek-session').resolve()
    args.session_dir.mkdir(parents=True, exist_ok=True)
    handle = exclusive_lock(args.session_dir / '.lock')
    try:
        propose(args)
    except ModelExhausted as error:
        response = failure_response(read_json(args.request), error)
        write_json(args.output.parent / 'model-failure.json', response)
        if not read_json(args.request).get('model_recovery'):
            raise
        write_json(args.output, response)
    finally:
        handle.close()


if __name__ == '__main__':
    main()
