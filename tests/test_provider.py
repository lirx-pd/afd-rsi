"""Proposer provider selection: recipe contract, frozen adapter identity, switching.

The contract cases are portable. The controller ones need the POSIX controller
(``fcntl``) and are skipped elsewhere rather than pretended.
"""

import json

import pytest

from afd_rsi import contracts
from afd_rsi.evidence import digest, read_json, write_json

ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]
HEAD = ('name = "r"\nbackend = "cpu-synthetic"\nsource = "source"\n'
        'workload_id = "w"\nscope = "s"\nunit = "us"\nmetric = "m"\n'
        'ranks = 1\nprimary_rank = 0\nrepetitions = 1\n'
        'timeout_seconds = 10\nbudget_seconds = 10\nmin_gain_pct = 1\n'
        'allowed_files = ["kernel.json"]\nartifacts = ["kernel.json"]\n'
        '[commands]\nbuild = ["{python}", "-c", "pass"]\n'
        'correctness = ["{python}", "{foundation}/benchmarks/x.py"]\n'
        'benchmark = ["{python}", "{foundation}/benchmarks/x.py"]\n'
        '[search]\n')
TAIL = ('max_rounds = 2\nmax_calls = 2\nagent_timeout_seconds = 10\n'
        'max_context_bytes = 8192\nexperience_dirs = []\n')


def recipe(tmp_path, search_body, tail=''):
    """Write a recipe; ``tail`` holds later tables such as [search.models]."""
    source = tmp_path / 'source'
    source.mkdir(exist_ok=True)
    (source / 'kernel.json').write_text('{}')
    path = tmp_path / 'recipe.toml'
    path.write_text(HEAD + search_body + TAIL + tail)
    return path


def test_provider_names_the_frozen_adapter_and_fills_the_command(tmp_path):
    config = contracts.load_recipe(recipe(tmp_path, 'provider = "deepseek"\n'))
    command = config['search']['agent_command']
    assert command == ['{python}', '{foundation}/benchmarks/agents/deepseek_agent.py',
                       '--request', '{request}', '--output', '{response}']
    assert contracts.agent_driver(command) == 'benchmarks/agents/deepseek_agent.py'


def test_explicit_command_stays_supported_and_is_derived(tmp_path):
    body = ('agent_command = ["{python}", "{foundation}/benchmarks/agents/codex_agent.py",'
            ' "--request", "{request}", "--output", "{response}"]\n')
    config = contracts.load_recipe(recipe(tmp_path, body))
    assert 'provider' not in config['search'], 'legacy recipes keep their plan shape'
    assert contracts.agent_driver(config['search']['agent_command']) == \
        'benchmarks/agents/codex_agent.py'


def test_provider_and_command_must_agree(tmp_path):
    body = ('provider = "deepseek"\n'
            'agent_command = ["{python}", "{foundation}/benchmarks/agents/codex_agent.py",'
            ' "--request", "{request}", "--output", "{response}"]\n')
    with pytest.raises(ValueError, match='disagree'):
        contracts.load_recipe(recipe(tmp_path, body))


@pytest.mark.parametrize('body, message', [
    ('provider = "openai"\n', 'provider must be one of'),
    ('', 'agent_command or provider'),
    ('agent_command = ["{python}", "{foundation}/benchmarks/x.py"]\n', 'benchmark/agents driver'),
    ('agent_command = ["{python}", "{foundation}/benchmarks/agents/../x.py"]\n', 'escapes foundation'),
])
def test_invalid_provider_settings_are_refused(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        contracts.load_recipe(recipe(tmp_path, body))


def test_declared_provider_changes_the_experience_fingerprint():
    try:
        from afd_rsi import search
    except ImportError as error:                     # pragma: no cover - non-POSIX hosts
        pytest.skip(f'the campaign controller needs a POSIX host: {error}')
    base = {'search': {}, 'backend': 'cpu-synthetic'}
    legacy = search.fingerprint(base)
    assert legacy == search.fingerprint({'search': {}, 'backend': 'cpu-synthetic'})
    codex = search.fingerprint({'search': {'provider': 'codex'}, 'backend': 'cpu-synthetic'})
    deepseek = search.fingerprint({'search': {'provider': 'deepseek'}, 'backend': 'cpu-synthetic'})
    assert len({legacy, codex, deepseek}) == 3, \
        'directions ranked from one model must not reuse another model history'


def _controller():
    try:
        from afd_rsi import campaign, search
    except ImportError as error:                     # pragma: no cover - non-POSIX hosts
        pytest.skip(f'the campaign controller needs a POSIX host: {error}')
    return campaign, search


def test_per_stage_models_are_frozen_with_defaults(tmp_path):
    # A declared table defaults the stage it does not name, so a recipe can swap
    # one stage's model later without restating the other.
    config = contracts.load_recipe(recipe(
        tmp_path, 'provider = "deepseek"\n',
        '\n[search.models]\nanalysis = "new-analyst"\n'))
    assert config['search']['models'] == {'analysis': 'new-analyst',
                                         'code': 'deepseek-v4.1-flash'}


def test_per_stage_models_are_rejected_where_the_adapter_owns_the_policy(tmp_path):
    with pytest.raises(ValueError, match='only accepted for a provider'):
        contracts.load_recipe(recipe(tmp_path, 'provider = "codex"\n',
                                     '\n[search.models]\nanalysis = "x"\n'))


@pytest.mark.parametrize('models, message', [
    ('analysis = ""', 'nonempty string'),
    ('planning = "x"', 'must name any of'),
])
def test_invalid_stage_model_settings_are_refused(tmp_path, models, message):
    with pytest.raises(ValueError, match=message):
        contracts.load_recipe(recipe(tmp_path, 'provider = "deepseek"\n',
                                     f'\n[search.models]\n{models}\n'))


def test_declared_models_change_the_experience_fingerprint():
    try:
        from afd_rsi import search
    except ImportError as error:                     # pragma: no cover - non-POSIX hosts
        pytest.skip(f'the campaign controller needs a POSIX host: {error}')
    base = {'backend': 'cpu-synthetic'}
    plain = search.fingerprint({'search': {'provider': 'deepseek'}, **base})
    analysis_swap = search.fingerprint(
        {'search': {'provider': 'deepseek',
                    'models': {'analysis': 'new-analyst', 'code': 'deepseek-v4.1-flash'}}, **base})
    code_swap = search.fingerprint(
        {'search': {'provider': 'deepseek',
                    'models': {'analysis': 'deepseek-v4.1-flash', 'code': 'new-coder'}}, **base})
    assert len({plain, analysis_swap, code_swap}) == 3, \
        'a different model per stage is a different method'


def test_request_carries_the_frozen_stage_models(tmp_path):
    campaign, search = _controller()
    path = _controller_recipe(tmp_path)
    text = path.read_text() + '\n[search.models]\nanalysis = "new-analyst"\n'
    path.write_text(text)
    run = campaign.prepare(path, tmp_path / 'run', ROOT)
    search.search(run, ROOT, external=True)
    plan = campaign.load(run, ROOT)
    best = read_json(run / 'search/best.json')
    context = search.request_context(run, plan, best, [], 1)
    assert context['models'] == {'analysis': 'new-analyst', 'code': 'deepseek-v4.1-flash'}


def test_request_without_declared_models_stays_unchanged(tmp_path):
    campaign, search = _controller()
    path = _controller_recipe(tmp_path)
    run = campaign.prepare(path, tmp_path / 'run', ROOT)
    search.search(run, ROOT, external=True)
    plan = campaign.load(run, ROOT)
    best = read_json(run / 'search/best.json')
    assert 'models' not in search.request_context(run, plan, best, [], 1)


def _controller_recipe(tmp_path, provider='deepseek'):
    """A real synthetic recipe whose proposer is the requested provider."""
    text = (ROOT / 'recipes/cpu_search.toml').read_text()
    text = text.replace('../examples/source', str(ROOT / 'examples/source'))
    text = text.replace('repetitions = 4', 'repetitions = 1')
    text = text.replace('max_rounds = 9', 'max_rounds = 2').replace('max_calls = 9', 'max_calls = 2')
    text = text.replace('provider = "codex"', f'provider = "{provider}"')
    text = text.replace('benchmarks/agents/codex_agent.py', f'benchmarks/agents/{provider}_agent.py')
    path = tmp_path / f'{provider}.toml'
    path.write_text(text)
    return path


def test_request_records_the_selected_adapter_identity(tmp_path):
    campaign, search = _controller()
    run = campaign.prepare(_controller_recipe(tmp_path), tmp_path / 'run', ROOT)
    search.search(run, ROOT, external=True)
    plan = campaign.load(run, ROOT)
    best = read_json(run / 'search/best.json')
    context = search.request_context(run, plan, best, [], 1)
    assert context['provider'] == 'benchmarks/agents/deepseek_agent.py'
    assert context['provider_hash'] == digest(run / 'foundation/benchmarks/agents/deepseek_agent.py')


def test_propose_refuses_a_proposer_that_differs_from_the_frozen_adapter(tmp_path, monkeypatch):
    campaign, search = _controller()
    run = campaign.prepare(_controller_recipe(tmp_path), tmp_path / 'run', ROOT)
    search.search(run, ROOT, external=True)
    plan = campaign.load(run, ROOT)
    best = read_json(run / 'search/best.json')
    request = tmp_path / 'request.json'
    write_json(request, search.request_context(run, plan, best, [], 1))

    # A DeepSeek request must not be served by the Codex adapter.
    tampered = read_json(request)
    tampered['provider'] = 'benchmarks/agents/codex_agent.py'
    write_json(request, tampered)
    with pytest.raises(ValueError, match='differs from the frozen campaign adapter'):
        search.propose(request, tmp_path / 'out', ROOT)

    # With the frozen identity restored, the adapter is located again and the
    # missing credential is the only remaining failure, which proves the request
    # routed to the DeepSeek adapter rather than to Codex.
    tampered['provider'] = 'benchmarks/agents/deepseek_agent.py'
    write_json(request, tampered)
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    monkeypatch.delenv('AFD_RSI_AGENT_SESSION_DIR', raising=False)
    with pytest.raises(ValueError, match='deepseek_agent proposal failed'):
        search.propose(request, tmp_path / 'out2', ROOT)
    assert 'DEEPSEEK_API_KEY' in (tmp_path / 'out2/stderr.log').read_text()


def test_session_env_reaches_both_adapters(tmp_path):
    _, search = _controller()
    env = search.session_env(tmp_path / 'session')
    assert env['AFD_RSI_AGENT_SESSION_DIR'] == env['AFD_RSI_CODEX_SESSION_DIR']
    assert env['AFD_RSI_AGENT_SESSION_DIR'] == str((tmp_path / 'session').resolve())
