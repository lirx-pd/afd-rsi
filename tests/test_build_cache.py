"""Build reuse: a cache hit must be re-verified, never trusted.

Every campaign paying a full recompile for byte-identical source is the cost this
cache removes; the risk it adds is stale artifacts, so a hit re-hashes the tree
against the source and against its own `artifacts.json`.
"""

from pathlib import Path

import pytest

from afd_rsi import native
from afd_rsi.evidence import read_json
from afd_rsi.runner import execute as real_execute

ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS = {
    'afd_plugin/__init__.py': 'plugin',
    'afd_plugin/_C_ascend.so': 'binding',
    'afd_plugin/_cann_ops_custom/vendors/afd-plugin/op_api/lib/libcust_opapi.so': 'api',
    'afd_plugin/_cann_ops_custom/vendors/afd-plugin/op_impl/ai_core/tbe/kernel/'
    'ascend910_93/afd_mega_moe_pull_native/kernel.o': 'kernel',
}


def source_tree(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / 'setup.py').write_text('print("build")\n')
    (path / 'csrc').mkdir()
    (path / 'csrc/kernel.cpp').write_text('void k() {}\n')
    return path


@pytest.fixture
def fake_build(monkeypatch):
    calls = []

    def fake_execute(argv, cwd, output, timeout, env):
        calls.append(list(argv))
        for name, content in ARTIFACTS.items():
            target = Path(cwd) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        return {'status': 'completed', 'cleanup_complete': True, 'argv': list(argv)}

    monkeypatch.setattr(native, 'execute', fake_execute)
    return calls


def test_second_identical_build_reuses_the_cache(tmp_path, fake_build):
    cache = tmp_path / 'cache/build'
    source = source_tree(tmp_path / 'afd-plugin')

    first = native.build(source, tmp_path / 'run1/native', 60, cache_root=cache)
    assert len(fake_build) == 1, 'a cold build compiles once'
    assert read_json(tmp_path / 'run1/native/build.json')['status'] == 'completed'

    second = native.build(source, tmp_path / 'run2/native', 60, cache_root=cache)
    assert len(fake_build) == 1, 'a hit must not invoke the compiler'
    assert read_json(tmp_path / 'run2/native/build.json')['status'] == 'reused'
    assert read_json(tmp_path / 'run2/native/artifacts.json') == \
        read_json(tmp_path / 'run1/native/artifacts.json')
    assert Path(second, 'afd_plugin/_C_ascend.so').read_text() == 'binding'
    assert Path(first, 'afd_plugin/_C_ascend.so').read_text() == 'binding'


def test_changed_source_misses_the_cache(tmp_path, fake_build):
    cache = tmp_path / 'cache/build'
    source = source_tree(tmp_path / 'afd-plugin')
    native.build(source, tmp_path / 'run1/native', 60, cache_root=cache)

    (source / 'csrc/kernel.cpp').write_text('void k() { /* tuned */ }\n')
    native.build(source, tmp_path / 'run2/native', 60, cache_root=cache)
    assert len(fake_build) == 2, 'a changed kernel must be rebuilt'


def test_tampered_cache_entry_is_refused(tmp_path, fake_build):
    cache = tmp_path / 'cache/build'
    source = source_tree(tmp_path / 'afd-plugin')
    native.build(source, tmp_path / 'run1/native', 60, cache_root=cache)

    entry = next(path for path in cache.iterdir() if path.is_dir())
    target = entry / 'source/afd_plugin/_cann_ops_custom/vendors/afd-plugin/op_impl/ai_core/tbe/kernel/ascend910_93/afd_mega_moe_pull_native/kernel.o'
    target.write_text('swapped binary')

    with pytest.raises(ValueError, match='cached build artifacts changed'):
        native.build(source, tmp_path / 'run2/native', 60, cache_root=cache)
