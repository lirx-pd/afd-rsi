import copy
import json

import pytest

from afd_rsi.model_protocol import ModelContext, compact
from afd_rsi.proposal import validate_patches
from afd_rsi.search import validate_proposal


def request():
    return dict(model_protocol=2, source_hash='parent', allowed_files=['*.cpp', 'inc/*.h'],
        source_files={'kernel.cpp': 'int x = 1;\nreturn x;\n', 'protocol.h': '// readonly\n'},
        read_only_source_files=['protocol.h'], knowledge='# Completion\nNever recycle live buffers.\n',
        dependency_context={'files': {
            'catlass/large.hpp': {'content': 'large_unused_template\n' * 1000},
            'mc2/mm.hpp': {'content': 'void Finalize();\n'}}},
        evidence=[dict(id='baseline', facts={}), dict(id='profile-incumbent', facts=dict(pipeline=dict(ffn_ranks=[
            dict(overlap={'hidden_cycles': 7}, selected_cores=[dict(core=0, lifetime_cycles=30,
                 events=[[1, 2, 40]] * 128)])])))])


def analysis():
    return dict(stop=False, reason='fixture', needs_context=[], proposals=[dict(hypothesis='reduce copies',
        category='memory', evidence=['baseline'], priority=3, files=['s0'])])


def response(wire, edits):
    return dict(base=wire.base, stop=False, reason='implemented', needs_context=[],
                proposals=[dict(id=0, edits=edits)])


def test_model_view_keeps_source_contracts_overlap_and_retrievable_raw_evidence():
    source = request()
    wire = ModelContext(source)
    text = wire.initial()
    assert 'int x = 1;' in text and 'void Finalize();' in text
    assert 'large_unused_template' not in text
    assert 'hidden_cycles' in text and 'lifetime_cycles' in text and '"events":' not in text
    assert '"events"' in wire.read([['e1', 1, 20]])
    assert source['evidence'][1]['facts']['pipeline']['ffn_ranks'][0]['selected_cores'][0]['events']
    assert 'large_unused_template' in wire.read([['d0', 20, 5]])
    code = wire.coding(analysis())
    assert 'large_unused_template' in code and 'void Finalize();' not in code
    assert len(text) < len(compact(source)) / 2


@pytest.mark.parametrize('new', ['int x = 2;\nreturn x;', 'int x = 2;\nreturn x;\n'])
def test_exact_edits_restore_canonical_metadata_and_applicable_diff(new):
    source = request()
    wire = ModelContext(source)
    plan = analysis()
    plan['proposals'][0]['notes'] = 'model-added commentary'
    value = wire.decode(response(wire, [dict(file='s0', old=source['source_files']['kernel.cpp'], new=new)]), plan)
    validate_proposal(value, {'baseline'})
    validate_patches(value, source)
    assert value['proposals'][0]['hypothesis'] == plan['proposals'][0]['hypothesis']
    assert 'files' not in value['proposals'][0]
    assert 'notes' not in value['proposals'][0]


def test_exact_edits_can_create_and_delete_allowed_files():
    source = request()
    wire = ModelContext(source)
    value = wire.decode(response(wire, [dict(file='new:inc/new file.h', old='', new='// new\n'),
        dict(file='s0', old=source['source_files']['kernel.cpp'], new=None)]), analysis())
    validate_patches(value, source)
    assert '/dev/null' in value['proposals'][0]['patch']


def test_empty_files_and_overlapping_preimage_matches():
    source = request()
    source['source_files']['kernel.cpp'] = ''
    wire = ModelContext(source)
    for text in (None, 'new\n'):
        validate_patches(wire.decode(response(wire, [dict(file='s0', old='', new=text)]), analysis()), source)
    validate_patches(wire.decode(response(wire, [dict(file='new:inc/empty.h', old='', new='')]), analysis()), source)
    source['source_files']['kernel.cpp'] = 'ababa'
    wire = ModelContext(source)
    with pytest.raises(ValueError, match='exactly once'):
        wire.decode(response(wire, [dict(file='s0', old='aba', new='x')]), analysis())


@pytest.mark.parametrize('edits,error', [
    ([dict(file='s0', old='missing', new='x')], 'exactly once'),
    ([dict(file='s1', old='// readonly', new='bad')], 'read-only'),
    ([dict(file='d0', old='large_unused_template', new='bad')], 'read-only'),
    ([dict(file='new:../escape.cpp', old='', new='bad')], 'invalid edit path'),
    ([dict(file='new:C:/escape.cpp', old='', new='bad')], 'invalid edit path'),
    ([dict(file='new:other.txt', old='', new='bad')], 'outside allowed_files'),
    ([dict(file='s0', old='return x;', new='return 1;'), dict(file='s0', old='x;', new='x+1;')], 'overlapping'),
])
def test_unsafe_or_ambiguous_edits_never_become_patches(edits, error):
    with pytest.raises(ValueError, match=error):
        ModelContext(request()).decode(response(ModelContext(request()), edits), analysis())


def test_repeated_preimage_stale_base_and_duplicate_proposal_are_rejected():
    source = request()
    source['source_files']['kernel.cpp'] = 'x\nx\n'
    wire = ModelContext(source)
    value = response(wire, [dict(file='s0', old='x', new='y')])
    with pytest.raises(ValueError, match='exactly once'):
        wire.decode(value, analysis())
    value['base'] = 'old'
    with pytest.raises(ValueError, match='base'):
        wire.decode(value, analysis())
    value['base'] = wire.base
    value['proposals'][0]['edits'][0]['old'] = 'x\nx\n'
    value['proposals'].append(copy.deepcopy(value['proposals'][0]))
    with pytest.raises(ValueError, match='duplicate'):
        wire.decode(value, analysis())


@pytest.mark.parametrize('ranges', [[['d0', 1, 401]], [['d0', 1001, 1]], [['unknown', 1, 1]],
                                  [['d0', 1, 1]] * 9])
def test_context_reads_are_bounded(ranges):
    wire = ModelContext(request())
    with pytest.raises(ValueError):
        wire.read(ranges)
    assert not wire.loaded


def test_context_read_clamps_at_eof_instead_of_spending_a_repair():
    wire = ModelContext(request())
    text = wire.read([['d0', 999, 400]])
    assert 'L999-1000' in text and text.count('large_unused_template') == 2


def test_debug_reuses_only_source_and_sdk_bound_reads():
    source = request()
    wire = ModelContext(source)
    wire.read([['d0', 2, 4]])
    saved = wire.reads()
    restored = ModelContext(source)
    restored.restore_reads(saved)
    assert restored.loaded == wire.loaded
    saved['files']['d0']['sha'] = 'foreign'
    with pytest.raises(ValueError, match='another build'):
        restored.restore_reads(saved)
