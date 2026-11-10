import pytest

from afd_rsi.proposal import require_patch_tools, validate_patches


def response(patch):
    return dict(stop=False, proposals=[dict(patch=patch)])


def test_preflight_applies_to_exact_supplied_parent_and_enforces_allowed_files():
    request = dict(source_hash='frozen', allowed_files=['kernel.cpp'],
                   source_files={'kernel.cpp': 'old\n', 'protocol.h': 'contract\n'})
    patch = '--- a/kernel.cpp\n+++ b/kernel.cpp\n@@ -1 +1 @@\n-old\n+new\n'
    validate_patches(response(patch), request)
    assert request['source_files']['kernel.cpp'] == 'old\n'
    with pytest.raises(ValueError, match='does not apply'):
        validate_patches(response(patch.replace('-old', '-missing')), request)
    with pytest.raises(ValueError, match='outside allowed_files'):
        validate_patches(response(patch.replace('kernel.cpp', 'protocol.h').replace('-old', '-contract')), request)


def test_preflight_rejects_source_path_escape(tmp_path):
    request = dict(source_hash='frozen', allowed_files=['*'], source_files={str(tmp_path / 'escaped'): 'bad'})
    with pytest.raises(ValueError, match='escapes'):
        validate_patches(response(''), request)
    assert not (tmp_path / 'escaped').exists()


def test_missing_git_is_detected_before_paid_calls(monkeypatch):
    monkeypatch.setattr('afd_rsi.proposal.shutil.which', lambda _: None)
    with pytest.raises(ValueError, match='no model call was made'):
        require_patch_tools(dict(source_hash='parent'))
