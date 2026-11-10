"""Local patch checks shared by the proposer adapters; no target checkout access."""

import fnmatch
import shutil
import subprocess
import tempfile
from pathlib import Path


DEBUG_TASK = (
    'Repair only the failed implementation described in request.debug. Do not analyze a new direction. '
    'Copy the original hypothesis, category and experiment verbatim; return at most one proposal. '
    'Return a complete replacement patch against request.source_files (the original parent), '
    'not an incremental patch against the failed candidate. Do not weaken validation or change gates. '
    'If the evidence does not support a safe repair, return stop=true with the exact reason.')


def require_patch_tools(request):
    if request.get('source_hash') and not shutil.which('git'):
        raise ValueError('Git is required on the proposer host for local patch preflight; no model call was made')


def validate_patches(value, request):
    """Apply in disposable copies, before spending a build or device trial."""
    if value.get('stop') or not request.get('source_hash'):
        return
    files = request.get('source_files', {})
    if not files:
        raise ValueError('patch preflight requires the supplied parent source')
    for proposal in value['proposals']:
        with tempfile.TemporaryDirectory(prefix='afd-rsi-patch-') as directory:
            source = Path(directory)
            for name, content in files.items():
                path = source / name
                if Path(name).is_absolute() or not path.resolve().is_relative_to(source):
                    raise ValueError('source context path escapes patch workspace')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content.encode())
            result = subprocess.run(
                ['git', '--git-dir=' + str(source / '.git'), '--work-tree=' + directory,
                 'apply', '--recount', '-'], input=proposal['patch'].encode(),
                cwd=source, capture_output=True, timeout=30)
            if result.returncode:
                raise ValueError('patch does not apply to supplied parent source: ' + result.stderr.decode(errors='replace')[-4000:])
            after = {}
            for path in source.rglob('*'):
                if path.is_symlink():
                    raise ValueError('patch creates a symlink')
                if path.is_file():
                    after[path.relative_to(source).as_posix()] = path.read_bytes()
            before = {name: content.encode() for name, content in files.items()}
            changed = [name for name in before.keys() | after.keys() if before.get(name) != after.get(name)]
            if not changed or any(not any(fnmatch.fnmatchcase(name, pattern)
                                          for pattern in request['allowed_files']) for name in changed):
                raise ValueError('patch has no content change or changes files outside allowed_files')
