"""Read-only MC2 contracts from the isolated build's resolved include paths."""

import hashlib
import re
from pathlib import Path

from .evidence import read_json, write_json

PULL_KERNEL = 'csrc/npu/ascend_kernels/afd_mega_moe_pull_native/op_kernel'


def verify_context(bundle, source_hash, artifacts, *, origins=False):
    if bundle['source_hash'] != source_hash or bundle['artifacts'] != artifacts:
        raise ValueError('dependency context belongs to another build')
    for name, item in bundle['files'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('dependency context path escapes workspace')
        content = item['content'].encode()
        if hashlib.sha256(content).hexdigest() != item['sha256']:
            raise ValueError(f'dependency context changed: {name}')
        if origins and item['kind'] == 'installed':
            if Path(item['origin']).read_bytes() != content:
                raise ValueError(f'installed build dependency changed: {name}; rebuild without this cache')
    return bundle


def capture_context(repo, output, source_hash, artifacts):
    if not (repo / PULL_KERNEL).is_dir():
        return
    roots = {}
    variables = {'AFD_MC2_INCLUDE_DIR': 'mc2', 'AFD_MC2_92_DIR': 'mc2',
                 'AFD_CATLASS_92_DIR': 'catlass'}
    for cache in sorted(repo.rglob('CMakeCache.txt')):
        for line in cache.read_text().splitlines():
            match = re.match(r'(\w+):[^=]+=(.*)', line)
            if match and match[1] in variables and Path(match[2]).is_dir():
                key, path = variables[match[1]], Path(match[2]).resolve()
                if key in roots and roots[key] != path:
                    raise ValueError(f'ambiguous compiled dependency root: {key}')
                roots[key] = path
    if 'mc2' not in roots:
        raise ValueError('build has no resolved MC2 include path; cannot supply GMM contracts')
    generated = sorted(repo.rglob('afd_mega_moe_activation.h'))
    if not generated or len({p.read_bytes() for p in generated}) != 1:
        raise ValueError('missing or inconsistent generated activation contract')
    seeds = [roots['mc2'] / 'utils/block_mmad_w4a4.hpp', generated[0]]
    files, unresolved, seen = {}, set(), set()
    for path in seeds:
        if not path.is_file():
            raise ValueError(f'missing compiled dependency: {path}')
    queue = seeds[:]
    for path in queue:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        owner = next((key for key, root in roots.items() if path.is_relative_to(root)), None)
        name = f'{owner}/{path.relative_to(roots[owner]).as_posix()}' if owner else f'generated/{path.name}'
        content = path.read_bytes().decode('utf-8')
        files[name] = dict(content=content, sha256=hashlib.sha256(content.encode()).hexdigest(),
                           origin=str(path), kind='installed' if owner else 'generated')
        # Follow SDK/Catlass contracts, leaving generic compiler/AscendC headers explicit.
        for include in re.findall(r'^\s*#\s*include\s*[<"]([^>"\n]+)[>"]', content, re.MULTILINE):
            candidates = [path.parent / include, *(root / include for root in roots.values())]
            target = next((p.resolve() for p in candidates if p.is_file()), None)
            if target and any(target.is_relative_to(root) for root in roots.values()):
                queue.append(target)
            else:
                unresolved.add(include)
    bundle = dict(schema_version=1, source_hash=source_hash, artifacts=artifacts,
                  roots={key: str(path) for key, path in roots.items()}, files=files,
                  unresolved_includes=sorted(unresolved), read_only=True)
    verify_context(bundle, source_hash, artifacts)
    write_json(output, bundle)


def load_context(path, source_hash, artifacts, *, origins=False):
    return verify_context(read_json(path), source_hash, artifacts, origins=origins)
