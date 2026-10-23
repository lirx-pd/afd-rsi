"""Content identity includes new files, deletions and executable modes."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from .evidence import digest


def inventory(root: Path) -> dict:
    root = root.resolve()
    git_root = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True,
    )
    if git_root.returncode == 0 and Path(git_root.stdout.strip()).resolve() == root:
        names = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
        ).decode().split("\0")
        paths = [root / name for name in names if name]
    else:
        paths = [p for p in root.rglob("*") if not {".git", "__pycache__"}.intersection(p.relative_to(root).parts)]
    files = {}
    for path in sorted(set(paths)):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root):
            raise ValueError(f"source symlink is not supported: {relative}")
        if path.is_file():
            files[relative] = {"sha256": digest(path), "executable": bool(path.stat().st_mode & 0o111)}
        elif path.exists() and not path.is_dir():
            raise ValueError(f"non-regular source: {relative}")
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"hash": identity, "files": files}


def copy_source(source: Path, destination: Path, expected: dict) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for name in expected["files"]:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    if inventory(destination) != expected or inventory(source) != expected:
        raise ValueError("source changed while creating snapshot")


def changed_files(before: dict, after: dict) -> set[str]:
    return {name for name in before["files"].keys() | after["files"].keys()
            if before["files"].get(name) != after["files"].get(name)}
