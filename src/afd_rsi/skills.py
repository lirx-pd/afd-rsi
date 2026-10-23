"""Repository-local foundation validation and context assembly."""

import re
from pathlib import Path

from .evidence import digest, read_json


def files(root: Path) -> dict:
    result = {}
    for directory in ("skills", "knowledge", "benchmarks"):
        if (root / directory).is_symlink():
            raise ValueError(f"foundation directory symlink: {directory}")
        for path in sorted((root / directory).rglob("*")):
            if "__pycache__" in path.parts:
                continue
            if path.is_symlink():
                raise ValueError(f"foundation symlink: {path}")
            if path.is_file() and path.relative_to(root).as_posix() != "skills/manifest.json":
                result[path.relative_to(root).as_posix()] = digest(path)
    return result


def validate(root: Path) -> dict:
    root = root.resolve()
    manifest = read_json(root / "skills/manifest.json")
    actual = files(root)
    if manifest["files"] != actual:
        raise ValueError("foundation content differs from manifest; review the change before freezing")
    for name in actual:
        if not name.endswith(".md"):
            continue
        path = root / name
        body = path.read_text()
        if re.search(r"https?://|skill://|/home/|/mnt/|~[/\\]", body):
            raise ValueError(f"external knowledge dependency in {name}")
        for link in re.findall(r"\]\(([^)]+)\)", body):
            target = link.split("#", 1)[0]
            if not target:
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file():
                raise ValueError(f"missing or external resource: {name}: {link}")
            if resolved.relative_to(root).as_posix() not in actual:
                raise ValueError(f"resource not frozen: {name}: {link}")
    for skill in manifest["skills"]:
        body = (root / "skills" / skill / "SKILL.md").read_text()
        if not body.startswith("---\n") or f"name: {skill}\n" not in body:
            raise ValueError(f"invalid skill entry: {skill}")
    return manifest


def context(root: Path, skill: str) -> str:
    manifest = validate(root)
    if skill not in manifest["skills"]:
        raise ValueError(f"unknown skill: {skill}")
    pending = [root / "skills" / skill / "SKILL.md"]
    seen, parts = set(), []
    while pending:
        path = pending.pop(0).resolve()
        if path in seen:
            continue
        seen.add(path)
        body = path.read_text()
        parts.append(f"# {path.relative_to(root)}\n\n{body}")
        for link in re.findall(r"\]\(([^)]+)\)", body):
            target = link.split("#", 1)[0]
            if target and target.endswith((".md", ".h", ".cpp")):
                pending.append(path.parent / target)
    return "\n\n".join(parts)
