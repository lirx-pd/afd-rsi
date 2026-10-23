"""Maintainer-only manifest generation; never called by candidate evaluation."""

from pathlib import Path

from afd_rsi.evidence import write_json
from afd_rsi.skills import files, validate

root = Path(__file__).resolve().parents[1]
write_json(root / "skills/manifest.json", {
    "schema_version": 1,
    "scope": "A3 MegaMoe local workflow; hardware performance unqualified",
    "skills": sorted(p.parent.name for p in (root / "skills").glob("*/SKILL.md")),
    "files": files(root),
})
validate(root)
print("Foundation frozen and internal resource closure validated")
