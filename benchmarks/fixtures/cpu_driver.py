"""Synthetic contract fixture. Values are input data, NOT measured performance."""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

build = Path(os.environ["AFD_RSI_BUILD_DIR"])
artifact = build / "kernel.json"
if sys.argv[1] == "build":
    shutil.copy2("kernel.json", artifact)
else:
    spec = json.loads(artifact.read_text())
    checks = {"affine_output": all(spec["multiplier"] * x + spec["bias"] == 2 * x + 1
                                   for x in (-3, 0, 1, 9))}
    result = {"backend": "cpu-synthetic", "workload_id": "cpu-fixture-v1",
              "scope": "synthetic-controller-only", "unit": "synthetic_us",
              "metric": "synthetic_latency", "ranks": []}
    for rank in range(2):
        result["ranks"].append({"rank": rank, "source_hash": os.environ["AFD_RSI_SOURCE_HASH"],
                                "artifacts": {"kernel.json": hashlib.sha256(artifact.read_bytes()).hexdigest()},
                                "checks": checks, "value": spec["synthetic_us"]})
    Path(os.environ["AFD_RSI_RESULT"]).write_text(json.dumps(result, allow_nan=False))
