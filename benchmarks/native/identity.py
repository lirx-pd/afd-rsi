"""Actual process import/mapping evidence after native warmup."""

import hashlib
import os
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mapped_api_paths(root, maps):
    mapped, other = set(), set()
    for line in maps.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith("/"):
            path = Path(parts[5]).resolve()
            if path.name == "libcust_opapi.so":
                # Other vendors (e.g. FLA) legitimately use the same basename.
                if "afd-plugin" not in path.parts:
                    other.add(path)
                elif not path.is_relative_to(root):
                    raise RuntimeError(f"foreign custom op library mapped: {path}")
                else:
                    mapped.add(path)
    if not mapped:
        raise RuntimeError("native warmup did not map the custom op API library")
    return mapped, other


def loaded_identity():
    root = Path(os.environ["AFD_RSI_TARGET"]).resolve()
    package = Path(sys.modules["afd_plugin"].__file__).resolve()
    binding = Path(sys.modules["afd_plugin._C_ascend"].__file__).resolve()
    if not package.is_relative_to(root) or not binding.is_relative_to(root):
        raise RuntimeError("actual AFD import is outside measured source")
    mapped, other = mapped_api_paths(root, Path("/proc/self/maps").read_text())
    files = {str(p.relative_to(root)): sha(p) for p in (package, binding, *sorted(mapped))}
    kernel_dir = root / "afd_plugin/_cann_ops_custom/vendors/afd-plugin/op_impl/ai_core/tbe/kernel/ascend910_93/afd_mega_moe_ffn"
    kernels = {str(p.relative_to(root)): sha(p) for p in sorted(kernel_dir.glob("*")) if p.suffix in (".o", ".json")}
    if not kernels:
        raise RuntimeError("MegaMoe kernel build artifacts missing")
    return {"files": files, "kernel_artifacts": kernels, "pid": os.getpid(),
            "other_vendor_api_paths": [str(p) for p in sorted(other)],
            "note": "mapped binding/API and built device objects; kernel execution requires profiler evidence"}
