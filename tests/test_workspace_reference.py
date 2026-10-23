import shutil
import subprocess
from pathlib import Path

import pytest


def test_workspace_reference_nonoverlap_and_alignment(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable; no device claim")
    reference = Path(__file__).resolve().parents[1] / "knowledge/reference-code"
    source = tmp_path / "check.cpp"
    source.write_text('''#include <cassert>
#include <initializer_list>
#include "mega_moe_workspace.h"
int main() {
  for (unsigned capacity : {1u, 127u, 128u, 129u, 1024u}) {
    Cam::MegaWorkspace w(capacity, 256, 512, 4, 2);
    assert(w.x == 0);
    assert(w.scales >= capacity * 256);
    assert(w.batch >= w.scales + capacity * sizeof(float));
    assert(w.counts >= w.batch + (5 + 2 + 4*2) * sizeof(int64_t));
    assert(w.y >= w.counts + 4 * sizeof(int64_t));
    assert(w.tileX >= w.y + capacity * 256 * sizeof(uint16_t));
    assert(w.bytes > w.compute && w.compute > w.zScales);
    assert(w.bytes % Cam::MEGA_ALIGNMENT == 0);
  }
}
''')
    subprocess.run([compiler, "-std=c++17", "-I", str(reference), str(source), "-o", str(tmp_path / "check")], check=True)
    subprocess.run([str(tmp_path / "check")], check=True)
