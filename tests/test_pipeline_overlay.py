"""Representative legacy and expert-packed diagnostic injection contracts."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KERNEL = Path('csrc/npu/ascend_kernels/afd_mega_moe_ffn/op_kernel')


def sources(packed):
    header = '''class PipelineFlags {
public:
    __aicore__ explicit PipelineFlags(GM_ADDR base) : base_(base) {}
        WaitAt(Address(stage, 0), sequence);
        if ASCEND_IS_AIC {
            Fence<HardEvent::FIX_S>();
        // Leaders publish after their members;
    }
private:
    __aicore__ GM_ADDR Address
        transport.Receive(next);
        transport.Receive(next);
        transport.Send(tile);
    transport.Finish();
    flags.Wait(first ? PRE1 : PRE2, tile.sequence);
    flags.Arrive(first ? G1 : G2,
'''
    header += ('    flags.Wait(RECEIVED, tile.sequence);\n    flags.Arrive(PRE1,\n' if packed else
               '    if (first) flags.Wait(RECEIVED, tile.sequence);\n    flags.Arrive(first ? PRE1 : PRE2,\n')
    header += '        InvalidateScalarInputs(slot + w.tileCounts);\n'
    header += ('        // The rounded INT8 activation\n' if packed else '        // PRE2 reads the same row partition\n')
    header += '        OutputTile(slot,\n        flags.Arrive(OUTPUT,\n'
    cpp = '''        PipelineFlags flags(scratch + w.flags);
                rx.Process(false);
'''
    if packed:
        cpp += '''                const auto result = ValidateTransaction(scratch);
                StoreControl(scratch + w.validation, 0, result);
'''
    cpp += ('        const auto layer = ReadControl(scratch + w.validation, 0);\n' if packed else '        const auto layerId = 0;\n')
    cpp += '''        TileCursor cursor();
        SyncAll<false>();
        WriteCompletion(completion, 0, pipe);
'''
    return header, cpp


@pytest.mark.parametrize('packed', [False, True])
def test_injection_has_paired_entry_validation_prepare_and_activation(tmp_path, monkeypatch, packed):
    monkeypatch.syspath_prepend(str(ROOT / 'benchmarks/native'))
    from pipeline_trace import instrument
    target = tmp_path / KERNEL
    target.mkdir(parents=True)
    header, cpp = sources(packed)
    (target / 'afd_mega_moe_pipeline.h').write_text(header)
    (target / 'afd_mega_moe_ffn.cpp').write_text(cpp)
    instrument(tmp_path)
    updated = (target / 'afd_mega_moe_ffn.cpp').read_text()
    assert updated.index('const auto validation =') < updated.index('flags.trace.End(31, 0, validation)')
    assert updated.count('flags.trace.Begin(29, 0)') == updated.count('flags.trace.End(29, 0, entry)') == 1
    assert updated.count('flags.trace.Finish()') == 1
    if packed:
        assert updated.index('trace.Begin(31, 0)') < updated.index('ValidateTransaction(') < updated.index('trace.End(31, 0, validation)')
    updated = (target / 'afd_mega_moe_pipeline.h').read_text()
    assert updated.count('const auto prep =') == updated.count('tile.sequence, prep)') == 1
    assert updated.count('const auto activation =') == updated.count('tile.sequence, activation)') == 1


def test_missing_anchor_fails_before_writing_either_file(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'benchmarks/native'))
    from pipeline_trace import instrument
    target = tmp_path / KERNEL
    target.mkdir(parents=True)
    header, cpp = sources(True)
    cpp = cpp.replace('                StoreControl(scratch + w.validation, 0, result);', '                ChangedValidation();')
    h, c = target / 'afd_mega_moe_pipeline.h', target / 'afd_mega_moe_ffn.cpp'
    h.write_text(header)
    c.write_text(cpp)
    with pytest.raises(ValueError, match='unsupported MegaMoe diagnostic source'):
        instrument(tmp_path)
    assert h.read_text() == header and c.read_text() == cpp
