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


PULL = Path('csrc/npu/ascend_kernels/afd_mega_moe_pull_native/op_kernel')
PROTOCOL = Path('csrc/npu/ascend_kernels/utils/op_kernel/afd_pull_protocol.h')

PULL_SOURCE = '''#include "afd_pull_protocol.h"
#include "afd_mega_moe_pull_pipeline.h"

namespace AfdNativeMega {
template <class Plan>
__aicore__ inline void Compute(
    GM_ADDR scales, int64_t layer, bool second, Resource &resource) {
        if (second) {
            if (index == 0 || SyncExpert(index - 1, round.activeCount)) CrossCoreWaitFlag<2>(ACT_READY);
        } else {
            CrossCoreWaitFlag<2>(index / FLAG_SETS);
        }
}
template <class Plan>
__aicore__ inline void Process(GM_ADDR state, GM_ADDR scratch) {
    const Cam::NativeMegaWorkspace w(1, 2, 3, 4, 5);
    Resource resource;
    TPipe pipe;
    const uint32_t id = GetBlockIdx();
    do {
        if ASCEND_IS_AIV {
            if (id == 0) {
                Cam::Pull::Transport rx(state, PullShape(t), c[7], t.aivNum, pipe, 0);
                rx.PrepareRound(scratch + w.batch, scratch + w.counts, c[10]);
                const auto modelLayer = rx.Read(scratch + w.batch + 16);
                int64_t slot = -1;
                rx.Write(scratch + w.layer, slot);
            }
            pipe.Reset();
        }
        SyncAll<false>();
        const auto layer = Load(state, Cam::Pull::FAILURE) ? -1 : Load(scratch + w.layer);
        if (layer >= 0 && layer < c[13]) {
            const ReceiveRound round(scratch, w, c[6]);
            if (round.total) {
                if ASCEND_IS_AIC {
                    Compute<Plan>(scratch, w, t, round, w13, s13, layer, false, resource);
                    Compute<Plan>(scratch, w, t, round, w2, s2, layer, true, resource);
                }
                if ASCEND_IS_AIV {
                    PullAndPrepare<Plan>(scratch, w, t, round, resource);
                    Activate<Plan>(scratch, w, t, round, Plan::kPackedInt4 ? b13 : s13, layer, resource);
                    Combine<Plan>(scratch, w, t, round, Plan::kPackedInt4 ? b2 : s2, layer, resource);
                }
            }
        }
        SyncAll<false>();
        if ASCEND_IS_AIV { pipe.Reset(); }
        if ASCEND_IS_AIV {
            if (id == 0) {
                Cam::Pull::Transport tx(state, PullShape(t), c[7], t.aivNum, pipe, 0);
                if (layer >= 0 && layer < c[13]) tx.CompleteRound(scratch + w.batch, c[10]);
                tx.Write(completion, tx.ReadState(Cam::Pull::FAILURE));
            }
        }
        // All GMM/epilogue events have drained before scratch and flag slots are reused.
        SyncAll<false>();
        if ASCEND_IS_AIV { pipe.Reset(); }
    } while (!Load(state, Cam::Pull::FAILURE) && Load(state, Cam::Pull::SOURCE) >= 0);
}
}
'''

PULL_ACTIVATION = '''template <class Plan>
__aicore__ inline void FUNCTION(GM_ADDR scratch, const Cam::NativeMegaWorkspace &w,
    const Cam::AfdMegaMoePullNativeTilingData &t, const ReceiveRound &round, GM_ADDR biases,
    int64_t layer, Resource &resource) {
NATIVE_CALL
        CrossCoreWaitFlag<2>(GMM1_READY);
        SyncAll<true>();
        activation.Work();
        SyncAll<true>();
        CrossCoreSetFlag<2, PIPE_MTE3>(ACT_READY);
    activation.Finalize();
}
'''

PULL_PIPELINE = (PULL_ACTIVATION.replace('FUNCTION', 'ActivateNative').replace('NATIVE_CALL', '')
    + PULL_ACTIVATION.replace('FUNCTION', 'Activate').replace('NATIVE_CALL',
        '            ActivateNative<Plan>(scratch, w, t, round, biases, layer, resource);')
    + '''template <class Plan>
__aicore__ inline void Combine(GM_ADDR scratch, const Cam::NativeMegaWorkspace &w,
    const Cam::AfdMegaMoePullNativeTilingData &t, const ReceiveRound &round, GM_ADDR biases,
    int64_t layer, Resource &resource) {
    const uint32_t h = t.config[1];
            for (; sync <= index; ++sync) CrossCoreWaitFlag<2>(sync / FLAG_SETS);
                output.Tile(raw, bias[e * h], scales, row, rows, coord.n() * CUBE_N, h, t.routedScale);
                rowTx.Pump(ExpertRowTx::PUMP_ROWS);
            output.DrainLocal();
            SyncAll<true>();
            rowTx.Enqueue(groupBegin,groupRows,e);
            rowTx.Pump(ExpertRowTx::PUMP_ROWS);
    for (; sync < round.activeCount; ++sync) CrossCoreWaitFlag<2>(sync / FLAG_SETS);
    output.Finalize();
    rowTx.Finalize();
}
''')

PULL_PROTOCOL = '''class Transport {
    __aicore__ bool Select() {
                    if (commit < expected) { ready = false; break; }
                        Fail(source, -2); return false;
                        Fail(source, -3); return false;
                    if (Read(header + COMMIT * ALIGNMENT) != commit) { ready = false; break; }
                if (!ready) continue;
    }
    __aicore__ void PrepareRound(GM_ADDR batch, GM_ADDR groupCounts, uint64_t capacity) {
        const bool continuing = ReadState(SOURCE) >= 0;
        Select();
        if (continuing) LoadSnapshot(batch);
        else LoadMetadata();
        if (!ReadState(FAILURE)) {
            for (uint32_t s = 0; s < shape_.tp; ++s) {
            }
        }
        if (ReadState(FAILURE)) {
            ClearMetadata();
        }
        const uint64_t end = begin + capacity < total ? begin + capacity : total;
            groups.SetValue(e, group);
        }
        SaveMetadata(batch, groupCounts);
        Fence<AscendC::HardEvent::MTE3_S>();
    }

    // Called only by the control owner
};
'''


def pull_checkout(tmp_path, source=PULL_SOURCE):
    target = tmp_path / PULL
    target.mkdir(parents=True)
    (target / 'afd_mega_moe_pull_native.cpp').write_text(source)
    (target / 'afd_mega_moe_pull_pipeline.h').write_text(PULL_PIPELINE)
    protocol = tmp_path / PROTOCOL
    protocol.parent.mkdir(parents=True)
    protocol.write_text(PULL_PROTOCOL)
    return target / 'afd_mega_moe_pull_native.cpp'


def test_pull_native_overlay_names_every_stage_of_a_round(tmp_path, monkeypatch):
    """The pull kernel is instrumented per stage instead of being skipped."""
    monkeypatch.syspath_prepend(str(ROOT / 'benchmarks/native'))
    from pipeline_trace import PULL_STAGES, PULL_SUBSTAGES, eligible, instrument
    path = pull_checkout(tmp_path)
    assert eligible(tmp_path) == 'pull'
    instrument(tmp_path)
    text = path.read_text()
    protocol = (tmp_path / PROTOCOL).read_text()
    pipeline = (tmp_path / PULL / 'afd_mega_moe_pull_pipeline.h').read_text()
    combined = text + protocol + pipeline
    codes = {int(code) for code in __import__('re').findall(r'trace\.Begin\((\d+)', combined)}
    assert codes == set(PULL_STAGES) | set(PULL_SUBSTAGES)
    assert text.count('trace.Finish()') == 1 and text.count('++traceRound') == 1
    # The tail lives past the workspace the executor allocated.
    assert 'AfdTrace::Phase trace(scratch + w.bytes)' in text
    assert protocol.index('trace.Begin(60, traceRound)') < protocol.index('Select(trace, traceRound)')
    assert protocol.index('trace.Begin(76, traceRound)') < protocol.index('if (!ready) continue;') < protocol.index('trace.End(76, traceRound, sourceWait)')
    assert pipeline.index('trace.Begin(70, traceRound)') < pipeline.index('output.Tile(')
    assert pipeline.index('trace.Begin(71, traceRound)') < pipeline.index('rowTx.Pump(')
    assert text.index('trace.End(44, traceRound, traceGmm2)') < text.index('trace.Begin(48, traceRound)')


def test_pull_native_overlay_refuses_an_unexpected_layout(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'benchmarks/native'))
    from pipeline_trace import instrument
    source = PULL_SOURCE.replace('                    Activate<Plan>(scratch, w, t, round, Plan::kPackedInt4 ? b13 : s13, layer, resource);\n', '')
    path = pull_checkout(tmp_path, source)
    with pytest.raises(ValueError, match='unsupported MegaMoe diagnostic source'):
        instrument(tmp_path)
    assert 'AfdTrace' not in path.read_text()
    assert (tmp_path / PROTOCOL).read_text() == PULL_PROTOCOL
    assert (tmp_path / PULL / 'afd_mega_moe_pull_pipeline.h').read_text() == PULL_PIPELINE
