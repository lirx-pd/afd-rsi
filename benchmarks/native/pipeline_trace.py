"""A3-only diagnostic overlay; never modify the production build.

Two kernel layouts exist. Older builds fuse the FFN into ``afd_mega_moe_ffn``;
the pull-native build splits it into ``afd_mega_moe_pull_native`` plus
``afd_mega_moe_comm``. Each layout has its own anchors, and both write the same
per-core record format so one decoder serves them.
"""
from pathlib import Path
import json

TRACE_CORES = 72
# One record per stage per receive round. Long sequences produce many rounds, so
# the tail must hold a whole suite: 2048 records x 32 bytes plus a 256-byte header.
TRACE_RECORDS = 2048
TRACE_STRIDE = 256 + TRACE_RECORDS * 32
TRACE_MAGIC = 0x4d4f455452414345

LEGACY_DIR = Path('csrc/npu/ascend_kernels/afd_mega_moe_ffn/op_kernel')
LEGACY_HEADER = 'afd_mega_moe_pipeline.h'
LEGACY_CPP = 'afd_mega_moe_ffn.cpp'
PULL_DIR = Path('csrc/npu/ascend_kernels/afd_mega_moe_pull_native/op_kernel')
PULL_CPP = 'afd_mega_moe_pull_native.cpp'
PULL_HEADER = 'afd_mega_moe_pull_pipeline.h'

# Codes recorded by the pull-native overlay. They name the stages a scheduler
# decision can actually move, so a hypothesis can point at one of them.
PULL_STAGES = {
    40: 'round_prepare',
    41: 'pull_rows',
    42: 'gmm1',
    43: 'activation',
    44: 'gmm2',
    45: 'combine',
    46: 'round_complete',
    47: 'sync_in',
    48: 'sync_out',
}


def eligible(repo):
    """Which overlay this checkout supports: 'pull', 'legacy' or None."""
    repo = Path(repo)
    pull = repo / PULL_DIR
    if (pull / PULL_CPP).is_file() and (pull / PULL_HEADER).is_file():
        return 'pull'
    legacy = repo / LEGACY_DIR
    if (legacy / LEGACY_HEADER).is_file() and (legacy / LEGACY_CPP).is_file():
        return 'legacy'
    return None


def supported(repo) -> bool:
    """Whether a per-stage overlay exists for this checkout."""
    return eligible(repo) is not None


def replace_checked(source, old, new, count=1):
    if source.count(old) != count:
        raise ValueError(f'unsupported MegaMoe diagnostic source: expected {count} occurrences of {old!r}')
    return source.replace(old, new)


def instrument(repo):
    mode = eligible(repo)
    if mode == 'pull':
        return instrument_pull(repo)
    if mode == 'legacy':
        return instrument_legacy(repo)
    raise ValueError(
        'no per-stage pipeline anchor exists for this checkout: neither '
        f'{PULL_DIR / PULL_CPP} nor {LEGACY_DIR / LEGACY_CPP} is present. '
        'Run the profile stage without --pipeline-trace-dir.')


PULL_TRACE_CLASS = r'''
namespace AfdTrace {
// The class is inserted before the file's own namespace, so it brings AscendC in.
using namespace AscendC;
// Diagnostic contract: the caller owns 72 extended storage tails, __RECORDS__ records/core.
class Phase {
public:
    __aicore__ Phase(GM_ADDR tail) : tail_(tail) {
        uint32_t core = GetBlockIdx();
        if ASCEND_IS_AIC { core += 48; }
        enabled_ = core == 0 || core == 4 || core == 8 || core == 28 || core == 47 ||
                   core == 48 || core == 60 || core == 71;
        tail_ += core * __STRIDE__;
        // The tail outlives one kernel launch: the fixture drives several
        // transactions and each launch reuses the same storage, so records
        // accumulate and the header keeps the earliest entry and the total count.
        GlobalTensor<int64_t> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(tail_));
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(gm);
        const auto seen = *reinterpret_cast<__gm__ volatile uint64_t *>(tail_);
        const auto total = *reinterpret_cast<__gm__ volatile uint64_t *>(tail_ + 8);
        const auto entry = *reinterpret_cast<__gm__ volatile uint64_t *>(tail_ + 16);
        const auto now = GetSystemCycle();
        count_ = (seen == 0x4d4f455452414345ULL && total <= __RECORDS__) ? uint32_t(total) : 0;
        start_ = count_ ? entry : now;
    }
    __aicore__ uint64_t Begin(uint32_t code, uint32_t seq) const {
        return enabled_ ? GetSystemCycle() : 0;
    }
    __aicore__ void End(uint32_t code, uint32_t seq, uint64_t start) const {
        if (!enabled_) return;
        const auto stop = GetSystemCycle();
        if (count_ < __RECORDS__) {
            auto dst = reinterpret_cast<volatile __gm__ uint64_t *>(tail_ + 256 + count_ * 32);
            dst[0] = start; dst[1] = stop; dst[2] = code; dst[3] = seq;
        }
        ++count_;
    }
    __aicore__ void Finish() const {
        const auto stop = GetSystemCycle();
        auto dst = reinterpret_cast<volatile __gm__ uint64_t *>(tail_);
        dst[0] = 0x4d4f455452414345ULL; dst[1] = count_; dst[2] = start_; dst[3] = stop;
        GlobalTensor<int64_t> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(tail_));
        DataCacheCleanAndInvalid<int64_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(gm);
    }
private:
    GM_ADDR tail_;
    uint64_t start_;
    bool enabled_;
    mutable uint32_t count_ = 0;
};
}  // namespace AfdTrace
'''


def instrument_pull(repo):
    """Instrument the pull-native round: prepare, pull, GMM1, activation, GMM2, combine."""
    path = Path(repo) / PULL_DIR / PULL_CPP
    source = path.read_text()
    # The workspace ends exactly where the caller's reserved tail begins.
    s = replace_checked(source, '#include "afd_mega_moe_pull_pipeline.h"\n',
                        '#include "afd_mega_moe_pull_pipeline.h"\n'
                        + PULL_TRACE_CLASS.replace('__RECORDS__', str(TRACE_RECORDS))
                                           .replace('__STRIDE__', str(TRACE_STRIDE)))
    s = replace_checked(s, '        const uint32_t id = GetBlockIdx();\n',
                        '        const uint32_t id = GetBlockIdx();\n'
                        '        ::AfdTrace::Phase trace(scratch + w.bytes);\n'
                        '        uint32_t traceRound = 0;\n')
    # Round entry: the AIV leader reads batch metadata and resolves the layer slot.
    s = replace_checked(s, '            if ASCEND_IS_AIV {\n                if (id == 0) {\n'
                           '                    Cam::Pull::Transport rx(',
                        '            const auto tracePrepare = trace.Begin(40, traceRound);\n'
                        '            if ASCEND_IS_AIV {\n                if (id == 0) {\n'
                        '                    Cam::Pull::Transport rx(')
    s = replace_checked(s, '                pipe.Reset();\n            }\n            SyncAll<false>();',
                        '                pipe.Reset();\n            }\n'
                        '            trace.End(40, traceRound, tracePrepare);\n'
                        '            const auto traceIn = trace.Begin(47, traceRound);\n'
                        '            SyncAll<false>();\n'
                        '            trace.End(47, traceRound, traceIn);')
    # The two cube calls and the three vector phases of one receive round.
    s = replace_checked(s, '                        Compute(scratch, w, t, round, w13, s13, layer, false, resource);\n'
                           '                        Compute(scratch, w, t, round, w2, s2, layer, true, resource);',
                        '                        const auto traceGmm1 = trace.Begin(42, traceRound);\n'
                        '                        Compute(scratch, w, t, round, w13, s13, layer, false, resource);\n'
                        '                        trace.End(42, traceRound, traceGmm1);\n'
                        '                        const auto traceGmm2 = trace.Begin(44, traceRound);\n'
                        '                        Compute(scratch, w, t, round, w2, s2, layer, true, resource);\n'
                        '                        trace.End(44, traceRound, traceGmm2);')
    s = replace_checked(s, '                        PullAndPrepare(scratch, w, t, round, resource);\n'
                           '                        Activate(scratch, w, t, round, b13, layer, resource);\n'
                           '                        Combine(scratch, w, t, round, b2, layer, resource);',
                        '                        const auto tracePull = trace.Begin(41, traceRound);\n'
                        '                        PullAndPrepare(scratch, w, t, round, resource);\n'
                        '                        trace.End(41, traceRound, tracePull);\n'
                        '                        const auto traceAct = trace.Begin(43, traceRound);\n'
                        '                        Activate(scratch, w, t, round, b13, layer, resource);\n'
                        '                        trace.End(43, traceRound, traceAct);\n'
                        '                        const auto traceCombine = trace.Begin(45, traceRound);\n'
                        '                        Combine(scratch, w, t, round, b2, layer, resource);\n'
                        '                        trace.End(45, traceRound, traceCombine);')
    # Round completion: publish the transaction result back to Attention.
    s = replace_checked(s, '                    if (layer >= 0 && layer < c[13]) tx.CompleteRound(scratch + w.batch, c[10]);\n'
                           '                    tx.Write(completion, tx.ReadState(Cam::Pull::FAILURE));',
                        '                    const auto traceComplete = trace.Begin(46, traceRound);\n'
                        '                    if (layer >= 0 && layer < c[13]) tx.CompleteRound(scratch + w.batch, c[10]);\n'
                        '                    tx.Write(completion, tx.ReadState(Cam::Pull::FAILURE));\n'
                        '                    trace.End(46, traceRound, traceComplete);')
    # Loop-closing barrier, then the per-core header the decoder relies on.
    s = replace_checked(s, '            // All GMM/epilogue events have drained before scratch and flag slots are reused.\n'
                           '            SyncAll<false>();\n'
                           '            if ASCEND_IS_AIV { pipe.Reset(); }\n',
                        '            // All GMM/epilogue events have drained before scratch and flag slots are reused.\n'
                        '            const auto traceOut = trace.Begin(48, traceRound);\n'
                        '            SyncAll<false>();\n'
                        '            trace.End(48, traceRound, traceOut);\n'
                        '            ++traceRound;\n'
                        '            if ASCEND_IS_AIV { pipe.Reset(); }\n')
    s = replace_checked(s, '        } while (!Load(state, Cam::Pull::FAILURE) && Load(state, Cam::Pull::SOURCE) >= 0);',
                        '        } while (!Load(state, Cam::Pull::FAILURE) && Load(state, Cam::Pull::SOURCE) >= 0);\n'
                        '        trace.Finish();')
    path.write_text(s)


def instrument_legacy(repo):
    """Instrument the fused afd_mega_moe_ffn pipeline (pre pull-native builds)."""
    trace_class=r'''
    // Diagnostic contract: the caller owns 72 extended storage tails, __RECORDS__ records/core.
    class PhaseTrace {
    public:
        __aicore__ PhaseTrace(GM_ADDR tail) : tail_(tail), start_(GetSystemCycle()) {
            uint32_t core = GetBlockIdx();
            if ASCEND_IS_AIC { core += 48; }
            enabled_ = !AFD_TRACE_SPARSE || core == 0 || core == 4 || core == 8 || core == 28 || core == 47 || core == 48 || core == 60 || core == 71;
            tail_ += core * __STRIDE__;
        }
        __aicore__ uint64_t Begin(uint32_t code, uint32_t seq) const {
            if (!enabled_) return 0;
    #if AFD_TRACE_NATIVE
            PrintTimeStamp(65536 + code * 128 + seq);
    #endif
            return GetSystemCycle();
        }
        __aicore__ void End(uint32_t code, uint32_t seq, uint64_t start) const {
            if (!enabled_) return;
            const auto stop = GetSystemCycle();
    #if AFD_TRACE_NATIVE
            PrintTimeStamp(65536 + code * 128 + seq);
    #endif
            if (count_ < __RECORDS__) {
                auto dst = reinterpret_cast<volatile __gm__ uint64_t *>(tail_ + 256 + count_ * 32);
                dst[0] = start; dst[1] = stop; dst[2] = code; dst[3] = seq;
            }
            ++count_;
        }
        __aicore__ void Finish() const {
            const auto stop = GetSystemCycle();
            auto dst = reinterpret_cast<volatile __gm__ uint64_t *>(tail_);
            dst[0] = 0x4d4f455452414345ULL; dst[1] = count_; dst[2] = start_; dst[3] = stop;
            GlobalTensor<int64_t> gm;
            gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(tail_));
            DataCacheCleanAndInvalid<int64_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(gm);
        }
    private:
        GM_ADDR tail_;
        uint64_t start_;
        bool enabled_;
        mutable uint32_t count_ = 0;
    };
    '''
    p = LEGACY_DIR
    original = (repo/p/'afd_mega_moe_pipeline.h').read_text()
    cpp = (repo/p/'afd_mega_moe_ffn.cpp').read_text()
    packed = 'const auto layer = ReadControl(scratch + w.validation, 0);' in cpp
    s = replace_checked(original, 'class PipelineFlags {',
                        trace_class.replace('__RECORDS__', str(TRACE_RECORDS))
                                   .replace('__STRIDE__', str(TRACE_STRIDE))
                        + '\nclass PipelineFlags {')
    s = replace_checked(s, 'PipelineFlags(GM_ADDR base) : base_(base) {}',
                        'PipelineFlags(GM_ADDR base, GM_ADDR tail) : trace(tail), base_(base) {}')
    s = replace_checked(s, 'public:\n    __aicore__ explicit PipelineFlags',
                        'public:\n    PhaseTrace trace;\n    __aicore__ explicit PipelineFlags')
    s = replace_checked(s, '        WaitAt(Address(stage, 0), sequence);',
                        '        const auto start = trace.Begin(8 + stage, sequence);\n'
                        '        WaitAt(Address(stage, 0), sequence);\n'
                        '        trace.End(8 + stage, sequence, start);')
    s = replace_checked(s, '        if ASCEND_IS_AIC {\n            Fence<HardEvent::FIX_S>();',
                        '        const auto drain = trace.Begin(15 + stage, sequence);\n'
                        '        if ASCEND_IS_AIC {\n            Fence<HardEvent::FIX_S>();')
    s = replace_checked(s, '        // Leaders publish after their members;',
                        '        trace.End(15 + stage, sequence, drain);\n'
                        '        const auto publish = trace.Begin(22 + stage, sequence);\n'
                        '        // Leaders publish after their members;')
    s = replace_checked(s, '    }\nprivate:\n    __aicore__ GM_ADDR Address',
                        '        trace.End(22 + stage, sequence, publish);\n'
                        '    }\nprivate:\n    __aicore__ GM_ADDR Address')
    s = replace_checked(s, '        transport.Receive(next);',
                        '        const auto rx = flags.trace.Begin(RECEIVED, next.sequence);\n'
                        '        transport.Receive(next);\n'
                        '        flags.trace.End(RECEIVED, next.sequence, rx);', 2)
    s = replace_checked(s, '        transport.Send(tile);',
                        '        const auto tx = flags.trace.Begin(SENT, tile.sequence);\n'
                        '        transport.Send(tile);\n        flags.trace.End(SENT, tile.sequence, tx);')
    s = replace_checked(s, '    transport.Finish();',
                        '    const auto finish = flags.trace.Begin(32, 0);\n'
                        '    transport.Finish();\n    flags.trace.End(32, 0, finish);')
    s = replace_checked(s, '    flags.Wait(first ? PRE1 : PRE2, tile.sequence);',
                        '    flags.Wait(first ? PRE1 : PRE2, tile.sequence);\n'
                        '    const auto compute = flags.trace.Begin(first ? G1 : G2, tile.sequence);')
    s = replace_checked(s, '    flags.Arrive(first ? G1 : G2,',
                        '    flags.trace.End(first ? G1 : G2, tile.sequence, compute);\n'
                        '    flags.Arrive(first ? G1 : G2,')
    wait = '    flags.Wait(RECEIVED, tile.sequence);' if packed else '    if (first) flags.Wait(RECEIVED, tile.sequence);'
    stage = 'PRE1' if packed else 'first ? PRE1 : PRE2'
    s = replace_checked(s, wait, wait+f'\n    const auto prep = flags.trace.Begin({stage}, tile.sequence);')
    s = replace_checked(s, f'    flags.Arrive({stage},',
                        f'    flags.trace.End({stage}, tile.sequence, prep);\n    flags.Arrive({stage},')
    s = replace_checked(s, '        InvalidateScalarInputs(slot + w.tileCounts);',
                        '        const auto activation = flags.trace.Begin(7, tile.sequence);\n'
                        '        InvalidateScalarInputs(slot + w.tileCounts);')
    activation_end = ('        // The rounded INT8 activation' if packed else '        // PRE2 reads the same row partition')
    s = replace_checked(s, activation_end,
                        '        flags.trace.End(7, tile.sequence, activation);\n'+activation_end)
    s = replace_checked(s, '        OutputTile(slot,',
                        '        const auto output = flags.trace.Begin(OUTPUT, tile.sequence);\n        OutputTile(slot,')
    s = replace_checked(s, '        flags.Arrive(OUTPUT,',
                        '        flags.trace.End(OUTPUT, tile.sequence, output);\n        flags.Arrive(OUTPUT,')
    cpp = replace_checked(cpp, 'PipelineFlags flags(scratch + w.flags);',
                          'PipelineFlags flags(scratch + w.flags, scratch + w.bytes);\n'
                          '        const auto entry = flags.trace.Begin(29, 0);')
    cpp = replace_checked(cpp, '                rx.Process(false);',
                          '                const auto traceDrStart = flags.trace.Begin(30, 0);\n'
                          '                rx.Process(false);\n                flags.trace.End(30, 0, traceDrStart);')
    if packed:
        cpp = replace_checked(cpp, '                const auto result = ValidateTransaction(',
                              '                const auto validation = flags.trace.Begin(31, 0);\n'
                              '                const auto result = ValidateTransaction(')
        cpp = replace_checked(cpp, '                StoreControl(scratch + w.validation, 0, result);',
                              '                flags.trace.End(31, 0, validation);\n'
                              '                StoreControl(scratch + w.validation, 0, result);')
        entry_end = '        const auto layer = ReadControl(scratch + w.validation, 0);'
        cpp = replace_checked(cpp, entry_end, '        flags.trace.End(29, 0, entry);\n'+entry_end)
    else:
        cpp = replace_checked(cpp, '        const auto layerId =',
                              '        flags.trace.End(29, 0, entry);\n'
                              '        const auto validation = flags.trace.Begin(31, 0);\n'
                              '        const auto layerId =')
        cpp = replace_checked(cpp, '        TileCursor cursor(',
                              '        flags.trace.End(31, 0, validation);\n        TileCursor cursor(')
    cpp = replace_checked(cpp, '        SyncAll<false>();\n        WriteCompletion(completion, 0, pipe);',
                          '        const auto exit = flags.trace.Begin(33, 0);\n        SyncAll<false>();\n'
                          '        flags.trace.End(33, 0, exit);\n        flags.trace.Finish();\n'
                          '        WriteCompletion(completion, 0, pipe);')
    # Validate every anchor before writing either diagnostic file.
    (repo/p/'afd_mega_moe_pipeline.h').write_text(s)
    (repo/p/'afd_mega_moe_ffn.cpp').write_text(
        '#define AFD_TRACE_NATIVE 0\n#define AFD_TRACE_SPARSE 0\n' + cpp)


def decode(data, work):
    cores = []
    for core, row in enumerate(data):
        magic, count, start, end = map(int, row[:4])
        if magic != TRACE_MAGIC or not 0 <= count <= TRACE_RECORDS or end < start:
            raise ValueError(f'incomplete or overflowed pipeline trace on core {core}')
        records = [list(map(int, row[32+i*4:36+i*4])) for i in range(count)]
        if any(not start <= a <= b <= end for a, b, code, seq in records):
            raise ValueError('pipeline timestamp outside kernel lifetime')
        cores.append(dict(core=core, start=start, end=end, events=records))
    if len(cores) != TRACE_CORES:
        raise ValueError('pipeline trace missing A3 cores')
    return dict(work=work, cores=cores)

