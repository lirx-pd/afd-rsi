"""A3-only diagnostic overlay; never modify the production build."""
from pathlib import Path
import json

TRACE_CORES = 72
TRACE_STRIDE = 16640
TRACE_MAGIC = 0x4d4f455452414345


def replace_checked(source, old, new, count=1):
    if source.count(old) != count:
        raise ValueError(f'unsupported MegaMoe diagnostic source: expected {count} occurrences of {old!r}')
    return source.replace(old, new)


def instrument(repo):
    p = Path('csrc/npu/ascend_kernels/afd_mega_moe_ffn/op_kernel')
    trace_class=r'''
    // Diagnostic contract: the caller owns 72 extended storage tails, 512 records/core.
    class PhaseTrace {
    public:
        __aicore__ PhaseTrace(GM_ADDR tail) : tail_(tail), start_(GetSystemCycle()) {
            uint32_t core = GetBlockIdx();
            if ASCEND_IS_AIC { core += 48; }
            enabled_ = !AFD_TRACE_SPARSE || core == 0 || core == 4 || core == 8 || core == 28 || core == 47 || core == 48 || core == 60 || core == 71;
            tail_ += core * 16640;
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
            if (count_ < 512) {
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
    original = (repo/p/'afd_mega_moe_pipeline.h').read_text()
    cpp = (repo/p/'afd_mega_moe_ffn.cpp').read_text()
    packed = 'const auto layer = ReadControl(scratch + w.validation, 0);' in cpp
    s = replace_checked(original, 'class PipelineFlags {', trace_class+'\nclass PipelineFlags {')
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
        if magic != TRACE_MAGIC or not 0 <= count <= 512 or end < start:
            raise ValueError(f'incomplete or overflowed pipeline trace on core {core}')
        records = [list(map(int, row[32+i*4:36+i*4])) for i in range(count)]
        if any(not start <= a <= b <= end for a, b, code, seq in records):
            raise ValueError('pipeline timestamp outside kernel lifetime')
        cores.append(dict(core=core, start=start, end=end, events=records))
    if len(cores) != TRACE_CORES:
        raise ValueError('pipeline trace missing A3 cores')
    return dict(work=work, cores=cores)

