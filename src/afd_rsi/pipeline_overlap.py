"""Conservative overlap of sampled core spans using round barrier bounds."""


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _subtract(intervals, excluded):
    result = []
    excluded = _union(excluded)
    for start, end in _union(intervals):
        cursor = start
        for left, right in excluded:
            if right <= cursor:
                continue
            if left >= end:
                break
            if left > cursor:
                result.append((cursor, min(left, end)))
            cursor = max(cursor, right)
        if cursor < end:
            result.append((cursor, end))
    return result


def overlap_bounds(first, second, lower, upper):
    """Exact extrema of interval-union overlap for one bounded clock offset."""
    first, second = _union(first), _union(second)
    if lower > upper:
        raise ValueError('empty clock offset range')
    if not first or not second:
        return None
    def at(offset):
        i = j = total = 0
        while i < len(first) and j < len(second):
            a, b = first[i]
            c, d = second[j]
            c, d = c + offset, d + offset
            total += max(0, min(b, d) - max(a, c))
            if b <= d:
                i += 1
            else:
                j += 1
        return total
    # Overlap is piecewise linear in the offset. Sweep its slope changes
    # rather than evaluating every pair of intervals at every breakpoint.
    slope = 0
    changes = {}
    for a, b in first:
        for c, d in second:
            for point, delta in ((a-d, 1), (a-c, -1), (b-d, -1), (b-c, 1)):
                if point <= lower:
                    slope += delta
                elif point < upper:
                    changes[point] = changes.get(point, 0) + delta
    position, value = lower, at(lower)
    minimum = maximum = value
    for point in sorted(changes):
        value += slope * (point - position)
        minimum, maximum = min(minimum, value), max(maximum, value)
        slope += changes[point]
        position = point
    value += slope * (upper - position)
    return min(minimum, value), max(maximum, value)


def _rounds(events):
    entries = sorted(event for event in events if event[2] == 47)
    exits = sorted(event for event in events if event[2] == 48)
    if not entries or len(entries) != len(exits):
        return None
    result = []
    previous = None
    for entry, exit_ in zip(entries, exits, strict=True):
        if (entry[3] != exit_[3] or entry[1] > exit_[0]
                or (previous is not None and entry[0] < previous)):
            return None
        previous = exit_[1]
        result.append((entry, exit_))
    return result


def _intervals(events, entry, exit_, code):
    return [(start, end) for start, end, kind, sequence in events
            if kind == code and sequence == entry[3] and entry[1] <= start <= end <= exit_[0]]


def _core_pair(transactions, first_core, second_core, definitions):
    stats = {name: {'rounds': 0, 'phase_overlap_cycles': [0, 0],
                    'rounds_with_marked_gmm_wait': 0,
                    'overlap_without_marked_gmm_wait_cycles': [0, 0],
                    'first_span_cycles': 0, 'second_span_cycles': 0,
                    'gmm_span_without_marked_wait_cycles': 0}
             for name, _, _, _ in definitions}
    max_uncertainty = 0
    total_rounds = 0
    for transaction in transactions:
        cores = {core['core']: core['events'] for core in transaction['cores']}
        first, second = cores.get(first_core, []), cores.get(second_core, [])
        vr, cr = _rounds(first), _rounds(second)
        if vr is None or cr is None or len(vr) != len(cr):
            return {'status': 'unavailable', 'reason': 'missing or unmatched round barriers'}
        for (vi, vo), (ci, co) in zip(vr, cr, strict=True):
            if vi[3] != ci[3]:
                return {'status': 'unavailable', 'reason': 'round sequence differs across cores'}
            # Offset maps the second clock to the first. Each barrier's
            # common instant lies inside both participating cores' spans.
            lower = max(vi[0] - ci[1], vo[0] - co[1])
            upper = min(vi[1] - ci[0], vo[1] - co[0])
            if lower > upper:
                return {'status': 'unavailable', 'reason': 'barrier clock offset bounds disagree'}
            total_rounds += 1
            max_uncertainty = max(max_uncertainty, upper - lower)
            for name, first_code, second_code, wait_code in definitions:
                a = _intervals(first, vi, vo, first_code)
                b = _intervals(second, ci, co, second_code)
                waits = _intervals(second, ci, co, wait_code) if wait_code else []
                phase = overlap_bounds(a, b, lower, upper)
                without_wait = _subtract(b, waits)
                useful = overlap_bounds(a, without_wait, lower, upper)
                if phase is None:
                    continue
                item = stats[name]
                item['rounds'] += 1
                item['rounds_with_marked_gmm_wait'] += bool(waits)
                for i in (0, 1):
                    item['phase_overlap_cycles'][i] += phase[i]
                    if waits:
                        item['overlap_without_marked_gmm_wait_cycles'][i] += useful[i] if useful else 0
                item['first_span_cycles'] += sum(end-start for start, end in _union(a))
                item['second_span_cycles'] += sum(end-start for start, end in _union(b))
                if waits:
                    item['gmm_span_without_marked_wait_cycles'] += sum(end-start for start, end in without_wait)
    for item in stats.values():
        if item['rounds_with_marked_gmm_wait'] != item['rounds']:
            item['overlap_without_marked_gmm_wait_cycles'] = None
            item['gmm_span_without_marked_wait_cycles'] = None
    return {'status': 'available', 'rounds': total_rounds,
            'max_offset_range_cycles': max_uncertainty,
            'stages': {name: value for name, value in stats.items() if value['rounds']}}


def analyze_overlap(transactions):
    pairs = []
    cube_vector = (('pull_gmm1', 41, 42, 74), ('pull_gmm2', 41, 44, 75),
                   ('gmm1_activation', 43, 42, 74), ('gmm2_activation', 43, 44, 75),
                   ('gmm1_activation_body', 79, 42, 74), ('gmm2_activation_body', 79, 44, 75),
                   ('gmm1_output', 70, 42, 74), ('gmm2_output', 70, 44, 75),
                   ('gmm1_tx_calls', 71, 42, 74), ('gmm2_tx_calls', 71, 44, 75))
    for vector, cube in ((0, 48), (4, 60), (8, 71)):
        pairs.append({'vector_core': vector, 'cube_core': cube,
                      **_core_pair(transactions, vector, cube, cube_vector)})
    vector_vector = (('pull_activation', 41, 43, None), ('pull_output', 41, 70, None),
                     ('pull_tx_calls', 41, 71, None),
                     ('activation_output', 43, 70, None), ('activation_tx_calls', 43, 71, None),
                     ('output_tx_calls', 70, 71, None))
    for first, second in ((0, 4), (4, 8)):
        pairs.append({'vector_cores': [first, second],
                      **_core_pair(transactions, first, second, vector_vector)})
    pairs.append({'cube_cores': [48, 60],
                  **_core_pair(transactions, 48, 60, (('gmm1_gmm2', 42, 44, None),))})
    return {'pairs': pairs,
            'method': 'per-round entry and exit barrier offset bounds; sampled core pairs',
            'assumptions': 'Per-core cycle counters have equal rate and a stable offset within each round; '
                           'consistent barrier bounds do not independently prove clock calibration.',
            'scope': 'Intervals are instrumented call spans, not hardware busy time or a full-rank critical path. '
                     'GMM marked waits can be removed; activation_body excludes marked GMM1 waits and outer '
                     'barriers when code 79 exists, but still includes internal epilogue synchronization. '
                     'TX intervals cover enqueue and drain calls, not asynchronous link occupancy. '
                     'Round totals aggregate captured cases without token-level attribution.'}
