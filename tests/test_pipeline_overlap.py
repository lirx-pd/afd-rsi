import random

from afd_rsi.pipeline_overlap import analyze_overlap, overlap_bounds


def transaction(vector, cube):
    return {'cores': [{'core': 0, 'events': vector}, {'core': 48, 'events': cube}]}


def test_overlap_bounds_include_clock_uncertainty_and_remove_gmm_wait():
    vector = [[100, 110, 47, 0], [120, 160, 43, 0], [190, 200, 48, 0]]
    cube = [[95, 105, 47, 0], [115, 125, 74, 0], [115, 155, 42, 0], [185, 195, 48, 0]]
    pair = analyze_overlap([transaction(vector, cube)])['pairs'][0]
    assert pair['status'] == 'available'
    assert pair['max_offset_range_cycles'] == 20
    stage = pair['stages']['gmm1_activation']
    assert stage['phase_overlap_cycles'] == [30, 40]
    assert stage['overlap_without_marked_gmm_wait_cycles'] == [20, 30]
    assert stage['first_span_cycles'] == 40
    assert stage['gmm_span_without_marked_wait_cycles'] == 30
    assert overlap_bounds([(0, 10), (5, 15)], [(0, 10)], 0, 0) == (10, 10)


def test_unmatched_or_inconsistent_barriers_never_report_overlap():
    vector = [[100, 110, 47, 0], [120, 160, 43, 0], [190, 200, 48, 0]]
    missing = [[95, 105, 47, 0], [115, 155, 42, 0]]
    pair = analyze_overlap([transaction(vector, missing)])['pairs'][0]
    assert pair['status'] == 'unavailable'
    inconsistent = missing + [[1000, 1010, 48, 0]]
    pair = analyze_overlap([transaction(vector, inconsistent)])['pairs'][0]
    assert pair['status'] == 'unavailable'
    assert 'bounds disagree' in pair['reason']


def test_pull_and_gmm1_overlap_excludes_marked_input_wait():
    vector = [[100, 110, 47, 0], [115, 145, 41, 0], [145, 160, 43, 0], [190, 200, 48, 0]]
    cube = [[95, 105, 47, 0], [115, 125, 74, 0], [115, 155, 42, 0], [185, 195, 48, 0]]
    stage = analyze_overlap([transaction(vector, cube)])['pairs'][0]['stages']['pull_gmm1']
    assert stage['phase_overlap_cycles'] == [15, 30]
    assert stage['overlap_without_marked_gmm_wait_cycles'] == [5, 25]


def test_vector_stages_compare_output_and_tx_calls_on_distinct_cores():
    first = [[100, 100, 47, 0], [110, 140, 43, 0], [145, 160, 70, 0], [190, 190, 48, 0]]
    second = [[100, 100, 47, 0], [110, 120, 70, 0], [135, 175, 71, 0], [190, 190, 48, 0]]
    tx = {'cores': [{'core': 0, 'events': first}, {'core': 4, 'events': second}]}
    pair = analyze_overlap([tx])['pairs'][3]
    assert pair['status'] == 'available'
    assert pair['stages']['activation_output']['phase_overlap_cycles'] == [10, 10]
    assert pair['stages']['activation_tx_calls']['phase_overlap_cycles'] == [5, 5]
    assert pair['stages']['output_tx_calls']['phase_overlap_cycles'] == [15, 15]


def test_missing_wait_markers_do_not_claim_wait_was_removed():
    vector = [[100, 110, 47, 0], [120, 160, 43, 0], [190, 200, 48, 0]]
    cube = [[95, 105, 47, 0], [115, 155, 42, 0], [185, 195, 48, 0]]
    stage = analyze_overlap([transaction(vector, cube)])['pairs'][0]['stages']['gmm1_activation']
    assert stage['phase_overlap_cycles'] == [30, 40]
    assert stage['overlap_without_marked_gmm_wait_cycles'] is None


def test_rounds_with_reused_sequence_are_paired_by_barrier_order():
    vector = [[100, 110, 47, 0], [120, 160, 43, 0], [190, 200, 48, 0]]
    cube = [[95, 105, 47, 0], [115, 125, 74, 0], [115, 155, 42, 0], [185, 195, 48, 0]]
    def later(events):
        return [[a + 1000, b + 1000, code, seq] for a, b, code, seq in events]
    pair = analyze_overlap([transaction(vector + later(vector), cube + later(cube))])['pairs'][0]
    assert pair['rounds'] == 2
    assert pair['stages']['gmm1_activation']['phase_overlap_cycles'] == [60, 80]
    assert pair['stages']['gmm1_activation']['overlap_without_marked_gmm_wait_cycles'] == [40, 60]


def test_overlap_extrema_match_exhaustive_offsets_with_fragmented_spans():
    rng = random.Random(7)
    for _ in range(200):
        first = [(x, x + rng.randint(1, 8)) for x in rng.sample(range(30), 4)]
        second = [(x, x + rng.randint(1, 8)) for x in rng.sample(range(30), 4)]
        lower, upper = -10, 10
        brute = []
        for offset in range(lower, upper + 1):
            a = {i for start, end in first for i in range(start, end)}
            b = {i + offset for start, end in second for i in range(start, end)}
            brute.append(len(a & b))
        assert overlap_bounds(first, second, lower, upper) == (min(brute), max(brute))
