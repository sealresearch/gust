"""Check complete call strings and summarize warm best-of-N whole-request timings."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--records', required=True, type=Path)
parser.add_argument('--expected', required=True, type=Path, help='JSON mapping request IDs to exact reference call strings')
parser.add_argument('--repetitions', type=int, default=3)
parser.add_argument('--correct-ids', type=Path, help='Optional JSON list of IDs known correct under the prescribed scorer')
parser.add_argument('--output', type=Path)
parser.add_argument('logs', nargs='+', type=Path)
args = parser.parse_args()
if args.repetitions < 2:
    parser.error('at least two repetitions are needed to discard the first call')
raw = args.records.read_bytes()
records = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
expected = json.loads(args.expected.read_text(encoding='utf-8'))
pattern = re.compile(r'^record=(\d+) tools=(\d+) repeat=(\d+) elapsed_ms=([\d.]+) calls=(.*)$', re.M)
samples = [[] for _ in records]
peaks, peak_kinds, guests, graphs = [], set(), [], set()
for path in args.logs:
    text = path.read_text(encoding='utf-8')
    data = pattern.findall(text)
    coverage = {(int(record), int(repeat)) for record, _, repeat, _, _ in data}
    wanted = {(r, n) for r in range(len(records)) for n in range(args.repetitions)}
    if len(data) != len(wanted) or coverage != wanted:
        raise RuntimeError(f'Incomplete measurement coverage: {path}')
    for record, tools, repeat, elapsed, calls in data:
        index, ms = int(record), float(elapsed)
        if int(tools) != len(records[index]['tools']) or ms < 0:
            raise RuntimeError(f'Invalid measurement: {path} record {record}')
        if calls != expected[str(records[index]['id'])]:
            raise RuntimeError(f'Prediction bytes differ: {path} record {record}')
        if int(repeat):
            samples[index].append(ms)
    peak = re.search(r'^peak_(rss|working_set)_kib=(\d+)', text, re.M)
    if not peak:
        raise RuntimeError(f'Missing process peak: {path}')
    peak_kinds.add(peak[1])
    peaks.append(int(peak[2]))
    graph = re.search(r'graph_bytes=(\d+)', text)
    if not graph:
        raise RuntimeError(f'Missing graph storage: {path}')
    graphs.add(int(graph[1]))
    if guest := re.search(r'guest_bytes=(\d+)', text):
        guests.append(int(guest[1]))
if len(peak_kinds) != 1 or len(graphs) != 1:
    raise RuntimeError('Mixed process memory metrics or graph storage')
warm_count = len(args.logs) * (args.repetitions - 1)
if not all(len(values) == warm_count for values in samples):
    raise RuntimeError('Missing warm samples')
best = [min(values) for values in samples]
result = {
    'requests': len(records), 'fresh_processes': len(args.logs), 'warm_samples_per_request': warm_count,
    'calls_verified': len(records) * len(args.logs) * args.repetitions,
    'all_prediction_strings_exact': True, 'records_sha256': hashlib.sha256(raw).hexdigest(),
    'median_warm_best_ms': statistics.median(best), 'max_warm_best_ms': max(best),
    'peak_process_kib': max(peaks), 'peak_kind': next(iter(peak_kinds)), 'graph_bytes': next(iter(graphs)),
    'guest_bytes': max(guests) if guests else None,
    'requests_with_abstentions': sum(not json.loads(expected[str(row['id'])]) for row in records),
    'by_request': [{'id': row['id'], 'warm_best_ms': ms} for row, ms in zip(records, best)],
}
if args.correct_ids:
    correct_ids = set(json.loads(args.correct_ids.read_text(encoding='utf-8')))
    correct_times = [ms for row, ms in zip(records, best) if row['id'] in correct_ids]
    if not correct_times:
        raise RuntimeError('No correct requests selected')
    result['correct_requests'] = len(correct_times)
    result['correct_median_warm_best_ms'] = statistics.median(correct_times)
    result['correct_max_warm_best_ms'] = max(correct_times)
output = json.dumps(result, indent=2) + '\n'
if args.output:
    args.output.write_text(output, encoding='utf-8')
print(output, end='')
