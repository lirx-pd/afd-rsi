"""Offline model-context comparison. UTF-8 bytes are not tokenizer counts."""

import argparse
import json
from pathlib import Path

from afd_rsi.model_protocol import ModelContext

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('request', type=Path)
args = parser.parse_args()
request = json.loads(args.request.read_text())
wire = ModelContext(request)
# A conservative coder receives every repository source and MC2/generated contract.
files = [key for key, item in wire.files.items() if key.startswith('s') or
         (key.startswith('d') and item['name'].startswith(('mc2/', 'generated/')))]
analysis = dict(proposals=[dict(hypothesis='context-size fixture', category='measurement',
    priority=3, evidence=[item['id'] for item in request['evidence']], files=files)])
old = len(json.dumps(request, ensure_ascii=False).encode())
initial = len(wire.initial().encode())
code = len(wire.coding(analysis).encode())
print(json.dumps(dict(units='UTF-8 bytes, not tokens; no model call',
    old_context_per_stage=old, analysis_context=initial, conservative_code_context=code,
    reduction_without_read_pct=100 * (1 - (initial + code) / (2 * old)),
    reduction_with_one_max_read_pct=100 * (1 - (2 * initial + 65536 + code) / (2 * old)),
    codex_initial_manifest=len(wire.initial(inline=False).encode())), indent=2))
