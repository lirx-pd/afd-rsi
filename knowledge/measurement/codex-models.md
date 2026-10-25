# Codex model policy

Use the authenticated proposer host's existing Codex provider/login. Pin model
and reasoning effort explicitly rather than inheriting a personal default:

| Work | Model | Reasoning effort |
| --- | --- | --- |
| Operate experiments with Codex | gpt-6.1-sol | medium |
| Generate code or patches | gpt-6.1-sol | xhigh |
| Analyze evidence, profiling, optimization direction | gpt-6-astra | xhigh |

The requested "ultra high" maps to `xhigh`; it does not select `ultra`.
The proposal adapter uses Astra to produce evidence-bound hypotheses without
patches, then Sol to turn those hypotheses into complete diffs. An analysis stop
skips coding. Both stages remain read-only. Preserve both stages' schema,
outputs, command records, JSONL, stderr and model-tagged token usage. A failed
stage fails the proposal; never silently fall back to another model or effort.

The controller executes build/validation/profiling/measurement directly and
retains sole ownership of acceptance. It does not need a model to run commands.
When Codex is separately used to operate an experiment, invoke it with
`--model gpt-6.1-sol -c 'model_reasoning_effort="medium"'`.

`search.max_calls` bounds Agent command dispatches, including external requests,
not individual internal CLI stages. Each built-in dispatch makes at most two
model calls (one when analysis stops), within the same frozen timeout/deadline.
Reserve no extra dispatch on resume. Preserve interrupted evidence and use a
new campaign for this changed adapter; never edit an existing frozen provider.
