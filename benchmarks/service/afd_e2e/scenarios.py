# SPDX-License-Identifier: Apache-2.0
"""Fixed-DP workloads using the existing service and strict streaming client."""
import asyncio
import json
import random
import time

from .performance import SEED, stream_request
from .scenario_contract import counter_delta, counters, summarize, verify_requests


async def snapshot(client, base, path, timeout, dp):
    response = await asyncio.wait_for(client.get(base + '/metrics'), min(timeout, 10))
    response.raise_for_status()
    text = response.text
    path.write_text(text)
    counters(text, dp)
    return text


async def drain_metrics(client, base, path, timeout, before, contract, warmup):
    # Metrics publication can lag the completed HTTP responses; never substitute planned counts.
    deadline = time.monotonic() + 10
    while True:
        after = await snapshot(client, base, path, timeout, contract['dp'])
        try:
            counter_delta(before, after, contract, warmup=warmup)
            return after
        except ValueError:
            if time.monotonic() >= deadline:
                raise
            await asyncio.sleep(0.5)


async def requests(client, base, model, vocabulary, name, contract, raw, timeout, *, warmup):
    count = contract['warmup_per_dp' if warmup else 'requests_per_dp']
    output = contract['warmup_output_tokens' if warmup else 'output_tokens']
    rate = None if warmup else contract['rps']
    records, tasks = [], []
    start = time.perf_counter()

    async def request(index, dp):
        seed = SEED + index * contract['dp'] + dp + (1000000 if warmup else 0)
        record = dict(dp_rank=dp, index=index, phase='warmup' if warmup else 'measured',
                      input_target=contract['input_lens'][dp], output_target=output,
                      prompt_seed=seed, headers={'X-data-parallel-rank': str(dp)}, status='running',
                      capture_token_times=name == 'long-decode' and not warmup)
        records.append(record)
        try:
            body = dict(model=model, prompt=random.Random(seed).choices(vocabulary, k=record['input_target']),
                        max_tokens=output, stream=True, stream_options={'include_usage': True},
                        temperature=0, ignore_eos=True, add_special_tokens=False, return_token_ids=True)
            record['sent_at'] = time.perf_counter()
            await asyncio.wait_for(stream_request(client, base + '/v1/completions', body, record,
                                                  record['headers']), timeout)
        except asyncio.CancelledError:
            record.update(status='cancelled', error='request cancelled')
            raise
        except Exception as error:
            record.update(status='failed', error=f'{type(error).__name__}: {error}')
        finally:
            raw.write(json.dumps(record, allow_nan=False) + '\n')
            raw.flush()

    try:
        for index in range(count):
            for dp in range(contract['dp']):
                if rate is not None:
                    scheduled = (index * contract['dp'] + dp) / rate
                    await asyncio.sleep(max(0, start + scheduled - time.perf_counter()))
                tasks.append(asyncio.create_task(request(index, dp)))
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    verify_requests(records, contract, warmup=warmup)
    return records


async def evaluate_scenario(client, base, model, vocabulary, name, contract, directory, timeout):
    directory = directory / 'scenarios' / name
    directory.mkdir(parents=True)
    try:
        before = await snapshot(client, base, directory / 'before-warmup.prom', timeout, contract['dp'])
        with (directory / 'requests.jsonl').open('x') as raw:
            await requests(client, base, model, vocabulary, name, contract, raw, timeout, warmup=True)
            before = await drain_metrics(client, base, directory / 'before.prom', timeout, before, contract, True)
            measured = await requests(client, base, model, vocabulary, name, contract, raw, timeout, warmup=False)
            after = await drain_metrics(client, base, directory / 'after.prom', timeout, before, contract, False)
        return summarize(name, contract, measured, before, after)
    except (KeyboardInterrupt, asyncio.CancelledError):
        raise
    except Exception as error:
        return dict(id=name, status='failed', configuration=contract,
                    error=f'{type(error).__name__}: {error}')
