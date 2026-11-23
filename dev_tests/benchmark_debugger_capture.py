"""Offline, identical debugger traffic against a dedicated test database."""
import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--source', default=str(ROOT))
parser.add_argument('--output', required=True)
parser.add_argument('--repetitions', type=int, default=3)
parser.add_argument('--concurrency', type=int, nargs='+', default=[1,24,32])
parser.add_argument('--env-file', type=Path, help='Private file containing DEBUGGER_TEST_DATABASE_URL')
args = parser.parse_args()
HERE = Path(args.output).resolve().parent
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, args.source)
sys.path.append(str(ROOT))
import asyncpg
from dotenv import dotenv_values

from dev_tests.benchmark_cpu_preparation import _memory

sys.path.insert(0, args.source)
import debug_logger
from debug_logger import DebugLogger
from debugger_postgres import debugger_schema

URL = (dotenv_values(args.env_file) if args.env_file else os.environ)['DEBUGGER_TEST_DATABASE_URL']
assert urlsplit(URL).path.endswith('_test')
logging.basicConfig(level=logging.ERROR)
assert Path(debug_logger.__file__).resolve().parent == Path(args.source).resolve()

def percentile(values, fraction):
    return sorted(values)[min(len(values)-1, int((len(values)-1)*fraction))] if values else 0

async def measure(concurrency, large):
    identity = 'test_perf_' + uuid4().hex[:24]
    instance = DebugLogger(enabled=True, db_path=str(HERE / 'never-history.db'), backend='postgresql',
                           database_url=URL, instance_id=identity)
    await instance.initialize()
    assert instance.storage_available
    prompt = ('Fictional source: café 海 🚀. ' * (48000 if large else 1500))
    chunk = ('Verbatim café 海\x00🚀. ' * (1600 if large else 200))
    expected_trace = chunk * 16
    request = {'prompt': prompt, 'request_name': 'synthetic benchmark'}
    latencies, lags, rss = [], [], []
    running = True
    async def sampler():
        while running:
            # Windows' coarse loop clock may fire asyncio.sleep(5ms) early.
            # A real 10ms wait measures resume delay without a busy probe loop.
            before = time.perf_counter()
            await asyncio.to_thread(time.sleep, .010)
            lags.append(max(0, time.perf_counter()-before-.010))
            rss.append(_memory()['rss_bytes'])
    async def call(coro):
        before=time.perf_counter()
        value=await coro
        latencies.append(time.perf_counter()-before)
        return value
    async def session(n):
        sid=f'synthetic-{n}'
        await call(instance.record_session_start(sid, request_payload=request, project_id='benchmark'))
        for k in range(16):
            if k % 4 == 0:
                await call(instance.record_event(sid, event_type='sample', payload={'n':k, 'text':'fictional'}))
            await call(instance.record_content_trace(session_id=sid,phase='generation',attempt=1,model='synthetic',
                       content_delta=chunk,expected_offset=k*len(chunk)))
            await asyncio.sleep(0)
        await call(instance.record_usage_summary(sid, {'input_tokens':17,'output_tokens':23}))
        await call(instance.update_session_status(sid,status='completed',final_payload={'content':expected_trace}))
    monitor=asyncio.create_task(sampler())
    cpu0=time.process_time(); t0=time.perf_counter(); base_rss=_memory()['rss_bytes']
    try:
        await asyncio.gather(*(session(n) for n in range(concurrency)))
        capture_s=time.perf_counter()-t0
        flush=getattr(instance,'flush',None)
        if flush is not None:
            await flush(timeout=120)
        durable_s=time.perf_counter()-t0
        transport=getattr(instance,'_transport',None)
        if transport:
            assert not transport.path.exists(), 'Healthy traffic must not spill locally'
        cpu_s=time.process_time()-cpu0
        running=False
        await monitor
        connection=await asyncpg.connect(URL)
        schema=debugger_schema(identity)
        try:
            rows=await connection.fetch(f'SELECT s.session_id,s.request_json,s.final_json,s.status,t.content FROM "{schema}".sessions s JOIN "{schema}".content_traces t USING(session_id) ORDER BY s.session_id')
            assert len(rows)==concurrency
            expected=expected_trace.encode('utf-8')
            for row in rows:
                assert json.loads(row['request_json'])==request
                assert json.loads(row['final_json'])=={'content':expected_trace}
                assert row['status']=='completed' and row['content']==expected
            assert await connection.fetchval(f'SELECT count(*) FROM "{schema}".session_events')==4*concurrency
            result={'concurrency':concurrency,'large':large,'capture_s':capture_s,'durable_s':durable_s,'cpu_s':cpu_s,
                    'call_p50_ms':percentile(latencies,.5)*1000,'call_p95_ms':percentile(latencies,.95)*1000,
                    'loop_p95_ms':percentile(lags,.95)*1000,'loop_max_ms':max(lags,default=0)*1000,
                    'loop_samples':len(lags),'capture_peak_bytes':transport._peak_ram if transport else None,
                    'rss_base_mib':base_rss/2**20,'rss_peak_mib':max(rss,default=base_rss)/2**20,
                    'prompt_bytes':len(prompt.encode('utf-8')),'trace_bytes_per_session':len(expected),
                    'trace_sha256':hashlib.sha256(expected).hexdigest(),'verified_sessions':len(rows)}
        finally:
            await instance.close()
            await connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
            await connection.close()
        return result
    finally:
        running=False
        await monitor
        await instance.close()

async def main():
    results=[]
    for repetition in range(args.repetitions):
        for large in (False,True):
            for concurrency in args.concurrency:
                row=await measure(concurrency,large)
                row['repetition']=repetition+1
                results.append(row)
                Path(args.output).write_text(json.dumps({'source':args.source,'results':results},indent=2),encoding='utf-8')
                print(json.dumps(row),flush=True)

asyncio.run(main())
