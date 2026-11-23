# Storage and operational output

Operational logs, debugger evidence and live generation state are separate.
Changing the output profile does not reduce request progress, streamed content,
debugger capture, model work or usage accounting. Changing a storage backend
does not migrate or delete existing SQLite history.

## Output profiles

| Setting | Effect |
| --- | --- |
| `LOG_MODE=normal` | INFO and above; the default. |
| `LOG_MODE=quiet` | Warnings/errors and storage incident/recovery notices; no HTTP access or routine stdout. |
| `LOG_MODE=verbose` | DEBUG and above. Full prompts/responses still require extra verbosity. |
| `GRANSABIO_FORCE_EXTRA_VERBOSE=true` | Full prompts/responses for every request, with progress verbosity enabled. |
| `LOG_BACKEND=local` | Console output, optionally mirrored to hourly files. The default. |
| `GRANSABIO_FILE_LOGGING=true` | Local hourly files; never changes the output level. |
| `LOG_BACKEND=postgresql` | The same operational output in PostgreSQL; independent of the debugger. |

`python main.py --quiet` and `--log-mode normal|quiet|verbose` select the process
profile before engine imports. `--force-verbose`, `--force-extra-verbose` and
`--file-logging` remain supported. Explicit quiet plus forced verbosity or
`EXTRA_VERBOSE=true` is a configuration error. File logging plus PostgreSQL
logging is also an error: choose one persistent destination. Explicit `LOG_MODE`
takes precedence over the legacy `LOG_LEVEL`; when no profile is selected,
`LOG_LEVEL` selects the threshold.

Quiet suppresses diagnostic bodies printed as part of rejected-content logs,
but retains the failure summary, session identity and actual exception. Raw
stderr remains an emergency channel; it is not classified by matching words.
Native writes to file descriptors and supervisor output are outside Python
capture. Uvicorn/Gunicorn application loggers use the same policy. Supervisor
processes do not start a second PostgreSQL writer. Use one serving worker per
installation identity (`APP_WORKERS=1` when using `python main.py` on Linux).

On Windows, `g.bat` / `tools/windows_launcher.ps1` respects `.env` and starts an
independent hidden process. Closing the menu or Codex leaves the server running.
Open the launcher again to inspect or stop the verified instance. Run
`python main.py` directly when a foreground console is wanted. The launcher no
longer forces file logging or extra verbosity: installations that need that
behavior must set their options explicitly before their next start.

## PostgreSQL operational logs

Example for a quiet installation (keep the password and CA path private):

```dotenv
LOG_MODE=quiet
LOG_BACKEND=postgresql
GRANSABIO_FILE_LOGGING=false
GRANSABIO_FORCE_VERBOSE=false
GRANSABIO_FORCE_EXTRA_VERBOSE=false
EXTRA_VERBOSE=false
LOG_DATABASE_URL=postgresql://logs:YOUR_PASSWORD@db.example:5432/logs?sslmode=verify-full&sslrootcert=/path/ca.crt
LOG_INSTANCE_ID=workstation_one
LOG_PG_SPILL=true
LOG_PG_STRICT=false
LOG_PG_RAM_BYTES=67108864
LOG_PG_RAM_SECONDS=30
LOG_PG_SPOOL_BYTES=1073741824
LOG_PG_MIN_FREE_BYTES=268435456
LOG_RETENTION_DAYS=14
# LOG_PG_SPOOL_PATH=/private/local/state/logs-pending/capture.sqlite
```

For development with full prompts/responses stored in the same PostgreSQL
destination, change only these output options from the example:

```dotenv
LOG_MODE=verbose
GRANSABIO_FORCE_EXTRA_VERBOSE=true
```

The destination, identity, spill and retention settings stay independent.

Use an application-owned role/database with permission to create its schema.
Each installation has a stable, distinct `LOG_INSTANCE_ID` (1–40 lowercase
letters/digits/underscores, starting with a letter). Tables live only in
`gransabio_logs_<instance_id>`. `LOG_DATABASE_URL` and `LOG_INSTANCE_ID` are
explicit; neither silently inherits debugger configuration. They may point to
the same application-owned database as the debugger with separate schemas.
Use PostgreSQL TLS verification and narrowly scoped network access.

One background writer captures structured Python logging and Python stdout/
stderr before the bounded console tail is trimmed. Level, logger, original
timestamp and available session/project/phase context are retained; startup and
framework logs need not belong to a debugger session. Immutable source UUID,
sequence and chained hashes identify each fragment. A PostgreSQL transaction
commits both content and its receipt. A lost commit acknowledgement is reconciled
against retained hashes; duplicate identity with different content blocks replay.
Text is encoded as a byte payload, preserving Unicode and NUL exactly.

Large messages are split into ordered fragments of one event, not truncated.
Normal emit only adds bounded data to RAM. Exceptional pressure, including a
single record larger than RAM, can hold its producer while the independent
writer drains bounded pieces. The RAM estimate includes queue, encoding/driver
and replay scratch; it excludes the producer's borrowed input and the separate
UI tail. It is not a process RSS limit. Half the configured RAM is reserved from
ingress to allow replay and encoding to make progress. Multiple unbounded calls
during an indefinite outage cannot be promised unlimited storage.

Healthy capture does not open a SQLite payload journal. An outage uses bounded
RAM first; age or pressure triggers optional private SQLite spill. The default
location is `<local-state>/GranSabio_LLM/logs/pending/<instance_id>/capture.sqlite`
(`LOCALAPPDATA` on Windows; `XDG_STATE_HOME` or `~/.local/state` elsewhere).
An explicit journal path requires its own empty private local directory. Its
brand differs from debugger journals; it cannot open historical SQLite as a
transport journal. The directory is owner-only (plus SYSTEM on Windows), and
the journal uses exclusive locking and `synchronous=EXTRA`. Quotas count pending
encoded payload; SQLite pages/rollback journals also need disk headroom.

Recovery replays the retained prefix before newer RAM. No accepted operation is
evicted, aged out, renamed as a new event or replaced by a different backend.
Connection retries use 1/2/5/10/30-second delays, resetting after validated
recovery. Ownership conflicts, corruption, replaced stores and differing
confirmed prefixes block automatic recovery. Preserve the journal and inspect
the destination; do not reset identities or delete pending files to make it pass.
The debugger recovery CLI is for debugger journals, not operational log journals.

`LOG_PG_SPILL=false` prevents new local payload spill; an existing journal still
recovers. `LOG_PG_STRICT=true` gates new generation/provider dispatch until the
connection, schema, ownership and retained prefix have all been validated.
Capacity pressure also gates new dispatch, reserving space for in-flight calls.
Logging callers never wait for storage capacity. Size the RAM budget to admit
the largest individual event; an oversized event reports incomplete capture.
No exception from a handler is relied upon to stop work: gates run outside
`emit`. Absolute capacity exhaustion or encoding failure latches
`capture_failed`, reports incomplete evidence and stops new work. This remains
visible after the accepted prefix drains; it is not silently cleared. Investigate
the failure, preserve available evidence and restart deliberately. A record
interrupted by exhaustion has fewer fragments than its advertised `parts`.

`LOG_RETENTION_DAYS=14` deletes only confirmed stored records; `0` keeps all.
Pending RAM/journal content never expires. Source receipts survive retention.
PostgreSQL requires `fsync=on` and the writer uses `synchronous_commit=on`.

### Status, reading and shutdown

`GET /health` includes a safe `logging` status. The console recent/SSE endpoints
also include persistence status, separate from the disposable live tail:
`state`, accepted/persisted sequences, pending operations, RAM/local bytes,
oldest age, admission blocking, capture failures and `loss_unknown`. No password
or URI is exposed. Storage pressure and recovery notices remain visible in
quiet; those diagnostics bypass the writer to prevent recursion. The HTTP
health status still describes process health; inspect `logging` for durability.

The `records` table contains one row per fragment. Inspect columns such as
`observed_at`, `level`, `logger`, `session_id` and `phase`, then decode
`payload` as UTF-8 JSON. Group by `(origin,event_id)`, order by `part`, verify
that the fragment count equals `parts`, and concatenate `event.text` exactly.
Do not use the live console's sequence or trimmed tail as a persistence receipt.

Normal application shutdown closes the writer before Python finalization:
a bounded drain is followed by journaling the remainder when enabled. Application
capture ends at that point; the final ASGI process-exit messages are console-only.
SIGKILL, power loss and the Windows immediate Ctrl+C/SIGBREAK path bypass this;
RAM can be lost. A new start reports possible loss when a prior remote owner
has no clean shutdown receipt (or the recovered journal was not closed cleanly).
An offline process that never created any durable receipt cannot prove its
prior volatile tail after a crash. `loss_unknown` is historical uncertainty,
not an invented loss count; operational log `complete` stays false when it is
present. These logs do not resume generation or repeat provider calls.

Full prompts/responses may contain sensitive data. Apply appropriate database
access, backups and retention, independently of the selected output level.

## Debugger capture and recovery

The debugger records evidence. It does not own generation state, restart work,
retry a provider, or settle usage or billing. SQLite history keeps its existing
behavior and location. PostgreSQL capture never migrates that history.

### Normal operation and failures

With `DEBUGGER_BACKEND=postgresql`, a single writer accepts immutable operations
into bounded RAM, then applies them in ordered PostgreSQL transactions. Each
operation retains its observed timestamp, exact payload, sequence and chained
checksum. The transaction updates both content and the confirmed sequence.
A lost COMMIT reply is reconciled against retained content before retrying.

RAM acceptance is volatile. It advances stream offsets but does not mean the
content is durable. Reads wait for the captured prefix and all registered stream
buffers; if that cut cannot be confirmed within two seconds, the API returns
503. Existing session/event/trace response bodies remain unchanged. The debugger
page and `GET /debugger/storage` separately report capture state, pending
operations, buffered stream bytes, estimated RAM, local pending bytes, age,
retry delay and safe failure reason.

| State | Meaning |
| --- | --- |
| `available` | PG is connected; inspect `complete` and pending counts for the current captured cut. |
| `buffering` | PG is unavailable; accepted data is still volatile. |
| `local` | A pending prefix is confirmed locally; newer RAM may also be pending. |
| `recovering` | The same writer is replaying the retained prefix before newer operations. |
| `blocked` | Capacity, ownership, integrity or configuration needs attention; new dispatch is paused. |

The writer retries connection failures with jittered delays of approximately
1, 2, 5, 10, 30 and 60 seconds. A brief successful batch does not reset the
backoff; a stable connection does. There is one reconnect/apply task, not one
per request. Healthy RAM bursts wait for capacity without writing a local
payload journal. When PG is unavailable, RAM age or pressure triggers spill if
enabled. Disk quota/free-space failures preserve accepted data and do not stop
PG from reconnecting and draining it.

### Bounds and modes

```dotenv
DEBUGGER_PG_SPILL=true
DEBUGGER_PG_STRICT=false
DEBUGGER_PG_RAM_BYTES=67108864
DEBUGGER_PG_RAM_SECONDS=30
DEBUGGER_PG_SPOOL_BYTES=1073741824
# Optional file in its own empty private directory on a local disk:
# DEBUGGER_PG_SPOOL_PATH=/private/state/pending/capture.sqlite
```

The RAM estimate covers retained operations, encoded representations, driver
scratch, replay scratch, and coalesced text/segment metadata. It is not a cap
on the whole engine process, application request bodies, or response objects.
A record larger than the capture budget keeps its producer awaiting confirmation
and encodes bounded pieces from that input. It is not truncated or given a new
prompt/result length limit. Backpressure preserves a reserve for calls already
in flight; it cannot guarantee that arbitrarily many unbounded provider results
fit in RAM during an indefinite outage.

`DEBUGGER_PG_SPILL=false` prevents new spill payloads. Pending RAM then needs PG
to return; age/capacity gates stop new dispatch. It does not skip a journal found
at startup. `DEBUGGER_PG_STRICT=true` also gates new work whenever PG is not
connected. These switches are independent. Neither is a guarantee of faster
model generation; measure the intended workload.

The default journal is `pending/<instance_id>/capture.sqlite` below the debugger
state directory. An explicit path must use its own empty directory when first
created. The directory is restricted to its owner (and SYSTEM on Windows), so
SQLite sidecars inherit private permissions. UNC/mapped network paths and
aliases of historical SQLite are refused. Always use local storage, not NAS/NFS.
One local owner lease also excludes a second engine/importer while PG is down.

The journal uses one SQLite connection, exclusive locking, DELETE journal mode
and `synchronous=EXTRA`. Its version/application marker is checked before a
writable open; normal SQLite rollback recovery handles a killed writer. Hashes
and continuity are then verified. Corruption, incompatible versions, store
replacement, missing parents/prefixes and conflicting trace overlap stop replay.
No automatic reset, truncation, silent backend switch or age-based deletion is
performed. The byte quota covers pending encoded payload; SQLite pages,
indexes and its rollback journal additionally consume disk space. A 256 MiB
free-space reserve and an estimate of transaction headroom are checked before
append. Exhaustion can still require operator action; the journal is retained.

PostgreSQL must have `fsync=on`; the writer sets `synchronous_commit=on`.
Large fields use transaction-scoped PostgreSQL large objects, unlinked before
commit or removed by rollback. No server filesystem import/export is used.
These guarantees rely on the database, OS and storage honoring their durability
settings; they are not a backup against loss of the machine or database.

### Process shutdown

The default guarantee is recovery from a PG outage while the engine remains
alive. If the engine is killed, reboots, or loses power, volatile RAM may be lost.
An orderly logger close attempts a drain and, when enabled, spill. This does
**not** change `main.py`'s Windows Ctrl+C/SIGBREAK handler, which forces
`os._exit(0)`; that path is an abrupt process exit, not a graceful flush.

PG records a clean-shutdown receipt only after a successful drain. A subsequent
start without that receipt reports `loss_unknown`: earlier volatile history may
be missing, with no invented exact loss count. The flag remains historical;
`complete` describes the current captured cut, not proof that a previous process
lost nothing. A journal can preserve its confirmed prefix even when additional
RAM was lost. The empty confirmed journal is retired before newer RAM is applied,
so its stale watermark cannot turn that accepted volatility into a false conflict.

### Inspect, retry and manually recover

Use the environment that runs GranSabio, with the same private `.env` and stable
`DEBUGGER_INSTANCE_ID`. Do not put a database password on the command line.

```sh
python debugger_cli.py status
python debugger_cli.py retry
```

These call the running engine's internal status/retry endpoints; `--url` selects
another explicitly managed engine. Retry wakes the existing writer. A consistency
conflict remains blocked and must be understood before recovery.

Normal recovery is automatic. For a separately preserved journal, stop the
writer for that installation, hash the source, verify its expected destination,
and run:

```sh
python debugger_cli.py recover /private/received/capture.sqlite --sha256 EXPECTED_SHA256
```

The command refuses a live local owner. It copies the source and any rollback
journal to a new private recovery directory, verifies transfer hashes, and uses
the normal version checks, target/store identity, owner fence and applier.
The source is never opened writable or removed. Re-importing an already
confirmed journal proves its prefix and does not duplicate data. A failed import
retains its working copy and a `receipt.json`; use that copy to continue a partial
recovery, without discarding the original. Credentials/URIs/payloads are absent
from the receipt. Changing an instance ID or resetting its schema to force an
import is not a supported repair.
