# Cooperating applications sharing a GPU

GranSabio can temporarily stop its local inference dispatches so another
application can use the same physical GPU. This is cooperative ownership, not
an operating-system GPU lock or a VRAM quota. Every participating application
must use the protocol; direct calls bypassing it are outside the guarantee.

The engine does not launch image jobs or depend on ComfyUI. It requires an
external service implementing the release acknowledgment below. The included
ComfyUI patch demonstrates how to connect that protocol to a real worker.

## Configure the service owner

Copy the root `local_resources.example.json` and replace its placeholders with
the actual GPU UUID, existing local endpoint IDs and `external_service_url`.
The URL is administrative configuration; client requests cannot override it.
Set `LOCAL_RESOURCES_CONFIG_PATH` to that file and keep
`LOCAL_RESOURCES_STATE_PATH` on persistent local disk. Run one GranSabio
coordinator process for that state file. The feature is disabled by default.

Native model unloading currently supports Ollama. A reservation refuses linked
adapters it cannot unload. An Ollama-only resource can omit the external service
for manual model management, but an external reservation requires its verified
release protocol. Other GPUs and cloud providers remain independent.

## The calling application

1. Read `GET /local-resources` and find the configured resource. Persist an
   acquire UUID **before** sending its request.
2. `POST /local-resources/{id}/reserve` with
   `{"request_id":"<acquire-uuid>","purpose":"video-render"}`. The purpose is
   a descriptive label, not a backend selector. GranSabio closes new local
   admission, drains admitted work, unloads Ollama and verifies the result.
3. HTTP 202 means pending: retry the identical request within the application's
   deadline, honoring `Retry-After`. Retry only `resource_busy` among 409 errors.
   Submit external work only after HTTP 200, `complete: true`, `state: "reserved"`
   and `reservation.ready: true`.
4. After the external work and any cancellations actually finish, persist a
   fresh release UUID and `POST /local-resources/{id}/release` with
   `{"request_id":"<release-uuid>","reservation_id":"<acquire-uuid>"}`.
   Retry this same body while pending. GranSabio reopens local admission only
   after the external service acknowledges completed memory cleanup.

The reservation ID always equals the acquire UUID, even while pending or after
a lost response. Release can also cancel a pending acquire without interrupting
admitted inference. An already-started native unload must be verified first.
A canceled/released acquire cannot be reused to start new external work.

Reservations never expire automatically. A lost response is not permission to
invent another operation ID or assume release. On an uncertain inference,
service restart or failed cleanup, retain ownership evidence and reconcile the
service before retrying. There is deliberately no unconditional force-release.

Manual cleanup uses `POST /local-resources/{id}/unload` with a fresh `request_id`
and either `target: "ollama"`, `endpoint_id`, optional exact `model`, or
`target: "external"`. It has the same 200/202 completion envelope and refuses
busy work. It never cancels jobs to free the resource.

## External service contract

Implement these three routes at `external_service_url`:

| Route | Behavior |
| --- | --- |
| `GET /gransabio/resource-barrier` | Current process/GPU identity, idle status and verified memory-release status. |
| `POST /gransabio/resource-barrier/release` | Accept only `{"request_id":"<uuid>"}`. Under the worker's queue lock, refuse active/queued work with 409 or schedule native cleanup. Repeat IDs identify the same operation. |
| `GET /gransabio/resource-barrier/{request_id}` | 404 for an unknown ID; otherwise that operation's state together with **current** process/GPU/queue/memory status. |

Status responses contain:

```json
{
  "protocol": "gransabio.gpu-barrier.v1",
  "instance_id": "new-uuid-for-each-service-process",
  "gpu_uuid": "GPU-actual-device-uuid",
  "idle": true,
  "memory_released": true
}
```

An operation response also includes `request_id` and `state` (`pending`,
`completed` or `failed`). POST returns JSON with 202 while pending, 200 when
completed. The worker may mark completion only after its real model unload,
executor/cache reset, collection and device synchronization have succeeded.
An HTTP acceptance response alone is never proof. New work invalidates the
memory receipt, even if that work is subsequently removed from the queue.

`memory_released` describes completion of this service's owned cleanup, not zero
physical VRAM usage. A process identity change invalidates an earlier receipt.
GranSabio verifies initial idle/released state before the first local dispatch;
use a verified manual external unload to establish it after a fresh service start.

Compatibility with the first ComfyUI consumer is retained: configuration may
use `comfyui_url` instead of `external_service_url` (not both), unload accepts
`target: "comfyui"`, and `purpose: "comfyui"` remains valid. Inventory includes
legacy `comfyui_url`/`comfyui` aliases; reservations retain `comfy_instance_id`
alongside `service_instance_id`. These aliases do not select different behavior.

## ComfyUI example

[`comfyui-0.37.0.patch`](comfyui-0.37.0.patch) targets ComfyUI 0.37.0 at revision
`b33e2b55cae074eca5aec96283cceac19aa249ba`. Base/result SHA-256 values are in
[`comfyui-source-manifest.json`](comfyui-source-manifest.json). It changes the
queue, native worker and routes, and adds `gransabio_resource_barrier.py`.
It requires exactly one visible CUDA device and obtains its physical UUID.
The example patch is distributed under ComfyUI's GPLv3 license, included as
[`COMFYUI-LICENSE.txt`](COMFYUI-LICENSE.txt). Upstream source:
[`Comfy-Org/ComfyUI`](https://github.com/Comfy-Org/ComfyUI/tree/b33e2b55cae074eca5aec96283cceac19aa249ba).

Review the patch against the installed version before applying it. For example,
from the ComfyUI checkout, with an absolute path to this example:

```text
git apply --check /path/to/GranSabio_LLM/examples/gpu_reservations/comfyui-0.37.0.patch
git apply /path/to/GranSabio_LLM/examples/gpu_reservations/comfyui-0.37.0.patch
```

Application and restart are explicit operator actions during an agreed idle
window. GranSabio never installs or restarts ComfyUI. Other versions may require
porting the worker hook; do not replace it with sleeps or free-VRAM heuristics.
The patch and protocol have offline tests. A bounded live check on the target
ComfyUI revision also verified Ollama unload on reserve, inference exclusion,
native VAE encode/decode on CUDA, worker-confirmed memory release and subsequent
GranSabio local generation. That check does not certify arbitrary workflows or
other ComfyUI versions; validate device placement and cleanup on each installation.
