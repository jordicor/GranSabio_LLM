"""Test the helper shipped in the public patch without importing ComfyUI/CUDA."""

from pathlib import Path
from threading import RLock
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest


@pytest.fixture
def example():
    patch = (Path(__file__).resolve().parents[1] / "examples/gpu_reservations/comfyui-0.37.0.patch").read_text(encoding="utf-8")
    helper_diff = patch.split("+++ b/gransabio_resource_barrier.py\n", 1)[1]
    source = "\n".join(line[1:] for line in helper_diff.splitlines() if line.startswith("+"))
    module = ModuleType("public_barrier_example")
    exec(compile(source, "<public-comfyui-helper>", "exec"), module.__dict__)
    flags, models = {}, ["example-model"]
    queue = SimpleNamespace(mutex=RLock(), queue=[], currently_running={}, set_flag=flags.__setitem__)
    barrier = module.ResourceBarrier(queue, gpu_uuid=lambda: "GPU-example", loaded_models=lambda: list(models))
    return SimpleNamespace(module=module, barrier=barrier, queue=queue, flags=flags, models=models)


def test_public_example_ack_requires_real_cleanup_and_invalidates_on_new_work(example):
    request_id = str(uuid4())
    pending = example.barrier.request_release(request_id)
    assert pending["state"] == "pending" and not pending["memory_released"]
    assert example.flags == {"gransabio_release_request": request_id, "unload_models": True, "free_memory": True}
    example.models.clear()  # The native worker performs this before acknowledging.
    example.barrier.complete(request_id)
    assert example.barrier.status(request_id)["memory_released"]
    example.barrier.work_queued()
    assert not example.barrier.status(request_id)["memory_released"]


def test_public_example_refuses_busy_queue_and_failed_cleanup(example):
    example.queue.queue.append("running-later")
    with pytest.raises(example.module.ResourceBarrierBusy):
        example.barrier.request_release(str(uuid4()))
    assert not example.flags
    example.queue.queue.clear()
    request_id = str(uuid4())
    example.barrier.request_release(request_id)
    example.barrier.complete(request_id)  # Models still resident: no success receipt.
    result = example.barrier.status(request_id)
    assert result["state"] == "failed" and not result["memory_released"]
