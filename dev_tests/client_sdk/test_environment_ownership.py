"""Offline subprocess regressions for SDK imports and executable demo bootstrap."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("configured", [False, True])
def test_sdk_import_leaves_host_environment_untouched(configured):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env.pop("GRANSABIO_BASE_URL", None)
    env.pop("GRANSABIO_API_KEY", None)
    if configured:
        env.update(GRANSABIO_BASE_URL="http://caller.invalid:9999/", GRANSABIO_API_KEY="unit-caller-key")
    script = r'''import json, os
import dotenv
before = dict(os.environ)
def forbidden(*args, **kwargs):
    raise AssertionError("SDK import must not load an environment file")
dotenv.load_dotenv = forbidden
from client import GranSabioClient, AsyncGranSabioClient
assert before == dict(os.environ)
for cls in (GranSabioClient, AsyncGranSabioClient):
    obj = cls()
    assert obj.base_url == os.environ.get("GRANSABIO_BASE_URL", "http://localhost:8000").rstrip("/")
    assert obj.api_key == os.environ.get("GRANSABIO_API_KEY")
    explicit = cls(base_url="http://explicit.invalid:8888/", api_key="unit-explicit-key")
    assert explicit.base_url == "http://explicit.invalid:8888"
    assert explicit.api_key == "unit-explicit-key"
assert before == dict(os.environ)
print(json.dumps({"environment_unchanged": True}))
'''
    result = subprocess.run([sys.executable, "-c", script], env=env, cwd=ROOT,
                            capture_output=True, text=True, check=True, timeout=20)
    assert json.loads(result.stdout) == {"environment_unchanged": True}


@pytest.mark.parametrize("external", [None, "http://caller.invalid:9999"])
def test_all_demo_entry_points_load_explicit_env_without_overwriting(tmp_path, external):
    fixture = tmp_path / ".env"
    fixture.write_text("GRANSABIO_BASE_URL=http://file.invalid:8888\n", encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env.pop("PYTHON_DOTENV_DISABLED", None)
    env.pop("GRANSABIO_BASE_URL", None)
    if external:
        env["GRANSABIO_BASE_URL"] = external
    script = r'''import json, os, runpy, socket, sys
from pathlib import Path
import dotenv
root = Path(sys.argv[1])
fixture = Path(sys.argv[2])
calls = []
original = dotenv.load_dotenv
def observed(path=None, *args, **kwargs):
    calls.append({"path": str(path), "override": kwargs.get("override")})
    assert path == root / ".env" and kwargs == {"override": False}
    return original(fixture, *args, **kwargs)
def no_network(*args, **kwargs):
    raise AssertionError("Demo import must not send requests")
socket.socket.connect = no_network
socket.socket.connect_ex = no_network
socket.getaddrinfo = no_network
dotenv.load_dotenv = observed
paths = sorted((root / "demos").glob("[0-9][0-9]_*.py"))
for path in paths:
    runpy.run_path(str(path), run_name="demo_import")
print(json.dumps({"loads": len(calls), "demos": len(paths), "endpoint": os.environ["GRANSABIO_BASE_URL"]}))
'''
    result = subprocess.run([sys.executable, "-c", script, str(ROOT), str(fixture)], env=env,
                            cwd=tmp_path, capture_output=True, text=True, check=True, timeout=20)
    assert json.loads(result.stdout) == {
        "loads": 12, "demos": 12, "endpoint": external or "http://file.invalid:8888",
    }
