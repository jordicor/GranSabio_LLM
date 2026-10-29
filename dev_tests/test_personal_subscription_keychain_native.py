"""Opt-in macOS tests: only fake accounts and keys owned by temporary stores.

Run with GRANSABIO_TEST_MACOS_KEYCHAIN=1. Never lock the user's real Keychain.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import personal_subscription as ps

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or os.getenv("GRANSABIO_TEST_MACOS_KEYCHAIN") != "1",
    reason="Requires explicit opt-in to temporary native macOS Keychain entries",
)


@pytest.fixture
def native_stores(tmp_path, monkeypatch):
    # Demonstrate that user-wide automatic backend overrides cannot disable protection.
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")
    backend = ps._macos_keyring()
    stores = []
    def create(name):
        store = ps.CredentialStore(tmp_path / name)
        assert backend.get_password(ps.KEYCHAIN_SERVICE, store.keychain_account) is None
        stores.append(store)
        return store
    yield create, backend
    for store in stores:
        store.close()
        if backend.get_password(ps.KEYCHAIN_SERVICE, store.keychain_account) is not None:
            backend.delete_password(ps.KEYCHAIN_SERVICE, store.keychain_account)
        assert backend.get_password(ps.KEYCHAIN_SERVICE, store.keychain_account) is None


@pytest.fixture
def fake_state():
    return {"host_id": "urn:uuid:native-test-only", "active": "a", "accounts": {
        "a": {"client_id": "fake-native-a", "refresh_token": "fake-native-refresh-a"},
        "b": {"client_id": "fake-native-b", "refresh_token": "fake-native-refresh-b"},
    }}


def test_native_write_update_and_read_in_another_process(native_stores, fake_state):
    create, _ = native_stores
    store = create("persistent")
    store.write(fake_state)
    fake_state["accounts"]["a"]["refresh_token"] = "fake-native-rotated"
    store.write(fake_state)
    assert store.read() == fake_state
    assert b"fake-native" not in store.path.read_bytes()
    store.close()
    program = """
import json, sys
from pathlib import Path
from personal_subscription import CredentialStore
payload = json.load(sys.stdin)
store = CredentialStore(Path(payload['directory']))
try:
    assert store.read() == payload['expected']
finally:
    store.close()
print('verified')
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        input=json.dumps({"directory": str(store.directory), "expected": fake_state}),
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "verified"


def test_native_stores_cannot_read_each_others_ciphertext(native_stores, fake_state):
    create, _ = native_stores
    first, second = create("first"), create("second")
    first.write(fake_state)
    second.write({"host_id": "other-fake-host", "accounts": {}, "active": None})
    second.path.write_bytes(first.path.read_bytes())
    with pytest.raises(ps.SubscriptionError, match="authentication"):
        second.read()
    assert first.read() == fake_state


def test_native_deleted_key_refuses_read_and_write_without_replacement(native_stores, fake_state):
    create, backend = native_stores
    store = create("missing-key")
    store.write(fake_state)
    original = store.path.read_bytes()
    backend.delete_password(ps.KEYCHAIN_SERVICE, store.keychain_account)
    for operation in (store.read, lambda: store.write(fake_state)):
        with pytest.raises(ps.SubscriptionError, match="missing"):
            operation()
    assert backend.get_password(ps.KEYCHAIN_SERVICE, store.keychain_account) is None
    assert store.path.read_bytes() == original


def test_native_security_api_denial_is_explicit_and_preserves_file(native_stores, fake_state, monkeypatch):
    from keyring.backends.macOS import api

    create, _ = native_stores
    store = create("denied")
    store.write(fake_state)
    original = store.path.read_bytes()
    # Inject Apple's locked/no-interaction result at the actual ctypes boundary.
    # Locking the login Keychain would interfere with the user's other applications.
    with monkeypatch.context() as patch:
        patch.setattr(api, "SecItemCopyMatching", lambda *args: -25308)
        for operation in (store.read, lambda: store.write(fake_state)):
            with pytest.raises(ps.SubscriptionError, match="Unlock/allow"):
                operation()
    assert store.path.read_bytes() == original
    assert store.read() == fake_state


def test_native_migration_command_respects_lock_and_preserves_accounts(native_stores, fake_state):
    create, _ = native_stores
    store = create("migration")
    original = json.dumps(fake_state).encode()
    store.path.write_bytes(original)
    store.path.chmod(0o600)
    command = [sys.executable, str(Path(ps.__file__)), "migrate-keychain", "--directory", str(store.directory)]
    blocked = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert blocked.returncode == 1
    assert "one engine process" in blocked.stderr
    assert store.path.read_bytes() == original
    store.close()
    migrated = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert migrated.returncode == 0, migrated.stderr
    assert migrated.stdout.strip() == "Migration verified and completed."
    reopened = ps.CredentialStore(store.directory)
    try:
        assert reopened.read() == fake_state
        assert b"fake-native" not in reopened.path.read_bytes()
        assert reopened.migrate_to_keychain() is False
    finally:
        reopened.close()
