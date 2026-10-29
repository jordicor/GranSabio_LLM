"""Storage regressions with fake keys; no real Keychain or credentials."""

import json
import os
from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet

import personal_subscription as ps


@pytest.fixture(autouse=True)
def macos(monkeypatch, personal_keychain):
    monkeypatch.setattr(ps, "sys", SimpleNamespace(platform="darwin"))


@pytest.fixture
def store(tmp_path):
    result = ps.CredentialStore(tmp_path / "private")
    yield result
    result.close()


@pytest.fixture
def state():
    return {"host_id": "urn:uuid:fake-host", "active": "first", "accounts": {
        "first": {"client_id": "fake-client-a", "access_token": "fake-a-access", "refresh_token": "fake-a-refresh"},
        "second": {"client_id": "fake-client-b", "access_token": "fake-b-access", "refresh_token": "fake-b-refresh"},
    }}


def legacy(store, state):
    data = json.dumps(state).encode()
    store.path.write_bytes(data)
    store.path.chmod(0o600)
    return data


def test_ciphertext_persists_accounts_and_rotates_only_selected_tokens(store, state, personal_keychain):
    store.write(state)
    assert store.path.read_bytes().startswith(ps.KEYCHAIN_HEADER)
    assert b"fake-a-refresh" not in store.path.read_bytes()
    assert store.read() == state
    key = dict(personal_keychain.entries)
    state["accounts"]["first"]["refresh_token"] = "rotated-fake"
    store.write(state)
    assert store.read() == state
    assert personal_keychain.entries == key
    if os.name != "nt":
        assert store.path.stat().st_mode & 0o077 == 0
        assert store.directory.stat().st_mode & 0o077 == 0


def test_legacy_requires_explicit_migration_and_preserves_all_fields(store, state, personal_keychain):
    original = legacy(store, state)
    for operation in (store.read, lambda: store.write(state)):
        with pytest.raises(ps.SubscriptionError, match="migrate-keychain"):
            operation()
    assert store.path.read_bytes() == original
    assert not personal_keychain.entries
    assert store.migrate_to_keychain() is True
    assert store.read() == state
    assert store.migrate_to_keychain() is False
    assert b"fake-a-refresh" not in store.path.read_bytes()
    assert not list(store.directory.glob("chatgpt-accounts-*"))


@pytest.mark.parametrize("operation", ["read", "write"])
def test_missing_key_never_regenerates_or_overwrites_existing_storage(store, state, personal_keychain, operation):
    store.write(state)
    original = store.path.read_bytes()
    personal_keychain.entries.clear()
    with pytest.raises(ps.SubscriptionError, match="missing"):
        store.read() if operation == "read" else store.write(state)
    assert not personal_keychain.entries
    assert store.path.read_bytes() == original


@pytest.mark.parametrize("stage", ["read", "create", "verify"])
def test_keychain_failure_keeps_legacy_source_private(store, state, personal_keychain, monkeypatch, stage):
    original = legacy(store, state)
    def fail(*args):
        raise RuntimeError("sensitive-provider-details-must-not-escape")
    if stage == "read":
        monkeypatch.setattr(personal_keychain, "get_password", fail)
    elif stage == "create":
        monkeypatch.setattr(personal_keychain, "set_password", fail)
    else:
        monkeypatch.setattr(personal_keychain, "set_password", lambda *args: None)
    with pytest.raises(ps.SubscriptionError) as error:
        store.migrate_to_keychain()
    assert "sensitive-provider-details" not in str(error.value)
    assert store.path.read_bytes() == original
    assert not list(store.directory.glob("chatgpt-accounts-*"))


def test_failed_atomic_replace_preserves_source_and_migration_can_retry(store, state, monkeypatch):
    original = legacy(store, state)
    with monkeypatch.context() as patch:
        def fail(*args):
            raise OSError("simulated replacement failure")
        patch.setattr(ps.os, "replace", fail)
        with pytest.raises(OSError):
            store.migrate_to_keychain()
    assert store.path.read_bytes() == original
    assert not list(store.directory.glob("chatgpt-accounts-*"))
    assert store.migrate_to_keychain() is True
    assert store.read() == state


@pytest.mark.parametrize("damage", ["ciphertext", "key"])
def test_authentication_failure_does_not_return_accounts(store, state, personal_keychain, damage):
    store.write(state)
    if damage == "ciphertext":
        original = store.path.read_bytes()
        store.path.write_bytes(original[:-12] + b"tampered-end")
    else:
        personal_keychain.entries[ps.KEYCHAIN_SERVICE, store.keychain_account] = Fernet.generate_key().decode()
    with pytest.raises(ps.SubscriptionError, match="authentication"):
        store.read()


def test_store_keys_are_isolated_and_copying_ciphertext_cannot_open_other_accounts(store, state, tmp_path, personal_keychain):
    other = ps.CredentialStore(tmp_path / "other")
    try:
        store.write(state)
        other.write({"host_id": "other-host", "accounts": {}, "active": None})
        assert len(personal_keychain.entries) == 2
        assert store.keychain_account != other.keychain_account
        other.path.write_bytes(store.path.read_bytes())
        with pytest.raises(ps.SubscriptionError, match="authentication"):
            other.read()
        assert store.read() == state
    finally:
        other.close()


def test_invalid_legacy_state_is_not_migrated(store, personal_keychain):
    original = legacy(store, {"accounts": []})
    with pytest.raises(ps.SubscriptionError, match="invalid"):
        store.migrate_to_keychain()
    assert store.path.read_bytes() == original
    assert not personal_keychain.entries


@pytest.mark.skipif(os.name == "nt", reason="Unix permissions")
def test_legacy_permissions_are_checked_before_keychain_access(store, state, personal_keychain):
    legacy(store, state)
    store.path.chmod(0o644)
    with pytest.raises(ps.SubscriptionError, match="0600"):
        store.migrate_to_keychain()
    assert not personal_keychain.entries


def test_unavailable_keychain_does_not_create_plaintext(store, state, monkeypatch):
    def unavailable():
        raise ps.SubscriptionError("macOS Keychain is unavailable")
    monkeypatch.setattr(ps, "_macos_keyring", unavailable)
    with pytest.raises(ps.SubscriptionError, match="unavailable"):
        store.write(state)
    assert not store.path.exists()


@pytest.mark.skipif(os.name == "nt", reason="Linux storage uses POSIX permissions")
def test_linux_plaintext_behavior_stays_unchanged(store, state, monkeypatch):
    monkeypatch.setattr(ps, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(ps, "_macos_keyring", lambda: pytest.fail("Linux must not open Keychain"))
    store.write(state)
    assert json.loads(store.path.read_bytes()) == state
    assert store.read() == state
    with pytest.raises(ps.SubscriptionError, match="only on macOS"):
        store.migrate_to_keychain()


def test_windows_still_dispatches_through_dpapi(store, state, monkeypatch):
    calls = []
    def dpapi(data, *, decrypt=False):
        calls.append(decrypt)
        return data.removeprefix(b"fake-dpapi:") if decrypt else b"fake-dpapi:" + data
    monkeypatch.setattr(ps, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(ps, "os", SimpleNamespace(
        name="nt", fdopen=os.fdopen, fsync=os.fsync, replace=os.replace))
    monkeypatch.setattr(ps, "_windows_protect", dpapi)
    store.write(state)
    assert store.read() == state
    assert calls == [False, True]
