"""Local-owner ChatGPT plan access through OpenAI's public SIWC OAuth flow.

No credentials enter the model catalog. Hosted callers can permanently deny
this capability for their request context, including detached child tasks.
"""

from __future__ import annotations

import asyncio
import base64
import ctypes
import hashlib
import os
import secrets
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from ctypes import wintypes
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit
from uuid import uuid4

import httpx
from cryptography.fernet import Fernet, InvalidToken

import json_utils as json

PROVIDER = "chatgpt"
MODEL_PREFIX = "chatgpt/"
ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = ISSUER + "/api/accounts/authorize"
TOKEN_URL = ISSUER + "/api/accounts/oauth/token"
JWKS_URL = ISSUER + "/.well-known/jwks.json"
DISCOVERY_URL = ISSUER + "/.well-known/openid-configuration"
API_URL = "https://api.openai.com/v1"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke " + PLAN_SCOPE
KEYCHAIN_SERVICE = "org.gransabio.personal.chatgpt.file-key.v1"
KEYCHAIN_HEADER = b"GRANSABIO-KEYCHAIN-V1\n"
personal_usage_blocked: ContextVar[bool] = ContextVar("personal_usage_blocked", default=False)


class SubscriptionError(ValueError):
    """A user-actionable refusal that must not fall back to paid API access."""


def subscriptions_allowed(settings: Any) -> bool:
    """Require operator opt-in and a personal execution context."""
    return bool(getattr(settings, "PERSONAL_SUBSCRIPTIONS_ENABLED", False)) and not personal_usage_blocked.get()


def require_personal_usage(settings: Any) -> None:
    """Reject disabled or hosted access before reading any owner credentials."""
    if not subscriptions_allowed(settings):
        raise SubscriptionError("Personal subscriptions are unavailable in this execution context.")


def _windows_protect(data: bytes, *, decrypt: bool = False) -> bytes:
    """Use the current Windows user's DPAPI identity, never machine-wide keys."""
    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]

    buffer = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    result = Blob()
    function = ctypes.windll.crypt32.CryptUnprotectData if decrypt else ctypes.windll.crypt32.CryptProtectData
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result)):
        raise SubscriptionError("Windows could not protect or open this user's ChatGPT credentials.")
    try:
        return ctypes.string_at(result.data, result.size)
    finally:
        ctypes.windll.kernel32.LocalFree(ctypes.cast(result.data, ctypes.c_void_p))


def _macos_keyring():
    """Use only the native backend, ignoring automatic/fallback backend selection."""
    try:
        from keyring.backends.macOS import Keyring

        backend = Keyring()
        backend.keychain = None  # Use the user's default Keychain, not KEYCHAIN_PATH.
        if backend.priority <= 0:
            raise RuntimeError()
        return backend
    except Exception:
        raise SubscriptionError("macOS Keychain is unavailable. Install the macOS requirements and retry.") from None


class CredentialStore:
    """Atomic owner storage; one process owns the rotating refresh credentials."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            directory.chmod(0o700)
        self.path = directory / "chatgpt-accounts.dat"
        self.keychain_account = hashlib.sha256(str(self.path.resolve()).encode("utf-8")).hexdigest()
        self.lock = (directory / "chatgpt-accounts.lock").open("a+b")
        self.lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.lock.close()
            raise SubscriptionError("ChatGPT personal access requires one engine process per credential directory.") from exc

    def read(self) -> dict:
        """Return saved registrations without printing their contents."""
        if not self.path.exists():
            return {"host_id": "urn:uuid:" + str(uuid4()), "accounts": {}, "active": None}
        data = self._read_bytes()
        if sys.platform == "darwin":
            self._require_keychain_format(data)
            try:
                data = self._macos_cipher().decrypt(data[len(KEYCHAIN_HEADER):])
            except InvalidToken:
                raise SubscriptionError("ChatGPT credentials failed authentication; restore the original file and Keychain key.") from None
        elif os.name == "nt":
            data = _windows_protect(data, decrypt=True)
        return self._decode_state(data)

    def _read_bytes(self) -> bytes:
        if os.name != "nt" and self.path.stat().st_mode & 0o077:
            raise SubscriptionError("ChatGPT credentials must have owner-only permissions (0600).")
        return self.path.read_bytes()

    @staticmethod
    def _decode_state(data: bytes) -> dict:
        try:
            state = json.loads(data)
            if not isinstance(state["accounts"], dict) or not isinstance(state["host_id"], str):
                raise ValueError()
            return state
        except (ValueError, KeyError, TypeError) as exc:
            raise SubscriptionError("ChatGPT credential storage is invalid; restore its private backup.") from exc

    @staticmethod
    def _require_keychain_format(data: bytes) -> None:
        if not data.startswith(KEYCHAIN_HEADER):
            raise SubscriptionError(
                "Legacy or unrecognized ChatGPT storage. For a legacy JSON store, stop the engine and run "
                "personal_subscription.py migrate-keychain --directory <credential-directory>."
            )

    def _macos_cipher(self, *, create: bool = False) -> Fernet:
        """Never cache the key or regenerate it for an existing encrypted file."""
        backend = _macos_keyring()
        try:
            key = backend.get_password(KEYCHAIN_SERVICE, self.keychain_account)
            if key is None and create:
                generated = Fernet.generate_key().decode("ascii")
                backend.set_password(KEYCHAIN_SERVICE, self.keychain_account, generated)
                key = backend.get_password(KEYCHAIN_SERVICE, self.keychain_account)
                if key is None or not secrets.compare_digest(key, generated):
                    raise SubscriptionError("macOS Keychain did not persist the encryption key; credentials were not replaced.")
            if key is None:
                raise SubscriptionError(
                    "The ChatGPT encryption key is missing from macOS Keychain. "
                    "Restore the original Keychain key and credential directory; no replacement key was created."
                )
            return Fernet(key.encode("ascii"))
        except SubscriptionError:
            raise
        except Exception:
            raise SubscriptionError(
                "macOS Keychain could not open or save the ChatGPT encryption key. "
                "Unlock/allow access to your Keychain and retry; plaintext storage is not permitted."
            ) from None

    def write(self, state: dict) -> None:
        """Replace tokens, expiry, scopes and catalog as one protected record."""
        data = json.dumps(state).encode("utf-8")
        if sys.platform == "darwin":
            exists = self.path.exists()
            if exists:
                self._require_keychain_format(self._read_bytes())
            data = KEYCHAIN_HEADER + self._macos_cipher(create=not exists).encrypt(data)
        elif os.name == "nt":
            data = _windows_protect(data)
        self._atomic_write(data)

    def migrate_to_keychain(self) -> bool:
        """Explicit, one-time conversion; preserve the source on any failure."""
        if sys.platform != "darwin":
            raise SubscriptionError("Keychain migration is available only on macOS.")
        data = self._read_bytes()
        if data.startswith(KEYCHAIN_HEADER):
            self.read()  # Confirm that an already migrated store is still readable.
            return False
        self._decode_state(data)
        encrypted = self._macos_cipher(create=True).encrypt(data)
        try:
            verified = self._macos_cipher().decrypt(encrypted)
        except InvalidToken:
            raise SubscriptionError("Keychain migration verification failed; the source was not replaced.") from None
        if verified != data:
            raise SubscriptionError("Keychain migration verification failed; the source was not replaced.")
        self._atomic_write(KEYCHAIN_HEADER + encrypted)
        return True

    def _atomic_write(self, data: bytes) -> None:
        fd, name = tempfile.mkstemp(prefix="chatgpt-accounts-", dir=self.directory)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(name, self.path)
        finally:
            Path(name).unlink(missing_ok=True)

    def close(self) -> None:
        """Release ownership without erasing the user's registration."""
        self.lock.close()


class PersonalSubscription:
    """One local user's saved accounts, explicit selection and OAuth lifecycle."""

    def __init__(self, store: CredentialStore):
        self.store = store
        self.state = store.read()
        self.pending: dict[str, dict] = {}
        self.lock = asyncio.Lock()
        self.active_calls = 0
        self.jwks: dict = {}
        self.jwks_loaded = 0.0

    def _idle(self) -> None:
        if self.active_calls:
            raise SubscriptionError("Wait for active ChatGPT requests before changing the connected account.")

    def summary(self) -> dict:
        """Expose only labels and connection state; never token material."""
        return {"active": self.state.get("active"), "accounts": [
            {"id": key, "label": f"{item.get('email') or 'ChatGPT account'} ({key[:8]})",
             "connected": bool(item.get("access_token")), "models": len(item.get("models", []))}
            for key, item in self.state["accounts"].items()
        ]}

    def catalog(self) -> dict[str, dict]:
        """Provide account-derived, namespaced models without changing model_specs.json."""
        account = self.state["accounts"].get(self.state.get("active"), {})
        if not account.get("access_token") or PLAN_SCOPE not in account.get("scopes", []):
            return {}
        result = {}
        for item in account.get("models", []):
            model = item.get("slug")
            if not isinstance(model, str) or not model or item.get("visibility") != "list":
                continue
            key = MODEL_PREFIX + model
            result[key] = {
                "model_id": model, "name": f"{item.get('display_name') or model} (ChatGPT subscription)",
                "description": "Personal ChatGPT plan; account usage limits apply.", "enabled": True,
                "capabilities": ["text", "streaming"],
                "provider_capabilities": {"text": True, "streaming": True, "json_object": False,
                                          "json_schema": False, "tool_calling": False, "logprobs": False,
                                          "vision_input": False},
                "supported_parameters": [], "pricing": {
                    "input_per_million": 0, "output_per_million": 0,
                    "cached_input_per_million": 0, "cache_write_per_million": 0,
                },
                "billing_mode": "subscription",
            }
        return result

    def begin_login(self, redirect_uri: str, account_id: str | None = None) -> tuple[str, str]:
        """Create a short-lived, browser-bound PKCE and nonce transaction."""
        self._idle()
        now = time.time()
        self.pending = {key: value for key, value in self.pending.items() if value["expires"] > now}
        if len(self.pending) >= 4:
            raise SubscriptionError("A sign-in is already pending. Finish it or wait for it to expire.")
        account = self.state["accounts"].get(account_id) if account_id else None
        if account_id and account is None:
            raise SubscriptionError("Unknown saved ChatGPT account.")
        state, nonce, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(32), secrets.token_urlsafe(64)
        client_id = account["client_id"] if account else "dynamic_agent_client"
        transaction = {"nonce": nonce, "verifier": verifier, "redirect_uri": redirect_uri,
                       "client_id": client_id, "account_id": account_id, "expires": now + 600}
        self.pending[state] = transaction
        self.store.write(self.state)  # Persist the host ID before first authorization.
        params = {"client_id": client_id, "ext_agent_host_id": self.state["host_id"],
                  "response_type": "code", "redirect_uri": redirect_uri, "scope": SCOPES,
                  "resource": API_URL, "state": state, "nonce": nonce, "code_challenge_method": "S256",
                  "code_challenge": base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")}
        if account is None:
            params["agent_name_hint"] = "GranSabio LLM"
        elif account.get("id_token"):
            params["id_token_hint"] = account["id_token"]
        return state, AUTHORIZE_URL + "?" + urlencode(params)

    async def _post_token(self, data: dict) -> dict:
        async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
            response = await client.post(TOKEN_URL, data=data)
        if response.status_code != 200:
            raise SubscriptionError("ChatGPT authorization could not be renewed. Sign in again.")
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("access_token"), str):
            raise SubscriptionError("ChatGPT returned an invalid authorization response.")
        return payload

    async def _identity(self, token: str, client_id: str, nonce: str) -> dict:
        import jwt

        try:
            header = jwt.get_unverified_header(token)
            if header.get("alg") != "RS256":
                raise ValueError()
            key = next((key for key in self.jwks.get("keys", []) if key.get("kid") == header.get("kid")), None)
            if key is None or time.time() - self.jwks_loaded > 3600:
                async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
                    response = await client.get(JWKS_URL)
                    response.raise_for_status()
                    self.jwks = response.json()
                self.jwks_loaded = time.time()
                key = next(key for key in self.jwks["keys"] if key.get("kid") == header.get("kid"))
            identity = jwt.decode(token, jwt.PyJWK.from_dict(key).key, algorithms=["RS256"],
                                  audience=client_id, issuer=ISSUER, leeway=5,
                                  options={"require": ["sub", "exp", "iat", "nonce"]})
            if not secrets.compare_digest(str(identity["nonce"]), nonce) or not identity["sub"]:
                raise ValueError()
            return identity
        except (jwt.PyJWTError, ValueError, KeyError, StopIteration, TypeError) as exc:
            raise SubscriptionError("ChatGPT identity could not be verified. Start sign-in again.") from exc

    @staticmethod
    def _credentials(tokens: dict, previous: dict | None = None) -> dict:
        previous = previous or {}
        scopes = tokens.get("scope", " ".join(previous.get("scopes", []))).split()
        if PLAN_SCOPE not in scopes:
            raise SubscriptionError("ChatGPT plan use was not authorized. Enable it when signing in.")
        if str(tokens.get("token_type", "Bearer")).lower() != "bearer":
            raise SubscriptionError("Unsupported ChatGPT credential type.")
        return {"access_token": tokens["access_token"],
                "refresh_token": tokens.get("refresh_token") or previous.get("refresh_token"),
                "id_token": tokens.get("id_token") or previous.get("id_token"),
                "scopes": scopes, "expires_at": time.time() + max(0, int(tokens.get("expires_in", 0)))}

    async def finish_login(self, params: dict, browser_state: str) -> None:
        """Consume one callback, validate identity, and activate only its own account."""
        returned = str(params.get("state") or "")
        if not browser_state or not secrets.compare_digest(browser_state, returned):
            raise SubscriptionError("The sign-in browser session could not be verified.")
        pending = self.pending.pop(returned, None)
        if not pending or pending["expires"] <= time.time() or params.get("error"):
            raise SubscriptionError("Sign-in expired or was declined. Start again when ready.")
        client_id = params.get("client_id") or pending["client_id"]
        if not params.get("code") or not isinstance(client_id, str) or not client_id.startswith("oaiapp_"):
            raise SubscriptionError("ChatGPT registration did not complete.")
        if pending["client_id"] != "dynamic_agent_client" and client_id != pending["client_id"]:
            raise SubscriptionError("The returned registration does not match the selected account.")
        async with self.lock:
            self._idle()
            try:
                tokens = await self._post_token({"grant_type": "authorization_code", "client_id": client_id,
                    "code": params["code"], "code_verifier": pending["verifier"],
                    "redirect_uri": pending["redirect_uri"], "resource": API_URL})
            except (SubscriptionError, httpx.HTTPError):
                if not pending["account_id"]:
                    # A code is single-use even if its exchange is interrupted.
                    # Keep the issued registration for a fresh authorization,
                    # without activating or trusting an unverified identity.
                    self.state["accounts"][uuid4().hex] = {"client_id": client_id, "subject": None}
                    self.store.write(self.state)
                raise
            identity = await self._identity(tokens.get("id_token", ""), client_id, pending["nonce"])
            previous = self.state["accounts"].get(pending["account_id"])
            if previous and previous.get("subject") and identity["sub"] != previous["subject"]:
                raise SubscriptionError("Sign-in returned a different ChatGPT identity.")
            credentials = self._credentials(tokens)
            key = pending["account_id"] or next((key for key, item in self.state["accounts"].items()
                if item["client_id"] == client_id and item.get("subject") == identity["sub"]), uuid4().hex)
            account = {"client_id": client_id, "subject": identity["sub"], "issuer": ISSUER,
                       "email": identity.get("email"), "models": [], **credentials}
            self.state["accounts"][key] = account
            self.state["active"] = key
            self.store.write(self.state)
            account["models"] = await self._models(account["access_token"])
            self.store.write(self.state)

    async def _models(self, token: str) -> list[dict]:
        async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
            response = await client.get(API_URL + "/models", headers={"Authorization": "Bearer " + token})
        if response.status_code != 200:
            raise SubscriptionError("The ChatGPT account's model list is unavailable. Reconnect or refresh it.")
        models = response.json().get("models")
        if not isinstance(models, list) or any(not isinstance(item, dict) for item in models):
            raise SubscriptionError("ChatGPT returned an invalid model list.")
        return [{key: item.get(key) for key in ("slug", "display_name", "visibility")}
                for item in models if item.get("visibility") == "list" and isinstance(item.get("slug"), str)]

    async def _access_token(self, account: dict) -> str:
        if not account.get("access_token") or PLAN_SCOPE not in account.get("scopes", []):
            raise SubscriptionError("Sign in to your ChatGPT account before selecting its models.")
        if account.get("expires_at", 0) <= time.time() + 60:
            if not account.get("refresh_token"):
                raise SubscriptionError("Your ChatGPT session expired. Sign in again.")
            tokens = await self._post_token({"grant_type": "refresh_token", "client_id": account["client_id"],
                                           "refresh_token": account["refresh_token"], "resource": API_URL})
            account.update(self._credentials(tokens, account))
            self.store.write(self.state)
        return account["access_token"]

    async def refresh_models(self) -> None:
        """Refresh only the selected account's current model entitlement."""
        async with self.lock:
            account = self.state["accounts"].get(self.state.get("active"), {})
            token = await self._access_token(account)
            account["models"] = await self._models(token)
            self.store.write(self.state)

    async def select(self, account_id: str) -> None:
        """Refresh an account's catalog before making it the selected account."""
        async with self.lock:
            self._idle()
            account = self.state["accounts"].get(account_id, {})
            token = await self._access_token(account)
            models = await self._models(token)
            account["models"] = models
            self.state["active"] = account_id
            self.store.write(self.state)

    async def _revoke_session(self, account: dict) -> None:
        """Revoke the saved renewable session without exposing its token to another host."""
        refresh_token, client_id = account.get("refresh_token"), account.get("client_id")
        if not all(isinstance(value, str) and value for value in (refresh_token, client_id)):
            raise SubscriptionError(
                "This account has no usable refresh registration. Sign in again before disconnecting, "
                "or revoke the app in ChatGPT settings."
            )
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
                discovery = await client.get(DISCOVERY_URL)
                discovery.raise_for_status()
                metadata = discovery.json()
                endpoint = metadata.get("revocation_endpoint") if isinstance(metadata, dict) else None
                if not isinstance(endpoint, str):
                    raise ValueError("Missing revocation endpoint")
                target = urlsplit(endpoint)
                if target.scheme != "https" or target.netloc != urlsplit(ISSUER).netloc:
                    raise ValueError("Untrusted revocation endpoint")
                response = await client.post(endpoint, data={
                    "token": refresh_token, "token_type_hint": "refresh_token", "client_id": client_id,
                })
                if response.status_code != 200:
                    raise ValueError("Revocation was not confirmed")
        except (httpx.HTTPError, ValueError) as exc:
            raise SubscriptionError(
                "ChatGPT could not confirm disconnection. Your saved credentials were kept. "
                "Retry Disconnect, or revoke the app in ChatGPT settings."
            ) from exc

    async def logout(self) -> None:
        """Revoke the selected session before removing its saved credentials."""
        async with self.lock:
            self._idle()
            account = self.state["accounts"].get(self.state.get("active"), {})
            if account.get("access_token") or account.get("refresh_token"):
                await self._revoke_session(account)
            for key in ("access_token", "refresh_token", "id_token", "scopes", "expires_at", "models"):
                account.pop(key, None)
            self.state["active"] = None
            self.pending.clear()
            self.store.write(self.state)

    @asynccontextmanager
    async def access(self, settings: Any, model_id: str):
        """Pin account ownership while a request is in flight; never use an API key."""
        require_personal_usage(settings)
        async with self.lock:
            account = self.state["accounts"].get(self.state.get("active"), {})
            token = await self._access_token(account)
            if MODEL_PREFIX + model_id not in self.catalog():
                raise SubscriptionError("This model is not in the selected ChatGPT account's catalog.")
            self.active_calls += 1
        try:
            yield token
        finally:
            self.active_calls -= 1


_service: PersonalSubscription | None = None
_directory: Path | None = None


def _credential_directory(value: str | None) -> Path:
    directory = Path(value or Path.home() / ".config" / "gransabio" / "personal").expanduser().resolve()
    if directory.is_relative_to(Path(__file__).resolve().parent):
        raise SubscriptionError("Personal credentials must be stored outside the repository.")
    return directory


def get_personal_subscription(settings: Any) -> PersonalSubscription:
    """Lazily open local credentials only when the personal capability is allowed."""
    global _service, _directory
    require_personal_usage(settings)
    directory = _credential_directory(getattr(settings, "PERSONAL_SUBSCRIPTION_DIR", ""))
    if _service is None:
        store = CredentialStore(directory)
        try:
            _service = PersonalSubscription(store)
        except BaseException:
            store.close()
            raise
        _directory = directory
    elif directory != _directory:
        raise SubscriptionError("Restart the engine after changing its personal credential directory.")
    return _service


def close_personal_subscription() -> None:
    """Release the credential owner lock during orderly engine shutdown."""
    global _service, _directory
    if _service is not None:
        _service.store.close()
        _service = None
        _directory = None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate local ChatGPT credentials to macOS Keychain-backed encryption.")
    parser.add_argument("command", choices=["migrate-keychain"])
    parser.add_argument("--directory", help="Credential directory; defaults to ~/.config/gransabio/personal")
    args = parser.parse_args()
    try:
        if sys.platform != "darwin":
            raise SubscriptionError("Keychain migration is available only on macOS.")
        store = CredentialStore(_credential_directory(args.directory))
        try:
            migrated = store.migrate_to_keychain()
        finally:
            store.close()
    except SubscriptionError as exc:
        parser.exit(1, str(exc) + "\n")
    except OSError:
        # No traceback or credential contents in this operator command.
        parser.exit(1, "Migration failed. Confirm the directory, owner-only permissions, unlocked Keychain, "
                    "and stopped engine. The original credential file was not replaced.\n")
    print("Migration verified and completed." if migrated else "Already encrypted; Keychain access verified.")
