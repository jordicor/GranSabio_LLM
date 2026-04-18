"""
Focused unit tests for core/security.py internal CIDR handling.

These tests validate the localhost/LAN allowlist behavior without
importing the FastAPI app.
"""

from __future__ import annotations

import importlib
import os
from unittest.mock import patch

import core.security as security


def test_default_internal_networks_allow_localhost_and_default_lan():
    with patch.dict(os.environ, {"INTERNAL_ALLOWED_CIDRS": ""}, clear=False):
        importlib.reload(security)

        assert security.is_ip_allowed("127.0.0.1") is True
        assert security.is_ip_allowed("::1") is True
        assert security.is_ip_allowed("192.168.50.23") is True
        assert security.is_ip_allowed("203.0.113.10") is False


def test_internal_allowed_cidrs_override_replaces_default_allowlist():
    with patch.dict(
        os.environ,
        {"INTERNAL_ALLOWED_CIDRS": "10.10.0.0/16,172.16.5.0/24"},
        clear=False,
    ):
        importlib.reload(security)

        assert security.is_ip_allowed("10.10.12.34") is True
        assert security.is_ip_allowed("172.16.5.77") is True
        assert security.is_ip_allowed("192.168.50.23") is False


def test_invalid_internal_allowed_cidrs_falls_back_to_defaults():
    with patch.dict(
        os.environ,
        {"INTERNAL_ALLOWED_CIDRS": "not-a-cidr"},
        clear=False,
    ):
        importlib.reload(security)

        assert security.is_ip_allowed("127.0.0.1") is True
        assert security.is_ip_allowed("192.168.50.23") is True
