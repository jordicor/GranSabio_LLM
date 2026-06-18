"""Static regressions for the admin model sync UI contract."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_ADMIN_JS = ROOT / "static" / "js" / "models_admin.js"
ADMIN_MODELS_TEMPLATE = ROOT / "templates" / "admin_models.html"


def test_models_admin_preserves_supported_parameters_through_sync_payload():
    source = MODELS_ADMIN_JS.read_text(encoding="utf-8")

    assert "const supportedParameters = normalizeCapabilities(" in source
    assert "supported_parameters: supportedParameters" in source
    assert "supported_parameters: [...(model.supported_parameters || [])].sort()" in source
    assert 'changes.push("supported parameters")' in source


def test_provider_health_uses_instance_fetch_helper():
    source = MODELS_ADMIN_JS.read_text(encoding="utf-8")

    assert "await this.fetchJson(`/api/admin/provider-health${suffix}`)" in source
    assert "await fetchJson(`/api/admin/provider-health${suffix}`)" not in source


def test_models_admin_template_exposes_native_provider_tabs():
    source = ADMIN_MODELS_TEMPLATE.read_text(encoding="utf-8")

    assert 'data-tab="minimax"' in source
    assert 'id="badge-minimax"' in source
    assert 'data-tab="moonshot"' in source
    assert 'id="badge-moonshot"' in source


def test_models_admin_template_cache_busts_script_asset():
    source = ADMIN_MODELS_TEMPLATE.read_text(encoding="utf-8")

    assert '/static/js/models_admin.js?v={{ models_admin_asset_version }}"' in source
