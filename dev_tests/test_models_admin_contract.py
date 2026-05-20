"""Static regressions for the admin model sync UI contract."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS_ADMIN_JS = ROOT / "static" / "js" / "models_admin.js"


def test_models_admin_preserves_supported_parameters_through_sync_payload():
    source = MODELS_ADMIN_JS.read_text(encoding="utf-8")

    assert "const supportedParameters = normalizeCapabilities(" in source
    assert "supported_parameters: supportedParameters" in source
    assert "supported_parameters: [...(model.supported_parameters || [])].sort()" in source
    assert 'changes.push("supported parameters")' in source
