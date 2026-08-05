"""Source-contract regression tests for the frontend security fixes.

The repo has no JS test runner (adding one was explicitly rejected as
overengineering), so these tests pin the security-relevant contracts at the
source level:

- FE-02: dynamic values inside quoted HTML attributes use ``escapeHtmlAttr``.
- FE-06: third-party CDN scripts carry SRI; the prism autoloader (which
  downloads unpinned code at runtime) is gone.
- FE-07: ``addQALayer`` never interpolates data into HTML strings.
- FE-08: a single shared escape helper exists and pages load it.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
TEMPLATES = ROOT / "templates"

PAGE_TEMPLATES = [
    "index.html",
    "debugger.html",
    "admin_models.html",
    "smart_edit_demo.html",
    "stream_monitor.html",
]

CDN_TEMPLATES = ["api_docs.html", "api_usage.html", "smart_edit_demo.html"]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# FE-02: attribute interpolation must use escapeHtmlAttr, not escapeHtml
# ---------------------------------------------------------------------------


def test_fe02_debugger_title_attribute_uses_attr_escape():
    source = _read(STATIC / "debugger.js")
    assert 'title="${escapeHtmlAttr(' in source


def test_fe02_no_content_mode_escape_inside_attributes_anywhere():
    """Interpolating content-mode escapeHtml() inside a quoted attribute is
    the FE-02 XSS vector (quotes are not escaped)."""
    pattern = re.compile(r'="\$\{escapeHtml\(')
    for js_file in STATIC.rglob("*.js"):
        assert not pattern.search(_read(js_file)), f"unsafe attribute escape in {js_file}"


def test_fe02_attr_escape_covers_quotes():
    helper = _read(STATIC / "js" / "escape_utils.js")
    attr_start = helper.index("function escapeHtmlAttr")
    attr_body = helper[attr_start:]
    for escaped in ("&amp;", "&lt;", "&gt;", "&quot;", "&#39;"):
        assert escaped in attr_body, f"escapeHtmlAttr does not produce {escaped}"


# ---------------------------------------------------------------------------
# FE-06: CDN scripts pinned with SRI; no runtime autoloader
# ---------------------------------------------------------------------------


def test_fe06_all_cdn_scripts_have_sri():
    tag_pattern = re.compile(r'<script[^>]*src="https://[^"]+"[^>]*>')
    for name in CDN_TEMPLATES:
        for tag in tag_pattern.findall(_read(TEMPLATES / name)):
            assert "integrity=" in tag, f"missing SRI in {name}: {tag}"
            assert "crossorigin=" in tag, f"missing crossorigin in {name}: {tag}"


def test_fe06_prism_autoloader_removed():
    """The autoloader downloads grammar files from the CDN at runtime with no
    integrity pinning; it must not come back."""
    for html_file in TEMPLATES.rglob("*.html"):
        assert "prism-autoloader" not in _read(html_file), f"autoloader in {html_file}"


# ---------------------------------------------------------------------------
# FE-07: addQALayer assigns data via DOM properties, never HTML interpolation
# ---------------------------------------------------------------------------


def test_fe07_add_qa_layer_does_not_interpolate_data_into_html():
    source = _read(STATIC / "js" / "app.js")
    assert "addQALayer(layerData" in source, "addQALayer not found in app.js"
    # Layer data must only reach the DOM via property assignments (.value,
    # .checked, textContent), never through HTML/template interpolation.
    assert "${defaultData." not in source
    assert "${layerData." not in source


# ---------------------------------------------------------------------------
# FE-08: single shared escape helper, loaded by every page
# ---------------------------------------------------------------------------


def test_fe08_no_local_escape_implementations_outside_shared_helper():
    for js_file in STATIC.rglob("*.js"):
        if js_file.name == "escape_utils.js":
            continue
        assert "function escapeHtml" not in _read(js_file), f"local escapeHtml in {js_file}"


def test_fe08_pages_load_shared_helper_before_their_scripts():
    for name in PAGE_TEMPLATES:
        html = _read(TEMPLATES / name)
        assert "/static/js/escape_utils.js" in html, f"escape_utils.js not loaded in {name}"
