"""Regression contracts for the main, Smart Edit, and monitor frontends."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "static" / "js" / "app.js"
SMART_EDIT_JS = ROOT / "static" / "js" / "smart_edit.js"
STREAM_MONITOR_JS = ROOT / "static" / "js" / "stream_monitor.js"
INDEX_TEMPLATE = ROOT / "templates" / "index.html"
OPENROUTER_TEMPLATE = ROOT / "templates" / "admin_openrouter.html"


def _function_source(source: str, start: str, next_start: str) -> str:
    start_index = source.index(start)
    end_index = source.index(next_start, start_index)
    return source[start_index:end_index]


def test_final_result_controls_use_current_dom_contract():
    source = APP_JS.read_text(encoding="utf-8")
    template = INDEX_TEMPLATE.read_text(encoding="utf-8")

    for element_id in (
        "finalContentDisplay",
        "copyFinalContent",
        "downloadFinalContent",
    ):
        assert f'id="{element_id}"' in template
        assert f"getElementById('{element_id}')" in source

    assert "getElementById('generatedContent')" not in source
    assert "getElementById('copyContent')" not in source
    assert "getElementById('downloadContent')" not in source
    assert 'id="saveConfigBtn" class="btn-tertiary" hidden disabled' in template
    assert 'id="refineContentBtn" class="btn-secondary" hidden disabled' in template


def test_reset_uses_existing_sections_and_null_guards():
    source = APP_JS.read_text(encoding="utf-8")
    reset_source = _function_source(source, "    resetForm() {", "    async copyGeneratedContent() {")

    assert "getElementById('resultsPanel')" not in source
    assert "getElementById('finalResultsSection')" in reset_source
    assert "this.stopProgressTracking();" in reset_source
    assert "this.currentSession = null;" in reset_source
    assert "if (finalResults)" in reset_source
    assert "if (waitingSection)" in reset_source
    assert "if (temperatureValue)" in reset_source
    assert "if (scoreValue)" in reset_source


def test_sse_failure_has_bounded_idempotent_polling_fallback():
    source = APP_JS.read_text(encoding="utf-8")

    assert "const SSE_MAX_ERRORS = 2;" in source
    assert "const SSE_FALLBACK_DELAY_MS = 8000;" in source
    assert "scheduleSSEFallback('SSE connection timed out')" in source
    assert "this.sseErrorCount >= SSE_MAX_ERRORS" in source
    assert "activatePollingFallback(reason)" in source
    assert "if (this.sseFallbackActive) return;" in source
    assert "this.closeEventStream();" in source
    assert "void this.updateProgress();" in source
    assert "EventSource.CLOSED" not in source


def test_stop_waits_for_server_before_terminal_ui_cleanup():
    source = APP_JS.read_text(encoding="utf-8")
    stop_source = _function_source(
        source,
        "    async handleStop(sessionId = this.currentSession",
        "    collectFormData() {",
    )

    assert "this.stopPromise" in source
    assert "return this.stopPromise;" in source
    assert stop_source.index("await fetch") < stop_source.index("this.stopProgressTracking()")
    assert "getElementById('processingStatusBadge')" in stop_source
    assert "getElementById('waitingSection')" in stop_source
    assert "await this.showFinalResults();" in stop_source
    assert "getElementById('statusBadge')" not in source


def test_smart_edit_action_labels_are_built_as_text_nodes():
    source = SMART_EDIT_JS.read_text(encoding="utf-8")
    item_source = _function_source(
        source,
        "function createActionItem(action, index) {",
        "function getTypeClass(type) {",
    )

    assert "type.textContent = actionType;" in item_source
    assert "description.textContent = action.description" in item_source
    assert "item.innerHTML" not in item_source
    assert "${action.type}" not in source
    assert "${action.description" not in source


def test_smart_edit_run_all_reports_partial_failure_without_confetti():
    source = SMART_EDIT_JS.read_text(encoding="utf-8")
    run_all_source = _function_source(
        source,
        "async function handleRunAll() {",
        "/**\n * Celebration effect",
    )

    assert "let succeeded = 0;" in run_all_source
    assert "let failed = 0;" in run_all_source
    assert "if (failed === 0)" in run_all_source
    assert "${succeeded} completed, ${failed} failed" in run_all_source
    assert run_all_source.index("if (failed === 0)") < run_all_source.index("celebrateCompletion();")


def test_openrouter_catalog_rows_do_not_interpolate_remote_metadata_as_html():
    source = OPENROUTER_TEMPLATE.read_text(encoding="utf-8")

    assert "tbody.replaceChildren();" in source
    assert "modelName.textContent = model.name || modelKey;" in source
    assert "modelId.textContent = modelKey;" in source
    assert "checkbox.addEventListener('change', () => toggleModel(modelKey));" in source
    assert "${model.id}" not in source
    assert "${model.name}" not in source
    assert "${model.provider}" not in source
    assert "onchange=\"toggleModel('" not in source


def test_stream_monitor_escapes_provider_model_labels():
    source = STREAM_MONITOR_JS.read_text(encoding="utf-8")

    assert "${escapeHtml(gen.model)}" in source
    assert "${escapeHtml(model)}" in source
    assert "${escapeHtml(sessionIdShort)}" in source
    assert "${escapeHtml(status)}" in source
    assert "${gen.model}</span>" not in source
    assert "Model: <span>${model}</span>" not in source
