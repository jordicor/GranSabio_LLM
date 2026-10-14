/** Run with: node dev_tests/test_generation_init_frontend.js */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

function createInterface(responseBody) {
    const elements = new Map();
    for (const id of [
        'configurationScreen', 'processingScreen', 'backToConfigBtn',
        'waitingSection', 'finalResultsSection', 'processingStatusBadge',
        'progressText', 'startGenerationBtn', 'stopGenerationBtn', 'sessionIdDisplay'
    ]) {
        elements.set(id, {
            style: {}, textContent: '', disabled: false, listeners: new Map(),
            addEventListener(event, handler) { this.listeners.set(event, handler); },
            querySelector() { return null; }
        });
    }
    const requests = [];
    const streams = [];
    const timers = [];
    const alerts = [];
    const context = vm.createContext({
        console,
        document: {
            getElementById: id => elements.get(id) || null,
            querySelectorAll: () => [],
            addEventListener() {}
        },
        alert: message => alerts.push(message),
        fetch: async (url, options) => {
            requests.push({ url, method: options.method });
            assert.equal(url, '/generate');
            return { ok: true, json: async () => responseBody };
        },
        EventSource: class {
            constructor(url) { streams.push(url); }
            close() {}
        },
        setInterval: () => { timers.push('interval'); return timers.length; },
        setTimeout: () => { timers.push('timeout'); return timers.length; },
        clearInterval() {}, clearTimeout() {}
    });
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/app.js'), 'utf8'), context);
    const Interface = vm.runInContext('GranSabioInterface', context);
    class TestInterface extends Interface {
        // Form bootstrap is unrelated; use the real event handler, HTTP call,
        // screen transitions, error display and progress/stream methods.
        init() { this.bindEvents(); }
        collectFormData() {
            return {
                prompt: 'Generate a draft', qa_models: ['test-model'],
                qa_layers: [{ name: 'Quality', description: 'Review', criteria: 'Clear prose' }]
            };
        }
    }
    return { app: new TestInterface(), elements, requests, streams, timers, alerts };
}

async function clickStart(ui) {
    let prevented = false;
    ui.elements.get('startGenerationBtn').listeners.get('click')({
        preventDefault() { prevented = true; }
    });
    await new Promise(setImmediate);
    assert.equal(prevented, true);
}

const refusals = [
    {
        name: 'preflight rejection', label: 'Generation did not start',
        reason: '<script>Contradictory requirements</script>',
        body: {
            status: 'preflight_rejected', session_id: null,
            preflight_feedback: { decision: 'reject', user_feedback: '<script>Contradictory requirements</script>', issues: [] }
        }
    },
    {
        name: 'manual review', label: 'Manual review required', reason: 'Please clarify the requirements',
        body: {
            status: 'preflight_rejected', session_id: null,
            preflight_feedback: { decision: 'manual_review', user_feedback: 'Please clarify the requirements', issues: [] }
        }
    },
    {
        name: 'Auto-QA rejection', label: 'Generation did not start', reason: 'No usable QA plan',
        body: {
            status: 'auto_qa_rejected', session_id: null,
            preflight_feedback: { decision: 'proceed', user_feedback: 'Preflight passed', issues: [] },
            auto_qa_feedback: { message: 'No usable QA plan' }
        }
    },
    {
        name: 'invalid initial response', label: 'Generation did not start',
        reason: 'No session was created (status: unknown)', body: {}
    },
    {
        name: 'null initial response', label: 'Generation did not start',
        reason: 'No session was created (status: unknown)', body: null
    }
];

for (const { name, label, reason, body } of refusals) {
    test(`connected startGeneration leaves processing after ${name}`, async () => {
        const ui = createInterface(body);
        ui.app.currentSession = 'previous-terminal-session';
        await clickStart(ui);

        assert.equal(ui.app.currentSession, null);
        assert.equal(ui.app.currentScreen, 'configuration');
        assert.equal(ui.elements.get('configurationScreen').style.display, 'flex');
        assert.equal(ui.elements.get('processingScreen').style.display, 'none');
        assert.equal(ui.elements.get('startGenerationBtn').disabled, false);
        assert.deepEqual(ui.alerts, [`Error: ${label}: ${reason}`]);
        assert.deepEqual(ui.requests, [{ url: '/generate', method: 'POST' }]);
        assert.deepEqual(ui.streams, []);
        assert.deepEqual(ui.timers, []);
    });
}

test('connected startGeneration tracks an initialized session', async () => {
    const ui = createInterface({ status: 'initialized', session_id: 'new-session' });
    await clickStart(ui);

    assert.equal(ui.app.currentSession, 'new-session');
    assert.equal(ui.app.currentScreen, 'processing');
    assert.equal(ui.elements.get('sessionIdDisplay').textContent, 'new-session');
    assert.deepEqual(ui.alerts, []);
    assert.deepEqual(ui.requests, [{ url: '/generate', method: 'POST' }]);
    assert.deepEqual(ui.streams, ['/stream/project/new-session']);
    assert.ok(ui.timers.includes('interval'));
});
