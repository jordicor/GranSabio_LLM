/** Run with: node dev_tests/test_generation_init_frontend.js */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

function modelElement() {
    return {
        value: '', textContent: '', style: {}, dataset: {}, children: [],
        addEventListener() {},
        set innerHTML(value) { this.children = []; this.value = ''; },
        appendChild(child) { this.children.push(child); },
        querySelectorAll() {
            return this.children.flatMap(child => child.children.length ? child.querySelectorAll() : [child]);
        },
        querySelector() { return this.querySelectorAll()[0] || null; }
    };
}

function createInterface(responseBody, catalogResponse) {
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
    for (const id of ['generatorModel', 'granSabioModel', 'qaModelSelect', 'addQaModelBtn']) {
        elements.set(id, modelElement());
    }
    const requests = [];
    const streams = [];
    const timers = [];
    const alerts = [];
    const context = vm.createContext({
        console,
        document: {
            getElementById: id => elements.get(id) || null,
            createElement: () => modelElement(),
            querySelectorAll: () => [],
            addEventListener() {}
        },
        alert: message => alerts.push(message),
        fetch: async (url, options = {}) => {
            requests.push({ url, method: options.method || 'GET' });
            if (url === '/models/qa/available') return catalogResponse;
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
                generator_model: 'test-model', gran_sabio_model: 'test-model',
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

test('personal model routes all text steps through the subscription without API defaults', () => {
    const { app } = createInterface({});
    const data = {
        generator_model: 'chatgpt/gpt-test', qa_models: ['paid-reviewer'],
        gran_sabio_model: 'paid-reviewer', auto_qa: { enabled: true },
        feedback_memory_enabled: true
    };
    app.usePersonalSubscriptionForRequest(data);
    assert.equal(data.llm_routing.global.model, 'chatgpt/gpt-test');
    assert.equal(data.qa_models.join(','), 'chatgpt/gpt-test');
    assert.equal(data.gran_sabio_model, 'chatgpt/gpt-test');
    assert.equal(data.arbiter_model, 'chatgpt/gpt-test');
    assert.equal(data.feedback_memory_enabled, false);
    assert.equal(data.auto_qa.enabled, true);
    const ordinary = { generator_model: 'api-model', qa_models: ['paid-reviewer'] };
    app.usePersonalSubscriptionForRequest(ordinary);
    assert.deepEqual(ordinary, { generator_model: 'api-model', qa_models: ['paid-reviewer'] });
});

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


for (const preferred of [null, 'chatgpt/gpt-test', 'gpt-test']) {
    test('model selectors preserve explicit defaults without substituting a missing model: ' + preferred, async () => {
        const responseBody = {
            qa_models: [
                { key: 'gpt-test', name: 'API model', provider: 'openai', qa_priority: 'fast', output_tokens: 2000 },
                { key: 'chatgpt/gpt-test', name: 'Personal model', provider: 'chatgpt', qa_priority: 'standard', output_tokens: 0 }
            ],
            defaults: { generator: preferred, gran_sabio: preferred, qa: [] },
            recommendations: {},
            default_errors: preferred ? [] : ['The configured model is unavailable. Select an available model.']
        };
        const ui = createInterface({}, { ok: true, json: async () => responseBody });
        ui.app.updateReasoningControls = () => {};
        await ui.app.loadQAModels();

        for (const id of ['generatorModel', 'granSabioModel']) {
            const select = ui.elements.get(id);
            assert.equal(select.value, preferred || '');
            const options = select.querySelectorAll('option');
            assert.deepEqual(options.map(option => option.value), ['gpt-test', 'chatgpt/gpt-test']);
            assert.equal(options[0].textContent, 'API model (2000 tokens)');
            assert.equal(options[1].textContent, 'Personal model');
        }
        assert.equal(ui.alerts.length, preferred ? 0 : 1);
        if (!preferred) {
            assert.match(ui.alerts[0], /configured model is unavailable/);
            const request = { prompt: 'Write', generator_model: '', gran_sabio_model: '',
                qa_models: ['gpt-test'], qa_layers: [{ name: 'Quality', description: 'Check', criteria: 'Clear' }] };
            assert.equal(ui.app.validateFormData(request), false);
        }
    });
}

test('model catalog HTTP errors are visible without trying to parse an error page', async () => {
    const ui = createInterface({}, {
        ok: false, status: 500,
        json: async () => { throw new Error('Error page must not be parsed as catalog JSON'); }
    });
    await ui.app.loadQAModels();
    assert.deepEqual(ui.alerts, ['Error: Failed to load models: Model catalog request failed (HTTP 500).']);
});
