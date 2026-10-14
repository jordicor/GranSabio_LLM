/** Run with: node dev_tests/test_models_admin_pricing.js */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

// The admin page script is one IIFE; its pure catalog helpers are evaluated on
// their own so the test exercises exactly the code the page runs.
function loadHelpers() {
    const source = fs.readFileSync(path.join(__dirname, '../static/js/models_admin.js'), 'utf8');
    const slice = (start, end) => {
        const from = source.indexOf(start);
        const to = source.indexOf(end, from);
        assert.ok(from >= 0 && to > from, `missing helper block ${start}`);
        return source.slice(from, to);
    };
    const context = vm.createContext({});
    vm.runInContext([
        slice('  function toArray(', '  function formatDate('),
        slice('  const PRICING_RATE_KEYS = [', '  function normalizeCapabilities('),
        slice('  function fingerprint(model) {', '  function modelKey('),
    ].join('\n'), context);
    // Objects built inside the vm realm have their own Object.prototype, so
    // results come back as plain JSON copies for strict deep comparison.
    const plain = (fn) => (...args) => JSON.parse(JSON.stringify(fn(...args)));
    return {
        normalizePricing: plain(vm.runInContext('normalizePricing', context)),
        formatLongContext: vm.runInContext('formatLongContext', context),
        fingerprint: vm.runInContext('fingerprint', context),
    };
}

// services/model_sync.openrouter_pricing_from_remote output for a cache-priced,
// long-context model, as /api/admin/models/providers/openrouter/remote serves it.
const LUNA_PRICING = {
    input_per_million: 0.1,
    output_per_million: 0.5,
    cached_input_per_million: 0.01,
    cache_write_per_million: 0.125,
    long_context: {
        threshold_tokens: 272000,
        input_per_million: 0.2,
        output_per_million: 0.75,
        cached_input_per_million: 0.02,
        cache_write_per_million: 0.25,
    },
};

test('the catalog pricing block survives the UI normalization unchanged', () => {
    const { normalizePricing } = loadHelpers();
    assert.deepEqual(normalizePricing(LUNA_PRICING), LUNA_PRICING);
    assert.deepEqual(normalizePricing({ input_per_million: 0.08, output_per_million: 0.3 }),
        { input_per_million: 0.08, output_per_million: 0.3 });
});

test('an unlisted rate stays absent instead of becoming a free rate', () => {
    const { normalizePricing } = loadHelpers();
    // Router models priced per routed request have no catalog rate at all.
    assert.deepEqual(normalizePricing({}), {});
    assert.deepEqual(normalizePricing({ input_per_million: null, output_per_million: '', cached_input_per_million: 'n/a' }), {});
    // A long-context block without a positive threshold or with junk rates is dropped or trimmed.
    assert.deepEqual(normalizePricing({ input_per_million: 1, long_context: { threshold_tokens: 0, input_per_million: 2 } }),
        { input_per_million: 1 });
    assert.deepEqual(normalizePricing({ input_per_million: 1, long_context: { threshold_tokens: 128000, input_per_million: 'x', output_per_million: 4 } }),
        { input_per_million: 1, long_context: { threshold_tokens: 128000, output_per_million: 4 } });
    assert.deepEqual(normalizePricing({ input_per_million: 1, long_context: [1, 2] }), { input_per_million: 1 });
});

test('remote and persisted entries with the same rates share one fingerprint', () => {
    const { normalizePricing, fingerprint } = loadHelpers();
    const base = { name: 'OpenAI: GPT-6 Luna', context_window: 1050000, input_tokens: 1050000, output_tokens: 128000,
        capabilities: ['text'], supported_parameters: ['tools'], source: 'https://openrouter.ai/models/openai/gpt-6-luna' };
    // Key order differs between the fetched and the persisted JSON; normalization fixes it.
    const persisted = { long_context: { cache_write_per_million: 0.25, threshold_tokens: 272000, input_per_million: 0.2,
        output_per_million: 0.75, cached_input_per_million: 0.02 }, cache_write_per_million: 0.125,
        cached_input_per_million: 0.01, output_per_million: 0.5, input_per_million: 0.1 };
    const remote = { ...base, pricing: normalizePricing(LUNA_PRICING) };
    const local = { ...base, pricing: normalizePricing(persisted) };
    assert.equal(fingerprint(remote), fingerprint(local));
    // Dropping the cache rates (the pre-fix normalization) is a real difference.
    const stripped = { ...base, pricing: normalizePricing({ input_per_million: 0.1, output_per_million: 0.5 }) };
    assert.notEqual(fingerprint(remote), fingerprint(stripped));
});

test('the long-context diff line names the threshold and every listed rate', () => {
    const { formatLongContext } = loadHelpers();
    assert.equal(formatLongContext(LUNA_PRICING.long_context),
        '>272.0K tokens: in $0.2000 / out $0.7500 / cached $0.0200 / write $0.2500');
    assert.equal(formatLongContext({ threshold_tokens: 128000, input_per_million: 1 }), '>128.0K tokens: in $1.0000');
    assert.equal(formatLongContext(undefined), '-');
});
