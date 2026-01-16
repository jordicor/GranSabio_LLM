/**
 * Smart Edit Playground - JavaScript
 *
 * Handles:
 * - Text analysis and action generation
 * - Action execution with animations
 * - Preview updates and view modes
 * - Settings and theme management
 */

// ============================================
// STATE MANAGEMENT
// ============================================

const state = {
    originalText: '',
    currentText: '',
    actions: [],
    executedActions: [],
    isRunning: false,
    animationSpeed: 1,
    settings: {
        autoScroll: true,
        highlightChanges: true,
        pauseBetween: false,
    }
};

// ============================================
// DOM ELEMENTS
// ============================================

const elements = {
    // Source panel
    sourceText: document.getElementById('source-text'),
    charCount: document.getElementById('char-count'),
    sampleSelect: document.getElementById('sample-select'),

    // Actions panel
    generateBtn: document.getElementById('generate-actions-btn'),
    analysisModel: document.getElementById('analysis-model'),
    runAllSection: document.getElementById('run-all-section'),
    runAllBtn: document.getElementById('run-all-btn'),
    runAllMeta: document.getElementById('run-all-meta'),
    actionsQueue: document.getElementById('actions-queue'),
    statsBar: document.getElementById('stats-bar'),
    statDirect: document.getElementById('stat-direct'),
    statAi: document.getElementById('stat-ai'),
    statChars: document.getElementById('stat-chars'),

    // Preview panel
    viewBtns: document.querySelectorAll('.view-btn'),
    previewLive: document.getElementById('preview-live'),
    previewDiff: document.getElementById('preview-diff'),
    previewSide: document.getElementById('preview-side'),
    previewText: document.getElementById('preview-text'),
    diffOutput: document.getElementById('diff-output'),
    sideOriginal: document.getElementById('side-original'),
    sideEdited: document.getElementById('side-edited'),
    copyResultBtn: document.getElementById('copy-result-btn'),

    // Theme & Settings
    themeToggle: document.getElementById('theme-toggle'),
    settingsPanel: document.getElementById('settings-panel'),
    settingsToggle: document.getElementById('settings-toggle'),
    animationSpeed: document.getElementById('animation-speed'),
    speedValue: document.getElementById('speed-value'),
    autoScroll: document.getElementById('auto-scroll'),
    highlightChanges: document.getElementById('highlight-changes'),
    pauseBetween: document.getElementById('pause-between'),

    // Loading & Toast
    loadingOverlay: document.getElementById('loading-overlay'),
    loadingText: document.getElementById('loading-text'),
    toastContainer: document.getElementById('toast-container'),
};

// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initializeTheme();
    initializeEventListeners();
    loadSampleTexts();
    updateCharCounter();
});

function initializeTheme() {
    const savedTheme = localStorage.getItem('smart-edit-theme');
    if (savedTheme === 'dark') {
        document.body.classList.remove('light-mode');
        document.body.classList.add('dark-mode');
        elements.themeToggle.querySelector('.theme-icon').textContent = '\u2600'; // Sun
    }
}

function initializeEventListeners() {
    // Source text
    elements.sourceText.addEventListener('input', handleSourceChange);
    elements.sampleSelect.addEventListener('change', handleSampleSelect);

    // Actions
    elements.generateBtn.addEventListener('click', handleGenerateActions);
    elements.runAllBtn.addEventListener('click', handleRunAll);

    // View mode
    elements.viewBtns.forEach(btn => {
        btn.addEventListener('click', () => handleViewModeChange(btn));
    });

    // Copy result
    elements.copyResultBtn.addEventListener('click', handleCopyResult);

    // Theme toggle
    elements.themeToggle.addEventListener('click', handleThemeToggle);

    // Settings
    elements.settingsToggle.addEventListener('click', () => {
        elements.settingsPanel.classList.toggle('open');
    });

    elements.animationSpeed.addEventListener('input', (e) => {
        state.animationSpeed = parseFloat(e.target.value);
        elements.speedValue.textContent = state.animationSpeed + 'x';
    });

    elements.autoScroll.addEventListener('change', (e) => {
        state.settings.autoScroll = e.target.checked;
    });

    elements.highlightChanges.addEventListener('change', (e) => {
        state.settings.highlightChanges = e.target.checked;
    });

    elements.pauseBetween.addEventListener('change', (e) => {
        state.settings.pauseBetween = e.target.checked;
    });
}

// ============================================
// SOURCE TEXT HANDLING
// ============================================

function handleSourceChange() {
    updateCharCounter();
    updatePreview(elements.sourceText.value);
    state.originalText = elements.sourceText.value;
    state.currentText = elements.sourceText.value;

    // Clear actions when text changes significantly
    if (state.actions.length > 0) {
        clearActions();
    }
}

function updateCharCounter() {
    const count = elements.sourceText.value.length;
    elements.charCount.textContent = count.toLocaleString();
}

function updatePreview(text) {
    if (!text) {
        elements.previewText.innerHTML = '<span class="placeholder">Preview will appear here...</span>';
    } else {
        elements.previewText.textContent = text;
    }
    elements.sideOriginal.textContent = state.originalText || text;
    elements.sideEdited.textContent = text;
}

// ============================================
// SAMPLE TEXTS
// ============================================

async function loadSampleTexts() {
    try {
        const response = await fetch('/smart-edit/samples');
        if (!response.ok) throw new Error('Failed to load samples');

        const samples = await response.json();
        populateSampleDropdown(samples);
    } catch (error) {
        console.error('Error loading samples:', error);
    }
}

function populateSampleDropdown(samples) {
    elements.sampleSelect.innerHTML = '<option value="">Load Example...</option>';

    samples.forEach(sample => {
        const option = document.createElement('option');
        option.value = sample.id;
        option.textContent = `${sample.title} (${sample.word_count} words)`;
        option.dataset.sample = JSON.stringify(sample);
        elements.sampleSelect.appendChild(option);
    });
}

function handleSampleSelect() {
    const selected = elements.sampleSelect.selectedOptions[0];
    if (!selected || !selected.dataset.sample) return;

    const sample = JSON.parse(selected.dataset.sample);
    elements.sourceText.value = sample.text;
    handleSourceChange();

    // Load pre-analyzed actions
    if (sample.suggested_actions && sample.suggested_actions.length > 0) {
        state.actions = sample.suggested_actions;
        renderActions();
        showToast(`Loaded ${state.actions.length} suggested actions`, 'info');
    }

    // Reset dropdown
    elements.sampleSelect.value = '';
}

// ============================================
// ACTION GENERATION
// ============================================

async function handleGenerateActions() {
    const text = elements.sourceText.value.trim();
    if (!text) {
        showToast('Please enter some text first', 'error');
        return;
    }

    showLoading('Analyzing text...');
    elements.generateBtn.disabled = true;

    try {
        const response = await fetch('/smart-edit/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                analysis_model: elements.analysisModel.value,
                max_actions: 20,
            }),
        });

        if (!response.ok) throw new Error('Analysis failed');

        const data = await response.json();
        state.actions = data.actions;
        state.originalText = text;
        state.currentText = text;

        renderActions();
        updateStats(data.stats);

        if (state.actions.length > 0) {
            showToast(`Found ${state.actions.length} suggested edits`, 'success');
        } else {
            showToast('No issues found - text looks good!', 'info');
        }
    } catch (error) {
        console.error('Analysis error:', error);
        showToast('Failed to analyze text: ' + error.message, 'error');
    } finally {
        hideLoading();
        elements.generateBtn.disabled = false;
    }
}

// ============================================
// ACTION RENDERING
// ============================================

function renderActions() {
    if (state.actions.length === 0) {
        elements.actionsQueue.innerHTML = `
            <div class="empty-state">
                <p>No actions yet.</p>
                <p class="hint">Enter text and click "Generate Actions" to analyze.</p>
            </div>
        `;
        elements.runAllSection.style.display = 'none';
        elements.statsBar.style.display = 'none';
        return;
    }

    elements.actionsQueue.innerHTML = '';
    state.actions.forEach((action, index) => {
        const item = createActionItem(action, index);
        elements.actionsQueue.appendChild(item);
    });

    // Show run all button
    elements.runAllSection.style.display = 'block';
    const totalTime = state.actions.reduce((sum, a) => sum + (a.estimated_ms || 10), 0);
    elements.runAllMeta.textContent = `${state.actions.length} actions | ~${(totalTime / 1000).toFixed(1)}s`;

    elements.statsBar.style.display = 'flex';
}

function createActionItem(action, index) {
    const item = document.createElement('div');
    item.className = 'action-item ready';
    item.dataset.index = index;
    item.dataset.actionId = action.id;

    const isAI = action.ai_required || ['rephrase', 'improve', 'fix_grammar', 'fix_style'].includes(action.type);
    const typeClass = getTypeClass(action.type);

    item.innerHTML = `
        <button class="action-play-btn" title="Execute this action">
            &#9654;
        </button>
        <div class="action-info">
            <span class="action-type ${typeClass}">${action.type}</span>
            <span class="action-description">${action.description || 'No description'}</span>
        </div>
        <span class="action-badge ${isAI ? 'ai' : 'direct'}">
            ${isAI ? '&#129302; AI' : '&#9889; Direct'}
        </span>
        <span class="action-status"></span>
    `;

    // Add click handler for play button
    const playBtn = item.querySelector('.action-play-btn');
    playBtn.addEventListener('click', () => executeAction(index));

    return item;
}

function getTypeClass(type) {
    const typeMap = {
        delete: 'delete',
        insert: 'insert',
        insert_before: 'insert',
        insert_after: 'insert',
        replace: 'replace',
        format: 'format',
        rephrase: 'ai',
        improve: 'ai',
        fix_grammar: 'ai',
        fix_style: 'ai',
    };
    return typeMap[type.toLowerCase()] || 'replace';
}

function updateStats(stats) {
    elements.statDirect.textContent = stats.direct_actions || 0;
    elements.statAi.textContent = stats.ai_actions || 0;
    elements.statChars.textContent = (state.currentText.length - state.originalText.length) || 0;
}

function clearActions() {
    state.actions = [];
    state.executedActions = [];
    renderActions();
}

// ============================================
// ACTION EXECUTION
// ============================================

async function executeAction(index) {
    if (state.isRunning) return;

    const action = state.actions[index];
    if (!action) return;

    const item = elements.actionsQueue.querySelector(`[data-index="${index}"]`);
    if (!item) return;

    // Update UI state
    state.isRunning = true;
    item.classList.remove('ready', 'pending');
    item.classList.add('running');
    item.querySelector('.action-status').textContent = '\u23F3'; // Hourglass

    try {
        const response = await fetch('/smart-edit/execute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: state.currentText,
                action: {
                    id: action.id,
                    type: action.type,
                    target: action.target,
                    content: action.content || action.replacement,
                    instruction: action.instruction,
                    description: action.description,
                    category: action.category,
                    metadata: action.metadata || {},
                },
            }),
        });

        const result = await response.json();

        if (result.success) {
            // Store the old text for animation reference
            const oldText = state.currentText;

            // Update state BEFORE animation (so animation can reference correct state)
            state.executedActions.push({ ...action, result });

            // Animate the change (this will update the preview at the end)
            await animateChange(result.change, action.type, result.text_after);

            // Now update state to new text
            state.currentText = result.text_after;

            // Update UI
            item.classList.remove('running');
            item.classList.add('completed');
            item.querySelector('.action-status').textContent = '\u2713'; // Checkmark

            // Update side views
            elements.sideOriginal.textContent = state.originalText;
            elements.sideEdited.textContent = state.currentText;

            // Update stats
            elements.statChars.textContent = state.currentText.length - state.originalText.length;
        } else {
            item.classList.remove('running');
            item.classList.add('failed');
            item.querySelector('.action-status').textContent = '\u2717'; // X mark
            showToast(result.error || 'Action failed', 'error');
        }
    } catch (error) {
        console.error('Execute error:', error);
        item.classList.remove('running');
        item.classList.add('failed');
        item.querySelector('.action-status').textContent = '\u2717';
        showToast('Execution failed: ' + error.message, 'error');
    } finally {
        state.isRunning = false;
    }
}

async function handleRunAll() {
    if (state.isRunning) return;

    const pendingActions = state.actions.filter((_, i) => {
        const item = elements.actionsQueue.querySelector(`[data-index="${i}"]`);
        return item && !item.classList.contains('completed');
    });

    if (pendingActions.length === 0) {
        showToast('All actions already executed', 'info');
        return;
    }

    elements.runAllBtn.disabled = true;

    for (let i = 0; i < state.actions.length; i++) {
        const item = elements.actionsQueue.querySelector(`[data-index="${i}"]`);
        if (item && !item.classList.contains('completed')) {
            await executeAction(i);

            // Pause between actions if enabled
            if (state.settings.pauseBetween && i < state.actions.length - 1) {
                await sleep(500 / state.animationSpeed);
            }
        }
    }

    elements.runAllBtn.disabled = false;
    showToast('All actions completed!', 'success');
    celebrateCompletion();
}

/**
 * Celebration effect when all actions complete successfully
 */
function celebrateCompletion() {
    if (typeof confetti !== 'function') return;

    // First burst - center
    confetti({
        particleCount: 100,
        spread: 70,
        origin: { y: 0.6 },
        colors: ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b']
    });

    // Second burst - left side (delayed)
    setTimeout(() => {
        confetti({
            particleCount: 50,
            angle: 60,
            spread: 55,
            origin: { x: 0, y: 0.6 },
            colors: ['#10b981', '#3b82f6']
        });
    }, 150);

    // Third burst - right side (delayed)
    setTimeout(() => {
        confetti({
            particleCount: 50,
            angle: 120,
            spread: 55,
            origin: { x: 1, y: 0.6 },
            colors: ['#8b5cf6', '#f59e0b']
        });
    }, 300);
}

// ============================================
// ANIMATIONS - Phase 2 Enhanced
// ============================================

/**
 * Animation timing configuration (in ms, adjusted by animationSpeed)
 */
const ANIMATION_TIMING = {
    highlight: 400,
    strikethrough: 300,
    fade: 250,
    collapse: 200,
    typewriter: 50,  // per character
    morphBlur: 300,
    glowFade: 400,
    markersPop: 200,
};

/**
 * Get adjusted timing based on animation speed setting
 */
function getTime(baseMs) {
    return baseMs / state.animationSpeed;
}

/**
 * Main animation orchestrator - routes to specific animation based on type
 */
async function animateChange(change, type, newText) {
    if (!change || !state.settings.highlightChanges) {
        // No animation - just update
        updatePreviewText(newText);
        return;
    }

    const normalizedType = type.toLowerCase();

    switch (normalizedType) {
        case 'delete':
            await animateDelete(change, newText);
            break;
        case 'insert':
        case 'insert_after':
        case 'insert_before':
            await animateInsert(change, newText);
            break;
        case 'replace':
            await animateReplace(change, newText);
            break;
        case 'format':
            await animateFormat(change, newText);
            break;
        case 'rephrase':
        case 'improve':
        case 'fix_grammar':
        case 'fix_style':
        case 'expand':
        case 'condense':
            await animateAI(change, newText, normalizedType);
            break;
        default:
            // Fallback: simple highlight
            await animateGeneric(change, newText);
    }
}

/**
 * DELETE Animation: highlight red -> strikethrough -> fade -> collapse -> update
 */
async function animateDelete(change, newText) {
    const { position, removed } = change;
    const text = state.currentText;

    // Phase 1: Render with target highlighted in red
    const beforeTarget = escapeHtml(text.substring(0, position.start));
    const target = escapeHtml(removed);
    const afterTarget = escapeHtml(text.substring(position.end));

    elements.previewText.innerHTML =
        `${beforeTarget}<span class="edit-target delete-highlight">${target}</span>${afterTarget}`;

    await sleep(getTime(ANIMATION_TIMING.highlight));

    // Phase 2: Add strikethrough
    const targetSpan = elements.previewText.querySelector('.edit-target');
    if (targetSpan) {
        targetSpan.classList.add('delete-strikethrough');
        await sleep(getTime(ANIMATION_TIMING.strikethrough));

        // Phase 3: Fade out
        targetSpan.classList.add('delete-fade');
        await sleep(getTime(ANIMATION_TIMING.fade));

        // Phase 4: Collapse (animate width to 0)
        targetSpan.classList.add('delete-collapse');
        await sleep(getTime(ANIMATION_TIMING.collapse));
    }

    // Final: Update to new text
    updatePreviewText(newText);
}

/**
 * INSERT Animation: cursor blink at position -> typewriter effect -> glow fade
 */
async function animateInsert(change, newText) {
    const { position, inserted } = change;
    const text = state.currentText;

    // Phase 1: Show cursor at insert position
    const beforeInsert = escapeHtml(text.substring(0, position.start));
    const afterInsert = escapeHtml(text.substring(position.start));

    elements.previewText.innerHTML =
        `${beforeInsert}<span class="edit-cursor"></span>${afterInsert}`;

    await sleep(getTime(300)); // Brief cursor blink

    // Phase 2: Typewriter effect - insert characters one by one
    const cursorSpan = elements.previewText.querySelector('.edit-cursor');
    if (cursorSpan) {
        const insertSpan = document.createElement('span');
        insertSpan.className = 'edit-target insert-appear';
        cursorSpan.parentNode.insertBefore(insertSpan, cursorSpan);

        // Type out the inserted text character by character
        for (let i = 0; i < inserted.length; i++) {
            insertSpan.textContent += inserted[i];
            await sleep(getTime(ANIMATION_TIMING.typewriter));
        }

        // Remove cursor
        cursorSpan.remove();

        // Phase 3: Glow fade
        insertSpan.classList.add('insert-glow');
        await sleep(getTime(ANIMATION_TIMING.glowFade));
    }

    // Final: Update to new text
    updatePreviewText(newText);
}

/**
 * REPLACE Animation: orange highlight on old -> blur morph -> new text with green glow
 */
async function animateReplace(change, newText) {
    const { position, removed, inserted } = change;
    const text = state.currentText;

    // Phase 1: Highlight old text in orange
    const beforeTarget = escapeHtml(text.substring(0, position.start));
    const target = escapeHtml(removed);
    const afterTarget = escapeHtml(text.substring(position.end));

    elements.previewText.innerHTML =
        `${beforeTarget}<span class="edit-target replace-old">${target}</span>${afterTarget}`;

    await sleep(getTime(ANIMATION_TIMING.highlight));

    // Phase 2: Blur/morph effect
    const targetSpan = elements.previewText.querySelector('.edit-target');
    if (targetSpan) {
        targetSpan.classList.add('replace-morph');
        await sleep(getTime(ANIMATION_TIMING.morphBlur));

        // Phase 3: Transform to new text with green glow
        targetSpan.textContent = inserted || '';
        targetSpan.classList.remove('replace-old', 'replace-morph');
        targetSpan.classList.add('replace-new');

        await sleep(getTime(ANIMATION_TIMING.glowFade));
    }

    // Final: Update to new text
    updatePreviewText(newText);
}

/**
 * FORMAT Animation: yellow highlight -> markers pop in -> apply style
 */
async function animateFormat(change, newText) {
    const { position, removed, inserted } = change;
    const text = state.currentText;

    // Phase 1: Yellow highlight on target
    const beforeTarget = escapeHtml(text.substring(0, position.start));
    const target = escapeHtml(removed);
    const afterTarget = escapeHtml(text.substring(position.end));

    elements.previewText.innerHTML =
        `${beforeTarget}<span class="edit-target format-highlight">${target}</span>${afterTarget}`;

    await sleep(getTime(ANIMATION_TIMING.highlight));

    // Phase 2: Show the formatted version with markers appearing
    const targetSpan = elements.previewText.querySelector('.edit-target');
    if (targetSpan) {
        // Detect format type from inserted text
        let prefix = '', suffix = '', innerText = removed;

        if (inserted.startsWith('**') && inserted.endsWith('**')) {
            prefix = '**'; suffix = '**';
            innerText = inserted.slice(2, -2);
        } else if (inserted.startsWith('*') && inserted.endsWith('*')) {
            prefix = '*'; suffix = '*';
            innerText = inserted.slice(1, -1);
        } else if (inserted.startsWith('`') && inserted.endsWith('`')) {
            prefix = '`'; suffix = '`';
            innerText = inserted.slice(1, -1);
        } else if (inserted.startsWith('~~') && inserted.endsWith('~~')) {
            prefix = '~~'; suffix = '~~';
            innerText = inserted.slice(2, -2);
        }

        // Animate markers appearing
        targetSpan.innerHTML =
            `<span class="format-marker format-pop">${escapeHtml(prefix)}</span>` +
            `<span class="format-inner">${escapeHtml(innerText)}</span>` +
            `<span class="format-marker format-pop">${escapeHtml(suffix)}</span>`;

        await sleep(getTime(ANIMATION_TIMING.markersPop));

        // Phase 3: Apply actual formatting style
        targetSpan.classList.remove('format-highlight');
        targetSpan.classList.add('format-applied');

        await sleep(getTime(ANIMATION_TIMING.glowFade));
    }

    // Final: Update to new text
    updatePreviewText(newText);
}

/**
 * AI Operation Animation: pulsing border -> blur content -> typewriter new text -> glow fade
 * Used for: rephrase, improve, fix_grammar, fix_style, expand, condense
 */
async function animateAI(change, newText, operationType) {
    const { position, removed, inserted } = change;
    const text = state.currentText;

    // Phase 1: Show the target segment with pulsing AI border
    const beforeTarget = escapeHtml(text.substring(0, position.start));
    const target = escapeHtml(removed);
    const afterTarget = escapeHtml(text.substring(position.end));

    // Create AI processing container with progress indicator
    elements.previewText.innerHTML =
        `${beforeTarget}<span class="edit-target ai-processing">` +
        `<span class="ai-content">${target}</span>` +
        `<span class="ai-progress-bar"><span class="ai-progress-fill"></span></span>` +
        `</span>${afterTarget}`;

    // Let the pulsing animation show for a moment
    await sleep(getTime(600));

    // Phase 2: Blur the old content
    const targetSpan = elements.previewText.querySelector('.edit-target');
    const contentSpan = targetSpan?.querySelector('.ai-content');
    if (contentSpan) {
        contentSpan.classList.add('ai-blur');
        await sleep(getTime(300));
    }

    // Phase 3: Typewriter effect for new content
    if (targetSpan && contentSpan) {
        // Clear the blurred content
        contentSpan.textContent = '';
        contentSpan.classList.remove('ai-blur');
        contentSpan.classList.add('ai-typewriter');

        // Type out the new text character by character (faster than regular insert)
        const typeSpeed = Math.max(10, Math.min(30, 1000 / inserted.length)); // Adaptive speed
        for (let i = 0; i < inserted.length; i++) {
            contentSpan.textContent += inserted[i];
            await sleep(getTime(typeSpeed));
        }

        // Remove progress bar and cursor
        const progressBar = targetSpan.querySelector('.ai-progress-bar');
        if (progressBar) progressBar.remove();
        contentSpan.classList.remove('ai-typewriter');

        // Phase 4: Success glow
        targetSpan.classList.remove('ai-processing');
        targetSpan.classList.add('ai-complete');
        await sleep(getTime(500));
    }

    // Final: Update to new text
    updatePreviewText(newText);
}

/**
 * Generic fallback animation
 */
async function animateGeneric(change, newText) {
    const { position, removed } = change;
    const text = state.currentText;

    if (removed && position) {
        const beforeTarget = escapeHtml(text.substring(0, position.start));
        const target = escapeHtml(removed);
        const afterTarget = escapeHtml(text.substring(position.end));

        elements.previewText.innerHTML =
            `${beforeTarget}<span class="edit-target generic-highlight">${target}</span>${afterTarget}`;

        await sleep(getTime(ANIMATION_TIMING.highlight * 2));
    }

    updatePreviewText(newText);
}

/**
 * Update preview text (plain text, no HTML)
 */
function updatePreviewText(text) {
    if (!text) {
        elements.previewText.innerHTML = '<span class="placeholder">Preview will appear here...</span>';
    } else {
        elements.previewText.textContent = text;
    }
}

// ============================================
// VIEW MODES
// ============================================

function handleViewModeChange(btn) {
    const view = btn.dataset.view;

    // Update button states
    elements.viewBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    // Update view visibility
    elements.previewLive.classList.remove('preview-active');
    elements.previewDiff.classList.remove('preview-active');
    elements.previewSide.classList.remove('preview-active');

    switch (view) {
        case 'live':
            elements.previewLive.classList.add('preview-active');
            break;
        case 'diff':
            elements.previewDiff.classList.add('preview-active');
            updateDiffView();
            break;
        case 'side':
            elements.previewSide.classList.add('preview-active');
            updateSideView();
            break;
    }
}

function updateDiffView() {
    const diff = generateSimpleDiff(state.originalText, state.currentText);
    elements.diffOutput.innerHTML = diff;
}

function updateSideView() {
    elements.sideOriginal.textContent = state.originalText;
    elements.sideEdited.textContent = state.currentText;
}

function generateSimpleDiff(original, edited) {
    if (original === edited) {
        return '<span class="text-muted">No changes yet</span>';
    }

    // Simple line-based diff visualization
    const originalLines = original.split('\n');
    const editedLines = edited.split('\n');

    let result = '';
    const maxLines = Math.max(originalLines.length, editedLines.length);

    for (let i = 0; i < maxLines; i++) {
        const origLine = originalLines[i] || '';
        const editLine = editedLines[i] || '';

        if (origLine !== editLine) {
            if (origLine) {
                result += `<span class="diff-remove">- ${escapeHtml(origLine)}</span>\n`;
            }
            if (editLine) {
                result += `<span class="diff-add">+ ${escapeHtml(editLine)}</span>\n`;
            }
        } else {
            result += `  ${escapeHtml(origLine)}\n`;
        }
    }

    return result;
}

// ============================================
// UTILITIES
// ============================================

function handleCopyResult() {
    navigator.clipboard.writeText(state.currentText).then(() => {
        showToast('Copied to clipboard!', 'success');
    }).catch(() => {
        showToast('Failed to copy', 'error');
    });
}

function handleThemeToggle() {
    const isDark = document.body.classList.contains('dark-mode');

    if (isDark) {
        document.body.classList.remove('dark-mode');
        document.body.classList.add('light-mode');
        elements.themeToggle.querySelector('.theme-icon').textContent = '\u263D'; // Moon
        localStorage.setItem('smart-edit-theme', 'light');
    } else {
        document.body.classList.remove('light-mode');
        document.body.classList.add('dark-mode');
        elements.themeToggle.querySelector('.theme-icon').textContent = '\u2600'; // Sun
        localStorage.setItem('smart-edit-theme', 'dark');
    }
}

function showLoading(text = 'Loading...') {
    elements.loadingText.textContent = text;
    elements.loadingOverlay.classList.add('active');
}

function hideLoading() {
    elements.loadingOverlay.classList.remove('active');
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    elements.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 3000);
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
