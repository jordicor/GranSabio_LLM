// ============================================
// GRAN SABIO LLM STREAM MONITOR
// Direct connection (no proxy needed)
// ============================================

// Direct connection - no proxy needed
const STREAM_BASE = '';
// UI panels (note: 'analysis' combines Auto-QA, preflight, and consensus)
const PHASES = ['generation', 'qa', 'arbiter', 'smartedit', 'analysis', 'gransabio', 'status', 'everything'];
// Phases that can be hard-switched (excluded from subscription)
const HARD_SWITCHABLE_PHASES = ['generation', 'qa', 'arbiter', 'smartedit', 'analysis', 'gransabio', 'status'];
const REQUEST_PANEL_PHASES = ['generation', 'qa', 'arbiter', 'analysis', 'gransabio', 'smartedit'];
const ITERATION_CONTENT_KEYS = ['auto_qa', 'preflight', 'generation', 'qa', 'arbiter', 'consensus', 'gransabio'];
const ANALYSIS_CONTENT_KEYS = ['auto_qa', 'preflight', 'consensus'];
const SERVER_PHASES_BY_PANEL = {
    analysis: ['auto_qa', 'preflight', 'consensus'],
    smartedit: ['smart_edit'],
    gransabio: ['gran_sabio'],
};
const TERMINAL_STATUSES = new Set([
    'completed',
    'rejected',
    'failed',
    'error',
    'cancelled',
    'auto_qa_rejected',
    'preflight_rejected',
]);
const ACTIVE_SESSION_STATUSES = new Set([
    'initializing',
    'running',
    'generating',
    'qa_evaluation',
    'consensus',
    'gran_sabio_review',
]);
const GRAN_SABIO_PHASES = new Set([
    'inline_deal_breaker_review',
    'gran_sabio_review',
    'gran_sabio_regeneration',
]);
const KNOWN_STRUCTURED_EVENTS = new Set([
    'retry_start',
    'retry',
    'error',
    'project_end',
    'project_cancelled',
    'validate_draft_oversize',
    'context_overflow_midloop',
    'force_finalize',
    'tool_call_start',
    'tool_call_delta',
    'tool_call_ready',
    'tool_call_result',
    'tool_call_error',
    'tool_loop_turn_start',
    'tool_loop_turn_done',
    'tool_loop_budget_state',
    'tool_loop_budget_warning',
    'assistant_delta',
    'thinking_delta',
    'assistant_text_done',
    'provider_usage_delta',
    'tool_loop_provider_terminal',
    'tool_loop_error',
    'tool_loop_complete',
]);
const DEFAULT_PANEL_MAX_CHARS = 80000;
const EVERYTHING_PANEL_MAX_CHARS = 40000;

// State
let eventSource = null;
let currentProjectId = null;
let stats = {};
let lastProjectData = null;  // Cache for re-rendering dashboard

// Helper to check if we're in Overview mode
function isOverviewMode() {
    return viewState.selectedRequestId === null;
}

function normalizePhaseName(phase, fallback = 'generation') {
    if (!phase) {
        return fallback;
    }
    if (phase === 'gran_sabio') {
        return 'gransabio';
    }
    if (phase === 'smart_edit') {
        return 'smartedit';
    }
    return phase;
}

function getDisplayPhase(phase) {
    const normalized = normalizePhaseName(phase, 'status');
    if (ANALYSIS_CONTENT_KEYS.includes(normalized)) {
        return 'analysis';
    }
    if (PHASES.includes(normalized) && normalized !== 'everything') {
        return normalized;
    }
    return 'status';
}

function getServerPhasesForPanel(panelPhase) {
    return SERVER_PHASES_BY_PANEL[panelPhase] || [panelPhase];
}

function isTerminalStatus(status) {
    return TERMINAL_STATUSES.has((status || '').toLowerCase());
}

function isActiveSessionStatus(status) {
    return ACTIVE_SESSION_STATUSES.has((status || '').toLowerCase());
}

function isRequestActiveStatus(status) {
    const normalized = (status || '').toLowerCase();
    return normalized === 'active' || isActiveSessionStatus(normalized);
}

// Panel toggle states: 'on' | 'soft' | 'hard'
// 'everything' panel only supports 'on' | 'soft' (no hard switch)
const panelStates = {};
PHASES.forEach(phase => {
    panelStates[phase] = 'on';
});

// Initialize stats
PHASES.forEach(phase => {
    stats[phase] = { chunks: 0, bytes: 0 };
});

// ============================================
// REQUEST TRACKING (Phase 1)
// ============================================

// Request data storage
const requestsData = {
    orderedRequestIds: [],  // Maintains order of arrival
    requests: {}            // Keyed by session_id
};

// View state for request selection
const viewState = {
    selectedRequestId: null,
    autoFollow: true,        // Auto-select new requests when they arrive
    tabsViewOffset: 0        // Offset for tabs sliding window (pagination)
};

// Maximum visible tabs before overflow (including Overview tab)
const MAX_VISIBLE_TABS = 6;

// QA Filter state - for filtering QA panel by layer/model
const qaFilterState = {
    layer: null,   // null = show all, string = filter by this layer
    model: null    // null = show all, string = filter by this model
};

// Analysis Filter state: null | 'auto_qa' | 'preflight' | 'consensus'
let analysisFilterState = null;

// ============================================
// PHASE TOGGLE MANAGEMENT
// ============================================

function initToggleListeners() {
    document.querySelectorAll('.phase-toggle').forEach(toggle => {
        const phase = toggle.dataset.phase;
        toggle.querySelectorAll('.toggle-segment').forEach(segment => {
            segment.addEventListener('click', () => {
                const newState = segment.dataset.state;
                setPhaseState(phase, newState);
            });
        });
    });
}

function setPhaseState(phase, newState) {
    const oldState = panelStates[phase];

    // No change
    if (oldState === newState) return;

    panelStates[phase] = newState;
    updateToggleUI(phase, newState);
    updatePanelVisual(phase, newState);

    log(`Phase '${phase}' switched: ${oldState} -> ${newState}`, 'info');

    // Hard switch requires reconnection (but not for 'everything' panel)
    if (phase !== 'everything' && (newState === 'hard' || oldState === 'hard')) {
        if (eventSource && currentProjectId) {
            log('Hard switch detected - reconnecting with updated phases...', 'warn');
            reconnectWithActivePhases();
        }
    }
}

function updateToggleUI(phase, state) {
    const toggle = document.querySelector(`.phase-toggle[data-phase="${phase}"]`);
    if (!toggle) return;

    toggle.querySelectorAll('.toggle-segment').forEach(segment => {
        segment.classList.remove('active');
        if (segment.dataset.state === state) {
            segment.classList.add('active');
        }
    });
}

function updatePanelVisual(phase, state) {
    const panel = document.querySelector(`.stream-panel[data-phase="${phase}"]`);
    if (!panel) return;

    panel.classList.remove('muted-soft', 'muted-hard');
    if (state === 'soft') {
        panel.classList.add('muted-soft');
    } else if (state === 'hard') {
        panel.classList.add('muted-hard');
    }
}

function getActivePhases() {
    // Return server-side phases that are NOT in 'hard' state
    const serverPhases = [];

    HARD_SWITCHABLE_PHASES.forEach(phase => {
        if (panelStates[phase] !== 'hard') {
            serverPhases.push(...getServerPhasesForPanel(phase));
        }
    });

    return serverPhases;
}

function reconnectWithActivePhases() {
    if (!currentProjectId) return;

    const activePhases = getActivePhases();

    if (activePhases.length === 0) {
        log('All phases are hard-muted. Disconnecting...', 'warn');
        disconnect();
        return;
    }

    // Close current connection
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }

    // Reconnect with only active phases
    // Use 'all' when all UI phases are active (none hard-muted)
    const allPhasesActive = HARD_SWITCHABLE_PHASES.every(phase => panelStates[phase] !== 'hard');
    const phasesParam = allPhasesActive ? 'all' : activePhases.join(',');
    const streamUrl = `${STREAM_BASE}/stream/project/${encodeURIComponent(currentProjectId)}?phases=${encodeURIComponent(phasesParam)}`;

    log(`Reconnecting with phases: ${phasesParam}`, 'info');
    setConnectionStatus('connecting', 'RECONNECTING...');

    try {
        eventSource = new EventSource(streamUrl);

        eventSource.onopen = function() {
            log('Reconnected successfully', 'success');
            setConnectionStatus('connected', 'ONLINE');
            // Update badges for active phases only
            PHASES.forEach(phase => {
                if (panelStates[phase] === 'hard') {
                    setBadge(phase, 'muted');
                } else {
                    setBadge(phase, 'active');
                }
            });
        };

        eventSource.onmessage = function(event) {
            handleMessage(event.data);
        };

        eventSource.onerror = function(error) {
            handleConnectionError(error);
        };

    } catch (error) {
        log(`Reconnection failed: ${error.message}`, 'error');
        setConnectionStatus('disconnected', 'ERROR');
    }
}

function handleConnectionError(error) {
    console.error('EventSource error:', error);

    if (eventSource.readyState === EventSource.CLOSED) {
        log('Connection closed by server', 'warn');
        setConnectionStatus('disconnected', 'CLOSED');
        PHASES.forEach(phase => setBadge(phase, 'idle'));
    } else if (eventSource.readyState === EventSource.CONNECTING) {
        log('Reconnecting...', 'warn');
        setConnectionStatus('connecting', 'RECONNECTING...');
    } else {
        log('Connection error', 'error');
        setConnectionStatus('disconnected', 'ERROR');
    }
}

function resetAllTogglesToOn() {
    PHASES.forEach(phase => {
        panelStates[phase] = 'on';
        updateToggleUI(phase, 'on');
        updatePanelVisual(phase, 'on');
    });
}

// ============================================
// LOGGING
// ============================================

function log(message, type = 'info') {
    const logContent = document.getElementById('logContent');
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.innerHTML = `<span class="timestamp">[${timestamp}]</span>${escapeHtml(message)}`;
    logContent.appendChild(entry);
    logContent.scrollTop = logContent.scrollHeight;

    // Also log to console for debugging
    console.log(`[${type.toUpperCase()}] ${message}`);
}

function clearLog() {
    document.getElementById('logContent').innerHTML = '';
    log('Log cleared', 'info');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================
// UI UPDATES
// ============================================

function setConnectionStatus(status, message) {
    const indicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    const connectBtn = document.getElementById('connectBtn');
    const disconnectBtn = document.getElementById('disconnectBtn');

    indicator.className = `status-indicator ${status}`;
    statusText.textContent = message;

    if (status === 'connected') {
        connectBtn.style.display = 'none';
        disconnectBtn.style.display = 'inline-block';
    } else {
        connectBtn.style.display = 'inline-block';
        disconnectBtn.style.display = 'none';
        connectBtn.disabled = (status === 'connecting');
    }
}

function setBadge(phase, status) {
    const badge = document.getElementById(`badge-${phase}`);
    if (!badge) return;
    badge.textContent = status;
    badge.className = 'stream-badge';
    if (status === 'active' || status === 'live') {
        badge.classList.add('active');
    } else if (status === 'receiving') {
        badge.classList.add('receiving');
    }
}

function resetPanelContent(phase, options = {}) {
    const contentEl = document.getElementById(`content-${phase}`);
    if (contentEl) {
        if (options.overviewMode) {
            contentEl.dataset.overviewMode = 'true';
        } else if (options.clearOverviewMode) {
            delete contentEl.dataset.overviewMode;
        }
        if (options.clearText !== false) {
            contentEl.textContent = '';
            delete contentEl.dataset.omittedChars;
        }
    }

    if (options.resetStats !== false) {
        const chunksEl = document.getElementById(`chunks-${phase}`);
        const bytesEl = document.getElementById(`bytes-${phase}`);
        if (chunksEl) chunksEl.textContent = options.statsText ?? '0';
        if (bytesEl) bytesEl.textContent = options.statsText ?? '0';
    }
    if (options.resetBadge !== false) {
        setBadge(phase, options.badge || 'idle');
    }
}

function getAnalysisStats(iterData) {
    return ANALYSIS_CONTENT_KEYS.reduce((total, phase) => {
        const phaseStats = iterData?.stats?.[phase] || { chunks: 0, bytes: 0 };
        total.chunks += phaseStats.chunks || 0;
        total.bytes += phaseStats.bytes || 0;
        return total;
    }, { chunks: 0, bytes: 0 });
}

function updateAnalysisStats(iterData) {
    const totals = getAnalysisStats(iterData);
    const analysisChunksEl = document.getElementById('chunks-analysis');
    const analysisBytesEl = document.getElementById('bytes-analysis');
    if (analysisChunksEl) analysisChunksEl.textContent = totals.chunks;
    if (analysisBytesEl) analysisBytesEl.textContent = totals.bytes;
}

function appendContent(phase, text) {
    const content = document.getElementById(`content-${phase}`);
    if (!content) return;

    // Check toggle state - soft/hard muted panels don't display content
    const state = panelStates[phase];
    if (state === 'soft' || state === 'hard') {
        // Still update stats for visibility, but don't append content
        stats[phase].chunks++;
        stats[phase].bytes += text.length;

        const chunksEl = document.getElementById(`chunks-${phase}`);
        const bytesEl = document.getElementById(`bytes-${phase}`);
        if (chunksEl) chunksEl.textContent = stats[phase].chunks;
        if (bytesEl) bytesEl.textContent = stats[phase].bytes;

        // Brief flash to show data is arriving but muted
        setBadge(phase, 'muted');
        return;
    }

    content.textContent += text;
    trimPanelContent(phase, content);
    content.scrollTop = content.scrollHeight;

    // Update stats
    stats[phase].chunks++;
    stats[phase].bytes += text.length;

    const chunksEl = document.getElementById(`chunks-${phase}`);
    const bytesEl = document.getElementById(`bytes-${phase}`);
    if (chunksEl) chunksEl.textContent = stats[phase].chunks;
    if (bytesEl) bytesEl.textContent = stats[phase].bytes;

    // Show receiving state with pulse - debounced return to active
    showReceiving(phase);
}

function getPanelMaxChars(phase) {
    return phase === 'everything' ? EVERYTHING_PANEL_MAX_CHARS : DEFAULT_PANEL_MAX_CHARS;
}

function trimPanelContent(phase, contentEl) {
    const maxChars = getPanelMaxChars(phase);
    const currentText = contentEl.textContent || '';
    if (currentText.length <= maxChars) return;

    const previousOmitted = Number(contentEl.dataset.omittedChars || 0);
    const overflow = currentText.length - maxChars;
    const omitted = previousOmitted + overflow;
    const retained = currentText.slice(-maxChars);
    contentEl.dataset.omittedChars = String(omitted);
    contentEl.textContent = `[... ${omitted.toLocaleString()} chars omitted from this panel ...]\n${retained}`;
}

// Debounce timers for each phase
const receivingTimers = {};

function showReceiving(phase) {
    const badge = document.getElementById(`badge-${phase}`);
    if (!badge) return;

    // Clear any existing timer for this phase
    if (receivingTimers[phase]) {
        clearTimeout(receivingTimers[phase]);
    }

    // Set to receiving with pulse effect
    badge.textContent = 'receiving';
    badge.className = 'stream-badge receiving pulse';

    // Debounced return to active - only after 400ms of no chunks
    receivingTimers[phase] = setTimeout(() => {
        badge.textContent = 'active';
        badge.className = 'stream-badge active';
    }, 400);
}

function clearAllContent() {
    // Clear request tracking structures
    requestsData.orderedRequestIds = [];
    requestsData.requests = {};
    viewState.selectedRequestId = null;  // Start in Overview mode
    viewState.autoFollow = true;
    lastProjectData = null;

    // Close overflow menu if open
    const overflowMenu = document.getElementById('tabsOverflowMenu');
    if (overflowMenu) overflowMenu.classList.add('hidden');

    // Clear phase panels and set overview mode
    PHASES.forEach(phase => {
        const content = document.getElementById(`content-${phase}`);
        if (content) {
            content.textContent = '';
            // Set overview mode flag for placeholder message
            if (phase !== 'status' && phase !== 'everything') {
                content.dataset.overviewMode = 'true';
            }
        }

        const chunksEl = document.getElementById(`chunks-${phase}`);
        const bytesEl = document.getElementById(`bytes-${phase}`);
        if (chunksEl) chunksEl.textContent = phase === 'status' || phase === 'everything' ? '0' : '-';
        if (bytesEl) bytesEl.textContent = phase === 'status' || phase === 'everything' ? '0' : '-';

        setBadge(phase, 'idle');
        stats[phase] = { chunks: 0, bytes: 0 };
    });

    // Render tabs (Overview will be shown)
    renderRequestTabs();

    // Hide iteration timeline (no session selected)
    const timeline = document.getElementById('iterationTimeline');
    if (timeline) timeline.classList.add('hidden');

    // Hide status dashboard until data arrives
    hideDashboard();

    // Reset QA filters
    resetQAFilters();

    // Reset Analysis filter
    resetAnalysisFilter();
}

// ============================================
// REQUEST PROCESSING
// ============================================

/**
 * Extract request info from event data and create/update request entry
 * @param {Object} data - Event data with session_id and request_name
 * @returns {Object|null} The request object, or null if no session_id
 */
function processRequestFromEvent(data) {
    const sessionId = data.session_id;
    const requestName = data.request_name || 'unknown';

    if (!sessionId) return null;

    // Create request if it doesn't exist
    if (!requestsData.requests[sessionId]) {
        const iteration = data.iteration || 1;

        requestsData.requests[sessionId] = {
            sessionId: sessionId,
            requestName: requestName,
            status: 'active',
            currentIteration: iteration,
            viewingIteration: iteration,      // Which iteration user is viewing
            autoFollowIteration: true,        // Auto-switch to new iterations
            maxIterations: data.max_iterations || 5,
            finalScore: null,
            // Content organized by iteration
            iterations: {}
        };

        // Initialize first iteration
        initializeIteration(requestsData.requests[sessionId], iteration);

        // Add to ordered list
        requestsData.orderedRequestIds.push(sessionId);

        // Auto-select if autoFollow is on (switch from Overview to this request)
        if (viewState.autoFollow) {
            viewState.selectedRequestId = sessionId;

            // Clear overview mode flag and prepare panels for content
            REQUEST_PANEL_PHASES.forEach(phase => {
                resetPanelContent(phase, { clearOverviewMode: true });
            });

            // Update dashboard for this specific session
            renderProjectStatus(lastProjectData, sessionId);
        }

        // Trigger UI update
        renderRequestTabs();
        renderIterationTimeline();

        log(`New request detected: ${requestName} (${sessionId.slice(0, 8)}...)`, 'info');
    }

    return requestsData.requests[sessionId];
}

/**
 * Initialize a new iteration structure for a request
 * @param {Object} request - The request object
 * @param {number} iteration - The iteration number to initialize
 */
function initializeIteration(request, iteration) {
    if (!request.iterations[iteration]) {
        const content = {};
        const phaseStats = {};
        ITERATION_CONTENT_KEYS.forEach(phase => {
            content[phase] = '';
            phaseStats[phase] = { chunks: 0, bytes: 0 };
        });

        request.iterations[iteration] = {
            content: content,
            stats: phaseStats,
            // QA filter structures - indexed content for filtering
            qaByLayer: {},       // { "Layer Name": "accumulated content..." }
            qaByModel: {},       // { "model-name": "accumulated content..." }
            qaByLayerModel: {},  // { "Layer Name": { "model-name": "content..." } }
            qaLayers: [],        // List of unique layer names seen
            qaModels: [],        // List of unique model names seen
            score: null,
            approved: false,
            status: 'in_progress',  // 'in_progress' | 'rejected' | 'approved'
            timestampStart: Date.now(),
            timestampEnd: null
        };
    }
}

// ============================================
// REQUEST TABS UI
// ============================================

function renderRequestTabs() {
    const container = document.getElementById('requestTabsContainer');
    const panel = document.getElementById('requestTabsPanel');
    const infoBar = document.getElementById('requestInfoBar');

    if (!container) return;

    // Always show the tabs panel (Overview is always available)
    if (panel) panel.style.display = 'block';
    if (infoBar) infoBar.style.display = 'flex';

    // Clear container
    container.innerHTML = '';

    // Always add Overview tab first (fixed, not closable)
    const overviewTab = createOverviewTab();
    container.appendChild(overviewTab);

    const requests = requestsData.orderedRequestIds;
    const totalRequests = requests.length;

    // Adjust max visible: -1 because Overview occupies one slot
    const maxSessionTabs = MAX_VISIBLE_TABS - 1;

    // Ensure offset is within valid bounds
    const maxOffset = Math.max(0, totalRequests - maxSessionTabs);
    if (viewState.tabsViewOffset > maxOffset) {
        viewState.tabsViewOffset = maxOffset;
    }
    if (viewState.tabsViewOffset < 0) {
        viewState.tabsViewOffset = 0;
    }

    const startIndex = viewState.tabsViewOffset;
    const endIndex = Math.min(startIndex + maxSessionTabs, totalRequests);
    const visibleCount = endIndex - startIndex;

    // Check if there are more tabs before/after the visible window
    const hasTabsBefore = startIndex > 0;
    const hasTabsAfter = endIndex < totalRequests;

    // Render session tabs from the current view window
    for (let i = startIndex; i < endIndex; i++) {
        const sessionId = requests[i];
        const request = requestsData.requests[sessionId];
        const tab = createRequestTab(request);
        container.appendChild(tab);
    }

    // Add overflow button if there are more tabs after
    if (hasTabsAfter) {
        const overflowCount = totalRequests - endIndex;
        const overflowBtn = document.createElement('button');
        overflowBtn.className = 'tabs-overflow-btn';
        overflowBtn.innerHTML = `+ ${overflowCount} more`;
        overflowBtn.onclick = (e) => {
            e.stopPropagation();
            toggleOverflowMenu();
        };
        container.appendChild(overflowBtn);

        // Render overflow menu items (all items not currently visible)
        renderOverflowMenu(requests.slice(endIndex));
    }

    // Update navigation buttons state
    updateTabsNavButtons(hasTabsBefore, hasTabsAfter);

    // Update info bar
    updateRequestInfoBar();

    // Update auto-follow button state
    updateAutoFollowButton();
}

/**
 * Toggle auto-follow mode for new requests
 */
function toggleAutoFollow() {
    if (viewState.autoFollow) {
        // Turn off
        viewState.autoFollow = false;
    } else {
        // Turn on and jump to latest request
        viewState.autoFollow = true;

        // Select the most recent request if any exist
        if (requestsData.orderedRequestIds.length > 0) {
            const latestSessionId = requestsData.orderedRequestIds[requestsData.orderedRequestIds.length - 1];
            viewState.selectedRequestId = latestSessionId;

            // Clear overview mode and load content
            REQUEST_PANEL_PHASES.forEach(phase => {
                resetPanelContent(phase, {
                    clearOverviewMode: true,
                    clearText: false,
                    resetBadge: false,
                    resetStats: false,
                });
            });

            loadRequestContent(latestSessionId);
            renderIterationTimeline();
            renderProjectStatus(lastProjectData, latestSessionId);
        }
    }

    // Update UI
    renderRequestTabs();
    log(`Auto-follow ${viewState.autoFollow ? 'enabled' : 'disabled'}`, 'info');
}

/**
 * Update the visual state of the auto-follow button
 */
function updateAutoFollowButton() {
    const btn = document.getElementById('autoFollowBtn');
    if (!btn) return;

    if (viewState.autoFollow) {
        btn.classList.add('active');
        btn.innerHTML = 'Follow &#9654;';  // right arrow
    } else {
        btn.classList.remove('active');
        btn.innerHTML = 'Follow';
    }
}

/**
 * Update the enabled/disabled state of tabs navigation buttons
 * @param {boolean} hasTabsBefore - Whether there are tabs before the current view
 * @param {boolean} hasTabsAfter - Whether there are tabs after the current view
 */
function updateTabsNavButtons(hasTabsBefore, hasTabsAfter) {
    const leftBtn = document.getElementById('tabsNavLeft');
    const rightBtn = document.getElementById('tabsNavRight');

    if (leftBtn) {
        leftBtn.disabled = !hasTabsBefore;
    }
    if (rightBtn) {
        rightBtn.disabled = !hasTabsAfter;
    }
}

/**
 * Navigate tabs view to the left (show earlier tabs)
 */
function navigateTabsLeft() {
    if (viewState.tabsViewOffset > 0) {
        viewState.tabsViewOffset--;
        renderRequestTabs();
        log('Navigated tabs left', 'info');
    }
}

/**
 * Navigate tabs view to the right (show later tabs)
 */
function navigateTabsRight() {
    const totalRequests = requestsData.orderedRequestIds.length;
    const maxSessionTabs = MAX_VISIBLE_TABS - 1;
    const maxOffset = Math.max(0, totalRequests - maxSessionTabs);

    if (viewState.tabsViewOffset < maxOffset) {
        viewState.tabsViewOffset++;
        renderRequestTabs();
        log('Navigated tabs right', 'info');
    }
}

/**
 * Ensure the selected request tab is visible in the current view window
 * Adjusts the offset if necessary to bring the selected tab into view
 * @param {string} sessionId - The session ID to ensure is visible
 */
function ensureTabVisible(sessionId) {
    const requests = requestsData.orderedRequestIds;
    const index = requests.indexOf(sessionId);

    if (index === -1) return;

    const maxSessionTabs = MAX_VISIBLE_TABS - 1;
    const startIndex = viewState.tabsViewOffset;
    const endIndex = startIndex + maxSessionTabs;

    // Check if the tab is outside the current view
    if (index < startIndex) {
        // Tab is before the view - scroll left
        viewState.tabsViewOffset = index;
    } else if (index >= endIndex) {
        // Tab is after the view - scroll right
        viewState.tabsViewOffset = index - maxSessionTabs + 1;
    }
    // else: tab is already visible, no change needed
}

function getRequestTabState(request) {
    const status = (request.status || 'active').toLowerCase();

    if (status === 'completed') {
        return {
            icon: '+',
            className: 'completed',
            infoText: request.finalScore !== null && request.finalScore !== undefined
                ? request.finalScore.toFixed(1)
                : 'done',
        };
    }
    if (status === 'pending') {
        return { icon: 'o', className: 'pending', infoText: 'pending' };
    }
    if (isTerminalStatus(status)) {
        const label = status.replace(/_/g, ' ');
        return { icon: status === 'cancelled' ? '-' : '!', className: 'failed', infoText: label };
    }

    return {
        icon: '*',
        className: 'active',
        infoText: `iter ${request.currentIteration}/${request.maxIterations}`,
    };
}

function createRequestTab(request) {
    const tab = document.createElement('div');
    tab.className = 'request-tab';
    tab.dataset.sessionId = request.sessionId;
    tab.title = `${request.requestName}\nSession: ${request.sessionId}`;

    if (request.sessionId === viewState.selectedRequestId) {
        tab.classList.add('selected');
    }

    // Determine icon and status text
    const tabState = getRequestTabState(request);

    // Truncate name if needed
    const shortName = request.requestName.length > 14
        ? request.requestName.slice(0, 11) + '...'
        : request.requestName;

    tab.innerHTML = `
        <div class="tab-header">
            <span class="tab-status-icon ${tabState.className}">[${tabState.icon}]</span>
            <span class="tab-name">${escapeHtml(shortName)}</span>
        </div>
        <div class="tab-info">${tabState.infoText}</div>
        <div class="tab-close-zone">
            <span class="tab-close-btn" title="Close tab">&times;</span>
        </div>
    `;

    // Click on tab selects the request
    tab.onclick = (e) => {
        // Don't select if clicking on close button
        if (e.target.classList.contains('tab-close-btn')) return;
        selectRequest(request.sessionId);
    };

    // Click on close button closes the tab
    const closeBtn = tab.querySelector('.tab-close-btn');
    closeBtn.onclick = (e) => {
        e.stopPropagation();
        closeRequest(request.sessionId);
    };

    return tab;
}

/**
 * Create the special Overview tab (always first, not closable)
 */
function createOverviewTab() {
    const tab = document.createElement('div');
    tab.className = 'request-tab overview-tab';
    tab.title = 'View all sessions overview';

    if (isOverviewMode()) {
        tab.classList.add('selected');
    }

    // Count active sessions for display
    const activeCount = Object.values(requestsData.requests)
        .filter(r => isRequestActiveStatus(r.status)).length;
    const totalCount = requestsData.orderedRequestIds.length;

    tab.innerHTML = `
        <div class="tab-header">
            <span class="tab-status-icon overview">[#]</span>
            <span class="tab-name">Overview</span>
        </div>
        <div class="tab-info">${activeCount}/${totalCount} active</div>
    `;
    // No close-zone - Overview tab cannot be closed

    tab.onclick = () => selectOverview();

    return tab;
}

/**
 * Select Overview mode - show all sessions in dashboard, clear streaming panels
 */
function selectOverview() {
    viewState.selectedRequestId = null;
    viewState.autoFollow = false;  // User manually selected Overview

    // Update UI
    renderRequestTabs();

    // Hide iteration timeline in overview mode
    renderIterationTimeline();

    // Clear streaming panels (show message instead)
    clearStreamingPanelsForOverview();

    // Render dashboard with ALL sessions (no filter)
    renderProjectStatus(lastProjectData, null);

    log('Switched to Overview mode', 'info');
}

/**
 * Clear streaming panels when in Overview mode
 */
function clearStreamingPanelsForOverview() {
    REQUEST_PANEL_PHASES.forEach(phase => {
        // Show dashes for stats in overview mode
        resetPanelContent(phase, { overviewMode: true, statsText: '-' });
    });
}

function selectRequest(sessionId) {
    viewState.selectedRequestId = sessionId;
    viewState.autoFollow = false;  // User manually selected, disable auto-follow

    // Reset QA filters when switching requests
    resetQAFilters();

    // Ensure the selected tab is visible in the view window
    ensureTabVisible(sessionId);

    // Update tabs UI
    renderRequestTabs();

    // Render iteration timeline for this session
    renderIterationTimeline();

    // Load streaming content for this session
    loadRequestContent(sessionId);

    // Clear overview mode flag from content elements
    REQUEST_PANEL_PHASES.forEach(phase => {
        resetPanelContent(phase, {
            clearOverviewMode: true,
            clearText: false,
            resetBadge: false,
            resetStats: false,
        });
    });

    // Render dashboard filtered to only this session
    renderProjectStatus(lastProjectData, sessionId);

    log(`Switched to request: ${requestsData.requests[sessionId]?.requestName || sessionId}`, 'info');
}

function closeRequest(sessionId) {
    const request = requestsData.requests[sessionId];
    const requestName = request?.requestName || sessionId.slice(0, 8);

    // Remove from ordered list
    const index = requestsData.orderedRequestIds.indexOf(sessionId);
    if (index > -1) {
        requestsData.orderedRequestIds.splice(index, 1);
    }

    // Remove from requests object
    delete requestsData.requests[sessionId];

    // If this was the selected request, select another one
    if (viewState.selectedRequestId === sessionId) {
        if (requestsData.orderedRequestIds.length > 0) {
            // Select the previous tab, or the first one if we closed the first
            const newIndex = Math.max(0, index - 1);
            viewState.selectedRequestId = requestsData.orderedRequestIds[newIndex];
            loadRequestContent(viewState.selectedRequestId);
            renderIterationTimeline();
            renderProjectStatus(lastProjectData, viewState.selectedRequestId);
        } else {
            // No more requests - go back to Overview mode
            selectOverview();
            log(`Closed request: ${requestName}`, 'info');
            return;  // selectOverview already renders tabs
        }
    }

    // Re-render tabs
    renderRequestTabs();

    log(`Closed request: ${requestName}`, 'info');
}

function loadRequestContent(sessionId) {
    const request = requestsData.requests[sessionId];
    if (!request) return;

    // Load content from the iteration we're currently viewing
    const iteration = request.viewingIteration || request.currentIteration;
    const iterData = request.iterations[iteration];

    if (!iterData) {
        // No data for this iteration - clear panels
        REQUEST_PANEL_PHASES.forEach(phase => {
            resetPanelContent(phase);
        });
        return;
    }

    // Load content into each phase panel (except status, everything, and analysis)
    // Note: analysis data is stored separately but displayed in the combined panel
    ['generation', 'qa', 'arbiter', 'gransabio'].forEach(phase => {
        const contentEl = document.getElementById(`content-${phase}`);
        if (contentEl) {
            contentEl.textContent = iterData.content[phase] || '';
            contentEl.scrollTop = contentEl.scrollHeight;
        }

        // Update stats display
        const phaseStats = iterData.stats[phase] || { chunks: 0, bytes: 0 };
        const chunksEl = document.getElementById(`chunks-${phase}`);
        const bytesEl = document.getElementById(`bytes-${phase}`);
        if (chunksEl) chunksEl.textContent = phaseStats.chunks;
        if (bytesEl) bytesEl.textContent = phaseStats.bytes;
    });

    // Render combined Analysis panel (Auto-QA + preflight + consensus)
    renderFilteredAnalysisContent(request);
    updateAnalysisStats(iterData);

    // Update history mode visual indicators
    updateHistoryModeIndicators(request);
}

function updateRequestInfoBar() {
    const nameEl = document.getElementById('currentRequestName');
    const sessionEl = document.getElementById('currentSessionId');

    // Overview mode
    if (isOverviewMode()) {
        const activeCount = Object.values(requestsData.requests)
            .filter(r => isRequestActiveStatus(r.status)).length;
        const totalCount = requestsData.orderedRequestIds.length;

        if (nameEl) nameEl.textContent = 'All Sessions';
        if (sessionEl) {
            sessionEl.textContent = `${activeCount} active / ${totalCount} total`;
            sessionEl.title = 'Overview mode - select a tab to view specific session';
        }
        return;
    }

    // Session mode
    const request = requestsData.requests[viewState.selectedRequestId];

    if (request) {
        if (nameEl) nameEl.textContent = request.requestName;
        if (sessionEl) {
            sessionEl.textContent = request.sessionId.slice(0, 12) + '...';
            sessionEl.title = request.sessionId;
        }
    } else {
        if (nameEl) nameEl.textContent = '-';
        if (sessionEl) {
            sessionEl.textContent = '-';
            sessionEl.title = '';
        }
    }
}

// ============================================
// OVERFLOW MENU
// ============================================

function renderOverflowMenu(overflowRequests) {
    const menu = document.getElementById('tabsOverflowMenu');
    if (!menu) return;

    menu.innerHTML = '';

    overflowRequests.forEach(sessionId => {
        const request = requestsData.requests[sessionId];
        const item = document.createElement('div');
        item.className = 'tabs-overflow-item';
        item.title = `${request.requestName}\nSession: ${sessionId}`;

        const tabState = getRequestTabState(request);

        item.innerHTML = `
            <span class="overflow-status">[${tabState.icon}]</span>
            <span class="overflow-name">${escapeHtml(request.requestName)}</span>
        `;

        item.onclick = () => {
            selectRequest(sessionId);
            toggleOverflowMenu();
        };

        menu.appendChild(item);
    });
}

function toggleOverflowMenu() {
    const menu = document.getElementById('tabsOverflowMenu');
    const overflowBtn = document.querySelector('.tabs-overflow-btn');

    if (!menu) return;

    const isHidden = menu.classList.contains('hidden');

    if (isHidden && overflowBtn) {
        // Position menu below the overflow button
        const btnRect = overflowBtn.getBoundingClientRect();
        menu.style.top = `${btnRect.bottom + 5}px`;
        menu.style.left = `${btnRect.left}px`;

        // Ensure menu doesn't go off-screen to the right
        const menuWidth = 200; // min-width from CSS
        if (btnRect.left + menuWidth > window.innerWidth) {
            menu.style.left = `${window.innerWidth - menuWidth - 10}px`;
        }
    }

    menu.classList.toggle('hidden');
}

// ============================================
// PROJECT STATUS DASHBOARD
// ============================================

function showDashboard() {
    const dashboard = document.getElementById('statusDashboard');
    if (dashboard) dashboard.classList.remove('hidden');
}

function hideDashboard() {
    const dashboard = document.getElementById('statusDashboard');
    if (dashboard) dashboard.classList.add('hidden');
}

function renderProjectStatus(projectData, filterSessionId = undefined) {
    if (!projectData) return;

    // Cache for re-rendering when switching tabs
    lastProjectData = projectData;

    showDashboard();

    // Determine filter: use provided value, or fall back to current viewState
    if (filterSessionId === undefined) {
        filterSessionId = viewState.selectedRequestId;
    }

    const isOverview = (filterSessionId === null);

    // Update dashboard title based on mode
    const dashboardTitle = document.querySelector('.dashboard-title');
    if (dashboardTitle) {
        if (isOverview) {
            dashboardTitle.textContent = '// All Sessions Overview';
            dashboardTitle.className = 'dashboard-title';
        } else {
            const request = requestsData.requests[filterSessionId];
            const name = request?.requestName || 'Session';
            dashboardTitle.textContent = `// ${name} Status`;
            dashboardTitle.className = 'dashboard-title';
        }
    }

    // Update project ID
    const projectIdEl = document.getElementById('dashboardProjectId');
    if (projectIdEl) {
        projectIdEl.textContent = projectData.project_id || '-';
    }

    // Update project status badge
    const statusBadgeEl = document.getElementById('dashboardProjectStatus');
    if (statusBadgeEl) {
        const status = (projectData.status || 'idle').toLowerCase();
        statusBadgeEl.textContent = status.toUpperCase();
        statusBadgeEl.className = `project-status-badge ${status}`;
    }

    // Render sessions
    const sessionsContainer = document.getElementById('sessionsContainer');
    if (!sessionsContainer) return;

    let sessions = projectData.sessions || [];

    // Filter sessions if not in overview mode
    if (!isOverview && filterSessionId) {
        sessions = sessions.filter(s => s.session_id === filterSessionId);
    }

    if (sessions.length === 0) {
        const message = isOverview
            ? 'No active sessions'
            : 'Session data not available yet';
        sessionsContainer.innerHTML = `<div class="no-sessions">${message}</div>`;
        return;
    }

    sessionsContainer.innerHTML = '';
    sessions.forEach(session => {
        const card = renderSessionCard(session);
        sessionsContainer.appendChild(card);
    });

    // Bind click listeners to QA filter pills
    bindQAFilterListeners();
}

function renderSessionCard(session) {
    const card = document.createElement('div');
    card.className = 'session-card';
    card.id = `session-card-${session.session_id}`;

    const sessionIdShort = (session.session_id || '').slice(0, 12);
    const status = session.status || 'unknown';
    const statusClass = status.toLowerCase().replace(/\s+/g, '_');

    card.innerHTML = `
        <div class="session-card-header">
            <div class="session-id-label">SESSION: <span>${sessionIdShort}</span></div>
            <div class="session-status-badge ${statusClass}">${status}</div>
        </div>
        <div class="session-card-body">
            ${renderPhaseTracker(session)}
            ${renderIterationProgress(session)}
            ${renderGenerationInfo(session)}
            ${renderQASection(session)}
            ${renderConsensusSection(session)}
            ${renderGranSabioSection(session)}
        </div>
    `;

    return card;
}

function getTrackerStage(session) {
    const status = (session.status || '').toLowerCase();
    const phase = (session.phase || '').toLowerCase();

    if (isTerminalStatus(status)) {
        return status === 'completed' ? 'completed' : 'terminal';
    }
    if (phase === 'generating' || status === 'generating') return 'generating';
    if (phase === 'qa_evaluation' || status === 'qa_evaluation') return 'qa_evaluation';
    if (phase === 'consensus' || status === 'consensus') return 'consensus';
    if (GRAN_SABIO_PHASES.has(phase) || status === 'gran_sabio_review') return 'gran_sabio';
    if (phase === 'initializing' || status === 'initializing' || status === 'running') return 'initializing';
    return phase || status || 'initializing';
}

function renderPhaseTracker(session) {
    const stages = [
        ['initializing', 'INIT'],
        ['generating', 'GEN'],
        ['qa_evaluation', 'QA'],
        ['consensus', 'CONS'],
        ['gran_sabio', 'GS'],
        ['completed', 'DONE'],
    ];
    const currentStage = getTrackerStage(session);
    const knownStage = stages.some(([stage]) => stage === currentStage);
    const terminalFailed = currentStage === 'terminal';

    let html = '<div class="phase-tracker">';
    stages.forEach(([phase, label]) => {
        let stateClass = 'pending';
        if (terminalFailed) {
            stateClass = 'terminal';
        } else if (currentStage === 'completed') {
            stateClass = 'completed';
        } else if (phase === currentStage) {
            stateClass = 'active';
        }

        html += `<span class="phase-pill ${stateClass}">${label}</span>`;
    });
    if (!knownStage && !terminalFailed) {
        html += `<span class="phase-pill active">${escapeHtml(currentStage.toUpperCase())}</span>`;
    }
    html += '</div>';
    return html;
}

function renderIterationProgress(session) {
    const current = session.iteration || 1;
    const max = session.max_iterations || 5;
    const percent = Math.min((current / max) * 100, 100);

    return `
        <div class="progress-row">
            <span class="progress-label">Iteration</span>
            <div class="progress-bar-container">
                <div class="progress-bar-fill iteration" style="width: ${percent}%"></div>
            </div>
            <span class="progress-value">${current}/${max}</span>
        </div>
    `;
}

function renderGenerationInfo(session) {
    const gen = session.generation || {};
    if (!gen.model && !gen.word_count) return '';

    return `
        <div class="info-section">
            <div class="info-section-title">Generation</div>
            <div class="info-grid">
                ${gen.model ? `<div class="info-item"><span class="info-item-label">Model:</span><span class="info-item-value">${gen.model}</span></div>` : ''}
                ${gen.word_count !== undefined ? `<div class="info-item"><span class="info-item-label">Words:</span><span class="info-item-value">${gen.word_count.toLocaleString()}</span></div>` : ''}
                ${gen.content_length !== undefined ? `<div class="info-item"><span class="info-item-label">Chars:</span><span class="info-item-value">${gen.content_length.toLocaleString()}</span></div>` : ''}
            </div>
        </div>
    `;
}

function renderQASection(session) {
    const qa = session.qa || {};
    if (!qa.models && !qa.layers) return '';

    const models = qa.models || [];
    const layers = qa.layers || [];
    const currentModel = qa.current_model || '';
    const currentLayer = qa.current_layer || '';
    const progress = qa.progress || {};

    let html = '<div class="info-section">';
    html += '<div class="info-section-title">QA Evaluation</div>';

    // Models row
    if (models.length > 0) {
        html += '<div style="margin-bottom: 8px;"><span class="info-item-label" style="font-size: 9px;">Models: <span style="color: var(--gray); font-style: italic;">(click to filter)</span></span></div>';
        html += '<div class="model-pills">';
        models.forEach(model => {
            const isActive = model === currentModel;
            const isCompleted = models.indexOf(model) < models.indexOf(currentModel);
            const isFilterSelected = qaFilterState.model === model;
            let stateClass = isActive ? 'active' : (isCompleted ? 'completed' : '');
            if (isFilterSelected) stateClass += ' filter-selected';
            let icon = isActive ? '*' : (isCompleted ? '+' : 'o');
            const escaped = escapeHtmlAttr(model);
            html += `<span class="model-pill ${stateClass}" data-filter-type="model" data-filter-value="${escaped}"><span class="status-icon">${icon}</span>${escapeHtml(model)}</span>`;
        });
        html += '</div>';
    }

    // Layers row
    if (layers.length > 0) {
        html += '<div style="margin: 8px 0;"><span class="info-item-label" style="font-size: 9px;">Layers: <span style="color: var(--gray); font-style: italic;">(click to filter)</span></span></div>';
        html += '<div class="layer-pills">';
        layers.forEach(layer => {
            const isActive = layer === currentLayer;
            const isCompleted = layers.indexOf(layer) < layers.indexOf(currentLayer);
            const isFilterSelected = qaFilterState.layer === layer;
            let stateClass = isActive ? 'active' : (isCompleted ? 'completed' : '');
            if (isFilterSelected) stateClass += ' filter-selected';
            let icon = isActive ? '*' : (isCompleted ? '+' : 'o');
            const escaped = escapeHtmlAttr(layer);
            html += `<span class="layer-pill ${stateClass}" data-filter-type="layer" data-filter-value="${escaped}"><span class="status-icon">${icon}</span>${escapeHtml(layer)}</span>`;
        });
        html += '</div>';
    }

    // Progress bar
    if (progress.total) {
        const completed = progress.completed || 0;
        const total = progress.total || 1;
        const percent = Math.min((completed / total) * 100, 100);

        html += `
            <div class="progress-row" style="margin-top: 10px;">
                <span class="progress-label">Progress</span>
                <div class="progress-bar-container">
                    <div class="progress-bar-fill qa" style="width: ${percent}%"></div>
                </div>
                <span class="progress-value">${completed}/${total}</span>
            </div>
        `;
    }

    html += '</div>';
    return html;
}

function renderConsensusSection(session) {
    const consensus = session.consensus || {};
    if (consensus.last_score === undefined && consensus.min_required === undefined) return '';

    const score = consensus.last_score || 0;
    const minRequired = consensus.min_required || 8.0;
    const approved = consensus.approved;
    const isAbove = score >= minRequired;

    // Calculate positions (0-10 scale)
    const scorePercent = Math.min((score / 10) * 100, 100);
    const thresholdPercent = (minRequired / 10) * 100;

    let statusText = approved === true ? 'APPROVED' : (approved === false ? 'BELOW THRESHOLD' : 'PENDING');
    if (approved === undefined && score > 0) {
        statusText = isAbove ? 'ABOVE MIN' : 'BELOW MIN';
    }

    return `
        <div class="info-section">
            <div class="info-section-title">Consensus</div>
            <div class="info-grid" style="margin-bottom: 8px;">
                <div class="info-item">
                    <span class="info-item-label">Score:</span>
                    <span class="info-item-value">${score.toFixed(1)}</span>
                </div>
                <div class="info-item">
                    <span class="info-item-label">Min:</span>
                    <span class="info-item-value">${minRequired.toFixed(1)}</span>
                </div>
                <div class="info-item">
                    <span class="info-item-label">Status:</span>
                    <span class="info-item-value" style="color: ${isAbove ? 'var(--green)' : 'var(--amber)'}">${statusText}</span>
                </div>
            </div>
            <div class="consensus-gauge">
                <div class="gauge-bar">
                    <div class="gauge-fill ${isAbove ? 'above' : 'below'}" style="width: ${scorePercent}%"></div>
                    <div class="gauge-threshold" style="left: ${thresholdPercent}%"></div>
                    ${score > 0 ? `<div class="gauge-current" style="left: ${scorePercent}%">${score.toFixed(1)}</div>` : ''}
                </div>
                <div class="gauge-labels">
                    <span>0</span>
                    <span>5</span>
                    <span>10</span>
                </div>
            </div>
        </div>
    `;
}

function renderGranSabioSection(session) {
    const gs = session.gran_sabio || {};
    if (!gs.model && gs.escalation_count === undefined) return '';

    const isActive = gs.active || false;
    const model = gs.model || 'claude-opus-4';
    const escalations = gs.escalation_count || 0;
    const maxEscalations = 15; // Default from config

    return `
        <div class="info-section">
            <div class="info-section-title">Gran Sabio</div>
            <div class="gran-sabio-status">
                <div class="gran-sabio-indicator">
                    <span class="gran-sabio-dot ${isActive ? 'active' : 'standby'}"></span>
                    <span>${isActive ? 'ACTIVE' : 'STANDBY'}</span>
                </div>
                <div class="gran-sabio-model">Model: <span>${model}</span></div>
                <div class="gran-sabio-escalations">Escalations: <span>${escalations}/${maxEscalations}</span></div>
            </div>
        </div>
    `;
}

// ============================================
// URL MANAGEMENT
// ============================================

function getProjectFromUrl() {
    const params = new URLSearchParams(window.location.search);
    return params.get('project');
}

function updateUrl(projectId) {
    const url = new URL(window.location);
    if (projectId) {
        url.searchParams.set('project', projectId);
    } else {
        url.searchParams.delete('project');
    }
    window.history.pushState({}, '', url);
}

// ============================================
// RECENT PROJECTS (ACTIVE CONNECTIONS)
// ============================================

async function loadRecentProjects() {
    const listEl = document.getElementById('recentList');
    const countEl = document.getElementById('recentCount');
    const filterEl = document.getElementById('recentStatusFilter');
    const filter = filterEl?.value || 'running';
    const includeReserved = filter === 'all' || filter === 'idle';

    try {
        const params = new URLSearchParams({
            status: filter,
            include_reserved: includeReserved ? 'true' : 'false',
        });
        const response = await fetch(`/monitor/active?${params.toString()}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();

        const projects = data.projects || [];

        countEl.textContent = `${projects.length} ${filter} projects`;

        if (projects.length === 0) {
            listEl.innerHTML = `<div class="empty-message">No ${escapeHtml(filter)} projects</div>`;
            return;
        }

        listEl.innerHTML = '';

        // Render projects (unified: project_id = session_id when not explicitly provided)
        projects.forEach(project => {
            const item = document.createElement('div');
            item.className = 'recent-item';
            if (project.status !== 'running') {
                item.classList.add('no-project');
            }

            const info = document.createElement('div');
            info.className = 'recent-info';

            const name = document.createElement('div');
            name.className = 'recent-name';
            name.textContent = project.project_id || '';

            const meta = document.createElement('div');
            meta.className = 'recent-meta';
            meta.textContent = `Sessions: ${project.session_count} | Active: ${project.active_sessions} | Status: ${project.status}`;

            const projectId = document.createElement('div');
            projectId.className = 'recent-project-id';
            projectId.textContent = project.project_id || '';

            info.appendChild(name);
            info.appendChild(meta);
            info.appendChild(projectId);

            const button = document.createElement('button');
            button.className = 'btn btn-connect btn-small';
            button.textContent = 'Connect';
            button.addEventListener('click', () => connectToProject(project.project_id || ''));

            item.appendChild(info);
            item.appendChild(button);
            listEl.appendChild(item);
        });

        log(`Loaded ${projects.length} ${filter} projects`, 'success');

    } catch (error) {
        log(`Failed to load active connections: ${error.message}`, 'error');
        listEl.innerHTML = `<div class="empty-message" style="color: var(--red);">Error: ${escapeHtml(error.message)}</div>`;
    }
}

// ============================================
// CONNECTION
// ============================================

function connectToProject(projectId) {
    document.getElementById('projectIdInput').value = projectId;
    connect();
}

function connect() {
    const inputValue = document.getElementById('projectIdInput').value.trim();

    if (!inputValue) {
        log('Please enter a project ID', 'error');
        return;
    }

    // Disconnect existing connection
    if (eventSource) {
        disconnect();
    }

    clearAllContent();
    setConnectionStatus('connecting', 'CONNECTING...');

    // Project mode - unified stream (project_id = session_id when not explicitly provided)
    currentProjectId = inputValue;

    const activePhases = getActivePhases();
    // Check if all UI phases are active (none hard-muted) to use 'all' param
    const allPhasesActive = HARD_SWITCHABLE_PHASES.every(phase => panelStates[phase] !== 'hard');
    const phasesParam = allPhasesActive ? 'all' : activePhases.join(',');
    const streamUrl = `${STREAM_BASE}/stream/project/${encodeURIComponent(currentProjectId)}?phases=${encodeURIComponent(phasesParam)}`;

    log(`Connecting to project stream: ${streamUrl}`, 'info');
    updateUrl(inputValue);

    try {
        eventSource = new EventSource(streamUrl);

        eventSource.onopen = function() {
            log('EventSource connection opened', 'success');
            setConnectionStatus('connected', 'ONLINE');
            PHASES.forEach(phase => setBadge(phase, 'active'));
        };

        eventSource.onmessage = function(event) {
            handleMessage(event.data);
        };

        eventSource.onerror = function(error) {
            handleConnectionError(error);
        };

    } catch (error) {
        log(`Failed to connect: ${error.message}`, 'error');
        setConnectionStatus('disconnected', 'ERROR');
    }
}

function disconnect() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
        log('Disconnected from stream', 'info');
    }

    currentProjectId = null;
    setConnectionStatus('disconnected', 'OFFLINE');
    PHASES.forEach(phase => setBadge(phase, 'idle'));
    updateUrl(null);
    resetQAFilters();
}

// ============================================
// QA FILTER MANAGEMENT
// ============================================

/**
 * Bind click listeners to QA filter pills in the dashboard.
 * Call this after renderProjectStatus() to make pills interactive.
 */
function bindQAFilterListeners() {
    const dashboard = document.getElementById('statusDashboard');
    if (!dashboard) return;

    // Find all model and layer pills with filter data attributes
    dashboard.querySelectorAll('.model-pill[data-filter-type], .layer-pill[data-filter-type]').forEach(pill => {
        pill.onclick = (e) => {
            e.stopPropagation();
            handleQAFilterClick(pill);
        };
    });
}

/**
 * Handle click on a QA filter pill in the dashboard
 * @param {HTMLElement} pill - The clicked pill element
 */
function handleQAFilterClick(pill) {
    const filterType = pill.dataset.filterType;
    const filterValue = pill.dataset.filterValue;

    // Toggle: if already selected, deselect
    if (filterType === 'layer') {
        qaFilterState.layer = (qaFilterState.layer === filterValue) ? null : filterValue;
    } else if (filterType === 'model') {
        qaFilterState.model = (qaFilterState.model === filterValue) ? null : filterValue;
    }

    // Re-render dashboard to update pill styles, then re-bind listeners
    if (lastProjectData) {
        renderProjectStatus(lastProjectData, viewState.selectedRequestId);
    }

    // Re-render QA content with filter applied
    const request = requestsData.requests[viewState.selectedRequestId];
    if (request) {
        renderFilteredQAContent(request);
    }
}

/**
 * Render QA content based on current filter state
 * @param {Object} request - The request object
 */
function renderFilteredQAContent(request) {
    const contentEl = document.getElementById('content-qa');
    if (!contentEl) return;

    const iter = request?.iterations?.[request.viewingIteration];
    if (!iter) {
        contentEl.textContent = '';
        return;
    }

    let content = '';

    if (qaFilterState.layer && qaFilterState.model) {
        // Filter by both layer AND model
        content = iter.qaByLayerModel?.[qaFilterState.layer]?.[qaFilterState.model] || '';
    } else if (qaFilterState.layer) {
        // Filter by layer only
        content = iter.qaByLayer?.[qaFilterState.layer] || '';
    } else if (qaFilterState.model) {
        // Filter by model only
        content = iter.qaByModel?.[qaFilterState.model] || '';
    } else {
        // No filter - show all
        content = iter.content?.qa || '';
    }

    contentEl.textContent = content;
    contentEl.scrollTop = contentEl.scrollHeight;
}

/**
 * Reset QA filters to default (show all)
 */
function resetQAFilters() {
    qaFilterState.layer = null;
    qaFilterState.model = null;
}

// ============================================
// ANALYSIS FILTER (AUTO-QA/PREFLIGHT/CONSENSUS)
// ============================================

/**
 * Render Analysis content based on current filter state
     * Combines Auto-QA, preflight, and consensus content with optional filtering
 * @param {Object} request - The request object
 */
function renderFilteredAnalysisContent(request) {
    const contentEl = document.getElementById('content-analysis');
    if (!contentEl) return;

    const iter = request?.iterations?.[request.viewingIteration];
    if (!iter) {
        contentEl.textContent = '';
        return;
    }

    let content = '';

    if (ANALYSIS_CONTENT_KEYS.includes(analysisFilterState)) {
        content = iter.content?.[analysisFilterState] || '';
    } else {
        const sections = [
            ['AUTO-QA', iter.content?.auto_qa || ''],
            ['PREFLIGHT', iter.content?.preflight || ''],
            ['CONSENSUS', iter.content?.consensus || ''],
        ].filter(([, sectionContent]) => Boolean(sectionContent));

        content = sections
            .map(([label, sectionContent]) => `--- ${label} ---\n\n${sectionContent}`)
            .join('\n\n');
    }

    contentEl.textContent = content;
    contentEl.scrollTop = contentEl.scrollHeight;
}

/**
 * Handle click on an analysis filter pill
 * @param {HTMLElement} pill - The clicked pill element
 */
function handleAnalysisFilterClick(pill) {
    const filterValue = pill.dataset.filter;

    // Toggle: if already selected, deselect (show both)
    analysisFilterState = (analysisFilterState === filterValue) ? null : filterValue;

    // Update pill visual state
    document.querySelectorAll('.analysis-pill').forEach(p => {
        p.classList.toggle('active', p.dataset.filter === analysisFilterState);
    });

    // Re-render content with filter applied
    const request = requestsData.requests[viewState.selectedRequestId];
    if (request) {
        renderFilteredAnalysisContent(request);
    }
}

/**
 * Bind click listeners to analysis filter pills
 */
function bindAnalysisFilterListeners() {
    document.querySelectorAll('.analysis-pill').forEach(pill => {
        pill.onclick = (e) => {
            e.stopPropagation();
            handleAnalysisFilterClick(pill);
        };
    });
}

/**
 * Reset Analysis filter to default (show both)
 */
function resetAnalysisFilter() {
    analysisFilterState = null;
    document.querySelectorAll('.analysis-pill').forEach(p => p.classList.remove('active'));
}

/**
 * Escape HTML for safe display
 * @param {string} text - Text to escape
 * @returns {string} Escaped text
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Escape text for use in HTML attributes
 * @param {string} text - Text to escape
 * @returns {string} Escaped text safe for attributes
 */
function escapeHtmlAttr(text) {
    return text.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// ============================================
// MESSAGE HANDLING (PROJECT MODE)
// ============================================

function summarizeStructuredPayload(payload) {
    if (!payload || typeof payload !== 'object') {
        return '';
    }
    if (payload.truncated) {
        const keys = Array.isArray(payload.preview_keys) ? payload.preview_keys.join(',') : '';
        return `truncated=true size_bytes=${payload.size_bytes ?? '?'} preview_keys=${keys}`;
    }

    const interestingKeys = [
        'turn',
        'model',
        'api_surface',
        'streaming',
        'tool',
        'tool_name',
        'tool_calls',
        'remaining_tool_calls',
        'arguments_chars',
        'assistant_text_chars',
        'finish_reason',
        'score',
        'hard_failed',
        'decision',
        'status',
        'reason',
        'error',
        'claims_count',
        'grounded_claims',
        'unsupported_claims',
    ];
    const parts = [];
    interestingKeys.forEach(key => {
        if (payload[key] !== undefined && payload[key] !== null) {
            parts.push(`${key}=${String(payload[key])}`);
        }
    });

    if (parts.length > 0) {
        return parts.join(' ');
    }

    return `keys=${Object.keys(payload).slice(0, 6).join(',')}`;
}

function parseGroundingPayload(data) {
    if (!data.content) {
        return data.data || {};
    }
    try {
        return JSON.parse(data.content);
    } catch (error) {
        return { preview: String(data.content).slice(0, 240) };
    }
}

function formatStructuredEventLine(data) {
    const eventType = data.type || 'unknown';
    const payloadSummary = summarizeStructuredPayload(data.data);

    if (eventType === 'assistant_delta') {
        const delta = data.data && typeof data.data.content === 'string'
            ? data.data.content
            : '';
        return `[TOOL_STREAM] assistant_delta${payloadSummary ? ' ' + payloadSummary : ''}${delta ? ` text=${JSON.stringify(delta)}` : ''}\n`;
    }

    if (eventType === 'tool_call_delta') {
        return '';
    }

    if (eventType === 'retry_start' || eventType === 'retry') {
        const attempt = data.attempt !== undefined ? `attempt ${data.attempt}/${data.max_attempts || '?'}` : 'attempt ?';
        const provider = data.provider ? ` provider=${data.provider}` : '';
        const retryIn = data.retry_in_seconds !== undefined ? ` retry_in=${data.retry_in_seconds}s` : '';
        const reason = data.reason ? ` reason=${data.reason}` : '';
        return `[RETRY] ${attempt}${provider}${retryIn}${reason}\n`;
    }

    if (eventType === 'error') {
        const label = data.phase === 'generation' ? 'RETRY_ERROR' : 'ERROR';
        const reason = data.reason || data.message || payloadSummary || 'unknown';
        return `[${label}] phase=${data.phase || 'status'} ${reason}\n`;
    }

    if (eventType === 'project_end') {
        return `[PROJECT_END] status=${data.status || 'unknown'} phase=${data.phase || 'unknown'}\n`;
    }

    if (eventType === 'project_cancelled') {
        return `[PROJECT_CANCELLED] project=${data.project_id || currentProjectId || '?'}\n`;
    }

    if (eventType.startsWith('grounding_')) {
        const groundingPayload = parseGroundingPayload(data);
        const groundingSummary = summarizeStructuredPayload(groundingPayload);
        return `[GROUNDING] ${eventType}${groundingSummary ? ' ' + groundingSummary : ''}\n`;
    }

    const group = eventType.startsWith('tool_') || eventType === 'force_finalize'
        || eventType === 'validate_draft_oversize'
        || eventType === 'context_overflow_midloop'
        ? 'TOOL'
        : 'EVENT';
    return `[${group}] ${eventType}${payloadSummary ? ' ' + payloadSummary : ''}\n`;
}

function appendStructuredEventForRequest(data, line, displayPhase) {
    const request = processRequestFromEvent(data);

    if (request && displayPhase !== 'status') {
        const iteration = request.currentIteration || 1;
        initializeIteration(request, iteration);
        const iterData = request.iterations[iteration];
        const contentKey = normalizePhaseName(data.phase, displayPhase);

        if (iterData.content.hasOwnProperty(contentKey)) {
            iterData.content[contentKey] += line;
            iterData.stats[contentKey].chunks++;
            iterData.stats[contentKey].bytes += line.length;
        }
    }

    if (request) {
        if (isOverviewMode()) {
            showReceiving(displayPhase);
            return;
        }
        if (request.sessionId !== viewState.selectedRequestId) {
            showReceiving(displayPhase);
            return;
        }
        if (request.viewingIteration !== request.currentIteration) {
            showLiveActivityIndicator(request.currentIteration);
            return;
        }
    }

    if (displayPhase === 'analysis' && request) {
        renderFilteredAnalysisContent(request);
        updateAnalysisStats(request.iterations[request.currentIteration]);
        showReceiving('analysis');
    } else if (displayPhase === 'qa' && request && (qaFilterState.layer || qaFilterState.model)) {
        renderFilteredQAContent(request);
        showReceiving('qa');
    } else {
        appendContent(displayPhase, line);
    }
}

function markRequestTerminal(sessionId, status) {
    if (!sessionId) return null;
    const request = requestsData.requests[sessionId];
    if (!request) return null;

    request.status = status || 'completed';
    request.autoFollowIteration = false;
    const iterData = request.iterations[request.currentIteration];
    if (iterData) {
        iterData.timestampEnd = Date.now();
        iterData.status = request.status === 'completed' ? 'approved' : 'rejected';
    }
    return request;
}

function handleSessionEnd(data) {
    const request = processRequestFromEvent(data);
    const status = data.status || 'completed';
    markRequestTerminal(data.session_id, status);

    log(`Session ${data.session_id?.slice(0, 8) || '?'} ended with ${status}`, 'warn');
    appendContent('status', `[SESSION_END] session=${data.session_id || '?'} status=${status}\n`);

    if (request && request.sessionId === viewState.selectedRequestId) {
        renderIterationTimeline();
    }
    renderRequestTabs();
}

function handleProjectTerminalEvent(data) {
    const status = data.status || (data.type === 'project_cancelled' ? 'cancelled' : undefined);
    if (status === 'cancelled') {
        Object.values(requestsData.requests).forEach(request => {
            if (!isTerminalStatus(request.status)) {
                markRequestTerminal(request.sessionId, 'cancelled');
            }
        });
        setConnectionStatus('disconnected', 'CANCELLED');
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
        renderRequestTabs();
        renderIterationTimeline();
    }
}

function handleStructuredEvent(data) {
    const eventType = data.type || 'unknown';
    let displayPhase = getDisplayPhase(data.phase);

    if (eventType === 'project_end' || eventType === 'project_cancelled') {
        displayPhase = 'status';
    } else if (eventType.startsWith('grounding_')) {
        displayPhase = 'qa';
    } else if (eventType === 'error' && data.phase !== 'generation') {
        displayPhase = 'status';
    }

    if (eventType === 'project_end' || eventType === 'project_cancelled') {
        handleProjectTerminalEvent(data);
    }

    const line = formatStructuredEventLine(data);
    if (!line) {
        return;
    }
    appendStructuredEventForRequest(data, line, displayPhase);
}

function handleMessage(rawData) {
    // Always append raw data to "everything" panel (unfiltered)
    appendContent('everything', rawData + '\n');

    try {
        const data = JSON.parse(rawData);
        const eventType = data.type || 'unknown';
        const phase = data.phase || 'status';

        // Log event type (except chunks to avoid spam)
        if (eventType !== 'chunk') {
            log(`Event: ${eventType} (phase: ${phase})`, 'event');
        }

        switch (eventType) {
            case 'connected':
                log(`Connected to project ${data.project_id}`, 'success');
                log(`Subscribed phases: ${data.subscribed_phases?.join(', ') || 'all'}`, 'info');
                break;

            case 'status_snapshot':
                handleStatusSnapshot(data);
                break;

            case 'chunk':
                handleChunk(data);
                break;

            case 'status_change':
                handleStatusChange(data);
                break;

            case 'session_end':
                handleSessionEnd(data);
                break;

            case 'stream_end':
                log(`Stream ended: ${data.reason}`, 'warn');
                appendContent('status', `[STREAM_END] Reason: ${data.reason}\n`);
                setConnectionStatus('disconnected', 'STREAM ENDED');
                if (eventSource) {
                    eventSource.close();
                    eventSource = null;
                }
                break;

            case 'edit_start':
                handleSmartEditStart(data);
                break;

            case 'edit_complete':
                handleSmartEditComplete(data);
                break;

            case 'edit_error':
                handleSmartEditError(data);
                break;

            default:
                if (KNOWN_STRUCTURED_EVENTS.has(eventType) || eventType.startsWith('grounding_')) {
                    handleStructuredEvent(data);
                } else {
                    // Unknown event type - show in status panel
                    log(`Unknown event type: ${eventType} (phase: ${phase})`, 'warn');
                    appendContent('status', `[UNKNOWN] type=${eventType} phase=${phase} ${JSON.stringify(data, null, 2)}\n`);
                }
        }

    } catch (parseError) {
        // Non-JSON data - show raw
        log(`Parse error: ${parseError.message}`, 'warn');
        appendContent('status', rawData + '\n');
    }
}

function handleChunk(data) {
    // Process and track by request
    const request = processRequestFromEvent(data);

    const phase = normalizePhaseName(data.phase, 'generation');
    const displayPhase = getDisplayPhase(data.phase);
    const content = data.content || '';

    if (!content) return;

    // Store content in request's iteration-specific data structure
    if (request && phase !== 'status') {
        const iteration = request.currentIteration;

        // Ensure iteration exists
        initializeIteration(request, iteration);

        const iterData = request.iterations[iteration];

        // Store in the correct iteration
        if (iterData.content.hasOwnProperty(phase)) {
            iterData.content[phase] += content;
            iterData.stats[phase].chunks++;
            iterData.stats[phase].bytes += content.length;
        }

        // Index QA content by layer and model for filtering
        if (phase === 'qa' && data.qa_layer && data.qa_model) {
            const layer = data.qa_layer;
            const model = data.qa_model;

            // Register layer if new
            if (!iterData.qaLayers.includes(layer)) {
                iterData.qaLayers.push(layer);
            }

            // Register model if new
            if (!iterData.qaModels.includes(model)) {
                iterData.qaModels.push(model);
            }

            // Accumulate by layer
            iterData.qaByLayer[layer] = (iterData.qaByLayer[layer] || '') + content;

            // Accumulate by model
            iterData.qaByModel[model] = (iterData.qaByModel[model] || '') + content;

            // Accumulate by layer+model
            if (!iterData.qaByLayerModel[layer]) {
                iterData.qaByLayerModel[layer] = {};
            }
            iterData.qaByLayerModel[layer][model] =
                (iterData.qaByLayerModel[layer][model] || '') + content;
        }
    }

    // In Overview mode: don't display in panels, but show generic activity
    if (isOverviewMode()) {
        if (request) {
            // Show activity indicator (generic receiving badge)
            showReceiving(displayPhase);
        }
        return;
    }

    // Session mode: display only if this is the selected request
    if (request && request.sessionId === viewState.selectedRequestId) {
        // Only show content if viewing the current iteration
        if (request.viewingIteration === request.currentIteration) {
            if (phase === 'qa' && (qaFilterState.layer || qaFilterState.model)) {
                // QA with filter active - re-render filtered content
                renderFilteredQAContent(request);
            } else if (ANALYSIS_CONTENT_KEYS.includes(phase)) {
                // Analysis phases render to a combined panel with filter
                if (analysisFilterState === null || analysisFilterState === phase) {
                    renderFilteredAnalysisContent(request);
                }
                // Update stats for analysis panel
                const iterData = request.iterations[request.currentIteration];
                if (iterData) {
                    updateAnalysisStats(iterData);
                }
                // Flash the analysis badge
                showReceiving('analysis');
            } else {
                // Normal append behavior
                appendContent(displayPhase, content);
            }
        } else {
            // Viewing history - show activity indicator that live data is coming
            showLiveActivityIndicator(request.currentIteration);
        }
    } else if (!request) {
        // No session_id in data - still show (legacy behavior)
        if (ANALYSIS_CONTENT_KEYS.includes(phase)) {
            // For legacy mode, append to analysis panel with separator
            const analysisEl = document.getElementById('content-analysis');
            if (analysisEl) {
                if (analysisFilterState === null || analysisFilterState === phase) {
                    analysisEl.textContent += content;
                    analysisEl.scrollTop = analysisEl.scrollHeight;
                }
            }
        } else {
            appendContent(displayPhase, content);
        }
    }

    // Note: 'everything' panel receives all content via handleMessage
    // before this function is called, so it remains unfiltered
}

function getSnapshotTextEntry(entry) {
    if (!entry) return null;
    if (typeof entry === 'string') {
        return { text: entry, truncated: false, omitted_chars: 0 };
    }
    if (typeof entry === 'object') {
        return {
            text: entry.text || '',
            truncated: Boolean(entry.truncated),
            omitted_chars: entry.omitted_chars || 0,
        };
    }
    return null;
}

function upsertRequestFromSessionSnapshot(sessionData) {
    const sessionId = sessionData.session_id;
    if (!sessionId) return null;

    let request = requestsData.requests[sessionId];
    if (!request) {
        const iteration = sessionData.iteration || 1;
        request = {
            sessionId: sessionId,
            requestName: sessionData.request_name || 'unknown',
            status: sessionData.status || 'active',
            currentIteration: iteration,
            viewingIteration: iteration,
            autoFollowIteration: true,
            maxIterations: sessionData.max_iterations || 5,
            finalScore: null,
            iterations: {}
        };
        requestsData.requests[sessionId] = request;
        requestsData.orderedRequestIds.push(sessionId);
        initializeIteration(request, iteration);
    }

    const iteration = sessionData.iteration || request.currentIteration || 1;
    request.currentIteration = iteration;
    if (!request.viewingIteration || request.autoFollowIteration) {
        request.viewingIteration = iteration;
    }
    request.status = sessionData.status || request.status;
    request.requestName = sessionData.request_name || request.requestName;
    request.maxIterations = sessionData.max_iterations || request.maxIterations;
    initializeIteration(request, iteration);

    const iterData = request.iterations[iteration];
    const snapshot = sessionData.content_snapshot || {};
    Object.entries(snapshot).forEach(([serverPhase, entry]) => {
        const snapshotEntry = getSnapshotTextEntry(entry);
        if (!snapshotEntry || !snapshotEntry.text) return;

        const phase = normalizePhaseName(serverPhase, serverPhase);
        if (!iterData.content.hasOwnProperty(phase)) return;

        const prefix = snapshotEntry.truncated
            ? `[... ${(snapshotEntry.omitted_chars || 0).toLocaleString()} chars omitted before monitor connection ...]\n`
            : '';
        const text = `${prefix}${snapshotEntry.text}`;
        iterData.content[phase] = text;
        iterData.stats[phase].chunks = 1;
        iterData.stats[phase].bytes = text.length;
    });

    if (sessionData.consensus && sessionData.consensus.last_score !== null && sessionData.consensus.last_score !== undefined) {
        iterData.score = sessionData.consensus.last_score;
        iterData.approved = sessionData.consensus.approved || false;
    }
    if (isTerminalStatus(request.status)) {
        markRequestTerminal(sessionId, request.status);
    }
    return request;
}

function hydrateRequestsFromProjectSnapshot(project) {
    const sessions = project.sessions || [];
    if (sessions.length === 0) return;

    sessions.forEach(sessionData => upsertRequestFromSessionSnapshot(sessionData));

    if (viewState.autoFollow && viewState.selectedRequestId === null) {
        const activeSession = [...sessions].reverse().find(session => isRequestActiveStatus(session.status));
        const selected = activeSession || sessions[sessions.length - 1];
        if (selected?.session_id) {
            viewState.selectedRequestId = selected.session_id;
            REQUEST_PANEL_PHASES.forEach(phase => {
                resetPanelContent(phase, {
                    clearOverviewMode: true,
                    clearText: false,
                    resetBadge: false,
                    resetStats: false,
                });
            });
        }
    }

    renderRequestTabs();
    renderIterationTimeline();
    if (!isOverviewMode()) {
        loadRequestContent(viewState.selectedRequestId);
    }
}

function handleStatusSnapshot(data) {
    log('Received status snapshot', 'info');
    const project = data.project || {};
    const summary = project.summary || {};
    const totalSessions = summary.total_sessions ?? summary.total ?? 0;
    const activeSessions = summary.active_sessions ?? summary.active ?? 0;
    const completedSessions = summary.completed_sessions ?? summary.completed ?? 0;

    hydrateRequestsFromProjectSnapshot(project);

    // Render visual dashboard
    renderProjectStatus(project);

    // Also output to text panel for debugging
    const statusText = `[STATUS SNAPSHOT]
Project: ${project.project_id || '?'}
Status: ${project.status || '?'}
Sessions: ${totalSessions} total, ${activeSessions} active, ${completedSessions} completed
`;
    appendContent('status', statusText);

    // Show session details if available
    if (project.sessions && project.sessions.length > 0) {
        project.sessions.forEach(session => {
            appendContent('status', `  Session ${session.session_id?.slice(0, 8) || '?'}: ${session.status} (${session.phase || '?'})\n`);
        });
    }
}

function handleStatusChange(data) {
    const project = data.project || {};
    const sessions = project.sessions || [];
    const triggeredBy = data.trigger_session?.slice(0, 8) || '?';

    // Process each session from the project status
    sessions.forEach(sessionData => {
        const sessionId = sessionData.session_id;
        if (!sessionId) return;

        // Get or create request
        let request = requestsData.requests[sessionId];
        if (!request) {
            // Create request from status_change data
            request = {
                sessionId: sessionId,
                requestName: sessionData.request_name || 'unknown',
                status: 'active',
                currentIteration: 1,
                viewingIteration: 1,
                autoFollowIteration: true,
                maxIterations: sessionData.max_iterations || 5,
                finalScore: null,
                iterations: {}
            };
            requestsData.requests[sessionId] = request;
            requestsData.orderedRequestIds.push(sessionId);
            initializeIteration(request, 1);
        }

        // DETECT ITERATION CHANGE
        const newIteration = sessionData.iteration || 1;
        const oldIteration = request.currentIteration;

        if (newIteration !== oldIteration) {
            // Iteration changed!
            handleIterationChange(request, oldIteration, newIteration, sessionData);
        }

        // Update request state
        if (sessionData.status) {
            request.status = sessionData.status;
            if (isTerminalStatus(sessionData.status)) {
                markRequestTerminal(sessionId, sessionData.status);
            }
        }
        if (sessionData.max_iterations) {
            request.maxIterations = sessionData.max_iterations;
        }

        // Update score of current iteration if consensus data available
        if (sessionData.consensus && sessionData.consensus.last_score !== null) {
            const iterData = request.iterations[request.currentIteration];
            if (iterData) {
                iterData.score = sessionData.consensus.last_score;
                iterData.approved = sessionData.consensus.approved || false;
            }
        }

        // Update final score for completed sessions
        if (sessionData.status === 'completed' && sessionData.consensus && sessionData.consensus.last_score !== null && sessionData.consensus.last_score !== undefined) {
            request.finalScore = sessionData.consensus.last_score;
        }
    });

    // Re-render UI components
    renderRequestTabs();
    renderIterationTimeline();

    // Re-render visual dashboard with updated data
    renderProjectStatus(project);

    // Also output to text panel for debugging
    appendContent('status', `[STATUS_CHANGE] Triggered by session ${triggeredBy}\n`);

    if (project.summary) {
        const activeSessions = project.summary.active_sessions ?? project.summary.active ?? 0;
        const completedSessions = project.summary.completed_sessions ?? project.summary.completed ?? 0;
        appendContent('status', `  Active: ${activeSessions}, Completed: ${completedSessions}\n`);
    }
}

/**
 * Handle iteration change - finalize old iteration and prepare new one
 * @param {Object} request - The request object
 * @param {number} oldIteration - Previous iteration number
 * @param {number} newIteration - New iteration number
 * @param {Object} sessionData - Session data from status_change event
 */
function handleIterationChange(request, oldIteration, newIteration, sessionData) {
    log(`Iteration changed: ${oldIteration} -> ${newIteration}`, 'info');

    // 1. Finalize old iteration
    const oldIterData = request.iterations[oldIteration];
    if (oldIterData) {
        oldIterData.timestampEnd = Date.now();
        oldIterData.status = oldIterData.approved ? 'approved' : 'rejected';

        // Save score from session data if available
        if (sessionData.consensus && sessionData.consensus.last_score !== null) {
            oldIterData.score = sessionData.consensus.last_score;
        }
    }

    // 2. Create new iteration
    initializeIteration(request, newIteration);

    // 3. Update current iteration
    request.currentIteration = newIteration;

    // 4. Auto-follow if enabled
    if (request.autoFollowIteration) {
        request.viewingIteration = newIteration;

        // Clear panels for new iteration if this is the selected request
        if (request.sessionId === viewState.selectedRequestId) {
            clearPanelsForNewIteration();
        }
    }

    // 5. Re-render timeline
    renderIterationTimeline();
}

/**
 * Clear all streaming panels when starting a new iteration
 */
function clearPanelsForNewIteration() {
    REQUEST_PANEL_PHASES.forEach(phase => {
        resetPanelContent(phase);
    });
}


// ============================================
// ITERATION TIMELINE
// ============================================

/**
 * Render the iteration timeline for the currently selected request
 */
function renderIterationTimeline() {
    const timeline = document.getElementById('iterationTimeline');
    const container = document.getElementById('iterationContainer');

    if (!timeline || !container) return;

    // Only show if a session is selected (not Overview mode)
    if (isOverviewMode()) {
        timeline.classList.add('hidden');
        return;
    }

    const request = requestsData.requests[viewState.selectedRequestId];
    if (!request) {
        timeline.classList.add('hidden');
        return;
    }

    timeline.classList.remove('hidden');
    container.innerHTML = '';

    // Create pills for each iteration
    for (let i = 1; i <= request.maxIterations; i++) {
        const pill = createIterationPill(request, i);
        container.appendChild(pill);

        // Add connector if not the last iteration
        if (i < request.maxIterations) {
            const connector = document.createElement('span');
            connector.className = 'iteration-connector';
            connector.textContent = '---';
            container.appendChild(connector);
        }
    }

    // Update navigation buttons
    updateIterationNavButtons(request);
}

/**
 * Create a single iteration pill element
 * @param {Object} request - The request object
 * @param {number} iteration - The iteration number
 * @returns {HTMLElement} The pill element
 */
function createIterationPill(request, iteration) {
    const pill = document.createElement('div');
    pill.className = 'iteration-pill';
    pill.dataset.iteration = iteration;

    const iterData = request.iterations[iteration];
    const isCurrent = iteration === request.currentIteration;
    const isViewing = iteration === request.viewingIteration;
    const isPending = !iterData;

    // Determine status and score
    let status = 'pending';
    let score = '--';

    if (iterData) {
        status = iterData.status;
        score = iterData.score !== null ? iterData.score.toFixed(1) : '...';
    }

    // Apply state classes
    if (isPending) {
        pill.classList.add('pending');
    } else if (status === 'approved') {
        pill.classList.add('completed-pass');
    } else if (status === 'rejected') {
        pill.classList.add('completed-fail');
    } else if (status === 'in_progress') {
        pill.classList.add('in-progress');
    }

    if (isViewing) {
        pill.classList.add('viewing');
    }

    // Build content
    const liveIndicator = isCurrent && status === 'in_progress' ? '<span class="live-dot"></span>' : '';
    const statusLabel = status === 'approved' ? 'PASS' :
                       status === 'rejected' ? 'FAIL' :
                       status === 'in_progress' ? 'LIVE' : 'PEND';

    pill.innerHTML = `
        <div class="pill-header">ITER ${iteration} ${liveIndicator}</div>
        <div class="pill-score">${score}</div>
        <div class="pill-status">${statusLabel}</div>
    `;

    // Click handler (only for iterations with data)
    if (!isPending) {
        pill.onclick = () => selectIteration(request.sessionId, iteration);
    }

    return pill;
}

/**
 * Select a specific iteration to view
 * @param {string} sessionId - The session ID
 * @param {number} iteration - The iteration number to view
 */
function selectIteration(sessionId, iteration) {
    const request = requestsData.requests[sessionId];
    if (!request) return;

    request.viewingIteration = iteration;
    request.autoFollowIteration = (iteration === request.currentIteration);

    // Reload content for this iteration
    loadRequestContent(sessionId);

    // Re-render timeline
    renderIterationTimeline();

    log(`Viewing iteration ${iteration}`, 'info');
}

/**
 * Navigate to live (current) iteration
 */
function goToLiveIteration() {
    if (isOverviewMode()) return;

    const request = requestsData.requests[viewState.selectedRequestId];
    if (!request) return;

    request.viewingIteration = request.currentIteration;
    request.autoFollowIteration = true;

    loadRequestContent(request.sessionId);
    renderIterationTimeline();

    log('Switched to live iteration', 'info');
}

/**
 * Navigate to previous or next iteration
 * @param {number} direction - -1 for previous, 1 for next
 */
function navigateIteration(direction) {
    if (isOverviewMode()) return;

    const request = requestsData.requests[viewState.selectedRequestId];
    if (!request) return;

    const newIteration = request.viewingIteration + direction;

    // Validate bounds
    if (newIteration < 1 || newIteration > request.currentIteration) return;

    selectIteration(request.sessionId, newIteration);
}

/**
 * Update the enabled/disabled state of navigation buttons
 * @param {Object} request - The request object
 */
function updateIterationNavButtons(request) {
    const prevBtn = document.getElementById('iterPrevBtn');
    const nextBtn = document.getElementById('iterNextBtn');
    const liveBtn = document.getElementById('iterLiveBtn');

    if (!prevBtn || !nextBtn || !liveBtn) return;

    const viewing = request.viewingIteration;
    const current = request.currentIteration;

    prevBtn.disabled = viewing <= 1;
    nextBtn.disabled = viewing >= current;

    // Live button appearance
    if (viewing === current) {
        liveBtn.classList.add('active');
        liveBtn.textContent = 'Live';
    } else {
        liveBtn.classList.remove('active');
        liveBtn.textContent = 'Go Live';
    }
}

/**
 * Update history mode visual indicators on stream panels
 * @param {Object} request - The request object
 */
function updateHistoryModeIndicators(request) {
    const isHistoryMode = request.viewingIteration !== request.currentIteration;

    REQUEST_PANEL_PHASES.forEach(phase => {
        const panel = document.querySelector(`.stream-panel[data-phase="${phase}"]`);
        if (!panel) return;

        if (isHistoryMode) {
            panel.classList.add('history-mode');

            // Add iteration badge if it doesn't exist
            let iterBadge = panel.querySelector('.iter-badge');
            const header = panel.querySelector('.stream-header');
            const streamBadge = panel.querySelector('.stream-badge');

            if (!iterBadge && header && streamBadge) {
                iterBadge = document.createElement('span');
                iterBadge.className = 'iter-badge';
                header.insertBefore(iterBadge, streamBadge);
            }
            if (iterBadge) {
                iterBadge.textContent = `ITER ${request.viewingIteration}`;
            }

            // Add history badge if it doesn't exist
            let historyBadge = panel.querySelector('.history-badge');
            if (!historyBadge && header && streamBadge) {
                historyBadge = document.createElement('span');
                historyBadge.className = 'history-badge';
                historyBadge.textContent = 'HISTORY';
                header.insertBefore(historyBadge, streamBadge);
            }
        } else {
            panel.classList.remove('history-mode');

            // Remove badges
            const iterBadge = panel.querySelector('.iter-badge');
            const historyBadge = panel.querySelector('.history-badge');
            if (iterBadge) iterBadge.remove();
            if (historyBadge) historyBadge.remove();
        }
    });
}

/**
 * Show activity indicator when live data arrives while viewing history
 * @param {number} iteration - The iteration receiving live data
 */
function showLiveActivityIndicator(iteration) {
    const timeline = document.getElementById('iterationTimeline');
    if (!timeline) return;

    const pill = timeline.querySelector(`[data-iteration="${iteration}"]`);
    if (pill && !pill.classList.contains('has-activity')) {
        pill.classList.add('has-activity');

        // Remove after animation
        setTimeout(() => {
            pill.classList.remove('has-activity');
        }, 300);
    }
}


// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    log('Gran Sabio LLM Stream Monitor initialized', 'success');
    log(`Stream endpoint: ${STREAM_BASE}/stream/project/{id}`, 'info');

    // Initialize toggle switch listeners
    initToggleListeners();
    log('Phase toggle switches ready', 'info');

    // Initialize analysis filter pill listeners
    bindAnalysisFilterListeners();
    log('Analysis filter pills ready', 'info');

    // Initialize tabs navigation buttons
    const tabsNavLeft = document.getElementById('tabsNavLeft');
    const tabsNavRight = document.getElementById('tabsNavRight');
    if (tabsNavLeft) {
        tabsNavLeft.addEventListener('click', navigateTabsLeft);
    }
    if (tabsNavRight) {
        tabsNavRight.addEventListener('click', navigateTabsRight);
    }
    log('Tabs navigation ready', 'info');

    // Load active connections
    loadRecentProjects();

    // Check URL for project parameter
    const urlProject = getProjectFromUrl();
    if (urlProject) {
        log(`Found project in URL: ${urlProject}`, 'info');
        document.getElementById('projectIdInput').value = urlProject;
        // Auto-connect after a short delay
        setTimeout(() => connect(), 500);
    }

    // Allow Enter key to connect
    document.getElementById('projectIdInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            // Unified: project_id = session_id when not explicitly provided
            connect();
        }
    });

    // Close overflow menu when clicking outside
    document.addEventListener('click', (e) => {
        const menu = document.getElementById('tabsOverflowMenu');
        const overflowBtn = document.querySelector('.tabs-overflow-btn');
        if (menu && !menu.contains(e.target) && e.target !== overflowBtn) {
            menu.classList.add('hidden');
        }
    });
});

// ============================================
// SMART EDIT VISUALIZATION
// ============================================

// Smart edit state for animations
const smartEditState = {
    totalEdits: 0,
    currentParagraph: null,
    totalParagraphs: null,
    charDelta: 0,
    animationSpeed: 1.0,  // Multiplier for animation timing
};

function shouldRenderRequestScopedPanelEvent(data, panelPhase) {
    const request = processRequestFromEvent(data);
    if (!request) {
        return true;
    }
    showReceiving(panelPhase);
    if (isOverviewMode()) {
        return false;
    }
    if (request.sessionId !== viewState.selectedRequestId) {
        return false;
    }
    if (request.viewingIteration !== request.currentIteration) {
        showLiveActivityIndicator(request.currentIteration);
        return false;
    }
    return true;
}

/**
 * Handle smart edit start event - show paragraph with fragment highlighted
 */
function handleSmartEditStart(data) {
    if (!shouldRenderRequestScopedPanelEvent(data, 'smartedit')) return;

    const editData = data.edit_data;
    if (!editData) return;

    // Update state
    smartEditState.currentParagraph = editData.paragraph_index;
    smartEditState.totalParagraphs = editData.total_paragraphs;

    // Update stats display
    updateSmartEditStats();

    // Show receiving indicator
    showReceiving('smartedit');
    const badge = document.getElementById('badge-smartedit');
    if (badge) {
        badge.textContent = 'editing...';
        badge.classList.add('badge-editing');
    }

    // Get the panel content area
    const panel = document.getElementById('content-smartedit');
    if (!panel) return;

    // Build the content with highlighted fragment
    const textBefore = editData.text_before || '';
    const fragment = editData.fragment;
    const fragPos = editData.fragment_position;

    let html = '';

    // Add operation header
    html += '<div class="smartedit-op-header">';
    html += `<span class="smartedit-op-type ai">EDIT</span>`;
    html += `<span class="smartedit-op-desc">${escapeHtml(editData.description || 'Processing...')}</span>`;
    html += `<span class="smartedit-op-progress">${editData.paragraph_index}/${editData.total_paragraphs}</span>`;
    html += '</div>';

    // Build paragraph with highlighted fragment
    if (fragment && fragPos) {
        const before = textBefore.substring(0, fragPos.start);
        const highlighted = textBefore.substring(fragPos.start, fragPos.end);
        const after = textBefore.substring(fragPos.end);

        html += '<div class="smartedit-text">';
        html += escapeHtml(before);
        html += `<span class="edit-target ai-processing"><span class="ai-content">${escapeHtml(highlighted)}</span><div class="ai-progress-bar"><div class="ai-progress-fill"></div></div></span>`;
        html += escapeHtml(after);
        html += '</div>';
    } else {
        // No specific fragment - highlight entire paragraph
        html += '<div class="smartedit-text">';
        html += `<span class="edit-target ai-processing"><span class="ai-content">${escapeHtml(textBefore)}</span><div class="ai-progress-bar"><div class="ai-progress-fill"></div></div></span>`;
        html += '</div>';
    }

    panel.innerHTML = html;

    log(`Smart Edit: Starting paragraph ${editData.paragraph_index}/${editData.total_paragraphs}`, 'info');
}

/**
 * Handle smart edit complete event - animate transition and show result
 */
function handleSmartEditComplete(data) {
    if (!shouldRenderRequestScopedPanelEvent(data, 'smartedit')) return;

    const editData = data.edit_data;
    if (!editData) return;

    // Update state
    smartEditState.totalEdits++;
    smartEditState.charDelta += editData.char_delta || 0;

    // Update stats display
    updateSmartEditStats();

    // Get the panel content area
    const panel = document.getElementById('content-smartedit');
    if (!panel) return;

    const badge = document.getElementById('badge-smartedit');

    // Determine operation type for animation
    const opType = (editData.operation_type || 'replace').toLowerCase();
    const isAI = editData.ai_assisted;

    // Get the current processing element
    const processingEl = panel.querySelector('.ai-processing');

    if (processingEl) {
        // Animate the transition
        animateSmartEditComplete(processingEl, editData, opType, isAI, () => {
            // After animation, show final result
            showSmartEditResult(panel, editData);

            // Update badge
            if (badge) {
                badge.textContent = 'done';
                badge.classList.remove('badge-editing');
                badge.classList.add('badge-success');
                setTimeout(() => {
                    badge.classList.remove('badge-success');
                    badge.textContent = 'idle';
                }, 2000);
            }
        });
    } else {
        // No animation element, just show result
        showSmartEditResult(panel, editData);
    }

    log(`Smart Edit: Completed paragraph ${editData.paragraph_index}/${editData.total_paragraphs} (${editData.char_delta > 0 ? '+' : ''}${editData.char_delta} chars)`, 'success');
}

/**
 * Handle smart edit error event
 */
function handleSmartEditError(data) {
    if (!shouldRenderRequestScopedPanelEvent(data, 'smartedit')) return;

    const editData = data.edit_data;
    if (!editData) return;

    const panel = document.getElementById('content-smartedit');
    const badge = document.getElementById('badge-smartedit');

    if (badge) {
        badge.textContent = 'error';
        badge.classList.remove('badge-editing');
        badge.classList.add('badge-error');
    }

    if (panel) {
        let html = '<div class="smartedit-op-header" style="background: rgba(239, 68, 68, 0.2);">';
        html += '<span class="smartedit-op-type" style="background: rgba(239, 68, 68, 0.3); color: #ef4444;">ERROR</span>';
        html += `<span class="smartedit-op-desc" style="color: #ef4444;">${escapeHtml(editData.error || 'Unknown error')}</span>`;
        html += '</div>';

        if (editData.text_before) {
            html += `<div class="smartedit-text" style="opacity: 0.5;">${escapeHtml(editData.text_before)}</div>`;
        }

        panel.innerHTML = html;
    }

    log(`Smart Edit: Error on paragraph ${editData.paragraph_index} - ${editData.error}`, 'error');
}

/**
 * Animate the completion of a smart edit operation
 */
function animateSmartEditComplete(element, editData, opType, isAI, onComplete) {
    const speed = smartEditState.animationSpeed;
    const getTime = (ms) => ms / speed;

    // Remove processing animation
    element.classList.remove('ai-processing');

    // Add completion class
    element.classList.add('ai-complete');

    // Remove progress bar
    const progressBar = element.querySelector('.ai-progress-bar');
    if (progressBar) {
        progressBar.remove();
    }

    // Get the content span
    const contentSpan = element.querySelector('.ai-content');

    if (contentSpan && editData.text_after) {
        // Blur effect
        contentSpan.classList.add('ai-blur');

        setTimeout(() => {
            // Replace content
            contentSpan.textContent = editData.text_after;
            contentSpan.classList.remove('ai-blur');

            // Success glow
            setTimeout(() => {
                if (onComplete) onComplete();
            }, getTime(500));
        }, getTime(300));
    } else {
        setTimeout(() => {
            if (onComplete) onComplete();
        }, getTime(500));
    }
}

/**
 * Show the final result of smart edit in the panel
 */
function showSmartEditResult(panel, editData) {
    let html = '';

    // Header showing completion
    html += '<div class="smartedit-op-header" style="background: rgba(16, 185, 129, 0.1);">';
    html += `<span class="smartedit-op-type" style="background: rgba(16, 185, 129, 0.2); color: #10b981;">${editData.ai_assisted ? 'AI' : 'DIRECT'}</span>`;
    html += `<span class="smartedit-op-desc" style="color: #10b981;">Edit applied successfully</span>`;
    html += `<span class="smartedit-op-progress">${editData.paragraph_index}/${editData.total_paragraphs}</span>`;
    html += '</div>';

    // Result text with success styling
    html += '<div class="smartedit-result">';
    html += '<div class="smartedit-result-label">Result:</div>';
    html += `<div class="smartedit-text">${escapeHtml(editData.text_after || '')}</div>`;
    html += '</div>';

    // Stats
    const charDelta = editData.char_delta || 0;
    const wordDelta = editData.word_delta || 0;
    html += '<div class="smartedit-stats" style="margin-top: 0.5rem; font-size: 0.75rem; color: var(--text-muted, #64748b);">';
    html += `<span>Chars: ${charDelta > 0 ? '+' : ''}${charDelta}</span>`;
    html += `<span style="margin-left: 1rem;">Words: ${wordDelta > 0 ? '+' : ''}${wordDelta}</span>`;
    html += '</div>';

    panel.innerHTML = html;
}

/**
 * Update smart edit stats display
 */
function updateSmartEditStats() {
    const editsEl = document.getElementById('edits-smartedit');
    const paragraphEl = document.getElementById('paragraph-smartedit');
    const deltaEl = document.getElementById('delta-smartedit');

    if (editsEl) {
        editsEl.textContent = smartEditState.totalEdits;
    }

    if (paragraphEl) {
        if (smartEditState.currentParagraph && smartEditState.totalParagraphs) {
            paragraphEl.textContent = `${smartEditState.currentParagraph}/${smartEditState.totalParagraphs}`;
        } else {
            paragraphEl.textContent = '-/-';
        }
    }

    if (deltaEl) {
        const delta = smartEditState.charDelta;
        deltaEl.textContent = delta > 0 ? `+${delta}` : delta.toString();
        deltaEl.style.color = delta > 0 ? '#10b981' : (delta < 0 ? '#ef4444' : 'inherit');
    }
}

/**
 * Escape HTML for safe display
 */
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (eventSource) {
        eventSource.close();
    }
});
