/**
 * Gran Sabio LLM - Web Interface JavaScript
 * Professional content generation interface with real-time progress tracking
 */

/**
 * Escapes HTML special characters to prevent XSS attacks.
 * @param {string} str - The string to escape
 * @returns {string} - The escaped string safe for innerHTML
 */
function escapeHtml(str) {
    if (str === null || str === undefined) {
        return '';
    }
    const text = String(str);
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

class GranSabioInterface {
    constructor() {
        this.currentSession = null;
        this.progressInterval = null;
        this.qaLayerCount = 0;
        this.currentScreen = 'configuration';
        this.eventSource = null;
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.initializeSliders();
        this.initializeTabs();
        this.loadDefaultQALayers();
        this.setupFormValidation();
        this.loadQAModels();
        this.initializeLogToggle();
        this.initializeDownloadButton();
        this.initializeVerboseToggle();
        this.initializeScreens();
    }
    
    // ==========================================
    // SCREEN MANAGEMENT
    // ==========================================
    
    initializeScreens() {
        // Ensure we start on configuration screen
        this.showConfigurationScreen();
        
        // Bind screen navigation events
        const backBtn = document.getElementById('backToConfigBtn');
        if (backBtn) {
            backBtn.addEventListener('click', () => {
                this.showConfigurationScreen();
            });
        }
        
        const newGenerationBtn = document.getElementById('newGenerationBtn');
        if (newGenerationBtn) {
            newGenerationBtn.addEventListener('click', () => {
                this.showConfigurationScreen();
                this.resetForm();
            });
        }
        
        const duplicateConfigBtn = document.getElementById('duplicateConfigBtn');
        if (duplicateConfigBtn) {
            duplicateConfigBtn.addEventListener('click', () => {
                this.showConfigurationScreen();
            });
        }
    }
    
    showConfigurationScreen() {
        this.currentScreen = 'configuration';
        
        const configScreen = document.getElementById('configurationScreen');
        const processingScreen = document.getElementById('processingScreen');
        const backBtn = document.getElementById('backToConfigBtn');
        
        if (configScreen && processingScreen) {
            configScreen.style.display = 'flex';
            processingScreen.style.display = 'none';
        }
        
        if (backBtn) {
            backBtn.style.display = 'none';
        }
        
        // Stop any active generation
        this.stopGeneration();
    }
    
    showProcessingScreen() {
        this.currentScreen = 'processing';
        
        const configScreen = document.getElementById('configurationScreen');
        const processingScreen = document.getElementById('processingScreen');
        const backBtn = document.getElementById('backToConfigBtn');
        
        if (configScreen && processingScreen) {
            configScreen.style.display = 'none';
            processingScreen.style.display = 'flex';
        }
        
        if (backBtn) {
            backBtn.style.display = 'block';
        }
        
        // Initialize processing screen
        this.initializeProcessingScreen();
    }
    
    initializeProcessingScreen() {
        // Show waiting section, hide final results
        const waitingSection = document.getElementById('waitingSection');
        const finalResults = document.getElementById('finalResultsSection');

        if (waitingSection) {
            waitingSection.style.display = 'block';
        }
        if (finalResults) {
            finalResults.style.display = 'none';
        }

        // Set initial status
        this.updateProcessingStatus('Initializing', 'Preparing generation...');

        // Sync iteration counter with form value
        const maxEl = document.getElementById('maxIterationsDisplay');
        const curEl = document.getElementById('currentIterationDisplay');
        if (maxEl) maxEl.textContent = document.getElementById('maxIterations')?.value || '3';
        if (curEl) curEl.textContent = '0';
    }
    
    updateProcessingStatus(status, progressText) {
        const statusBadge = document.getElementById('processingStatusBadge');
        const progressTextElement = document.getElementById('progressText');
        
        if (statusBadge) {
            statusBadge.textContent = status;
        }
        
        if (progressTextElement) {
            progressTextElement.textContent = progressText;
        }
    }
    
    // ==========================================
    // NEW GENERATION FLOW
    // ==========================================
    
    async startGeneration() {
        const formData = this.collectFormData();
        
        if (!this.validateFormData(formData)) {
            this.showError('Please fill in all required fields');
            return;
        }
        
        try {
            // Switch to processing screen
            this.showProcessingScreen();
            
            // Start generation
            const response = await this.submitGeneration(formData);
            
            if (response.ok) {
                const result = await response.json();
                this.currentSession = result.session_id;
                
                // Update session display
                const sessionIdDisplay = document.getElementById('sessionIdDisplay');
                if (sessionIdDisplay) {
                    sessionIdDisplay.textContent = this.currentSession;
                }
                
                // Start progress tracking and streaming
                this.startProgressTracking();
                
            } else {
                const error = await response.text();
                this.showError(`Generation failed: ${error}`);
                this.showConfigurationScreen();
            }
        } catch (error) {
            this.showError(`Network error: ${error.message}`);
            this.showConfigurationScreen();
        }
    }
    
    stopGeneration() {
        if (this.currentSession) {
            this.handleStop();
        }
        
        // Stop streaming
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        
        // Stop progress tracking
        this.stopProgressTracking();
        
        // Update status
        this.updateProcessingStatus('Stopped', 'Generation stopped by user');
    }
    
    startContentStreaming() {
        if (!this.currentSession) return;
        
        this.eventSource = new EventSource(`/stream/project/${this.currentSession}`);
        
        this.eventSource.onmessage = (event) => {
            try {
                const update = JSON.parse(event.data);
                this.handleStreamUpdate(update);
            } catch (error) {
                console.error('Failed to parse stream data:', error);
            }
        };
        
        this.eventSource.onerror = (error) => {
            console.error('EventSource error:', error);
            // Continue with polling as fallback
        };
    }
    
    handleStreamUpdate(update) {
        if (!update || typeof update !== 'object') {
            return;
        }

        // Debug logging
        console.log('[Stream Update]', update.type || update.status, update);

        if (typeof update.type === 'string') {
            const eventType = update.type;

            if (eventType === 'connected') {
                return;
            }

            if (eventType === 'snapshot' || eventType === 'status_change') {
                const statusValue = this.getStreamStatusValue(update);
                this.applyStreamStatusUpdate(statusValue, update);
                if (this.isFinalStatus(statusValue)) {
                    this.finishStream(statusValue, update.reason || '');
                }
                return;
            }

            if (eventType === 'chunk') {
                const phase = update.phase === 'gran_sabio' ? 'gransabio' : update.phase;
                if (update.is_thinking === true) {
                    return;
                }
                if (typeof update.content === 'string' && (phase === 'generation' || phase === 'consensus')) {
                    this.updateContentStream(update.content, { append: true });
                }
                return;
            }

            if (eventType === 'log') {
                if (typeof update.message === 'string' && update.message.length > 0) {
                    this.addNewVerboseLogs([update.message]);
                }
                return;
            }

            if (eventType === 'stream_end') {
                const statusValue = this.getStreamStatusValue(update);
                this.finishStream(statusValue, update.reason || '');
                return;
            }

            // Handle session_end event (final completion)
            if (eventType === 'session_end') {
                const statusValue = this.normalizeStatus(update.status);
                if (this.isFinalStatus(statusValue)) {
                    this.finishStream(statusValue, '');
                }
                return;
            }
        }

        // 1) Estado y progreso
        if (update.status) {
            this.updateProcessingStatus(this.formatStatus(update.status), update.progress_text || '');
        }

        // Barra de progreso
        const progressBar = document.getElementById('progressBar') || document.getElementById('mainProgressBar')?.querySelector('.progress-fill');
        if (progressBar) {
            const progress = this.calculateProgress(update);
            progressBar.style.width = `${progress}%`;
        }

        // 2) Iteraciones (separado en current y max)
        const curEl = document.getElementById('currentIterationDisplay');
        const maxEl = document.getElementById('maxIterationsDisplay');
        if (curEl && update.current_iteration !== undefined) curEl.textContent = update.current_iteration;
        if (maxEl && update.max_iterations !== undefined) maxEl.textContent = update.max_iterations;

        // 3) Streaming de contenido
        // Soporta tanto deltas como contenido completo incremental
        if (typeof update.delta === 'string') {
            this.updateContentStream(update.delta, { append: true });
        } else if (typeof update.content_chunk === 'string') {
            this.updateContentStream(update.content_chunk, { append: true });
        } else if (typeof update.generated_content === 'string') {
            // Si el backend envía el texto completo acumulado
            this.updateContentStream(update.generated_content, { append: false });
        }

        // 4) QA feedback and verbose logs - now shown in Live Matrix (/monitor)
        // Keeping console.log for debugging purposes
        if (update.qa_feedback) {
            console.log('[QA Feedback]', update.qa_feedback);
        }
        if (update.verbose_log) {
            console.log('[Verbose]', update.verbose_log);
        }

        // 5) Fin de proceso
        const normalizedStatus = this.normalizeStatus(update.status);
        if (this.isFinalStatus(normalizedStatus)) {
            if (this.eventSource) {
                this.eventSource.close();
                this.eventSource = null;
            }
            this.stopProgressTracking();
            this.handleGenerationComplete({ ...update, status: normalizedStatus });
        }
    }

    getStreamStatusValue(update) {
        // Handle status object with value property
        if (update && typeof update.status === 'object' && update.status !== null && 'value' in update.status) {
            return this.normalizeStatus(update.status.value);
        }
        // Handle project.status (from status_change events)
        if (update && update.project && update.project.status) {
            return this.normalizeStatus(update.project.status);
        }
        // Handle direct status string
        if (update && update.status) {
            return this.normalizeStatus(update.status);
        }
        return update.current_phase || null;
    }

    applyStreamStatusUpdate(statusValue, update) {
        if (statusValue) {
            this.updateProcessingStatus(this.formatStatus(statusValue), update.progress_text || '');
        }

        const progressBar = document.getElementById('progressBar') || document.getElementById('mainProgressBar')?.querySelector('.progress-fill');
        if (progressBar && statusValue) {
            const progress = this.calculateProgress({
                status: statusValue,
                current_iteration: update.current_iteration ?? 0,
                max_iterations: update.max_iterations ?? 0
            });
            progressBar.style.width = `${progress}%`;
        }

        const curEl = document.getElementById('currentIterationDisplay');
        const maxEl = document.getElementById('maxIterationsDisplay');
        if (curEl && update.current_iteration !== undefined) curEl.textContent = update.current_iteration;
        if (maxEl && update.max_iterations !== undefined) maxEl.textContent = update.max_iterations;
    }

    isFinalStatus(statusValue) {
        return ['completed', 'failed', 'cancelled'].includes(statusValue);
    }

    normalizeStatus(status) {
        if (!status) return null;
        // Handle "GenerationStatus.COMPLETED" format
        if (typeof status === 'string' && status.includes('.')) {
            const parts = status.split('.');
            return parts[parts.length - 1].toLowerCase();
        }
        return status.toLowerCase ? status.toLowerCase() : status;
    }

    finishStream(statusValue, reason) {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        this.stopProgressTracking();
        this.setLoadingState(false);

        // Hide waiting section
        const waitingSection = document.getElementById('waitingSection');
        if (waitingSection) {
            waitingSection.style.display = 'none';
        }

        if (statusValue) {
            this.updateProcessingStatus(this.formatStatus(statusValue), reason || '');
        }

        if (this.isFinalStatus(statusValue)) {
            this.showFinalResults();
        }
    }
    
    updateContentStream(text, { append = false } = {}) {
        // Content streaming removed - final content shown only after completion
        // See Live Matrix (/monitor) for real-time streaming
        if (text) {
            console.log('[Content Stream]', text.substring(0, 100) + (text.length > 100 ? '...' : ''));
        }
    }
    
    updateQAFeedback(qaFeedback) {
        // QA feedback is now shown in Live Matrix (/monitor)
        // Keeping console.log for debugging
        if (Array.isArray(qaFeedback)) {
            console.log('[QA Feedback Update]', qaFeedback);
        }
    }
    
    addToProcessLog(logText) {
        // Process log is now shown in Live Matrix (/monitor)
        if (logText) {
            console.log(`[Process Log] ${logText}`);
        }
    }
    
    handleGenerationComplete(update) {
        // Stop streaming
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }

        // Stop progress tracking
        this.stopProgressTracking();

        // Reset loading state (hides Stop button)
        this.setLoadingState(false);

        // Update status text
        const statusText = update.status === 'failed' ? 'Failed' :
                          update.status === 'cancelled' ? 'Cancelled' : 'Completed';
        this.updateProcessingStatus(statusText, 'Loading results...');

        // Fetch and display final results from the API
        this.showFinalResults();
    }

    updateResultStatusBadge(isApproved, status) {
        const badge = document.getElementById('resultStatusBadge');
        if (!badge) return;

        const iconSpan = badge.querySelector('.status-icon');
        const textSpan = badge.querySelector('.status-text');

        if (status === 'cancelled') {
            badge.className = 'result-status-badge cancelled';
            if (iconSpan) iconSpan.textContent = '[X]';
            if (textSpan) textSpan.textContent = 'Cancelled';
        } else if (status === 'failed' || !isApproved) {
            badge.className = 'result-status-badge rejected';
            if (iconSpan) iconSpan.textContent = '[!]';
            if (textSpan) textSpan.textContent = 'Rejected';
        } else {
            badge.className = 'result-status-badge approved';
            if (iconSpan) iconSpan.textContent = '[OK]';
            if (textSpan) textSpan.textContent = 'Approved';
        }
    }

    bindEvents() {
        // Form submission - now switches to processing screen
        document.getElementById('startGenerationBtn')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.startGeneration();
        });
        
        // Legacy form submission for compatibility
        document.getElementById('generationForm')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.startGeneration();
        });
        
        // Stop generation buttons
        document.getElementById('stopGenerationBtn')?.addEventListener('click', () => {
            this.stopGeneration();
        });

        // Reset form
        document.getElementById('resetBtn')?.addEventListener('click', () => {
            this.resetForm();
        });

        // Legacy stop button (if exists)
        document.getElementById('stopBtn')?.addEventListener('click', () => {
            this.handleStop();
        });

        // Content type change
        document.getElementById('contentType')?.addEventListener('change', (e) => {
            this.loadQALayersForContentType(e.target.value);
        });

        // Generator model change - show/hide reasoning controls
        document.getElementById('generatorModel')?.addEventListener('change', (e) => {
            this.updateReasoningControls(e.target.value);
        });

        // Add QA Layer
        document.getElementById('addQaLayer')?.addEventListener('click', () => {
            this.addQALayer();
        });

        // Copy content buttons
        document.getElementById('copyContent')?.addEventListener('click', () => {
            this.copyGeneratedContent();
        });
        
        document.getElementById('copyFinalContent')?.addEventListener('click', () => {
            this.copyGeneratedContent();
        });

        // Word count enforcement checkbox
        document.getElementById('wordCountEnabled')?.addEventListener('change', (e) => {
            this.toggleWordCountOptions(e.target.checked);
        });

        // Word count parameters change
        document.getElementById('minWords')?.addEventListener('input', () => {
            this.updateWordCountPreview();
        });
        document.getElementById('maxWords')?.addEventListener('input', () => {
            this.updateWordCountPreview();
        });
        document.getElementById('flexibilityPercent')?.addEventListener('input', () => {
            this.updateWordCountPreview();
        });
        
        // Handle radio button changes for flexibility direction
        document.querySelectorAll('input[name="flexibilityDirection"]').forEach(radio => {
            radio.addEventListener('change', () => {
                this.updateWordCountPreview();
            });
        });
    }

    initializeSliders() {
        // Temperature slider
        const tempSlider = document.getElementById('temperature');
        const tempValue = document.querySelector('label[for="temperature"] + .slider-container .slider-value');
        
        if (tempSlider && tempValue) {
            tempSlider.addEventListener('input', (e) => {
                tempValue.textContent = e.target.value;
            });
        }

        // Global score slider
        const scoreSlider = document.getElementById('minGlobalScore');
        const scoreValue = document.querySelector('label[for="minGlobalScore"] + .slider-container .slider-value');
        
        if (scoreSlider && scoreValue) {
            scoreSlider.addEventListener('input', (e) => {
                scoreValue.textContent = e.target.value;
            });
        }

        // Flexibility percentage slider
        const flexSlider = document.getElementById('flexibilityPercent');
        const flexValue = document.querySelector('label[for="flexibilityPercent"] + .slider-container .slider-value');
        
        if (flexSlider && flexValue) {
            flexSlider.addEventListener('input', (e) => {
                flexValue.textContent = e.target.value + '%';
            });
        }

        // Thinking budget tokens slider
        const thinkingSlider = document.getElementById('thinkingBudgetTokens');
        const thinkingValue = document.querySelector('#thinkingBudgetGroup .slider-value');
        
        if (thinkingSlider && thinkingValue) {
            thinkingSlider.addEventListener('input', (e) => {
                const value = parseInt(e.target.value);
                const displayValue = value >= 1024 ? `${Math.round(value / 1024)}K tokens` : `${value} tokens`;
                thinkingValue.textContent = displayValue;
            });
        }
    }

    loadDefaultQALayers() {
        // Load default QA layers for the current content type
        this.loadQALayersForContentType('article');
    }

    loadQALayersForContentType(contentType) {
        const qaLayersContainer = document.getElementById('qaLayers');
        qaLayersContainer.innerHTML = '';
        this.qaLayerCount = 0;

        const defaultLayers = this.getDefaultQALayers(contentType);
        defaultLayers.forEach(layer => {
            this.addQALayer(layer);
        });
    }

    getDefaultQALayers(contentType) {
        const layerTemplates = {
            biography: [
                {
                    name: 'Historical Accuracy',
                    description: 'Verification of historical facts',
                    criteria: 'Verify all dates, events, and biographical details. Detect fabricated information.',
                    minScore: 8.5,
                    isMandatory: true,
                    dealBreakerCriteria: 'invents dates, historical events, or false biographical data'
                },
                {
                    name: 'Literary Quality',
                    description: 'Writing style and narrative',
                    criteria: 'Evaluate prose, narrative flow, and ability to maintain reader interest.',
                    minScore: 7.5,
                    isMandatory: false,
                    dealBreakerCriteria: null
                }
            ],
            script: [
                {
                    name: 'Script Format',
                    description: 'Compliance with standard format',
                    criteria: 'Verify correct formatting: headers, character names, dialogue, action descriptions.',
                    minScore: 8.0,
                    isMandatory: true,
                    dealBreakerCriteria: 'does not follow standard screenplay format'
                },
                {
                    name: 'Dialogue Quality',
                    description: 'Naturalness and effectiveness of dialogue',
                    criteria: 'Evaluate naturalness, voice differentiation, and subtext.',
                    minScore: 7.5,
                    isMandatory: false,
                    dealBreakerCriteria: null
                }
            ],
            article: [
                {
                    name: 'Technical Accuracy',
                    description: 'Information accuracy',
                    criteria: 'Verify that information is technically correct and up-to-date. Exception allowed for parody, satire and jokes.',
                    minScore: 8.0,
                    isMandatory: false,
                    dealBreakerCriteria: 'invents facts or presents false information as true'
                },
                {
                    name: 'Clarity and Accessibility',
                    description: 'Clarity for target audience',
                    criteria: 'Evaluate comprehensibility, use of examples, and clear explanations.',
                    minScore: 7.0,
                    isMandatory: false,
                    dealBreakerCriteria: null
                }
            ],
            novel: [
                {
                    name: 'Character Development',
                    description: 'Depth and consistency of characters',
                    criteria: 'Evaluate character development, consistency, and realistic character arcs.',
                    minScore: 7.5,
                    isMandatory: false,
                    dealBreakerCriteria: null
                },
                {
                    name: 'Plot Coherence',
                    description: 'Logic and narrative consistency',
                    criteria: 'Verify absence of plot holes and cause-effect logic.',
                    minScore: 8.0,
                    isMandatory: false,
                    dealBreakerCriteria: 'contains major plot holes or significant logical inconsistencies'
                }
            ],
            json: [
                {
                    name: 'Functional Validation',
                    description: 'Confirm generated JSON meets required fields and types.',
                    criteria: 'Verify all required keys are present, values use correct types, and explicit prompt rules are followed.',
                    minScore: 8.5,
                    isMandatory: true,
                    dealBreakerCriteria: 'missing required keys or invalid data types detected'
                }
            ]
        };

        return layerTemplates[contentType] || layerTemplates.article;
    }

    addQALayer(layerData = null) {
        const qaLayersContainer = document.getElementById('qaLayers');
        const layerId = `qa-layer-${this.qaLayerCount++}`;
        
        const defaultData = layerData || {
            name: '',
            description: '',
            criteria: '',
            minScore: 7.0,
            isMandatory: false,
            dealBreakerCriteria: null,
            // Backward compatibility
            isDealBreaker: false
        };

        const layerHtml = `
            <div class="qa-layer" id="${layerId}">
                <div class="qa-layer-header">
                    <span class="qa-layer-title">QA Layer ${this.qaLayerCount}</span>
                    <button type="button" class="qa-layer-remove" onclick="gransabio.removeQALayer('${layerId}')">
                        Remove
                    </button>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">Layer Name</label>
                        <input type="text" class="form-input" name="layerName" value="${defaultData.name}" required>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Description</label>
                        <input type="text" class="form-input" name="layerDescription" value="${defaultData.description}" required>
                    </div>
                </div>
                <div class="form-group">
                    <label class="form-label">Evaluation Criteria</label>
                    <textarea class="form-textarea" name="layerCriteria" rows="3" required>${defaultData.criteria}</textarea>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">Minimum Score</label>
                        <div class="slider-container">
                            <input type="range" class="form-slider" name="layerMinScore" 
                                   min="1" max="10" step="0.1" value="${defaultData.minScore}"
                                   oninput="this.nextElementSibling.textContent = this.value">
                            <span class="slider-value">${defaultData.minScore}</span>
                        </div>
                    </div>
                    <div class="form-group">
                        <label class="checkbox-label">
                            <input type="checkbox" name="layerIsMandatory" ${defaultData.isMandatory ? 'checked' : ''}>
                            <span class="checkmark"></span>
                            Mandatory Layer
                        </label>
                    </div>
                </div>
                <div class="form-group">
                    <label class="form-label">Deal-Breaker Criteria (Optional)</label>
                    <input type="text" class="form-input" name="layerDealBreakerCriteria" 
                           value="${defaultData.dealBreakerCriteria || ''}"
                           placeholder="e.g., 'invents facts', 'uses offensive language'">
                    <small class="form-help">Specific facts that would immediately reject content</small>
                </div>
            </div>
        `;

        qaLayersContainer.insertAdjacentHTML('beforeend', layerHtml);
    }

    removeQALayer(layerId) {
        document.getElementById(layerId).remove();
    }

    setupFormValidation() {
        const form = document.getElementById('generationForm');
        const inputs = form.querySelectorAll('input[required], textarea[required], select[required]');
        
        inputs.forEach(input => {
            input.addEventListener('blur', () => {
                this.validateField(input);
            });
        });
    }

    validateField(field) {
        if (!field.value.trim()) {
            field.style.borderColor = 'var(--error-500)';
            return false;
        } else {
            field.style.borderColor = 'var(--success-500)';
            return true;
        }
    }

    async handleSubmit() {
        const formData = this.collectFormData();
        
        if (!this.validateFormData(formData)) {
            this.showError('Please fill in all required fields');
            return;
        }

        try {
            this.setLoadingState(true);
            const response = await this.submitGeneration(formData);
            
            if (response.ok) {
                const result = await response.json();
                this.currentSession = result.session_id;
                this.showResults();
                this.startProgressTracking();
                // Keep loading state true - will be set to false when generation completes
            } else {
                const error = await response.text();
                this.showError(`Generation failed: ${error}`);
                this.setLoadingState(false);
            }
        } catch (error) {
            this.showError(`Network error: ${error.message}`);
            this.setLoadingState(false);
        }
    }

    async handleStop() {
        if (!this.currentSession) {
            this.showError('No active session to stop');
            return;
        }

        try {
            const response = await fetch(`/stop/${this.currentSession}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                const result = await response.json();
                if (result.stopped) {
                    this.stopProgressTracking();
                    this.setLoadingState(false);
                    
                    // Update status badge
                    const statusBadge = document.getElementById('statusBadge');
                    if (statusBadge) {
                        statusBadge.textContent = 'Cancelled';
                        statusBadge.className = 'status-badge failed';
                    }
                    
                    // Show cancellation message in progress log
                    this.addNewVerboseLogs(['Generation cancelled by user request']);
                } else {
                    this.showError(result.message || 'Could not stop session');
                }
            } else {
                const error = await response.text();
                this.showError(`Stop request failed: ${error}`);
            }
        } catch (error) {
            this.showError(`Network error: ${error.message}`);
        }
    }

    collectFormData() {
        const form = document.getElementById('generationForm');
        const formData = new FormData(form);
        
        // Collect QA models from the new interface
        const qaModels = this.getSelectedQAModels();

        // Collect QA layers
        const qaLayers = [];
        const qaLayerElements = form.querySelectorAll('.qa-layer');
        
        qaLayerElements.forEach((layer, index) => {
            const layerData = {
                name: layer.querySelector('input[name="layerName"]').value,
                description: layer.querySelector('input[name="layerDescription"]').value,
                criteria: layer.querySelector('textarea[name="layerCriteria"]').value,
                min_score: parseFloat(layer.querySelector('input[name="layerMinScore"]').value),
                is_mandatory: layer.querySelector('input[name="layerIsMandatory"]').checked,
                deal_breaker_criteria: layer.querySelector('input[name="layerDealBreakerCriteria"]').value || null,
                order: index + 1,
                // Backward compatibility
                is_deal_breaker: false
            };
            qaLayers.push(layerData);
        });

        const requestData = {
            prompt: formData.get('prompt'),
            content_type: formData.get('contentType'),
            generator_model: formData.get('generatorModel'),
            temperature: parseFloat(formData.get('temperature')),
            qa_models: qaModels,
            qa_layers: qaLayers,
            min_global_score: parseFloat(formData.get('minGlobalScore')),
            max_iterations: parseInt(formData.get('maxIterations')),
            gran_sabio_model: formData.get('granSabioModel'),
            gran_sabio_fallback: formData.has('granSabioFallback'),
            verbose: formData.has('verbose'),
            extra_verbose: formData.has('extra_verbose') && formData.has('verbose')
        };

        // Add word limits if specified
        const minWords = formData.get('minWords');
        const maxWords = formData.get('maxWords');
        
        if (minWords && parseInt(minWords) > 0) {
            requestData.min_words = parseInt(minWords);
        }
        
        if (maxWords && parseInt(maxWords) > 0) {
            requestData.max_words = parseInt(maxWords);
        }

        // Add word count enforcement configuration if enabled
        const wordCountEnabled = document.getElementById('wordCountEnabled').checked;
        if (wordCountEnabled && (requestData.min_words || requestData.max_words)) {
            requestData.word_count_enforcement = {
                enabled: true,
                flexibility_percent: parseInt(formData.get('flexibilityPercent') || '15'),
                direction: formData.get('flexibilityDirection') || 'both',
                severity: formData.get('wordCountSeverity') || 'important'
            };
        }
        
        // Keep max_tokens as fallback if no word limits are set
        if (!requestData.min_words && !requestData.max_words && maxWords) {
            requestData.max_tokens = this.wordsToTokens(parseInt(maxWords));
        } else if (!requestData.min_words && !requestData.max_words) {
            requestData.max_tokens = 4000; // Default
        }

        // Add reasoning effort parameter for supported models
        const reasoningEffortGroup = document.getElementById('reasoningEffortGroup');
        if (reasoningEffortGroup && reasoningEffortGroup.style.display !== 'none') {
            const reasoningEffort = formData.get('reasoningEffort');
            if (reasoningEffort) {
                requestData.reasoning_effort = reasoningEffort;
            }
        }

        // Add thinking budget tokens for supported models
        const thinkingBudgetGroup = document.getElementById('thinkingBudgetGroup');
        if (thinkingBudgetGroup && thinkingBudgetGroup.style.display !== 'none') {
            const thinkingBudget = formData.get('thinkingBudgetTokens');
            if (thinkingBudget) {
                requestData.thinking_budget_tokens = parseInt(thinkingBudget);
            }
        }
        
        return requestData;
    }

    validateFormData(data) {
        if (!data.prompt || data.prompt.trim() === '') return false;
        if (data.qa_models.length === 0) return false;
        if (data.qa_layers.length === 0) return false;
        
        for (const layer of data.qa_layers) {
            if (!layer.name || !layer.description || !layer.criteria) {
                return false;
            }
        }
        
        return true;
    }

    async submitGeneration(data) {
        return fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
    }

    showResults() {
        const resultsPanel = document.getElementById('resultsPanel');
        const finalResults = document.getElementById('finalResults');
        
        resultsPanel.style.display = 'block';
        finalResults.style.display = 'none';
        
        // Set session ID
        document.getElementById('sessionId').textContent = this.currentSession;
        
        // Scroll to results
        resultsPanel.scrollIntoView({ behavior: 'smooth' });
    }

    startProgressTracking() {
        if (!this.currentSession) return;

        // Use EventSource for real-time status updates
        this.startEventStream();

        // Fallback polling as last resort
        this.progressInterval = setInterval(() => {
            if (!this.eventSource || this.eventSource.readyState === EventSource.CLOSED) {
                this.updateProgress();
            }
        }, 3000);
    }

    startEventStream() {
        if (!this.currentSession) return;

        // Cerrar stream previo si existiera
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }

        this.eventSource = new EventSource(`/stream/project/${this.currentSession}`);

        this.eventSource.onmessage = (event) => {
            try {
                const update = JSON.parse(event.data);
                // Unificamos: progreso + contenido + QA + logs + cierre
                this.handleStreamUpdate(update);
            } catch (err) {
                console.error('Failed to parse stream data:', err);
            }
        };

        this.eventSource.onerror = (error) => {
            console.error('EventSource error:', error);
            // El fallback por polling ya está en startProgressTracking()
        };
    }

    async updateProgress() {
        if (!this.currentSession) return;

        try {
            const response = await fetch(`/status/${this.currentSession}`);
            const status = await response.json();

            this.updateProgressUI(status);

            if (status.status === 'completed' || status.status === 'failed' || status.status === 'cancelled') {
                this.stopProgressTracking();
                this.setLoadingState(false);

                // Hide waiting section
                const waitingSection = document.getElementById('waitingSection');
                if (waitingSection) {
                    waitingSection.style.display = 'none';
                }

                // Show final results
                await this.showFinalResults();
            }
        } catch (error) {
            console.error('Failed to fetch progress:', error);
        }
    }

    updateProgressUI(status) {
        // Update status badge
        const statusBadge = document.getElementById('statusBadge') || document.getElementById('processingStatusBadge');
        if (statusBadge) {
            statusBadge.textContent = this.formatStatus(status.status);
            statusBadge.className = `status-badge ${status.status}`;
        }
        
        // Update progress bar
        const progressBar = document.getElementById('progressBar') || document.getElementById('mainProgressBar')?.querySelector('.progress-fill');
        if (progressBar) {
            const progress = this.calculateProgress(status);
            progressBar.style.width = `${progress}%`;
        }
        
        // Update current iteration
        const currentIterationElement = document.getElementById('currentIteration') || document.getElementById('currentIterationDisplay');
        if (currentIterationElement) {
            currentIterationElement.textContent = `${status.current_iteration}/${status.max_iterations}`;
        }
        
        // Update verbose logs (only last 10 from polling)
        this.updateVerboseLogs(status.verbose_log);
    }

    formatStatus(status) {
        const normalized = this.normalizeStatus(status);
        const statusMap = {
            'initializing': 'Initializing',
            'generating': 'Generating',
            'qa_evaluation': 'QA Evaluation',
            'gran_sabio_review': 'Gran Sabio Review',
            'completed': 'Completed',
            'failed': 'Failed',
            'cancelled': 'Cancelled'
        };
        return statusMap[normalized] || status;
    }

    calculateProgress(status) {
        const statusProgress = {
            'initializing': 5,
            'generating': 25,
            'qa_evaluation': 70,
            'gran_sabio_review': 90,
            'completed': 100,
            'failed': 100
        };
        
        let baseProgress = statusProgress[status.status] || 0;
        
        // Add iteration progress
        if (status.max_iterations > 0) {
            const iterationProgress = (status.current_iteration / status.max_iterations) * 10;
            baseProgress += iterationProgress;
        }
        
        return Math.min(baseProgress, 100);
    }

    updateProgressFromStream(update) {
        // Update status badge
        const statusBadge = document.getElementById('statusBadge') || document.getElementById('processingStatusBadge');
        if (statusBadge) {
            statusBadge.textContent = this.formatStatus(update.status);
            statusBadge.className = `status-badge ${update.status}`;
        }
        
        // Update progress bar  
        const progressBar = document.getElementById('progressBar') || document.getElementById('mainProgressBar')?.querySelector('.progress-fill');
        if (progressBar) {
            const progress = this.calculateProgress(update);
            progressBar.style.width = `${progress}%`;
        }
        
        // Update current iteration
        const currentIterationElement = document.getElementById('currentIteration') || document.getElementById('currentIterationDisplay');
        if (currentIterationElement && update.max_iterations !== undefined) {
            currentIterationElement.textContent = `${update.current_iteration}/${update.max_iterations}`;
        }
        
        // Add new verbose logs from stream
        this.addNewVerboseLogs(update.verbose_log);
        
        // Check if complete
        if (update.status === 'completed' || update.status === 'failed' || update.status === 'cancelled') {
            this.stopProgressTracking();
            this.setLoadingState(false);  // Reset UI state
            
            // Show final results for completed, failed and cancelled generations
            this.showFinalResults();
        }
    }

    updateVerboseLogs(logs) {
        // Verbose logs are now shown in Live Matrix (/monitor)
        if (logs && logs.length > 0) {
            console.log('[Verbose Logs]', logs);
        }
    }

    addNewVerboseLogs(newLogs) {
        // Verbose logs are now shown in Live Matrix (/monitor)
        if (newLogs && newLogs.length > 0) {
            newLogs.forEach(log => console.log('[Verbose]', log));
        }
    }

    async showFinalResults() {
        try {
            const response = await fetch(`/result/${this.currentSession}`);
            const result = await response.json();

            // Hide waiting section
            const waitingSection = document.getElementById('waitingSection');
            if (waitingSection) {
                waitingSection.style.display = 'none';
            }

            // Determine if content was approved
            const isApproved = result.approved !== false;
            const status = result.approved === false ? 'rejected' : 'completed';

            // Update status badge
            this.updateResultStatusBadge(isApproved, status);

            // Get elements (using correct IDs)
            const finalScoreDisplay = document.getElementById('finalScoreDisplay');
            const finalIterationsDisplay = document.getElementById('finalIterationsDisplay');
            const finalWordCountDisplay = document.getElementById('finalWordCountDisplay');
            const finalContentDisplay = document.getElementById('finalContentDisplay');
            const finalResultsSection = document.getElementById('finalResultsSection');

            // Update metrics
            if (finalScoreDisplay && result.final_score !== undefined) {
                finalScoreDisplay.textContent = result.final_score.toFixed(1);
                finalScoreDisplay.style.color = isApproved ? '#10b981' : '#ef4444';
            }
            if (finalIterationsDisplay) {
                finalIterationsDisplay.textContent = result.final_iteration || '-';
            }

            // Count words
            if (finalWordCountDisplay && result.content) {
                const wordCount = result.content.trim().split(/\s+/).filter(word => word.length > 0).length;
                finalWordCountDisplay.textContent = wordCount;
            }

            // Clean up any previous failure reason elements
            const existingFailureReason = document.querySelector('.failure-reason');
            if (existingFailureReason) {
                existingFailureReason.remove();
            }

            // Apply styling based on approval status
            if (finalResultsSection) {
                if (isApproved) {
                    finalResultsSection.classList.remove('rejected');
                    finalResultsSection.classList.add('approved');
                } else {
                    finalResultsSection.classList.remove('approved');
                    finalResultsSection.classList.add('rejected');
                }
            }

            // Show content
            if (finalContentDisplay) {
                finalContentDisplay.textContent = result.content || '';
                finalContentDisplay.style.borderColor = isApproved ? '#10b981' : '#ef4444';
                finalContentDisplay.style.backgroundColor = isApproved ? '#f0fdf4' : '#fef2f2';
            }

            // Add failure reason if rejected
            if (!isApproved && result.failure_reason) {
                console.log('Content rejection reason:', result.failure_reason);

                const failureReasonElement = document.createElement('div');
                failureReasonElement.className = 'failure-reason';
                failureReasonElement.style.cssText = `
                    background-color: #fee2e2;
                    border: 1px solid #fecaca;
                    border-radius: 6px;
                    padding: 12px;
                    margin-bottom: 16px;
                    color: #dc2626;
                    font-size: 14px;
                `;
                failureReasonElement.innerHTML = `<strong>Rejection reason:</strong> ${escapeHtml(result.failure_reason)}`;

                // Insert before the content display
                if (finalContentDisplay && finalContentDisplay.parentNode) {
                    finalContentDisplay.parentNode.insertBefore(failureReasonElement, finalContentDisplay);
                }
            }

            // Log to console
            console.log('Generated content:', result.content);

            // Show final results section
            if (finalResultsSection) {
                finalResultsSection.style.display = 'block';
            }

        } catch (error) {
            this.showError(`Failed to load results: ${error.message}`);
        }
    }

    stopProgressTracking() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
        
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
    }

    setLoadingState(isLoading) {
        const generateBtn = document.getElementById('generateBtn') || document.getElementById('startGenerationBtn');
        const stopBtn = document.getElementById('stopBtn') || document.getElementById('stopGenerationBtn');
        
        if (!generateBtn) return; // Skip if no generation button found
        
        const btnText = generateBtn.querySelector('.btn-text');
        const btnSpinner = generateBtn.querySelector('.btn-spinner');
        
        if (isLoading) {
            generateBtn.disabled = true;
            if (stopBtn) {
                stopBtn.style.display = 'inline-flex';
            }
            if (btnText) btnText.textContent = 'Generating...';
            if (btnSpinner) btnSpinner.style.display = 'inline-block';
        } else {
            generateBtn.disabled = false;
            if (stopBtn) {
                stopBtn.style.display = 'none';
            }
            if (btnText) btnText.textContent = 'Start Generation';
            if (btnSpinner) btnSpinner.style.display = 'none';
        }
    }

    showError(message) {
        // Simple error display - could be enhanced with a modal or toast
        alert(`Error: ${message}`);
    }

    resetForm() {
        document.getElementById('generationForm').reset();
        this.stopProgressTracking();
        document.getElementById('resultsPanel').style.display = 'none';
        this.loadDefaultQALayers();
        
        // Reset sliders
        document.querySelector('label[for="temperature"] + .slider-container .slider-value').textContent = '0.7';
        document.querySelector('label[for="minGlobalScore"] + .slider-container .slider-value').textContent = '7.5';
    }

    copyGeneratedContent() {
        const content = document.getElementById('generatedContent').textContent;
        navigator.clipboard.writeText(content).then(() => {
            // Visual feedback for successful copy
            const copyBtn = document.getElementById('copyContent');
            const originalText = copyBtn.textContent;
            copyBtn.textContent = 'Copied!';
            copyBtn.style.background = 'var(--success-200)';
            
            setTimeout(() => {
                copyBtn.textContent = originalText;
                copyBtn.style.background = '';
            }, 2000);
        }).catch(err => {
            this.showError('Failed to copy content to clipboard');
        });
    }

    async loadQAModels() {
        try {
            const response = await fetch('/models/qa/available');
            const data = await response.json();
            
            this.qaModels = data.qa_models;
            this.qaRecommendations = data.recommendations;
            
            this.populateQAModelSelect();
            this.populateGeneratorModelSelect();
            this.populateGranSabioModelSelect();
            this.bindQAModelEvents();
            this.loadDefaultQAModels();
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    }

    populateQAModelSelect() {
        const select = document.getElementById('qaModelSelect');
        select.innerHTML = '<option value="">Select a model for QA...</option>';
        
        this.qaModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model.key;
            option.textContent = `${model.name} (${model.qa_priority})`;
            option.dataset.provider = model.provider;
            option.dataset.priority = model.qa_priority;
            select.appendChild(option);
        });
    }

    populateGeneratorModelSelect() {
        const select = document.getElementById('generatorModel');
        select.innerHTML = '';
        
        // Group models by priority for better organization
        const groupedModels = {
            recommended: [],
            advanced: [],
            experimental: []
        };
        
        this.qaModels.forEach(model => {
            const modelInfo = {
                key: model.key,
                name: model.name,
                provider: model.provider,
                priority: model.qa_priority,
                description: model.description,
                tokens: model.output_tokens
            };
            
            if (model.qa_priority === 'fast') {
                groupedModels.recommended.push(modelInfo);
            } else if (model.qa_priority === 'standard') {
                groupedModels.advanced.push(modelInfo);
            } else {
                groupedModels.experimental.push(modelInfo);
            }
        });
        
        // Add recommended models
        if (groupedModels.recommended.length > 0) {
            const recommendedGroup = document.createElement('optgroup');
            recommendedGroup.label = 'Recommended (Fast & Efficient)';
            groupedModels.recommended.forEach(model => {
                const option = document.createElement('option');
                option.value = model.key;
                option.textContent = `${model.name} (${model.tokens}k tokens)`;
                recommendedGroup.appendChild(option);
            });
            select.appendChild(recommendedGroup);
        }
        
        // Add advanced models
        if (groupedModels.advanced.length > 0) {
            const advancedGroup = document.createElement('optgroup');
            advancedGroup.label = 'Advanced (Balanced Performance)';
            groupedModels.advanced.forEach(model => {
                const option = document.createElement('option');
                option.value = model.key;
                option.textContent = `${model.name} (${model.tokens}k tokens)`;
                advancedGroup.appendChild(option);
            });
            select.appendChild(advancedGroup);
        }
        
        // Add experimental models  
        if (groupedModels.experimental.length > 0) {
            const experimentalGroup = document.createElement('optgroup');
            experimentalGroup.label = 'Premium (Highest Quality)';
            groupedModels.experimental.forEach(model => {
                const option = document.createElement('option');
                option.value = model.key;
                option.textContent = `${model.name} (${model.tokens}k tokens)`;
                experimentalGroup.appendChild(option);
            });
            select.appendChild(experimentalGroup);
        }
        
        // Set default selection to Gemini 3 Flash
        select.value = 'gemini-3-flash-preview';
        
        // Initialize reasoning controls for the default model
        this.updateReasoningControls(select.value);
    }

    populateGranSabioModelSelect() {
        const select = document.getElementById('granSabioModel');
        select.innerHTML = '';
        
        // Filter and prioritize premium models for Gran Sabio
        const premiumModels = this.qaModels.filter(model => 
            model.qa_priority === 'premium' || 
            model.key.includes('opus') || 
            model.key.includes('gpt-4o')
        );
        
        const standardModels = this.qaModels.filter(model => 
            model.qa_priority === 'standard' && 
            !premiumModels.includes(model)
        );
        
        // Add premium models first
        if (premiumModels.length > 0) {
            const premiumGroup = document.createElement('optgroup');
            premiumGroup.label = 'Premium Models (Recommended for Gran Sabio)';
            premiumModels.forEach(model => {
                const option = document.createElement('option');
                option.value = model.key;
                option.textContent = `${model.name} (${model.output_tokens} tokens)`;
                premiumGroup.appendChild(option);
            });
            select.appendChild(premiumGroup);
        }
        
        // Add standard models
        if (standardModels.length > 0) {
            const standardGroup = document.createElement('optgroup');
            standardGroup.label = 'Standard Models';
            standardModels.forEach(model => {
                const option = document.createElement('option');
                option.value = model.key;
                option.textContent = `${model.name} (${model.output_tokens} tokens)`;
                standardGroup.appendChild(option);
            });
            select.appendChild(standardGroup);
        }
        
        // Set default to Claude Opus 4.5 (best for final decisions)
        select.value = 'claude-opus-4-5-20251101';
    }

    loadDefaultQAModels() {
        // Add some recommended QA models by default
        const defaultModels = [
            'claude-sonnet-4-20250514',
            'gpt-4o',
            'gemini-3-flash-preview'
        ];

        defaultModels.forEach(modelKey => {
            const modelData = this.qaModels.find(m => m.key === modelKey);
            if (modelData) {
                this.createQAModelTag(modelData);
            }
        });
    }

    bindQAModelEvents() {
        // Add model button
        document.getElementById('addQaModelBtn').addEventListener('click', () => {
            this.addQAModel();
        });
        
        // Enter key on select
        document.getElementById('qaModelSelect').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.addQAModel();
            }
        });
    }

    addQAModel() {
        const select = document.getElementById('qaModelSelect');
        const selectedValue = select.value;
        
        if (!selectedValue) return;
        
        // Check if already added
        const existing = document.querySelector(`[data-model="${selectedValue}"]`);
        if (existing) {
            this.showError('Model already added to QA list');
            select.value = '';
            return;
        }
        
        const modelData = this.qaModels.find(m => m.key === selectedValue);
        if (!modelData) return;
        
        this.createQAModelTag(modelData);
        select.value = '';
    }

    createQAModelTag(modelData) {
        const container = document.getElementById('qaSelectedModels');

        const tag = document.createElement('div');
        tag.className = 'qa-model-tag';
        tag.dataset.model = modelData.key;

        const modelName = escapeHtml(modelData.name);
        const modelPriority = escapeHtml(modelData.qa_priority);
        tag.innerHTML = `
            <span class="model-name">${modelName}</span>
            <span class="model-priority">${modelPriority}</span>
            <button type="button" class="remove-model" onclick="removeQaModel(this)">×</button>
        `;

        container.appendChild(tag);
    }

    addRecommendedModels(type) {
        const recommendations = this.qaRecommendations[type] || [];
        const existingModels = Array.from(document.querySelectorAll('[data-model]'))
            .map(el => el.dataset.model);
        
        recommendations.forEach(modelKey => {
            if (!existingModels.includes(modelKey)) {
                const modelData = this.qaModels.find(m => m.key === modelKey);
                if (modelData) {
                    this.createQAModelTag(modelData);
                }
            }
        });
    }

    getSelectedQAModels() {
        return Array.from(document.querySelectorAll('#qaSelectedModels [data-model]'))
            .map(el => el.dataset.model);
    }

    wordsToTokens(words) {
        // Approximate conversion: 1 word ≈ 1.3 tokens on average
        return Math.round(words * 1.3);
    }

    tokensToWords(tokens) {
        // Approximate conversion: 1 token ≈ 0.77 words on average
        return Math.round(tokens * 0.77);
    }

    toggleWordCountOptions(enabled) {
        const optionsDiv = document.getElementById('wordCountOptions');
        optionsDiv.style.display = enabled ? 'block' : 'none';
        
        if (enabled) {
            // Update preview when enabling
            this.updateWordCountPreview();
        }
    }

    updateWordCountPreview() {
        const minWords = parseInt(document.getElementById('minWords')?.value) || 0;
        const maxWords = parseInt(document.getElementById('maxWords')?.value) || 0;
        const flexPercent = parseInt(document.getElementById('flexibilityPercent')?.value) || 15;
        const direction = document.querySelector('input[name="flexibilityDirection"]:checked')?.value || 'both';
        
        const rangeSpan = document.getElementById('acceptableRange');
        
        if (!minWords && !maxWords) {
            rangeSpan.textContent = 'Please set min/max words first';
            return;
        }

        // Calculate range based on the same logic as the server
        let baseMin, baseMax;
        
        if (minWords && maxWords) {
            baseMin = minWords;
            baseMax = maxWords;
        } else if (minWords) {
            baseMin = minWords;
            baseMax = Math.round(minWords * 1.5); // 50% buffer
        } else {
            baseMin = Math.max(1, Math.round(maxWords * 0.7)); // 30% buffer
            baseMax = maxWords;
        }

        const flexFactor = flexPercent / 100;
        let absoluteMin, absoluteMax;

        if (direction === 'both') {
            absoluteMin = Math.max(1, Math.round(baseMin * (1 - flexFactor)));
            absoluteMax = Math.round(baseMax * (1 + flexFactor));
        } else if (direction === 'less') {
            absoluteMin = Math.max(1, Math.round(baseMin * (1 - flexFactor)));
            absoluteMax = baseMax;
        } else if (direction === 'more') {
            absoluteMin = baseMin;
            absoluteMax = Math.round(baseMax * (1 + flexFactor));
        }

        rangeSpan.textContent = `${absoluteMin}-${absoluteMax} words`;
        
        // Also update token estimate
        updateTokenEstimate();
    }

    initializeTabs() {
        // Initialize configuration tab functionality
        const configTabBtns = document.querySelectorAll('.config-tab-btn');
        const configTabPanels = document.querySelectorAll('.config-tab-panel');
        
        console.log('Initializing tabs:', configTabBtns.length, 'buttons,', configTabPanels.length, 'panels');
        
        configTabBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const targetTab = btn.getAttribute('data-tab');
                console.log('Tab clicked:', targetTab);
                
                // Remove active class from all buttons and panels
                configTabBtns.forEach(b => b.classList.remove('active'));
                configTabPanels.forEach(p => p.classList.remove('active'));
                
                // Add active class to clicked button and corresponding panel
                btn.classList.add('active');
                const targetPanel = document.getElementById(targetTab + '-panel');
                if (targetPanel) {
                    targetPanel.classList.add('active');
                    console.log('Panel activated:', targetTab + '-panel');
                } else {
                    console.error('Panel not found:', targetTab + '-panel');
                }
            });
        });
    }

    initializeLogToggle() {
        // Log toggle removed - logs now shown in Live Matrix (/monitor)
    }

    initializeDownloadButton() {
        // Add download functionality
        const downloadBtn = document.getElementById('downloadContent');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => {
                this.downloadGeneratedContent();
            });
        }
    }

    downloadGeneratedContent() {
        const content = document.getElementById('generatedContent').textContent;
        const contentType = document.getElementById('contentType').value;
        const timestamp = new Date().toISOString().slice(0, -5).replace(/:/g, '-');
        const filename = `gransabio-${contentType}-${timestamp}.txt`;
        
        // Create download link
        const blob = new Blob([content], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        // Visual feedback
        const originalText = downloadBtn.textContent;
        downloadBtn.textContent = 'Downloaded!';
        downloadBtn.style.background = 'var(--success-bg)';
        downloadBtn.style.color = 'var(--success)';
        
        setTimeout(() => {
            downloadBtn.textContent = originalText;
            downloadBtn.style.background = '';
            downloadBtn.style.color = '';
        }, 2000);
    }

    initializeVerboseToggle() {
        const verboseCheckbox = document.getElementById('verbose');
        const extraVerboseCheckbox = document.getElementById('extra_verbose');
        
        // Function to update extra_verbose state
        const updateExtraVerboseState = () => {
            if (verboseCheckbox.checked) {
                extraVerboseCheckbox.disabled = false;
                extraVerboseCheckbox.parentElement.style.opacity = '1';
            } else {
                extraVerboseCheckbox.disabled = true;
                extraVerboseCheckbox.checked = false;
                extraVerboseCheckbox.parentElement.style.opacity = '0.5';
            }
        };
        
        // Initialize state
        updateExtraVerboseState();
        
        // Add event listener for verbose checkbox changes
        verboseCheckbox.addEventListener('change', updateExtraVerboseState);
    }

    updateReasoningControls(selectedModel) {
        const reasoningEffortGroup = document.getElementById('reasoningEffortGroup');
        const thinkingBudgetGroup = document.getElementById('thinkingBudgetGroup');
        const reasoningEffortSelect = document.getElementById('reasoningEffort');
        
        // Hide both controls by default
        reasoningEffortGroup.style.display = 'none';
        thinkingBudgetGroup.style.display = 'none';
        
        if (!selectedModel) return;
        
        // Check if the model supports reasoning_effort (GPT-5, O1, O3 models, except O1-mini)
        const supportsReasoningEffort = selectedModel.includes('gpt-5') || 
                                       selectedModel.includes('o3') ||
                                       (selectedModel.includes('o1') && !selectedModel.includes('o1-mini'));
        
        // Check if the model supports thinking_budget_tokens (Claude 3.7/4 models)
        const supportsThinkingBudget = selectedModel.includes('claude-3-7') ||
                                      selectedModel.includes('claude-opus-4') ||
                                      selectedModel.includes('claude-sonnet-4');
        
        if (supportsReasoningEffort) {
            reasoningEffortGroup.style.display = 'block';
            
            // Update reasoning effort options based on model type
            if (selectedModel.includes('gpt-5')) {
                // GPT-5 supports none, low, medium, high
                reasoningEffortSelect.innerHTML = `
                    <option value="none">None (Fastest)</option>
                    <option value="low" selected>Low (Quick)</option>
                    <option value="medium">Medium (Balanced)</option>
                    <option value="high">High (Deep thinking)</option>
                `;
            } else if (selectedModel.includes('o3') || (selectedModel.includes('o1') && !selectedModel.includes('o1-mini'))) {
                // O3 and O1 models support low, medium, high (no minimal)
                reasoningEffortSelect.innerHTML = `
                    <option value="low">Low</option>
                    <option value="medium" selected>Medium (Balanced)</option>
                    <option value="high">High (Deep thinking)</option>
                `;
            }
        }
        
        if (supportsThinkingBudget) {
            thinkingBudgetGroup.style.display = 'block';
        }
        
        // Trigger word count range update since layout changed
        this.updateWordCountPreview();
    }
}

// Global functions for HTML onclick events
function removeQaModel(button) {
    button.closest('.qa-model-tag').remove();
}

function addRecommendedModels(type) {
    if (window.gransabio && window.gransabio.addRecommendedModels) {
        window.gransabio.addRecommendedModels(type);
    }
}

function updateTokenEstimate() {
    const minWordsInput = document.getElementById('minWords');
    const maxWordsInput = document.getElementById('maxWords');
    const tokenEstimate = document.getElementById('tokenEstimate');
    
    if (tokenEstimate) {
        const minWords = parseInt(minWordsInput?.value) || 0;
        const maxWords = parseInt(maxWordsInput?.value) || 0;
        
        let estimateText = '';
        
        if (minWords && maxWords) {
            const minTokens = Math.round(minWords * 1.3);
            const maxTokens = Math.round(maxWords * 1.3);
            estimateText = `≈ ${minTokens}-${maxTokens} tokens`;
        } else if (maxWords) {
            const maxTokens = Math.round(maxWords * 1.3);
            estimateText = `≈ ${maxTokens} tokens max`;
        } else if (minWords) {
            const minTokens = Math.round(minWords * 1.3);
            const bufferTokens = Math.round(minTokens * 1.5);
            estimateText = `≈ ${minTokens}-${bufferTokens} tokens (min + buffer)`;
        } else {
            estimateText = 'No word limits set';
        }
        
        tokenEstimate.textContent = estimateText;
    }
}

// Initialize the interface when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.gransabio = new GranSabioInterface();
});
