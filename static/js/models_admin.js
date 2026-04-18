(() => {
  /* ═══════════════════════════════════════════════════
   * Constants
   * ═══════════════════════════════════════════════════ */
  const TABS = ["catalog", "openai", "anthropic", "google", "xai", "openrouter"];
  const PROVIDER_TABS = ["openai", "anthropic", "google", "xai", "openrouter"];
  const PROVIDER_LABELS = {
    catalog: "Catalog",
    openai: "OpenAI",
    anthropic: "Anthropic",
    google: "Google",
    xai: "xAI",
    openrouter: "OpenRouter",
  };
  const FULL_SYNC_PROVIDERS = new Set(["openrouter", "xai"]);

  /* ═══════════════════════════════════════════════════
   * Utility functions (kept from previous version)
   * ═══════════════════════════════════════════════════ */

  function escapeHtml(value) {
    if (value === null || value === undefined) return "";
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function toArray(value) {
    if (Array.isArray(value)) return value;
    if (value === null || value === undefined || value === "") return [];
    return [value];
  }

  function formatNumber(value) {
    const num = Number(value);
    if (!Number.isFinite(num) || num === 0) return "-";
    if (Math.abs(num) >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (Math.abs(num) >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return `${num}`;
  }

  function formatMoney(value) {
    const num = Number(value);
    if (!Number.isFinite(num)) return "-";
    return `$${num.toFixed(4)}`;
  }

  function formatDate(value) {
    if (!value) return "-";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  function normalizePricing(raw = {}) {
    const input =
      raw.input_per_million ??
      raw.input ??
      raw.prompt ??
      raw.prompt_per_million ??
      raw.prompt_price ??
      raw.input_price ??
      0;
    const output =
      raw.output_per_million ??
      raw.output ??
      raw.completion ??
      raw.completion_per_million ??
      raw.completion_price ??
      raw.output_price ??
      0;
    return {
      input_per_million: Number(input) || 0,
      output_per_million: Number(output) || 0,
    };
  }

  function normalizeCapabilities(raw) {
    if (Array.isArray(raw)) return raw.filter(Boolean).map((item) => String(item));
    if (typeof raw === "string" && raw.trim()) return [raw.trim()];
    return [];
  }

  function normalizeModel(entry = {}, providerHint = "catalog") {
    const modelId = String(
      entry.model_id ??
        entry.id ??
        entry.key ??
        entry.model ??
        entry.slug ??
        ""
    );
    const provider = entry.provider || providerHint;
    const pricing = normalizePricing(entry.pricing || entry.costs || entry.price || {});
    const capabilities = normalizeCapabilities(
      entry.capabilities || entry.features || entry.modalities || entry.tags
    );
    const contextWindow =
      Number(
        entry.context_window ??
          entry.contextLength ??
          entry.context_length ??
          entry.input_tokens ??
          entry.max_context_tokens ??
          entry.max_context ??
          0
      ) || 0;
    const inputTokens =
      Number(entry.input_tokens ?? entry.context_length ?? entry.context_window ?? contextWindow) ||
      contextWindow;
    const outputTokens =
      Number(entry.output_tokens ?? entry.max_output_tokens ?? entry.max_completion_tokens ?? 0) || 0;
    const name = entry.name || entry.display_name || entry.title || modelId;
    const status = entry.sync_state || entry.state || entry.status || "";
    const source = entry.source || entry.url || "";
    const enabled = entry.enabled !== undefined ? Boolean(entry.enabled) : true;

    return {
      id: modelId,
      provider,
      name,
      description: entry.description || "",
      input_tokens: inputTokens,
      output_tokens: outputTokens,
      context_window: contextWindow,
      pricing,
      capabilities,
      verified_at: entry.verified_at || entry.updated_at || entry.created_at || "",
      created_at: entry.created_at || entry.remote_created_at || "",
      source,
      status,
      enabled,
      raw: entry,
    };
  }

  function flattenModelCollection(value, providerHint = "catalog") {
    if (!value) return [];
    if (Array.isArray(value)) {
      return value.map((item) => normalizeModel(item, providerHint));
    }
    if (typeof value === "object") {
      const rows = [];
      for (const [key, item] of Object.entries(value)) {
        if (item && typeof item === "object" && !Array.isArray(item)) {
          rows.push(normalizeModel({ ...item, model_id: item.model_id || key }, providerHint));
        }
      }
      return rows;
    }
    return [];
  }

  function parseCatalogResponse(payload) {
    const summary = payload?.summary || payload?.stats || {};
    let models = [];

    if (Array.isArray(payload?.models)) {
      models = payload.models.map((item) => normalizeModel(item, item.provider || "catalog"));
    } else if (Array.isArray(payload?.data)) {
      models = payload.data.map((item) => normalizeModel(item, item.provider || "catalog"));
    } else if (payload?.catalog && typeof payload.catalog === "object") {
      for (const [provider, value] of Object.entries(payload.catalog)) {
        models.push(...flattenModelCollection(value, provider));
      }
    } else if (payload?.providers && typeof payload.providers === "object") {
      for (const [provider, value] of Object.entries(payload.providers)) {
        models.push(...flattenModelCollection(value, provider));
      }
    } else if (payload?.model_specifications && typeof payload.model_specifications === "object") {
      for (const [provider, value] of Object.entries(payload.model_specifications)) {
        models.push(...flattenModelCollection(value, provider));
      }
    } else if (payload && typeof payload === "object") {
      for (const provider of PROVIDER_TABS) {
        if (payload[provider]) {
          models.push(...flattenModelCollection(payload[provider], provider));
        }
      }
    }

    models = models.filter((model) => model.id);
    models.sort((a, b) => {
      const providerA = (a.provider || "").toLowerCase();
      const providerB = (b.provider || "").toLowerCase();
      if (providerA !== providerB) return providerA.localeCompare(providerB);
      return a.name.localeCompare(b.name);
    });

    return { summary, models, raw: payload || {} };
  }

  function parseRemoteResponse(provider, payload) {
    let models = [];
    if (Array.isArray(payload?.models)) {
      models = payload.models.map((item) => normalizeModel(item, provider));
    } else if (Array.isArray(payload?.data)) {
      models = payload.data.map((item) => normalizeModel(item, provider));
    } else if (Array.isArray(payload?.items)) {
      models = payload.items.map((item) => normalizeModel(item, provider));
    } else if (payload && typeof payload === "object") {
      const directListKeys = ["models", "data", "items"];
      for (const key of directListKeys) {
        if (Array.isArray(payload[key])) {
          models = payload[key].map((item) => normalizeModel(item, provider));
          break;
        }
      }
      if (models.length === 0) {
        for (const [key, value] of Object.entries(payload)) {
          if (key === "summary" || key === "stats" || key === "meta") continue;
          if (Array.isArray(value)) {
            models.push(...value.map((item) => normalizeModel(item, provider)));
          }
        }
      }
    }

    models = models.filter((model) => model.id);
    models.sort((a, b) => a.name.localeCompare(b.name));
    return {
      provider,
      models,
      summary: payload?.summary || payload?.stats || payload?.meta || {},
      raw: payload || {},
    };
  }

  function fingerprint(model) {
    const payload = {
      name: model.name || "",
      description: model.description || "",
      context_window: model.context_window || 0,
      input_tokens: model.input_tokens || 0,
      output_tokens: model.output_tokens || 0,
      pricing: model.pricing || {},
      capabilities: [...(model.capabilities || [])].sort(),
      source: model.source || "",
    };
    return JSON.stringify(payload);
  }

  function modelKey(model) {
    return model?.id || "";
  }

  /* ═══════════════════════════════════════════════════
   * ModelsAdminApp
   * ═══════════════════════════════════════════════════ */

  class ModelsAdminApp {
    constructor() {
      this.catalog = { models: [], summary: {} };
      this.catalogIndex = new Map();
      this.remote = new Map();
      this.activeTab = "catalog";
      this.loading = new Set();
      this.abortControllers = new Map();
      this.sortState = this._loadSortState();
      this.providerSelections = new Map();
      this._deletePopover = null;
      this.init();
    }

    /* ─── Bootstrap ─────────────────────────── */

    init() {
      this.bindTabs();
      this.bindCatalogControls();
      this.bindProviderControls();
      this.bindModals();

      const boot = window.ADMIN_BOOT || {};
      const savedTab = localStorage.getItem("gransabio_admin_tab");
      const initialTab = savedTab && TABS.includes(savedTab) ? savedTab : boot.initialTab || "catalog";

      this.refreshCatalog();
      this.activateTab(TABS.includes(initialTab) ? initialTab : "catalog");
    }

    /* ─── Sort persistence ──────────────────── */

    _loadSortState() {
      try {
        const raw = localStorage.getItem("gransabio_admin_sort");
        if (raw) {
          const parsed = JSON.parse(raw);
          if (parsed && parsed.key && parsed.dir) return parsed;
        }
      } catch { /* ignore */ }
      return { key: "provider", dir: "asc" };
    }

    _saveSortState() {
      localStorage.setItem("gransabio_admin_sort", JSON.stringify(this.sortState));
    }

    /* ─── Tab binding ───────────────────────── */

    bindTabs() {
      const strip = document.getElementById("tabStrip");
      if (!strip) return;
      strip.addEventListener("click", (e) => {
        const btn = e.target.closest("[data-tab]");
        if (!btn) return;
        this.activateTab(btn.dataset.tab);
      });
    }

    /* ─── Catalog controls ──────────────────── */

    bindCatalogControls() {
      const search = document.getElementById("catalogSearch");
      if (search) {
        search.addEventListener("input", () => this.renderCatalog());
      }
      const providerFilter = document.getElementById("catalogProviderFilter");
      if (providerFilter) {
        providerFilter.addEventListener("change", () => this.renderCatalog());
      }
      const statusFilter = document.getElementById("catalogStatusFilter");
      if (statusFilter) {
        statusFilter.addEventListener("change", () => this.renderCatalog());
      }

      // Sort headers — use event delegation on the table
      const catalogPanel = document.getElementById("catalogPanel");
      if (catalogPanel) {
        catalogPanel.addEventListener("click", (e) => {
          const th = e.target.closest("th.sortable");
          if (th && th.dataset.sort) {
            this.handleSort(th.dataset.sort);
          }
        });
      }

      // Sync All button
      const syncAllBtn = document.getElementById("syncAllBtn");
      if (syncAllBtn) {
        syncAllBtn.addEventListener("click", () => this.handleSyncAll());
      }

      // Event delegation for catalog body (toggles + deletes)
      const catalogBody = document.getElementById("catalogBody");
      if (catalogBody) {
        catalogBody.addEventListener("change", (e) => {
          const toggle = e.target.closest(".toggle input[type='checkbox']");
          if (!toggle) return;
          const tr = toggle.closest("tr");
          if (!tr) return;
          const provider = tr.dataset.provider;
          const modelId = tr.dataset.modelId;
          if (provider && modelId) {
            this.handleToggle(provider, modelId, toggle.checked);
          }
        });
        catalogBody.addEventListener("click", (e) => {
          const deleteBtn = e.target.closest(".icon-btn[data-action='delete']");
          if (!deleteBtn) return;
          const tr = deleteBtn.closest("tr");
          if (!tr) return;
          const provider = tr.dataset.provider;
          const modelId = tr.dataset.modelId;
          if (provider && modelId) {
            this.handleDelete(provider, modelId, deleteBtn);
          }
        });
      }
    }

    /* ─── Provider controls ─────────────────── */

    bindProviderControls() {
      const refreshBtn = document.getElementById("refreshProviderBtn");
      if (refreshBtn) {
        refreshBtn.addEventListener("click", () => {
          if (this.activeTab !== "catalog") {
            this.loadRemote(this.activeTab, true);
          }
        });
      }

      const applyBtn = document.getElementById("applyChangesBtn");
      if (applyBtn) {
        applyBtn.addEventListener("click", () => {
          if (this.activeTab !== "catalog") {
            this.showConfirmModal(this.activeTab);
          }
        });
      }

      // Event delegation for provider sections (checkboxes, select/deselect all, collapse toggle)
      const sections = document.getElementById("providerSections");
      if (sections) {
        sections.addEventListener("click", (e) => {
          // Section collapse toggle
          const header = e.target.closest(".section-header");
          if (header && !e.target.closest(".section-actions") && !e.target.closest("input")) {
            const section = header.closest(".section");
            if (section) section.classList.toggle("open");
            return;
          }
          // Select all / Deselect all buttons
          const actionBtn = e.target.closest("[data-select-action]");
          if (actionBtn) {
            const section = actionBtn.closest(".section");
            const sectionType = section?.dataset.sectionType;
            const selectAll = actionBtn.dataset.selectAction === "all";
            if (sectionType) {
              this._handleSectionSelectAll(sectionType, selectAll);
            }
            return;
          }
        });
        sections.addEventListener("change", (e) => {
          const cb = e.target.closest("input[type='checkbox'][data-model-id]");
          if (!cb) return;
          const sectionType = cb.closest(".section")?.dataset.sectionType;
          const modelId = cb.dataset.modelId;
          if (sectionType && modelId) {
            this._handleProviderCheckbox(sectionType, modelId, cb.checked);
          }
        });
      }
    }

    /* ─── Modal binding ─────────────────────── */

    bindModals() {
      // Confirm modal
      const confirmCancel = document.getElementById("confirmCancel");
      if (confirmCancel) {
        confirmCancel.addEventListener("click", () => this._hideModal("confirmModal"));
      }
      const confirmApply = document.getElementById("confirmApply");
      if (confirmApply) {
        confirmApply.addEventListener("click", () => {
          this.handleApplyChanges(this.activeTab);
        });
      }

      // Sync-all modal
      const syncAllCancel = document.getElementById("syncAllCancel");
      if (syncAllCancel) {
        syncAllCancel.addEventListener("click", () => this._hideModal("syncAllModal"));
      }
      const syncAllApply = document.getElementById("syncAllApply");
      if (syncAllApply) {
        syncAllApply.addEventListener("click", () => this.handleSyncAllApply());
      }

      // Close modals on overlay click
      document.querySelectorAll(".modal-overlay").forEach((overlay) => {
        overlay.addEventListener("click", (e) => {
          if (e.target === overlay) this._hideModal(overlay.id);
        });
      });

      // Close delete popover on outside click
      document.addEventListener("click", (e) => {
        if (this._deletePopover && !this._deletePopover.contains(e.target)) {
          this._removeDeletePopover();
        }
      });
    }

    /* ─── Tab activation ────────────────────── */

    activateTab(tab) {
      if (!TABS.includes(tab)) return;
      this.activeTab = tab;
      localStorage.setItem("gransabio_admin_tab", tab);

      // Toggle active class on tab buttons
      document.querySelectorAll("#tabStrip [data-tab]").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.tab === tab);
      });

      // Show correct panel
      const catalogPanel = document.getElementById("catalogPanel");
      const providerPanel = document.getElementById("providerPanel");
      if (catalogPanel) catalogPanel.classList.toggle("active", tab === "catalog");
      if (providerPanel) providerPanel.classList.toggle("active", tab !== "catalog");

      if (tab === "catalog") {
        this.renderCatalog();
      } else {
        // Set provider title and meta
        const titleEl = document.getElementById("providerTitle");
        const metaEl = document.getElementById("providerMeta");
        if (titleEl) titleEl.textContent = PROVIDER_LABELS[tab] || tab;
        if (metaEl) {
          metaEl.textContent = FULL_SYNC_PROVIDERS.has(tab)
            ? "Full sync - replaces provider section in catalog"
            : "Discovery sync - adds new models to catalog";
        }
        this.loadRemote(tab);
      }
    }

    /* ─── Catalog refresh ───────────────────── */

    async refreshCatalog() {
      this.loading.add("catalog");
      this.showNotice("globalNotice", "loading", "Loading catalog...", "");
      try {
        const data = await this.fetchJson("/api/admin/models/catalog");
        this.catalog = parseCatalogResponse(data);
        this.catalogIndex = new Map(this.catalog.models.map((m) => [modelKey(m), m]));
        this._updateTabBadges();
        this._updateTitleBarCount();
        this.hideNotice("globalNotice");
        if (this.activeTab === "catalog") {
          this.renderCatalog();
        }
      } catch (err) {
        console.error("refreshCatalog failed:", err);
        this.showNotice("globalNotice", "error", "Failed to load catalog", err.message);
      } finally {
        this.loading.delete("catalog");
      }
    }

    _updateTabBadges() {
      // Count per provider
      const counts = {};
      for (const m of this.catalog.models) {
        counts[m.provider] = (counts[m.provider] || 0) + 1;
      }
      const totalBadge = document.getElementById("badge-catalog");
      if (totalBadge) totalBadge.textContent = this.catalog.models.length;

      for (const p of PROVIDER_TABS) {
        const badge = document.getElementById(`badge-${p}`);
        if (badge) badge.textContent = counts[p] || 0;
      }
    }

    _updateTitleBarCount() {
      const el = document.getElementById("modelCount");
      if (el) el.textContent = `${this.catalog.models.length} models`;
    }

    /* ─── Catalog rendering ─────────────────── */

    renderCatalog() {
      const tbody = document.getElementById("catalogBody");
      const emptyEl = document.getElementById("catalogEmpty");
      if (!tbody) return;

      // Update provider filter options
      this._updateProviderFilterOptions();

      const filtered = this._getFilteredCatalogModels();
      const sorted = this.sortModels(filtered);

      if (sorted.length === 0) {
        tbody.innerHTML = "";
        if (emptyEl) emptyEl.style.display = "block";
        return;
      }
      if (emptyEl) emptyEl.style.display = "none";

      tbody.innerHTML = sorted.map((model) => {
        const isEnabled = model.enabled !== false;
        const disabledClass = isEnabled ? "" : " disabled";
        const checked = isEnabled ? "checked" : "";
        const providerClass = escapeHtml(model.provider);
        const inputPrice = model.pricing?.input_per_million;
        const outputPrice = model.pricing?.output_per_million;

        return `<tr class="${disabledClass}" data-provider="${escapeHtml(model.provider)}" data-model-id="${escapeHtml(model.id)}">
          <td>
            <label class="toggle"><input type="checkbox" ${checked}><span class="slider"></span></label>
          </td>
          <td><span class="provider-pill ${providerClass}">${escapeHtml(model.provider)}</span></td>
          <td>
            <span class="model-name">${escapeHtml(model.name)}</span>
            <span class="model-id">${escapeHtml(model.id)}</span>
          </td>
          <td class="ctx">${formatNumber(model.context_window || model.input_tokens)}</td>
          <td class="price">${formatMoney(inputPrice)}</td>
          <td class="price">${formatMoney(outputPrice)}</td>
          <td style="position:relative">
            <div class="row-actions">
              <button type="button" class="icon-btn" data-action="delete" title="Delete model">&#x2715;</button>
            </div>
          </td>
        </tr>`;
      }).join("");

      this._updateSortIcons();
    }

    _updateProviderFilterOptions() {
      const select = document.getElementById("catalogProviderFilter");
      if (!select) return;
      const providerSet = new Set(this.catalog.models.map((m) => m.provider));
      const currentValue = select.value;
      select.innerHTML = '<option value="">All providers</option>';
      [...providerSet].sort().forEach((p) => {
        select.innerHTML += `<option value="${escapeHtml(p)}">${escapeHtml(PROVIDER_LABELS[p] || p)}</option>`;
      });
      if ([...providerSet].includes(currentValue)) {
        select.value = currentValue;
      }
    }

    _getFilteredCatalogModels() {
      const searchEl = document.getElementById("catalogSearch");
      const providerEl = document.getElementById("catalogProviderFilter");
      const statusEl = document.getElementById("catalogStatusFilter");
      const search = (searchEl?.value || "").trim().toLowerCase();
      const providerFilter = providerEl?.value || "";
      const statusFilter = statusEl?.value || "";

      return this.catalog.models.filter((model) => {
        if (providerFilter && model.provider !== providerFilter) return false;
        if (search) {
          const haystack = `${model.id} ${model.name} ${model.description} ${model.provider}`.toLowerCase();
          if (!haystack.includes(search)) return false;
        }
        if (statusFilter === "enabled" && model.enabled === false) return false;
        if (statusFilter === "disabled" && model.enabled !== false) return false;
        return true;
      });
    }

    /* ─── Sorting ───────────────────────────── */

    handleSort(key) {
      if (this.sortState.key === key) {
        // Cycle: asc -> desc -> none
        if (this.sortState.dir === "asc") {
          this.sortState.dir = "desc";
        } else if (this.sortState.dir === "desc") {
          this.sortState = { key, dir: "none" };
        } else {
          this.sortState = { key, dir: "asc" };
        }
      } else {
        this.sortState = { key, dir: "asc" };
      }
      this._saveSortState();
      this.renderCatalog();
    }

    sortModels(models) {
      if (this.sortState.dir === "none") return [...models];
      const { key, dir } = this.sortState;
      const mult = dir === "desc" ? -1 : 1;

      return [...models].sort((a, b) => {
        let va, vb;
        switch (key) {
          case "provider":
            va = (a.provider || "").toLowerCase();
            vb = (b.provider || "").toLowerCase();
            return mult * va.localeCompare(vb) || a.name.localeCompare(b.name);
          case "name":
            va = (a.name || "").toLowerCase();
            vb = (b.name || "").toLowerCase();
            return mult * va.localeCompare(vb);
          case "context":
            va = a.context_window || a.input_tokens || 0;
            vb = b.context_window || b.input_tokens || 0;
            return mult * (va - vb);
          case "input_price":
            va = a.pricing?.input_per_million || 0;
            vb = b.pricing?.input_per_million || 0;
            return mult * (va - vb);
          case "output_price":
            va = a.pricing?.output_per_million || 0;
            vb = b.pricing?.output_per_million || 0;
            return mult * (va - vb);
          default:
            return 0;
        }
      });
    }

    _updateSortIcons() {
      document.querySelectorAll("#catalogPanel th.sortable").forEach((th) => {
        const icon = th.querySelector(".sort-icon");
        if (!icon) return;
        const key = th.dataset.sort;
        if (key === this.sortState.key && this.sortState.dir !== "none") {
          icon.textContent = this.sortState.dir === "asc" ? "\u25B2" : "\u25BC";
        } else {
          icon.textContent = "";
        }
      });
    }

    /* ─── Toggle enable/disable ─────────────── */

    async handleToggle(provider, modelId, enabled) {
      const encodedId = encodeURIComponent(modelId);
      try {
        await this.fetchJson(
          `/api/admin/models/catalog/${encodeURIComponent(provider)}?model_id=${encodedId}&enabled=${enabled}`,
          { method: "PATCH" }
        );
        // Update local state
        const model = this.catalogIndex.get(modelId);
        if (model) {
          model.enabled = enabled;
        }
        this.renderCatalog();
      } catch (err) {
        this.showNotice("globalNotice", "error", "Toggle failed", err.message);
        // Revert: re-render will restore the correct checkbox state
        this.renderCatalog();
      }
    }

    /* ─── Delete model ──────────────────────── */

    handleDelete(provider, modelId, buttonEl) {
      this._removeDeletePopover();

      const popover = document.createElement("div");
      popover.className = "delete-popover";
      popover.innerHTML = `
        <p>Delete <strong>${escapeHtml(modelId)}</strong> from <strong>${escapeHtml(provider)}</strong>?</p>
        <div class="actions">
          <button type="button" class="btn-secondary btn-sm" data-action="cancel-delete">Cancel</button>
          <button type="button" class="btn-danger btn-sm" data-action="confirm-delete">Delete</button>
        </div>
      `;

      // Position near the button
      const td = buttonEl.closest("td");
      if (td) {
        td.style.position = "relative";
        td.appendChild(popover);
      } else {
        document.body.appendChild(popover);
      }

      // Defer popover registration so the current click event finishes
      // before the document-level outside-click handler can act on it
      requestAnimationFrame(() => {
        this._deletePopover = popover;
      });

      popover.addEventListener("click", async (e) => {
        e.stopPropagation();
        const action = e.target.closest("[data-action]")?.dataset.action;
        if (action === "cancel-delete") {
          this._removeDeletePopover();
        } else if (action === "confirm-delete") {
          const confirmBtn = e.target.closest("[data-action='confirm-delete']");
          if (confirmBtn) confirmBtn.disabled = true;
          try {
            const encodedId = encodeURIComponent(modelId);
            await this.fetchJson(
              `/api/admin/models/catalog/${encodeURIComponent(provider)}?model_id=${encodedId}`,
              { method: "DELETE" }
            );
            this._removeDeletePopover();
            // Remove from local catalog
            this.catalog.models = this.catalog.models.filter((m) => !(m.id === modelId && m.provider === provider));
            this.catalogIndex.delete(modelId);
            this._updateTabBadges();
            this._updateTitleBarCount();
            this.renderCatalog();
            this.showNotice("globalNotice", "success", "Model deleted", `${modelId} removed from ${provider}.`);
          } catch (err) {
            this._removeDeletePopover();
            this.showNotice("globalNotice", "error", "Delete failed", err.message);
          }
        }
      });
    }

    _removeDeletePopover() {
      if (this._deletePopover) {
        this._deletePopover.remove();
        this._deletePopover = null;
      }
    }

    /* ─── Remote provider loading ───────────── */

    async loadRemote(provider, force = false) {
      if (provider === "catalog") return;
      if (!force && this.remote.has(provider)) {
        this.renderProvider();
        return;
      }

      // Cancel in-flight request for this provider
      if (this.abortControllers.has(provider)) {
        this.abortControllers.get(provider).abort();
      }
      const controller = new AbortController();
      this.abortControllers.set(provider, controller);

      this.loading.add(`remote:${provider}`);
      this.showNotice("providerNotice", "loading", `Loading ${PROVIDER_LABELS[provider] || provider}...`, "Fetching models from provider API.");

      try {
        const data = await this.fetchJson(
          `/api/admin/models/providers/${encodeURIComponent(provider)}/remote`,
          { signal: controller.signal }
        );
        const parsed = parseRemoteResponse(provider, data);
        this.remote.set(provider, parsed);
        this.initSelections(provider);
        this.hideNotice("providerNotice");
        this.renderProvider();
      } catch (err) {
        if (err.name === "AbortError") return;
        this.showNotice("providerNotice", "error", `Failed to load ${PROVIDER_LABELS[provider] || provider}`, err.message);
      } finally {
        this.loading.delete(`remote:${provider}`);
        this.abortControllers.delete(provider);
      }
    }

    /* ─── Selection management ──────────────── */

    initSelections(provider) {
      const { newModels, updatedModels, missingModels } = this.classifyModels(provider);

      const selections = {
        new: new Set(newModels.map((m) => m.id)),           // all new checked by default
        updated: new Set(updatedModels.map((m) => m.remote.id)),   // all updated checked by default
        missing: new Set(),                                  // none checked = keep all
      };
      this.providerSelections.set(provider, selections);
    }

    classifyModels(provider) {
      const remoteData = this.remote.get(provider);
      if (!remoteData) return { newModels: [], updatedModels: [], currentModels: [], missingModels: [] };

      const remoteModels = remoteData.models;
      const remoteIdSet = new Set(remoteModels.map((m) => m.id));

      // Local models for this provider
      const localModels = this.catalog.models.filter((m) => m.provider === provider);

      const newModels = [];
      const updatedModels = [];
      const currentModels = [];

      for (const rm of remoteModels) {
        const local = this.catalogIndex.get(rm.id);
        if (!local || local.provider !== provider) {
          newModels.push(rm);
        } else if (fingerprint(rm) !== fingerprint(local)) {
          updatedModels.push({ remote: rm, local });
        } else {
          currentModels.push(rm);
        }
      }

      // Missing: local models for this provider that are NOT in remote
      // Only relevant for FULL_SYNC_PROVIDERS
      let missingModels = [];
      if (FULL_SYNC_PROVIDERS.has(provider)) {
        missingModels = localModels.filter((m) => !remoteIdSet.has(m.id));
      }

      return { newModels, updatedModels, currentModels, missingModels };
    }

    _handleProviderCheckbox(sectionType, modelId, checked) {
      const provider = this.activeTab;
      const selections = this.providerSelections.get(provider);
      if (!selections) return;

      const set = selections[sectionType];
      if (!set) return;

      if (checked) {
        set.add(modelId);
      } else {
        set.delete(modelId);
      }
      this.updateActionButton();
    }

    _handleSectionSelectAll(sectionType, selectAll) {
      const provider = this.activeTab;
      const selections = this.providerSelections.get(provider);
      if (!selections) return;

      const { newModels, updatedModels, missingModels } = this.classifyModels(provider);

      let models;
      switch (sectionType) {
        case "new":
          models = newModels;
          break;
        case "updated":
          models = updatedModels.map((m) => m.remote);
          break;
        case "missing":
          models = missingModels;
          break;
        default:
          return;
      }

      const set = selections[sectionType];
      if (selectAll) {
        models.forEach((m) => set.add(m.id));
      } else {
        set.clear();
      }

      this.renderProvider();
    }

    /* ─── Provider rendering ────────────────── */

    renderProvider() {
      const provider = this.activeTab;
      if (provider === "catalog") return;

      const sectionsEl = document.getElementById("providerSections");
      const actionBar = document.getElementById("providerActionBar");
      if (!sectionsEl) { console.error("providerSections element not found"); return; }

      if (!this.remote.has(provider)) {
        sectionsEl.innerHTML = "";
        if (actionBar) actionBar.style.display = "none";
        console.log("renderProvider: no remote data for", provider);
        return;
      }

      const { newModels, updatedModels, currentModels, missingModels } = this.classifyModels(provider);
      console.log("renderProvider:", provider, "new:", newModels.length, "updated:", updatedModels.length, "current:", currentModels.length, "missing:", missingModels.length);
      const selections = this.providerSelections.get(provider) || { new: new Set(), updated: new Set(), missing: new Set() };

      let html = "";

      // New models section
      if (newModels.length > 0) {
        html += this.renderSection("new", "New models", `${newModels.length} models found in API but not in your catalog.`, newModels, {
          expanded: true,
          showCheckboxes: true,
          selectedSet: selections.new,
        });
      }

      // Updated models section
      if (updatedModels.length > 0) {
        html += this.renderSection("updated", "Updated models", `${updatedModels.length} models with changes from the API.`, updatedModels.map((m) => m.remote), {
          expanded: true,
          showCheckboxes: true,
          showDiffs: true,
          diffPairs: updatedModels,
          selectedSet: selections.updated,
        });
      }

      // Up to date section
      if (currentModels.length > 0) {
        html += this.renderSection("current", "Up to date", `${currentModels.length} models match the API.`, currentModels, {
          expanded: false,
          showCheckboxes: false,
        });
      }

      // Not in API section (only for FULL_SYNC providers)
      if (FULL_SYNC_PROVIDERS.has(provider) && missingModels.length > 0) {
        html += this.renderSection("missing", "Not in API", `${missingModels.length} models in catalog but not found in remote API. Check = remove from catalog.`, missingModels, {
          expanded: false,
          showCheckboxes: true,
          selectedSet: selections.missing,
          checkboxMeaning: "remove",
        });
      }

      if (!html) {
        html = '<div class="empty-state">No remote models loaded for this provider.</div>';
      }

      sectionsEl.innerHTML = html;

      // Show/hide action bar
      if (actionBar) {
        actionBar.style.display = (newModels.length > 0 || updatedModels.length > 0 || missingModels.length > 0) ? "flex" : "none";
      }

      this.updateActionButton();
    }

    renderSection(type, title, description, models, options = {}) {
      const { expanded, showCheckboxes, showDiffs, diffPairs, selectedSet, checkboxMeaning } = options;
      const openClass = expanded ? " open" : "";

      let headerActions = "";
      if (showCheckboxes) {
        headerActions = `
          <div class="section-actions">
            <button type="button" class="btn-secondary btn-sm" data-select-action="all">Select all</button>
            <button type="button" class="btn-secondary btn-sm" data-select-action="none">Deselect all</button>
          </div>
        `;
      }

      let bodyHtml = "";
      models.forEach((model, idx) => {
        const isChecked = selectedSet ? selectedSet.has(model.id) : false;
        const checkboxHtml = showCheckboxes
          ? `<input type="checkbox" data-model-id="${escapeHtml(model.id)}" ${isChecked ? "checked" : ""}>`
          : "";

        let diffHtml = "";
        if (showDiffs && diffPairs && diffPairs[idx]) {
          diffHtml = this.renderDiff(diffPairs[idx].remote, diffPairs[idx].local);
        }

        const ctxStr = model.context_window ? formatNumber(model.context_window) : "-";
        const inputStr = formatMoney(model.pricing?.input_per_million);
        const outputStr = formatMoney(model.pricing?.output_per_million);

        bodyHtml += `
          <div class="section-row">
            ${checkboxHtml}
            <div class="info">
              <div class="model-name">${escapeHtml(model.name)}</div>
              <div class="model-meta">${escapeHtml(model.id)} | ctx: ${ctxStr} | in: ${inputStr} | out: ${outputStr}</div>
              ${diffHtml}
            </div>
          </div>
        `;
      });

      return `
        <div class="section ${escapeHtml(type)}${openClass}" data-section-type="${escapeHtml(type)}">
          <div class="section-header">
            <div class="left">
              <span class="chevron">&#9654;</span>
              <h3 class="section-title">${escapeHtml(title)}</h3>
              <span class="count">${models.length}</span>
            </div>
            ${headerActions}
          </div>
          <div class="section-body">
            <p class="section-desc">${escapeHtml(description)}</p>
            ${bodyHtml}
          </div>
        </div>
      `;
    }

    renderDiff(remoteModel, localModel) {
      const fields = [
        { key: "context_window", label: "Context" },
        { key: "input_tokens", label: "Input tokens" },
        { key: "output_tokens", label: "Output tokens" },
        { key: "name", label: "Name" },
        { key: "description", label: "Description" },
        { key: "source", label: "Source" },
      ];
      const pricingFields = [
        { key: "input_per_million", label: "Input $/M" },
        { key: "output_per_million", label: "Output $/M" },
      ];

      let diffs = "";

      for (const f of fields) {
        const oldVal = localModel[f.key];
        const newVal = remoteModel[f.key];
        if (oldVal !== newVal && (oldVal || newVal)) {
          const oldStr = f.key.includes("token") || f.key === "context_window" ? formatNumber(oldVal) : escapeHtml(String(oldVal || "-"));
          const newStr = f.key.includes("token") || f.key === "context_window" ? formatNumber(newVal) : escapeHtml(String(newVal || "-"));
          diffs += `<div class="diff-field">${escapeHtml(f.label)}: <span class="diff-old">${oldStr}</span> -> <span class="diff-new">${newStr}</span></div>`;
        }
      }

      for (const f of pricingFields) {
        const oldVal = localModel.pricing?.[f.key];
        const newVal = remoteModel.pricing?.[f.key];
        if (oldVal !== newVal) {
          diffs += `<div class="diff-field">${escapeHtml(f.label)}: <span class="diff-old">${formatMoney(oldVal)}</span> -> <span class="diff-new">${formatMoney(newVal)}</span></div>`;
        }
      }

      return diffs;
    }

    /* ─── Action button ─────────────────────── */

    updateActionButton() {
      const provider = this.activeTab;
      const btn = document.getElementById("applyChangesBtn");
      const summary = document.getElementById("actionSummary");
      if (!btn) return;

      const selections = this.providerSelections.get(provider);
      if (!selections) {
        btn.disabled = true;
        btn.textContent = "No changes to apply";
        if (summary) summary.textContent = "";
        return;
      }

      const addCount = selections.new.size;
      const updateCount = selections.updated.size;
      const removeCount = FULL_SYNC_PROVIDERS.has(provider) ? selections.missing.size : 0;
      const total = addCount + updateCount + removeCount;

      if (total === 0) {
        btn.disabled = true;
        btn.textContent = "No changes to apply";
        if (summary) summary.textContent = "";
        return;
      }

      btn.disabled = false;
      const parts = [];
      if (addCount > 0) parts.push(`add ${addCount}`);
      if (updateCount > 0) parts.push(`update ${updateCount}`);
      if (removeCount > 0) parts.push(`remove ${removeCount}`);
      btn.textContent = `Apply changes: ${parts.join(", ")}`;
      if (summary) summary.textContent = `${total} change${total !== 1 ? "s" : ""} pending`;
    }

    /* ─── Confirm modal ─────────────────────── */

    showConfirmModal(provider) {
      const selections = this.providerSelections.get(provider);
      if (!selections) return;

      const { newModels, updatedModels, missingModels } = this.classifyModels(provider);

      const selectedNew = newModels.filter((m) => selections.new.has(m.id));
      const selectedUpdated = updatedModels.filter((m) => selections.updated.has(m.remote.id));
      const selectedMissing = FULL_SYNC_PROVIDERS.has(provider) ? missingModels.filter((m) => selections.missing.has(m.id)) : [];

      const titleEl = document.getElementById("confirmTitle");
      const subtitleEl = document.getElementById("confirmSubtitle");
      const bodyEl = document.getElementById("confirmBody");
      if (titleEl) titleEl.textContent = `Confirm sync - ${PROVIDER_LABELS[provider] || provider}`;
      if (subtitleEl) subtitleEl.textContent = "The following changes will be applied to model_specs.json:";

      let bodyHtml = "";

      if (selectedNew.length > 0) {
        bodyHtml += `<div class="modal-section">
          <div class="label add">Add ${selectedNew.length} model${selectedNew.length !== 1 ? "s" : ""}</div>
          <ul>${selectedNew.map((m) => `<li>${escapeHtml(m.name)} <span class="diff-detail">${escapeHtml(m.id)}</span></li>`).join("")}</ul>
        </div>`;
      }

      if (selectedUpdated.length > 0) {
        bodyHtml += `<div class="modal-section">
          <div class="label update">Update ${selectedUpdated.length} model${selectedUpdated.length !== 1 ? "s" : ""}</div>
          <ul>${selectedUpdated.map((item) => {
            const diffDetail = this._summarizeDiff(item.remote, item.local);
            return `<li>${escapeHtml(item.remote.name)} <span class="diff-detail">${diffDetail}</span></li>`;
          }).join("")}</ul>
        </div>`;
      }

      if (selectedMissing.length > 0) {
        bodyHtml += `<div class="modal-section">
          <div class="label remove">Remove ${selectedMissing.length} model${selectedMissing.length !== 1 ? "s" : ""}</div>
          <ul>${selectedMissing.map((m) => `<li>${escapeHtml(m.name)} <span class="diff-detail">${escapeHtml(m.id)}</span></li>`).join("")}</ul>
        </div>`;
      }

      if (!bodyHtml) {
        bodyHtml = '<p>No changes selected.</p>';
      }

      if (bodyEl) bodyEl.innerHTML = bodyHtml;

      // Reset apply button state
      const applyBtn = document.getElementById("confirmApply");
      if (applyBtn) {
        applyBtn.disabled = false;
        applyBtn.textContent = "Apply changes";
      }

      this._showModal("confirmModal");
    }

    _summarizeDiff(remote, local) {
      const changes = [];
      if (remote.context_window !== local.context_window) changes.push("context");
      if (remote.pricing?.input_per_million !== local.pricing?.input_per_million) changes.push("input price");
      if (remote.pricing?.output_per_million !== local.pricing?.output_per_million) changes.push("output price");
      if (remote.output_tokens !== local.output_tokens) changes.push("output tokens");
      if (remote.name !== local.name) changes.push("name");
      return changes.length > 0 ? changes.join(", ") : "metadata";
    }

    /* ─── Apply changes ─────────────────────── */

    async handleApplyChanges(provider) {
      const applyBtn = document.getElementById("confirmApply");
      if (applyBtn) {
        applyBtn.disabled = true;
        applyBtn.textContent = "Applying...";
      }

      const selections = this.providerSelections.get(provider);
      if (!selections) {
        this._hideModal("confirmModal");
        return;
      }

      const { newModels, updatedModels, currentModels, missingModels } = this.classifyModels(provider);

      let modelsToSync;

      if (FULL_SYNC_PROVIDERS.has(provider)) {
        // FULL_SYNC: send ALL models that should remain in the catalog for this provider
        // = current (up to date) + selected new + selected updated + non-removed missing
        const keep = [];

        // Current (up to date) models - always keep
        for (const m of currentModels) {
          keep.push(m);
        }

        // Selected new models
        for (const m of newModels) {
          if (selections.new.has(m.id)) {
            keep.push(m);
          }
        }

        // Selected updated models (send remote version)
        for (const pair of updatedModels) {
          if (selections.updated.has(pair.remote.id)) {
            keep.push(pair.remote);
          } else {
            // Not selected for update - keep the local version
            keep.push(pair.local);
          }
        }

        // Missing models: if NOT checked for removal, keep them
        for (const m of missingModels) {
          if (!selections.missing.has(m.id)) {
            keep.push(m);
          }
        }

        modelsToSync = keep;
      } else {
        // DISCOVERY mode: only send new + updated selections
        const toSync = [];
        for (const m of newModels) {
          if (selections.new.has(m.id)) toSync.push(m);
        }
        for (const pair of updatedModels) {
          if (selections.updated.has(pair.remote.id)) toSync.push(pair.remote);
        }
        modelsToSync = toSync;
      }

      try {
        await this.fetchJson(`/api/admin/models/providers/${encodeURIComponent(provider)}/sync`, {
          method: "POST",
          body: JSON.stringify({ models: modelsToSync }),
        });
        this._hideModal("confirmModal");
        this.showNotice("globalNotice", "success", "Sync complete", `${PROVIDER_LABELS[provider] || provider} models updated successfully.`);
        await this.refreshCatalog();
        await this.loadRemote(provider, true);
      } catch (err) {
        this.showNotice("globalNotice", "error", "Sync failed", err.message);
        if (applyBtn) {
          applyBtn.disabled = false;
          applyBtn.textContent = "Apply changes";
        }
      }
    }

    /* ─── Sync All ──────────────────────────── */

    async handleSyncAll() {
      const bodyEl = document.getElementById("syncAllBody");
      const totalEl = document.getElementById("syncAllTotal");
      const applyBtn = document.getElementById("syncAllApply");
      if (!bodyEl) return;

      // Reset state
      if (applyBtn) {
        applyBtn.disabled = true;
        applyBtn.textContent = "Fetching...";
      }
      if (totalEl) totalEl.style.display = "none";

      // Build initial rows with spinners
      bodyEl.innerHTML = PROVIDER_TABS.map((p) => `
        <div class="sync-all-row" data-provider="${escapeHtml(p)}">
          <span class="provider-name">${escapeHtml(PROVIDER_LABELS[p] || p)}</span>
          <span class="stats">Loading...</span>
          <span class="status-icon"><span class="spinner"></span></span>
        </div>
      `).join("");

      this._showModal("syncAllModal");

      // Store per-provider results for the apply step
      this._syncAllResults = new Map();

      // Fetch all providers in parallel
      const results = await Promise.allSettled(
        PROVIDER_TABS.map(async (p) => {
          try {
            const data = await this.fetchJson(`/api/admin/models/providers/${encodeURIComponent(p)}/remote`);
            const parsed = parseRemoteResponse(p, data);
            this.remote.set(p, parsed);
            this.initSelections(p);
            return { provider: p, success: true, data: parsed };
          } catch (err) {
            return { provider: p, success: false, error: err.message };
          }
        })
      );

      let totalNew = 0;
      let totalUpdated = 0;
      let totalRemoved = 0;
      let anyChanges = false;

      for (const result of results) {
        const value = result.status === "fulfilled" ? result.value : { provider: "unknown", success: false, error: "Promise rejected" };
        const row = bodyEl.querySelector(`[data-provider="${value.provider}"]`);
        if (!row) continue;

        const statsEl = row.querySelector(".stats");
        const iconEl = row.querySelector(".status-icon");

        if (!value.success) {
          if (statsEl) statsEl.textContent = `Error: ${value.error}`;
          if (iconEl) iconEl.textContent = "!";
          continue;
        }

        const { newModels, updatedModels, missingModels } = this.classifyModels(value.provider);
        const selections = this.providerSelections.get(value.provider);
        const addCount = selections ? selections.new.size : 0;
        const updateCount = selections ? selections.updated.size : 0;
        const removeCount = FULL_SYNC_PROVIDERS.has(value.provider) && selections ? selections.missing.size : 0;

        totalNew += addCount;
        totalUpdated += updateCount;
        totalRemoved += removeCount;
        if (addCount + updateCount + removeCount > 0) anyChanges = true;

        const parts = [];
        if (addCount > 0) parts.push(`${addCount} new`);
        if (updateCount > 0) parts.push(`${updateCount} updated`);
        if (removeCount > 0) parts.push(`${removeCount} missing`);

        if (statsEl) statsEl.textContent = parts.length > 0 ? parts.join(", ") : "Up to date";
        if (iconEl) iconEl.textContent = parts.length > 0 ? "*" : "OK";

        this._syncAllResults.set(value.provider, { newModels, updatedModels, missingModels });
      }

      // Show totals
      if (totalEl) {
        const parts = [];
        if (totalNew > 0) parts.push(`${totalNew} new`);
        if (totalUpdated > 0) parts.push(`${totalUpdated} updated`);
        if (totalRemoved > 0) parts.push(`${totalRemoved} to remove`);
        totalEl.textContent = parts.length > 0 ? `Total: ${parts.join(", ")}` : "Everything is up to date.";
        totalEl.style.display = "block";
      }

      if (applyBtn) {
        applyBtn.disabled = !anyChanges;
        applyBtn.textContent = anyChanges ? "Apply all changes" : "No changes needed";
      }
    }

    async handleSyncAllApply() {
      const applyBtn = document.getElementById("syncAllApply");
      if (applyBtn) {
        applyBtn.disabled = true;
        applyBtn.textContent = "Applying...";
      }

      // Build combined payload
      const providersPayload = {};

      for (const provider of PROVIDER_TABS) {
        const classifications = this._syncAllResults?.get(provider);
        const selections = this.providerSelections.get(provider);
        if (!classifications || !selections) continue;

        const { newModels, updatedModels, currentModels, missingModels } = this.classifyModels(provider);

        // Check if there are any changes for this provider
        const addCount = selections.new.size;
        const updateCount = selections.updated.size;
        const removeCount = FULL_SYNC_PROVIDERS.has(provider) ? selections.missing.size : 0;
        if (addCount + updateCount + removeCount === 0) continue;

        let modelsToSync;

        if (FULL_SYNC_PROVIDERS.has(provider)) {
          const keep = [];
          for (const m of currentModels) keep.push(m);
          for (const m of newModels) {
            if (selections.new.has(m.id)) keep.push(m);
          }
          for (const pair of updatedModels) {
            if (selections.updated.has(pair.remote.id)) {
              keep.push(pair.remote);
            } else {
              keep.push(pair.local);
            }
          }
          for (const m of missingModels) {
            if (!selections.missing.has(m.id)) keep.push(m);
          }
          modelsToSync = keep;
        } else {
          const toSync = [];
          for (const m of newModels) {
            if (selections.new.has(m.id)) toSync.push(m);
          }
          for (const pair of updatedModels) {
            if (selections.updated.has(pair.remote.id)) toSync.push(pair.remote);
          }
          modelsToSync = toSync;
        }

        if (modelsToSync.length > 0) {
          providersPayload[provider] = modelsToSync;
        }
      }

      if (Object.keys(providersPayload).length === 0) {
        this._hideModal("syncAllModal");
        return;
      }

      try {
        await this.fetchJson("/api/admin/models/sync-all", {
          method: "POST",
          body: JSON.stringify({ providers: providersPayload }),
        });
        this._hideModal("syncAllModal");
        this.showNotice("globalNotice", "success", "Bulk sync complete", "All provider models have been updated.");
        await this.refreshCatalog();
      } catch (err) {
        this.showNotice("globalNotice", "error", "Bulk sync failed", err.message);
        if (applyBtn) {
          applyBtn.disabled = false;
          applyBtn.textContent = "Apply all changes";
        }
      }
    }

    /* ─── Fetch helper ──────────────────────── */

    async fetchJson(url, options = {}) {
      const { headers: extraHeaders, signal, ...rest } = options;
      const response = await fetch(url, {
        ...rest,
        headers: {
          "Content-Type": "application/json",
          ...(extraHeaders || {}),
        },
        signal,
      });
      const text = await response.text();
      let data = {};
      if (text) {
        try {
          data = JSON.parse(text);
        } catch {
          data = { raw: text };
        }
      }
      if (!response.ok) {
        const message = data.error || data.detail || data.message || `Request failed (${response.status})`;
        throw new Error(message);
      }
      return data;
    }

    /* ─── Notice helpers ────────────────────── */

    showNotice(elementId, tone, title, message) {
      const el = document.getElementById(elementId);
      if (!el) return;
      const spinnerHtml = tone === "loading" ? '<span class="spinner"></span>' : "";
      el.className = `notice ${tone} visible`;
      el.innerHTML = `${spinnerHtml}<span class="title">${escapeHtml(title)}</span> ${escapeHtml(message)}`;
    }

    hideNotice(elementId) {
      const el = document.getElementById(elementId);
      if (!el) return;
      el.classList.remove("visible");
    }

    /* ─── Modal helpers ─────────────────────── */

    _showModal(id) {
      const el = document.getElementById(id);
      if (el) el.classList.add("visible");
    }

    _hideModal(id) {
      const el = document.getElementById(id);
      if (el) el.classList.remove("visible");
    }
  }

  /* ─── Bootstrap ───────────────────────────── */

  document.addEventListener("DOMContentLoaded", () => {
    new ModelsAdminApp();
  });
})();
