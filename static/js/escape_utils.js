/**
 * Shared HTML escaping helpers for the engine frontend.
 *
 * Two modes, named after the context where the result is interpolated:
 * - escapeHtml(value): HTML content context. Escapes &, <, >.
 * - escapeHtmlAttr(value): quoted attribute context. Escapes &, <, >, ", '.
 *
 * Never use escapeHtml() inside an HTML attribute; use escapeHtmlAttr() there.
 * This file must be loaded before the other frontend scripts that rely on
 * these globals (see the <script> tags in templates/*.html).
 */

/**
 * Escapes a value for safe interpolation into HTML content.
 * @param {*} value - The value to escape
 * @returns {string} - The escaped string, safe for innerHTML content
 */
function escapeHtml(value) {
    if (value === null || value === undefined) {
        return '';
    }
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

/**
 * Escapes a value for safe interpolation into a quoted HTML attribute.
 * @param {*} value - The value to escape
 * @returns {string} - The escaped string, safe for attribute values
 */
function escapeHtmlAttr(value) {
    if (value === null || value === undefined) {
        return '';
    }
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
