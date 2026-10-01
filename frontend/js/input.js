/* ============================================================
   TruthLens · js/input.js
   The Text / Image / URL input modes, the character counter, the
    language badge and URL validation.

   getCurrentClaimText() is the single place a real backend would
   read the submitted claim from.
   ============================================================ */

let currentMode = 'text';

const TAB_ACTIVE = 'flex items-center gap-2 px-4 py-2 bg-surface rounded-lg border border-glass-stroke font-label-md text-label-md text-primary shadow-sm hover:border-primary transition-colors';
const TAB_IDLE = 'flex items-center gap-2 px-4 py-2 bg-surface/50 rounded-lg border border-glass-stroke font-label-md text-label-md text-on-surface-variant hover:bg-surface transition-colors';

function setInputMode(mode) {
    currentMode = mode;
    ['text', 'image', 'url'].forEach(m => {
        const tab = document.getElementById('tab-' + m);
        const panel = document.getElementById('mode-' + m);
        const isActive = (m === mode);
        if (tab) tab.className = isActive ? TAB_ACTIVE : TAB_IDLE;
        if (panel) panel.classList.toggle('hidden', !isActive);
    });
}

/* Reads the claim from whichever input mode is active.
   Returns '' when that mode has nothing to submit yet. */
function getCurrentClaimText() {
    if (currentMode === 'image') {
        const ocr = document.getElementById('ocr-text');
        const preview = document.getElementById('img-preview');
        const file = document.getElementById('img-upload')?.files?.[0];
        const hasImage = preview && !preview.classList.contains('hidden');
        if (!hasImage || !file) return '';
        return ocr?.value.trim() || 'Image claim';
    }
    if (currentMode === 'url') {
        const preview = document.getElementById('url-preview');
        const fetched = preview && !preview.classList.contains('hidden');
        const url = document.getElementById('url-input');
        return (fetched && url) ? url.value.trim() : '';
    }
    const ta = document.getElementById('claim-input');
    return ta ? ta.value.trim() : '';
}

/* ---------- URL mode ----------
    Validate the URL locally, then submit it to the backend for extraction. */
function fetchUrl() {
    const val = document.getElementById('url-input').value.trim();
    const err = document.getElementById('url-error');
    const preview = document.getElementById('url-preview');

    if (!val.startsWith('http://') && !val.startsWith('https://')) {
        err.classList.remove('hidden');
        preview.classList.add('hidden');
        return;
    }
    err.classList.add('hidden');
    preview.classList.remove('hidden');
}

/* ---------- Language badge ----------
    This is only immediate UI feedback; the backend is authoritative. */
function detectLanguage(text) {
    for (const { pattern, lang } of LANG_PATTERNS) {
        if (pattern.test(text)) return lang;
    }
    return LANG_DEFAULT;
}

/* Keeps the character counter and language badge in step with the textarea. */
function refreshInputMeta() {
    const ta = document.getElementById('claim-input');
    const counter = document.getElementById('char-counter');
    const langText = document.getElementById('lang-text');
    if (!ta) return;
    if (counter) counter.textContent = `${ta.value.length} / 1000 characters`;
    if (langText) langText.textContent = detectLanguage(ta.value);
}
