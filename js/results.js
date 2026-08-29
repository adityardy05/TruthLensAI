/* ============================================================
   TruthLens · js/results.js
   The Results view: verdict readout, the expandable evidence /
   source cards, Recent Claims, and the reset back to Verify.

   The verdict, sub-claims, sources and reviewer findings are static
   markup in index.html for now. readVerdictFromResults() is the one
   place the rest of the app asks "what did this come out as", so a
   real backend response only needs wiring in here.
   ============================================================ */

/* Reads the verdict off the Results view rather than hard-coding it,
   so the state, the badge and the PDF can never disagree. */
function readVerdictFromResults() {
    const el = document.querySelector('#view-results .font-display') || document.getElementById('results-verdict');
    const text = el ? el.innerText.trim() : '';
    return text || 'FALSE';
}
// Kept so any existing reference still works
const currentVerdictText = readVerdictFromResults;

/* ---------- Verdict Card & Dynamic Confidence Radial ---------- */
function renderVerdictCard() {
    const confStr = (typeof REPORT_META !== 'undefined' && REPORT_META.confidence) ? REPORT_META.confidence : '91%';
    const confNum = parseInt(confStr, 10) || 91;

    const radial = document.getElementById('results-confidence-radial');
    const valEl = document.getElementById('results-confidence-val');
    if (radial) {
        radial.style.background = `conic-gradient(#ef4444 ${confNum}%, #fee2e2 0)`;
    }
    if (valEl) {
        valEl.innerHTML = `${confNum}<span class="text-sm font-body-md">%</span>`;
    }

    const sourcesEl = document.getElementById('results-sources-count');
    if (sourcesEl && typeof REPORT_META !== 'undefined' && REPORT_META.sources) {
        const srcCount = REPORT_META.sources.replace(/[^0-9]/g, '');
        sourcesEl.textContent = srcCount || '9';
    }

    const explEl = document.getElementById('results-explanation');
    if (explEl && typeof REPORT_META !== 'undefined' && REPORT_META.explanation) {
        explEl.textContent = REPORT_META.explanation;
    }
}

/* ---------- Evidence / source accordion ---------- */
function toggleSource(id) {
    const body = document.getElementById(id + '-body');
    const icon = document.getElementById(id + '-icon');
    if (!body || !icon) return;
    const isHidden = body.classList.contains('hidden');
    body.classList.toggle('hidden', !isHidden);
    icon.innerText = isHidden ? 'expand_less' : 'expand_more';
}

/* ---------- Recent Claims ----------
   Rendered from claimHistory in js/state.js. Independent of the
   current claim: starting a new verification replaces the current
   claim but only adds to this list. */
function escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function renderHistory() {
    const empty = document.getElementById('recent-empty');
    const list = document.getElementById('recent-list');
    if (!list || !empty) return;

    const history = getHistory();

    if (history.length === 0) {
        empty.classList.remove('hidden');
        list.classList.add('hidden');
        return;
    }

    empty.classList.add('hidden');
    list.classList.remove('hidden');
    list.innerHTML = history.slice(0, 5).map(h => `
                <div class="w-full py-3 px-4 glass-panel rounded-lg font-body-md flex items-center gap-3 cursor-pointer hover:bg-surface-container-low transition-colors bg-white" onclick="switchView('results')">
                    <div class="w-2 h-2 rounded-full bg-error shrink-0"></div>
                    <span class="flex-1 text-left text-on-surface truncate text-sm">${escapeHtml(h.claim)}</span>
                    <span class="font-label-sm text-text-muted text-xs shrink-0">${escapeHtml(h.time)}</span>
                </div>
            `).join('');
}

/* ---------- "Analyze another claim" ----------
   The one explicit reset in the UI. Unlike navigation, this does
   clear the current claim, so Verify re-centers with no checking card.
   Recent Claims is left alone. */
function resetAndGoHome() {
    clearCheckingTimers();
    clearCurrentClaim();

    const claimInput = document.getElementById('claim-input');
    if (claimInput) claimInput.value = DEFAULT_CLAIM;
    refreshInputMeta();

    resetSideCheckingToIdle();
    resetCheckingPipeline();
    setTargetClaim(DEFAULT_CLAIM, true);

    switchView('home');
}
