/* ============================================================
   TruthLens · js/results.js
   The Results view: verdict readout, the expandable evidence /
   source cards, Recent Claims, and the reset back to Verify.

   Results are populated from the current backend verification response.
   The HTML only provides empty containers; live evidence, reviewer
   findings, patterns and confidence are rendered after verification.
   ============================================================ */

/* Reads the verdict off the Results view rather than hard-coding it,
   so the state, the badge and the PDF can never disagree. */
function readVerdictFromResults() {
    const el = document.querySelector('#view-results .font-display') || document.getElementById('results-verdict');
    const text = el ? el.innerText.trim() : '';
    return text || 'UNVERIFIABLE';
}
// Kept so any existing reference still works
const currentVerdictText = readVerdictFromResults;

/* ---------- Verdict Card & Dynamic Confidence Radial ---------- */
function renderVerdictCard() {
    const result = getCurrentClaim()?.result;
    if (result) {
        const confidence = Math.round(Number(result.confidence) || 0);
        document.getElementById('results-verdict').textContent = result.verdict;
        document.getElementById('results-confidence-label').textContent = confidence >= 80 ? 'High Confidence' : 'Evidence-based assessment';
        document.getElementById('results-explanation').textContent = result.justification;
        document.getElementById('results-sources-count').textContent = result.evidence.length;
        const radial = document.getElementById('results-confidence-radial');
        const value = document.getElementById('results-confidence-val');
        if (radial) radial.style.background = `conic-gradient(#ef4444 ${confidence}%, #fee2e2 0)`;
        if (value) value.innerHTML = `${confidence}<span class="text-sm font-body-md">%</span>`;
        renderLiveEvidence(result.evidence);
        renderLiveReviewers(result.persona_insights);
        renderLiveAgentFeed(result.persona_insights, result.rounds_executed, result.verdict, result.confidence);
        renderLivePatterns(result.patterns_detected);
        return;
    }
    const confNum = 0;

    const radial = document.getElementById('results-confidence-radial');
    const valEl = document.getElementById('results-confidence-val');
    if (radial) {
        radial.style.background = `conic-gradient(#ef4444 ${confNum}%, #fee2e2 0)`;
    }
    if (valEl) {
        valEl.innerHTML = `${confNum}<span class="text-sm font-body-md">%</span>`;
    }

}

function renderLiveReviewers(insights) {
    const cards = document.getElementById('reviewer-cards');
    if (!cards) return;
    cards.innerHTML = Object.entries(insights || {}).map(([name, insight]) => `<div class="flex gap-4 p-5 rounded-xl bg-surface-container-low/50 border border-glass-stroke"><div><span class="font-label-md text-primary font-bold">${escapeHtml(name.replace(/_/g, ' '))}</span><p class="font-body-md text-sm text-on-surface-variant mt-1">${escapeHtml(String(insight))}</p></div></div>`).join('');
}

function renderLiveAgentFeed(insights, rounds, verdict, confidence) {
    const feed = document.getElementById('agent-feed');
    if (!feed) return;
    const entries = Object.entries(insights || {}).filter(([, value]) => String(value || '').trim());
    if (!entries.length) {
        feed.innerHTML = '<div class="p-5 rounded-xl bg-surface-container-low/50 border border-glass-stroke text-text-muted text-sm">No reviewer findings were returned by the verification pipeline.</div>';
        return;
    }
    feed.innerHTML = entries.map(([name, insight]) => `
        <div class="flex gap-4 p-4 rounded-xl bg-surface/50 border border-glass-stroke hover:bg-surface transition-colors duration-300">
            <div class="flex-shrink-0 w-10 h-10 rounded-full bg-surface-container-low flex items-center justify-center text-primary border border-outline-variant">
                <span class="material-symbols-outlined text-sm">smart_toy</span>
            </div>
            <div>
                <div class="flex items-center gap-2 mb-1">
                    <span class="font-label-md text-label-md text-primary font-bold">${escapeHtml(name.replace(/_/g, ' '))}</span>
                    <span class="text-xs text-text-muted">${escapeHtml(String(rounds || 0))} round(s)</span>
                </div>
                <p class="font-body-md text-body-md text-on-surface-variant">${escapeHtml(String(insight))}</p>
            </div>
        </div>`
    ).join('') + `
        <div class="p-4 rounded-xl bg-surface-container-low border border-glass-stroke text-sm text-text-muted">
            Final judgment: <strong class="text-primary">${escapeHtml(String(verdict || 'UNVERIFIABLE'))}</strong>
            · confidence ${escapeHtml(String(Math.round(Number(confidence) || 0)))}%
        </div>`;
}

function renderLivePatterns(patterns) {
    const cards = document.getElementById('pattern-cards');
    if (!cards) return;
    const values = Array.isArray(patterns) ? patterns.filter(Boolean).slice(0, 3) : [];
    if (!values.length) {
        cards.innerHTML = '<div class="glass-panel p-6 rounded-2xl border border-glass-stroke bg-white shadow-md md:col-span-3"><h4 class="font-headline-md text-primary text-xl mb-2">No specific pattern reported</h4><p class="font-body-md text-text-muted text-sm">The current verification did not return a specific manipulation pattern from the analysis.</p></div>';
        return;
    }
    cards.innerHTML = values.map((pattern, index) => `
        <div class="glass-panel p-6 rounded-2xl border border-glass-stroke bg-white shadow-md">
            <div class="w-12 h-12 rounded-full bg-secondary-container text-secondary flex items-center justify-center mb-4">
                <span class="material-symbols-outlined">${index === 0 ? 'analytics' : index === 1 ? 'psychology' : 'visibility'}</span>
            </div>
            <h4 class="font-headline-md text-primary text-xl mb-2">Detected pattern ${index + 1}</h4>
            <p class="font-body-md text-text-muted text-sm">${escapeHtml(String(pattern))}</p>
        </div>`
    ).join('');
}

function renderLiveEvidence(evidence) {
    const cards = document.getElementById('source-cards');
    if (!cards) return;
    const items = Array.isArray(evidence) ? evidence : [];
    const support = items.filter(item => String(item.stance || '').toUpperCase() === 'SUPPORT').length;
    const contradict = items.filter(item => String(item.stance || '').toUpperCase() === 'CONTRADICT').length;
    const supportEl = document.getElementById('source-supporting');
    const contradictEl = document.getElementById('source-contradicting');
    if (supportEl) supportEl.textContent = String(support);
    if (contradictEl) contradictEl.textContent = String(contradict);
    cards.innerHTML = items.map((item, index) => {
        const id = `live-source-${index}`;
        const score = Math.round((item.combined_reliability || 0) * 100);
        return `<div class="border border-glass-stroke rounded-xl overflow-hidden"><button class="w-full flex justify-between items-center gap-3 p-4 bg-surface text-left" onclick="toggleSource('${id}')"><span class="font-label-md text-primary font-bold">${escapeHtml(item.source_domain)}</span><span class="font-label-md text-text-muted">${score}%</span><span class="material-symbols-outlined text-sm" id="${id}-icon">expand_more</span></button><div class="hidden p-4 bg-surface-container-low border-t border-glass-stroke space-y-2" id="${id}-body"><p class="font-body-md text-sm text-on-surface-variant">${escapeHtml(item.content)}</p><a class="text-primary hover:underline text-sm" href="${escapeHtml(item.url)}" target="_blank" rel="noopener noreferrer">Open source</a></div></div>`;
    }).join('') || '<p class="text-text-muted">No evidence was retrieved.</p>';
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
                    <div class="w-2 h-2 rounded-full ${String(h.verdict || '').toUpperCase() === 'TRUE' ? 'bg-tertiary-fixed-dim' : String(h.verdict || '').toUpperCase() === 'FALSE' ? 'bg-error' : 'bg-secondary'} shrink-0"></div>
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
    if (claimInput) claimInput.value = '';
    refreshInputMeta();

    resetSideCheckingToIdle();
    resetCheckingPipeline();
    setTargetClaim('No claim submitted yet.', true);

    switchView('home');
}
