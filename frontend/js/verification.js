/* ============================================================
   TruthLens · js/verification.js

   This file owns BOTH checking concepts. They are deliberately
   separate and must stay that way:

   1. The Verify-page checking card — the small panel that appears
      beside the claim block after Analyze and runs the five-stage
      animation. Its element id is #how-it-works for historical
      reasons; it is NOT the How It Works page. Always reach it
      through the showVerifyCheckingCard / hideVerifyCheckingCard
      helpers below rather than by that id.

   2. The full verification presentation — #view-checking, with
      Verification in Progress / Target Claim / Analysis Pipeline /
      the five stages / Agent Processing Log. This is what the
      How It Works nav item opens, and it is also the live screen
      the pipeline runs on.

   The backend verification request is real. Timers only control the
   visual hand-off between pipeline stages while the synchronous API
   request runs. Once the backend responds, the terminal log and
   result panels are populated from that real response.
   ============================================================ */

/* Every simulated stage timer, so a new claim or a reset can cancel
   an in-flight run instead of letting two runs fight over the DOM. */
let checkingTimers = [];
let activeVerificationRun = 0;

function clearCheckingTimers() {
    checkingTimers.forEach(clearTimeout);
    checkingTimers = [];
}

/* ============================================================
   1. VERIFY-PAGE CHECKING CARD
   ============================================================ */

function showVerifyCheckingCard(animate) {
    const card = document.getElementById('how-it-works');
    if (!card) return;
    card.classList.remove('hidden');
    if (animate) {
        card.classList.remove('fade-in');
        void card.offsetWidth;
        card.classList.add('fade-in');
    }
}

function hideVerifyCheckingCard() {
    const card = document.getElementById('how-it-works');
    if (card) card.classList.add('hidden');
}

function setSideCheckingStep(n, state) {
    const el = document.getElementById('side-step-' + n);
    if (!el) return;
    const label = SIDE_STEP_LABELS[n - 1];
    if (state === 'done') {
        el.className = 'flex items-center gap-4 text-primary';
        el.innerHTML = `<span class="material-symbols-outlined text-tertiary-fixed-dim">check_circle</span><span class="font-body-md">${label}</span>`;
    } else if (state === 'active') {
        el.className = 'flex items-center gap-4 text-primary font-semibold';
        el.innerHTML = `<div class="w-5 h-5 rounded-full border-2 border-surface-variant border-t-primary animate-spin ml-0.5 mr-0.5"></div><span class="font-body-md">${label}</span>`;
    } else {
        el.className = 'flex items-center gap-4 text-surface-tint opacity-70';
        el.innerHTML = `<div class="w-5 h-5 rounded-full border-2 border-outline ml-0.5 mr-0.5"></div><span class="font-body-md">${label}</span>`;
    }
}

function resetSideCheckingSteps(mode = 'idle') {
    setSideCheckingStep(1, mode === 'active' ? 'active' : 'pending');
    for (let i = 2; i <= 5; i++) setSideCheckingStep(i, 'pending');
}

/* Back to the fresh state: card hidden, no claim, all stages pending. */
function resetSideCheckingToIdle() {
    hideVerifyCheckingCard();

    const badge = document.getElementById('side-status-badge');
    if (badge) badge.textContent = 'checking';

    const spinner = document.getElementById('side-spinner');
    if (spinner) spinner.classList.add('hidden');

    const claimEl = document.getElementById('side-check-claim');
    if (claimEl) {
        claimEl.textContent = '';
        claimEl.classList.add('hidden');
    }

    resetSideCheckingSteps('idle');
}

// Kept so any existing reference still works
function hideHomeCheckingPanel() {
    resetSideCheckingToIdle();
}

/* ---------- Centered -> left transition (Verify only) ----------
   The claim block starts centered and slides into its left column
   once a claim is submitted. The offset itself is in
   css/responsive.css and only applies at lg and above. */
function setClaimBlockCentered(centered) {
    const block = document.getElementById('claim-block');
    if (block) block.classList.toggle('is-centered', !!centered);
}

/* Re-applies the current claim to the Verify page on navigation.
   Must NOT replay the left transition or restart the animation —
   returning to Verify should look like you never left. */
function restoreVerifyForCurrentClaim() {
    if (!hasCurrentClaim()) {
        resetSideCheckingToIdle();
        setClaimBlockCentered(true);
        return;
    }

    const claim = getCurrentClaim();

    setClaimBlockCentered(false);
    showVerifyCheckingCard(false);

    const claimEl = document.getElementById('side-check-claim');
    if (claimEl) {
        claimEl.textContent = '"' + claim.text + '"';
        claimEl.classList.remove('hidden');
    }

    const running = isClaimRunning();

    const spinner = document.getElementById('side-spinner');
    if (spinner) spinner.classList.toggle('hidden', !running);

    const badge = document.getElementById('side-status-badge');
    if (badge) badge.textContent = running ? 'checking' : (claim.verdict || 'checking');

    // While the side sequence is still running, its own timers own the
    // stage states; only settle them once it is past that stage.
    if (!running) {
        for (let i = 1; i <= 5; i++) setSideCheckingStep(i, 'done');
    }
}

/* ============================================================
   2. FULL VERIFICATION PRESENTATION (#view-checking)
   ============================================================ */

function pipelineStepMarkup(n, state) {
    const step = PIPELINE_STEPS[n - 1];
    if (state === 'done') {
        return `
                    <div class="w-6 h-6 rounded-full bg-primary text-on-primary flex items-center justify-center shrink-0 step-icon">
                        <span class="material-symbols-outlined text-[14px] font-bold">check</span>
                    </div>
                    <div>
                        <h4 class="font-body-md text-body-md text-primary font-semibold step-title">${step.title}</h4>
                        <p class="font-body-md text-body-md text-text-muted step-desc">${step.agent}</p>
                    </div>
                `;
    }
    if (state === 'running') {
        return `
                    <div class="w-6 h-6 rounded-full bg-surface border-2 border-tertiary-fixed-dim flex items-center justify-center shrink-0 relative step-icon">
                        <div class="w-2 h-2 rounded-full bg-tertiary-fixed-dim"></div>
                        <div class="absolute inset-0 rounded-full border-2 border-tertiary-fixed-dim agent-pulse"></div>
                    </div>
                    <div>
                        <h4 class="font-body-md text-body-md text-primary font-bold step-title">${step.title}</h4>
                        <p class="font-body-md text-body-md text-tertiary-fixed-dim step-desc">${step.agent}${step.suffix}</p>
                    </div>
                `;
    }
    return `
                    <div class="w-6 h-6 rounded-full bg-surface border-2 border-outline flex items-center justify-center shrink-0 step-icon"></div>
                    <div>
                        <h4 class="font-body-md text-body-md text-on-surface-variant font-semibold step-title">${step.title}</h4>
                        <p class="font-body-md text-body-md text-text-muted step-desc">${step.agent}</p>
                    </div>
                `;
}

/* state: 'done' | 'running' | 'pending' */
function setPipelineStep(n, state) {
    const el = document.getElementById('step-' + n);
    if (!el) return;
    el.classList.toggle('opacity-50', state === 'pending');
    el.innerHTML = pipelineStepMarkup(n, state);
}

/* The illustrative default: first two stages done, third running,
   last two still to come. This is the state the screen ships in. */
function resetPipelineSteps() {
    setPipelineStep(1, 'done');
    setPipelineStep(2, 'done');
    setPipelineStep(3, 'running');
    setPipelineStep(4, 'pending');
    setPipelineStep(5, 'pending');
}

function setPipelineAllComplete() {
    for (let i = 1; i <= 5; i++) setPipelineStep(i, 'done');
}

/* state: 'default' | 'judging' | 'finalizing' | 'complete' */
function setTerminalState(state) {
    const evidence = document.getElementById('terminal-evidence-line');
    const judge = document.getElementById('terminal-judge-line');
    const cursor = document.getElementById('terminal-cursor-line');
    const placeholder = document.getElementById('terminal-placeholder');

    if (placeholder) placeholder.style.display = state === 'default' ? 'block' : 'none';
    if (evidence) evidence.style.display = state === 'judging' || state === 'finalizing' || state === 'complete' ? 'block' : 'none';
    if (judge) judge.style.display = state === 'finalizing' || state === 'complete' ? 'block' : 'none';

    const line = state === 'default'
        ? 'Waiting for the verification backend...'
        : state === 'judging'
            ? 'Backend evidence and reviewer analysis are being evaluated...'
            : state === 'finalizing'
                ? 'Final supervisor is synthesizing the judgment...'
                : 'Verification completed.';
    if (cursor) cursor.innerHTML = '&gt; ' + line + '<span class="terminal-cursor"></span>';
}

function renderDynamicProcessingLog(result) {
    const terminal = document.getElementById('terminal-content');
    if (!terminal) return;
    const evidence = Array.isArray(result.evidence) ? result.evidence : [];
    const subClaims = Array.isArray(result.sub_claims) ? result.sub_claims : [];
    const meta = result.retrieval_metadata || {};
    const insights = result.persona_insights || {};

    const lines = [
        '<div class="text-white/60">TruthLensAI verification run started.</div>',
        '<div class="text-white/60">Claim received by the verification backend.</div>',
        '<div class="mt-2"><span class="text-secondary-fixed">[DECOMPOSE]</span> ' +
            escapeHtml(String(subClaims.length)) + ' sub-claim(s) returned.</div>',
    ];

    subClaims.slice(0, 5).forEach((claim, index) => {
        lines.push('<div>  ├─ Sub ' + (index + 1) + ': "' + escapeHtml(String(claim)) + '"</div>');
    });

    lines.push(
        '<div class="mt-2"><span class="text-tertiary-fixed-dim">[RETRIEVAL]</span> ' +
        escapeHtml(String(meta.num_live_web_sources || 0)) + ' live web source(s), ' +
        escapeHtml(String(meta.num_sources_kept || evidence.length)) + ' retained evidence item(s).</div>'
    );

    const domains = [...new Set(evidence.map(item => item.source_domain).filter(Boolean))].slice(0, 5);
    if (domains.length) {
        lines.push('<div><span class="text-tertiary-fixed-dim">[SOURCES]</span> ' + domains.map(escapeHtml).join(', ') + '</div>');
    }

    Object.entries(insights).forEach(([persona, insight]) => {
        if (String(insight || '').trim()) {
            lines.push(
                '<div class="mt-2"><span class="text-secondary-fixed-dim">[' +
                escapeHtml(persona.replace(/_/g, ' ').toUpperCase()) +
                ']</span> ' + escapeHtml(String(insight)) + '</div>'
            );
        }
    });

    lines.push(
        '<div class="mt-2 text-white/80"><span class="text-[#c2b9fd]">[JUDGE]</span> ' +
        'Verdict: ' + escapeHtml(String(result.verdict || 'UNVERIFIABLE')) +
        ' · confidence: ' + escapeHtml(String(Math.round(Number(result.confidence) || 0))) + '%</div>'
    );

    terminal.innerHTML = lines.join('') +
        '<div class="mt-1 flex items-center text-[#00dbe9]" id="terminal-cursor-line">&gt; Verification completed<span class="terminal-cursor"></span></div>';
}

function setTargetClaim(text, isExample) {
    const claimEl = document.getElementById('checking-claim');
    if (claimEl) claimEl.textContent = '"' + text + '"';

    // Makes clear when the screen is showing the shipped example rather
    // than something the user actually submitted.
    const labelEl = document.getElementById('checking-claim-label');
    if (labelEl) labelEl.textContent = isExample ? 'Target Claim · Example' : 'Target Claim';
}

/* Full reset of the pipeline screen, used before a run starts. */
function resetCheckingPipeline() {
    clearCheckingTimers();
    resetPipelineSteps();
    setTerminalState('default');
}
const resetPipeline = resetCheckingPipeline;

/* ---------- How It Works ----------
   Renders #view-checking as a read-only presentation of the
   verification process. It starts nothing and restarts nothing.

   With a finished claim it shows that claim's completed pipeline.
   With no claim it shows the shipped example, clearly labelled.
   While a run is in flight it leaves the live screen alone, because
   the run's own timers own it. */
function renderVerificationPresentation() {
    if (!hasCurrentClaim()) {
        setTargetClaim('No claim submitted yet.', true);
        resetPipelineSteps();
        setTerminalState('default');
        return;
    }

    const claim = getCurrentClaim();
    setTargetClaim(claim.text, false);

    if (claim.stage === 'done') {
        setPipelineAllComplete();
        setTerminalState('complete');
    } else {
        resetPipelineSteps();
        setTerminalState('default');
    }
}

/* ============================================================
   3. THE FLOW: Analyze -> card -> full screen -> Results
   ============================================================ */

async function analyzeClaim() {
    const claim = getCurrentClaimText();

    if (!claim) {
        // Nothing to check yet — nudge the user to the right control.
        // NOTE: never trigger #img-upload from here. Analyze must never
        // open the OS file picker; the drop-zone is the only thing that does.
        if (currentMode === 'text') {
            const ta = document.getElementById('claim-input');
            if (ta) ta.focus();
        } else if (currentMode === 'url') {
            const urlIn = document.getElementById('url-input');
            if (urlIn) urlIn.focus();
        } else {
            const ocr = document.getElementById('ocr-text');
            if (ocr) ocr.focus();
        }
        return;
    }

    // A new submission becomes the current claim and replaces any
    // previous one, along with the pipeline built for it.
    const runId = ++activeVerificationRun;
    setCurrentClaim(claim, currentMode);
    let verificationRequest;
    try {
        verificationRequest = submitVerification(currentMode);
    } catch (error) {
        alert(error.message);
        return;
    }
    resetCheckingPipeline();

    // Slide the centered claim block into its left column and reveal the
    // checking card on the right with a subtle fade. Verify page only.
    setClaimBlockCentered(false);
    showVerifyCheckingCard(true);

    const badge = document.getElementById('side-status-badge');
    if (badge) badge.textContent = 'checking';
    const spinner = document.getElementById('side-spinner');
    if (spinner) spinner.classList.remove('hidden');

    const claimEl = document.getElementById('side-check-claim');
    if (claimEl) {
        claimEl.textContent = '"' + claim + '"';
        claimEl.classList.remove('hidden');
    }

    clearCheckingTimers();
    resetSideCheckingSteps('active');

    // Five-stage animation on the card
    const cardStages = [600, 1200, 1800, 2400];
    cardStages.forEach((delay, i) => {
        checkingTimers.push(setTimeout(() => {
            if (runId !== activeVerificationRun) return;
            setSideCheckingStep(i + 1, 'done');
            setSideCheckingStep(i + 2, 'active');
        }, delay));
    });
    checkingTimers.push(setTimeout(() => {
        if (runId !== activeVerificationRun) return;
        setSideCheckingStep(5, 'done');
    }, 3000));

    // Card finishes -> hand over to the full verification screen
    checkingTimers.push(setTimeout(async () => {
        if (runId !== activeVerificationRun) return;
        markClaimChecking();
        setTargetClaim(claim, false);
        resetCheckingPipeline();
        switchView('checking');
        await simulateCheckingProcess(verificationRequest, runId);
    }, 3400));
}

/* The full pipeline run on #view-checking. Timers only — this is
   where real backend progress events would be wired in. */
async function simulateCheckingProcess(verificationRequest, runId) {
    // Stage 3 completes, stage 4 begins
    checkingTimers.push(setTimeout(() => {
        if (runId !== activeVerificationRun) return;
        setPipelineStep(3, 'done');
        setPipelineStep(4, 'running');
        setTerminalState('judging');
    }, 2000));

    // Stage 4 completes, stage 5 begins
    checkingTimers.push(setTimeout(() => {
        if (runId !== activeVerificationRun) return;
        setPipelineStep(4, 'done');
        setPipelineStep(5, 'running');
        setTerminalState('finalizing');
    }, 4000));

    // Verdict reached -> Results
    try {
        const result = await verificationRequest;
        if (runId !== activeVerificationRun) return;
        setPipelineAllComplete();
        renderDynamicProcessingLog(result);
        setTerminalState('complete');
        setCurrentResult(result);
        addToHistory(result.original_claim, result.verdict);
        switchView('results');
    } catch (error) {
        if (runId !== activeVerificationRun) return;
        clearCheckingTimers();
        alert(error.message || 'Verification could not be completed.');
        resetAndGoHome();
    }
}
