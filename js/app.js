/* ============================================================
   TruthLens · js/app.js
   Boot sequence. Loaded last, after every other script.

   State lives only in memory (js/state.js), so this runs on every
   open with nothing carried over — which is exactly the required
   reset-on-reopen behaviour. Nothing here starts a verification.
   ============================================================ */

function initInputs() {
    const ta = document.getElementById('claim-input');
    if (ta) ta.addEventListener('input', refreshInputMeta);
    refreshInputMeta();
    renderHistory();
}

/* The fresh Verify page: claim block centered, checking card hidden,
   no claim, and the pipeline screen showing the labelled example. */
function initFreshState() {
    clearCurrentClaim();
    clearCheckingTimers();
    setInputMode('text');
    resetSideCheckingToIdle();
    setClaimBlockCentered(true);
    resetPipelineSteps();
    setTerminalState('default');
    setTargetClaim(DEFAULT_CLAIM, true);
}

function initApp() {
    initInputs();
    initFreshState();
    if (typeof renderVerdictCard === 'function') renderVerdictCard();
    switchView('home');
}

initApp();
