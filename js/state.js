/* ============================================================
   TruthLens · js/state.js
   The single shared in-memory verification state.

   Every view reads the current claim from here, which is what keeps
   Verify, How It Works, Evidence, About and Results consistent while
   the app is open.

   Deliberately NOT persisted — no localStorage, sessionStorage,
   IndexedDB, cookies or disk. Closing and reopening the application
   therefore starts from a clean Verify page with no claim, which is
   the required behaviour.

   This file touches no DOM at all. Rendering lives in the view
   modules; they call in here to ask what the current claim is.
   ============================================================ */

/* The claim currently being verified, or null when there is none.
   Shape: { text, mode, stage, verdict, startedAt }
     mode    'text' | 'image' | 'url'  — which input it came from
     stage   'side' -> 'checking' -> 'done'
     verdict filled in from the Results view once stage is 'done' */
let currentClaim = null;

/* A new submission always replaces the previous current claim. */
function setCurrentClaim(text, mode) {
    currentClaim = {
        text: text,
        mode: mode,
        stage: 'side',
        verdict: null,
        startedAt: Date.now()
    };
}

function clearCurrentClaim() {
    currentClaim = null;
}

function getCurrentClaim() {
    return currentClaim;
}

function hasCurrentClaim() {
    return currentClaim !== null;
}

/* Stage transitions. Kept as named helpers so the pipeline in
   js/verification.js never pokes at the state object directly. */
function markClaimChecking() {
    if (currentClaim) currentClaim.stage = 'checking';
}

function markClaimDone(verdict) {
    if (currentClaim) {
        currentClaim.stage = 'done';
        currentClaim.verdict = verdict;
    }
}

function isClaimRunning() {
    return !!currentClaim && currentClaim.stage === 'side';
}

function isClaimComplete() {
    return !!currentClaim && currentClaim.stage === 'done';
}

/* ---------- Recent Claims history ----------
   Separate from the current claim: submitting a new claim replaces
   the current verification but appends to this list, which is the
   existing Recent Claims behaviour. Also session-only. */
let claimHistory = [];

function addToHistory(claim) {
    claimHistory.unshift({ claim: claim, verdict: 'FALSE', time: 'just now' });
    renderHistory();
}

function getHistory() {
    return claimHistory;
}
