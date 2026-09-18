/* ============================================================
   TruthLens · js/navigation.js
   View switching, navbar highlighting and the nav entry points.

   There are exactly five destinations:
     Verify        -> #view-home
     How It Works  -> #view-checking, as a read-only presentation
     Evidence      -> #view-results, scrolled to #evidence
     About         -> #view-about
     Results       -> #view-results

   Navigation never clears the current claim. Returning to Verify
   re-applies it (js/verification.js), it does not reset it.
   ============================================================ */

const NAV_IDLE = 'text-on-surface-variant font-medium hover:text-primary transition-colors duration-200 cursor-pointer';
const NAV_ACTIVE = 'text-primary font-bold border-b-2 border-primary pb-1 cursor-pointer';

/* Which nav link is highlighted for each view. #view-checking maps to
   null because during a live verification no nav item is "current";
   arriving there via How It Works passes an explicit override. */
const NAV_MAP = {
    home: 'nav-verify',
    about: 'nav-about',
    results: 'nav-evidence',
    checking: null
};

function setActiveNav(viewId, navIdOverride) {
    const activeId = navIdOverride || NAV_MAP[viewId] || null;
    ['nav-verify', 'nav-how', 'nav-evidence', 'nav-about'].forEach(id => {
        const link = document.getElementById(id);
        if (link) link.className = (id === activeId) ? NAV_ACTIVE : NAV_IDLE;
    });
}

/* Show one view and hide the rest.
   navIdOverride forces a navbar link active regardless of the view —
   used by How It Works, which shares #view-checking with the live run. */
function switchView(viewId, navIdOverride) {
    document.querySelectorAll('.view-content').forEach(view => {
        view.classList.add('view-hidden');
        view.classList.remove('fade-in');
    });

    const targetView = document.getElementById('view-' + viewId);
    if (!targetView) return;
    targetView.classList.remove('view-hidden');

    // Trigger reflow so the entrance animation restarts
    void targetView.offsetWidth;
    targetView.classList.add('fade-in');

    setActiveNav(viewId, navIdOverride);

    // Returning to Verify re-applies the current claim (or the clean
    // centered state when there is none). It must not clear it.
    if (viewId === 'home') restoreVerifyForCurrentClaim();
    if (viewId === 'results' && typeof renderVerdictCard === 'function') renderVerdictCard();

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* Smooth-scroll to a section; fall back to a view if it is not visible. */
function navScroll(sectionId, fallbackView) {
    const el = document.getElementById(sectionId);
    const parentView = el ? el.closest('.view-content') : null;
    const isVisible = parentView && !parentView.classList.contains('view-hidden') && !el.classList.contains('hidden');

    if (isVisible) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        return;
    }

    // Section lives in a hidden view or is hidden — go to the fallback view first
    switchView(fallbackView);
    const target = document.getElementById(sectionId);
    if (target && target.classList.contains('hidden')) {
        target.classList.remove('hidden');
        target.classList.remove('fade-in');
        void target.offsetWidth;
        target.classList.add('fade-in');
    }
    const targetParent = target ? target.closest('.view-content') : null;
    if (target && targetParent && !targetParent.classList.contains('view-hidden')) {
        setTimeout(() => target.scrollIntoView({ behavior: 'smooth', block: 'start' }), 350);
    }
}

/* ---------- How It Works ----------
   When there is a current claim, opens the full verification
   presentation (#view-checking: Verification in Progress /
   Target Claim / Analysis Pipeline / the five stages / Agent
   Processing Log).

   When there is no current claim, behaves like Evidence navigation
   and redirects to the Verify page.

   It is read-only. It does not start, restart or continue a
   verification, and it never replays the Verify-page
   centered-to-left transition. */
function navHowItWorks() {
    if (!hasCurrentClaim()) {
        switchView('home');
        return;
    }
    renderVerificationPresentation();
    switchView('checking', 'nav-how');
}

/* ---------- Evidence ----------
   #evidence lives inside the Results view, so it is only reachable
   once the current claim actually has a result to show evidence for. */
function navEvidence() {
    if (isClaimComplete()) {
        navScroll('evidence', 'results');
    } else {
        navScroll('evidence', 'home');
    }
}

function startVerifying() {
    switchView('home');
    setInputMode('text');
    const ta = document.getElementById('claim-input');
    if (ta) setTimeout(() => ta.focus(), 350);
}
