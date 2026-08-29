# TruthLens Cowork Instructions

Use `truthlens_final_production_ready_application.html` as the sole source of truth.

Use `TruthLens_Cowork_Current_Claim_And_Transition_Prompt.md` as the implementation specification.

Rules:
- Inspect the actual current file before editing.
- Make the smallest possible changes.
- Do not rebuild or redesign.
- Preserve the Stitch visual design.
- Do not introduce Streamlit, React, Next.js, Vite, or another framework.
- Do not add unnecessary dependencies.
- Do not persist current claim state with localStorage, sessionStorage, IndexedDB, cookies, disk, or server persistence.
- Preserve all previous fixes, especially image upload, OCR, URL, PDF, evidence, reviewers, subclaims, recent claims, and results.
- Keep both existing checking components: `#how-it-works` and `#view-checking`.
- The Verify-page centered-to-left transition is ONLY for Verify.
- Current claim state must remain available across Verify, How It Works, Evidence, About, and Results during the current open application session.
- A new submitted claim replaces the current claim state.
- Closing and reopening the application resets the current claim state.
- Prefer real browser testing when available.
- Stop when the requested behavior is verified; do not perform unrelated cleanup.

Final report should contain only:
1. Files changed
2. Concise changes
3. Tests performed
4. Remaining limitations
