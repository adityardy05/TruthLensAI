# TruthLens Cowork Instructions

Read `TruthLens_Frontend_Reorganization_Prompt.md` completely before editing.

Use the existing `truthlens_final_production_ready_application.html` as the source of truth.

- Inspect the actual current code first.
- Preserve the approved Stitch design and all existing functionality.
- Make the smallest safe changes.
- Do not rebuild or redesign.
- Do not introduce React, Next.js, Vite, Streamlit, or another framework.
- Do not add unnecessary dependencies.
- Do not persist current claim state with localStorage, sessionStorage, IndexedDB, cookies, or other persistent storage.
- Keep the Verify-page dynamic checking card separate from the full verification presentation used by How It Works.
- The centered-to-left transition happens only on Verify.
- How It Works must not replay the transition or restart checking.
- Keep current claim state in memory across all existing views; a new claim replaces it; closing/reopening resets it.
- Preserve the image upload/OCR fix, Text/Image/URL, Results, Evidence, reviewers, subclaims, Recent Claims, PDF export, and existing animations.

Do not blindly split the monolithic HTML. Organize responsibilities while keeping behavior equivalent and ensuring the local server still works.

Test the actual browser if available. Verify initial state, Analyze transition, checking card, full verification, Results, How It Works, navigation persistence, new-claim replacement, close/reopen reset, image flow, URL flow, duplicate IDs, valid JS, and imports/assets.

Final report:
1. Files changed
2. Changes made
3. Tests performed
4. Remaining limitations

Stop after the requested work. Do not perform unrelated cleanup.
