# TruthLens Frontend Reorganization & How It Works

Use the existing `truthlens_final_production_ready_application.html` as the source of truth. Do not rebuild or redesign it.

## Required UX

### Verify
Initial state:
- Verify/Home claim block is centered.
- Text/Image/URL, Analyze, language indicator and Recent Claims remain.
- Active checking card is hidden.
- No mock active claim and no automatic pipeline.

After Analyze:
- On the Verify page ONLY, slowly/smoothly move the claim block left.
- Dynamically reveal the existing small checking box on the right.
- Put the actual submitted claim into it.
- Run its existing five-stage animation.
- Then enter the existing full `#view-checking` “Verification in Progress” screen.
- Continue the existing pipeline and then Results.

Do not create a new transition page.

### How It Works
The How It Works page must show the FULL verification-process presentation represented by the approved first screenshot, not the small live checking card from the Verify page.

It should present:
- Verification in Progress
- Target Claim
- Analysis Pipeline
- Breaking down the claim
- Checking sources
- Consulting reviewers
- Weighing confidence
- Reaching a verdict
- Agent Processing Log

It must NOT replay the Verify-page left transition, restart checking, or start a new verification.

If a current claim exists, show information belonging to that current claim. If none exists, show a neutral/default state.

### Current claim
Create/use ONE shared in-memory current-verification state.

It must remain consistent across:
- Verify
- How It Works
- Evidence
- About
- Results

Navigation must not clear it.

Submitting a new claim replaces/resets the current verification state. Recent Claims history can remain as existing behavior.

Closing and reopening the application must reset current claim state. Do not use localStorage, sessionStorage, IndexedDB, cookies, or other persistence.

## Frontend organization

Reorganize the monolithic HTML into a maintainable vanilla frontend structure, for example:

TruthLens_frontend/
├── index.html
├── assets/
│   ├── images/
│   └── icons/
├── css/
│   ├── main.css
│   ├── components.css
│   └── responsive.css
├── js/
│   ├── app.js
│   ├── state.js
│   ├── navigation.js
│   ├── verification.js
│   ├── input.js
│   ├── image.js
│   ├── results.js
│   ├── evidence.js
│   └── export.js
├── components/
│   ├── navbar.html
│   ├── verify.html
│   ├── how-it-works.html
│   ├── checking.html
│   ├── verification.html
│   ├── evidence.html
│   ├── results.html
│   └── about.html
├── data/
│   └── mock-data.js
└── README.md

This is guidance, not permission to blindly split files. Choose a simple reliable structure that works with the existing local server. Do NOT introduce React, Next.js, Vite, Streamlit, or another framework.

Do not simply copy the 1900-line file into separate files. Separate responsibilities: shared state, navigation, verification, inputs/images, results/evidence/export, components, and mock data.

## Preserve
Do not regress:
- Stitch visual design
- fonts/colors/spacing/design tokens/keyframes
- Text/Image/URL
- image preview inside upload placeholder
- OCR mock
- 5 MB validation
- image remove/replace
- Analyze never reopening the file picker
- language detection
- checking/full verification pipeline
- Results
- Evidence
- reviewers
- subclaims
- Recent Claims
- PDF export

Do not connect FastAPI/LangGraph/Ollama in this task. Organize the frontend so backend integration can be added later.

## Important
Keep the two checking concepts separate:
1. Verify-page small dynamic checking card.
2. How It Works full verification-process presentation.

Do not delete or duplicate existing components unnecessarily.

## Testing
Test:
1. Fresh open: centered Verify, no active checking.
2. Analyze: slow left movement + dynamic right checking card + actual claim.
3. Full Verification in Progress → Results.
4. Verify → How It Works → Evidence → About → Results → Verify: current claim remains.
5. New claim replaces old current state.
6. Close/reopen resets current state.
7. Image flow works and picker does not reopen.
8. URL flow works.
9. No duplicate IDs, valid JS, no broken imports/assets.

Make the smallest safe changes and stop when requirements pass. Do not perform unrelated visual improvements.
