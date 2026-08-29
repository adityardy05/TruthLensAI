# TruthLens — Current Claim State & Verify Transition

## Files
Use:
- `truthlens_final_production_ready_application.html` — existing frontend, source of truth
- `TruthLens_Cowork_Instructions.md` — working rules

Do not rebuild or redesign the application.

## Goal
Implement the following exact UX/state behavior while preserving the existing Stitch design.

### 1. Initial Verify/Home
On a fresh application open:
- Show the Verify/Home page.
- The entire “What claim would you like to check?” input block is centered.
- Show Text / Image / URL tabs, input, language indicator, Analyze, and Recent Claims.
- Do NOT show an active checking box.
- Do NOT show a mock/current claim in checking.
- Do NOT run any pipeline automatically.

### 2. Analyze transition — Verify page only
When the user submits a valid claim:
- The centered claim-input block slowly and smoothly transitions toward the left.
- The existing checking box/grid dynamically appears on the right.
- This transition happens ONLY on the Verify page.
- Do not create a transition page or new route.
- Do not reproduce this transition on How It Works, Evidence, About, or Results.
- The checking box must contain the user’s actual submitted claim.

Desired sequence:
`centered claim block → slow left transition + checking box appears → checking stages run`

### 3. Keep the existing checking box/grid
Do not delete or duplicate the existing checking component.
Do not create a new one.
Use the existing `#how-it-works` / side checking component and its existing styling/animation.

Before submission it is hidden.
After Analyze it becomes visible and active.

### 4. Full verification process
After the Verify-page checking sequence reaches the full verification stage:
- Use the existing `#view-checking`.
- Preserve its “Verification in Progress” UI.
- Preserve Target Claim, Analysis Pipeline, five stages, Agent Processing Log, animations, and layout.
- Do not create another verification screen.
- Then continue to the existing Results view.

Flow:
`Verify → dynamic side checking → existing full Verification in Progress → Results`

### 5. Current claim state must persist across ALL existing views
After a claim is submitted, its current verification information must remain available while navigating between:
- Verify
- How It Works
- Evidence
- About
- Results

Navigation must not clear or replace the current claim.

The state should include whatever existing frontend data is needed for the latest verification, including where applicable:
- current claim text
- input mode
- checking status/stage
- latest result/verdict
- evidence
- reviewers
- subclaims
- other existing claim-dependent UI data

Reuse existing state structures where possible. Do not invent backend data.

### 6. New claim replaces the current state
The current claim remains current until the user submits a new claim.

When a new claim is submitted:
- replace/reset the previous current verification state
- update checking, results, evidence, reviewers, subclaims, etc. to the new claim
- reset the verification pipeline for the new claim
- Recent Claims history may remain according to the existing behavior

### 7. How It Works
How It Works must show information related to the user’s current claim when a current claim exists.

Important:
- Do NOT replay the Verify-page centered-to-left transition here.
- Do NOT restart the checking animation merely because the user navigates here.
- Do NOT clear the current claim.
- If there is no current claim, show the normal neutral/intro state.
- Do not show a fake claim as current.

### 8. Evidence / About / Results
- Evidence must remain associated with the current claim.
- About must not reset current claim state.
- Results must remain associated with the latest submitted claim.
- Navigation between these views must not reset the current claim.
- Do not redesign any of these pages.

### 9. Close/reopen reset
Current claim state must exist only for the current open application session.

When the application is closed and opened again:
- reset current claim details
- start from clean Verify/Home
- no old current claim is restored
- no checking pipeline auto-starts

Do NOT use:
- localStorage
- sessionStorage
- IndexedDB
- cookies
- disk persistence
- server persistence

Do not add persistence merely to survive reopening.

### 10. Image behavior must not regress
Preserve the existing image fix:
- select image
- preview appears inside the existing upload placeholder
- OCR behavior remains
- remove/replace remains
- 5 MB validation remains
- Analyze must NEVER reopen the file picker

### 11. Text / URL
Preserve existing Text and URL behavior.

## Existing code to inspect first
Inspect the actual current implementations of:
- `switchView()`
- `analyzeClaim()`
- `getCurrentClaimText()`
- `simulateCheckingProcess()`
- `resetCheckingPipeline()`
- `resetAndGoHome()`
- `hideHomeCheckingPanel()`
- `setInputMode()`
- `handleImageUpload()`
- `removeImage()`

Inspect:
- `#how-it-works`
- `#view-checking`
- `#side-check-claim`
- `#checking-claim`
- `#side-step-1` through `#side-step-5`
- `#view-results`

Do not assume the current file matches earlier versions.

## Avoid previous mistakes
Do NOT:
- show the checking box active on initial load
- immediately replace Home with only the full checking screen
- use only `#how-it-works` as a substitute for `#view-checking`
- remove either checking component
- duplicate checking markup

Correct behavior:
- Initial: centered Verify input, no active checking box
- Analyze: slow left transition on Verify + dynamic right checking box
- Then: existing full Verification in Progress
- Then: Results

## Visual preservation
The Stitch-generated design is the source of truth.

Do not:
- redesign
- regenerate
- change fonts, colors, typography, spacing, design tokens, cards, navbar, Results, Evidence, About
- change existing animations unnecessarily
- introduce React, Next.js, Vite, Streamlit, or another frontend framework
- add unnecessary dependencies

Prefer minimal JavaScript/state/visibility changes. Only minimal CSS/layout changes needed for the centered-to-left transition are allowed.

## Testing
Test in the real browser if possible.

1. Fresh open:
   - centered input
   - checking box hidden
   - no mock claim
   - no auto pipeline

2. Submit text claim:
   - centered block slowly moves left
   - checking box dynamically appears right
   - actual claim appears
   - stages animate
   - full existing Verification in Progress appears
   - Results appears

3. Navigation:
   - Verify → How It Works → Evidence → About → Results → Verify
   - current claim and related information remain consistent
   - no navigation restarts checking

4. New claim:
   - previous current state is replaced
   - new verification starts cleanly
   - history may retain older claims

5. Close/reopen:
   - old current claim is gone
   - clean initial state

6. Image:
   - preview inside placeholder
   - Analyze does not reopen file picker

7. URL:
   - valid URL flow works

Also verify:
- no duplicate IDs
- valid JS
- balanced HTML
- existing CSS/keyframes preserved
- no unwanted dependencies

## Stop condition
Once these requirements work, stop. Do not perform unrelated cleanup, refactoring, or visual improvements.
