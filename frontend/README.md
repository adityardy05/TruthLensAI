# TruthLens — Frontend

Multi-agent misinformation-detection UI. Vanilla HTML, Tailwind (CDN) and
plain JavaScript. No build step, no bundler, no framework.

## Running it

Open `index.html` directly in a browser, or serve the folder over any static
server:

```
python -m http.server 8000
```

Both work. The scripts are classic `<script src>` tags rather than ES modules
specifically so that opening the file from disk behaves the same as serving it.
An internet connection is needed for the Tailwind, jsPDF and Google Fonts CDNs.

## Layout

```
index.html              markup for all four views + the Tailwind token config
css/
  main.css              base body, glass panels, neon accents, view show/hide
  components.css        agent pulse, pipeline flow, terminal cursor, download
                        menu, PDF drop animation, claim-block transition
  responsive.css        the lg-and-up centered offset for the claim block
data/
  mock-data.js          every hard-coded demo value; no DOM access
js/
  state.js              the shared in-memory current claim + history; no DOM
  navigation.js         switchView, navbar highlighting, nav entry points
  input.js              Text / Image / URL modes, char counter, language badge
  image.js              upload validation, preview, server OCR result, remove
  verification.js       checking components and live API-backed pipeline
  results.js            verdict readout, evidence accordion, Recent Claims
  export.js             PDF export via jsPDF
  app.js               boot sequence
truthlens_final_production_ready_application.html
                        the original single-file version, kept for reference
```

Script load order is fixed by `index.html`: data, then state, then the view
modules, then `app.js` last.

## The two checking components

These are separate on purpose and should stay that way.

The **Verify-page checking card** is the small panel that appears next to the
claim block after Analyze and runs a five-stage animation. Its element id is
`#how-it-works` for historical reasons — it is not the How It Works page. Reach
it through `showVerifyCheckingCard()` / `hideVerifyCheckingCard()`.

The **full verification presentation** is `#view-checking`: Verification in
Progress, Target Claim, Analysis Pipeline, the five stages, and the Agent
Processing Log. It serves two roles. During a run it is the live screen. Opened
from the How It Works nav item it is a read-only presentation of the process —
`renderVerificationPresentation()` starts nothing and restarts nothing, and it
never replays the Verify-page transition.

## Current claim state

`js/state.js` holds one shared claim object, `{ text, mode, stage, verdict,
startedAt }`, with `stage` moving `side` → `checking` → `done`. Every view reads
from it, which is what keeps Verify, How It Works, Evidence, About and Results
consistent while the app is open.

- Navigation never clears it. Returning to Verify re-applies it without
  replaying the transition or restarting the animation.
- Submitting a new claim replaces it.
- "Analyze another claim" clears it explicitly.
- It is **not persisted** — no localStorage, sessionStorage, IndexedDB or
  cookies. Closing and reopening the app therefore starts clean, which is the
  intended behaviour.

Recent Claims is a separate list, also session-only.

## Backend integration

The frontend submits claims to the Flask API under `/api/v1`. The request runs
through OCR or URL extraction, retrieval, and the LangGraph reviewer pipeline.
The timers in `verification.js` only provide visual stage transitions while the
real request is running.

| What a backend would replace | Where |
| --- | --- |
| the claim being submitted | `getCurrentClaimText()` in `js/input.js` |
| URL validation | `fetchUrl()` in `js/input.js` |
| image upload | `submitVerification()` in `js/api.js` |
| OCR result rendering | `renderImageOcrResult()` in `js/verification.js` |
| stage presentation | `simulateCheckingProcess()` in `js/verification.js` |
| result rendering | `renderVerdictCard()` in `js/results.js` |

`DEFAULT_CLAIM` in `data/mock-data.js` is also duplicated as a markup default
in `index.html` (`#claim-input`, `#ocr-text`, `#url-extracted`,
`#checking-claim`) because markup cannot read from JS without a build step.
Keep them in sync.
