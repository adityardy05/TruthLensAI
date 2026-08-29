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
  image.js              upload validation, preview, mock OCR, remove
  verification.js       both checking components and the simulated pipeline
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

## This is a prototype

The verification is simulated with `setTimeout`. There is no FastAPI, LangGraph
or Ollama backend connected, and the verdict, sub-claims, sources and reviewer
findings are static.

The seams for a real backend are deliberately narrow:

| What a backend would replace | Where |
| --- | --- |
| the claim being submitted | `getCurrentClaimText()` in `js/input.js` |
| URL extraction | `fetchUrl()` in `js/input.js` |
| OCR | `handleImageUpload()` in `js/image.js` |
| language detection | `detectLanguage()` in `js/input.js` |
| stage progress events | `simulateCheckingProcess()` in `js/verification.js` |
| the verdict | `readVerdictFromResults()` in `js/results.js` |
| all demo content | `data/mock-data.js` |

`DEFAULT_CLAIM` in `data/mock-data.js` is also duplicated as a markup default
in `index.html` (`#claim-input`, `#ocr-text`, `#url-extracted`,
`#checking-claim`) because markup cannot read from JS without a build step.
Keep them in sync.
