# TruthLens — Claude Code Fix Prompt
**Project 45 | KMIT 2025–26**
**Version:** 5.0 — Targeted Fixes on Base File
**Start with:** `TruthLens_Base.html` — do NOT rewrite from scratch. Fix only what is listed below.
   
   
---

## CONTEXT

The base file `TruthLens_Base.html` already has:
- Full 4-view SPA (Home, Checking, Results, About)
- Animated pipeline steps with `setStepActive` / `setStepDone`
- Terminal log with staggered lines
- Confidence ring SVG with `stroke-dashoffset` animation
- About page with architecture diagram + tech stack grid
- `switchView()` fade transitions
- `exportReport()` button stub
- `resetApp()` back to home
- Exact design tokens: `#f9f9f7` background, `glass-panel`, `agent-pulse`, `pipeline-flow`
- Fonts: Plus Jakarta Sans + JetBrains Mono
- Icons: Material Symbols Outlined
- Footer with Project 45 / KMIT credit

**Do not touch anything that already works.**
**Only fix the 8 gaps listed below.**

---

## FIX 1 — Navbar cleanup

**Current:** Navbar has "Platform", "Academy", "Support", "Sign In" links that are irrelevant.

**Replace navbar center links with:**
```
Verify | How It Works | Evidence | About
```

- `Verify` → `switchView('home')`
- `How It Works` → smooth scroll to `#how-it-works` section on home page
- `Evidence` → smooth scroll to `#evidence` section on results page (or home if results not shown)
- `About` → `switchView('about')`

**Remove:** "Sign In" button entirely.

**Keep:** "Get Started" right button → `switchView('home')` and focus textarea.

**Active state:** whichever page is currently shown gets `border-b-2 border-primary font-bold` underline.

---

## FIX 2 — Real PDF export using jsPDF

**Add to `<head>`:**
```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
```

**Replace the `exportReport()` stub function with this full implementation:**

```javascript
function exportReport() {
  const btn = document.getElementById('btn-export');
  const text = document.getElementById('export-text');
  const icon = document.getElementById('export-icon');

  text.innerText = 'Exporting...';
  icon.innerText = 'hourglass_empty';
  btn.classList.add('opacity-70', 'pointer-events-none');

  setTimeout(() => {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    const claim = document.getElementById('claim-input')?.value || 'Claim not available';

    // Header
    doc.setFontSize(20);
    doc.setFont('helvetica', 'bold');
    doc.text('TruthLens Verification Report', 20, 22);

    doc.setFontSize(9);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(120);
    doc.text(`Generated: ${new Date().toLocaleString()}`, 20, 30);
    doc.text('Project 45 · KMIT · 2025-26 · LangGraph · DeepSeek · FAISS · Tavily', 20, 36);

    doc.setDrawColor(220);
    doc.line(20, 41, 190, 41);

    // Claim
    doc.setTextColor(0);
    doc.setFontSize(11);
    doc.setFont('helvetica', 'bold');
    doc.text('Claim Analyzed', 20, 51);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10);
    const claimLines = doc.splitTextToSize(claim, 170);
    doc.text(claimLines, 20, 59);

    let y = 59 + claimLines.length * 6 + 6;
    doc.line(20, y, 190, y);
    y += 8;

    // Verdict
    doc.setFontSize(11);
    doc.setFont('helvetica', 'bold');
    doc.text('Verdict', 20, y);
    y += 8;
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10);
    doc.text('Result:       FALSE', 20, y); y += 6;
    doc.text('Confidence:   91%', 20, y); y += 6;
    doc.text('Language:     English', 20, y); y += 6;
    doc.text('Sources:      9 retrieved', 20, y); y += 6;
    doc.text('Sub-claims:   3 decomposed', 20, y); y += 8;

    doc.line(20, y, 190, y); y += 8;

    // Sub-claims
    doc.setFontSize(11);
    doc.setFont('helvetica', 'bold');
    doc.text('Sub-Claim Breakdown', 20, y); y += 8;
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10);
    const subclaims = [
      ['01', 'Government made an official announcement', 'NOT SUPPORTED'],
      ['02', 'Electricity would be free for all households', 'NOT SUPPORTED'],
      ['03', 'Policy takes effect next month', 'UNVERIFIED'],
    ];
    subclaims.forEach(([num, text, verdict]) => {
      doc.text(`${num}  ${text}`, 20, y); y += 5;
      doc.setTextColor(150);
      doc.text(`     → ${verdict}`, 20, y); y += 7;
      doc.setTextColor(0);
    });

    doc.line(20, y, 190, y); y += 8;

    // Sources
    doc.setFontSize(11);
    doc.setFont('helvetica', 'bold');
    doc.text('Sources', 20, y); y += 8;
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10);
    const sources = [
      ['ndtv.com', '0.91', 'high factual', 'contradicts'],
      ['altnews.in', '0.87', 'high factual', 'contradicts'],
      ['randomblog.net', '0.34', 'low factual', 'supports'],
    ];
    sources.forEach(([domain, score, rating, rel]) => {
      doc.text(`• ${domain}   Score: ${score}   ${rating}   — ${rel}`, 20, y); y += 7;
    });

    doc.line(20, y, 190, y); y += 8;

    // Reviewer findings
    doc.setFontSize(11);
    doc.setFont('helvetica', 'bold');
    doc.text('Reviewer Findings', 20, y); y += 8;
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10);
    const agents = [
      ['Fact Checker', 'Scanned 42 recent government press releases. No mention of nationwide free electricity found. Found a localized subsidy program in Region 4 only.'],
      ['Logical Analyst', 'Claim extrapolates a regional program to a national announcement. Classic "False Generalization" pattern.'],
      ['Bias Detector', 'High emotional language indicates sensationalism. Phrasing designed to elicit reactions rather than inform.'],
    ];
    agents.forEach(([name, finding]) => {
      doc.setFont('helvetica', 'bold');
      doc.text(name + ':', 20, y); y += 6;
      doc.setFont('helvetica', 'normal');
      const lines = doc.splitTextToSize(finding, 162);
      doc.text(lines, 25, y); y += lines.length * 5 + 4;
    });

    // Footer
    doc.setFontSize(8);
    doc.setTextColor(160);
    doc.text('TruthLens · Project 45 · KMIT · LangGraph · DeepSeek · FAISS · Tavily · Ollama', 20, 284);

    doc.save(`TruthLens_Report_${Date.now()}.pdf`);

    text.innerText = '✓ Downloaded';
    icon.innerText = 'check';
    btn.classList.remove('opacity-70');
    btn.classList.add('bg-tertiary-fixed/20', 'text-tertiary-container');

    setTimeout(() => {
      text.innerText = 'Download Full Report';
      icon.innerText = 'download';
      btn.classList.remove('bg-tertiary-fixed/20', 'text-tertiary-container', 'pointer-events-none');
    }, 3000);
  }, 800);
}
```

---

## FIX 3 — Three reviewer cards in Results view

**Current:** Results has only "Agent Deliberation" feed with 2 generic message bubbles.

**Add a full "What the reviewers say" section** after the Agent Deliberation section in `view-results`:

```html
<!-- What the reviewers say -->
<section class="space-y-6">
  <h2 class="font-headline-lg text-headline-lg text-primary border-b border-glass-stroke pb-4">
    What the reviewers say
  </h2>
  <div class="space-y-4">

    <!-- Fact Checker -->
    <div class="flex gap-4 p-5 rounded-xl bg-surface-container-low/50 border border-glass-stroke hover:bg-surface-container-low transition-colors">
      <div class="flex-shrink-0 w-10 h-10 rounded-full bg-primary-fixed flex items-center justify-center text-primary border border-outline-variant agent-pulse">
        <span class="material-symbols-outlined text-sm">fact_check</span>
      </div>
      <div>
        <div class="flex items-center gap-3 mb-1">
          <span class="font-label-md text-primary font-bold">Fact Checker</span>
          <span class="text-xs px-2 py-0.5 rounded-full bg-primary-fixed text-on-primary-fixed font-label-sm">High confidence</span>
          <span class="text-xs text-text-muted">Sources: 1, 2</span>
        </div>
        <p class="font-body-md text-sm text-on-surface-variant">
          Scanned 42 recent government press releases and main utility provider announcements.
          No mention of nationwide free electricity found. Did find a localized subsidy program
          in Region 4 for low-income households.
        </p>
      </div>
    </div>

    <!-- Logical Analyst -->
    <div class="flex gap-4 p-5 rounded-xl bg-surface-container-low/50 border border-glass-stroke hover:bg-surface-container-low transition-colors">
      <div class="flex-shrink-0 w-10 h-10 rounded-full bg-secondary-container flex items-center justify-center text-secondary border border-outline-variant agent-pulse" style="animation-delay: 0.5s;">
        <span class="material-symbols-outlined text-sm">psychology</span>
      </div>
      <div>
        <div class="flex items-center gap-3 mb-1">
          <span class="font-label-md text-secondary font-bold">Logical Analyst</span>
          <span class="text-xs px-2 py-0.5 rounded-full bg-secondary-fixed text-on-secondary-fixed font-label-sm">High confidence</span>
          <span class="text-xs text-text-muted">Sources: 1</span>
        </div>
        <p class="font-body-md text-sm text-on-surface-variant">
          Based on evidence, the claim extrapolates a highly specific regional program to a
          generalized national announcement. This represents a classic "False Generalization" pattern.
        </p>
      </div>
    </div>

    <!-- Bias Detector -->
    <div class="flex gap-4 p-5 rounded-xl bg-surface-container-low/50 border border-glass-stroke hover:bg-surface-container-low transition-colors">
      <div class="flex-shrink-0 w-10 h-10 rounded-full bg-tertiary-fixed/30 flex items-center justify-center text-tertiary-container border border-outline-variant agent-pulse" style="animation-delay: 1s;">
        <span class="material-symbols-outlined text-sm">visibility</span>
      </div>
      <div>
        <div class="flex items-center gap-3 mb-1">
          <span class="font-label-md text-tertiary-container font-bold">Bias Detector</span>
          <span class="text-xs px-2 py-0.5 rounded-full bg-tertiary-fixed/20 text-tertiary-container font-label-sm">Medium confidence</span>
          <span class="text-xs text-text-muted">Sources: 2</span>
        </div>
        <p class="font-body-md text-sm text-on-surface-variant">
          High emotional language used in the source blog post indicates sensationalism rather
          than objective reporting. The framing is designed to elicit strong reactions rather than inform.
        </p>
      </div>
    </div>

  </div>
</section>
```

---

## FIX 4 — Expandable source cards

**Current:** Source Credibility section shows only two generic rows (Gov. Official Portal, Social Media Post).

**Replace** the Source Credibility glass-panel content with real expandable source cards:

```html
<div class="glass-panel p-6 rounded-2xl border border-glass-stroke">
  <h3 class="font-label-md text-label-md text-primary font-bold mb-2 uppercase tracking-wider">Sources</h3>
  <div class="flex gap-4 text-xs font-label-sm text-text-muted mb-4">
    <span>Contradicting: 2</span>
    <span>Supporting: 1</span>
  </div>
  <div class="space-y-3" id="source-cards">

    <!-- Source 1 -->
    <div class="border border-glass-stroke rounded-xl overflow-hidden">
      <div class="flex justify-between items-center p-4 bg-surface cursor-pointer hover:bg-surface-container-low transition-colors" onclick="toggleSource('src1')">
        <div>
          <span class="font-label-md text-primary font-bold">ndtv.com</span>
          <span class="ml-3 text-xs text-on-tertiary-container bg-tertiary-fixed/20 px-2 py-0.5 rounded font-label-sm">high factual</span>
          <span class="ml-2 text-xs text-error bg-error/10 px-2 py-0.5 rounded font-label-sm">Contradicts</span>
        </div>
        <div class="flex items-center gap-3">
          <span class="font-label-md text-text-muted">0.91</span>
          <span class="material-symbols-outlined text-sm text-text-muted" id="src1-icon">expand_more</span>
        </div>
      </div>
      <div class="w-full h-1 bg-surface-variant"><div class="h-full bg-on-tertiary-container rounded-r" style="width:91%"></div></div>
      <div class="hidden p-4 bg-surface-container-low border-t border-glass-stroke space-y-2" id="src1-body">
        <p class="font-body-md text-sm text-on-surface-variant italic">
          "Ministry officials confirmed no such nationwide policy exists, calling the viral claim 'completely fabricated'."
        </p>
        <div class="flex items-center gap-4 text-xs text-text-muted font-label-sm mt-2">
          <span>Published: 24 Aug 2026</span>
          <a href="#" class="flex items-center gap-1 text-primary hover:underline">
            Open source <span class="material-symbols-outlined text-sm">open_in_new</span>
          </a>
        </div>
      </div>
    </div>

    <!-- Source 2 -->
    <div class="border border-glass-stroke rounded-xl overflow-hidden">
      <div class="flex justify-between items-center p-4 bg-surface cursor-pointer hover:bg-surface-container-low transition-colors" onclick="toggleSource('src2')">
        <div>
          <span class="font-label-md text-primary font-bold">altnews.in</span>
          <span class="ml-3 text-xs text-on-tertiary-container bg-tertiary-fixed/20 px-2 py-0.5 rounded font-label-sm">high factual</span>
          <span class="ml-2 text-xs text-error bg-error/10 px-2 py-0.5 rounded font-label-sm">Contradicts</span>
        </div>
        <div class="flex items-center gap-3">
          <span class="font-label-md text-text-muted">0.87</span>
          <span class="material-symbols-outlined text-sm text-text-muted" id="src2-icon">expand_more</span>
        </div>
      </div>
      <div class="w-full h-1 bg-surface-variant"><div class="h-full bg-on-tertiary-container rounded-r" style="width:87%"></div></div>
      <div class="hidden p-4 bg-surface-container-low border-t border-glass-stroke space-y-2" id="src2-body">
        <p class="font-body-md text-sm text-on-surface-variant italic">
          "AltNews investigation found the claim originated from a regional subsidy misquoted on social media."
        </p>
        <div class="flex items-center gap-4 text-xs text-text-muted font-label-sm mt-2">
          <span>Published: 23 Aug 2026</span>
          <a href="#" class="flex items-center gap-1 text-primary hover:underline">
            Open source <span class="material-symbols-outlined text-sm">open_in_new</span>
          </a>
        </div>
      </div>
    </div>

    <!-- Source 3 -->
    <div class="border border-glass-stroke rounded-xl overflow-hidden">
      <div class="flex justify-between items-center p-4 bg-surface cursor-pointer hover:bg-surface-container-low transition-colors" onclick="toggleSource('src3')">
        <div>
          <span class="font-label-md text-primary font-bold">randomblog.net</span>
          <span class="ml-3 text-xs text-error bg-error/10 px-2 py-0.5 rounded font-label-sm">low factual</span>
          <span class="ml-2 text-xs text-on-tertiary-container bg-tertiary-fixed/20 px-2 py-0.5 rounded font-label-sm">Supports</span>
        </div>
        <div class="flex items-center gap-3">
          <span class="font-label-md text-text-muted">0.34</span>
          <span class="material-symbols-outlined text-sm text-text-muted" id="src3-icon">expand_more</span>
        </div>
      </div>
      <div class="w-full h-1 bg-surface-variant"><div class="h-full bg-error rounded-r" style="width:34%"></div></div>
      <div class="hidden p-4 bg-surface-container-low border-t border-glass-stroke space-y-2" id="src3-body">
        <p class="font-body-md text-sm text-on-surface-variant italic">
          "Source unverified. Domain registered 2 months ago with no editorial history found."
        </p>
        <div class="flex items-center gap-4 text-xs text-text-muted font-label-sm mt-2">
          <span>Published: 22 Aug 2026</span>
          <a href="#" class="flex items-center gap-1 text-primary hover:underline">
            Open source <span class="material-symbols-outlined text-sm">open_in_new</span>
          </a>
        </div>
      </div>
    </div>

  </div>
</div>
```

**Add this JS function:**
```javascript
function toggleSource(id) {
  const body = document.getElementById(id + '-body');
  const icon = document.getElementById(id + '-icon');
  const isHidden = body.classList.contains('hidden');
  body.classList.toggle('hidden', !isHidden);
  icon.innerText = isHidden ? 'expand_less' : 'expand_more';
}
```

---

## FIX 5 — Sub-claim breakdown in Results

**Add this section** inside `view-results`, after the verdict header card and before "Live Analysis Dashboard":

```html
<!-- Sub-Claim Breakdown -->
<section class="space-y-4">
  <h2 class="font-headline-lg text-headline-lg text-primary border-b border-glass-stroke pb-4">Claim Breakdown</h2>
  <div class="glass-panel p-6 rounded-2xl border border-glass-stroke space-y-4">

    <div class="flex items-start gap-4 p-4 rounded-xl bg-surface border border-glass-stroke">
      <span class="font-label-md text-text-muted shrink-0">01</span>
      <div class="flex-1">
        <p class="font-body-md text-on-surface">Government made an official announcement</p>
        <div class="flex items-center gap-2 mt-1">
          <span class="material-symbols-outlined text-error text-sm">close</span>
          <span class="text-xs text-error font-label-sm">Not supported</span>
        </div>
      </div>
    </div>

    <div class="flex items-start gap-4 p-4 rounded-xl bg-surface border border-glass-stroke">
      <span class="font-label-md text-text-muted shrink-0">02</span>
      <div class="flex-1">
        <p class="font-body-md text-on-surface">Electricity would be free for all households</p>
        <div class="flex items-center gap-2 mt-1">
          <span class="material-symbols-outlined text-error text-sm">close</span>
          <span class="text-xs text-error font-label-sm">Not supported</span>
        </div>
      </div>
    </div>

    <div class="flex items-start gap-4 p-4 rounded-xl bg-surface border border-glass-stroke">
      <span class="font-label-md text-text-muted shrink-0">03</span>
      <div class="flex-1">
        <p class="font-body-md text-on-surface">Policy takes effect next month</p>
        <div class="flex items-center gap-2 mt-1">
          <span class="material-symbols-outlined text-outline text-sm">help</span>
          <span class="text-xs text-text-muted font-label-sm">Unverified</span>
        </div>
      </div>
    </div>

  </div>
</section>
```

---

## FIX 6 — Recent claims list updates after analysis

**Current:** Recent section is a static button that never changes.

**Replace** the recent section in `view-home` with:

```html
<div class="text-center">
  <span class="font-label-sm text-label-sm text-surface-tint uppercase tracking-wider block mb-2">recent</span>
  <div id="recent-empty" class="w-full py-3 glass-panel rounded-lg font-body-md text-on-surface-variant flex items-center justify-center gap-2">
    claims you check will show up here.
    <span class="material-symbols-outlined text-sm">expand_more</span>
  </div>
  <div id="recent-list" class="space-y-2 hidden"></div>
</div>
```

**Add these JS functions and update `analyzeClaim()`:**

```javascript
// History store
let claimHistory = [];

function addToHistory(claim) {
  claimHistory.unshift({ claim, verdict: 'FALSE', time: 'just now' });
  renderHistory();
}

function renderHistory() {
  const empty = document.getElementById('recent-empty');
  const list = document.getElementById('recent-list');
  if (!list) return;
  if (claimHistory.length === 0) {
    empty.classList.remove('hidden');
    list.classList.add('hidden');
    return;
  }
  empty.classList.add('hidden');
  list.classList.remove('hidden');
  list.innerHTML = claimHistory.slice(0, 5).map(h => `
    <div class="w-full py-3 px-4 glass-panel rounded-lg font-body-md flex items-center gap-3 cursor-pointer hover:bg-surface-container-low transition-colors" onclick="switchView('results')">
      <div class="w-2 h-2 rounded-full bg-error shrink-0"></div>
      <span class="flex-1 text-left text-on-surface truncate text-sm">${h.claim}</span>
      <span class="font-label-sm text-text-muted text-xs shrink-0">${h.time}</span>
    </div>
  `).join('');
}

// Update analyzeClaim to record history
function analyzeClaim() {
  const claim = document.getElementById('claim-input').value.trim();
  if (!claim) {
    document.getElementById('claim-input').focus();
    return;
  }
  addToHistory(claim);
  switchView('checking');
  resetPipeline();
  setTimeout(() => { simulateCheckingProcess(); }, 800);
}
```

---

## FIX 7 — Text/Image/URL tab mode switching

**Current:** Three tab buttons are visual only — clicking them does nothing.

**Replace** the tab buttons and input area in `view-home` with:

```html
<!-- Tabs -->
<div class="flex gap-3 mb-6">
  <button onclick="setInputMode('text')" id="tab-text"
    class="flex items-center gap-2 px-4 py-2 bg-surface rounded-lg border border-glass-stroke font-label-md text-label-md text-primary shadow-sm">
    <span class="material-symbols-outlined text-sm">text_fields</span> text
  </button>
  <button onclick="setInputMode('image')" id="tab-image"
    class="flex items-center gap-2 px-4 py-2 bg-surface/50 rounded-lg border border-glass-stroke font-label-md text-label-md text-on-surface-variant hover:bg-surface transition-colors">
    <span class="material-symbols-outlined text-sm">image</span> image
  </button>
  <button onclick="setInputMode('url')" id="tab-url"
    class="flex items-center gap-2 px-4 py-2 bg-surface/50 rounded-lg border border-glass-stroke font-label-md text-label-md text-on-surface-variant hover:bg-surface transition-colors">
    <span class="material-symbols-outlined text-sm">link</span> url
  </button>
</div>

<!-- Text mode -->
<div id="mode-text">
  <textarea class="w-full bg-transparent border-none focus:ring-0 p-0 font-body-lg text-body-lg text-primary placeholder:text-surface-tint resize-none h-32"
    id="claim-input" placeholder="Enter claim here...">The government announced free electricity for every household starting next month.</textarea>
  <div class="text-right font-label-sm text-text-muted text-xs mt-1" id="char-counter">0 / 1000 characters</div>
</div>

<!-- Image mode -->
<div id="mode-image" class="hidden">
  <div class="border-2 border-dashed border-glass-stroke rounded-xl p-8 text-center cursor-pointer hover:border-outline transition-colors" onclick="document.getElementById('img-upload').click()">
    <span class="material-symbols-outlined text-3xl text-text-muted mb-2 block">upload_file</span>
    <p class="font-body-md text-on-surface-variant text-sm">Drop an image or click to upload</p>
    <p class="font-label-sm text-text-muted text-xs mt-1">PNG, JPG · max 5MB</p>
  </div>
  <input type="file" id="img-upload" accept="image/png,image/jpeg" class="hidden" onchange="handleImageUpload(event)" />
  <div id="img-preview" class="hidden mt-4 space-y-3">
    <img id="img-thumb" class="w-full max-h-40 object-cover rounded-lg border border-glass-stroke" />
    <p class="font-label-sm text-text-muted text-xs" id="img-name"></p>
    <div class="p-3 bg-surface-container-low rounded-lg border border-glass-stroke">
      <p class="font-label-sm text-text-muted text-xs mb-1">Extracted text (OCR mock):</p>
      <textarea id="ocr-text" class="w-full bg-transparent text-sm font-body-md text-on-surface resize-none focus:ring-0 border-none" rows="3">The government announced free electricity for every household starting next month.</textarea>
    </div>
    <button onclick="document.getElementById('img-preview').classList.add('hidden');document.getElementById('img-upload').value='';" class="text-xs text-text-muted hover:text-error transition-colors font-label-sm">Remove image</button>
  </div>
</div>

<!-- URL mode -->
<div id="mode-url" class="hidden">
  <div class="flex gap-2">
    <div class="flex-1 flex items-center gap-2 bg-surface rounded-lg border border-glass-stroke px-3">
      <span class="material-symbols-outlined text-sm text-text-muted">link</span>
      <input type="url" id="url-input" placeholder="https://example.com/article"
        class="flex-1 bg-transparent border-none focus:ring-0 font-body-md text-sm py-3 text-primary placeholder:text-surface-tint" />
    </div>
    <button onclick="fetchUrl()" class="px-4 py-2 bg-surface border border-glass-stroke rounded-lg font-label-md text-sm hover:bg-surface-variant transition-colors">Fetch</button>
  </div>
  <div id="url-preview" class="hidden mt-3 p-3 bg-surface-container-low rounded-lg border border-glass-stroke">
    <p class="font-label-sm text-text-muted text-xs mb-1">Extracted from URL:</p>
    <p class="font-body-md text-sm text-on-surface" id="url-extracted">The government announced free electricity for every household starting next month.</p>
  </div>
  <p class="text-xs text-error mt-2 hidden" id="url-error">That doesn't look like a valid URL.</p>
</div>
```

**Add these JS functions:**

```javascript
let currentMode = 'text';

function setInputMode(mode) {
  currentMode = mode;
  ['text', 'image', 'url'].forEach(m => {
    const tab = document.getElementById('tab-' + m);
    const panel = document.getElementById('mode-' + m);
    const isActive = m === mode;
    tab.className = `flex items-center gap-2 px-4 py-2 rounded-lg border font-label-md text-label-md transition-colors ${
      isActive
        ? 'bg-surface border-glass-stroke text-primary shadow-sm'
        : 'bg-surface/50 border-glass-stroke text-on-surface-variant hover:bg-surface'
    }`;
    panel.classList.toggle('hidden', !isActive);
  });
}

function handleImageUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  if (file.size > 5 * 1024 * 1024) { alert('Image must be under 5MB.'); return; }
  const reader = new FileReader();
  reader.onload = ev => {
    document.getElementById('img-thumb').src = ev.target.result;
    document.getElementById('img-name').textContent = file.name;
    document.getElementById('img-preview').classList.remove('hidden');
  };
  reader.readAsDataURL(file);
}

function fetchUrl() {
  const val = document.getElementById('url-input').value.trim();
  const err = document.getElementById('url-error');
  const preview = document.getElementById('url-preview');
  if (!val.startsWith('http://') && !val.startsWith('https://')) {
    err.classList.remove('hidden');
    preview.classList.add('hidden');
    return;
  }
  err.classList.add('hidden');
  preview.classList.remove('hidden');
}

// Character counter
document.addEventListener('DOMContentLoaded', () => {
  const ta = document.getElementById('claim-input');
  const counter = document.getElementById('char-counter');
  if (ta && counter) {
    ta.addEventListener('input', () => {
      counter.textContent = `${ta.value.length} / 1000 characters`;
    });
    counter.textContent = `${ta.value.length} / 1000 characters`;
  }
});
```

---

## FIX 8 — Language detection badge

**Add** inside the input card, below the textarea and above the send button:

```html
<div class="flex items-center justify-between mt-3">
  <div class="flex items-center gap-2 px-3 py-1 rounded-full bg-tertiary-fixed/10 border border-tertiary-fixed/20 font-label-sm text-label-sm text-tertiary-container" id="lang-badge">
    <span class="material-symbols-outlined text-sm">language</span>
    <span id="lang-text">🌐 English detected · 98%</span>
  </div>
</div>
```

**Add this JS** (runs on textarea input to simulate detection):

```javascript
const langPatterns = [
  { pattern: /[\u0900-\u097F]/, lang: '🇮🇳 Hindi detected · 96%' },
  { pattern: /[\u0C00-\u0C7F]/, lang: '🇮🇳 Telugu detected · 95%' },
  { pattern: /[\u0B80-\u0BFF]/, lang: '🇮🇳 Tamil detected · 94%' },
  { pattern: /[\u0600-\u06FF]/, lang: '🇸🇦 Arabic detected · 93%' },
];

function detectLanguage(text) {
  for (const { pattern, lang } of langPatterns) {
    if (pattern.test(text)) return lang;
  }
  return '🌐 English detected · 98%';
}

// Wire up to textarea
document.addEventListener('DOMContentLoaded', () => {
  const ta = document.getElementById('claim-input');
  const langText = document.getElementById('lang-text');
  if (ta && langText) {
    ta.addEventListener('input', () => {
      langText.textContent = detectLanguage(ta.value);
    });
  }
});
```

---

## SUMMARY OF ALL CHANGES

| Fix | What | Where |
|---|---|---|
| 1 | Navbar — remove Sign In, clean links | `<nav>` |
| 2 | Real PDF export with jsPDF | `exportReport()` + `<head>` CDN |
| 3 | Three reviewer cards (Fact Checker, Logical Analyst, Bias Detector) | `view-results` |
| 4 | Expandable source cards with excerpt + relationship | `view-results` source section |
| 5 | Sub-claim breakdown 01/02/03 | `view-results` |
| 6 | Recent claims list updates after each analysis | `view-home` + JS |
| 7 | Text/Image/URL tab switching with char counter | `view-home` input area |
| 8 | Language detection badge on textarea | `view-home` input area |

**Do not change:** design tokens, fonts, animations, About page architecture diagram, terminal log, pipeline steps, `switchView()`, `simulateCheckingProcess()`, footer, or any CSS.

---

*TruthLens · Project 45 · KMIT 2025–26*
*LangGraph · DeepSeek · FAISS · Tavily · Ollama*
