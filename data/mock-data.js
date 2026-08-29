/* ============================================================
   TruthLens · data/mock-data.js
   Every hard-coded demo value the prototype needs, in one place.

   NOTHING here talks to the DOM. When a real backend arrives, this
   is the file that gets replaced by API responses — the rest of
   the app reads from these names and should not need changes.
   ============================================================ */

/* The sample claim the prototype ships with. This string is also
   duplicated in index.html as the default value of #claim-input,
   #ocr-text, #url-extracted and #checking-claim (markup defaults
   cannot read from JS without a build step). Keep them in sync. */
const DEFAULT_CLAIM = 'The government announced free electricity for every household starting next month.';

/* ---------- Verify-page checking card (5 stages) ---------- */
const SIDE_STEP_LABELS = [
    'Breaking down the claim',
    'Checking sources',
    'Consulting reviewers',
    'Weighing confidence',
    'Reaching a verdict'
];

/* ---------- Full verification pipeline (#view-checking) ----------
   Same five stages, plus the agent credited with each one. The
   suffix is appended to the agent name while that stage is live. */
const PIPELINE_STEPS = [
    { title: 'Breaking down the claim', agent: 'Decompose Agent',   suffix: ' · Running...' },
    { title: 'Checking sources',        agent: 'Credibility Agent', suffix: ' · Running...' },
    { title: 'Consulting reviewers',    agent: 'Evidence Agent',    suffix: ' · Running...' },
    { title: 'Weighing confidence',     agent: 'Judge Agent',       suffix: ' · Running...' },
    { title: 'Reaching a verdict',      agent: 'Synthesis',         suffix: ' · Finalizing...' }
];

/* ---------- Agent Processing Log cursor lines ---------- */
const TERMINAL_LINES = {
    awaiting:   '> Awaiting consensus from reviewer nodes',
    judging:    '> Evaluating judgment weights',
    finalizing: '> Synthesizing final verdict'
};

/* ---------- Language detection (script-range heuristic) ----------
   Order matters: first matching range wins. A real detector belongs
   on the backend; this only inspects Unicode blocks. */
const LANG_PATTERNS = [
    { pattern: /[\u0900-\u097F]/, lang: '🇮🇳 Hindi detected · 96%' },
    { pattern: /[\u0C00-\u0C7F]/, lang: '🇮🇳 Telugu detected · 95%' },
    { pattern: /[\u0B80-\u0BFF]/, lang: '🇮🇳 Tamil detected · 94%' },
    { pattern: /[\u0600-\u06FF]/, lang: '🇸🇦 Arabic detected · 93%' }
];
const LANG_DEFAULT = '🌐 English detected · 98%';

/* ---------- PDF report content ----------
   Mirrors what the Results view shows on screen. */
const REPORT_SUBCLAIMS = [
    ['01', 'Government made an official announcement', 'NOT SUPPORTED'],
    ['02', 'Electricity would be free for all households', 'NOT SUPPORTED'],
    ['03', 'Policy takes effect next month', 'UNVERIFIED']
];

const REPORT_SOURCES = [
    ['ndtv.com', '0.91', 'high factual', 'contradicts'],
    ['altnews.in', '0.87', 'high factual', 'contradicts'],
    ['randomblog.net', '0.34', 'low factual', 'supports']
];

const REPORT_AGENTS = [
    ['Fact Checker', 'Scanned 42 recent government press releases. No mention of nationwide free electricity found. Found a localized subsidy program in Region 4 only.'],
    ['Logical Analyst', 'Claim extrapolates a regional program to a national announcement. Classic "False Generalization" pattern.'],
    ['Bias Detector', 'High emotional language indicates sensationalism. Phrasing designed to elicit reactions rather than inform.']
];

const REPORT_META = {
    confidence: '91%',
    sources: '9 retrieved',
    subclaims: '3 decomposed',
    explanation: "No official press release or gazette notification supports this claim, and it doesn't appear on any ministry website.",
    footer: 'TruthLens · Project 45 · KMIT · LangGraph · DeepSeek · FAISS · Tavily · Ollama',
    strapline: 'Project 45 · KMIT · 2025-26 · LangGraph · DeepSeek · FAISS · Tavily'
};
